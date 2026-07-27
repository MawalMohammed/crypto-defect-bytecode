"""Neural arm 2: conv-downsampled transformer over opcode sequences (local GPU).

RTX 3050 Ti 4 GB sizing: seq cap 8192 -> conv frontend (2x stride-2) -> <=2048
positions -> 4-layer TransformerEncoder (d=128, 4 heads, FFN 512, SDPA kernels,
AMP fp16) -> masked mean+max pooling -> 12 sigmoids.
BCEWithLogits with per-class pos_weight (capped 100). Early stop on val mAUPRC.
Keep-awake + per-epoch checkpoint resume (survives system sleep).

Output: results/transformer_metrics.csv, model -> results/transformer_best.pt
"""
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

BASE = Path(r"G:\Claude\new blockchain\crypto_defect_ml")
DATA = BASE / "data"
RESULTS = BASE / "results"
DEFECTS = ["SSR", "CSR", "CCR", "SF", "SM", "ISV", "MR", "MF", "HC", "ES", "WR", "WRT"]

SEQ_CAP = 8192
BATCH = 8
ACCUM = 4
EPOCHS = 15
PATIENCE = 3
LR = 3e-4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

try:
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
except Exception:
    pass


def load_data():
    ops = {}
    for line in open(DATA / "opcodes.jsonl", encoding="utf-8"):
        d = json.loads(line)
        ops[d["address"]] = d["ops"]
    rows = []
    with open(DATA / "dataset.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["address"] in ops:
                rows.append(row)
    vocab = {"<PAD>": 0}
    seqs = []
    for r in rows:
        toks = ops[r["address"]].split()[:SEQ_CAP]
        ids = np.empty(len(toks), dtype=np.int16)
        for i, t in enumerate(toks):
            v = vocab.get(t)
            if v is None:
                v = len(vocab)
                vocab[t] = v
            ids[i] = v
        seqs.append(ids)
    del ops
    gc.collect()
    splits = np.array([r["split"] for r in rows])
    Y = np.array([[int(r[d]) for d in DEFECTS] for r in rows], dtype=np.float32)
    print(f"contracts: {len(seqs):,} | vocab: {len(vocab)}", flush=True)
    return seqs, splits, Y, vocab


class SeqDataset(Dataset):
    def __init__(self, seqs, Y, idx):
        self.seqs, self.Y, self.idx = seqs, Y, idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.seqs[j], self.Y[j]


def collate(batch):
    maxlen = max(len(s) for s, _ in batch)
    maxlen = max(maxlen, 8)  # conv frontend minimum
    x = torch.zeros(len(batch), maxlen, dtype=torch.long)
    y = torch.stack([torch.from_numpy(np.asarray(yy)) for _, yy in batch])
    for i, (s, _) in enumerate(batch):
        x[i, : len(s)] = torch.from_numpy(s.astype(np.int64))
    return x, y


class OpcodeTransformer(nn.Module):
    def __init__(self, vocab_size, d=128, heads=4, layers=4, ffn=512, n_out=12,
                 max_pos=SEQ_CAP // 8 + 1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.front = nn.Sequential(
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
        )
        pe = torch.zeros(max_pos, d)
        pos = torch.arange(max_pos).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(2 * d, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n_out))

    def forward(self, x):
        valid = (x != 0).float().unsqueeze(1)                    # B,1,L
        e = self.emb(x).transpose(1, 2)                          # B,d,L
        h = self.front(e).transpose(1, 2)                        # B,L/4,d
        vmask = torch.nn.functional.max_pool1d(valid, 8, 8, ceil_mode=True)
        vmask = vmask.squeeze(1)[:, : h.size(1)] > 0             # B,L/4 valid
        h = h + self.pe[: h.size(1)].unsqueeze(0)
        h = self.enc(h, src_key_padding_mask=~vmask)
        m = vmask.unsqueeze(2).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = h.masked_fill(~vmask.unsqueeze(2), float("-inf")).amax(1)
        return self.head(torch.cat([mean, mx], dim=1))


def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            ps.append(torch.sigmoid(logits.float()).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def main():
    RESULTS.mkdir(exist_ok=True)
    device = "cuda"
    seqs, splits, Y, vocab = load_data()
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    trainable = [k for k in range(12) if Y[idx["train"], k].sum() >= 5]
    print(f"trainable classes: {[DEFECTS[k] for k in trainable]}", flush=True)

    pos = Y[idx["train"]].sum(axis=0)
    neg = len(idx["train"]) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 100), dtype=torch.float32).to(device)

    loaders = {
        s: DataLoader(SeqDataset(seqs, Y, idx[s]), batch_size=BATCH, shuffle=(s == "train"),
                      collate_fn=collate, num_workers=0, pin_memory=True)
        for s in ("train", "val", "test")
    }

    model = OpcodeTransformer(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_score, best_epoch, start_ep = -1, -1, 1
    ckpt_path = RESULTS / "transformer_ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        best_score, best_epoch, start_ep = ck["best_score"], ck["best_epoch"], ck["epoch"] + 1
        print(f"resumed from checkpoint at epoch {ck['epoch']} (best {best_score:.4f})", flush=True)

    for ep in range(start_ep, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total = steps = 0
        opt.zero_grad(set_to_none=True)
        for bi, (x, y) in enumerate(loaders["train"]):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = crit(model(x.to(device, non_blocking=True)),
                            y.to(device, non_blocking=True)) / ACCUM
            scaler.scale(loss).backward()
            if (bi + 1) % ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += loss.item() * ACCUM
            steps += 1
        p, yv = evaluate(model, loaders["val"], device)
        aps = [average_precision_score(yv[:, k], p[:, k]) for k in trainable
               if yv[:, k].sum() > 0]
        score = float(np.mean(aps))
        print(f"epoch {ep}: loss {total/max(steps,1):.4f} | val mAUPRC {score:.4f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        if score > best_score:
            best_score, best_epoch = score, ep
            torch.save({"model": model.state_dict(), "vocab": vocab, "epoch": ep},
                       RESULTS / "transformer_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep,
                    "best_score": best_score, "best_epoch": best_epoch}, ckpt_path)
        if score <= best_score and ep - best_epoch >= PATIENCE:
            print(f"early stop at epoch {ep} (best {best_epoch})", flush=True)
            break

    ckpt = torch.load(RESULTS / "transformer_best.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    pv, yv = evaluate(model, loaders["val"], device)
    pt, yt = evaluate(model, loaders["test"], device)

    out = []
    print(f"{'class':>5} {'n_test+':>7} | {'TRF AUPRC':>9} {'TRF F1':>6} {'AUROC':>6}", flush=True)
    for k, d in enumerate(DEFECTS):
        rec = {"class": d, "n_test_pos": int(yt[:, k].sum())}
        if k not in trainable or yt[:, k].sum() < 1:
            rec["status"] = "skipped_few_pos"
            out.append(rec)
            continue
        rec["status"] = "ok"
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.05, 0.95, 19):
            if yv[:, k].sum() and (pv[:, k] >= t).any():
                f1v = f1_score(yv[:, k], pv[:, k] >= t, zero_division=0)
                if f1v > best_f1:
                    best_t, best_f1 = t, f1v
        rec["trf_auprc"] = round(float(average_precision_score(yt[:, k], pt[:, k])), 4)
        rec["trf_f1"] = round(float(f1_score(yt[:, k], pt[:, k] >= best_t, zero_division=0)), 4)
        rec["trf_auroc"] = round(float(roc_auc_score(yt[:, k], pt[:, k])), 4)
        rec["trf_threshold"] = round(float(best_t), 2)
        print(f"{d:>5} {rec['n_test_pos']:>7} | {rec['trf_auprc']:>9.3f} "
              f"{rec['trf_f1']:>6.3f} {rec['trf_auroc']:>6.3f}", flush=True)
        out.append(rec)

    keys = sorted({k for r in out for k in r}, key=lambda s: (s != "class", s))
    with open(RESULTS / "transformer_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {RESULTS / 'transformer_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
