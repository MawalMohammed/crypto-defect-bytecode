"""Neural arm 1: multi-label 1D-CNN over opcode sequences (local GPU).

Sized for RTX 3050 Ti 4 GB: AMP, seq cap 8192, batch 16 (+grad-accum 2),
embedding 96, conv widths 3/5/7 x 192 channels, global max-pool.
Multi-label head (12 sigmoids), BCEWithLogits with per-class pos_weight
(neg/pos capped at 100) for the long tail.

Early stopping on mean val AUPRC over trainable classes (patience 3).
Output: results/cnn_metrics.csv, model -> results/cnn_best.pt
"""
import csv
import gc
import json
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
BATCH = 16
ACCUM = 2
EPOCHS = 15
PATIENCE = 3
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# keep the system awake while this process runs (cleared automatically on exit);
# does not stop lid-close sleep, but checkpointing below covers that case
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
        self.seqs = seqs
        self.Y = Y
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.seqs[j], self.Y[j]


def collate(batch):
    maxlen = max(len(s) for s, _ in batch)
    x = torch.zeros(len(batch), maxlen, dtype=torch.long)
    y = torch.stack([torch.from_numpy(np.asarray(yy)) for _, yy in batch])
    for i, (s, _) in enumerate(batch):
        x[i, : len(s)] = torch.from_numpy(s.astype(np.int64))
    return x, y


class OpcodeCNN(nn.Module):
    def __init__(self, vocab_size, emb=96, channels=192, widths=(3, 5, 7), n_out=12):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb, channels, w, padding=w // 2) for w in widths])
        self.head = nn.Sequential(
            nn.Linear(channels * len(widths), 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_out))
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)          # B, emb, L
        feats = [torch.relu(c(e)).amax(dim=2) for c in self.convs]
        return self.head(torch.cat(feats, dim=1))


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

    model = OpcodeCNN(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_score, best_epoch, start_ep = -1, -1, 1
    ckpt_path = RESULTS / "cnn_ckpt.pt"
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
                       RESULTS / "cnn_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep,
                    "best_score": best_score, "best_epoch": best_epoch},
                   ckpt_path)
        if score <= best_score and ep - best_epoch >= PATIENCE:
            print(f"early stop at epoch {ep} (best {best_epoch})", flush=True)
            break

    ckpt = torch.load(RESULTS / "cnn_best.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    pv, yv = evaluate(model, loaders["val"], device)
    pt, yt = evaluate(model, loaders["test"], device)

    out = []
    print(f"{'class':>5} {'n_test+':>7} | {'CNN AUPRC':>9} {'CNN F1':>6} {'AUROC':>6}", flush=True)
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
        rec["cnn_auprc"] = round(float(average_precision_score(yt[:, k], pt[:, k])), 4)
        rec["cnn_f1"] = round(float(f1_score(yt[:, k], pt[:, k] >= best_t, zero_division=0)), 4)
        rec["cnn_auroc"] = round(float(roc_auc_score(yt[:, k], pt[:, k])), 4)
        rec["cnn_threshold"] = round(float(best_t), 2)
        print(f"{d:>5} {rec['n_test_pos']:>7} | {rec['cnn_auprc']:>9.3f} "
              f"{rec['cnn_f1']:>6.3f} {rec['cnn_auroc']:>6.3f}", flush=True)
        out.append(rec)

    keys = sorted({k for r in out for k in r}, key=lambda s: (s != "class", s))
    with open(RESULTS / "cnn_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {RESULTS / 'cnn_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
