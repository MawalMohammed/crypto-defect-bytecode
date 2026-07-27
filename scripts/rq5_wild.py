"""RQ5: in-the-wild study — score recent, live mainnet contracts (bytecode-only)
with the trained per-class XGBoost detectors.

Steps (all free, keyless):
  1. sample ~150 recent blocks spread over the last ~30 days via public RPC
  2. collect unique `to` addresses, exclude the labeled corpus, eth_getCode each,
     keep real contracts (runtime code >= 500 bytes)
  3. disassemble (same tokenizer as training), refit the training TF-IDF
     (deterministic), score with the 11 saved boosters
  4. add a semantic sanity column for WR (TIMESTAMP/BLOCKHASH/DIFFICULTY present)
  5. if ETHERSCAN_API_KEY is set, annotate verification status for the top hits

Output: results/wild_scores.csv, results/wild_top_report.md
"""
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

from extract_opcodes import OPCODES, disassemble, strip_metadata

BASE = Path(r"G:\Claude\new blockchain\crypto_defect_ml")
DATA = BASE / "data"
RESULTS = BASE / "results"
CORPUS_LIST = Path(r"G:\Claude\new blockchain\CryptoScan_repo\Dataset\contract-list.txt")
DEFECTS = ["SSR", "CSR", "CCR", "SF", "SM", "ISV", "MR", "MF", "HC", "WR", "WRT"]

RPC = "https://ethereum-rpc.publicnode.com"
N_BLOCKS = 150
SPAN_BLOCKS = 216_000          # ~30 days
TARGET_CONTRACTS = 1500
MIN_CODE_BYTES = 500
TOP_K = 20


def rpc(method, params, retries=8):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                RPC, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                                      "params": params}).encode(),
                headers={"Content-Type": "application/json",
                         "User-Agent": "research-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.load(r)
            if "result" in resp:
                return resp["result"]
        except Exception:
            pass
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"rpc {method} failed after {retries} tries")


def sample_candidates():
    latest = int(rpc("eth_blockNumber", []), 16)
    print(f"latest block: {latest:,}", flush=True)
    corpus = {l.strip().lower() for l in open(CORPUS_LIST) if l.strip()}
    step = SPAN_BLOCKS // N_BLOCKS
    cands = []
    seen = set()
    for i in range(N_BLOCKS):
        bn = latest - i * step
        blk = rpc("eth_getBlockByNumber", [hex(bn), True])
        for tx in blk.get("transactions", []):
            to = (tx.get("to") or "").lower()
            if to and to not in seen and to not in corpus:
                seen.add(to)
                cands.append(to)
        if (i + 1) % 25 == 0:
            print(f"  blocks {i + 1}/{N_BLOCKS}, candidates {len(cands):,}", flush=True)
        time.sleep(0.3)
    print(f"candidate addresses: {len(cands):,}", flush=True)
    return cands


def fetch_codes(cands):
    out = {}
    for a in cands:
        if len(out) >= TARGET_CONTRACTS:
            break
        try:
            code = rpc("eth_getCode", [a, "latest"], retries=4)
        except RuntimeError:
            continue
        if code and len(code) >= 2 + 2 * MIN_CODE_BYTES:
            out[a] = code
        if len(out) % 200 == 0 and len(out) > 0:
            print(f"  contracts kept: {len(out):,}", flush=True)
        time.sleep(0.12)
    print(f"wild contracts: {len(out):,}", flush=True)
    return out


def load_train_texts():
    ops = {}
    for line in open(DATA / "opcodes.jsonl", encoding="utf-8"):
        d = json.loads(line)
        ops[d["address"]] = d["ops"]
    texts = []
    with open(DATA / "dataset.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train" and row["address"] in ops:
                texts.append(ops[row["address"]])
    return texts


def etherscan_status(addr, key):
    url = ("https://api.etherscan.io/v2/api?chainid=1&module=contract"
           f"&action=getsourcecode&address={addr}&apikey={key}")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.load(r)
        src = d.get("result", [{}])[0].get("SourceCode", "")
        return "verified" if src else "unverified"
    except Exception:
        return "unknown"


def main():
    RESULTS.mkdir(exist_ok=True)
    cands = sample_candidates()
    codes = fetch_codes(cands)

    wild_addrs, wild_texts, wr_rule, sizes = [], [], [], []
    for a, hexcode in codes.items():
        code = bytes.fromhex(hexcode[2:])
        toks = disassemble(strip_metadata(code))
        wild_addrs.append(a)
        wild_texts.append(" ".join(toks))
        s = set(toks)
        wr_rule.append(int(bool(s & {"TIMESTAMP", "BLOCKHASH", "DIFFICULTY"}) and "SHA3" in s))
        sizes.append(len(code))
    print("disassembled", flush=True)

    train_texts = load_train_texts()
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3),
                          max_features=40_000, sublinear_tf=True, min_df=10,
                          dtype=np.float32)
    vec.fit(np.array(train_texts, dtype=object))
    del train_texts
    Xw = vec.transform(np.array(wild_texts, dtype=object))
    dw = xgb.DMatrix(Xw)
    print("vectorized", flush=True)

    scores = {}
    for d in DEFECTS:
        mp = RESULTS / "xgb_models" / f"{d}.json"
        if not mp.exists():
            continue
        bst = xgb.Booster()
        bst.load_model(str(mp))
        scores[d] = bst.predict(dw)
        print(f"scored {d}", flush=True)

    key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    rows = []
    for i, a in enumerate(wild_addrs):
        row = {"address": a, "code_bytes": sizes[i], "wr_rule": wr_rule[i]}
        for d in scores:
            row[d] = round(float(scores[d][i]), 4)
        rows.append(row)
    with open(RESULTS / "wild_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report = ["# RQ5: top wild-contract flags per class\n",
              f"Sampled {len(wild_addrs):,} recent active mainnet contracts "
              f"(>= {MIN_CODE_BYTES}B code, not in labeled corpus).\n"]
    for d in ["WR", "SM", "CCR", "CSR", "ISV", "SSR"]:
        if d not in scores:
            continue
        order = np.argsort(-scores[d])[:TOP_K]
        report.append(f"\n## {d} top-{TOP_K}\n")
        report.append("| address | score | code B | WR-rule | verif |")
        report.append("|---|---|---|---|---|")
        for j in order:
            a = wild_addrs[j]
            status = etherscan_status(a, key) if key else "n/a (no key)"
            if key:
                time.sleep(0.25)
            report.append(f"| {a} | {scores[d][j]:.3f} | {sizes[j]} | "
                          f"{wr_rule[j]} | {status} |")
    (RESULTS / "wild_top_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {RESULTS / 'wild_top_report.md'} and wild_scores.csv", flush=True)


if __name__ == "__main__":
    main()
