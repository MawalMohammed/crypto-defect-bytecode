"""Hardening 3: statistical rigor for the XGBoost result.
  - 3 training seeds per class (row/col subsample variation) -> mean +/- std AUPRC
  - 1000x bootstrap over the test set -> 95% percentile CI on AUPRC (seed-0 model)
Output: results/xgb_ci.md, results/xgb_ci.csv
"""
import csv
import gc
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score

BASE = Path(r"G:\Claude\new blockchain\crypto_defect_ml")
DATA = BASE / "data"
RESULTS = BASE / "results"
DEFECTS = ["SSR", "CSR", "CCR", "SF", "SM", "ISV", "MR", "MF", "HC", "ES", "WR", "WRT"]
SEEDS = [42, 7, 123]
N_BOOT = 1000
ROUNDS = 300


def load():
    ops = {}
    for line in open(DATA / "opcodes.jsonl", encoding="utf-8"):
        d = json.loads(line)
        ops[d["address"]] = d["ops"]
    rows = []
    with open(DATA / "dataset.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["address"] in ops:
                rows.append(row)
    texts = [ops[r["address"]] for r in rows]
    del ops
    splits = np.array([r["split"] for r in rows])
    Y = np.array([[int(r[d]) for d in DEFECTS] for r in rows], dtype=np.int8)
    return texts, splits, Y


def boot_ci(y, p, rng, n=N_BOOT):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0:
            continue
        vals.append(average_precision_score(y[b], p[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    texts, splits, Y = load()
    tr, te = splits == "train", splits == "test"
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3),
                          max_features=40_000, sublinear_tf=True, min_df=10,
                          dtype=np.float32)
    arr = np.array(texts, dtype=object)
    del texts
    Xtr = vec.fit_transform(arr[tr])
    Xte = vec.transform(arr[te])
    del arr
    gc.collect()
    dtr = xgb.DMatrix(Xtr)
    dte = xgb.DMatrix(Xte)
    rng = np.random.RandomState(2024)

    out = []
    print(f"{'class':>5} | {'mean':>6} {'std':>5} | {'95% CI':>16}", flush=True)
    for k, d in enumerate(DEFECTS):
        ytr, yte = Y[tr, k], Y[te, k]
        if ytr.sum() < 5 or yte.sum() < 1:
            continue
        spw = max(float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0), 1.0)
        aps, p0 = [], None
        for s in SEEDS:
            dtr.set_label(ytr.astype(np.float32))
            params = {"objective": "binary:logistic", "eval_metric": "aucpr",
                      "max_depth": 6, "eta": 0.1, "tree_method": "hist", "max_bin": 128,
                      "subsample": 0.8, "colsample_bytree": 0.5, "scale_pos_weight": spw,
                      "nthread": -1, "verbosity": 0, "seed": s}
            bst = xgb.train(params, dtr, num_boost_round=ROUNDS)
            p = bst.predict(dte)
            aps.append(average_precision_score(yte, p))
            if s == SEEDS[0]:
                p0 = p
            del bst
            gc.collect()
        lo, hi = boot_ci(yte.astype(int), p0, rng)
        rec = {"class": d, "mean_auprc": round(float(np.mean(aps)), 4),
               "std_auprc": round(float(np.std(aps)), 4),
               "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
               "seeds": ";".join(f"{a:.4f}" for a in aps)}
        out.append(rec)
        print(f"{d:>5} | {rec['mean_auprc']:.3f} {rec['std_auprc']:.3f} | "
              f"[{lo:.3f}, {hi:.3f}]", flush=True)

    with open(RESULTS / "xgb_ci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "mean_auprc", "std_auprc",
                                          "ci_lo", "ci_hi", "seeds"])
        w.writeheader()
        w.writerows(out)
    lines = ["# Hardening 3: XGBoost AUPRC, 3 seeds (mean+/-std) and 1000x bootstrap 95% CI\n",
             "| class | mean AUPRC | std (seeds) | 95% CI (bootstrap) |",
             "|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['class']} | {r['mean_auprc']:.3f} | {r['std_auprc']:.3f} "
                     f"| [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |")
    (RESULTS / "xgb_ci.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote xgb_ci.md / xgb_ci.csv", flush=True)


if __name__ == "__main__":
    main()
