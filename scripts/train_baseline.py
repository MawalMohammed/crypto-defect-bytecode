"""Baseline: TF-IDF over opcode n-grams -> per-class classifiers -> AUPRC.

Memory-lean for a 16 GB machine:
  - float32 TF-IDF, max 40k features, min_df=10
  - XGBoost via native API: DMatrix built ONCE, labels swapped per class
    (avoids 12x multi-GB matrix constructions), max_bin=128
  - gc between classes

Arms: logistic (one-vs-rest, sparse) and xgboost.
Metrics: AUPRC (primary), ROC-AUC, F1 at val-tuned threshold.
Output: results/baseline_metrics.csv + console table.
"""
import csv
import gc
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

BASE = Path(r"G:\Claude\new blockchain\crypto_defect_ml")
DATA = BASE / "data"
RESULTS = BASE / "results"
DEFECTS = ["SSR", "CSR", "CCR", "SF", "SM", "ISV", "MR", "MF", "HC", "ES", "WR", "WRT"]


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


def tune_threshold(yva, pva):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        if yva.sum() and (pva >= t).any():
            f1v = f1_score(yva, pva >= t, zero_division=0)
            if f1v > best_f1:
                best_t, best_f1 = t, f1v
    return best_t


def main():
    RESULTS.mkdir(exist_ok=True)
    texts, splits, Y = load()
    tr, va, te = splits == "train", splits == "val", splits == "test"
    print(f"train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}", flush=True)

    vec = TfidfVectorizer(
        analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3),
        max_features=40_000, sublinear_tf=True, min_df=10, dtype=np.float32,
    )
    arr = np.array(texts, dtype=object)
    del texts
    Xtr = vec.fit_transform(arr[tr])
    Xva = vec.transform(arr[va])
    Xte = vec.transform(arr[te])
    del arr, vec
    gc.collect()
    print(f"features: {Xtr.shape[1]:,} | train nnz: {Xtr.nnz:,}", flush=True)

    dtr = xgb.DMatrix(Xtr)
    dva = xgb.DMatrix(Xva)
    dte = xgb.DMatrix(Xte)
    print("DMatrices built", flush=True)

    out = []
    print(f"{'class':>5} {'n_test+':>7} {'prev':>7} | {'LR AUPRC':>8} {'LR F1':>6} | {'XGB AUPRC':>9} {'XGB F1':>6}", flush=True)

    for k, d in enumerate(DEFECTS):
        ytr, yva, yte = Y[tr, k], Y[va, k], Y[te, k]
        prev = float(yte.mean())
        rec = {"class": d, "n_train_pos": int(ytr.sum()), "n_test_pos": int(yte.sum()),
               "test_prevalence": round(prev, 5)}
        if ytr.sum() < 5 or yte.sum() < 1:
            print(f"{d:>5} {int(yte.sum()):>7} {prev:>7.4f} | insufficient positives, skipped", flush=True)
            rec["status"] = "skipped_few_pos"
            out.append(rec)
            continue
        rec["status"] = "ok"
        line = f"{d:>5} {int(yte.sum()):>7} {prev:>7.4f}"

        # --- logistic ---
        lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        lr.fit(Xtr, ytr)
        pva = lr.predict_proba(Xva)[:, 1]
        pte = lr.predict_proba(Xte)[:, 1]
        t = tune_threshold(yva, pva)
        rec.update(lr_auprc=round(float(average_precision_score(yte, pte)), 4),
                   lr_f1=round(float(f1_score(yte, pte >= t, zero_division=0)), 4),
                   lr_auroc=round(float(roc_auc_score(yte, pte)), 4),
                   lr_threshold=round(float(t), 2))
        line += f" | {rec['lr_auprc']:>8.3f} {rec['lr_f1']:>6.3f}"
        del lr, pva, pte
        gc.collect()

        # --- xgboost (reuse DMatrices, swap labels) ---
        dtr.set_label(ytr.astype(np.float32))
        spw = max(float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0), 1.0)
        params = {"objective": "binary:logistic", "eval_metric": "aucpr",
                  "max_depth": 6, "eta": 0.1, "tree_method": "hist",
                  "max_bin": 128, "subsample": 0.8, "colsample_bytree": 0.5,
                  "scale_pos_weight": spw, "nthread": -1, "verbosity": 0}
        bst = xgb.train(params, dtr, num_boost_round=300)
        pva = bst.predict(dva)
        pte = bst.predict(dte)
        t = tune_threshold(yva, pva)
        rec.update(xgb_auprc=round(float(average_precision_score(yte, pte)), 4),
                   xgb_f1=round(float(f1_score(yte, pte >= t, zero_division=0)), 4),
                   xgb_auroc=round(float(roc_auc_score(yte, pte)), 4),
                   xgb_threshold=round(float(t), 2))
        line += f" | {rec['xgb_auprc']:>9.3f} {rec['xgb_f1']:>6.3f}"
        print(line, flush=True)
        out.append(rec)
        del bst, pva, pte
        gc.collect()

    keys = sorted({k for r in out for k in r}, key=lambda s: (s != "class", s))
    with open(RESULTS / "baseline_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {RESULTS / 'baseline_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
