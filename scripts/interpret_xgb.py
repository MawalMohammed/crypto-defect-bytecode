"""RQ4: retrain the winning XGBoost per class, save boosters, and dump the
top gain-ranked opcode n-gram features per defect class for semantic inspection.

Also dumps top +/- logistic-regression coefficients as a linear cross-check.
Output: results/xgb_models/<class>.json (boosters),
        results/feature_importance.md
"""
import csv
import gc
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE = Path(r"G:\Claude\new blockchain\crypto_defect_ml")
DATA = BASE / "data"
RESULTS = BASE / "results"
DEFECTS = ["SSR", "CSR", "CCR", "SF", "SM", "ISV", "MR", "MF", "HC", "ES", "WR", "WRT"]
TOP_N = 25


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


def main():
    (RESULTS / "xgb_models").mkdir(parents=True, exist_ok=True)
    texts, splits, Y = load()
    tr = splits == "train"
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3),
                          max_features=40_000, sublinear_tf=True, min_df=10,
                          dtype=np.float32)
    Xtr = vec.fit_transform(np.array(texts, dtype=object)[tr])
    names = vec.get_feature_names_out()
    del texts
    gc.collect()
    print(f"features: {len(names):,}", flush=True)
    dtr = xgb.DMatrix(Xtr)

    report = ["# RQ4: top features per defect class (XGBoost gain / LR coefficients)\n"]
    for k, d in enumerate(DEFECTS):
        ytr = Y[tr, k]
        if ytr.sum() < 5:
            report.append(f"\n## {d}: skipped (insufficient positives)\n")
            continue
        print(f"training {d} ...", flush=True)
        dtr.set_label(ytr.astype(np.float32))
        spw = max(float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0), 1.0)
        params = {"objective": "binary:logistic", "eval_metric": "aucpr",
                  "max_depth": 6, "eta": 0.1, "tree_method": "hist", "max_bin": 128,
                  "subsample": 0.8, "colsample_bytree": 0.5, "scale_pos_weight": spw,
                  "nthread": -1, "verbosity": 0}
        bst = xgb.train(params, dtr, num_boost_round=300)
        bst.save_model(str(RESULTS / "xgb_models" / f"{d}.json"))
        gain = bst.get_score(importance_type="gain")
        top = sorted(gain.items(), key=lambda kv: -kv[1])[:TOP_N]
        report.append(f"\n## {d} — XGB top-{TOP_N} by gain\n")
        for feat, g in top:
            idx = int(feat[1:])
            report.append(f"- `{names[idx]}`  (gain {g:.1f})")
        del bst
        gc.collect()

        lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        lr.fit(Xtr, ytr)
        coef = lr.coef_[0]
        pos_i = np.argsort(-coef)[:10]
        report.append(f"\n{d} — LR top positive coefficients:")
        for i in pos_i:
            report.append(f"- `{names[i]}` (+{coef[i]:.2f})")
        del lr
        gc.collect()

    (RESULTS / "feature_importance.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {RESULTS / 'feature_importance.md'}", flush=True)


if __name__ == "__main__":
    main()
