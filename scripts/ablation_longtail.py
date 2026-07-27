"""RQ3: imbalance-technique ablations on the 5 weakest (rare) classes.

Classes: SSR, SF, MR, MF, HC  (baseline XGB AUPRC 0.42/0.57/0.24/0.23/0.23)
Arms (all XGBoost, same base params as baseline):
  a_baseline    : scale_pos_weight = neg/pos (reference re-run)
  b_no_weight   : no class weighting
  c_under10     : undersample negatives to 10:1
  d_under3      : undersample negatives to 3:1
  e_over10      : duplicate positives x10 (no weighting)
  f_focal       : focal loss (gamma=2, alpha=0.75), numeric hessian

Metrics on the untouched test split: AUPRC, P@100 (precision in top-100),
recall@FPR1% proxy via threshold at 99th pct of negative scores.
Output: results/ablation_longtail.csv + console table.
"""
import csv
import gc
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score

BASE = Path(r"G:\Claude\new blockchain\crypto_defect_ml")
DATA = BASE / "data"
RESULTS = BASE / "results"
DEFECTS = ["SSR", "CSR", "CCR", "SF", "SM", "ISV", "MR", "MF", "HC", "ES", "WR", "WRT"]
TARGETS = ["SSR", "SF", "MR", "MF", "HC"]
SEED = 42
BASE_PARAMS = {"objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": 6,
               "eta": 0.1, "tree_method": "hist", "max_bin": 128, "subsample": 0.8,
               "colsample_bytree": 0.5, "nthread": -1, "verbosity": 0, "seed": SEED}
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


def focal_obj(alpha=0.75, gamma=2.0):
    def obj(preds, dtrain):
        y = dtrain.get_label()
        def grad_at(x):
            p = 1.0 / (1.0 + np.exp(-x))
            s = 2.0 * y - 1.0
            pt = np.clip(y * p + (1 - y) * (1 - p), 1e-7, 1 - 1e-7)
            at = y * alpha + (1 - y) * (1 - alpha)
            dpt = s * p * (1 - p)
            dFL_dpt = -at * (-gamma * (1 - pt) ** (gamma - 1) * np.log(pt)
                             + (1 - pt) ** gamma / pt)
            return dFL_dpt * dpt
        g = grad_at(preds)
        eps = 1e-3
        h = (grad_at(preds + eps) - grad_at(preds - eps)) / (2 * eps)
        return g, np.maximum(h, 1e-6)
    return obj


def metrics(yte, pte):
    auprc = float(average_precision_score(yte, pte))
    top100 = np.argsort(-pte)[:100]
    p_at_100 = float(yte[top100].mean())
    neg_thr = np.percentile(pte[yte == 0], 99)
    rec_fpr1 = float(((pte >= neg_thr) & (yte == 1)).sum() / max(yte.sum(), 1))
    return auprc, p_at_100, rec_fpr1


def main():
    RESULTS.mkdir(exist_ok=True)
    texts, splits, Y = load()
    tr, te = splits == "train", splits == "test"
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3),
                          max_features=40_000, sublinear_tf=True, min_df=10,
                          dtype=np.float32)
    arr = np.array(texts, dtype=object)
    del texts
    Xtr_full = vec.fit_transform(arr[tr])
    Xte = vec.transform(arr[te])
    del arr, vec
    gc.collect()
    dte = xgb.DMatrix(Xte)
    print("data ready", flush=True)

    rng = np.random.RandomState(SEED)
    out = []
    print(f"{'class':>5} {'arm':>10} | {'AUPRC':>6} {'P@100':>6} {'R@FPR1%':>7}", flush=True)
    for d in TARGETS:
        k = DEFECTS.index(d)
        ytr_full = Y[tr, k].astype(np.float32)
        yte = Y[te, k].astype(np.int8)
        pos_idx = np.where(ytr_full == 1)[0]
        neg_idx = np.where(ytr_full == 0)[0]

        def fit_predict(X, y, params=None, obj=None):
            dtr = xgb.DMatrix(X, label=y)
            p = dict(BASE_PARAMS)
            if params:
                p.update(params)
            if obj is not None:
                p.pop("eval_metric", None)
                bst = xgb.train(p, dtr, num_boost_round=ROUNDS, obj=obj)
                raw = bst.predict(dte, output_margin=True)
                pred = 1.0 / (1.0 + np.exp(-raw))
            else:
                bst = xgb.train(p, dtr, num_boost_round=ROUNDS)
                pred = bst.predict(dte)
            del dtr, bst
            gc.collect()
            return pred

        spw = max(len(neg_idx) / max(len(pos_idx), 1), 1.0)
        arms = {}
        arms["a_baseline"] = lambda: fit_predict(Xtr_full, ytr_full, {"scale_pos_weight": spw})
        arms["b_no_weight"] = lambda: fit_predict(Xtr_full, ytr_full)

        def make_under(ratio):
            def f():
                n = min(len(neg_idx), int(len(pos_idx) * ratio))
                sel = np.concatenate([pos_idx, rng.choice(neg_idx, n, replace=False)])
                rng.shuffle(sel)
                return fit_predict(Xtr_full[sel], ytr_full[sel])
            return f
        arms["c_under10"] = make_under(10)
        arms["d_under3"] = make_under(3)

        def over():
            reps = [Xtr_full] + [Xtr_full[pos_idx]] * 9
            Xo = sparse.vstack(reps).tocsr()
            yo = np.concatenate([ytr_full] + [ytr_full[pos_idx]] * 9)
            return fit_predict(Xo, yo)
        arms["e_over10"] = over
        arms["f_focal"] = lambda: fit_predict(Xtr_full, ytr_full, obj=focal_obj())

        for arm, fn in arms.items():
            pred = fn()
            auprc, p100, rf = metrics(yte, pred)
            print(f"{d:>5} {arm:>10} | {auprc:>6.3f} {p100:>6.3f} {rf:>7.3f}", flush=True)
            out.append({"class": d, "arm": arm, "auprc": round(auprc, 4),
                        "p_at_100": round(p100, 4), "recall_at_fpr1": round(rf, 4)})

    with open(RESULTS / "ablation_longtail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "arm", "auprc", "p_at_100", "recall_at_fpr1"])
        w.writeheader()
        w.writerows(out)
    print(f"wrote {RESULTS / 'ablation_longtail.csv'}", flush=True)


if __name__ == "__main__":
    main()
