"""Hardening 1: evaluate detectors against HUMAN ground truth (gold TP/FP),
not CryptoScan pseudo-labels. Restricted to the TEST split (held out).

Gold set = contracts CryptoScan flagged, then human-audited T (real defect) or
F (false positive). For test-split gold contracts we ask:
  - recall_on_T: fraction of human-confirmed defects our model scores >= 0.5
  - false_confirm_F: fraction of human-rejected flags our model ALSO scores >=0.5
    (i.e. did we merely replicate CryptoScan's mistakes?)
Pooled across classes because per-class F counts are tiny.

Output: results/gold_eval.md
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(r"G:\Claude\new blockchain\crypto_defect_ml")
DATA = BASE / "data"
RESULTS = BASE / "results"
GOLD = Path(r"G:\Claude\new blockchain\CryptoScan_repo\Experiments\Dataset1")

NAME_TO_CODE = {
    "Cross_Chain_Signature_Replay": "CCR",
    "Cross_Contract_Signature_Replay": "CSR",
    "Ecmul_Scalar_Input_Overflow": "ES",
    "Hash_Collision_With_Dynamic_Length_Arguments": "HC",
    "Insufficient_Signature_Verification": "ISV",
    "Merkle_Proof_Front_Running": "MF",
    "Merkle_Proof_Replay": "MR",
    "Signature_Frontrunning": "SF",
    "Signature_Malleability": "SM",
    "Single_Contract_Signature_Replay": "SSR",
    "Weak_Randomness_From_Hashing_Chain_Attributes": "WR",
    "Weak_Randomness_From_Hashing_Tx_Attributes": "WRT",
}


def main():
    ops = {}
    for line in open(DATA / "opcodes.jsonl", encoding="utf-8"):
        d = json.loads(line)
        ops[d["address"]] = d["ops"]
    split_of = {}
    train_texts = []
    with open(DATA / "dataset.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_of[row["address"]] = row["split"]
            if row["split"] == "train" and row["address"] in ops:
                train_texts.append(ops[row["address"]])

    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3),
                          max_features=40_000, sublinear_tf=True, min_df=10,
                          dtype=np.float32)
    vec.fit(np.array(train_texts, dtype=object))
    del train_texts

    report = ["# Hardening 1: gold (human-label) evaluation on the TEST split\n"]
    pooled_T_scores, pooled_F_scores = [], []
    per_class = []
    for fp in sorted(glob.glob(str(GOLD / "*.csv"))):
        code = NAME_TO_CODE.get(Path(fp).stem)
        if code is None:
            continue
        mp = RESULTS / "xgb_models" / f"{code}.json"
        if not mp.exists():
            per_class.append((code, "no model (ES untrainable)", None))
            continue
        bst = xgb.Booster()
        bst.load_model(str(mp))

        addrs, labels = [], []
        for row in csv.DictReader(open(fp, encoding="utf-8-sig")):
            a = (row.get("Contract Address") or "").strip().lower()
            lab = (row.get("TP/FP") or "").strip().upper()
            if not a or a not in ops or split_of.get(a) != "test":
                continue
            addrs.append(a)
            labels.append(1 if lab.startswith("T") else 0)
        if not addrs:
            per_class.append((code, "no test-split gold contracts", None))
            continue
        X = vec.transform(np.array([ops[a] for a in addrs], dtype=object))
        scores = bst.predict(xgb.DMatrix(X))
        labels = np.array(labels)
        nT, nF = int((labels == 1).sum()), int((labels == 0).sum())
        recall_T = float((scores[labels == 1] >= 0.5).mean()) if nT else float("nan")
        fconf = float((scores[labels == 0] >= 0.5).mean()) if nF else float("nan")
        pooled_T_scores += list(scores[labels == 1])
        pooled_F_scores += list(scores[labels == 0])
        per_class.append((code, f"T={nT} F={nF} | recall@.5(T)={recall_T:.2f} "
                          f"| false-confirm(F)={fconf if nF else float('nan'):.2f}",
                          (nT, nF)))

    report.append("Per class (test-split gold only):\n")
    for code, msg, _ in per_class:
        report.append(f"- **{code}**: {msg}")

    T = np.array(pooled_T_scores)
    F = np.array(pooled_F_scores)
    report.append(f"\n## Pooled (all classes, test-split gold)\n")
    report.append(f"- human-confirmed defects (T): n={len(T)}, "
                  f"mean model score {T.mean():.3f}, recall@0.5 {float((T>=0.5).mean()):.3f}")
    report.append(f"- human-rejected flags (F): n={len(F)}, "
                  f"mean model score {F.mean():.3f}, "
                  f"false-confirm@0.5 {float((F>=0.5).mean()) if len(F) else float('nan'):.3f}")
    if len(F) and len(T):
        y = np.r_[np.ones(len(T)), np.zeros(len(F))]
        s = np.r_[T, F]
        report.append(f"- T-vs-F separation: AUROC {roc_auc_score(y, s):.3f}, "
                      f"AUPRC {average_precision_score(y, s):.3f}")
    report.append(f"\nCryptoScan human-audited FP rate in gold set overall: "
                  f"53/993 = 5.3% (context for how few F exist).")
    (RESULTS / "gold_eval.md").write_text("\n".join(report), encoding="utf-8")
    print("wrote gold_eval.md", flush=True)
    print("\n".join(report[-6:]), flush=True)


if __name__ == "__main__":
    main()
