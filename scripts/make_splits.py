"""Join opcodes + clone clusters + 12-defect labels, assign cluster-level splits.

Split protocol (leakage-safe): entire clone clusters go to one of train/val/test
(70/10/20), assigned greedily largest-first to the split whose current share is
furthest below target (seeded shuffle for ties). No address-level splitting.

Inputs : data/opcodes.jsonl, data/clusters.csv, ../panel.csv (repo root)
Output : data/dataset.csv  (address, cluster_id, split, SSR..WRT, analysis_ok)
         -- opcode text stays in opcodes.jsonl; join on address at train time.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

BASE = Path(r"G:\Claude\new blockchain")
DATA = BASE / "crypto_defect_ml" / "data"
DEFECTS = ["SSR", "CSR", "CCR", "SF", "SM", "ISV", "MR", "MF", "HC", "ES", "WR", "WRT"]
TARGETS = {"train": 0.70, "val": 0.10, "test": 0.20}
SEED = 42


def main():
    labels = {}
    with open(BASE / "panel.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["address"].lower()] = row

    cluster_of = {}
    with open(DATA / "clusters.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cluster_of[row["address"]] = row["cluster_id"]

    have_ops = set()
    for line in open(DATA / "opcodes.jsonl", encoding="utf-8"):
        have_ops.add(json.loads(line)["address"])

    # usable = has bytecode + cluster + valid CryptoScan analysis
    members = defaultdict(list)
    skipped_no_label = skipped_bad_analysis = 0
    for a in have_ops:
        if a not in cluster_of:
            continue
        row = labels.get(a)
        if row is None:
            skipped_no_label += 1
            continue
        if row.get("analysis_ok") != "1":
            skipped_bad_analysis += 1
            continue
        members[cluster_of[a]].append(a)

    clusters = sorted(members.items(), key=lambda kv: -len(kv[1]))
    rng = random.Random(SEED)

    total = sum(len(v) for v in members.values())
    counts = {s: 0 for s in TARGETS}
    assign = {}
    for cid, ads in clusters:
        # deficit = how far below target share each split currently is
        deficits = {s: TARGETS[s] - counts[s] / max(total, 1) for s in TARGETS}
        best = max(deficits.values())
        cands = [s for s, d in deficits.items() if abs(d - best) < 1e-12]
        s = rng.choice(cands)
        assign[cid] = s
        counts[s] += len(ads)

    with open(DATA / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["address", "cluster_id", "split"] + DEFECTS)
        for cid, ads in clusters:
            for a in ads:
                row = labels[a]
                w.writerow([a, cid, assign[cid]] + [row[d] for d in DEFECTS])

    print(f"usable contracts: {total:,} in {len(members):,} clusters "
          f"(skipped: {skipped_no_label} no-label, {skipped_bad_analysis} analysis-failed)")
    for s in TARGETS:
        print(f"  {s}: {counts[s]:,} ({counts[s]/total:.1%})")
    # per-class positives per split
    pos = defaultdict(lambda: defaultdict(int))
    for cid, ads in clusters:
        for a in ads:
            for d in DEFECTS:
                if labels[a][d] == "1":
                    pos[d][assign[cid]] += 1
    print("positives per class (train/val/test):")
    for d in DEFECTS:
        print(f"  {d:>4}: {pos[d]['train']:>6} / {pos[d]['val']:>5} / {pos[d]['test']:>5}")


if __name__ == "__main__":
    main()
