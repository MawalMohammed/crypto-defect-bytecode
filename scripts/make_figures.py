"""Render the 5 paper figures into results/figs/ (all from on-disk data, CPU-only).

fig1_clone_sizes.png     clone-cluster size distribution (log-log CCDF + hist)
fig2_auprc_prevalence.png AUPRC vs class prevalence scatter (with CIs)
fig3_representation.png   XGB vs CNN vs Transformer grouped bars
fig4_pr_curves.png        per-class precision-recall curves (recomputed on test)
fig5_ablation_heatmap.png imbalance ablation AUPRC heatmap
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve

BASE = Path(r"G:\Claude\new blockchain\crypto_defect_ml")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
DEFECTS = ["SSR", "CSR", "CCR", "SF", "SM", "ISV", "MR", "MF", "HC", "ES", "WR", "WRT"]
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight"})

# ---- static result tables (final numbers) ----
AUPRC = {"SM": .934, "CCR": .928, "CSR": .881, "WR": .857, "ISV": .818, "WRT": .625,
         "SF": .564, "SSR": .391, "HC": .239, "MR": .232, "MF": .225}
CI = {"SM": (.925, .940), "CCR": (.919, .937), "CSR": (.867, .893), "WR": (.839, .870),
      "ISV": (.777, .853), "WRT": (.596, .666), "SF": (.528, .623), "SSR": (.353, .436),
      "HC": (.183, .321), "MR": (.163, .300), "MF": (.129, .323)}
PREV = {"CCR": .1305, "SM": .1241, "CSR": .0742, "WR": .0567, "SF": .0472, "SSR": .0317,
        "WRT": .0290, "ISV": .0181, "HC": .0112, "MR": .0080, "MF": .0023}
REP = {  # class: (xgb, cnn, trf)
    "SM": (.933, .841, .664), "CCR": (.928, .863, .769), "CSR": (.882, .725, .493),
    "WR": (.857, .754, .278), "ISV": (.822, .744, .691), "WRT": (.627, .482, .216),
    "SF": (.567, .430, .190), "SSR": (.415, .265, .189), "MR": (.235, .120, .028),
    "MF": (.230, .064, .012), "HC": (.225, .092, .051)}
ABL_CLASSES = ["SSR", "SF", "MR", "MF", "HC"]
ABL_ARMS = ["baseline", "no-weight", "under-10", "under-3", "over-10", "focal"]
ABL = {  # class: [baseline, no_weight, under10, under3, over10, focal]
    "SSR": [.393, .424, .352, .379, .458, .449],
    "SF":  [.573, .564, .580, .529, .572, .561],
    "MR":  [.228, .209, .203, .130, .244, .215],
    "MF":  [.224, .176, .091, .043, .218, .217],
    "HC":  [.248, .219, .211, .173, .257, .243]}


def fig1_clone_sizes():
    sizes = []
    seen = set()
    for row in csv.DictReader(open(DATA / "clusters.csv", encoding="utf-8")):
        cid = row["cluster_id"]
        if cid not in seen:
            seen.add(cid)
            sizes.append(int(row["cluster_size"]))
    sizes = np.array(sizes)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts = Counter(sizes)
    xs = np.array(sorted(counts))
    ys = np.array([counts[x] for x in xs])
    ax[0].loglog(xs, ys, "o", ms=4, color="#1f77b4")
    ax[0].set_xlabel("cluster size (contracts)")
    ax[0].set_ylabel("number of clusters")
    ax[0].set_title("Clone-cluster size distribution")
    ax[0].grid(True, which="both", ls=":", alpha=.4)
    s = np.sort(sizes)[::-1]
    ccdf = np.arange(1, len(s) + 1) / len(s)
    ax[1].semilogx(s, 1 - ccdf + 1e-9, color="#d62728")
    ax[1].set_xlabel("cluster size (contracts)")
    ax[1].set_ylabel("fraction of clusters ≤ size")
    ax[1].set_title(f"{len(sizes):,} clusters; largest = {sizes.max():,}")
    ax[1].grid(True, which="both", ls=":", alpha=.4)
    fig.suptitle("67% of 79,286 contracts are clones — motivating cluster-level splits",
                 y=1.03, fontsize=12)
    fig.savefig(FIGS / "fig1_clone_sizes.png")
    plt.close(fig)
    print("fig1 done", flush=True)


def fig2_auprc_prevalence():
    cls = sorted(AUPRC, key=lambda c: PREV[c])
    x = np.array([PREV[c] for c in cls])
    y = np.array([AUPRC[c] for c in cls])
    lo = np.array([AUPRC[c] - CI[c][0] for c in cls])
    hi = np.array([CI[c][1] - AUPRC[c] for c in cls])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=7, color="#1f77b4",
                ecolor="#888", capsize=3, zorder=3)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, xs, "--", color="#d62728", alpha=.6, label="prevalence (random baseline)")
    for c in cls:
        ax.annotate(c, (PREV[c], AUPRC[c]), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("class prevalence (log scale)")
    ax.set_ylabel("test AUPRC (XGBoost, 3-seed mean ± 95% CI)")
    ax.set_title("Detectability rises with prevalence; rare classes remain signal-starved")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, ls=":", alpha=.4)
    fig.savefig(FIGS / "fig2_auprc_prevalence.png")
    plt.close(fig)
    print("fig2 done", flush=True)


def fig3_representation():
    cls = sorted(REP, key=lambda c: -REP[c][0])
    xgb_v = [REP[c][0] for c in cls]
    cnn_v = [REP[c][1] for c in cls]
    trf_v = [REP[c][2] for c in cls]
    x = np.arange(len(cls))
    w = .26
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w, xgb_v, w, label="XGBoost (0.611)", color="#1f77b4")
    ax.bar(x, cnn_v, w, label="CNN (0.489)", color="#ff7f0e")
    ax.bar(x + w, trf_v, w, label="Transformer (0.326)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(cls)
    ax.set_ylabel("test AUPRC")
    ax.set_title("Representation comparison — n-gram boosting wins every class")
    ax.legend(title="mean AUPRC")
    ax.grid(True, axis="y", ls=":", alpha=.4)
    fig.savefig(FIGS / "fig3_representation.png")
    plt.close(fig)
    print("fig3 done", flush=True)


def fig4_pr_curves():
    ops = {}
    for line in open(DATA / "opcodes.jsonl", encoding="utf-8"):
        d = json.loads(line)
        ops[d["address"]] = d["ops"]
    tr_txt, te_txt, te_addr = [], [], []
    with open(DATA / "dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["address"] not in ops:
            continue
        if r["split"] == "train":
            tr_txt.append(ops[r["address"]])
        elif r["split"] == "test":
            te_txt.append(ops[r["address"]])
            te_addr.append(r["address"])
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3),
                          max_features=40_000, sublinear_tf=True, min_df=10,
                          dtype=np.float32)
    vec.fit(np.array(tr_txt, dtype=object))
    Xte = vec.transform(np.array(te_txt, dtype=object))
    dte = xgb.DMatrix(Xte)
    yt = {r["address"]: r for r in rows}
    show = ["SM", "CCR", "CSR", "WR", "ISV", "WRT", "SF", "SSR"]
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(show):
        mp = RESULTS / "xgb_models" / f"{c}.json"
        if not mp.exists():
            continue
        bst = xgb.Booster()
        bst.load_model(str(mp))
        p = bst.predict(dte)
        y = np.array([int(yt[a][c]) for a in te_addr])
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=cmap(i), lw=1.6,
                label=f"{c} (AP {AUPRC.get(c, float('nan')):.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Per-class precision–recall (XGBoost, test split)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, ls=":", alpha=.4)
    fig.savefig(FIGS / "fig4_pr_curves.png")
    plt.close(fig)
    print("fig4 done", flush=True)


def fig5_ablation_heatmap():
    M = np.array([ABL[c] for c in ABL_CLASSES])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=.6)
    ax.set_xticks(range(len(ABL_ARMS)))
    ax.set_xticklabels(ABL_ARMS, rotation=20, ha="right")
    ax.set_yticks(range(len(ABL_CLASSES)))
    ax.set_yticklabels(ABL_CLASSES)
    for i in range(M.shape[0]):
        best = M[i].argmax()
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < .35 else "black",
                    fontweight="bold" if j == best else "normal", fontsize=9)
    ax.set_title("Imbalance ablation (test AUPRC) — no scheme rescues the tail")
    fig.colorbar(im, ax=ax, label="AUPRC")
    fig.savefig(FIGS / "fig5_ablation_heatmap.png")
    plt.close(fig)
    print("fig5 done", flush=True)


if __name__ == "__main__":
    fig1_clone_sizes()
    fig2_auprc_prevalence()
    fig3_representation()
    fig5_ablation_heatmap()
    fig4_pr_curves()  # last (heaviest: vectorizes test set)
    print("ALL FIGURES DONE", flush=True)
