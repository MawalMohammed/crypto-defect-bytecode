"""Annotate the RQ5 wild top-hits with Etherscan verification status.
Reads existing results/wild_scores.csv (no re-sampling). Key from env only.

Output: overwrites results/wild_top_report.md with a verif column, and prints
the headline unverified-rate among high-confidence flags.
"""
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

RESULTS = Path(r"G:\Claude\new blockchain\crypto_defect_ml\results")
CLASSES = ["WR", "SM", "CCR", "CSR", "ISV", "SSR"]
TOP_K = 20
KEY = os.environ.get("ETHERSCAN_API_KEY", "").strip()
assert KEY, "ETHERSCAN_API_KEY not set"

_cache = {}


def verif(addr):
    if addr in _cache:
        return _cache[addr]
    url = ("https://api.etherscan.io/v2/api?chainid=1&module=contract"
           f"&action=getsourcecode&address={addr}&apikey={KEY}")
    status = "unknown"
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                d = json.load(r)
            res = d.get("result")
            if isinstance(res, list) and res:
                status = "verified" if res[0].get("SourceCode") else "unverified"
                break
        except Exception:
            time.sleep(1.0)
        time.sleep(0.25)
    _cache[addr] = status
    return status


def main():
    rows = list(csv.DictReader(open(RESULTS / "wild_scores.csv", encoding="utf-8")))
    report = ["# RQ5: top wild-contract flags per class (with verification status)\n",
              f"Scored {len(rows):,} recent active mainnet contracts (not in corpus).\n"]
    hi_total = hi_unverified = 0
    for d in CLASSES:
        ranked = sorted(rows, key=lambda r: -float(r[d]))[:TOP_K]
        report.append(f"\n## {d} top-{TOP_K}\n")
        report.append("| address | score | code B | WR-rule | verif |")
        report.append("|---|---|---|---|---|")
        for r in ranked:
            a = r["address"]
            v = verif(a)
            sc = float(r[d])
            if sc >= 0.9:
                hi_total += 1
                if v == "unverified":
                    hi_unverified += 1
            report.append(f"| {a} | {sc:.3f} | {r['code_bytes']} | {r['wr_rule']} | {v} |")
            time.sleep(0.22)

    frac = (hi_unverified / hi_total) if hi_total else float("nan")
    headline = (f"\n**Headline:** among high-confidence (score>=0.9) flags shown above, "
                f"{hi_unverified}/{hi_total} = {frac:.0%} are UNVERIFIED on Etherscan "
                f"— contracts no source-level analyzer (incl. CryptoScan) can audit.")
    report.append(headline)
    (RESULTS / "wild_top_report.md").write_text("\n".join(report), encoding="utf-8")
    print(headline, flush=True)
    print(f"wrote {RESULTS / 'wild_top_report.md'}", flush=True)


if __name__ == "__main__":
    main()
