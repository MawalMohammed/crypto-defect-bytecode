"""Fetch deployed runtime bytecode for the CryptoScan corpus via free public JSON-RPC.

Zero-cost: uses public Ethereum RPC endpoints (no API key), batched eth_getCode
calls, polite rate, endpoint rotation on failure, resume-safe.

Input : CryptoScan_repo/Dataset/contract-list.txt  (79,598 addresses)
Output: crypto_defect_ml/data/bytecode.jsonl  (one line per address:
        {"address": ..., "code": "0x..."} ; code == "0x" means self-destructed /
        no code at latest block -- kept, useful to report)

Usage:  python fetch_bytecode.py [--limit N]
Re-running resumes where it left off.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"G:\Claude\new blockchain")
LIST_FILE = ROOT / "CryptoScan_repo" / "Dataset" / "contract-list.txt"
OUT_FILE = ROOT / "crypto_defect_ml" / "data" / "bytecode.jsonl"

ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://eth.llamarpc.com",
    "https://eth.drpc.org",
    "https://eth.merkle.io",
    "https://rpc.flashbots.net",
    "https://eth-mainnet.public.blastapi.io",
    "https://1rpc.io/eth",
    "https://cloudflare-eth.com",
]

BATCH_SIZE = 10          # addresses per JSON-RPC batch request (gentle)
SLEEP_BETWEEN = 1.0      # seconds between batches (gentle: ~10 addr/s max)
MAX_RETRIES = 16         # per batch, rotating endpoints with backoff
COOLDOWNS = [90, 180, 300, 600, 600]  # after all-endpoint failure, wait then retry batch


def rpc_batch(endpoint, addrs):
    payload = [
        {"jsonrpc": "2.0", "id": i, "method": "eth_getCode", "params": [a, "latest"]}
        for i, a in enumerate(addrs)
    ]
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "research-fetch/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    if isinstance(resp, dict):  # endpoint rejected batching -> treat as failure
        raise RuntimeError(f"non-batch response: {str(resp)[:200]}")
    out = {}
    for item in resp:
        if "result" not in item:
            raise RuntimeError(f"rpc error: {str(item)[:200]}")
        out[addrs[item["id"]]] = item["result"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N new fetches (for testing)")
    args = ap.parse_args()

    addrs = [l.strip().lower() for l in open(LIST_FILE) if l.strip()]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["address"])
                except Exception:
                    pass
    todo = [a for a in addrs if a not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"total {len(addrs):,} | done {len(done):,} | fetching {len(todo):,}")

    ep_idx = 0
    fetched = 0
    t0 = time.time()
    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i : i + BATCH_SIZE]
            # sticky endpoint: keep using the last one that worked; rotate only on failure
            results = None
            for cooldown in [0] + COOLDOWNS:
                if cooldown:
                    print(f"  all endpoints failing; cooling down {cooldown}s ...")
                    time.sleep(cooldown)
                for attempt in range(MAX_RETRIES):
                    ep = ENDPOINTS[ep_idx % len(ENDPOINTS)]
                    try:
                        results = rpc_batch(ep, batch)
                        break
                    except Exception as e:
                        print(f"  [{ep}] failed ({str(e)[:100]}), rotating")
                        ep_idx += 1
                        time.sleep(min(2 ** attempt, 45))
                if results is not None:
                    break
            if results is None:
                print(f"batch at offset {i} failed despite cooldowns; stopping (re-run to resume)")
                return
            for a in batch:
                fout.write(json.dumps({"address": a, "code": results[a]}) + "\n")
            fout.flush()
            fetched += len(batch)
            if fetched % 1000 < BATCH_SIZE:
                rate = fetched / max(time.time() - t0, 1)
                eta_h = (len(todo) - fetched) / max(rate, 0.1) / 3600
                print(f"  {fetched:,}/{len(todo):,}  ({rate:.0f} addr/s, eta {eta_h:.1f} h)")
            time.sleep(SLEEP_BETWEEN)
    print(f"finished: {fetched:,} new records -> {OUT_FILE}")


if __name__ == "__main__":
    main()
