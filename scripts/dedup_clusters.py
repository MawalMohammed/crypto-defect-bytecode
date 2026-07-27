"""Cluster contracts into clone groups for leakage-free splits.

Memory-safe + deterministic rewrite:
  - shingle hashes computed vectorized in numpy (uint64), stored as np.unique
    arrays (8 B/element, ~1 GB total) instead of Python sets (10+ GB -> OOM)
  - token codes from blake2b (stable across runs, unlike built-in hash())
  - exact Jaccard verification via np.intersect1d on sorted unique arrays

Levels:
  1. exact: identical sha256 of metadata-stripped runtime bytecode
  2. near : MinHash (128 perms) over opcode 4-gram shingles + LSH banding
            (32 bands x 4 rows); candidates verified with true Jaccard >= 0.9.

Output: data/clusters.csv  (address, code_hash, cluster_id, cluster_size)
"""
import csv
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(r"G:\Claude\new blockchain\crypto_defect_ml\data")
NUM_PERM = 128
BANDS, ROWS = 32, 4
JACCARD_THRESHOLD = 0.9

U64 = np.uint64
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
P1, P2, P3, P4 = (U64(0x9E3779B97F4A7C15), U64(0xC2B2AE3D27D4EB4F),
                  U64(0x165667B19E3779F9), U64(0x27D4EB2F165667C5))

SEEDS = np.array(
    [struct.unpack("<Q", hashlib.sha256(f"perm{i}".encode()).digest()[:8])[0]
     for i in range(NUM_PERM)],
    dtype=np.uint64,
)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


_token_code = {}


def token_code(tok):
    c = _token_code.get(tok)
    if c is None:
        c = struct.unpack("<Q", hashlib.blake2b(tok.encode(), digest_size=8).digest())[0]
        _token_code[tok] = c
    return c


def rot(x, k):
    k = U64(k)
    return ((x << k) | (x >> (U64(64) - k))) & MASK


def shingle_array(ops):
    """Deterministic 4-gram shingle hashes as sorted-unique uint64 array."""
    codes = np.fromiter((token_code(t) for t in ops.split()), dtype=np.uint64)
    if codes.size < 4:
        codes = np.pad(codes, (0, 4 - codes.size), constant_values=token_code("<PAD>"))
    with np.errstate(over="ignore"):
        h = (codes[:-3] * P1) ^ (rot(codes[1:-2], 17) * P2) \
            ^ (rot(codes[2:-1], 31) * P3) ^ (rot(codes[3:], 47) * P4)
    return np.unique(h)


def minhash(arr):
    with np.errstate(over="ignore"):
        mixed = (arr[:, None] ^ SEEDS[None, :]) * P1
    return mixed.min(axis=0)


def jaccard(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def main():
    reps = {}
    order = []
    for line in open(DATA / "opcodes.jsonl", encoding="utf-8"):
        d = json.loads(line)
        h = d["code_hash"]
        if h not in reps:
            reps[h] = [d["ops"], []]
            order.append(h)
        reps[h][1].append(d["address"])
    total_contracts = sum(len(v[1]) for v in reps.values())
    print(f"contracts: {total_contracts:,} | exact-unique: {len(order):,}", flush=True)

    shingle_sets = []
    sigs = np.empty((len(order), NUM_PERM), dtype=np.uint64)
    for k, h in enumerate(order):
        arr = shingle_array(reps[h][0])
        reps[h][0] = None  # free the ops string
        shingle_sets.append(arr)
        sigs[k] = minhash(arr)
        if (k + 1) % 5000 == 0:
            print(f"  minhash {k + 1:,}/{len(order):,}", flush=True)

    dsu = DSU(len(order))
    pairs_checked = merged = 0
    for b in range(BANDS):
        buckets = defaultdict(list)
        lo = b * ROWS
        for idx in range(len(order)):
            buckets[sigs[idx, lo:lo + ROWS].tobytes()].append(idx)
        for members in buckets.values():
            if len(members) < 2:
                continue
            anchor = members[0]
            for other in members[1:]:
                if dsu.find(anchor) == dsu.find(other):
                    continue
                pairs_checked += 1
                if jaccard(shingle_sets[anchor], shingle_sets[other]) >= JACCARD_THRESHOLD:
                    dsu.union(anchor, other)
                    merged += 1
        print(f"  band {b + 1}/{BANDS} done (pairs so far {pairs_checked:,}, merges {merged:,})", flush=True)

    cluster_of = {h: dsu.find(idx) for idx, h in enumerate(order)}
    sizes = defaultdict(int)
    for h, (_, ads) in reps.items():
        sizes[cluster_of[h]] += len(ads)

    with open(DATA / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["address", "code_hash", "cluster_id", "cluster_size"])
        for h in order:
            cid = cluster_of[h]
            for a in reps[h][1]:
                w.writerow([a, h, cid, sizes[cid]])

    n_clusters = len(set(cluster_of.values()))
    print(f"clusters: {n_clusters:,} (from {len(order):,} exact-unique) | "
          f"biggest cluster: {max(sizes.values()):,} contracts", flush=True)


if __name__ == "__main__":
    main()
