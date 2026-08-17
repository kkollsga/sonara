#!/usr/bin/env python3
"""P6 genre feasibility — step 4: fresh default-path embeddings.

Stratified subsample of pool.jsonl (up to 100 train + 50 eval per class,
seed 42), re-analyzed at sonara's default path (22050/compact) with
features=["embedding"]. Verifies embedding_version == 2 and the default-path
provenance on the first result. Kills the 44100/playlist transfer trap.

Output: dev-docs/bench/out/genre-feasibility/fresh.jsonl
  rows: {hash, path, artist, label, split, emb, sr, mode, emb_v}

Usage: fresh_embed.py [--limit-per-class-train N] [--limit-per-class-eval N]
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np

import sonara

MUSIC = "/Volumes/EksternalHome/Downloads/Music"
OUTDIR = ("/Volumes/EksternalHome/Koding/Rust/sonara/dev-docs/bench/out/"
          "genre-feasibility")
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-class-train", type=int, default=100)
    ap.add_argument("--limit-per-class-eval", type=int, default=50)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{OUTDIR}/pool.jsonl",
                                        encoding="utf-8")]
    rng = np.random.default_rng(SEED)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["split"])].append(r)

    chosen = []
    for (label, split), items in sorted(groups.items()):
        lim = (args.limit_per_class_train if split == "train"
               else args.limit_per_class_eval)
        items = sorted(items, key=lambda r: r["hash"])
        if len(items) > lim:
            idx = rng.choice(len(items), size=lim, replace=False)
            items = [items[i] for i in sorted(idx)]
        chosen.extend(items)
        print(f"  {label:18s} {split:5s} -> {len(items)} files")
    print(f"analyzing {len(chosen)} files at the default path ...")

    paths = [f"{MUSIC}/{r['path']}" for r in chosen]
    t0 = time.time()
    res = sonara.analyze_batch(paths, features=["embedding"])
    dt = time.time() - t0
    print(f"analyze_batch: {dt:.1f}s ({dt/len(paths):.2f}s/file)")

    n_ok = n_fail = 0
    checked = False
    with open(f"{OUTDIR}/fresh.jsonl", "w", encoding="utf-8") as out:
        for r, a in zip(chosen, res):
            if a.failed:
                n_fail += 1
                continue
            emb = a.get("embedding")
            if emb is None:
                n_fail += 1
                continue
            emb_v = a.get("embedding_version")
            prov = a.get("provenance") or {}
            sr = prov.get("sample_rate")
            mode = prov.get("mode")
            if not checked:
                checked = True
                print(f"  first result provenance: sr={sr} mode={mode} "
                      f"embedding_version={emb_v}")
                assert emb_v == 2, f"embedding_version {emb_v} != 2"
                assert sr == 22050, f"sample_rate {sr} != 22050 default"
            out.write(json.dumps({
                "hash": r["hash"], "path": r["path"], "artist": r["artist"],
                "label": r["label"], "split": r["split"],
                "emb": list(emb), "sr": sr, "mode": mode, "emb_v": emb_v,
            }) + "\n")
            n_ok += 1
    print(f"fresh embeddings: {n_ok} ok, {n_fail} failed "
          f"-> {OUTDIR}/fresh.jsonl")


if __name__ == "__main__":
    main()
