#!/usr/bin/env python3
"""Real-music similarity neighbor-quality gate (local-dataset tier).

Evaluates the similarity profiles' style coherence against the labeled set in
tests/reference_data/similarity_labels.csv, resolved against the local music
library. For every labeled track it ranks all other labeled tracks by each
profile's embedding distance and reports neighbor-precision@k: the fraction of
the k nearest neighbors sharing the track's coarse style label (k=5, 10),
per label and macro-averaged.

Usage:
    python tests/python/test_similarity_real.py [--gate] [--dump]
        [--cache FILE]

--cache stores computed embeddings keyed by path+mtime (local convenience for
repeated runs; embeddings are deterministic for an unchanged file). --gate
enforces the release criteria (used by the phased-plan fidelity gate):
  * the dataset root and every labeled pattern must resolve (fail closed,
    never silently shrink)
  * the timbre profile beats the default profile on macro precision@10 by at
    least TIMBRE_MARGIN (the profile's reason to exist: style-coherent
    neighbors)
  * the default profile's macro precision@10 matches its pinned baseline
    (the default table is untouched by construction; this catches drift)
The audio-free exact regressions live in test_similarity_frozen.py.
"""

import csv
import json
import os
import sys

import sonara

ROOTS = [
    "/Volumes/EksternalHome/Downloads/Music",
]
LABELS = os.path.join(
    os.path.dirname(__file__), "..", "reference_data", "similarity_labels.csv"
)
MAX_MATCHES_PER_PATTERN = 3
KS = (5, 10)

# --gate criteria. Baselines measured 2026-08-17 over the 210-track curated
# set (30 per style, <=2 per artist): default macro P@10 = 0.2124, timbre
# macro P@10 = 0.2310 (chance ~= 0.139 with 7 balanced styles). The observed
# timbre delta is +0.0186 with a stratified paired-bootstrap 95% CI of
# [+0.0048, +0.0329], P(delta<=0) = 0.0046 — a real improvement, not noise.
# The margin is set at the CI lower bound: any future timbre table must keep
# a statistically defensible advantage, with 0.014 headroom over v1.
DEFAULT_BASELINE_P10 = 0.2124
DEFAULT_BASELINE_TOL = 0.02
TIMBRE_MARGIN = 0.005


def list_audio(roots):
    exts = (".mp3", ".flac", ".m4a", ".ogg", ".wav")
    files = []
    for root in roots:
        for dirpath, _dirs, names in os.walk(root):
            for n in names:
                if n.lower().endswith(exts):
                    files.append(os.path.join(dirpath, n))
    return sorted(files)


def resolve(labels_csv, files):
    lower = [(f, f.lower()) for f in files]
    resolved = []  # (path, label, pattern)
    missing = []
    with open(labels_csv, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().startswith("#"):
                continue
            label, pattern = row[0].strip(), ",".join(row[1:]).rstrip()
            pat = pattern.lower()
            hits = [f for f, fl in lower if pat in fl][:MAX_MATCHES_PER_PATTERN]
            if not hits:
                missing.append(pattern)
            for h in hits:
                resolved.append((h, label, pattern))
    return resolved, missing


def embeddings_for(paths, cache_file):
    cache = {}
    if cache_file and os.path.isfile(cache_file):
        with open(cache_file, encoding="utf-8") as fh:
            cache = json.load(fh)
    out = {}
    todo = []
    for p in paths:
        entry = cache.get(p)
        if entry and entry.get("mtime") == os.path.getmtime(p):
            out[p] = entry["emb"]
        else:
            todo.append(p)
    if todo:
        print(f"analyzing {len(todo)} tracks (features=['embedding']) ...")
        results = sonara.analyze_batch(todo, features=["embedding"])
        for p, res in zip(todo, results):
            if res.failed:
                print(f"  analyze failed: {os.path.basename(p)}: "
                      f"{res.get('error', '')[:80]}")
                continue
            emb = res.get("embedding")
            if emb is None:
                print(f"  no embedding: {os.path.basename(p)}")
                continue
            out[p] = [float(v) for v in emb]
            cache[p] = {"mtime": os.path.getmtime(p), "emb": out[p]}
        if cache_file:
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
    return out


def neighbor_precision(rows, embs, profile):
    """Per-track precision@k for every k in KS under one profile."""
    n = len(rows)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ei = embs[rows[i][0]]
        for j in range(i + 1, n):
            d = sonara.embedding_distance(ei, embs[rows[j][0]],
                                          profile=profile)
            dist[i][j] = dist[j][i] = d
    per_track = []  # (label, {k: precision})
    for i in range(n):
        order = sorted((j for j in range(n) if j != i),
                       key=lambda j: (dist[i][j], rows[j][0]))
        label = rows[i][1]
        precs = {}
        for k in KS:
            top = order[:k]
            precs[k] = sum(1 for j in top if rows[j][1] == label) / k
        per_track.append((label, precs))
    return per_track


def summarize(per_track):
    labels = sorted({label for label, _ in per_track})
    table = {}
    for label in labels:
        rows = [precs for lbl, precs in per_track if lbl == label]
        table[label] = {
            k: sum(r[k] for r in rows) / len(rows) for k in KS
        }
    macro = {k: sum(table[label][k] for label in labels) / len(labels)
             for k in KS}
    return table, macro


def main():
    gate = "--gate" in sys.argv
    cache_file = None
    if "--cache" in sys.argv:
        cache_file = sys.argv[sys.argv.index("--cache") + 1]

    missing_roots = [r for r in ROOTS if not os.path.isdir(r)]
    if missing_roots:
        print(f"FAIL: local dataset roots not present: {missing_roots}")
        return 1

    files = list_audio(ROOTS)
    resolved, missing = resolve(LABELS, files)
    n_patterns = len({p for _, _, p in resolved}) + len(missing)
    print(f"library files: {len(files)}; patterns resolved: "
          f"{n_patterns - len(missing)}/{n_patterns}; tracks: {len(resolved)}")
    for m in missing:
        print(f"  unresolved: {m}")
    if missing and gate:
        print("FAIL: fidelity gate requires every labeled pattern")
        return 1
    if len(missing) > 0.05 * n_patterns:
        print("FAIL: too many unresolved patterns")
        return 1

    paths = [p for p, _, _ in resolved]
    embs = embeddings_for(paths, cache_file)
    rows = [(p, label, pattern) for p, label, pattern in resolved
            if p in embs]
    if len(rows) != len(resolved):
        print(f"FAIL: embeddings for {len(rows)}/{len(resolved)} resolved "
              "tracks; the labeled set must not silently shrink")
        return 1

    profiles = sorted(sonara.SIMILARITY_PROFILES)
    macros = {}
    tables = {}
    for profile in profiles:
        per_track = neighbor_precision(rows, embs, profile)
        tables[profile], macros[profile] = summarize(per_track)

    header = "label".ljust(18) + "".join(
        f"{p}@{k}".rjust(13) for p in profiles for k in KS
    )
    print("\nneighbor-precision (mean over tracks):")
    print(header)
    for label in sorted(tables[profiles[0]]):
        cells = "".join(
            f"{tables[p][label][k]:13.4f}" for p in profiles for k in KS
        )
        print(label.ljust(18) + cells)
    macro_cells = "".join(
        f"{macros[p][k]:13.4f}" for p in profiles for k in KS
    )
    print("macro".ljust(18) + macro_cells)

    if "--dump" in sys.argv:
        print("\nper-profile macro summary:")
        for p in profiles:
            for k in KS:
                print(f"  {p:8s} P@{k:<3d} {macros[p][k]:.4f}")

    if gate:
        ok = True
        d10, t10 = macros["default"][10], macros["timbre"][10]
        if abs(d10 - DEFAULT_BASELINE_P10) > DEFAULT_BASELINE_TOL:
            print(f"GATE FAIL: default macro P@10 {d10:.4f} drifted from "
                  f"pinned {DEFAULT_BASELINE_P10:.4f} "
                  f"(+/- {DEFAULT_BASELINE_TOL})")
            ok = False
        if t10 < d10 + TIMBRE_MARGIN:
            print(f"GATE FAIL: timbre macro P@10 {t10:.4f} does not beat "
                  f"default {d10:.4f} by margin {TIMBRE_MARGIN}")
            ok = False
        print("GATE PASS" if ok else "GATE FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
