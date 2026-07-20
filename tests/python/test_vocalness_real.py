#!/usr/bin/env python3
"""Real-music vocal-presence gate (local-dataset tier, like test_tonal_batch).

Evaluates the `vocalness` score against the labeled set in
tests/reference_data/vocal_labels.csv, resolved against the local music
library. Reports AUC, per-class stats, and the downstream curation gate
(vocal tracks wrongly below 0.35 / instrumentals wrongly above it).

Usage:
    python tests/python/test_vocalness_real.py [--model path.json] [--gate]

Without --model the built-in heuristic is scored (baseline). With --model
the vocalness model is applied. --gate additionally enforces the release
criteria (used by the phased-plan fidelity gate):
  * every labeled pattern and analysis result is present
  * instrumental mean stays < vocal mean with AUC >= 0.94
  * <= 10% of labeled vocal tracks below the 0.35 curation threshold
The content-addressed exact regressions live in test_vocalness_frozen.py.
Under --gate, absent local data or incomplete results fail closed.
"""

import csv
import os
import sys

import sonara

ROOTS = [
    "/Volumes/EksternalHome/Downloads/Music",
    "/Volumes/EksternalHome/Downloads/Apps/Music",
]
LABELS = os.path.join(
    os.path.dirname(__file__), "..", "reference_data", "vocal_labels.csv"
)
MAX_MATCHES_PER_PATTERN = 3
CURATION_THRESHOLD = 0.35  # sonagram's max_vocalness focus gate


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
            # Only strip the trailing side: a deliberate leading space in a
            # pattern (e.g. " Mother Nature's Son") excludes hyphen-joined
            # false matches like "Narrative-Mother Nature's Son".
            label, pattern = row[0].strip(), ",".join(row[1:]).rstrip()
            pat = pattern.lower()
            hits = [f for f, fl in lower if pat in fl][:MAX_MATCHES_PER_PATTERN]
            if not hits:
                missing.append(pattern)
            for h in hits:
                resolved.append((h, label, pattern))
    return resolved, missing


def auc(scores_pos, scores_neg):
    """Rank-based AUC: P(vocal score > instrumental score)."""
    wins = ties = 0
    for p in scores_pos:
        for n in scores_neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    total = len(scores_pos) * len(scores_neg)
    return (wins + 0.5 * ties) / total if total else float("nan")


def main():
    model = None
    gate = "--gate" in sys.argv
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    if not all(os.path.isdir(r) for r in ROOTS):
        print("FAIL: local dataset roots not present" if gate
              else "SKIP: local dataset roots not present")
        return 1 if gate else 0

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
    if len(missing) > 0.2 * n_patterns:
        print("FAIL: too many unresolved patterns")
        return 1

    paths = [p for p, _, _ in resolved]
    kwargs = {"features": ["vocalness"]}
    if model:
        kwargs["vocalness_model"] = model
    results = sonara.analyze_batch(paths, **kwargs)

    rows = []
    incomplete = False
    for (path, label, pattern), res in zip(resolved, results):
        if res.failed:
            print(f"  analyze failed: {os.path.basename(path)}: "
                  f"{res.get('error', '')[:80]}")
            incomplete = True
            continue
        v = res.get("vocalness")
        if v is None:
            print(f"  no vocalness: {os.path.basename(path)}")
            incomplete = True
            continue
        rows.append((path, label, pattern, float(v)))

    if gate and (incomplete or len(rows) != len(resolved)):
        print("FAIL: fidelity gate requires one score for every resolved track")
        return 1

    voc = [v for _, l, _, v in rows if l == "vocal"]
    ins = [v for _, l, _, v in rows if l == "instrumental"]
    if not voc or not ins:
        print("FAIL: both vocal and instrumental results are required")
        return 1
    a = auc(voc, ins)

    def stats(xs):
        xs = sorted(xs)
        return (f"n={len(xs)} mean={sum(xs)/len(xs):.3f} "
                f"median={xs[len(xs)//2]:.3f} min={xs[0]:.3f} max={xs[-1]:.3f}")

    print(f"\nvocal:        {stats(voc)}")
    print(f"instrumental: {stats(ins)}")
    print(f"AUC(vocal > instrumental) = {a:.4f}")

    fn = sum(1 for v in voc if v < CURATION_THRESHOLD)
    fp = sum(1 for v in ins if v >= CURATION_THRESHOLD)
    print(f"vocal below {CURATION_THRESHOLD} (focus-gate FNs): {fn}/{len(voc)}"
          f" ({100*fn/len(voc):.0f}%)")
    print(f"instrumental at/above {CURATION_THRESHOLD}: {fp}/{len(ins)}"
          f" ({100*fp/len(ins):.0f}%)")

    if "--dump" in sys.argv:
        print("\nmisclassified (vocal < 0.35 or instrumental >= 0.35):")
        for p, l, _pat, v in sorted(rows, key=lambda r: r[3]):
            if (l == "vocal" and v < CURATION_THRESHOLD) or (
                    l == "instrumental" and v >= CURATION_THRESHOLD):
                print(f"  {v:.3f}  {l:<12}  {os.path.basename(p)}")

    if gate:
        ok = True
        if not (a >= 0.94):
            print("GATE FAIL: AUC < 0.94")
            ok = False
        if fn > 0.10 * len(voc):
            print("GATE FAIL: >10% of vocal tracks below curation threshold")
            ok = False
        print("GATE PASS" if ok else "GATE FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
