#!/usr/bin/env python3
"""Train the bundled vocalness model from album-level weak labels.

Training pool: whole albums/playlists whose vocal-presence label is uniform
(orchestral/easy-listening = instrumental; pop/rock/crooner/opera = vocal).
Any file that also resolves from the curated eval set
(tests/reference_data/vocal_labels.csv) is EXCLUDED from training, so
tests/python/test_vocalness_real.py --model <out> is a fully held-out gate.

Usage: python dev-docs/bench/scripts/train_vocalness.py out_model.json
"""

import csv
import os
import sys

import numpy as np

import sonara
from sonara import vocal_model

MUSIC = "/Volumes/EksternalHome/Downloads/Music"
APPS = "/Volumes/EksternalHome/Downloads/Apps/Music"
GH2 = os.path.join(MUSIC, "Greatest Hits - Collection 1958-2021 _1074 ALBUMS_ MP3 Part 2 Of 2-TL")

INSTRUMENTAL_DIRS = [
    os.path.join(GH2, "2003 - GREATEST HITS • Herbert von Karajan - Karajan Forever The Greatest Classical Hits [Russia] [2CD]"),
    os.path.join(GH2, "2007 - GREATEST HITS • Paul Mauriat & His Orchestra - Star Mark Greatest Hits [Russia] [2CD]"),
    os.path.join(GH2, "2016 - GREATEST HITS • Royal Philharmonic Orchestra - Symphonic Queen The Greatest Hits [EU]"),
    # solo-instrument / smooth-jazz / electronic instrumentals (the FP class
    # missing from the first training round):
    os.path.join(GH2, "2006 - GREATEST HITS • Kenny G - Greatest Hits [Russia] [2CD]"),
    os.path.join(GH2, "2021 - GREATEST HITS • DJ BoBo - Greatest Hits New Versions [Instrumentals] [Switserland]"),
    os.path.join(GH2, "2001 - GREATEST HITS • Classic Dream Orchestra - Greatest Hits Go Classic Phil Collins [EU]"),
    # solo melodic leads (sax/piano/guitar/new-age) — the residual FP class:
    os.path.join(GH2, "2003 - GREATEST HITS • Fausto Papetti - 50 Greatest Hits [Russia] [2CD]"),
    os.path.join(GH2, "2004 - GREATEST HITS • Jim Brickman - Greatest Hits [Canada]"),
    os.path.join(GH2, "2008 - GREATEST HITS • Armik - Star Mark Greatest Hits [Russia] [2CD]"),
    os.path.join(GH2, "2017 - GREATEST HITS • Chris Spheeris & Paul Voudouris - Greatest Hits & Unreleased Masters [US]"),
    os.path.join(GH2, "1998 - GREATEST HITS • Greatest New Age Hits Vol. II [EU]"),
    os.path.join(APPS, "42 Instrumentals"),
]
VOCAL_DIRS = [
    os.path.join(APPS, "11 Workout Energy"),
    os.path.join(APPS, "16 British Invasion"),
    os.path.join(APPS, "17 Classic Rock Hits"),
    # acoustic folk / country / ballads (the reported-FN class):
    os.path.join(APPS, "38 Songwriter & Folk"),
    os.path.join(APPS, "28 Country Roads"),
    os.path.join(APPS, "23 Power Ballads"),
    os.path.join(GH2, "2012 - GREATEST HITS • Andy Williams - Greatest Hits [Russia] [2CD]"),
    # crooner vocals over lush orchestral backing (the boundary the
    # easy-listening instrumentals push against):
    os.path.join(GH2, "2009 - GREATEST HITS • Paul Anka - Greatest Hits [Russia] [2CD]"),
    os.path.join(GH2, "2006 - GREATEST HITS • Tony Bennett - Greatest Hits of the '50s [US]"),
    os.path.join(GH2, "2006 - GREATEST HITS • Tony Bennett - Greatest Hits of the '60s [US]"),
    os.path.join(GH2, "2019 - GREATEST HITS • Luciano Pavarotti - The Greatest Hits [EU]"),
    os.path.join(GH2, "1997 - GREATEST HITS • The Barry Sisters - Their Greatest Yiddish Hits [US] [2CD]"),
]
LABELS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "tests", "reference_data", "vocal_labels.csv")
EXTS = (".mp3", ".flac", ".m4a", ".ogg", ".wav")


def files_under(d):
    out = []
    for dirpath, _dirs, names in os.walk(d):
        out.extend(os.path.join(dirpath, n) for n in names
                   if n.lower().endswith(EXTS))
    return sorted(out)


def eval_set_paths():
    """Paths the curated eval harness resolves — excluded from training."""
    all_files = files_under(MUSIC) + files_under(APPS)
    lower = [(f, f.lower()) for f in all_files]
    out = set()
    with open(LABELS_CSV, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().startswith("#"):
                continue
            pat = ",".join(row[1:]).strip().lower()
            out.update([f for f, fl in lower if pat in fl][:3])
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "vocalness_model.json"
    excluded = eval_set_paths()
    print(f"eval-set exclusion list: {len(excluded)} paths")

    pool = []  # (path, is_vocal)
    for d in INSTRUMENTAL_DIRS:
        fs = [f for f in files_under(d) if f not in excluded]
        print(f"  instrumental {len(fs):4d}  {os.path.basename(d)}")
        pool += [(f, False) for f in fs]
    for d in VOCAL_DIRS:
        fs = [f for f in files_under(d) if f not in excluded]
        print(f"  vocal        {len(fs):4d}  {os.path.basename(d)}")
        pool += [(f, True) for f in fs]

    paths = [p for p, _ in pool]
    print(f"training pool: {len(paths)} files; extracting embeddings ...")
    res = sonara.analyze_batch(paths, features=["embedding"])
    X, y, skipped = [], [], 0
    for (p, lab), r in zip(pool, res):
        emb = None if r.failed else r.get("embedding")
        if emb is None:
            skipped += 1
            continue
        X.append(emb)
        y.append(lab)
    print(f"embeddings: {len(X)} ok, {skipped} skipped "
          f"({sum(y)} vocal / {len(y) - sum(y)} instrumental)")

    # 5-fold CV over the training pool for an internal sanity number.
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_arr))
    folds = np.array_split(idx, 5)
    aucs = []
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(idx, test_idx)
        m = vocal_model.train(X_arr[train_idx], y_arr[train_idx],
                              model_id=f"cv-{k}", hidden=32, epochs=3500)
        s = np.array([m.predict_vocalness(r) for r in X_arr[test_idx]])
        t = y_arr[test_idx]
        pos, neg = s[t], s[~t]
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        aucs.append(wins / (len(pos) * len(neg)))
    print(f"train-pool 5-fold CV AUC: {np.mean(aucs):.4f} "
          f"(folds: {', '.join(f'{a:.3f}' for a in aucs)})")

    model = vocal_model.train(X_arr, y_arr, model_id="sonara-vocalness-v1",
                              hidden=32, epochs=3500)
    model.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
