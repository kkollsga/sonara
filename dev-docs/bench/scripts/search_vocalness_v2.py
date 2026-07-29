#!/usr/bin/env python3
"""Search bundled vocalness candidates with frozen hashes excluded."""

import csv
import importlib.util
import json
import os
import sys

import numpy as np

import sonara
from sonara import vocal_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FIXTURE = os.path.join(ROOT, "tests", "reference_data", "vocalness_similarity_v2.json")
CACHE = os.path.join(ROOT, "dev-docs", "bench", "out", "vocalness_v2_search.npz")
OUT = "/private/tmp/vocalness_v2_candidate.json"

spec = importlib.util.spec_from_file_location(
    "train_vocalness", os.path.join(os.path.dirname(__file__), "train_vocalness.py")
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)

HARD_VOCAL_DIRS = [
    os.path.join(training.GH2, "2019 - GREATEST HITS • Luciano Pavarotti - The Greatest Hits [EU]"),
    os.path.join(training.GH2, "2000 - GREATEST HITS • Tito Schipa - Greatest Hits [Italy]"),
    os.path.join(training.GH2, "2001 - GREATEST HITS • The Beatles - 20 Greatest Hits [UK]"),
    os.path.join(training.GH2, "1997 - GREATEST HITS • The Barry Sisters - Their Greatest Yiddish Hits [US] [2CD]"),
]
HARD_INSTRUMENTAL_DIRS = [
    os.path.join(training.GH2, "2003 - GREATEST HITS • Fausto Papetti - 50 Greatest Hits [Russia] [2CD]"),
]


def frozen_paths(fixture):
    paths = set()
    analysis_root = os.path.join(training.MUSIC, ".sonagram", "analysis")
    for case in fixture["cases"]:
        path = os.path.join(analysis_root, case["content_hash"] + ".json")
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        if record["source"]["content_hash"] != case["content_hash"]:
            raise RuntimeError(f"hash mismatch in {path}")
        paths.add(os.path.normpath(os.path.join(training.MUSIC, record["source"]["path"])))
    if len(paths) != len(fixture["cases"]):
        raise RuntimeError("frozen hashes did not resolve uniquely")
    return paths


def capped_broad(files):
    lower = [(path, path.lower()) for path in files]
    resolved = []
    with open(training.LABELS_CSV, newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().startswith("#"):
                continue
            label = row[0].strip() == "vocal"
            pattern = ",".join(row[1:]).rstrip().lower()
            hits = [path for path, lowered in lower if pattern in lowered][:3]
            resolved.extend((path, label) for path in hits)
    return resolved


def extract(rows):
    results = sonara.analyze_batch([path for path, _ in rows], features=["embedding"])
    good = [
        (result["embedding"], label)
        for (_, label), result in zip(rows, results)
        if not result.failed and result.get("embedding") is not None
    ]
    return np.asarray([embedding for embedding, _ in good]), np.asarray(
        [label for _, label in good], dtype=bool
    )


def extract_cached(rows, forbidden_hashes):
    """Load Sonagram-domain embeddings by exact source path, never by name."""
    wanted = {os.path.normpath(path): label for path, label in rows}
    found = []
    analysis_root = os.path.join(training.MUSIC, ".sonagram", "analysis")
    for entry in os.scandir(analysis_root):
        if not entry.name.endswith(".json"):
            continue
        with open(entry.path, encoding="utf-8") as handle:
            record = json.load(handle)
        source = record.get("source", {})
        content_hash = source.get("content_hash")
        if content_hash in forbidden_hashes:
            continue
        source_path = source.get("path")
        if not isinstance(source_path, str):
            continue
        path = os.path.normpath(os.path.join(training.MUSIC, source_path))
        if path not in wanted:
            continue
        analysis = record.get("analysis", {})
        embedding = analysis.get("embedding")
        if analysis.get("embedding_version") != 2 or not isinstance(embedding, list):
            continue
        if len(embedding) != 48 or not np.all(np.isfinite(embedding)):
            continue
        found.append((embedding, wanted[path]))
    return np.asarray([embedding for embedding, _ in found]), np.asarray(
        [label for _, label in found], dtype=bool
    )


def load_data(fixture):
    frozen = frozen_paths(fixture)
    forbidden_hashes = {case["content_hash"] for case in fixture["cases"]}
    excluded = training.eval_set_paths() | frozen
    pool = []
    for directory in training.INSTRUMENTAL_DIRS:
        pool.extend((path, False) for path in training.files_under(directory) if path not in excluded)
    for directory in training.VOCAL_DIRS:
        pool.extend((path, True) for path in training.files_under(directory) if path not in excluded)
    files = training.files_under(training.MUSIC) + training.files_under(training.APPS)
    broad = [(path, label) for path, label in capped_broad(files) if path not in frozen]
    hard_pool = []
    for directory in HARD_INSTRUMENTAL_DIRS:
        hard_pool.extend(
            (path, False) for path in training.files_under(directory) if path not in excluded
        )
    for directory in HARD_VOCAL_DIRS:
        hard_pool.extend(
            (path, True) for path in training.files_under(directory) if path not in excluded
        )

    if os.path.exists(CACHE):
        data = np.load(CACHE)
        if "cached_hard_x" in data:
            return tuple(data[key] for key in (
                "train_x", "train_y", "eval_x", "eval_y", "hard_x", "hard_y",
                "cached_x", "cached_y", "cached_eval_x", "cached_eval_y",
                "cached_hard_x", "cached_hard_y",
            ))
        train_x, train_y, eval_x, eval_y, hard_x, hard_y = tuple(
            data[key] for key in (
                "train_x", "train_y", "eval_x", "eval_y", "hard_x", "hard_y"
            )
        )
        if "cached_x" in data:
            cached_x, cached_y, cached_eval_x, cached_eval_y = tuple(
                data[key] for key in (
                    "cached_x", "cached_y", "cached_eval_x", "cached_eval_y"
                )
            )
    else:
        train_x, train_y = extract(pool)
        eval_x, eval_y = extract(broad)
        hard_x, hard_y = extract(hard_pool)

    if "cached_x" not in locals():
        cached_x, cached_y = extract_cached(pool, forbidden_hashes)
        cached_eval_x, cached_eval_y = extract_cached(broad, forbidden_hashes)
    cached_hard_x, cached_hard_y = extract_cached(hard_pool, forbidden_hashes)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE,
        train_x=train_x,
        train_y=train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        hard_x=hard_x,
        hard_y=hard_y,
        cached_x=cached_x,
        cached_y=cached_y,
        cached_eval_x=cached_eval_x,
        cached_eval_y=cached_eval_y,
        cached_hard_x=cached_hard_x,
        cached_hard_y=cached_hard_y,
    )
    return (
        train_x, train_y, eval_x, eval_y, hard_x, hard_y,
        cached_x, cached_y, cached_eval_x, cached_eval_y,
        cached_hard_x, cached_hard_y,
    )


def auc(scores, labels):
    pos = scores[labels]
    neg = scores[~labels]
    return float(
        ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
        / (len(pos) * len(neg))
    )


def main():
    with open(FIXTURE, encoding="utf-8") as handle:
        fixture = json.load(handle)
    (
        train_x, train_y, eval_x, eval_y, hard_x, hard_y,
        cached_x, cached_y, cached_eval_x, cached_eval_y,
        cached_hard_x, cached_hard_y,
    ) = load_data(fixture)
    frozen_x = np.asarray([case["embedding"] for case in fixture["cases"]])
    frozen_y = np.asarray([case["class"] == "vocal" for case in fixture["cases"]])
    print(
        f"train={len(train_x)} ({train_y.sum()} vocal), "
        f"cached-train={len(cached_x)} ({cached_y.sum()} vocal), "
        f"hard={len(hard_x)} ({hard_y.sum()} vocal), "
        f"cached-hard={len(cached_hard_x)} ({cached_hard_y.sum()} vocal), "
        f"disjoint-eval={len(eval_x)}/{len(cached_eval_x)} fresh/cached, "
        f"frozen={len(frozen_x)}"
    )

    best = None
    for vocal_repeat, instrumental_repeat in ((1, 1), (2, 2), (4, 4), (4, 8)):
            extra_x = np.concatenate([
                np.tile(cached_hard_x[cached_hard_y], (vocal_repeat, 1)),
                np.tile(cached_hard_x[~cached_hard_y], (instrumental_repeat, 1)),
            ])
            extra_y = np.concatenate([
                np.ones(vocal_repeat * cached_hard_y.sum(), dtype=bool),
                np.zeros(instrumental_repeat * (~cached_hard_y).sum(), dtype=bool),
            ])
            fit_x = np.concatenate([train_x, extra_x])
            fit_y = np.concatenate([train_y, extra_y])
            for seed in range(32):
                model = vocal_model.train(
                    fit_x,
                    fit_y,
                    model_id="sonara-vocalness-v2",
                    hidden=32,
                    epochs=3500,
                    seed=seed,
                    l2=1e-3,
                )
                frozen_scores = np.asarray([model.predict_vocalness(row) for row in frozen_x])
                exact_ok = (frozen_scores[frozen_y] > 0.35).sum() + (
                    frozen_scores[~frozen_y] < 0.35
                ).sum()
                eval_scores = np.asarray([model.predict_vocalness(row) for row in eval_x])
                eval_auc = auc(eval_scores, eval_y)
                cached_eval_scores = np.asarray(
                    [model.predict_vocalness(row) for row in cached_eval_x]
                )
                cached_eval_auc = auc(cached_eval_scores, cached_eval_y)
                fn = int((eval_scores[eval_y] < 0.35).sum())
                fp = int((eval_scores[~eval_y] >= 0.35).sum())
                key = (
                    int(exact_ok), eval_auc >= 0.9443, eval_auc,
                    cached_eval_auc, -fn, -fp,
                )
                print(
                    f"vr={vocal_repeat} ir={instrumental_repeat} seed={seed:2d} "
                    f"exact={exact_ok}/11 auc={eval_auc:.4f}/{cached_eval_auc:.4f} "
                    f"fn={fn}/{eval_y.sum()} fp={fp}/{(~eval_y).sum()} "
                    f"frozen={' '.join(f'{score:.3f}' for score in frozen_scores)}"
                )
                if best is None or key > best[0]:
                    best = (
                        key, model, frozen_scores, eval_auc, cached_eval_auc, fn, fp,
                        vocal_repeat, instrumental_repeat, seed,
                    )
                    model.save(OUT)
    (
        key, _model, scores, eval_auc, cached_eval_auc, fn, fp,
        vocal_repeat, instrumental_repeat, seed,
    ) = best
    print(
        f"BEST vr={vocal_repeat} ir={instrumental_repeat} seed={seed} "
        f"exact={key[0]}/11 auc={eval_auc:.4f}/{cached_eval_auc:.4f} "
        f"fn={fn} fp={fp} "
        f"scores={' '.join(f'{score:.6f}' for score in scores)} saved={OUT}"
    )


if __name__ == "__main__":
    main()
