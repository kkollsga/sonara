#!/usr/bin/env python3
"""Freeze predictions and build a track/group-disjoint, blind audio audit."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SONARA = HERE.parents[3]
POOL = SONARA / "dev-docs/bench/out/sonagram-aggression-ranking"
OUT = SONARA / "dev-docs/bench/out/ferricml-aggression-audit"
OLD_SCRIPTS = SONARA / "dev-docs/bench/scripts/sonagram-aggression-ranking"
sys.path.insert(0, str(OLD_SCRIPTS))
from pack_rust_artifact import predict as old_predict  # noqa: E402

SEED = "ferricml-aggression-fresh-audit-v1"
INITIAL_TRAIN_TRACKS = 74
EXCERPT_SEC = 20


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(path: Path):
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return {
        row[0]: [float(value) for value in row[2:]]
        for row in (line.split("\t") for line in lines[1:])
    }, header[2:]


def duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def excerpt(path: str, output: Path):
    length = duration(path)
    start = max(0.0, min(length * 0.30, max(0.0, length - EXCERPT_SEC)))
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", str(EXCERPT_SEC),
            "-i", path, "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le",
            str(output), "-loglevel", "error",
        ],
        check=True,
        capture_output=True,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = OUT / "clips"
    clips.mkdir(exist_ok=True)
    private_paths = json.loads((POOL / "private_paths.json").read_text())
    fit_ids = {
        json.loads(line)["sample_id"]
        for line in (POOL / "fit_train_labels.jsonl").read_text().splitlines()
    }
    features, names = table(POOL / "train_features.tsv")
    if len(names) != 39:
        raise RuntimeError("unexpected feature width")
    fit_groups = {
        os.path.dirname(private_paths[sample_id])
        for sample_id in fit_ids
        if sample_id in private_paths
    }
    eligible_train = [
        sample_id
        for sample_id in features
        if sample_id not in fit_ids
        and os.path.dirname(private_paths[sample_id]) not in fit_groups
    ]
    eligible_train.sort(key=lambda sample_id: digest(f"{SEED}:{sample_id}".encode()))
    if len(eligible_train) < INITIAL_TRAIN_TRACKS:
        raise RuntimeError(
            f"need {INITIAL_TRAIN_TRACKS} group-disjoint train tracks, found {len(eligible_train)}"
        )
    corpus_manifest = [
        json.loads(line) for line in (POOL / "manifest.jsonl").read_text().splitlines()
    ]
    eligible_reserved = [
        row["sample_id"]
        for row in corpus_manifest
        if row["split"] == "locked"
        and os.path.dirname(private_paths[row["sample_id"]]) not in fit_groups
    ]
    eligible_reserved.sort(key=lambda sample_id: digest(f"reserved:{SEED}:{sample_id}".encode()))
    selected = eligible_train[:INITIAL_TRAIN_TRACKS] + eligible_reserved

    new_scores = json.loads((POOL / "ferric_all_scores.json").read_text())
    old_model = json.loads((POOL / "compact_candidate.json").read_text())
    predictions = {
        sample_id: {
            "released_v2": old_predict(old_model, features[sample_id]),
            "ferric_candidate": new_scores[sample_id],
        }
        for sample_id in selected
        if sample_id in features and sample_id in new_scores
    }
    freeze = {
        "protocol": 1,
        "selection_seed": SEED,
        "track_count": len(selected),
        "pair_tie_band": 0.07,
        "label_tie_band": 0.08,
        "candidate": json.loads((POOL / "ferric_candidate.json").read_text()),
        "ranker_sha256": digest((POOL / "ferric_ranker_candidate.ferricml").read_bytes()),
        "hgb_sha256": digest((POOL / "ferric_hgb_candidate.ferricml").read_bytes()),
        "predictions": predictions,
    }
    blob = json.dumps(freeze, indent=2, sort_keys=True).encode()
    (OUT / "predictions.freeze.json").write_bytes(blob)
    (OUT / "predictions.freeze.sha256").write_text(digest(blob) + "\n")

    manifest = []
    for index, sample_id in enumerate(selected, 1):
        temporary = clips / f"audit-{index:03d}.wav"
        excerpt(private_paths[sample_id], temporary)
        clip_blob = temporary.read_bytes()
        clip_hash = digest(clip_blob)
        final = clips / f"{clip_hash}.wav"
        temporary.replace(final)
        manifest.append({
            "audit_id": f"audit-{index:03d}",
            "sample_id": sample_id,
            "clip_sha256": clip_hash,
            "clip": f"clips/{clip_hash}.wav",
            "source_group": digest(os.path.dirname(private_paths[sample_id]).encode())[:16],
        })
    (OUT / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    )
    order = sorted(manifest, key=lambda row: digest(f"pair:{SEED}:{row['sample_id']}".encode()))
    pairs = []
    for index in range(0, len(order) - 1, 2):
        left, right = order[index:index + 2]
        pairs.append({
            "pair_id": f"fresh-{index // 2 + 1:03d}",
            "left_id": left["sample_id"],
            "right_id": right["sample_id"],
            "left_clip": left["clip"],
            "right_clip": right["clip"],
        })
    (OUT / "pairs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pairs)
    )
    print(f"frozen predictions: {digest(blob)}")
    print(f"fresh audit: {len(manifest)} tracks / {len(pairs)} blind pairs")


if __name__ == "__main__":
    main()
