#!/usr/bin/env python3
"""Pack the frozen compact candidate into Sonara's bounded Rust artifact."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

from compare_model_families import labels, metrics, table


MAGIC = b"SNRAGGR2"
FORMAT_VERSION = 2
MODEL_VERSION = 3


def f32(value):
    return float(np.float32(value))


def predict(candidate, vector):
    linear = candidate["linear"]
    raw = np.float32(0.0)
    for weight, value, center in zip(linear["coefficients"], vector, linear["feature_center"]):
        raw = np.float32(raw + np.float32(weight) * np.float32(np.float32(value) - np.float32(center)))
    calibrated = np.float32(np.float32(linear["calibration"]["slope"]) * raw + np.float32(linear["calibration"]["intercept"]))
    linear_score = np.float32(1.0 / (1.0 + np.exp(-calibrated)))
    ensemble = candidate["tree_ensemble"]
    tree_score = np.float32(ensemble["baseline"])
    for tree in ensemble["trees"]:
        index = 0
        while not tree[index]["leaf"]:
            node = tree[index]
            index = node["left"] if np.float32(vector[node["feature"]]) <= np.float32(node["threshold"]) else node["right"]
        tree_score = np.float32(tree_score + np.float32(tree[index]["value"]))
    tree_score = np.clip(tree_score, 0, 1)
    blended = np.float32(
        np.float32(candidate["linear_weight"]) * linear_score
        + np.float32(candidate["tree_weight"]) * tree_score
    )
    corrected = np.float32(
        blended
        + np.float32(candidate["harshness_correction"])
        * np.float32(np.float32(vector[26]) - np.float32(0.5))
    )
    return float(np.clip(corrected, 0, 1))


def main() -> int:
    pool = Path(sys.argv[1])
    candidate = json.loads((pool / "compact_candidate.json").read_text())
    names = candidate["feature_names"]
    ensemble = candidate["tree_ensemble"]
    trees = ensemble["trees"]
    header = bytearray()
    header += MAGIC
    header += struct.pack("<IIII", FORMAT_VERSION, len(names), len(trees), MODEL_VERSION)
    header += bytes.fromhex(candidate["feature_schema_sha256"])
    header += struct.pack(
        "<7f",
        f32(ensemble["baseline"]),
        f32(candidate["linear_weight"]),
        f32(candidate["tree_weight"]),
        f32(candidate["linear"]["calibration"]["slope"]),
        f32(candidate["linear"]["calibration"]["intercept"]),
        f32(candidate["tie_band"]),
        f32(candidate["harshness_correction"]),
    )
    header += struct.pack(f"<{len(names)}f", *map(f32, candidate["linear"]["feature_center"]))
    payload = bytearray(header)
    for tree in trees:
        payload += struct.pack("<H", len(tree))
        for node in tree:
            payload += struct.pack(
                "<BBHHff",
                255 if node["leaf"] else node["feature"],
                int(node["leaf"]),
                node["left"],
                node["right"],
                f32(node["threshold"]),
                f32(node["value"]),
            )
    payload += hashlib.sha256(payload).digest()
    output = pool / "aggression_model.bin"
    output.write_bytes(payload)

    _, features = table(pool / "evaluation_development_features.tsv")
    truth = labels(pool / "evaluation_development_labels.jsonl")
    pairs = [json.loads(line) for line in (pool / "development_pairs.jsonl").read_text().splitlines()]
    scores = {sample_id: predict(candidate, vector) for sample_id, vector in features.items()}
    result = metrics(pairs, truth, scores, candidate["tie_band"])
    if not (result[0] >= 52 and result[1] >= 20 and result[2] >= 12):
        raise RuntimeError(f"f32 artifact regressed open pair gates: {result}")
    print(f"artifact -> {output}")
    print(f"bytes={len(payload)} sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"f32 decisive={result[0]} hard={result[1]} ties={result[2]} rho={result[3]:.6f} mae={result[4]:.6f} range={result[5]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
