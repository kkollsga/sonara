#!/usr/bin/env python3
"""Explain development pair failures without exposing track metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_features(path: Path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    names = [name for name in rows[0] if name not in {"sample_id", "support"}]
    values = {
        row["sample_id"]: [float(row[name]) for name in names]
        for row in rows
    }
    return names, values


def score(features, candidate):
    raw = sum(
        weight * (value - center)
        for weight, value, center in zip(
            candidate["coefficients"], features, candidate["feature_center"]
        )
    )
    value = (
        candidate["calibration"]["slope"] * raw
        + candidate["calibration"]["intercept"]
    )
    return 1.0 / (1.0 + math.exp(-value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    args = parser.parse_args()
    candidate = json.loads((args.pool / "candidate.json").read_text())
    names, features = read_features(args.pool / "evaluation_development_features.tsv")
    pairs = [
        json.loads(line)
        for line in (args.pool / "development_pairs.jsonl").read_text().splitlines()
    ]
    scores = {sample_id: score(vector, candidate) for sample_id, vector in features.items()}
    for category in ("hard", "tie", "near", "broad"):
        print(f"\n{category.upper()}")
        selected = [pair for pair in pairs if pair["category"] == category]
        selected.sort(
            key=lambda pair: abs(scores[pair["left_id"]] - scores[pair["right_id"]]),
            reverse=True,
        )
        for pair in selected:
            delta = scores[pair["left_id"]] - scores[pair["right_id"]]
            expected = 0.0 if pair["decision"] == "tie" else (
                1.0 if pair["decision"] == "left" else -1.0
            )
            correctness = "ok" if expected == 0.0 or delta * expected > 0 else "WRONG"
            contributions = [
                (
                    abs(weight * (left - right)),
                    name,
                    weight * (left - right),
                    left - right,
                )
                for name, weight, left, right in zip(
                    names,
                    candidate["coefficients"],
                    features[pair["left_id"]],
                    features[pair["right_id"]],
                )
            ]
            top = ", ".join(
                f"{name}:{contribution:+.3f}/{difference:+.3f}"
                for _, name, contribution, difference in sorted(contributions, reverse=True)[:5]
            )
            print(
                f"{pair['pair_id']} {correctness} decision={pair['decision']} "
                f"teacher={pair['left_target'] - pair['right_target']:+.3f} "
                f"pred={delta:+.3f} context={pair['context_distance']:.3f} top=[{top}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
