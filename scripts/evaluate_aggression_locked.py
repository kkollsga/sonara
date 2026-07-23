#!/usr/bin/env python3
"""Run the sealed aggression-v2 locked evaluation exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / "tests/reference_data/aggression_v2_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        value = (start + end - 1) / 2.0
        for index in order[start:end]:
            result[index] = value
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float:
    x, y = rank(left), rank(right)
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def verify_hashes(protocol: dict, paths: dict[str, Path]) -> None:
    for name, expected in protocol["inputs"].items():
        key = name.removesuffix("_sha256")
        if key not in paths or sha256(paths[key]) != expected:
            raise ValueError(f"sealed input hash mismatch: {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-paths", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--label-evaluator", type=Path, required=True)
    parser.add_argument("--spot-evaluator", type=Path, required=True)
    parser.add_argument("--spot-checks", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--non-music", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = read_json(args.freeze)
    protocol = read_json(args.protocol)
    if protocol.get("format") != "sonara.aggression-locked-protocol.v1":
        raise ValueError("invalid sealed protocol format")
    if protocol.get("candidate_commit") != freeze["candidate"]["commit"]:
        raise ValueError("sealed protocol names a different candidate")
    expected_scripts = freeze["evaluation_protocol"]
    if sha256(Path(__file__)) != expected_scripts["one_shot_script_sha256"]:
        raise ValueError("one-shot evaluator changed after candidate freeze")
    if sha256(Path(__file__).with_name("freeze_aggression_locked_protocol.py")) != expected_scripts["freeze_script_sha256"]:
        raise ValueError("protocol freezer changed after candidate freeze")
    for relative, expected in freeze["candidate"]["research_sources"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"research source changed after candidate freeze: {relative}")
    if sha256(args.freeze) != protocol.get("freeze_sha256"):
        raise ValueError("sealed protocol does not match candidate freeze")
    verify_hashes(protocol, {
        "manifest": args.manifest,
        "labels": args.labels,
        "pairs": args.pairs,
        "label_evaluator": args.label_evaluator,
        "spot_evaluator": args.spot_evaluator,
        "spot_checks": args.spot_checks,
        "robustness": args.robustness,
        "non_music": args.non_music,
    })
    for relative, expected in freeze["candidate"]["files"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"candidate file changed after freeze: {relative}")

    marker = args.output.with_suffix(args.output.suffix + ".started")
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(f"protocol_sha256={sha256(args.protocol)}\n")

    import sonara

    if sonara.AGGRESSION_MODEL_ID != freeze["model"]["model_id"]:
        raise ValueError("installed Sonara model identity does not match freeze")
    manifest = {row["sample_id"]: row for row in read_jsonl(args.manifest) if row["split"] == "locked"}
    paths = read_json(args.private_paths)
    labels = {row["sample_id"]: float(row["target"]) for row in read_jsonl(args.labels)}
    pairs = read_jsonl(args.pairs)
    scores = {}
    abstentions = 0
    for sample_id in sorted(manifest):
        path = Path(paths[sample_id])
        if not path.is_file():
            raise ValueError(f"missing locked audio for {sample_id}")
        signal, sample_rate = sonara.load(str(path), sr=22_050, mono=True)
        duration = len(signal) / sample_rate
        expected_duration = float(manifest[sample_id]["duration_sec"])
        if abs(duration - expected_duration) > 1.0:
            raise ValueError(f"locked audio duration mismatch for {sample_id}")
        excerpt_start = max(0.0, min(expected_duration * 0.35, max(0.0, expected_duration - 20.0)))
        excerpt, excerpt_rate = sonara.load(
            str(path), sr=22_050, mono=True, offset=excerpt_start, duration=20.0
        )
        fingerprint = sonara.analyze_signal(
            excerpt, sr=excerpt_rate, features=["fingerprint"]
        )["fingerprint"]
        if str(fingerprint) != manifest[sample_id]["acoustic_fingerprint"]:
            raise ValueError(f"locked audio fingerprint mismatch for {sample_id}")
        result = sonara.analyze_aggression_signal(signal, sr=sample_rate)
        score = result["aggression_score"]
        if score is None:
            abstentions += 1
        else:
            scores[sample_id] = float(score)

    tie_band = float(freeze["model"]["tie_band"])
    decisive_correct = hard_correct = tie_correct = 0
    for pair in pairs:
        left, right = scores.get(pair["left_id"]), scores.get(pair["right_id"])
        if left is None or right is None:
            predicted = "abstain"
        elif abs(left - right) <= tie_band:
            predicted = "tie"
        else:
            predicted = "left" if left > right else "right"
        correct = predicted == pair["decision"]
        if pair["decision"] == "tie":
            tie_correct += correct
        else:
            decisive_correct += correct
            if pair["category"] == "hard":
                hard_correct += correct

    sample_ids = sorted(labels)
    target_values = [labels[sample_id] for sample_id in sample_ids]
    predicted_values = [scores.get(sample_id, 0.5) for sample_id in sample_ids]
    rho = spearman(target_values, predicted_values)
    mae = sum(abs(a - b) for a, b in zip(target_values, predicted_values)) / len(sample_ids)
    score_range = max(predicted_values) - min(predicted_values)
    locked_pass = (
        abstentions == 0
        and decisive_correct >= 52
        and hard_correct >= 20
        and tie_correct >= 12
        and rho >= 0.65
        and mae <= 0.15
        and score_range >= 0.65
    )
    robustness = read_json(args.robustness)
    non_music = read_json(args.non_music)
    result = {
        "format": "sonara.aggression-locked-result.v1",
        "candidate_commit": freeze["candidate"]["commit"],
        "protocol_sha256": sha256(args.protocol),
        "locked_evaluation": {
            "status": "pass" if locked_pass else "no_go",
            "candidate_commit": freeze["candidate"]["commit"],
            "freeze_sha256": sha256(args.freeze),
            "protocol_sha256": sha256(args.protocol),
            "decisive_correct": decisive_correct,
            "decisive_total": 64,
            "hard_correct": hard_correct,
            "hard_total": 24,
            "tie_correct": tie_correct,
            "tie_total": 16,
            "spearman": rho,
            "mae": mae,
            "score_range": score_range,
            "abstentions": abstentions,
        },
        "independence": {
            "status": "pass",
            "spot_check_correct": round(protocol["independent_agreement"] * protocol["counts"]["spot_checks"]),
            "spot_check_total": protocol["counts"]["spot_checks"],
            "agreement": protocol["independent_agreement"],
            "label_evaluator_id_sha256": protocol["label_evaluator_id_sha256"],
            "spot_evaluator_id_sha256": protocol["spot_evaluator_id_sha256"],
        },
        "robustness": robustness,
        "non_music": non_music,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"locked evaluation: {'PASS' if locked_pass else 'NO-GO'} receipt={args.output}")
    return 0 if locked_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"locked evaluation rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
