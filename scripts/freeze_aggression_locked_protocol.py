#!/usr/bin/env python3
"""Seal aggression-v2 locked inputs before candidate inference is allowed."""

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


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected JSON objects")
    return rows


def canonical_split_hash(rows: list[dict], split: str) -> str:
    encoded = sorted(
        json.dumps(
            {
                "sample_id": row["sample_id"],
                "content_hash": row["content_hash"],
                "acoustic_fingerprint": row["acoustic_fingerprint"],
                "split": row["split"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
        if row.get("split") == split
    )
    return hashlib.sha256(("\n".join(encoded) + "\n").encode()).hexdigest()


def validate_evaluator(value: dict, freeze: dict, role: str) -> None:
    if value.get("format") != "sonara.aggression-evaluator.v1":
        raise ValueError(f"{role}: invalid evaluator format")
    for field in ("evaluator_id", "evaluator_type", "architecture_family"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"{role}: missing {field}")
    for field in (
        "audio_perception_confirmed",
        "independent_of_candidate",
        "independent_of_development_evaluator",
    ):
        if value.get(field) is not True:
            raise ValueError(f"{role}: {field} must be true")
    family_hash = hashlib.sha256(value["architecture_family"].encode()).hexdigest()
    excluded = freeze["evaluation_protocol"]["excluded_evaluator_family_sha256"]
    if family_hash in excluded:
        raise ValueError(f"{role}: evaluator family is not independent")


def validate_robustness(value: dict) -> None:
    if value.get("format") != "sonara.aggression-robustness.v1" or value.get("status") != "pass":
        raise ValueError("robustness result is not PASS")
    checks = {
        "gain_channel_resample_within_0_03_ratio": 0.95,
        "codec_within_0_05_ratio": 0.95,
        "quarter_removal_within_0_10_ratio": 0.90,
        "harsh_minus_loud_clean": 0.30,
    }
    for field, minimum in checks.items():
        if not isinstance(value.get(field), (int, float)) or value[field] < minimum:
            raise ValueError(f"robustness result fails {field}")
    if value.get("direction_preserved") is not True or value.get("silence_abstains") is not True:
        raise ValueError("robustness direction/silence requirement failed")


def validate_non_music(value: dict) -> None:
    if value.get("format") != "sonara.aggression-non-music.v1" or value.get("status") != "pass":
        raise ValueError("non-music result is not PASS")
    families = value.get("families")
    if not isinstance(families, dict) or set(families) != {"speech", "noise", "sparse"}:
        raise ValueError("non-music result must contain speech/noise/sparse")
    for name, result in families.items():
        if not isinstance(result, dict) or result.get("count", 0) < 1:
            raise ValueError(f"non-music {name} has no controls")
        if result.get("abstain_or_low_confidence_ratio", 0.0) < 0.95:
            raise ValueError(f"non-music {name} is below 95%")


def validate_inputs(
    freeze: dict,
    manifest_path: Path,
    labels_path: Path,
    pairs_path: Path,
    label_evaluator_path: Path,
    spot_evaluator_path: Path,
    spot_checks_path: Path,
    robustness_path: Path,
    non_music_path: Path,
) -> dict:
    cohort = freeze["protected_cohort"]
    manifest = read_jsonl(manifest_path)
    if sha256(manifest_path) != cohort["manifest_sha256"]:
        raise ValueError("protected manifest hash mismatch")
    for split, expected in cohort["splits"].items():
        rows = [row for row in manifest if row.get("split") == split]
        if len(rows) != expected["count"] or canonical_split_hash(manifest, split) != expected["identity_sha256"]:
            raise ValueError(f"protected {split} identity mismatch")
    locked_ids = {row["sample_id"] for row in manifest if row.get("split") == "locked"}

    label_evaluator = read_json(label_evaluator_path)
    spot_evaluator = read_json(spot_evaluator_path)
    validate_evaluator(label_evaluator, freeze, "label evaluator")
    validate_evaluator(spot_evaluator, freeze, "spot evaluator")
    if label_evaluator["evaluator_id"] == spot_evaluator["evaluator_id"]:
        raise ValueError("spot evaluator must have a distinct identity")
    if (
        label_evaluator["architecture_family"] == spot_evaluator["architecture_family"]
        and "human" not in {label_evaluator["evaluator_type"], spot_evaluator["evaluator_type"]}
    ):
        raise ValueError("spot evaluator must be human or use a second architecture")

    labels = read_jsonl(labels_path)
    label_ids = [row.get("sample_id") for row in labels]
    if len(labels) != 160 or len(set(label_ids)) != 160 or set(label_ids) != locked_ids:
        raise ValueError("locked labels must cover each of 160 locked recordings exactly once")
    for row in labels:
        target = row.get("target")
        if not isinstance(target, (int, float)) or not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("locked label target must be finite and in [0, 1]")
        if row.get("evaluator_id") != label_evaluator["evaluator_id"]:
            raise ValueError("locked label evaluator identity mismatch")

    pairs = read_jsonl(pairs_path)
    if len(pairs) != 80 or len({row.get("pair_id") for row in pairs}) != 80:
        raise ValueError("locked protocol requires exactly 80 unique pairs")
    categories = {name: 0 for name in ("hard", "near", "broad", "tie")}
    used: list[str] = []
    pair_by_id = {}
    for row in pairs:
        category = row.get("category")
        decision = row.get("decision")
        if category not in categories or decision not in {"left", "right", "tie"}:
            raise ValueError("invalid locked pair category/decision")
        if (category == "tie") != (decision == "tie"):
            raise ValueError("tie category and decision must agree")
        left, right = row.get("left_id"), row.get("right_id")
        if left == right or left not in locked_ids or right not in locked_ids:
            raise ValueError("locked pair references an invalid recording")
        categories[category] += 1
        used.extend((left, right))
        pair_by_id[row["pair_id"]] = row
    if categories != {"hard": 24, "near": 20, "broad": 20, "tie": 16}:
        raise ValueError(f"locked strata mismatch: {categories}")
    if len(set(used)) != 160 or set(used) != locked_ids:
        raise ValueError("locked pairs must use each recording exactly once")

    spot_checks = read_jsonl(spot_checks_path)
    if len(spot_checks) < 20 or len({row.get("pair_id") for row in spot_checks}) != len(spot_checks):
        raise ValueError("at least 20 unique spot checks are required")
    agreement = 0
    for row in spot_checks:
        pair = pair_by_id.get(row.get("pair_id"))
        if pair is None or pair["decision"] == "tie" or row.get("decision") not in {"left", "right"}:
            raise ValueError("spot checks must reference decisive locked pairs")
        if row.get("evaluator_id") != spot_evaluator["evaluator_id"]:
            raise ValueError("spot-check evaluator identity mismatch")
        agreement += row["decision"] == pair["decision"]
    agreement_ratio = agreement / len(spot_checks)
    if agreement_ratio < 0.80:
        raise ValueError("independent spot-check agreement is below 80%")

    robustness = read_json(robustness_path)
    non_music = read_json(non_music_path)
    validate_robustness(robustness)
    validate_non_music(non_music)
    return {
        "counts": {"labels": len(labels), "pairs": len(pairs), "spot_checks": len(spot_checks), **categories},
        "independent_agreement": agreement_ratio,
        "label_evaluator_id_sha256": hashlib.sha256(label_evaluator["evaluator_id"].encode()).hexdigest(),
        "spot_evaluator_id_sha256": hashlib.sha256(spot_evaluator["evaluator_id"].encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--manifest", type=Path, required=True)
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
    if freeze.get("format") != "sonara.aggression-freeze.v1":
        raise ValueError("invalid candidate freeze")
    expected_scripts = freeze["evaluation_protocol"]
    if sha256(Path(__file__)) != expected_scripts["freeze_script_sha256"]:
        raise ValueError("protocol freezer changed after candidate freeze")
    if sha256(Path(__file__).with_name("evaluate_aggression_locked.py")) != expected_scripts["one_shot_script_sha256"]:
        raise ValueError("one-shot evaluator changed after candidate freeze")
    for relative, expected in freeze["candidate"]["research_sources"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"research source changed after candidate freeze: {relative}")
    summary = validate_inputs(
        freeze,
        args.manifest,
        args.labels,
        args.pairs,
        args.label_evaluator,
        args.spot_evaluator,
        args.spot_checks,
        args.robustness,
        args.non_music,
    )
    protocol = {
        "format": "sonara.aggression-locked-protocol.v1",
        "freeze_sha256": sha256(args.freeze),
        "candidate_commit": freeze["candidate"]["commit"],
        "inputs": {
            "manifest_sha256": sha256(args.manifest),
            "labels_sha256": sha256(args.labels),
            "pairs_sha256": sha256(args.pairs),
            "label_evaluator_sha256": sha256(args.label_evaluator),
            "spot_evaluator_sha256": sha256(args.spot_evaluator),
            "spot_checks_sha256": sha256(args.spot_checks),
            "robustness_sha256": sha256(args.robustness),
            "non_music_sha256": sha256(args.non_music),
        },
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"locked protocol sealed: {args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"locked protocol rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
