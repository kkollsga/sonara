#!/usr/bin/env python3
"""Execute the custodian-authorized aggression-v2 audit and emit aggregates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from freeze_aggression_locked_protocol import verify_declared_candidate, verify_ssh_signature


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / "tests/reference_data/aggression_v2_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


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


def content_hash(binary: Path, path: Path) -> str:
    output = subprocess.run(
        [str(binary), str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.rstrip("\n")
    fields = output.split("\t", 2)
    if len(fields) != 3:
        raise ValueError("content hasher returned an invalid record")
    return fields[1]


def score_path(sonara, path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing sealed audio: {path.name}")
    return sonara.analyze_aggression_file(str(path))


def verify_runtime(attestation: dict, site: Path, content_hasher: Path, freeze: dict):
    if tree_manifest(site) != attestation["site_manifest"]:
        raise ValueError("sealed runtime site changed after attestation")
    if sha256(Path(sys.executable).resolve()) != attestation["probe"]["python_sha256"]:
        raise ValueError("evaluator Python executable differs from runtime attestation")
    if sha256(content_hasher) != attestation["content_hasher_sha256"]:
        raise ValueError("content-hash executable differs from runtime attestation")
    sys.path.insert(0, str(site.resolve()))
    import sonara

    # Force all analysis-time native dependencies to load before checking them.
    import numpy as np
    sonara.analyze_aggression_signal(np.zeros(22_050, dtype=np.float32), sr=22_050)

    native = Path(sonara._sonara.__file__).resolve()
    if not native.is_relative_to(site.resolve()) or sha256(native) != attestation["probe"]["native_sha256"]:
        raise ValueError("loaded native module differs from runtime attestation")
    if sonara.AGGRESSION_MODEL_ID != freeze["model"]["model_id"]:
        raise ValueError("loaded native model identity differs from candidate")
    for name, expected in attestation["probe"]["modules"].items():
        module = sys.modules.get(name)
        path = Path(getattr(module, "__file__", ""))
        if not path.is_file() or sha256(path.resolve()) != expected["sha256"]:
            raise ValueError(f"loaded runtime dependency differs: {name}")
    return sonara


def synthetic_physical_controls(sonara) -> dict:
    import numpy as np

    sample_rate = 22_050
    time = np.arange(sample_rate * 8, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(20260722)
    harsh = 0.7 * np.tanh(
        8 * (0.38 * np.sin(2 * np.pi * 110 * time) + 0.25 * np.sin(2 * np.pi * 173 * time) + 0.20 * rng.normal(size=time.size))
    )
    phase = (time * 2) % 1
    loud_clean = 0.9 * np.exp(-phase * 12) * np.sin(2 * np.pi * 80 * time)
    values = {
        "harsh": sonara.analyze_aggression_signal(harsh.astype(np.float32), sr=sample_rate),
        "loud_clean": sonara.analyze_aggression_signal(loud_clean.astype(np.float32), sr=sample_rate),
        "silence": sonara.analyze_aggression_signal(np.zeros(sample_rate * 8, dtype=np.float32), sr=sample_rate),
    }
    margin = values["harsh"]["aggression_score"] - values["loud_clean"]["aggression_score"]
    return {
        "harsh_minus_loud_clean": margin,
        "silence_abstains": values["silence"]["aggression_score"] is None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-paths", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--label-evaluator", type=Path, required=True)
    parser.add_argument("--spot-evaluator", type=Path, required=True)
    parser.add_argument("--spot-checks", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--runtime-site", type=Path, required=True)
    parser.add_argument("--content-hasher", type=Path, required=True)
    parser.add_argument("--robustness-cases", type=Path, required=True)
    parser.add_argument("--robustness-paths", type=Path, required=True)
    parser.add_argument("--non-music-cases", type=Path, required=True)
    parser.add_argument("--non-music-paths", type=Path, required=True)
    parser.add_argument("--custody-authorization", type=Path, required=True)
    parser.add_argument("--custody-signature", type=Path, required=True)
    parser.add_argument("--allowed-signers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = read_json(args.freeze)
    verify_declared_candidate(freeze)
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
    if sha256(args.freeze) != protocol.get("freeze_sha256"):
        raise ValueError("sealed protocol does not match candidate freeze")

    input_paths = {
        "manifest": args.manifest,
        "locked_paths": args.locked_paths,
        "labels": args.labels,
        "pairs": args.pairs,
        "label_evaluator": args.label_evaluator,
        "spot_evaluator": args.spot_evaluator,
        "spot_checks": args.spot_checks,
        "runtime_attestation": args.runtime_attestation,
        "robustness_cases": args.robustness_cases,
        "robustness_paths": args.robustness_paths,
        "non_music_cases": args.non_music_cases,
        "non_music_paths": args.non_music_paths,
    }
    verify_hashes(protocol, input_paths)
    if sha256(args.custody_authorization) != protocol["custody"]["authorization_sha256"]:
        raise ValueError("custody authorization substitution")
    if sha256(args.custody_signature) != protocol["custody"]["signature_sha256"]:
        raise ValueError("custody signature substitution")
    if sha256(args.allowed_signers) != protocol["custody"]["allowed_signers_sha256"]:
        raise ValueError("custodian root substitution")
    verify_ssh_signature(
        args.custody_authorization,
        args.custody_signature,
        args.allowed_signers,
        freeze["custody"]["signer_identity"],
        freeze["custody"]["signature_namespace"],
    )
    authorization = read_json(args.custody_authorization)
    if authorization.get("evaluation_identity") != protocol["evaluation_identity"] or not authorization.get("cohort_retired"):
        raise ValueError("custodian did not retire the cohort to this evaluation")

    attestation = read_json(args.runtime_attestation)
    sonara = verify_runtime(attestation, args.runtime_site, args.content_hasher, freeze)

    manifest_rows = read_jsonl(args.manifest)
    if len({row["sample_id"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("cohort sample IDs overlap")
    if len({row["content_hash"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("cohort content identities overlap")
    if len({row["acoustic_fingerprint"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("cohort acoustic identities overlap")
    split_ids = {
        split: {row["sample_id"] for row in manifest_rows if row["split"] == split}
        for split in ("train", "development", "locked")
    }
    if any(split_ids[left] & split_ids[right] for left, right in (("train", "development"), ("train", "locked"), ("development", "locked"))):
        raise ValueError("cohort splits overlap")

    manifest = {row["sample_id"]: row for row in manifest_rows if row["split"] == "locked"}
    paths = read_json(args.locked_paths)
    labels = {row["sample_id"]: float(row["target"]) for row in read_jsonl(args.labels)}
    pairs = read_jsonl(args.pairs)
    scores = {}
    abstentions = 0
    for sample_id in sorted(manifest):
        path = Path(paths[sample_id])
        if content_hash(args.content_hasher, path) != manifest[sample_id]["content_hash"]:
            raise ValueError(f"full locked content mismatch for {sample_id}")
        result = score_path(sonara, path)
        score = result["aggression_score"]
        if score is None:
            abstentions += 1
        else:
            scores[sample_id] = float(score)

    tie_band = float(freeze["model"]["tie_band"])
    decisive_correct = hard_correct = tie_correct = 0
    for pair in pairs:
        left, right = scores.get(pair["left_id"]), scores.get(pair["right_id"])
        predicted = "abstain" if left is None or right is None else "tie" if abs(left - right) <= tie_band else "left" if left > right else "right"
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
    locked_pass = abstentions == 0 and decisive_correct >= 52 and hard_correct >= 20 and tie_correct >= 12 and rho >= 0.65 and mae <= 0.15 and score_range >= 0.65

    robustness_rows = read_jsonl(args.robustness_cases)
    robustness_paths = {key: Path(value) for key, value in read_json(args.robustness_paths).items()}
    robustness_scores = {}
    family_results = {}
    for row in robustness_rows:
        for role in ("base", "variant", "contrast"):
            control_id = row[f"{role}_id"]
            path = robustness_paths[control_id]
            if content_hash(args.content_hasher, path) != row[f"{role}_content_hash"]:
                raise ValueError(f"robustness content mismatch: {control_id}")
            if control_id not in robustness_scores:
                robustness_scores[control_id] = score_path(sonara, path)["aggression_score"]
        base = robustness_scores[row["base_id"]]
        variant = robustness_scores[row["variant_id"]]
        contrast = robustness_scores[row["contrast_id"]]
        stable = base is not None and variant is not None and abs(base - variant) <= row["tolerance"]
        direction = row["expected_direction"]
        direction_ok = base is not None and variant is not None and contrast is not None and ((base > contrast and variant > contrast) if direction == "higher" else (base < contrast and variant < contrast))
        family_results.setdefault(row["family"], []).append((stable, direction_ok))
    robustness = {
        family: {
            "count": len(values),
            "within_tolerance_ratio": sum(value[0] for value in values) / len(values),
            "direction_preserved_ratio": sum(value[1] for value in values) / len(values),
        }
        for family, values in family_results.items()
    }
    physical = synthetic_physical_controls(sonara)
    robustness_pass = all(value["within_tolerance_ratio"] >= (0.90 if family == "quarter_removal" else 0.95) and value["direction_preserved_ratio"] == 1.0 for family, value in robustness.items()) and physical["harsh_minus_loud_clean"] >= 0.30 and physical["silence_abstains"]

    non_music_paths = {key: Path(value) for key, value in read_json(args.non_music_paths).items()}
    non_music_values = {family: [] for family in ("speech", "noise", "sparse")}
    for row in read_jsonl(args.non_music_cases):
        path = non_music_paths[row["control_id"]]
        if content_hash(args.content_hasher, path) != row["content_hash"]:
            raise ValueError(f"non-music content mismatch: {row['control_id']}")
        analysis = score_path(sonara, path)
        non_music_values[row["family"]].append(
            analysis["aggression_score"] is None or analysis["aggression_confidence"] <= 0.25
        )
    non_music = {
        family: {"count": len(values), "abstain_or_low_confidence_ratio": sum(values) / len(values)}
        for family, values in non_music_values.items()
    }
    non_music_pass = all(value["abstain_or_low_confidence_ratio"] >= 0.95 for value in non_music.values())

    result = {
        "format": "sonara.aggression-locked-result.v2",
        "candidate_commit": freeze["candidate"]["commit"],
        "evaluation_identity": protocol["evaluation_identity"],
        "protocol_sha256": sha256(args.protocol),
        "runtime_attestation_sha256": sha256(args.runtime_attestation),
        "custody_authorization_sha256": sha256(args.custody_authorization),
        "locked_evaluation": {
            "status": "pass" if locked_pass else "no_go",
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
            "label_evaluator_identity_sha256": protocol["label_evaluator_identity_sha256"],
            "spot_evaluator_identity_sha256": protocol["spot_evaluator_identity_sha256"],
        },
        "robustness": {"status": "pass" if robustness_pass else "no_go", "families": robustness, **physical},
        "non_music": {"status": "pass" if non_music_pass else "no_go", "families": non_music},
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    all_pass = locked_pass and robustness_pass and non_music_pass
    print(f"locked evaluation: {'PASS' if all_pass else 'NO-GO'} receipt={args.output} sha256={sha256(args.output)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"locked evaluation rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
