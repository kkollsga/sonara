#!/usr/bin/env python3
"""Seal aggression-v2 locked inputs before candidate inference is allowed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
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


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_output(*args: str, binary: bool = False):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=not binary,
    ).stdout


def verify_declared_candidate(freeze: dict) -> None:
    candidate = freeze["candidate"]
    if git_output("rev-parse", f"{candidate['commit']}^{{tree}}").strip() != candidate["tree"]:
        raise ValueError("candidate tree mismatch")
    if git_output("rev-parse", f"{candidate['commit']}^").strip() != candidate["base_commit"]:
        raise ValueError("candidate base mismatch")
    diff = git_output("diff", "--binary", f"{candidate['base_commit']}..{candidate['commit']}", binary=True)
    if hashlib.sha256(diff).hexdigest() != candidate["binary_diff_sha256"]:
        raise ValueError("candidate binary diff mismatch")
    declared = {**candidate["files"], **freeze["evaluation_protocol"].get("immutable_dependencies", {})}
    for relative, expected in declared.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen dependency mismatch: {relative}")


def verify_ssh_signature(
    payload_path: Path,
    signature_path: Path,
    allowed_signers_path: Path,
    identity: str,
    namespace: str,
) -> None:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise ValueError("ssh-keygen is required for custodian signature verification")
    result = subprocess.run(
        [
            executable,
            "-Y",
            "verify",
            "-f",
            str(allowed_signers_path),
            "-I",
            identity,
            "-n",
            namespace,
            "-s",
            str(signature_path),
        ],
        input=payload_path.read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError("custodian signature verification failed")


def validate_custody_authorization(
    freeze: dict,
    evaluation_identity: str,
    authorization_path: Path,
    signature_path: Path,
    allowed_signers_path: Path,
) -> dict:
    custody = freeze.get("custody", {})
    if custody.get("status") != "ready":
        raise ValueError("Sonagram custodian root is not frozen")
    if sha256(allowed_signers_path) != custody.get("allowed_signers_sha256"):
        raise ValueError("custodian public-key root mismatch")
    authorization = read_json(authorization_path)
    required = {
        "format": "sonagram.aggression-custody-authorization.v1",
        "evaluation_identity": evaluation_identity,
        "candidate_commit": freeze["candidate"]["commit"],
        "cohort_manifest_sha256": freeze["protected_cohort"]["manifest_sha256"],
        "action": "retire-cohort-and-authorize-one-candidate",
        "cohort_retired": True,
        "ledger_repository": custody["ledger_repository"],
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"custody authorization mismatch: {field}")
    if not isinstance(authorization.get("sequence"), int) or authorization["sequence"] < 1:
        raise ValueError("custody authorization lacks append-only sequence")
    if not isinstance(authorization.get("previous_entry_sha256"), str) or len(authorization["previous_entry_sha256"]) != 64:
        raise ValueError("custody authorization lacks previous ledger identity")
    verify_ssh_signature(
        authorization_path, signature_path, allowed_signers_path,
        custody["signer_identity"], custody["signature_namespace"],
    )
    return authorization


def validate_runtime_attestation(value: dict, freeze: dict) -> None:
    if value.get("format") != "sonara.aggression-runtime-attestation.v1":
        raise ValueError("invalid runtime attestation")
    if value.get("freeze_identity_sha256") != freeze["freeze_identity_sha256"]:
        raise ValueError("runtime attestation names a different freeze")
    if value.get("candidate_commit") != freeze["candidate"]["commit"]:
        raise ValueError("runtime attestation names a different candidate")
    if value.get("candidate_tree") != freeze["candidate"]["tree"]:
        raise ValueError("runtime attestation names a different candidate tree")
    probe = value.get("probe", {})
    if probe.get("model_id") != freeze["model"]["model_id"]:
        raise ValueError("runtime attestation model mismatch")
    if probe.get("native_sha256") not in value.get("site_manifest", {}).values():
        raise ValueError("attested native module is absent from sealed site")


def validate_control_manifests(
    robustness_rows: list[dict], non_music_rows: list[dict]
) -> dict:
    tolerances = {
        "gain": 0.03,
        "channel": 0.03,
        "resample": 0.03,
        "codec": 0.05,
        "quarter_removal": 0.10,
    }
    counts = {name: 0 for name in tolerances}
    case_ids = set()
    for row in robustness_rows:
        case_id, family = row.get("case_id"), row.get("family")
        if not isinstance(case_id, str) or case_id in case_ids or family not in tolerances:
            raise ValueError("invalid/duplicate robustness case")
        case_ids.add(case_id)
        if row.get("tolerance") != tolerances[family]:
            raise ValueError(f"robustness tolerance mismatch for {family}")
        if row.get("expected_direction") not in {"higher", "lower"}:
            raise ValueError("robustness case lacks frozen direction")
        for field in ("base_id", "variant_id", "contrast_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"robustness case lacks {field}")
        for field in ("base_content_hash", "variant_content_hash", "contrast_content_hash"):
            if not isinstance(row.get(field), str) or len(row[field]) != 64:
                raise ValueError(f"robustness case lacks {field}")
        counts[family] += 1
    if any(count < 20 for count in counts.values()):
        raise ValueError(f"insufficient raw robustness controls: {counts}")

    non_music_counts = {name: 0 for name in ("speech", "noise", "sparse")}
    control_ids = set()
    for row in non_music_rows:
        control_id, family = row.get("control_id"), row.get("family")
        if not isinstance(control_id, str) or control_id in control_ids or family not in non_music_counts:
            raise ValueError("invalid/duplicate non-music control")
        if not isinstance(row.get("content_hash"), str) or len(row["content_hash"]) != 64:
            raise ValueError("non-music control lacks full content identity")
        control_ids.add(control_id)
        non_music_counts[family] += 1
    if any(count < 20 for count in non_music_counts.values()):
        raise ValueError(f"insufficient raw non-music controls: {non_music_counts}")
    return {"robustness": counts, "non_music": non_music_counts}


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


def evaluator_artifact_identity(value: dict) -> str:
    material = {
        key: value.get(key)
        for key in (
            "evaluator_type", "architecture_family", "panel_member_commitments",
            "artifacts", "implementation_sha256", "protocol_sha256",
        )
    }
    return canonical_sha256(material)


def validate_evaluator(value: dict, freeze: dict, role: str) -> None:
    if value.get("format") != "sonara.aggression-evaluator.v2":
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
    if value["evaluator_type"] == "human":
        commitments = value.get("panel_member_commitments")
        if not isinstance(commitments, list) or len(set(commitments)) < 3:
            raise ValueError(f"{role}: human panel requires three identity commitments")
        if any(not isinstance(item, str) or len(item) != 64 for item in commitments):
            raise ValueError(f"{role}: invalid human-panel commitment")
    else:
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"{role}: audio evaluator artifacts are not bound")
        excluded_artifacts = set(
            freeze["evaluation_protocol"].get("excluded_evaluator_artifact_sha256", [])
        )
        for artifact in artifacts:
            path = Path(artifact.get("path", ""))
            expected = artifact.get("sha256")
            if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
                raise ValueError(f"{role}: evaluator artifact hash mismatch")
            if expected in excluded_artifacts:
                raise ValueError(f"{role}: evaluator reuses an excluded artifact")
    for field in ("implementation_sha256", "protocol_sha256"):
        if not isinstance(value.get(field), str) or len(value[field]) != 64:
            raise ValueError(f"{role}: missing {field}")


def validate_inputs(
    freeze: dict,
    manifest_path: Path,
    locked_paths_path: Path,
    labels_path: Path,
    pairs_path: Path,
    label_evaluator_path: Path,
    spot_evaluator_path: Path,
    spot_checks_path: Path,
    runtime_attestation_path: Path,
    robustness_cases_path: Path,
    robustness_paths_path: Path,
    non_music_cases_path: Path,
    non_music_paths_path: Path,
) -> dict:
    cohort = freeze["protected_cohort"]
    manifest = read_jsonl(manifest_path)
    if sha256(manifest_path) != cohort["manifest_sha256"]:
        raise ValueError("protected manifest hash mismatch")
    for field in ("sample_id", "content_hash", "acoustic_fingerprint"):
        values = [row.get(field) for row in manifest]
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"protected cohort has duplicate/invalid {field}")
    split_sets = {
        split: {row["sample_id"] for row in manifest if row.get("split") == split}
        for split in ("train", "development", "locked")
    }
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "development"), ("train", "locked"), ("development", "locked"))):
        raise ValueError("protected cohort splits overlap")
    for split, expected in cohort["splits"].items():
        rows = [row for row in manifest if row.get("split") == split]
        if len(rows) != expected["count"] or canonical_split_hash(manifest, split) != expected["identity_sha256"]:
            raise ValueError(f"protected {split} identity mismatch")
    locked_ids = {row["sample_id"] for row in manifest if row.get("split") == "locked"}
    locked_paths = read_json(locked_paths_path)
    if set(locked_paths) != locked_ids:
        raise ValueError("locked private-path map does not exactly cover cohort")

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
    if evaluator_artifact_identity(label_evaluator) == evaluator_artifact_identity(spot_evaluator):
        raise ValueError("evaluator artifact identities must be distinct")

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
    label_by_id = {item["sample_id"]: item for item in labels}
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
        left_target = float(label_by_id[left]["target"])
        right_target = float(label_by_id[right]["target"])
        margin = abs(left_target - right_target)
        expected_decision = "tie" if margin <= 0.04 else "left" if left_target > right_target else "right"
        if not math.isclose(float(row.get("left_target", math.nan)), left_target, abs_tol=1e-12) or not math.isclose(float(row.get("right_target", math.nan)), right_target, abs_tol=1e-12):
            raise ValueError("pair targets do not match frozen labels")
        if not math.isclose(float(row.get("margin", math.nan)), margin, abs_tol=1e-12) or decision != expected_decision:
            raise ValueError("pair decision/margin does not derive from frozen targets")
        valid_stratum = (
            (category == "tie" and margin <= 0.04)
            or (category == "hard" and margin >= 0.25 and row.get("context_distance", 1.0) <= 0.15)
            or (category == "near" and 0.08 <= margin <= 0.20)
            or (category == "broad" and margin >= 0.35)
        )
        if not valid_stratum:
            raise ValueError("pair does not satisfy its frozen stratum")
        used.extend((left, right))
        pair_by_id[row["pair_id"]] = row
    if categories != {"hard": 24, "near": 20, "broad": 20, "tie": 16}:
        raise ValueError(f"locked strata mismatch: {categories}")
    if len(set(used)) != 160 or set(used) != locked_ids:
        raise ValueError("locked pairs must use each recording exactly once")

    spot_checks = read_jsonl(spot_checks_path)
    if len(spot_checks) != 20 or len({row.get("pair_id") for row in spot_checks}) != 20:
        raise ValueError("exactly 20 preregistered spot checks are required")
    required_spot_ids = set()
    for category, count in (("hard", 8), ("near", 6), ("broad", 6)):
        candidates = sorted(
            (row["pair_id"] for row in pairs if row["category"] == category),
            key=lambda pair_id: hashlib.sha256(f"aggression-spot-v1:{pair_id}".encode()).hexdigest(),
        )
        required_spot_ids.update(candidates[:count])
    if {row.get("pair_id") for row in spot_checks} != required_spot_ids:
        raise ValueError("spot checks are not the preregistered stratified sample")
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

    runtime_attestation = read_json(runtime_attestation_path)
    validate_runtime_attestation(runtime_attestation, freeze)
    robustness_rows = read_jsonl(robustness_cases_path)
    non_music_rows = read_jsonl(non_music_cases_path)
    control_counts = validate_control_manifests(robustness_rows, non_music_rows)
    robustness_paths = read_json(robustness_paths_path)
    non_music_paths = read_json(non_music_paths_path)
    required_robustness_ids = {
        row[field]
        for row in robustness_rows
        for field in ("base_id", "variant_id", "contrast_id")
    }
    if set(robustness_paths) != required_robustness_ids:
        raise ValueError("robustness private-path map does not exactly cover controls")
    if set(non_music_paths) != {row["control_id"] for row in non_music_rows}:
        raise ValueError("non-music private-path map does not exactly cover controls")
    return {
        "counts": {"labels": len(labels), "pairs": len(pairs), "spot_checks": len(spot_checks), **categories},
        "independent_agreement": agreement_ratio,
        "label_evaluator_identity_sha256": evaluator_artifact_identity(label_evaluator),
        "spot_evaluator_identity_sha256": evaluator_artifact_identity(spot_evaluator),
        "control_counts": control_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-paths", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--label-evaluator", type=Path, required=True)
    parser.add_argument("--spot-evaluator", type=Path, required=True)
    parser.add_argument("--spot-checks", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
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
    if freeze.get("format") != "sonara.aggression-freeze.v1":
        raise ValueError("invalid candidate freeze")
    verify_declared_candidate(freeze)
    expected_scripts = freeze["evaluation_protocol"]
    if sha256(Path(__file__)) != expected_scripts["freeze_script_sha256"]:
        raise ValueError("protocol freezer changed after candidate freeze")
    if sha256(Path(__file__).with_name("evaluate_aggression_locked.py")) != expected_scripts["one_shot_script_sha256"]:
        raise ValueError("one-shot evaluator changed after candidate freeze")
    summary = validate_inputs(
        freeze,
        args.manifest,
        args.locked_paths,
        args.labels,
        args.pairs,
        args.label_evaluator,
        args.spot_evaluator,
        args.spot_checks,
        args.runtime_attestation,
        args.robustness_cases,
        args.robustness_paths,
        args.non_music_cases,
        args.non_music_paths,
    )
    custody = freeze.get("custody", {})
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
    input_hashes = {f"{name}_sha256": sha256(path) for name, path in input_paths.items()}
    evaluation_identity = canonical_sha256(
        {
            "freeze_identity_sha256": freeze["freeze_identity_sha256"],
            "candidate_commit": freeze["candidate"]["commit"],
            "cohort_manifest_sha256": freeze["protected_cohort"]["manifest_sha256"],
            "inputs": input_hashes,
        }
    )
    authorization = validate_custody_authorization(
        freeze, evaluation_identity, args.custody_authorization,
        args.custody_signature, args.allowed_signers,
    )
    protocol = {
        "format": "sonara.aggression-locked-protocol.v1",
        "freeze_sha256": sha256(args.freeze),
        "candidate_commit": freeze["candidate"]["commit"],
        "evaluation_identity": evaluation_identity,
        "inputs": input_hashes,
        "custody": {
            "authorization_sha256": sha256(args.custody_authorization),
            "signature_sha256": sha256(args.custody_signature),
            "allowed_signers_sha256": sha256(args.allowed_signers),
            "sequence": authorization["sequence"],
            "previous_entry_sha256": authorization["previous_entry_sha256"],
            "ledger_repository": authorization["ledger_repository"],
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
