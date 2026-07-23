#!/usr/bin/env python3
"""Verify Sonagram's signature over the first-and-final aggression result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freeze_aggression_locked_protocol import verify_ssh_signature


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def verify_result_attestation(
    freeze_path: Path,
    protocol_path: Path,
    result_path: Path,
    attestation_path: Path,
    signature_path: Path,
    allowed_signers_path: Path,
) -> dict:
    freeze = read_json(freeze_path)
    protocol = read_json(protocol_path)
    result = read_json(result_path)
    attestation = read_json(attestation_path)
    custody = freeze.get("custody", {})
    if custody.get("status") != "ready":
        raise ValueError("Sonagram custodian root is not frozen")
    if sha256(allowed_signers_path) != custody.get("allowed_signers_sha256"):
        raise ValueError("custodian public-key root mismatch")
    if result.get("format") != "sonara.aggression-locked-result.v2":
        raise ValueError("invalid locked result")
    expected_result = {
        "candidate_commit": freeze["candidate"]["commit"],
        "evaluation_identity": protocol["evaluation_identity"],
        "protocol_sha256": sha256(protocol_path),
        "custody_authorization_sha256": protocol["custody"]["authorization_sha256"],
    }
    for field, expected in expected_result.items():
        if result.get(field) != expected:
            raise ValueError(f"locked result substitution: {field}")
    for section in ("locked_evaluation", "independence", "robustness", "non_music"):
        if result.get(section, {}).get("status") != "pass":
            raise ValueError(f"locked result is not PASS: {section}")
    expected_attestation = {
        "format": "sonagram.aggression-result-attestation.v1",
        "evaluation_identity": protocol["evaluation_identity"],
        "protocol_sha256": sha256(protocol_path),
        "result_sha256": sha256(result_path),
        "custody_authorization_sha256": protocol["custody"]["authorization_sha256"],
        "previous_entry_sha256": protocol["custody"]["authorization_sha256"],
        "ledger_repository": custody["ledger_repository"],
        "action": "accept-first-and-final-result",
        "cohort_retired": True,
    }
    for field, expected in expected_attestation.items():
        if attestation.get(field) != expected:
            raise ValueError(f"result attestation mismatch: {field}")
    if not isinstance(attestation.get("sequence"), int) or attestation["sequence"] <= protocol["custody"]["sequence"]:
        raise ValueError("result attestation does not advance the custody ledger")
    verify_ssh_signature(
        attestation_path, signature_path, allowed_signers_path,
        custody["signer_identity"], custody["signature_namespace"],
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--allowed-signers", type=Path, required=True)
    args = parser.parse_args()
    verify_result_attestation(
        args.freeze, args.protocol, args.result, args.attestation,
        args.signature, args.allowed_signers,
    )
    print(f"custodian result: VERIFIED sha256={sha256(args.result)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"custodian result rejected: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
