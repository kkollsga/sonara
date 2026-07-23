#!/usr/bin/env python3
"""Verify an aggression receipt through Sonara's generic custody protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonara import validation


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def verify_result_attestation(
    capsule_path: Path,
    receipt_path: Path,
    proof_path: Path,
    trust_root_path: Path,
) -> dict:
    capsule = read_json(capsule_path)
    receipt = read_json(receipt_path)
    validation.verify(receipt_path, proof_path, trust_root_path)
    if capsule.get("feature") != "aggression-ranking":
        raise ValueError("capsule is not an aggression-ranking evaluation")
    if receipt.get("evaluation_digest", {}).get("value") != validation.capsule_digest(capsule_path):
        raise ValueError("receipt names a different aggression capsule")
    if receipt.get("outcome") != "pass":
        raise ValueError("aggression receipt is not PASS")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--trust-root", type=Path, required=True)
    args = parser.parse_args()
    verify_result_attestation(
        args.capsule, args.receipt, args.proof, args.trust_root,
    )
    print("aggression validation result: VERIFIED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"custodian result rejected: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
