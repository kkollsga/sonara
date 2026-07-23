#!/usr/bin/env python3
"""Python API and installed-command contract for generic validation custody."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import sonara


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def digest(content: bytes) -> dict[str, object]:
    return {"algorithm": "sha256", "value": sha256(content)}


def artifact(identity: str, role: str, content: bytes) -> dict[str, object]:
    return {
        "digest": digest(content),
        "id": identity,
        "role": role,
        "size_bytes": len(content),
    }


with tempfile.TemporaryDirectory(prefix="sonara-validation-python-") as temp:
    root = Path(temp)
    output = sonara.validation.canonical_json({
        "disclosure": "aggregate_only",
        "evidence": [],
        "metrics": [{"name": "python-score", "value": "1"}],
        "outcome": "pass",
    })
    if os.name == "nt":
        runner_name = "runner.cmd"
        runner = f"@echo off\r\n<nul set /p ={output}\r\n".encode()
    else:
        runner_name = "runner.sh"
        runner = f"#!/bin/sh\nprintf '%s' '{output}'\n".encode()
    private_input = b"private-evaluator-input"
    runner_path = root / runner_name
    input_path = root / "input.bin"
    runner_path.write_bytes(runner)
    input_path.write_bytes(private_input)

    capsule = {
        "artifacts": [],
        "candidate": {
            "binary_diff": digest(b"candidate-diff"),
            "commit": "a" * 40,
            "tree": "b" * 40,
        },
        "command_digest": digest(
            b"sonara-validation-command-v1\0" + sonara.validation.canonical_json(
                {"arguments": [], "executable_id": runner_name}
            ).encode()
        ),
        "feature": "python-validation",
        "format": "sonara.validation-capsule.v1",
        "model_id": "python-candidate",
        "runtime": {
            "artifacts": [artifact(runner_name, "runtime-executable", runner)],
            "model_id": "python-candidate",
            "schema_version": 1,
        },
        "suite": {
            "controls": [],
            "evaluator": {
                "artifacts": [artifact("input.bin", "evaluator-input", private_input)],
                "attestation_digest": digest(b"attestation"),
                "id": "python-evaluator",
                "kind": "executable",
                "output_digest": digest(b"expected-output"),
            },
            "kind": "python-suite",
            "schema_version": 1,
            "thresholds": [],
            "tooling": [],
        },
    }
    bindings = {
        "format": "sonara.validation-bindings.v1",
        "resources": [
            {"id": runner_name, "path": str(runner_path)},
            {"id": "input.bin", "path": str(input_path)},
        ],
    }
    command = {"arguments": [], "executable_id": runner_name}
    capsule_path = root / "capsule.json"
    bindings_path = root / "bindings.json"
    capsule_path.write_text(sonara.validation.canonical_json(capsule), encoding="utf-8")
    bindings_path.write_text(json.dumps(bindings, indent=2), encoding="utf-8")

    prepared = sonara.validation.prepare(capsule_path, bindings_path)
    assert prepared.resource_count == 2
    assert prepared.evaluation_digest == sonara.validation.capsule_digest(capsule_path)

    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "sonara.cli",
            "validate",
            "prepare",
            "--capsule",
            str(capsule_path),
            "--bindings",
            str(bindings_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    expected_prepared = sonara.validation.canonical_json({
        "evaluation_digest": {"algorithm": "sha256", "value": prepared.evaluation_digest},
        "format": "sonara.prepared-validation.v1",
        "resource_count": 2,
    })
    assert cli.stdout == expected_prepared

    key_path = root / "custodian-key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
        check=True,
    )
    public_key = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(key_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run = sonara.validation.run(
        capsule,
        bindings,
        command,
        ledger=root / "ledger.db",
        ledger_id="python-ledger",
        private_key=key_path,
        principal="python-custodian",
    )
    assert run.receipt["outcome"] == "pass"
    assert run.receipt["evaluation_digest"]["value"] == prepared.evaluation_digest

    trust_root = {
        "principal": "python-custodian",
        "public_key_openssh": public_key,
    }
    runner_path.unlink()
    input_path.unlink()
    sonara.validation.verify(run.receipt, run.proof, trust_root)

    tampered = json.loads(run.receipt_json)
    tampered["metrics"][0]["value"] = "0"
    try:
        sonara.validation.verify(tampered, run.proof, trust_root)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered receipt was accepted")

    noncanonical = root / "noncanonical-capsule.json"
    noncanonical.write_text(sonara.validation.canonical_json(capsule) + "\n", encoding="utf-8")
    try:
        sonara.validation.capsule_digest(noncanonical)
    except ValueError:
        pass
    else:
        raise AssertionError("non-canonical capsule bytes were accepted")

print("validation Python API/CLI: PASS")
