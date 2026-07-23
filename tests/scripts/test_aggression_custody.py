#!/usr/bin/env python3
"""Adversarial tests for the aggression adapter over generic custody."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import sonara


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load("aggression_validation_adapter", ROOT / "scripts/aggression_validation_adapter.py")
VERIFY = load("verify_aggression_custody", ROOT / "scripts/verify_aggression_custody.py")


def write_json(path: Path, value: dict, *, canonical: bool = False) -> None:
    content = sonara.validation.canonical_json(value) if canonical else json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")


@unittest.skipUnless(shutil.which("ssh-keygen"), "ssh-keygen is required")
class AggressionCustodyTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict:
        output = sonara.validation.canonical_json({
            "disclosure": "aggregate_only",
            "evidence": [],
            "metrics": [{"name": "decisive-correct", "value": "52"}],
            "outcome": "pass",
        })
        if os.name == "nt":
            runner = root / "runner.cmd"
            runner.write_bytes(f"@echo off\r\n<nul set /p ={output}\r\n".encode())
        else:
            runner = root / "runner.sh"
            runner.write_bytes(f"#!/bin/sh\nprintf '%s' '{output}'\n".encode())
        evaluator_output = root / "labels.bin"
        evaluator_attestation = root / "evaluator-attestation.json"
        evaluator_output.write_bytes(b"sealed-independent-labels")
        evaluator_attestation.write_bytes(b'{"audio_perception_confirmed":true}')
        freeze = root / "freeze.json"
        freeze_value = {
            "candidate": {
                "binary_diff_sha256": "1" * 64,
                "commit": "2" * 40,
                "tree": "3" * 40,
            },
            "evaluation_protocol": {"required_thresholds": {"decisive_correct": 52, "mae_max": 0.15}},
            "format": "sonara.aggression-freeze.v1",
            "model": {"analysis_schema_version": 5, "model_id": "aggression-rank-v2"},
        }
        write_json(freeze, freeze_value)
        manifest = root / "manifest.json"
        manifest_value = {
            "command": {"arguments": [], "executable_id": runner.name},
            "evaluator": {
                "attestation_path": str(evaluator_attestation),
                "id": "independent-audio-rater",
                "kind": "executable",
                "output_path": str(evaluator_output),
            },
            "format": ADAPTER.MANIFEST_FORMAT,
            "resources": [
                {"id": runner.name, "path": str(runner), "role": "runtime-executable", "section": "runtime"},
                {"id": "labels.bin", "path": str(evaluator_output), "role": "evaluator-output", "section": "evaluator"},
                {"id": "evaluator-attestation.json", "path": str(evaluator_attestation), "role": "evaluator-attestation", "section": "evaluator"},
            ],
        }
        write_json(manifest, manifest_value)
        capsule, bindings, command = ADAPTER.build_artifacts(freeze, manifest)
        capsule_path = root / "capsule.json"
        write_json(capsule_path, capsule, canonical=True)

        key = root / "custodian"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
        public_key = subprocess.run(
            ["ssh-keygen", "-y", "-f", str(key)], check=True, capture_output=True, text=True
        ).stdout.strip()
        run = sonara.validation.run(
            capsule, bindings, command, ledger=root / "ledger.db", ledger_id="aggression-locked",
            private_key=key, principal="aggression-custodian",
        )
        receipt = root / "receipt.json"
        proof = root / "proof.json"
        trust_root = root / "trust-root.json"
        receipt.write_text(run.receipt_json, encoding="utf-8")
        proof.write_text(run.proof_json, encoding="utf-8")
        write_json(trust_root, {"principal": "aggression-custodian", "public_key_openssh": public_key}, canonical=True)
        return {
            "capsule": capsule_path, "receipt": receipt, "proof": proof, "trust_root": trust_root,
            "freeze": freeze, "manifest": manifest, "freeze_value": freeze_value,
        }

    def verify(self, fixture: dict) -> dict:
        return VERIFY.verify_result_attestation(
            fixture["capsule"], fixture["receipt"], fixture["proof"], fixture["trust_root"]
        )

    def test_generic_pass_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            receipt = self.verify(self.fixture(Path(raw)))
            self.assertEqual(receipt["outcome"], "pass")

    def test_modified_freeze_cannot_reuse_old_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self.fixture(Path(raw))
            changed = dict(fixture["freeze_value"])
            changed["model"] = {**changed["model"], "model_id": "changed-ranker"}
            write_json(fixture["freeze"], changed)
            capsule, _, _ = ADAPTER.build_artifacts(fixture["freeze"], fixture["manifest"])
            write_json(fixture["capsule"], capsule, canonical=True)
            with self.assertRaisesRegex(ValueError, "different aggression capsule"):
                self.verify(fixture)

    def test_receipt_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self.fixture(Path(raw))
            receipt = json.loads(fixture["receipt"].read_text())
            receipt["metrics"][0]["value"] = "51"
            write_json(fixture["receipt"], receipt, canonical=True)
            with self.assertRaises(ValueError):
                self.verify(fixture)

    def test_wrong_trust_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self.fixture(Path(raw))
            root = json.loads(fixture["trust_root"].read_text())
            root["principal"] = "other-custodian"
            write_json(fixture["trust_root"], root, canonical=True)
            with self.assertRaises(ValueError):
                self.verify(fixture)

    def test_public_artifacts_do_not_disclose_private_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = self.fixture(Path(raw))
            private_root = str(Path(raw).resolve())
            for name in ("capsule", "receipt", "proof"):
                content = fixture[name].read_text(encoding="utf-8")
                self.assertNotIn(private_root, content)
                self.assertNotIn("sealed-independent-labels", content)


if __name__ == "__main__":
    unittest.main()
