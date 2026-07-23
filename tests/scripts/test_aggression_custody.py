#!/usr/bin/env python3
"""Adversarial tests for custodian-authorized aggression results."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "verify_aggression_custody", ROOT / "scripts/verify_aggression_custody.py"
)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
import freeze_aggression_locked_protocol as FREEZE


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@unittest.skipUnless(shutil.which("ssh-keygen"), "ssh-keygen is required")
class CustodyTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, Path]:
        key = root / "custodian"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
            check=True,
        )
        identity = "sonagram-aggression-custodian"
        namespace = "sonara-aggression-v2"
        allowed = root / "allowed_signers"
        allowed.write_text(f"{identity} {key.with_suffix('.pub').read_text()}", encoding="utf-8")
        authorization_hash = "a" * 64
        freeze = {
            "candidate": {"commit": "c" * 40},
            "custody": {
                "status": "ready",
                "allowed_signers_sha256": hashlib.sha256(allowed.read_bytes()).hexdigest(),
                "signer_identity": identity,
                "signature_namespace": namespace,
                "ledger_repository": "https://github.com/kkollsga/sonagram.git",
            },
        }
        protocol = {
            "evaluation_identity": "e" * 64,
            "custody": {"authorization_sha256": authorization_hash, "sequence": 4},
        }
        paths = {name: root / f"{name}.json" for name in ("freeze", "protocol", "result", "attestation")}
        paths.update({"allowed": allowed, "key": key})
        write(paths["freeze"], freeze)
        write(paths["protocol"], protocol)
        result = {
            "format": "sonara.aggression-locked-result.v2",
            "candidate_commit": freeze["candidate"]["commit"],
            "evaluation_identity": protocol["evaluation_identity"],
            "protocol_sha256": hashlib.sha256(paths["protocol"].read_bytes()).hexdigest(),
            "custody_authorization_sha256": authorization_hash,
            **{name: {"status": "pass"} for name in ("locked_evaluation", "independence", "robustness", "non_music")},
        }
        write(paths["result"], result)
        attestation = {
            "format": "sonagram.aggression-result-attestation.v1",
            "evaluation_identity": protocol["evaluation_identity"],
            "protocol_sha256": hashlib.sha256(paths["protocol"].read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256(paths["result"].read_bytes()).hexdigest(),
            "custody_authorization_sha256": authorization_hash,
            "previous_entry_sha256": authorization_hash,
            "ledger_repository": freeze["custody"]["ledger_repository"],
            "action": "accept-first-and-final-result",
            "cohort_retired": True,
            "sequence": 5,
        }
        write(paths["attestation"], attestation)
        subprocess.run(
            ["ssh-keygen", "-q", "-Y", "sign", "-f", str(key), "-n", namespace, str(paths["attestation"])],
            check=True,
        )
        paths["signature"] = Path(f"{paths['attestation']}.sig")
        return paths

    def verify(self, paths: dict[str, Path]):
        return VERIFY.verify_result_attestation(
            paths["freeze"], paths["protocol"], paths["result"], paths["attestation"],
            paths["signature"], paths["allowed"],
        )

    def test_signed_first_and_final_result_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            self.verify(self.fixture(Path(raw)))

    def test_result_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = self.fixture(Path(raw))
            result = json.loads(paths["result"].read_text())
            result["locked_evaluation"]["decisive_correct"] = 64
            write(paths["result"], result)
            with self.assertRaisesRegex(ValueError, "result attestation mismatch"):
                self.verify(paths)

    def test_signature_replay_on_new_attestation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = self.fixture(Path(raw))
            attestation = json.loads(paths["attestation"].read_text())
            attestation["sequence"] += 1
            write(paths["attestation"], attestation)
            with self.assertRaisesRegex(ValueError, "signature verification failed"):
                self.verify(paths)

    def test_authorization_cannot_be_retargeted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = self.fixture(Path(raw))
            freeze = json.loads(paths["freeze"].read_text())
            freeze["protected_cohort"] = {"manifest_sha256": "m" * 64}
            write(paths["freeze"], freeze)
            authorization = {
                "format": "sonagram.aggression-custody-authorization.v1",
                "evaluation_identity": "e" * 64,
                "candidate_commit": freeze["candidate"]["commit"],
                "cohort_manifest_sha256": "m" * 64,
                "action": "retire-cohort-and-authorize-one-candidate",
                "cohort_retired": True,
                "ledger_repository": freeze["custody"]["ledger_repository"],
                "sequence": 4,
                "previous_entry_sha256": "0" * 64,
            }
            auth_path = Path(raw) / "authorization.json"
            write(auth_path, authorization)
            subprocess.run(
                ["ssh-keygen", "-q", "-Y", "sign", "-f", str(paths["key"]), "-n", freeze["custody"]["signature_namespace"], str(auth_path)],
                check=True,
            )
            signature = Path(f"{auth_path}.sig")
            FREEZE.validate_custody_authorization(
                freeze, "e" * 64, auth_path, signature, paths["allowed"]
            )
            authorization["evaluation_identity"] = "x" * 64
            write(auth_path, authorization)
            with self.assertRaisesRegex(ValueError, "authorization mismatch"):
                FREEZE.validate_custody_authorization(
                    freeze, "e" * 64, auth_path, signature, paths["allowed"]
                )


if __name__ == "__main__":
    unittest.main()
