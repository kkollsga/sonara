#!/usr/bin/env python3
"""Adversarial contracts for the sealed aggression-v2 inputs."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "freeze_aggression_locked_protocol",
    ROOT / "scripts/freeze_aggression_locked_protocol.py",
)
assert SPEC is not None and SPEC.loader is not None
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class LockedProtocolTests(unittest.TestCase):
    def fixture(self, root: Path):
        manifest = []
        for index in range(1024):
            split = "train" if index < 704 else "development" if index < 864 else "locked"
            manifest.append({
                "sample_id": f"rank-{index:04d}",
                "content_hash": f"{index:064x}",
                "acoustic_fingerprint": f"fingerprint-{index:04d}",
                "duration_sec": 100.0,
                "split": split,
            })
        manifest_path = root / "manifest.jsonl"
        write_jsonl(manifest_path, manifest)
        locked = [row["sample_id"] for row in manifest if row["split"] == "locked"]
        locked_paths = {sample_id: f"/sealed/{sample_id}.wav" for sample_id in locked}
        freeze = {
            "format": "sonara.aggression-freeze.v1",
            "freeze_identity_sha256": "f" * 64,
            "candidate": {"commit": "a" * 40, "tree": "b" * 40},
            "model": {"model_id": "aggression-rank-v2"},
            "protected_cohort": {
                "manifest_sha256": PROTOCOL.sha256(manifest_path),
                "splits": {
                    split: {
                        "count": sum(row["split"] == split for row in manifest),
                        "identity_sha256": PROTOCOL.canonical_split_hash(manifest, split),
                    }
                    for split in ("train", "development", "locked")
                },
            },
            "evaluation_protocol": {
                "excluded_evaluator_family_sha256": [],
                "excluded_evaluator_artifact_sha256": [],
            },
        }
        evaluator_base = {
            "format": "sonara.aggression-evaluator.v2",
            "evaluator_type": "human",
            "architecture_family": "human-panel",
            "audio_perception_confirmed": True,
            "independent_of_candidate": True,
            "independent_of_development_evaluator": True,
            "implementation_sha256": "1" * 64,
            "protocol_sha256": "2" * 64,
        }
        label_evaluator = {
            **evaluator_base,
            "evaluator_id": "label-panel",
            "panel_member_commitments": ["3" * 64, "4" * 64, "5" * 64],
        }
        spot_evaluator = {
            **evaluator_base,
            "evaluator_id": "spot-panel",
            "panel_member_commitments": ["6" * 64, "7" * 64, "8" * 64],
        }
        categories = ["hard"] * 24 + ["near"] * 20 + ["broad"] * 20 + ["tie"] * 16
        target_pairs = {"hard": (0.1, 0.5), "near": (0.4, 0.54), "broad": (0.1, 0.7), "tie": (0.5, 0.52)}
        labels, pairs = [], []
        for index, category in enumerate(categories):
            left_id, right_id = locked[index * 2 : index * 2 + 2]
            left_target, right_target = target_pairs[category]
            labels.extend([
                {"sample_id": left_id, "target": left_target, "evaluator_id": "label-panel"},
                {"sample_id": right_id, "target": right_target, "evaluator_id": "label-panel"},
            ])
            margin = abs(left_target - right_target)
            pairs.append({
                "pair_id": f"pair-{index:03d}",
                "left_id": left_id,
                "right_id": right_id,
                "left_target": left_target,
                "right_target": right_target,
                "margin": margin,
                "context_distance": 0.1,
                "category": category,
                "decision": "tie" if category == "tie" else "right",
            })
        spot_ids = set()
        for category, count in (("hard", 8), ("near", 6), ("broad", 6)):
            ids = sorted(
                (row["pair_id"] for row in pairs if row["category"] == category),
                key=lambda pair_id: hashlib.sha256(f"aggression-spot-v1:{pair_id}".encode()).hexdigest(),
            )
            spot_ids.update(ids[:count])
        spot_checks = [
            {"pair_id": pair_id, "decision": "right", "evaluator_id": "spot-panel"}
            for pair_id in sorted(spot_ids)
        ]
        runtime = {
            "format": "sonara.aggression-runtime-attestation.v1",
            "freeze_identity_sha256": freeze["freeze_identity_sha256"],
            "candidate_commit": freeze["candidate"]["commit"],
            "candidate_tree": freeze["candidate"]["tree"],
            "site_manifest": {"sonara/_sonara.so": "9" * 64},
            "probe": {"model_id": "aggression-rank-v2", "native_sha256": "9" * 64},
        }
        robustness = []
        robustness_paths = {}
        for family in ("gain", "channel", "resample", "codec", "quarter_removal"):
            tolerance = 0.10 if family == "quarter_removal" else 0.05 if family == "codec" else 0.03
            for index in range(20):
                ids = {role: f"{family}-{index}-{role}" for role in ("base", "variant", "contrast")}
                hashes = {role: hashlib.sha256(value.encode()).hexdigest() for role, value in ids.items()}
                robustness.append({
                    "case_id": f"{family}-{index}",
                    "family": family,
                    "tolerance": tolerance,
                    "expected_direction": "higher",
                    **{f"{role}_id": value for role, value in ids.items()},
                    **{f"{role}_content_hash": hashes[role] for role in ids},
                })
                robustness_paths.update({value: f"/sealed/{value}.wav" for value in ids.values()})
        non_music = []
        non_music_paths = {}
        for family in ("speech", "noise", "sparse"):
            for index in range(20):
                control_id = f"{family}-{index}"
                non_music.append({"control_id": control_id, "family": family, "content_hash": hashlib.sha256(control_id.encode()).hexdigest()})
                non_music_paths[control_id] = f"/sealed/{control_id}.wav"
        values = {
            "locked_paths": locked_paths,
            "labels": labels,
            "pairs": pairs,
            "label_evaluator": label_evaluator,
            "spot_evaluator": spot_evaluator,
            "spot_checks": spot_checks,
            "runtime_attestation": runtime,
            "robustness_cases": robustness,
            "robustness_paths": robustness_paths,
            "non_music_cases": non_music,
            "non_music_paths": non_music_paths,
        }
        paths = {"manifest": manifest_path}
        jsonl_names = {"labels", "pairs", "spot_checks", "robustness_cases", "non_music_cases"}
        for name, value in values.items():
            path = root / f"{name}.{'jsonl' if name in jsonl_names else 'json'}"
            (write_jsonl if name in jsonl_names else write_json)(path, value)
            paths[name] = path
        return freeze, paths

    def validate(self, freeze: dict, paths: dict[str, Path]):
        return PROTOCOL.validate_inputs(
            freeze,
            paths["manifest"], paths["locked_paths"], paths["labels"], paths["pairs"],
            paths["label_evaluator"], paths["spot_evaluator"], paths["spot_checks"],
            paths["runtime_attestation"], paths["robustness_cases"], paths["robustness_paths"],
            paths["non_music_cases"], paths["non_music_paths"],
        )

    def test_valid_protocol_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            result = self.validate(freeze, paths)
            self.assertEqual(result["counts"]["pairs"], 80)
            self.assertEqual(result["independent_agreement"], 1.0)

    def test_evaluator_alias_is_rejected_by_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            evaluator = copy.deepcopy(json.loads(paths["label_evaluator"].read_text()))
            evaluator["evaluator_id"] = "alias"
            write_json(paths["spot_evaluator"], evaluator)
            checks = PROTOCOL.read_jsonl(paths["spot_checks"])
            for row in checks:
                row["evaluator_id"] = "alias"
            write_jsonl(paths["spot_checks"], checks)
            with self.assertRaisesRegex(ValueError, "artifact identities"):
                self.validate(freeze, paths)

    def test_pair_stratum_cannot_be_forged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            pairs = PROTOCOL.read_jsonl(paths["pairs"])
            pairs[0]["margin"] = 0.01
            write_jsonl(paths["pairs"], pairs)
            with self.assertRaisesRegex(ValueError, "decision/margin"):
                self.validate(freeze, paths)

    def test_spot_checks_cannot_be_cherry_picked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            checks = PROTOCOL.read_jsonl(paths["spot_checks"])
            checks[0]["pair_id"] = "pair-023"
            write_jsonl(paths["spot_checks"], checks)
            with self.assertRaisesRegex(ValueError, "preregistered stratified"):
                self.validate(freeze, paths)

    def test_cohort_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            manifest = PROTOCOL.read_jsonl(paths["manifest"])
            manifest[-1]["sample_id"] = manifest[0]["sample_id"]
            write_jsonl(paths["manifest"], manifest)
            freeze["protected_cohort"]["manifest_sha256"] = PROTOCOL.sha256(paths["manifest"])
            with self.assertRaisesRegex(ValueError, "duplicate/invalid|identity mismatch|exactly cover"):
                self.validate(freeze, paths)


if __name__ == "__main__":
    unittest.main()
