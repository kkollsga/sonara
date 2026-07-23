#!/usr/bin/env python3
"""Contract tests for sealed aggression-v2 evaluation inputs."""

from __future__ import annotations

import copy
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
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class LockedProtocolTests(unittest.TestCase):
    def fixture(self, root: Path):
        manifest = []
        for index in range(1024):
            split = "train" if index < 704 else "development" if index < 864 else "locked"
            manifest.append({
                "sample_id": f"rank-{index:04d}",
                "content_hash": f"content-{index:04d}",
                "acoustic_fingerprint": f"fingerprint-{index:04d}",
                "duration_sec": 100.0,
                "split": split,
            })
        manifest_path = root / "manifest.jsonl"
        write_jsonl(manifest_path, manifest)
        freeze = {
            "format": "sonara.aggression-freeze.v1",
            "candidate": {"commit": "a" * 40},
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
            "evaluation_protocol": {"excluded_evaluator_family_sha256": []},
        }
        locked = [row["sample_id"] for row in manifest if row["split"] == "locked"]
        label_evaluator = {
            "format": "sonara.aggression-evaluator.v1",
            "evaluator_id": "label-rater",
            "evaluator_type": "audio_model",
            "architecture_family": "independent-audio-family-a",
            "audio_perception_confirmed": True,
            "independent_of_candidate": True,
            "independent_of_development_evaluator": True,
        }
        spot_evaluator = {
            **label_evaluator,
            "evaluator_id": "spot-rater",
            "evaluator_type": "human",
            "architecture_family": "human-panel",
        }
        labels = [
            {"sample_id": sample_id, "target": index / 159, "evaluator_id": "label-rater"}
            for index, sample_id in enumerate(locked)
        ]
        categories = ["hard"] * 24 + ["near"] * 20 + ["broad"] * 20 + ["tie"] * 16
        pairs = []
        for index, category in enumerate(categories):
            pairs.append({
                "pair_id": f"pair-{index:03d}",
                "left_id": locked[index * 2],
                "right_id": locked[index * 2 + 1],
                "category": category,
                "decision": "tie" if category == "tie" else "left",
            })
        spot_checks = [
            {"pair_id": f"pair-{index:03d}", "decision": "left", "evaluator_id": "spot-rater"}
            for index in range(20)
        ]
        robustness = {
            "format": "sonara.aggression-robustness.v1",
            "status": "pass",
            "gain_channel_resample_within_0_03_ratio": 0.95,
            "codec_within_0_05_ratio": 0.95,
            "quarter_removal_within_0_10_ratio": 0.90,
            "direction_preserved": True,
            "harsh_minus_loud_clean": 0.30,
            "silence_abstains": True,
        }
        non_music = {
            "format": "sonara.aggression-non-music.v1",
            "status": "pass",
            "families": {
                name: {"count": 20, "abstain_or_low_confidence_ratio": 0.95}
                for name in ("speech", "noise", "sparse")
            },
        }
        values = {
            "labels": labels,
            "pairs": pairs,
            "label_evaluator": label_evaluator,
            "spot_evaluator": spot_evaluator,
            "spot_checks": spot_checks,
            "robustness": robustness,
            "non_music": non_music,
        }
        paths = {"manifest": manifest_path}
        for name, value in values.items():
            path = root / f"{name}.jsonl" if name in {"labels", "pairs", "spot_checks"} else root / f"{name}.json"
            (write_jsonl if path.suffix == ".jsonl" else write_json)(path, value)
            paths[name] = path
        return freeze, paths

    def validate(self, freeze: dict, paths: dict[str, Path]):
        return PROTOCOL.validate_inputs(
            freeze,
            paths["manifest"],
            paths["labels"],
            paths["pairs"],
            paths["label_evaluator"],
            paths["spot_evaluator"],
            paths["spot_checks"],
            paths["robustness"],
            paths["non_music"],
        )

    def test_valid_protocol_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            result = self.validate(freeze, paths)
            self.assertEqual(result["counts"]["pairs"], 80)
            self.assertEqual(result["independent_agreement"], 1.0)

    def test_development_evaluator_family_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            evaluator = json.loads(paths["label_evaluator"].read_text())
            family_hash = __import__("hashlib").sha256(
                evaluator["architecture_family"].encode()
            ).hexdigest()
            freeze["evaluation_protocol"]["excluded_evaluator_family_sha256"] = [family_hash]
            with self.assertRaisesRegex(ValueError, "not independent"):
                self.validate(freeze, paths)

    def test_locked_strata_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            pairs = PROTOCOL.read_jsonl(paths["pairs"])
            pairs[0]["category"] = "near"
            write_jsonl(paths["pairs"], pairs)
            with self.assertRaisesRegex(ValueError, "strata mismatch"):
                self.validate(freeze, paths)

    def test_spot_evaluator_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            freeze, paths = self.fixture(Path(raw))
            evaluator = copy.deepcopy(json.loads(paths["label_evaluator"].read_text()))
            write_json(paths["spot_evaluator"], evaluator)
            with self.assertRaisesRegex(ValueError, "distinct identity"):
                self.validate(freeze, paths)


if __name__ == "__main__":
    unittest.main()
