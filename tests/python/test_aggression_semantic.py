"""Fail-closed semantic checks for the bundled aggression ranker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import sonara
from verify_aggression_custody import verify_result_attestation


def check_frozen_candidate() -> dict:
    freeze_path = ROOT / "tests/reference_data/aggression_v2_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    assert freeze["format"] == "sonara.aggression-freeze.v1"
    assert freeze["candidate"]["commit"] == "33d905acab2e2b38c86fbeb1aecfa76fb9ed9d2d"
    assert freeze["candidate"]["tree"] == "df1a306952b365ea71c9f619a6e5d9477a85138a"
    assert freeze["model"]["model_id"] == sonara.AGGRESSION_MODEL_ID
    assert freeze["model"]["analysis_schema_version"] == 5
    assert abs(freeze["model"]["tie_band"] - sonara.AGGRESSION_TIE_BAND) <= 1.0e-7
    for relative, expected in freeze["candidate"]["files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    scripts = freeze["evaluation_protocol"]
    assert hashlib.sha256(
        (ROOT / "scripts/freeze_aggression_locked_protocol.py").read_bytes()
    ).hexdigest() == scripts["freeze_script_sha256"]
    assert hashlib.sha256(
        (ROOT / "scripts/evaluate_aggression_locked.py").read_bytes()
    ).hexdigest() == scripts["one_shot_script_sha256"]
    for relative, expected in scripts["immutable_dependencies"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    performance = freeze["performance"]
    evidence_path = ROOT / performance["evidence_path"]
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == performance["evidence_sha256"]
    evidence = json.loads(evidence_path.read_text())
    assert evidence["candidate_commit"] == freeze["candidate"]["commit"]
    assert evidence["status"] == "pass"
    assert evidence["conservative_cross_bound_overhead_ratio"] <= evidence["criterion_maximum_ratio"]
    assert performance["conservative_cross_bound_overhead_ratio"] <= performance["maximum_allowed_ratio"]
    return freeze


def check_receipt(freeze: dict) -> None:
    receipt = json.loads(
        (ROOT / "tests/reference_data/aggression_v2_acceptance.json").read_text()
    )
    assert receipt["format"] == "sonara.aggression-acceptance.v1"
    assert receipt["model_id"] == sonara.AGGRESSION_MODEL_ID
    assert receipt["analysis_schema_version"] == 5
    for relative, expected in receipt["artifacts"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    development = receipt["development"]
    assert development["status"] == "pass"
    assert development["decisive_correct"] >= 52
    assert development["hard_correct"] >= 20
    assert development["tie_correct"] >= 12
    assert development["spearman"] >= 0.65
    assert development["mae"] <= 0.15
    assert development["score_range"] >= 0.65
    assert development["grouped_fold_spearman_min"] >= 0.65
    assert development["grouped_fold_mae_max"] <= 0.15
    assert 0.45 <= development["shuffled_directional_accuracy"] <= 0.55
    assert abs(development["shuffled_spearman"]) <= 0.10

    controls = receipt["physical_controls"]
    assert controls["status"] == "pass"
    assert controls["harsh_minus_loud_clean"] >= 0.30
    assert controls["silence_abstains"] is True
    assert controls["offline_rust_max_abs_error"] <= 0.0001

    locked = receipt["locked_evaluation"]
    assert locked["status"] == "pass", "sealed/pending locked evidence is not release acceptance"
    required = ("capsule_path", "receipt_path", "proof_path", "trust_root_path")
    assert all(isinstance(locked.get(field), str) for field in required)
    result = verify_result_attestation(
        ROOT / locked["capsule_path"], ROOT / locked["receipt_path"],
        ROOT / locked["proof_path"], ROOT / locked["trust_root_path"],
    )
    metrics = {row["name"]: float(row["value"]) for row in result["metrics"]}
    assert metrics["decisive-total"] == 64 and metrics["decisive-correct"] >= 52
    assert metrics["hard-total"] == 24 and metrics["hard-correct"] >= 20
    assert metrics["tie-total"] == 16 and metrics["tie-correct"] >= 12
    assert metrics["spearman"] >= 0.65
    assert metrics["mae"] <= 0.15
    assert metrics["score-range"] >= 0.65
    assert metrics["abstentions"] == 0
    assert metrics["spot-check-total"] >= 20
    assert metrics["independent-agreement"] >= 0.80
    assert metrics["evaluator-identities-distinct"] == 1

    for family in ("gain", "channel", "resample", "codec"):
        assert metrics[f"robustness-{family}-within-tolerance-ratio"] >= 0.95
        assert metrics[f"robustness-{family}-direction-preserved-ratio"] == 1.0
    assert metrics["robustness-quarter-removal-within-tolerance-ratio"] >= 0.90
    assert metrics["robustness-quarter-removal-direction-preserved-ratio"] == 1.0
    assert metrics["harsh-minus-loud-clean"] >= 0.30
    assert metrics["silence-abstains"] == 1
    for family in ("speech", "noise", "sparse"):
        assert metrics[f"non-music-{family}-count"] >= 1
        assert metrics[f"non-music-{family}-abstain-or-low-confidence-ratio"] >= 0.95


def synthesized_controls() -> None:
    sr = 22_050
    time = np.arange(sr * 8, dtype=np.float32) / sr
    rng = np.random.default_rng(20260722)
    harsh = 0.7 * np.tanh(
        8
        * (
            0.38 * np.sin(2 * np.pi * 110 * time)
            + 0.25 * np.sin(2 * np.pi * 173 * time)
            + 0.20 * rng.normal(size=time.size)
        )
    )
    phase = (time * 2) % 1
    loud_clean = 0.9 * np.exp(-phase * 12) * np.sin(2 * np.pi * 80 * time)
    calm = (
        0.18 * np.sin(2 * np.pi * 110 * time)
        + 0.14 * np.sin(2 * np.pi * 165 * time)
        + 0.10 * np.sin(2 * np.pi * 220 * time)
    )
    signals = {
        "harsh": harsh.astype(np.float32),
        "loud_clean": loud_clean.astype(np.float32),
        "calm": calm.astype(np.float32),
    }
    results = {
        name: sonara.analyze_aggression_signal(signal, sr=sr)
        for name, signal in signals.items()
    }
    harsh_score = results["harsh"]["aggression_score"]
    loud_score = results["loud_clean"]["aggression_score"]
    calm_score = results["calm"]["aggression_score"]
    assert harsh_score is not None and loud_score is not None and calm_score is not None
    assert harsh_score - loud_score >= 0.30
    assert harsh_score > calm_score
    assert results["harsh"]["aggression_harshness"] > results["loud_clean"]["aggression_harshness"]

    quiet = sonara.analyze_aggression_signal((signals["harsh"] * 0.25).astype(np.float32), sr=sr)
    assert quiet["aggression_score"] is not None
    assert abs(quiet["aggression_score"] - harsh_score) <= 0.03

    silence = sonara.analyze_aggression_signal(np.zeros(sr * 8, dtype=np.float32), sr=sr)
    assert silence["aggression_score"] is None
    assert silence["aggression_confidence"] == 0.0


freeze = check_frozen_candidate()
synthesized_controls()
check_receipt(freeze)
print("aggression semantic fidelity: PASS")
