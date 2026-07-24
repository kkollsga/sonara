"""Repository-owned semantic checks for the bundled aggression analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import sonara


def check_evidence() -> None:
    evidence = json.loads(
        (ROOT / "tests/reference_data/aggression_v2_evidence.json").read_text()
    )
    assert set(evidence) == {
        "format",
        "model_id",
        "analysis_schema_version",
        "feature_schema_sha256",
        "artifacts",
        "development",
        "physical_controls",
        "sample_rate_robustness",
    }
    assert evidence["format"] == "sonara.aggression-evidence.v1"
    assert evidence["model_id"] == sonara.AGGRESSION_MODEL_ID
    assert evidence["analysis_schema_version"] == 6
    for relative, expected in evidence["artifacts"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    development = evidence["development"]
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

    controls = evidence["physical_controls"]
    assert controls["status"] == "pass"
    assert controls["harsh_minus_loud_clean"] >= 0.30
    assert controls["silence_abstains"] is True
    assert controls["offline_rust_max_abs_error"] <= 0.0001

    robustness = evidence["sample_rate_robustness"]
    assert robustness["status"] == "pass"
    assert robustness["canonical_sample_rate"] == sonara.AGGRESSION_SAMPLE_RATE
    assert robustness["requested_sample_rates"] == [22_050, 32_000, 44_100, 48_000]
    file_routes = robustness["real_music_file_routes"]
    assert file_routes["track_count"] == 117
    assert file_routes["within_0_03_fraction"] >= 0.95
    assert file_routes["max_abs_delta"] <= 0.05
    assert file_routes["min_spearman"] >= 0.99
    assert file_routes["decisive_pair_flips"] == 0
    assert file_routes["component_p90_max_abs_delta"] <= 0.03
    anchors = robustness["named_file_anchors"]
    assert anchors["count"] == 6
    assert anchors["max_abs_delta"] <= 0.03
    assert anchors["directions_preserved"] == anchors["directions_total"] == 9


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


def check_performance_evidence() -> None:
    performance = json.loads(
        (ROOT / "tests/reference_data/aggression_v2_performance.json").read_text()
    )
    assert performance["format"] == "sonara.aggression-performance.v1"
    acceptance = performance["acceptance"]
    assert acceptance["status"] == "pass"
    assert (
        performance["compact_default"]["max_abs_change_percent"]
        <= acceptance["max_compact_abs_change_percent"]
    )
    assert (
        performance["compact_with_aggression_compiled_but_not_requested"][
            "max_abs_change_percent"
        ]
        <= acceptance["max_compact_abs_change_percent"]
    )
    assert (
        performance["requested_aggression_overhead"]["max_overhead_percent"]
        <= acceptance["max_requested_overhead_percent"]
    )
    assert (
        performance["canonical_requested_vs_v2"]["max_regression_percent"]
        <= acceptance["max_canonical_requested_regression_percent"]
    )
    paths = performance["sample_rate_paths"]
    assert set(paths["rates_hz"]) == {"22050", "32000", "44100", "48000"}
    assert (
        paths["max_standalone_over_canonical_plus_resample_ratio"]
        <= acceptance["max_noncanonical_composition_ratio"]
    )
    assert (
        paths["max_fused_over_native_plus_standalone_ratio"]
        <= acceptance["max_noncanonical_composition_ratio"]
    )
    assert performance["model_only_rank_midpoint_ns"] < 500.0
    assert performance["package"]["wheel_change_percent"] < 1.0
    assert all(performance["architecture"].values())


check_evidence()
synthesized_controls()
check_performance_evidence()
print("aggression semantic fidelity: PASS")
