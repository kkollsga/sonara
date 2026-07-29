#!/usr/bin/env python3
"""Run the frozen 117-clip aggression sample-rate acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sonara


ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "dev-docs/bench/out/ferricml-aggression-audit"
OUTPUT = ROOT / "dev-docs/bench/out/aggression-sample-rate-v3.json"
RATES = (22_050, 32_000, 44_100, 48_000)
COMPONENTS = (
    "aggression_confidence",
    "aggression_forcefulness",
    "aggression_harshness",
    "aggression_tension",
    "aggression_rhythm",
)
# Selected prospectively and spread across the immutable manifest order.
ANCHORS = ("audit-001", "audit-020", "audit-040", "audit-060", "audit-080", "audit-117")


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranked


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def summarize(by_rate, audit_ids):
    reference = np.asarray(
        [result["aggression_score"] for result in by_rate[22_050]], dtype=np.float64
    )
    anchor_indices = [audit_ids.index(anchor) for anchor in ANCHORS]
    per_rate = {}
    all_score_deltas = []
    all_component_deltas = {component: [] for component in COMPONENTS}
    for rate in RATES[1:]:
        candidate = np.asarray(
            [result["aggression_score"] for result in by_rate[rate]], dtype=np.float64
        )
        deltas = np.abs(candidate - reference)
        all_score_deltas.extend(deltas.tolist())
        flips = 0
        decisive = 0
        for left in range(len(reference)):
            for right in range(left + 1, len(reference)):
                margin = reference[left] - reference[right]
                if abs(margin) > 0.05:
                    decisive += 1
                    flips += int(np.sign(margin) != np.sign(candidate[left] - candidate[right]))
        component_p90 = {}
        for component in COMPONENTS:
            component_reference = np.asarray(
                [result[component] for result in by_rate[22_050]], dtype=np.float64
            )
            component_candidate = np.asarray(
                [result[component] for result in by_rate[rate]], dtype=np.float64
            )
            component_deltas = np.abs(component_candidate - component_reference)
            all_component_deltas[component].extend(component_deltas.tolist())
            component_p90[component] = float(np.quantile(component_deltas, 0.90))
        per_rate[str(rate)] = {
            "within_0_03_fraction": float(np.mean(deltas <= 0.03)),
            "median_abs_delta": float(np.median(deltas)),
            "p90_abs_delta": float(np.quantile(deltas, 0.90)),
            "max_abs_delta": float(np.max(deltas)),
            "spearman": spearman(reference, candidate),
            "decisive_pairs": decisive,
            "direction_flips": flips,
            "component_p90_abs_delta": component_p90,
            "anchor_max_abs_delta": float(np.max(deltas[anchor_indices])),
        }

    aggregate = {
        "within_0_03_fraction": float(np.mean(np.asarray(all_score_deltas) <= 0.03)),
        "max_abs_delta": float(np.max(all_score_deltas)),
        "min_spearman": min(result["spearman"] for result in per_rate.values()),
        "direction_flips": sum(result["direction_flips"] for result in per_rate.values()),
        "component_p90_abs_delta": {
            component: float(np.quantile(deltas, 0.90))
            for component, deltas in all_component_deltas.items()
        },
        "anchor_max_abs_delta": max(
            result["anchor_max_abs_delta"] for result in per_rate.values()
        ),
    }
    return per_rate, aggregate, {
        "at_least_95_percent_within_0_03": aggregate["within_0_03_fraction"] >= 0.95,
        "max_at_most_0_05": aggregate["max_abs_delta"] <= 0.05,
        "spearman_at_least_0_99": aggregate["min_spearman"] >= 0.99,
        "zero_decisive_pair_flips": aggregate["direction_flips"] == 0,
        "component_p90_at_most_0_03": max(aggregate["component_p90_abs_delta"].values()) <= 0.03,
        "anchors_at_most_0_03": aggregate["anchor_max_abs_delta"] <= 0.03,
    }


def main() -> int:
    manifest = [json.loads(line) for line in (POOL / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 117
    paths = [str(POOL / row["clip"]) for row in manifest]
    audit_ids = [row["audit_id"] for row in manifest]

    file_by_rate = {}
    for rate in RATES:
        results = sonara.analyze_aggression_batch(paths, sr=rate)
        assert len(results) == len(paths)
        assert all("error" not in result for result in results)
        assert all(result["aggression_score"] is not None for result in results)
        file_by_rate[rate] = results
        print(f"file route: analyzed {len(results)} clips at {rate} Hz", flush=True)
    file_per_rate, file_aggregate, file_acceptance = summarize(file_by_rate, audit_ids)

    canonical_audio = [
        sonara.load(path, sr=sonara.AGGRESSION_SAMPLE_RATE)[0] for path in paths
    ]
    signal_by_rate = {}
    for rate in RATES:
        results = []
        for audio in canonical_audio:
            signal = (
                audio
                if rate == sonara.AGGRESSION_SAMPLE_RATE
                else sonara.resample(
                    audio,
                    orig_sr=sonara.AGGRESSION_SAMPLE_RATE,
                    target_sr=rate,
                )
            )
            results.append(sonara.analyze_aggression_signal(signal, sr=rate))
        signal_by_rate[rate] = results
        print(f"signal route: analyzed {len(results)} clips at {rate} Hz", flush=True)
    signal_per_rate, signal_aggregate, signal_acceptance = summarize(signal_by_rate, audit_ids)

    acceptance = {
        f"file_{name}": value for name, value in file_acceptance.items()
    }
    acceptance.update(
        {f"signal_{name}": value for name, value in signal_acceptance.items()}
    )
    report = {
        "format": "sonara.aggression-sample-rate-evidence.v1",
        "model_id": sonara.AGGRESSION_MODEL_ID,
        "canonical_sample_rate": sonara.AGGRESSION_SAMPLE_RATE,
        "track_count": len(manifest),
        "requested_sample_rates": list(RATES),
        "anchors": list(ANCHORS),
        "file_routes": {"per_rate": file_per_rate, "aggregate": file_aggregate},
        "signal_routes": {"per_rate": signal_per_rate, "aggregate": signal_aggregate},
        "acceptance": acceptance,
        "status": "pass" if all(acceptance.values()) else "fail",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
