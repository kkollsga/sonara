#!/usr/bin/env python3
"""Label the already-frozen fresh audit with CLAP and compare both models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
SONARA = HERE.parents[3]
OUT = SONARA / "dev-docs/bench/out/ferricml-aggression-audit"
RATER = SONARA / "dev-docs/bench/scripts/aggression-blind-review/rater"
sys.path.insert(0, str(RATER))
from clap_rater import ClapRater  # noqa: E402


def metric(truth, predicted):
    return {
        "spearman": float(spearmanr(truth, predicted).statistic),
        "kendall": float(kendalltau(truth, predicted).statistic),
        "mae": float(np.mean(np.abs(np.asarray(truth) - np.asarray(predicted)))),
    }


def bootstrap(truth, old, new, repeats=10000):
    rng = np.random.default_rng(20260724)
    values = {"spearman": [], "kendall": [], "mae": []}
    n = len(truth)
    for _ in range(repeats):
        index = rng.integers(0, n, n)
        sample_truth = np.asarray(truth)[index]
        sample_old = np.asarray(old)[index]
        sample_new = np.asarray(new)[index]
        old_metrics = metric(sample_truth, sample_old)
        new_metrics = metric(sample_truth, sample_new)
        if all(np.isfinite(list(old_metrics.values()) + list(new_metrics.values()))):
            values["spearman"].append(new_metrics["spearman"] - old_metrics["spearman"])
            values["kendall"].append(new_metrics["kendall"] - old_metrics["kendall"])
            values["mae"].append(new_metrics["mae"] - old_metrics["mae"])
    return {
        name: {
            "delta": float(np.mean(samples)),
            "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
        }
        for name, samples in values.items()
    }


def pair_metrics(pairs, labels, predictions, model):
    correct = decisive = ties = tie_correct = 0
    for pair in pairs:
        left_truth = labels[pair["left_id"]]["target"]
        right_truth = labels[pair["right_id"]]["target"]
        truth_delta = left_truth - right_truth
        expected = "tie" if abs(truth_delta) < 0.08 else ("left" if truth_delta > 0 else "right")
        delta = predictions[pair["left_id"]][model] - predictions[pair["right_id"]][model]
        predicted = "tie" if abs(delta) <= 0.07 else ("left" if delta > 0 else "right")
        if expected == "tie":
            ties += 1
            tie_correct += int(predicted == expected)
        else:
            decisive += 1
            correct += int(predicted == expected)
    return {
        "decisive_correct": correct,
        "decisive_total": decisive,
        "tie_correct": tie_correct,
        "tie_total": ties,
    }


def main():
    freeze = json.loads((OUT / "predictions.freeze.json").read_text())
    manifest = [json.loads(line) for line in (OUT / "manifest.jsonl").read_text().splitlines()]
    pairs = [json.loads(line) for line in (OUT / "pairs.jsonl").read_text().splitlines()]
    label_path = OUT / "labels.json"
    labels = json.loads(label_path.read_text()) if label_path.exists() else {}
    missing = [row for row in manifest if row["sample_id"] not in labels]
    if missing:
        rater = ClapRater()
        for index, row in enumerate(missing, 1):
            verdict = rater.score_clip(str(OUT / row["clip"]))
            labels[row["sample_id"]] = {
                "target": verdict.aggression / 100.0,
                "confidence": verdict.confidence,
                "insufficient": verdict.insufficient,
            }
            print(f"rated new {index}/{len(missing)}")
        label_path.write_text(json.dumps(labels, indent=2, sort_keys=True))

    aligned_old = OUT / "released_v2_scores_aligned.json"
    aligned_new = OUT / "ferric_scores_aligned.json"
    if aligned_old.exists() and aligned_new.exists():
        old_scores = json.loads(aligned_old.read_text())
        new_scores = json.loads(aligned_new.read_text())
        predictions = {
            sample_id: {
                "released_v2": old_scores[sample_id],
                "ferric_candidate": new_scores[sample_id],
            }
            for sample_id in labels
        }
        input_scope = "exact_20_second_audit_excerpts"
    else:
        predictions = freeze["predictions"]
        input_scope = "full_tracks"

    usable = [sample_id for sample_id, label in labels.items() if not label["insufficient"]]
    truth = [labels[sample_id]["target"] for sample_id in usable]
    old = [predictions[sample_id]["released_v2"] for sample_id in usable]
    new = [predictions[sample_id]["ferric_candidate"] for sample_id in usable]
    result = {
        "input_scope": input_scope,
        "usable_tracks": len(usable),
        "released_v2": metric(truth, old),
        "ferric_candidate": metric(truth, new),
        "paired_bootstrap": bootstrap(truth, old, new),
        "released_v2_pairs": pair_metrics(pairs, labels, predictions, "released_v2"),
        "ferric_candidate_pairs": pair_metrics(pairs, labels, predictions, "ferric_candidate"),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
