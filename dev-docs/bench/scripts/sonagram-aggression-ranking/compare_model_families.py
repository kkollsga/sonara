#!/usr/bin/env python3
"""Diagnostic-only established-model comparison on the frozen open cohort."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def table(path: Path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    names = [key for key in rows[0] if key not in {"sample_id", "support"}]
    return names, {row["sample_id"]: np.asarray([float(row[key]) for key in names]) for row in rows}


def labels(path: Path):
    return {
        row["sample_id"]: float(row["target"])
        for row in (json.loads(line) for line in path.read_text().splitlines())
    }


def metrics(pairs, truth, scores, band):
    decisive = hard = ties = 0
    for pair in pairs:
        delta = scores[pair["left_id"]] - scores[pair["right_id"]]
        predicted = "tie" if abs(delta) <= band else "left" if delta > 0 else "right"
        correct = predicted == pair["decision"]
        if pair["decision"] == "tie":
            ties += int(correct)
        else:
            decisive += int(correct)
            hard += int(correct and pair["category"] == "hard")
    ids = sorted(scores)
    expected = np.asarray([truth[sample_id] for sample_id in ids])
    actual = np.asarray([scores[sample_id] for sample_id in ids])
    return decisive, hard, ties, float(spearmanr(expected, actual).statistic), float(np.mean(abs(expected - actual))), float(np.ptp(actual))


def pairwise_scores(features, candidate):
    weights = np.asarray(candidate["coefficients"])
    center = np.asarray(candidate["feature_center"])
    slope = candidate["calibration"]["slope"]
    intercept = candidate["calibration"]["intercept"]
    return {
        sample_id: float(1.0 / (1.0 + np.exp(-(slope * np.dot(weights, vector - center) + intercept))))
        for sample_id, vector in features.items()
    }


def pairwise_raw(features, candidate):
    weights = np.asarray(candidate["coefficients"])
    center = np.asarray(candidate["feature_center"])
    return {
        sample_id: float(np.dot(weights, vector - center))
        for sample_id, vector in features.items()
    }


def main() -> int:
    pool = Path(sys.argv[1])
    names, train_x = table(pool / "fit_train_features.tsv")
    _, dev_x = table(pool / "evaluation_development_features.tsv")
    _, anchor_x = table(pool / "anchor_features.tsv")
    train_y = labels(pool / "fit_train_labels.jsonl")
    dev_y = labels(pool / "evaluation_development_labels.jsonl")
    pairs = [json.loads(line) for line in (pool / "development_pairs.jsonl").read_text().splitlines()]
    train_ids = sorted(train_x)
    x = np.stack([train_x[sample_id] for sample_id in train_ids])
    y = np.asarray([train_y[sample_id] for sample_id in train_ids])
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "rf": RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=20260722),
        "extra": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=20260722),
        "gbr": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, loss="huber", random_state=20260722),
        "hist": HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=20260722),
    }
    model_scores = {}
    model_anchors = {}
    for name, model in models.items():
        model.fit(x, y)
        scores = {sample_id: float(np.clip(model.predict(vector[None])[0], 0, 1)) for sample_id, vector in dev_x.items()}
        candidates = []
        for step in range(1, 31):
            band = step / 100
            result = metrics(pairs, dev_y, scores, band)
            gates = sum((result[0] >= 52, result[1] >= 20, result[2] >= 12))
            candidates.append((gates, sum(result[:3]), result[2], band, result))
        _, _, _, band, result = max(candidates)
        anchors = {sample_id: float(np.clip(model.predict(vector[None])[0], 0, 1)) for sample_id, vector in anchor_x.items()}
        model_scores[name] = scores
        model_anchors[name] = anchors
        margins = [anchors[f"heavy-{i}"] - anchors[f"dance-{j}"] for i in range(1, 4) for j in range(1, 4)]
        print(f"{name:5} band={band:.2f} decisive={result[0]}/64 hard={result[1]}/24 ties={result[2]}/16 rho={result[3]:.3f} mae={result[4]:.3f} range={result[5]:.3f} anchors={sum(value > 0 for value in margins)}/9 min={min(margins):.3f}")
    candidate = json.loads((pool / "candidate.json").read_text())
    pair_scores = pairwise_scores(dev_x, candidate)
    pair_anchors = pairwise_scores(anchor_x, candidate)
    raw_train = pairwise_raw(train_x, candidate)
    raw_dev = pairwise_raw(dev_x, candidate)
    raw_anchors = pairwise_raw(anchor_x, candidate)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        [raw_train[sample_id] for sample_id in train_ids],
        [train_y[sample_id] for sample_id in train_ids],
    )
    isotonic_scores = {
        sample_id: float(calibrator.predict([value])[0]) for sample_id, value in raw_dev.items()
    }
    isotonic_anchors = {
        sample_id: float(calibrator.predict([value])[0]) for sample_id, value in raw_anchors.items()
    }
    isotonic_candidates = []
    for band_step in range(1, 31):
        band = band_step / 100
        result = metrics(pairs, dev_y, isotonic_scores, band)
        margins = [isotonic_anchors[f"heavy-{i}"] - isotonic_anchors[f"dance-{j}"] for i in range(1, 4) for j in range(1, 4)]
        gates = sum((result[0] >= 52, result[1] >= 20, result[2] >= 12))
        isotonic_candidates.append((gates, sum(result[:3]), result[2], band, result, margins))
    _, _, _, band, result, margins = max(isotonic_candidates)
    print(f"isotonic knots={len(calibrator.X_thresholds_)} band={band:.2f} decisive={result[0]}/64 hard={result[1]}/24 ties={result[2]}/16 rho={result[3]:.3f} mae={result[4]:.3f} range={result[5]:.3f} anchors={sum(value > 0 for value in margins)}/9 min={min(margins):.3f}")
    for partner in models:
        blended = []
        for step in range(21):
            alpha = step / 20
            scores = {
                sample_id: alpha * pair_scores[sample_id] + (1 - alpha) * model_scores[partner][sample_id]
                for sample_id in dev_x
            }
            anchors = {
                sample_id: alpha * pair_anchors[sample_id] + (1 - alpha) * model_anchors[partner][sample_id]
                for sample_id in anchor_x
            }
            for band_step in range(1, 31):
                band = band_step / 100
                result = metrics(pairs, dev_y, scores, band)
                margins = [anchors[f"heavy-{i}"] - anchors[f"dance-{j}"] for i in range(1, 4) for j in range(1, 4)]
                gates = sum((result[0] >= 52, result[1] >= 20, result[2] >= 12))
                blended.append((gates, sum(result[:3]), result[2], alpha, band, result, margins))
        _, _, _, alpha, band, result, margins = max(blended)
        print(f"blend-{partner:5} alpha={alpha:.2f} band={band:.2f} decisive={result[0]}/64 hard={result[1]}/24 ties={result[2]}/16 rho={result[3]:.3f} mae={result[4]:.3f} range={result[5]:.3f} anchors={sum(value > 0 for value in margins)}/9 min={min(margins):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
