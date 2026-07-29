#!/usr/bin/env python3
"""Grouped-fold and shuffled-label controls for the compact candidate family."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from compare_model_families import labels, table


def fold_of(sample_id):
    value = 0xCBF29CE484222325
    for byte in sample_id.encode():
        value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return value % 5


def fit_tree(x, y):
    return RandomForestRegressor(
        n_estimators=16,
        max_depth=4,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=20260722,
    ).fit(x, y)


def fit_linear(feature_map, target_map, pairs, ids, ties):
    matrix = []
    output = []
    allowed = set(ids)
    for pair in pairs:
        if pair["left_id"] not in allowed or pair["right_id"] not in allowed:
            continue
        left = feature_map[pair["left_id"]]
        right = feature_map[pair["right_id"]]
        difference = left - right
        if ties and pair["decision"] == "tie":
            for _ in range(32):
                for sign in (1.0, -1.0):
                    matrix.extend((sign * difference, sign * difference))
                    output.extend((0, 1))
        else:
            sign = 1.0 if target_map[pair["left_id"]] > target_map[pair["right_id"]] else -1.0
            matrix.extend((sign * difference, -sign * difference))
            output.extend((1, 0))
    matrix = np.stack(matrix)
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        C=0.1,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=1000,
        tol=1.0e-7,
    ).fit(scaler.transform(matrix), output)
    raw = np.asarray([
        model.decision_function(scaler.transform(feature_map[sample_id][None]))[0]
        for sample_id in ids
    ])
    target = np.asarray([target_map[sample_id] for sample_id in ids])
    logit = np.log(np.clip(target, 1.0e-4, 1.0 - 1.0e-4) / np.clip(1.0 - target, 1.0e-4, 1.0))
    slope = max(float(np.cov(raw, logit, bias=True)[0, 1] / max(np.var(raw), 1.0e-12)), 1.0e-8)
    intercept = float(np.mean(logit) - slope * np.mean(raw))
    return scaler, model, slope, intercept


def linear_predict(fitted, matrix):
    scaler, model, slope, intercept = fitted
    raw = model.decision_function(scaler.transform(matrix))
    value = slope * raw + intercept
    return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))


def directional_accuracy(pairs, scores):
    decisive = [pair for pair in pairs if pair["decision"] != "tie"]
    return sum(
        ((scores[pair["left_id"]] - scores[pair["right_id"]]) > 0)
        == (pair["decision"] == "left")
        for pair in decisive
    ) / len(decisive)


def apply_harshness_correction(prediction, matrix):
    return np.clip(prediction + 0.10 * (matrix[:, 26] - 0.5), 0, 1)


def main() -> int:
    pool = Path(sys.argv[1])
    _, train_x = table(pool / "fit_train_features.tsv")
    _, dev_x = table(pool / "evaluation_development_features.tsv")
    train_y = labels(pool / "fit_train_labels.jsonl")
    dev_y = labels(pool / "evaluation_development_labels.jsonl")
    train_pairs = [json.loads(line) for line in (pool / "train_pairs.jsonl").read_text().splitlines()]
    dev_pairs = [json.loads(line) for line in (pool / "development_pairs.jsonl").read_text().splitlines()]
    train_ids = sorted(train_x)
    dev_ids = sorted(dev_x)
    dev_matrix = np.stack([dev_x[sample_id] for sample_id in dev_ids])
    truth = np.asarray([dev_y[sample_id] for sample_id in dev_ids])

    shuffle_accuracy = []
    shuffle_rho = []
    values = np.asarray([train_y[sample_id] for sample_id in train_ids])
    for repeat in range(10):
        shuffled_values = np.random.default_rng(0x5EED + repeat).permutation(values)
        shuffled = dict(zip(train_ids, shuffled_values))
        tree = fit_tree(np.stack([train_x[sample_id] for sample_id in train_ids]), shuffled_values)
        linear = fit_linear(train_x, shuffled, train_pairs, train_ids, ties=False)
        prediction = 0.80 * np.clip(tree.predict(dev_matrix), 0, 1) + 0.20 * linear_predict(linear, dev_matrix)
        scores = dict(zip(dev_ids, prediction))
        shuffle_accuracy.append(directional_accuracy(dev_pairs, scores))
        shuffle_rho.append(float(spearmanr(truth, prediction).statistic))
    print(
        f"shuffle accuracy_mean={np.mean(shuffle_accuracy):.6f} "
        f"rho_mean={np.mean(shuffle_rho):.6f} "
        f"accuracy_range={min(shuffle_accuracy):.6f}..{max(shuffle_accuracy):.6f}"
    )

    fold_rhos = []
    fold_maes = []
    for fold in range(5):
        fit_ids = [sample_id for sample_id in train_ids if fold_of(sample_id) != fold]
        held_ids = [sample_id for sample_id in train_ids if fold_of(sample_id) == fold]
        fit_matrix = np.stack([train_x[sample_id] for sample_id in fit_ids])
        held_matrix = np.stack([train_x[sample_id] for sample_id in held_ids])
        tree = fit_tree(fit_matrix, np.asarray([train_y[sample_id] for sample_id in fit_ids]))
        linear = fit_linear(train_x, train_y, train_pairs, fit_ids, ties=True)
        prediction = apply_harshness_correction(
            0.80 * np.clip(tree.predict(held_matrix), 0, 1)
            + 0.20 * linear_predict(linear, held_matrix),
            held_matrix,
        )
        held_truth = np.asarray([train_y[sample_id] for sample_id in held_ids])
        rho = float(spearmanr(held_truth, prediction).statistic)
        mae = float(np.mean(abs(held_truth - prediction)))
        fold_rhos.append(rho)
        fold_maes.append(mae)
        print(f"fold={fold} tracks={len(held_ids)} rho={rho:.6f} mae={mae:.6f}")
    print(
        f"fold rho_min={min(fold_rhos):.6f} rho_mean={np.mean(fold_rhos):.6f} "
        f"mae_max={max(fold_maes):.6f}"
    )
    go = (
        0.45 <= np.mean(shuffle_accuracy) <= 0.55
        and abs(np.mean(shuffle_rho)) <= 0.10
        and min(fold_rhos) >= 0.65
        and max(fold_maes) <= 0.15
    )
    print(f"COMPACT CONTROLS: {'GO' if go else 'NO-GO'}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
