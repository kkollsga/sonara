#!/usr/bin/env python3
"""Find the smallest nonlinear residual model that clears every open gate."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from compare_model_families import labels, metrics, pairwise_scores, table
from score_physical_controls import CONTROLS, load_extractor


def evaluate(
    name, model, nodes, dev_x, dev_y, anchor_x, control_x, pairs,
    linear, linear_anchors, linear_controls,
):
    dev_ids = list(dev_x)
    anchor_ids = list(anchor_x)
    tree_scores = dict(zip(
        dev_ids,
        np.clip(model.predict(np.stack([dev_x[sample_id] for sample_id in dev_ids])), 0, 1),
    ))
    tree_anchors = dict(zip(
        anchor_ids,
        np.clip(model.predict(np.stack([anchor_x[sample_id] for sample_id in anchor_ids])), 0, 1),
    ))
    tree_controls = {
        sample_id: float(np.clip(model.predict(vector[None])[0], 0, 1))
        for sample_id, vector in control_x.items()
    }
    passing = []
    for alpha_step in range(12):
        alpha = alpha_step * 0.05
        base_scores = {
            sample_id: alpha * linear[sample_id] + (1 - alpha) * tree_scores[sample_id]
            for sample_id in dev_x
        }
        base_anchors = {
            sample_id: alpha * linear_anchors[sample_id] + (1 - alpha) * tree_anchors[sample_id]
            for sample_id in anchor_x
        }
        base_controls = {
            sample_id: alpha * linear_controls[sample_id]
            + (1 - alpha) * tree_controls[sample_id]
            for sample_id in control_x
        }
        for correction_step in range(11):
            correction = correction_step * 0.02
            scores = {
                sample_id: float(np.clip(value + correction * (dev_x[sample_id][26] - 0.5), 0, 1))
                for sample_id, value in base_scores.items()
            }
            anchors = {
                sample_id: float(np.clip(value + correction * (anchor_x[sample_id][26] - 0.5), 0, 1))
                for sample_id, value in base_anchors.items()
            }
            controls = {
                sample_id: float(np.clip(value + correction * (control_x[sample_id][26] - 0.5), 0, 1))
                for sample_id, value in base_controls.items()
            }
            anchor_margins = [
                anchors[f"heavy-{i}"] - anchors[f"dance-{j}"]
                for i in range(1, 4)
                for j in range(1, 4)
            ]
            control_margin = controls["harsh"] - controls["loud_clean"]
            for band_step in range(1, 21):
                band = band_step / 100
                result = metrics(pairs, dev_y, scores, band)
                if (
                    result[0] >= 52
                    and result[1] >= 20
                    and result[2] >= 12
                    and result[3] >= 0.65
                    and result[4] <= 0.15
                    and result[5] >= 0.65
                    and min(anchor_margins) >= 0.15
                    and controls["harsh"] > controls["calm"]
                    and control_margin >= 0.30
                ):
                    passing.append(
                        (
                            nodes, name, alpha, correction, band, result,
                            min(anchor_margins), control_margin,
                        )
                    )
    return passing


def main() -> int:
    pool = Path(sys.argv[1])
    _, train_x = table(pool / "fit_train_features.tsv")
    _, dev_x = table(pool / "evaluation_development_features.tsv")
    _, anchor_x = table(pool / "anchor_features.tsv")
    train_y = labels(pool / "fit_train_labels.jsonl")
    dev_y = labels(pool / "evaluation_development_labels.jsonl")
    pairs = [json.loads(line) for line in (pool / "development_pairs.jsonl").read_text().splitlines()]
    candidate = json.loads((pool / "candidate.json").read_text())
    linear = pairwise_scores(dev_x, candidate)
    linear_anchors = pairwise_scores(anchor_x, candidate)
    extractor = load_extractor()
    import sonara

    probe = extractor.load_probe()
    control_x = {}
    for sample_id, filename in {
        "harsh": "harsh_distortion.wav",
        "calm": "calm_pad.wav",
        "loud_clean": "loud_clean_pulse.wav",
    }.items():
        _, features = extractor.extract(sonara, probe, CONTROLS / filename)
        control_x[sample_id] = np.asarray([features[name] for name in candidate["feature_names"]])
    linear_controls = pairwise_scores(control_x, candidate)
    train_ids = sorted(train_x)
    x = np.stack([train_x[sample_id] for sample_id in train_ids])
    y = np.asarray([train_y[sample_id] for sample_id in train_ids])
    passing = []

    for estimators, depth, leaf, max_features in itertools.product(
        (16, 32, 64), (4, 6, 8), (2, 4, 8), (0.6, 1.0)
    ):
        model = RandomForestRegressor(
            n_estimators=estimators,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=20260722,
        ).fit(x, y)
        nodes = sum(tree.tree_.node_count for tree in model.estimators_)
        passing.extend(evaluate(
            f"rf-n{estimators}-d{depth}-l{leaf}-f{max_features}", model, nodes,
            dev_x, dev_y, anchor_x, control_x, pairs, linear, linear_anchors,
            linear_controls,
        ))

    for iterations, leaves, min_samples, l2, learning_rate in itertools.product(
        (16, 32, 64, 96, 128, 160, 200), (7, 15), (10, 20, 40), (1.0, 5.0), (0.03, 0.05, 0.08)
    ):
        model = HistGradientBoostingRegressor(
            max_iter=iterations,
            max_leaf_nodes=leaves,
            min_samples_leaf=min_samples,
            learning_rate=learning_rate,
            l2_regularization=l2,
            random_state=20260722,
        ).fit(x, y)
        nodes = sum(predictor[0].nodes.shape[0] for predictor in model._predictors)
        passing.extend(evaluate(
            f"hist-n{iterations}-l{leaves}-m{min_samples}-r{l2}-e{learning_rate}", model, nodes,
            dev_x, dev_y, anchor_x, control_x, pairs, linear, linear_anchors,
            linear_controls,
        ))

    for nodes, name, alpha, correction, band, result, anchor_margin, control_margin in sorted(passing)[:200]:
        print(
            f"nodes={nodes:4} {name} alpha={alpha:.2f} correction={correction:.2f} band={band:.2f} "
            f"decisive={result[0]} hard={result[1]} ties={result[2]} rho={result[3]:.3f} "
            f"mae={result[4]:.3f} range={result[5]:.3f} anchor={anchor_margin:.3f} "
            f"control={control_margin:.3f}"
        )
    print(f"passing configurations: {len(passing)}")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
