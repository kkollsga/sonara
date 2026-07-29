#!/usr/bin/env python3
"""Freeze and self-check the smallest passing open-development candidate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from compare_model_families import labels, metrics, pairwise_scores, table


def flatten_tree(tree, scale):
    rows = []
    for index in range(tree.node_count):
        leaf = tree.children_left[index] < 0
        rows.append(
            {
                "feature": 0 if leaf else int(tree.feature[index]),
                "threshold": 0.0 if leaf else float(tree.threshold[index]),
                "left": 0 if leaf else int(tree.children_left[index]),
                "right": 0 if leaf else int(tree.children_right[index]),
                "value": float(tree.value[index].reshape(-1)[0]) * scale if leaf else 0.0,
                "leaf": bool(leaf),
            }
        )
    return rows


def predict_flat(vector, baseline, trees):
    value = baseline
    for tree in trees:
        index = 0
        while not tree[index]["leaf"]:
            node = tree[index]
            index = node["left"] if vector[node["feature"]] <= node["threshold"] else node["right"]
        value += tree[index]["value"]
    return value


def main() -> int:
    pool = Path(sys.argv[1])
    names, train_x = table(pool / "fit_train_features.tsv")
    _, dev_x = table(pool / "evaluation_development_features.tsv")
    _, anchor_x = table(pool / "anchor_features.tsv")
    train_y = labels(pool / "fit_train_labels.jsonl")
    dev_y = labels(pool / "evaluation_development_labels.jsonl")
    train_ids = sorted(train_x)
    x = np.stack([train_x[sample_id] for sample_id in train_ids])
    y = np.asarray([train_y[sample_id] for sample_id in train_ids])
    model = RandomForestRegressor(
        n_estimators=16,
        max_depth=4,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=20260722,
    ).fit(x, y)
    trees = [flatten_tree(estimator.tree_, 1.0 / len(model.estimators_)) for estimator in model.estimators_]
    baseline = 0.0
    for vector in list(dev_x.values())[:16]:
        expected = float(model.predict(vector[None])[0])
        actual = predict_flat(vector, baseline, trees)
        if abs(expected - actual) > 1.0e-12:
            raise RuntimeError(f"tree export mismatch: {expected} != {actual}")

    linear = json.loads((pool / "candidate.json").read_text())
    linear_scores = pairwise_scores(dev_x, linear)
    linear_anchors = pairwise_scores(anchor_x, linear)
    tree_scores = {
        sample_id: float(np.clip(model.predict(vector[None])[0], 0, 1))
        for sample_id, vector in dev_x.items()
    }
    tree_anchors = {
        sample_id: float(np.clip(model.predict(vector[None])[0], 0, 1))
        for sample_id, vector in anchor_x.items()
    }
    harshness_index = names.index("interaction_harshness")
    harshness_correction = 0.10
    scores = {
        sample_id: float(np.clip(
            0.20 * linear_scores[sample_id]
            + 0.80 * tree_scores[sample_id]
            + harshness_correction * (dev_x[sample_id][harshness_index] - 0.5),
            0,
            1,
        ))
        for sample_id in dev_x
    }
    anchors = {
        sample_id: float(np.clip(
            0.20 * linear_anchors[sample_id]
            + 0.80 * tree_anchors[sample_id]
            + harshness_correction * (anchor_x[sample_id][harshness_index] - 0.5),
            0,
            1,
        ))
        for sample_id in anchor_x
    }
    pairs = [json.loads(line) for line in (pool / "development_pairs.jsonl").read_text().splitlines()]
    result = metrics(pairs, dev_y, scores, 0.07)
    anchor_margins = [
        anchors[f"heavy-{i}"] - anchors[f"dance-{j}"]
        for i in range(1, 4)
        for j in range(1, 4)
    ]
    if not (
        result[0] >= 52
        and result[1] >= 20
        and result[2] >= 12
        and result[3] >= 0.65
        and result[4] <= 0.15
        and result[5] >= 0.65
        and min(anchor_margins) >= 0.15
    ):
        raise RuntimeError("frozen compact candidate no longer passes open gates")
    schema = "\n".join(names).encode()
    artifact = {
        "format": "sonara.aggression-rank-candidate.v1",
        "feature_names": names,
        "feature_schema_sha256": hashlib.sha256(schema).hexdigest(),
        "rank_target_transform": json.loads((pool / "rank_target_transform.json").read_text()),
        "linear_weight": 0.20,
        "tree_weight": 0.80,
        "harshness_correction": harshness_correction,
        "tie_band": 0.07,
        "linear": linear,
        "tree_ensemble": {
            "baseline": baseline,
            "kind": "random_forest_regressor",
            "n_estimators": 16,
            "max_depth": 4,
            "min_samples_leaf": 2,
            "max_features": 1.0,
            "trees": trees,
        },
        "open_metrics": {
            "decisive_correct": result[0],
            "hard_correct": result[1],
            "tie_correct": result[2],
            "spearman": result[3],
            "mae": result[4],
            "range": result[5],
            "anchor_min_margin": min(anchor_margins),
        },
    }
    encoded = json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n"
    output = pool / "compact_candidate.json"
    output.write_bytes(encoded)
    print(f"candidate -> {output}")
    print(f"bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")
    print(
        f"open decisive={result[0]}/64 hard={result[1]}/24 ties={result[2]}/16 "
        f"rho={result[3]:.6f} mae={result[4]:.6f} range={result[5]:.6f} "
        f"anchor_min={min(anchor_margins):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
