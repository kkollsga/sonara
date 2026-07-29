#!/usr/bin/env python3
"""Diagnose whether the feasibility miss is representation- or backend-limited."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestRegressor as PermutationForest,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2] / "out"


def load(path: Path, open_set: bool = False):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    fixed = {"pair_id", "source", "sha256", "decision", "target"} if open_set else {
        "sample_id", "group_id", "sha256", "target"
    }
    names = [name for name in rows[0] if name not in fixed]
    x = np.asarray([[float(row[name]) for name in names] for row in rows])
    y = np.asarray([float(row["target"]) for row in rows])
    return rows, names, x, y


def pair_score(rows, predictions, tie_band=0.08):
    pairs = sorted({row["pair_id"] for row in rows})
    decisive_ok = decisive_n = all_ok = 0
    for pair in pairs:
        indices = [i for i, row in enumerate(rows) if row["pair_id"] == pair]
        truth = {"left": 1, "right": -1, "tie": 0}[rows[indices[0]]["decision"]]
        delta = predictions[indices[0]] - predictions[indices[1]]
        predicted = 0 if abs(delta) < tie_band else (1 if delta > 0 else -1)
        all_ok += truth == predicted
        if truth:
            decisive_n += 1
            decisive_ok += truth == predicted
    return decisive_ok, decisive_n, all_ok, len(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    suffix = args.suffix
    pilot_rows, names, x, y = load(ROOT / f"aggression-clap-pilot-features{suffix}.tsv")
    open_rows, open_names, xo, yo = load(ROOT / f"aggression-clap-open-features{suffix}.tsv", True)
    assert names == open_names
    cv = KFold(5, shuffle=True, random_state=20260722)
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 33))),
        "rf-matched": RandomForestRegressor(
            n_estimators=32, max_depth=10, min_samples_leaf=6,
            max_features="sqrt", random_state=20260722, n_jobs=-1,
        ),
        "rf-large": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, max_features=1.0,
            random_state=20260722, n_jobs=-1,
        ),
        "extra-trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=3, max_features=1.0,
            random_state=20260722, n_jobs=-1,
        ),
        "gradient-boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=2, loss="huber", random_state=20260722,
        ),
        "hist-gradient": HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=20260722,
        ),
    }
    print("model\tpilot_spearman\tpilot_mae\topen_spearman\topen_mae\tdecisive\tall_pairs\trange")
    for name, model in models.items():
        cvp = np.clip(cross_val_predict(model, x, y, cv=cv, n_jobs=1), 0, 1)
        fitted = clone(model).fit(x, y)
        op = np.clip(fitted.predict(xo), 0, 1)
        decisive, total, all_ok, pairs = pair_score(open_rows, op)
        print(
            f"{name}\t{spearmanr(y, cvp).statistic:.3f}\t{mean_absolute_error(y, cvp):.3f}"
            f"\t{spearmanr(yo, op).statistic:.3f}\t{mean_absolute_error(yo, op):.3f}"
            f"\t{decisive}/{total}\t{all_ok}/{pairs}\t{np.ptp(op):.3f}"
        )

    forest = PermutationForest(
        n_estimators=500, min_samples_leaf=3, max_features=1.0,
        random_state=20260722, n_jobs=-1,
    ).fit(x, y)
    perm = permutation_importance(
        forest, x, y, scoring="neg_mean_absolute_error", n_repeats=20,
        random_state=20260722, n_jobs=-1,
    )
    print("\nfeature\tunivariate_spearman\tpermutation_mae_loss")
    for index in np.argsort(perm.importances_mean)[::-1]:
        rho = spearmanr(x[:, index], y).statistic
        print(f"{names[index]}\t{rho:.3f}\t{perm.importances_mean[index]:.4f}")


if __name__ == "__main__":
    main()
