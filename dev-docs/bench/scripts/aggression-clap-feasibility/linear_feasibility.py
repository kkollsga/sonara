#!/usr/bin/env python3
"""Select a compact linear distillation model on pilot CV, then score open dev."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT = Path(__file__).resolve().parents[2] / "out"


def load(path: Path, open_set=False):
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
    decisive_ok = decisive_n = all_ok = 0
    pairs = sorted({row["pair_id"] for row in rows})
    for pair in pairs:
        indices = [index for index, row in enumerate(rows) if row["pair_id"] == pair]
        truth = {"left": 1, "right": -1, "tie": 0}[rows[indices[0]]["decision"]]
        delta = predictions[indices[0]] - predictions[indices[1]]
        predicted = 0 if abs(delta) < tie_band else (1 if delta > 0 else -1)
        all_ok += predicted == truth
        if truth:
            decisive_n += 1
            decisive_ok += predicted == truth
    return decisive_ok, decisive_n, all_ok, len(pairs)


def identity(value):
    return value


def clipped_logit(value):
    return logit(np.clip(value, 1e-4, 1 - 1e-4))


def arcsine(value):
    return np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def inverse_arcsine(value):
    return np.sin(np.clip(value, 0, np.pi / 2)) ** 2


def estimator(k, alpha, transform):
    steps = [StandardScaler()]
    if k != "all":
        steps.append(SelectKBest(f_regression, k=k))
    steps.append(Ridge(alpha=alpha))
    model = make_pipeline(*steps)
    if transform == "raw":
        return model
    function, inverse = {
        "logit": (clipped_logit, expit),
        "arcsine": (arcsine, inverse_arcsine),
    }[transform]
    return TransformedTargetRegressor(
        regressor=model, func=function, inverse_func=inverse, check_inverse=False
    )


def main():
    rows, names, x, y = load(OUT / "aggression-clap-pilot-600-features-plus-temporal.tsv")
    open_rows, open_names, xo, yo = load(
        OUT / "aggression-clap-open-features-plus-temporal.tsv", True
    )
    assert names == open_names
    cv = KFold(5, shuffle=True, random_state=20260722)
    best = None
    for k in [20, 40, 63, 100, 160, 220, "all"]:
        for alpha in np.logspace(-3, 5, 25):
            for transform in ["raw", "logit", "arcsine"]:
                model = estimator(k, alpha, transform)
                predictions = np.clip(cross_val_predict(model, x, y, cv=cv), 0, 1)
                rho = float(spearmanr(y, predictions).statistic)
                mae = float(mean_absolute_error(y, predictions))
                candidate = (rho, -mae, k, float(alpha), transform, predictions)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    rho, neg_mae, k, alpha, transform, predictions = best
    model = estimator(k, alpha, transform).fit(x, y)
    open_predictions = np.clip(model.predict(xo), 0, 1)
    open_rho = float(spearmanr(yo, open_predictions).statistic)
    pair = pair_score(open_rows, open_predictions)
    print(f"features={len(names)} selected_k={k} alpha={alpha:.9g} transform={transform}")
    print(f"pilot_cv_spearman={rho:.6f} pilot_cv_mae={-neg_mae:.6f}")
    print(
        f"open_spearman={open_rho:.6f} open_mae={mean_absolute_error(yo, open_predictions):.6f} "
        f"range={np.ptp(open_predictions):.6f}"
    )
    print(
        f"open_pairs={pair[0]}/{pair[1]} decisive, {pair[2]}/{pair[3]} all"
    )
    go = pair[0] >= 14 and open_rho >= 0.70 and np.ptp(open_predictions) >= 0.55
    print(f"OPEN LINEAR FEASIBILITY: {'GO' if go else 'NO-GO'}")


if __name__ == "__main__":
    main()
