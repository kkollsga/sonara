#!/usr/bin/env python3
"""Build frozen train and development pairs from anonymous tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SEED = "sonagram-aggression-ranking-pairs-v2"
TIE_MARGIN = 0.04
TIE_CONTEXT_DISTANCE = 0.05
TIE_CONTEXT_MAX = 0.15
TRAIN_TIE_CONTEXT_DISTANCE = 0.08
TRAIN_TIE_CONTEXT_MAX = 0.25
FIT_TRACKS = 544
DEVELOPMENT_CANDIDATES = 320


def read_labels(path: Path):
    return {
        row["sample_id"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        if not row["insufficient"]
    }


def latent_target(row):
    logits = []
    for excerpt in row["excerpts"]:
        if excerpt["insufficient"]:
            continue
        probability = min(1.0 - 1.0e-6, max(1.0e-6, float(excerpt["score"])))
        logits.append(math.log(probability / (1.0 - probability)))
    if len(logits) < 3:
        raise RuntimeError("usable label lacks three excerpt logits")
    return sum(sorted(logits, reverse=True)[:2]) / 2.0


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def rank_labels(labels, low, high):
    return {
        sample_id: {
            **row,
            "target": max(0.0, min(1.0, (latent_target(row) - low) / (high - low))),
        }
        for sample_id, row in labels.items()
    }


def read_features(path: Path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return {
        row["sample_id"]: {key: float(value) for key, value in row.items() if key != "sample_id"}
        for row in rows
    }


def stable(left: str, right: str, category: str) -> str:
    return hashlib.sha256(f"{SEED}:{category}:{left}:{right}".encode()).hexdigest()


def role_order(sample_id: str) -> str:
    return hashlib.sha256(f"{SEED}:role:{sample_id}".encode()).hexdigest()


def energy_context_distance(features, left, right):
    keys = ("embedding_46", "onset_density", "embedding_37")
    return sum(abs(features[left][key] - features[right][key]) for key in keys) / len(keys)


def tie_context_metrics(features, left, right):
    # A scalar tie is only meaningful when the tracks occupy the same local
    # physical neighbourhood.  Energy/onset-only matching admitted unrelated
    # force/harshness trade-offs whose equal teacher totals did not constrain a
    # stable local rank band.
    keys = sorted(key for key in features[left] if not key.startswith("interaction_"))
    distances = [abs(features[left][key] - features[right][key]) for key in keys]
    return sum(distances) / len(distances), max(distances)


def all_candidates(labels, features):
    ids = sorted(set(labels) & set(features))
    pairs = []
    for offset, left in enumerate(ids):
        for right in ids[offset + 1 :]:
            delta = float(labels[left]["target"]) - float(labels[right]["target"])
            margin = abs(delta)
            tie_mean, tie_max = tie_context_metrics(features, left, right)
            pairs.append(
                {
                    "left_id": left,
                    "right_id": right,
                    "decision": "tie" if margin <= 0.04 else "left" if delta > 0 else "right",
                    "margin": margin,
                    "context_distance": energy_context_distance(features, left, right),
                    "tie_context_distance": tie_mean,
                    "tie_context_max": tie_max,
                    "left_target": float(labels[left]["target"]),
                    "right_target": float(labels[right]["target"]),
                }
            )
    return pairs


def capped_select(candidates, count, category, used_counts, cap):
    selected = []
    for row in sorted(candidates, key=lambda item: stable(item["left_id"], item["right_id"], category)):
        if used_counts.get(row["left_id"], 0) >= cap or used_counts.get(row["right_id"], 0) >= cap:
            continue
        selected.append({**row, "category": category})
        used_counts[row["left_id"]] = used_counts.get(row["left_id"], 0) + 1
        used_counts[row["right_id"]] = used_counts.get(row["right_id"], 0) + 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only selected {len(selected)}/{count} {category} training pairs")
    return selected


def training_pairs(labels, features):
    candidates = all_candidates(labels, features)
    used = {}
    ties = [
        row
        for row in candidates
        if row["margin"] <= TIE_MARGIN
        and row["tie_context_distance"] <= TRAIN_TIE_CONTEXT_DISTANCE
        and row["tie_context_max"] <= TRAIN_TIE_CONTEXT_MAX
    ]
    hard = [row for row in candidates if row["margin"] >= 0.25 and row["context_distance"] <= 0.15]
    broad = [row for row in candidates if row["margin"] >= 0.15]
    selected = capped_select(ties, 1000, "tie", used, 20)
    already = {(row["left_id"], row["right_id"]) for row in selected}
    selected += capped_select(
        [row for row in hard if (row["left_id"], row["right_id"]) not in already],
        1500,
        "hard",
        used,
        20,
    )
    already = {(row["left_id"], row["right_id"]) for row in selected}
    selected += capped_select(
        [row for row in broad if (row["left_id"], row["right_id"]) not in already],
        2500,
        "broad",
        used,
        20,
    )
    return selected


def cohort_pairs(labels, features):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    candidates = all_candidates(labels, features)
    specs = [
        (
            "tie",
            16,
            lambda row: row["margin"] <= TIE_MARGIN
            and row["tie_context_distance"] <= TIE_CONTEXT_DISTANCE
            and row["tie_context_max"] <= TIE_CONTEXT_MAX,
            lambda row: row["margin"] / TIE_MARGIN
            + row["tie_context_distance"] / TIE_CONTEXT_DISTANCE
            + row["tie_context_max"] / TIE_CONTEXT_MAX,
        ),
        (
            "hard",
            24,
            lambda row: row["margin"] >= 0.25 and row["context_distance"] <= 0.15,
            lambda row: row["context_distance"],
        ),
        (
            "near",
            20,
            lambda row: 0.08 <= row["margin"] <= 0.20,
            lambda row: abs(row["margin"] - 0.14),
        ),
        ("broad", 20, lambda row: row["margin"] >= 0.35, lambda row: 1.0 - row["margin"]),
    ]
    track_ids = sorted(labels)
    track_row = {sample_id: index for index, sample_id in enumerate(track_ids)}
    category_row = {
        category: len(track_ids) + index for index, (category, _, _, _) in enumerate(specs)
    }
    options = []
    costs = []
    for category, _, predicate, objective in specs:
        for pair in candidates:
            if not predicate(pair):
                continue
            tie_break = int(stable(pair["left_id"], pair["right_id"], category)[:8], 16) / 2**32
            options.append((category, pair))
            costs.append(float(objective(pair)) + 1e-6 * tie_break)
    matrix_rows = []
    matrix_columns = []
    matrix_values = []
    for column, (category, pair) in enumerate(options):
        for row in (track_row[pair["left_id"]], track_row[pair["right_id"]], category_row[category]):
            matrix_rows.append(row)
            matrix_columns.append(column)
            matrix_values.append(1.0)
    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(track_ids) + len(specs), len(options)),
    ).tocsr()
    lower = np.asarray([0.0] * len(track_ids) + [count for _, count, _, _ in specs])
    upper = np.asarray([1.0] * len(track_ids) + [count for _, count, _, _ in specs])
    solution = milp(
        c=np.asarray(costs),
        integrality=np.ones(len(options)),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 120},
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"exact cohort matching failed: {solution.message}")
    result = [
        {**pair, "category": category}
        for selected, (category, pair) in zip(solution.x, options)
        if selected >= 0.5
    ]
    result.sort(key=lambda row: (row["category"], stable(row["left_id"], row["right_id"], row["category"])))
    used = {row[key] for row in result for key in ("left_id", "right_id")}
    assert len(result) == 80 and len(used) == 160
    return result


def write(path: Path, rows):
    path.write_text(
        "".join(
            json.dumps({"pair_id": f"{path.stem}-{index:03d}", **row}, sort_keys=True) + "\n"
            for index, row in enumerate(rows)
        )
    )


def write_label_subset(path: Path, labels, ids):
    rows = (
        {
            "sample_id": sample_id,
            "target": float(labels[sample_id]["target"]),
            "insufficient": False,
        }
        for sample_id in sorted(ids)
    )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_feature_subset(path: Path, source_paths, ids):
    rows = {}
    fieldnames = None
    for source_path in source_paths:
        with source_path.open() as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise RuntimeError("feature source schema mismatch")
            for row in reader:
                if row["sample_id"] in ids:
                    rows[row["sample_id"]] = row
    if len(rows) != len(ids) or fieldnames is None:
        raise RuntimeError(f"missing feature rows: {len(rows)}/{len(ids)}")
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[sample_id] for sample_id in sorted(rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    args = parser.parse_args()
    original_train_labels = read_labels(args.pool / "train_labels.jsonl")
    original_development_labels = read_labels(args.pool / "development_labels.jsonl")
    original_train_features = read_features(args.pool / "train_features.tsv")
    original_development_features = read_features(args.pool / "development_features.tsv")
    all_labels = {**original_train_labels, **original_development_labels}
    all_features = {**original_train_features, **original_development_features}
    ordered_ids = sorted(set(all_labels) & set(all_features), key=role_order)
    if len(ordered_ids) != FIT_TRACKS + DEVELOPMENT_CANDIDATES:
        raise RuntimeError(f"unexpected open pool size: {len(ordered_ids)}")
    fit_ids = set(ordered_ids[:FIT_TRACKS])
    development_candidate_ids = set(ordered_ids[FIT_TRACKS:])
    fit_latent = [latent_target(all_labels[sample_id]) for sample_id in fit_ids]
    rank_low = percentile(fit_latent, 0.02)
    rank_high = percentile(fit_latent, 0.98)
    if rank_high <= rank_low:
        raise RuntimeError("invalid latent-rank calibration")
    all_labels = rank_labels(all_labels, rank_low, rank_high)
    train_labels = {sample_id: all_labels[sample_id] for sample_id in fit_ids}
    train_features = {sample_id: all_features[sample_id] for sample_id in fit_ids}
    development_labels = {
        sample_id: all_labels[sample_id] for sample_id in development_candidate_ids
    }
    development_features = {
        sample_id: all_features[sample_id] for sample_id in development_candidate_ids
    }
    train = training_pairs(train_labels, train_features)
    development = cohort_pairs(development_labels, development_features)
    evaluation_ids = {
        row[key] for row in development for key in ("left_id", "right_id")
    }
    write_label_subset(args.pool / "fit_train_labels.jsonl", all_labels, fit_ids)
    write_label_subset(
        args.pool / "evaluation_development_labels.jsonl", all_labels, evaluation_ids
    )
    anchor_labels = rank_labels(read_labels(args.pool / "anchor_labels.jsonl"), rank_low, rank_high)
    write_label_subset(
        args.pool / "rank_anchor_labels.jsonl", anchor_labels, set(anchor_labels)
    )
    (args.pool / "rank_target_transform.json").write_text(
        json.dumps(
            {
                "source": "mean-top-two-excerpt-logit-margin",
                "lower_train_quantile": 0.02,
                "upper_train_quantile": 0.98,
                "lower": rank_low,
                "upper": rank_high,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    sources = [args.pool / "train_features.tsv", args.pool / "development_features.tsv"]
    write_feature_subset(args.pool / "fit_train_features.tsv", sources, fit_ids)
    write_feature_subset(
        args.pool / "evaluation_development_features.tsv", sources, evaluation_ids
    )
    write(args.pool / "train_pairs.jsonl", train)
    write(args.pool / "development_pairs.jsonl", development)
    print(f"train pairs: {len(train)}; development pairs: {len(development)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
