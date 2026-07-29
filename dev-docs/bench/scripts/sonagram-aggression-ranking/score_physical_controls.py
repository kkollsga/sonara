#!/usr/bin/env python3
"""Score the rater's frozen synthetic controls with the compact candidate."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXTRACTOR = Path(__file__).with_name("extract_features.py")
CONTROLS = ROOT / "dev-docs" / "bench" / "scripts" / "aggression-blind-review" / "controls"


def load_extractor():
    spec = importlib.util.spec_from_file_location("rank_features", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict(candidate, vector):
    linear = candidate["linear"]
    raw = sum(
        weight * (value - center)
        for weight, value, center in zip(
            linear["coefficients"], vector, linear["feature_center"]
        )
    )
    value = linear["calibration"]["slope"] * raw + linear["calibration"]["intercept"]
    linear_score = 1.0 / (1.0 + math.exp(-value))
    ensemble = candidate["tree_ensemble"]
    tree_score = ensemble["baseline"]
    for tree in ensemble["trees"]:
        index = 0
        while not tree[index]["leaf"]:
            node = tree[index]
            index = node["left"] if vector[node["feature"]] <= node["threshold"] else node["right"]
        tree_score += tree[index]["value"]
    tree_score = max(0.0, min(1.0, tree_score))
    blend = candidate["linear_weight"] * linear_score + candidate["tree_weight"] * tree_score
    return max(
        0.0,
        min(1.0, blend + candidate["harshness_correction"] * (vector[26] - 0.5)),
    )


def main() -> int:
    pool = Path(sys.argv[1])
    candidate = json.loads((pool / "compact_candidate.json").read_text())
    extractor = load_extractor()
    import sonara

    probe = extractor.load_probe()
    names = candidate["feature_names"]
    controls = {
        "harsh": CONTROLS / "harsh_distortion.wav",
        "calm": CONTROLS / "calm_pad.wav",
        "loud_clean": CONTROLS / "loud_clean_pulse.wav",
        "silence": CONTROLS / "silence.wav",
    }
    scores = {}
    support = {}
    parity_errors = {}
    for name, path in controls.items():
        confidence, features = extractor.extract(sonara, probe, path)
        vector = [features[feature] for feature in names]
        scores[name] = predict(candidate, vector)
        support[name] = confidence
        signal, sample_rate = sonara.load(str(path), sr=22_050, mono=True)
        production = sonara.analyze_aggression_signal(signal, sr=sample_rate)
        production_score = production["aggression_score"]
        parity_errors[name] = (
            0.0 if production_score is None and confidence <= 0.10
            else abs(scores[name] - production_score)
        )
        print(
            f"{name:10} score={scores[name]:.6f} content_support={confidence:.6f} "
            f"rust={production_score} parity_error={parity_errors[name]:.6g}"
        )
    checks = {
        "harsh_gt_calm": scores["harsh"] > scores["calm"],
        "harsh_gt_loud_clean": scores["harsh"] > scores["loud_clean"],
        "harsh_margin_over_loud_clean": scores["harsh"] - scores["loud_clean"] >= 0.30,
        "silence_low_support": support["silence"] <= 0.10,
        "offline_rust_parity": max(parity_errors.values()) <= 0.0001,
    }
    for name, passed in checks.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    print(f"PHYSICAL CONTROLS: {'GO' if all(checks.values()) else 'NO-GO'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
