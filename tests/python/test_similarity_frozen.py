#!/usr/bin/env python3
"""Mandatory, audio-free gate for the content-addressed similarity metric.

Recomputes every pinned pairwise distance / similarity and nearest-neighbor
ordering over 14 frozen 48-dim embeddings under BOTH selectable profiles
("default" and "timbre") and fails on any divergence. This is the gate that
catches a weight-table edit (WEIGHTS / WEIGHTS_TIMBRE), a normalization
change, or a profile-version drift without the corresponding fixture refresh.
Deterministic, no audio, runs anywhere. The labeled real-music
neighbor-quality gate lives in test_similarity_real.py (local-only).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

import sonara


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "reference_data" / "similarity_frozen_v1.json"
HASH_RE = re.compile(r"[0-9a-f]{64}")
N_CASES = 14
N_PAIRS = N_CASES * (N_CASES - 1) // 2
TOL = 5e-7  # f32 metric recomputed from identical inputs; platform guard only


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if fixture.get("fixture_version") != 1:
        fail("fixture_version must be 1")
    if fixture.get("similarity_version") != sonara.SIMILARITY_VERSION:
        fail("fixture similarity_version does not match sonara.SIMILARITY_VERSION")
    profiles = fixture.get("profiles")
    if profiles != dict(sonara.SIMILARITY_PROFILES):
        fail(
            "fixture profiles/versions do not match sonara.SIMILARITY_PROFILES "
            f"({profiles!r} vs {dict(sonara.SIMILARITY_PROFILES)!r}); a weight-"
            "table change must bump its profile version AND refresh this fixture"
        )

    cases = fixture.get("cases")
    if not isinstance(cases, list) or len(cases) != N_CASES:
        fail(f"fixture must contain exactly {N_CASES} cases")
    ids = [case.get("case_id") for case in cases]
    hashes = [case.get("content_hash") for case in cases]
    if len(set(ids)) != N_CASES or len(set(hashes)) != N_CASES:
        fail("case ids and content hashes must be unique")
    if any(not isinstance(value, str) or not HASH_RE.fullmatch(value) for value in hashes):
        fail("every case requires a full lowercase SHA-256 content hash")
    labels = sorted({case.get("label") for case in cases})
    if len(labels) != 7 or any(
        sum(1 for case in cases if case["label"] == label) != 2 for label in labels
    ):
        fail("fixture must contain exactly 2 cases for each of 7 style labels")

    emb = {}
    for case in cases:
        if case.get("hash_kind") != "mp3-audio-v1":
            fail(f"{case['case_id']}: unsupported hash_kind")
        if case.get("embedding_version") != 2 or case.get("similarity_version") != 2:
            fail(f"{case['case_id']}: wrong embedding/similarity version")
        vector = case.get("embedding")
        if not isinstance(vector, list) or len(vector) != 48:
            fail(f"{case['case_id']}: embedding must contain exactly 48 values")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in vector):
            fail(f"{case['case_id']}: embedding must be finite")
        emb[case["case_id"]] = vector

    expected = fixture.get("expected")
    if not isinstance(expected, dict) or set(expected) != set(profiles):
        fail("expected values must cover exactly the declared profiles")

    checked = 0
    for profile in sorted(profiles):
        table = expected[profile]
        distances = table.get("distances")
        similarities = table.get("similarities")
        nearest = table.get("nearest")
        if (
            not isinstance(distances, dict) or len(distances) != N_PAIRS
            or not isinstance(similarities, dict) or len(similarities) != N_PAIRS
            or not isinstance(nearest, dict) or len(nearest) != N_CASES
        ):
            fail(f"{profile}: expected {N_PAIRS} pairs and {N_CASES} orderings")
        for key, pinned in distances.items():
            a, b = key.split("|")
            got = sonara.embedding_distance(emb[a], emb[b], profile=profile)
            if sonara.embedding_distance(emb[b], emb[a], profile=profile) != got:
                fail(f"{profile} {key}: distance is not symmetric")
            if abs(got - pinned) > TOL:
                fail(
                    f"{profile} {key}: distance {got!r} diverged from "
                    f"pinned {pinned!r}"
                )
            got_sim = sonara.similarity(emb[a], emb[b], profile=profile)
            pinned_sim = similarities[key]
            if abs(got_sim - pinned_sim) > TOL:
                fail(
                    f"{profile} {key}: similarity {got_sim!r} diverged from "
                    f"pinned {pinned_sim!r}"
                )
            if profile == "default":
                if sonara.embedding_distance(emb[a], emb[b]) != got:
                    fail(f"{key}: profile='default' is not the default metric")
            checked += 1
        for case_id, vector in emb.items():
            if sonara.embedding_distance(vector, vector, profile=profile) != 0.0:
                fail(f"{profile} {case_id}: self-distance must be 0.0")
            order = sorted(
                (other for other in emb if other != case_id),
                key=lambda other: (
                    sonara.embedding_distance(emb[case_id], emb[other],
                                              profile=profile),
                    other,
                ),
            )
            if order != nearest[case_id]:
                fail(
                    f"{profile} {case_id}: nearest-neighbor ordering diverged "
                    f"from pinned (got {order[:3]}..., "
                    f"pinned {nearest[case_id][:3]}...)"
                )
        print(f"{profile:8s} v{profiles[profile]}: {N_PAIRS} pairs, "
              f"{N_CASES} orderings OK")

    print(f"PASS: {checked} pinned pairs verified under "
          f"{len(profiles)} profiles")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
