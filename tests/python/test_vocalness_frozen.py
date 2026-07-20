#!/usr/bin/env python3
"""Mandatory, audio-free gate for the content-addressed vocalness regressions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sys

from sonara import vocal_model


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "reference_data" / "vocalness_similarity_v2.json"
VALIDATION = ROOT / "tests" / "reference_data" / "vocalness_v2_validation.json"
CRATE_MODEL = ROOT / "sonara" / "models" / "vocalness_v2.json"
PACKAGE_MODEL = ROOT / "python" / "sonara" / "models" / "vocalness_v2.json"
HASH_RE = re.compile(r"[0-9a-f]{64}")


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    fixture_bytes = FIXTURE.read_bytes()
    fixture = json.loads(fixture_bytes)
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if fixture.get("fixture_version") != 1:
        fail("fixture_version must be 1")
    if fixture.get("threshold") != 0.35:
        fail("frozen threshold must remain 0.35")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or len(cases) != 11:
        fail("fixture must contain exactly 11 cases")

    ids = [case.get("case_id") for case in cases]
    hashes = [case.get("content_hash") for case in cases]
    if len(set(ids)) != 11 or len(set(hashes)) != 11:
        fail("case ids and content hashes must be unique")
    if any(not isinstance(value, str) or not HASH_RE.fullmatch(value) for value in hashes):
        fail("every case requires a full lowercase SHA-256 content hash")
    classes = [case.get("class") for case in cases]
    if classes.count("vocal") != 5 or classes.count("instrumental") != 6:
        fail("fixture must contain exactly 5 vocal and 6 instrumental cases")

    fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()
    hash_set_sha = hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(hashes)).encode()
    ).hexdigest()
    if validation.get("frozen_fixture_sha256") != fixture_sha:
        fail("validation provenance does not match the frozen fixture")
    training = validation.get("training", {})
    if training.get("frozen_hash_exclusions") != 11:
        fail("validation provenance must exclude all 11 frozen hashes")
    if training.get("frozen_hash_set_sha256") != hash_set_sha:
        fail("training exclusion set does not match the frozen hashes")

    if CRATE_MODEL.read_bytes() != PACKAGE_MODEL.read_bytes():
        fail("Rust and Python bundled model artifacts differ")
    model = vocal_model.load(PACKAGE_MODEL)
    if model.id != "sonara-vocalness-v2":
        fail(f"unexpected bundled model id: {model.id!r}")
    model_sha = hashlib.sha256(PACKAGE_MODEL.read_bytes()).hexdigest()
    if validation.get("model_id") != model.id or validation.get("model_sha256") != model_sha:
        fail("validation provenance does not match the bundled model")

    threshold = fixture["threshold"]
    failures = []
    for case in cases:
        if case.get("hash_kind") != "mp3-audio-v1":
            fail(f"{case['case_id']}: unsupported hash_kind")
        if case.get("embedding_version") != 2 or case.get("similarity_version") != 2:
            fail(f"{case['case_id']}: wrong embedding/similarity version")
        embedding = case.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != 48:
            fail(f"{case['case_id']}: embedding must contain exactly 48 values")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in embedding):
            fail(f"{case['case_id']}: embedding must be finite")

        score = model.predict_vocalness(embedding)
        if not math.isfinite(score):
            fail(f"{case['case_id']}: model produced a non-finite score")
        expected_vocal = case["class"] == "vocal"
        margin = score - threshold if expected_vocal else threshold - score
        print(
            f"{case['case_id']:<10} {case['class']:<12} "
            f"score={score:.9f} margin={margin:+.9f}"
        )
        if margin <= 0:
            failures.append(case["case_id"])

    print(f"fixture_sha256={fixture_sha}")
    print(f"model_sha256={model_sha}")
    if failures:
        fail(f"misclassified frozen cases: {', '.join(failures)}")
    print("PASS: 5/5 vocal above 0.35; 6/6 instrumental below 0.35")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
