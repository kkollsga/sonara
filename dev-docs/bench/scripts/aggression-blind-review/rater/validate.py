"""Validate the blind-review packet + label files against the protocol.

Checks (no label values printed for the locked cohort):
  - dev >= 24 pairs, locked >= 20 pairs
  - dev and locked hash sets are DISJOINT
  - >= 2 independent sources represented overall and in each cohort
  - every label row is schema-conformant (ids, score ranges, tag vocabulary,
    valid more_aggressive)

Usage:  python validate.py
"""
from __future__ import annotations

import json
import os

from base import REASON_TAGS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACKETS = os.path.join(ROOT, "packets")

VALID_DECISIONS = {"left", "right", "tie", "abstain"}


def read_jsonl(name):
    rows = []
    with open(os.path.join(PACKETS, name)) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load(name):
    with open(os.path.join(ROOT, name)) as fh:
        return json.load(fh)


def cohort_hashes(pairs):
    hs = set()
    for p in pairs:
        hs.add(p["left_sha256"])
        hs.add(p["right_sha256"])
    return hs


def cohort_sources(manifest):
    s = set()
    for p in manifest:
        s.add(p["left_source"])
        s.add(p["right_source"])
    return s


def check_schema(doc, min_pairs):
    errs = []
    for key in ("protocol_version", "evaluator", "audio_perception_confirmed",
                "blindness_statement", "cohort", "pairs"):
        if key not in doc:
            errs.append(f"missing top-level key: {key}")
    pairs = doc.get("pairs", [])
    if len(pairs) < min_pairs:
        errs.append(f"{len(pairs)} pairs < required {min_pairs}")
    for p in pairs:
        pid = p.get("pair_id", "?")
        if p.get("more_aggressive") not in VALID_DECISIONS:
            errs.append(f"{pid}: bad more_aggressive={p.get('more_aggressive')}")
        for f in ("left_score", "right_score"):
            v = p.get(f)
            if not isinstance(v, (int, float)) or not (0 <= v <= 100):
                errs.append(f"{pid}: {f} out of range: {v}")
        c = p.get("confidence")
        if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0):
            errs.append(f"{pid}: confidence out of range: {c}")
        for t in p.get("reason_tags", []):
            if t not in REASON_TAGS:
                errs.append(f"{pid}: unknown reason_tag {t}")
    return errs


def main():
    dev_m = read_jsonl("dev_pairs.jsonl")
    locked_m = read_jsonl("locked_pairs.jsonl")
    dev = load("development_labels.json")
    locked = load("locked_labels.json")

    dev_h = cohort_hashes(dev["pairs"])
    locked_h = cohort_hashes(locked["pairs"])
    overlap = dev_h & locked_h
    dev_src = cohort_sources(dev_m)
    locked_src = cohort_sources(locked_m)
    all_src = dev_src | locked_src

    results = {
        "dev_pairs>=24": len(dev["pairs"]) >= 24,
        "locked_pairs>=20": len(locked["pairs"]) >= 20,
        "hash_disjoint(dev,locked)": len(overlap) == 0,
        "sources>=2 overall": len(all_src) >= 2,
        "sources>=2 in dev": len(dev_src) >= 2,
        "sources>=2 in locked": len(locked_src) >= 2,
        "dev schema ok": not check_schema(dev, 24),
        "locked schema ok": not check_schema(locked, 20),
    }
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    schema_errs = check_schema(dev, 24) + check_schema(locked, 20)
    if schema_errs:
        print("\nschema errors (first 10):")
        for e in schema_errs[:10]:
            print("   -", e)

    print(f"\ndev hashes: {len(dev_h)}   locked hashes: {len(locked_h)}   "
          f"overlap: {len(overlap)}")
    print(f"sources overall: {sorted(all_src)}")
    print(f"OVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
