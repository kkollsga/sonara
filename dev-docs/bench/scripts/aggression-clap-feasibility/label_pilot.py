#!/usr/bin/env python3
"""Generate deterministic CLAP teacher scores for the anonymous pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SONARA_ROOT = Path(__file__).resolve().parents[4]
RATER_ROOT = SONARA_ROOT / "dev-docs" / "bench" / "scripts" / "aggression-blind-review" / "rater"
sys.path.insert(0, str(RATER_ROOT))

from clap_rater import ClapRater  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--reuse-labels", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.pilot / "manifest.jsonl").read_text().splitlines()]
    reused = {}
    if args.reuse_labels:
        reused = {
            row["sha256"]: row
            for row in (
                json.loads(line) for line in args.reuse_labels.read_text().splitlines()
            )
        }
    rater = ClapRater()
    output = args.pilot / "clap_labels.jsonl"
    with output.open("w") as stream:
        for index, row in enumerate(rows, 1):
            record = reused.get(row["sha256"])
            if record is None:
                verdict = rater.score_clip(str(args.pilot / row["path"]))
                record = {
                    "sample_id": row["sample_id"],
                    "sha256": row["sha256"],
                    "target": verdict.aggression / 100.0,
                    "confidence": verdict.confidence,
                    "insufficient": verdict.insufficient,
                }
            else:
                record = {**record, "sample_id": row["sample_id"]}
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            if index % 8 == 0:
                print(f"labeled {index}/{len(rows)} (reused {min(index, len(reused))})", flush=True)
    print(f"labels -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
