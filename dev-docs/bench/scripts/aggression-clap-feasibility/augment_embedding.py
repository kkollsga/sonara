#!/usr/bin/env python3
"""Add Sonara's existing 48D analysis embedding to feasibility TSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--review-package", type=Path)
    args = parser.parse_args()
    if (args.pilot is None) == (args.review_package is None):
        parser.error("provide exactly one of --pilot or --review-package")

    import sonara

    with args.input.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if args.pilot:
        manifest = {
            row["sha256"]: args.pilot / row["path"]
            for row in (
                json.loads(line)
                for line in (args.pilot / "manifest.jsonl").read_text().splitlines()
            )
        }
    else:
        manifest = {
            row["sha256"]: args.review_package / "packets" / "clips" / f"{row['sha256']}.wav"
            for row in rows
        }

    names = [f"embedding_{index:02d}" for index in range(48)]
    for index, row in enumerate(rows, 1):
        y, sr = sonara.load(str(manifest[row["sha256"]]), sr=22_050, mono=True)
        analysis = sonara.analyze_signal(y, sr=sr, features=["embedding"])
        embedding = analysis["embedding"]
        if len(embedding) != len(names) or analysis["embedding_version"] != 2:
            raise RuntimeError("unexpected Sonara embedding schema")
        row.update(zip(names, (f"{float(value):.9g}" for value in embedding)))
        if index % 32 == 0:
            print(f"embedding {index}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0].keys()], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
