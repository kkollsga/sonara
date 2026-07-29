#!/usr/bin/env python3
"""Generate multi-excerpt CLAP targets without exposing metadata to the rater."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


SONARA_ROOT = Path(__file__).resolve().parents[4]
RATER_ROOT = SONARA_ROOT / "dev-docs" / "bench" / "scripts" / "aggression-blind-review" / "rater"
sys.path.insert(0, str(RATER_ROOT))

from clap_rater import ClapRater  # noqa: E402


POSITIONS = (0.10, 0.35, 0.60, 0.85)
SECONDS = 20.0


def excerpt(source: Path, duration: float, position: float, output: Path) -> None:
    start = max(0.0, min(duration * position, max(0.0, duration - SECONDS)))
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{start:.6f}", "-t", str(SECONDS),
            "-i", str(source), "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le",
            str(output), "-loglevel", "error",
        ],
        check=True,
        capture_output=True,
    )


def score_track(rater: ClapRater, path: Path, duration: float) -> dict:
    excerpts = []
    with tempfile.TemporaryDirectory(prefix="sonagram-rank-") as directory:
        root = Path(directory)
        for index, position in enumerate(POSITIONS):
            audio = root / f"excerpt-{index}.wav"
            excerpt(path, duration, position, audio)
            verdict = rater.score_clip(str(audio))
            excerpts.append(
                {
                    "position": position,
                    "score": verdict.aggression / 100.0,
                    "confidence": verdict.confidence,
                    "insufficient": verdict.insufficient,
                }
            )
    usable = [row["score"] for row in excerpts if not row["insufficient"]]
    if len(usable) < 3:
        return {
            "target": None,
            "median": None,
            "range": None,
            "support": len(usable),
            "insufficient": True,
            "excerpts": excerpts,
        }
    strongest = sorted(usable, reverse=True)[:2]
    return {
        "target": statistics.mean(strongest),
        "median": statistics.median(usable),
        "range": max(usable) - min(usable),
        "support": len(usable),
        "insufficient": False,
        "excerpts": excerpts,
    }


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        row["sample_id"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "development", "locked"])
    parser.add_argument("--anchors", action="store_true")
    args = parser.parse_args()
    if not args.splits and not args.anchors:
        parser.error("select --splits and/or --anchors")

    manifest = [json.loads(line) for line in (args.pool / "manifest.jsonl").read_text().splitlines()]
    private_paths = json.loads((args.pool / "private_paths.json").read_text())
    rater = ClapRater()

    for split in args.splits or []:
        selected = [row for row in manifest if row["split"] == split]
        output = args.pool / f"{split}_labels.jsonl"
        existing = load_existing(output)
        rows = []
        for index, row in enumerate(selected, 1):
            record = existing.get(row["sample_id"])
            if record is None:
                result = score_track(
                    rater, Path(private_paths[row["sample_id"]]), float(row["duration_sec"])
                )
                record = {"sample_id": row["sample_id"], **result}
            rows.append(record)
            write_rows(output, rows)
            if index % 8 == 0:
                print(f"{split} {index}/{len(selected)}", flush=True)
        print(f"labels -> {output}", flush=True)

    if args.anchors:
        anchors = json.loads((args.pool / "anchors-private.json").read_text())
        output = args.pool / "anchor_labels.jsonl"
        rows = []
        for index, row in enumerate(anchors, 1):
            path = Path(row["path"])
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = score_track(rater, path, float(probe.stdout.strip()))
            rows.append(
                {
                    "sample_id": row["anchor_id"],
                    "anchor_id": row["anchor_id"],
                    "role": row["role"],
                    **result,
                }
            )
            write_rows(output, rows)
            print(f"anchors {index}/{len(anchors)}", flush=True)
        print(f"labels -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
