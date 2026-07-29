#!/usr/bin/env python3
"""Build an anonymous, deterministic open-corpus pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile


DEFAULT_CORPUS = Path(
    "/Volumes/EksternalHome/Koding/Rust/ferricml/research/sonara-aggression/out/fma-corpus.json"
)
DEFAULT_AUDIO = Path(
    "/Volumes/EksternalHome/Koding/Rust/ferricml/research/sonara-aggression/data/permissive/fma-labeled/audio"
)
SEED = 20260722
COUNT = 256
SECONDS = 20


def excerpt(source: Path) -> bytes | None:
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        seconds = float(duration.stdout.strip())
    except ValueError:
        return None
    if seconds < 8:
        return None
    start = max(0.0, min(seconds * 0.30, max(0.0, seconds - SECONDS)))
    with tempfile.NamedTemporaryFile(suffix=".wav") as output:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", str(SECONDS),
                "-i", str(source), "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le",
                output.name, "-loglevel", "error",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            return None
        return Path(output.name).read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=COUNT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clips = args.output / "clips"
    clips.mkdir(exist_ok=True)

    corpus = json.loads(args.corpus.read_text())
    candidates = list(corpus["tracks"])
    random.Random(SEED).shuffle(candidates)
    rows = []
    seen_audio: set[str] = set()
    seen_groups: set[str] = set()
    for track in candidates:
        if len(rows) >= args.count:
            break
        if track["audio_sha256"] in seen_audio or track["group_id"] in seen_groups:
            continue
        payload = excerpt(args.audio_root / track["canonical_audio_path"])
        if payload is None:
            continue
        digest = hashlib.sha256(payload).hexdigest()
        destination = clips / f"{digest}.wav"
        destination.write_bytes(payload)
        rows.append(
            {
                "sample_id": f"pilot-{len(rows):04d}",
                "sha256": digest,
                "group_id": track["group_id"],
                "path": f"clips/{digest}.wav",
            }
        )
        seen_audio.add(track["audio_sha256"])
        seen_groups.add(track["group_id"])
        if len(rows) % 32 == 0:
            print(f"prepared {len(rows)}/{args.count}", flush=True)
    if len(rows) != args.count:
        raise RuntimeError(f"prepared {len(rows)} clips, expected {args.count}")
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"pilot -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
