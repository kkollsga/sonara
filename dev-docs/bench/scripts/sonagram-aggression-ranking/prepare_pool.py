#!/usr/bin/env python3
"""Prepare an anonymous, recording-deduplicated Sonagram candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEED = "sonagram-aggression-ranking-v1"
POSITION = 0.35
SECONDS = 20.0


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fingerprint(sonara, path: Path, duration: float) -> str:
    start = max(0.0, min(duration * POSITION, max(0.0, duration - SECONDS)))
    y, sr = sonara.load(str(path), sr=22_050, mono=True, offset=start, duration=SECONDS)
    return str(sonara.analyze_signal(y, sr=sr, features=["fingerprint"])["fingerprint"])


def stable_key(content_hash: str) -> str:
    return hashlib.sha256(f"{SEED}:{content_hash}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-cache", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--exclude-clips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    args = parser.parse_args()

    import sonara

    args.output.mkdir(parents=True, exist_ok=True)
    anchors = json.loads(args.anchor_manifest.read_text())
    anchor_paths = {str(Path(row["path"]).resolve()) for row in anchors}
    anchor_hashes = {row.get("content_hash") for row in anchors if row.get("content_hash")}

    excluded_fingerprints = set()
    for index, clip in enumerate(sorted(args.exclude_clips.glob("*.wav")), 1):
        y, sr = sonara.load(str(clip), sr=22_050, mono=True)
        value = str(sonara.analyze_signal(y, sr=sr, features=["fingerprint"])["fingerprint"])
        excluded_fingerprints.add(value)
        if index % 32 == 0:
            print(f"excluded fingerprints {index}", flush=True)

    records = []
    for cache_file in args.analysis_cache.glob("*.json"):
        record = json.loads(cache_file.read_text())
        source = record.get("source", {})
        analysis = record.get("analysis", {})
        relative = source.get("path")
        content_hash = source.get("content_hash")
        duration = analysis.get("duration_sec")
        if not relative or not content_hash or not isinstance(duration, (int, float)) or duration < 70:
            continue
        path = (args.library_root / relative).resolve()
        if not path.is_file() or str(path) in anchor_paths or content_hash in anchor_hashes:
            continue
        records.append((stable_key(content_hash), content_hash, float(duration), path))
    records.sort()

    manifest = []
    private_paths = {}
    seen_fingerprints = set(excluded_fingerprints)
    for _, content_hash, duration, path in records:
        if len(manifest) >= args.count:
            break
        try:
            acoustic = fingerprint(sonara, path, duration)
        except Exception as error:
            print(f"skip decode failure: {type(error).__name__}", flush=True)
            continue
        if acoustic in seen_fingerprints:
            continue
        seen_fingerprints.add(acoustic)
        index = len(manifest)
        sample_id = f"rank-{index:04d}"
        split = "train" if index < 704 else "development" if index < 864 else "locked"
        manifest.append(
            {
                "sample_id": sample_id,
                "content_hash": content_hash,
                "acoustic_fingerprint": acoustic,
                "duration_sec": duration,
                "split": split,
            }
        )
        private_paths[sample_id] = str(path)
        if len(manifest) % 32 == 0:
            print(f"prepared {len(manifest)}/{args.count}", flush=True)

    if len(manifest) != args.count:
        raise RuntimeError(f"prepared {len(manifest)} candidates, expected {args.count}")
    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    )
    (args.output / "private_paths.json").write_text(json.dumps(private_paths, sort_keys=True))
    print(
        f"pool: {len(manifest)} tracks, {len(seen_fingerprints)} fingerprints including exclusions",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
