"""Build blind pairwise packets for the aggression review.

For each sampled track: extract a fixed excerpt, content-hash it (sha256 of the
excerpt PCM), and store it content-addressed. Tracks are partitioned into two
cohorts that are DISJOINT BY HASH (dev vs locked), pairs are formed within each
cohort, and provenance is recorded only as an opaque source-A/B/C label. No
artist, title, filename, tag, or Sonara signal enters the manifest.

Usage:  python packet_builder.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLIPS = os.path.join(ROOT, "packets", "clips")
PACKETS = os.path.join(ROOT, "packets")

SEED = 20260722
EXCERPT_SEC = 20
N_DEV_PAIRS = 24
N_LOCKED_PAIRS = 20
AUDIO_EXTS = (".mp3", ".flac", ".m4a", ".wav", ".aac", ".ogg")

# Opaque source label -> real collection root. Labels are all that ship.
SOURCES = {
    "source-A": "/Volumes/EksternalHome/Downloads/Music/Greatest Hits - Collection 1958-2021 _1074 ALBUMS_ MP3 Part 2 Of 2-TL",
    "source-B": "/Volumes/EksternalHome/Downloads/Music/Greatest Hits & Best Of (HipHop-PoP-RaP-RnB)-budyzer",
    "source-C": "/Volumes/EksternalHome/Downloads/Music/The Rolling Stones",
}


def list_audio(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(AUDIO_EXTS):
                out.append(os.path.join(dirpath, f))
    return out


def probe_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def extract_excerpt(path: str) -> bytes | None:
    """Deterministic excerpt: EXCERPT_SEC from 30% into the track, as PCM bytes."""
    dur = probe_duration(path)
    if dur < 8.0:
        return None
    start = max(0.0, min(dur * 0.30, max(0.0, dur - EXCERPT_SEC)))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", str(EXCERPT_SEC),
             "-i", path, "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le",
             out, "-loglevel", "error"],
            check=True, capture_output=True)
        with open(out, "rb") as fh:
            return fh.read()
    except Exception:
        return None
    finally:
        if os.path.exists(out):
            os.remove(out)


def build():
    os.makedirs(CLIPS, exist_ok=True)
    rng = random.Random(SEED)

    n_clips = (N_DEV_PAIRS + N_LOCKED_PAIRS) * 2      # distinct clips needed
    per_source = n_clips // len(SOURCES) + 6          # oversample for skips/dupes

    clips = []          # {sha, source, rel_path}
    seen = set()
    for label, root in SOURCES.items():
        files = list_audio(root)
        rng.shuffle(files)
        taken = 0
        for path in files:
            if taken >= per_source:
                break
            pcm = extract_excerpt(path)
            if pcm is None:
                continue
            sha = hashlib.sha256(pcm).hexdigest()
            if sha in seen:                            # global dedup => disjointness safe
                continue
            seen.add(sha)
            dst = os.path.join(CLIPS, f"{sha}.wav")
            with open(dst, "wb") as fh:
                fh.write(pcm)
            clips.append({"sha256": sha, "source": label,
                          "audio_path": f"clips/{sha}.wav"})
            taken += 1
        print(f"{label}: {taken} clips")

    rng.shuffle(clips)
    need = (N_DEV_PAIRS + N_LOCKED_PAIRS) * 2
    if len(clips) < need:
        raise SystemExit(f"only {len(clips)} clips, need {need}")
    clips = clips[:need]

    dev_clips = clips[:N_DEV_PAIRS * 2]
    locked_clips = clips[N_DEV_PAIRS * 2:]

    def make_pairs(pool, prefix):
        # Prefer cross-source pairs: sort so adjacent items tend to differ.
        pool = sorted(pool, key=lambda c: (c["source"], c["sha256"]))
        half = len(pool) // 2
        left, right = pool[:half], pool[half:]
        rng.shuffle(left)
        rng.shuffle(right)
        rows = []
        for i, (l, r) in enumerate(zip(left, right)):
            rows.append({
                "pair_id": f"{prefix}-{i:03d}",
                "left_sha256": l["sha256"], "left_path": l["audio_path"],
                "left_source": l["source"],
                "right_sha256": r["sha256"], "right_path": r["audio_path"],
                "right_source": r["source"],
            })
        return rows

    dev = make_pairs(dev_clips, "dev")
    locked = make_pairs(locked_clips, "locked")

    with open(os.path.join(PACKETS, "dev_pairs.jsonl"), "w") as fh:
        for row in dev:
            fh.write(json.dumps(row) + "\n")
    with open(os.path.join(PACKETS, "locked_pairs.jsonl"), "w") as fh:
        for row in locked:
            fh.write(json.dumps(row) + "\n")

    print(f"\ndev pairs: {len(dev)}  locked pairs: {len(locked)}")
    print(f"dev sources: {sorted({c['source'] for c in dev_clips})}")
    print(f"locked sources: {sorted({c['source'] for c in locked_clips})}")
    print(f"manifests -> {PACKETS}/dev_pairs.jsonl, locked_pairs.jsonl")


if __name__ == "__main__":
    build()
