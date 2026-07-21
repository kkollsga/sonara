#!/usr/bin/env python3
"""Offline, deterministic aggression-evidence probe (not a public scorer).

The probe resolves opaque content hashes through a Sonagram index.  Paths and
tag metadata are never written to its NDJSON output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "reference_data" / "mood_aggression_development.json"
DEFAULT_DECISION = ROOT / "tests" / "reference_data" / "mood_aggression_phase6_decision.json"
SCHEMA = "sonara.mood-aggression-evidence.v1"
N_FFT = 2048
HOP = 512
EPS = 1e-12


def _dependencies():
    try:
        import numpy as np
        import sonara
    except ImportError as error:
        raise RuntimeError(
            "probe requires NumPy and a release-mode sonara development install"
        ) from error
    return np, sonara


def _finite(value: float) -> float:
    return round(float(value), 8) if math.isfinite(float(value)) else 0.0


def _quantile(np: Any, values: Any, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return _finite(np.quantile(values, q)) if values.size else 0.0


def extract_evidence(y: Any, sr: int) -> dict[str, float]:
    """Extract preregistered physical evidence; intentionally returns no score."""
    np, sonara = _dependencies()
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1), copy=True)
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))

    signal_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    scaled = y / max(signal_rms, EPS)
    frames = np.lib.stride_tricks.sliding_window_view(scaled, N_FFT)[::HOP]
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPS)
    frame_peak = np.max(np.abs(frames), axis=1)
    crest_db = 20.0 * np.log10((frame_peak + EPS) / (frame_rms + EPS))

    onset = np.asarray(sonara.onset_strength(scaled, sr=sr, hop_length=HOP), dtype=np.float64)
    onset_scale = max(_quantile(np, onset, 0.90), EPS)
    onset_norm = np.clip(onset / onset_scale, 0.0, 4.0)
    onset_threshold = max(0.30, _quantile(np, onset_norm, 0.50) + 0.25)
    onset_frames = np.flatnonzero(onset_norm >= onset_threshold)
    intervals = np.diff(onset_frames).astype(np.float64)
    interval_cv = (
        float(np.std(intervals) / max(float(np.mean(intervals)), EPS))
        if intervals.size >= 2
        else 0.0
    )

    spectrum = np.asarray(sonara.stft(scaled, n_fft=N_FFT, hop_length=HOP))
    power = np.abs(spectrum).astype(np.float64) ** 2
    freqs = np.linspace(0.0, sr / 2.0, power.shape[0], dtype=np.float32)
    total = np.sum(power, axis=0) + EPS
    high = power[freqs >= 4000.0]
    high_total = np.sum(high, axis=0) + EPS
    high_ratio = high_total / total
    high_flatness = np.exp(np.mean(np.log(high + EPS), axis=0)) / (
        np.mean(high + EPS, axis=0) + EPS
    )
    high_flux = np.sum(np.maximum(np.diff(high, axis=1), 0.0), axis=0) / high_total[1:]
    top_bins = min(8, power.shape[0])
    peak_ratio = np.sum(np.partition(power, -top_bins, axis=0)[-top_bins:], axis=0) / total

    sample_count = min(48, power.shape[1])
    sample_frames = np.linspace(0, power.shape[1] - 1, sample_count, dtype=int)
    dissonance = np.asarray(
        [
            sonara.dissonance(power[:, index : index + 1].astype(np.float32), freqs)
            for index in sample_frames
        ],
        dtype=np.float64,
    )
    non_silent = float(np.mean(frame_rms >= 0.10 * max(_quantile(np, frame_rms, 0.90), EPS)))
    content_confidence = non_silent * (
        0.5 * _quantile(np, peak_ratio, 0.50)
        + 0.5 * (1.0 - _quantile(np, high_flatness, 0.50))
    )

    return {
        "content_confidence": _finite(np.clip(content_confidence, 0.0, 1.0)),
        "crest_db_p50": _quantile(np, crest_db, 0.50),
        "crest_db_p90": _quantile(np, crest_db, 0.90),
        "dissonance_p50": _quantile(np, dissonance, 0.50),
        "dissonance_p90": _quantile(np, dissonance, 0.90),
        "high_energy_ratio_p50": _quantile(np, high_ratio, 0.50),
        "high_energy_ratio_p90": _quantile(np, high_ratio, 0.90),
        "high_flatness_p50": _quantile(np, high_flatness, 0.50),
        "high_flux_p90": _quantile(np, high_flux, 0.90),
        "onset_density_hz": _finite(onset_frames.size / max(y.size / sr, EPS)),
        "onset_interval_cv": _finite(interval_cv),
        "onset_strength_p50": _quantile(np, onset_norm, 0.50),
        "onset_strength_p90": _quantile(np, onset_norm, 0.90),
        "rms_dynamic_ratio": _finite(
            _quantile(np, frame_rms, 0.90) / max(_quantile(np, frame_rms, 0.10), EPS)
        ),
        "spectral_peak_ratio_p50": _quantile(np, peak_ratio, 0.50),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"version", "schema", "hash_kind", "decode", "cases"}
    if set(manifest) != required or manifest["version"] != 1 or manifest["schema"] != SCHEMA:
        raise ValueError("invalid development manifest schema")
    if manifest["hash_kind"] != "mp3-audio-v1":
        raise ValueError("unsupported content hash kind")
    seen: set[str] = set()
    for case in manifest["cases"]:
        if set(case) != {"case_id", "content_hash", "expected_presentation"}:
            raise ValueError("invalid manifest case")
        digest = case["content_hash"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("content_hash must be lowercase SHA-256")
        if digest in seen or case["expected_presentation"] not in {"dance", "rough"}:
            raise ValueError("duplicate hash or invalid presentation label")
        seen.add(digest)
    return manifest


def resolve_cases(manifest: dict[str, Any], library_root: Path, index_path: Path):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_hash = {
        value["content_hash"]: relative
        for relative, value in index.items()
        if isinstance(value, dict) and isinstance(value.get("content_hash"), str)
    }
    for case in manifest["cases"]:
        digest = case["content_hash"]
        relative = by_hash.get(digest)
        if relative is None:
            raise FileNotFoundError(f"content hash missing from index: {digest}")
        audio = library_root / relative
        if not audio.is_file():
            raise FileNotFoundError(f"indexed audio is missing for hash: {digest}")
        yield case, audio


def run_probe(manifest_path: Path, library_root: Path, index_path: Path) -> list[str]:
    _, sonara = _dependencies()
    manifest = load_manifest(manifest_path)
    decode = manifest["decode"]
    lines: list[str] = []
    for case, audio in resolve_cases(manifest, library_root, index_path):
        y, sr = sonara.load(
            str(audio), sr=decode["sample_rate"], mono=True,
            offset=decode["offset_seconds"], duration=decode["duration_seconds"],
        )
        record = {
            "case_id": case["case_id"],
            "content_hash": case["content_hash"],
            "evidence": extract_evidence(y, sr),
            "expected_presentation": case["expected_presentation"],
            "schema": SCHEMA,
        }
        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return lines


def self_test() -> None:
    np, _ = _dependencies()
    load_manifest(DEFAULT_MANIFEST)
    decision = json.loads(DEFAULT_DECISION.read_text(encoding="utf-8"))
    assert decision["decision"] == "NO-GO"
    assert decision["public_mood_formula_change"] is False
    assert decision["consequence"].startswith("Keep heuristic v1 public")
    sr = 22050
    seconds = 8
    t = np.arange(sr * seconds, dtype=np.float32) / sr
    rng = np.random.default_rng(27062026)
    broadband = rng.standard_normal(t.size).astype(np.float32)
    # A tiny deterministic broadband floor keeps the high-band flatness probe
    # away from platform-specific FFT roundoff of a mathematically pure tone.
    base = (
        0.45 * np.sin(2 * np.pi * 220 * t)
        + 0.20 * np.sin(2 * np.pi * 440 * t)
        + 0.02 * np.sin(2 * np.pi * 6000 * t)
        + 0.0002 * broadband
    ).astype(np.float32)
    noisy = (base + 0.08 * broadband).astype(np.float32)
    first = extract_evidence(base, sr)
    assert first == extract_evidence(base, sr), "probe is not exactly deterministic"
    gained = extract_evidence(base * np.float32(0.17), sr)
    for key in first:
        # The Rust FFT consumes f32, so rescaling/re-normalizing may move the
        # final rounded statistic by a few ulps without changing its meaning.
        assert abs(first[key] - gained[key]) <= 1e-4, f"gain invariance failed: {key}"
    rough = extract_evidence(noisy, sr)
    assert rough["high_flatness_p50"] > first["high_flatness_p50"]
    assert rough["high_energy_ratio_p50"] > first["high_energy_ratio_p50"]
    regular = np.zeros(sr * seconds, dtype=np.float32)
    jittered = np.zeros_like(regular)
    regular_times = np.arange(0.5, seconds, 0.5)
    jittered_times = regular_times + np.resize(np.array([-0.12, 0.08, 0.02]), regular_times.size)
    for signal, times in ((regular, regular_times), (jittered, jittered_times)):
        for sample in (times * sr).astype(int):
            signal[sample : sample + 64] = np.hanning(64).astype(np.float32)
    regular_stats = extract_evidence(regular, sr)
    jittered_stats = extract_evidence(jittered, sr)
    assert regular_stats["onset_interval_cv"] < jittered_stats["onset_interval_cv"]
    assert set(first) == {
        "content_confidence", "crest_db_p50", "crest_db_p90", "dissonance_p50",
        "dissonance_p90", "high_energy_ratio_p50", "high_energy_ratio_p90",
        "high_flatness_p50", "high_flux_p90", "onset_density_hz",
        "onset_interval_cv", "onset_strength_p50", "onset_strength_p90",
        "rms_dynamic_ratio", "spectral_peak_ratio_p50",
    }
    print("mood aggression evidence probe: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--library-root", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.library_root is None or args.index is None:
        parser.error("--library-root and --index are required unless --self-test is used")
    for line in run_probe(args.manifest, args.library_root, args.index):
        print(line)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
