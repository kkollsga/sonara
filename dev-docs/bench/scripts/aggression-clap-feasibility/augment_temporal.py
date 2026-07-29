#!/usr/bin/env python3
"""Prototype richer temporal/spectral Sonara evidence without runtime changes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


N_FFT = 2048
HOP = 512
EPS = 1e-12


def summaries(np, prefix, values, quantiles=(0.10, 0.50, 0.90)):
    values = np.asarray(values, dtype=np.float64)
    result = {f"{prefix}_q{round(q * 100):02d}": float(np.quantile(values, q)) for q in quantiles}
    result[f"{prefix}_std"] = float(np.std(values))
    return result


def extract(np, sonara, y, sr):
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1), copy=True)
    scaled = y / max(float(np.sqrt(np.mean(y.astype(np.float64) ** 2))), EPS)
    magnitude = np.abs(np.asarray(sonara.stft(scaled, n_fft=N_FFT, hop_length=HOP))).astype(np.float64)
    power = magnitude**2
    freqs = np.linspace(0.0, sr / 2.0, power.shape[0])
    total = np.sum(power, axis=0) + EPS
    result = {}

    centroid = np.sum(freqs[:, None] * power, axis=0) / total
    bandwidth = np.sqrt(np.sum((freqs[:, None] - centroid) ** 2 * power, axis=0) / total)
    cdf = np.cumsum(power, axis=0)
    rolloff = freqs[np.argmax(cdf >= 0.85 * total, axis=0)]
    flatness = np.exp(np.mean(np.log(power + EPS), axis=0)) / (np.mean(power + EPS, axis=0) + EPS)
    for name, values in (
        ("frame_centroid", centroid),
        ("frame_bandwidth", bandwidth),
        ("frame_rolloff", rolloff),
        ("frame_flatness", flatness),
    ):
        result.update(summaries(np, name, values))

    edges = np.geomspace(50.0, sr / 2.0, 9)
    for band, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (freqs >= lo) & (freqs < hi)
        band_power = power[mask]
        band_magnitude = magnitude[mask]
        energy_ratio = np.sum(band_power, axis=0) / total
        band_flatness = np.exp(np.mean(np.log(band_power + EPS), axis=0)) / (
            np.mean(band_power + EPS, axis=0) + EPS
        )
        flux = np.sum(np.maximum(np.diff(band_power, axis=1), 0.0), axis=0) / (
            np.sum(band_power[:, 1:], axis=0) + EPS
        )
        valley = np.quantile(band_magnitude, 0.02, axis=0)
        peak = np.quantile(band_magnitude, 0.98, axis=0)
        contrast = np.log10(peak + EPS) - np.log10(valley + EPS)
        prefix = f"band{band}"
        result.update(summaries(np, f"{prefix}_energy", energy_ratio))
        result.update(summaries(np, f"{prefix}_flatness", band_flatness))
        result.update(summaries(np, f"{prefix}_flux", flux))
        result.update(summaries(np, f"{prefix}_contrast", contrast))

    mfcc = np.asarray(
        sonara.mfcc(y=scaled, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP, n_mels=64),
        dtype=np.float64,
    )
    for coefficient in range(mfcc.shape[0]):
        result.update(summaries(np, f"mfcc{coefficient:02d}", mfcc[coefficient]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--review-package", type=Path)
    args = parser.parse_args()
    if (args.pilot is None) == (args.review_package is None):
        parser.error("provide exactly one of --pilot or --review-package")

    import numpy as np
    import sonara

    with args.input.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if args.pilot:
        paths = {
            row["sha256"]: args.pilot / row["path"]
            for row in (json.loads(line) for line in (args.pilot / "manifest.jsonl").read_text().splitlines())
        }
    else:
        paths = {
            row["sha256"]: args.review_package / "packets" / "clips" / f"{row['sha256']}.wav"
            for row in rows
        }

    feature_names = None
    for index, row in enumerate(rows, 1):
        y, sr = sonara.load(str(paths[row["sha256"]]), sr=22_050, mono=True)
        evidence = extract(np, sonara, y, sr)
        names = sorted(evidence)
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise RuntimeError("temporal evidence schema drift")
        row.update({name: f"{evidence[name]:.9g}" for name in names})
        if index % 32 == 0:
            print(f"temporal {index}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows with {len(feature_names or [])} temporal features -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
