#!/usr/bin/env python3
"""Extract the frozen, production-shaped Sonagram aggression rank schema."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path


SONARA_ROOT = Path(__file__).resolve().parents[4]
PROBE_PATH = SONARA_ROOT / "scripts" / "mood_aggression_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("mood_aggression_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clamp(value: float, high: float = 1.0) -> float:
    return max(0.0, min(1.0, float(value) / high))


def production_evidence(y, sr: int, sonara):
    """Mirror the fused Rust evidence schema with one research STFT."""
    import numpy as np

    n_fft = 2048
    hop = 512
    eps = 1e-12
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1), copy=True)
    signal_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if y.size else 0.0
    scaled = y / max(signal_rms, eps)
    centered = np.pad(scaled, (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(centered, n_fft)[::hop]
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    crest = 20.0 * np.log10(
        (np.max(np.abs(frames), axis=1) + eps) / (frame_rms + eps)
    )
    spectrum = np.asarray(sonara.stft(scaled, n_fft=n_fft, hop_length=hop))
    power = np.abs(spectrum).astype(np.float32) ** 2
    mel = sonara.melspectrogram(
        y=scaled, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128
    )
    mel_db = np.asarray(sonara.power_to_db(mel), dtype=np.float64)
    onset_unpadded = np.mean(np.maximum(np.diff(mel_db, axis=1), 0.0), axis=0)
    onset = np.pad(onset_unpadded, (1 + n_fft // (2 * hop), 0))
    onset_scale = max(float(np.quantile(onset, 0.90)) if onset.size else 0.0, eps)
    onset = np.clip(onset / onset_scale, 0.0, 4.0)
    freqs = np.linspace(0.0, sr / 2.0, power.shape[0], dtype=np.float32)
    total = np.sum(power, axis=0) + eps
    high = power[freqs >= 4000.0]
    high_total = np.sum(high, axis=0)
    high_ratio = high_total / total
    flat_eps = 1e-10
    high_floored = np.maximum(high, flat_eps)
    high_flatness = np.exp(np.mean(np.log(high_floored), axis=0)) / np.mean(high_floored, axis=0)
    top_bins = min(8, power.shape[0])
    peak_ratio = np.sum(np.partition(power, -top_bins, axis=0)[-top_bins:], axis=0) / total
    high_flux = np.maximum(np.diff(high_total), 0.0) / np.maximum(high_total[1:], eps)
    sample_count = min(48, power.shape[1])
    sample_frames = np.linspace(0, power.shape[1] - 1, sample_count, dtype=int)
    dissonance = np.asarray(
        [
            sonara.dissonance(power[:, index : index + 1].astype(np.float32), freqs)
            for index in sample_frames
        ],
        dtype=np.float64,
    )
    length = min(len(crest), len(onset), len(high_ratio))
    block = max(1, round(20.0 * sr / hop))
    step = max(1, block // 2)
    starts = list(range(0, max(1, length - block + 1), step))
    if not starts or starts[-1] + block < length:
        starts.append(max(0, length - block))
    windows = []
    for start in sorted(set(starts)):
        stop = min(length, start + block)
        section = slice(start, stop)
        threshold = max(0.30, float(np.quantile(onset[section], 0.50)) + 0.25)
        onset_density = float(np.count_nonzero(onset[section] >= threshold)) / max((stop - start) * hop / sr, eps)
        force = (
            1.0 - clamp(float(np.quantile(crest[section], 0.50)), 20.0)
            + clamp(onset_density, 15.0)
            + clamp(float(np.quantile(onset[section], 0.50)), 2.0)
        ) / 3.0
        harshness = (
            clamp(float(np.quantile(high_ratio[section], 0.50)), 0.35)
            + clamp(float(np.quantile(high_flatness[section], 0.50)))
            + 1.0 - clamp(float(np.quantile(peak_ratio[section], 0.50)))
        ) / 3.0
        windows.append((force, harshness, force * harshness))
    top_two = lambda index: sum(sorted((row[index] for row in windows), reverse=True)[:2]) / min(2, len(windows))
    impacts = [row[2] for row in windows]
    temporal = {
        "window_force_top2": top_two(0),
        "window_harshness_top2": top_two(1),
        "window_impact_top2": top_two(2),
        "window_impact_persistence": sum(value >= 0.25 for value in impacts) / len(impacts),
    }
    onset_threshold = max(0.30, float(np.quantile(onset, 0.50)) + 0.25)
    onset_frames = np.flatnonzero(onset >= onset_threshold)
    intervals = np.diff(onset_frames).astype(np.float64)
    interval_cv = (
        float(np.std(intervals) / max(float(np.mean(intervals)), eps))
        if intervals.size >= 2
        else 0.0
    )
    rms_p90 = float(np.quantile(frame_rms, 0.90))
    non_silent = float(np.mean(frame_rms >= 0.10 * max(rms_p90, eps)))
    support = 0.0 if signal_rms <= 1.0e-6 else non_silent * (
        0.5 * float(np.quantile(peak_ratio, 0.50))
        + 0.5 * (1.0 - float(np.quantile(high_flatness, 0.50)))
    )
    evidence = {
        "crest_db_p50": float(np.quantile(crest, 0.50)),
        "crest_db_p90": float(np.quantile(crest, 0.90)),
        "dissonance_p50": float(np.quantile(dissonance, 0.50)),
        "dissonance_p90": float(np.quantile(dissonance, 0.90)),
        "high_energy_ratio_p50": float(np.quantile(high_ratio, 0.50)),
        "high_energy_ratio_p90": float(np.quantile(high_ratio, 0.90)),
        "high_flatness_p50": float(np.quantile(high_flatness, 0.50)),
        "high_flux_p90": float(np.quantile(high_flux, 0.90)) if high_flux.size else 0.0,
        "onset_density_hz": float(onset_frames.size / max(y.size / sr, eps)),
        "onset_interval_cv": interval_cv,
        "onset_strength_p50": float(np.quantile(onset, 0.50)),
        "onset_strength_p90": float(np.quantile(onset, 0.90)),
        "rms_dynamic_ratio": rms_p90 / max(float(np.quantile(frame_rms, 0.10)), eps),
        "spectral_peak_ratio_p50": float(np.quantile(peak_ratio, 0.50)),
    }
    return max(0.0, min(1.0, support)), evidence, temporal


def schema_features(
    evidence: dict[str, float], embedding: list[float], temporal: dict[str, float]
) -> dict[str, float]:
    physical = {
        "crest_p50": clamp(evidence["crest_db_p50"], 20.0),
        "crest_p90": clamp(evidence["crest_db_p90"], 20.0),
        "dissonance_p50": clamp(evidence["dissonance_p50"], 0.15),
        "dissonance_p90": clamp(evidence["dissonance_p90"], 0.15),
        "high_energy_p50": clamp(evidence["high_energy_ratio_p50"], 0.35),
        "high_energy_p90": clamp(evidence["high_energy_ratio_p90"], 0.35),
        "high_flatness_p50": clamp(evidence["high_flatness_p50"]),
        "high_flux_p90": clamp(math.log1p(max(0.0, evidence["high_flux_p90"])) / math.log(11.0)),
        "onset_density": clamp(evidence["onset_density_hz"], 15.0),
        "onset_interval_cv": clamp(evidence["onset_interval_cv"], 2.0),
        "onset_strength_p50": clamp(evidence["onset_strength_p50"], 2.0),
        "onset_strength_p90": clamp(evidence["onset_strength_p90"], 4.0),
        "rms_dynamic": clamp(math.log(max(1.0, evidence["rms_dynamic_ratio"])) / math.log(10.0)),
        "spectral_peak_ratio": clamp(evidence["spectral_peak_ratio_p50"]),
    }
    selected_embedding = [0, 2, 25, 26, 27, 28, 29, 31, 32, 35, 36, 37, 38, 40, 46]
    features = dict(physical)
    for index in selected_embedding:
        features[f"embedding_{index:02d}"] = clamp(embedding[index])

    force = (
        (1.0 - physical["crest_p50"])
        + physical["onset_density"]
        + physical["onset_strength_p50"]
    ) / 3.0
    harsh = (
        physical["high_energy_p50"]
        + physical["high_flatness_p50"]
        + (1.0 - physical["spectral_peak_ratio"])
    ) / 3.0
    tension = physical["dissonance_p90"]
    dance = clamp(embedding[37])
    regularity = clamp(embedding[38])
    features.update(
        {
            "interaction_force": force,
            "interaction_harshness": harsh,
            "interaction_tension": tension,
            "interaction_force_harshness": force * harsh,
            "interaction_force_tension": force * tension,
            "interaction_regular_clean": dance * regularity * (1.0 - harsh),
        }
    )
    features.update(temporal)
    assert len(features) == 39
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in features.values())
    return dict(sorted(features.items()))


def extract(sonara, _probe, path: Path):
    y, sr = sonara.load(str(path), sr=22_050, mono=True)
    support, evidence, temporal = production_evidence(y, sr, sonara)
    analysis = sonara.analyze_signal(y, sr=sr, features=["embedding"])
    embedding = list(map(float, analysis["embedding"]))
    if len(embedding) != 48 or analysis["embedding_version"] != 2:
        raise RuntimeError("unexpected Sonara embedding schema")
    return support, schema_features(evidence, embedding, temporal)


_WORKER_MODULES = None


def extract_worker(task):
    global _WORKER_MODULES
    if _WORKER_MODULES is None:
        import sonara

        _WORKER_MODULES = (sonara, load_probe())
    sample_id, path = task
    support, features = extract(*_WORKER_MODULES, Path(path))
    return {"sample_id": sample_id, "support": support, **features}


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as stream:
        return {row["sample_id"] for row in csv.DictReader(stream, delimiter="\t")}


def append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    new = not path.exists()
    with path.open("a") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "development", "locked"])
    parser.add_argument("--anchors", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if not args.splits and not args.anchors:
        parser.error("select --splits and/or --anchors")

    import sonara

    probe = load_probe()
    manifest = [json.loads(line) for line in (args.pool / "manifest.jsonl").read_text().splitlines()]
    private_paths = json.loads((args.pool / "private_paths.json").read_text())

    for split in args.splits or []:
        selected = [row for row in manifest if row["split"] == split]
        output = args.pool / f"{split}_features.tsv"
        if args.force and output.exists():
            output.unlink()
        done = existing_ids(output)
        pending = [
            (row["sample_id"], private_paths[row["sample_id"]])
            for row in selected
            if row["sample_id"] not in done
        ]
        if args.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                records = executor.map(extract_worker, pending)
                for index, record in enumerate(records, 1):
                    append_row(output, record, list(record))
                    if index % 16 == 0:
                        print(f"{split} features {index}/{len(pending)}", flush=True)
        else:
            for index, task in enumerate(pending, 1):
                record = extract_worker(task)
                append_row(output, record, list(record))
                if index % 16 == 0:
                    print(f"{split} features {index}/{len(pending)}", flush=True)
        print(f"features -> {output}", flush=True)

    if args.anchors:
        anchors = json.loads((args.pool / "anchors-private.json").read_text())
        output = args.pool / "anchor_features.tsv"
        if output.exists():
            output.unlink()
        for index, row in enumerate(anchors, 1):
            support, features = extract(sonara, probe, Path(row["path"]))
            record = {"sample_id": row["anchor_id"], "support": support, **features}
            append_row(output, record, list(record))
            print(f"anchor features {index}/{len(anchors)}", flush=True)
        print(f"features -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
