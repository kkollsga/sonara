#!/usr/bin/env python3
"""Extract the open CLAP development cohort into an anonymous TSV."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SONARA_ROOT = Path(__file__).resolve().parents[4]
PROBE_PATH = SONARA_ROOT / "scripts" / "mood_aggression_probe.py"
DEFAULT_PACKAGE = (
    SONARA_ROOT / "dev-docs" / "bench" / "scripts" / "aggression-blind-review"
)


def load_probe():
    spec = importlib.util.spec_from_file_location("mood_aggression_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sonara

    probe = load_probe()
    labels = json.loads((args.package / "development_labels.json").read_text())
    manifest = {
        row["pair_id"]: row
        for row in read_jsonl(args.package / "packets" / "dev_pairs.jsonl")
    }
    feature_names: list[str] | None = None
    rows: list[tuple[str, str, str, str, float, dict[str, float]]] = []
    seen: set[str] = set()

    for pair in labels["pairs"]:
        pair_manifest = manifest[pair["pair_id"]]
        for side in ("left", "right"):
            digest = pair[f"{side}_sha256"]
            if digest in seen:
                raise RuntimeError(f"development clip reused: {digest}")
            seen.add(digest)
            audio = args.package / "packets" / "clips" / f"{digest}.wav"
            y, sr = sonara.load(str(audio), sr=22_050, mono=True)
            evidence = probe.extract_evidence(y, sr)
            names = sorted(evidence)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("evidence schema drift")
            rows.append(
                (
                    pair["pair_id"],
                    pair_manifest[f"{side}_source"],
                    digest,
                    pair["more_aggressive"],
                    float(pair[f"{side}_score"]) / 100.0,
                    evidence,
                )
            )
            print(f"extracted {len(rows):02d}/48 {pair['pair_id']} {side}", flush=True)

    assert feature_names is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write(
            "pair_id\tsource\tsha256\tdecision\ttarget\t"
            + "\t".join(feature_names)
            + "\n"
        )
        for pair_id, source, digest, decision, target, evidence in rows:
            values = "\t".join(f"{evidence[name]:.9g}" for name in feature_names)
            output.write(
                f"{pair_id}\t{source}\t{digest}\t{decision}\t{target:.9g}\t{values}\n"
            )
    print(f"wrote {len(rows)} rows x {len(feature_names)} features -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
