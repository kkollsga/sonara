#!/usr/bin/env python3
"""Extract Sonara physical evidence for the labeled pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sonara

    probe = load_probe()
    manifest = [json.loads(line) for line in (args.pilot / "manifest.jsonl").read_text().splitlines()]
    labels = {
        row["sha256"]: row
        for row in (
            json.loads(line) for line in (args.pilot / "clap_labels.jsonl").read_text().splitlines()
        )
    }
    output_rows = []
    feature_names = None
    for index, row in enumerate(manifest, 1):
        label = labels[row["sha256"]]
        if label["insufficient"]:
            continue
        y, sr = sonara.load(str(args.pilot / row["path"]), sr=22_050, mono=True)
        evidence = probe.extract_evidence(y, sr)
        names = sorted(evidence)
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise RuntimeError("evidence schema drift")
        output_rows.append((row, label, evidence))
        if index % 32 == 0:
            print(f"features {index}/{len(manifest)}", flush=True)
    if feature_names is None:
        raise RuntimeError("no usable pilot rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        stream.write("sample_id\tgroup_id\tsha256\ttarget\t" + "\t".join(feature_names) + "\n")
        for row, label, evidence in output_rows:
            values = "\t".join(f"{evidence[name]:.9g}" for name in feature_names)
            stream.write(
                f"{row['sample_id']}\t{row['group_id']}\t{row['sha256']}\t"
                f"{label['target']:.9g}\t{values}\n"
            )
    print(f"wrote {len(output_rows)} pilot rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
