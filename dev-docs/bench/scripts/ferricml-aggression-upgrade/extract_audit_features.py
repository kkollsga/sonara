#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SONARA = HERE.parents[3]
OUT = SONARA / "dev-docs/bench/out/ferricml-aggression-audit"
EXTRACTOR_PATH = SONARA / "dev-docs/bench/scripts/sonagram-aggression-ranking/extract_features.py"


def load_extractor():
    spec = importlib.util.spec_from_file_location("aggression_extract", EXTRACTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(task):
    sample_id, path = task
    import sonara
    extractor = load_extractor()
    support, features = extractor.extract(sonara, None, Path(path))
    return {"sample_id": sample_id, "support": support, **features}


def main():
    manifest = [json.loads(line) for line in (OUT / "manifest.jsonl").read_text().splitlines()]
    tasks = [(row["sample_id"], str(OUT / row["clip"])) for row in manifest]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        for index, row in enumerate(executor.map(worker, tasks), 1):
            rows.append(row)
            print(f"features {index}/{len(tasks)}", flush=True)
    output = OUT / "features.tsv"
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"features -> {output}")


if __name__ == "__main__":
    main()
