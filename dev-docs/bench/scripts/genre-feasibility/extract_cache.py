#!/usr/bin/env python3
"""P6 genre feasibility — step 1: one-pass extraction from the sonagram cache.

Reads every record in the sonagram analysis cache and writes one minimal JSONL
row per record (content_hash, path, artist, ID3 genre, 48-dim embedding,
embedding_version, provenance sample_rate/mode). Everything downstream works
from this file, so the 32,890-file scan happens once.

Output: dev-docs/bench/out/genre-feasibility/cache_extract.jsonl
"""

import json
import os
import sys

CACHE = "/Volumes/EksternalHome/Downloads/Music/.sonagram/analysis"
OUT = ("/Volumes/EksternalHome/Koding/Rust/sonara/dev-docs/bench/out/"
       "genre-feasibility/cache_extract.jsonl")


def main():
    names = sorted(n for n in os.listdir(CACHE) if n.endswith(".json"))
    n_ok = n_no_emb = n_err = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for i, name in enumerate(names):
            if i % 5000 == 0:
                print(f"  {i}/{len(names)} ...", flush=True)
            try:
                with open(os.path.join(CACHE, name), encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception as e:  # noqa: BLE001 - survey pass
                n_err += 1
                print(f"  ERROR {name}: {e}", file=sys.stderr)
                continue
            a = d.get("analysis") or {}
            emb = a.get("embedding")
            if not emb or len(emb) != 48:
                n_no_emb += 1
                continue
            tags = d.get("tags") or {}
            prov = a.get("provenance") or {}
            src = d.get("source") or {}
            row = {
                "hash": src.get("content_hash"),
                "path": src.get("path"),
                "artist": tags.get("artist"),
                "id3_genre": tags.get("genre"),
                "emb": emb,
                "emb_v": a.get("embedding_version"),
                "sr": prov.get("sample_rate"),
                "mode": prov.get("mode"),
                "schema": prov.get("schema_version"),
            }
            out.write(json.dumps(row) + "\n")
            n_ok += 1
    print(f"extracted {n_ok} rows ({n_no_emb} without 48-dim embedding, "
          f"{n_err} unreadable) -> {OUT}")


if __name__ == "__main__":
    main()
