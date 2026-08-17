#!/usr/bin/env python3
"""P6 genre feasibility — step 2: survey label sources before fixing taxonomy.

Reads cache_extract.jsonl + Last.fm artists.json, reports:
  - ID3 genre raw distribution (top 60, after basic cleanup)
  - artist-name join rate between cache and Last.fm entries
  - Last.fm first-tag distribution over joined artists
No taxonomy decisions here; output informs the prereg doc.
"""

import json
import re
from collections import Counter

OUTDIR = ("/Volumes/EksternalHome/Koding/Rust/sonara/dev-docs/bench/out/"
          "genre-feasibility")
EXTRACT = f"{OUTDIR}/cache_extract.jsonl"
LASTFM = "/Volumes/EksternalHome/Downloads/Music/.sonagram/lastfm/artists.json"

_PAREN_NUM = re.compile(r"^\(\d+\)\s*")


def clean_id3(g):
    """Strip numeric ID3v1 residue, lower-case, split multi-values."""
    if not g:
        return []
    g = _PAREN_NUM.sub("", g.strip())
    parts = re.split(r"[;/,]| \| ", g)
    return [p.strip().lower() for p in parts if p.strip()]


def norm_artist(a):
    if not a:
        return None
    return re.sub(r"\s+", " ", a.strip().lower())


def main():
    rows = [json.loads(l) for l in open(EXTRACT, encoding="utf-8")]
    print(f"{len(rows)} cache rows")

    lf = json.load(open(LASTFM, encoding="utf-8"))["entries"]
    lf_norm = {}
    for name, e in lf.items():
        if e.get("fetched") and e.get("tags"):
            lf_norm[norm_artist(name)] = [t.lower() for t in e["tags"]]
    print(f"{len(lf_norm)} lastfm artists with tags")

    id3_counter = Counter()
    joined = 0
    lf_first_tag = Counter()
    artists_seen = set()
    both = 0
    for r in rows:
        gs = clean_id3(r["id3_genre"])
        for g in gs[:1]:
            id3_counter[g] += 1
        a = norm_artist(r["artist"])
        tags = lf_norm.get(a)
        if tags:
            joined += 1
            if a not in artists_seen:
                artists_seen.add(a)
                lf_first_tag[tags[0]] += 1
            if gs:
                both += 1
    print(f"tracks joined to a tagged lastfm artist: {joined} "
          f"({100*joined/len(rows):.1f}%)")
    print(f"tracks with BOTH id3 genre and lastfm artist tags: {both}")
    print(f"distinct joined artists: {len(artists_seen)}")

    print("\n== top 60 ID3 first-genres (cleaned) ==")
    for g, c in id3_counter.most_common(60):
        print(f"  {c:6d}  {g}")
    print("\n== lastfm first-tag over joined artists (top 40) ==")
    for g, c in lf_first_tag.most_common(40):
        print(f"  {c:6d}  {g}")


if __name__ == "__main__":
    main()
