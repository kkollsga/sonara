#!/usr/bin/env python3
"""P6 genre feasibility — step 3: build the weak-labeled pool + artist split.

Label rule (high-precision weak labels):
  - ID3 coarse label: split tags.genre on ; / , and |, strip "(NN)" residue,
    map every part through TAG2COARSE. Exactly one distinct coarse label
    -> that's the ID3 label; conflicting parts -> track discarded (ambiguous);
    no mappable part -> no ID3 label.
  - Last.fm coarse label: artist tags scanned in listed (popularity) order,
    first mappable tag wins.
  - Keep a track when BOTH labels exist and AGREE; additionally (uniform
    fallback, all classes alike) keep a track on the Last.fm label alone
    when the file carries NO ID3 genre at all — with no ID3 evidence there
    is nothing to disagree with. An unmapped or conflicting ID3 genre is
    evidence of a straddle and still discards the track.

Split rule (artist-disjoint): every normalized artist is wholly train or
wholly eval, judged by its majority class. Classes are processed thinnest
first; each class's artists are shuffled (seed 42) and assigned to eval
until eval has >= 5 artists and >= max(40, 25% of the class's eval-capped
track total) tracks (ceiling 300), never taking more than 40% of a class's
artists into eval. Per-artist caps: 60 train tracks / 15 eval tracks per
class (random subsample, seed 42) so no single artist defines a class.

Class viability (preregistered): a class ships in the spike only with
eval >= 40 tracks from >= 5 artists AND train >= 100 tracks from >= 8
artists. Fewer than 6 surviving classes = NO-GO on pool grounds.

Outputs (dev-docs/bench/out/genre-feasibility/):
  pool.jsonl   — one row per kept track: hash, path, artist, label, split, emb
  pool_summary.json — counts per class/split, surviving classes
"""

import json
import re
from collections import Counter, defaultdict

import numpy as np

OUTDIR = ("/Volumes/EksternalHome/Koding/Rust/sonara/dev-docs/bench/out/"
          "genre-feasibility")
EXTRACT = f"{OUTDIR}/cache_extract.jsonl"
LASTFM = "/Volumes/EksternalHome/Downloads/Music/.sonagram/lastfm/artists.json"

EVAL_FRAC_MIN = 0.25          # of the class's eval-capped track total
EVAL_TRACKS_CEIL = 300        # capped eval tracks per class, absolute ceiling
EVAL_ARTIST_FRAC_MAX = 0.40   # never take more artists than this into eval
TRAIN_CAP = 60
EVAL_CAP = 15
MIN_EVAL_TRACKS, MIN_EVAL_ARTISTS = 40, 5
MIN_TRAIN_TRACKS, MIN_TRAIN_ARTISTS = 100, 8
SEED = 42

# ---- coarse taxonomy ------------------------------------------------------
# Ambiguous straddlers (pop rock, new wave, synthpop, folk rock, doo wop,
# country rock, dance-pop, grunge...) are deliberately UNMAPPED: with two
# noisy sources, discard-on-ambiguity buys precision cheaply.
TAG2COARSE = {}


def _m(coarse, *tags):
    for t in tags:
        TAG2COARSE[t] = coarse


_m("rock", "rock", "classic rock", "hard rock", "progressive rock",
   "alternative rock", "alt. rock", "alternative", "punk rock", "punk",
   "glam rock", "latin rock", "christian rock", "southern rock",
   "rock & roll", "rock and roll", "rock'n'roll", "rock n roll",
   "rockabilly", "beat music", "garage rock", "psychedelic rock",
   "psychedelic", "blues rock", "indie rock", "surf rock", "surf",
   "heavy metal", "thrash metal", "metal", "power metal", "nu metal",
   "speed metal", "glam metal", "britpop", "arena rock", "soft rock")
_m("pop", "pop", "pop music", "поп", "europop", "teen pop", "bubblegum",
   "dance pop", "schlager")
_m("soul-funk-disco", "soul", "r&b", "rnb", "soul and r&b", "r&b/soul",
   "funk", "disco", "motown", "quiet storm", "neo-soul", "neo soul",
   "funk / soul", "soul, r&b")
_m("hip-hop-rap", "hip-hop", "hip hop", "rap", "rap & hip-hop",
   "rap & hip hop", "hip hop rap", "hip-hop/rap", "gangsta rap", "trap",
   "east coast rap", "west coast rap", "gangsta")
_m("electronic-dance", "electronic", "electronica", "dance", "eurodance",
   "house", "deep house", "trance", "vocal trance", "techno", "edm",
   "italo disco", "italo-disco", "eurodisco", "euro disco", "electro",
   "drum and bass", "drum & bass", "dubstep", "synthwave", "dance & dj",
   "club", "hi-nrg", "eurohouse", "progressive trance", "progressive house")
_m("jazz-blues", "jazz", "blues", "big band", "swing", "smooth jazz",
   "bebop", "delta blues", "vocal jazz", "jazz vocal", "blues & jazz")
_m("folk-country", "folk", "country", "singer-songwriter",
   "singer songwriter", "americana", "bluegrass", "songwriter & folk",
   "country & folk", "acoustic")
_m("reggae", "reggae", "ska", "dancehall", "dub")

_PAREN_NUM = re.compile(r"^\(\d+\)\s*")


def id3_coarse(g):
    if not g:
        return None, "none"
    g = _PAREN_NUM.sub("", g.strip())
    parts = [p.strip().lower() for p in re.split(r"[;/,|]", g) if p.strip()]
    mapped = {TAG2COARSE[p] for p in parts if p in TAG2COARSE}
    if len(mapped) == 1:
        return next(iter(mapped)), "ok"
    return None, ("conflict" if len(mapped) > 1 else "unmapped")


def lastfm_coarse(tags):
    for t in tags:
        c = TAG2COARSE.get(t.lower())
        if c:
            return c
    return None


def norm_artist(a):
    if not a:
        return None
    return re.sub(r"\s+", " ", a.strip().lower())


def main():
    rows = [json.loads(l) for l in open(EXTRACT, encoding="utf-8")]
    lf = json.load(open(LASTFM, encoding="utf-8"))["entries"]
    lf_coarse = {}
    for name, e in lf.items():
        if e.get("fetched") and e.get("tags"):
            c = lastfm_coarse(e["tags"])
            if c:
                lf_coarse[norm_artist(name)] = c

    stats = Counter()
    kept = []
    for r in rows:
        a = norm_artist(r["artist"])
        lfc = lf_coarse.get(a)
        idc, why = id3_coarse(r["id3_genre"])
        stats[f"id3_{why}"] += 1
        if lfc is None:
            stats["no_lastfm_label"] += 1
            continue
        if why == "none":
            stats["lastfm_only"] += 1  # no ID3 evidence -> lastfm fallback
            label = lfc
        elif idc is None:
            continue  # unmapped/conflicting ID3 = straddle evidence
        elif idc != lfc:
            stats["disagree"] += 1
            continue
        else:
            stats["agree"] += 1
            label = idc
        kept.append({"hash": r["hash"], "path": r["path"], "artist": a,
                     "label": label, "emb": r["emb"]})

    print("label funnel:", dict(stats))
    by_class = Counter(k["label"] for k in kept)
    print("agreed tracks per class:", dict(by_class.most_common()))

    # ---- artist-disjoint split (per-class aware, thinnest classes first) --
    by_artist = defaultdict(list)
    for k in kept:
        by_artist[k["artist"]].append(k)
    rng = np.random.default_rng(SEED)
    total = Counter(k["label"] for k in kept)
    maj_of = {a: Counter(k["label"] for k in items).most_common(1)[0][0]
              for a, items in by_artist.items()}
    split_of = {}
    for cls in sorted(total, key=lambda c: total[c]):
        cls_artists = sorted(a for a, m in maj_of.items() if m == cls)
        rng.shuffle(cls_artists)
        n_of = {a: sum(1 for k in by_artist[a] if k["label"] == cls)
                for a in cls_artists}
        capped_total = sum(min(n, EVAL_CAP) for n in n_of.values())
        target = min(max(MIN_EVAL_TRACKS, EVAL_FRAC_MIN * capped_total),
                     EVAL_TRACKS_CEIL)
        max_eval_artists = max(MIN_EVAL_ARTISTS,
                               int(EVAL_ARTIST_FRAC_MAX * len(cls_artists)))
        ev_art = ev_tracks = 0
        for a in cls_artists:
            need = ev_art < MIN_EVAL_ARTISTS or ev_tracks < target
            room = ev_art < max_eval_artists
            if need and room:
                split_of[a] = "eval"
                ev_art += 1
                ev_tracks += min(n_of[a], EVAL_CAP)
            else:
                split_of[a] = "train"

    # ---- per-artist-per-class caps ----
    final = []
    for a in sorted(by_artist):
        split = split_of[a]
        cap = EVAL_CAP if split == "eval" else TRAIN_CAP
        per_cls = defaultdict(list)
        for k in by_artist[a]:
            per_cls[k["label"]].append(k)
        for cls, items in per_cls.items():
            items = sorted(items, key=lambda k: k["hash"])
            if len(items) > cap:
                idx = rng.choice(len(items), size=cap, replace=False)
                items = [items[i] for i in sorted(idx)]
            for k in items:
                k = dict(k)
                k["split"] = split
                final.append(k)

    # ---- viability ----
    summary = {}
    surviving = []
    for cls in sorted(total):
        tr = [k for k in final if k["label"] == cls and k["split"] == "train"]
        ev = [k for k in final if k["label"] == cls and k["split"] == "eval"]
        tra = len({k["artist"] for k in tr})
        eva = len({k["artist"] for k in ev})
        ok = (len(ev) >= MIN_EVAL_TRACKS and eva >= MIN_EVAL_ARTISTS
              and len(tr) >= MIN_TRAIN_TRACKS and tra >= MIN_TRAIN_ARTISTS)
        summary[cls] = {"train_tracks": len(tr), "train_artists": tra,
                        "eval_tracks": len(ev), "eval_artists": eva,
                        "viable": ok}
        if ok:
            surviving.append(cls)
        print(f"  {cls:18s} train {len(tr):5d} ({tra:3d} artists)  "
              f"eval {len(ev):4d} ({eva:3d} artists)  "
              f"{'OK' if ok else 'DROP'}")
    print(f"surviving classes: {len(surviving)}: {surviving}")

    final = [k for k in final if k["label"] in surviving]
    with open(f"{OUTDIR}/pool.jsonl", "w", encoding="utf-8") as out:
        for k in final:
            out.write(json.dumps(k) + "\n")
    with open(f"{OUTDIR}/pool_summary.json", "w", encoding="utf-8") as out:
        json.dump({"funnel": dict(stats), "classes": summary,
                   "surviving": surviving, "seed": SEED,
                   "caps": {"train": TRAIN_CAP, "eval": EVAL_CAP},
                   "eval_frac_min": EVAL_FRAC_MIN,
                   "eval_tracks_ceil": EVAL_TRACKS_CEIL,
                   "eval_artist_frac_max": EVAL_ARTIST_FRAC_MAX},
                  out, indent=2)
    print(f"wrote {len(final)} rows -> {OUTDIR}/pool.jsonl")


if __name__ == "__main__":
    main()
