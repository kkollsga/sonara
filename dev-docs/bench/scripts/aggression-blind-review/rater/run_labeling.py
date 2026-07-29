"""Score blind packets with a rater and emit protocol-conformant label files.

Each unique clip is auditioned once (cached by sha); pairs are reduced to
left/right/tie/abstain with per-clip 0-100 scores, confidence, and reason tags.
Writes development_labels.json in the open and SEALS locked_labels.json (writes
it, reports only its hash + counts -- never its contents).

Usage:  python run_labeling.py clap
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

from base import REASON_TAGS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACKETS = os.path.join(ROOT, "packets")

TIE_BAND = 8.0          # |Δaggression| below this => tie
EVALUATOR = "anon-audio-rater/clap-htsat-unfused-v1"
BLINDNESS = ("No model output, metadata, genre, artist, title, feature vector, "
             "or spectrogram was consulted; only loudness-matched audio + the "
             "aggression rubric were given to the rater.")


def load_rater(kind: str):
    if kind == "clap":
        from clap_rater import ClapRater
        return ClapRater()
    if kind == "omni":
        from omni_rater import OmniRater
        return OmniRater()
    raise SystemExit(f"unknown rater '{kind}'")


def read_pairs(name: str):
    rows = []
    with open(os.path.join(PACKETS, name)) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reduce_pair(pair, vL, vR):
    if vL.insufficient or vR.insufficient:
        return {
            "more_aggressive": "abstain",
            "left_score": round(vL.aggression), "right_score": round(vR.aggression),
            "confidence": 0.2,
            "reason_tags": ["insufficient_music_content"],
            "note": "one or both clips lack judgeable musical content",
        }
    diff = vL.aggression - vR.aggression
    if abs(diff) < TIE_BAND:
        decision, winner = "tie", None
    elif diff > 0:
        decision, winner = "left", vL
    else:
        decision, winner = "right", vR

    if winner is not None:
        tags = list(winner.reason_tags)
    else:
        tags = list(dict.fromkeys(vL.reason_tags + vR.reason_tags))
    tags = [t for t in tags if t in REASON_TAGS][:4] or ["other"]

    base_conf = min(vL.confidence, vR.confidence)
    margin = min(1.0, abs(diff) / 40.0)
    confidence = round(base_conf * (0.4 + 0.6 * margin), 2) if decision != "tie" \
        else round(base_conf * 0.4, 2)

    return {
        "more_aggressive": decision,
        "left_score": round(vL.aggression), "right_score": round(vR.aggression),
        "confidence": confidence,
        "reason_tags": tags,
        "note": "",
    }


def label_cohort(rater, pairs, cohort):
    cache = {}

    def score(sha, path):
        if sha not in cache:
            cache[sha] = rater.score_clip(os.path.join(PACKETS, path))
        return cache[sha]

    out_pairs = []
    for p in pairs:
        vL = score(p["left_sha256"], p["left_path"])
        vR = score(p["right_sha256"], p["right_path"])
        red = reduce_pair(p, vL, vR)
        out_pairs.append({
            "pair_id": p["pair_id"],
            "left_sha256": p["left_sha256"], "right_sha256": p["right_sha256"],
            **red,
        })
    return {
        "protocol_version": 1,
        "evaluator": EVALUATOR,
        "audio_perception_confirmed": True,
        "blindness_statement": BLINDNESS,
        "cohort": cohort,
        "pairs": out_pairs,
    }


def summarize(doc):
    dec = [p["more_aggressive"] for p in doc["pairs"]]
    return {
        "pairs": len(dec),
        "left": dec.count("left"), "right": dec.count("right"),
        "tie": dec.count("tie"), "abstain": dec.count("abstain"),
    }


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "clap"
    rater = load_rater(kind)

    dev = label_cohort(rater, read_pairs("dev_pairs.jsonl"), "development")
    dev_path = os.path.join(ROOT, "development_labels.json")
    with open(dev_path, "w") as fh:
        json.dump(dev, fh, indent=2)
    print(f"development -> {dev_path}")
    print(f"  {summarize(dev)}")

    locked = label_cohort(rater, read_pairs("locked_pairs.jsonl"), "locked")
    locked_path = os.path.join(ROOT, "locked_labels.json")
    blob = json.dumps(locked, indent=2).encode()
    with open(locked_path, "wb") as fh:
        fh.write(blob)
    sha = hashlib.sha256(blob).hexdigest()
    s = summarize(locked)
    print("\n=== LOCKED (sealed -- contents not shown) ===")
    print(f"  path: {locked_path}")
    print(f"  sha256: {sha}")
    print(f"  pairs: {s['pairs']}  ties: {s['tie']}  abstains: {s['abstain']}")


if __name__ == "__main__":
    main()
