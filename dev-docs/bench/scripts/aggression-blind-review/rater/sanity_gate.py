"""Perception sanity gate -- the rater's version of `audio_perception_confirmed`.

Feeds synthesized controls with KNOWN ground truth through a rater and checks
it actually hears: harsh > calm, harsh > loud-but-clean (the loudness-confound
test, since all clips are matched to -23 LUFS first), and silence flagged as
insufficient. If a rater cannot pass this, it must not label real pairs --
the same refusal logic a human evaluator owes when they cannot audition audio.

Usage:  python sanity_gate.py [clap|omni]
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONTROLS = os.path.join(ROOT, "controls")
OUT = os.path.join(ROOT, "out")


def load_rater(kind: str):
    if kind == "clap":
        from clap_rater import ClapRater
        return ClapRater()
    if kind == "omni":
        from omni_rater import OmniRater  # built once weights land
        return OmniRater()
    raise SystemExit(f"unknown rater '{kind}' (use clap|omni)")


def main() -> int:
    kind = sys.argv[1] if len(sys.argv) > 1 else "clap"
    rater = load_rater(kind)

    clips = {
        "harsh": "harsh_distortion.wav",
        "calm": "calm_pad.wav",
        "silence": "silence.wav",
        "loud_clean": "loud_clean_pulse.wav",
    }
    verdicts = {}
    for label, fname in clips.items():
        v = rater.score_clip(os.path.join(CONTROLS, fname))
        verdicts[label] = v.to_dict()
        print(f"{label:11s} aggr={v.aggression:6.1f}  conf={v.confidence:.2f}  "
              f"insuff={v.insufficient}  tags={v.reason_tags}")

    h = verdicts["harsh"]["aggression"]
    c = verdicts["calm"]["aggression"]
    l = verdicts["loud_clean"]["aggression"]

    checks = {
        "harsh_gt_calm": h > c,
        "harsh_gt_loud_clean (loudness-confound)": h > l,
        "harsh_reads_aggressive (>=55)": h >= 55.0,
        "calm_not_aggressive (<=55)": c <= 55.0,
        "silence_flagged_insufficient": bool(verdicts["silence"]["insufficient"]),
    }
    passed = all(checks.values())

    print("\n--- perception checks ---")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nGATE: {'PASS -- rater perceives audio' if passed else 'FAIL -- do not label'}")

    os.makedirs(OUT, exist_ok=True)
    report = {
        "rater": rater.name,
        "audio_perception_confirmed": passed,
        "checks": checks,
        "verdicts": verdicts,
    }
    dst = os.path.join(OUT, f"perception_gate_{kind}.json")
    with open(dst, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"report -> {dst}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
