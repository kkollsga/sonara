#!/usr/bin/env python3
"""Tests for the opt-in extras: silence offsets, key candidates, vocalness,
mood (heuristic v1), instrumentalness.

These features are opt-in only (never enabled by any mode) and must appear in
the result dict only when requested via `features=[...]`. Runs as a standalone
script (exits nonzero on failure), matching the repo's test convention.
"""

import sys
import numpy as np

import sonara

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


SR = 22050
CAMELOT = {f"{n}{s}" for n in range(1, 13) for s in ("A", "B")}


def tone(freq, dur, amp=0.5):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def a_minor_triad(dur=4.0):
    t = np.arange(int(dur * SR)) / SR
    y = (0.5 * np.sin(2 * np.pi * 220.0 * t)
         + 0.4 * np.sin(2 * np.pi * 261.63 * t)
         + 0.35 * np.sin(2 * np.pi * 329.63 * t))
    return y.astype(np.float32)


def signal_with_silence(lead=1.5, mid=3.0, trail=2.25):
    y = np.concatenate([
        np.zeros(int(lead * SR), dtype=np.float32),
        tone(440.0, mid),
        np.zeros(int(trail * SR), dtype=np.float32),
    ])
    return y.astype(np.float32)


# ------------------------------------------------------------
# Opt-in: absent by default in every mode
# ------------------------------------------------------------

OPTIN_KEYS = [
    "leading_silence_sec", "trailing_silence_sec", "key_candidates", "vocalness",
    "mood_happy", "mood_aggressive", "mood_relaxed", "mood_sad", "instrumentalness",
]
MOOD_KEYS = ["mood_happy", "mood_aggressive", "mood_relaxed", "mood_sad"]


def test_absent_by_default():
    y = a_minor_triad()
    for mode in ("compact", "playlist", "full"):
        r = sonara.analyze_signal(y, sr=SR, mode=mode)
        for k in OPTIN_KEYS:
            assert k not in r, f"{k} must be absent in mode={mode}"


test("opt-in keys absent by default (compact/playlist/full)", test_absent_by_default)


# ------------------------------------------------------------
# Silence
# ------------------------------------------------------------

def test_silence_present_and_bounded():
    y = signal_with_silence()
    r = sonara.analyze_signal(y, sr=SR, features=["silence"])
    assert "leading_silence_sec" in r and "trailing_silence_sec" in r
    lead = r["leading_silence_sec"]
    trail = r["trailing_silence_sec"]
    dur = r["duration_sec"]
    assert 0.0 <= lead <= dur, f"lead {lead} out of [0,{dur}]"
    assert 0.0 <= trail <= dur, f"trail {trail} out of [0,{dur}]"
    assert abs(lead - 1.5) < 0.05, f"lead {lead} != 1.5"
    assert abs(trail - 2.25) < 0.05, f"trail {trail} != 2.25"


def test_silence_no_silence_zero():
    y = tone(440.0, 3.0)
    r = sonara.analyze_signal(y, sr=SR, features=["silence"])
    assert r["leading_silence_sec"] < 0.05
    assert r["trailing_silence_sec"] < 0.05


test("silence present + bounded [0, duration]", test_silence_present_and_bounded)
test("silence ~0 for gapless signal", test_silence_no_silence_zero)


# ------------------------------------------------------------
# Key candidates
# ------------------------------------------------------------

def test_key_candidates_shape_and_consistency():
    y = a_minor_triad()
    r = sonara.analyze_signal(y, sr=SR, features=["key", "key_candidates"])
    assert "key_candidates" in r
    cands = r["key_candidates"]
    assert len(cands) == 3, f"expected 3 candidates, got {len(cands)}"
    scores = []
    for entry in cands:
        key, cam, score = entry
        assert isinstance(key, str) and " " in key
        assert cam in CAMELOT, f"invalid camelot {cam}"
        assert 0.0 <= score <= 1.0 and np.isfinite(score)
        scores.append(score)
    assert scores == sorted(scores, reverse=True), "scores must be descending"
    # candidate[0] consistent with r['key']
    assert cands[0][0] == r["key"], f"{cands[0][0]} != {r['key']}"


def test_key_candidates_camelot_matches_key():
    y = a_minor_triad()
    # Also request key_camelot if a parallel feature exposes it; otherwise just
    # verify the first candidate's camelot is a valid code and pairs with key.
    r = sonara.analyze_signal(y, sr=SR, features=["key", "key_candidates"])
    cands = r["key_candidates"]
    assert cands[0][1] in CAMELOT
    if "key_camelot" in r:
        assert cands[0][1] == r["key_camelot"]


test("key_candidates shape + [0] == key + descending", test_key_candidates_shape_and_consistency)
test("key_candidates[0] camelot valid / matches key_camelot", test_key_candidates_camelot_matches_key)


# ------------------------------------------------------------
# Vocalness
# ------------------------------------------------------------

def test_vocalness_present_and_in_range():
    y = a_minor_triad()
    r = sonara.analyze_signal(y, sr=SR, features=["vocalness"])
    assert "vocalness" in r
    v = r["vocalness"]
    assert 0.0 <= v <= 1.0 and np.isfinite(v), f"vocalness {v} out of [0,1]"


def test_vocalness_silence_low():
    y = np.zeros(int(3.0 * SR), dtype=np.float32)
    r = sonara.analyze_signal(y, sr=SR, features=["vocalness"])
    assert r["vocalness"] < 0.3


test("vocalness present + in [0,1]", test_vocalness_present_and_in_range)
test("vocalness low for silence", test_vocalness_silence_low)


# ------------------------------------------------------------
# Mood (heuristic v1)
# ------------------------------------------------------------

def test_mood_present_and_in_range():
    y = a_minor_triad()
    r = sonara.analyze_signal(y, sr=SR, features=["mood"])
    for k in MOOD_KEYS:
        assert k in r, f"{k} missing when mood requested"
        v = r[k]
        assert 0.0 <= v <= 1.0 and np.isfinite(v), f"{k}={v} out of [0,1]"
    # Requesting mood must not leak the key / valence fields.
    assert "key" not in r, "mood must not leak key"
    assert "valence" not in r, "mood must not leak valence"


test("mood present (all four) + in [0,1] + no key/valence leak", test_mood_present_and_in_range)


# ------------------------------------------------------------
# Instrumentalness (heuristic v1: 1 - vocalness)
# ------------------------------------------------------------

def test_instrumentalness_present_and_inverse():
    y = a_minor_triad()
    r = sonara.analyze_signal(y, sr=SR, features=["instrumentalness"])
    assert "instrumentalness" in r
    i = r["instrumentalness"]
    assert 0.0 <= i <= 1.0 and np.isfinite(i), f"instrumentalness {i} out of [0,1]"
    assert "vocalness" not in r, "vocalness must stay absent when only instrumentalness requested"
    # Both together: instrumentalness == 1 - vocalness.
    r2 = sonara.analyze_signal(y, sr=SR, features=["vocalness", "instrumentalness"])
    assert abs(r2["instrumentalness"] - (1.0 - r2["vocalness"])) < 1e-5


test("instrumentalness present + [0,1] + == 1 - vocalness", test_instrumentalness_present_and_inverse)


# ------------------------------------------------------------
# Independence: each opt-in works alone
# ------------------------------------------------------------

def test_independent_optins():
    y = a_minor_triad()
    r_s = sonara.analyze_signal(y, sr=SR, features=["silence"])
    assert "leading_silence_sec" in r_s and "key_candidates" not in r_s and "vocalness" not in r_s
    r_v = sonara.analyze_signal(y, sr=SR, features=["vocalness"])
    assert "vocalness" in r_v and "leading_silence_sec" not in r_v and "key_candidates" not in r_v
    r_m = sonara.analyze_signal(y, sr=SR, features=["mood"])
    assert all(k in r_m for k in MOOD_KEYS)
    assert "vocalness" not in r_m and "instrumentalness" not in r_m and "leading_silence_sec" not in r_m
    r_i = sonara.analyze_signal(y, sr=SR, features=["instrumentalness"])
    assert "instrumentalness" in r_i and "vocalness" not in r_i and not any(k in r_i for k in MOOD_KEYS)


test("opt-ins are independent (each alone)", test_independent_optins)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print(f"\n{'='*70}")
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
print(f"{'='*70}")
if errors:
    print("\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")
sys.exit(1 if failed > 0 else 0)
