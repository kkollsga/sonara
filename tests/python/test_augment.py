#!/usr/bin/env python3
"""Augment lane: dict ingestion round-trip, decode-free recompute equality,
blockers, the feature-dependency map, provenance bpm range, and the
model-driven augment paths (bundled vocalness; trained genre; empty request).

Follows the test(name, fn) harness style of test_vocalness_model.py. All
records come from synthetic signals; the audio-fallback test writes a
temporary WAV with the stdlib wave module.
"""

import copy
import math
import os
import sys
import tempfile
import wave

import numpy as np

import sonara
from sonara import genre

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


print("=" * 70)
print("  Testing sonara augment lane")
print("=" * 70)

SR = 22050
DUR = 5.0
_t = np.arange(int(SR * DUR)) / SR
# A tonal signal with a beat-like amplitude pattern so key/tempo paths engage.
_y = (
    0.5 * np.sin(2 * np.pi * 220.0 * _t)
    + 0.3 * np.sin(2 * np.pi * 277.18 * _t)
    + 0.2 * np.sin(2 * np.pi * 329.63 * _t)
).astype(np.float32)
_y *= (0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 2.0 * _t))).astype(np.float32)

REC_PLAYLIST = sonara.analyze_signal(_y, sr=SR, mode="playlist")
REC_COMPACT = sonara.analyze_signal(_y, sr=SR, mode="compact")


# ------------------------------------------------------------
# Round trip: dict → struct → dict (augment with no request is a clone)
# ------------------------------------------------------------
def _roundtrip_playlist():
    aug = sonara.augment_analysis(REC_PLAYLIST, [])
    assert isinstance(aug, sonara.TrackAnalysis)
    assert aug == REC_PLAYLIST, "empty augment must round-trip the record exactly"
    assert aug is not REC_PLAYLIST


test("empty augment round-trips a playlist record exactly", _roundtrip_playlist)


def _roundtrip_optin():
    rec = sonara.analyze_signal(
        _y, sr=SR,
        features=["bpm", "beats", "chroma", "structure", "beatgrid", "silence",
                  "key_candidates", "chords", "fingerprint", "embedding"],
    )
    aug = sonara.augment_analysis(rec, [])
    assert aug == rec, "opt-in fields (events, fingerprint, embedding) must survive"


test("empty augment round-trips opt-in fields exactly", _roundtrip_optin)


# ------------------------------------------------------------
# Decode-free recompute equals a direct request over the same audio
# ------------------------------------------------------------
def _augment_equals_direct():
    aug = sonara.augment_analysis(REC_PLAYLIST, ["vocalness", "key_candidates", "embedding"])
    direct_v = sonara.analyze_signal(_y, sr=SR, features=["vocalness"])
    direct_k = sonara.analyze_signal(_y, sr=SR, features=["key_candidates"])
    direct_e = sonara.analyze_signal(_y, sr=SR, features=["embedding"])
    assert aug["vocalness"] == direct_v["vocalness"], (aug["vocalness"], direct_v["vocalness"])
    # vocalness/instrumentalness recompute together from one shared value.
    assert math.isclose(aug["instrumentalness"], 1.0 - aug["vocalness"], abs_tol=1e-6)
    assert aug["key_candidates"] == direct_k["key_candidates"]
    assert aug["embedding"] == direct_e["embedding"]
    assert aug["embedding_version"] == direct_e["embedding_version"]
    # Untouched fields never cleared.
    assert aug["mfcc_mean"] == REC_PLAYLIST["mfcc_mean"]
    assert aug["bpm"] == REC_PLAYLIST["bpm"]


test("decode-free augment equals direct request (vocalness/key_candidates/embedding)", _augment_equals_direct)


def _augment_mood_equals_direct():
    aug = sonara.augment_analysis(REC_PLAYLIST, ["mood"])
    direct = sonara.analyze_signal(_y, sr=SR, features=["mood", "dissonance"])
    for k in ("mood_happy", "mood_aggressive", "mood_relaxed", "mood_sad"):
        assert aug[k] == direct[k], (k, aug[k], direct[k])


test("decode-free mood augment equals direct co-request with dissonance", _augment_mood_equals_direct)


def _requested_features_union():
    rec = sonara.analyze_signal(_y, sr=SR, features=["bpm", "chroma"])
    aug = sonara.augment_analysis(rec, ["key"])
    req = aug["provenance"]["requested_features"]
    assert "key" in req and "bpm" in req and "chroma" in req, req
    assert req == sorted(req), "requested_features must stay sorted"
    assert "key" in aug and "key_camelot" in aug


test("requested_features becomes the sorted union after augment", _requested_features_union)


# ------------------------------------------------------------
# Blockers and errors
# ------------------------------------------------------------
def _unknown_feature():
    assert sonara.can_augment(REC_PLAYLIST, "nope") is False
    assert sonara.augment_blocker(REC_PLAYLIST, "nope") == "unknown feature"
    try:
        sonara.augment_analysis(REC_PLAYLIST, ["nope"])
    except ValueError as e:
        assert "unknown feature" in str(e), e
    else:
        raise AssertionError("unknown feature must be a hard error")


test("unknown feature: can_augment False, blocker string, hard error", _unknown_feature)


def _blocker_strings():
    assert sonara.can_augment(REC_PLAYLIST, "energy") is True
    assert sonara.augment_blocker(REC_PLAYLIST, "energy") is None
    # Case-insensitive lookup.
    assert sonara.can_augment(REC_PLAYLIST, "EnErGy") is True
    assert sonara.augment_blocker(REC_PLAYLIST, "zcr") == "needs audio (audio-class feature)"
    assert sonara.augment_blocker(REC_PLAYLIST, "bpm") == "needs audio (frame_curves-class feature)"


test("blocker strings: None when recomputable, needs-audio classes named", _blocker_strings)


def _missing_evidence():
    # A compact record has no spectral_bandwidth_mean → energy is blocked.
    assert sonara.can_augment(REC_COMPACT, "energy") is False
    blocker = sonara.augment_blocker(REC_COMPACT, "energy")
    assert blocker.startswith("missing evidence:"), blocker
    assert "spectral_bandwidth_mean" in blocker, blocker
    try:
        sonara.augment_analysis(REC_COMPACT, ["energy"])
    except ValueError as e:
        assert "audio" in str(e), e
    else:
        raise AssertionError("missing evidence without audio_path must raise")


test("missing evidence: blocker names the fields, no-audio augment raises", _missing_evidence)


def _schema_mismatch():
    stale = copy.deepcopy(dict(REC_PLAYLIST))
    stale["provenance"]["schema_version"] = 999
    blocker = sonara.augment_blocker(stale, "energy")
    assert blocker == f"schema version mismatch (record 999, current {REC_PLAYLIST['provenance']['schema_version']})", blocker
    assert sonara.can_augment(stale, "energy") is False
    try:
        sonara.augment_analysis(stale, ["energy"])
    except ValueError as e:
        assert "schema_version" in str(e), e
    else:
        raise AssertionError("schema mismatch must be a hard error")


test("schema version mismatch: stable blocker string + hard error", _schema_mismatch)


def _ingestion_errors():
    try:
        sonara.augment_analysis({}, [])
    except ValueError as e:
        assert "provenance" in str(e), e
    else:
        raise AssertionError("a record without provenance must be rejected")
    broken = copy.deepcopy(dict(REC_PLAYLIST))
    broken["chroma_mean"] = "not a list"
    try:
        sonara.augment_analysis(broken, [])
    except ValueError as e:
        assert "chroma_mean" in str(e), e
    else:
        raise AssertionError("a mistyped field must be rejected naming the field")


test("ingestion fails fast naming the missing/mistyped field", _ingestion_errors)


def _json_lists_accepted():
    # JSON round-trips turn tuples into lists; ingestion must accept them.
    import json
    rec = sonara.analyze_signal(_y, sr=SR, features=["key_candidates", "bpm", "chroma"])
    revived = json.loads(json.dumps(dict(rec)))
    aug = sonara.augment_analysis(revived, ["key"])
    direct = sonara.analyze_signal(_y, sr=SR, features=["key"])
    assert aug["key"] == direct["key"], (aug["key"], direct["key"])


test("a JSON-revived record (tuples as lists) augments cleanly", _json_lists_accepted)


# ------------------------------------------------------------
# feature_dependencies map
# ------------------------------------------------------------
def _dependency_map_shape():
    deps = sonara.feature_dependencies()
    assert isinstance(deps, list) and len(deps) > 20
    by_name = {}
    for row in deps:
        assert set(row) == {"name", "class", "required_evidence",
                            "needs_extended", "opt_in_only", "full_only"}, row
        assert row["class"] in {"audio", "frame_curves", "scalars", "embedding"}, row
        assert isinstance(row["required_evidence"], list)
        for flag in ("needs_extended", "opt_in_only", "full_only"):
            assert isinstance(row[flag], bool)
        by_name[row["name"]] = row
    key = by_name["key"]
    assert key["class"] == "scalars" and key["required_evidence"] == ["chroma_mean"]
    assert key["needs_extended"] is True and key["opt_in_only"] is False
    assert by_name["zcr"]["class"] == "audio"
    assert by_name["bpm"]["class"] == "frame_curves"
    assert by_name["embedding"]["class"] == "embedding"
    assert by_name["beatgrid"]["opt_in_only"] is True
    assert by_name["tempo_curve"]["full_only"] is True
    # The map agrees with per-record augmentability on a fully-evidenced record.
    for row in deps:
        if row["class"] in ("scalars", "embedding"):
            assert sonara.can_augment(REC_PLAYLIST, row["name"]) is True, row["name"]
        else:
            assert sonara.can_augment(REC_PLAYLIST, row["name"]) is False, row["name"]


test("feature_dependencies rows carry class, evidence and routing flags", _dependency_map_shape)


# ------------------------------------------------------------
# Provenance bpm range
# ------------------------------------------------------------
def _bpm_range_visible():
    rec = sonara.analyze_signal(_y, sr=SR, bpm_min=70.0, bpm_max=140.0)
    assert rec["provenance"]["bpm_min"] == 70.0
    assert rec["provenance"]["bpm_max"] == 140.0
    assert "bpm_min" not in REC_PLAYLIST["provenance"]
    assert "bpm_max" not in REC_PLAYLIST["provenance"]
    # The recorded range survives the dict → struct → dict round trip.
    aug = sonara.augment_analysis(rec, [])
    assert aug["provenance"]["bpm_min"] == 70.0
    assert aug["provenance"]["bpm_max"] == 140.0


test("provenance carries the configured bpm range (absent when unset)", _bpm_range_visible)


# ------------------------------------------------------------
# Model-driven augment paths
# ------------------------------------------------------------
def _vocalness_model_augment():
    aug = sonara.augment_analysis(REC_PLAYLIST, ["vocalness"], vocalness_model="bundled")
    direct = sonara.analyze_signal(
        _y, sr=SR, features=["vocalness", "embedding"], vocalness_model="bundled"
    )
    assert aug["vocalness"] == direct["vocalness"], (aug["vocalness"], direct["vocalness"])
    assert math.isclose(aug["instrumentalness"], 1.0 - aug["vocalness"], abs_tol=1e-6)
    assert aug["provenance"]["vocalness_model_id"] == direct["provenance"]["vocalness_model_id"]


test("bundled vocalness model augment equals the direct model run", _vocalness_model_augment)


def _genre_model_empty_request():
    # Genre has no feature name: passing the model IS the request.
    rng = np.random.default_rng(0)
    dim = sonara.EMBEDDING_DIM
    shift = np.zeros(dim)
    shift[:6] = 1.5
    X = np.vstack([
        rng.normal(0.0, 0.25, size=(100, dim)),
        rng.normal(0.0, 0.25, size=(100, dim)) + shift,
    ])
    labels = ["a"] * 100 + ["b"] * 100
    model = genre.train(X, labels, hidden=0, epochs=300, lr=0.5, seed=0)
    path = os.path.join(tempfile.mkdtemp(prefix="sonara_augment_"), "genre.json")
    model.save(path)
    aug = sonara.augment_analysis(REC_PLAYLIST, [], genre_model=path)
    assert aug["genre"] in ("a", "b")
    assert 0.0 < aug["genre_confidence"] <= 1.0
    # An empty feature request changes no recorded request.
    assert "requested_features" not in aug["provenance"]
    assert "genre" not in REC_PLAYLIST


test("genre model + empty feature list is a valid request", _genre_model_empty_request)


# ------------------------------------------------------------
# Audio fallback
# ------------------------------------------------------------
def _audio_fallback():
    wav_path = os.path.join(tempfile.mkdtemp(prefix="sonara_augment_wav_"), "tone.wav")
    pcm = np.clip(_y, -1.0, 1.0)
    frames = (pcm * 32767.0).astype("<i2")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(frames.tobytes())
    rec = sonara.analyze_file(wav_path, sr=SR, mode="compact")
    aug = sonara.augment_analysis(rec, ["chroma"], audio_path=wav_path)
    direct = sonara.analyze_file(wav_path, sr=SR, features=["chroma"])
    assert aug["chroma_mean"] == direct["chroma_mean"]
    # Untouched compact fields keep their record values.
    assert aug["bpm"] == rec["bpm"]


test("audio fallback recomputes a frame-curves feature from the file", _audio_fallback)


print()
print(f"passed: {passed}  failed: {failed}")
if failed:
    for name, err in errors:
        print(f"  FAILED {name}: {err}")
    sys.exit(1)
