#!/usr/bin/env python3
"""Bring-your-own vocalness model: trainer, round-trip, analyze-time override,
provenance identity, and fail-fast validation.

Follows the test(name, fn) harness style of test_genre.py: trains a tiny numpy
classifier on synthetic separable embeddings, checks save/load round-trip and
numpy/Rust parity, verifies analyze-time override of vocalness/instrumentalness
plus provenance.vocalness_model_id, and asserts a missing id or stale
embedding_version fails fast.
"""

import json
import math
import os
import sys
import tempfile

import numpy as np

import sonara
from sonara import vocal_model

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
print("  Testing sonara bring-your-own vocalness model")
print("=" * 70)

DIM = sonara.EMBEDDING_DIM  # 48

# Synthetic separable data: instrumental blob at origin, vocal blob shifted.
rng = np.random.default_rng(0)
n_per = 120
shift = np.zeros(DIM)
shift[:8] = 1.2
Xi = rng.normal(0.0, 0.3, size=(n_per, DIM))
Xv = rng.normal(0.0, 0.3, size=(n_per, DIM)) + shift
X = np.vstack([Xi, Xv]).astype(np.float64)
y = [False] * n_per + [True] * n_per

tmp = tempfile.mkdtemp(prefix="sonara_vocal_model_")
model_path = os.path.join(tmp, "vocal.json")


def _train_separates():
    model = vocal_model.train(X, y, model_id="synthetic-v1", hidden=8, epochs=800)
    pi = [model.predict_vocalness(r) for r in Xi]
    pv = [model.predict_vocalness(r) for r in Xv]
    assert sum(p < 0.5 for p in pi) > 0.9 * n_per, "instrumental blob not separated"
    assert sum(p > 0.5 for p in pv) > 0.9 * n_per, "vocal blob not separated"
    model.save(model_path)
    assert os.path.exists(model_path)


test("train separates synthetic blobs + save", _train_separates)


def _roundtrip_parity():
    loaded = vocal_model.load(model_path)
    assert loaded.id == "synthetic-v1"
    orig = vocal_model.train(X, y, model_id="synthetic-v1", hidden=8, epochs=800)
    for row in X[:20]:
        a, b = orig.predict_vocalness(row), loaded.predict_vocalness(row)
        assert math.isclose(a, b, abs_tol=1e-6), f"roundtrip drift: {a} vs {b}"


test("save/load round-trip parity", _roundtrip_parity)


def _requires_model_id():
    try:
        vocal_model.train(X, y, model_id="  ")
        raise AssertionError("blank model_id must be rejected")
    except ValueError:
        pass
    try:
        vocal_model.train(X, y)  # type: ignore[call-arg]
        raise AssertionError("missing model_id must be rejected")
    except TypeError:
        pass


test("model_id is required", _requires_model_id)


# A signal to analyze (2s of a tone + noise bursts is fine — any signal works).
sig = (0.4 * np.sin(2 * np.pi * 440.0 * np.arange(44100) / 22050)).astype(np.float32)


def _analyze_override_and_provenance():
    r = sonara.analyze_signal(sig, sr=22050, vocalness_model=model_path)
    assert "vocalness" in r, "model must populate vocalness without features list"
    assert "instrumentalness" in r
    v, i = r["vocalness"], r["instrumentalness"]
    assert 0.0 <= v <= 1.0 and 0.0 <= i <= 1.0
    assert math.isclose(v + i, 1.0, abs_tol=1e-5), "instrumentalness must be 1 - v"
    prov = r["provenance"]
    assert prov.get("vocalness_model_id") == "synthetic-v1", prov
    # Rust-side score must match the numpy-side model on the same embedding.
    r_emb = sonara.analyze_signal(sig, sr=22050, features=["embedding"])
    loaded = vocal_model.load(model_path)
    expected = loaded.predict_vocalness(r_emb["embedding"])
    assert math.isclose(v, expected, rel_tol=1e-4, abs_tol=1e-5), (
        f"Rust {v} vs numpy {expected}")
    # The embedding itself must not leak from the model-only run.
    assert "embedding" not in r


test("analyze override + provenance id + Rust/numpy parity",
     _analyze_override_and_provenance)


def _no_model_no_id():
    r = sonara.analyze_signal(sig, sr=22050, features=["vocalness"])
    assert "vocalness" in r
    assert "vocalness_model_id" not in r["provenance"]


test("no model → heuristic + no provenance id", _no_model_no_id)


def _batch_carries_model():
    wav = os.path.join(tmp, "tone.wav")
    import struct
    import wave
    with wave.open(wav, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        frames = b"".join(
            struct.pack("<h", int(32767 * 0.4 * math.sin(2 * math.pi * 440 * i / 22050)))
            for i in range(44100)
        )
        w.writeframes(frames)
    res = sonara.analyze_batch([wav], vocalness_model=model_path)
    assert len(res) == 1 and not res[0].failed
    assert res[0]["provenance"].get("vocalness_model_id") == "synthetic-v1"
    assert "vocalness" in res[0]


test("analyze_batch carries the model", _batch_carries_model)


def _stale_embedding_version_fails_fast():
    stale = os.path.join(tmp, "stale.json")
    d = vocal_model.load(model_path).to_dict()
    d["embedding_version"] = d["embedding_version"] + 1
    with open(stale, "w") as f:
        json.dump(d, f)
    try:
        sonara.analyze_signal(sig, sr=22050, vocalness_model=stale)
        raise AssertionError("stale embedding_version must fail")
    except ValueError as e:
        assert "embedding_version" in str(e), e


test("stale embedding_version fails fast", _stale_embedding_version_fails_fast)


def _bad_model_json_rejected():
    bad = os.path.join(tmp, "bad.json")
    d = vocal_model.load(model_path).to_dict()
    del d["id"]
    with open(bad, "w") as f:
        json.dump(d, f)
    try:
        sonara.analyze_signal(sig, sr=22050, vocalness_model=bad)
        raise AssertionError("model without id must be rejected")
    except ValueError:
        pass
    with open(bad, "w") as f:
        f.write("{not json")
    try:
        sonara.analyze_signal(sig, sr=22050, vocalness_model=bad)
        raise AssertionError("malformed JSON must be rejected")
    except ValueError:
        pass


test("id-less / malformed model rejected", _bad_model_json_rejected)


def _non_finite_model_state_rejected_everywhere():
    zeros = ",".join(["0"] * DIM)
    overflow_json = (
        '{"format_version":1,'
        f'"embedding_version":{sonara.SIMILARITY_VERSION},'
        '"id":"overflow","labels":["instrumental","vocal"],"layers":['
        f'{{"weights":[[{zeros}],[{zeros}]],'
        '"bias":[1e999,1e999],"activation":"softmax"}]}'
    )
    path = os.path.join(tmp, "overflow.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(overflow_json)
    try:
        vocal_model.load(path)
        raise AssertionError("numpy loader must reject overflow")
    except ValueError:
        pass
    try:
        sonara.analyze_signal(sig, sr=22050, vocalness_model=path)
        raise AssertionError("Rust loader must reject overflow")
    except ValueError:
        pass

    model = vocal_model.train(X, y, model_id="mutable", epochs=1)
    model.layers[-1]["b"][0] = np.inf
    try:
        model.predict_vocalness(np.zeros(DIM))
        raise AssertionError("mutated non-finite model must not predict")
    except ValueError:
        pass
    try:
        model.save(os.path.join(tmp, "nonfinite.json"))
        raise AssertionError("non-finite model must not serialize")
    except (ValueError, OverflowError):
        pass

    bad_x = X.copy()
    bad_x[0, 0] = -np.inf
    try:
        vocal_model.train(bad_x, y, model_id="bad-training")
        raise AssertionError("training data must be finite")
    except ValueError:
        pass


test("non-finite model state rejected by numpy and Rust",
     _non_finite_model_state_rejected_everywhere)


def _bundled_model_stable():
    path = vocal_model.bundled_path()
    assert os.path.exists(path), path
    m = vocal_model.load(path)
    assert m.id == "sonara-vocalness-v1"
    r = sonara.analyze_signal(sig, sr=22050, vocalness_model="bundled")
    assert r["provenance"].get("vocalness_model_id") == "sonara-vocalness-v1"
    # Rust-crate embedded copy (sonara/models/) must match the package data
    # copy when the repo layout is present.
    crate_copy = os.path.join(os.path.dirname(__file__), "..", "..",
                              "sonara", "models", "vocalness_v1.json")
    if os.path.exists(crate_copy):
        with open(crate_copy, "rb") as a, open(path, "rb") as b:
            assert a.read() == b.read(), "crate and package model copies diverged"


test("bundled model stable id + crate/package parity", _bundled_model_stable)

print(f"\n{'='*70}")
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
print(f"{'='*70}")
for name, err in errors:
    print(f"  FAILED: {name}: {err}")
sys.exit(1 if failed else 0)
