#!/usr/bin/env python3
"""Bring-your-own genre model: trainer, round-trip, and analyze-time parity.

Follows the test(name, fn) harness style of test_api.py. Trains a tiny numpy
classifier on synthetic separable embeddings, checks save/load round-trip,
verifies numpy predict matches the Rust analyze-time label, and asserts genre is
absent without a model and that a stale embedding_version fails fast.
"""

import os
import sys
import tempfile

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
print("  Testing sonara bring-your-own genre model")
print("=" * 70)

DIM = sonara.EMBEDDING_DIM  # 48

# ------------------------------------------------------------
# Synthetic separable data: two Gaussian blobs in 48-dim.
# ------------------------------------------------------------
rng = np.random.default_rng(0)
n_per = 100
# Blob A centered near the origin; blob B shifted along the first few dims.
shift = np.zeros(DIM)
shift[:6] = 1.5
Xa = rng.normal(0.0, 0.25, size=(n_per, DIM))
Xb = rng.normal(0.0, 0.25, size=(n_per, DIM)) + shift
X = np.vstack([Xa, Xb]).astype(np.float64)
y = ["a"] * n_per + ["b"] * n_per


def _train_accuracy():
    model = genre.train(X, y, hidden=0, epochs=400, lr=0.5, seed=0)
    preds = [model.predict(row)[0] for row in X]
    acc = np.mean([p == t for p, t in zip(preds, y)])
    assert acc > 0.9, f"train accuracy {acc:.3f} should exceed 0.9"


test("train reaches >0.9 accuracy on separable blobs", _train_accuracy)


def _train_hidden_layer():
    # A one-hidden-ReLU-layer model must also fit and produce a 2-layer JSON.
    model = genre.train(X, y, hidden=8, epochs=400, lr=0.3, seed=1)
    d = model.to_dict()
    assert len(d["layers"]) == 2, d
    assert d["layers"][0]["activation"] == "relu"
    assert d["layers"][1]["activation"] == "softmax"
    preds = [model.predict(row)[0] for row in X]
    acc = np.mean([p == t for p, t in zip(preds, y)])
    assert acc > 0.9, f"hidden-layer train accuracy {acc:.3f}"


test("train with hidden ReLU layer fits + emits 2 layers", _train_hidden_layer)


def _save_load_roundtrip():
    model = genre.train(X, y, hidden=0, epochs=300, lr=0.5, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        model.save(path)
        loaded = genre.load(path)
        assert loaded.labels == model.labels
        assert loaded.embedding_version == model.embedding_version
        # Predictions identical after round-trip.
        for row in X[::10]:
            l1, c1 = model.predict(row)
            l2, c2 = loaded.predict(row)
            assert l1 == l2, f"{l1} != {l2}"
            assert abs(c1 - c2) < 1e-9, f"confidence drift {c1} vs {c2}"


test("save/load JSON round-trip preserves predictions", _save_load_roundtrip)


def _deterministic_given_seed():
    m1 = genre.train(X, y, hidden=0, epochs=200, lr=0.5, seed=7)
    m2 = genre.train(X, y, hidden=0, epochs=200, lr=0.5, seed=7)
    assert np.allclose(m1.layers[0]["W"], m2.layers[0]["W"]), "same seed → same weights"


test("training is deterministic given seed", _deterministic_given_seed)


def _embedding_version_default():
    model = genre.train(X, y, seed=0)
    assert model.embedding_version == sonara.SIMILARITY_VERSION


test("model defaults embedding_version to SIMILARITY_VERSION", _embedding_version_default)


# ------------------------------------------------------------
# Analyze-time parity: numpy predict == Rust analyze-time label.
# ------------------------------------------------------------
def _real_signal(sr=22050):
    # A tonal, structured signal so the embedding is well-defined.
    t = np.arange(4 * sr) / sr
    y_sig = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2 * np.pi * 330.0 * t)
    )
    return y_sig.astype(np.float32)


def _predict_parity():
    model = genre.train(X, y, hidden=0, epochs=400, lr=0.5, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        model.save(path)

        sig = _real_signal()
        # Get the actual embedding this signal produces.
        r_emb = sonara.analyze_signal(sig, sr=22050, features=["embedding"])
        emb = r_emb["embedding"]
        assert len(emb) == DIM

        # numpy-side prediction on that embedding.
        py_label, py_conf = model.predict(emb)

        # Rust analyze-time prediction via genre_model=path.
        r = sonara.analyze_signal(sig, sr=22050, genre_model=path)
        assert "genre" in r, "genre must be populated when a model is set"
        assert "genre_confidence" in r
        assert r["genre"] == py_label, f"rust {r['genre']!r} != numpy {py_label!r}"
        assert 0.5 < r["genre_confidence"] <= 1.0, r["genre_confidence"]
        # Confidence agrees to f32 tolerance (Rust is f32, numpy f64).
        assert abs(r["genre_confidence"] - py_conf) < 1e-3, (r["genre_confidence"], py_conf)


test("numpy predict == analyze-time (Rust) label + confidence", _predict_parity)


def _genre_and_embedding_both_when_requested():
    model = genre.train(X, y, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        model.save(path)
        sig = _real_signal()
        r = sonara.analyze_signal(sig, sr=22050, features=["embedding"], genre_model=path)
        assert "genre" in r and "genre_confidence" in r
        # With features=["embedding"], the embedding IS emitted alongside genre.
        assert "embedding" in r and "embedding_version" in r


test("genre + embedding both present when embedding requested", _genre_and_embedding_both_when_requested)


def _genre_absent_without_model():
    sig = _real_signal()
    r = sonara.analyze_signal(sig, sr=22050, mode="playlist")
    assert "genre" not in r, "genre must be absent without a model"
    assert "genre_confidence" not in r


test("genre absent when no model given", _genre_absent_without_model)


def _no_embedding_leak_without_feature():
    model = genre.train(X, y, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        model.save(path)
        sig = _real_signal()
        r = sonara.analyze_signal(sig, sr=22050, genre_model=path)
        assert "genre" in r
        # Embedding must NOT leak when not explicitly requested.
        assert "embedding" not in r, "embedding must not be emitted without feature"
        assert "embedding_version" not in r


test("embedding not leaked when only genre_model given", _no_embedding_leak_without_feature)


def _stale_embedding_version_raises():
    # Force a mismatched embedding_version → analyze must raise fast.
    model = genre.train(X, y, seed=0, embedding_version=999)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stale.json")
        model.save(path)
        sig = _real_signal()
        raised = False
        try:
            sonara.analyze_signal(sig, sr=22050, genre_model=path)
        except ValueError as e:
            raised = True
            assert "999" in str(e), f"error should name the model version: {e}"
        assert raised, "stale embedding_version must raise ValueError"


test("stale embedding_version fails fast (ValueError)", _stale_embedding_version_raises)


def _malformed_model_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.json")
        with open(path, "w") as f:
            f.write('{"format_version": 1, "labels": ["a"')  # truncated
        sig = _real_signal()
        raised = False
        try:
            sonara.analyze_signal(sig, sr=22050, genre_model=path)
        except ValueError:
            raised = True
        assert raised, "malformed model JSON must raise ValueError"


test("malformed model JSON raises ValueError", _malformed_model_raises)


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
