"""Bring-your-own vocalness model: a pure-numpy trainer + loader for a tiny
vocal-presence classifier over sonara's similarity embedding.

The built-in ``vocalness`` heuristic cannot separate clean solo voice from
clean solo pitched instruments. This module trains the precision upgrade and
hands it back to ``analyze_*`` via ``vocalness_model=<path>``:

    import sonara
    from sonara import vocal_model

    # 1. Embeddings for a labeled set (True = vocal, False = instrumental).
    lib = sonara.analyze_batch(files, features=["embedding"])
    X = [r["embedding"] for r in lib]      # (n, 48)
    y = [True, False, ...]                 # vocal presence per track

    # 2. Train (numpy only) and save. `model_id` is REQUIRED — it is stamped
    #    into every result's provenance as `vocalness_model_id` so downstream
    #    caches can invalidate scores produced by a different model.
    model = vocal_model.train(X, y, model_id="my-vocal-v1", hidden=16)
    model.save("vocal_model.json")

    # 3. Score new tracks: vocalness/instrumentalness now come from the model.
    r = sonara.analyze_file("track.mp3", vocalness_model="vocal_model.json")
    r["vocalness"], r["provenance"]["vocalness_model_id"]

The JSON is the genre-model format (last layer softmax over exactly two
labels, one of them ``"vocal"``) plus the required ``id``. Training
standardizes features internally and folds the standardization into the first
layer's weights, so the saved model operates on raw embeddings — identical
inference in Rust and numpy.
"""

from __future__ import annotations

import json

import numpy as np

from sonara._sonara import SIMILARITY_VERSION as _SIMILARITY_VERSION
from sonara._sonara import EMBEDDING_DIM as _EMBEDDING_DIM

FORMAT_VERSION = 1
LABELS = ["instrumental", "vocal"]  # fixed; index 1 is P(vocal)

__all__ = ["VocalnessModel", "train", "load", "bundled_path", "FORMAT_VERSION",
           "LABELS"]


def bundled_path() -> str:
    """Path to the vocalness model bundled with the package.

    Trained on ~900 album-level labeled tracks; on a held-out 205-track
    curated real-music set: AUC 0.94 vs 0.63 for the built-in heuristic,
    clear-vocal false negatives 5% vs 53%. Pass it (or the shorthand
    ``vocalness_model="bundled"``) to ``analyze_*``. Known limitations:
    solo melodic instrument leads (sax/guitar) can score vocal-high, and
    spoken narration scores low.
    """
    import os

    return os.path.join(os.path.dirname(__file__), "models", "vocalness_v1.json")


def _softmax_rows(m: np.ndarray) -> np.ndarray:
    z = m - m.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class VocalnessModel:
    """A trained (or loaded) vocal-presence classifier over the embedding."""

    def __init__(self, model_id, layers, embedding_version):
        self.id = str(model_id)
        if not self.id.strip():
            raise ValueError("model_id must be a non-empty string")
        self.layers = layers  # [{"W": (in, out), "b": (out,), "activation": str}]
        self.embedding_version = int(embedding_version)

    def predict_vocalness(self, x) -> float:
        """P(vocal) for one embedding vector, in ``[0, 1]`` (numpy parity
        with the Rust ``VocalnessModel::predict_vocalness``)."""
        v = np.asarray(x, dtype=np.float64).ravel()
        v = np.where(np.isfinite(v), v, 0.0)
        for layer in self.layers:
            v = v @ layer["W"] + layer["b"]
            act = layer["activation"]
            if act == "relu":
                v = np.maximum(v, 0.0)
            elif act == "softmax":
                z = v - np.max(v)
                e = np.exp(z)
                v = e / e.sum()
        return float(np.clip(v[1], 0.0, 1.0))

    def to_dict(self):
        return {
            "format_version": FORMAT_VERSION,
            "embedding_version": self.embedding_version,
            "id": self.id,
            "labels": list(LABELS),
            "layers": [
                {
                    # JSON is row-major out x in — transpose the internal (in, out).
                    "weights": np.asarray(layer["W"]).T.tolist(),
                    "bias": np.asarray(layer["b"]).ravel().tolist(),
                    "activation": layer["activation"],
                }
                for layer in self.layers
            ],
        }

    def save(self, path):
        """Write the model as JSON to ``path`` (loadable by ``analyze_*``)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        return path


def load(path) -> VocalnessModel:
    """Load a vocalness model JSON written by :meth:`VocalnessModel.save`."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    labels = [str(x).lower() for x in d["labels"]]
    if labels != LABELS:
        raise ValueError(f"labels must be {LABELS}; got {d['labels']}")
    layers = []
    for layer in d["layers"]:
        w = np.asarray(layer["weights"], dtype=np.float64)
        layers.append(
            {
                "W": w.T.copy(),
                "b": np.asarray(layer["bias"], dtype=np.float64),
                "activation": str(layer["activation"]),
            }
        )
    return VocalnessModel(d["id"], layers, d["embedding_version"])


def train(X, y, *, model_id, hidden=16, epochs=2000, lr=0.05, seed=0, l2=1e-3,
          embedding_version=None) -> VocalnessModel:
    """Train a vocal-presence classifier with plain numpy (no sklearn/torch).

    Parameters
    ----------
    X : array-like, shape ``(n, EMBEDDING_DIM)``
        One similarity embedding per training example.
    y : sequence of bool (or 0/1), length ``n``
        ``True``/1 = vocal, ``False``/0 = instrumental.
    model_id : str (required)
        Model identity stamped into analysis provenance
        (``vocalness_model_id``). Version it like an artifact
        (e.g. ``"myset-vocal-v3"``).
    hidden : int, default 16
        ``0`` → logistic regression; ``> 0`` → one hidden ReLU layer.
    epochs, lr, l2, seed : training hyperparameters (deterministic given seed).
    embedding_version : int, optional
        Recorded in the model; defaults to ``sonara.SIMILARITY_VERSION``.

    Features are standardized (zero mean, unit variance over the training
    set) internally; the standardization is folded into the first layer's
    weights so the saved model operates on raw embeddings.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != _EMBEDDING_DIM:
        raise ValueError(f"X must have shape (n, {_EMBEDDING_DIM}); got {X.shape}")
    X = np.where(np.isfinite(X), X, 0.0)
    yi = np.array([1 if bool(v) else 0 for v in y], dtype=np.int64)
    if len(yi) != X.shape[0]:
        raise ValueError(f"len(y) ({len(yi)}) must equal X.shape[0] ({X.shape[0]})")
    if len(set(yi.tolist())) < 2:
        raise ValueError("need both vocal and instrumental examples")

    # Standardize (folded into the first layer below).
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    n, d = X.shape
    k = 2
    Y = np.zeros((n, k), dtype=np.float64)
    Y[np.arange(n), yi] = 1.0
    # Class weighting: balance vocal/instrumental contributions.
    counts = np.bincount(yi, minlength=2).astype(np.float64)
    w_sample = (n / (2.0 * counts))[yi][:, None]

    rng = np.random.default_rng(seed)
    ev = _SIMILARITY_VERSION if embedding_version is None else int(embedding_version)

    if hidden and hidden > 0:
        h = int(hidden)
        W1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, h))
        b1 = np.zeros(h)
        W2 = rng.normal(0.0, np.sqrt(2.0 / h), size=(h, k))
        b2 = np.zeros(k)
        for _ in range(int(epochs)):
            Z1 = Xs @ W1 + b1
            A1 = np.maximum(Z1, 0.0)
            P = _softmax_rows(A1 @ W2 + b2)
            dZ2 = (P - Y) * w_sample / n
            dW2 = A1.T @ dZ2 + l2 * W2
            db2 = dZ2.sum(axis=0)
            dZ1 = (dZ2 @ W2.T) * (Z1 > 0.0)
            dW1 = Xs.T @ dZ1 + l2 * W1
            db1 = dZ1.sum(axis=0)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        # Fold standardization into layer 1: x_s = (x - mu) / sigma.
        W1_raw = W1 / sigma[:, None]
        b1_raw = b1 - (mu / sigma) @ W1
        layers = [
            {"W": W1_raw, "b": b1_raw, "activation": "relu"},
            {"W": W2, "b": b2, "activation": "softmax"},
        ]
    else:
        W = rng.normal(0.0, 0.1, size=(d, k))
        b = np.zeros(k)
        for _ in range(int(epochs)):
            P = _softmax_rows(Xs @ W + b)
            dZ = (P - Y) * w_sample / n
            dW = Xs.T @ dZ + l2 * W
            db = dZ.sum(axis=0)
            W -= lr * dW
            b -= lr * db
        W_raw = W / sigma[:, None]
        b_raw = b - (mu / sigma) @ W
        layers = [{"W": W_raw, "b": b_raw, "activation": "softmax"}]

    return VocalnessModel(model_id, layers, ev)
