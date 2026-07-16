"""Bring-your-own genre model: a pure-numpy trainer + loader for a tiny
classifier over sonara's similarity embedding.

sonara ships **no** genre model. This module lets you train one from your own
labeled library and hand it back to ``analyze_*`` via ``genre_model=<path>``:

    import sonara
    from sonara import genre

    # 1. Scan your library for embeddings (48-dim vectors) + your labels.
    lib = sonara.analyze_batch(files, features=["embedding"])
    X = [r["embedding"] for r in lib]      # (n, 48)
    y = ["rock", "electronic", ...]        # one label per track

    # 2. Train (numpy only — no sklearn/torch) and save.
    model = genre.train(X, y, hidden=0, epochs=300, lr=0.1, seed=0)
    model.save("genre_model.json")

    # 3. Classify new tracks.
    r = sonara.analyze_file("track.mp3", genre_model="genre_model.json")
    r["genre"], r["genre_confidence"]

The saved JSON is the exact format the Rust core loads (see ``sonara.genre`` in
the Rust docs): a small feed-forward net over the embedding, last layer softmax.
``model.predict(x)`` runs the same inference numpy-side for parity with Rust.

The model records the ``embedding_version`` it was trained against (defaults to
``sonara.SIMILARITY_VERSION``); analysis refuses to classify if that does not
match the running build's embedding layout.
"""

from __future__ import annotations

import json

import numpy as np

# Pulled from the compiled module (fully loaded before this file is imported by
# sonara/__init__.py — no circular import). These are the single source of truth
# for the embedding dimensionality and version.
from sonara._sonara import SIMILARITY_VERSION as _SIMILARITY_VERSION
from sonara._sonara import EMBEDDING_DIM as _EMBEDDING_DIM

FORMAT_VERSION = 1

__all__ = ["GenreModel", "train", "load", "FORMAT_VERSION"]


def _softmax(v: np.ndarray) -> np.ndarray:
    """Numerically stable softmax of a 1-D vector."""
    z = v - np.max(v)
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else e


def _softmax_rows(m: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax of a 2-D array."""
    z = m - m.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class GenreModel:
    """A trained (or loaded) genre classifier over the similarity embedding.

    Internally holds each dense layer as ``{"W": (in, out), "b": (out,),
    "activation": str}``. The JSON on disk stores ``W`` transposed to row-major
    ``out x in`` (matching the Rust loader).
    """

    def __init__(self, labels, layers, embedding_version):
        self.labels = [str(x) for x in labels]
        self.layers = layers
        self.embedding_version = int(embedding_version)

    # -- inference (numpy parity with the Rust `predict`) --
    def predict(self, x):
        """Classify one embedding vector → ``(label, confidence)``.

        ``confidence`` is the winning softmax probability in ``(0, 1]``. Mirrors
        the Rust ``GenreModel::predict`` (same layer chain + stable softmax), so
        the predicted label matches ``analyze_*(genre_model=...)`` on the same
        embedding.
        """
        v = np.asarray(x, dtype=np.float64).ravel()
        for layer in self.layers:
            v = v @ layer["W"] + layer["b"]
            act = layer["activation"]
            if act == "relu":
                v = np.maximum(v, 0.0)
            elif act == "softmax":
                v = _softmax(v)
            # "identity": pass through
        idx = int(np.argmax(v))
        return self.labels[idx], float(v[idx])

    # -- serialization --
    def to_dict(self):
        return {
            "format_version": FORMAT_VERSION,
            "embedding_version": self.embedding_version,
            "labels": list(self.labels),
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
        """Write the model as JSON to ``path`` (the format ``analyze_*`` loads)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        return path


def load(path) -> GenreModel:
    """Load a genre model JSON written by :meth:`GenreModel.save` (or hand-authored)."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    layers = []
    for layer in d["layers"]:
        # JSON weights are row-major out x in → store internally as (in, out).
        w = np.asarray(layer["weights"], dtype=np.float64)
        layers.append(
            {
                "W": w.T.copy(),
                "b": np.asarray(layer["bias"], dtype=np.float64),
                "activation": str(layer["activation"]),
            }
        )
    return GenreModel(d["labels"], layers, d["embedding_version"])


def train(X, y, *, hidden=0, epochs=300, lr=0.1, seed=0, l2=0.0, embedding_version=None) -> GenreModel:
    """Train a genre classifier over embeddings with plain numpy (no sklearn/torch).

    Parameters
    ----------
    X : array-like, shape ``(n, EMBEDDING_DIM)``
        One similarity embedding per training example (from
        ``analyze_*(features=["embedding"])``).
    y : sequence of str, length ``n``
        Genre label per example. The sorted unique labels become the model's
        output classes.
    hidden : int, default 0
        ``0`` → softmax (logistic) regression; ``> 0`` → one hidden ReLU layer of
        that width, then softmax.
    epochs, lr, l2 : training hyperparameters
        Full-batch gradient descent for ``epochs`` steps at learning rate ``lr``,
        with optional L2 weight decay ``l2``.
    seed : int, default 0
        Seeds weight initialization — training is deterministic given the inputs.
    embedding_version : int, optional
        Recorded in the model; defaults to ``sonara.SIMILARITY_VERSION``.

    Returns
    -------
    GenreModel
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != _EMBEDDING_DIM:
        raise ValueError(
            f"X must have shape (n, {_EMBEDDING_DIM}); got {X.shape}"
        )
    y = [str(v) for v in y]
    if len(y) != X.shape[0]:
        raise ValueError(f"len(y) ({len(y)}) must equal X.shape[0] ({X.shape[0]})")

    labels = sorted(set(y))
    if len(labels) < 2:
        raise ValueError("need at least 2 distinct labels to train a classifier")
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    yi = np.array([label_to_idx[v] for v in y], dtype=np.int64)

    n, d = X.shape
    k = len(labels)
    Y = np.zeros((n, k), dtype=np.float64)
    Y[np.arange(n), yi] = 1.0

    rng = np.random.default_rng(seed)
    ev = _SIMILARITY_VERSION if embedding_version is None else int(embedding_version)

    if hidden and hidden > 0:
        h = int(hidden)
        W1 = rng.normal(0.0, 0.1, size=(d, h))
        b1 = np.zeros(h)
        W2 = rng.normal(0.0, 0.1, size=(h, k))
        b2 = np.zeros(k)
        for _ in range(int(epochs)):
            Z1 = X @ W1 + b1
            A1 = np.maximum(Z1, 0.0)
            P = _softmax_rows(A1 @ W2 + b2)
            dZ2 = (P - Y) / n
            dW2 = A1.T @ dZ2 + l2 * W2
            db2 = dZ2.sum(axis=0)
            dZ1 = (dZ2 @ W2.T) * (Z1 > 0.0)
            dW1 = X.T @ dZ1 + l2 * W1
            db1 = dZ1.sum(axis=0)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        layers = [
            {"W": W1, "b": b1, "activation": "relu"},
            {"W": W2, "b": b2, "activation": "softmax"},
        ]
    else:
        W = rng.normal(0.0, 0.1, size=(d, k))
        b = np.zeros(k)
        for _ in range(int(epochs)):
            P = _softmax_rows(X @ W + b)
            dZ = (P - Y) / n
            dW = X.T @ dZ + l2 * W
            db = dZ.sum(axis=0)
            W -= lr * dW
            b -= lr * db
        layers = [{"W": W, "b": b, "activation": "softmax"}]

    return GenreModel(labels, layers, ev)
