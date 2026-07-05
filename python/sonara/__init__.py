"""Sonara: High-performance audio analysis library for music information retrieval."""

from sonara._sonara import *  # noqa: F401, F403
from sonara._sonara import __version__  # noqa: F401 — sourced from Cargo.toml
from sonara._sonara import (
    analyze_file as _analyze_file,
    analyze_signal as _analyze_signal,
    analyze_batch as _analyze_batch,
    # --- similarity ---
    similarity as _similarity,
)
from sonara._sonara import fingerprint_match  # noqa: F401 — duplicate detection
from sonara._result import TrackAnalysis
from sonara import display  # noqa: F401


def analyze_file(path, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None):
    """Analyze an audio file and return a `TrackAnalysis` (dict subclass with `.print()`)."""
    return TrackAnalysis(_analyze_file(
        path, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
    ))


def analyze_signal(y, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None):
    """Analyze a signal array and return a `TrackAnalysis` (dict subclass with `.print()`)."""
    return TrackAnalysis(_analyze_signal(
        y, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
    ))


def analyze_batch(paths, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None):
    """Analyze a list of audio files in parallel; returns a `list[TrackAnalysis]`.

    Errors are isolated per file: the returned list has exactly one entry per
    input path, in the same order as ``paths``, and every entry carries its
    input ``path``. A file that fails to decode does
    not abort the batch — instead its entry is a failure ``TrackAnalysis`` with
    ``path``, ``error`` (human-readable, includes container/codec and cause) and
    ``error_kind`` (a short stable category such as ``"decode"``, ``"io"`` or
    ``"unsupported_format"``). Use ``result.failed`` to distinguish them.
    """
    return [
        TrackAnalysis(r)
        for r in _analyze_batch(
            paths, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
        )
    ]


# --- similarity ---
def _as_embedding(x):
    """Extract an embedding vector (list of float) from a TrackAnalysis dict or
    accept a raw vector (list / numpy array) as-is. Returns (vector, version)."""
    # Mapping (TrackAnalysis is a dict subclass) → pull the stored embedding.
    if isinstance(x, dict):
        if "embedding" not in x:
            raise ValueError(
                "TrackAnalysis has no 'embedding'; analyze with "
                "features=['embedding'] to compute the similarity vector"
            )
        return list(x["embedding"]), x.get("embedding_version")
    # Raw vector (list, tuple, or numpy array).
    return [float(v) for v in x], None


def similarity(a, b):
    """Similarity of two tracks in ``[0, 1]`` (higher = more similar).

    Accepts two ``TrackAnalysis`` results (analyzed with
    ``features=['embedding']``) or two raw embedding vectors (lists / numpy
    arrays). Uses a weighted, normalized Euclidean metric over the hand-crafted
    similarity vector; identical inputs return ``1.0``.

    Raises ``ValueError`` if two dict inputs carry different
    ``embedding_version`` values (their vectors are not comparable).
    """
    va, ver_a = _as_embedding(a)
    vb, ver_b = _as_embedding(b)
    if ver_a is not None and ver_b is not None and ver_a != ver_b:
        raise ValueError(
            f"embedding_version mismatch ({ver_a} != {ver_b}); "
            "vectors from different layout versions are not comparable"
        )
    return _similarity(va, vb)
