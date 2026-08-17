"""Sonara: High-performance audio analysis library for music information retrieval."""

from sonara._sonara import *  # noqa: F401, F403
from sonara._sonara import __version__  # noqa: F401 — sourced from Cargo.toml
from sonara._sonara import (
    analyze_file as _analyze_file,
    analyze_signal as _analyze_signal,
    analyze_batch as _analyze_batch,
    # --- augment lane ---
    augment_analysis as _augment_analysis,
    # --- similarity ---
    similarity as _similarity,
)
from sonara._sonara import (  # noqa: F401 — augment lane introspection
    augment_blocker,
    can_augment,
    feature_dependencies,
)
from sonara._sonara import fingerprint_match  # noqa: F401 — duplicate detection
from sonara._result import TrackAnalysis
from sonara import display  # noqa: F401
from sonara import genre  # noqa: F401 — bring-your-own genre model trainer/loader
from sonara import vocal_model  # noqa: F401 — bring-your-own vocalness model trainer/loader


def analyze_file(path, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None):
    """Analyze an audio file and return a `TrackAnalysis` (dict subclass with `.print()`).

    ``features`` selects features explicitly (overriding ``mode``) and is the
    only way to enable the opt-in features: ``beatgrid``, ``structure``,
    ``embedding``, ``aggression``, ``fingerprint``, ``loudness``, ``silence``,
    ``key_candidates``, ``vocalness``. See the README for the full list.

    ``genre_model`` is a path to a user-trained genre model (JSON). When given,
    the result carries ``genre`` and ``genre_confidence``. See ``sonara.genre``
    for the trainer and the JSON format; the model's ``embedding_version`` must
    match ``sonara.SIMILARITY_VERSION`` (else a ``ValueError`` is raised).

    ``vocalness_model`` is a path to a user-trained vocal-presence model (JSON;
    see ``sonara.vocal_model``). When given, ``vocalness`` and
    ``instrumentalness`` are the model's calibrated P(vocal) / its inverse
    (overriding the built-in heuristic), and the result's
    ``provenance.vocalness_model_id`` carries the model's required ``id``.
    """
    if vocalness_model == "bundled":
        vocalness_model = vocal_model.bundled_path()
    return TrackAnalysis(_analyze_file(
        path, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
        genre_model=genre_model, vocalness_model=vocalness_model,
    ))


def analyze_signal(y, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None):
    """Analyze a signal array and return a `TrackAnalysis` (dict subclass with `.print()`).

    ``genre_model`` (path to a user-trained model JSON) adds ``genre`` and
    ``genre_confidence`` to the result. See ``sonara.genre``.
    """
    if vocalness_model == "bundled":
        vocalness_model = vocal_model.bundled_path()
    return TrackAnalysis(_analyze_signal(
        y, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
        genre_model=genre_model, vocalness_model=vocalness_model,
    ))


def analyze_batch(paths, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None, progress=None, genre_model=None, vocalness_model=None):
    """Analyze a list of audio files in parallel; returns a `list[TrackAnalysis]`.

    Errors are isolated per file: the returned list has exactly one entry per
    input path, in the same order as ``paths``, and every entry carries its
    input ``path``. A file that fails to decode does
    not abort the batch — instead its entry is a failure ``TrackAnalysis`` with
    ``path``, ``error`` (human-readable, includes container/codec and cause) and
    ``error_kind`` (a short stable category such as ``"decode"``, ``"io"`` or
    ``"unsupported_format"``). Use ``result.failed`` to distinguish them.

    ``progress``, if given, must be callable and is invoked as
    ``progress(done, total)`` after **each** file finishes (success or failure).
    ``done`` counts completions in *completion order* (not input order) and
    ``total == len(paths)``. A raising/broken callback never aborts the batch —
    its exception is swallowed (per-file isolation is a contract). Passing
    ``progress=None`` (the default) runs the original zero-overhead path.

    ``genre_model`` (path to a user-trained model JSON) adds ``genre`` and
    ``genre_confidence`` to each successful entry. See ``sonara.genre``.
    """
    if vocalness_model == "bundled":
        vocalness_model = vocal_model.bundled_path()
    return [
        TrackAnalysis(r)
        for r in _analyze_batch(
            paths, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
            progress=progress, genre_model=genre_model, vocalness_model=vocalness_model,
        )
    ]


# --- augment lane ---
def augment_analysis(cached, features=None, *, audio_path=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None):
    """Recompute named features onto a copy of a cached analysis dict.

    ``cached`` is a ``TrackAnalysis`` (or plain dict of the same shape, e.g.
    loaded back from JSON). Returns a new ``TrackAnalysis``; the input is never
    mutated, and fields you did not ask about are never cleared.

    Feature names are case-insensitive; an unknown name raises ``ValueError``
    (no silent fallback). Decode-free features (``feature_dependencies()``
    class ``"scalars"``/``"embedding"`` with all ``required_evidence`` fields
    present on this record — see ``can_augment``/``augment_blocker``) are
    recomputed from the cached fields alone, reproducing the standalone meaning
    of the feature. Anything else (class ``"audio"``/``"frame_curves"``, or
    missing evidence) needs the audio again: pass ``audio_path`` to enable one
    re-analysis at the record's own ``provenance.sample_rate`` computing exactly
    the blocked features (an ``aggression`` request auto-routes through its
    dedicated 22.05 kHz lane); without it the call raises ``ValueError`` naming
    each blocked feature and why.

    ``vocalness``/``instrumentalness`` are one shared value and always update
    together (either name requests both), along with
    ``provenance.vocalness_model_id`` — the model when ``vocalness_model`` is
    given (``"bundled"`` selects the packaged model), the built-in heuristic
    otherwise. Genre has no feature name: passing ``genre_model`` IS the
    request (``features=[]`` plus a model is valid) and populates
    ``genre``/``genre_confidence`` + ``provenance.genre_model_id``. There is no
    bundled genre model.

    A record whose ``provenance.schema_version`` differs from this build's
    schema raises ``ValueError`` (augmenting would mix field eras —
    re-analyze instead), as does a recorded ``embedding_version`` differing
    from ``sonara.SIMILARITY_VERSION`` for embedding-consuming requests.
    The re-analysis fallback inherits the record's recorded
    ``provenance.bpm_min``/``bpm_max``; ``bpm_min``/``bpm_max`` here apply only
    to records predating range recording. ``provenance.requested_features``
    becomes the union of the record's request and the augmented names.
    """
    if vocalness_model == "bundled":
        vocalness_model = vocal_model.bundled_path()
    return TrackAnalysis(_augment_analysis(
        cached, features, audio_path=audio_path, bpm_min=bpm_min, bpm_max=bpm_max,
        genre_model=genre_model, vocalness_model=vocalness_model,
    ))


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


def similarity(a, b, *, profile="default"):
    """Similarity of two tracks in ``[0, 1]`` (higher = more similar).

    Accepts two ``TrackAnalysis`` results (analyzed with
    ``features=['embedding']``) or two raw embedding vectors (lists / numpy
    arrays). Uses a weighted, normalized Euclidean metric over the hand-crafted
    similarity vector; identical inputs return ``1.0``.

    ``profile`` selects the distance-time weight table: ``"default"`` (the
    historical balanced metric) or ``"timbre"`` (spectral texture dominates;
    tempo/energy demoted, so neighbors share sonic style). Profiles never
    change the stored vector — see ``sonara.SIMILARITY_PROFILES`` for the
    name → weight-table-version map. Unknown names raise ``ValueError``.

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
    return _similarity(va, vb, profile=profile)
