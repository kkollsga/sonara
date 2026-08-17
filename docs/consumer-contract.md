# Consumer contract — sonara's core Rust API

**Owner:** sonara. This is the single durable copy.
**Status:** current, verified against sonara 0.3.5 / `ANALYSIS_SCHEMA_VERSION = 6`
on 2026-07-29.
**Consumer of record:** sonagram (since 2026-07-18). Originally raised by
kglite on 2026-07-15.

sonara produces the API this contract is about, so sonara owns the obligation
and keeps the durable copy. Downstreams hold a pointer, not a second copy.

## Why this file is tracked

Between 2026-07-15 and 2026-07-29 this contract existed as two "durable copies"
in two *gitignored* `dev-docs/designs/` folders — one here, one in kglite —
each of which described itself as the durable one. They diverged, and both
still listed as **unmet** three API requirements sonara had already shipped in
0.2.1 on 2026-07-15. The closure was recorded only in a third repo's notes
(`sonagram:dev-docs/designs/upstream-contracts.md`). One tracked copy, in the
repo that owes the obligation, is the fix.

## Ownership split

Confirmed by kglite 2026-07-18 (kglite inbox
`2026-07-16-from-sonagram-ownership-boundary.md`, status footer), after
sonagram took the mapping seat kglite's original 2026-07-15 note had assumed
for itself.

| Concern | Owner |
|---|---|
| Analysis → typed `TrackAnalysis` | **sonara** |
| Music-graph schema, source hashing, mapping, library scan | **sonagram** |
| Graph storage, Cypher, MCP graph exposure, embedding *engine* | **kglite** |

sonagram may persist sonara's 48-dimensional similarity vector in kglite's
`EmbeddingStore` (as `model_id = "sonara-similarity-v1"`, re-keyed on sonara
version bumps). The stored vector is **unweighted** — the per-dimension weights
live inside `distance()` and are applied at query time, so a weighting change
(e.g. a new similarity profile) never alters a persisted vector. Ownership of
that music-domain vector and its provenance stays with sonagram; kglite owns
the generic storage and query machinery, not music-specific mapping or
lifecycle.

## The consumed surface

Typed Rust values directly from the `sonara` crate — `analyze_file` /
`analyze_signal` / `AnalysisConfig` → `TrackAnalysis`. **Not** Python dicts,
not JSON. `SonaraError` is a dependency-neutral structured error. sonara stays
graph-agnostic; it is never asked to know about graphs.

## Compatibility requests — all three CLOSED in 0.2.1 (2026-07-15)

The original asks and what shipped. Kept for lineage; do not re-raise them.

| Ask (2026-07-15) | Shipped |
|---|---|
| **Self-describing time.** `beats` / `onset_frames` / downbeats were frame indices; effective sample rate and `hop_length` were internal to `analyze.rs`. | `TrackAnalysis.provenance: AnalysisProvenance` (always present) carries effective post-resample `sample_rate` and `hop_length`, plus `frame_to_sec` / `beats_sec` / `onsets_sec` / `downbeats_sec` helpers. |
| **Typed event records with spans.** `chord_sequence: Option<Vec<String>>` had no temporal alignment; segments were tuples. | `chord_events: Option<Vec<ChordEvent>>` (`{ label, start_sec, end_sec }`, merged runs, contiguous, covering the track); `segments` is a named `SegmentEvent { start_sec, end_sec, energy }`. |
| **Persistable provenance** to detect stale analysis. | `AnalysisProvenance { schema_version, sample_rate, hop_length, mode, requested_features (sorted), vocalness_model_id, … }`, pinned to `ANALYSIS_SCHEMA_VERSION` (now 6), bumped whenever field meaning or units change. |

## Standing boundaries — what sonara undertakes to hold

Verified against sonara 0.3.5 on 2026-07-29 (`sonara/Cargo.toml`,
`cargo tree -p sonara`):

- **The core crate stays PyO3-free.** PyO3 and NumPy are confined to
  `sonara-python`. A consumer's core dependency path must never acquire them
  transitively. *(Verified: no `pyo3` / `numpy` in the `sonara` tree.)*
- **`default = []`.** Platform acceleration (`accelerate` → `ndarray/blas` +
  `blas-src/accelerate`) is opt-in. *(Verified.)*
- **No ML/ONNX runtime in the default feature set.** The `aggression` feature
  is the only path that pulls a learning runtime (`ferricml`, `sha2`), and it
  is off by default. *(Verified.)*
  - Clarification added 2026-07-29: the **bundled/bring-your-own vocalness
    classifier** (`sonara::vocal_model`) is *in* the default build and does not
    breach this. It is a pure-Rust feed-forward evaluator over the existing
    hand-crafted embedding, using the same JSON MLP format as `genre`, with no
    runtime dependency. It extends the versioning discipline rather than
    weakening it: the model `id` is required and is carried into
    `AnalysisProvenance::vocalness_model_id` so downstream caches can
    invalidate scores produced by a different model or none.
- **Versioned vector layouts.** Fingerprint and similarity vectors carry
  explicit version constants and must remain distinguishable on persisted
  records. Never reinterpret an existing layout without a version bump; a
  future learned-vector layout gets a new explicit version/model identity
  rather than silently reusing the `embedding` field.
- **`analyze_signal` stays the low-level, dependency-light entry point.**
  Symphonia currently enables all codecs and formats unconditionally. Do not
  churn this speculatively — but if a signal-only or reduced-codec feature
  split is ever introduced, `analyze_signal` must stay buildable without the
  full decoder bundle.
- **No serde/JSON bridge is required** solely to serve a downstream. Direct
  typed access is preferable and avoids another dependency contract.

## Versioning

Downstreams pin a compatible pre-1.0 sonara release and key stored records on
`provenance.schema_version` (plus the per-subsystem `embedding_version` /
fingerprint version where those are consumed). sonagram's current floor is
`sonara >= 0.2.2`. Heuristic-semantics changes bump `ANALYSIS_SCHEMA_VERSION`.

### Per-feature freshness

Additive fields and APIs ship **without** an `ANALYSIS_SCHEMA_VERSION` bump —
the whole-record version moves only when existing field meaning or units
change. A consumer deciding whether a stored record needs refreshing should
therefore key **per feature**, not on the whole-record version: field presence
on the record, the per-feature model ids in provenance (`vocalness_model_id`,
`genre_model_id`), and the declared dependency map `feature_dependencies()` —
which names each feature's dependency class and the record fields a decode-free
recompute reads (`augment_analysis` / `can_augment` consume the same map, so a
missing feature can often be filled in without re-decoding the audio).

Similarity weighting profiles version independently of the vector layout:
`SIMILARITY_VERSION` (unchanged at 2) still identifies the stored vector and
the default metric, while each named profile (e.g. `timbre`) carries its own
weight-table version, surfaced through `SIMILARITY_PROFILES`. Profiles apply at
query time and never change stored vectors, so a profile bump forces no
re-keying. This shape was communicated to sonagram on 2026-08-17 (sonagram
inbox, `2026-08-17-from-sonara-augment-api-shape-genre-nogo-and-a-question.md`);
this section records the standing policy.

## Notification obligations

sonara notifies (via `notify`) **before** any change that:
- couples the core crate to PyO3 or NumPy,
- moves an ML/ONNX runtime into the default feature set,
- reinterprets a fingerprint or similarity layout without a version bump.

sonara replies if a requested boundary conflicts with its intended API shape.

**Recipient: sonagram. kglite is deliberately NOT on this list** (decided
2026-07-29). sonagram consumes the API and can act on the notice; kglite
cannot. The obligation was written on 2026-07-15 naming kglite, and stayed
that way through the 2026-07-18 handover to sonagram.

kglite was considered and ruled out on evidence, recorded here so it is not
re-added out of habit: it has **no cargo dependency on sonara**, no 48-dim
assumption anywhere in the engine, and `EmbeddingStore` is dimension-agnostic
(it carries `dimension` as data — `EmbeddingStore::new(src_store.dimension)`).
kglite stores vectors it is handed and never interprets them, so a change to
what they *mean* cannot break it. Notifying a repo that cannot act is the
noise that trains people to ignore the inbox.

## History

- 2026-07-15 — kglite raises the contract
  (`sonara:inbox/read/2026-07-15-from-kglite-future-core-consumer-contract.md`).
- 2026-07-15 — sonara ships all three asks in 0.2.1 and reports back
  (`kglite:inbox/read/2026-07-15-from-sonara-consumer-contract-shipped.md`).
- 2026-07-16 — sonagram claims the mapping seat
  (`kglite:inbox/read/2026-07-16-from-sonagram-ownership-boundary.md`).
- 2026-07-18 — kglite confirms the narrower split.
- 2026-07-29 — the two gitignored copies are collapsed into this tracked file;
  the notification-routing gap above is opened.
- 2026-08-17 — augment lane and similarity profiles ship additively (no schema
  bump); the "pre-weighted vector" mis-description is corrected (the stored
  vector is unweighted — weights apply inside `distance()` at query time) and
  the per-feature freshness policy above is recorded. sonagram notified the
  same day (API shape, profile versioning, genre NO-GO).
