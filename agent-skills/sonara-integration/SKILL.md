---
name: sonara-integration
description: Analyze music with Sonara's Python API, choose efficient feature modes, interpret confidence and abstention, process libraries safely in batches, and verify content-addressed validation receipts. Use when an agent needs to integrate Sonara audio analysis or assess a Sonara validation result.
---

# Sonara integration

## Analyze audio

1. Use `sonara.analyze_file(path)` for one file, `analyze_signal(y, sr=...)` for decoded mono samples, or `analyze_batch(paths)` for a library.
2. Start with `mode="compact"`; select `playlist` only when organization features are needed. Supplying `features=[...]` replaces mode selection rather than adding to it, so list exactly the outputs required; internal dependencies are resolved automatically.
3. Preserve the returned `provenance`, model IDs, schema versions, and error records with persisted results.

- `compact`: Fast core scan: tempo, beats, onsets, RMS, centroid, zero crossings, and dynamic range.
- `playlist`: Adds spectral, tonal, and perceptual features for library organization.
- `full`: Playlist analysis plus expensive rhythm curves and time signature.

- **core**: Timing and low-cost signal summaries. Request with `features=['bpm', 'beats', 'onsets', 'rms', 'dynamic_range', 'centroid', 'zcr', 'onset_density']` as needed.
- **spectral-tonal**: Timbre, harmony, key, chord, and consonance evidence. Request with `features=['bandwidth', 'rolloff', 'flatness', 'contrast', 'mfcc', 'chroma', 'chords', 'dissonance', 'key', 'key_candidates']` as needed.
- **perceptual-rhythm**: Interpretable perceptual and longer-timescale rhythm estimates. Request with `features=['energy', 'danceability', 'valence', 'acousticness', 'tempo_curve', 'time_signature', 'beatgrid', 'structure', 'loudness', 'silence']` as needed.
- **models-and-identity**: Opt-in similarity, learned ranking, vocal, mood, and content-identity outputs. Request with `features=['embedding', 'aggression', 'fingerprint', 'vocalness', 'mood', 'instrumentalness', 'tags']` as needed.

## Interpret results

- A confidence or support field describes evidence strength for that estimator; it is not a universal probability of correctness.
- Retain model/schema versions and requested feature provenance when comparing stored results. Do not compare incompatible versions.
- Treat a null estimate as an intentional abstention. In particular, aggression_score is null when content support is insufficient; do not coerce it to zero.
- analyze_batch preserves input order and isolates per-file failures in error/error_kind records. Progress callback failures never cancel the batch.

## Verify validation evidence

1. Obtain the receipt, custody proof, and trust root through independent channels.
2. Call `sonara.validation.verify(receipt, proof, trust_root)` or `sonara validate verify --receipt ... --proof ... --trust-root ...`.
3. Accept the outcome only if verification succeeds and the receipt's evaluation digest names the intended candidate and suite.
4. Never tune from sealed inputs, infer private membership from aggregate receipts, or trust a root merely because it is embedded in the proof.

Interpret a PASS only after offline verification against a separately pinned trust root. A receipt bundled with its own unpinned trust root is not independently trusted.

Packaged validation contract identities:

- `command-spec.schema.json` — SHA-256 `1539fb3e9e2cf13435bfc1b6225ec40dfc11f8b856d0f55d6d8f4d7ec96226a7`
- `custody-proof.schema.json` — SHA-256 `61ac9ab6e31b29d293df2abbef0d92b2f4ddf238de3882e523a031b43c4e17a3`
- `evaluation-receipt.schema.json` — SHA-256 `4ba11a3b83e0c78c8223590885208d0ef4972162f78e6e71a65c6eaeca133743`
- `validation-bindings.schema.json` — SHA-256 `2452eaae8d89680292495791eb6244e36fc31b85c27b003597a3682447579d84`
- `validation-capsule.schema.json` — SHA-256 `8d90c1ca3e40961dbb0d422edda55002f2201691733c08cc99a835f5d7dba808`
