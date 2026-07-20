# Changelog

All notable changes to sonara are documented in this file.

## [0.2.8] - 2026-07-20

### Validated on real music

Before the fix, one unchanged track with three equally frequent chord labels
returned `A`, `Gm`, or `G#m` across five fresh processes. After the fix, a
seed-pinned 400-track commercial-library sample analyzed 400/400 without
error, found five tied predominant-chord cases, and selected the documented
lexicographically smallest winner in all five; every tonal sanity check
passed.

### Fixed

- `predominant_chord` now resolves equal top counts by lexicographically
  ascending label instead of process-randomized hash-map iteration.
  `ANALYSIS_SCHEMA_VERSION` is now 4 so persisted downstream analyses
  invalidate the potentially nondeterministic cached summary.
- Exact genre-model probability ties now select the lowest label index in
  Rust, matching NumPy's documented inference behavior.
- The real-music tonal gate now sorts its corpus before seeded sampling and
  uses explicit secondary label ordering in tied frequency reports.

## [0.2.7] - 2026-07-20

### Fixed
- MP3 loading now retries extension-confirmed streams with the MPEG audio
  reader when generic probing aborts on malformed ID3 text metadata or a false
  ADTS marker. Invalid metadata no longer hides decodable audio; fake MP3s
  remain rejected and failed recovery preserves the original probe diagnostic.

## [0.2.6] - 2026-07-20

### Added
- `sonara::vocal_model::bundled()`: Rust-native accessor for the validated
  vocalness model (`sonara-vocalness-v1`). The artifact is embedded in the
  crate at compile time (`sonara/models/vocalness_v1.json`, ~33 KB,
  dead-stripped when unused), so in-process Rust consumers no longer depend
  on the Python package data. Byte-identical to the Python
  `vocalness_model="bundled"` artifact (enforced by tests in both layers).
- `sonara.genre.train(model_id=...)` / `GenreModel(model_id=...)`: the Python
  genre trainer can now write the optional JSON `id`, making
  `provenance.genre_model_id` reachable for models trained the documented
  way. Omitting it keeps the previous JSON shape and provenance (backward
  compatible).

## [0.2.5] - 2026-07-20

### Validated on real music

**MP3 decode recovery** (35,898-file commercial library): a 2000-file random
sample contained 51 MP3s (2.6%) rejected wholesale by 0.2.4 with recoverable
packet-local errors (`invalid main_data offset`, `huffman decode overrun` —
damaged bit reservoirs); with packet-level recovery all 51 analyze
successfully, the representative file recovers 175.4 of 179.1 s (146 of 6862
packets skipped), and decode time on healthy files is unchanged (2.537 s vs
2.538 s over 25 files). **Vocalness model** (205-track curated labeled set,
disjoint from the 908-track training pool): the built-in contrast heuristic
scores AUC 0.63 with 53% of clearly-vocal tracks under a 0.35 curation
threshold (pop/folk/opera false-negative cluster); the bundled model scores
AUC 0.944 with 5% under the threshold, and the five downstream-reported false
negatives move from ≤ 0.12 to ≥ 0.95. Known model limitations: solo melodic
instrument leads (sax/guitar) can score vocal-high (17% of instrumental
controls ≥ 0.35); spoken narration scores low. Default-path speed held
(compact 1 s / 5 s / 30 s within historical envelopes; interleaved A/B
mixed-sign).

### Added
- Vocalness model socket: `AnalysisConfig::vocalness_model` /
  `vocalness_model=` (Python) loads a JSON MLP (genre-model format + required
  `id`, exactly two labels, one `"vocal"`); its calibrated P(vocal) overrides
  `vocalness` and `instrumentalness`.
- Bundled validated model `sonara-vocalness-v1`
  (`vocalness_model="bundled"`; ships as package data, ~33 KB).
- `sonara.vocal_model`: pure-numpy trainer (standardization folded into the
  first layer), `save`/`load`, `bundled_path()`.
- `AnalysisProvenance.genre_model_id` / `.vocalness_model_id` (additive; also
  in the Python provenance dict): model identity for downstream cache
  invalidation. `None` means the built-in paths produced the fields.
- Genre-model JSON format: optional `id` field (additive in format v1).

### Fixed
- MP3 decoding no longer fails an entire file on packet-local bitstream
  damage: recoverable `DecodeError` packets are skipped (decoder rebuilt on
  `ResetRequired`), with guardrails — non-empty all-finite PCM required, and
  a stream where nothing decodes still surfaces its first decode error.
  Fake/garbage MP3s still fail at probe, unchanged.

## [0.2.4] - 2026-07-17

### Validated on real music

All changes validated on labeled samples from a commercial library.
**Vocalness v2** (62-track probe set: 21 harsh-vocal metal, 23 clean-vocal
pop, 18 instrumental): the v1 heuristic was inverted on real music (pooled
AUC 0.29 — screamed metal scored lowest, solo sax/flute high); v2 scores
monotone harsh (0.92 mean) > clean (0.57) > instrumental (0.23), AUC 0.92;
acceptance tracks: Slipknot/Slayer 1.00, Kenny G/Galway 0.00. Known
ambiguous cases documented (sparse voice+piano ballads, voice-mimicking solo
violin). **Acousticness/danceability** (199-track distributions + 28 genre
anchors): electronic anchors now 0.11 mean acousticness (was 0.45), acoustic
anchors 0.71; danceability spreads to p5 0.20 / p95 0.90 (was p5 0.69).
**bpm_confidence** separates rhythmic from ambient/rubato material at
d=+1.07 (implausible ambient tempi score ~0.39, steady dance 0.83-0.90).

### Fixed

- **`vocalness`/`instrumentalness` inverted on real music.** v2 derives
  vocal presence from mid-band spectral contrast (voice and screams fill the
  0.8-5.6 kHz spectral valleys; clean solo instruments leave them deep)
  instead of tonality+modulation. Both features now require the extended
  pass. `vocal.rs` remains as a documented legacy pitched-melodic-content
  heuristic. **`ANALYSIS_SCHEMA_VERSION` → 3.**
- **`acousticness` and `danceability` were compressed rankers, not absolute
  scales** (floors of ~0.37 / ~0.39 baked into their normalizations).
  Recalibrated; rankings preserved (Spearman 0.97 / 1.00). Consumers with
  pre-0.2.4 cutoffs must re-derive them.

### Added

- **`bpm_confidence`** — always-present [0,1] trust signal for the tempo
  estimate (dominant ACF-peak strength + bpm/beat-rate agreement + onset
  density). Low values flag ambient/rubato material where BPM is unreliable.
- **`tags.original_year`** — original release year from ID3v2.4 `TDOR` /
  v2.3 `TORY` / Vorbis `ORIGINALDATE` (+ tolerant `TXXX:originalyear`-style
  raw keys). `tags.year` now reflects only the file/edition date — on
  reissues prefer `original_year` for era reasoning.

## [0.2.3] - 2026-07-17

### Validated on real music

The chroma fix was validated on a 100-track random sample from a commercial
library (seed-pinned, analyzed at both 22050 Hz and native 44.1 kHz): before,
native-rate key detection collapsed to "F major" on **72/100** tracks with
near-zero confidence; after, the native histogram matches the healthy 22050
spread (top key ≤ 16%), cross-rate key agreement on a 15-track spot set went
from 4/15 to **13/15**, and native F-major dropped to 1/15. A 400-track tonal
batch confirms chords/dissonance (separate HPCP path) unaffected: 400/400
analyzed, all sanity checks pass, flat key distribution (top key 9%).
Deterministic multi-rate regression tests (C-major cadence at
22050/44100/48000) now pin the behavior.

### Fixed

- **Chroma filterbank corrupted all chroma-derived output at sample rates
  above 22050** (key, key_candidates, key_camelot, valence, mood, tonnetz,
  embedding dims 13-25). Two librosa-parity gaps: the missing octave-domain
  Gaussian weighting let broadband energy above the 22050 Nyquist flood the
  chroma sum, and linear two-class bin assignment concentrated wide
  low-frequency bins arbitrarily. Both fixed to librosa 0.10 semantics.
  Chroma values change at every rate: **`ANALYSIS_SCHEMA_VERSION` → 2** and
  **`SIMILARITY_VERSION` → 2** (persisted analyses and stored embeddings
  should be re-generated). Known cost: extended-path (playlist/full) analysis
  +8-13%; the compact default path is unaffected.

### Added

- **Bring-your-own genre model** — `analyze_*(..., genre_model=<path>)` runs a
  user-trained classifier (JSON: softmax/ReLU layers over the versioned 48-dim
  similarity embedding) in pure Rust, populating `genre` + `genre_confidence`.
  Training is numpy-only: `sonara.genre.train(X, y)` → `.save(path)` — no
  PyTorch/ONNX/sklearn. Models carry `embedding_version` and fail fast on
  mismatch. sonara ships no model; the field stays `None` without one.
- **Core-side batch progress** — `analyze_batch_with(paths, sr, config,
  on_done)` for Rust consumers; the Python `progress=` callback now wraps it.

## [0.2.2] - 2026-07-16

### Added

- **File tag passthrough** — opt-in `features=["tags"]`: `analyze_file` /
  `analyze_batch` now surface the ID3v2/Vorbis metadata Symphonia already
  parses during decoding as a `tags` sub-dict (Rust: `tags: Option<TrackTags>`)
  with `title`, `artist`, `album`, `genre`, `year`, `track_no`. No second file
  parse needed downstream. Always absent for `analyze_signal`; the WAV fast
  path carries no tags. Zero cost when not requested.
- **Mood heuristics (v1)** — opt-in `features=["mood"]` populates
  `mood_happy`, `mood_aggressive`, `mood_relaxed`, `mood_sad` in `[0, 1]`:
  documented weighted-term heuristics over key mode, tempo, brightness,
  energy, onset density, and dissonance. Explicitly rough hints, not an ML
  classifier; `genre` remains reserved for a real ML tier.
- **Instrumentalness (v1)** — opt-in `features=["instrumentalness"]`: the
  inverse of the vocal-presence heuristic (`1 - vocalness`, clamped).
- **Batch progress** — `analyze_batch(..., progress=callable)` calls
  `progress(done, total)` after each file completes (completion order;
  results stay input-ordered). Callback exceptions never abort the batch.

## [0.2.1] - 2026-07-15

### Added

- **Analysis provenance** — every result now carries a `provenance` block
  (`schema_version`, effective `sample_rate`, `hop_length`, `mode`,
  `requested_features`), so persisted results are self-describing: frame
  indices convert to seconds as `frame * hop_length / sample_rate`, and
  stale records are detectable via `ANALYSIS_SCHEMA_VERSION`.
- **`chord_events`** — typed, time-spanned chords alongside `chord_sequence`:
  merged runs as `{label, start_sec, end_sec}`, contiguous and covering the
  track. Present whenever chords are computed.
- **Seconds helpers** (Rust) — `TrackAnalysis::frame_to_sec`, `beats_sec`,
  `onsets_sec`, `downbeats_sec`; `TrackAnalysis.print()` shows a Provenance
  line.

### Changed

- **`segments` is now a typed `SegmentEvent` struct** in the Rust API
  (previously a `(start_sec, end_sec, energy)` tuple). The Python dict shape
  is unchanged. Default-path performance verified unchanged (interleaved A/B,
  ≤0.4%).

### Validated on real music

All new features were validated against a 9,400-track commercial library
(60-track random sample + targeted cases): zero batch failures, BPM within
78-186, `energy_level` spreading across the full 1-10 range, fingerprint
separating a real library duplicate pair (0.69) and a gain-changed re-encode
(1.00) from unrelated tracks (0.01), and similarity ranking a same-artist
track #5 of 61 against random material.

### Changed

- **`analyze_batch` entries always carry `path`** — successful results now
  include their input path (failures already did), so consumers no longer
  need to zip results against the input list.
- **`energy_level` recalibrated to real music** — the 1-10 mapping now
  stretches the measured 0.25-0.60 mean-energy band of commercial libraries
  (was 0.30-0.85, which never produced levels above 6 on real tracks).
- **`vocalness` hardened** — a relative modulation-depth gate stops sustained
  chords/pads (whose envelope ripple is numerical, not syllabic) from scoring
  as vocal; per-frame flatness is now energy-weighted. Steady instrumental
  pads score < 0.3 (previously ~0.8); regression tests cover the failure cases.
- **`similarity()` rescaled for interpretability** — raw embedding distances on
  real music occupy a narrow band (~0.08-0.27), so `1 - distance` scored
  everything ~0.85. A calibrated linear stretch (`SIMILARITY_SCALE`, measured
  on a commercial library) now puts a median random pair at ~0.5 and close
  neighbors at 0.65+. The stretch is monotone, so nearest-neighbor rankings
  are unchanged; `distance()` still returns the raw value.
- **Segmentation threshold recalibrated** — the novelty peak threshold rose
  from `mean + 0.5·std` to `mean + 2·std` (named constant). On real pop the
  old threshold hit the 12-segment cap on most tracks; the new one centres at
  ~8 segments per 3-4 minute track with the cap binding <10%, while synthetic
  known-structure boundaries are still recovered. Accuracy-level tuning
  against annotated structure data remains future work.
- CI now runs all nine Python test suites (previously only `test_api.py`).

### Added — opt-in analysis features

All of the following are strictly **opt-in** via `features=[...]` — no default mode
(compact/playlist/full) computes them, and default-mode performance is unchanged
(verified before/after on a 3-minute track: compact ~37 ms, playlist ~72 ms, full ~517 ms).

- **Beat grid** (`features=["beatgrid"]`) — `grid_offset_sec` (first-beat anchor), `downbeats` (bar-starting beats, 4/4 by default or the detected time signature), and `grid_stability` (0-1 grid rigidity). Reuses tracked beats and onset envelope; ~0.2 ms opt-in cost.
- **Structure & energy** (`features=["structure"]`) — `energy_curve` (+`energy_curve_hop_sec`), novelty-based `segments` (contiguous `{start_sec, end_sec, energy}`), `intro_end_sec`/`outro_start_sec` heuristics, and a 1-10 `energy_level`. ~1 ms opt-in cost on a 3-minute track.
- **Similarity embedding** (`features=["embedding"]`) — versioned 48-dim normalized feature vector (`embedding` + `embedding_version`) built from timbre/harmony/rhythm/dynamics/tonal blocks, plus `sonara.similarity(a, b)` (weighted-Euclidean, 0-1) accepting results or raw vectors. No ML dependency; the field is designed so model-based embeddings can replace it behind the same version constant.
- **Acoustic fingerprint** (`features=["fingerprint"]`) — gain-invariant band-energy-difference fingerprint (base64 `fingerprint` + `fingerprint_version`, ~8 subprints/sec) and `sonara.fingerprint_match(a, b)` (0-1, alignment-searched) for duplicate detection across re-encodes. ~6 ms opt-in cost per 3-minute track.
- **Loudness/gain suite** (`features=["loudness"]`) — `true_peak_db` (4x oversampled, BS.1770-4), `replaygain_db` (gain to -18 LUFS), `loudness_curve` (short-term LUFS, 3 s/1 s), `loudness_momentary_max_db`, and `loudness_range_lu` (EBU R128 LRA). Existing `loudness_lufs`/`dynamic_range_db` unchanged.
- **Silence offsets** (`features=["silence"]`) — `leading_silence_sec`/`trailing_silence_sec` at -60 dBFS with click-proof hysteresis, from already-computed frame RMS.
- **Key candidates** (`features=["key_candidates"]`) — top-3 `(key, camelot, score)` list mirroring `bpm_candidates`; first entry always matches `key`.
- **Vocal presence** (`features=["vocalness"]`) — documented 0-1 heuristic (vocal-band energy ratio, tonality, 4-8 Hz syllabic modulation via an O(n) band-pass biquad). ~1.4 ms opt-in cost.

### Added

- **Optional BPM range** — `bpm_min` / `bpm_max` parameters on `analyze_file` / `analyze_signal` / `analyze_batch` (and `AnalysisConfig`). When both are set, detected tempos outside the range are deterministically doubled/halved into it (e.g. a house/techno project range of 79–192). `bpm_max` must be at least `2 * bpm_min`.
- **Tempo candidates in results** — new `bpm_raw` (selected tempo before range alignment) and `bpm_candidates` (top-5 `(bpm, score)` pairs) fields in all modes, so downstream apps can apply their own octave-disambiguation policy. Exposed in Rust via `beat_track_detailed()` returning a `TempoEstimate`.
- **Camelot wheel notation** — new `key_camelot` field (e.g. A minor → `8A`, C major → `8B`) alongside `key` in playlist/full modes, for harmonic mixing workflows.
- **Structured per-file batch errors** — `analyze_batch` now always returns one entry per input path in order; a file that fails to decode yields `{path, error, error_kind}` (`io`, `decode`, `unsupported_format`, …) instead of aborting the whole batch. New `TrackAnalysis.failed` property.
- **Accuracy regression harness** — `sonara/tests/bpm_accuracy.rs` (synthetic ground-truth suite: 15 tempos × 3 patterns, octave-error and drift metrics with hard-asserted thresholds) and `sonara/examples/accuracy_eval.rs` (CSV-driven evaluator for labeled corpora with worst-offenders table).
- **Contribution infrastructure** — `CONTRIBUTING.md`, issue templates, and PR template; detection changes now require accuracy evidence.

### Fixed

- **Half-BPM octave errors** — tempo estimation now collects all ACF candidates and lifts a well-supported 2x/1.5x metrical multiple when the raw pick is suspiciously low. Fixes the classic electronic-music failure where ~126–140 BPM tracks were reported at half tempo. Synthetic-suite octave errors at 126/140 BPM eliminated across all test patterns.
- **1–3 BPM quantization drift** — ACF peak selection now uses parabolic (fractional-lag) interpolation instead of integer lags. Synthetic-suite median absolute error dropped from 0.90 to 0.27 BPM.
- **Degenerate onset envelopes** — flat/silent onset envelopes now fall back to `start_bpm` with no beats instead of producing arbitrary output.

### Known limitations

- 192 BPM material can still be halved to ~96 (outside the metrical-lift tiers); use a `bpm_min`/`bpm_max` range whose floor excludes ~96 to rescue it.
- The metrical lift can double genuinely slow (60–70 BPM) tracks that have dense offbeat hats; a `bpm_max` bound disambiguates. Beat-grid regularity scoring across candidates is the planned principled fix. Both cases are tracked as `#[ignore]`d tests in the accuracy suite.

## [0.1.6] - 2026-04-14

### Added

- **Tonal analysis module** (`tonal.rs`) — HPCP, chord detection, and sensory dissonance.
  - `hpcp()` — Harmonic Pitch Class Profile with spectral peak detection and harmonic weighting (Gomez 2006).
  - `chords_from_beats()` / `chords_from_frames()` — chord recognition via template correlation against 24 major/minor profiles.
  - `chord_descriptors()` — predominant chord, chord change rate, unique chord count.
  - `dissonance()` / `dissonance_from_peaks()` — Sethares (1998) Plomp-Levelt sensory dissonance model.
- **Playlist pipeline integration** — new `TrackAnalysis` fields: `chord_sequence`, `chord_change_rate`, `predominant_chord`, `dissonance`, `embedding` (placeholder for future ONNX).
- **Python bindings** for all tonal functions.

### Performance (9x pipeline speedup)

- **Fused all post-loop features into the per-frame FFT pass** — spectral contrast, HPCP, and dissonance now compute inline during FFT processing. Eliminated the 32MB power spectrum matrix and 3 post-loop passes with cold cache misses.
- **Pre-computed DCT matrix for MFCCs** — replaced 12.9M `cos()` calls with a cached (13, 128) coefficient matrix. ~100x faster for this stage.
- **Compute magnitude once per frame** — `sqrt(power)` was called 4x per bin in extended mode. Now computed once and reused for centroid, bandwidth, rolloff, and contrast.
- **Sparse chroma filterbank** — applied same sparse representation as mel, reducing 95M multiply-adds to ~5M.
- **Partial sort for spectral contrast** — `select_nth_unstable` (O(n)) instead of full `sort` (O(n log n)) for percentile finding.
- **Moved time_signature/tempo_curve to Full mode** — metrogram (O(n^3)) was costing 445ms; now only computed in Full mode or when explicitly requested.
- **Result**: Playlist mode 726ms → 80ms for a 3-minute track. Per 10s track: ~4ms.

### Fixed

- **Beat tracking DP** — removed erroneous `score_thresh` gate in the forward pass that broke backtracking chains on real music (only 1 beat detected). Now matches the original Ellis 2007 algorithm.

### Changed

- **Removed `accurate` flag** — all features now always use their most accurate algorithms (proper chroma filterbank, log-spaced spectral contrast, top-N peak detection). The DFA danceability function remains available standalone as `perceptual::danceability_dfa()`.

## [0.1.5] - 2026-04-14

### Fixed

- **Key detection C minor bias eliminated** — replaced mel-to-chroma approximation with proper chroma filterbank (`filters::chroma()`) in all analysis modes. C minor dropped from 39% to 5.5% on a 200-song test. The mel approximation had two flaws: frequencies below C0 defaulted to bin 0 (C), and mel bands were too wide to resolve semitones accurately.
- **Spectral contrast accuracy** — now always uses proper log-spaced frequency bands on the magnitude spectrum instead of the mel sub-band approximation. Removed the separate `accurate` vs fast branch.
- **Key profiles upgraded** — switched from Krumhansl (1990, classical-biased) to Temperley MIREX 2005 profiles (corpus-derived, better for popular music).
- **Energy scaling improved** — expanded RMS normalization range from 0.4 to 0.5, onset density ceiling from 8 to 10, reduced sigmoid steepness from 5.0 to 4.0 with shifted center (0.5 → 0.45). Energy floor raised from 0.007 to 0.14 on quiet music.
- **Python test dtype** — fixed `test_api.py` and benchmarks to use `np.float32` arrays (was `np.float64`), matching the f32 Rust bindings.

### Changed

- **`accurate` flag simplified** — now only controls danceability (heuristic vs DFA). Chroma and spectral contrast are always computed with the accurate method since the cost is negligible.
- Playlist mode stores full power spectrogram per frame for chroma filterbank projection and proper spectral contrast.

## [0.1.4] - 2026-04-14

### Performance

- **Switch to f32 precision** — halves memory bandwidth across the entire pipeline; eliminates f32-to-f64 conversion during MP3 decode (Symphonia decodes to f32 natively). Standard practice for audio processing libraries.
- **Parallelize fused analysis loop** — per-frame FFT + mel + centroid computation now uses rayon when frame count exceeds 32. Thread-local FFT caches avoid lock contention.
- **Fast-path 2:1 decimation** — 31-tap half-band FIR filter for the common 44100-to-22050 Hz case, replacing full sinc resampling (~20x faster for this ratio).
- **Pre-computed ln() table in beat DP** — eliminates transcendental function calls in the beat tracking inner loop.
- **Mel filterbank caching** — thread-local cache avoids recomputing the mel filterbank across batch calls with the same parameters.

### Added

- **Analysis modes**: new `mode` parameter (`compact`, `playlist`, `full`) replaces the old boolean flags. Cleaner API with preset configurations.
- **`accurate` flag**: trades speed for precision on chroma (filterbank vs mel-approx), spectral contrast (log-spaced bands vs mel sub-bands), and danceability (DFA vs heuristic).
- **Custom feature selection**: `features=["bpm", "energy", "key"]` parameter to cherry-pick specific features regardless of mode.
- **LUFS loudness** (`loudness_lufs`): integrated loudness per ITU-R BS.1770-4 / EBU R128 standard, computed via K-weighting filter. Added to compact set (always computed).
- **Energy** (0-1): perceptual intensity from RMS, spectral centroid, onset density, bandwidth.
- **Danceability** (0-1): heuristic from beat regularity + tempo + onsets (fast), or Detrended Fluctuation Analysis per Streich & Herrera 2005 (accurate).
- **Key detection**: Krumhansl-Schmuckler algorithm with Pearson correlation against major/minor profiles, returns key + mode + confidence.
- **Valence** (0-1): heuristic mood from key mode + tempo + brightness.
- **Acousticness** (0-1): heuristic from spectral flatness + rolloff + onset density.
- **Spectral features in fused pipeline**: bandwidth, rolloff, flatness computed per-frame alongside existing centroid/RMS with minimal overhead.
- **MFCCs**: 13-coefficient mean via DCT-II of log mel spectrogram.
- **Chroma**: mel-approximated (fast) or chroma filterbank (accurate) pitch class distribution.
- **Spectral contrast**: mel sub-band (fast) or log-spaced frequency band (accurate) peak-valley ratios.
- **Tier 3 placeholders**: `mood_happy`, `mood_aggressive`, `mood_relaxed`, `mood_sad`, `instrumentalness`, `genre` fields reserved for future ML models.
- New `perceptual.rs` module with all perceptual computation logic.
- Analysis benchmarks (`bench_analyze.rs`) for compact, playlist, and playlist+accurate modes.
- Accuracy test suite (`tests/accuracy.rs`) with 33 tests validating feature correctness.

### Changed

- **Python API**: `analyze_file`, `analyze_signal`, `analyze_batch` now accept `mode`, `accurate`, and `features` keyword arguments.
- **NumPy dtype**: all arrays are now `float32` / `complex64` (was `float64` / `complex128`).
- **RMS computation**: fixed to use time-domain calculation instead of windowed FFT (which underestimated by the Hann window power factor).

## [0.1.3] - 2026-04-14

### Performance

- Switch all computation from f64 to f32 precision.
- Parallelize the fused analysis loop with rayon for frames >= 32.
- Add fast-path half-band FIR decimation for 44100->22050 Hz resampling.
- Pre-compute ln() lookup table in beat tracking DP.
- Add thread-local mel filterbank cache for batch processing.

### Changed

- Replaced release workflow with CI workflow.
- Bumped version to 0.1.3.

## [0.1.2] - 2026-04-14

### Changed

- Rename project from **canora** to **sonara** across all crates, Python modules, imports, CI, and documentation.
- Rename `canora/` crate directory to `sonara/`, `canora-python/` to `sonara-python/`.
- Rename Python package from `canora` to `sonara` (including `canora.display` to `sonara.display`).
- Update all internal error types from `CanoraError` to `SonaraError`.
- Update README, examples, and benchmark scripts for the new name.

## [0.1.1] - 2026-04-14

### Fixed

- Remove unused imports (`std::f64::consts::PI`, `canora::core::convert`) causing warnings.
- Fix conditional import of `ndarray::s` (only needed in test cfg).
- Update rubato resampler API from deprecated `FftFixedInOut` to new `Fft` + `SequentialSliceOfVecs` API.
- Fix dependency resolution issues in `Cargo.toml` (workspace dependencies).
- Add `#![allow(non_snake_case)]` for Python binding parameter names.
- Fix `display.py` import paths and formatting.
- Fix license field in `pyproject.toml`.
- CI workflow fixes for cross-platform builds.

## [0.1.0] - 2026-04-13

### Added

- Initial release as **canora**.
- Pure Rust core library with 214 functions across 22 source files.
- PyO3 Python bindings with numpy zero-copy interop.
- 92 audio analysis functions: STFT/ISTFT, mel spectrogram, MFCC, chroma, beat tracking, onset detection, pitch estimation (YIN/pYIN), CQT/VQT/hybrid CQT/pseudo CQT, Griffin-Lim, and more.
- Fused single-pass FFT pipeline with sparse mel projection (~97% sparsity).
- Rayon-based parallel batch file analysis and parallel STFT.
- Audio I/O via Symphonia (MP3, FLAC, OGG) and hound (WAV).
- High-quality resampling via rubato (sinc interpolation).
- BLAS acceleration support (Apple Accelerate, OpenBLAS).
- Display module with `specshow` for matplotlib-based visualization.
- Criterion benchmarks for STFT, CQT, sequence, and utilities.
- Full conversion suite: Hz/mel/MIDI/note/octave/svara, frequency weighting (A/B/C/D/Z), frame/time/sample conversions.
- Effects: time stretch, pitch shift, trim, split, preemphasis/deemphasis.
- Notation: key_to_notes, key_to_degrees, Indian music notation (mela/thaat/svara).
- CI/CD workflows for testing and PyPI release.
