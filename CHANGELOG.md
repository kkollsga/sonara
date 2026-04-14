# Changelog

All notable changes to sonara are documented in this file.

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
