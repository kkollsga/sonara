# sonara

**High-performance audio analysis library for Python, written in Rust.**

High-performance audio feature extraction, batch analysis, and built-in perceptual features for playlist generation.

> *sonara* — from Latin *sonare*, "to sound, to resonate"

## Installation

```bash
pip install sonara
```

Requires Python 3.9+. Pre-built wheels available for Linux, macOS (Intel & Apple Silicon), and Windows.

Build from source:

```bash
git clone https://github.com/kkollsga/sonara.git
cd sonara
pip install maturin
maturin develop --release
```

## Quick Start

```python
import sonara
import numpy as np

# Load audio
y, sr = sonara.load("track.mp3", sr=22050)

# STFT
D = sonara.stft(y)
S_db = sonara.amplitude_to_db(np.abs(D))

# Mel spectrogram + MFCC
mel = sonara.melspectrogram(y=y, sr=22050.0)
mfcc = sonara.mfcc(y=y, sr=22050.0, n_mfcc=13)

# Beat tracking
tempo, beats = sonara.beat_track(y=y, sr=22050)

# Chroma & HPCP
chroma = sonara.chroma_stft(y=y, sr=22050.0)
hpcp = sonara.hpcp(power_spec, freqs)

# Pitch estimation
f0, voiced, prob = sonara.pyin(y, fmin=65.0, fmax=2093.0, sr=22050)
```

## Analysis Pipeline

sonara includes a fused analysis pipeline that extracts all features in a single optimized pass. Three modes control the depth of analysis:

### Modes

| Mode | Features | Time (10s track) | Use case |
|------|----------|-------------------|----------|
| **`compact`** | 11 core features | ~1.2 ms | Fast scanning, metadata |
| **`playlist`** | 30+ features incl. tonal & perceptual | ~4 ms | Playlist generation, music discovery |
| **`full`** | All features incl. time signature | ~50 ms | Research, comprehensive analysis |

### Compact mode (default)

Core signal features, always computed:

```python
r = sonara.analyze_file("track.mp3", mode="compact")

r['bpm']                    # Tempo (BPM)
r['beats']                  # Beat frame positions
r['onset_frames']           # Onset positions
r['onset_density']          # Onsets per second
r['rms_mean']               # Average loudness (RMS)
r['rms_max']                # Peak loudness (RMS)
r['loudness_lufs']          # Integrated loudness (LUFS, ITU-R BS.1770-4)
r['dynamic_range_db']       # Loudness range (p95 - p5, dB)
r['spectral_centroid_mean'] # Brightness (Hz)
r['zero_crossing_rate']     # Percussiveness proxy
r['duration_sec']           # Track length
```

### Playlist mode

Everything for playlist generation: spectral features, MFCCs (timbre fingerprint), chroma (harmony), tonal analysis (chords, dissonance), plus perceptual features:

```python
r = sonara.analyze_file("track.mp3", mode="playlist")

# Perceptual features (0.0 - 1.0)
r['energy']           # Perceived intensity (loudness + brightness + activity)
r['danceability']     # Beat regularity + tempo sweet spot + rhythm
r['valence']          # Mood (0 = sad/dark, 1 = happy/bright)
r['acousticness']     # Acoustic vs electronic character

# Musical key
r['key']              # e.g. "C major", "A minor"
r['key_confidence']   # How confident the key detection is (0.0 - 1.0)

# Tonal analysis
r['chord_sequence']        # Beat-synchronous chord labels, e.g. ["Am", "F", "C", "G"]
r['predominant_chord']     # Most frequent chord
r['chord_change_rate']     # Chord changes per second (harmonic complexity)
r['dissonance']            # Sensory dissonance (0 = consonant, 1 = rough)

# Spectral features
r['spectral_bandwidth_mean']   # Frequency spread
r['spectral_rolloff_mean']     # Frequency below which 85% of energy sits
r['spectral_flatness_mean']    # Tonal (0) vs noise-like (1)
r['spectral_contrast_mean']    # Peak-valley ratio per band (7 values)
r['mfcc_mean']                 # Timbre fingerprint (13 coefficients)
r['chroma_mean']               # Pitch class distribution (12 values)
```

### Full mode

Adds expensive rhythm analysis features on top of playlist mode:

```python
r = sonara.analyze_file("track.mp3", mode="full")

r['tempo_curve']                # Per-beat BPM values
r['tempo_variability']          # Coefficient of variation of tempo
r['time_signature']             # e.g. "4/4", "3/4"
r['time_signature_confidence']  # Detection confidence
```

### Custom feature selection

Cherry-pick specific features regardless of mode:

```python
r = sonara.analyze_file("track.mp3", features=["bpm", "energy", "key", "chords"])
```

Valid feature names: `bpm`, `beats`, `onsets`, `rms`, `dynamic_range`, `centroid`, `zcr`, `onset_density`, `bandwidth`, `rolloff`, `flatness`, `contrast`, `mfcc`, `chroma`, `chords`, `dissonance`, `energy`, `danceability`, `key`, `valence`, `acousticness`, `tempo_curve`, `time_signature`

### Batch analysis

Analyze entire music libraries in parallel using all CPU cores:

```python
import sonara
from pathlib import Path

files = [str(p) for p in Path("~/Music").rglob("*.mp3")]
results = sonara.analyze_batch(files, mode="playlist")

for r in results:
    print(f"{r['bpm']:5.0f} BPM | {r['energy']:.2f} energy | "
          f"{r['key']:>10} | {r['predominant_chord']:>4} | "
          f"{r['dissonance']:.3f} diss | {r['valence']:.2f} valence")
```

## Tonal Analysis

Standalone tonal functions for detailed harmonic analysis:

```python
import sonara
import numpy as np

y, sr = sonara.load("track.mp3", sr=22050)
S = sonara.stft(y, n_fft=2048, hop_length=512)
power = np.abs(S) ** 2
freqs = sonara.fft_frequencies(sr=float(sr), n_fft=2048)

# HPCP — Harmonic Pitch Class Profile (Gomez 2006)
# More robust than energy-based chroma: uses spectral peaks + harmonic weighting
hpcp = sonara.hpcp(power, freqs)  # shape (12, n_frames)

# Chord detection from HPCP + beats
tempo, beats = sonara.beat_track(y=y, sr=sr)
chords = sonara.chords_from_beats(hpcp, list(beats))  # ["Am", "F", "C", "G", ...]
desc = sonara.chord_descriptors(chords, len(y) / sr)
print(f"Predominant: {desc['predominant_chord']}, "
      f"Changes: {desc['chord_change_rate']:.2f}/s, "
      f"Unique: {desc['n_unique']}")

# Dissonance — Sethares (1998) Plomp-Levelt model
diss = sonara.dissonance(power, freqs)  # mean dissonance (0-1)

# Or from specific peaks
d = sonara.dissonance_from_peaks([440.0, 466.16], [1.0, 1.0])  # minor 2nd
```

## Display

```python
import sonara
import sonara.display as display
import matplotlib.pyplot as plt

y, sr = sonara.load("track.mp3", sr=22050)
mel = sonara.melspectrogram(y=y, sr=22050.0)
mel_db = sonara.power_to_db(mel)

fig, ax = plt.subplots()
display.specshow(mel_db, x_axis='time', y_axis='mel', sr=22050, ax=ax)
plt.show()
```

## Performance

All arithmetic uses f32 precision (matching native decoder format), with a parallelized fused FFT pipeline where all features (spectral, tonal, contrast) are computed in a single pass per frame — eliminating redundant FFT computation and keeping data in L1 cache.

### Analysis pipeline benchmarks (Apple Silicon)

| Mode | 10s track | 3-min track | Features |
|------|-----------|-------------|----------|
| `compact` | ~1.2 ms | ~39 ms | 11 core features |
| `playlist` | ~4 ms | ~80 ms | 30+ features |
| `full` | ~50 ms | ~510 ms | All features incl. time signature |

### Feature benchmarks (vs Python/librosa)

| Feature | Speedup |
|---------|---------|
| Mel spectrogram | ~3x |
| MFCC | ~3x |
| Beat tracking | ~4x |
| Onset detection | ~3x |
| Cold start (first call) | ~20-30x |
| **Batch analysis (parallel)** | **~5x** |

### Key optimizations

- **Fused single-pass pipeline** — one FFT per frame simultaneously produces mel, chroma, centroid, RMS, bandwidth, rolloff, flatness, spectral contrast, HPCP, and dissonance. No power spectrum matrix stored.
- **Pre-computed DCT matrix** — MFCCs use cached DCT-II coefficients (matrix multiply instead of per-element cos())
- **Sparse filterbanks** — both mel and chroma filterbanks skip zero entries (~97% sparsity for mel)
- **Partial sort for contrast** — uses O(n) selection instead of O(n log n) sort for percentile computation
- **Top-N peak detection** — spectral peaks sorted by magnitude for HPCP/dissonance, shared between both algorithms
- **f32 precision** — halves memory bandwidth vs f64, matches Symphonia's native decode format
- **Parallel FFT frames** — rayon parallelism across frames (for signals > 32 frames)
- **Fast 2:1 decimation** — half-band FIR filter for 44100-to-22050 Hz instead of full sinc resampling
- **Thread-local caches** — FFT plans, mel/chroma filterbanks, DCT matrix reused across calls

## API Reference

sonara provides 100+ audio analysis functions:

**Core Audio:** `load`, `stream`, `stft`, `istft`, `resample`, `to_mono`, `tone`, `chirp`, `clicks`, `autocorrelate`, `lpc`, `zero_crossings`, `mu_compress`, `mu_expand`

**Spectral Features:** `melspectrogram`, `mfcc`, `chroma_stft`, `tonnetz`, `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_contrast`, `rms`, `zero_crossing_rate`, `poly_features`

**Tonal Analysis:** `hpcp`, `chords_from_beats`, `chords_from_frames`, `chord_descriptors`, `dissonance`, `dissonance_from_peaks`

**Rhythm:** `beat_track`, `onset_detect`, `onset_strength`, `onset_strength_multi`, `tempo`, `tempo_curve`, `tempo_variability`, `tempogram`, `fourier_tempogram`, `metrogram`, `detect_time_signature`, `plp`

**Pitch:** `yin`, `pyin`, `piptrack`, `estimate_tuning`, `pitch_tuning`, `salience`, `interp_harmonics`, `f0_harmonics`

**Transforms:** `cqt`, `vqt`, `icqt`, `hybrid_cqt`, `pseudo_cqt`, `griffinlim`, `griffinlim_cqt`, `phase_vocoder`, `iirt`, `reassigned_spectrogram`, `pcen`, `perceptual_weighting`

**Source Separation:** `hpss`, `harmonic`, `percussive`, `nn_filter`, `decompose_nmf`

**Effects:** `time_stretch`, `pitch_shift`, `trim`, `split`, `split_with_constraints`, `remix`, `melody_separate`, `preemphasis`, `deemphasis`

**Sequence Analysis:** `dtw`, `rqa`, `viterbi`, `viterbi_discriminative`, `viterbi_binary`, `recurrence_matrix`, `cross_similarity`, `path_enhance`

**Perceptual:** `loudness_lufs`, `energy`, `danceability`, `detect_key`, `valence`, `acousticness`

**Conversions (50+):** `hz_to_mel`, `mel_to_hz`, `hz_to_midi`, `midi_to_hz`, `note_to_hz`, `note_to_midi`, `hz_to_note`, `hz_to_octs`, `hz_to_svara_h`, `hz_to_svara_c`, `hz_to_fjs`, `fft_frequencies`, `mel_frequencies`, `cqt_frequencies`, `frames_to_time`, `time_to_frames`, frequency weighting (A/B/C/D/Z), notation helpers, and more

**Filters & DSP:** `mel` filterbank, `chroma` filterbank, `lfilter`, `filtfilt`, `sosfiltfilt`, window functions (Hann, Hamming, Blackman, Kaiser, Tukey, Gaussian)

**Pipeline:** `analyze_file`, `analyze_signal`, `analyze_batch`

## Architecture

sonara is a two-crate Rust workspace:

- **`sonara`** — Pure Rust core library (~18,000 LOC)
- **`sonara-python`** — PyO3 bindings (~1,200 LOC)

```text
sonara/src/
  analyze.rs      — Fused analysis pipeline (compact/playlist/full modes)
  perceptual.rs   — LUFS, energy, danceability, key detection, valence, acousticness
  tonal.rs        — HPCP, chord detection, dissonance (Sethares 1998)
  beat.rs         — Beat tracking (Ellis 2007 DP algorithm)
  onset.rs        — Onset detection (spectral flux + peak picking)
  decompose.rs    — HPSS, NMF
  effects.rs      — Time stretch, pitch shift, trim, split
  segment.rs      — Recurrence matrix, cross-similarity, path enhancement
  sequence.rs     — DTW, RQA, Viterbi, transition matrices
  core/
    audio.rs      — Audio I/O, resampling, fast 2:1 decimation
    spectrum.rs   — STFT, CQT/VQT, phase vocoder, Griffin-Lim
    fft.rs        — FFT with thread-local plan caching
    pitch.rs      — YIN / pYIN pitch estimation
    harmonic.rs   — Harmonic salience, interpolation
    convert.rs    — Hz/mel/MIDI/note/SVara/FJS conversions, frequency weighting
  feature/
    spectral.rs   — Mel, MFCC, chroma, centroid, bandwidth, rolloff, flatness, contrast
    rhythm.rs     — Tempogram, metrogram, time signature detection
  dsp/
    windows.rs    — Window functions (Hann, Hamming, Blackman, Kaiser, Tukey, Gaussian)
    iir.rs        — IIR filters (lfilter, filtfilt, sosfiltfilt)
    extrema.rs    — Local maxima/minima detection
  filters.rs      — Mel/chroma filterbanks
```

## License

[MIT](LICENSE)
