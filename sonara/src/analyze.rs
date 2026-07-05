//! Fused audio analysis pipeline.
//!
//! Computes all common audio features in a single optimized pass,
//! eliminating redundant STFT computation.
//!
//! ## Modes
//!
//! - **Compact** (default): core signal features — tempo, beats, onsets, RMS,
//!   centroid, ZCR, dynamic range. ~0.6ms per 10s track.
//! - **Playlist**: everything needed for playlist generation — adds spectral
//!   features, MFCCs, chroma, plus perceptual features (energy, danceability,
//!   key, valence, acousticness). ~3ms per 10s track.
//! - **Full**: same features as Playlist (currently identical, reserved for
//!   future additions like per-frame arrays or segment-level analysis).
//!
//! ## Algorithms
//!
//! All features use their most accurate algorithms by default:
//! - **Chroma**: proper chroma filterbank (sparse projection)
//! - **Spectral contrast**: log-spaced frequency bands with partial sort
//! - **HPCP / chords**: spectral peak detection with harmonic weighting (Gomez 2006)
//! - **Dissonance**: Sethares (1998) Plomp-Levelt model
//! - All features are fused into a single FFT pass to minimize cache pressure.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;

use ndarray::{s, Array1, Array2};
use rayon::prelude::*;

use crate::core::{audio, convert, fft, spectrum};
use crate::dsp::windows;
use crate::error::{SonaraError, Result};
use crate::filters;
use crate::perceptual;
use crate::types::*;
use crate::util::utils;

/// Minimum number of frames to justify rayon thread overhead.
const PARALLEL_THRESHOLD: usize = 32;

// ============================================================
// Analysis mode & feature selection
// ============================================================

/// Analysis depth — controls which features are computed.
///
/// Use `AnalysisMode::Compact` for fast scanning, `Playlist` for music discovery
/// and playlist generation, or `Full` for comprehensive analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMode {
    /// Core signal features only: tempo, beats, onsets, RMS, centroid, ZCR,
    /// dynamic range. (~0.6ms per 10s track)
    Compact,
    /// All features for playlist generation: adds spectral bandwidth/rolloff/
    /// flatness/contrast, MFCCs, chroma, plus perceptual features (energy,
    /// danceability, key, valence, acousticness). (~3ms per 10s track)
    Playlist,
    /// All available features including expensive rhythm analysis
    /// (tempo_curve, time_signature via metrogram).
    Full,
}

impl AnalysisMode {
    /// Parse mode from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "compact" => Some(Self::Compact),
            "playlist" => Some(Self::Playlist),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
}

impl Default for AnalysisMode {
    fn default() -> Self {
        Self::Compact
    }
}

/// Configuration for a single analysis run.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Analysis depth — which feature groups to compute.
    pub mode: AnalysisMode,
    /// Optional: override which features to include, regardless of mode.
    /// When `Some`, only the listed features are computed.
    /// Valid feature names (case-insensitive):
    ///
    /// **Core signal:**
    /// `bpm`, `beats`, `onsets`, `rms`, `dynamic_range`, `centroid`, `zcr`, `onset_density`
    ///
    /// **Spectral:**
    /// `bandwidth`, `rolloff`, `flatness`, `contrast`, `mfcc`, `chroma`
    ///
    /// **Tonal:**
    /// `chords`, `dissonance`
    ///
    /// **Perceptual:**
    /// `energy`, `danceability`, `key`, `valence`, `acousticness`
    ///
    /// **Beat grid (opt-in only — never on by mode):**
    /// `beatgrid` (populates `grid_offset_sec`, `downbeats`, `grid_stability`)
    /// **Opt-in only (never enabled by any mode; request explicitly):**
    /// `silence` (leading/trailing silence offsets), `key_candidates`
    /// (top-3 keys), `vocalness` (vocal-presence heuristic)
    /// **Similarity (opt-in):**
    /// `embedding` — hand-crafted similarity vector; automatically pulls in the
    /// features it is assembled from. Never produced by a bare mode.
    ///
    /// Note: `duration` is always included. Some features depend on others
    /// (e.g., `key` requires `chroma`, `valence` requires `key`); dependencies
    /// are resolved automatically.
    pub features: Option<HashSet<String>>,
    /// Optional lower bound for octave-folding tempo normalization.
    ///
    /// When both `bpm_min` and `bpm_max` are set, BPM values outside the range
    /// are doubled or halved by octaves until they fit the requested range.
    pub bpm_min: Option<Float>,
    /// Optional upper bound for octave-folding tempo normalization.
    pub bpm_max: Option<Float>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            mode: AnalysisMode::Compact,
            features: None,
            bpm_min: None,
            bpm_max: None,
        }
    }
}

impl AnalysisConfig {
    /// Check if a feature should be computed.
    fn wants(&self, name: &str) -> bool {
        if let Some(ref features) = self.features {
            // Explicit feature list — check if requested
            if features.contains(name) {
                return true;
            }
            // Requesting "embedding" implies the features it is assembled from.
            features.contains("embedding") && EMBEDDING_DEPS.contains(&name)
        } else {
            // Opt-in-only features are never enabled by a mode's defaults —
            // not even Full — only by an explicit `features=[...]` request
            // (performance-first policy).
            if OPT_IN_ONLY_FEATURES.contains(&name) {
                return false;
            }
            // Mode-based defaults
            match self.mode {
                AnalysisMode::Compact => false,
                // Expensive rhythm analysis features are Full-only
                // (metrogram is O(n³) and costs ~445ms for a 3-min track).
                AnalysisMode::Playlist => !matches!(name, "tempo_curve" | "time_signature"),
                AnalysisMode::Full => true,
            }
        }
    }

    /// Check if extended features (anything beyond compact) are needed.
    fn needs_extended(&self) -> bool {
        if let Some(ref features) = self.features {
            // If any non-core feature is requested
            const EXTENDED_FEATURES: &[&str] = &[
                "bandwidth", "rolloff", "flatness", "contrast", "mfcc", "chroma",
                "chords", "dissonance",
                "energy", "danceability", "key", "valence", "acousticness",
                // --- structure ---
                "structure",
                // key_candidates needs the chroma filterbank (extended pass).
                // silence + vocalness derive from always-computed RMS / mel data
                // and do NOT require the extended pass.
                "key_candidates",
                // --- similarity ---: embedding needs the playlist-level features
                "embedding",
            ];
            EXTENDED_FEATURES.iter().any(|&f| features.contains(f))
        } else {
            self.mode != AnalysisMode::Compact
        }
    }

}

/// Feature names that are only ever computed when explicitly requested via
/// `features=[...]`, never enabled by an analysis mode's defaults — not even
/// Full. Every new analysis feature belongs here unless it is provably free
/// (performance-first policy).
const OPT_IN_ONLY_FEATURES: &[&str] = &[
    "loudness",
    "beatgrid",
    "structure",
    "embedding",
    "fingerprint",
    "silence",
    "key_candidates",
    "vocalness",
];
// --- similarity ---
/// Feature names the similarity embedding is assembled from. Requesting
/// "embedding" implies each of these (see `AnalysisConfig::wants`). The spectral
/// timbre features (mfcc/chroma/contrast/bandwidth/rolloff/flatness) are computed
/// automatically whenever extended analysis runs, so only the wants-gated tonal
/// and perceptual features need to be listed here.
const EMBEDDING_DEPS: &[&str] = &[
    "energy", "danceability", "key", "valence", "dissonance", "chords",
];

/// Number of spectral contrast bands.
const N_CONTRAST_BANDS: usize = 6;
/// Number of MFCC coefficients.
const N_MFCC: usize = 13;
/// Number of HPCP harmonics.
const N_HPCP_HARMONICS: usize = 4;
/// Max spectral peaks for HPCP/dissonance.
const MAX_PEAKS: usize = 50;

/// Cached mel filterbank, sparse chroma, DCT matrix, and analysis constants.
struct AnalysisCache {
    key: (u32, usize, usize), // (sr, n_fft, n_mels)
    sparse_mel: Vec<(usize, Vec<Float>)>,
    sparse_chroma: Vec<(usize, Vec<Float>)>,
    freqs: Array1<Float>,
    win_padded: Array1<Float>,
    /// Pre-computed DCT-II matrix (n_mfcc, n_mels) for fast MFCC.
    dct_matrix: Array2<Float>,
    /// Spectral contrast band boundaries (bin indices).
    contrast_bands: Vec<(usize, usize)>,
    /// HPCP harmonic weights: 1/(h+1).
    harmonic_weights: [Float; N_HPCP_HARMONICS],
}

thread_local! {
    static ANALYSIS_CACHE: RefCell<Option<AnalysisCache>> = const { RefCell::new(None) };
}

/// Complete analysis result for a single track.
///
/// Core fields are always populated. Extended/perceptual fields are `Some`
/// only when the selected mode or feature list includes them.
pub struct TrackAnalysis {
    // -- Basic (always computed) --
    pub duration_sec: Float,
    pub bpm: Float,
    /// Selected tempo before optional `bpm_min`/`bpm_max` range alignment.
    pub bpm_raw: Float,
    /// Strongest tempo candidates as `(bpm, score)` pairs, sorted by score
    /// descending (up to the top 5).
    pub bpm_candidates: Vec<(Float, Float)>,
    pub beats: Vec<usize>,
    pub onset_frames: Vec<usize>,
    pub rms_mean: Float,
    pub rms_max: Float,
    pub loudness_lufs: Float,
    pub dynamic_range_db: Float,

    // --- loudness ---
    // Extended loudness / gain metrics (opt-in via `features=["loudness"]`).
    // `Some` only when the "loudness" group was requested; `None` otherwise.
    /// True peak in dBTP (4x oversampled, ITU-R BS.1770-4 Annex 2). ~0 dBTP is
    /// full scale; > 0 dBTP means inter-sample overs that can clip on playback.
    pub true_peak_db: Option<Float>,
    /// ReplayGain-style track gain in dB to reach the -18 LUFS reference:
    /// `-18 - loudness_lufs`.
    pub replaygain_db: Option<Float>,
    /// Short-term loudness curve: one LUFS value per 3 s window at a 1 s hop
    /// (ITU-R BS.1770 short-term integration). Empty for tracks under one window.
    pub loudness_curve: Option<Vec<Float>>,
    /// Maximum momentary (400 ms window) loudness, dB (EBU R128 momentary).
    pub loudness_momentary_max_db: Option<Float>,
    /// EBU R128 loudness range (LRA) in LU: gated 95th-10th percentile spread of
    /// the short-term loudness distribution. The standardized counterpart to the
    /// approximate `dynamic_range_db` (which is a raw p95-p5 of RMS).
    pub loudness_range_lu: Option<Float>,
    // --- end loudness ---

    pub spectral_centroid_mean: Float,
    pub zero_crossing_rate: Float,
    pub onset_density: Float,

    // -- Extended (extended or full) --
    pub spectral_bandwidth_mean: Option<Float>,
    pub spectral_rolloff_mean: Option<Float>,
    pub spectral_flatness_mean: Option<Float>,
    pub spectral_contrast_mean: Option<Vec<Float>>,
    pub mfcc_mean: Option<Vec<Float>>,
    pub chroma_mean: Option<Vec<Float>>,

    // -- Rhythm (extended or full) --
    pub tempo_curve: Option<Vec<Float>>,
    pub tempo_variability: Option<Float>,
    pub time_signature: Option<String>,
    pub time_signature_confidence: Option<Float>,

    // -- Tonal (extended or full) --
    pub chord_sequence: Option<Vec<String>>,
    pub chord_change_rate: Option<Float>,
    pub predominant_chord: Option<String>,
    pub dissonance: Option<Float>,

    // -- Perceptual (extended or full) --
    pub energy: Option<Float>,
    pub danceability: Option<Float>,
    pub key: Option<String>,
    pub key_confidence: Option<Float>,
    /// Camelot wheel code for the detected key (e.g. "8A" for A minor), for DJ harmonic mixing.
    pub key_camelot: Option<String>,
    pub valence: Option<Float>,
    pub acousticness: Option<Float>,

    // -- Embedding (future ML models) --
    /// Learned audio embedding vector (future ONNX integration).
    pub embedding: Option<Vec<Float>>,

    // -- Tier 3 placeholders (future ML models) --
    /// Requires ML model (future).
    pub mood_happy: Option<Float>,
    pub mood_aggressive: Option<Float>,
    pub mood_relaxed: Option<Float>,
    pub mood_sad: Option<Float>,
    pub instrumentalness: Option<Float>,
    pub genre: Option<String>,

    // --- beat grid ---
    // Opt-in only (request via features=["beatgrid"]); `None` in the default
    // compact/playlist/full modes.
    /// Time (seconds) of the first beat — the grid anchor.
    pub grid_offset_sec: Option<Float>,
    /// Frame indices of bar-starting beats (subset of `beats`).
    pub downbeats: Option<Vec<usize>>,
    /// How rigidly beats fit a constant-tempo grid, in `[0, 1]`.
    pub grid_stability: Option<Float>,
    // --- structure ---
    /// Time-resolved perceptual energy (0-1), one value per window.
    pub energy_curve: Option<Vec<Float>>,
    /// Seconds between successive `energy_curve` samples.
    pub energy_curve_hop_sec: Option<Float>,
    /// Structural sections as `(start_sec, end_sec, mean_energy)`.
    pub segments: Option<Vec<(Float, Float, Float)>>,
    /// End of the initial low-energy / pre-first-drop region (seconds).
    pub intro_end_sec: Option<Float>,
    /// Start of the final fade / low-energy region (seconds).
    pub outro_start_sec: Option<Float>,
    /// Coarse 1-10 energy level derived from mean energy.
    pub energy_level: Option<u8>,
    // --- silence ---
    /// Leading silence duration in seconds — audio below the silence threshold
    /// (-60 dBFS relative to full scale) at the very start. Opt-in via
    /// `features=["silence"]`. `None` unless requested.
    pub leading_silence_sec: Option<Float>,
    /// Trailing silence duration in seconds — audio below the silence threshold
    /// at the very end. Opt-in via `features=["silence"]`. `None` unless requested.
    pub trailing_silence_sec: Option<Float>,

    // --- key candidates ---
    /// Top-3 ranked key candidates as `(key string, Camelot code, score)`.
    /// Opt-in via `features=["key_candidates"]`. The first entry equals `key`.
    pub key_candidates: Option<Vec<(String, String, Float)>>,

    // --- vocalness ---
    /// Vocal-presence heuristic in `[0, 1]` (rough indicator, not a classifier).
    /// Opt-in via `features=["vocalness"]`. `None` unless requested.
    pub vocalness: Option<Float>,
    // --- fingerprint ---
    /// Acoustic fingerprint (raw sub-fingerprint sequence, ~8 `u32`/sec) for
    /// duplicate detection. `Some` only when the `"fingerprint"` feature is
    /// explicitly requested; `None` in every mode by default. See
    /// [`crate::fingerprint`]. Compare two with [`crate::fingerprint::match_score`].
    pub fingerprint: Option<Vec<u32>>,
    // --- similarity ---
    /// Version of the `embedding` layout + normalization (see
    /// `crate::similarity::SIMILARITY_VERSION`). `Some` iff `embedding` is `Some`.
    /// Present only when the `"embedding"` feature is explicitly requested.
    pub embedding_version: Option<u32>,
}

/// Per-frame results from the fused FFT pass.
struct FrameResult {
    mel_col: Vec<Float>,
    centroid: Float,
    rms: Float,
    bandwidth: Float,
    rolloff: Float,
    flatness: Float,
    chroma_col: [Float; 12],
    // Fused tonal + contrast (only populated when extended)
    contrast_bands: [Float; N_CONTRAST_BANDS + 1],
    hpcp_col: [Float; 12],
    dissonance: Float,
}

// ============================================================
// Public API
// ============================================================

/// Analyze a track from a file path with the given configuration.
pub fn analyze_file(path: &Path, sr: u32, config: &AnalysisConfig) -> Result<TrackAnalysis> {
    let (y, actual_sr) = audio::load(path, sr, true, 0.0, 0.0)?;
    analyze_signal(y.view(), actual_sr, config)
}

/// Analyze a pre-loaded audio signal with the given configuration.
pub fn analyze_signal(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    let extended = config.needs_extended();
    analyze_signal_inner(y, sr, extended, config)
}

/// Analyze multiple files in parallel.
///
/// Failures are isolated per file: the returned vector has exactly one entry
/// per input path, in the same order as `paths`, and a decode/IO failure on one
/// file yields an `Err` for that entry only — it never aborts or poisons the
/// rest of the batch. This is the robustness contract the Python `analyze_batch`
/// binding relies on when analyzing large libraries.
pub fn analyze_batch(paths: &[&Path], sr: u32, config: &AnalysisConfig) -> Vec<Result<TrackAnalysis>> {
    paths.par_iter().map(|path| analyze_file(path, sr, config)).collect()
}

// ============================================================
// Core implementation
// ============================================================

fn analyze_signal_inner(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    extended: bool,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    let sr_f = sr as Float;
    let n_fft = 2048;
    let hop_length = 512;
    let n_mels = 128;
    let n_bins = n_fft / 2 + 1;

    let duration_sec = y.len() as Float / sr_f;

    // ================================================================
    // SETUP: mel filterbank, window, padding (cached across calls)
    // ================================================================

    let cache_key = (sr, n_fft, n_mels);

    let cache_data = ANALYSIS_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(ref c) = *cache {
            if c.key == cache_key {
                return (
                    c.sparse_mel.clone(), c.sparse_chroma.clone(), c.freqs.clone(),
                    c.win_padded.clone(), c.dct_matrix.clone(),
                    c.contrast_bands.clone(), c.harmonic_weights,
                );
            }
        }

        let mel_fb = filters::mel(sr_f, n_fft, n_mels, 0.0, sr_f / 2.0, false, "slaney");
        let sparse_mel: Vec<(usize, Vec<Float>)> = (0..n_mels)
            .map(|m| {
                let row = mel_fb.row(m);
                let first = row.iter().position(|&v| v > 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&v| v > 0.0).unwrap_or(0);
                if first > last { (0, vec![]) }
                else { (first, row.slice(s![first..=last]).to_vec()) }
            })
            .collect();
        let f = convert::fft_frequencies(sr_f, n_fft);
        let win = windows::get_window(&WindowSpec::Named("hann".into()), n_fft, true)
            .expect("hann window");
        let wp = utils::pad_center(win.view(), n_fft).expect("pad_center");

        // Sparse chroma filterbank
        let cfb = filters::chroma(sr_f, n_fft, 12, 0.0);
        let sparse_chroma: Vec<(usize, Vec<Float>)> = (0..12)
            .map(|c| {
                let row = cfb.row(c);
                let first = row.iter().position(|&v| v > 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&v| v > 0.0).unwrap_or(0);
                if first > last { (0, vec![]) }
                else { (first, row.slice(s![first..=last]).to_vec()) }
            })
            .collect();

        // Pre-computed DCT-II matrix (n_mfcc × n_mels)
        let dct_matrix = Array2::from_shape_fn((N_MFCC, n_mels), |(k, m)| {
            let norm = if k == 0 { (1.0 / n_mels as Float).sqrt() } else { (2.0 / n_mels as Float).sqrt() };
            norm * (std::f32::consts::PI * k as Float * (2 * m + 1) as Float / (2.0 * n_mels as Float)).cos()
        });

        // Spectral contrast band bin boundaries
        let fmin: Float = 200.0;
        let fmax = sr_f / 2.0;
        let freqs_slice = f.as_slice().unwrap();
        let mut band_edges = vec![fmin];
        for i in 1..=N_CONTRAST_BANDS {
            band_edges.push(fmin * (fmax / fmin).powf(i as Float / N_CONTRAST_BANDS as Float));
        }
        let contrast_bands: Vec<(usize, usize)> = (0..N_CONTRAST_BANDS)
            .map(|b| {
                let lo = band_edges[b];
                let hi = band_edges[b + 1];
                let start = freqs_slice.iter().position(|&freq| freq >= lo).unwrap_or(0);
                let end = freqs_slice.iter().position(|&freq| freq >= hi).unwrap_or(n_bins);
                (start, end)
            })
            .collect();

        let harmonic_weights: [Float; N_HPCP_HARMONICS] = std::array::from_fn(|h| 1.0 / (h as Float + 1.0));

        *cache = Some(AnalysisCache {
            key: cache_key,
            sparse_mel: sparse_mel.clone(),
            sparse_chroma: sparse_chroma.clone(),
            freqs: f.clone(),
            win_padded: wp.clone(),
            dct_matrix: dct_matrix.clone(),
            contrast_bands: contrast_bands.clone(),
            harmonic_weights,
        });

        (sparse_mel, sparse_chroma, f, wp, dct_matrix, contrast_bands, harmonic_weights)
    });
    let (sparse_mel, sparse_chroma, freqs, win_padded, dct_matrix, contrast_bands, harmonic_weights) = cache_data;

    let pad = n_fft / 2;
    let mut y_padded = Array1::<Float>::zeros(y.len() + 2 * pad);
    y_padded.slice_mut(s![pad..pad + y.len()]).assign(&y);
    let n = y_padded.len();
    if n < n_fft {
        return Err(SonaraError::InsufficientData { needed: n_fft, got: n });
    }
    let y_raw = y_padded.as_slice().unwrap();
    let win_raw = win_padded.as_slice().unwrap();

    // ================================================================
    // SINGLE PASS: FFT → mel + centroid + rms + (extended features)
    // Also computes chroma via proper filterbank and stores power spectrum
    // All extended features are fused into this single FFT pass.
    // ================================================================

    let n_frames = 1 + (n - n_fft) / hop_length;
    let mut mel_spec = Array2::<Float>::zeros((n_mels, n_frames));
    let mut centroids = Array1::<Float>::zeros(n_frames);
    let mut rms_frames = Array1::<Float>::zeros(n_frames);
    let mut bandwidths = if extended { Array1::<Float>::zeros(n_frames) } else { Array1::zeros(0) };
    let mut rolloffs = if extended { Array1::<Float>::zeros(n_frames) } else { Array1::zeros(0) };
    let mut flatnesses = if extended { Array1::<Float>::zeros(n_frames) } else { Array1::zeros(0) };
    let mut chroma_raw = if extended { Array2::<Float>::zeros((12, n_frames)) } else { Array2::zeros((0, 0)) };
    // Fused HPCP (accumulated per-frame, normalized+averaged post-loop)
    let mut hpcp_raw = if extended { Array2::<Float>::zeros((12, n_frames)) } else { Array2::zeros((0, 0)) };
    // Fused contrast + dissonance accumulators
    let contrast_acc;
    let dissonance_acc;

    let freqs_raw = freqs.as_slice().unwrap();
    let roll_percent: Float = 0.85;
    let contrast_quantile: Float = 0.02;
    let c_ref: Float = 261.6256; // C4 reference for HPCP

    // Sethares (1998) dissonance model constants
    let diss_b1: Float = 3.5144;
    let diss_b2: Float = 5.7564;
    let diss_d_max: Float = 0.24;
    let diss_s1: Float = 0.0207;
    let diss_s2: Float = 18.96;

    let compute_frame = |t: usize| -> FrameResult {
        let start = t * hop_length;
        let mut fft_in = vec![0.0_f32; n_fft];
        for i in 0..n_fft { fft_in[i] = y_raw[start + i] * win_raw[i]; }
        let mut fft_out = vec![num_complex::Complex::new(0.0, 0.0); n_bins];
        fft::rfft(&mut fft_in, &mut fft_out).expect("FFT failed");

        // Compute power and magnitude ONCE
        let mut power_col = vec![0.0_f32; n_bins];
        let mut mag_col = vec![0.0_f32; n_bins];
        let mut cent_num = 0.0_f32;
        let mut cent_den = 0.0_f32;

        for i in 0..n_bins {
            let pwr = fft_out[i].norm_sqr();
            let mag = pwr.sqrt();
            power_col[i] = pwr;
            mag_col[i] = mag;
            cent_num += freqs_raw[i] * mag;
            cent_den += mag;
        }

        let centroid = if cent_den > 0.0 { cent_num / cent_den } else { 0.0 };

        // RMS from time-domain
        let mut sum_sq = 0.0_f32;
        for i in 0..n_fft { sum_sq += y_raw[start + i] * y_raw[start + i]; }
        let rms = (sum_sq / n_fft as Float).sqrt();

        let (bandwidth, rolloff, flatness) = if extended {
            // Bandwidth — reuse mag_col
            let bw = if cent_den > 0.0 {
                let mut bw_num = 0.0_f32;
                for i in 0..n_bins {
                    let dev = freqs_raw[i] - centroid;
                    bw_num += mag_col[i] * dev * dev;
                }
                (bw_num / cent_den).sqrt()
            } else { 0.0 };

            // Rolloff — reuse mag_col
            let threshold = roll_percent * cent_den; // cent_den == sum of mag
            let mut cumsum = 0.0_f32;
            let mut ro = 0.0_f32;
            for i in 0..n_bins {
                cumsum += mag_col[i];
                if cumsum >= threshold {
                    ro = freqs_raw[i];
                    break;
                }
            }

            // Flatness — on power_col directly
            let amin: Float = 1e-10;
            let mut log_sum = 0.0_f32;
            let mut arith_sum = 0.0_f32;
            for i in 0..n_bins {
                let v = power_col[i].max(amin);
                log_sum += v.ln();
                arith_sum += v;
            }
            let geo_mean = (log_sum / n_bins as Float).exp();
            let arith_mean = arith_sum / n_bins as Float;
            let fl = if arith_mean > 0.0 { geo_mean / arith_mean } else { 0.0 };

            (bw, ro, fl)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Sparse mel projection
        let mel_col: Vec<Float> = sparse_mel.iter().map(|(start_bin, weights)| {
            let mut sum = 0.0;
            for (k, &w) in weights.iter().enumerate() { sum += w * power_col[start_bin + k]; }
            sum
        }).collect();

        // Sparse chroma projection
        let mut chroma_col = [0.0_f32; 12];
        if extended {
            for (c, (sb, weights)) in sparse_chroma.iter().enumerate() {
                let mut sum = 0.0;
                for (k, &w) in weights.iter().enumerate() { sum += w * power_col[sb + k]; }
                chroma_col[c] = sum;
            }
        }

        // --- Fused spectral contrast (inline, using mag_col) ---
        let mut contrast_bands_out = [0.0_f32; N_CONTRAST_BANDS + 1];
        if extended {
            for (b, &(sb, eb)) in contrast_bands.iter().enumerate() {
                if sb >= eb { continue; }
                let bn = eb - sb;
                // Collect magnitudes for this band into a small buffer
                let mut band_vals: Vec<Float> = (sb..eb)
                    .map(|f| mag_col[f].max(1e-10))
                    .collect();
                // Partial sort: O(n) instead of O(n log n)
                let q_idx = ((bn as Float * contrast_quantile) as usize).min(bn - 1);
                band_vals.select_nth_unstable_by(q_idx, |a, b| a.partial_cmp(b).unwrap());
                let valley = band_vals[q_idx];
                let peak_idx = (bn - 1).saturating_sub(q_idx);
                band_vals.select_nth_unstable_by(peak_idx, |a, b| a.partial_cmp(b).unwrap());
                let peak = band_vals[peak_idx];
                contrast_bands_out[b] = peak.log10() - valley.log10();
            }
            let mean_mag = cent_den / n_bins as Float; // cent_den = sum of mag
            contrast_bands_out[N_CONTRAST_BANDS] = mean_mag.max(1e-10).log10();
        }

        // --- Fused HPCP + dissonance (inline, sharing peak detection) ---
        let mut hpcp_col = [0.0_f32; 12];
        let mut frame_diss = 0.0_f32;
        if extended {
            // Spectral peak detection with parabolic interpolation.
            // Collect ALL peaks, then keep top MAX_PEAKS by magnitude
            // (matches standalone tonal::detect_spectral_peaks behavior).
            let mut all_peaks_freq = Vec::new();
            let mut all_peaks_mag = Vec::new();

            for i in 1..n_bins - 1 {
                if mag_col[i] <= mag_col[i - 1] || mag_col[i] <= mag_col[i + 1] { continue; }
                if freqs_raw[i] < 40.0 || freqs_raw[i] > 5000.0 { continue; }

                // Parabolic interpolation
                let alpha = mag_col[i - 1];
                let beta = mag_col[i];
                let gamma = mag_col[i + 1];
                let denom = alpha - 2.0 * beta + gamma;
                let (freq, mag) = if denom.abs() > 1e-10 {
                    let p = 0.5 * (alpha - gamma) / denom;
                    let bin_frac = i as Float + p;
                    let f = if bin_frac >= 0.0 && (bin_frac as usize) < n_bins - 1 {
                        let lo = bin_frac as usize;
                        let frac = bin_frac - lo as Float;
                        freqs_raw[lo] * (1.0 - frac) + freqs_raw[lo + 1] * frac
                    } else {
                        freqs_raw[i]
                    };
                    (f, beta - 0.25 * (alpha - gamma) * p)
                } else {
                    (freqs_raw[i], beta)
                };

                all_peaks_freq.push(freq);
                all_peaks_mag.push(mag);
            }

            // Sort by magnitude descending, keep top MAX_PEAKS
            let mut indices: Vec<usize> = (0..all_peaks_freq.len()).collect();
            indices.sort_unstable_by(|&a, &b| all_peaks_mag[b].partial_cmp(&all_peaks_mag[a]).unwrap());
            indices.truncate(MAX_PEAKS);
            let n_peaks = indices.len();

            let peaks_freq: Vec<Float> = indices.iter().map(|&i| all_peaks_freq[i]).collect();
            let peaks_mag: Vec<Float> = indices.iter().map(|&i| all_peaks_mag[i]).collect();

            // HPCP from peaks
            for p in 0..n_peaks {
                let pmag_sq = peaks_mag[p] * peaks_mag[p];
                for h in 0..N_HPCP_HARMONICS {
                    let freq = peaks_freq[p] / (h as Float + 1.0);
                    if freq < 20.0 { continue; }
                    let semitones = 12.0 * (freq / c_ref).log2();
                    let pitch_class = ((semitones % 12.0) + 12.0) % 12.0;
                    let center = pitch_class.round() as usize % 12;
                    let dist = (pitch_class - center as Float).abs();
                    if dist < 0.5 {
                        let w = (std::f32::consts::PI * dist).cos();
                        hpcp_col[center] += harmonic_weights[h] * pmag_sq * w;
                    }
                }
            }

            // Dissonance from same peaks (Sethares 1998)
            if n_peaks >= 2 {
                let mut diss_sum = 0.0_f32;
                let mut weight_sum = 0.0_f32;
                for i in 0..n_peaks {
                    for j in (i + 1)..n_peaks {
                        let f_min = peaks_freq[i].min(peaks_freq[j]);
                        let f_diff = (peaks_freq[i] - peaks_freq[j]).abs();
                        let s = diss_d_max / (diss_s1 * f_min + diss_s2);
                        let d = (-diss_b1 * s * f_diff).exp() - (-diss_b2 * s * f_diff).exp();
                        let d = d.max(0.0);
                        let w = peaks_mag[i] * peaks_mag[j];
                        diss_sum += w * d;
                        weight_sum += w;
                    }
                }
                if weight_sum > 0.0 {
                    frame_diss = (diss_sum / weight_sum).clamp(0.0, 1.0);
                }
            }
        }

        FrameResult {
            mel_col, centroid, rms, bandwidth, rolloff, flatness,
            chroma_col, contrast_bands: contrast_bands_out,
            hpcp_col, dissonance: frame_diss,
        }
    };

    // Scatter frame results into arrays
    let mut scatter_results = |frame_results: Vec<FrameResult>| -> (
        [Float; N_CONTRAST_BANDS + 1], Float,
    ) {
        let mut c_acc = [0.0_f32; N_CONTRAST_BANDS + 1];
        let mut d_acc = 0.0_f32;

        for (t, fr) in frame_results.into_iter().enumerate() {
            centroids[t] = fr.centroid;
            rms_frames[t] = fr.rms;
            if extended {
                bandwidths[t] = fr.bandwidth;
                rolloffs[t] = fr.rolloff;
                flatnesses[t] = fr.flatness;
                for c in 0..12 { chroma_raw[(c, t)] = fr.chroma_col[c]; }
                for b in 0..N_CONTRAST_BANDS + 1 { c_acc[b] += fr.contrast_bands[b]; }
                for c in 0..12 { hpcp_raw[(c, t)] = fr.hpcp_col[c]; }
                d_acc += fr.dissonance;
            }
            for (m, val) in fr.mel_col.into_iter().enumerate() {
                mel_spec[(m, t)] = val;
            }
        }
        (c_acc, d_acc)
    };

    if n_frames >= PARALLEL_THRESHOLD {
        let frame_results: Vec<FrameResult> = (0..n_frames)
            .into_par_iter()
            .map(|t| compute_frame(t))
            .collect();
        let (ca, da) = scatter_results(frame_results);
        contrast_acc = ca;
        dissonance_acc = da;
    } else {
        let frame_results: Vec<FrameResult> = (0..n_frames)
            .map(|t| compute_frame(t))
            .collect();
        let (ca, da) = scatter_results(frame_results);
        contrast_acc = ca;
        dissonance_acc = da;
    }

    // ================================================================
    // ONSET STRENGTH from mel spectrogram (no additional FFT)
    // ================================================================

    let s_db = spectrum::power_to_db(mel_spec.view(), 1.0, 1e-10, Some(80.0));
    let lag = 1usize;

    let out_frames = if n_frames > lag { n_frames - lag } else { 0 };
    let mut onset_env = Array1::<Float>::zeros(out_frames);
    for t in 0..out_frames {
        let mut sum = 0.0;
        for m in 0..n_mels {
            sum += (s_db[(m, t + lag)] - s_db[(m, t)]).max(0.0);
        }
        onset_env[t] = sum / n_mels as Float;
    }

    let pad_left = lag + n_fft / (2 * hop_length);
    let total_oenv_frames = out_frames + pad_left;
    let mut oenv_padded = Array1::<Float>::zeros(total_oenv_frames);
    for t in 0..out_frames { oenv_padded[pad_left + t] = onset_env[t]; }

    // ================================================================
    // BEAT TRACKING + ONSET DETECTION
    // ================================================================

    let (tempo_estimate, beats) = crate::beat::beat_track_detailed(
        None, Some(oenv_padded.view()), sr, hop_length, 120.0, 100.0, true,
        config.bpm_min, config.bpm_max,
    )?;
    let bpm = tempo_estimate.tempo;
    let bpm_raw = tempo_estimate.tempo_raw;
    let bpm_candidates = tempo_estimate.candidates;

    let onset_frames = crate::onset::onset_detect(
        None, Some(oenv_padded.view()), sr, hop_length, false, 0.07, 0,
    )?;

    // ================================================================
    // Zero crossings (trivial, time-domain)
    // ================================================================

    let zc = audio::zero_crossings(y, 0.0);
    let zcr = zc.iter().filter(|&&v| v).count() as Float / y.len() as Float;

    // ================================================================
    // LUFS integrated loudness (ITU-R BS.1770-4, K-weighted)
    // ================================================================

    let loudness_lufs = perceptual::loudness_lufs(y, sr);

    // --- loudness ---
    // Extended loudness / gain metrics — strictly opt-in via `features=["loudness"]`.
    // Default modes (compact/playlist/full) skip this entirely, so they pay nothing.
    let (
        true_peak_db,
        replaygain_db,
        loudness_curve,
        loudness_momentary_max_db,
        loudness_range_lu,
    ) = if config.wants("loudness") {
        let tp = crate::loudness_ext::true_peak_db(y);
        let rg = crate::loudness_ext::replaygain_db(loudness_lufs);
        // Short-term curve: 3 s window, 1 s hop (ITU-R BS.1770 short-term).
        // One K-weighting pass feeds the curve, momentary max and LRA.
        let m = crate::loudness_ext::loudness_metrics(y, sr, 3.0, 1.0);
        (Some(tp), Some(rg), Some(m.curve), Some(m.momentary_max_db), Some(m.range_lu))
    } else {
        (None, None, None, None, None)
    };
    // --- end loudness ---

    // ================================================================
    // EXTENDED: MFCCs via pre-computed DCT matrix (no per-frame cos())
    // ================================================================

    let mfcc_mean = if extended {
        // dct_matrix is (N_MFCC, n_mels), s_db is (n_mels, n_frames)
        // Compute MFCC mean = mean over frames of (dct_matrix @ s_db[:, t])
        let mut mfcc_avg = vec![0.0_f32; N_MFCC];
        let dct_raw = dct_matrix.as_slice().unwrap();
        for t in 0..n_frames {
            for k in 0..N_MFCC {
                let mut sum = 0.0_f32;
                let row_start = k * n_mels;
                for m in 0..n_mels {
                    sum += dct_raw[row_start + m] * s_db[(m, t)];
                }
                mfcc_avg[k] += sum;
            }
        }
        for v in mfcc_avg.iter_mut() { *v /= n_frames.max(1) as Float; }
        Some(mfcc_avg)
    } else {
        None
    };

    // ================================================================
    // CHROMA: proper filterbank (always, computed in the fused loop)
    // L-inf normalize per frame, then average across frames.
    // ================================================================

    let chroma_mean = if extended && n_frames > 0 {
        let mut chroma_avg = vec![0.0_f32; 12];
        for t in 0..n_frames {
            let mut frame_chroma = [0.0_f32; 12];
            for c in 0..12 { frame_chroma[c] = chroma_raw[(c, t)]; }
            // L-inf normalize per frame
            let max_val = frame_chroma.iter().copied().fold(0.0_f32, Float::max);
            if max_val > 0.0 {
                for v in frame_chroma.iter_mut() { *v /= max_val; }
            }
            for (i, &v) in frame_chroma.iter().enumerate() {
                chroma_avg[i] += v;
            }
        }
        for v in chroma_avg.iter_mut() { *v /= n_frames as Float; }
        Some(chroma_avg)
    } else {
        None
    };

    // ================================================================
    // SPECTRAL CONTRAST: aggregated from fused frame loop
    // ================================================================

    let spectral_contrast_mean = if extended && n_frames > 0 {
        let mut contrast_avg = contrast_acc.to_vec();
        for v in contrast_avg.iter_mut() { *v /= n_frames as Float; }
        Some(contrast_avg)
    } else {
        None
    };

    // ================================================================
    // Aggregate results
    // ================================================================

    let rms_mean = rms_frames.iter().sum::<Float>() / rms_frames.len() as Float;
    let rms_max = rms_frames.iter().copied().fold(0.0_f32, Float::max);

    let rms_nonzero: Vec<Float> = rms_frames.iter().copied().filter(|&v| v > 1e-10).collect();
    let dynamic_range_db = if rms_nonzero.len() > 10 {
        let mut sorted = rms_nonzero.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p5 = sorted[sorted.len() * 5 / 100];
        let p95 = sorted[sorted.len() * 95 / 100];
        if p5 > 0.0 { 20.0 * (p95 / p5).log10() } else { 0.0 }
    } else {
        0.0
    };

    let centroid_mean = centroids.iter().sum::<Float>() / centroids.len().max(1) as Float;
    let onset_density = onset_frames.len() as Float / duration_sec;

    let spectral_bandwidth_mean = if extended {
        Some(bandwidths.iter().sum::<Float>() / bandwidths.len().max(1) as Float)
    } else { None };

    let spectral_rolloff_mean = if extended {
        Some(rolloffs.iter().sum::<Float>() / rolloffs.len().max(1) as Float)
    } else { None };

    let spectral_flatness_mean = if extended {
        Some(flatnesses.iter().sum::<Float>() / flatnesses.len().max(1) as Float)
    } else { None };

    // ================================================================
    // RHYTHM: Tempo curve & time signature
    // ================================================================

    let (tempo_curve, tempo_variability) = if extended && config.wants("tempo_curve") {
        let tc = crate::beat::tempo_curve(&beats, sr, hop_length, Some(5))
            .unwrap_or_default();
        let tv = crate::beat::tempo_variability(&tc);
        (Some(tc), Some(tv))
    } else {
        (None, None)
    };

    let (time_signature, time_signature_confidence) = if extended && config.wants("time_signature") {
        let win = 384.min(oenv_padded.len());
        if win >= 4 {
            match crate::feature::rhythm::metrogram(
                None, Some(oenv_padded.view()), sr, hop_length, win, None,
            ) {
                Ok(mg) => {
                    let (label, conf) = crate::feature::rhythm::detect_time_signature(mg.view(), None);
                    (Some(label), Some(conf))
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    // ================================================================
    // BEAT GRID: offset, downbeats, stability (opt-in via features)
    // Reuses the already-computed beats + onset envelope (O(n_beats)),
    // so it never runs in the default modes — only when explicitly
    // requested via features=["beatgrid"].
    // ================================================================

    let (grid_offset_sec, downbeats, grid_stability) = if config.wants("beatgrid") {
        // Prefer the detected meter (full mode) when it was also requested;
        // otherwise assume 4/4.
        let beats_per_bar = time_signature
            .as_deref()
            .and_then(|ts| ts.split('/').next())
            .and_then(|n| n.trim().parse::<usize>().ok())
            .filter(|&n| n >= 2)
            .unwrap_or(crate::beatgrid::DEFAULT_BEATS_PER_BAR);
        let grid = crate::beatgrid::analyze_grid(
            &beats, oenv_padded.view(), sr, hop_length, beats_per_bar,
        );
        (Some(grid.grid_offset_sec), Some(grid.downbeats), Some(grid.grid_stability))
    } else {
        (None, None, None)
    };

    // ================================================================
    // TONAL: chords from fused HPCP, dissonance from fused accumulator
    // ================================================================

    let wants_chords = extended && config.wants("chords");
    let wants_diss = extended && config.wants("dissonance");

    let (chord_sequence, chord_change_rate, predominant_chord, dissonance_val) =
        if (wants_chords || wants_diss) && n_frames > 0 {
            // L1-normalize HPCP per frame (in-place on hpcp_raw)
            for t in 0..n_frames {
                let sum: Float = (0..12).map(|c| hpcp_raw[(c, t)]).sum();
                if sum > 0.0 {
                    for c in 0..12 { hpcp_raw[(c, t)] /= sum; }
                }
            }

            let (cs, ccr, pc) = if wants_chords {
                let chords = crate::tonal::chords_from_beats(hpcp_raw.view(), &beats);
                let desc = crate::tonal::chord_descriptors(&chords, duration_sec);
                (Some(chords), Some(desc.change_rate), Some(desc.predominant_chord))
            } else {
                (None, None, None)
            };

            let dv = if wants_diss {
                Some(dissonance_acc / n_frames as Float)
            } else { None };

            (cs, ccr, pc, dv)
        } else {
            (None, None, None, None)
        };

    // ================================================================
    // PERCEPTUAL FEATURES (from already-computed scalars, ~0 extra cost)
    // ================================================================

    let bw_mean = spectral_bandwidth_mean.unwrap_or(0.0);
    let fl_mean = spectral_flatness_mean.unwrap_or(0.0);
    let ro_mean = spectral_rolloff_mean.unwrap_or(0.0);

    let wants_energy = extended && config.wants("energy");
    let wants_dance = extended && config.wants("danceability");
    let wants_key = extended && config.wants("key");
    let wants_valence = extended && (config.wants("valence") || config.wants("key"));
    let wants_acoustic = extended && config.wants("acousticness");

    let energy = if wants_energy {
        Some(perceptual::energy(rms_mean, centroid_mean, onset_density, bw_mean))
    } else { None };

    let danceability = if wants_dance {
        Some(perceptual::danceability_heuristic(bpm, &beats, onset_density))
    } else { None };

    // Key detection requires chroma (resolved as dependency)
    let key_result = if wants_key || wants_valence {
        chroma_mean.as_ref().map(|c| perceptual::detect_key(c))
    } else { None };

    let valence = if config.wants("valence") {
        key_result.as_ref().map(|kr| perceptual::valence(kr, bpm, centroid_mean))
    } else { None };

    let acousticness = if wants_acoustic {
        Some(perceptual::acousticness(fl_mean, ro_mean, onset_density))
    } else { None };

    // --- fingerprint ---
    // Strictly opt-in (see OPT_IN_ONLY_FEATURES): never runs unless
    // the caller explicitly requested the "fingerprint" feature, so default modes
    // pay exactly zero cost. Operates on its own downsampled mono copy of `y`.
    let fingerprint = if config.wants("fingerprint") {
        let fp = crate::fingerprint::compute(y, sr);
        if fp.is_empty() { None } else { Some(fp) }
    } else {
        None
    };

    let key = key_result.as_ref().map(|kr| perceptual::format_key(kr));
    let key_confidence = key_result.as_ref().map(|kr| kr.confidence);
    let key_camelot = key_result
        .as_ref()
        .and_then(|kr| perceptual::camelot(kr.key, kr.mode))
        .map(|c| c.to_string());

    // ================================================================
    // STRUCTURE (opt-in): energy curve + novelty segmentation
    // Reuses per-frame RMS/centroid/bandwidth and the mel dB spectrogram
    // already computed above — no extra decode or FFT pass.
    // ================================================================
    // --- structure ---
    let structure = if extended && config.wants("structure") && n_frames > 0 {
        let fps = sr_f / hop_length as Float;
        Some(crate::structure::analyze_structure(
            rms_frames.as_slice().unwrap(),
            centroids.as_slice().unwrap(),
            bandwidths.as_slice().unwrap_or(&[]),
            s_db.view(),
            dct_matrix.view(),
            &onset_frames,
            fps,
            duration_sec,
        ))
    } else {
        None
    };

    // ================================================================
    // OPT-IN FEATURES (only computed when explicitly requested via
    // `features=[...]`; never enabled by mode — performance-first policy)
    // ================================================================

    // --- silence ---
    // Nearly free: pure arithmetic over the RMS frames already computed above.
    // Kept opt-in per the performance-first policy so default modes are unchanged.
    let (leading_silence_sec, trailing_silence_sec) = if config.wants("silence") {
        let rms_slice = rms_frames.as_slice().unwrap();
        let (lead, trail) = silence_offsets(rms_slice, sr, hop_length, -60.0);
        (Some(lead), Some(trail))
    } else {
        (None, None)
    };

    // --- key candidates ---
    // Requires chroma (resolved as an extended-pass dependency).
    let key_candidates = if config.wants("key_candidates") {
        chroma_mean.as_ref().map(|c| {
            perceptual::detect_key_candidates(c)
                .into_iter()
                .map(|kc| (kc.key, kc.camelot.to_string(), kc.score))
                .collect::<Vec<_>>()
        })
    } else {
        None
    };

    // --- vocalness ---
    // Derived from the always-computed mel spectrogram (no extra FFT work).
    let vocalness = if config.wants("vocalness") {
        Some(crate::vocal::vocalness(mel_spec.view(), sr, hop_length))
    } else {
        None
    };

    let mut result = TrackAnalysis {
        duration_sec,
        bpm,
        bpm_raw,
        bpm_candidates,
        beats,
        onset_frames,
        rms_mean,
        rms_max,
        loudness_lufs,
        dynamic_range_db,
        // --- loudness ---
        true_peak_db,
        replaygain_db,
        loudness_curve,
        loudness_momentary_max_db,
        loudness_range_lu,
        // --- end loudness ---
        spectral_centroid_mean: centroid_mean,
        zero_crossing_rate: zcr,
        onset_density,
        spectral_bandwidth_mean,
        spectral_rolloff_mean,
        spectral_flatness_mean,
        spectral_contrast_mean,
        mfcc_mean,
        chroma_mean,
        tempo_curve,
        tempo_variability,
        time_signature,
        time_signature_confidence,
        chord_sequence,
        chord_change_rate,
        predominant_chord,
        dissonance: dissonance_val,
        energy,
        danceability,
        key,
        key_confidence,
        key_camelot,
        valence,
        acousticness,
        // Embedding placeholder — future ONNX integration
        embedding: None,
        // Tier 3 placeholders — requires ML models
        mood_happy: None,
        mood_aggressive: None,
        mood_relaxed: None,
        mood_sad: None,
        instrumentalness: None,
        genre: None,

        // --- beat grid ---
        grid_offset_sec,
        downbeats,
        grid_stability,
        // --- structure ---
        energy_curve: structure.as_ref().map(|s| s.energy_curve.clone()),
        energy_curve_hop_sec: structure.as_ref().map(|s| s.energy_curve_hop_sec),
        segments: structure.as_ref().map(|s| s.segments.clone()),
        intro_end_sec: structure.as_ref().map(|s| s.intro_end_sec),
        outro_start_sec: structure.as_ref().map(|s| s.outro_start_sec),
        energy_level: structure.as_ref().map(|s| s.energy_level),
        // --- silence ---
        leading_silence_sec,
        trailing_silence_sec,
        // --- key candidates ---
        key_candidates,
        // --- vocalness ---
        vocalness,
        // --- fingerprint ---
        fingerprint,
        // --- similarity ---
        embedding_version: None,
    };

    // --- similarity ---
    // Populate the hand-crafted similarity vector only when explicitly opted in
    // via `features=["embedding"]`. This keeps compact/playlist/full unchanged
    // and adds near-zero cost (the vector is assembled from features already
    // computed above). A future ML embedding can replace `embed` behind the same
    // version field.
    if config.wants("embedding") {
        result.embedding = Some(crate::similarity::embed(&result));
        result.embedding_version = Some(crate::similarity::SIMILARITY_VERSION);
    }

    Ok(result)
}

// ============================================================
// Silence offsets (opt-in)
// ============================================================

/// Leading/trailing silence duration (seconds) from per-frame RMS.
///
/// A frame counts as silent when its RMS is below `threshold_db` dBFS relative to
/// full scale (amplitude `10^(threshold_db/20)`; default -60 dBFS ≈ 0.001).
///
/// Hysteresis rule: leading silence ends at the first frame that *begins a
/// sustained run* of at least `HYST_FRAMES` consecutive above-threshold frames.
/// A single loud click surrounded by silence is shorter than the run and is
/// therefore ignored — it does not terminate the silence. Trailing silence is
/// the symmetric quantity measured from the end.
///
/// Returns `(leading_sec, trailing_sec)`, each clamped to `[0, duration]`.
fn silence_offsets(
    rms: &[Float],
    sr: u32,
    hop_length: usize,
    threshold_db: Float,
) -> (Float, Float) {
    /// Consecutive above-threshold frames required to count as real audio onset.
    const HYST_FRAMES: usize = 3;

    let n = rms.len();
    let sec_per_frame = hop_length as Float / sr as Float;
    if n == 0 {
        return (0.0, 0.0);
    }
    let thresh = 10.0_f32.powf(threshold_db / 20.0);
    let need = HYST_FRAMES.min(n);

    // Leading: first index that starts a sustained above-threshold run.
    let mut lead_frames = n; // all-silence fallback
    for i in 0..n {
        if rms[i] >= thresh {
            let end = (i + need).min(n);
            if (end - i) >= need && (i..end).all(|k| rms[k] >= thresh) {
                lead_frames = i;
                break;
            }
        }
    }

    // Trailing: last index that ends a sustained above-threshold run.
    let mut trail_frames = n; // all-silence fallback
    for i in (0..n).rev() {
        if rms[i] >= thresh {
            let start = i + 1 - need; // i - (need-1)
            // `start` underflow-safe because i >= need-1 is required for a run.
            if i + 1 >= need && (i + 1 - need..=i).all(|k| rms[k] >= thresh) {
                trail_frames = n - 1 - i;
                break;
            }
            let _ = start;
        }
    }

    let dur = n as Float * sec_per_frame;
    let lead = (lead_frames as Float * sec_per_frame).clamp(0.0, dur);
    let trail = (trail_frames as Float * sec_per_frame).clamp(0.0, dur);
    (lead, trail)
}

// ============================================================
// Convenience constructors
// ============================================================

/// Shorthand for compact mode analysis.
pub fn compact() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Compact, ..Default::default() }
}

/// Shorthand for playlist mode analysis.
pub fn playlist() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Playlist, ..Default::default() }
}

/// Shorthand for full mode analysis.
pub fn full() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Full, ..Default::default() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: Float, sr: u32, dur: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin())
    }

    #[test]
    fn test_analyze_compact() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(result.duration_sec > 1.9 && result.duration_sec < 2.1);
        assert!(result.bpm > 30.0 && result.bpm < 320.0);
        assert!(result.rms_mean > 0.0);
        assert!(result.spectral_centroid_mean > 0.0);
        // Compact: no extended features
        assert!(result.spectral_bandwidth_mean.is_none());
        assert!(result.mfcc_mean.is_none());
        assert!(result.energy.is_none());
    }

    #[test]
    fn test_analyze_playlist() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        assert!(result.spectral_bandwidth_mean.unwrap() > 0.0);
        assert!(result.mfcc_mean.unwrap().len() == 13);
        assert!(result.chroma_mean.unwrap().len() == 12);
        assert!(result.energy.unwrap() >= 0.0);
        assert!(result.danceability.unwrap() >= 0.0);
        assert!(result.key.is_some());
        assert!(result.valence.unwrap() >= 0.0);
        assert!(result.acousticness.unwrap() >= 0.0);
    }

    #[test]
    fn test_analyze_accurate_chroma() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        // Chroma should map A440 to bin 9
        let chroma = result.chroma_mean.unwrap();
        assert_eq!(chroma.len(), 12);
        let max_bin = chroma.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        assert_eq!(max_bin, 9, "A440 should map to chroma bin 9 (A), got {}", max_bin);
    }

    #[test]
    fn test_analyze_custom_features() {
        let y = sine(440.0, 22050, 2.0);
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: Some(["energy", "key", "chroma"].iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        // Requested features should be present
        assert!(result.energy.is_some());
        assert!(result.key.is_some());
        assert!(result.chroma_mean.is_some());
        // Non-requested extended features should be absent
        assert!(result.danceability.is_none());
        assert!(result.acousticness.is_none());
    }

    #[test]
    fn test_analyze_config_accepts_runtime_bpm_range() {
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: None,
            bpm_min: Some(79.0),
            bpm_max: Some(192.0),
        };
        assert_eq!(config.bpm_min, Some(79.0));
        assert_eq!(config.bpm_max, Some(192.0));
    }

    #[test]
    fn test_analyze_exposes_bpm_candidates() {
        let sr = 22050u32;
        let n = (4.0 * sr as Float) as usize;
        let interval = (60.0 / 120.0 * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        let result = analyze_signal(y.view(), sr, &compact()).unwrap();
        assert!(!result.bpm_candidates.is_empty(), "expected tempo candidates");
        assert!(result.bpm_candidates.len() <= 5);
        // Candidates are sorted by score descending.
        for w in result.bpm_candidates.windows(2) {
            assert!(w[0].1 >= w[1].1, "candidates must be sorted by score descending");
        }
        assert!(result.bpm_raw > 30.0 && result.bpm_raw < 320.0);
        // Without a bpm range, the final bpm equals the raw selection.
        assert!((result.bpm - result.bpm_raw).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_click_train() {
        let sr = 22050u32;
        let n = (4.0 * sr as Float) as usize;
        let interval = (60.0 / 120.0 * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        let result = analyze_signal(y.view(), sr, &compact()).unwrap();
        assert!(result.bpm > 50.0 && result.bpm < 250.0);
        assert!(result.onset_frames.len() >= 3);
    }

    #[test]
    fn test_analyze_features_reasonable() {
        let y = Array1::from_shape_fn(44100, |i| {
            (2.0 * PI * 440.0 * i as Float / 22050.0).sin() * 0.5
        });
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(result.rms_mean > 0.1 && result.rms_mean < 0.6,
            "RMS {} unexpected", result.rms_mean);
        assert!(result.spectral_centroid_mean > 300.0 && result.spectral_centroid_mean < 600.0,
            "Centroid {} unexpected", result.spectral_centroid_mean);
    }

    #[test]
    fn test_analyze_playlist_sine_vs_noise() {
        let sine_sig = sine(440.0, 22050, 2.0);
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let noise = Array1::from_shape_fn(44100, |i| {
            let mut h = DefaultHasher::new();
            (i as u64 ^ 0xDEADBEEF).hash(&mut h);
            (h.finish() as Float / u64::MAX as Float) * 2.0 - 1.0
        });

        let cfg = playlist();
        let r_sine = analyze_signal(sine_sig.view(), 22050, &cfg).unwrap();
        let r_noise = analyze_signal(noise.view(), 22050, &cfg).unwrap();

        assert!(r_sine.spectral_flatness_mean.unwrap() < r_noise.spectral_flatness_mean.unwrap());
        assert!(r_sine.spectral_bandwidth_mean.unwrap() < r_noise.spectral_bandwidth_mean.unwrap());
    }

    // ---- structure (opt-in) ----

    fn structure_config() -> AnalysisConfig {
        AnalysisConfig {
            mode: AnalysisMode::Playlist,
            features: Some(["structure"].iter().map(|s| s.to_string()).collect()),
            ..AnalysisConfig::default()
        }
    }

    // ---- opt-in features: silence, key candidates, vocalness ----

    fn feature_config(names: &[&str]) -> AnalysisConfig {
        AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: Some(names.iter().map(|s| s.to_string()).collect()),
            ..AnalysisConfig::default()
        }
    }

    const SR: u32 = 22050;
    const HOP: usize = 512;
    const SPF: Float = HOP as Float / SR as Float; // seconds per frame

    #[test]
    fn test_silence_offsets_leading_trailing() {
        // 65 silent frames, 100 loud, 97 silent → 1.5s lead, 2.25s trail (approx).
        let mut rms = vec![0.0_f32; 65];
        rms.extend(std::iter::repeat(0.5).take(100));
        rms.extend(std::iter::repeat(0.0).take(97));
        let (lead, trail) = silence_offsets(&rms, SR, HOP, -60.0);
        assert!((lead - 65.0 * SPF).abs() < SPF, "lead {} vs {}", lead, 65.0 * SPF);
        assert!((trail - 97.0 * SPF).abs() < SPF, "trail {} vs {}", trail, 97.0 * SPF);
        assert!((lead - 1.5).abs() < 2.0 * SPF);
        assert!((trail - 2.25).abs() < 2.0 * SPF);
    }

    #[test]
    fn test_silence_offsets_no_silence() {
        let rms = vec![0.5_f32; 200];
        let (lead, trail) = silence_offsets(&rms, SR, HOP, -60.0);
        assert_eq!(lead, 0.0);
        assert_eq!(trail, 0.0);
    }

    #[test]
    fn test_silence_offsets_all_silence() {
        let rms = vec![0.0_f32; 200];
        let dur = 200.0 * SPF;
        let (lead, trail) = silence_offsets(&rms, SR, HOP, -60.0);
        assert!((lead - dur).abs() < 1e-4, "leading {} should ~= duration {}", lead, dur);
        assert!((trail - dur).abs() < 1e-4);
    }

    #[test]
    fn test_silence_offsets_click_hysteresis() {
        // 30 silent, 1 loud click, 30 silent, then 100 sustained loud, then silent.
        let mut rms = vec![0.0_f32; 30];
        rms.push(0.9); // isolated click
        rms.extend(std::iter::repeat(0.0).take(30));
        rms.extend(std::iter::repeat(0.5).take(100));
        rms.extend(std::iter::repeat(0.0).take(20));
        let (lead, _trail) = silence_offsets(&rms, SR, HOP, -60.0);
        // Sustained audio starts at frame 61, not at the click (frame 30).
        assert!((lead - 61.0 * SPF).abs() < SPF,
            "click should not end silence: lead {} expected ~{}", lead, 61.0 * SPF);
    }

    #[test]
    fn test_silence_pipeline_optin_and_bounds() {
        // Real pipeline: 1.5s leading + 2.25s trailing silence around a tone.
        let sr = SR;
        let lead_n = (1.5 * sr as Float) as usize;
        let trail_n = (2.25 * sr as Float) as usize;
        let mid_n = (3.0 * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(lead_n + mid_n + trail_n);
        for i in 0..mid_n {
            y[lead_n + i] = 0.5 * (2.0 * PI * 440.0 * i as Float / sr as Float).sin();
        }
        let r = analyze_signal(y.view(), sr, &feature_config(&["silence"])).unwrap();
        let lead = r.leading_silence_sec.unwrap();
        let trail = r.trailing_silence_sec.unwrap();
        assert!((lead - 1.5).abs() < 0.05, "lead {}", lead);
        assert!((trail - 2.25).abs() < 0.05, "trail {}", trail);
        assert!(lead >= 0.0 && lead <= r.duration_sec);
        assert!(trail >= 0.0 && trail <= r.duration_sec);
    }

    #[test]
    fn test_optin_absent_by_default() {
        let y = sine(440.0, SR, 3.0);
        for cfg in [compact(), playlist(), full()] {
            let r = analyze_signal(y.view(), SR, &cfg).unwrap();
            assert!(r.leading_silence_sec.is_none(), "silence must be opt-in");
            assert!(r.trailing_silence_sec.is_none());
            assert!(r.key_candidates.is_none(), "key_candidates must be opt-in");
            assert!(r.vocalness.is_none(), "vocalness must be opt-in");
        }
    }

    #[test]
    fn test_structure_is_opt_in() {
        let y = sine(440.0, 22050, 15.0);
        // Default playlist/full modes must NOT compute structure.
        for cfg in [playlist(), full()] {
            let r = analyze_signal(y.view(), 22050, &cfg).unwrap();
            assert!(r.energy_curve.is_none(), "structure must be absent by default");
            assert!(r.segments.is_none());
            assert!(r.energy_level.is_none());
        }
        // Compact obviously not.
        let rc = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(rc.energy_curve.is_none());
        // Opt-in via features=["structure"] turns it on.
        let rs = analyze_signal(y.view(), 22050, &structure_config()).unwrap();
        assert!(rs.energy_curve.as_ref().unwrap().len() > 0);
        assert!(rs.segments.is_some());
        assert!(rs.energy_curve_hop_sec.unwrap() > 0.0);
        let lvl = rs.energy_level.unwrap();
        assert!((1..=10).contains(&lvl));
    }

    #[test]
    fn test_structure_pipeline_known_shape() {
        // Synthetic audio with known structure:
        // 30s quiet 200 Hz sine -> 60s loud broadband -> 30s quiet sine.
        let sr = 22050u32;
        let seg = |dur: Float, loud: bool| -> Vec<Float> {
            let n = (dur * sr as Float) as usize;
            (0..n)
                .map(|i| {
                    if loud {
                        // Broadband: sum of several partials, high amplitude.
                        let t = i as Float / sr as Float;
                        0.5 * ((2.0 * PI * 200.0 * t).sin()
                            + (2.0 * PI * 1500.0 * t).sin()
                            + (2.0 * PI * 4000.0 * t).sin())
                            / 3.0
                            * 3.0
                    } else {
                        0.04 * (2.0 * PI * 200.0 * i as Float / sr as Float).sin()
                    }
                })
                .collect()
        };
        let mut samples = seg(30.0, false);
        samples.extend(seg(60.0, true));
        samples.extend(seg(30.0, false));
        let y = Array1::from(samples);

        let r = analyze_signal(y.view(), sr, &structure_config()).unwrap();
        let segs = r.segments.as_ref().unwrap();
        // Covering + ordered + non-overlapping.
        assert!(segs.first().unwrap().0.abs() < 1e-2, "first segment must start at 0");
        assert!((segs.last().unwrap().1 - r.duration_sec).abs() < 0.5, "last must end at duration");
        for w in segs.windows(2) {
            assert!((w[0].1 - w[1].0).abs() < 1e-2, "segments must be contiguous");
        }
        // Boundaries near 30s and 90s.
        let interior: Vec<Float> = segs.iter().skip(1).map(|s| s.0).collect();
        let near = |target: Float| interior.iter().any(|&b| (b - target).abs() < 8.0);
        assert!(near(30.0), "expected boundary near 30s, interior={:?}", interior);
        assert!(near(90.0), "expected boundary near 90s, interior={:?}", interior);
        // Intro/outro land in the quiet regions.
        assert!(r.intro_end_sec.unwrap() < 45.0);
        assert!(r.outro_start_sec.unwrap() > 80.0);
        // Middle (loud) segment has clearly higher mean energy than the ends.
        let mid = segs.iter().find(|s| s.0 < 60.0 && s.1 > 60.0).map(|s| s.2).unwrap_or(0.0);
        assert!(mid > segs.first().unwrap().2 + 0.15, "loud section should be more energetic");
    }

    #[test]
    fn test_key_candidates_pipeline_a_minor() {
        // Synthesized A-minor triad: A(220), C(~261.6), E(~329.6).
        let sr = SR;
        let n = (4.0 * sr as Float) as usize;
        let y = Array1::from_shape_fn(n, |i| {
            let t = i as Float / sr as Float;
            0.5 * (2.0 * PI * 220.0 * t).sin()
                + 0.4 * (2.0 * PI * 261.63 * t).sin()
                + 0.35 * (2.0 * PI * 329.63 * t).sin()
        });
        let r = analyze_signal(y.view(), sr, &feature_config(&["key_candidates"])).unwrap();
        let cands = r.key_candidates.unwrap();
        assert_eq!(cands.len(), 3, "exactly 3 candidates");
        // Scores descending, finite, in [0,1].
        for (_, _, s) in &cands {
            assert!(*s >= 0.0 && *s <= 1.0 && s.is_finite());
        }
        assert!(cands[0].2 >= cands[1].2 && cands[1].2 >= cands[2].2);
        // Camelot codes valid.
        let valid: HashSet<&str> = [
            "1A","2A","3A","4A","5A","6A","7A","8A","9A","10A","11A","12A",
            "1B","2B","3B","4B","5B","6B","7B","8B","9B","10B","11B","12B",
        ].into_iter().collect();
        for (_, cam, _) in &cands {
            assert!(valid.contains(cam.as_str()), "invalid camelot {}", cam);
        }
        // First candidate is A minor and matches the separately requested `key`.
        assert_eq!(cands[0].0, "A minor", "got {}", cands[0].0);
        let r2 = analyze_signal(y.view(), sr, &feature_config(&["key", "key_candidates"])).unwrap();
        assert_eq!(r2.key_candidates.unwrap()[0].0, r2.key.unwrap());
    }

    #[test]
    fn test_vocalness_pipeline_in_range() {
        let y = sine(440.0, SR, 3.0);
        let r = analyze_signal(y.view(), SR, &feature_config(&["vocalness"])).unwrap();
        let v = r.vocalness.unwrap();
        assert!(v >= 0.0 && v <= 1.0 && v.is_finite(), "vocalness {}", v);
    }
}
