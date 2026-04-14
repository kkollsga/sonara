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
    /// Note: `duration` is always included. Some features depend on others
    /// (e.g., `key` requires `chroma`, `valence` requires `key`); dependencies
    /// are resolved automatically.
    pub features: Option<HashSet<String>>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            mode: AnalysisMode::Compact,
            features: None,
        }
    }
}

impl AnalysisConfig {
    /// Check if a feature should be computed.
    fn wants(&self, name: &str) -> bool {
        if let Some(ref features) = self.features {
            // Explicit feature list — check if requested
            features.contains(name)
        } else {
            // Mode-based defaults
            match self.mode {
                AnalysisMode::Compact => false,
                AnalysisMode::Playlist => {
                    // Expensive rhythm analysis features are Full-only
                    // (metrogram is O(n³) and costs ~445ms for a 3-min track)
                    !matches!(name, "tempo_curve" | "time_signature")
                }
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
            ];
            EXTENDED_FEATURES.iter().any(|&f| features.contains(f))
        } else {
            self.mode != AnalysisMode::Compact
        }
    }

}

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
    pub beats: Vec<usize>,
    pub onset_frames: Vec<usize>,
    pub rms_mean: Float,
    pub rms_max: Float,
    pub loudness_lufs: Float,
    pub dynamic_range_db: Float,
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

    let (bpm, beats) = crate::beat::beat_track(
        None, Some(oenv_padded.view()), sr, hop_length, 120.0, 100.0, true,
    )?;

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

    let key = key_result.as_ref().map(|kr| perceptual::format_key(kr));
    let key_confidence = key_result.as_ref().map(|kr| kr.confidence);

    Ok(TrackAnalysis {
        duration_sec,
        bpm,
        beats,
        onset_frames,
        rms_mean,
        rms_max,
        loudness_lufs,
        dynamic_range_db,
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
    })
}

// ============================================================
// Convenience constructors
// ============================================================

/// Shorthand for compact mode analysis.
pub fn compact() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Compact, features: None }
}

/// Shorthand for playlist mode analysis.
pub fn playlist() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Playlist, features: None }
}

/// Shorthand for full mode analysis.
pub fn full() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Full, features: None }
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
}
