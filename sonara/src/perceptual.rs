//! Perceptual audio features: LUFS loudness, energy, danceability, key, valence, acousticness.
//!
//! These are higher-level features derived from the signal-level measurements
//! already computed by the fused analysis pipeline. No additional FFT work.
//!
//! Tier 0 (standardized): LUFS integrated loudness (ITU-R BS.1770-4)
//! Tier 1 (signal-grounded): energy, danceability, key detection
//! Tier 2 (heuristic approximations): valence, acousticness, mood_*, instrumentalness
//! Tier 3 (requires ML, future): genre
//!
//! `mood_*` and `instrumentalness` ship as **heuristic v1** (rough hints in the
//! same spirit as valence/acousticness), not ML classifiers. `genre` remains a
//! future-ML placeholder.

use ndarray::ArrayView1;

use crate::types::Float;

// ============================================================
// LUFS Integrated Loudness (ITU-R BS.1770-4 / EBU R128)
// ============================================================

/// K-weighting filter coefficients for a given sample rate.
///
/// Two cascaded biquad sections:
/// 1. High-shelf (~+4dB above 1500 Hz) — head-related transfer function
/// 2. High-pass at ~38 Hz — removes subsonic content
///
/// Coefficients are computed via bilinear transform from the analog prototypes
/// defined in ITU-R BS.1770-4.
struct KWeightCoeffs {
    // Stage 1: high-shelf (b0, b1, b2, a1, a2) — a0 normalized to 1.0
    s1_b: [Float; 3],
    s1_a: [Float; 2], // a1, a2 (a0 = 1.0)
    // Stage 2: high-pass
    s2_b: [Float; 3],
    s2_a: [Float; 2],
}

impl KWeightCoeffs {
    fn for_sample_rate(sr: u32) -> Self {
        let sr = sr as f64;

        // Stage 1: High shelf filter
        // Pre-warped analog prototype from BS.1770-4
        let f0: f64 = 1681.974450955533;
        let g: f64 = 3.999843853572914; // ~+4 dB
        let q: f64 = 0.7071752369554196;

        let k = (std::f64::consts::PI * f0 / sr).tan();
        let vh = (10.0_f64).powf(g / 20.0);
        let vb = vh.powf(0.4996667741545416);

        let a0 = 1.0 + k / q + k * k;
        let s1_b0 = ((vh + vb * k / q + k * k) / a0) as Float;
        let s1_b1 = (2.0 * (k * k - vh) / a0) as Float;
        let s1_b2 = ((vh - vb * k / q + k * k) / a0) as Float;
        let s1_a1 = (2.0 * (k * k - 1.0) / a0) as Float;
        let s1_a2 = ((1.0 - k / q + k * k) / a0) as Float;

        // Stage 2: High-pass filter at ~38 Hz
        let f0: f64 = 38.13547087602444;
        let q: f64 = 0.5003270373238773;

        let k = (std::f64::consts::PI * f0 / sr).tan();
        let a0 = 1.0 + k / q + k * k;
        let s2_b0 = (1.0 / a0) as Float;
        let s2_b1 = (-2.0 / a0) as Float;
        let s2_b2 = (1.0 / a0) as Float;
        let s2_a1 = (2.0 * (k * k - 1.0) / a0) as Float;
        let s2_a2 = ((1.0 - k / q + k * k) / a0) as Float;

        KWeightCoeffs {
            s1_b: [s1_b0, s1_b1, s1_b2],
            s1_a: [s1_a1, s1_a2],
            s2_b: [s2_b0, s2_b1, s2_b2],
            s2_a: [s2_a1, s2_a2],
        }
    }
}

/// Compute integrated LUFS loudness per ITU-R BS.1770-4 / EBU R128.
///
/// This is the industry standard for loudness measurement, used by Spotify,
/// YouTube, and broadcast. It applies K-weighting (models human loudness
/// perception) then computes mean-square energy.
///
/// Returns the integrated loudness in LUFS (typically -60 to 0 for music).
/// Silence returns -70.0 (the EBU R128 "absolute gate" threshold).
///
/// Performance: ~0.2-0.5ms for a 3-minute track (two biquad IIR passes).
pub fn loudness_lufs(y: ArrayView1<Float>, sr: u32) -> Float {
    let n = y.len();
    if n == 0 {
        return -70.0;
    }

    let c = KWeightCoeffs::for_sample_rate(sr);
    let raw = y.as_slice().unwrap();

    // Apply K-weighting: two cascaded biquad sections (Direct Form II Transposed)
    // We process both stages in a single pass to stay cache-friendly.
    let mut s1_z1: Float = 0.0;
    let mut s1_z2: Float = 0.0;
    let mut s2_z1: Float = 0.0;
    let mut s2_z2: Float = 0.0;
    let mut sum_sq: Float = 0.0;

    for i in 0..n {
        // Stage 1: high shelf
        let x = raw[i];
        let y1 = c.s1_b[0] * x + s1_z1;
        s1_z1 = c.s1_b[1] * x - c.s1_a[0] * y1 + s1_z2;
        s1_z2 = c.s1_b[2] * x - c.s1_a[1] * y1;

        // Stage 2: high pass
        let y2 = c.s2_b[0] * y1 + s2_z1;
        s2_z1 = c.s2_b[1] * y1 - c.s2_a[0] * y2 + s2_z2;
        s2_z2 = c.s2_b[2] * y1 - c.s2_a[1] * y2;

        sum_sq += y2 * y2;
    }

    let mean_sq = sum_sq / n as Float;

    if mean_sq < 1e-20 {
        -70.0 // EBU R128 absolute gate
    } else {
        -0.691 + 10.0 * mean_sq.log10()
    }
}

// --- loudness ---
/// Apply the ITU-R BS.1770-4 K-weighting filter and return the filtered samples.
///
/// This shares the exact `KWeightCoeffs` biquad cascade that [`loudness_lufs`]
/// uses, so block-based measurements built on top of it (momentary / short-term
/// loudness, loudness range) are filtered identically to the integrated value —
/// no duplicated filter definition. It does not gate or aggregate; callers window
/// the returned mean-square energy themselves. Returns an empty vec for empty input.
///
/// (Appended in a marked block so parallel edits to this file stay isolated.)
pub(crate) fn k_weighted_signal(y: ArrayView1<Float>, sr: u32) -> Vec<Float> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let c = KWeightCoeffs::for_sample_rate(sr);
    let mut out = vec![0.0 as Float; n];

    // Two cascaded biquad sections (Direct Form II Transposed), single pass.
    let mut s1_z1: Float = 0.0;
    let mut s1_z2: Float = 0.0;
    let mut s2_z1: Float = 0.0;
    let mut s2_z2: Float = 0.0;

    for (i, &x) in y.iter().enumerate() {
        // Stage 1: high shelf
        let y1 = c.s1_b[0] * x + s1_z1;
        s1_z1 = c.s1_b[1] * x - c.s1_a[0] * y1 + s1_z2;
        s1_z2 = c.s1_b[2] * x - c.s1_a[1] * y1;

        // Stage 2: high pass
        let y2 = c.s2_b[0] * y1 + s2_z1;
        s2_z1 = c.s2_b[1] * y1 - c.s2_a[0] * y2 + s2_z2;
        s2_z2 = c.s2_b[2] * y1 - c.s2_a[1] * y2;

        out[i] = y2;
    }
    out
}
// --- end loudness ---

// ============================================================
// Key detection types
// ============================================================

/// Musical key detection result.
pub struct KeyResult {
    /// Root note: "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    pub key: &'static str,
    /// "major" or "minor"
    pub mode: &'static str,
    /// Pearson correlation strength (0.0 - 1.0). Higher = more confident.
    pub confidence: Float,
}

const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// Temperley MIREX 2005 key profiles — corpus-derived, better for popular music
// than the original Krumhansl profiles. Corpus-derived profiles are generally
// recommended over Krumhansl for non-classical music.
// Source: D. Temperley "What's Key for Key?" (1999/2005)
const KEY_PROFILE_MAJOR: [Float; 12] = [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400];
const KEY_PROFILE_MINOR: [Float; 12] = [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330];

// ============================================================
// Tier 1: Energy
// ============================================================

/// Compute perceptual energy (0.0 - 1.0).
///
/// Normalized combination of loudness, spectral brightness, rhythmic activity,
/// and frequency spread. Loosely modeled on Spotify's energy descriptor.
///
/// - Loud, bright, rhythmically active music → high energy
/// - Quiet, dark, sparse music → low energy
pub fn energy(
    rms_mean: Float,
    spectral_centroid_mean: Float,
    onset_density: Float,
    spectral_bandwidth_mean: Float,
) -> Float {
    // Normalize each feature to [0, 1] via empirical music ranges
    let norm_rms = (rms_mean / 0.5).clamp(0.0, 1.0);              // compressed pop reaches 0.5+
    let norm_centroid = ((spectral_centroid_mean - 500.0) / 4500.0).clamp(0.0, 1.0);
    let norm_onset = (onset_density / 10.0).clamp(0.0, 1.0);      // complex rhythms exceed 8
    let norm_bw = ((spectral_bandwidth_mean - 500.0) / 3500.0).clamp(0.0, 1.0);

    // Weighted combination
    let weighted = 0.35 * norm_rms + 0.25 * norm_centroid + 0.25 * norm_onset + 0.15 * norm_bw;

    // Gentler sigmoid centered lower — reaches 0.9+ on energetic music
    1.0 / (1.0 + (-4.0 * (weighted - 0.45)).exp())
}

// ============================================================
// Tier 1: Danceability
// ============================================================

/// Fast danceability estimate from beat regularity and tempo (0.0 - 1.0).
///
/// Uses heuristics: regular beats + tempo in 100-140 BPM range + moderate
/// onset density → high danceability. No extra signal processing needed.
pub fn danceability_heuristic(bpm: Float, beats: &[usize], onset_density: Float) -> Float {
    // Beat regularity: coefficient of variation of inter-beat intervals
    let beat_reg = if beats.len() >= 3 {
        let intervals: Vec<Float> = beats.windows(2)
            .map(|w| (w[1] - w[0]) as Float)
            .collect();
        let mean_interval = intervals.iter().sum::<Float>() / intervals.len() as Float;
        if mean_interval > 0.0 {
            let std_interval = (intervals.iter()
                .map(|&i| (i - mean_interval).powi(2))
                .sum::<Float>() / intervals.len() as Float)
                .sqrt();
            let cv = std_interval / mean_interval;
            1.0 - cv.clamp(0.0, 1.0) // low CV = regular beats = danceable
        } else {
            0.0
        }
    } else {
        0.0 // too few beats to judge
    };

    // Tempo sweet spot: Gaussian centered at 120 BPM
    let tempo_score = (-0.5 * ((bpm - 120.0) / 30.0).powi(2)).exp();

    // Onset density sweet spot: 2-6 onsets/sec is danceable
    let onset_score = (-0.5 * ((onset_density - 4.0) / 2.0).powi(2)).exp();

    0.4 * beat_reg + 0.35 * tempo_score + 0.25 * onset_score
}

/// Accurate danceability via Detrended Fluctuation Analysis (DFA).
///
/// Based on Streich & Herrera 2005.
/// Operates on the raw audio signal. Returns 0.0 - 1.0 (normalized from raw DFA
/// values which typically range 0 to ~3).
pub fn danceability_dfa(y: ArrayView1<Float>, sr: u32) -> Float {
    let frame_size = (0.01 * sr as Float) as usize; // 10ms frames
    let n_samples = y.len();
    let n_frames = n_samples / frame_size;

    if n_frames < 10 {
        return 0.0;
    }

    let raw = y.as_slice().unwrap();

    // Step 1: Compute stddev per 10ms frame
    let mut s = vec![0.0_f32; n_frames];
    for i in 0..n_frames {
        let start = i * frame_size;
        let end = ((i + 1) * frame_size).min(n_samples);
        let n = (end - start) as Float;
        let mean: Float = raw[start..end].iter().sum::<Float>() / n;
        let var: Float = raw[start..end].iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / (n - 1.0).max(1.0);
        s[i] = var.sqrt();
    }

    // Step 2: Subtract mean and integrate
    let mean_s: Float = s.iter().sum::<Float>() / n_frames as Float;
    for v in s.iter_mut() {
        *v -= mean_s;
    }
    for i in 1..n_frames {
        s[i] += s[i - 1];
    }

    // Step 3: Compute DFA for each tau
    // Tau values: geometrically spaced from 30 (0.3s) to min(800, n_frames) (×1.1)
    let min_tau = 30usize;
    let max_tau = 800.min(n_frames);
    let mut taus = Vec::new();
    let mut tau = min_tau as Float;
    while (tau as usize) <= max_tau {
        taus.push(tau as usize);
        tau *= 1.1;
    }

    if taus.len() < 2 {
        return 0.0;
    }

    let mut f_values: Vec<Float> = Vec::with_capacity(taus.len());

    for &tau in &taus {
        if n_frames < tau {
            break;
        }

        let jump = (tau / 50).max(1);
        let mut total_error = 0.0_f32;
        let mut n_blocks = 0usize;

        let mut k = 0;
        while k + tau <= n_frames {
            total_error += residual_error(&s, k, k + tau);
            n_blocks += 1;
            k += jump;
        }

        if n_blocks > 0 {
            f_values.push((total_error / n_blocks as Float).sqrt());
        } else {
            f_values.push(0.0);
        }
    }

    // Step 4: Compute DFA exponent from log-log slope
    let n_f = f_values.len();
    if n_f < 2 {
        return 0.0;
    }

    let mut dfa_sum = 0.0_f32;
    let mut n_valid = 0usize;

    for i in 0..n_f - 1 {
        if f_values[i + 1] > 0.0 && f_values[i] > 0.0 {
            let dfa_i = (f_values[i + 1] / f_values[i]).log10()
                / ((taus[i + 1] as Float + 3.0) / (taus[i] as Float + 3.0)).log10();
            dfa_sum += dfa_i;
            n_valid += 1;
        }
    }

    if n_valid == 0 {
        return 0.0;
    }

    let dfa_exponent = dfa_sum / n_valid as Float;
    let raw_danceability = if dfa_exponent > 0.0 { 1.0 / dfa_exponent } else { 0.0 };

    // Normalize to 0-1 (typical range is 0 to ~3)
    (raw_danceability / 3.0).clamp(0.0, 1.0)
}

/// Least-squares residual error for a segment (used by DFA).
/// Formula from Mathworld: ssyy - ssxy^2 / ssxx
fn residual_error(array: &[Float], start: usize, end: usize) -> Float {
    let size = end - start;
    let mean_x = (size - 1) as Float * 0.5;
    let mean_y: Float = array[start..end].iter().sum::<Float>() / size as Float;

    let mut ssxx = 0.0_f32;
    let mut ssyy = 0.0_f32;
    let mut ssxy = 0.0_f32;

    for i in 0..size {
        let dx = i as Float - mean_x;
        let dy = array[start + i] - mean_y;
        ssxx += dx * dx;
        ssxy += dx * dy;
        ssyy += dy * dy;
    }

    if ssxx > 0.0 {
        (ssyy - ssxy * ssxy / ssxx) / size as Float
    } else {
        0.0
    }
}

// ============================================================
// Tier 1: Key detection
// ============================================================

/// Detect musical key from a chroma vector using the Krumhansl-Schmuckler algorithm.
///
/// The chroma vector should have 12 elements (C, C#, D, ..., B).
/// Returns the best-matching key, mode (major/minor), and correlation confidence.
pub fn detect_key(chroma: &[Float]) -> KeyResult {
    if chroma.len() != 12 {
        return KeyResult { key: "C", mode: "major", confidence: 0.0 };
    }

    let mut best_key = 0usize;
    let mut best_mode = "major";
    let mut best_corr: Float = -2.0;
    let mut second_best: Float = -2.0;

    for shift in 0..12 {
        let corr_major = pearson_correlation(chroma, &KEY_PROFILE_MAJOR, shift);
        if corr_major > best_corr {
            second_best = best_corr;
            best_corr = corr_major;
            best_key = shift;
            best_mode = "major";
        } else if corr_major > second_best {
            second_best = corr_major;
        }

        let corr_minor = pearson_correlation(chroma, &KEY_PROFILE_MINOR, shift);
        if corr_minor > best_corr {
            second_best = best_corr;
            best_corr = corr_minor;
            best_key = shift;
            best_mode = "minor";
        } else if corr_minor > second_best {
            second_best = corr_minor;
        }
    }

    // Confidence: how much the best key stands out from the second best
    let confidence = if best_corr > -1.0 {
        ((best_corr - second_best) / best_corr.abs().max(0.001)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    KeyResult {
        key: NOTE_NAMES[best_key],
        mode: best_mode,
        confidence,
    }
}

/// Format key result as a string like "C major" or "A minor".
pub fn format_key(result: &KeyResult) -> String {
    format!("{} {}", result.key, result.mode)
}

// ============================================================
// Camelot wheel notation (DJ harmonic mixing)
// ============================================================

// Camelot codes indexed by pitch class (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
// Minor keys form the "A" ring, major keys the "B" ring. Used by DJs
// (Mixed In Key / Rekordbox) for harmonic mixing.
const CAMELOT_MINOR: [&str; 12] =
    ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"];
const CAMELOT_MAJOR: [&str; 12] =
    ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"];

/// Map a note name to its pitch class (0-11), accepting sharp or flat spellings.
fn pitch_class(tonic: &str) -> Option<usize> {
    Some(match tonic {
        "C" | "B#" => 0,
        "C#" | "Db" => 1,
        "D" => 2,
        "D#" | "Eb" => 3,
        "E" | "Fb" => 4,
        "F" | "E#" => 5,
        "F#" | "Gb" => 6,
        "G" => 7,
        "G#" | "Ab" => 8,
        "A" => 9,
        "A#" | "Bb" => 10,
        "B" | "Cb" => 11,
        _ => return None,
    })
}

/// Map a detected key `(tonic, mode)` to its Camelot wheel code.
///
/// Minor keys map to the "A" ring (e.g. A minor -> "8A"), major keys to the
/// "B" ring (e.g. C major -> "8B"). `tonic` accepts sharp or flat spellings;
/// `mode` is "major"/"maj" or "minor"/"min". Returns `None` for unrecognized input.
pub fn camelot(tonic: &str, mode: &str) -> Option<&'static str> {
    let pc = pitch_class(tonic)?;
    match mode {
        "minor" | "min" => Some(CAMELOT_MINOR[pc]),
        "major" | "maj" => Some(CAMELOT_MAJOR[pc]),
        _ => None,
    }
}

// --- key candidates ---
// Opt-in top-3 key detection. Mirrors the `bpm_candidates` design: a ranked
// list of `(key string, Camelot code, score)`. This does NOT alter the existing
// `detect_key` / `key` / `key_confidence` outputs — it re-runs the same 24
// major/minor profile correlations and exposes the ranking rather than only the
// argmax. The first candidate is guaranteed to equal `detect_key`'s result.

/// One ranked key candidate.
///
/// `key` is a human string like `"A minor"`, `camelot` is the Camelot-wheel
/// code (e.g. `"8A"`) for harmonic mixing, and `score` is the Pearson
/// correlation of the chroma against that key profile, clamped to `[0, 1]`
/// (higher = better match). Scores are comparable to `key_confidence` in scale.
pub struct KeyCandidate {
    /// Human-readable key, e.g. "A minor".
    pub key: String,
    /// Camelot-wheel code, e.g. "8A".
    pub camelot: &'static str,
    /// Correlation strength against this key profile, clamped to [0, 1].
    pub score: Float,
}

/// Map a root note + mode to its Camelot-wheel code (used for harmonic mixing).
///
/// Thin wrapper over [`camelot`] that returns `"?"` instead of `None` for
/// unrecognized input (should not happen for pipeline output).
pub fn key_camelot(key: &str, mode: &str) -> &'static str {
    camelot(key, mode).unwrap_or("?")
}

/// Detect the top-3 candidate keys from a 12-bin chroma vector.
///
/// Correlates the chroma against all 24 major/minor Temperley profiles (the same
/// correlation used by [`detect_key`]), ranks them, and returns the best three as
/// `(key string, Camelot code, score)`. `score` is the Pearson correlation
/// clamped to `[0, 1]`; scores are in descending order. The first entry matches
/// [`detect_key`]'s `key` / mode exactly.
pub fn detect_key_candidates(chroma: &[Float]) -> Vec<KeyCandidate> {
    if chroma.len() != 12 {
        return Vec::new();
    }
    // (shift, mode, correlation). Push major before minor per shift, in shift
    // order, so a *stable* descending sort breaks ties the same way `detect_key`
    // does (first-encountered maximum wins) — guaranteeing candidate[0] == key.
    let mut scored: Vec<(usize, &'static str, Float)> = Vec::with_capacity(24);
    for shift in 0..12 {
        scored.push((shift, "major", pearson_correlation(chroma, &KEY_PROFILE_MAJOR, shift)));
        scored.push((shift, "minor", pearson_correlation(chroma, &KEY_PROFILE_MINOR, shift)));
    }
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    scored.into_iter().take(3).map(|(shift, mode, corr)| {
        let key = NOTE_NAMES[shift];
        KeyCandidate {
            key: format!("{} {}", key, mode),
            camelot: key_camelot(key, mode),
            score: corr.clamp(0.0, 1.0),
        }
    }).collect()
}
// --- end key candidates ---

/// Pearson correlation between chroma (12 values) and a profile rotated by `shift`.
fn pearson_correlation(chroma: &[Float], profile: &[Float; 12], shift: usize) -> Float {
    let n = 12;
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;

    for i in 0..n {
        sum_x += chroma[(i + shift) % n];
        sum_y += profile[i];
    }
    let mean_x = sum_x / n as Float;
    let mean_y = sum_y / n as Float;

    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    let mut var_y = 0.0_f32;

    for i in 0..n {
        let dx = chroma[(i + shift) % n] - mean_x;
        let dy = profile[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 0.0 { cov / denom } else { 0.0 }
}

// ============================================================
// Tier 2: Valence
// ============================================================

/// Heuristic valence/mood estimate (0.0 = sad/dark, 1.0 = happy/bright).
///
/// Combines musical mode (major=happy), tempo (fast=happy), and spectral
/// brightness (bright=happy). This is an approximation — true mood perception
/// is subjective and context-dependent.
pub fn valence(key_result: &KeyResult, bpm: Float, spectral_centroid_mean: Float) -> Float {
    // Mode contribution
    let mode_score = if key_result.confidence < 0.05 {
        0.5 // low confidence → neutral
    } else if key_result.mode == "major" {
        0.7
    } else {
        0.3
    };

    // Tempo contribution (faster = happier, range 60-180)
    let tempo_score = ((bpm - 60.0) / 120.0).clamp(0.0, 1.0);

    // Brightness contribution (brighter = happier)
    let brightness = ((spectral_centroid_mean - 1000.0) / 3000.0).clamp(0.0, 1.0);

    0.45 * mode_score + 0.30 * tempo_score + 0.25 * brightness
}

// ============================================================
// Tier 2: Acousticness
// ============================================================

/// Heuristic acousticness estimate (0.0 = electronic/synthetic, 1.0 = acoustic).
///
/// Acoustic music tends to be more tonal (low spectral flatness), have less
/// high-frequency energy (low rolloff), and fewer percussive onsets.
pub fn acousticness(
    spectral_flatness_mean: Float,
    spectral_rolloff_mean: Float,
    onset_density: Float,
) -> Float {
    // Low flatness = tonal = more acoustic
    let tonal_score = (1.0 - (spectral_flatness_mean * 5.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    // Low rolloff = less high-frequency energy = more acoustic
    let hf_score = (1.0 - ((spectral_rolloff_mean - 2000.0) / 6000.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    // Fewer onsets = calmer = more acoustic
    let calm_score = (1.0 - (onset_density / 6.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    0.45 * tonal_score + 0.30 * hf_score + 0.25 * calm_score
}

// ============================================================
// Tier 2: Mood (heuristic v1)
// ============================================================

/// Four rough mood affinities, each in `[0, 1]`. Heuristic v1 — NOT an ML
/// classifier. The four scores are correlated (a happy track tends to be
/// un-sad) but are computed independently and are **not** constrained to sum
/// to 1.
pub struct MoodScores {
    /// Bright/upbeat affinity: major mode + moderate-fast tempo + brightness + danceability.
    pub happy: Float,
    /// Intense/harsh affinity: energy + rhythmic density + dissonance + minor nudge.
    pub aggressive: Float,
    /// Calm affinity: low energy + slow tempo + sparse onsets + narrow dynamics.
    pub relaxed: Float,
    /// Melancholic affinity: minor mode + slow tempo + darkness + low energy.
    pub sad: Float,
}

/// Heuristic mood affinities (happy / aggressive / relaxed / sad), each `[0, 1]`.
///
/// **Heuristic v1, not ML.** These are rough hints derived — like [`valence`]
/// and [`acousticness`] — from scalars the fused pipeline already produced. No
/// extra signal processing. Perceived mood is subjective and context-dependent;
/// treat these as coarse tags, not ground truth.
///
/// The two composite drivers (`energy`, `danceability_heuristic`) are recomputed
/// internally from the same raw scalars, so a mood request does not depend on
/// whether the caller also asked for those fields.
///
/// Terms (each normalized to `[0, 1]` over empirical music ranges):
/// - `mode_major` / `mode_minor`: 1/0 for a confident major key, 0/1 for minor,
///   0.5/0.5 when key confidence `< 0.05` (mirrors [`valence`]'s neutral term).
///   `None` `key_result` is treated as neutral (defensive — mood is extended-gated).
/// - `tempo` = `((bpm-60)/120)`; `slow` = `1-tempo`.
/// - `brightness` = `((centroid-1000)/3000)`; `darkness` = `1-brightness`.
/// - `onset` = `onset_density/8` (saturates at 8 onsets/sec); `low_onset` = `1-onset`.
/// - `diss` = clamped `dissonance` (0 when unavailable).
/// - `narrow_dyn` = `1 - dynamic_range_db/20` (narrow dynamics → relaxed).
///
/// Weighted sums (weights per score sum to 1, so each is already in `[0, 1]`;
/// clamped defensively):
/// - happy      = 0.35·major   + 0.25·tempo     + 0.20·brightness + 0.20·dance
/// - aggressive = 0.35·energy  + 0.30·onset      + 0.20·diss       + 0.15·minor
/// - relaxed    = 0.30·(1-energy) + 0.25·slow    + 0.25·low_onset  + 0.20·narrow_dyn
/// - sad        = 0.35·minor   + 0.25·slow       + 0.20·darkness   + 0.20·(1-energy)
#[allow(clippy::too_many_arguments)]
pub fn mood_scores(
    key_result: Option<&KeyResult>,
    bpm: Float,
    rms_mean: Float,
    spectral_centroid_mean: Float,
    onset_density: Float,
    spectral_bandwidth_mean: Float,
    beats: &[usize],
    dissonance: Option<Float>,
    dynamic_range_db: Float,
) -> MoodScores {
    // Composite drivers, recomputed from raw scalars (cheap, self-contained).
    let energy_val = energy(rms_mean, spectral_centroid_mean, onset_density, spectral_bandwidth_mean);
    let dance_val = danceability_heuristic(bpm, beats, onset_density);

    // Mode term (neutral 0.5/0.5 at low confidence or missing key).
    let (mode_major, mode_minor) = match key_result {
        Some(kr) if kr.confidence >= 0.05 => {
            if kr.mode == "major" { (1.0, 0.0) } else { (0.0, 1.0) }
        }
        _ => (0.5, 0.5),
    };

    let tempo = ((bpm - 60.0) / 120.0).clamp(0.0, 1.0);
    let slow = 1.0 - tempo;
    let brightness = ((spectral_centroid_mean - 1000.0) / 3000.0).clamp(0.0, 1.0);
    let darkness = 1.0 - brightness;
    let onset = (onset_density / 8.0).clamp(0.0, 1.0);
    let low_onset = 1.0 - onset;
    let diss = dissonance.map(|d| d.clamp(0.0, 1.0)).unwrap_or(0.0);
    let narrow_dyn = (1.0 - (dynamic_range_db / 20.0)).clamp(0.0, 1.0);

    let happy = (0.35 * mode_major + 0.25 * tempo + 0.20 * brightness + 0.20 * dance_val).clamp(0.0, 1.0);
    let aggressive = (0.35 * energy_val + 0.30 * onset + 0.20 * diss + 0.15 * mode_minor).clamp(0.0, 1.0);
    let relaxed = (0.30 * (1.0 - energy_val) + 0.25 * slow + 0.25 * low_onset + 0.20 * narrow_dyn).clamp(0.0, 1.0);
    let sad = (0.35 * mode_minor + 0.25 * slow + 0.20 * darkness + 0.20 * (1.0 - energy_val)).clamp(0.0, 1.0);

    MoodScores { happy, aggressive, relaxed, sad }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_energy_loud_vs_quiet() {
        let loud = energy(0.35, 3000.0, 5.0, 2500.0);
        let quiet = energy(0.05, 800.0, 1.0, 600.0);
        assert!(loud > quiet, "loud energy {} should be > quiet energy {}", loud, quiet);
        assert!(loud > 0.5, "loud signal energy should be > 0.5, got {}", loud);
        assert!(quiet < 0.5, "quiet signal energy should be < 0.5, got {}", quiet);
    }

    #[test]
    fn test_energy_range() {
        // Extreme low
        let low = energy(0.0, 0.0, 0.0, 0.0);
        assert!(low >= 0.0 && low <= 1.0);
        // Extreme high
        let high = energy(1.0, 10000.0, 20.0, 8000.0);
        assert!(high >= 0.0 && high <= 1.0);
    }

    #[test]
    fn test_danceability_heuristic_regular_beats() {
        // Regular 120 BPM with consistent intervals
        let beats: Vec<usize> = (0..16).map(|i| i * 43).collect(); // ~43 frames apart
        let d = danceability_heuristic(120.0, &beats, 4.0);
        assert!(d > 0.5, "Regular 120 BPM should be danceable, got {}", d);
    }

    #[test]
    fn test_danceability_heuristic_irregular() {
        // Irregular beats, weird tempo
        let beats = vec![0, 10, 50, 52, 100, 200];
        let d = danceability_heuristic(45.0, &beats, 1.0);
        assert!(d < 0.4, "Irregular slow music should not be very danceable, got {}", d);
    }

    #[test]
    fn test_danceability_dfa_basic() {
        // Generate a simple signal and verify DFA returns a reasonable value
        let n = 22050 * 3; // 3 seconds
        let y = Array1::from_shape_fn(n, |i| {
            (2.0 * std::f32::consts::PI * 440.0 * i as Float / 22050.0).sin()
        });
        let d = danceability_dfa(y.view(), 22050);
        assert!(d >= 0.0 && d <= 1.0, "DFA danceability should be in [0,1], got {}", d);
    }

    #[test]
    fn test_key_detection_a440() {
        // A major chord: A(9), C#(1), E(4) → should detect A
        let mut chroma = [0.01_f32; 12];
        chroma[9] = 1.0; // A
        chroma[1] = 0.6; // C#
        chroma[4] = 0.5; // E
        let result = detect_key(&chroma);
        assert_eq!(result.key, "A", "A major chroma should detect key A, got {}", result.key);
    }

    #[test]
    fn test_key_detection_c_major() {
        // C, E, G prominent → C major
        let mut chroma = [0.01_f32; 12];
        chroma[0] = 1.0; // C
        chroma[4] = 0.7; // E
        chroma[7] = 0.5; // G
        let result = detect_key(&chroma);
        assert_eq!(result.key, "C", "C major chroma should detect C, got {}", result.key);
        assert_eq!(result.mode, "major", "C major chroma should detect major, got {}", result.mode);
    }

    #[test]
    fn test_key_detection_a_minor() {
        // A, C, E prominent → A minor
        let mut chroma = [0.01_f32; 12];
        chroma[9] = 1.0; // A
        chroma[0] = 0.7; // C
        chroma[4] = 0.5; // E
        let result = detect_key(&chroma);
        assert_eq!(result.key, "A", "A minor chroma should detect A, got {}", result.key);
    }

    #[test]
    fn test_valence_major_fast_bright() {
        let major_key = KeyResult { key: "C", mode: "major", confidence: 0.5 };
        let v = valence(&major_key, 140.0, 3500.0);
        assert!(v > 0.5, "Major + fast + bright should have high valence, got {}", v);
    }

    #[test]
    fn test_valence_minor_slow_dark() {
        let minor_key = KeyResult { key: "A", mode: "minor", confidence: 0.5 };
        let v = valence(&minor_key, 70.0, 800.0);
        assert!(v < 0.5, "Minor + slow + dark should have low valence, got {}", v);
    }

    // ---- mood (heuristic v1) ----

    /// A regular ~120-BPM beat grid so danceability has something to chew on.
    fn regular_beats() -> Vec<usize> {
        (0..16).map(|i| i * 43).collect()
    }

    #[test]
    fn test_mood_happy_fast_bright_major() {
        let major = KeyResult { key: "C", mode: "major", confidence: 0.5 };
        let beats = regular_beats();
        // fast, bright, major, high-energy
        let m = mood_scores(Some(&major), 140.0, 0.4, 3500.0, 4.0, 2500.0, &beats, None, 8.0);
        assert!(m.happy > 0.6, "happy {} should be > 0.6", m.happy);
        assert!(m.happy > m.sad, "happy {} should exceed sad {}", m.happy, m.sad);
        for v in [m.happy, m.aggressive, m.relaxed, m.sad] {
            assert!((0.0..=1.0).contains(&v) && v.is_finite(), "out of range {}", v);
        }
    }

    #[test]
    fn test_mood_sad_relaxed_slow_dark_minor() {
        let minor = KeyResult { key: "A", mode: "minor", confidence: 0.5 };
        let beats = regular_beats();
        // slow, dark, minor, low-energy, sparse onsets, narrow dynamics
        let m = mood_scores(Some(&minor), 70.0, 0.05, 800.0, 1.0, 600.0, &beats, None, 5.0);
        assert!(m.sad > 0.6, "sad {} should be > 0.6", m.sad);
        assert!(m.sad > m.happy, "sad {} should exceed happy {}", m.sad, m.happy);
        assert!(m.relaxed > m.aggressive, "relaxed {} should exceed aggressive {}", m.relaxed, m.aggressive);
    }

    #[test]
    fn test_mood_aggressive_dense_dissonant() {
        let minor = KeyResult { key: "E", mode: "minor", confidence: 0.5 };
        let beats = regular_beats();
        // high onset density + dissonance + loud/bright → aggressive high
        let m = mood_scores(Some(&minor), 150.0, 0.45, 4000.0, 9.0, 3000.0, &beats, Some(0.8), 12.0);
        assert!(m.aggressive > 0.6, "aggressive {} should be > 0.6", m.aggressive);
        assert!(m.aggressive > m.relaxed, "aggressive {} should exceed relaxed {}", m.aggressive, m.relaxed);
    }

    #[test]
    fn test_mood_range_on_boundaries() {
        let beats = regular_beats();
        // Extreme inputs and a None key must all stay within [0,1].
        for kr in [None, Some(&KeyResult { key: "C", mode: "major", confidence: 0.0 })] {
            let lo = mood_scores(kr, 0.0, 0.0, 0.0, 0.0, 0.0, &beats, Some(0.0), 0.0);
            let hi = mood_scores(kr, 400.0, 2.0, 12000.0, 40.0, 9000.0, &beats, Some(2.0), 60.0);
            for v in [lo.happy, lo.aggressive, lo.relaxed, lo.sad,
                      hi.happy, hi.aggressive, hi.relaxed, hi.sad] {
                assert!((0.0..=1.0).contains(&v) && v.is_finite(), "out of range {}", v);
            }
        }
    }

    #[test]
    fn test_acousticness_tonal_vs_noisy() {
        let acoustic = acousticness(0.01, 1500.0, 2.0); // tonal, low HF, calm
        let electronic = acousticness(0.5, 7000.0, 6.0); // flat, high HF, busy
        assert!(acoustic > electronic,
            "Tonal signal ({}) should be more acoustic than noisy signal ({})",
            acoustic, electronic);
    }

    #[test]
    fn test_acousticness_range() {
        let low = acousticness(1.0, 11025.0, 10.0);
        let high = acousticness(0.0, 500.0, 0.0);
        assert!(low >= 0.0 && low <= 1.0);
        assert!(high >= 0.0 && high <= 1.0);
    }

    #[test]
    fn test_format_key() {
        let result = KeyResult { key: "F#", mode: "minor", confidence: 0.7 };
        assert_eq!(format_key(&result), "F# minor");
    }

    #[test]
    fn test_camelot_all_minor_keys() {
        // (tonic, expected Camelot code) for the full "A" ring.
        let cases = [
            ("G#", "1A"), ("D#", "2A"), ("A#", "3A"), ("F", "4A"),
            ("C", "5A"), ("G", "6A"), ("D", "7A"), ("A", "8A"),
            ("E", "9A"), ("B", "10A"), ("F#", "11A"), ("C#", "12A"),
        ];
        for (tonic, code) in cases {
            assert_eq!(camelot(tonic, "minor"), Some(code), "{tonic} minor");
        }
    }

    // ---- key candidates ----

    #[test]
    fn test_key_candidates_a_minor_triad() {
        // A-minor triad: A(9), C(0), E(4)
        let mut chroma = [0.01_f32; 12];
        chroma[9] = 1.0;
        chroma[0] = 0.7;
        chroma[4] = 0.5;
        let cands = detect_key_candidates(&chroma);
        // Exactly 3 entries
        assert_eq!(cands.len(), 3);
        // First candidate matches detect_key exactly
        let dk = detect_key(&chroma);
        assert_eq!(cands[0].key, format_key(&dk),
            "first candidate {} should equal detect_key {}", cands[0].key, format_key(&dk));
        assert_eq!(cands[0].key, "A minor");
        // Scores strictly descending (non-increasing) and in [0,1]
        for c in &cands {
            assert!(c.score >= 0.0 && c.score <= 1.0 && c.score.is_finite());
        }
        assert!(cands[0].score >= cands[1].score);
        assert!(cands[1].score >= cands[2].score);
        // Camelot codes valid (match a known code table)
        let valid: std::collections::HashSet<&str> = [
            "1A","2A","3A","4A","5A","6A","7A","8A","9A","10A","11A","12A",
            "1B","2B","3B","4B","5B","6B","7B","8B","9B","10B","11B","12B",
        ].into_iter().collect();
        for c in &cands {
            assert!(valid.contains(c.camelot), "invalid camelot {}", c.camelot);
        }
    }

    #[test]
    fn test_camelot_all_major_keys() {
        // (tonic, expected Camelot code) for the full "B" ring.
        let cases = [
            ("B", "1B"), ("F#", "2B"), ("C#", "3B"), ("G#", "4B"),
            ("D#", "5B"), ("A#", "6B"), ("F", "7B"), ("C", "8B"),
            ("G", "9B"), ("D", "10B"), ("A", "11B"), ("E", "12B"),
        ];
        for (tonic, code) in cases {
            assert_eq!(camelot(tonic, "major"), Some(code), "{tonic} major");
        }
    }

    #[test]
    fn test_camelot_enharmonic_spellings() {
        // Flat spellings must resolve to the same code as their sharp equivalent.
        assert_eq!(camelot("Ab", "minor"), Some("1A")); // = G# minor
        assert_eq!(camelot("Eb", "minor"), Some("2A")); // = D# minor
        assert_eq!(camelot("Bb", "minor"), Some("3A")); // = A# minor
        assert_eq!(camelot("Gb", "minor"), Some("11A")); // = F# minor
        assert_eq!(camelot("Db", "minor"), Some("12A")); // = C# minor
        assert_eq!(camelot("Gb", "major"), Some("2B")); // = F# major
        assert_eq!(camelot("Db", "major"), Some("3B")); // = C# major
        assert_eq!(camelot("Ab", "major"), Some("4B")); // = G# major
        assert_eq!(camelot("Eb", "major"), Some("5B")); // = D# major
        assert_eq!(camelot("Bb", "major"), Some("6B")); // = A# major
    }

    #[test]
    fn test_camelot_accepts_short_mode_names() {
        assert_eq!(camelot("A", "min"), Some("8A"));
        assert_eq!(camelot("C", "maj"), Some("8B"));
    }

    #[test]
    fn test_camelot_rejects_invalid_input() {
        assert_eq!(camelot("H", "minor"), None);
        assert_eq!(camelot("A", "dorian"), None);
    }

    #[test]
    fn test_camelot_matches_detector_output() {
        // Every note name the detector can emit must map to a Camelot code.
        for name in NOTE_NAMES {
            assert!(camelot(name, "minor").is_some(), "{name} minor");
            assert!(camelot(name, "major").is_some(), "{name} major");
        }
    }

    #[test]
    fn test_key_candidates_camelot_relatives() {
        // C major and A minor are relative keys → 8B / 8A
        assert_eq!(key_camelot("C", "major"), "8B");
        assert_eq!(key_camelot("A", "minor"), "8A");
        assert_eq!(key_camelot("G", "major"), "9B");
        assert_eq!(key_camelot("E", "minor"), "9A");
    }

    #[test]
    fn test_key_candidates_empty_on_bad_len() {
        assert!(detect_key_candidates(&[0.0; 5]).is_empty());
    }

    // ---- LUFS tests ----

    #[test]
    fn test_lufs_silence() {
        let y = Array1::<Float>::zeros(22050);
        let lufs = loudness_lufs(y.view(), 22050);
        assert_eq!(lufs, -70.0, "Silence should be -70 LUFS");
    }

    #[test]
    fn test_lufs_loud_vs_quiet() {
        let loud = Array1::from_shape_fn(22050, |i| {
            (2.0 * std::f32::consts::PI * 1000.0 * i as Float / 22050.0).sin()
        });
        let quiet = loud.mapv(|v| v * 0.1);

        let lufs_loud = loudness_lufs(loud.view(), 22050);
        let lufs_quiet = loudness_lufs(quiet.view(), 22050);

        assert!(lufs_loud > lufs_quiet,
            "Loud LUFS ({}) should be > quiet LUFS ({})", lufs_loud, lufs_quiet);
        // 10x amplitude = 20 dB difference
        let diff = lufs_loud - lufs_quiet;
        assert!((diff - 20.0).abs() < 2.0,
            "LUFS difference {} should be ~20 dB for 10x amplitude ratio", diff);
    }

    #[test]
    fn test_lufs_range() {
        // Unit amplitude 1kHz sine should be around -3 LUFS
        // (K-weighting boosts HF slightly, so slightly higher than pure RMS)
        let y = Array1::from_shape_fn(44100, |i| {
            (2.0 * std::f32::consts::PI * 1000.0 * i as Float / 22050.0).sin()
        });
        let lufs = loudness_lufs(y.view(), 22050);
        assert!(lufs > -10.0 && lufs < 5.0,
            "Unit sine LUFS {} should be in reasonable range", lufs);
    }
}
