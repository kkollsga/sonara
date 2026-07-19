//! Beat tracking.
//!
//! Beat tracking via dynamic programming (Ellis 2007 algorithm).
//! Includes beat_track, plp, tempo_curve, and tempo_variability.

use ndarray::{Array1, ArrayView1};

use crate::error::{Result, SonaraError};
use crate::onset;
use crate::types::Float;

/// Number of strongest tempo candidates surfaced in a [`TempoEstimate`].
const MAX_TEMPO_CANDIDATES: usize = 5;

/// Result of tempo estimation, including diagnostic tempo candidates.
///
/// - `tempo`: final BPM used for beat tracking (after metrical selection,
///   fractional ACF refinement, optional `bpm_min`/`bpm_max` range alignment,
///   and clamping to `[30, 320]`).
/// - `tempo_raw`: the same estimate *before* optional BPM-range alignment.
/// - `candidates`: the strongest ACF tempo candidates as `(bpm, score)` pairs,
///   sorted by score descending.
#[derive(Debug, Clone)]
pub struct TempoEstimate {
    /// Final tempo in BPM (post range-alignment and clamping).
    pub tempo: Float,
    /// Selected tempo in BPM before optional BPM-range alignment.
    pub tempo_raw: Float,
    /// Strongest `(bpm, score)` candidates, sorted by score descending.
    pub candidates: Vec<(Float, Float)>,
}

impl TempoEstimate {
    /// Fallback estimate carrying a single tempo and no ACF candidates.
    fn fallback(tempo: Float) -> Self {
        Self {
            tempo,
            tempo_raw: tempo,
            candidates: Vec::new(),
        }
    }
}

/// Track beats in an audio signal.
///
/// Returns `(tempo, beat_frames)`:
/// - `tempo`: Estimated tempo in BPM
/// - `beat_frames`: Frame indices of detected beats
pub fn beat_track(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    tightness: Float,
    trim: bool,
) -> Result<(Float, Vec<usize>)> {
    beat_track_with_bpm_range(
        y,
        onset_envelope,
        sr,
        hop_length,
        start_bpm,
        tightness,
        trim,
        None,
        None,
    )
}

/// Track beats in an audio signal with optional octave-folding BPM range.
///
/// If both `bpm_min` and `bpm_max` are provided, the estimated tempo is doubled
/// or halved by octaves until it falls inside the requested range. This mirrors
/// DJ-library workflows that constrain BPM display to a preferred range.
pub fn beat_track_with_bpm_range(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    tightness: Float,
    trim: bool,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
) -> Result<(Float, Vec<usize>)> {
    let (estimate, beats) = beat_track_detailed(
        y,
        onset_envelope,
        sr,
        hop_length,
        start_bpm,
        tightness,
        trim,
        bpm_min,
        bpm_max,
    )?;
    Ok((estimate.tempo, beats))
}

/// Track beats and return the full [`TempoEstimate`] alongside the beat frames.
///
/// Like [`beat_track_with_bpm_range`], but also surfaces the pre-range-alignment
/// tempo and the strongest ACF tempo candidates for reporting.
#[allow(clippy::too_many_arguments)]
pub fn beat_track_detailed(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    tightness: Float,
    trim: bool,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
) -> Result<(TempoEstimate, Vec<usize>)> {
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;

    // Get onset envelope
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "either y or onset_envelope must be provided".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    if oenv.len() < 4 {
        return Ok((TempoEstimate::fallback(start_bpm), vec![]));
    }

    // Guard against flat / degenerate onset envelopes (silence, DC): with no
    // dynamic range there is no meaningful autocorrelation peak to track.
    let (min_onset, max_onset) = oenv.iter().copied().fold(
        (Float::INFINITY, Float::NEG_INFINITY),
        |(min_v, max_v), v| (min_v.min(v), max_v.max(v)),
    );
    if !min_onset.is_finite()
        || !max_onset.is_finite()
        || max_onset <= 1e-10
        || (max_onset - min_onset) <= 1e-10
    {
        return Ok((TempoEstimate::fallback(start_bpm), vec![]));
    }

    // Estimate tempo
    let estimate = estimate_tempo(&oenv, sr, hop_length, start_bpm, bpm_min, bpm_max)?;
    let tempo = estimate.tempo;
    let frames_per_beat = (60.0 * frame_rate / tempo).round() as usize;

    if frames_per_beat == 0 {
        return Ok((estimate, vec![]));
    }

    // Normalize onset envelope
    let mean = oenv.iter().sum::<Float>() / oenv.len() as Float;
    let std =
        (oenv.iter().map(|&v| (v - mean).powi(2)).sum::<Float>() / oenv.len() as Float).sqrt();
    let oenv_norm = if std > 0.0 {
        oenv.mapv(|v| (v - mean) / (std + 1e-10))
    } else {
        oenv.clone()
    };

    // Compute local score via Gaussian-windowed autocorrelation
    let local_score = beat_local_score(oenv_norm.view(), frames_per_beat);

    // Run DP beat tracker
    let beats = beat_track_dp(local_score.view(), frames_per_beat, tightness);

    // Trim beats with low onset strength
    let beats = if trim {
        trim_beats(local_score.view(), &beats)
    } else {
        beats
    };

    Ok((estimate, beats))
}

/// Estimate tempo from onset envelope using autocorrelation.
fn estimate_tempo(
    oenv: &Array1<Float>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
) -> Result<TempoEstimate> {
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;

    // Autocorrelate onset envelope
    let max_lag = (4.0 * frame_rate).min(oenv.len() as Float) as usize; // up to 4 seconds
    let acf = crate::core::audio::autocorrelate(oenv.view(), Some(max_lag))?;

    if acf.is_empty() {
        return Ok(TempoEstimate::fallback(start_bpm));
    }

    // Find peaks in BPM range [30, 300]
    let min_lag = (60.0 * frame_rate / 300.0).ceil() as usize;
    let max_lag = (60.0 * frame_rate / 30.0).floor() as usize;
    let max_lag = max_lag.min(acf.len() - 1);

    if min_lag >= max_lag {
        return Ok(TempoEstimate::fallback(start_bpm));
    }

    // Weight by log-normal prior centered at start_bpm, collecting every
    // candidate so downstream metrical-multiple lifting can inspect them.
    let mut candidates = Vec::with_capacity(max_lag - min_lag + 1);

    for lag in min_lag..=max_lag {
        let bpm = 60.0 * frame_rate / lag as Float;
        let log_prior = -0.5 * ((bpm.log2() - start_bpm.log2()) / 1.0).powi(2);
        let score = acf[lag] * (1.0 + log_prior.exp());
        candidates.push((lag, bpm, score));
    }

    let (lag, _tempo, _score) =
        select_preferred_tempo_candidate(&candidates).unwrap_or((min_lag, start_bpm, 0.0));
    let refined = refine_tempo_from_acf_peak(acf.view(), lag, frame_rate);
    let tempo_raw = refined.clamp(30.0, 320.0);
    let tempo = align_tempo_to_bpm_range(refined, bpm_min, bpm_max)?.clamp(30.0, 320.0);

    // Surface the strongest candidates (by score) as (bpm, score) pairs.
    let mut ranked: Vec<(Float, Float)> = candidates
        .iter()
        .filter(|(_, bpm, score)| bpm.is_finite() && score.is_finite())
        .map(|&(_, bpm, score)| (bpm, score))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(MAX_TEMPO_CANDIDATES);

    Ok(TempoEstimate {
        tempo,
        tempo_raw,
        candidates: ranked,
    })
}

/// Refine a tempo from an integer ACF lag using parabolic interpolation of the
/// autocorrelation peak. This removes the 1–3 BPM quantization drift caused by
/// snapping tempo to integer lags.
fn refine_tempo_from_acf_peak(acf: ArrayView1<Float>, lag: usize, frame_rate: Float) -> Float {
    let integer_tempo = if lag > 0 {
        60.0 * frame_rate / lag as Float
    } else {
        return 0.0;
    };

    if lag + 1 >= acf.len() || !frame_rate.is_finite() || frame_rate <= 0.0 {
        return integer_tempo;
    }

    let left = acf[lag - 1];
    let center = acf[lag];
    let right = acf[lag + 1];
    if !left.is_finite()
        || !center.is_finite()
        || !right.is_finite()
        || center < left
        || center < right
    {
        return integer_tempo;
    }

    let denominator = left - 2.0 * center + right;
    if denominator.abs() <= 1e-12 {
        return integer_tempo;
    }

    let offset = (0.5 * (left - right) / denominator).clamp(-0.5, 0.5);
    let refined_lag = lag as Float + offset;
    if refined_lag <= 0.0 || !refined_lag.is_finite() {
        integer_tempo
    } else {
        60.0 * frame_rate / refined_lag
    }
}

/// Choose a preferred tempo candidate, lifting a supported metrical multiple
/// (2x / 1.5x) when the raw best BPM is suspiciously low. Electronic music
/// frequently produces a strong half-tempo ACF peak; when a supported
/// double/dotted multiple exists in the higher, danceable range we prefer it.
fn select_preferred_tempo_candidate(
    candidates: &[(usize, Float, Float)],
) -> Option<(usize, Float, Float)> {
    let best = candidates
        .iter()
        .copied()
        .filter(|(_, bpm, score)| bpm.is_finite() && score.is_finite())
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))?;

    if best.1 < 75.0 {
        if let Some(candidate) =
            best_supported_metrical_candidate(candidates, best, 115.0, 150.0, 1.80, 2.20, 0.50)
        {
            return Some(candidate);
        }
    } else if best.1 < 90.0 {
        if let Some(candidate) =
            best_supported_metrical_candidate(candidates, best, 120.0, 145.0, 1.42, 1.62, 0.75)
        {
            return Some(candidate);
        }
    } else if best.1 < 95.0 && best.2 >= 4.0 {
        if let Some(candidate) =
            best_supported_metrical_candidate(candidates, best, 120.0, 145.0, 1.42, 1.62, 0.85)
        {
            return Some(candidate);
        }
    }

    Some(best)
}

/// Find the strongest candidate that is a supported metrical multiple of `best`
/// (within the given BPM window, multiple range, and score-ratio floor).
fn best_supported_metrical_candidate(
    candidates: &[(usize, Float, Float)],
    best: (usize, Float, Float),
    min_bpm: Float,
    max_bpm: Float,
    min_multiple: Float,
    max_multiple: Float,
    min_score_ratio: Float,
) -> Option<(usize, Float, Float)> {
    candidates
        .iter()
        .copied()
        .filter(|candidate| {
            let multiple = candidate.1 / best.1;
            candidate.1 >= min_bpm
                && candidate.1 <= max_bpm
                && multiple >= min_multiple
                && multiple <= max_multiple
                && candidate.2 >= best.2 * min_score_ratio
        })
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
}

/// Deterministically double/halve a tempo into an optional user-supplied BPM
/// range. Both bounds must be supplied together and span at least one octave.
fn align_tempo_to_bpm_range(
    mut tempo: Float,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
) -> Result<Float> {
    let (min_bpm, max_bpm) = match (bpm_min, bpm_max) {
        (None, None) => return Ok(tempo),
        (Some(min_bpm), Some(max_bpm)) => (min_bpm, max_bpm),
        _ => {
            return Err(SonaraError::InvalidParameter {
                param: "bpm_range",
                reason: "bpm_min and bpm_max must be provided together".into(),
            });
        }
    };

    if !min_bpm.is_finite() || !max_bpm.is_finite() || min_bpm <= 0.0 || max_bpm <= min_bpm {
        return Err(SonaraError::InvalidParameter {
            param: "bpm_range",
            reason: "expected finite values with 0 < bpm_min < bpm_max".into(),
        });
    }
    if max_bpm < min_bpm * 2.0 {
        return Err(SonaraError::InvalidParameter {
            param: "bpm_range",
            reason: "bpm_max must be at least double bpm_min for octave folding".into(),
        });
    }
    if !tempo.is_finite() || tempo <= 0.0 {
        return Ok(tempo);
    }

    while tempo < min_bpm {
        tempo *= 2.0;
    }
    while tempo > max_bpm {
        tempo /= 2.0;
    }
    Ok(tempo)
}

/// Compute local beat score via Gaussian convolution.
fn beat_local_score(oenv: ArrayView1<Float>, frames_per_beat: usize) -> Array1<Float> {
    let n = oenv.len();
    let fpb = frames_per_beat as Float;

    // Build Gaussian window: exp(-0.5 * (lag * 32 / fpb)^2)
    let half_win = frames_per_beat;
    let win_len = 2 * half_win + 1;
    let window: Vec<Float> = (0..win_len)
        .map(|i| {
            let lag = (i as Float - half_win as Float) * 32.0 / fpb;
            (-0.5 * lag * lag).exp()
        })
        .collect();
    let win_sum: Float = window.iter().sum();

    // Convolve (same mode)
    let mut score = Array1::<Float>::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..win_len {
            let idx = i as i64 + j as i64 - half_win as i64;
            if idx >= 0 && (idx as usize) < n {
                sum += oenv[idx as usize] * window[j];
            }
        }
        score[i] = sum / win_sum;
    }

    score
}

/// Dynamic programming beat tracker.
///
/// Scoring: cumscore[i] = localscore[i] + max_j(cumscore[j] - tightness * (log(i-j) - log(fpb))^2)
/// where j ranges over [i - 2*fpb, i - fpb/2]
fn beat_track_dp(
    local_score: ArrayView1<Float>,
    frames_per_beat: usize,
    tightness: Float,
) -> Vec<usize> {
    let n = local_score.len();
    if n == 0 {
        return vec![];
    }

    let fpb = frames_per_beat;
    let log_fpb = (fpb as Float).ln();

    // Pre-compute ln() lookup table to avoid transcendental calls in the inner loop.
    // Intervals range from fpb/2 to 2*fpb, so we need ln(1) through ln(2*fpb).
    let max_interval = 2 * fpb + 1;
    let ln_table: Vec<Float> = (0..=max_interval)
        .map(|i| if i == 0 { 0.0 } else { (i as Float).ln() })
        .collect();

    let mut cumscore = vec![0.0_f32; n];
    let mut backlink = vec![0usize; n];

    // Forward pass — every frame gets a backlink to its best predecessor.
    // No score threshold here; trimming happens post-hoc (Ellis 2007).
    for i in 0..n {
        // Search backwards for best predecessor
        let search_start = i.saturating_sub(2 * fpb);
        let search_end = i.saturating_sub(fpb / 2);

        let mut best_score = Float::NEG_INFINITY;
        let mut best_j = 0usize;

        for j in search_start..search_end.min(i) {
            let interval = i - j;
            let ln_interval = if interval <= max_interval {
                ln_table[interval]
            } else {
                (interval as Float).ln()
            };
            let penalty = tightness * (ln_interval - log_fpb).powi(2);
            let score = cumscore[j] - penalty;
            if score > best_score {
                best_score = score;
                best_j = j;
            }
        }

        if best_score > Float::NEG_INFINITY {
            cumscore[i] = local_score[i] + best_score;
            backlink[i] = best_j;
        } else {
            cumscore[i] = local_score[i];
            backlink[i] = i; // self-link (no predecessor found yet)
        }
    }

    // Find last beat: highest cumscore in the tail region
    let tail_start = n.saturating_sub(2 * fpb);
    let mut last_beat = tail_start;
    let mut best = Float::NEG_INFINITY;
    for i in tail_start..n {
        if cumscore[i] > best {
            best = cumscore[i];
            last_beat = i;
        }
    }

    // Backtrack
    let mut beats = vec![last_beat];
    let mut current = last_beat;
    while backlink[current] != current && backlink[current] < current {
        current = backlink[current];
        beats.push(current);
    }

    beats.reverse();
    beats
}

/// Trim beats with low onset strength at the edges.
fn trim_beats(local_score: ArrayView1<Float>, beats: &[usize]) -> Vec<usize> {
    if beats.is_empty() {
        return vec![];
    }

    let threshold = 0.01 * local_score.iter().copied().fold(0.0_f32, Float::max);

    let start = beats
        .iter()
        .position(|&b| b < local_score.len() && local_score[b] >= threshold)
        .unwrap_or(0);

    let end = beats
        .iter()
        .rposition(|&b| b < local_score.len() && local_score[b] >= threshold)
        .unwrap_or(beats.len() - 1);

    beats[start..=end].to_vec()
}

/// Predominant Local Pulse.
///
/// Estimates a pulse curve from the onset envelope using Fourier tempogram.
pub fn plp(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    tempo_min: Float,
    tempo_max: Float,
) -> Result<Array1<Float>> {
    let oenv = onset::onset_strength(y, sr, hop_length)?;
    let n = oenv.len();

    if n < 4 {
        return Ok(Array1::zeros(n));
    }

    // Simple PLP: autocorrelate onset envelope in windows
    let win_length = 384.min(n);
    let acf = crate::core::audio::autocorrelate(oenv.view(), Some(win_length))?;

    // Weight by tempo range
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;
    let min_lag = (60.0 * frame_rate / tempo_max).ceil() as usize;
    let max_lag = (60.0 * frame_rate / tempo_min).floor() as usize;

    let mut pulse = Array1::<Float>::zeros(n);

    // Find peak in tempo range
    let mut best_lag = min_lag;
    let mut best_val = 0.0;
    for lag in min_lag..max_lag.min(acf.len()) {
        if acf[lag] > best_val {
            best_val = acf[lag];
            best_lag = lag;
        }
    }

    // Generate pulse at detected tempo
    if best_lag > 0 {
        let period = best_lag;
        for i in (0..n).step_by(period) {
            pulse[i] = 1.0;
        }
    }

    Ok(pulse)
}

/// Compute a per-beat tempo curve from beat frame positions.
///
/// Returns a vector of BPM values, one per inter-beat interval
/// (length = `beat_frames.len() - 1`).
///
/// - `smooth`: optional median filter window size for smoothing.
///   Use an odd number (e.g., 5) to reduce jitter.
pub fn tempo_curve(
    beat_frames: &[usize],
    sr: u32,
    hop_length: usize,
    smooth: Option<usize>,
) -> Result<Vec<Float>> {
    if beat_frames.len() < 2 {
        return Ok(vec![]);
    }

    let sr_f = sr as Float;
    let hop_f = hop_length as Float;

    // Convert frame intervals to BPM
    let mut bpms: Vec<Float> = beat_frames
        .windows(2)
        .map(|w| {
            let dt = (w[1] as Float - w[0] as Float) * hop_f / sr_f;
            if dt > 0.0 {
                60.0 / dt
            } else {
                0.0
            }
        })
        .collect();

    // Optional median filter smoothing
    if let Some(k) = smooth {
        if k >= 3 && bpms.len() >= k {
            let half = k / 2;
            let orig = bpms.clone();
            for i in half..orig.len().saturating_sub(half) {
                let mut window: Vec<Float> = orig[i - half..i + half + 1].to_vec();
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                bpms[i] = window[window.len() / 2];
            }
        }
    }

    Ok(bpms)
}

/// Compute the coefficient of variation (std/mean) of the tempo curve.
///
/// A low value (< 0.05) indicates steady tempo; a high value (> 0.1)
/// indicates significant tempo variation.
pub fn tempo_variability(tempo_curve: &[Float]) -> Float {
    if tempo_curve.is_empty() {
        return 0.0;
    }
    let n = tempo_curve.len() as Float;
    let mean = tempo_curve.iter().sum::<Float>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = tempo_curve
        .iter()
        .map(|&b| (b - mean).powi(2))
        .sum::<Float>()
        / n;
    variance.sqrt() / mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn click_train(sr: u32, dur: Float, bpm: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        let interval = (60.0 / bpm * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        y
    }

    #[test]
    fn test_beat_track_clicks() {
        let y = click_train(22050, 4.0, 120.0);
        let (tempo, beats) =
            beat_track(Some(y.view()), None, 22050, 512, 120.0, 100.0, true).unwrap();
        assert!(
            tempo > 80.0 && tempo < 180.0,
            "tempo {tempo} should be near 120"
        );
        assert!(beats.len() >= 3, "expected >=3 beats, got {}", beats.len());
    }

    #[test]
    fn test_beat_track_tempo() {
        let y = click_train(22050, 4.0, 120.0);
        let (tempo, _) = beat_track(Some(y.view()), None, 22050, 512, 120.0, 100.0, true).unwrap();
        assert!(
            (tempo - 120.0).abs() < 30.0,
            "tempo {tempo} should be ~120 BPM"
        );
    }

    #[test]
    fn test_tempo_candidate_selection_lifts_supported_half_tempo() {
        let selected = select_preferred_tempo_candidate(&[
            (41, 63.024010, 11.464),
            (31, 83.354332, 9.647),
            (21, 123.046875, 7.414),
            (20, 129.199219, 7.407),
        ])
        .unwrap();
        assert_eq!(selected.0, 21);
    }

    #[test]
    fn test_tempo_candidate_selection_lifts_supported_three_halves_tempo() {
        let selected = select_preferred_tempo_candidate(&[
            (29, 89.102913, 9.025),
            (19, 135.999176, 8.613),
            (39, 66.256012, 7.820),
            (20, 129.199219, 6.865),
        ])
        .unwrap();
        assert_eq!(selected.0, 19);
    }

    #[test]
    fn test_tempo_candidate_selection_keeps_weakly_supported_true_90_bpm() {
        let selected = select_preferred_tempo_candidate(&[
            (29, 89.102913, 2.050),
            (20, 129.199219, 1.338),
            (19, 135.999176, 0.939),
            (39, 66.256012, 0.747),
        ])
        .unwrap();
        assert_eq!(selected.0, 29);
    }

    #[test]
    fn test_tempo_candidate_selection_keeps_low_confidence_92_bpm() {
        let selected = select_preferred_tempo_candidate(&[
            (28, 92.285156, 2.170),
            (37, 69.837418, 2.145),
            (18, 143.554688, 2.144),
            (19, 135.999176, 2.082),
        ])
        .unwrap();
        assert_eq!(selected.0, 28);
    }

    #[test]
    fn test_tempo_refinement_uses_fractional_acf_peak() {
        let mut acf = Array1::<Float>::zeros(24);
        acf[19] = 7.0;
        acf[20] = 10.0;
        acf[21] = 10.0;

        let frame_rate = 22050.0 / 512.0;
        let refined = refine_tempo_from_acf_peak(acf.view(), 20, frame_rate);

        let expected = 60.0 * frame_rate / 20.5;
        assert!(
            (refined - expected).abs() < 1e-5,
            "expected fractional-lag tempo {expected}, got {refined}"
        );
    }

    #[test]
    fn test_align_tempo_to_bpm_range_doubles_low_values() {
        assert_eq!(
            align_tempo_to_bpm_range(63.02401, Some(79.0), Some(192.0)).unwrap(),
            126.04802
        );
        assert_eq!(
            align_tempo_to_bpm_range(66.25601, Some(79.0), Some(192.0)).unwrap(),
            132.51202
        );
    }

    #[test]
    fn test_align_tempo_to_bpm_range_halves_high_values() {
        assert!(
            (align_tempo_to_bpm_range(250.0, Some(79.0), Some(192.0)).unwrap() - 125.0).abs()
                < 1e-6
        );
        assert!(
            (align_tempo_to_bpm_range(401.0, Some(79.0), Some(192.0)).unwrap() - 100.25).abs()
                < 1e-6
        );
    }

    #[test]
    fn test_align_tempo_to_bpm_range_requires_complete_valid_range() {
        assert!(align_tempo_to_bpm_range(120.0, None, None).is_ok());
        assert!(align_tempo_to_bpm_range(120.0, Some(79.0), None).is_err());
        assert!(align_tempo_to_bpm_range(120.0, None, Some(192.0)).is_err());
        assert!(align_tempo_to_bpm_range(120.0, Some(192.0), Some(79.0)).is_err());
        assert!(align_tempo_to_bpm_range(120.0, Some(100.0), Some(150.0)).is_err());
    }

    #[test]
    fn test_beat_track_flat_onset_envelope_falls_back_without_beats() {
        let env = Array1::<Float>::zeros(128);
        let (tempo, beats) =
            beat_track(None, Some(env.view()), 22050, 512, 120.0, 100.0, true).unwrap();
        assert!(
            (tempo - 120.0).abs() < 1e-6,
            "flat onset envelope should fall back to start BPM, got {tempo}"
        );
        assert!(
            beats.is_empty(),
            "flat onset envelope should not produce beats, got {}",
            beats.len()
        );
    }

    #[test]
    fn test_beat_track_silence() {
        let y = Array1::<Float>::zeros(44100);
        let (_, beats) = beat_track(Some(y.view()), None, 22050, 512, 120.0, 100.0, true).unwrap();
        // DP tracker may find some beats in silence; the important thing is it doesn't crash
        // and produces fewer beats than a click train would
        assert!(beats.len() < 50, "silence produced {} beats", beats.len());
    }

    #[test]
    fn test_plp_basic() {
        let y = click_train(22050, 2.0, 120.0);
        let pulse = plp(y.view(), 22050, 512, 30.0, 300.0).unwrap();
        assert!(pulse.len() > 0);
    }

    #[test]
    fn test_tempo_curve_steady() {
        // Steady 120 BPM → each beat every ~43 frames at sr=22050, hop=512
        let frames_per_beat = (60.0_f32 / 120.0 * 22050.0 / 512.0).round() as usize;
        let beats: Vec<usize> = (0..10).map(|i| i * frames_per_beat).collect();
        let curve = tempo_curve(&beats, 22050, 512, None).unwrap();
        assert_eq!(curve.len(), 9);
        for &bpm in &curve {
            assert!((bpm - 120.0).abs() < 5.0, "expected ~120 BPM, got {bpm}");
        }
        let var = tempo_variability(&curve);
        assert!(
            var < 0.01,
            "steady tempo should have low variability, got {var}"
        );
    }

    #[test]
    fn test_tempo_curve_empty() {
        let curve = tempo_curve(&[], 22050, 512, None).unwrap();
        assert!(curve.is_empty());
        let curve = tempo_curve(&[10], 22050, 512, None).unwrap();
        assert!(curve.is_empty());
    }
}
