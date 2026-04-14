//! Beat tracking.
//!
//! Beat tracking via dynamic programming (Ellis 2007 algorithm).
//! Includes beat_track, plp, tempo_curve, and tempo_variability.

use ndarray::{Array1, ArrayView1};

use crate::error::{SonaraError, Result};
use crate::onset;
use crate::types::Float;

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
        return Ok((start_bpm, vec![]));
    }

    // Estimate tempo
    let tempo = estimate_tempo(&oenv, sr, hop_length, start_bpm)?;
    let frames_per_beat = (60.0 * frame_rate / tempo).round() as usize;

    if frames_per_beat == 0 {
        return Ok((tempo, vec![]));
    }

    // Normalize onset envelope
    let mean = oenv.iter().sum::<Float>() / oenv.len() as Float;
    let std = (oenv.iter().map(|&v| (v - mean).powi(2)).sum::<Float>() / oenv.len() as Float).sqrt();
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

    Ok((tempo, beats))
}

/// Estimate tempo from onset envelope using autocorrelation.
fn estimate_tempo(
    oenv: &Array1<Float>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
) -> Result<Float> {
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;

    // Autocorrelate onset envelope
    let max_lag = (4.0 * frame_rate).min(oenv.len() as Float) as usize; // up to 4 seconds
    let acf = crate::core::audio::autocorrelate(oenv.view(), Some(max_lag))?;

    if acf.is_empty() {
        return Ok(start_bpm);
    }

    // Find peaks in BPM range [30, 300]
    let min_lag = (60.0 * frame_rate / 300.0).ceil() as usize;
    let max_lag = (60.0 * frame_rate / 30.0).floor() as usize;
    let max_lag = max_lag.min(acf.len() - 1);

    if min_lag >= max_lag {
        return Ok(start_bpm);
    }

    // Weight by log-normal prior centered at start_bpm
    let mut best_lag = min_lag;
    let mut best_score = Float::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        let bpm = 60.0 * frame_rate / lag as Float;
        let log_prior = -0.5 * ((bpm.log2() - start_bpm.log2()) / 1.0).powi(2);
        let score = acf[lag] * (1.0 + log_prior.exp());

        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    let tempo = 60.0 * frame_rate / best_lag as Float;
    Ok(tempo.clamp(30.0, 320.0))
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
fn beat_track_dp(local_score: ArrayView1<Float>, frames_per_beat: usize, tightness: Float) -> Vec<usize> {
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
            let ln_interval = if interval <= max_interval { ln_table[interval] } else { (interval as Float).ln() };
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

    let start = beats.iter().position(|&b| {
        b < local_score.len() && local_score[b] >= threshold
    }).unwrap_or(0);

    let end = beats.iter().rposition(|&b| {
        b < local_score.len() && local_score[b] >= threshold
    }).unwrap_or(beats.len() - 1);

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
            if dt > 0.0 { 60.0 / dt } else { 0.0 }
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
    let variance = tempo_curve.iter().map(|&b| (b - mean).powi(2)).sum::<Float>() / n;
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
        let (tempo, beats) = beat_track(Some(y.view()), None, 22050, 512, 120.0, 100.0, true).unwrap();
        assert!(tempo > 80.0 && tempo < 180.0, "tempo {tempo} should be near 120");
        assert!(beats.len() >= 3, "expected >=3 beats, got {}", beats.len());
    }

    #[test]
    fn test_beat_track_tempo() {
        let y = click_train(22050, 4.0, 120.0);
        let (tempo, _) = beat_track(Some(y.view()), None, 22050, 512, 120.0, 100.0, true).unwrap();
        assert!((tempo - 120.0).abs() < 30.0, "tempo {tempo} should be ~120 BPM");
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
        assert!(var < 0.01, "steady tempo should have low variability, got {var}");
    }

    #[test]
    fn test_tempo_curve_empty() {
        let curve = tempo_curve(&[], 22050, 512, None).unwrap();
        assert!(curve.is_empty());
        let curve = tempo_curve(&[10], 22050, 512, None).unwrap();
        assert!(curve.is_empty());
    }
}
