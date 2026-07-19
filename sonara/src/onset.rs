//! Onset detection.
//!
//! Onset detection via spectral flux, energy, phase, and complex-domain methods.
//! Includes onset_detect, onset_strength, onset_strength_multi, and onset_backtrack.

use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex;

use crate::core::spectrum;
use crate::error::{Result, SonaraError};
use crate::feature::spectral as feat;
use crate::types::*;
use crate::util::utils;

/// Detect onset events from an audio signal.
///
/// Returns frame indices where onsets occur.
pub fn onset_detect(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    backtrack: bool,
    delta: Float,
    wait: usize,
) -> Result<Vec<usize>> {
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "either y or onset_envelope must be provided".into(),
            })?;
            onset_strength(y, sr, hop_length)?
        }
    };

    if oenv.is_empty() {
        return Ok(vec![]);
    }

    // Normalize to [0, 1]
    let max_val = oenv.iter().copied().fold(0.0_f32, Float::max);
    let oenv_norm = if max_val > 0.0 {
        oenv.mapv(|v| v / max_val)
    } else {
        return Ok(vec![]);
    };

    // Peak picking with standard defaults
    let sr_f = sr as Float;
    let pre_max = ((0.03 * sr_f / hop_length as Float) as usize).max(1);
    let post_max = 1;
    let pre_avg = ((0.10 * sr_f / hop_length as Float) as usize).max(1);
    let post_avg = ((0.10 * sr_f / hop_length as Float) as usize + 1).max(1);
    let wait_frames = if wait > 0 {
        wait
    } else {
        ((0.03 * sr_f / hop_length as Float) as usize).max(1)
    };

    let peaks = utils::peak_pick(
        oenv_norm.view(),
        pre_max,
        post_max,
        pre_avg,
        post_avg,
        delta,
        wait_frames,
    );

    if backtrack && !peaks.is_empty() {
        Ok(onset_backtrack(&peaks, oenv.view()))
    } else {
        Ok(peaks)
    }
}

/// Compute onset strength envelope.
///
/// Wrapper for onset_strength_multi that returns a single-channel envelope.
pub fn onset_strength(y: ArrayView1<Float>, sr: u32, hop_length: usize) -> Result<Array1<Float>> {
    let multi = onset_strength_multi(y, sr, hop_length, 1, None)?;
    Ok(multi.row(0).to_owned())
}

/// Compute multi-band onset strength envelope.
///
/// Algorithm: mel spectrogram → log power → spectral flux (positive first-order difference).
///
/// Returns shape `(n_channels, n_frames)`.
pub fn onset_strength_multi(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    lag: usize,
    max_size: Option<usize>,
) -> Result<Array2<Float>> {
    let sr_f = sr as Float;
    let n_fft = 2048;

    // Compute mel spectrogram in dB
    let mel = feat::melspectrogram(
        Some(y),
        None,
        sr_f,
        n_fft,
        hop_length,
        128,
        0.0,
        sr_f / 2.0,
        2.0,
    )?;
    let s_db = spectrum::power_to_db(mel.view(), 1.0, 1e-10, Some(80.0));

    let n_mels = s_db.nrows();
    let n_frames = s_db.ncols();

    if n_frames <= lag {
        return Ok(Array2::zeros((1, n_frames)));
    }

    // Reference spectrum (local max filter along frequency axis)
    let ref_spec = match max_size {
        Some(ms) if ms > 1 => {
            let mut filtered = s_db.clone();
            for t in 0..n_frames {
                for m in 0..n_mels {
                    let lo = m.saturating_sub(ms / 2);
                    let hi = (m + ms / 2 + 1).min(n_mels);
                    let max_val = (lo..hi)
                        .map(|k| s_db[(k, t)])
                        .fold(Float::NEG_INFINITY, Float::max);
                    filtered[(m, t)] = max_val;
                }
            }
            filtered
        }
        _ => s_db.clone(), // max_size=1: ref = S itself
    };

    // Spectral flux: positive first-order difference
    let out_frames = n_frames - lag;
    let mut onset_env = Array2::<Float>::zeros((1, out_frames));

    for t in 0..out_frames {
        let mut sum = 0.0;
        for m in 0..n_mels {
            let diff = s_db[(m, t + lag)] - ref_spec[(m, t)];
            sum += diff.max(0.0); // half-wave rectification
        }
        onset_env[(0, t)] = sum / n_mels as Float;
    }

    // Pad to align with STFT centering
    let pad_left = lag + n_fft / (2 * hop_length);
    let total_frames = out_frames + pad_left;
    let mut padded = Array2::<Float>::zeros((1, total_frames));
    for t in 0..out_frames {
        padded[(0, pad_left + t)] = onset_env[(0, t)];
    }

    Ok(padded)
}

// ============================================================
// Advanced onset detection methods
// ============================================================

/// Onset detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnsetMethod {
    /// Spectral flux on mel spectrogram (default).
    SpectralFlux,
    /// Energy-based: half-wave rectified power spectrum difference.
    Energy,
    /// Phase deviation weighted by magnitude (good for tonal onsets).
    Phase,
    /// Complex-domain: deviation from predicted complex spectrum.
    Complex,
}

impl OnsetMethod {
    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "spectral_flux" | "spectralflux" | "flux" => Some(Self::SpectralFlux),
            "energy" | "rms" => Some(Self::Energy),
            "phase" => Some(Self::Phase),
            "complex" => Some(Self::Complex),
            _ => None,
        }
    }
}

/// Compute onset strength envelope using a specified method.
///
/// Unlike [`onset_strength`] (which always uses spectral flux on mel),
/// this supports alternative methods: energy, phase deviation, and
/// complex-domain detection.
pub fn onset_strength_method(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    method: OnsetMethod,
) -> Result<Array1<Float>> {
    match method {
        OnsetMethod::SpectralFlux => onset_strength(y, sr, hop_length),
        OnsetMethod::Energy => onset_strength_energy(y, sr, hop_length),
        OnsetMethod::Phase => onset_strength_phase(y, sr, hop_length),
        OnsetMethod::Complex => onset_strength_complex(y, sr, hop_length),
    }
}

/// Energy-based onset strength: half-wave rectified power difference.
fn onset_strength_energy(
    y: ArrayView1<Float>,
    _sr: u32,
    hop_length: usize,
) -> Result<Array1<Float>> {
    let n_fft = 2048;
    let window = WindowSpec::Named("hann".into());
    let stft = spectrum::stft(
        y,
        n_fft,
        Some(hop_length),
        None,
        &window,
        true,
        PadMode::Constant,
    )?;
    let n_frames = stft.ncols();
    let n_bins = stft.nrows();

    if n_frames < 2 {
        return Ok(Array1::zeros(n_frames));
    }

    // Pad to align with STFT centering
    let pad_left = 1 + n_fft / (2 * hop_length);
    let out_len = n_frames - 1 + pad_left;
    let mut env = Array1::<Float>::zeros(out_len);

    for t in 1..n_frames {
        let mut diff_sum = 0.0;
        for b in 0..n_bins {
            let pow_cur = stft[(b, t)].norm_sqr();
            let pow_prev = stft[(b, t - 1)].norm_sqr();
            diff_sum += (pow_cur - pow_prev).max(0.0);
        }
        env[pad_left + t - 1] = diff_sum / n_bins as Float;
    }

    Ok(env)
}

/// Phase-based onset strength: phase deviation weighted by magnitude.
fn onset_strength_phase(
    y: ArrayView1<Float>,
    _sr: u32,
    hop_length: usize,
) -> Result<Array1<Float>> {
    let n_fft = 2048;
    let window = WindowSpec::Named("hann".into());
    let stft = spectrum::stft(
        y,
        n_fft,
        Some(hop_length),
        None,
        &window,
        true,
        PadMode::Constant,
    )?;
    let n_frames = stft.ncols();
    let n_bins = stft.nrows();

    if n_frames < 3 {
        return Ok(Array1::zeros(n_frames));
    }

    let pad_left = 2 + n_fft / (2 * hop_length);
    let out_len = n_frames - 2 + pad_left;
    let mut env = Array1::<Float>::zeros(out_len);

    // phase[t] - 2*phase[t-1] + phase[t-2], weighted by |S[t]|
    for t in 2..n_frames {
        let mut sum = 0.0;
        for b in 0..n_bins {
            let p0 = stft[(b, t - 2)].arg();
            let p1 = stft[(b, t - 1)].arg();
            let p2 = stft[(b, t)].arg();
            let mag = stft[(b, t)].norm();
            // Second-order phase deviation
            let mut phase_dev = p2 - 2.0 * p1 + p0;
            // Wrap to [-pi, pi]
            phase_dev = ((phase_dev + std::f32::consts::PI) % (2.0 * std::f32::consts::PI))
                - std::f32::consts::PI;
            sum += mag * phase_dev.abs();
        }
        env[pad_left + t - 2] = sum / n_bins as Float;
    }

    Ok(env)
}

/// Complex-domain onset strength.
fn onset_strength_complex(
    y: ArrayView1<Float>,
    _sr: u32,
    hop_length: usize,
) -> Result<Array1<Float>> {
    let n_fft = 2048;
    let window = WindowSpec::Named("hann".into());
    let stft = spectrum::stft(
        y,
        n_fft,
        Some(hop_length),
        None,
        &window,
        true,
        PadMode::Constant,
    )?;
    let n_frames = stft.ncols();
    let n_bins = stft.nrows();

    if n_frames < 3 {
        return Ok(Array1::zeros(n_frames));
    }

    let pad_left = 2 + n_fft / (2 * hop_length);
    let out_len = n_frames - 2 + pad_left;
    let mut env = Array1::<Float>::zeros(out_len);

    // Predicted spectrum: |S[t-1]| * exp(j * (2*phase[t-1] - phase[t-2]))
    for t in 2..n_frames {
        let mut sum = 0.0;
        for b in 0..n_bins {
            let p0 = stft[(b, t - 2)].arg();
            let p1 = stft[(b, t - 1)].arg();
            let mag_prev = stft[(b, t - 1)].norm();
            let predicted_phase = 2.0 * p1 - p0;
            let predicted = Complex::new(
                mag_prev * predicted_phase.cos(),
                mag_prev * predicted_phase.sin(),
            );
            let actual = stft[(b, t)];
            sum += (actual - predicted).norm();
        }
        env[pad_left + t - 2] = sum / n_bins as Float;
    }

    Ok(env)
}

/// Backtrack onset events to the nearest preceding energy minimum.
pub fn onset_backtrack(events: &[usize], energy: ArrayView1<Float>) -> Vec<usize> {
    let mut backtracked = Vec::with_capacity(events.len());

    for &event in events {
        let mut best = event;
        let mut best_val = energy.get(event).copied().unwrap_or(Float::INFINITY);

        // Search backwards for minimum
        let search_start = if backtracked.is_empty() {
            0
        } else {
            *backtracked.last().unwrap()
        };
        for i in (search_start..event).rev() {
            if let Some(&val) = energy.get(i) {
                if val < best_val {
                    best_val = val;
                    best = i;
                }
            }
        }
        backtracked.push(best);
    }

    backtracked
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: Float, sr: u32, dur: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin())
    }

    fn click_train(sr: u32, dur: Float, bpm: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        let interval = (60.0 / bpm * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            // Short click: 100 samples of 1kHz sine
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        y
    }

    #[test]
    fn test_onset_strength_shape() {
        let y = sine(440.0, 22050, 1.0);
        let env = onset_strength(y.view(), 22050, 512).unwrap();
        assert!(env.len() > 0);
    }

    #[test]
    fn test_onset_detect_clicks() {
        let y = click_train(22050, 2.0, 120.0);
        let onsets = onset_detect(Some(y.view()), None, 22050, 512, false, 0.05, 0).unwrap();
        // 120 BPM for 2s → ~4 beats → ~4 onsets
        assert!(
            onsets.len() >= 2,
            "expected >=2 onsets, got {}",
            onsets.len()
        );
    }

    #[test]
    fn test_onset_detect_silence() {
        let y = Array1::<Float>::zeros(22050);
        let onsets = onset_detect(Some(y.view()), None, 22050, 512, false, 0.07, 0).unwrap();
        assert!(onsets.is_empty(), "silence should have no onsets");
    }

    #[test]
    fn test_onset_backtrack() {
        let energy = Array1::from_vec(vec![0.5, 0.1, 0.3, 0.8, 0.2, 0.9, 0.1]);
        let events = vec![3, 5];
        let bt = onset_backtrack(&events, energy.view());
        assert!(bt[0] <= 3);
        assert!(bt[1] <= 5);
    }

    #[test]
    fn test_onset_strength_multi() {
        let y = click_train(22050, 1.0, 120.0);
        let env = onset_strength_multi(y.view(), 22050, 512, 1, None).unwrap();
        assert_eq!(env.nrows(), 1);
        assert!(env.ncols() > 0);
    }

    #[test]
    fn test_onset_strength_methods() {
        let y = click_train(22050, 2.0, 120.0);
        for method in [
            OnsetMethod::SpectralFlux,
            OnsetMethod::Energy,
            OnsetMethod::Phase,
            OnsetMethod::Complex,
        ] {
            let env = onset_strength_method(y.view(), 22050, 512, method).unwrap();
            assert!(env.len() > 0, "method {:?} produced empty envelope", method);
            // Each method should produce non-zero values for a click train
            let max = env.iter().copied().fold(0.0_f32, Float::max);
            assert!(max > 0.0, "method {:?} max={max}, expected >0", method);
        }
    }
}
