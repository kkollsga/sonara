//! Onset detection.
//!
//! Mirrors librosa.onset — onset_detect, onset_strength, onset_strength_multi, onset_backtrack.
//! Computes spectral flux from mel spectrogram to detect note onsets.

use ndarray::{Array1, Array2, ArrayView1};

use crate::core::spectrum;
use crate::error::{CanoraError, Result};
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
            let y = y.ok_or(CanoraError::InvalidParameter {
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
    let max_val = oenv.iter().copied().fold(0.0_f64, Float::max);
    let oenv_norm = if max_val > 0.0 {
        oenv.mapv(|v| v / max_val)
    } else {
        return Ok(vec![]);
    };

    // Peak picking with librosa-compatible defaults
    let sr_f = sr as Float;
    let pre_max = ((0.03 * sr_f / hop_length as Float) as usize).max(1);
    let post_max = 1;
    let pre_avg = ((0.10 * sr_f / hop_length as Float) as usize).max(1);
    let post_avg = ((0.10 * sr_f / hop_length as Float) as usize + 1).max(1);
    let wait_frames = if wait > 0 { wait } else {
        ((0.03 * sr_f / hop_length as Float) as usize).max(1)
    };

    let peaks = utils::peak_pick(oenv_norm.view(), pre_max, post_max, pre_avg, post_avg, delta, wait_frames);

    if backtrack && !peaks.is_empty() {
        Ok(onset_backtrack(&peaks, oenv.view()))
    } else {
        Ok(peaks)
    }
}

/// Compute onset strength envelope.
///
/// Wrapper for onset_strength_multi that returns a single-channel envelope.
pub fn onset_strength(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
) -> Result<Array1<Float>> {
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
        Some(y), None, sr_f, n_fft, hop_length, 128, 0.0, sr_f / 2.0, 2.0,
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
                    let max_val = (lo..hi).map(|k| s_db[(k, t)]).fold(Float::NEG_INFINITY, Float::max);
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

/// Backtrack onset events to the nearest preceding energy minimum.
pub fn onset_backtrack(events: &[usize], energy: ArrayView1<Float>) -> Vec<usize> {
    let mut backtracked = Vec::with_capacity(events.len());

    for &event in events {
        let mut best = event;
        let mut best_val = energy.get(event).copied().unwrap_or(Float::INFINITY);

        // Search backwards for minimum
        let search_start = if backtracked.is_empty() { 0 } else { *backtracked.last().unwrap() };
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
    use std::f64::consts::PI;

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
        assert!(onsets.len() >= 2, "expected >=2 onsets, got {}", onsets.len());
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
}
