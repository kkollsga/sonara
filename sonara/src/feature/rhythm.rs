//! Rhythm features — tempogram, tempo estimation.
//!
//! Tempogram, Fourier tempogram, tempo estimation, tempogram ratio, metrogram,
//! and time signature detection.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex;
#[cfg(test)]
use std::f32::consts::PI;

use crate::core::{audio, fft};
use crate::dsp::windows;
use crate::error::{Result, SonaraError};
use crate::onset;
use crate::types::*;
use crate::util::utils;

/// Compute autocorrelation tempogram.
///
/// Returns shape `(win_length, n_frames)` — lag × time.
pub fn tempogram(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<Float>> {
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "provide y or onset_envelope".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    let n = oenv.len();
    if n < win_length {
        return Err(SonaraError::InsufficientData {
            needed: win_length,
            got: n,
        });
    }

    // Frame the onset envelope
    let frames = utils::frame(oenv.view(), win_length, 1)?;
    let n_frames = frames.ncols();

    // Window
    let window = windows::hann(win_length, true);

    let mut tgram = Array2::<Float>::zeros((win_length, n_frames));

    for t in 0..n_frames {
        // Window the frame
        let mut windowed = vec![0.0; win_length];
        for i in 0..win_length {
            windowed[i] = frames[(i, t)] * window[i];
        }

        // Autocorrelate
        let windowed_arr = Array1::from_vec(windowed);
        let acf = audio::autocorrelate(windowed_arr.view(), Some(win_length))?;

        // Normalize
        let max_val = acf.iter().copied().fold(0.0_f32, Float::max).max(1e-10);
        for i in 0..win_length {
            tgram[(i, t)] = acf[i] / max_val;
        }
    }

    Ok(tgram)
}

/// Compute Fourier tempogram.
///
/// Returns complex-valued shape `(win_length/2 + 1, n_frames)`.
pub fn fourier_tempogram(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<ComplexFloat>> {
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "provide y or onset_envelope".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    let n = oenv.len();
    if n < win_length {
        return Err(SonaraError::InsufficientData {
            needed: win_length,
            got: n,
        });
    }

    let frames = utils::frame(oenv.view(), win_length, 1)?;
    let n_frames = frames.ncols();
    let n_bins = win_length / 2 + 1;

    let window = windows::hann(win_length, true);

    let mut ftgram = Array2::<ComplexFloat>::zeros((n_bins, n_frames));

    let mut fft_in = vec![0.0_f32; win_length];
    let mut fft_out = vec![Complex::new(0.0, 0.0); n_bins];

    for t in 0..n_frames {
        for i in 0..win_length {
            fft_in[i] = frames[(i, t)] * window[i];
        }
        fft::rfft(&mut fft_in, &mut fft_out)?;
        for i in 0..n_bins {
            ftgram[(i, t)] = fft_out[i];
        }
    }

    Ok(ftgram)
}

/// Estimate tempo from onset envelope.
///
/// Returns estimated tempo in BPM.
pub fn tempo(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    max_tempo: Float,
) -> Result<Float> {
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "provide y or onset_envelope".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    // Compute tempogram
    let win_length = 384.min(oenv.len());
    if win_length < 4 {
        return Ok(start_bpm);
    }

    let tg = tempogram(None, Some(oenv.view()), sr, hop_length, win_length)?;

    // Aggregate across time (mean)
    let avg: Array1<Float> = tg
        .mean_axis(ndarray::Axis(1))
        .unwrap_or(Array1::zeros(win_length));

    // Convert lag to BPM and find peak
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;

    let mut best_bpm = start_bpm;
    let mut best_score = Float::NEG_INFINITY;

    for lag in 1..avg.len() {
        let bpm = 60.0 * frame_rate / lag as Float;
        if bpm < 30.0 || bpm > max_tempo {
            continue;
        }

        // Log-normal prior centered at start_bpm
        let log_prior = -0.5 * ((bpm.log2() - start_bpm.log2()) / 1.0).powi(2);
        let score = avg[lag] * (1.0 + log_prior.exp());

        if score > best_score {
            best_score = score;
            best_bpm = bpm;
        }
    }

    Ok(best_bpm)
}

/// Compute tempogram ratio features.
///
/// Ratio of energy at different tempo multiples.
pub fn tempogram_ratio(
    tg: ArrayView2<Float>,
    _sr: u32,
    _hop_length: usize,
) -> Result<Array2<Float>> {
    let n_lags = tg.nrows();
    let n_frames = tg.ncols();

    // Compute ratio at 1x, 2x, 3x tempo
    let mut ratios = Array2::<Float>::zeros((3, n_frames));

    for t in 0..n_frames {
        // Find peak lag
        let mut peak_lag = 1;
        let mut peak_val = 0.0;
        for lag in 1..n_lags {
            if tg[(lag, t)] > peak_val {
                peak_val = tg[(lag, t)];
                peak_lag = lag;
            }
        }

        if peak_val > 0.0 {
            ratios[(0, t)] = tg[(peak_lag, t)] / peak_val;
            if peak_lag * 2 < n_lags {
                ratios[(1, t)] = tg[(peak_lag * 2, t)] / peak_val;
            }
            if peak_lag * 3 < n_lags {
                ratios[(2, t)] = tg[(peak_lag * 3, t)] / peak_val;
            }
        }
    }

    Ok(ratios)
}

// ============================================================
// Time signature detection (metrogram)
// ============================================================

/// Default subharmonic ratios mapping to common time signatures.
///
/// `[0.5, 1/3, 0.25, 0.2, 1/6, 1/7]` → meters `[2, 3, 4, 5, 6, 7]`
const DEFAULT_METER_RATIOS: &[Float] = &[
    0.5,       // 2/4
    1.0 / 3.0, // 3/4
    0.25,      // 4/4
    0.2,       // 5/4
    1.0 / 6.0, // 6/4 (or 6/8)
    1.0 / 7.0, // 7/4
];

/// Numerators corresponding to each ratio in `DEFAULT_METER_RATIOS`.
const METER_NUMERATORS: &[usize] = &[2, 3, 4, 5, 6, 7];

/// Compute a metrogram — a time-varying meter estimate.
///
/// The metrogram measures the strength of different metric subdivisions
/// (2, 3, 4, 5, 6, 7 beats per bar) over time by analysing subharmonic
/// ratios in the Fourier tempogram.
///
/// Returns shape `(n_ratios, n_frames)` where each row is the meter
/// strength for that ratio over time.
pub fn metrogram(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    win_length: usize,
    ratios: Option<&[Float]>,
) -> Result<Array2<Float>> {
    let ratios = ratios.unwrap_or(DEFAULT_METER_RATIOS);
    let n_ratios = ratios.len();

    // Compute Fourier tempogram magnitude
    let ft = fourier_tempogram(y, onset_envelope, sr, hop_length, win_length)?;
    let n_bins = ft.nrows();
    let n_frames = ft.ncols();
    let ft_mag = ft.mapv(|c| c.norm());

    // Compute Fourier tempogram frequencies (BPM)
    let freqs =
        crate::core::convert::fourier_tempo_frequencies(sr as Float, win_length, hop_length);

    // Interpolate at subharmonic frequencies and compute product
    let mut meter_scores = Array2::<Float>::zeros((n_ratios, n_frames));

    for (ri, &ratio) in ratios.iter().enumerate() {
        for t in 0..n_frames {
            let mut score = 0.0;
            for b in 1..n_bins {
                let target_freq = freqs[b] * ratio;
                if target_freq <= 0.0 {
                    continue;
                }

                // Linear interpolation in the frequency axis
                let mut lo = 0;
                for k in 0..n_bins {
                    if freqs[k] <= target_freq {
                        lo = k;
                    }
                }
                let hi = (lo + 1).min(n_bins - 1);

                let interp_val = if hi > lo && freqs[hi] > freqs[lo] {
                    let frac =
                        ((target_freq - freqs[lo]) / (freqs[hi] - freqs[lo])).clamp(0.0, 1.0);
                    (1.0 - frac) * ft_mag[(lo, t)] + frac * ft_mag[(hi, t)]
                } else {
                    ft_mag[(lo, t)]
                };

                // Product: original magnitude × subharmonic magnitude
                score += ft_mag[(b, t)] * interp_val;
            }
            meter_scores[(ri, t)] = score;
        }
    }

    // Normalize each frame by its max value (column-wise)
    for t in 0..n_frames {
        let max_val = (0..n_ratios)
            .map(|r| meter_scores[(r, t)])
            .fold(0.0_f32, Float::max);
        if max_val > 0.0 {
            for r in 0..n_ratios {
                meter_scores[(r, t)] /= max_val;
            }
        }
    }

    Ok(meter_scores)
}

/// Detect time signature from a metrogram.
///
/// Aggregates the metrogram over time and picks the dominant meter.
/// Returns `(label, confidence)` e.g. `("4/4", 0.85)`.
pub fn detect_time_signature(
    metrogram: ArrayView2<Float>,
    ratios: Option<&[Float]>,
) -> (String, Float) {
    let nums = if ratios.is_some() {
        // If custom ratios, just use indices
        (0..metrogram.nrows()).collect::<Vec<_>>()
    } else {
        METER_NUMERATORS.to_vec()
    };

    if metrogram.ncols() == 0 || metrogram.nrows() == 0 {
        return ("4/4".to_string(), 0.0);
    }

    // Aggregate: mean score per ratio across all frames
    let n_ratios = metrogram.nrows();
    let mut scores = vec![0.0_f32; n_ratios];
    for r in 0..n_ratios {
        scores[r] = metrogram.row(r).sum() / metrogram.ncols() as Float;
    }

    // Find best and second-best
    let mut best_idx = 0;
    let mut best_score = scores[0];
    let mut second_score = 0.0_f32;
    for (i, &s) in scores.iter().enumerate() {
        if s > best_score {
            second_score = best_score;
            best_score = s;
            best_idx = i;
        } else if s > second_score {
            second_score = s;
        }
    }

    let confidence = if best_score > 0.0 {
        ((best_score - second_score) / best_score).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let numerator = if best_idx < nums.len() {
        nums[best_idx]
    } else {
        4
    };
    let label = format!("{}/4", numerator);

    (label, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_tempogram_shape() {
        let y = click_train(22050, 3.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let tg = tempogram(None, Some(oenv.view()), 22050, 512, 384.min(oenv.len())).unwrap();
        assert!(tg.nrows() > 0);
        assert!(tg.ncols() > 0);
    }

    #[test]
    fn test_tempo_120bpm() {
        let y = click_train(22050, 4.0, 120.0);
        let t = tempo(Some(y.view()), None, 22050, 512, 120.0, 320.0).unwrap();
        assert!(t > 80.0 && t < 180.0, "tempo {t} should be near 120");
    }

    #[test]
    fn test_fourier_tempogram_shape() {
        let y = click_train(22050, 3.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let win = 384.min(oenv.len());
        let ft = fourier_tempogram(None, Some(oenv.view()), 22050, 512, win).unwrap();
        assert_eq!(ft.nrows(), win / 2 + 1);
    }

    #[test]
    fn test_tempogram_ratio_shape() {
        let y = click_train(22050, 3.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let tg = tempogram(None, Some(oenv.view()), 22050, 512, 384.min(oenv.len())).unwrap();
        let ratios = tempogram_ratio(tg.view(), 22050, 512).unwrap();
        assert_eq!(ratios.nrows(), 3);
    }

    #[test]
    fn test_metrogram_shape() {
        let y = click_train(22050, 4.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let win = 384.min(oenv.len());
        let mg = metrogram(None, Some(oenv.view()), 22050, 512, win, None).unwrap();
        assert_eq!(mg.nrows(), DEFAULT_METER_RATIOS.len());
        assert!(mg.ncols() > 0);
    }

    #[test]
    fn test_detect_time_signature_basic() {
        let y = click_train(22050, 4.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let win = 384.min(oenv.len());
        let mg = metrogram(None, Some(oenv.view()), 22050, 512, win, None).unwrap();
        let (label, confidence) = detect_time_signature(mg.view(), None);
        // A simple click train is ambiguous in meter, but should return a valid label
        assert!(
            label.contains('/'),
            "label should be N/4 format, got {label}"
        );
        assert!(
            confidence >= 0.0 && confidence <= 1.0,
            "confidence {confidence} out of range"
        );
    }
}
