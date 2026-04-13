//! Rhythm features — tempogram, tempo estimation.
//!
//! Mirrors librosa.feature.rhythm — tempogram, fourier_tempogram, tempo, tempogram_ratio.

#[cfg(test)]
use std::f64::consts::PI;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex;

use crate::core::{audio, fft};
use crate::dsp::windows;
use crate::error::{CanoraError, Result};
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
            let y = y.ok_or(CanoraError::InvalidParameter {
                param: "y",
                reason: "provide y or onset_envelope".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    let n = oenv.len();
    if n < win_length {
        return Err(CanoraError::InsufficientData { needed: win_length, got: n });
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
        let max_val = acf.iter().copied().fold(0.0_f64, Float::max).max(1e-10);
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
            let y = y.ok_or(CanoraError::InvalidParameter {
                param: "y",
                reason: "provide y or onset_envelope".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    let n = oenv.len();
    if n < win_length {
        return Err(CanoraError::InsufficientData { needed: win_length, got: n });
    }

    let frames = utils::frame(oenv.view(), win_length, 1)?;
    let n_frames = frames.ncols();
    let n_bins = win_length / 2 + 1;

    let window = windows::hann(win_length, true);

    let mut ftgram = Array2::<ComplexFloat>::zeros((n_bins, n_frames));

    let mut fft_in = vec![0.0_f64; win_length];
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
            let y = y.ok_or(CanoraError::InvalidParameter {
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
    let avg: Array1<Float> = tg.mean_axis(ndarray::Axis(1)).unwrap_or(Array1::zeros(win_length));

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
}
