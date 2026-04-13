//! Harmonic analysis — salience, interp_harmonics, f0_harmonics.
//!
//! Mirrors librosa.core.harmonic.

#[cfg(test)]
use ndarray::Array1;
use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::error::{CanoraError, Result};
use crate::types::Float;

/// Compute harmonic salience.
///
/// Sums spectrogram energy at harmonics of each frequency bin,
/// producing a pitch salience function.
///
/// - `s`: Magnitude spectrogram (n_bins, n_frames)
/// - `freqs`: Frequency array for each bin
/// - `harmonics`: Which harmonics to sum (e.g., [1, 2, 3, 4, 5])
/// - `weights`: Weight for each harmonic (None = uniform)
/// - `filter_peaks`: If true, only keep local maxima
pub fn salience(
    s: ArrayView2<Float>,
    freqs: ArrayView1<Float>,
    harmonics: &[usize],
    weights: Option<&[Float]>,
    _fill_value: Float,
) -> Result<Array2<Float>> {
    let n_bins = s.nrows();
    let n_frames = s.ncols();

    let default_weights: Vec<Float> = vec![1.0; harmonics.len()];
    let w = weights.unwrap_or(&default_weights);

    let mut sal = Array2::<Float>::zeros((n_bins, n_frames));

    for t in 0..n_frames {
        for i in 0..n_bins {
            if freqs[i] <= 0.0 {
                continue;
            }
            let mut sum = 0.0;
            for (hi, &h) in harmonics.iter().enumerate() {
                let target_freq = freqs[i] * h as Float;
                // Find nearest bin
                let bin = freqs
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        ((**a) - target_freq)
                            .abs()
                            .partial_cmp(&((**b) - target_freq).abs())
                            .unwrap()
                    })
                    .map(|(idx, _)| idx);

                if let Some(bin_idx) = bin {
                    let weight = if hi < w.len() { w[hi] } else { 1.0 };
                    sum += weight * s[(bin_idx, t)];
                }
            }
            sal[(i, t)] = sum;
        }
    }

    Ok(sal)
}

/// Interpolate energy at harmonics of given frequencies.
///
/// - `s`: Spectrogram (n_bins, n_frames)
/// - `freqs`: Frequency for each bin
/// - `harmonics`: Harmonic numbers to extract
///
/// Returns shape (n_harmonics, n_bins, n_frames) flattened to (n_harmonics * n_bins, n_frames)
/// for simplicity. Use reshape as needed.
pub fn interp_harmonics(
    s: ArrayView2<Float>,
    freqs: ArrayView1<Float>,
    harmonics: &[Float],
    fill_value: Float,
) -> Result<Array2<Float>> {
    let n_bins = s.nrows();
    let n_frames = s.ncols();
    let n_harm = harmonics.len();

    let mut result = Array2::from_elem((n_harm * n_bins, n_frames), fill_value);

    for (hi, &h) in harmonics.iter().enumerate() {
        for i in 0..n_bins {
            let target = freqs[i] * h;
            // Linear interpolation between bins
            let mut lo = 0;
            for k in 0..n_bins {
                if freqs[k] <= target {
                    lo = k;
                }
            }
            let hi_bin = (lo + 1).min(n_bins - 1);

            if lo < n_bins && hi_bin < n_bins && freqs[hi_bin] > freqs[lo] {
                let frac = (target - freqs[lo]) / (freqs[hi_bin] - freqs[lo]);
                let frac = frac.clamp(0.0, 1.0);
                for t in 0..n_frames {
                    result[(hi * n_bins + i, t)] =
                        (1.0 - frac) * s[(lo, t)] + frac * s[(hi_bin, t)];
                }
            }
        }
    }

    Ok(result)
}

/// Extract energy at harmonics of a time-varying f0.
///
/// - `s`: Spectrogram (n_bins, n_frames)
/// - `freqs`: Frequency for each bin
/// - `f0`: Fundamental frequency per frame (n_frames,)
/// - `harmonics`: Which harmonics to extract
///
/// Returns shape (n_harmonics, n_frames).
pub fn f0_harmonics(
    s: ArrayView2<Float>,
    freqs: ArrayView1<Float>,
    f0: ArrayView1<Float>,
    harmonics: &[Float],
    fill_value: Float,
) -> Result<Array2<Float>> {
    let n_bins = s.nrows();
    let n_frames = s.ncols();
    let n_harm = harmonics.len();

    if f0.len() != n_frames {
        return Err(CanoraError::ShapeMismatch {
            expected: format!("{n_frames}"),
            got: format!("{}", f0.len()),
        });
    }

    let mut result = Array2::from_elem((n_harm, n_frames), fill_value);

    for t in 0..n_frames {
        if f0[t].is_nan() || f0[t] <= 0.0 {
            continue;
        }
        for (hi, &h) in harmonics.iter().enumerate() {
            let target = f0[t] * h;
            // Find nearest bin via linear interpolation
            let mut lo = 0;
            for k in 0..n_bins {
                if freqs[k] <= target {
                    lo = k;
                }
            }
            let hi_bin = (lo + 1).min(n_bins - 1);
            if freqs[hi_bin] > freqs[lo] {
                let frac = ((target - freqs[lo]) / (freqs[hi_bin] - freqs[lo])).clamp(0.0, 1.0);
                result[(hi, t)] = (1.0 - frac) * s[(lo, t)] + frac * s[(hi_bin, t)];
            } else {
                result[(hi, t)] = s[(lo, t)];
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_salience_shape() {
        let s = Array2::from_shape_fn((128, 20), |(i, j)| (i + j) as Float * 0.01);
        let freqs = Array1::from_shape_fn(128, |i| i as Float * 100.0);
        let sal = salience(s.view(), freqs.view(), &[1, 2, 3], None, 0.0).unwrap();
        assert_eq!(sal.shape(), &[128, 20]);
    }

    #[test]
    fn test_interp_harmonics_shape() {
        let s = Array2::from_shape_fn((64, 10), |(i, j)| (i + j) as Float * 0.01);
        let freqs = Array1::from_shape_fn(64, |i| i as Float * 50.0 + 50.0);
        let result = interp_harmonics(s.view(), freqs.view(), &[1.0, 2.0, 3.0], 0.0).unwrap();
        assert_eq!(result.nrows(), 3 * 64);
        assert_eq!(result.ncols(), 10);
    }

    #[test]
    fn test_f0_harmonics_shape() {
        let s = Array2::from_shape_fn((64, 10), |(i, j)| (i + j) as Float * 0.01);
        let freqs = Array1::from_shape_fn(64, |i| i as Float * 50.0 + 50.0);
        let f0 = Array1::from_elem(10, 440.0);
        let result = f0_harmonics(s.view(), freqs.view(), f0.view(), &[1.0, 2.0, 3.0], 0.0).unwrap();
        assert_eq!(result.shape(), &[3, 10]);
    }
}
