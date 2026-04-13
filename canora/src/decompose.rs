//! Source separation and matrix decomposition.
//!
//! Mirrors librosa.decompose — hpss, decompose (NMF), nn_filter.

use ndarray::{Array2, ArrayView2};

use crate::error::Result;
use crate::types::*;

/// Harmonic-Percussive Source Separation on a spectrogram.
///
/// Applies median filtering along time (harmonic) and frequency (percussive) axes,
/// then uses soft masking to separate the two components.
///
/// Returns `(harmonic, percussive)` spectrograms.
pub fn hpss(
    s: ArrayView2<Float>,
    kernel_size: usize,
    margin: Float,
    power: Float,
) -> Result<(Array2<Float>, Array2<Float>)> {
    let n_bins = s.nrows();
    let n_frames = s.ncols();

    // Median filter along time axis → harmonic estimate
    let harmonic_med = median_filter_2d(s, 1, kernel_size);
    // Median filter along frequency axis → percussive estimate
    let percussive_med = median_filter_2d(s, kernel_size, 1);

    // Soft masking
    let margin_pow = margin.powf(power);
    let mut harmonic = Array2::<Float>::zeros((n_bins, n_frames));
    let mut percussive = Array2::<Float>::zeros((n_bins, n_frames));

    for i in 0..n_bins {
        for j in 0..n_frames {
            let h = harmonic_med[(i, j)].powf(power) * margin_pow;
            let p = percussive_med[(i, j)].powf(power) * margin_pow;
            let total = h + p;
            if total > 0.0 {
                harmonic[(i, j)] = s[(i, j)] * h / total;
                percussive[(i, j)] = s[(i, j)] * p / total;
            }
        }
    }

    Ok((harmonic, percussive))
}

/// Non-negative Matrix Factorization.
///
/// Factorizes `V ≈ W @ H` where W ≥ 0, H ≥ 0 using multiplicative updates.
/// Returns `(W, H)` with shapes `(n_bins, n_components)` and `(n_components, n_frames)`.
pub fn decompose_nmf(
    v: ArrayView2<Float>,
    n_components: usize,
    n_iter: usize,
) -> Result<(Array2<Float>, Array2<Float>)> {
    let n_bins = v.nrows();
    let n_frames = v.ncols();

    // Initialize with random values
    let mut rng: u64 = 42;
    let mut next_rand = || -> Float {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as Float / u64::MAX as Float) * 0.1 + 0.01
    };

    let mut w = Array2::from_shape_fn((n_bins, n_components), |_| next_rand());
    let mut h = Array2::from_shape_fn((n_components, n_frames), |_| next_rand());

    let eps = 1e-10;

    for _ in 0..n_iter {
        // Update H: H *= (W^T @ V) / (W^T @ W @ H + eps)
        let wt_v = w.t().dot(&v);
        let wt_w_h = w.t().dot(&w).dot(&h);
        for i in 0..n_components {
            for j in 0..n_frames {
                h[(i, j)] *= wt_v[(i, j)] / (wt_w_h[(i, j)] + eps);
            }
        }

        // Update W: W *= (V @ H^T) / (W @ H @ H^T + eps)
        let v_ht = v.dot(&h.t());
        let w_h_ht = w.dot(&h).dot(&h.t());
        for i in 0..n_bins {
            for j in 0..n_components {
                w[(i, j)] *= v_ht[(i, j)] / (w_h_ht[(i, j)] + eps);
            }
        }
    }

    Ok((w, h))
}

/// Nearest-neighbor filtering.
///
/// Smooths a spectrogram by replacing each point with the average of its k nearest neighbors.
pub fn nn_filter(
    s: ArrayView2<Float>,
    k: usize,
    metric: &str,
) -> Result<Array2<Float>> {
    let n_bins = s.nrows();
    let n_frames = s.ncols();
    let mut result = Array2::<Float>::zeros((n_bins, n_frames));

    // For each frame, find k nearest neighbors and average
    for t in 0..n_frames {
        // Compute distance to all other frames
        let mut distances: Vec<(usize, Float)> = (0..n_frames)
            .filter(|&t2| t2 != t)
            .map(|t2| {
                let dist = match metric {
                    "cosine" => {
                        let dot: Float = (0..n_bins).map(|i| s[(i, t)] * s[(i, t2)]).sum();
                        let n1: Float = (0..n_bins).map(|i| s[(i, t)].powi(2)).sum::<Float>().sqrt();
                        let n2: Float = (0..n_bins).map(|i| s[(i, t2)].powi(2)).sum::<Float>().sqrt();
                        1.0 - dot / (n1 * n2 + 1e-10)
                    }
                    _ => { // euclidean
                        (0..n_bins).map(|i| (s[(i, t)] - s[(i, t2)]).powi(2)).sum::<Float>().sqrt()
                    }
                };
                (t2, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let k_actual = k.min(distances.len());

        // Average the k nearest neighbors
        for i in 0..n_bins {
            let mut sum = s[(i, t)]; // include self
            for nn in 0..k_actual {
                sum += s[(i, distances[nn].0)];
            }
            result[(i, t)] = sum / (k_actual + 1) as Float;
        }
    }

    Ok(result)
}

/// 2D median filter (separable: row-wise then column-wise, or single-axis).
fn median_filter_2d(s: ArrayView2<Float>, row_size: usize, col_size: usize) -> Array2<Float> {
    let n_rows = s.nrows();
    let n_cols = s.ncols();
    let mut result = s.to_owned();

    // Filter along columns (time axis) if col_size > 1
    if col_size > 1 {
        let half = col_size / 2;
        let mut temp = Array2::<Float>::zeros((n_rows, n_cols));
        for i in 0..n_rows {
            for j in 0..n_cols {
                let lo = j.saturating_sub(half);
                let hi = (j + half + 1).min(n_cols);
                let mut vals: Vec<Float> = (lo..hi).map(|k| result[(i, k)]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                temp[(i, j)] = vals[vals.len() / 2];
            }
        }
        result = temp;
    }

    // Filter along rows (frequency axis) if row_size > 1
    if row_size > 1 {
        let half = row_size / 2;
        let mut temp = Array2::<Float>::zeros((n_rows, n_cols));
        for i in 0..n_rows {
            for j in 0..n_cols {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(n_rows);
                let mut vals: Vec<Float> = (lo..hi).map(|k| result[(k, j)]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                temp[(i, j)] = vals[vals.len() / 2];
            }
        }
        result = temp;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hpss_shape() {
        let s = Array2::from_shape_fn((128, 50), |(i, j)| (i + j) as Float * 0.01);
        let (h, p) = hpss(s.view(), 31, 1.0, 2.0).unwrap();
        assert_eq!(h.shape(), s.shape());
        assert_eq!(p.shape(), s.shape());
    }

    #[test]
    fn test_hpss_energy_conservation() {
        let s = Array2::from_shape_fn((64, 30), |(i, j)| ((i * j) as Float * 0.1).sin().abs() + 0.01);
        let (h, p) = hpss(s.view(), 11, 1.0, 2.0).unwrap();
        // h + p should approximately equal s
        for i in 0..s.nrows() {
            for j in 0..s.ncols() {
                let sum = h[(i, j)] + p[(i, j)];
                assert!((sum - s[(i, j)]).abs() < s[(i, j)] * 0.5 + 0.01,
                    "h+p should be close to s at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_decompose_nmf_nonneg() {
        let v = Array2::from_shape_fn((20, 10), |(i, j)| (i + j + 1) as Float * 0.1);
        let (w, h) = decompose_nmf(v.view(), 4, 50).unwrap();
        assert_eq!(w.shape(), &[20, 4]);
        assert_eq!(h.shape(), &[4, 10]);
        for &val in w.iter() { assert!(val >= 0.0); }
        for &val in h.iter() { assert!(val >= 0.0); }
    }

    #[test]
    fn test_decompose_nmf_reconstruction() {
        let v = Array2::from_shape_fn((20, 10), |(i, j)| (i + j + 1) as Float * 0.1);
        let (w, h) = decompose_nmf(v.view(), 4, 100).unwrap();
        let recon = w.dot(&h);
        let error: Float = (&v - &recon).mapv(|x| x.powi(2)).sum();
        let energy: Float = v.mapv(|x| x.powi(2)).sum();
        assert!(error / energy < 0.5, "reconstruction error too high: {}", error / energy);
    }

    #[test]
    fn test_nn_filter_shape() {
        let s = Array2::from_shape_fn((32, 20), |(i, j)| (i + j) as Float * 0.01);
        let filtered = nn_filter(s.view(), 3, "euclidean").unwrap();
        assert_eq!(filtered.shape(), s.shape());
    }
}
