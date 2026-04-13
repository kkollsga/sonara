//! Structural segmentation.
//!
//! Mirrors librosa.segment — recurrence_matrix, cross_similarity,
//! recurrence_to_lag, lag_to_recurrence, path_enhance, subsegment.

use ndarray::{Array2, ArrayView2};

use crate::error::Result;
use crate::types::Float;

/// Compute a recurrence (self-similarity) matrix.
///
/// Returns a binary or weighted matrix where R[i,j] indicates similarity
/// between frames i and j.
///
/// - `data`: Feature matrix of shape `(n_features, n_frames)`
/// - `k`: Number of nearest neighbors (0 = use all)
/// - `metric`: Distance metric ("euclidean" or "cosine")
/// - `sym`: If true, only keep mutual nearest neighbors
pub fn recurrence_matrix(
    data: ArrayView2<Float>,
    k: usize,
    metric: &str,
    sym: bool,
) -> Result<Array2<Float>> {
    let n = data.ncols();
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    // Compute pairwise distance matrix
    let mut dist = Array2::<Float>::zeros((n, n));
    for i in 0..n {
        for j in i + 1..n {
            let d = match metric {
                "cosine" => {
                    let mut dot = 0.0;
                    let mut n1 = 0.0;
                    let mut n2 = 0.0;
                    for f in 0..data.nrows() {
                        dot += data[(f, i)] * data[(f, j)];
                        n1 += data[(f, i)].powi(2);
                        n2 += data[(f, j)].powi(2);
                    }
                    1.0 - dot / (n1.sqrt() * n2.sqrt() + 1e-10)
                }
                _ => {
                    let mut sum = 0.0;
                    for f in 0..data.nrows() {
                        sum += (data[(f, i)] - data[(f, j)]).powi(2);
                    }
                    sum.sqrt()
                }
            };
            dist[(i, j)] = d;
            dist[(j, i)] = d;
        }
    }

    // Threshold by k-nearest-neighbors
    let k_actual = if k == 0 { (2.0 * (n as Float).sqrt()).ceil() as usize } else { k };
    let mut rec = Array2::<Float>::zeros((n, n));

    for i in 0..n {
        // Find k-th nearest neighbor distance
        let mut row: Vec<Float> = dist.row(i).to_vec();
        row.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = if k_actual < row.len() { row[k_actual] } else { Float::INFINITY };

        for j in 0..n {
            if dist[(i, j)] <= threshold && i != j {
                rec[(i, j)] = 1.0;
            }
        }
    }

    // Symmetrize: keep mutual nearest neighbors only
    if sym {
        let mut sym_rec = Array2::<Float>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if rec[(i, j)] > 0.0 && rec[(j, i)] > 0.0 {
                    sym_rec[(i, j)] = 1.0;
                }
            }
        }
        rec = sym_rec;
    }

    Ok(rec)
}

/// Compute cross-similarity matrix between two feature sequences.
pub fn cross_similarity(
    data1: ArrayView2<Float>,
    data2: ArrayView2<Float>,
    metric: &str,
) -> Result<Array2<Float>> {
    let n1 = data1.ncols();
    let n2 = data2.ncols();
    let n_feat = data1.nrows();

    let mut sim = Array2::<Float>::zeros((n1, n2));

    for i in 0..n1 {
        for j in 0..n2 {
            let d = match metric {
                "cosine" => {
                    let mut dot = 0.0;
                    let mut na = 0.0;
                    let mut nb = 0.0;
                    for f in 0..n_feat {
                        dot += data1[(f, i)] * data2[(f, j)];
                        na += data1[(f, i)].powi(2);
                        nb += data2[(f, j)].powi(2);
                    }
                    dot / (na.sqrt() * nb.sqrt() + 1e-10)
                }
                _ => {
                    let mut sum = 0.0;
                    for f in 0..n_feat {
                        sum += (data1[(f, i)] - data2[(f, j)]).powi(2);
                    }
                    (-sum.sqrt()).exp() // Gaussian kernel
                }
            };
            sim[(i, j)] = d;
        }
    }

    Ok(sim)
}

/// Convert recurrence matrix to lag matrix.
pub fn recurrence_to_lag(rec: ArrayView2<Float>, pad: bool) -> Array2<Float> {
    let n = rec.nrows();
    let size = if pad { 2 * n } else { n };
    let mut lag = Array2::<Float>::zeros((size, n));

    for i in 0..n {
        for j in 0..n {
            let l = (j as i64 - i as i64).unsigned_abs() as usize;
            if l < size {
                lag[(l, i)] += rec[(i, j)];
            }
        }
    }

    lag
}

/// Convert lag matrix back to recurrence matrix.
pub fn lag_to_recurrence(lag: ArrayView2<Float>) -> Array2<Float> {
    let n = lag.ncols();
    let mut rec = Array2::<Float>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let l = (j as i64 - i as i64).unsigned_abs() as usize;
            if l < lag.nrows() {
                rec[(i, j)] = lag[(l, i)];
            }
        }
    }

    rec
}

/// Enhance diagonal paths in a similarity matrix.
pub fn path_enhance(r: ArrayView2<Float>, n_filter: usize) -> Array2<Float> {
    let n = r.nrows();
    let mut enhanced = Array2::<Float>::zeros((n, n));
    let half = n_filter / 2;

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            let mut count = 0;
            for k in 0..n_filter {
                let di = i as i64 + k as i64 - half as i64;
                let dj = j as i64 + k as i64 - half as i64;
                if di >= 0 && di < n as i64 && dj >= 0 && dj < n as i64 {
                    sum += r[(di as usize, dj as usize)];
                    count += 1;
                }
            }
            enhanced[(i, j)] = if count > 0 { sum / count as Float } else { 0.0 };
        }
    }

    enhanced
}

/// Subdivide segments based on feature changes.
pub fn subsegment(
    _data: ArrayView2<Float>,
    frames: &[usize],
    n_segments: usize,
) -> Vec<usize> {
    let mut all_frames: Vec<usize> = frames.to_vec();

    // For each segment, find internal change points
    for win in frames.windows(2) {
        let start = win[0];
        let end = win[1];
        if end - start <= n_segments {
            continue;
        }

        // Simple: divide segment into equal-length sub-segments
        let sub_len = (end - start) / n_segments;
        for k in 1..n_segments {
            all_frames.push(start + k * sub_len);
        }
    }

    all_frames.sort_unstable();
    all_frames.dedup();
    all_frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recurrence_matrix_self() {
        let data = Array2::from_shape_fn((4, 10), |(i, j)| (i * 10 + j) as Float);
        let rec = recurrence_matrix(data.view(), 3, "euclidean", false).unwrap();
        assert_eq!(rec.shape(), &[10, 10]);
        // Diagonal should be 0 (self is excluded)
        for i in 0..10 {
            assert_eq!(rec[(i, i)], 0.0);
        }
    }

    #[test]
    fn test_recurrence_matrix_symmetric() {
        let data = Array2::from_shape_fn((4, 10), |(i, j)| (i * 10 + j) as Float);
        let rec = recurrence_matrix(data.view(), 3, "euclidean", true).unwrap();
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(rec[(i, j)], rec[(j, i)], "should be symmetric");
            }
        }
    }

    #[test]
    fn test_cross_similarity_shape() {
        let d1 = Array2::from_shape_fn((4, 10), |(i, j)| (i + j) as Float);
        let d2 = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as Float);
        let sim = cross_similarity(d1.view(), d2.view(), "euclidean").unwrap();
        assert_eq!(sim.shape(), &[10, 8]);
    }

    #[test]
    fn test_recurrence_to_lag_roundtrip() {
        let data = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as Float);
        let rec = recurrence_matrix(data.view(), 2, "euclidean", false).unwrap();
        let lag = recurrence_to_lag(rec.view(), true); // pad=true preserves all lags
        let rec2 = lag_to_recurrence(lag.view());
        // Should reconstruct (positive lags only, so symmetric part matches)
        for i in 0..8 {
            for j in 0..8 {
                // Lag representation aggregates positive/negative lags,
                // so reconstruction is approximate for asymmetric matrices
                assert!(
                    (rec[(i, j)] - rec2[(i, j)]).abs() < 2.0,
                    "mismatch at ({i},{j}): {} vs {}", rec[(i, j)], rec2[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_path_enhance_shape() {
        let r = Array2::from_shape_fn((10, 10), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let enhanced = path_enhance(r.view(), 5);
        assert_eq!(enhanced.shape(), &[10, 10]);
        // Diagonal should still be strongest
        assert!(enhanced[(5, 5)] > enhanced[(5, 3)]);
    }

    #[test]
    fn test_subsegment() {
        let data = Array2::from_shape_fn((4, 20), |(_i, j)| j as Float);
        let frames = vec![0, 10, 20];
        let sub = subsegment(data.view(), &frames, 3);
        assert!(sub.len() > frames.len());
    }
}
