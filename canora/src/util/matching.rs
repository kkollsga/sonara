//! Event and interval matching utilities.
//!
//! Mirrors librosa.util.matching — match_intervals, match_events.

#[cfg(test)]
use ndarray::Array2;
use ndarray::ArrayView2;

use crate::error::{CanoraError, Result};
use crate::types::Float;

/// Match intervals by Jaccard overlap.
///
/// For each interval in `intervals_from`, find the best-matching interval
/// in `intervals_to` based on the Jaccard index (intersection / union).
///
/// - `intervals_from`: shape `(n, 2)` — start/end pairs
/// - `intervals_to`: shape `(m, 2)` — start/end pairs
///
/// Returns indices into `intervals_to` for each row of `intervals_from`.
pub fn match_intervals(
    intervals_from: ArrayView2<Float>,
    intervals_to: ArrayView2<Float>,
) -> Result<Vec<usize>> {
    if intervals_from.ncols() != 2 || intervals_to.ncols() != 2 {
        return Err(CanoraError::InvalidParameter {
            param: "intervals",
            reason: "intervals must have 2 columns (start, end)".into(),
        });
    }

    let n = intervals_from.nrows();
    let m = intervals_to.nrows();

    if m == 0 {
        return Err(CanoraError::InvalidParameter {
            param: "intervals_to",
            reason: "target intervals must not be empty".into(),
        });
    }

    let mut matches = Vec::with_capacity(n);

    for i in 0..n {
        let a_start = intervals_from[(i, 0)];
        let a_end = intervals_from[(i, 1)];

        let mut best_j = 0;
        let mut best_jaccard = -1.0_f64;

        for j in 0..m {
            let b_start = intervals_to[(j, 0)];
            let b_end = intervals_to[(j, 1)];

            let jaccard = jaccard_overlap(a_start, a_end, b_start, b_end);
            if jaccard > best_jaccard {
                best_jaccard = jaccard;
                best_j = j;
            }
        }

        matches.push(best_j);
    }

    Ok(matches)
}

/// Compute Jaccard index between two intervals.
fn jaccard_overlap(a_start: Float, a_end: Float, b_start: Float, b_end: Float) -> Float {
    let inter_start = a_start.max(b_start);
    let inter_end = a_end.min(b_end);
    let intersection = (inter_end - inter_start).max(0.0);

    let union = (a_end - a_start) + (b_end - b_start) - intersection;
    if union <= 0.0 {
        return 0.0;
    }

    intersection / union
}

/// Match events (time points) to the nearest targets.
///
/// For each event in `events_from`, find the nearest event in `events_to`.
///
/// Returns indices into `events_to`.
pub fn match_events(events_from: &[Float], events_to: &[Float]) -> Result<Vec<usize>> {
    if events_to.is_empty() {
        return Err(CanoraError::InvalidParameter {
            param: "events_to",
            reason: "target events must not be empty".into(),
        });
    }

    let mut matches = Vec::with_capacity(events_from.len());

    for &event in events_from {
        let mut best_j = 0;
        let mut best_dist = Float::INFINITY;

        for (j, &target) in events_to.iter().enumerate() {
            let dist = (event - target).abs();
            if dist < best_dist {
                best_dist = dist;
                best_j = j;
            }
        }

        matches.push(best_j);
    }

    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_match_intervals_exact() {
        let from = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let to = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let matches = match_intervals(from.view(), to.view()).unwrap();
        assert_eq!(matches, vec![0, 1]);
    }

    #[test]
    fn test_match_intervals_overlap() {
        let from = Array2::from_shape_vec((1, 2), vec![0.5, 1.5]).unwrap();
        let to = Array2::from_shape_vec((3, 2), vec![
            0.0, 1.0,  // overlaps [0.5, 1.0] = 0.5
            1.0, 2.0,  // overlaps [1.0, 1.5] = 0.5
            0.0, 2.0,  // overlaps [0.5, 1.5] = 1.0
        ]).unwrap();
        let matches = match_intervals(from.view(), to.view()).unwrap();
        // Best match should be interval 2 (full overlap / smallest union)
        assert_eq!(matches[0], 2);
    }

    #[test]
    fn test_match_events_exact() {
        let from = vec![0.5, 1.5, 2.5];
        let to = vec![0.5, 1.5, 2.5];
        let matches = match_events(&from, &to).unwrap();
        assert_eq!(matches, vec![0, 1, 2]);
    }

    #[test]
    fn test_match_events_nearest() {
        let from = vec![0.3, 1.7];
        let to = vec![0.0, 1.0, 2.0];
        let matches = match_events(&from, &to).unwrap();
        // 0.3 is closest to 0.0 (dist=0.3)
        // 1.7 is closest to 2.0 (dist=0.3)
        assert_eq!(matches[0], 0);
        assert_eq!(matches[1], 2);
    }

    #[test]
    fn test_jaccard_overlap() {
        // Full overlap
        assert_abs_diff_eq!(jaccard_overlap(0.0, 1.0, 0.0, 1.0), 1.0, epsilon = 1e-10);
        // No overlap
        assert_abs_diff_eq!(jaccard_overlap(0.0, 1.0, 2.0, 3.0), 0.0, epsilon = 1e-10);
        // Half overlap: [0,2] ∩ [1,3] = [1,2] = 1, union = 3
        assert_abs_diff_eq!(jaccard_overlap(0.0, 2.0, 1.0, 3.0), 1.0 / 3.0, epsilon = 1e-10);
    }
}
