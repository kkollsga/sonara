//! Core utility functions used pervasively across sonara.
//!
//! Frame, pad_center, normalize, peak_pick, and other core utilities.

use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use num_complex::Complex;
use num_traits::Float as NumFloat;

use crate::error::{Result, SonaraError};
use crate::types::Float;

/// Slice a signal into overlapping frames.
///
/// Returns a 2-D array of shape `(frame_length, n_frames)` where column `t` is
/// `y[t * hop_length .. t * hop_length + frame_length]`.
pub fn frame(
    y: ArrayView1<Float>,
    frame_length: usize,
    hop_length: usize,
) -> Result<Array2<Float>> {
    if hop_length == 0 {
        return Err(SonaraError::InvalidParameter {
            param: "hop_length",
            reason: "must be > 0".into(),
        });
    }
    if frame_length == 0 {
        return Err(SonaraError::InvalidParameter {
            param: "frame_length",
            reason: "must be > 0".into(),
        });
    }
    let n = y.len();
    if n < frame_length {
        return Err(SonaraError::InsufficientData {
            needed: frame_length,
            got: n,
        });
    }

    let n_frames = 1 + (n - frame_length) / hop_length;
    let mut frames = Array2::<Float>::zeros((frame_length, n_frames));

    for t in 0..n_frames {
        let start = t * hop_length;
        frames
            .column_mut(t)
            .assign(&y.slice(s![start..start + frame_length]));
    }

    Ok(frames)
}

/// Validate that `y` is a valid audio buffer (finite, real-valued, non-empty).
pub fn valid_audio(y: ArrayView1<Float>) -> Result<bool> {
    if y.is_empty() {
        return Err(SonaraError::InvalidAudio("audio buffer is empty".into()));
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(SonaraError::InvalidAudio(format!(
                "non-finite value at index {i}: {v}"
            )));
        }
    }
    Ok(true)
}

/// Check that `x` is a valid positive integer.
pub fn is_positive_int(x: f32) -> bool {
    x > 0.0 && x == x.floor() && x.is_finite()
}

/// Cast a float to integer, optionally applying a rounding function.
pub fn valid_int(x: f32, cast: Option<fn(f32) -> f32>) -> Result<i64> {
    if !x.is_finite() {
        return Err(SonaraError::InvalidParameter {
            param: "x",
            reason: format!("non-finite value: {x}"),
        });
    }
    let val = match cast {
        Some(f) => f(x),
        None => x,
    };
    Ok(val as i64)
}

/// Validate that intervals are well-formed: shape (n, 2) with each start <= end.
pub fn valid_intervals(intervals: &Array2<Float>) -> Result<bool> {
    if intervals.ncols() != 2 {
        return Err(SonaraError::InvalidParameter {
            param: "intervals",
            reason: format!("expected 2 columns, got {}", intervals.ncols()),
        });
    }
    for row in intervals.rows() {
        if row[0] > row[1] {
            return Err(SonaraError::InvalidParameter {
                param: "intervals",
                reason: format!("invalid interval: [{}, {}]", row[0], row[1]),
            });
        }
    }
    Ok(true)
}

/// Pad an array to length `size`, centering the original data.
/// Pads with zeros on both sides.
pub fn pad_center(data: ArrayView1<Float>, size: usize) -> Result<Array1<Float>> {
    let n = data.len();
    if size < n {
        return Err(SonaraError::InvalidParameter {
            param: "size",
            reason: format!("target size {size} < input length {n}"),
        });
    }
    let mut result = Array1::<Float>::zeros(size);
    let pad_left = (size - n) / 2;
    result.slice_mut(s![pad_left..pad_left + n]).assign(&data);
    Ok(result)
}

/// Broadcast an array to `ndim` dimensions by inserting size-1 axes.
///
/// Given `x` with shape (n,) and `ndim=3, axes=-2`, produces shape (1, n, 1).
/// The `axes` parameter specifies which axis of the output should hold the data.
pub fn expand_to(
    x: ArrayView1<Float>,
    ndim: usize,
    axes: i32,
) -> Result<crate::types::ArrayDynFloat> {
    use ndarray::IxDyn;

    if ndim == 0 {
        return Err(SonaraError::InvalidParameter {
            param: "ndim",
            reason: "must be > 0".into(),
        });
    }

    // Resolve negative axis
    let axis = if axes < 0 {
        (ndim as i32 + axes) as usize
    } else {
        axes as usize
    };

    if axis >= ndim {
        return Err(SonaraError::InvalidParameter {
            param: "axes",
            reason: format!("axis {axes} out of range for ndim {ndim}"),
        });
    }

    let mut shape = vec![1usize; ndim];
    shape[axis] = x.len();

    let result =
        crate::types::ArrayDynFloat::from_shape_vec(IxDyn(&shape), x.to_vec()).map_err(|e| {
            SonaraError::ShapeMismatch {
                expected: format!("{shape:?}"),
                got: e.to_string(),
            }
        })?;

    Ok(result)
}

/// Fix the length of an array to exactly `size`, by truncating or zero-padding.
pub fn fix_length(data: ArrayView1<Float>, size: usize) -> Array1<Float> {
    let n = data.len();
    if n >= size {
        data.slice(s![..size]).to_owned()
    } else {
        let mut result = Array1::<Float>::zeros(size);
        result.slice_mut(s![..n]).assign(&data);
        result
    }
}

/// Fix frame indices to be valid: ensure sorted, positive, bounded, and include boundaries.
pub fn fix_frames(frames: &[usize], x_min: usize, x_max: usize) -> Vec<usize> {
    let mut result: Vec<usize> = frames.iter().copied().filter(|&f| f < x_max).collect();

    // Ensure boundaries are included
    if result.is_empty() || result[0] != x_min {
        result.insert(0, x_min);
    }

    result.sort_unstable();
    result.dedup();
    result
}

/// Normalize an array along an axis.
///
/// - `norm`: "l1", "l2", "max", or "inf"
/// - `threshold`: small value to avoid divide-by-zero (default: tiny)
pub fn normalize(
    data: ArrayView1<Float>,
    norm: &str,
    threshold: Option<Float>,
) -> Result<Array1<Float>> {
    let thresh = threshold.unwrap_or_else(|| tiny(1.0f32));

    let scale = match norm {
        "l1" => {
            let s: Float = data.iter().map(|x| x.abs()).sum();
            s.max(thresh)
        }
        "l2" => {
            let s: Float = data.iter().map(|x| x * x).sum();
            s.sqrt().max(thresh)
        }
        "max" | "inf" => {
            let s: Float = data.iter().map(|x| x.abs()).fold(0.0, Float::max);
            s.max(thresh)
        }
        _ => {
            return Err(SonaraError::InvalidParameter {
                param: "norm",
                reason: format!("unsupported norm type: '{norm}'"),
            })
        }
    };

    Ok(data.mapv(|x| x / scale))
}

/// Find local maxima in a 1-D array.
/// Returns a boolean array where `true` indicates `x[i] > x[i-1]` and `x[i] > x[i+1]`.
pub fn localmax(x: ArrayView1<Float>) -> Array1<bool> {
    let n = x.len();
    if n < 3 {
        return Array1::from_elem(n, false);
    }
    Array1::from_shape_fn(n, |i| {
        if i == 0 || i == n - 1 {
            false
        } else {
            x[i] > x[i - 1] && x[i] > x[i + 1]
        }
    })
}

/// Find local minima in a 1-D array.
pub fn localmin(x: ArrayView1<Float>) -> Array1<bool> {
    let n = x.len();
    if n < 3 {
        return Array1::from_elem(n, false);
    }
    Array1::from_shape_fn(n, |i| {
        if i == 0 || i == n - 1 {
            false
        } else {
            x[i] < x[i - 1] && x[i] < x[i + 1]
        }
    })
}

/// Peak picking from an onset strength envelope.
///
/// A sample `n` is a peak if:
/// - `x[n] == max(x[n - pre_max : n + post_max])`
/// - `x[n] >= mean(x[n - pre_avg : n + post_avg]) + delta`
/// - `n - previous_peak >= wait`
pub fn peak_pick(
    x: ArrayView1<Float>,
    pre_max: usize,
    post_max: usize,
    pre_avg: usize,
    post_avg: usize,
    delta: Float,
    wait: usize,
) -> Vec<usize> {
    let n = x.len();
    let mut peaks = Vec::new();

    for i in 0..n {
        // Check local max condition
        let max_start = i.saturating_sub(pre_max);
        let max_end = (i + post_max + 1).min(n);
        let local_max = x
            .slice(s![max_start..max_end])
            .iter()
            .copied()
            .fold(Float::NEG_INFINITY, Float::max);

        if (x[i] - local_max).abs() > 1e-15 {
            continue;
        }

        // Check average threshold condition
        let avg_start = i.saturating_sub(pre_avg);
        let avg_end = (i + post_avg + 1).min(n);
        let avg_slice = x.slice(s![avg_start..avg_end]);
        let local_avg: Float = avg_slice.iter().sum::<Float>() / avg_slice.len() as Float;

        if x[i] < local_avg + delta {
            continue;
        }

        // Check wait condition
        if let Some(&last) = peaks.last() {
            if i - last < wait {
                continue;
            }
        }

        peaks.push(i);
    }

    peaks
}

/// Compute a soft mask for signal separation.
///
/// `softmask(X, ref, power) = X^power / (X^power + ref^power)`
///
/// Result is bounded in [0, 1].
pub fn softmask(
    x: ArrayView1<Float>,
    reference: ArrayView1<Float>,
    power: Float,
) -> Result<Array1<Float>> {
    if x.len() != reference.len() {
        return Err(SonaraError::ShapeMismatch {
            expected: format!("{}", x.len()),
            got: format!("{}", reference.len()),
        });
    }

    let eps = tiny(1.0f32);
    Ok(Array1::from_shape_fn(x.len(), |i| {
        let xp = x[i].abs().powf(power).max(eps);
        let rp = reference[i].abs().powf(power).max(eps);
        xp / (xp + rp)
    }))
}

/// Return the smallest positive usable value for a float type.
/// Equivalent to `np.finfo(x).tiny`.
pub fn tiny<T: NumFloat>(x: T) -> T {
    let _ = x; // Use the type, not the value
    T::min_positive_value()
}

/// Compute |x|^2 efficiently for complex or real values.
pub fn abs2_real(x: Float) -> Float {
    x * x
}

/// Compute |x|^2 for complex values: re^2 + im^2.
pub fn abs2_complex(x: Complex<Float>) -> Float {
    x.norm_sqr()
}

/// Compute |x|^2 element-wise for a real array.
pub fn abs2(x: ArrayView1<Float>) -> Array1<Float> {
    x.mapv(|v| v * v)
}

/// Compute unit phasors: exp(j * angles).
pub fn phasor(angles: ArrayView1<Float>) -> Array1<Complex<Float>> {
    angles.mapv(|a| Complex::new(a.cos(), a.sin()))
}

/// Compute unit phasor for a single angle.
pub fn phasor_scalar(angle: Float) -> Complex<Float> {
    Complex::new(angle.cos(), angle.sin())
}

/// Map a real dtype to its complex equivalent (conceptually).
/// In Rust we just track this as a type-level concern, but this function
/// documents the mapping f32→c64 for f32 inputs and f64→c128 for f64.
/// Since we use f32 internally, this returns "complex64".
pub fn dtype_r2c() -> &'static str {
    "complex64"
}

/// Map a complex dtype to its real equivalent.
pub fn dtype_c2r() -> &'static str {
    "float32"
}

/// Synchronize (aggregate) a data matrix `data` at specified frame boundaries.
///
/// For each segment defined by consecutive `frames` indices, aggregate along `axis`
/// using the specified function (mean, median, min, max).
pub fn sync(data: &Array2<Float>, frames: &[usize], aggregate: &str) -> Result<Array2<Float>> {
    if frames.is_empty() {
        return Err(SonaraError::InvalidParameter {
            param: "frames",
            reason: "frames must not be empty".into(),
        });
    }

    let n_cols = data.ncols();
    let n_segments = frames.len() + 1;
    let mut result = Array2::<Float>::zeros((data.nrows(), n_segments));

    // Build segment boundaries: [0, frames[0], frames[1], ..., n_cols]
    let mut boundaries = vec![0usize];
    boundaries.extend_from_slice(frames);
    if *boundaries.last().unwrap() != n_cols {
        boundaries.push(n_cols);
    }

    for (seg_idx, window) in boundaries.windows(2).enumerate() {
        let start = window[0];
        let end = window[1];
        if start >= end || end > n_cols {
            continue;
        }

        let segment = data.slice(s![.., start..end]);

        let agg = match aggregate {
            "mean" => segment.mean_axis(Axis(1)).unwrap(),
            "max" => Array1::from_shape_fn(data.nrows(), |r| {
                segment
                    .row(r)
                    .iter()
                    .copied()
                    .fold(Float::NEG_INFINITY, Float::max)
            }),
            "min" => Array1::from_shape_fn(data.nrows(), |r| {
                segment
                    .row(r)
                    .iter()
                    .copied()
                    .fold(Float::INFINITY, Float::min)
            }),
            _ => {
                return Err(SonaraError::InvalidParameter {
                    param: "aggregate",
                    reason: format!("unsupported aggregate: '{aggregate}'"),
                })
            }
        };

        result.column_mut(seg_idx).assign(&agg);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::f32::consts::PI;

    #[test]
    fn test_frame_basic() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let frames = frame(y.view(), 3, 2).unwrap();
        assert_eq!(frames.shape(), &[3, 2]);
        assert_eq!(frames.column(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(frames.column(1).to_vec(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_frame_hop_1() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let frames = frame(y.view(), 3, 1).unwrap();
        assert_eq!(frames.shape(), &[3, 2]);
        assert_eq!(frames.column(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(frames.column(1).to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_frame_insufficient_data() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        assert!(frame(y.view(), 3, 1).is_err());
    }

    #[test]
    fn test_frame_zero_hop() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(frame(y.view(), 2, 0).is_err());
    }

    #[test]
    fn test_valid_audio_good() {
        let y = Array1::from_vec(vec![0.1, -0.5, 0.9]);
        assert!(valid_audio(y.view()).is_ok());
    }

    #[test]
    fn test_valid_audio_nan() {
        let y = Array1::from_vec(vec![0.1, Float::NAN, 0.9]);
        assert!(valid_audio(y.view()).is_err());
    }

    #[test]
    fn test_valid_audio_inf() {
        let y = Array1::from_vec(vec![0.1, Float::INFINITY, 0.9]);
        assert!(valid_audio(y.view()).is_err());
    }

    #[test]
    fn test_valid_audio_empty() {
        let y = Array1::<Float>::zeros(0);
        assert!(valid_audio(y.view()).is_err());
    }

    #[test]
    fn test_is_positive_int() {
        assert!(is_positive_int(1.0));
        assert!(is_positive_int(512.0));
        assert!(!is_positive_int(0.0));
        assert!(!is_positive_int(-1.0));
        assert!(!is_positive_int(1.5));
        assert!(!is_positive_int(Float::NAN));
    }

    #[test]
    fn test_pad_center() {
        let data = array![1.0, 2.0, 3.0];
        let padded = pad_center(data.view(), 7).unwrap();
        assert_eq!(padded.to_vec(), vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pad_center_no_padding() {
        let data = array![1.0, 2.0, 3.0];
        let padded = pad_center(data.view(), 3).unwrap();
        assert_eq!(padded.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pad_center_too_small() {
        let data = array![1.0, 2.0, 3.0];
        assert!(pad_center(data.view(), 2).is_err());
    }

    #[test]
    fn test_fix_length_truncate() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let fixed = fix_length(data.view(), 3);
        assert_eq!(fixed.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fix_length_pad() {
        let data = array![1.0, 2.0];
        let fixed = fix_length(data.view(), 5);
        assert_eq!(fixed.to_vec(), vec![1.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_l2() {
        let data = array![3.0, 4.0];
        let normed = normalize(data.view(), "l2", None).unwrap();
        assert_abs_diff_eq!(normed[0], 0.6, epsilon = 1e-5);
        assert_abs_diff_eq!(normed[1], 0.8, epsilon = 1e-5);
    }

    #[test]
    fn test_normalize_max() {
        let data = array![1.0, -3.0, 2.0];
        let normed = normalize(data.view(), "max", None).unwrap();
        assert_abs_diff_eq!(normed[1], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_normalize_l1() {
        let data = array![1.0, 2.0, 3.0];
        let normed = normalize(data.view(), "l1", None).unwrap();
        let sum: Float = normed.iter().map(|x| x.abs()).sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_localmax() {
        let x = array![1.0, 3.0, 2.0, 4.0, 1.0];
        let result = localmax(x.view());
        assert_eq!(result.to_vec(), vec![false, true, false, true, false]);
    }

    #[test]
    fn test_localmin() {
        let x = array![3.0, 1.0, 2.0, 0.0, 4.0];
        let result = localmin(x.view());
        assert_eq!(result.to_vec(), vec![false, true, false, true, false]);
    }

    #[test]
    fn test_localmax_short() {
        let x = array![1.0, 2.0];
        let result = localmax(x.view());
        assert_eq!(result.to_vec(), vec![false, false]);
    }

    #[test]
    fn test_peak_pick_basic() {
        // Simple envelope with clear peaks
        let x = array![0.0, 1.0, 0.5, 2.0, 0.3, 0.0];
        let peaks = peak_pick(x.view(), 1, 1, 1, 1, 0.0, 1);
        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
    }

    #[test]
    fn test_softmask_bounds() {
        let x = array![1.0, 2.0, 3.0];
        let r = array![3.0, 2.0, 1.0];
        let mask = softmask(x.view(), r.view(), 1.0).unwrap();
        for &v in mask.iter() {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_softmask_equal() {
        let x = array![1.0, 2.0, 3.0];
        let mask = softmask(x.view(), x.view(), 1.0).unwrap();
        for &v in mask.iter() {
            assert_abs_diff_eq!(v, 0.5, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_tiny() {
        let t: f32 = tiny(1.0f32);
        assert!(t > 0.0);
        assert!(t < 1e-37);
    }

    #[test]
    fn test_abs2() {
        let x = array![3.0, -4.0, 0.0];
        let result = abs2(x.view());
        assert_eq!(result.to_vec(), vec![9.0, 16.0, 0.0]);
    }

    #[test]
    fn test_abs2_complex() {
        let c = Complex::new(3.0, 4.0);
        assert_abs_diff_eq!(abs2_complex(c), 25.0, epsilon = 1e-5);
    }

    #[test]
    fn test_phasor() {
        let angles = array![0.0, PI / 2.0, PI];
        let result = phasor(angles.view());
        assert_abs_diff_eq!(result[0].re, 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result[0].im, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result[1].re, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result[1].im, 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result[2].re, -1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result[2].im, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_fix_frames() {
        let frames = fix_frames(&[5, 3, 10, 1], 0, 20);
        assert_eq!(frames[0], 0); // boundary included
        assert!(frames.windows(2).all(|w| w[0] < w[1])); // sorted, deduped
    }

    #[test]
    fn test_fix_frames_clips_max() {
        let frames = fix_frames(&[5, 30], 0, 20);
        assert!(!frames.contains(&30));
    }

    #[test]
    fn test_valid_intervals_good() {
        let intervals = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 2.0]).unwrap();
        assert!(valid_intervals(&intervals).is_ok());
    }

    #[test]
    fn test_valid_intervals_bad() {
        let intervals = Array2::from_shape_vec((1, 2), vec![2.0, 1.0]).unwrap();
        assert!(valid_intervals(&intervals).is_err());
    }

    #[test]
    fn test_expand_to() {
        let x = array![1.0, 2.0, 3.0];
        let result = expand_to(x.view(), 3, -2).unwrap();
        assert_eq!(result.shape(), &[1, 3, 1]);
    }
}
