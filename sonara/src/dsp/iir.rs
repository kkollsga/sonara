//! IIR (Infinite Impulse Response) digital filter implementations.
//!
//! Replaces scipy.signal.lfilter, filtfilt, and sosfiltfilt.
//! Used for PCEN, preemphasis/deemphasis, and iirt.

use ndarray::{Array1, ArrayView1};

use crate::error::{Result, SonaraError};
use crate::types::Float;

/// Apply an IIR/FIR filter using Direct Form II Transposed.
///
/// Implements `y = lfilter(b, a, x)` matching scipy.signal.lfilter.
///
/// - `b`: Numerator coefficients (feedforward)
/// - `a`: Denominator coefficients (feedback). `a[0]` must be non-zero.
/// - `x`: Input signal
///
/// Returns the filtered signal.
pub fn lfilter(
    b: ArrayView1<Float>,
    a: ArrayView1<Float>,
    x: ArrayView1<Float>,
) -> Result<Array1<Float>> {
    if a.is_empty() || b.is_empty() {
        return Err(SonaraError::InvalidParameter {
            param: "b/a",
            reason: "filter coefficients must not be empty".into(),
        });
    }
    if a[0] == 0.0 {
        return Err(SonaraError::InvalidParameter {
            param: "a",
            reason: "a[0] must be non-zero".into(),
        });
    }

    let n = x.len();
    let order = b.len().max(a.len());

    // Normalize by a[0]
    let a0 = a[0];
    let b_norm: Vec<Float> = b.iter().map(|&v| v / a0).collect();
    let a_norm: Vec<Float> = a.iter().map(|&v| v / a0).collect();

    let mut y = Array1::<Float>::zeros(n);
    let mut state = vec![0.0_f32; order];

    for i in 0..n {
        // Output: y[i] = b[0]*x[i] + state[0]
        y[i] = b_norm[0] * x[i] + state[0];

        // Update state (Direct Form II Transposed)
        for j in 0..order - 1 {
            let b_j = if j + 1 < b_norm.len() {
                b_norm[j + 1]
            } else {
                0.0
            };
            let a_j = if j + 1 < a_norm.len() {
                a_norm[j + 1]
            } else {
                0.0
            };
            state[j] = b_j * x[i] - a_j * y[i] + state[j + 1];
        }
        state[order - 1] = 0.0;
    }

    Ok(y)
}

/// Compute initial conditions for lfilter to produce steady-state output.
///
/// Equivalent to scipy.signal.lfilter_zi. Used by filtfilt for edge padding.
#[allow(dead_code)]
fn lfilter_zi(b: &[Float], a: &[Float]) -> Vec<Float> {
    let order = b.len().max(a.len());
    if order <= 1 {
        return vec![];
    }

    let n = order - 1;
    let mut zi = vec![0.0; n];

    // Build companion matrix and solve for steady-state
    // Simplified approach: assume unit step input and solve for zi
    // such that the output is constant b_sum/a_sum
    let b_sum: Float = b.iter().sum();
    let a_sum: Float = a.iter().sum();

    if a_sum.abs() < 1e-15 {
        return zi;
    }

    let ss = b_sum / a_sum;

    // Forward substitution: if input is constant ss/b[0] and output is constant ss,
    // what initial conditions produce this?
    // Direct computation from the filter difference equation
    for i in 0..n {
        let b_i = if i + 1 < b.len() { b[i + 1] } else { 0.0 };
        let a_i = if i + 1 < a.len() { a[i + 1] } else { 0.0 };
        zi[i] = b_i - a_i * ss;
        // Accumulate from remaining terms
        for j in (i + 1)..n {
            let b_j = if j + 1 < b.len() { b[j + 1] } else { 0.0 };
            let a_j = if j + 1 < a.len() { a[j + 1] } else { 0.0 };
            zi[i] += b_j - a_j * ss;
        }
    }

    zi
}

/// Apply an IIR filter forward and backward for zero-phase filtering.
///
/// Equivalent to scipy.signal.filtfilt. The result has zero phase distortion
/// and double the filter order.
///
/// - `b`: Numerator coefficients
/// - `a`: Denominator coefficients
/// - `x`: Input signal
pub fn filtfilt(
    b: ArrayView1<Float>,
    a: ArrayView1<Float>,
    x: ArrayView1<Float>,
) -> Result<Array1<Float>> {
    let n = x.len();
    if n < 3 {
        return Err(SonaraError::InsufficientData { needed: 3, got: n });
    }

    // Pad length: 3 * max(len(a), len(b))
    let pad_len = 3 * (a.len().max(b.len()));
    let pad_len = pad_len.min(n - 1);

    // Edge padding by reflection
    let mut padded = Array1::<Float>::zeros(n + 2 * pad_len);

    // Reflect left edge
    for i in 0..pad_len {
        padded[pad_len - 1 - i] = 2.0 * x[0] - x[i + 1];
    }
    // Copy signal
    for i in 0..n {
        padded[pad_len + i] = x[i];
    }
    // Reflect right edge
    for i in 0..pad_len {
        padded[pad_len + n + i] = 2.0 * x[n - 1] - x[n - 2 - i];
    }

    // Forward pass
    let forward = lfilter(b, a, padded.view())?;

    // Reverse
    let mut reversed = Array1::<Float>::zeros(forward.len());
    for i in 0..forward.len() {
        reversed[i] = forward[forward.len() - 1 - i];
    }

    // Backward pass
    let backward = lfilter(b, a, reversed.view())?;

    // Reverse again and trim padding
    let mut result = Array1::<Float>::zeros(n);
    for i in 0..n {
        result[i] = backward[backward.len() - 1 - pad_len - i];
    }

    Ok(result)
}

/// Second-order sections (biquad cascade) filter — forward and backward.
///
/// Equivalent to scipy.signal.sosfiltfilt.
///
/// - `sos`: Second-order sections, shape (n_sections, 6). Each row is [b0, b1, b2, a0, a1, a2].
/// - `x`: Input signal
pub fn sosfiltfilt(sos: &[&[Float; 6]], x: ArrayView1<Float>) -> Result<Array1<Float>> {
    let mut result = x.to_owned();

    // Apply each section as filtfilt
    for section in sos {
        let b = Array1::from_vec(vec![section[0], section[1], section[2]]);
        let a = Array1::from_vec(vec![section[3], section[4], section[5]]);
        result = filtfilt(b.view(), a.view(), result.view())?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_lfilter_identity() {
        // Passthrough: b=[1], a=[1]
        let b = array![1.0];
        let a = array![1.0];
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = lfilter(b.view(), a.view(), x.view()).unwrap();
        for i in 0..x.len() {
            assert_abs_diff_eq!(y[i], x[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_lfilter_fir() {
        // Simple FIR: moving average of 2 samples
        // b=[0.5, 0.5], a=[1]
        let b = array![0.5, 0.5];
        let a = array![1.0];
        let x = array![1.0, 3.0, 5.0, 7.0];
        let y = lfilter(b.view(), a.view(), x.view()).unwrap();
        assert_abs_diff_eq!(y[0], 0.5, epsilon = 1e-5); // 0.5*1 + 0.5*0
        assert_abs_diff_eq!(y[1], 2.0, epsilon = 1e-5); // 0.5*3 + 0.5*1
        assert_abs_diff_eq!(y[2], 4.0, epsilon = 1e-5); // 0.5*5 + 0.5*3
        assert_abs_diff_eq!(y[3], 6.0, epsilon = 1e-5); // 0.5*7 + 0.5*5
    }

    #[test]
    fn test_lfilter_first_order_iir() {
        // First-order IIR: y[n] = x[n] + 0.5*y[n-1]
        // b=[1], a=[1, -0.5]
        let b = array![1.0];
        let a = array![1.0, -0.5];
        let x = array![1.0, 0.0, 0.0, 0.0, 0.0];
        let y = lfilter(b.view(), a.view(), x.view()).unwrap();
        // Impulse response: 1, 0.5, 0.25, 0.125, 0.0625
        assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(y[1], 0.5, epsilon = 1e-5);
        assert_abs_diff_eq!(y[2], 0.25, epsilon = 1e-5);
        assert_abs_diff_eq!(y[3], 0.125, epsilon = 1e-5);
        assert_abs_diff_eq!(y[4], 0.0625, epsilon = 1e-5);
    }

    #[test]
    fn test_lfilter_empty_error() {
        let b = Array1::<Float>::zeros(0);
        let a = array![1.0];
        let x = array![1.0];
        assert!(lfilter(b.view(), a.view(), x.view()).is_err());
    }

    #[test]
    fn test_filtfilt_zero_phase() {
        // filtfilt of a sine signal should preserve phase (no group delay shift)
        let b = array![0.2, 0.6, 0.2]; // low-pass FIR
        let a = array![1.0];
        let n = 200;
        let x = Array1::from_shape_fn(n, |i| {
            (2.0 * std::f32::consts::PI * 5.0 * i as Float / n as Float).sin()
        });
        let y = filtfilt(b.view(), a.view(), x.view()).unwrap();

        // Zero-phase: peaks of output should align with peaks of input
        // Find peak of input in the middle region
        let mid = n / 2;
        let input_peak = (mid - 10..mid + 10)
            .max_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap())
            .unwrap();
        let output_peak = (mid - 10..mid + 10)
            .max_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap())
            .unwrap();
        // Peaks should be within 1 sample of each other (zero phase)
        assert!(
            (input_peak as i64 - output_peak as i64).unsigned_abs() <= 1,
            "filtfilt should be zero-phase: input peak at {input_peak}, output at {output_peak}"
        );
    }

    #[test]
    fn test_filtfilt_vs_double_lfilter() {
        // filtfilt should produce a smoother result than single lfilter
        let b = array![0.1, 0.2, 0.4, 0.2, 0.1];
        let a = array![1.0];
        let x = Array1::from_shape_fn(200, |i| (i as Float * 0.3).sin() + (i as Float * 1.7).sin());

        let single = lfilter(b.view(), a.view(), x.view()).unwrap();
        let double = filtfilt(b.view(), a.view(), x.view()).unwrap();

        // Double filtering has same length
        assert_eq!(double.len(), x.len());

        // Double filtering attenuates high frequencies more
        let diff_single: Float = single
            .windows(2)
            .into_iter()
            .map(|w| (w[1] - w[0]).abs())
            .sum::<Float>();
        let diff_double: Float = double
            .windows(2)
            .into_iter()
            .map(|w| (w[1] - w[0]).abs())
            .sum::<Float>();
        assert!(diff_double < diff_single, "filtfilt should be smoother");
    }
}
