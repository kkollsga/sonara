//! Window function generation.
//!
//! Implements the standard window functions used in spectral analysis.
//! These replace scipy.signal.get_window for the subset used in audio analysis.

use std::f32::consts::PI;

use ndarray::Array1;

use crate::error::{Result, SonaraError};
use crate::types::Float;

/// Generate a Hann (raised cosine) window of length `n`.
///
/// If `fftbins` is true, the window is periodic (length N, suitable for FFT).
/// If false, the window is symmetric (suitable for filter design).
pub fn hann(n: usize, fftbins: bool) -> Array1<Float> {
    general_cosine(n, &[0.5, 0.5], fftbins)
}

/// Generate a Hamming window.
pub fn hamming(n: usize, fftbins: bool) -> Array1<Float> {
    general_cosine(n, &[0.54, 0.46], fftbins)
}

/// Generate a Blackman window.
pub fn blackman(n: usize, fftbins: bool) -> Array1<Float> {
    general_cosine(n, &[0.42, 0.50, 0.08], fftbins)
}

/// Generate a Bartlett (triangular) window.
pub fn bartlett(n: usize, fftbins: bool) -> Array1<Float> {
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1 {
        return Array1::ones(1);
    }
    let m = if fftbins { n } else { n - 1 };
    Array1::from_shape_fn(n, |i| {
        let x = i as Float / m as Float;
        if x <= 0.5 {
            2.0 * x
        } else {
            2.0 - 2.0 * x
        }
    })
}

/// Generate a rectangular (boxcar) window — all ones.
pub fn boxcar(n: usize) -> Array1<Float> {
    Array1::ones(n)
}

/// Generate a Kaiser window with parameter `beta`.
pub fn kaiser(n: usize, beta: Float, fftbins: bool) -> Array1<Float> {
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1 {
        return Array1::ones(1);
    }
    let m = if fftbins { n } else { n - 1 };
    let denom = bessel_i0(beta);
    Array1::from_shape_fn(n, |i| {
        let alpha = 2.0 * i as Float / m as Float - 1.0;
        let arg = beta * (1.0 - alpha * alpha).sqrt();
        bessel_i0(arg) / denom
    })
}

/// Generate a Tukey (tapered cosine) window with parameter `alpha`.
/// alpha=0 → rectangular, alpha=1 → Hann.
pub fn tukey(n: usize, alpha: Float, fftbins: bool) -> Array1<Float> {
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1 {
        return Array1::ones(1);
    }
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha == 0.0 {
        return Array1::ones(n);
    }
    let m = if fftbins { n } else { n - 1 };
    Array1::from_shape_fn(n, |i| {
        let x = i as Float / m as Float;
        if x < alpha / 2.0 {
            0.5 * (1.0 - (2.0 * PI * x / alpha).cos())
        } else if x <= 1.0 - alpha / 2.0 {
            1.0
        } else {
            0.5 * (1.0 - (2.0 * PI * (1.0 - x) / alpha).cos())
        }
    })
}

/// Generate a Gaussian window with standard deviation `std`.
pub fn gaussian(n: usize, std: Float, fftbins: bool) -> Array1<Float> {
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1 {
        return Array1::ones(1);
    }
    let m = if fftbins { n } else { n - 1 };
    let center = m as Float / 2.0;
    Array1::from_shape_fn(n, |i| {
        let x = (i as Float - center) / (std * center);
        (-0.5 * x * x).exp()
    })
}

/// Generalized cosine window from a sequence of coefficients.
/// w[n] = sum_k (-1)^k * a_k * cos(2*pi*k*n/M)
fn general_cosine(n: usize, coeffs: &[Float], fftbins: bool) -> Array1<Float> {
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1 {
        return Array1::ones(1);
    }
    let m = if fftbins { n } else { n - 1 };
    Array1::from_shape_fn(n, |i| {
        let mut val = 0.0;
        let mut sign = 1.0;
        for (k, &a) in coeffs.iter().enumerate() {
            val += sign * a * (2.0 * PI * k as Float * i as Float / m as Float).cos();
            sign = -sign;
        }
        val
    })
}

/// Modified Bessel function of the first kind, order 0.
/// Used by the Kaiser window. Series expansion converges rapidly.
fn bessel_i0(x: Float) -> Float {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;
    for k in 1..50 {
        term *= (x_half / k as Float) * (x_half / k as Float);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// Get a window by name/specification, returning an array of length `n`.
///
/// Supported windows: "hann", "hamming", "blackman", "bartlett", "boxcar",
/// "ones", "rectangular", "triangular", "kaiser" (requires beta via WindowSpec::Parameterized),
/// "tukey", "gaussian".
pub fn get_window(
    spec: &crate::types::WindowSpec,
    n: usize,
    fftbins: bool,
) -> Result<Array1<Float>> {
    use crate::types::WindowSpec;
    match spec {
        WindowSpec::Named(name) => match name.to_lowercase().as_str() {
            "hann" | "hanning" => Ok(hann(n, fftbins)),
            "hamming" => Ok(hamming(n, fftbins)),
            "blackman" => Ok(blackman(n, fftbins)),
            "bartlett" | "triangular" | "triang" => Ok(bartlett(n, fftbins)),
            "boxcar" | "rectangular" | "ones" => Ok(boxcar(n)),
            "kaiser" => Ok(kaiser(n, 14.0, fftbins)), // default beta
            "tukey" => Ok(tukey(n, 0.5, fftbins)),    // default alpha
            "gaussian" => Ok(gaussian(n, 0.4, fftbins)), // default std
            _ => Err(SonaraError::InvalidParameter {
                param: "window",
                reason: format!("unknown window function: '{name}'"),
            }),
        },
        WindowSpec::Array(arr) => {
            if arr.len() == n {
                Ok(arr.clone())
            } else {
                Err(SonaraError::ShapeMismatch {
                    expected: format!("{n}"),
                    got: format!("{}", arr.len()),
                })
            }
        }
        WindowSpec::Parameterized(name, param) => match name.to_lowercase().as_str() {
            "kaiser" => Ok(kaiser(n, *param, fftbins)),
            "tukey" => Ok(tukey(n, *param, fftbins)),
            "gaussian" => Ok(gaussian(n, *param, fftbins)),
            _ => Err(SonaraError::InvalidParameter {
                param: "window",
                reason: format!("unknown parameterized window: '{name}'"),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_hann_symmetry() {
        let w = hann(256, false);
        for i in 0..128 {
            assert_abs_diff_eq!(w[i], w[255 - i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_hann_endpoints() {
        let w = hann(256, false);
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(w[255], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hamming_known_values() {
        // Hamming(5, symmetric): [0.08, 0.54, 1.0, 0.54, 0.08]
        let w = hamming(5, false);
        assert_abs_diff_eq!(w[0], 0.08, epsilon = 1e-5);
        assert_abs_diff_eq!(w[2], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(w[4], 0.08, epsilon = 1e-5);
    }

    #[test]
    fn test_kaiser_beta_zero_is_rectangular() {
        let w = kaiser(64, 0.0, false);
        for &v in w.iter() {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_boxcar_all_ones() {
        let w = boxcar(100);
        for &v in w.iter() {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_window_energy_positive() {
        for win_fn in [
            hann(256, true),
            hamming(256, true),
            blackman(256, true),
            bartlett(256, true),
        ] {
            let energy: Float = win_fn.iter().sum();
            assert!(energy > 0.0, "Window energy must be positive");
        }
    }

    #[test]
    fn test_get_window_named() {
        use crate::types::WindowSpec;
        let w = get_window(&WindowSpec::Named("hann".into()), 512, true).unwrap();
        assert_eq!(w.len(), 512);
    }

    #[test]
    fn test_get_window_parameterized_kaiser() {
        use crate::types::WindowSpec;
        let w = get_window(&WindowSpec::Parameterized("kaiser".into(), 8.0), 256, true).unwrap();
        assert_eq!(w.len(), 256);
        // Kaiser with beta > 0 tapers at edges
        assert!(w[0] < w[128]);
    }

    #[test]
    fn test_get_window_array() {
        use crate::types::WindowSpec;
        let arr = Array1::ones(64);
        let w = get_window(&WindowSpec::Array(arr), 64, true).unwrap();
        assert_eq!(w.len(), 64);
    }

    #[test]
    fn test_get_window_array_size_mismatch() {
        use crate::types::WindowSpec;
        let arr = Array1::ones(32);
        assert!(get_window(&WindowSpec::Array(arr), 64, true).is_err());
    }

    #[test]
    fn test_tukey_alpha_zero_is_rectangular() {
        let w = tukey(64, 0.0, false);
        for &v in w.iter() {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_empty_windows() {
        assert_eq!(hann(0, true).len(), 0);
        assert_eq!(hamming(0, true).len(), 0);
        assert_eq!(kaiser(0, 5.0, true).len(), 0);
    }

    #[test]
    fn test_single_sample_windows() {
        assert_abs_diff_eq!(hann(1, true)[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(hamming(1, true)[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(kaiser(1, 5.0, true)[0], 1.0, epsilon = 1e-5);
    }
}
