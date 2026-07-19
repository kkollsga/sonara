//! FFT abstraction layer wrapping `realfft`.
//!
//! Provides real-to-complex and complex-to-real FFT with thread-local plan caching
//! and scratch buffer reuse for zero-allocation hot-loop performance.
//!
//! Optimization over naive approach:
//! - RefCell instead of Mutex (thread_local is already single-threaded)
//! - Plan caching by size (avoids HashMap lookup in RealFftPlanner on every call)
//! - Scratch buffer reuse via process_with_scratch() (avoids per-call allocation)

use std::cell::RefCell;
use std::sync::Arc;

use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::error::{Result, SonaraError};
use crate::types::Float;

/// Maximum memory block size for chunked FFT processing (in bytes).
pub const MAX_MEM_BLOCK: usize = 256 * 1024;

/// Cached FFT state: planner + last-used plans + reusable scratch buffer.
struct FftCache {
    planner: RealFftPlanner<Float>,
    // Cache last-used plan to avoid HashMap lookup for repeated same-size FFTs
    last_fwd: Option<(usize, Arc<dyn RealToComplex<Float>>)>,
    last_inv: Option<(usize, Arc<dyn ComplexToReal<Float>>)>,
    scratch: Vec<Complex<Float>>,
}

impl FftCache {
    fn new() -> Self {
        Self {
            planner: RealFftPlanner::new(),
            last_fwd: None,
            last_inv: None,
            scratch: Vec::new(),
        }
    }

    fn get_forward(&mut self, n: usize) -> Arc<dyn RealToComplex<Float>> {
        if let Some((cached_n, ref plan)) = self.last_fwd {
            if cached_n == n {
                return Arc::clone(plan);
            }
        }
        let plan = self.planner.plan_fft_forward(n);
        // Ensure scratch is large enough
        let scratch_len = plan.get_scratch_len();
        if self.scratch.len() < scratch_len {
            self.scratch.resize(scratch_len, Complex::new(0.0, 0.0));
        }
        self.last_fwd = Some((n, Arc::clone(&plan)));
        plan
    }

    fn get_inverse(&mut self, n: usize) -> Arc<dyn ComplexToReal<Float>> {
        if let Some((cached_n, ref plan)) = self.last_inv {
            if cached_n == n {
                return Arc::clone(plan);
            }
        }
        let plan = self.planner.plan_fft_inverse(n);
        let scratch_len = plan.get_scratch_len();
        if self.scratch.len() < scratch_len {
            self.scratch.resize(scratch_len, Complex::new(0.0, 0.0));
        }
        self.last_inv = Some((n, Arc::clone(&plan)));
        plan
    }
}

// Thread-local FFT cache: RefCell instead of Mutex (no lock needed for thread_local).
thread_local! {
    static FFT_CACHE: RefCell<FftCache> = RefCell::new(FftCache::new());
}

/// Perform a real-to-complex FFT in-place with scratch buffer reuse.
pub fn rfft(input: &mut [Float], output: &mut [Complex<Float>]) -> Result<()> {
    let n = input.len();
    if output.len() != n / 2 + 1 {
        return Err(SonaraError::ShapeMismatch {
            expected: format!("{}", n / 2 + 1),
            got: format!("{}", output.len()),
        });
    }
    FFT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let plan = cache.get_forward(n);
        plan.process_with_scratch(input, output, &mut cache.scratch)
            .map_err(|e| SonaraError::Fft(e.to_string()))
    })
}

/// Perform a complex-to-real inverse FFT with scratch buffer reuse.
///
/// **Note:** The output is NOT normalized. Caller must divide by N if needed.
pub fn irfft(input: &mut [Complex<Float>], output: &mut [Float]) -> Result<()> {
    let n = output.len();
    if input.len() != n / 2 + 1 {
        return Err(SonaraError::ShapeMismatch {
            expected: format!("{}", n / 2 + 1),
            got: format!("{}", input.len()),
        });
    }
    FFT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let plan = cache.get_inverse(n);
        plan.process_with_scratch(input, output, &mut cache.scratch)
            .map_err(|e| SonaraError::Fft(e.to_string()))
    })
}

/// Compute real-to-complex FFT, allocating the output vector.
pub fn rfft_alloc(input: &mut [Float]) -> Result<Vec<Complex<Float>>> {
    let n = input.len();
    let out_len = n / 2 + 1;
    let mut output = vec![Complex::new(0.0, 0.0); out_len];
    rfft(input, &mut output)?;
    Ok(output)
}

/// Compute complex-to-real inverse FFT, allocating the output vector.
/// Applies 1/N normalization.
pub fn irfft_alloc(input: &mut [Complex<Float>], n: usize) -> Result<Vec<Float>> {
    let mut output = vec![0.0; n];
    irfft(input, &mut output)?;
    let scale = 1.0 / n as Float;
    for sample in output.iter_mut() {
        *sample *= scale;
    }
    Ok(output)
}

/// Compute the maximum number of spectral columns that fit in [`MAX_MEM_BLOCK`].
pub fn max_columns_in_block(n_fft: usize) -> usize {
    let col_bytes = (n_fft / 2 + 1) * std::mem::size_of::<Complex<Float>>();
    (MAX_MEM_BLOCK / col_bytes).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f32::consts::PI;

    #[test]
    fn test_rfft_sine() {
        let n = 2048;
        let sr = 22050.0;
        let freq = 440.0;
        let mut signal: Vec<Float> = (0..n)
            .map(|i| (2.0 * PI * freq * i as Float / sr).sin())
            .collect();

        let spectrum = rfft_alloc(&mut signal).unwrap();
        assert_eq!(spectrum.len(), n / 2 + 1);

        let magnitudes: Vec<Float> = spectrum.iter().map(|c| c.norm()).collect();
        let max_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let expected_bin = (freq * n as Float / sr).round() as usize;
        assert!(
            (max_bin as i64 - expected_bin as i64).unsigned_abs() <= 1,
            "expected bin ~{expected_bin}, got {max_bin}"
        );
    }

    #[test]
    fn test_rfft_irfft_roundtrip() {
        let n = 1024;
        let original: Vec<Float> = (0..n).map(|i| (i as Float * 0.1).sin()).collect();
        let mut input = original.clone();

        let mut spectrum = rfft_alloc(&mut input).unwrap();
        let reconstructed = irfft_alloc(&mut spectrum, n).unwrap();

        for (a, b) in original.iter().zip(reconstructed.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_rfft_output_length() {
        for n in [256, 512, 1024, 2048, 4096] {
            let mut signal = vec![0.0; n];
            let spectrum = rfft_alloc(&mut signal).unwrap();
            assert_eq!(spectrum.len(), n / 2 + 1);
        }
    }

    #[test]
    fn test_parseval_theorem() {
        let n = 2048;
        let original: Vec<Float> = (0..n).map(|i| (i as Float * 0.3).sin()).collect();
        let time_energy: Float = original.iter().map(|x| x * x).sum();

        let mut input = original;
        let spectrum = rfft_alloc(&mut input).unwrap();

        let mut freq_energy = spectrum[0].norm_sqr();
        for c in &spectrum[1..spectrum.len() - 1] {
            freq_energy += 2.0 * c.norm_sqr();
        }
        freq_energy += spectrum[spectrum.len() - 1].norm_sqr();

        assert_abs_diff_eq!(freq_energy / n as Float, time_energy, epsilon = 1e-1);
    }

    #[test]
    fn test_max_columns_in_block() {
        assert!(max_columns_in_block(2048) >= 1);
        assert!(max_columns_in_block(512) >= 1);
        assert!(max_columns_in_block(8192) <= max_columns_in_block(512));
    }
}
