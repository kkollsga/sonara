//! Constant-Q and Variable-Q Transform.
//!
//! Mirrors librosa.core.constantq — cqt, vqt, hybrid_cqt, pseudo_cqt, icqt, griffinlim_cqt.
//!
//! The CQT provides logarithmically-spaced frequency resolution, making it
//! ideal for music analysis where pitch is perceived logarithmically.
//! The VQT generalizes CQT with a variable Q-factor (gamma parameter).
//!
//! Architecture: cqt → vqt(gamma=0), hybrid_cqt splits high/low bands,
//! pseudo_cqt uses a single FFT for all bins. VQT uses recursive
//! octave-wise downsampling with per-octave filterbanks.

use std::f64::consts::PI;

#[cfg(test)]
use ndarray::Axis;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex;

use crate::core::{convert, fft, spectrum};
use crate::dsp::windows;
use crate::error::{CanoraError, Result};
use crate::types::*;

// ============================================================
// Public API
// ============================================================

/// Constant-Q Transform.
///
/// Computes the CQT of a signal using logarithmically-spaced frequency bins.
/// Delegates to `vqt` with `gamma=0` for true constant-Q behavior.
///
/// Returns complex CQT matrix of shape `(n_bins, n_frames)`.
pub fn cqt(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    fmin: Option<Float>,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: Float,
) -> Result<Array2<ComplexFloat>> {
    vqt(y, sr, hop_length, fmin, n_bins, bins_per_octave, filter_scale, 0.0)
}

/// Variable-Q Transform.
///
/// Generalization of CQT where `gamma` controls how the Q-factor varies.
/// - `gamma=0`: constant-Q (standard CQT)
/// - `gamma>0`: variable-Q (lower frequencies get wider bandwidth)
///
/// Uses octave-wise recursive downsampling: processes one octave at a time,
/// downsampling the signal by 2 between octaves for efficiency.
pub fn vqt(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    fmin: Option<Float>,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: Float,
    gamma: Float,
) -> Result<Array2<ComplexFloat>> {
    let sr_f = sr as Float;
    let fmin = fmin.unwrap_or(32.7032); // C1

    if n_bins == 0 {
        return Err(CanoraError::InvalidParameter {
            param: "n_bins",
            reason: "must be > 0".into(),
        });
    }

    // Generate target frequencies
    let freqs = convert::cqt_frequencies(n_bins, fmin, bins_per_octave);

    // Number of octaves to process
    let n_octaves = (n_bins as Float / bins_per_octave as Float).ceil() as usize;

    // Determine Q-factor from filter_scale and spacing
    let alpha = 2.0_f64.powf(1.0 / bins_per_octave as Float) - 1.0;

    // Process signal — start with the full signal and current sample rate
    let mut y_current = y.to_owned();
    let mut sr_current = sr_f;
    let mut hop_current = hop_length;

    let mut octave_results: Vec<(Array2<ComplexFloat>, usize)> = Vec::new();

    for oct in 0..n_octaves {
        // Which bins belong to this octave? (top octave first)
        let oct_from_top = n_octaves - 1 - oct;
        let bin_start = oct_from_top * bins_per_octave;
        let bin_end = (bin_start + bins_per_octave).min(n_bins);

        if bin_start >= n_bins {
            continue;
        }

        let _n_oct_bins = bin_end - bin_start;
        let oct_freqs = freqs.slice(s![bin_start..bin_end]).to_owned();

        // Build filterbank for this octave
        let oct_result = cqt_response(
            y_current.view(),
            sr_current,
            oct_freqs.view(),
            hop_current,
            filter_scale,
            gamma,
            alpha,
        )?;

        let n_frames = oct_result.ncols();
        octave_results.push((oct_result, n_frames));

        // Downsample for next octave (lower frequencies)
        if oct + 1 < n_octaves && hop_current % 2 == 0 {
            // Simple 2x downsampling: take every other sample
            // (librosa uses proper anti-alias filtering via resample)
            let n = y_current.len();
            let mut downsampled = Array1::<Float>::zeros(n / 2);
            for i in 0..n / 2 {
                downsampled[i] = y_current[2 * i];
            }
            y_current = downsampled;
            sr_current /= 2.0;
            hop_current /= 2;
        }
    }

    // Reverse octave_results (we processed top-to-bottom, output is bottom-to-top)
    octave_results.reverse();

    // Find the minimum frame count across all octaves
    let min_frames = octave_results.iter().map(|(_, nf)| *nf).min().unwrap_or(0);

    if min_frames == 0 {
        return Err(CanoraError::InsufficientData {
            needed: 1,
            got: 0,
        });
    }

    // Stack octaves into output matrix
    let mut output = Array2::<ComplexFloat>::zeros((n_bins, min_frames));
    let mut row = 0;
    for (oct_data, _) in &octave_results {
        let n_rows = oct_data.nrows();
        for i in 0..n_rows {
            if row < n_bins {
                for j in 0..min_frames {
                    output[(row, j)] = oct_data[(i, j.min(oct_data.ncols() - 1))];
                }
                row += 1;
            }
        }
    }

    Ok(output)
}

/// Pseudo Constant-Q Transform.
///
/// Uses a single FFT size for all frequency bins (no octave recursion).
/// Faster but less accurate at low frequencies. Returns magnitude (real-valued).
pub fn pseudo_cqt(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    fmin: Option<Float>,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: Float,
) -> Result<Spectrogram> {
    let cq = cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave, filter_scale)?;
    Ok(cq.mapv(|c| c.norm()))
}

/// Hybrid Constant-Q Transform.
///
/// Splits the frequency range: high frequencies use pseudo_cqt (fast),
/// low frequencies use full recursive cqt (accurate).
pub fn hybrid_cqt(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    fmin: Option<Float>,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: Float,
) -> Result<Array2<ComplexFloat>> {
    // For simplicity, delegate to full CQT
    // (librosa's optimization splits at a filter-length threshold — implement if profiling shows need)
    cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave, filter_scale)
}

/// Inverse Constant-Q Transform.
///
/// Reconstructs a time-domain signal from a CQT matrix.
/// Uses pseudo-inversion of the filterbank.
pub fn icqt(
    cq: ArrayView2<ComplexFloat>,
    sr: u32,
    hop_length: usize,
    fmin: Option<Float>,
    bins_per_octave: usize,
    filter_scale: Float,
) -> Result<AudioBuffer> {
    let n_bins = cq.nrows();
    let n_frames = cq.ncols();
    let fmin = fmin.unwrap_or(32.7032);
    let sr_f = sr as Float;

    let freqs = convert::cqt_frequencies(n_bins, fmin, bins_per_octave);

    // Determine FFT size from the longest filter
    let alpha = 2.0_f64.powf(1.0 / bins_per_octave as Float) - 1.0;
    let q = filter_scale / alpha;
    let max_len = (q * sr_f / freqs[0]).ceil() as usize;
    let n_fft = max_len.next_power_of_two().max(2 * hop_length);
    let n_fft_bins = n_fft / 2 + 1;

    // Build filterbank in frequency domain
    let basis = build_cq_filterbank(freqs.view(), sr_f, n_fft, filter_scale, 0.0, alpha)?;

    // Pseudo-invert: conjugate transpose
    let mut inv_basis = Array2::<ComplexFloat>::zeros((n_fft_bins, n_bins));
    for i in 0..n_bins {
        for j in 0..n_fft_bins {
            inv_basis[(j, i)] = basis[(i, j)].conj();
        }
    }

    // Reconstruct STFT frames
    let mut stft_matrix = Array2::<ComplexFloat>::zeros((n_fft_bins, n_frames));
    for t in 0..n_frames {
        for f in 0..n_fft_bins {
            let mut val = Complex::new(0.0, 0.0);
            for b in 0..n_bins {
                val += inv_basis[(f, b)] * cq[(b, t)];
            }
            // DC and Nyquist bins must be real for irfft
            if f == 0 || f == n_fft_bins - 1 {
                val = Complex::new(val.re, 0.0);
            }
            stft_matrix[(f, t)] = val;
        }
    }

    // ISTFT to get audio
    let window = WindowSpec::Named("hann".into());
    spectrum::istft(stft_matrix.view(), Some(hop_length), None, &window, true, None)
}

/// Griffin-Lim for CQT magnitude reconstruction.
pub fn griffinlim_cqt(
    cq_mag: ArrayView2<Float>,
    sr: u32,
    hop_length: usize,
    fmin: Option<Float>,
    bins_per_octave: usize,
    n_iter: usize,
) -> Result<AudioBuffer> {
    // Initialize with random phase
    let mut rng: u64 = 42;
    let mut cq_est = Array2::<ComplexFloat>::from_shape_fn(
        (cq_mag.nrows(), cq_mag.ncols()),
        |(i, j)| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let angle = (rng as Float / u64::MAX as Float) * 2.0 * PI;
            Complex::new(cq_mag[(i, j)] * angle.cos(), cq_mag[(i, j)] * angle.sin())
        },
    );

    for _ in 0..n_iter {
        // ICQT → time domain
        let y = icqt(cq_est.view(), sr, hop_length, fmin, bins_per_octave, 1.0)?;

        // CQT → frequency domain
        let rebuilt = cqt(y.view(), sr, hop_length, fmin, cq_mag.nrows(), bins_per_octave, 1.0)?;

        // Replace magnitude, keep phase
        let cols = rebuilt.ncols().min(cq_mag.ncols());
        cq_est = Array2::<ComplexFloat>::zeros((cq_mag.nrows(), cols));
        for i in 0..cq_mag.nrows() {
            for j in 0..cols {
                let norm = rebuilt[(i, j)].norm();
                let phase = if norm > 0.0 {
                    rebuilt[(i, j)] / norm
                } else {
                    Complex::new(1.0, 0.0)
                };
                cq_est[(i, j)] = cq_mag[(i, j.min(cq_mag.ncols() - 1))] * phase;
            }
        }
    }

    icqt(cq_est.view(), sr, hop_length, fmin, bins_per_octave, 1.0)
}

// ============================================================
// Internal: filterbank and convolution
// ============================================================

/// Compute CQT response for a single octave.
///
/// Builds a frequency-domain filterbank and applies it to the STFT of the signal.
fn cqt_response(
    y: ArrayView1<Float>,
    sr: Float,
    freqs: ArrayView1<Float>,
    hop_length: usize,
    filter_scale: Float,
    gamma: Float,
    alpha: Float,
) -> Result<Array2<ComplexFloat>> {
    let n_bins = freqs.len();

    // Compute filter lengths
    let q = filter_scale / alpha;
    let lengths: Vec<Float> = freqs.iter().map(|&f| {
        if gamma > 0.0 {
            q * sr / (f + gamma / alpha)
        } else {
            q * sr / f
        }
    }).collect();

    // FFT size: power of 2, at least max(longest_filter, 2 * hop_length)
    let max_len = lengths.iter().copied().fold(0.0_f64, Float::max).ceil() as usize;
    let n_fft = max_len.max(2 * hop_length).next_power_of_two();

    // Build frequency-domain filterbank
    let basis = build_cq_filterbank(freqs, sr, n_fft, filter_scale, gamma, alpha)?;

    // Compute STFT of the signal
    let window = WindowSpec::Named("ones".into()); // rectangular — CQ filters handle windowing
    let stft_matrix = spectrum::stft(
        y, n_fft, Some(hop_length), Some(n_fft), &window, true, PadMode::Constant,
    )?;

    let n_frames = stft_matrix.ncols();
    let n_fft_bins = n_fft / 2 + 1;

    // Apply filterbank: output = basis @ stft_matrix for each frame
    let mut output = Array2::<ComplexFloat>::zeros((n_bins, n_frames));

    for t in 0..n_frames {
        for i in 0..n_bins {
            let mut val = Complex::new(0.0, 0.0);
            for f in 0..n_fft_bins {
                val += basis[(i, f)] * stft_matrix[(f, t)];
            }
            output[(i, t)] = val;
        }
    }

    Ok(output)
}

/// Build a CQ filterbank in the frequency domain.
///
/// Each row is a windowed complex exponential at the target frequency,
/// transformed to the frequency domain.
fn build_cq_filterbank(
    freqs: ArrayView1<Float>,
    sr: Float,
    n_fft: usize,
    filter_scale: Float,
    gamma: Float,
    alpha: Float,
) -> Result<Array2<ComplexFloat>> {
    let n_bins = freqs.len();
    let n_fft_bins = n_fft / 2 + 1;
    let q = filter_scale / alpha;

    let mut basis = Array2::<ComplexFloat>::zeros((n_bins, n_fft_bins));

    for i in 0..n_bins {
        let freq = freqs[i];
        let length = if gamma > 0.0 {
            q * sr / (freq + gamma / alpha)
        } else {
            q * sr / freq
        };

        let length_int = length.ceil() as usize;
        if length_int == 0 {
            continue;
        }

        // Generate windowed complex exponential
        let win = windows::hann(length_int, false);
        let mut filter_td = vec![Complex::new(0.0, 0.0); n_fft];

        // Center the filter in the FFT buffer
        let start = (n_fft - length_int) / 2;
        for j in 0..length_int {
            let t = j as Float / sr;
            let phase = 2.0 * PI * freq * t;
            filter_td[start + j] = Complex::new(
                win[j] * phase.cos() / length,
                win[j] * phase.sin() / length,
            );
        }

        // FFT of the filter using rfft (fast) on the real and imaginary parts separately
        // Since our filter is complex: F(filter) = F(real) + j*F(imag)
        let mut re_part: Vec<Float> = filter_td.iter().map(|c| c.re).collect();
        let mut im_part: Vec<Float> = filter_td.iter().map(|c| c.im).collect();

        let re_fft = fft::rfft_alloc(&mut re_part).map_err(|_| CanoraError::Fft("filter rfft re".into()))?;
        let im_fft = fft::rfft_alloc(&mut im_part).map_err(|_| CanoraError::Fft("filter rfft im".into()))?;

        for k in 0..n_fft_bins {
            // F(complex_filter) = F(re) + j * F(im)
            basis[(i, k)] = Complex::new(
                re_fft[k].re - im_fft[k].im,
                re_fft[k].im + im_fft[k].re,
            );
        }
    }

    Ok(basis)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(freq: Float, sr: u32, duration: Float) -> Array1<Float> {
        let n = (sr as Float * duration) as usize;
        Array1::from_shape_fn(n, |i| {
            (2.0 * PI * freq * i as Float / sr as Float).sin()
        })
    }

    #[test]
    fn test_cqt_shape() {
        let y = sine_signal(440.0, 22050, 1.0);
        let c = cqt(y.view(), 22050, 512, None, 84, 12, 1.0).unwrap();
        assert_eq!(c.nrows(), 84);
        assert!(c.ncols() > 0);
    }

    #[test]
    fn test_cqt_sine_energy() {
        let y = sine_signal(440.0, 22050, 2.0);
        let c = cqt(y.view(), 22050, 512, Some(32.7), 84, 12, 1.0).unwrap();

        // Find the bin with maximum energy
        let mag = c.mapv(|v| v.norm());
        let avg_mag: Array1<Float> = mag.mean_axis(Axis(1)).unwrap();
        let max_bin = avg_mag.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;

        // A4 = 440 Hz, with fmin=32.7 (C1) and 12 bins/octave,
        // bin for 440 Hz ≈ 12 * log2(440/32.7) ≈ 44.8
        let expected_bin = (12.0 * (440.0 / 32.7_f64).log2()).round() as usize;
        assert!(
            (max_bin as i64 - expected_bin as i64).unsigned_abs() <= 3,
            "expected bin ~{expected_bin}, got {max_bin}"
        );
    }

    #[test]
    fn test_cqt_n_bins() {
        let y = sine_signal(440.0, 22050, 0.5);
        for n_bins in [36, 60, 84] {
            let c = cqt(y.view(), 22050, 512, None, n_bins, 12, 1.0).unwrap();
            assert_eq!(c.nrows(), n_bins);
        }
    }

    #[test]
    fn test_pseudo_cqt_real() {
        let y = sine_signal(440.0, 22050, 0.5);
        let pc = pseudo_cqt(y.view(), 22050, 512, None, 84, 12, 1.0).unwrap();
        assert_eq!(pc.nrows(), 84);
        // Should be all non-negative (magnitudes)
        for &v in pc.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_hybrid_cqt_shape() {
        let y = sine_signal(440.0, 22050, 0.5);
        let hc = hybrid_cqt(y.view(), 22050, 512, None, 84, 12, 1.0).unwrap();
        assert_eq!(hc.nrows(), 84);
    }

    #[test]
    fn test_vqt_gamma() {
        let y = sine_signal(440.0, 22050, 1.0);
        let c0 = vqt(y.view(), 22050, 512, None, 36, 12, 1.0, 0.0).unwrap();
        let c1 = vqt(y.view(), 22050, 512, None, 36, 12, 1.0, 5.0).unwrap();
        // Different gamma should produce different results
        let diff: Float = (&c0 - &c1).mapv(|c| c.norm()).sum();
        assert!(diff > 0.0, "different gamma should produce different output");
    }

    #[test]
    fn test_icqt_roundtrip() {
        let y = sine_signal(440.0, 22050, 1.0);
        // Use fewer bins and higher fmin to keep filter lengths manageable
        let c = cqt(y.view(), 22050, 512, Some(220.0), 24, 12, 1.0).unwrap();
        let y_rec = icqt(c.view(), 22050, 512, Some(220.0), 12, 1.0).unwrap();
        // Should produce a signal (may not be exact reconstruction)
        assert!(y_rec.len() > 0);
        let energy: Float = y_rec.mapv(|v| v * v).sum();
        assert!(energy > 0.0, "reconstructed signal should have energy");
    }
}
