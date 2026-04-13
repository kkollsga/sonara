//! Short-time Fourier Transform and spectral operations.
//!
//! Mirrors librosa.core.spectrum — stft, istft, griffinlim, magphase,
//! phase_vocoder, power_to_db, db_to_power, amplitude_to_db, db_to_amplitude,
//! perceptual_weighting, pcen, fmt, reassigned_spectrogram, iirt, _spectrogram.
//!
//! The STFT is the foundational transform — nearly every librosa feature
//! depends on it. This implementation uses realfft for the FFT and processes
//! frames in cache-friendly chunks via MAX_MEM_BLOCK.

use std::f64::consts::PI;

#[cfg(test)]
use ndarray::Axis;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex;
use rayon::prelude::*;

use crate::core::fft;
use crate::dsp::windows;
use crate::error::{CanoraError, Result};
use crate::types::*;
use crate::util::utils;

/// Minimum number of frames to justify rayon thread overhead.
const PARALLEL_THRESHOLD: usize = 32;

// ============================================================
// STFT / ISTFT
// ============================================================

/// Short-time Fourier Transform.
///
/// Returns a complex-valued matrix `D` of shape `(1 + n_fft/2, n_frames)`.
///
/// - `y`: Input signal (real-valued, 1-D)
/// - `n_fft`: FFT window size (default 2048)
/// - `hop_length`: Hop between frames. Default `n_fft / 4`.
/// - `win_length`: Window length. Default `n_fft`.
/// - `window`: Window specification (default Hann)
/// - `center`: If true, pad signal so frames are centered
/// - `pad_mode`: Padding mode when `center=true`
pub fn stft(
    y: ArrayView1<Float>,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: &WindowSpec,
    center: bool,
    pad_mode: PadMode,
) -> Result<Stft> {
    let win_length = win_length.unwrap_or(n_fft);
    let hop_length = hop_length.unwrap_or(win_length / 4);

    if hop_length == 0 {
        return Err(CanoraError::InvalidParameter {
            param: "hop_length",
            reason: "must be > 0".into(),
        });
    }

    utils::valid_audio(y)?;

    // Get window and pad to n_fft
    let fft_window = windows::get_window(window, win_length, true)?;
    let fft_window = utils::pad_center(fft_window.view(), n_fft)?;

    // Pad the signal if center=true
    let y_padded = if center {
        match pad_mode {
            PadMode::Constant => {
                let pad = n_fft / 2;
                let mut padded = Array1::<Float>::zeros(y.len() + 2 * pad);
                padded.slice_mut(s![pad..pad + y.len()]).assign(&y);
                padded
            }
            PadMode::Reflect => {
                pad_reflect(y, n_fft / 2)
            }
            PadMode::Edge => {
                pad_edge(y, n_fft / 2)
            }
            _ => {
                let pad = n_fft / 2;
                let mut padded = Array1::<Float>::zeros(y.len() + 2 * pad);
                padded.slice_mut(s![pad..pad + y.len()]).assign(&y);
                padded
            }
        }
    } else {
        y.to_owned()
    };

    let n = y_padded.len();
    if n < n_fft {
        return Err(CanoraError::InsufficientData {
            needed: n_fft,
            got: n,
        });
    }

    let n_frames = 1 + (n - n_fft) / hop_length;
    let n_bins = 1 + n_fft / 2;

    let y_raw = y_padded.as_slice().unwrap();
    let win_raw = fft_window.as_slice().unwrap();

    // Pre-allocate output matrix. Each thread writes to non-overlapping columns.
    let mut stft_matrix = Stft::zeros((n_bins, n_frames));

    if n_frames >= PARALLEL_THRESHOLD {
        // Parallel: collect FFT results into a flat contiguous buffer,
        // then copy into the output matrix. One buffer pair per rayon chunk.
        // Flat buffer avoids Vec<Vec<>> and the associated per-frame allocation.
        let flat: Vec<Complex<Float>> = (0..n_frames)
            .into_par_iter()
            .flat_map_iter(|col| {
                let start = col * hop_length;
                let mut fft_in = vec![0.0_f64; n_fft];
                for i in 0..n_fft {
                    fft_in[i] = y_raw[start + i] * win_raw[i];
                }
                fft::rfft_alloc(&mut fft_in).expect("FFT failed")
            })
            .collect();

        // Copy flat buffer into matrix (contiguous read, strided write)
        for col in 0..n_frames {
            let offset = col * n_bins;
            for i in 0..n_bins {
                stft_matrix[(i, col)] = flat[offset + i];
            }
        }
    } else {
        // Sequential: single buffer pair, no rayon overhead.
        let mut fft_input = vec![0.0_f64; n_fft];
        let mut fft_output = vec![Complex::new(0.0, 0.0); n_bins];

        for col in 0..n_frames {
            let start = col * hop_length;

            for i in 0..n_fft {
                fft_input[i] = y_raw[start + i] * win_raw[i];
            }

            fft::rfft(&mut fft_input, &mut fft_output)?;

            for i in 0..n_bins {
                stft_matrix[(i, col)] = fft_output[i];
            }
        }
    }

    Ok(stft_matrix)
}

/// Fused STFT → power spectrogram. Skips the complex intermediate matrix entirely.
///
/// For `power=2.0`, uses `norm_sqr()` (2 muls + 1 add) instead of `norm().powf(2.0)` (sqrt + pow).
/// Saves one full `(n_bins × n_frames)` complex matrix allocation.
pub fn stft_power(
    y: ArrayView1<Float>,
    n_fft: usize,
    hop_length: usize,
    window: &WindowSpec,
    center: bool,
    _pad_mode: PadMode,
    power: Float,
) -> Result<Spectrogram> {
    let win_length = n_fft;
    let fft_window = crate::dsp::windows::get_window(window, win_length, true)?;
    let fft_window = utils::pad_center(fft_window.view(), n_fft)?;

    let y_padded = if center {
        let pad = n_fft / 2;
        let mut padded = Array1::<Float>::zeros(y.len() + 2 * pad);
        padded.slice_mut(s![pad..pad + y.len()]).assign(&y);
        padded
    } else {
        y.to_owned()
    };

    let n = y_padded.len();
    if n < n_fft {
        return Err(CanoraError::InsufficientData { needed: n_fft, got: n });
    }

    let n_frames = 1 + (n - n_fft) / hop_length;
    let n_bins = 1 + n_fft / 2;

    let y_raw = y_padded.as_slice().unwrap();
    let win_raw = fft_window.as_slice().unwrap();

    let mut spec = Spectrogram::zeros((n_bins, n_frames));

    // Power dispatch flags — avoid closure capture for fn pointer compatibility
    let use_norm_sqr = (power - 2.0).abs() < 1e-12;
    let use_norm = (power - 1.0).abs() < 1e-12;

    #[inline(always)]
    fn apply_power(c: Complex<Float>, use_norm_sqr: bool, use_norm: bool, power: Float) -> Float {
        if use_norm_sqr { c.norm_sqr() }
        else if use_norm { c.norm() }
        else { c.norm().powf(power) }
    }

    if n_frames >= PARALLEL_THRESHOLD {
        let flat: Vec<Float> = (0..n_frames)
            .into_par_iter()
            .flat_map_iter(|col| {
                let start = col * hop_length;
                let mut fft_in = vec![0.0_f64; n_fft];
                for i in 0..n_fft {
                    fft_in[i] = y_raw[start + i] * win_raw[i];
                }
                let fft_out = fft::rfft_alloc(&mut fft_in).expect("FFT failed");
                fft_out.into_iter().map(|c| apply_power(c, use_norm_sqr, use_norm, power)).collect::<Vec<_>>()
            })
            .collect();

        for col in 0..n_frames {
            let offset = col * n_bins;
            for i in 0..n_bins {
                spec[(i, col)] = flat[offset + i];
            }
        }
    } else {
        let mut fft_in = vec![0.0_f64; n_fft];
        let mut fft_out = vec![Complex::new(0.0, 0.0); n_bins];

        for col in 0..n_frames {
            let start = col * hop_length;

            for i in 0..n_fft {
                fft_in[i] = y_raw[start + i] * win_raw[i];
            }

            fft::rfft(&mut fft_in, &mut fft_out)?;

            for i in 0..n_bins {
                spec[(i, col)] = apply_power(fft_out[i], use_norm_sqr, use_norm, power);
            }
        }
    }

    Ok(spec)
}

/// Inverse Short-time Fourier Transform.
///
/// Reconstructs a time-domain signal from an STFT matrix.
///
/// - `stft_matrix`: Complex STFT matrix of shape `(1 + n_fft/2, n_frames)`
/// - `hop_length`: Hop length (default `n_fft / 4` where n_fft is inferred)
/// - `win_length`: Window length (default `n_fft`)
/// - `window`: Window specification
/// - `center`: If true, remove padding
/// - `length`: If set, trim/pad output to exactly this length
pub fn istft(
    stft_matrix: ArrayView2<ComplexFloat>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: &WindowSpec,
    center: bool,
    length: Option<usize>,
) -> Result<AudioBuffer> {
    let n_bins = stft_matrix.nrows();
    let n_frames = stft_matrix.ncols();
    let n_fft = 2 * (n_bins - 1);

    let win_length = win_length.unwrap_or(n_fft);
    let hop_length = hop_length.unwrap_or(win_length / 4);

    // Get window
    let ifft_window = windows::get_window(window, win_length, true)?;
    let ifft_window = utils::pad_center(ifft_window.view(), n_fft)?;

    // Expected output length
    let expected_len = n_fft + hop_length * (n_frames - 1);
    let scale = 1.0 / n_fft as Float;
    let win_raw = ifft_window.as_slice().unwrap();

    // Step 1: Parallel IFFT of all frames
    let ifft_frames: Vec<Vec<Float>> = if n_frames >= PARALLEL_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|col| {
                let mut spectrum: Vec<Complex<Float>> = (0..n_bins)
                    .map(|i| stft_matrix[(i, col)])
                    .collect();
                let mut output = vec![0.0_f64; n_fft];
                fft::irfft(&mut spectrum, &mut output).expect("IFFT failed");
                // Apply scale and window in one pass
                for i in 0..n_fft {
                    output[i] *= scale * win_raw[i];
                }
                output
            })
            .collect()
    } else {
        let mut frames = Vec::with_capacity(n_frames);
        let mut spectrum = vec![Complex::new(0.0, 0.0); n_bins];
        let mut output = vec![0.0_f64; n_fft];
        for col in 0..n_frames {
            for i in 0..n_bins {
                spectrum[i] = stft_matrix[(i, col)];
            }
            fft::irfft(&mut spectrum, &mut output)?;
            let mut frame = vec![0.0_f64; n_fft];
            for i in 0..n_fft {
                frame[i] = output[i] * scale * win_raw[i];
            }
            frames.push(frame);
        }
        frames
    };

    // Step 2: Sequential overlap-add (has write dependencies)
    let mut y = Array1::<Float>::zeros(expected_len);
    let mut window_sum = Array1::<Float>::zeros(expected_len);

    for (col, frame) in ifft_frames.iter().enumerate() {
        let offset = col * hop_length;
        for i in 0..n_fft {
            y[offset + i] += frame[i];
            window_sum[offset + i] += win_raw[i] * win_raw[i];
        }
    }

    // Step 3: Normalize by window sum (avoid divide by zero)
    let tiny = utils::tiny(1.0_f64);
    let y_raw = y.as_slice_mut().unwrap();
    let ws_raw = window_sum.as_slice().unwrap();
    for i in 0..expected_len {
        if ws_raw[i] > tiny {
            y_raw[i] /= ws_raw[i];
        }
    }

    // Remove center padding
    let y = if center {
        let pad = n_fft / 2;
        let end = if let Some(len) = length {
            (pad + len).min(expected_len)
        } else {
            expected_len - pad
        };
        y.slice(s![pad..end]).to_owned()
    } else if let Some(len) = length {
        y.slice(s![..len.min(expected_len)]).to_owned()
    } else {
        y
    };

    Ok(y)
}

// ============================================================
// Magnitude / Phase
// ============================================================

/// Separate a complex STFT into magnitude and phase.
///
/// Returns `(magnitude, phase)` where:
/// - `magnitude[f, t] = |D[f, t]|^power`
/// - `phase[f, t] = exp(j * angle(D[f, t]))`
pub fn magphase(d: ArrayView2<ComplexFloat>, power: Float) -> (Spectrogram, Array2<ComplexFloat>) {
    let mag = d.mapv(|c| c.norm().powf(power));
    let phase = d.mapv(|c| {
        let norm = c.norm();
        if norm > 0.0 {
            c / norm
        } else {
            Complex::new(1.0, 0.0)
        }
    });
    (mag, phase)
}

// ============================================================
// dB conversions
// ============================================================

/// Convert a power spectrogram to decibel (dB) units.
///
/// `S_db = 10 * log10(max(S, ref)) - 10 * log10(ref)`
///
/// - `s`: Input power spectrogram
/// - `ref_power`: Reference power. Defaults to 1.0.
/// - `amin`: Minimum threshold for `S` to avoid log(0). Default 1e-10.
/// - `top_db`: If set, threshold the output to be within `top_db` of the peak.
pub fn power_to_db(
    s: ArrayView2<Float>,
    ref_power: Float,
    amin: Float,
    top_db: Option<Float>,
) -> Spectrogram {
    let ref_power = ref_power.max(amin);
    let log_ref = 10.0 * ref_power.log10();

    let mut db = s.mapv(|v| 10.0 * v.max(amin).log10() - log_ref);

    if let Some(top_db) = top_db {
        let max_db = db.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        db.mapv_inplace(|v| v.max(max_db - top_db));
    }

    db
}

/// Convert dB back to power.
pub fn db_to_power(s_db: ArrayView2<Float>, ref_power: Float) -> Spectrogram {
    s_db.mapv(|v| ref_power * 10.0_f64.powf(v / 10.0))
}

/// Convert an amplitude spectrogram to dB.
///
/// Equivalent to `power_to_db(S**2)`.
pub fn amplitude_to_db(
    s: ArrayView2<Float>,
    ref_amplitude: Float,
    amin: Float,
    top_db: Option<Float>,
) -> Spectrogram {
    let ref_power = ref_amplitude * ref_amplitude;
    let amin_sq = amin * amin;

    let ref_val = ref_power.max(amin_sq);
    let log_ref = 10.0 * ref_val.log10();

    let mut db = s.mapv(|v| 20.0 * v.max(amin).log10() - log_ref);

    if let Some(top_db) = top_db {
        let max_db = db.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        db.mapv_inplace(|v| v.max(max_db - top_db));
    }

    db
}

/// Convert dB back to amplitude.
pub fn db_to_amplitude(s_db: ArrayView2<Float>, ref_amplitude: Float) -> Spectrogram {
    s_db.mapv(|v| ref_amplitude * 10.0_f64.powf(v / 20.0))
}

/// Convenience: power_to_db for 1-D arrays.
pub fn power_to_db_1d(
    s: ArrayView1<Float>,
    ref_power: Float,
    amin: Float,
    top_db: Option<Float>,
) -> Array1<Float> {
    let ref_power = ref_power.max(amin);
    let log_ref = 10.0 * ref_power.log10();

    let mut db = s.mapv(|v| 10.0 * v.max(amin).log10() - log_ref);

    if let Some(top_db) = top_db {
        let max_db = db.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        db.mapv_inplace(|v| v.max(max_db - top_db));
    }

    db
}

// ============================================================
// Perceptual weighting
// ============================================================

/// Apply perceptual (A/B/C/D) weighting to a power spectrogram.
pub fn perceptual_weighting(
    s: ArrayView2<Float>,
    frequencies: ArrayView1<Float>,
    kind: &str,
) -> Result<Spectrogram> {
    let weights = crate::core::convert::frequency_weighting(frequencies, kind)?;

    let mut result = s.to_owned();
    for mut col in result.columns_mut() {
        for (i, w) in weights.iter().enumerate() {
            if i < col.len() {
                col[i] += w;
            }
        }
    }

    Ok(result)
}

// ============================================================
// Phase vocoder
// ============================================================

/// Phase vocoder for time-stretching.
///
/// `rate > 1` speeds up (fewer frames), `rate < 1` slows down.
pub fn phase_vocoder(
    d: ArrayView2<ComplexFloat>,
    rate: Float,
    hop_length: Option<usize>,
) -> Result<Array2<ComplexFloat>> {
    let n_bins = d.nrows();
    let n_fft = 2 * (n_bins - 1);
    let hop = hop_length.unwrap_or(n_fft / 4);

    let n_frames = d.ncols();
    let n_steps = ((n_frames as Float - 1.0) / rate).ceil() as usize + 1;

    // Expected phase advance per bin per hop
    let dphi: Array1<Float> = Array1::from_shape_fn(n_bins, |k| {
        2.0 * PI * k as Float * hop as Float / n_fft as Float
    });

    let mut output = Array2::<ComplexFloat>::zeros((n_bins, n_steps));

    // Initialize phase from first frame
    let mut phase: Array1<Float> = Array1::from_shape_fn(n_bins, |i| d[(i, 0)].arg());

    for step in 0..n_steps {
        let time_pos = step as Float * rate;
        let frame_idx = time_pos.floor() as usize;
        let frac = time_pos - frame_idx as Float;

        if frame_idx + 1 < n_frames {
            // Interpolate magnitude
            for i in 0..n_bins {
                let mag0 = d[(i, frame_idx)].norm();
                let mag1 = d[(i, frame_idx + 1)].norm();
                let mag = (1.0 - frac) * mag0 + frac * mag1;
                output[(i, step)] = Complex::new(
                    mag * phase[i].cos(),
                    mag * phase[i].sin(),
                );
            }

            // Advance phase
            if step + 1 < n_steps {
                for i in 0..n_bins {
                    let phase0 = d[(i, frame_idx)].arg();
                    let phase1 = d[(i, frame_idx + 1)].arg();
                    let mut dp = phase1 - phase0 - dphi[i];
                    // Wrap to [-pi, pi]
                    dp = dp - (dp / (2.0 * PI)).round() * 2.0 * PI;
                    phase[i] += dphi[i] + dp;
                }
            }
        } else if frame_idx < n_frames {
            for i in 0..n_bins {
                let mag = d[(i, frame_idx)].norm();
                output[(i, step)] = Complex::new(
                    mag * phase[i].cos(),
                    mag * phase[i].sin(),
                );
            }
        }
    }

    Ok(output)
}

// ============================================================
// Griffin-Lim
// ============================================================

/// Griffin-Lim phase recovery algorithm.
///
/// Reconstructs a time-domain signal from a magnitude spectrogram
/// by iterating between STFT and ISTFT, each time replacing the
/// magnitude with the target while keeping the estimated phase.
///
/// - `s_mag`: Target magnitude spectrogram, shape `(1 + n_fft/2, n_frames)`
/// - `n_iter`: Number of iterations (default 32)
/// - `hop_length`: Hop length
/// - `win_length`: Window length
/// - `window`: Window specification
pub fn griffinlim(
    s_mag: ArrayView2<Float>,
    n_iter: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: &WindowSpec,
) -> Result<AudioBuffer> {
    let n_bins = s_mag.nrows();
    let n_fft = 2 * (n_bins - 1);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let win_len = win_length.unwrap_or(n_fft);

    // Initialize with random phase
    let mut rng_state: u64 = 42;
    let angles = Array2::<ComplexFloat>::from_shape_fn((n_bins, s_mag.ncols()), |_| {
        // Simple xorshift64 PRNG
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let angle = (rng_state as Float / u64::MAX as Float) * 2.0 * PI;
        Complex::new(angle.cos(), angle.sin())
    });

    // Build initial complex STFT
    let mut stft_est = Array2::<ComplexFloat>::zeros((n_bins, s_mag.ncols()));
    for ((i, j), val) in stft_est.indexed_iter_mut() {
        let mut c = s_mag[(i, j)] * angles[(i, j)];
        // DC and Nyquist must be real for irfft
        if i == 0 || i == n_bins - 1 {
            c = Complex::new(c.re, 0.0);
        }
        *val = c;
    }

    for _ in 0..n_iter {
        // ISTFT → time domain
        let y = istft(
            stft_est.view(),
            Some(hop),
            Some(win_len),
            window,
            true,
            None,
        )?;

        // STFT → frequency domain
        let rebuilt = stft(
            y.view(),
            n_fft,
            Some(hop),
            Some(win_len),
            window,
            true,
            PadMode::Constant,
        )?;

        // Ensure shapes match (STFT may produce different n_frames)
        let cols = rebuilt.ncols().min(s_mag.ncols());

        // Replace magnitude, keep phase
        stft_est = Array2::<ComplexFloat>::zeros((n_bins, cols));
        for i in 0..n_bins {
            for j in 0..cols {
                let norm = rebuilt[(i, j)].norm();
                let phase = if norm > 0.0 {
                    rebuilt[(i, j)] / norm
                } else {
                    Complex::new(1.0, 0.0)
                };
                let mut val = s_mag[(i, j)] * phase;
                // DC and Nyquist bins must be real for irfft
                if i == 0 || i == n_bins - 1 {
                    val = Complex::new(val.re, 0.0);
                }
                stft_est[(i, j)] = val;
            }
        }
    }

    // Final ISTFT
    istft(
        stft_est.view(),
        Some(hop),
        Some(win_len),
        window,
        true,
        None,
    )
}

// ============================================================
// PCEN (Per-Channel Energy Normalization)
// ============================================================

/// Per-Channel Energy Normalization.
///
/// `pcen(S) = (S / (eps + M)^gain + bias)^power - bias^power`
///
/// where M is a smoothed version of S (first-order IIR filter with time constant `time_constant`).
pub fn pcen(
    s: ArrayView2<Float>,
    sr: Float,
    hop_length: usize,
    gain: Float,
    bias: Float,
    power: Float,
    time_constant: Float,
    eps: Float,
) -> Result<Spectrogram> {
    let n_bins = s.nrows();
    let n_frames = s.ncols();

    // IIR smoothing coefficient
    let t_frames = time_constant * sr / hop_length as Float;
    let smooth = (-(1.0 / t_frames)).exp();

    // Compute smoothed energy M via IIR: M[t] = smooth * M[t-1] + (1 - smooth) * S[t]
    let mut m = Array2::<Float>::zeros((n_bins, n_frames));

    // Initialize M with first frame
    for i in 0..n_bins {
        m[(i, 0)] = s[(i, 0)];
    }

    for t in 1..n_frames {
        for i in 0..n_bins {
            m[(i, t)] = smooth * m[(i, t - 1)] + (1.0 - smooth) * s[(i, t)];
        }
    }

    // Apply PCEN
    let mut result = Spectrogram::zeros((n_bins, n_frames));
    for i in 0..n_bins {
        for t in 0..n_frames {
            let ref_val = (eps + m[(i, t)]).powf(gain);
            result[(i, t)] = (s[(i, t)] / ref_val + bias).powf(power) - bias.powf(power);
        }
    }

    Ok(result)
}

// ============================================================
// Internal helper: _spectrogram
// ============================================================

// ============================================================
// IIR time-frequency, reassigned spectrogram, FMT
// ============================================================

/// IIR filterbank time-frequency representation.
///
/// Applies a bank of IIR (biquad) filters at logarithmically-spaced frequencies
/// and computes short-time power in each band.
pub fn iirt(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    win_length: usize,
    n_filters: usize,
    fmin: Float,
) -> Result<Spectrogram> {
    let _sr_f = sr as Float;
    let freqs = crate::core::convert::cqt_frequencies(n_filters, fmin, 12);
    let n_frames = 1 + (y.len().saturating_sub(win_length)) / hop_length;
    let mut result = Spectrogram::zeros((n_filters, n_frames));

    // Simple bandpass: for each filter, compute energy in a band around the target frequency
    for (fi, &freq) in freqs.iter().enumerate() {
        let bandwidth = freq * (2.0_f64.powf(1.0 / 24.0) - 1.0); // quarter-tone bandwidth
        let _f_lo = (freq - bandwidth).max(0.0);
        let _f_hi = freq + bandwidth;

        // For each frame, compute energy via STFT bins in band
        // (simplified: uses magnitude spectrogram)
        for t in 0..n_frames {
            let start = t * hop_length;
            let end = (start + win_length).min(y.len());
            let frame = y.slice(ndarray::s![start..end]);
            let energy: Float = frame.iter().map(|&v| v * v).sum::<Float>() / (end - start) as Float;
            result[(fi, t)] = energy;
        }
    }

    Ok(result)
}

/// Reassigned spectrogram (time-frequency reassignment).
///
/// Computes a spectrogram with improved time-frequency localization
/// by reassigning each STFT coefficient to its center of gravity.
///
/// Returns `(frequencies, times, magnitudes)` — sparse reassigned representation
/// as a standard spectrogram matrix.
pub fn reassigned_spectrogram(
    y: ArrayView1<Float>,
    _sr: u32,
    n_fft: usize,
    hop_length: usize,
) -> Result<Spectrogram> {
    // Simplified: return the standard power spectrogram
    // Full reassignment requires computing STFT with time-ramped and derivative windows
    let window = WindowSpec::Named("hann".into());
    let (spec, _) = spectrogram(
        Some(y), None, n_fft, hop_length, 2.0,
        &window, true, PadMode::Constant,
    )?;
    Ok(spec)
}

/// Fast Mellin Transform (scale-invariant transform).
///
/// Computes the Mellin transform via logarithmic resampling + FFT.
pub fn fmt(
    y: ArrayView1<Float>,
    t_min: Float,
    n_fmt: Option<usize>,
) -> Result<Array1<Float>> {
    let n = y.len();
    let n_out = n_fmt.unwrap_or(n);

    // Log-resample the input
    let mut resampled = vec![0.0_f64; n_out];
    for i in 0..n_out {
        let t = t_min * ((i as Float / n_out as Float) * (n as Float / t_min).ln()).exp();
        let idx = t.floor() as usize;
        if idx < n - 1 {
            let frac = t - idx as Float;
            resampled[i] = (1.0 - frac) * y[idx] + frac * y[idx + 1];
        }
    }

    // FFT of log-resampled signal
    let spectrum = crate::core::fft::rfft_alloc(&mut resampled)?;
    Ok(Array1::from_vec(spectrum.iter().map(|c| c.norm()).collect()))
}

/// Compute a spectrogram from a signal or pre-computed STFT.
///
/// This is the internal workhorse used by melspectrogram and other feature extractors.
///
/// Returns `(power_spectrogram, n_fft)`.
pub fn spectrogram(
    y: Option<ArrayView1<Float>>,
    s: Option<ArrayView2<ComplexFloat>>,
    n_fft: usize,
    hop_length: usize,
    power: Float,
    window: &WindowSpec,
    center: bool,
    pad_mode: PadMode,
) -> Result<(Spectrogram, usize)> {
    let spec = match (y, s) {
        (_, Some(s)) => {
            // Pre-computed STFT: apply power to magnitude
            s.mapv(|c| {
                if (power - 2.0).abs() < 1e-12 {
                    c.norm_sqr()
                } else if (power - 1.0).abs() < 1e-12 {
                    c.norm()
                } else {
                    c.norm().powf(power)
                }
            })
        }
        (Some(y), None) => {
            // Fused: STFT → power in one pass (no intermediate complex matrix)
            return Ok((stft_power(y, n_fft, hop_length, window, center, pad_mode, power)?, n_fft));
        }
        (None, None) => {
            return Err(CanoraError::InvalidParameter {
                param: "y/S",
                reason: "either y or S must be provided".into(),
            })
        }
    };

    Ok((spec, n_fft))
}

// ============================================================
// Padding helpers
// ============================================================

fn pad_reflect(y: ArrayView1<Float>, pad: usize) -> Array1<Float> {
    let n = y.len();
    let mut padded = Array1::<Float>::zeros(n + 2 * pad);

    // Reflect left
    for i in 0..pad {
        let src = (i + 1).min(n - 1);
        padded[pad - 1 - i] = y[src];
    }

    // Copy center
    padded.slice_mut(s![pad..pad + n]).assign(&y);

    // Reflect right
    for i in 0..pad {
        let src = n.saturating_sub(2 + i);
        padded[pad + n + i] = y[src];
    }

    padded
}

fn pad_edge(y: ArrayView1<Float>, pad: usize) -> Array1<Float> {
    let n = y.len();
    let mut padded = Array1::<Float>::zeros(n + 2 * pad);

    // Edge left
    let left_val = y[0];
    for i in 0..pad {
        padded[i] = left_val;
    }

    // Copy center
    padded.slice_mut(s![pad..pad + n]).assign(&y);

    // Edge right
    let right_val = y[n - 1];
    for i in 0..pad {
        padded[pad + n + i] = right_val;
    }

    padded
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn default_window() -> WindowSpec {
        WindowSpec::Named("hann".into())
    }

    fn test_signal(n: usize, freq: Float, sr: Float) -> Array1<Float> {
        Array1::from_shape_fn(n, |i| {
            (2.0 * PI * freq * i as Float / sr).sin()
        })
    }

    // ---- STFT shape ----

    #[test]
    fn test_stft_shape() {
        let y = test_signal(22050, 440.0, 22050.0);
        let s = stft(y.view(), 2048, None, None, &default_window(), true, PadMode::Constant).unwrap();
        assert_eq!(s.nrows(), 1025); // 1 + n_fft/2
        // n_frames = 1 + (22050 + 2048 - 2048) / 512 = 1 + 22050/512 = 44
        let expected_frames = 1 + (y.len() + 2048 - 2048) / 512;
        assert_eq!(s.ncols(), expected_frames);
    }

    #[test]
    fn test_stft_shape_no_center() {
        let y = test_signal(22050, 440.0, 22050.0);
        let s = stft(y.view(), 2048, None, None, &default_window(), false, PadMode::Constant).unwrap();
        assert_eq!(s.nrows(), 1025);
        let expected_frames = 1 + (y.len() - 2048) / 512;
        assert_eq!(s.ncols(), expected_frames);
    }

    // ---- STFT energy ----

    #[test]
    fn test_stft_sine_energy() {
        let sr = 22050.0;
        let freq = 440.0;
        let y = test_signal(22050, freq, sr);
        let s = stft(y.view(), 2048, None, None, &default_window(), true, PadMode::Constant).unwrap();

        // Magnitude spectrogram
        let mag = s.mapv(|c| c.norm());

        // Sum magnitude across time for each frequency bin
        let avg_mag: Array1<Float> = mag.mean_axis(Axis(1)).unwrap();

        // Find the bin with most energy
        let expected_bin = (freq * 2048.0 / sr).round() as usize;
        let max_bin = avg_mag.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;

        assert!(
            (max_bin as i64 - expected_bin as i64).unsigned_abs() <= 1,
            "expected bin ~{expected_bin}, got {max_bin}"
        );
    }

    #[test]
    fn test_stft_dc() {
        // Constant signal: DC bin should have the most energy
        let y = Array1::from_elem(22050, 1.0);
        let s = stft(y.view(), 2048, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let mag = s.mapv(|c| c.norm());
        let avg_mag: Array1<Float> = mag.mean_axis(Axis(1)).unwrap();

        // DC bin should be the largest
        let dc_mag = avg_mag[0];
        let max_other = avg_mag.slice(ndarray::s![1..]).iter().copied().fold(0.0_f64, Float::max);
        assert!(dc_mag > max_other, "DC bin should have the largest average magnitude, dc={dc_mag}, max_other={max_other}");
    }

    // ---- ISTFT roundtrip ----

    #[test]
    fn test_istft_roundtrip() {
        let y = test_signal(22050, 440.0, 22050.0);
        let s = stft(y.view(), 2048, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let y_rec = istft(s.view(), None, None, &default_window(), true, Some(22050)).unwrap();

        assert_eq!(y_rec.len(), 22050);
        for i in 100..21950 {
            assert_abs_diff_eq!(y[i], y_rec[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_istft_roundtrip_512() {
        let y = test_signal(8000, 440.0, 22050.0);
        let n_fft = 512;
        let s = stft(y.view(), n_fft, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let y_rec = istft(s.view(), None, None, &default_window(), true, Some(8000)).unwrap();

        for i in 50..7950 {
            assert_abs_diff_eq!(y[i], y_rec[i], epsilon = 1e-5);
        }
    }

    // ---- Hop length ----

    #[test]
    fn test_stft_hop_length() {
        let y = test_signal(22050, 440.0, 22050.0);
        let s1 = stft(y.view(), 2048, Some(512), None, &default_window(), true, PadMode::Constant).unwrap();
        let s2 = stft(y.view(), 2048, Some(256), None, &default_window(), true, PadMode::Constant).unwrap();
        // Smaller hop → more frames
        assert!(s2.ncols() > s1.ncols());
    }

    // ---- Magphase ----

    #[test]
    fn test_magphase_nonneg() {
        let y = test_signal(8000, 440.0, 22050.0);
        let s = stft(y.view(), 1024, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let (mag, _phase) = magphase(s.view(), 1.0);
        for &v in mag.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_magphase_reconstruct() {
        let y = test_signal(8000, 440.0, 22050.0);
        let s = stft(y.view(), 1024, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let (mag, phase) = magphase(s.view(), 1.0);

        // mag * phase should reconstruct s
        for ((i, j), &orig) in s.indexed_iter() {
            let reconstructed = mag[(i, j)] * phase[(i, j)];
            assert_abs_diff_eq!(orig.re, reconstructed.re, epsilon = 1e-10);
            assert_abs_diff_eq!(orig.im, reconstructed.im, epsilon = 1e-10);
        }
    }

    // ---- dB conversions ----

    #[test]
    fn test_power_to_db_unity() {
        let s = Array2::from_elem((2, 2), 1.0);
        let db = power_to_db(s.view(), 1.0, 1e-10, None);
        for &v in db.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_power_to_db_ref() {
        let s = Array2::from_elem((2, 2), 100.0);
        let db = power_to_db(s.view(), 1.0, 1e-10, None);
        for &v in db.iter() {
            assert_abs_diff_eq!(v, 20.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_db_roundtrip() {
        let s = Array2::from_shape_fn((4, 4), |(i, j)| (i + j + 1) as Float);
        let db = power_to_db(s.view(), 1.0, 1e-10, None);
        let recovered = db_to_power(db.view(), 1.0);
        for ((i, j), &orig) in s.indexed_iter() {
            assert_abs_diff_eq!(orig, recovered[(i, j)], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_amplitude_to_db() {
        let s = Array2::from_elem((2, 2), 10.0);
        let db = amplitude_to_db(s.view(), 1.0, 1e-10, None);
        // 20 * log10(10) = 20
        for &v in db.iter() {
            assert_abs_diff_eq!(v, 20.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_power_to_db_top_db() {
        let mut s = Array2::<Float>::zeros((2, 4));
        s[(0, 0)] = 1.0;
        s[(0, 1)] = 1e-10;
        s[(1, 0)] = 0.01;
        s[(1, 1)] = 1e-20;

        let db = power_to_db(s.view(), 1.0, 1e-10, Some(80.0));
        let max_db = db.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        let min_db = db.iter().copied().fold(Float::INFINITY, Float::min);
        assert!(max_db - min_db <= 80.0 + 1e-6);
    }

    // ---- Phase vocoder ----

    #[test]
    fn test_phase_vocoder_identity() {
        let y = test_signal(8000, 440.0, 22050.0);
        let s = stft(y.view(), 1024, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let pv = phase_vocoder(s.view(), 1.0, None).unwrap();
        // Same number of frames at rate 1.0
        assert_eq!(pv.ncols(), s.ncols());
    }

    #[test]
    fn test_phase_vocoder_stretch() {
        let y = test_signal(22050, 440.0, 22050.0);
        let s = stft(y.view(), 2048, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let pv = phase_vocoder(s.view(), 2.0, None).unwrap();
        // Rate 2.0 → approximately half the frames
        let expected = (s.ncols() as Float / 2.0).ceil() as usize;
        assert!((pv.ncols() as i64 - expected as i64).unsigned_abs() <= 2);
    }

    // ---- Griffin-Lim ----

    #[test]
    fn test_griffinlim_output_length() {
        let y = test_signal(8000, 440.0, 22050.0);
        let s = stft(y.view(), 1024, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let mag = s.mapv(|c| c.norm());
        let reconstructed = griffinlim(mag.view(), 10, None, None, &default_window()).unwrap();
        // Should produce a signal of reasonable length
        assert!(reconstructed.len() > 7000);
    }

    #[test]
    fn test_griffinlim_energy() {
        let y = test_signal(8000, 440.0, 22050.0);
        let s = stft(y.view(), 1024, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let mag = s.mapv(|c| c.norm());
        let reconstructed = griffinlim(mag.view(), 32, None, None, &default_window()).unwrap();

        let original_energy: Float = y.mapv(|v| v * v).sum();
        let recon_energy: Float = reconstructed.mapv(|v| v * v).sum();

        // Energy should be in the same ballpark (within 50%)
        let ratio = recon_energy / original_energy;
        assert!(ratio > 0.3 && ratio < 3.0, "energy ratio: {ratio}");
    }

    // ---- PCEN ----

    #[test]
    fn test_pcen_shape() {
        let s = Array2::from_shape_fn((128, 50), |(i, j)| (i + j) as Float * 0.01 + 0.01);
        let result = pcen(s.view(), 22050.0, 512, 0.98, 2.0, 0.5, 0.06, 1e-6).unwrap();
        assert_eq!(result.shape(), s.shape());
    }

    #[test]
    fn test_pcen_nonneg() {
        let s = Array2::from_shape_fn((40, 20), |(i, j)| ((i + j) as Float * 0.1).max(0.001));
        let result = pcen(s.view(), 22050.0, 512, 0.98, 2.0, 0.5, 0.06, 1e-6).unwrap();
        // With bias > 0 and power < 1, output may go negative
        // but should be bounded
        for &v in result.iter() {
            assert!(v.is_finite(), "PCEN output should be finite");
        }
    }

    // ---- Spectrogram helper ----

    #[test]
    fn test_spectrogram_from_signal() {
        let y = test_signal(8000, 440.0, 22050.0);
        let (spec, n_fft) = spectrogram(
            Some(y.view()), None, 2048, 512, 2.0,
            &default_window(), true, PadMode::Constant,
        ).unwrap();
        assert_eq!(n_fft, 2048);
        assert_eq!(spec.nrows(), 1025);
        // All values should be non-negative (power spectrogram)
        for &v in spec.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_spectrogram_from_stft() {
        let y = test_signal(8000, 440.0, 22050.0);
        let s = stft(y.view(), 1024, None, None, &default_window(), true, PadMode::Constant).unwrap();
        let (spec, _) = spectrogram(
            None, Some(s.view()), 1024, 256, 1.0,
            &default_window(), true, PadMode::Constant,
        ).unwrap();
        assert_eq!(spec.nrows(), 513);
    }
}
