//! Feature inversion — reconstruct audio from spectral features.
//!
//! Mirrors librosa.feature.inverse — mel_to_stft, mel_to_audio, mfcc_to_mel, mfcc_to_audio.

use std::f64::consts::PI;

use ndarray::{Array2, ArrayView2};

use crate::core::spectrum;
use crate::error::Result;
use crate::filters;
use crate::types::*;

/// Approximate STFT magnitude from mel spectrogram via NNLS.
///
/// Inverts the mel filterbank projection using non-negative least squares.
pub fn mel_to_stft(
    m: ArrayView2<Float>,
    sr: Float,
    n_fft: usize,
    power: Float,
) -> Result<Array2<Float>> {
    let n_mels = m.nrows();
    let n_frames = m.ncols();
    let n_bins = n_fft / 2 + 1;

    let mel_fb = filters::mel(sr, n_fft, n_mels, 0.0, 0.0, false, "slaney");

    // Solve: mel_fb @ X ≈ M, X ≥ 0
    // Simple approach: use pseudo-inverse with non-negativity clipping
    // mel_fb: (n_mels, n_bins), M: (n_mels, n_frames)
    // X ≈ pinv(mel_fb) @ M, clipped to ≥ 0

    // Compute pseudo-inverse via normal equations: X = (FB^T @ FB)^{-1} @ FB^T @ M
    // For simplicity, use iterative projected gradient descent
    let mut x = Array2::<Float>::zeros((n_bins, n_frames));

    let learning_rate = 0.01;
    let n_iter = 50;

    for _ in 0..n_iter {
        // Gradient: 2 * mel_fb^T @ (mel_fb @ X - M)
        let residual = mel_fb.dot(&x) - &m;
        let grad = mel_fb.t().dot(&residual);

        // Update with non-negativity projection
        for i in 0..n_bins {
            for j in 0..n_frames {
                x[(i, j)] = (x[(i, j)] - learning_rate * grad[(i, j)]).max(0.0);
            }
        }
    }

    // Apply power inversion
    if (power - 1.0).abs() > 1e-10 {
        let inv_power = 1.0 / power;
        x.mapv_inplace(|v| v.powf(inv_power));
    }

    Ok(x)
}

/// Reconstruct audio from mel spectrogram.
pub fn mel_to_audio(
    m: ArrayView2<Float>,
    sr: Float,
    n_fft: usize,
    hop_length: usize,
    n_iter: usize,
) -> Result<AudioBuffer> {
    let stft_mag = mel_to_stft(m, sr, n_fft, 2.0)?;
    let window = WindowSpec::Named("hann".into());
    spectrum::griffinlim(stft_mag.view(), n_iter, Some(hop_length), None, &window)
}

/// Convert MFCC back to approximate mel spectrogram via inverse DCT.
pub fn mfcc_to_mel(
    mfcc: ArrayView2<Float>,
    n_mels: usize,
) -> Result<Array2<Float>> {
    let n_mfcc = mfcc.nrows();
    let n_frames = mfcc.ncols();

    // Inverse DCT-II (orthonormal): reconstruct n_mels from n_mfcc coefficients
    // IDCT: x[n] = sum_k X[k] * cos(pi * k * (2n+1) / (2N)) * norm
    let mut mel = Array2::<Float>::zeros((n_mels, n_frames));

    for t in 0..n_frames {
        for n in 0..n_mels {
            let mut sum = 0.0;
            for k in 0..n_mfcc {
                let norm = if k == 0 {
                    (1.0 / n_mels as Float).sqrt()
                } else {
                    (2.0 / n_mels as Float).sqrt()
                };
                sum += mfcc[(k, t)] * norm
                    * (PI * k as Float * (2.0 * n as Float + 1.0) / (2.0 * n_mels as Float)).cos();
            }
            mel[(n, t)] = sum;
        }
    }

    // Convert from dB back to power
    mel.mapv_inplace(|v| 10.0_f64.powf(v / 10.0));

    Ok(mel)
}

/// Reconstruct audio from MFCC.
pub fn mfcc_to_audio(
    mfcc: ArrayView2<Float>,
    sr: Float,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    n_iter: usize,
) -> Result<AudioBuffer> {
    let mel = mfcc_to_mel(mfcc, n_mels)?;
    mel_to_audio(mel.view(), sr, n_fft, hop_length, n_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_to_stft_shape() {
        let m = Array2::from_shape_fn((128, 20), |(i, j)| (i + j + 1) as Float * 0.01);
        let stft = mel_to_stft(m.view(), 22050.0, 2048, 2.0).unwrap();
        assert_eq!(stft.nrows(), 1025);
        assert_eq!(stft.ncols(), 20);
    }

    #[test]
    fn test_mel_to_stft_nonneg() {
        let m = Array2::from_shape_fn((40, 10), |(i, j)| (i + j + 1) as Float * 0.01);
        let stft = mel_to_stft(m.view(), 22050.0, 2048, 2.0).unwrap();
        for &v in stft.iter() {
            assert!(v >= 0.0, "STFT magnitude should be non-negative");
        }
    }

    #[test]
    fn test_mfcc_to_mel_shape() {
        let mfcc = Array2::from_shape_fn((20, 10), |(i, _j)| (i as Float - 10.0) * 0.1);
        let mel = mfcc_to_mel(mfcc.view(), 128).unwrap();
        assert_eq!(mel.nrows(), 128);
        assert_eq!(mel.ncols(), 10);
    }

    #[test]
    fn test_mel_to_audio_produces_signal() {
        let m = Array2::from_shape_fn((40, 20), |(i, j)| ((i + j) as Float * 0.1).max(0.001));
        let y = mel_to_audio(m.view(), 22050.0, 2048, 512, 10).unwrap();
        assert!(y.len() > 0);
        let energy: Float = y.mapv(|v| v * v).sum();
        assert!(energy > 0.0);
    }
}
