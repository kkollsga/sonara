//! Fused audio analysis pipeline.
//!
//! Computes all common audio features in a single optimized pass,
//! eliminating redundant STFT computation. A typical per-track analysis
//! that previously required 3 separate STFT passes now uses 1.
//!
//! Key optimization: one FFT per frame simultaneously produces mel projection
//! (for onset/beat), spectral centroid, and RMS — no intermediate matrices.

use std::path::Path;

use ndarray::{s, Array1, Array2};

use crate::core::{audio, convert, fft, spectrum};
use crate::dsp::windows;
use crate::error::{CanoraError, Result};
use crate::filters;
use crate::types::*;
use crate::util::utils;

/// Complete analysis result for a single track.
pub struct TrackAnalysis {
    pub duration_sec: Float,
    pub bpm: Float,
    pub beats: Vec<usize>,
    pub onset_frames: Vec<usize>,
    pub rms_mean: Float,
    pub rms_max: Float,
    pub dynamic_range_db: Float,
    pub spectral_centroid_mean: Float,
    pub zero_crossing_rate: Float,
    pub onset_density: Float,
}

/// Analyze a track from a file path — decode, resample, and extract all features.
pub fn analyze_file(path: &Path, sr: u32) -> Result<TrackAnalysis> {
    let (y, actual_sr) = audio::load(path, sr, true, 0.0, 0.0)?;
    analyze_signal(y.view(), actual_sr)
}

/// Analyze a pre-loaded audio signal — single-pass feature extraction.
///
/// This is the core optimization: computes mel spectrogram, spectral centroid,
/// RMS, onset envelope, beat tracking, and onset detection in one fused pass
/// instead of 3 separate STFT computations.
pub fn analyze_signal(y: ndarray::ArrayView1<Float>, sr: u32) -> Result<TrackAnalysis> {
    let sr_f = sr as Float;
    let n_fft = 2048;
    let hop_length = 512;
    let n_mels = 128;
    let n_bins = n_fft / 2 + 1;

    let duration_sec = y.len() as Float / sr_f;

    // ================================================================
    // SETUP: mel filterbank, window, padding (computed once)
    // ================================================================

    let mel_fb = filters::mel(sr_f, n_fft, n_mels, 0.0, sr_f / 2.0, false, "slaney");
    let sparse_mel: Vec<(usize, Vec<Float>)> = (0..n_mels)
        .map(|m| {
            let row = mel_fb.row(m);
            let first = row.iter().position(|&v| v > 0.0).unwrap_or(0);
            let last = row.iter().rposition(|&v| v > 0.0).unwrap_or(0);
            if first > last { (0, vec![]) }
            else { (first, row.slice(s![first..=last]).to_vec()) }
        })
        .collect();
    let freqs = convert::fft_frequencies(sr_f, n_fft);
    let win = windows::get_window(&WindowSpec::Named("hann".into()), n_fft, true)?;
    let win_padded = utils::pad_center(win.view(), n_fft)?;

    let pad = n_fft / 2;
    let mut y_padded = Array1::<Float>::zeros(y.len() + 2 * pad);
    y_padded.slice_mut(s![pad..pad + y.len()]).assign(&y);
    let n = y_padded.len();
    if n < n_fft {
        return Err(CanoraError::InsufficientData { needed: n_fft, got: n });
    }
    let y_raw = y_padded.as_slice().unwrap();
    let win_raw = win_padded.as_slice().unwrap();

    // ================================================================
    // SINGLE PASS: FFT → mel + centroid + rms simultaneously
    // One FFT per frame produces all features. No redundant computation.
    // ================================================================

    let n_frames = 1 + (n - n_fft) / hop_length;
    let mut mel_spec = Array2::<Float>::zeros((n_mels, n_frames));
    let mut centroids = Array1::<Float>::zeros(n_frames);
    let mut rms_frames = Array1::<Float>::zeros(n_frames);

    let mut fft_in = vec![0.0_f64; n_fft];
    let mut fft_out = vec![num_complex::Complex::new(0.0, 0.0); n_bins];
    let mut power_col = vec![0.0_f64; n_bins];

    for t in 0..n_frames {
        let start = t * hop_length;
        for i in 0..n_fft { fft_in[i] = y_raw[start + i] * win_raw[i]; }
        fft::rfft(&mut fft_in, &mut fft_out)?;

        let mut cent_num = 0.0_f64;
        let mut cent_den = 0.0_f64;

        for i in 0..n_bins {
            let pwr = fft_out[i].norm_sqr();
            power_col[i] = pwr;
            let mag = pwr.sqrt();
            cent_num += freqs[i] * mag;
            cent_den += mag;
        }

        centroids[t] = if cent_den > 0.0 { cent_num / cent_den } else { 0.0 };
        let spectral_energy = (power_col[0] + power_col[n_bins - 1]
            + 2.0 * power_col[1..n_bins - 1].iter().sum::<Float>()) / (n_fft as Float * n_fft as Float);
        rms_frames[t] = spectral_energy.sqrt();

        // Sparse mel projection
        for (m, (start_bin, weights)) in sparse_mel.iter().enumerate() {
            let mut sum = 0.0;
            for (k, &w) in weights.iter().enumerate() { sum += w * power_col[start_bin + k]; }
            mel_spec[(m, t)] = sum;
        }
    }

    // ================================================================
    // ONSET STRENGTH from mel spectrogram (no additional FFT)
    // ================================================================

    let s_db = spectrum::power_to_db(mel_spec.view(), 1.0, 1e-10, Some(80.0));
    let lag = 1usize;

    let out_frames = if n_frames > lag { n_frames - lag } else { 0 };
    let mut onset_env = Array1::<Float>::zeros(out_frames);
    for t in 0..out_frames {
        let mut sum = 0.0;
        for m in 0..n_mels {
            sum += (s_db[(m, t + lag)] - s_db[(m, t)]).max(0.0);
        }
        onset_env[t] = sum / n_mels as Float;
    }

    let pad_left = lag + n_fft / (2 * hop_length);
    let total_oenv_frames = out_frames + pad_left;
    let mut oenv_padded = Array1::<Float>::zeros(total_oenv_frames);
    for t in 0..out_frames { oenv_padded[pad_left + t] = onset_env[t]; }

    // ================================================================
    // BEAT TRACKING + ONSET DETECTION (reuse cached onset envelope)
    // ================================================================

    let (bpm, beats) = crate::beat::beat_track(
        None, Some(oenv_padded.view()), sr, hop_length, 120.0, 100.0, true,
    )?;

    let onset_frames = crate::onset::onset_detect(
        None, Some(oenv_padded.view()), sr, hop_length, false, 0.07, 0,
    )?;

    // ================================================================
    // PHASE 5: Zero crossings (trivial, time-domain)
    // ================================================================

    let zc = audio::zero_crossings(y, 0.0);
    let zcr = zc.iter().filter(|&&v| v).count() as Float / y.len() as Float;

    // ================================================================
    // Aggregate results
    // ================================================================

    let rms_mean = rms_frames.iter().sum::<Float>() / rms_frames.len() as Float;
    let rms_max = rms_frames.iter().copied().fold(0.0_f64, Float::max);

    let rms_nonzero: Vec<Float> = rms_frames.iter().copied().filter(|&v| v > 1e-10).collect();
    let dynamic_range_db = if rms_nonzero.len() > 10 {
        let mut sorted = rms_nonzero.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p5 = sorted[sorted.len() * 5 / 100];
        let p95 = sorted[sorted.len() * 95 / 100];
        if p5 > 0.0 { 20.0 * (p95 / p5).log10() } else { 0.0 }
    } else {
        0.0
    };

    let centroid_mean = centroids.iter().sum::<Float>() / centroids.len().max(1) as Float;
    let onset_density = onset_frames.len() as Float / duration_sec;

    Ok(TrackAnalysis {
        duration_sec,
        bpm,
        beats,
        onset_frames,
        rms_mean,
        rms_max,
        dynamic_range_db,
        spectral_centroid_mean: centroid_mean,
        zero_crossing_rate: zcr,
        onset_density,
    })
}

/// Analyze multiple files in parallel using rayon.
pub fn analyze_batch(paths: &[&Path], sr: u32) -> Vec<Result<TrackAnalysis>> {
    use rayon::prelude::*;

    paths
        .par_iter()
        .map(|path| analyze_file(path, sr))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(freq: Float, sr: u32, dur: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin())
    }

    #[test]
    fn test_analyze_signal_basic() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050).unwrap();
        assert!(result.duration_sec > 1.9 && result.duration_sec < 2.1);
        assert!(result.bpm > 30.0 && result.bpm < 320.0);
        assert!(result.rms_mean > 0.0);
        assert!(result.spectral_centroid_mean > 0.0);
    }

    #[test]
    fn test_analyze_signal_click_train() {
        // 120 BPM click train
        let sr = 22050u32;
        let n = (4.0 * sr as Float) as usize;
        let interval = (60.0 / 120.0 * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }

        let result = analyze_signal(y.view(), sr).unwrap();
        // Coarse hop (2048) may detect half-time or double-time
        assert!(result.bpm > 50.0 && result.bpm < 250.0,
            "BPM {} should be near 120 (or harmonic)", result.bpm);
        assert!(result.onset_frames.len() >= 3);
        assert!(result.onset_density > 0.5);
    }

    #[test]
    fn test_analyze_features_reasonable() {
        let y = Array1::from_shape_fn(44100, |i| {
            (2.0 * PI * 440.0 * i as Float / 22050.0).sin() * 0.5
        });
        let result = analyze_signal(y.view(), 22050).unwrap();

        // RMS of 0.5 amplitude sine ≈ 0.354
        assert!(result.rms_mean > 0.1 && result.rms_mean < 0.6,
            "RMS {} unexpected", result.rms_mean);

        // Centroid should be near 440 Hz
        assert!(result.spectral_centroid_mean > 300.0 && result.spectral_centroid_mean < 600.0,
            "Centroid {} unexpected", result.spectral_centroid_mean);
    }
}
