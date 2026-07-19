//! Audio effects.
//!
//! Audio effects — HPSS, time_stretch, pitch_shift, trim, split, remix,
//! preemphasis, deemphasis, and melody separation.

use ndarray::{s, Array1, Array2, ArrayView1};
use num_complex::Complex;

use crate::core::pitch;
use crate::core::{audio, convert, spectrum};
use crate::decompose;
use crate::dsp::iir;
use crate::error::Result;
use crate::feature::spectral;
use crate::types::*;

/// Harmonic-Percussive Source Separation (audio domain).
///
/// Returns `(harmonic, percussive)` audio signals.
pub fn hpss(
    y: ArrayView1<Float>,
    kernel_size: usize,
    margin: Float,
) -> Result<(AudioBuffer, AudioBuffer)> {
    let window = WindowSpec::Named("hann".into());
    let stft = spectrum::stft(y, 2048, None, None, &window, true, PadMode::Constant)?;
    let mag = stft.mapv(|c| c.norm());

    let (h_mag, p_mag) = decompose::hpss(mag.view(), kernel_size, margin, 2.0)?;

    // Reconstruct phase from original STFT
    let h_stft = ndarray::Array2::from_shape_fn(stft.raw_dim(), |(i, j)| {
        let norm = stft[(i, j)].norm();
        if norm > 0.0 {
            stft[(i, j)] * h_mag[(i, j)] / norm
        } else {
            num_complex::Complex::new(0.0, 0.0)
        }
    });
    let p_stft = ndarray::Array2::from_shape_fn(stft.raw_dim(), |(i, j)| {
        let norm = stft[(i, j)].norm();
        if norm > 0.0 {
            stft[(i, j)] * p_mag[(i, j)] / norm
        } else {
            num_complex::Complex::new(0.0, 0.0)
        }
    });

    let length = Some(y.len());
    let h = spectrum::istft(h_stft.view(), None, None, &window, true, length)?;
    let p = spectrum::istft(p_stft.view(), None, None, &window, true, length)?;

    Ok((h, p))
}

/// Extract harmonic component only.
pub fn harmonic(y: ArrayView1<Float>, kernel_size: usize, margin: Float) -> Result<AudioBuffer> {
    let (h, _) = hpss(y, kernel_size, margin)?;
    Ok(h)
}

/// Extract percussive component only.
pub fn percussive(y: ArrayView1<Float>, kernel_size: usize, margin: Float) -> Result<AudioBuffer> {
    let (_, p) = hpss(y, kernel_size, margin)?;
    Ok(p)
}

/// Time-stretch an audio signal without changing pitch.
pub fn time_stretch(y: ArrayView1<Float>, rate: Float) -> Result<AudioBuffer> {
    let window = WindowSpec::Named("hann".into());
    let stft = spectrum::stft(y, 2048, None, None, &window, true, PadMode::Constant)?;
    let mut stretched = spectrum::phase_vocoder(stft.view(), rate, None)?;

    // Enforce real DC/Nyquist for irfft compatibility
    let n_bins = stretched.nrows();
    for t in 0..stretched.ncols() {
        stretched[(0, t)] = num_complex::Complex::new(stretched[(0, t)].re, 0.0);
        stretched[(n_bins - 1, t)] = num_complex::Complex::new(stretched[(n_bins - 1, t)].re, 0.0);
    }

    let length = (y.len() as Float / rate).round() as usize;
    spectrum::istft(stretched.view(), None, None, &window, true, Some(length))
}

/// Shift the pitch of an audio signal.
pub fn pitch_shift(
    y: ArrayView1<Float>,
    sr: u32,
    n_steps: Float,
    bins_per_octave: usize,
) -> Result<AudioBuffer> {
    let rate = 2.0_f32.powf(-n_steps / bins_per_octave as Float);
    let stretched = time_stretch(y, rate)?;
    let target_sr = (sr as Float * rate) as u32;
    audio::resample(stretched.view(), target_sr, sr)
}

/// Trim silence from the edges of an audio signal.
///
/// Returns `(trimmed_signal, (start_sample, end_sample))`.
pub fn trim(
    y: ArrayView1<Float>,
    top_db: Float,
    frame_length: usize,
    hop_length: usize,
) -> Result<(AudioBuffer, (usize, usize))> {
    let rms = spectral::rms(Some(y), None, frame_length, hop_length)?;
    let rms_row = rms.row(0);

    let ref_val = rms_row.iter().copied().fold(0.0_f32, Float::max);
    if ref_val <= 0.0 {
        return Ok((Array1::zeros(0), (0, 0)));
    }

    let threshold = ref_val * 10.0_f32.powf(-top_db / 20.0);

    // Find first and last frames above threshold
    let first_frame = rms_row.iter().position(|&v| v >= threshold).unwrap_or(0);
    let last_frame = rms_row.iter().rposition(|&v| v >= threshold).unwrap_or(0);

    let start = first_frame * hop_length;
    let end = ((last_frame + 1) * hop_length).min(y.len());

    Ok((y.slice(s![start..end]).to_owned(), (start, end)))
}

/// Split an audio signal into non-silent intervals.
///
/// Returns a vector of `(start_sample, end_sample)` pairs.
pub fn split(
    y: ArrayView1<Float>,
    top_db: Float,
    frame_length: usize,
    hop_length: usize,
) -> Result<Vec<(usize, usize)>> {
    let rms = spectral::rms(Some(y), None, frame_length, hop_length)?;
    let rms_row = rms.row(0);

    let ref_val = rms_row.iter().copied().fold(0.0_f32, Float::max);
    if ref_val <= 0.0 {
        return Ok(vec![]);
    }

    let threshold = ref_val * 10.0_f32.powf(-top_db / 20.0);

    // Find non-silent frames
    let non_silent: Vec<bool> = rms_row.iter().map(|&v| v >= threshold).collect();

    // Find edges (transitions between silent and non-silent)
    let mut intervals = Vec::new();
    let mut in_segment = false;
    let mut seg_start = 0usize;

    for (i, &active) in non_silent.iter().enumerate() {
        if active && !in_segment {
            seg_start = i * hop_length;
            in_segment = true;
        } else if !active && in_segment {
            let seg_end = (i * hop_length).min(y.len());
            intervals.push((seg_start, seg_end));
            in_segment = false;
        }
    }
    if in_segment {
        intervals.push((seg_start, y.len()));
    }

    Ok(intervals)
}

/// Split with duration constraints.
///
/// Works like [`split`], then merges intervals separated by silences shorter
/// than `min_silence_duration` and drops segments shorter than
/// `min_signal_duration`.
///
/// - `sr`: sample rate (needed to convert seconds to samples)
/// - `min_silence_duration`: silences shorter than this (seconds) are bridged
/// - `min_signal_duration`: segments shorter than this (seconds) are dropped
pub fn split_with_constraints(
    y: ArrayView1<Float>,
    sr: u32,
    top_db: Float,
    frame_length: usize,
    hop_length: usize,
    min_silence_duration: Option<Float>,
    min_signal_duration: Option<Float>,
) -> Result<Vec<(usize, usize)>> {
    let mut intervals = split(y, top_db, frame_length, hop_length)?;

    // Merge intervals separated by short silences
    if let Some(min_sil) = min_silence_duration {
        let min_sil_samples = (min_sil * sr as Float) as usize;
        let mut merged = Vec::with_capacity(intervals.len());
        for iv in intervals {
            if let Some(last) = merged.last_mut() {
                let (_, ref mut prev_end): &mut (usize, usize) = last;
                if iv.0.saturating_sub(*prev_end) < min_sil_samples {
                    // Silence gap is too short — bridge it
                    *prev_end = iv.1;
                    continue;
                }
            }
            merged.push(iv);
        }
        intervals = merged;
    }

    // Drop segments shorter than min_signal_duration
    if let Some(min_sig) = min_signal_duration {
        let min_sig_samples = (min_sig * sr as Float) as usize;
        intervals.retain(|&(start, end)| end - start >= min_sig_samples);
    }

    Ok(intervals)
}

/// Remix audio by reordering segments.
pub fn remix(y: ArrayView1<Float>, intervals: &[(usize, usize)]) -> AudioBuffer {
    let total_len: usize = intervals.iter().map(|(s, e)| e - s).sum();
    let mut result = Array1::<Float>::zeros(total_len);
    let mut pos = 0;
    for &(start, end) in intervals {
        let len = (end - start).min(y.len().saturating_sub(start));
        result
            .slice_mut(s![pos..pos + len])
            .assign(&y.slice(s![start..start + len]));
        pos += len;
    }
    result
}

/// Apply preemphasis filter: y[n] = x[n] - coef * x[n-1].
pub fn preemphasis(y: ArrayView1<Float>, coef: Float) -> Result<AudioBuffer> {
    let b = ndarray::array![1.0, -coef];
    let a = ndarray::array![1.0];
    iir::lfilter(b.view(), a.view(), y)
}

/// Apply deemphasis filter (inverse of preemphasis).
pub fn deemphasis(y: ArrayView1<Float>, coef: Float) -> Result<AudioBuffer> {
    let b = ndarray::array![1.0];
    let a = ndarray::array![1.0, -coef];
    iir::lfilter(b.view(), a.view(), y)
}

// ============================================================
// Melody separation by f0
// ============================================================

/// Separate melody from accompaniment using pitch-guided harmonic masking.
///
/// 1. Estimates f0 with pYIN
/// 2. Builds a smooth harmonic mask at harmonics of f0
/// 3. Applies soft mask to the STFT
/// 4. Reconstructs melody and accompaniment via ISTFT
///
/// Returns `(melody, accompaniment)` as audio buffers.
pub fn melody_separate(
    y: ArrayView1<Float>,
    sr: u32,
    fmin: Float,
    fmax: Float,
    n_harmonics: usize,
    n_fft: usize,
    hop_length: usize,
) -> Result<(AudioBuffer, AudioBuffer)> {
    let window = WindowSpec::Named("hann".into());

    // 1. Pitch estimation via pYIN
    let (f0, voiced, _probs) = pitch::pyin(y, fmin, fmax, sr, n_fft, Some(hop_length))?;

    // 2. Compute STFT
    let stft_matrix = spectrum::stft(
        y,
        n_fft,
        Some(hop_length),
        None,
        &window,
        true,
        PadMode::Constant,
    )?;
    let n_bins = stft_matrix.nrows();
    let n_frames = stft_matrix.ncols();

    // STFT frequency bins
    let freqs = convert::fft_frequencies(sr as Float, n_fft);
    let freq_res = sr as Float / n_fft as Float; // Hz per bin

    // 3. Build harmonic mask
    // Align frame counts — pYIN may produce fewer frames than STFT
    let n_f0 = f0.len().min(n_frames);
    let mask_width = freq_res * 2.0; // Gaussian sigma in Hz

    let mut mask = Array2::<Float>::zeros((n_bins, n_frames));

    for t in 0..n_f0 {
        if !voiced[t] || f0[t] <= 0.0 {
            continue;
        }
        for h in 1..=n_harmonics {
            let harmonic_freq = f0[t] * h as Float;
            if harmonic_freq > sr as Float / 2.0 {
                break;
            }
            // Add Gaussian peak at harmonic frequency
            for b in 0..n_bins {
                let diff = freqs[b] - harmonic_freq;
                let gauss = (-0.5 * (diff / mask_width).powi(2)).exp();
                // Keep max of existing mask and new peak
                if gauss > mask[(b, t)] {
                    mask[(b, t)] = gauss;
                }
            }
        }
    }

    // Clamp mask to [0, 1]
    mask.mapv_inplace(|v| v.clamp(0.0, 1.0));

    // 4. Apply soft mask to complex STFT
    let mut melody_stft = Array2::<ComplexFloat>::zeros((n_bins, n_frames));
    let mut accomp_stft = Array2::<ComplexFloat>::zeros((n_bins, n_frames));

    for t in 0..n_frames {
        for b in 0..n_bins {
            let m = mask[(b, t)];
            melody_stft[(b, t)] = stft_matrix[(b, t)] * Complex::new(m, 0.0);
            accomp_stft[(b, t)] = stft_matrix[(b, t)] * Complex::new(1.0 - m, 0.0);
        }
    }

    // 5. Reconstruct via ISTFT
    let melody = spectrum::istft(
        melody_stft.view(),
        Some(hop_length),
        None,
        &window,
        true,
        Some(y.len()),
    )?;
    let accomp = spectrum::istft(
        accomp_stft.view(),
        Some(hop_length),
        None,
        &window,
        true,
        Some(y.len()),
    )?;

    Ok((melody, accomp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f32::consts::PI;

    fn sine(freq: Float, sr: u32, dur: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin())
    }

    #[test]
    fn test_trim() {
        // Silence + signal + silence
        let mut y = Array1::<Float>::zeros(22050);
        for i in 5000..15000 {
            y[i] = (2.0 * PI * 440.0 * i as Float / 22050.0).sin();
        }
        let (trimmed, (start, end)) = trim(y.view(), 60.0, 2048, 512).unwrap();
        assert!(start > 0, "should trim leading silence");
        assert!(end < 22050, "should trim trailing silence");
        assert!(trimmed.len() < y.len());
    }

    #[test]
    fn test_split() {
        let mut y = Array1::<Float>::zeros(44100);
        // Two bursts of signal
        for i in 5000..10000 {
            y[i] = 0.5 * (i as Float * 0.1).sin();
        }
        for i in 25000..35000 {
            y[i] = 0.5 * (i as Float * 0.1).sin();
        }
        let intervals = split(y.view(), 40.0, 2048, 512).unwrap();
        assert!(intervals.len() >= 1, "should find at least 1 segment");
    }

    #[test]
    fn test_split_with_constraints() {
        let sr: u32 = 22050;
        let mut y = Array1::<Float>::zeros(sr as usize * 4); // 4 seconds
                                                             // Burst 1: 0.2s–0.6s
        for i in (0.2 * sr as Float) as usize..(0.6 * sr as Float) as usize {
            y[i] = 0.5 * (i as Float * 0.1).sin();
        }
        // Burst 2: 0.8s–1.2s (gap = 0.2s from burst 1)
        for i in (0.8 * sr as Float) as usize..(1.2 * sr as Float) as usize {
            y[i] = 0.5 * (i as Float * 0.1).sin();
        }
        // Burst 3: 2.5s–3.5s (gap = 1.3s from burst 2)
        for i in (2.5 * sr as Float) as usize..(3.5 * sr as Float) as usize {
            y[i] = 0.5 * (i as Float * 0.1).sin();
        }

        let raw = split(y.view(), 40.0, 2048, 512).unwrap();
        assert!(
            raw.len() >= 2,
            "should find at least 2 segments, got {}",
            raw.len()
        );

        // Merge silences shorter than 0.5s — merges bursts 1+2
        let merged =
            split_with_constraints(y.view(), sr, 40.0, 2048, 512, Some(0.5), None).unwrap();
        assert!(
            merged.len() < raw.len(),
            "merging should reduce segment count"
        );

        // Drop segments shorter than 0.5s
        let filtered =
            split_with_constraints(y.view(), sr, 40.0, 2048, 512, None, Some(0.5)).unwrap();
        assert!(filtered.len() >= 1, "long burst should survive");
    }

    #[test]
    fn test_preemphasis_deemphasis_roundtrip() {
        let y = sine(440.0, 22050, 0.5);
        let pre = preemphasis(y.view(), 0.97).unwrap();
        let de = deemphasis(pre.view(), 0.97).unwrap();
        for i in 10..y.len() - 10 {
            // skip edges
            assert_abs_diff_eq!(y[i], de[i], epsilon = 0.01);
        }
    }

    #[test]
    fn test_time_stretch_identity() {
        let y = sine(440.0, 22050, 0.5);
        let stretched = time_stretch(y.view(), 1.0).unwrap();
        assert!((stretched.len() as f32 - y.len() as f32).abs() < 100.0);
    }

    #[test]
    fn test_time_stretch_double() {
        let y = sine(440.0, 22050, 1.0);
        let stretched = time_stretch(y.view(), 2.0).unwrap();
        // Rate 2.0 → half the length
        assert!(stretched.len() < y.len());
    }

    #[test]
    fn test_remix() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let remixed = remix(y.view(), &[(2, 4), (0, 2)]);
        assert_eq!(remixed.to_vec(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_melody_separate_basic() {
        // 2s sine at 440Hz — verify function runs and produces valid output
        let y = sine(440.0, 22050, 2.0);
        let (melody, accomp) =
            melody_separate(y.view(), 22050, 65.0, 2100.0, 10, 2048, 512).unwrap();
        assert_eq!(melody.len(), y.len());
        assert_eq!(accomp.len(), y.len());
        // Both outputs should be finite
        assert!(
            melody.iter().all(|x| x.is_finite()),
            "melody should be finite"
        );
        assert!(
            accomp.iter().all(|x| x.is_finite()),
            "accompaniment should be finite"
        );
        // Sum of melody + accompaniment should approximate original (energy conservation)
        let orig_energy: Float = y.iter().map(|x| x * x).sum();
        let mel_energy: Float = melody.iter().map(|x| x * x).sum();
        let acc_energy: Float = accomp.iter().map(|x| x * x).sum();
        // Total energy should be in same order of magnitude
        assert!(
            (mel_energy + acc_energy) > 0.0,
            "should produce non-zero output"
        );
        assert!(
            (mel_energy + acc_energy) < orig_energy * 4.0,
            "energy shouldn't explode"
        );
    }
}
