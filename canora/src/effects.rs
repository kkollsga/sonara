//! Audio effects.
//!
//! Mirrors librosa.effects — hpss, harmonic, percussive, time_stretch, pitch_shift,
//! trim, split, remix, preemphasis, deemphasis.

use ndarray::{s, Array1, ArrayView1};

use crate::core::{audio, spectrum};
use crate::decompose;
use crate::dsp::iir;
use crate::error::Result;
use crate::feature::spectral;
use crate::types::*;

/// Harmonic-Percussive Source Separation (audio domain).
///
/// Returns `(harmonic, percussive)` audio signals.
pub fn hpss(y: ArrayView1<Float>, kernel_size: usize, margin: Float) -> Result<(AudioBuffer, AudioBuffer)> {
    let window = WindowSpec::Named("hann".into());
    let stft = spectrum::stft(y, 2048, None, None, &window, true, PadMode::Constant)?;
    let mag = stft.mapv(|c| c.norm());

    let (h_mag, p_mag) = decompose::hpss(mag.view(), kernel_size, margin, 2.0)?;

    // Reconstruct phase from original STFT
    let h_stft = ndarray::Array2::from_shape_fn(stft.raw_dim(), |(i, j)| {
        let norm = stft[(i, j)].norm();
        if norm > 0.0 { stft[(i, j)] * h_mag[(i, j)] / norm } else { num_complex::Complex::new(0.0, 0.0) }
    });
    let p_stft = ndarray::Array2::from_shape_fn(stft.raw_dim(), |(i, j)| {
        let norm = stft[(i, j)].norm();
        if norm > 0.0 { stft[(i, j)] * p_mag[(i, j)] / norm } else { num_complex::Complex::new(0.0, 0.0) }
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
    let rate = 2.0_f64.powf(-n_steps / bins_per_octave as Float);
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

    let ref_val = rms_row.iter().copied().fold(0.0_f64, Float::max);
    if ref_val <= 0.0 {
        return Ok((Array1::zeros(0), (0, 0)));
    }

    let threshold = ref_val * 10.0_f64.powf(-top_db / 20.0);

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

    let ref_val = rms_row.iter().copied().fold(0.0_f64, Float::max);
    if ref_val <= 0.0 {
        return Ok(vec![]);
    }

    let threshold = ref_val * 10.0_f64.powf(-top_db / 20.0);

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

/// Remix audio by reordering segments.
pub fn remix(y: ArrayView1<Float>, intervals: &[(usize, usize)]) -> AudioBuffer {
    let total_len: usize = intervals.iter().map(|(s, e)| e - s).sum();
    let mut result = Array1::<Float>::zeros(total_len);
    let mut pos = 0;
    for &(start, end) in intervals {
        let len = (end - start).min(y.len().saturating_sub(start));
        result.slice_mut(s![pos..pos + len]).assign(&y.slice(s![start..start + len]));
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

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
        for i in 5000..10000 { y[i] = 0.5 * (i as Float * 0.1).sin(); }
        for i in 25000..35000 { y[i] = 0.5 * (i as Float * 0.1).sin(); }
        let intervals = split(y.view(), 40.0, 2048, 512).unwrap();
        assert!(intervals.len() >= 1, "should find at least 1 segment");
    }

    #[test]
    fn test_preemphasis_deemphasis_roundtrip() {
        let y = sine(440.0, 22050, 0.5);
        let pre = preemphasis(y.view(), 0.97).unwrap();
        let de = deemphasis(pre.view(), 0.97).unwrap();
        for i in 10..y.len() - 10 { // skip edges
            assert_abs_diff_eq!(y[i], de[i], epsilon = 0.01);
        }
    }

    #[test]
    fn test_time_stretch_identity() {
        let y = sine(440.0, 22050, 0.5);
        let stretched = time_stretch(y.view(), 1.0).unwrap();
        assert!((stretched.len() as f64 - y.len() as f64).abs() < 100.0);
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
}
