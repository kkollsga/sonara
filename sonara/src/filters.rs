//! Filter bank design for spectral analysis.
//!
//! Mel, chroma, constant-Q, wavelet, and other filter banks for
//! spectral analysis. Includes get_window, window_sumsquare,
//! window_bandwidth, and cq_to_chroma.

use ndarray::{Array1, Array2, ArrayView1};

use crate::core::convert;
use crate::dsp::windows;
use crate::error::Result;
use crate::types::{FilterBank, Float, WindowSpec};

/// Create a Mel filter bank.
///
/// Returns a matrix of shape `(n_mels, 1 + n_fft/2)` where each row is a
/// triangular mel-scale filter.
///
/// - `sr`: Sample rate of the audio
/// - `n_fft`: FFT window size
/// - `n_mels`: Number of mel bands
/// - `fmin`: Minimum frequency (Hz)
/// - `fmax`: Maximum frequency (Hz). 0 means sr/2.
/// - `htk`: Use HTK formula (default false = Slaney)
/// - `norm`: "slaney" for area normalization, "" for none
pub fn mel(
    sr: Float,
    n_fft: usize,
    n_mels: usize,
    fmin: Float,
    fmax: Float,
    htk: bool,
    norm: &str,
) -> FilterBank {
    let fmax = if fmax <= 0.0 { sr / 2.0 } else { fmax };
    let n_bins = 1 + n_fft / 2;

    // Compute mel points: n_mels + 2 points (includes edges)
    let mel_min = convert::hz_to_mel(fmin, htk);
    let mel_max = convert::hz_to_mel(fmax, htk);

    let n_points = n_mels + 2;
    let mel_points: Vec<Float> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as Float / (n_points - 1) as Float)
        .collect();

    // Convert mel points to Hz
    let hz_points: Vec<Float> = mel_points
        .iter()
        .map(|&m| convert::mel_to_hz(m, htk))
        .collect();

    // Convert Hz to FFT bin indices (fractional)
    let _bin_points: Vec<Float> = hz_points
        .iter()
        .map(|&f| f * n_fft as Float / sr)
        .collect();

    // FFT frequencies
    let fft_freqs = convert::fft_frequencies(sr, n_fft);

    let mut fb = FilterBank::zeros((n_mels, n_bins));

    for i in 0..n_mels {
        let f_left = hz_points[i];
        let f_center = hz_points[i + 1];
        let f_right = hz_points[i + 2];

        for j in 0..n_bins {
            let f = fft_freqs[j];

            if f >= f_left && f <= f_center && f_center > f_left {
                fb[(i, j)] = (f - f_left) / (f_center - f_left);
            } else if f > f_center && f <= f_right && f_right > f_center {
                fb[(i, j)] = (f_right - f) / (f_right - f_center);
            }
        }

        // Slaney normalization: divide by bandwidth (area = 1)
        if norm == "slaney" {
            let bandwidth = hz_points[i + 2] - hz_points[i];
            if bandwidth > 0.0 {
                let enorm = 2.0 / bandwidth;
                fb.row_mut(i).mapv_inplace(|v| v * enorm);
            }
        }
    }

    fb
}

/// Create a chroma filter bank.
///
/// Maps DFT bins to chroma (pitch class) bins.
/// Returns shape `(n_chroma, 1 + n_fft/2)`.
pub fn chroma(
    sr: Float,
    n_fft: usize,
    n_chroma: usize,
    tuning: Float,
) -> FilterBank {
    let n_bins = 1 + n_fft / 2;
    let fft_freqs = convert::fft_frequencies(sr, n_fft);

    let mut fb = FilterBank::zeros((n_chroma, n_bins));

    // librosa `filters.chroma` soft assignment (librosa 0.10). Each FFT bin's
    // energy is spread over pitch classes with a Gaussian over the wrapped
    // chroma-bin distance, scaled by the bin's WIDTH in chroma units — a bass
    // bin at 44.1k/n_fft=2048 spans ~2.7 semitones and must feed every class it
    // covers, not just its two nearest (which biased bass-heavy real music).
    //
    // `frqbins[k]` = position of FFT bin k in chroma-bin units (octaves*n_chroma)
    // relative to A. We need one bin past Nyquist (index n_bins = n_fft/2 + 1)
    // so the last kept column has a forward difference for its bin width.
    let ncf = n_chroma as Float;
    // C-based row offset: librosa computes A-based weights then rolls rows by
    // -3*(n_chroma/12) so row 0 = C. Fold that shift into the class index here
    // (sonara's rows are C-based: A440 -> row 9, C261.63 -> row 0).
    let roll = 3.0 * (n_chroma / 12) as Float;
    let n_chroma2 = (ncf / 2.0).round();
    let frqbins: Vec<Float> = (0..=n_bins)
        .map(|k| {
            let freq = k as Float * sr / n_fft as Float;
            if freq <= 0.0 {
                0.0 // DC placeholder; column 0 is left at zero below
            } else {
                ncf * convert::hz_to_octs(freq, tuning, n_chroma, None)
            }
        })
        .collect();

    for j in 1..n_bins {
        // Skip DC (column 0 stays zero, matching sonara's convention).
        // Bin width in chroma units, clamped to >= 1 like librosa.
        let binwidth = (frqbins[j + 1] - frqbins[j]).max(1.0);
        for c in 0..n_chroma {
            // Distance of bin j to class c (C-based) in chroma units, wrapped
            // to (-n_chroma/2, +n_chroma/2].
            let mut d = frqbins[j] - (c as Float + roll);
            d = (d + n_chroma2 + 10.0 * ncf).rem_euclid(ncf) - n_chroma2;
            // Gaussian bump; the factor of 2 narrows it (librosa parity).
            fb[(c, j)] = (-0.5 * (2.0 * d / binwidth).powi(2)).exp();
        }
    }

    // Normalize each column
    for j in 0..n_bins {
        let col_sum: Float = fb.column(j).sum();
        if col_sum > 0.0 {
            fb.column_mut(j).mapv_inplace(|v| v / col_sum);
        }
    }

    // Octave-domain weighting (librosa parity): Gaussian over octaves centred at
    // ctroct=5.0 (~880 Hz) with octwidth=2.0, suppressing the sr-dependent high
    // band that otherwise floods chroma with broadband energy at sr > 22050.
    // Without this, every FFT bin up to sr/2 contributes fully and real-music
    // chroma collapses toward a single class at 44.1k/48k.
    let ctroct: Float = 5.0;
    let octwidth: Float = 2.0;
    for j in 1..n_bins {
        let freq = fft_freqs[j];
        if freq <= 0.0 {
            continue;
        }
        let octs = convert::hz_to_octs(freq, tuning, n_chroma, None);
        let w = (-0.5 * ((octs - ctroct) / octwidth).powi(2)).exp();
        fb.column_mut(j).mapv_inplace(|v| v * w);
    }

    fb
}

/// Compute constant-Q filter lengths for given frequencies.
///
/// Returns the filter lengths (in samples) for each CQ frequency.
pub fn constant_q_lengths(
    sr: Float,
    fmin: Float,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: Float,
    window: &str,
) -> Result<Array1<Float>> {
    let q = filter_scale / (2.0_f32.powf(1.0 / bins_per_octave as Float) - 1.0);
    let freqs = convert::cqt_frequencies(n_bins, fmin, bins_per_octave);

    let bw = window_bandwidth_by_name(window);

    Ok(freqs.mapv(|f| q * sr / (f * bw)))
}

/// Compute wavelet filter lengths.
pub fn wavelet_lengths(
    sr: Float,
    freqs: ArrayView1<Float>,
    filter_scale: Float,
    window: &str,
) -> Array1<Float> {
    let bw = window_bandwidth_by_name(window);
    freqs.mapv(|f| {
        if f > 0.0 {
            filter_scale * sr / (f * bw)
        } else {
            0.0
        }
    })
}

/// Get a window function by specification.
///
/// Wraps `dsp::windows::get_window` with a convenient interface.
pub fn get_window(window: &WindowSpec, n: usize, fftbins: bool) -> Result<Array1<Float>> {
    windows::get_window(window, n, fftbins)
}

/// Compute the bandwidth of a window function in bins.
fn window_bandwidth_by_name(window: &str) -> Float {
    match window.to_lowercase().as_str() {
        "hann" | "hanning" => 1.5018,
        "hamming" => 1.3629,
        "blackman" => 1.7268,
        "blackmanharris" => 2.0044,
        "bartlett" | "triangular" => 1.3333,
        "kaiser" => 1.7029, // default beta
        "flattop" => 3.7702,
        "boxcar" | "rectangular" | "ones" => 1.0,
        _ => 1.5018, // default to Hann
    }
}

/// Return the bandwidth of a window function.
pub fn window_bandwidth(window: &str, n: usize) -> Float {
    let _ = n;
    window_bandwidth_by_name(window)
}

/// Compute the sum-of-squares envelope of a window function.
///
/// Used to verify the Constant Overlap-Add (COLA) condition for ISTFT.
///
/// Returns an array of length `n_fft` representing the normalized
/// sum-of-squared window values across all frames.
pub fn window_sumsquare(
    window: &WindowSpec,
    n_frames: usize,
    hop_length: usize,
    win_length: usize,
    n_fft: usize,
) -> Result<Array1<Float>> {
    let win = get_window(window, win_length, true)?;

    // Pad window to n_fft
    let win_padded = crate::util::utils::pad_center(win.view(), n_fft)?;
    let win_sq = win_padded.mapv(|v| v * v);

    let output_len = n_fft + hop_length * (n_frames - 1);
    let mut wss = Array1::<Float>::zeros(output_len);

    for t in 0..n_frames {
        let offset = t * hop_length;
        for i in 0..n_fft {
            if offset + i < output_len {
                wss[offset + i] += win_sq[i];
            }
        }
    }

    Ok(wss)
}

/// Map CQT bins to chroma.
///
/// Returns a matrix of shape `(n_chroma, n_input)` that maps
/// CQT frequency bins to chroma bins.
pub fn cq_to_chroma(
    n_input: usize,
    _bins_per_octave: usize,
    n_chroma: usize,
    fmin: Option<Float>,
) -> FilterBank {
    let _ = fmin; // Used for tuning, simplified here
    let mut fb = FilterBank::zeros((n_chroma, n_input));

    for i in 0..n_input {
        let chroma_bin = i % n_chroma;
        fb[(chroma_bin, i)] = 1.0;
    }

    fb
}

/// Compute multi-rate filter center frequencies.
///
/// Returns (frequencies, sample_rates) for multi-resolution analysis.
pub fn mr_frequencies(tuning: Float) -> (Array1<Float>, Array1<Float>) {
    let a4 = convert::tuning_to_a4(tuning, 12);
    let n_bins = 84; // 7 octaves
    let fmin = a4 * 2.0_f32.powf(-4.75); // ~C1

    let freqs = convert::cqt_frequencies(n_bins, fmin, 12);
    let sample_rates = freqs.mapv(|f| (f * 8.0).max(1000.0)); // heuristic

    (freqs, sample_rates)
}

/// Create a diagonal filter matrix.
///
/// Returns an identity-like sparse filter of shape `(n, n)` with bandwidth `width`.
pub fn diagonal_filter(n: usize, width: usize) -> Array2<Float> {
    let mut filt = Array2::<Float>::zeros((n, n));
    let half = width / 2;

    for i in 0..n {
        for j in i.saturating_sub(half)..=(i + half).min(n - 1) {
            filt[(i, j)] = 1.0 / width as Float;
        }
    }

    filt
}

/// Create a semitone filter bank.
pub fn semitone_filterbank(
    sr: Float,
    n_fft: usize,
    n_bands: usize,
    fmin: Float,
) -> Result<FilterBank> {
    let freqs = convert::cqt_frequencies(n_bands, fmin, 12);
    let n_bins = 1 + n_fft / 2;
    let fft_freqs = convert::fft_frequencies(sr, n_fft);

    let mut fb = FilterBank::zeros((n_bands, n_bins));

    for i in 0..n_bands {
        let f_center = freqs[i];
        let f_lower = f_center / 2.0_f32.powf(1.0 / 24.0);
        let f_upper = f_center * 2.0_f32.powf(1.0 / 24.0);

        for j in 0..n_bins {
            let f = fft_freqs[j];
            if f >= f_lower && f < f_center {
                fb[(i, j)] = (f - f_lower) / (f_center - f_lower);
            } else if f >= f_center && f <= f_upper {
                fb[(i, j)] = (f_upper - f) / (f_upper - f_center);
            }
        }
    }

    Ok(fb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel(22050.0, 2048, 128, 0.0, 0.0, false, "slaney");
        assert_eq!(fb.shape(), &[128, 1025]);
    }

    #[test]
    fn test_mel_filterbank_nonnegative() {
        let fb = mel(22050.0, 2048, 128, 0.0, 0.0, false, "slaney");
        for &v in fb.iter() {
            assert!(v >= 0.0, "mel filterbank values must be non-negative");
        }
    }

    #[test]
    fn test_mel_filterbank_triangular() {
        let fb = mel(22050.0, 2048, 40, 0.0, 0.0, false, "");
        // Each row should rise then fall (triangular shape)
        for i in 0..40 {
            let row = fb.row(i);
            let nonzero: Vec<Float> = row.iter().copied().filter(|&v| v > 0.0).collect();
            if nonzero.len() >= 3 {
                // Find peak
                let max_val = nonzero.iter().copied().fold(0.0_f32, Float::max);
                let peak_idx = nonzero.iter().position(|&v| v == max_val).unwrap();
                // Values before peak should be ascending
                for j in 1..peak_idx {
                    assert!(nonzero[j] >= nonzero[j - 1] - 1e-10);
                }
                // Values after peak should be descending
                for j in (peak_idx + 1)..nonzero.len() {
                    assert!(nonzero[j] <= nonzero[j - 1] + 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_mel_filterbank_slaney_normalization() {
        let fb = mel(22050.0, 2048, 40, 0.0, 0.0, false, "slaney");
        // With Slaney normalization, rows should have approximately equal area
        // (though not exactly equal because of discrete binning)
        let areas: Vec<Float> = (0..40).map(|i| fb.row(i).sum()).collect();
        for &area in &areas {
            assert!(area > 0.0, "each mel band should have positive area");
        }
    }

    #[test]
    fn test_mel_different_n_mels() {
        for n_mels in [20, 64, 128, 256] {
            let fb = mel(22050.0, 2048, n_mels, 0.0, 0.0, false, "slaney");
            assert_eq!(fb.nrows(), n_mels);
            assert_eq!(fb.ncols(), 1025);
        }
    }

    #[test]
    fn test_chroma_shape() {
        let fb = chroma(22050.0, 2048, 12, 0.0);
        assert_eq!(fb.shape(), &[12, 1025]);
    }

    #[test]
    fn test_chroma_class_mapping_pins() {
        // Pin the C-based pitch-class mapping across octaves BEFORE trusting the
        // soft assignment: the argmax row for a pure tone at a note frequency
        // must equal that note's chroma class (C=0 .. B=11). Uses a high n_fft
        // so each tone resolves cleanly into one bin/class.
        let sr = 44100.0;
        let n_fft = 16384;
        let fb = chroma(sr, n_fft, 12, 0.0);
        let fft_freqs = convert::fft_frequencies(sr, n_fft);
        // (freq, expected class): A across octaves -> 9; C -> 0; plus a spread.
        let cases = [
            (110.0_f32, 9),   // A2
            (220.0, 9),       // A3
            (440.0, 9),       // A4
            (880.0, 9),       // A5
            (261.63, 0),      // C4
            (523.25, 0),      // C5
            (329.63, 4),      // E4
            (392.00, 7),      // G4
            (493.88, 11),     // B4
            (277.18, 1),      // C#4
        ];
        for (freq, expected) in cases {
            // nearest FFT bin to the tone
            let j = (freq * n_fft as Float / sr).round() as usize;
            // sanity: bin frequency within a few Hz of the target
            assert!((fft_freqs[j] - freq).abs() < sr / n_fft as Float);
            let col = fb.column(j);
            let argmax = col
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                argmax, expected,
                "freq {freq} Hz should map to class {expected}, got {argmax}"
            );
        }
    }

    #[test]
    fn test_chroma_column_sums() {
        // After column-normalization the filterbank multiplies each column by the
        // librosa octave-domain Gaussian weight, so a column no longer sums to 1
        // but to exactly that weight (in (0, 1]) — the low/mid band near ctroct
        // stays ~1 while the sr/2 broadband band is suppressed toward 0.
        let sr = 22050.0;
        let n_fft = 2048;
        let n_chroma = 12;
        let fb = chroma(sr, n_fft, n_chroma, 0.0);
        let fft_freqs = convert::fft_frequencies(sr, n_fft);
        let ctroct: Float = 5.0;
        let octwidth: Float = 2.0;
        for j in 1..fb.ncols() {
            let col_sum: Float = fb.column(j).sum();
            if col_sum > 0.0 {
                let octs = convert::hz_to_octs(fft_freqs[j], 0.0, n_chroma, None);
                let w = (-0.5 * ((octs - ctroct) / octwidth).powi(2)).exp();
                assert_abs_diff_eq!(col_sum, w, epsilon = 1e-5);
                assert!(col_sum <= 1.0 + 1e-6);
            }
        }
    }

    #[test]
    fn test_constant_q_lengths() {
        let lengths = constant_q_lengths(22050.0, 32.7, 84, 12, 1.0, "hann").unwrap();
        assert_eq!(lengths.len(), 84);
        // Lower frequencies should have longer filters
        assert!(lengths[0] > lengths[83]);
    }

    #[test]
    fn test_get_window_delegates() {
        let w = get_window(&WindowSpec::Named("hann".into()), 512, true).unwrap();
        assert_eq!(w.len(), 512);
    }

    #[test]
    fn test_window_sumsquare_cola() {
        // Hann window with hop = n_fft/4 satisfies COLA
        let n_fft = 1024;
        let hop = n_fft / 4;
        let n_frames = 20;
        let wss = window_sumsquare(
            &WindowSpec::Named("hann".into()),
            n_frames,
            hop,
            n_fft,
            n_fft,
        )
        .unwrap();

        // In the middle region (away from edges), the sum-square should be constant
        let start = n_fft;
        let end = wss.len() - n_fft;
        let mid_val = wss[start + hop]; // reference value
        for i in start..end {
            assert_abs_diff_eq!(wss[i], mid_val, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_diagonal_filter() {
        let filt = diagonal_filter(5, 3);
        assert_eq!(filt.shape(), &[5, 5]);
        // Center diagonal should have value 1/3
        assert_abs_diff_eq!(filt[(2, 2)], 1.0 / 3.0, epsilon = 1e-5);
        // Neighbors should also have value
        assert_abs_diff_eq!(filt[(2, 1)], 1.0 / 3.0, epsilon = 1e-5);
        assert_abs_diff_eq!(filt[(2, 3)], 1.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cq_to_chroma_shape() {
        let fb = cq_to_chroma(84, 12, 12, None);
        assert_eq!(fb.shape(), &[12, 84]);
    }

    #[test]
    fn test_window_bandwidth_known() {
        assert_abs_diff_eq!(window_bandwidth("hann", 1024), 1.5018, epsilon = 1e-4);
        assert_abs_diff_eq!(window_bandwidth("boxcar", 1024), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_semitone_filterbank_shape() {
        let fb = semitone_filterbank(22050.0, 2048, 84, 32.7).unwrap();
        assert_eq!(fb.shape(), &[84, 1025]);
    }
}
