//! Tonal analysis — HPCP, chord detection, dissonance.
//!
//! Harmonic Pitch Class Profile (HPCP) provides a more robust pitch class
//! representation than energy-based chroma, using spectral peak detection
//! and harmonic weighting (Gomez 2006).
//!
//! Chord detection matches HPCP frames against chord templates via
//! correlation. Dissonance measures perceived roughness using the
//! Plomp-Levelt model.

use ndarray::{Array2, ArrayView1, ArrayView2};
#[cfg(test)]
use ndarray::Array1;

use crate::types::Float;

// ============================================================
// Constants
// ============================================================

/// Reference frequency for pitch class 0 (C) at A4 = 440 Hz.
const C_REF: Float = 261.6256; // C4

/// Note names for chord labels.
const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// ============================================================
// Spectral peak detection
// ============================================================

/// A spectral peak with interpolated frequency and magnitude.
#[derive(Clone, Copy)]
struct SpectralPeak {
    freq: Float,
    mag: Float,
}

/// Detect spectral peaks via local maxima with parabolic interpolation.
///
/// Returns peaks sorted by magnitude (descending), limited to `max_peaks`.
fn detect_spectral_peaks(
    power_spectrum: &[Float],
    freqs: &[Float],
    threshold: Float,
    max_peaks: usize,
    min_freq: Float,
    max_freq: Float,
) -> Vec<SpectralPeak> {
    let n = power_spectrum.len();
    if n < 3 {
        return Vec::new();
    }

    let mag: Vec<Float> = power_spectrum.iter().map(|&p| p.sqrt()).collect();

    let mut peaks = Vec::new();

    for i in 1..n - 1 {
        if mag[i] <= mag[i - 1] || mag[i] <= mag[i + 1] {
            continue;
        }
        if mag[i] < threshold {
            continue;
        }
        if freqs[i] < min_freq || freqs[i] > max_freq {
            continue;
        }

        // Parabolic interpolation for sub-bin accuracy
        let alpha = mag[i - 1];
        let beta = mag[i];
        let gamma = mag[i + 1];
        let denom = alpha - 2.0 * beta + gamma;
        let (interp_freq, interp_mag) = if denom.abs() > 1e-10 {
            let p = 0.5 * (alpha - gamma) / denom;
            let bin_frac = i as Float + p;
            let freq = if bin_frac >= 0.0 && (bin_frac as usize) < n - 1 {
                let lo = bin_frac as usize;
                let hi = lo + 1;
                let frac = bin_frac - lo as Float;
                freqs[lo] * (1.0 - frac) + freqs[hi] * frac
            } else {
                freqs[i]
            };
            let mag_interp = beta - 0.25 * (alpha - gamma) * p;
            (freq, mag_interp)
        } else {
            (freqs[i], beta)
        };

        peaks.push(SpectralPeak {
            freq: interp_freq,
            mag: interp_mag,
        });
    }

    // Sort by magnitude descending, take top max_peaks
    peaks.sort_by(|a, b| b.mag.partial_cmp(&a.mag).unwrap());
    peaks.truncate(max_peaks);
    peaks
}

// ============================================================
// HPCP — Harmonic Pitch Class Profile
// ============================================================

/// Compute Harmonic Pitch Class Profile from a power spectrogram.
///
/// HPCP is a more robust pitch class representation than energy-based chroma.
/// It uses spectral peak detection, frequency-to-pitch-class mapping with
/// cosine weighting, and optional harmonic contribution.
///
/// - `power_spec`: Power spectrogram (n_bins, n_frames)
/// - `freqs`: Frequency of each bin (n_bins,)
/// - `n_harmonics`: Number of harmonics to consider (default: 4)
/// - `min_freq`: Minimum frequency for peak detection (default: 40 Hz)
/// - `max_freq`: Maximum frequency for peak detection (default: 5000 Hz)
/// - `peak_threshold`: Minimum magnitude for peaks (default: 0.0)
/// - `max_peaks`: Max peaks per frame (default: 50)
///
/// Returns: Array2<Float> shape (12, n_frames), L1-normalized per frame.
pub fn hpcp(
    power_spec: ArrayView2<Float>,
    freqs: ArrayView1<Float>,
    n_harmonics: usize,
    min_freq: Float,
    max_freq: Float,
    peak_threshold: Float,
    max_peaks: usize,
) -> Array2<Float> {
    let n_frames = power_spec.ncols();
    let n_bins = power_spec.nrows();
    let mut result = Array2::<Float>::zeros((12, n_frames));

    let freqs_slice = freqs.as_slice().unwrap();

    // Harmonic weights: 1.0 for fundamental, decreasing for higher harmonics
    let harmonic_weights: Vec<Float> = (0..n_harmonics)
        .map(|h| 1.0 / (h as Float + 1.0))
        .collect();

    for t in 0..n_frames {
        let power_col: Vec<Float> = (0..n_bins).map(|i| power_spec[(i, t)]).collect();
        let peaks = detect_spectral_peaks(
            &power_col,
            freqs_slice,
            peak_threshold,
            max_peaks,
            min_freq,
            max_freq,
        );

        for peak in &peaks {
            // Contribute to pitch class for each harmonic
            for (h, &weight) in harmonic_weights.iter().enumerate() {
                let freq = peak.freq / (h as Float + 1.0);
                if freq < 20.0 {
                    continue;
                }

                // Frequency to pitch class (continuous)
                let semitones = 12.0 * (freq / C_REF).log2();
                let pitch_class = ((semitones % 12.0) + 12.0) % 12.0;

                // Cosine weighting: contribute to nearest pitch class with
                // weight based on distance (Gomez 2006 style)
                let center = pitch_class.round() as usize % 12;
                let dist = (pitch_class - center as Float).abs();

                // Cosine window: full contribution at center, zero at 0.5 semitone away
                let w = if dist < 0.5 {
                    (std::f32::consts::PI * dist).cos()
                } else {
                    0.0
                };

                result[(center, t)] += weight * peak.mag * peak.mag * w;
            }
        }

        // L1 normalize
        let sum: Float = (0..12).map(|c| result[(c, t)]).sum();
        if sum > 0.0 {
            for c in 0..12 {
                result[(c, t)] /= sum;
            }
        }
    }

    result
}

// ============================================================
// Chord detection
// ============================================================

/// Major and minor chord templates (pitch class profiles).
///
/// Each template is a 12-element array with 1.0 for chord tones and 0.0 for others.
/// Templates are rotated for all 12 root notes.
fn chord_templates() -> Vec<([Float; 12], &'static str, usize)> {
    // (template, quality, root_index)
    // Major triad: root, major third (+4), fifth (+7)
    let major = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    // Minor triad: root, minor third (+3), fifth (+7)
    let minor = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];

    let mut templates = Vec::with_capacity(24);

    for root in 0..12 {
        let mut maj_rot = [0.0_f32; 12];
        let mut min_rot = [0.0_f32; 12];
        for i in 0..12 {
            maj_rot[(i + root) % 12] = major[i];
            min_rot[(i + root) % 12] = minor[i];
        }
        templates.push((maj_rot, "maj", root));
        templates.push((min_rot, "min", root));
    }

    templates
}

/// Correlate an HPCP vector with all chord templates, return best match.
///
/// Returns (chord_label, correlation).
fn match_chord(hpcp_frame: &[Float]) -> (&'static str, usize, Float) {
    let templates = chord_templates();
    let mut best_corr = -1.0_f32;
    let mut best_quality = "maj";
    let mut best_root = 0;

    // Normalize hpcp for correlation
    let hpcp_norm: Float = hpcp_frame.iter().map(|v| v * v).sum::<Float>().sqrt();
    if hpcp_norm < 1e-10 {
        return ("maj", 0, 0.0);
    }

    for (template, quality, root) in &templates {
        let t_norm: Float = template.iter().map(|v| v * v).sum::<Float>().sqrt();
        if t_norm < 1e-10 {
            continue;
        }
        let dot: Float = hpcp_frame.iter().zip(template.iter()).map(|(a, b)| a * b).sum();
        let corr = dot / (hpcp_norm * t_norm);
        if corr > best_corr {
            best_corr = corr;
            best_quality = quality;
            best_root = *root;
        }
    }

    (best_quality, best_root, best_corr)
}

/// Format a chord label from root index and quality.
fn format_chord(root: usize, quality: &str) -> String {
    let note = NOTE_NAMES[root % 12];
    if quality == "min" {
        format!("{note}m")
    } else {
        note.to_string()
    }
}

/// Detect chords from an HPCP, aligned to beat boundaries.
///
/// Averages HPCP within each beat interval, then matches against
/// major/minor chord templates via correlation.
///
/// - `hpcp`: Harmonic Pitch Class Profile (12, n_frames)
/// - `beats`: Beat frame indices from beat tracker
///
/// Returns chord label per beat segment.
pub fn chords_from_beats(
    hpcp: ArrayView2<Float>,
    beats: &[usize],
) -> Vec<String> {
    let n_frames = hpcp.ncols();

    if beats.is_empty() || n_frames == 0 {
        return Vec::new();
    }

    // Create segment boundaries from beats
    let mut boundaries: Vec<usize> = Vec::with_capacity(beats.len() + 2);
    if beats[0] > 0 {
        boundaries.push(0);
    }
    boundaries.extend_from_slice(beats);
    if *beats.last().unwrap() < n_frames {
        boundaries.push(n_frames);
    }

    let mut chords = Vec::with_capacity(boundaries.len() - 1);

    for seg in boundaries.windows(2) {
        let start = seg[0].min(n_frames);
        let end = seg[1].min(n_frames);
        if start >= end {
            chords.push("N".to_string()); // no chord
            continue;
        }

        // Average HPCP over segment
        let mut avg = [0.0_f32; 12];
        for t in start..end {
            for c in 0..12 {
                avg[c] += hpcp[(c, t)];
            }
        }
        let n = (end - start) as Float;
        for v in avg.iter_mut() {
            *v /= n;
        }

        let (quality, root, _corr) = match_chord(&avg);
        chords.push(format_chord(root, quality));
    }

    chords
}

/// Detect chords from an HPCP using fixed-length segments.
///
/// Falls back to this when beat information is unavailable.
/// Uses segments of `segment_frames` frames (default ~0.5s).
pub fn chords_from_frames(
    hpcp: ArrayView2<Float>,
    segment_frames: usize,
) -> Vec<String> {
    let n_frames = hpcp.ncols();
    if n_frames == 0 {
        return Vec::new();
    }

    let seg_len = segment_frames.max(1);
    let n_segments = (n_frames + seg_len - 1) / seg_len;
    let mut chords = Vec::with_capacity(n_segments);

    for s in 0..n_segments {
        let start = s * seg_len;
        let end = ((s + 1) * seg_len).min(n_frames);

        let mut avg = [0.0_f32; 12];
        for t in start..end {
            for c in 0..12 {
                avg[c] += hpcp[(c, t)];
            }
        }
        let n = (end - start) as Float;
        for v in avg.iter_mut() {
            *v /= n;
        }

        let (quality, root, _corr) = match_chord(&avg);
        chords.push(format_chord(root, quality));
    }

    chords
}

// ============================================================
// Chord descriptors
// ============================================================

/// Summary statistics for a chord sequence.
pub struct ChordDescriptors {
    /// Most frequent chord in the sequence.
    pub predominant_chord: String,
    /// Chord changes per second.
    pub change_rate: Float,
    /// Number of distinct chords.
    pub n_unique: usize,
}

/// Compute chord descriptors from a chord sequence.
///
/// - `chords`: Chord labels (one per segment)
/// - `duration_sec`: Total duration in seconds
pub fn chord_descriptors(chords: &[String], duration_sec: Float) -> ChordDescriptors {
    if chords.is_empty() {
        return ChordDescriptors {
            predominant_chord: "N".to_string(),
            change_rate: 0.0,
            n_unique: 0,
        };
    }

    // Count occurrences
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    for chord in chords {
        *counts.entry(chord.as_str()).or_insert(0) += 1;
    }

    let predominant_chord = counts
        .iter()
        .max_by_key(|(_, &count)| count)
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "N".to_string());

    // Count transitions (adjacent chords that differ)
    let changes = chords.windows(2).filter(|w| w[0] != w[1]).count();
    let change_rate = if duration_sec > 0.0 {
        changes as Float / duration_sec
    } else {
        0.0
    };

    let n_unique = counts.len();

    ChordDescriptors {
        predominant_chord,
        change_rate,
        n_unique,
    }
}

// ============================================================
// Dissonance — Plomp-Levelt model
// ============================================================

/// Compute sensory dissonance from spectral peaks using the Sethares (1998) model.
///
/// Based on Plomp-Levelt's psychoacoustic experiments, parameterized by Sethares
/// in "Tuning, Timbre, Spectrum, Scale" (1998). For each pair of peaks, dissonance
/// depends on frequency separation relative to a frequency-dependent critical band.
///
/// Returns a value in [0, 1] where 0 = consonant, 1 = maximally dissonant.
pub fn dissonance_from_peaks(peaks_freq: &[Float], peaks_mag: &[Float]) -> Float {
    let n = peaks_freq.len();
    if n < 2 {
        return 0.0;
    }

    // Sethares (1998) parameters
    let b1: Float = 3.5144;
    let b2: Float = 5.7564;
    let d_max: Float = 0.24;
    let s1: Float = 0.0207;
    let s2: Float = 18.96;

    let mut diss_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;

    for i in 0..n {
        for j in (i + 1)..n {
            let f_min = peaks_freq[i].min(peaks_freq[j]);
            let f_diff = (peaks_freq[i] - peaks_freq[j]).abs();

            // Frequency-dependent scaling
            let s = d_max / (s1 * f_min + s2);

            // Plomp-Levelt curve
            let d = (-b1 * s * f_diff).exp() - (-b2 * s * f_diff).exp();
            let d = d.max(0.0);

            let w = peaks_mag[i] * peaks_mag[j];
            diss_sum += w * d;
            weight_sum += w;
        }
    }

    if weight_sum > 0.0 {
        (diss_sum / weight_sum).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute mean dissonance across all frames of a power spectrogram.
///
/// - `power_spec`: Power spectrogram (n_bins, n_frames)
/// - `freqs`: Frequency of each bin
/// - `peak_threshold`: Minimum magnitude for peak detection
/// - `max_peaks`: Maximum number of peaks per frame
///
/// Returns mean dissonance in [0, 1].
pub fn dissonance(
    power_spec: ArrayView2<Float>,
    freqs: ArrayView1<Float>,
    peak_threshold: Float,
    max_peaks: usize,
) -> Float {
    let n_frames = power_spec.ncols();
    let n_bins = power_spec.nrows();
    if n_frames == 0 {
        return 0.0;
    }

    let freqs_slice = freqs.as_slice().unwrap();
    let mut total = 0.0_f32;

    for t in 0..n_frames {
        let power_col: Vec<Float> = (0..n_bins).map(|i| power_spec[(i, t)]).collect();
        let peaks = detect_spectral_peaks(
            &power_col,
            freqs_slice,
            peak_threshold,
            max_peaks,
            40.0,
            5000.0,
        );

        if peaks.len() >= 2 {
            let pf: Vec<Float> = peaks.iter().map(|p| p.freq).collect();
            let pm: Vec<Float> = peaks.iter().map(|p| p.mag).collect();
            total += dissonance_from_peaks(&pf, &pm);
        }
    }

    total / n_frames as Float
}

/// Compute HPCP and dissonance from a power spectrogram in a single pass,
/// sharing peak detection between both algorithms.
///
/// Returns (hpcp, dissonance_mean).
pub fn hpcp_and_dissonance(
    power_spec: ArrayView2<Float>,
    freqs: ArrayView1<Float>,
    n_harmonics: usize,
    min_freq: Float,
    max_freq: Float,
    peak_threshold: Float,
    max_peaks: usize,
) -> (Array2<Float>, Float) {
    let n_frames = power_spec.ncols();
    let n_bins = power_spec.nrows();
    let mut hpcp_result = Array2::<Float>::zeros((12, n_frames));
    let mut diss_total = 0.0_f32;

    let freqs_slice = freqs.as_slice().unwrap();

    let harmonic_weights: Vec<Float> = (0..n_harmonics)
        .map(|h| 1.0 / (h as Float + 1.0))
        .collect();

    for t in 0..n_frames {
        let power_col: Vec<Float> = (0..n_bins).map(|i| power_spec[(i, t)]).collect();
        let peaks = detect_spectral_peaks(
            &power_col,
            freqs_slice,
            peak_threshold,
            max_peaks,
            min_freq,
            max_freq,
        );

        // --- HPCP from peaks ---
        for peak in &peaks {
            for (h, &weight) in harmonic_weights.iter().enumerate() {
                let freq = peak.freq / (h as Float + 1.0);
                if freq < 20.0 {
                    continue;
                }
                let semitones = 12.0 * (freq / C_REF).log2();
                let pitch_class = ((semitones % 12.0) + 12.0) % 12.0;
                let center = pitch_class.round() as usize % 12;
                let dist = (pitch_class - center as Float).abs();
                let w = if dist < 0.5 {
                    (std::f32::consts::PI * dist).cos()
                } else {
                    0.0
                };
                hpcp_result[(center, t)] += weight * peak.mag * peak.mag * w;
            }
        }

        // L1 normalize HPCP
        let sum: Float = (0..12).map(|c| hpcp_result[(c, t)]).sum();
        if sum > 0.0 {
            for c in 0..12 {
                hpcp_result[(c, t)] /= sum;
            }
        }

        // --- Dissonance from same peaks ---
        if peaks.len() >= 2 {
            let pf: Vec<Float> = peaks.iter().map(|p| p.freq).collect();
            let pm: Vec<Float> = peaks.iter().map(|p| p.mag).collect();
            diss_total += dissonance_from_peaks(&pf, &pm);
        }
    }

    let diss_mean = if n_frames > 0 {
        diss_total / n_frames as Float
    } else {
        0.0
    };

    (hpcp_result, diss_mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_tone_spectrum(freqs: &[Float], tone_freqs: &[Float], sr: Float, n_fft: usize) -> Array1<Float> {
        let n_bins = n_fft / 2 + 1;
        let _bin_freqs = Array1::from_shape_fn(n_bins, |i| i as Float * sr / n_fft as Float);
        let mut power = Array1::<Float>::zeros(n_bins);
        for &tf in tone_freqs {
            // Place energy at nearest bin
            let bin = ((tf / sr * n_fft as Float).round() as usize).min(n_bins - 1);
            power[bin] = 1.0;
        }
        let _ = freqs; // freqs used for clarity in call site
        power
    }

    #[test]
    fn test_hpcp_c_major_chord() {
        // C major = C4 (261.63), E4 (329.63), G4 (392.00)
        let sr = 22050.0;
        let n_fft = 4096;
        let n_bins = n_fft / 2 + 1;
        let freqs = Array1::from_shape_fn(n_bins, |i| i as Float * sr / n_fft as Float);
        let tone_freqs = [261.63, 329.63, 392.00];

        let power = make_tone_spectrum(freqs.as_slice().unwrap(), &tone_freqs, sr, n_fft);
        let power_2d = power.to_shape((n_bins, 1)).unwrap().to_owned();

        let result = hpcp(power_2d.view(), freqs.view(), 1, 40.0, 5000.0, 0.0, 50);

        // C=0, E=4, G=7 should be the dominant pitch classes
        let frame: Vec<Float> = (0..12).map(|c| result[(c, 0)]).collect();
        let max_val = frame.iter().copied().fold(0.0_f32, Float::max);
        assert!(max_val > 0.0, "HPCP should have non-zero values");

        // C, E, G should collectively dominate
        let chord_energy = frame[0] + frame[4] + frame[7];
        let total: Float = frame.iter().sum();
        assert!(
            chord_energy / total > 0.5,
            "C, E, G should dominate: chord_energy={chord_energy}, total={total}"
        );
    }

    #[test]
    fn test_chord_detection_major() {
        // Pure C major HPCP profile
        let mut hpcp_frame = [0.0_f32; 12];
        hpcp_frame[0] = 0.4; // C
        hpcp_frame[4] = 0.3; // E
        hpcp_frame[7] = 0.3; // G

        let (quality, root, corr) = match_chord(&hpcp_frame);
        assert_eq!(root, 0, "Should detect root C");
        assert_eq!(quality, "maj", "Should detect major");
        assert!(corr > 0.9, "Correlation should be high: {corr}");
    }

    #[test]
    fn test_chord_detection_a_minor() {
        let mut hpcp_frame = [0.0_f32; 12];
        hpcp_frame[9] = 0.4; // A
        hpcp_frame[0] = 0.3; // C
        hpcp_frame[4] = 0.3; // E

        let (quality, root, corr) = match_chord(&hpcp_frame);
        assert_eq!(root, 9, "Should detect root A");
        assert_eq!(quality, "min", "Should detect minor");
        assert!(corr > 0.9, "Correlation should be high: {corr}");
    }

    #[test]
    fn test_dissonance_consonant() {
        // Octave: should be very consonant
        let freqs = [440.0, 880.0];
        let mags = [1.0, 1.0];
        let d = dissonance_from_peaks(&freqs, &mags);
        assert!(d < 0.1, "Octave should be consonant: {d}");
    }

    #[test]
    fn test_dissonance_dissonant() {
        // Minor second: should be dissonant
        let freqs = [440.0, 466.16]; // A4 and A#4
        let mags = [1.0, 1.0];
        let d = dissonance_from_peaks(&freqs, &mags);
        assert!(d > 0.1, "Minor second should be dissonant: {d}");

        // Should be more dissonant than an octave
        let d_octave = dissonance_from_peaks(&[440.0, 880.0], &[1.0, 1.0]);
        assert!(d > d_octave * 10.0, "Minor second ({d}) should be much more dissonant than octave ({d_octave})");
    }

    #[test]
    fn test_dissonance_single_peak() {
        let d = dissonance_from_peaks(&[440.0], &[1.0]);
        assert_eq!(d, 0.0, "Single peak has no dissonance");
    }

    #[test]
    fn test_chord_descriptors() {
        let chords = vec![
            "C".to_string(), "C".to_string(), "Am".to_string(),
            "F".to_string(), "G".to_string(), "C".to_string(),
        ];
        let desc = chord_descriptors(&chords, 10.0);
        assert_eq!(desc.predominant_chord, "C");
        assert_eq!(desc.n_unique, 4);
        assert!((desc.change_rate - 0.4).abs() < 0.01); // 4 changes in 10s
    }

    #[test]
    fn test_chord_descriptors_empty() {
        let desc = chord_descriptors(&[], 10.0);
        assert_eq!(desc.predominant_chord, "N");
        assert_eq!(desc.change_rate, 0.0);
        assert_eq!(desc.n_unique, 0);
    }

    #[test]
    fn test_chords_from_frames() {
        let n_bins = 12;
        let n_frames = 20;
        let mut hpcp_data = Array2::<Float>::zeros((n_bins, n_frames));
        // Set C major profile for all frames
        for t in 0..n_frames {
            hpcp_data[(0, t)] = 0.4;
            hpcp_data[(4, t)] = 0.3;
            hpcp_data[(7, t)] = 0.3;
        }
        let chords = chords_from_frames(hpcp_data.view(), 5);
        assert_eq!(chords.len(), 4); // 20 frames / 5 = 4 segments
        for chord in &chords {
            assert_eq!(chord, "C");
        }
    }

    #[test]
    fn test_hpcp_and_dissonance_combined() {
        // Verify that the combined function produces the same results
        let sr = 22050.0;
        let n_fft = 4096;
        let n_bins = n_fft / 2 + 1;
        let freqs = Array1::from_shape_fn(n_bins, |i| i as Float * sr / n_fft as Float);

        // Simple spectrum with a few peaks
        let mut power = Array2::<Float>::zeros((n_bins, 3));
        let c_bin = (261.63 / sr * n_fft as Float).round() as usize;
        let e_bin = (329.63 / sr * n_fft as Float).round() as usize;
        let g_bin = (392.00 / sr * n_fft as Float).round() as usize;
        for t in 0..3 {
            power[(c_bin, t)] = 1.0;
            power[(e_bin, t)] = 0.8;
            power[(g_bin, t)] = 0.6;
        }

        let (hpcp_result, diss) = hpcp_and_dissonance(
            power.view(), freqs.view(), 4, 40.0, 5000.0, 0.0, 50,
        );

        assert_eq!(hpcp_result.shape(), &[12, 3]);
        // C major chord components should be relatively consonant
        assert!(diss < 0.5, "C major chord should not be highly dissonant: {diss}");
    }
}
