//! Unit conversion functions for time, frequency, pitch, and musical notation.
//!
//! Mirrors librosa.core.convert — 42 conversion functions covering
//! Hz, mel, MIDI, note names, octs, frames, samples, time, and weighting.

use ndarray::{Array1, ArrayView1};

use crate::types::Float;

// ============================================================
// Constants
// ============================================================

/// Standard concert pitch (Hz).
const A4_HZ: Float = 440.0;

/// MIDI note number for A4.
const A4_MIDI: Float = 69.0;

/// Reference frequency for mel scale (Hz).
const F_MEL_REF: Float = 700.0;

/// log(6.4) / 27.0 — used in Slaney mel scale
const MEL_LOG_STEP: Float = 0.068_751_777_473_377_22;

/// Break frequency between linear and log regions (Slaney mel scale).
const MEL_BREAK_FREQ: Float = 1000.0;

/// Mel value at the break frequency (Slaney).
const MEL_BREAK_MEL: Float = 15.0; // 1000 / (200/3)

/// Linear step for Slaney mel (Hz per mel below break).
const MEL_LIN_STEP: Float = 200.0 / 3.0; // = 66.667 Hz/mel

// ============================================================
// Frame / Sample / Time conversions
// ============================================================

/// Convert frame indices to sample indices.
pub fn frames_to_samples(frames: &[usize], hop_length: usize) -> Vec<usize> {
    frames.iter().map(|&f| f * hop_length).collect()
}

/// Convert frame indices to time (seconds).
pub fn frames_to_time(frames: &[usize], sr: Float, hop_length: usize) -> Vec<Float> {
    frames
        .iter()
        .map(|&f| f as Float * hop_length as Float / sr)
        .collect()
}

/// Convert sample indices to frame indices (floor division).
pub fn samples_to_frames(samples: &[usize], hop_length: usize) -> Vec<usize> {
    samples.iter().map(|&s| s / hop_length).collect()
}

/// Convert sample indices to time (seconds).
pub fn samples_to_time(samples: &[usize], sr: Float) -> Vec<Float> {
    samples.iter().map(|&s| s as Float / sr).collect()
}

/// Convert time (seconds) to frame indices.
pub fn time_to_frames(times: &[Float], sr: Float, hop_length: usize) -> Vec<usize> {
    times
        .iter()
        .map(|&t| (t * sr / hop_length as Float).floor() as usize)
        .collect()
}

/// Convert time (seconds) to sample indices.
pub fn time_to_samples(times: &[Float], sr: Float) -> Vec<usize> {
    times.iter().map(|&t| (t * sr).round() as usize).collect()
}

/// Convert block counts to frame indices.
pub fn blocks_to_frames(blocks: &[usize], block_length: usize, hop_length: usize) -> Vec<usize> {
    blocks
        .iter()
        .map(|&b| b * block_length / hop_length)
        .collect()
}

/// Convert block counts to sample indices.
pub fn blocks_to_samples(blocks: &[usize], block_length: usize) -> Vec<usize> {
    blocks.iter().map(|&b| b * block_length).collect()
}

/// Convert block counts to time (seconds).
pub fn blocks_to_time(blocks: &[usize], block_length: usize, sr: Float) -> Vec<Float> {
    blocks
        .iter()
        .map(|&b| b as Float * block_length as Float / sr)
        .collect()
}

// ============================================================
// Hz ↔ Mel
// ============================================================

/// Convert Hz to mel scale.
///
/// Uses the Slaney (linear below 1kHz, log above) scale by default.
/// Set `htk=true` for the HTK formula: `m = 2595 * log10(1 + f/700)`.
pub fn hz_to_mel(freq: Float, htk: bool) -> Float {
    if htk {
        2595.0 * (1.0 + freq / F_MEL_REF).log10()
    } else {
        if freq < MEL_BREAK_FREQ {
            freq / MEL_LIN_STEP
        } else {
            MEL_BREAK_MEL + (freq / MEL_BREAK_FREQ).ln() / MEL_LOG_STEP
        }
    }
}

/// Convert mel to Hz.
pub fn mel_to_hz(mel: Float, htk: bool) -> Float {
    if htk {
        F_MEL_REF * (10.0_f64.powf(mel / 2595.0) - 1.0)
    } else {
        if mel < MEL_BREAK_MEL {
            mel * MEL_LIN_STEP
        } else {
            MEL_BREAK_FREQ * ((mel - MEL_BREAK_MEL) * MEL_LOG_STEP).exp()
        }
    }
}

/// Vectorized Hz to mel.
pub fn hz_to_mel_array(freqs: ArrayView1<Float>, htk: bool) -> Array1<Float> {
    freqs.mapv(|f| hz_to_mel(f, htk))
}

/// Vectorized mel to Hz.
pub fn mel_to_hz_array(mels: ArrayView1<Float>, htk: bool) -> Array1<Float> {
    mels.mapv(|m| mel_to_hz(m, htk))
}

// ============================================================
// Hz ↔ MIDI ↔ Note
// ============================================================

/// Convert Hz to MIDI note number (fractional).
pub fn hz_to_midi(freq: Float) -> Float {
    if freq <= 0.0 {
        return Float::NAN;
    }
    A4_MIDI + 12.0 * (freq / A4_HZ).log2()
}

/// Convert MIDI note number to Hz.
pub fn midi_to_hz(midi: Float) -> Float {
    A4_HZ * 2.0_f64.powf((midi - A4_MIDI) / 12.0)
}

/// Convert Hz to note name string (e.g., "A4", "C#5").
pub fn hz_to_note(freq: Float) -> String {
    let midi = hz_to_midi(freq);
    midi_to_note(midi)
}

/// Convert MIDI note number to note name.
pub fn midi_to_note(midi: Float) -> String {
    if midi.is_nan() {
        return String::new();
    }
    let midi_round = midi.round() as i64;
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let note_idx = ((midi_round % 12) + 12) % 12;
    let octave = (midi_round / 12) - 1;
    format!("{}{}", note_names[note_idx as usize], octave)
}

/// Convert note name to Hz.
pub fn note_to_hz(note: &str) -> crate::Result<Float> {
    let midi = note_to_midi(note)?;
    Ok(midi_to_hz(midi))
}

/// Convert note name to MIDI note number.
pub fn note_to_midi(note: &str) -> crate::Result<Float> {
    let note = note.trim();
    if note.is_empty() {
        return Err(crate::CanoraError::InvalidParameter {
            param: "note",
            reason: "empty note string".into(),
        });
    }

    let note_map: std::collections::HashMap<&str, i32> = [
        ("C", 0), ("D", 2), ("E", 4), ("F", 5), ("G", 7), ("A", 9), ("B", 11),
    ]
    .into_iter()
    .collect();

    let bytes = note.as_bytes();
    let base_note = (bytes[0] as char).to_uppercase().to_string();
    let base_pitch = *note_map.get(base_note.as_str()).ok_or_else(|| {
        crate::CanoraError::InvalidParameter {
            param: "note",
            reason: format!("invalid note name: '{note}'"),
        }
    })?;

    // Parse accidentals and octave
    let mut offset = 0i32;
    let mut idx = 1;
    while idx < bytes.len() {
        match bytes[idx] as char {
            '#' | '♯' => {
                offset += 1;
                idx += 1;
            }
            'b' | '♭' => {
                offset -= 1;
                idx += 1;
            }
            _ => break,
        }
    }

    // Remaining should be the octave number
    let octave_str = &note[idx..];
    let octave: i32 = octave_str.parse().map_err(|_| {
        crate::CanoraError::InvalidParameter {
            param: "note",
            reason: format!("invalid octave in note: '{note}'"),
        }
    })?;

    Ok((12 * (octave + 1) + base_pitch + offset) as Float)
}

// ============================================================
// Hz ↔ Octave
// ============================================================

/// Convert Hz to octave number (relative to A4 by default).
pub fn hz_to_octs(freq: Float, tuning: Float, bins_per_octave: usize) -> Float {
    let a4_tuned = A4_HZ * 2.0_f64.powf(tuning / bins_per_octave as Float);
    (freq / (a4_tuned / 16.0)).log2()
}

/// Convert octave number to Hz.
pub fn octs_to_hz(octs: Float, tuning: Float, bins_per_octave: usize) -> Float {
    let a4_tuned = A4_HZ * 2.0_f64.powf(tuning / bins_per_octave as Float);
    (a4_tuned / 16.0) * 2.0_f64.powf(octs)
}

// ============================================================
// Tuning
// ============================================================

/// Convert A4 frequency to tuning deviation in fractional bins.
pub fn a4_to_tuning(a4: Float, bins_per_octave: usize) -> Float {
    bins_per_octave as Float * (a4 / A4_HZ).log2()
}

/// Convert tuning deviation to A4 frequency.
pub fn tuning_to_a4(tuning: Float, bins_per_octave: usize) -> Float {
    A4_HZ * 2.0_f64.powf(tuning / bins_per_octave as Float)
}

// ============================================================
// Frequency range generators
// ============================================================

/// Generate the frequencies for each FFT bin.
///
/// Returns `n_fft/2 + 1` frequencies from 0 to `sr/2`.
pub fn fft_frequencies(sr: Float, n_fft: usize) -> Array1<Float> {
    let n_bins = n_fft / 2 + 1;
    Array1::from_shape_fn(n_bins, |i| i as Float * sr / n_fft as Float)
}

/// Generate `n_mels` mel-spaced frequencies between `fmin` and `fmax`.
pub fn mel_frequencies(n_mels: usize, fmin: Float, fmax: Float, htk: bool) -> Array1<Float> {
    let mel_min = hz_to_mel(fmin, htk);
    let mel_max = hz_to_mel(fmax, htk);

    Array1::from_shape_fn(n_mels, |i| {
        let mel = mel_min + (mel_max - mel_min) * i as Float / (n_mels - 1).max(1) as Float;
        mel_to_hz(mel, htk)
    })
}

/// Generate CQT frequencies for `n_bins` bins starting at `fmin`.
pub fn cqt_frequencies(n_bins: usize, fmin: Float, bins_per_octave: usize) -> Array1<Float> {
    Array1::from_shape_fn(n_bins, |i| {
        fmin * 2.0_f64.powf(i as Float / bins_per_octave as Float)
    })
}

/// Generate tempo frequencies for a tempogram.
pub fn tempo_frequencies(n_bins: usize, hop_length: usize, sr: Float) -> Array1<Float> {
    let bin_frequencies = Array1::from_shape_fn(n_bins, |i| i as Float);
    bin_frequencies.mapv(|b| {
        if b == 0.0 {
            0.0
        } else {
            60.0 * sr / (hop_length as Float * b)
        }
    })
}

/// Generate Fourier tempogram frequencies.
pub fn fourier_tempo_frequencies(sr: Float, win_length: usize, hop_length: usize) -> Array1<Float> {
    let n_bins = win_length / 2 + 1;
    Array1::from_shape_fn(n_bins, |i| {
        i as Float * sr * 60.0 / (hop_length as Float * win_length as Float)
    })
}

// ============================================================
// Weighting functions (A, B, C, D, Z)
// ============================================================

/// A-weighting curve (dB) for a given frequency in Hz.
pub fn a_weighting(freq: Float) -> Float {
    let f2 = freq * freq;
    let num = 12194.0_f64.powi(2) * f2 * f2;
    let denom = (f2 + 20.6_f64.powi(2))
        * ((f2 + 107.7_f64.powi(2)) * (f2 + 737.9_f64.powi(2))).sqrt()
        * (f2 + 12194.0_f64.powi(2));
    if denom == 0.0 {
        return Float::NEG_INFINITY;
    }
    20.0 * (num / denom).log10() + 2.0
}

/// B-weighting curve (dB).
pub fn b_weighting(freq: Float) -> Float {
    let f2 = freq * freq;
    let num = 12194.0_f64.powi(2) * f2 * freq;
    let denom = (f2 + 20.6_f64.powi(2))
        * (f2 + 158.5_f64.powi(2)).sqrt()
        * (f2 + 12194.0_f64.powi(2));
    if denom == 0.0 {
        return Float::NEG_INFINITY;
    }
    20.0 * (num / denom).log10() + 0.17
}

/// C-weighting curve (dB).
pub fn c_weighting(freq: Float) -> Float {
    let f2 = freq * freq;
    let num = 12194.0_f64.powi(2) * f2;
    let denom = (f2 + 20.6_f64.powi(2)) * (f2 + 12194.0_f64.powi(2));
    if denom == 0.0 {
        return Float::NEG_INFINITY;
    }
    20.0 * (num / denom).log10() + 0.06
}

/// D-weighting curve (dB).
pub fn d_weighting(freq: Float) -> Float {
    let f = freq;
    let f2 = f * f;

    let h_f = ((1037918.48 - f2).powi(2) + 1080768.16 * f2)
        / ((9837328.0 - f2).powi(2) + 11723776.0 * f2);

    let num = f / 6.8966888496476e-5;
    let denom = ((f2 + 79919.29).sqrt()) * ((f2 + 1345600.0).sqrt());

    if denom == 0.0 || h_f <= 0.0 {
        return Float::NEG_INFINITY;
    }
    20.0 * (num / denom * h_f.sqrt()).log10()
}

/// Z-weighting (zero weighting = flat response, 0 dB everywhere).
pub fn z_weighting(_freq: Float) -> Float {
    0.0
}

/// Apply a frequency weighting function to an array of frequencies.
pub fn frequency_weighting(
    freqs: ArrayView1<Float>,
    kind: &str,
) -> crate::Result<Array1<Float>> {
    let weight_fn: fn(Float) -> Float = match kind.to_uppercase().as_str() {
        "A" => a_weighting,
        "B" => b_weighting,
        "C" => c_weighting,
        "D" => d_weighting,
        "Z" => z_weighting,
        _ => {
            return Err(crate::CanoraError::InvalidParameter {
                param: "kind",
                reason: format!("unknown weighting: '{kind}'"),
            })
        }
    };
    Ok(freqs.mapv(weight_fn))
}

/// Apply multiple frequency weightings, returning one row per weighting type.
pub fn multi_frequency_weighting(
    freqs: ArrayView1<Float>,
    kinds: &[&str],
) -> crate::Result<ndarray::Array2<Float>> {
    let mut result = ndarray::Array2::<Float>::zeros((kinds.len(), freqs.len()));
    for (i, &kind) in kinds.iter().enumerate() {
        let weights = frequency_weighting(freqs, kind)?;
        result.row_mut(i).assign(&weights);
    }
    Ok(result)
}

// ============================================================
// Utility: samples_like, times_like
// ============================================================

/// Generate sample indices matching the frames of a spectrogram-like array.
pub fn samples_like(n_frames: usize, hop_length: usize) -> Array1<usize> {
    Array1::from_shape_fn(n_frames, |i| i * hop_length)
}

/// Generate time values matching the frames of a spectrogram-like array.
pub fn times_like(n_frames: usize, sr: Float, hop_length: usize) -> Array1<Float> {
    Array1::from_shape_fn(n_frames, |i| {
        i as Float * hop_length as Float / sr
    })
}

// ============================================================
// Tests
// ============================================================

// ============================================================
// Svara conversions (Hindustani / Carnatic)
// ============================================================

const SVARA_H: [&str; 12] = ["Sa", "re", "Re", "ga", "Ga", "ma", "Ma", "Pa", "dha", "Dha", "ni", "Ni"];
const SVARA_C: [&str; 12] = ["Sa", "Ri1", "Ri2", "Ga1", "Ga2", "Ma1", "Ma2", "Pa", "Da1", "Da2", "Ni1", "Ni2"];

/// Convert Hz to Hindustani svara name.
pub fn hz_to_svara_h(freq: Float, sa: Float, _abbr: bool) -> String {
    if freq <= 0.0 || sa <= 0.0 { return String::new(); }
    let midi_offset = (12.0 * (freq / sa).log2()).round() as i64;
    let idx = ((midi_offset % 12) + 12) % 12;
    SVARA_H[idx as usize].to_string()
}

/// Convert Hz to Carnatic svara name.
pub fn hz_to_svara_c(freq: Float, sa: Float, _abbr: bool) -> String {
    if freq <= 0.0 || sa <= 0.0 { return String::new(); }
    let midi_offset = (12.0 * (freq / sa).log2()).round() as i64;
    let idx = ((midi_offset % 12) + 12) % 12;
    SVARA_C[idx as usize].to_string()
}

/// Convert MIDI note to Hindustani svara.
pub fn midi_to_svara_h(midi: Float, sa_midi: Float, abbr: bool) -> String {
    hz_to_svara_h(midi_to_hz(midi), midi_to_hz(sa_midi), abbr)
}

/// Convert MIDI note to Carnatic svara.
pub fn midi_to_svara_c(midi: Float, sa_midi: Float, abbr: bool) -> String {
    hz_to_svara_c(midi_to_hz(midi), midi_to_hz(sa_midi), abbr)
}

/// Convert note name to Hindustani svara.
pub fn note_to_svara_h(note: &str, sa: &str, abbr: bool) -> crate::Result<String> {
    let freq = note_to_hz(note)?;
    let sa_freq = note_to_hz(sa)?;
    Ok(hz_to_svara_h(freq, sa_freq, abbr))
}

/// Convert note name to Carnatic svara.
pub fn note_to_svara_c(note: &str, sa: &str, abbr: bool) -> crate::Result<String> {
    let freq = note_to_hz(note)?;
    let sa_freq = note_to_hz(sa)?;
    Ok(hz_to_svara_c(freq, sa_freq, abbr))
}

/// Convert Hz to FJS (Functional Just System) notation.
pub fn hz_to_fjs(freq: Float, ref_freq: Float) -> String {
    if freq <= 0.0 || ref_freq <= 0.0 { return String::new(); }
    let ratio = freq / ref_freq;
    // Fold to [1, 2)
    let mut r = ratio;
    while r >= 2.0 { r /= 2.0; }
    while r < 1.0 { r *= 2.0; }

    // Common just intonation intervals
    let known: [(Float, &str); 12] = [
        (1.0, "P1"), (16.0/15.0, "m2"), (9.0/8.0, "M2"), (6.0/5.0, "m3"),
        (5.0/4.0, "M3"), (4.0/3.0, "P4"), (45.0/32.0, "A4"),
        (3.0/2.0, "P5"), (8.0/5.0, "m6"), (5.0/3.0, "M6"),
        (9.0/5.0, "m7"), (15.0/8.0, "M7"),
    ];

    for &(r_known, name) in &known {
        if (r - r_known).abs() < 0.01 {
            return name.to_string();
        }
    }
    format!("{:.4}", r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---- Hz ↔ Mel ----

    #[test]
    fn test_hz_to_mel_htk() {
        // 440 Hz → 549.64 mel (HTK)
        let mel = hz_to_mel(440.0, true);
        assert_abs_diff_eq!(mel, 2595.0 * (1.0 + 440.0_f64 / 700.0).log10(), epsilon = 1e-8);
    }

    #[test]
    fn test_hz_to_mel_slaney_below_break() {
        // Below 1000 Hz: linear mapping
        let mel = hz_to_mel(500.0, false);
        assert_abs_diff_eq!(mel, 500.0 / MEL_LIN_STEP, epsilon = 1e-8);
    }

    #[test]
    fn test_mel_roundtrip() {
        for freq in [100.0, 440.0, 1000.0, 4000.0, 8000.0] {
            for htk in [true, false] {
                let mel = hz_to_mel(freq, htk);
                let recovered = mel_to_hz(mel, htk);
                assert_abs_diff_eq!(freq, recovered, epsilon = 1e-8);
            }
        }
    }

    // ---- Hz ↔ MIDI ↔ Note ----

    #[test]
    fn test_hz_to_midi_a4() {
        assert_abs_diff_eq!(hz_to_midi(440.0), 69.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midi_to_hz_a4() {
        assert_abs_diff_eq!(midi_to_hz(69.0), 440.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hz_midi_roundtrip() {
        for freq in [261.63, 440.0, 880.0, 1760.0] {
            let midi = hz_to_midi(freq);
            let recovered = midi_to_hz(midi);
            assert_abs_diff_eq!(freq, recovered, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_note_to_hz_a4() {
        let hz = note_to_hz("A4").unwrap();
        assert_abs_diff_eq!(hz, 440.0, epsilon = 1e-6);
    }

    #[test]
    fn test_note_to_hz_c4() {
        let hz = note_to_hz("C4").unwrap();
        assert_abs_diff_eq!(hz, 261.6255653, epsilon = 0.01);
    }

    #[test]
    fn test_note_to_hz_sharp() {
        let hz = note_to_hz("C#4").unwrap();
        let expected = 440.0 * 2.0_f64.powf(-8.0 / 12.0); // C#4 = MIDI 61
        assert_abs_diff_eq!(hz, expected, epsilon = 0.01);
    }

    #[test]
    fn test_midi_to_note_a4() {
        assert_eq!(midi_to_note(69.0), "A4");
    }

    #[test]
    fn test_midi_to_note_c4() {
        assert_eq!(midi_to_note(60.0), "C4");
    }

    // ---- Frame / Sample / Time ----

    #[test]
    fn test_frames_to_samples() {
        assert_eq!(frames_to_samples(&[0, 1, 2], 512), vec![0, 512, 1024]);
    }

    #[test]
    fn test_frames_to_time() {
        let times = frames_to_time(&[0, 1, 2], 22050.0, 512);
        assert_abs_diff_eq!(times[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(times[1], 512.0 / 22050.0, epsilon = 1e-10);
    }

    #[test]
    fn test_time_frames_roundtrip() {
        let frames = vec![0, 10, 50, 100];
        let times = frames_to_time(&frames, 22050.0, 512);
        let recovered = time_to_frames(&times, 22050.0, 512);
        assert_eq!(frames, recovered);
    }

    // ---- FFT frequencies ----

    #[test]
    fn test_fft_frequencies() {
        let freqs = fft_frequencies(22050.0, 2048);
        assert_eq!(freqs.len(), 1025);
        assert_abs_diff_eq!(freqs[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(freqs[1024], 11025.0, epsilon = 1e-6);
    }

    // ---- Mel frequencies ----

    #[test]
    fn test_mel_frequencies() {
        let mels = mel_frequencies(128, 0.0, 11025.0, false);
        assert_eq!(mels.len(), 128);
        assert_abs_diff_eq!(mels[0], 0.0, epsilon = 1e-6);
        // Last should be close to fmax
        assert!((mels[127] - 11025.0).abs() < 1.0);
    }

    // ---- CQT frequencies ----

    #[test]
    fn test_cqt_frequencies() {
        let freqs = cqt_frequencies(84, 32.70, 12);
        assert_eq!(freqs.len(), 84);
        assert_abs_diff_eq!(freqs[0], 32.70, epsilon = 1e-6);
        // 84 bins = 7 octaves → last freq ≈ fmin * 2^7
        assert_abs_diff_eq!(freqs[83], 32.70 * 2.0_f64.powf(83.0 / 12.0), epsilon = 0.1);
    }

    // ---- Weighting ----

    #[test]
    fn test_a_weighting_1khz() {
        // A-weighting at 1kHz ≈ 0 dB (reference point)
        let w = a_weighting(1000.0);
        assert!(w.abs() < 1.0, "A-weighting at 1kHz should be ~0 dB, got {w}");
    }

    #[test]
    fn test_z_weighting() {
        assert_abs_diff_eq!(z_weighting(1000.0), 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(z_weighting(100.0), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_frequency_weighting_array() {
        let freqs = array![100.0, 1000.0, 10000.0];
        let weights = frequency_weighting(freqs.view(), "A").unwrap();
        assert_eq!(weights.len(), 3);
        // Low frequencies are heavily attenuated by A-weighting
        assert!(weights[0] < weights[1]);
    }

    // ---- Tuning ----

    #[test]
    fn test_a4_tuning_standard() {
        let tuning = a4_to_tuning(440.0, 12);
        assert_abs_diff_eq!(tuning, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tuning_roundtrip() {
        let a4 = 442.0;
        let tuning = a4_to_tuning(a4, 12);
        let recovered = tuning_to_a4(tuning, 12);
        assert_abs_diff_eq!(a4, recovered, epsilon = 1e-10);
    }

    // ---- samples_like / times_like ----

    #[test]
    fn test_samples_like() {
        let s = samples_like(5, 512);
        assert_eq!(s.to_vec(), vec![0, 512, 1024, 1536, 2048]);
    }

    #[test]
    fn test_times_like() {
        let t = times_like(3, 22050.0, 512);
        assert_abs_diff_eq!(t[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(t[1], 512.0 / 22050.0, epsilon = 1e-10);
    }
}
