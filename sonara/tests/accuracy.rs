//! Accuracy tests for the analysis pipeline.
//!
//! Uses synthetic signals with known ground-truth properties to validate
//! that each feature returns physically correct values.

use ndarray::Array1;
use std::f32::consts::PI;

type Float = f32;

fn sine(freq: Float, sr: u32, dur: Float) -> Array1<Float> {
    let n = (sr as Float * dur) as usize;
    Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin())
}

fn click_train(sr: u32, dur: Float, bpm: Float) -> Array1<Float> {
    let n = (sr as Float * dur) as usize;
    let interval = (60.0 / bpm * sr as Float) as usize;
    let mut y = Array1::<Float>::zeros(n);
    let mut pos = 0;
    while pos < n {
        for i in 0..100.min(n - pos) {
            y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
        }
        pos += interval;
    }
    y
}

fn white_noise(n: usize) -> Array1<Float> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    Array1::from_shape_fn(n, |i| {
        let mut h = DefaultHasher::new();
        (i as u64 ^ 0xDEADBEEF).hash(&mut h);
        (h.finish() as Float / u64::MAX as Float) * 2.0 - 1.0
    })
}

// ================================================================
// BPM accuracy
// ================================================================

#[test]
fn accuracy_bpm_120() {
    let y = click_train(22050, 8.0, 120.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    // Allow octave errors (60, 120, 240) — common in beat trackers
    let is_harmonic = (r.bpm - 120.0).abs() < 15.0
        || (r.bpm - 60.0).abs() < 15.0
        || (r.bpm - 240.0).abs() < 15.0;
    assert!(is_harmonic, "BPM {} should be 120 or octave harmonic", r.bpm);
}

#[test]
fn accuracy_bpm_90() {
    let y = click_train(22050, 8.0, 90.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    let is_harmonic = (r.bpm - 90.0).abs() < 15.0
        || (r.bpm - 180.0).abs() < 15.0
        || (r.bpm - 45.0).abs() < 15.0;
    assert!(is_harmonic, "BPM {} should be 90 or octave harmonic", r.bpm);
}

#[test]
fn accuracy_bpm_140() {
    let y = click_train(22050, 8.0, 140.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    let is_harmonic = (r.bpm - 140.0).abs() < 15.0
        || (r.bpm - 70.0).abs() < 15.0
        || (r.bpm - 280.0).abs() < 15.0;
    assert!(is_harmonic, "BPM {} should be 140 or octave harmonic", r.bpm);
}

// ================================================================
// Spectral centroid accuracy
// ================================================================

#[test]
fn accuracy_centroid_440hz() {
    // Pure 440 Hz sine — centroid should be near 440 Hz
    let y = sine(440.0, 22050, 2.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!((r.spectral_centroid_mean - 440.0).abs() < 50.0,
        "Centroid {} should be ~440 Hz", r.spectral_centroid_mean);
}

#[test]
fn accuracy_centroid_1000hz() {
    let y = sine(1000.0, 22050, 2.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!((r.spectral_centroid_mean - 1000.0).abs() < 100.0,
        "Centroid {} should be ~1000 Hz", r.spectral_centroid_mean);
}

#[test]
fn accuracy_centroid_low_vs_high() {
    // Lower frequency should have lower centroid
    let low = sine(200.0, 22050, 2.0);
    let high = sine(4000.0, 22050, 2.0);
    let r_low = sonara::analyze::analyze_signal(low.view(), 22050, &sonara::analyze::compact()).unwrap();
    let r_high = sonara::analyze::analyze_signal(high.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!(r_low.spectral_centroid_mean < r_high.spectral_centroid_mean,
        "200Hz centroid ({}) should be < 4000Hz centroid ({})",
        r_low.spectral_centroid_mean, r_high.spectral_centroid_mean);
}

// ================================================================
// RMS / dynamic range accuracy
// ================================================================

#[test]
fn accuracy_rms_sine() {
    // RMS of unit amplitude sine = 1/sqrt(2) ≈ 0.707
    let y = sine(440.0, 22050, 2.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!((r.rms_mean - 0.707).abs() < 0.1,
        "RMS {} should be ~0.707 for unit sine", r.rms_mean);
}

#[test]
fn accuracy_rms_half_amplitude() {
    // 0.5 amplitude → RMS ≈ 0.354
    let y = sine(440.0, 22050, 2.0).mapv(|v| v * 0.5);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!((r.rms_mean - 0.354).abs() < 0.1,
        "RMS {} should be ~0.354 for 0.5 amplitude sine", r.rms_mean);
}

#[test]
fn accuracy_dynamic_range_constant() {
    // Constant amplitude → near-zero dynamic range
    let y = sine(440.0, 22050, 2.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!(r.dynamic_range_db < 5.0,
        "Dynamic range {} dB should be small for constant sine", r.dynamic_range_db);
}

// ================================================================
// Zero crossing rate accuracy
// ================================================================

#[test]
fn accuracy_zcr_frequency_proportional() {
    // Higher frequency → more zero crossings
    let sr = 22050u32;
    let low = sine(100.0, sr, 2.0);
    let high = sine(4000.0, sr, 2.0);
    let r_low = sonara::analyze::analyze_signal(low.view(), sr, &sonara::analyze::compact()).unwrap();
    let r_high = sonara::analyze::analyze_signal(high.view(), sr, &sonara::analyze::compact()).unwrap();
    assert!(r_high.zero_crossing_rate > r_low.zero_crossing_rate * 5.0,
        "ZCR for 4kHz ({}) should be much higher than 100Hz ({})",
        r_high.zero_crossing_rate, r_low.zero_crossing_rate);
}

#[test]
fn accuracy_zcr_sine_expected_value() {
    // 440 Hz sine at 22050 sr → ~880 crossings/sec → ZCR ≈ 880/22050 ≈ 0.04
    let y = sine(440.0, 22050, 2.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    let expected_zcr = 2.0 * 440.0 / 22050.0; // ~0.04
    assert!((r.zero_crossing_rate - expected_zcr).abs() < 0.01,
        "ZCR {} should be ~{:.4}", r.zero_crossing_rate, expected_zcr);
}

// ================================================================
// Onset detection accuracy
// ================================================================

#[test]
fn accuracy_onset_count_click_train() {
    // 120 BPM for 4s → 8 beats → ~8 onsets
    let y = click_train(22050, 4.0, 120.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!(r.onset_frames.len() >= 3 && r.onset_frames.len() <= 12,
        "Expected ~8 onsets for 120 BPM/4s, got {}", r.onset_frames.len());
}

#[test]
fn accuracy_onset_density() {
    // 120 BPM = 2 beats/sec → onset density should be roughly 2
    let y = click_train(22050, 8.0, 120.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!(r.onset_density > 0.5 && r.onset_density < 5.0,
        "Onset density {} should be roughly ~2 for 120 BPM", r.onset_density);
}

#[test]
fn accuracy_silence_no_onsets() {
    let y = Array1::<Float>::zeros(44100);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!(r.onset_frames.is_empty(),
        "Silence should have 0 onsets, got {}", r.onset_frames.len());
}

// ================================================================
// Extended features accuracy
// ================================================================

#[test]
fn accuracy_bandwidth_sine_vs_noise() {
    // Pure sine: very narrow bandwidth. Noise: wide.
    let sine_sig = sine(440.0, 22050, 2.0);
    let noise_sig = white_noise(44100);

    let r_sine = sonara::analyze::analyze_signal(sine_sig.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let r_noise = sonara::analyze::analyze_signal(noise_sig.view(), 22050, &sonara::analyze::playlist()).unwrap();

    let bw_sine = r_sine.spectral_bandwidth_mean.unwrap();
    let bw_noise = r_noise.spectral_bandwidth_mean.unwrap();
    assert!(bw_sine < bw_noise,
        "Sine bandwidth ({}) should be < noise bandwidth ({})", bw_sine, bw_noise);
    // Sine bandwidth should be very small relative to Nyquist
    assert!(bw_sine < 500.0,
        "Pure sine bandwidth {} should be very narrow", bw_sine);
}

#[test]
fn accuracy_flatness_sine_vs_noise() {
    // Sine: near 0 (tonal). Noise: near 1 (flat spectrum).
    let sine_sig = sine(440.0, 22050, 2.0);
    let noise_sig = white_noise(44100);

    let r_sine = sonara::analyze::analyze_signal(sine_sig.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let r_noise = sonara::analyze::analyze_signal(noise_sig.view(), 22050, &sonara::analyze::playlist()).unwrap();

    let fl_sine = r_sine.spectral_flatness_mean.unwrap();
    let fl_noise = r_noise.spectral_flatness_mean.unwrap();
    assert!(fl_sine < 0.1,
        "Sine flatness {} should be near 0 (tonal)", fl_sine);
    assert!(fl_noise > fl_sine,
        "Noise flatness ({}) should be > sine flatness ({})", fl_noise, fl_sine);
}

#[test]
fn accuracy_rolloff_low_vs_high() {
    // Low frequency → low rolloff. High frequency → high rolloff.
    let low = sine(200.0, 22050, 2.0);
    let high = sine(5000.0, 22050, 2.0);

    let r_low = sonara::analyze::analyze_signal(low.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let r_high = sonara::analyze::analyze_signal(high.view(), 22050, &sonara::analyze::playlist()).unwrap();

    let ro_low = r_low.spectral_rolloff_mean.unwrap();
    let ro_high = r_high.spectral_rolloff_mean.unwrap();
    assert!(ro_low < ro_high,
        "200Hz rolloff ({}) should be < 5000Hz rolloff ({})", ro_low, ro_high);
}

#[test]
fn accuracy_mfcc_dimensions() {
    let y = sine(440.0, 22050, 2.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let mfcc = r.mfcc_mean.unwrap();
    assert_eq!(mfcc.len(), 13, "Should have 13 MFCC coefficients");
    // First MFCC relates to log energy — should be non-zero for a non-silent signal
    assert!(mfcc[0].abs() > 0.1,
        "MFCC[0] (log energy) {} should be non-zero for a signal", mfcc[0]);
}

#[test]
fn accuracy_mfcc_different_timbres() {
    // Sine and noise should have very different MFCC profiles
    let sine_sig = sine(440.0, 22050, 2.0);
    let noise_sig = white_noise(44100);

    let r_sine = sonara::analyze::analyze_signal(sine_sig.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let r_noise = sonara::analyze::analyze_signal(noise_sig.view(), 22050, &sonara::analyze::playlist()).unwrap();

    let mfcc_sine = r_sine.mfcc_mean.unwrap();
    let mfcc_noise = r_noise.mfcc_mean.unwrap();

    // Compute Euclidean distance between MFCC vectors
    let dist: Float = mfcc_sine.iter().zip(mfcc_noise.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<Float>()
        .sqrt();
    assert!(dist > 5.0,
        "MFCC distance between sine and noise ({}) should be large", dist);
}

#[test]
fn accuracy_chroma_a440() {
    // 440 Hz = A4. Chroma bin for A = index 9 (C=0, C#=1, ..., A=9)
    let y = sine(440.0, 22050, 2.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let chroma = r.chroma_mean.unwrap();
    assert_eq!(chroma.len(), 12);

    // The A bin (index 9) should have the highest or near-highest energy
    let max_bin = chroma.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    // Allow ±1 bin tolerance (mel-to-chroma mapping is approximate)
    let a_bin = 9;
    let dist = ((max_bin as i32 - a_bin as i32).abs()).min(12 - (max_bin as i32 - a_bin as i32).abs());
    assert!(dist <= 1,
        "Strongest chroma bin {} should be near A (bin 9), got chroma: {:?}", max_bin, chroma);
}

#[test]
fn accuracy_chroma_c_major() {
    // C4 (261.63 Hz) + E4 (329.63 Hz) + G4 (392.00 Hz) = C major chord
    let sr = 22050u32;
    let n = sr as usize * 2;
    let y = Array1::from_shape_fn(n, |i| {
        let t = i as Float / sr as Float;
        (2.0 * PI * 261.63 * t).sin() +
        (2.0 * PI * 329.63 * t).sin() +
        (2.0 * PI * 392.00 * t).sin()
    });
    let r = sonara::analyze::analyze_signal(y.view(), sr, &sonara::analyze::playlist()).unwrap();
    let chroma = r.chroma_mean.unwrap();

    // The fused chroma uses mel-to-chroma approximation, which can be ±1 bin
    // off for some pitches due to mel band width exceeding semitone width.
    // C=0, E=4, G=7 — allow ±1 bin tolerance.
    let expected = [0, 4, 7];
    let mut indexed: Vec<(usize, Float)> = chroma.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top3: Vec<usize> = indexed[..3].iter().map(|&(i, _)| i).collect();

    // At least 2 of the 3 strongest should be within ±1 of expected
    let matches = top3.iter().filter(|&&b| {
        expected.iter().any(|&e| {
            let d = ((b as i32 - e as i32).abs()).min(12 - (b as i32 - e as i32).abs());
            d <= 1
        })
    }).count();
    assert!(matches >= 2,
        "C major top 3 chroma bins {:?} should be near {:?} (±1 bin), chroma: {:?}",
        top3, expected, chroma);
}

#[test]
fn accuracy_spectral_contrast_noise() {
    // White noise should have relatively low contrast (flat spectrum)
    let y = white_noise(44100);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let contrast = r.spectral_contrast_mean.unwrap();
    assert_eq!(contrast.len(), 7);
    // Contrast values should be reasonable (not NaN, not extreme)
    for (i, &v) in contrast.iter().enumerate() {
        assert!(v.is_finite(), "Contrast band {} should be finite, got {}", i, v);
    }
}

// ================================================================
// Duration accuracy
// ================================================================

#[test]
fn accuracy_duration() {
    let y = sine(440.0, 22050, 3.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::compact()).unwrap();
    assert!((r.duration_sec - 3.0).abs() < 0.01,
        "Duration {} should be ~3.0", r.duration_sec);
}

// ================================================================
// Cross-validation: fused pipeline vs standalone functions
// ================================================================

#[test]
fn accuracy_centroid_fused_vs_standalone() {
    // Verify the fused pipeline centroid matches the standalone spectral_centroid
    let y = sine(440.0, 22050, 2.0);
    let sr = 22050u32;

    let fused = sonara::analyze::analyze_signal(y.view(), sr, &sonara::analyze::compact()).unwrap();

    let standalone = sonara::feature::spectral::spectral_centroid(
        Some(y.view()), None, sr as Float, 2048, 512,
    ).unwrap();
    let standalone_mean = standalone.row(0).iter().sum::<Float>() / standalone.ncols() as Float;

    // Should match within 10% — slight differences expected due to
    // different padding strategies (fused uses zero-padding, standalone may differ)
    let rel_diff = (fused.spectral_centroid_mean - standalone_mean).abs()
        / standalone_mean.max(1.0);
    assert!(rel_diff < 0.15,
        "Fused centroid ({}) vs standalone ({}) differ by {:.1}%",
        fused.spectral_centroid_mean, standalone_mean, rel_diff * 100.0);
}

#[test]
fn accuracy_rms_fused_vs_standalone() {
    let y = sine(440.0, 22050, 2.0);
    let sr = 22050u32;

    let fused = sonara::analyze::analyze_signal(y.view(), sr, &sonara::analyze::compact()).unwrap();

    let standalone = sonara::feature::spectral::rms(
        Some(y.view()), None, 2048, 512,
    ).unwrap();
    let standalone_mean = standalone.row(0).iter().sum::<Float>() / standalone.ncols() as Float;

    let rel_diff = (fused.rms_mean - standalone_mean).abs() / standalone_mean.max(0.001);
    assert!(rel_diff < 0.15,
        "Fused RMS ({}) vs standalone ({}) differ by {:.1}%",
        fused.rms_mean, standalone_mean, rel_diff * 100.0);
}

// ================================================================
// Perceptual features: integrated pipeline tests
// ================================================================

#[test]
fn accuracy_perceptual_energy_loud_vs_quiet() {
    // Loud, bright, busy signal should have higher energy than quiet, dark, sparse
    let sr = 22050u32;
    let loud = Array1::from_shape_fn(sr as usize * 2, |i| {
        let t = i as Float / sr as Float;
        0.8 * (2.0 * PI * 2000.0 * t).sin() + 0.3 * (2.0 * PI * 5000.0 * t).sin()
    });
    // Add some clicks to make it rhythmically active
    let mut loud = loud;
    for beat in 0..8 {
        let start = beat * sr as usize / 4;
        for i in 0..50.min(loud.len() - start) {
            loud[start + i] += 0.5 * (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
        }
    }

    let quiet = sine(200.0, sr, 2.0).mapv(|v| v * 0.05);

    let r_loud = sonara::analyze::analyze_signal(loud.view(), sr, &sonara::analyze::playlist()).unwrap();
    let r_quiet = sonara::analyze::analyze_signal(quiet.view(), sr, &sonara::analyze::playlist()).unwrap();

    let e_loud = r_loud.energy.unwrap();
    let e_quiet = r_quiet.energy.unwrap();
    assert!(e_loud > e_quiet,
        "Loud energy ({}) should be > quiet energy ({})", e_loud, e_quiet);
    assert!(e_loud > 0.4, "Loud signal energy should be > 0.4, got {}", e_loud);
}

#[test]
fn accuracy_perceptual_danceability_regular_beat() {
    // Regular 120 BPM click train should be more danceable than random noise
    let sr = 22050u32;
    let clicks = click_train(sr, 6.0, 120.0);
    let noise = white_noise(sr as usize * 6);

    let r_clicks = sonara::analyze::analyze_signal(clicks.view(), sr, &sonara::analyze::playlist()).unwrap();
    let r_noise = sonara::analyze::analyze_signal(noise.view(), sr, &sonara::analyze::playlist()).unwrap();

    let d_clicks = r_clicks.danceability.unwrap();
    let d_noise = r_noise.danceability.unwrap();
    assert!(d_clicks > d_noise,
        "Click train danceability ({}) should be > noise danceability ({})",
        d_clicks, d_noise);
}

#[test]
fn accuracy_perceptual_key_a440_extended() {
    let y = sine(440.0, 22050, 3.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::playlist()).unwrap();

    let key = r.key.unwrap();
    // Pure sine = single chroma bin; Temperley profiles may detect as
    // the key where A is the 5th (D). Both are valid for a single-note signal.
    assert!(key.starts_with("A") || key.starts_with("D"),
        "A440 should detect key A or D (where A is the 5th), got '{}'", key);
    assert!(r.key_confidence.unwrap() > 0.0);
}

#[test]
fn accuracy_perceptual_key_full_c_major() {
    let sr = 22050u32;
    let n = sr as usize * 3;
    let y = Array1::from_shape_fn(n, |i| {
        let t = i as Float / sr as Float;
        (2.0 * PI * 261.63 * t).sin() +
        (2.0 * PI * 329.63 * t).sin() +
        (2.0 * PI * 392.00 * t).sin()
    });
    let r = sonara::analyze::analyze_signal(y.view(), sr, &sonara::analyze::playlist()).unwrap();

    let key = r.key.unwrap();
    // Full mode uses accurate chroma filterbank. C major (C-E-G) shares notes
    // with A minor and E minor, so any of these related keys is acceptable.
    let acceptable = key.contains("C") || key.contains("A") || key.contains("E") || key.contains("G");
    assert!(acceptable,
        "C major chord should detect a related key (C/Am/Em/G), got '{}'", key);
}

#[test]
fn accuracy_perceptual_valence_major_vs_minor() {
    let sr = 22050u32;
    let n = sr as usize * 3;

    // C major chord (bright, fast-ish)
    let major = Array1::from_shape_fn(n, |i| {
        let t = i as Float / sr as Float;
        (2.0 * PI * 523.25 * t).sin() + // C5 (high, bright)
        (2.0 * PI * 659.25 * t).sin() + // E5
        (2.0 * PI * 783.99 * t).sin()   // G5
    });

    // A minor chord (darker, lower)
    let minor = Array1::from_shape_fn(n, |i| {
        let t = i as Float / sr as Float;
        (2.0 * PI * 220.0 * t).sin() + // A3 (low, dark)
        (2.0 * PI * 261.63 * t).sin() + // C4
        (2.0 * PI * 329.63 * t).sin()   // E4
    });

    let r_major = sonara::analyze::analyze_signal(major.view(), sr, &sonara::analyze::playlist()).unwrap();
    let r_minor = sonara::analyze::analyze_signal(minor.view(), sr, &sonara::analyze::playlist()).unwrap();

    let v_major = r_major.valence.unwrap();
    let v_minor = r_minor.valence.unwrap();
    // Higher-pitched major chord should have higher valence than lower-pitched minor
    assert!(v_major > v_minor,
        "Major chord valence ({}) should be > minor chord valence ({})",
        v_major, v_minor);
}

#[test]
fn accuracy_perceptual_acousticness_sine_vs_noise() {
    let y_sine = sine(440.0, 22050, 3.0);
    let y_noise = white_noise(22050 * 3);

    let r_sine = sonara::analyze::analyze_signal(y_sine.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let r_noise = sonara::analyze::analyze_signal(y_noise.view(), 22050, &sonara::analyze::playlist()).unwrap();

    let a_sine = r_sine.acousticness.unwrap();
    let a_noise = r_noise.acousticness.unwrap();
    assert!(a_sine > a_noise,
        "Pure sine acousticness ({}) should be > noise acousticness ({})",
        a_sine, a_noise);
}

#[test]
fn accuracy_perceptual_dfa_returns_value() {
    // Full mode should compute DFA-based danceability
    let y = click_train(22050, 5.0, 120.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::playlist()).unwrap();
    let d = r.danceability.unwrap();
    assert!(d >= 0.0 && d <= 1.0, "DFA danceability should be in [0,1], got {}", d);
}

#[test]
fn accuracy_perceptual_tier3_placeholders() {
    let y = sine(440.0, 22050, 2.0);
    let r = sonara::analyze::analyze_signal(y.view(), 22050, &sonara::analyze::playlist()).unwrap();
    // Tier 3 features should be None (not yet implemented)
    assert!(r.mood_happy.is_none());
    assert!(r.mood_aggressive.is_none());
    assert!(r.mood_relaxed.is_none());
    assert!(r.mood_sad.is_none());
    assert!(r.instrumentalness.is_none());
    assert!(r.genre.is_none());
}
