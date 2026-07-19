//! Legacy pitched-melodic-content heuristic (`vocalness` v1).
//!
//! **Superseded.** This module measures *tonal + syllabic modulation* — NOT
//! vocal presence. It inverts on real music: distorted/screamed vocals are
//! broadband (they fail the tonal gate and score LOW) while clean solo pitched
//! instruments (sax, flute, violin) pass both gates and score HIGH. On a labeled
//! research set its AUC(vocal>instrumental) was 0.289 — worse than chance.
//!
//! The `TrackAnalysis::vocalness` field no longer uses this function. As of 0.2.4
//! it is derived from **mid-band spectral contrast** (voice/broadband energy fills
//! the ~0.8-5.6 kHz spectral valleys → low contrast → high vocalness), which
//! orders harsh > clean > instrumental correctly (AUC ≈ 0.92). See
//! `crate::analyze` for the current implementation.
//!
//! This function is retained only because its public signature
//! (`vocalness(mel_spec, sr, hop)`) may have external callers; its unit tests
//! verify what it actually computes (tonal+syllabic modulation), which is still
//! correct for that narrower quantity. Do not treat its output as vocal presence.
//!
//! ---
//! *(v1 mechanism, for reference.)*
//! Human singing/speech in the ~200–4000 Hz band is (a) harmonic — so the vocal
//! band is *tonal* (low spectral flatness) — and (b) modulated at the syllabic
//! rate (~4–8 Hz). Sustained pads are tonal but not syllabically modulated;
//! percussion (kicks/hats) is modulated but not tonal in the vocal band. Vocals
//! tend to have **both**, so the score gates the two together.
//!
//! ## Formula
//!
//! From the mel spectrogram (power), restricted to mel bins whose center
//! frequency falls in `[VOCAL_LO_HZ, VOCAL_HI_HZ]`:
//!
//! - `er`    = mean(vocal-band energy) / mean(total energy)          (energy ratio)
//! - `fl`    = **energy-weighted** per-frame spectral flatness within the vocal
//!             band (weighting by frame energy stops near-silent frames from
//!             diluting the estimate)
//! - `tonal` = clamp(1 − fl / FLATNESS_REF, 0, 1)                    (harmonicity)
//! - `mod48` = fraction of the vocal-band energy envelope's AC power that lies in
//!             the 4–8 Hz modulation band (syllabic rate)
//! - `depth` = rms(AC envelope) / mean envelope — **relative** modulation depth.
//!             `mod48` is a ratio of AC powers, so a near-constant envelope's
//!             numerical ripple could otherwise dominate it; `mod48` only counts
//!             when the modulation is meaningfully deep (see `DEPTH_LO/HI`).
//! - `cv`    = std(vocal-band energy) / mean(vocal-band energy)      (temporal variance)
//!
//! Then, with each component scaled to `[0, 1]`:
//!
//! ```text
//! mod_eff   = (mod48 / MOD_REF) * depth_gate      // syllabic AND deep
//! core      = sqrt(tonal * mod_eff)               // AND gate: needs both
//! vocalness = clamp(0.75*core + 0.15*er_c + 0.10*cv_c, 0, 1)
//! ```
//!
//! Calibrated so pure instrumental synthetic content — kicks, hats, *and
//! sustained pads/chords with no syllabic modulation* — scores < 0.3, and a
//! harmonic tone with 4–6 Hz amplitude/frequency modulation and formant-like
//! peaks scores > 0.6. Silence and white noise score low. Treat the output as a
//! soft hint, not ground truth.

use ndarray::ArrayView2;

use crate::core::convert;
use crate::types::Float;

/// Low edge of the vocal band (Hz).
pub const VOCAL_LO_HZ: Float = 200.0;
/// High edge of the vocal band (Hz).
pub const VOCAL_HI_HZ: Float = 4000.0;
/// Flatness reference: vocal-band flatness at/above this reads as fully non-tonal.
/// Voiced speech/singing sits around 0.1–0.3 (broad formants); noise approaches 1.
const FLATNESS_REF: Float = 0.4;
/// Energy-ratio scale: vocal-band fraction at/above this reads as full presence.
const ER_REF: Float = 0.5;
/// Modulation scale: 4–8 Hz AC fraction at/above this reads as full modulation.
const MOD_REF: Float = 0.30;
/// Coefficient-of-variation scale for temporal variance.
const CV_REF: Float = 1.0;
/// Relative modulation depth (rms(AC)/mean) below which the envelope counts as
/// constant — its 4–8 Hz "fraction" is then numerical ripple, not modulation.
const DEPTH_LO: Float = 0.05;
/// Depth at/above which modulation counts fully. Syllabic singing/speech
/// typically modulates the vocal-band envelope by well over 25%.
const DEPTH_HI: Float = 0.25;

/// Compute the legacy v1 `vocalness` heuristic (0.0 – 1.0) from a mel power
/// spectrogram. **Not vocal presence** — see the module docs; superseded by the
/// contrast-based `TrackAnalysis::vocalness` field in 0.2.4.
///
/// `mel_spec` is `(n_mels, n_frames)` mel-band **power**, `sr` the sample rate,
/// and `hop_length` the STFT hop used to build it (needed to map the envelope's
/// frame rate onto the 4–8 Hz modulation band). Returns a finite value in
/// `[0, 1]`; degenerate input (no frames, silence) returns `0.0`.
pub fn vocalness(mel_spec: ArrayView2<Float>, sr: u32, hop_length: usize) -> Float {
    let n_mels = mel_spec.nrows();
    let n_frames = mel_spec.ncols();
    if n_mels == 0 || n_frames == 0 {
        return 0.0;
    }

    // Mel-band center frequencies (slaney, matching the pipeline filterbank).
    // mel_frequencies(n_mels + 2, ...) gives band edges; the interior points are
    // the filter centers.
    let edges = convert::mel_frequencies(n_mels + 2, 0.0, sr as Float / 2.0, false);
    let vocal_bins: Vec<usize> = (0..n_mels)
        .filter(|&m| {
            let f = edges[m + 1];
            f >= VOCAL_LO_HZ && f <= VOCAL_HI_HZ
        })
        .collect();
    if vocal_bins.is_empty() {
        return 0.0;
    }

    // Per-frame: vocal-band energy, total energy, vocal-band flatness.
    let amin: Float = 1e-10;
    let mut vocal_env = vec![0.0_f32; n_frames];
    let mut total_energy = 0.0_f32;
    let mut flatness_sum = 0.0_f32;
    let mut flatness_weight = 0.0_f32;

    for t in 0..n_frames {
        let mut band_sum = 0.0_f32;
        let mut log_sum = 0.0_f32;
        let mut frame_total = 0.0_f32;
        for m in 0..n_mels {
            frame_total += mel_spec[(m, t)];
        }
        for &m in &vocal_bins {
            let v = mel_spec[(m, t)].max(amin);
            band_sum += v;
            log_sum += v.ln();
        }
        vocal_env[t] = band_sum;
        total_energy += frame_total;
        // Spectral flatness within the vocal band (geo mean / arith mean),
        // energy-weighted so near-silent frames don't dilute the estimate.
        let nb = vocal_bins.len() as Float;
        let geo = (log_sum / nb).exp();
        let arith = band_sum / nb;
        if arith > amin {
            flatness_sum += (geo / arith).clamp(0.0, 1.0) * band_sum;
            flatness_weight += band_sum;
        }
    }

    let mean_vocal = vocal_env.iter().sum::<Float>() / n_frames as Float;
    let mean_total = total_energy / n_frames as Float;
    if mean_total < amin || mean_vocal < amin {
        return 0.0; // silence / no energy
    }

    // Energy ratio.
    let er = mean_vocal / mean_total;

    // Tonality from vocal-band flatness (low flatness → tonal → vocal-like).
    let fl = if flatness_weight > amin {
        flatness_sum / flatness_weight
    } else {
        1.0
    };
    let tonal = (1.0 - fl / FLATNESS_REF).clamp(0.0, 1.0);

    // Temporal variance (coefficient of variation) of the vocal-band envelope.
    let var = vocal_env
        .iter()
        .map(|&v| (v - mean_vocal) * (v - mean_vocal))
        .sum::<Float>()
        / n_frames as Float;
    let cv = var.sqrt() / mean_vocal;

    // 4–8 Hz modulation fraction of the envelope's AC power.
    let mod48 = modulation_fraction(&vocal_env, mean_vocal, sr, hop_length);

    // Relative modulation depth: rms of the AC envelope over its mean. Gates
    // `mod48`, which is a ratio of AC powers and therefore meaningless when the
    // envelope is near-constant (a steady chord's numerical ripple can land
    // anywhere in the spectrum, including the 4-8 Hz band).
    let depth = (cv).min(10.0); // cv IS rms(AC)/mean by construction
    let depth_gate = ((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO)).clamp(0.0, 1.0);

    // Scale components to [0, 1].
    let er_c = (er / ER_REF).clamp(0.0, 1.0);
    let mod_c = (mod48 / MOD_REF).clamp(0.0, 1.0) * depth_gate;
    let cv_c = (cv / CV_REF).clamp(0.0, 1.0);

    // AND gate: vocal presence needs both harmonic content and syllabic modulation.
    let core = (tonal * mod_c).sqrt();
    (0.75 * core + 0.15 * er_c + 0.10 * cv_c).clamp(0.0, 1.0)
}

/// Fraction of an envelope's AC (mean-removed) power that lies in the 4–8 Hz
/// modulation band — the syllabic rate of speech/singing.
///
/// Runs an RBJ band-pass biquad (center ≈ √(4·8) Hz, 1-octave bandwidth) over the
/// mean-removed envelope and returns band energy / total AC energy. This is O(n)
/// in the number of frames (a full DFT would be O(n²) and dominate the pipeline
/// on long tracks). Returns 0.0 when the envelope is (near-)constant, or when the
/// envelope frame rate is too low to resolve the band.
fn modulation_fraction(env: &[Float], mean: Float, sr: u32, hop_length: usize) -> Float {
    let n = env.len();
    if n < 4 {
        return 0.0;
    }
    let fs = sr as Float / hop_length as Float; // envelope frame rate (Hz)
                                                // Need the band below Nyquist of the envelope signal.
    if fs <= 16.0 {
        return 0.0;
    }

    // RBJ band-pass (constant 0 dB peak gain), 1-octave band centered at √(4·8).
    let f0 = (4.0_f32 * 8.0).sqrt();
    let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
    let (sin_w0, cos_w0) = w0.sin_cos();
    let bw_octaves = 1.0_f32; // 4 Hz → 8 Hz spans one octave
    let alpha = sin_w0 * (std::f32::consts::LN_2 / 2.0 * bw_octaves * w0 / sin_w0).sinh();

    let a0 = 1.0 + alpha;
    let b0 = alpha / a0;
    let b2 = -alpha / a0;
    let a1 = -2.0 * cos_w0 / a0;
    let a2 = (1.0 - alpha) / a0;

    // Direct Form II Transposed biquad over the mean-removed envelope.
    let mut z1 = 0.0_f32;
    let mut z2 = 0.0_f32;
    let mut band_energy = 0.0_f32;
    let mut total_ac = 0.0_f32;
    for &v in env {
        let x = v - mean;
        total_ac += x * x;
        let y = b0 * x + z1;
        z1 = z2 - a1 * y; // b1 == 0
        z2 = b2 * x - a2 * y;
        band_energy += y * y;
    }

    if total_ac < 1e-12 {
        0.0
    } else {
        (band_energy / total_ac).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::f32::consts::PI;

    // Build a mel-power spectrogram from a synthetic signal via the same pipeline
    // path is overkill for a unit test; instead we construct mel spectra directly
    // by placing energy in the mel bins that cover the intended frequencies.
    // We approximate a mel filterbank of `n_mels` bins over [0, sr/2].

    fn mel_center(m: usize, n_mels: usize, sr: u32) -> Float {
        let edges = convert::mel_frequencies(n_mels + 2, 0.0, sr as Float / 2.0, false);
        edges[m + 1]
    }

    /// Synthesize a mel spectrogram: for each frame, deposit `amp(t)` of energy
    /// into mel bins whose center is near one of `freqs`, plus a `noise_floor`.
    fn synth_mel(
        n_mels: usize,
        n_frames: usize,
        sr: u32,
        freqs: &[Float],
        amp: impl Fn(usize) -> Float,
        noise_floor: Float,
        spread: Float,
    ) -> Array2<Float> {
        let mut mel = Array2::<Float>::from_elem((n_mels, n_frames), noise_floor);
        for t in 0..n_frames {
            let a = amp(t);
            for &f in freqs {
                for m in 0..n_mels {
                    let fc = mel_center(m, n_mels, sr);
                    let d = (fc - f) / (spread * f.max(1.0));
                    mel[(m, t)] += a * (-0.5 * d * d).exp();
                }
            }
        }
        mel
    }

    #[test]
    fn test_vocalness_silence_low() {
        let mel = Array2::<Float>::zeros((128, 200));
        let v = vocalness(mel.view(), 22050, 512);
        assert!(v >= 0.0 && v.is_finite());
        assert!(v < 0.3, "silence vocalness {} should be low", v);
    }

    #[test]
    fn test_vocalness_white_noise_low() {
        // Flat energy across all mel bins, steady over time → high flatness,
        // low modulation → low vocalness.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let n_mels = 128;
        let n_frames = 300;
        let mut mel = Array2::<Float>::zeros((n_mels, n_frames));
        for m in 0..n_mels {
            for t in 0..n_frames {
                let mut h = DefaultHasher::new();
                ((m * 7919 + t) as u64).hash(&mut h);
                mel[(m, t)] = (h.finish() as Float / u64::MAX as Float) + 0.5;
            }
        }
        let v = vocalness(mel.view(), 22050, 512);
        assert!(v >= 0.0 && v <= 1.0 && v.is_finite());
        assert!(v < 0.3, "white-noise vocalness {} should be low", v);
    }

    #[test]
    fn test_vocalness_instrumental_low() {
        // Kick/hats/pad: a sustained tonal pad (no syllabic modulation) plus a
        // broadband "hat" layer. Tonal but not modulated at 4-8 Hz → low.
        let n_mels = 128;
        let n_frames = 430; // ~10s at 22050/512
        let sr = 22050;
        // Sustained pad at 300 Hz + 600 Hz (constant amplitude).
        let mut mel = synth_mel(n_mels, n_frames, sr, &[300.0, 600.0], |_| 5.0, 0.05, 0.06);
        // Add a periodic "hat" burst (broadband) every ~0.5s (2 Hz, not 4-8 Hz).
        let hat_period = (0.5 * sr as Float / 512.0) as usize;
        for t in 0..n_frames {
            if t % hat_period.max(1) < 2 {
                for m in 0..n_mels {
                    let fc = mel_center(m, n_mels, sr);
                    if fc > 5000.0 {
                        mel[(m, t)] += 3.0;
                    }
                }
            }
        }
        let v = vocalness(mel.view(), sr, 512);
        assert!(v >= 0.0 && v <= 1.0 && v.is_finite());
        assert!(v < 0.3, "instrumental vocalness {} should be < 0.3", v);
    }

    #[test]
    fn test_vocalness_vocal_like_high() {
        // Harmonic tone with formant-ish peaks and 5 Hz amplitude modulation.
        let n_mels = 128;
        let n_frames = 430;
        let sr = 22050;
        let fr = sr as Float / 512.0; // frame rate ~43 Hz
                                      // Formant-ish harmonic peaks in the vocal band.
        let freqs = [250.0_f32, 500.0, 1000.0, 2000.0, 3000.0];
        // 5 Hz amplitude modulation (syllabic), never fully zero.
        let amp = |t: usize| {
            let ph = 2.0 * PI * 5.0 * t as Float / fr;
            2.0 + 1.8 * (0.5 * (1.0 + ph.sin()))
        };
        let mel = synth_mel(n_mels, n_frames, sr, &freqs, amp, 0.02, 0.05);
        let v = vocalness(mel.view(), sr, 512);
        assert!(v >= 0.0 && v <= 1.0 && v.is_finite());
        assert!(v > 0.6, "vocal-like vocalness {} should be > 0.6", v);
    }

    #[test]
    fn test_vocalness_always_finite_in_range() {
        // A grab-bag of shapes must always yield a finite [0,1] value.
        for (nm, nf) in [(1, 1), (4, 4), (128, 2), (64, 500)] {
            let mel = Array2::<Float>::from_elem((nm, nf), 1.0);
            let v = vocalness(mel.view(), 22050, 512);
            assert!(
                v >= 0.0 && v <= 1.0 && v.is_finite(),
                "v={} for {}x{}",
                v,
                nm,
                nf
            );
        }
    }

    #[test]
    fn test_vocalness_steady_chord_low() {
        // Regression: a sustained triad (organ-like pad, constant amplitude) is
        // tonal but has NO syllabic modulation — its envelope AC is numerical
        // ripple. Without the depth gate this scored ~0.8; it must be low.
        let n_mels = 128;
        let n_frames = 430;
        let sr = 22050;
        let freqs = [440.0_f32, 554.0, 659.0]; // A major triad
        let mel = synth_mel(n_mels, n_frames, sr, &freqs, |_| 5.0, 0.02, 0.05);
        let v = vocalness(mel.view(), sr, 512);
        assert!(v >= 0.0 && v <= 1.0 && v.is_finite());
        assert!(v < 0.3, "steady-chord vocalness {} should be < 0.3", v);
    }

    #[test]
    fn test_vocalness_pulse_train_moderate_at_most() {
        // A 2 Hz pulse train leaks harmonics into the 4-8 Hz modulation band and
        // its bursts are broadband (high flatness). The energy-weighted flatness
        // must keep it from reading as tonal — score stays below the vocal range.
        let n_mels = 128;
        let n_frames = 430;
        let sr = 22050;
        let fr = sr as Float / 512.0;
        let period = (fr / 2.0) as usize; // 2 Hz
                                          // Broadband bursts across the whole spectrum incl. vocal band.
        let mut mel = ndarray::Array2::<Float>::from_elem((n_mels, n_frames), 0.02);
        for t in 0..n_frames {
            if t % period.max(1) < 3 {
                for m in 0..n_mels {
                    mel[(m, t)] += 4.0;
                }
            }
        }
        let v = vocalness(mel.view(), sr, 512);
        assert!(v >= 0.0 && v <= 1.0 && v.is_finite());
        assert!(
            v < 0.45,
            "broadband pulse-train vocalness {} should stay below vocal range",
            v
        );
    }

    #[test]
    fn test_vocalness_depth_gate_monotone() {
        // Deeper 5 Hz modulation must never score lower than shallower modulation
        // of the same content (sanity for the depth gate).
        let n_mels = 128;
        let n_frames = 430;
        let sr = 22050;
        let fr = sr as Float / 512.0;
        let freqs = [250.0_f32, 500.0, 1000.0, 2000.0];
        let score = |depth: Float| {
            let amp = move |t: usize| {
                let ph = 2.0 * PI * 5.0 * t as Float / fr;
                2.0 * (1.0 + depth * ph.sin())
            };
            let mel = synth_mel(n_mels, n_frames, sr, &freqs, amp, 0.02, 0.05);
            vocalness(mel.view(), sr, 512)
        };
        let shallow = score(0.02);
        let mid = score(0.3);
        let deep = score(0.8);
        assert!(
            shallow < 0.3,
            "2% depth should read as constant, got {shallow}"
        );
        assert!(
            mid <= deep + 1e-3,
            "depth response should be monotone: {mid} vs {deep}"
        );
        assert!(
            deep > 0.6,
            "80% depth harmonic modulation should be vocal-like, got {deep}"
        );
    }
}
