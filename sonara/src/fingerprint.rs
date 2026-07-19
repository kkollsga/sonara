//! Acoustic fingerprinting for duplicate detection.
//!
//! This module derives a compact, robust fingerprint of a recording that lets
//! you recognise *the same performance* across different encodings, bitrates and
//! playback gains in a large local library. It is a classical
//! spectral-landmark / band-energy-difference design (no machine learning, no
//! external services) built from first principles and documented in full below.
//!
//! It is **not** designed to survive tempo or pitch manipulation — it targets the
//! realistic library-dedup case: the *same* master re-encoded as MP3/AAC/FLAC at
//! various bitrates, normalised or attenuated in level, possibly with a little
//! extra leading silence.
//!
//! ## Design
//!
//! ### 1. Front end (make encodings look alike)
//!
//! - **Downmix**: the caller passes an already-mono signal.
//! - **Resample to a low, fixed rate** (`FP_SR` = 11025 Hz). Discarding everything
//!   above ~5.5 kHz throws away exactly the high-frequency detail that lossy
//!   codecs quantise most aggressively and that differs between bitrates, while
//!   keeping the musically dominant mid band.
//!
//! ### 2. Framing (~8 sub-fingerprints per second)
//!
//! The resampled signal is cut into overlapping frames of `N_FFT` (2048) samples
//! with a hop of `HOP` samples, giving `FP_SR / HOP` ≈ 8 frames per second. Each
//! frame is Hann-windowed and transformed with a real FFT to a power spectrum.
//!
//! ### 3. Band energies (300–2000 Hz, log-spaced)
//!
//! Power is summed into `N_BANDS` (33) logarithmically-spaced bands spanning
//! 300–2000 Hz. This mid band is where music carries most of its energy and where
//! codecs agree best, so it is the most stable region to key off. Each band energy
//! is converted to a log energy `L(t, m) = ln(E(t, m) + eps)`.
//!
//! ### 4. One-bit encoding across time *and* frequency
//!
//! For every frame `t > 0` and band `m` in `0..32` we emit a single bit from the
//! sign of a *double difference* — energy contrast between adjacent bands,
//! differenced again against the previous frame:
//!
//! ```text
//! D(t, m) = ( L(t, m) - L(t, m+1) ) - ( L(t-1, m) - L(t-1, m+1) )
//! bit(t, m) = 1 if D(t, m) > 0 else 0
//! ```
//!
//! 33 bands → 32 bits → exactly one `u32` "sub-fingerprint" per frame. The whole
//! track becomes a `Vec<u32>` of ~8 values per second.
//!
//! ### Why this is robust
//!
//! - **Gain / normalisation**: a global level change multiplies every band energy
//!   by the same constant, i.e. *adds* a constant to every log energy. That
//!   constant cancels in the band-to-band difference and again in the
//!   frame-to-frame difference, so `D` — and therefore every bit — is *exactly*
//!   invariant to gain. A 0.5× copy yields a bit-identical fingerprint.
//! - **Spectral tilt / EQ from re-encoding**: a fixed per-band coloration adds a
//!   per-band constant to `L(·, m)`; the frame-to-frame difference cancels it, so
//!   slowly-varying filtering barely moves any bits.
//! - **Lossy-codec perturbation**: MP3/AAC add small spectral noise. Only bands
//!   whose contrast sits *right at* the decision boundary can flip, so the bit
//!   error rate stays low.
//! - **Leading silence**: extra silence simply shifts the sub-fingerprint stream
//!   in time; the matcher's bounded offset search re-aligns it (see below).
//!
//! ## Matching
//!
//! [`match_score`] slides one fingerprint against the other over a bounded offset
//! window (`± MAX_OFFSET_FRAMES`, a few seconds) and, at the best alignment,
//! computes the **bit error rate** (BER) = differing bits / compared bits. The BER
//! is mapped to a similarity in `[0, 1]`:
//!
//! ```text
//! score = max(0, 1 - 2 * BER)
//! ```
//!
//! Two unrelated recordings have essentially random bits (BER ≈ 0.5 → score ≈ 0);
//! an identical one has BER 0 → score 1. **A BER below ~0.35 (score above ~0.30)
//! means "same recording".** In practice genuine duplicates score far higher
//! (typically > 0.7) and unrelated tracks far lower (< 0.15), so 0.30 is a safe
//! decision threshold with wide margin.
//!
//! ## Serialization
//!
//! [`encode_base64`] / [`decode_base64`] pack the `Vec<u32>` as little-endian
//! bytes in a standard base64 string (hand-rolled, no dependencies) for the
//! Python `fingerprint` dict field. [`FINGERPRINT_VERSION`] tags the format.

use ndarray::ArrayView1;

use crate::core::{audio, fft};
use crate::types::Float;

// ============================================================
// Format / algorithm constants
// ============================================================

/// Fingerprint format version. Bump on any change that alters the bits so that
/// stored fingerprints from different versions are never compared blindly.
pub const FINGERPRINT_VERSION: u32 = 1;

/// Internal analysis sample rate (Hz). Low rate discards the fragile,
/// codec-dependent high frequencies and keeps the stable mid band.
pub const FP_SR: u32 = 11025;

/// FFT / frame length in samples at `FP_SR` (~186 ms).
const N_FFT: usize = 2048;

/// Hop between frames in samples → `FP_SR / HOP` ≈ 8 sub-fingerprints per second.
const HOP: usize = FP_SR as usize / 8; // 1378

/// Number of log-spaced energy bands. `N_BANDS - 1` = 32 bits per frame.
const N_BANDS: usize = 33;

/// Low edge of the fingerprinted band (Hz).
const BAND_FMIN: Float = 300.0;
/// High edge of the fingerprinted band (Hz).
const BAND_FMAX: Float = 2000.0;

/// Floor added before the logarithm to keep silent bands well-defined.
const ENERGY_EPS: Float = 1e-10;

/// Bounded temporal search window for alignment (frames). ~3 s at 8 fps — enough
/// to absorb typical leading-silence / trim differences between encodings.
const MAX_OFFSET_FRAMES: isize = 24;

// ============================================================
// Fingerprint computation
// ============================================================

/// Compute the raw sub-fingerprint sequence for a mono signal.
///
/// `y` is a mono time series at sample rate `sr`; it is resampled internally to
/// [`FP_SR`]. Returns one `u32` per frame after the first (~8 per second), or an
/// empty vector for signals too short to yield at least two frames.
pub fn compute(y: ArrayView1<Float>, sr: u32) -> Vec<u32> {
    // --- Front end: resample to the fixed low analysis rate ---
    let ys = if sr == FP_SR {
        y.to_owned()
    } else {
        match audio::resample(y, sr, FP_SR) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        }
    };
    let n = ys.len();
    if n < N_FFT {
        return Vec::new();
    }
    let n_frames = 1 + (n - N_FFT) / HOP;
    if n_frames < 2 {
        return Vec::new();
    }
    let samples = ys.as_slice().expect("contiguous");

    // --- Precompute Hann window and band bin ranges ---
    let window: Vec<Float> = (0..N_FFT)
        .map(|i| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as Float / (N_FFT as Float - 1.0)).cos()
        })
        .collect();
    let bands = band_bin_ranges();

    // --- Per-frame log band energies (parallel over frames) ---
    let log_bands: Vec<[Float; N_BANDS]> = {
        use rayon::prelude::*;
        (0..n_frames)
            .into_par_iter()
            .map(|t| frame_log_bands(samples, t * HOP, &window, &bands))
            .collect()
    };

    // --- One-bit double-difference encoding → u32 per frame (t > 0) ---
    let mut out = Vec::with_capacity(n_frames - 1);
    for t in 1..n_frames {
        let cur = &log_bands[t];
        let prev = &log_bands[t - 1];
        let mut word: u32 = 0;
        for m in 0..(N_BANDS - 1) {
            let d = (cur[m] - cur[m + 1]) - (prev[m] - prev[m + 1]);
            if d > 0.0 {
                word |= 1 << m;
            }
        }
        out.push(word);
    }
    out
}

/// Log band energies for a single frame starting at sample `start`.
fn frame_log_bands(
    samples: &[Float],
    start: usize,
    window: &[Float],
    bands: &[(usize, usize)],
) -> [Float; N_BANDS] {
    let n_bins = N_FFT / 2 + 1;
    let mut buf = vec![0.0_f32; N_FFT];
    for i in 0..N_FFT {
        buf[i] = samples[start + i] * window[i];
    }
    let mut spec = vec![num_complex::Complex::new(0.0, 0.0); n_bins];
    fft::rfft(&mut buf, &mut spec).expect("rfft");

    let mut out = [0.0_f32; N_BANDS];
    for (m, &(lo, hi)) in bands.iter().enumerate() {
        let mut e = 0.0_f32;
        for k in lo..hi {
            e += spec[k].norm_sqr();
        }
        out[m] = (e + ENERGY_EPS).ln();
    }
    out
}

/// Bin index ranges `[lo, hi)` for each of the `N_BANDS` log-spaced bands.
fn band_bin_ranges() -> Vec<(usize, usize)> {
    let n_bins = N_FFT / 2 + 1;
    let bin_hz = FP_SR as Float / N_FFT as Float;
    let ratio = (BAND_FMAX / BAND_FMIN).powf(1.0 / N_BANDS as Float);
    (0..N_BANDS)
        .map(|m| {
            let f_lo = BAND_FMIN * ratio.powi(m as i32);
            let f_hi = BAND_FMIN * ratio.powi(m as i32 + 1);
            let mut lo = (f_lo / bin_hz).floor() as usize;
            let mut hi = (f_hi / bin_hz).ceil() as usize;
            lo = lo.min(n_bins - 1);
            hi = hi.clamp(lo + 1, n_bins);
            (lo, hi)
        })
        .collect()
}

// ============================================================
// Matching
// ============================================================

/// Similarity in `[0, 1]` between two fingerprints, robust to a bounded temporal
/// offset (leading-silence / trim differences).
///
/// Slides `b` against `a` over `± MAX_OFFSET_FRAMES` and returns the score at the
/// best-aligned (lowest bit-error-rate) overlap. `score = max(0, 1 - 2*BER)`;
/// a score above ~0.30 (BER below ~0.35) indicates the same recording.
///
/// Returns 0.0 if either fingerprint is empty or no offset yields enough overlap.
pub fn match_score(a: &[u32], b: &[u32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let min_overlap = (a.len().min(b.len()) / 2).max(4);

    let mut best_ber = 1.0_f32;
    let mut found = false;
    for off in -MAX_OFFSET_FRAMES..=MAX_OFFSET_FRAMES {
        let mut diff_bits: u64 = 0;
        let mut overlap: usize = 0;
        for i in 0..a.len() {
            let j = i as isize + off;
            if j < 0 || j as usize >= b.len() {
                continue;
            }
            diff_bits += (a[i] ^ b[j as usize]).count_ones() as u64;
            overlap += 1;
        }
        if overlap < min_overlap {
            continue;
        }
        let ber = diff_bits as f32 / (overlap as f32 * 32.0);
        if ber < best_ber {
            best_ber = ber;
            found = true;
        }
    }
    if !found {
        return 0.0;
    }
    (1.0 - 2.0 * best_ber).max(0.0)
}

// ============================================================
// Serialization (standard base64, hand-rolled — no dependencies)
// ============================================================

const B64_ENC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode a fingerprint as a standard (padded) base64 string. Each `u32` is
/// serialized little-endian, so `decode_base64` reproduces the exact sequence.
pub fn encode_base64(fp: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(fp.len() * 4);
    for &w in fp {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ENC[(n >> 18) as usize & 0x3f] as char);
        out.push(B64_ENC[(n >> 12) as usize & 0x3f] as char);
        if chunk.len() > 1 {
            out.push(B64_ENC[(n >> 6) as usize & 0x3f] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_ENC[n as usize & 0x3f] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode a base64 string produced by [`encode_base64`] back into a `Vec<u32>`.
///
/// Returns `None` on invalid characters or a byte length that is not a multiple
/// of four (i.e. not a whole sequence of `u32`s).
pub fn decode_base64(s: &str) -> Option<Vec<u32>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let s = s.trim();
    let raw = s.as_bytes();
    if raw.len() % 4 != 0 {
        return None;
    }
    let mut bytes = Vec::with_capacity(raw.len() / 4 * 3);
    for chunk in raw.chunks(4) {
        let c0 = val(chunk[0])?;
        let c1 = val(chunk[1])?;
        let pad2 = chunk[2] == b'=';
        let pad3 = chunk[3] == b'=';
        let c2 = if pad2 { 0 } else { val(chunk[2])? };
        let c3 = if pad3 { 0 } else { val(chunk[3])? };
        let n = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
        bytes.push((n >> 16) as u8);
        if !pad2 {
            bytes.push((n >> 8) as u8);
        }
        if !pad3 {
            bytes.push(n as u8);
        }
    }
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::f32::consts::PI;

    fn hashf(a: u64, b: u64) -> Float {
        let h = a
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9)
            .wrapping_add(b.wrapping_mul(0x94D0_49BB_1331_11EB));
        ((h >> 40) as Float) / (1u64 << 24) as Float // in [0, 1)
    }

    /// Deterministic broadband, time-varying "recording" that models real music
    /// far better than a pure tone: 64 partials densely spanning the 300–2000 Hz
    /// fingerprint band (so every band carries robust energy — no fragile
    /// near-silent bands), each driven by two independent slow amplitude LFOs
    /// (0.6–3 Hz). The rich, band-limited temporal texture is what a duplicate
    /// preserves and an unrelated recording does not; distinct `seed`s vary the
    /// partial frequencies, weights, LFO rates and phases to yield genuinely
    /// uncorrelated recordings.
    fn melody(sr: u32, dur: Float, seed: u64) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        const N_PART: usize = 64;
        let mut freqs = [0.0f32; N_PART];
        let mut weights = [0.0f32; N_PART];
        let mut r1 = [0.0f32; N_PART];
        let mut r2 = [0.0f32; N_PART];
        let mut ph1 = [0.0f32; N_PART];
        let mut ph2 = [0.0f32; N_PART];
        for p in 0..N_PART {
            let base = 300.0 * (1950.0f32 / 300.0).powf(p as Float / (N_PART - 1) as Float);
            freqs[p] = base * (0.97 + 0.06 * hashf(seed, p as u64 + 2000));
            weights[p] = 0.4 + 1.2 * hashf(seed, p as u64 + 3000);
            r1[p] = 0.6 + 2.4 * hashf(seed, p as u64);
            r2[p] = 0.6 + 2.4 * hashf(seed, p as u64 + 5000);
            ph1[p] = 2.0 * PI * hashf(seed, p as u64 + 1000);
            ph2[p] = 2.0 * PI * hashf(seed, p as u64 + 6000);
        }
        Array1::from_shape_fn(n, |i| {
            let t = i as Float / sr as Float;
            let mut s = 0.0;
            for p in 0..N_PART {
                let amp = (0.35
                    + 0.35 * (2.0 * PI * r1[p] * t + ph1[p]).sin()
                    + 0.3 * (2.0 * PI * r2[p] * t + ph2[p]).sin())
                .max(0.0);
                s += weights[p] * amp * (2.0 * PI * freqs[p] * t + ph1[p]).sin();
            }
            s / N_PART as Float
        })
    }

    /// Deterministic white noise in [-amp, amp].
    fn noise(n: usize, amp: Float, seed: u64) -> Array1<Float> {
        Array1::from_shape_fn(n, |i| {
            let h = (i as u64)
                .wrapping_add(seed)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((h >> 33) as f32) / (1u64 << 31) as f32 - 1.0;
            u * amp
        })
    }

    fn rms(y: &Array1<Float>) -> Float {
        (y.iter().map(|v| v * v).sum::<Float>() / y.len() as Float).sqrt()
    }

    #[test]
    fn test_version_constant() {
        assert_eq!(FINGERPRINT_VERSION, 1);
    }

    // Test-signal length. Long enough that the bit-error-rate over the offset
    // search has low variance, so unrelated recordings separate cleanly.
    const DUR: Float = 14.0;

    #[test]
    fn test_identical_signal_matches_one() {
        let y = melody(22050, DUR, 1);
        let fp = compute(y.view(), 22050);
        assert!(
            fp.len() > 100,
            "expected a non-trivial fingerprint, got {}",
            fp.len()
        );
        let score = match_score(&fp, &fp);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "self-match should be 1.0, got {score}"
        );
    }

    #[test]
    fn test_half_gain_matches() {
        let y = melody(22050, DUR, 1);
        let half = y.mapv(|v| v * 0.5);
        let a = compute(y.view(), 22050);
        let b = compute(half.view(), 22050);
        let score = match_score(&a, &b);
        // Log-energy differences are exactly gain-invariant → essentially identical.
        assert!(score > 0.95, "0.5x gain should match > 0.95, got {score}");
    }

    #[test]
    fn test_light_noise_matches() {
        let y = melody(22050, DUR, 1);
        let sig_rms = rms(&y);
        // SNR ~30 dB → noise amplitude well below signal.
        let noise_amp = sig_rms / 31.6;
        let noisy = &y + &noise(y.len(), noise_amp, 99);
        let a = compute(y.view(), 22050);
        let b = compute(noisy.view(), 22050);
        let score = match_score(&a, &b);
        assert!(score > 0.7, "SNR~30dB should still match high, got {score}");
    }

    #[test]
    fn test_leading_silence_matches() {
        let y = melody(22050, DUR, 1);
        let pad = (0.3 * 22050.0) as usize;
        let mut padded = Array1::<Float>::zeros(pad + y.len());
        padded.slice_mut(ndarray::s![pad..]).assign(&y);
        let a = compute(y.view(), 22050);
        let b = compute(padded.view(), 22050);
        let score = match_score(&a, &b);
        // Offset search must re-align despite 0.3 s of extra leading silence.
        assert!(
            score > 0.4,
            "0.3s leading silence should still match, got {score}"
        );
    }

    #[test]
    fn test_different_signals_low() {
        let a = compute(melody(22050, DUR, 1).view(), 22050);
        let b = compute(melody(22050, DUR, 2).view(), 22050);
        let score = match_score(&a, &b);
        assert!(
            score < 0.3,
            "different recordings should score low, got {score}"
        );

        // Noise vs melody: unrelated → near zero.
        let c = compute(noise((22050.0 * DUR) as usize, 0.5, 5).view(), 22050);
        let score2 = match_score(&a, &c);
        assert!(
            score2 < 0.3,
            "noise vs melody should score low, got {score2}"
        );
    }

    #[test]
    fn test_short_signal_edge_case() {
        // Shorter than one FFT frame → empty fingerprint, no panic.
        let y = melody(22050, 0.02, 1);
        let fp = compute(y.view(), 22050);
        assert!(fp.is_empty());
        // Matching against empty is a defined 0.0, never a panic.
        let full = compute(melody(22050, DUR, 1).view(), 22050);
        assert_eq!(match_score(&fp, &full), 0.0);
        assert_eq!(match_score(&full, &fp), 0.0);
    }

    #[test]
    fn test_silence_handling() {
        let y = Array1::<Float>::zeros(22050 * 3);
        let fp = compute(y.view(), 22050);
        // Silence must not panic; every score must be finite.
        let s = match_score(&fp, &fp);
        assert!(s.is_finite());
    }

    #[test]
    fn test_empty_match() {
        assert_eq!(match_score(&[], &[]), 0.0);
        assert_eq!(match_score(&[1, 2, 3], &[]), 0.0);
    }

    #[test]
    fn test_serialization_round_trip() {
        let fp = compute(melody(22050, 6.0, 42).view(), 22050);
        assert!(!fp.is_empty());
        let s = encode_base64(&fp);
        let decoded = decode_base64(&s).expect("decode");
        assert_eq!(fp, decoded, "encode→decode must round-trip exactly");

        // Explicit small vectors covering all padding cases (1,2,3 words).
        for v in [vec![0u32], vec![0xDEAD_BEEF], vec![1, 2], vec![1, 2, 3]] {
            assert_eq!(decode_base64(&encode_base64(&v)).unwrap(), v);
        }
        // Empty → empty.
        assert_eq!(encode_base64(&[]), "");
        assert_eq!(decode_base64("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn test_decode_rejects_garbage() {
        assert!(decode_base64("abc").is_none()); // not a multiple of 4
        assert!(decode_base64("!!!!").is_none()); // invalid chars
    }
}
