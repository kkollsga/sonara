//! Extended loudness / gain utilities — the values players and mix software
//! consume for auto-gain and clip protection.
//!
//! These build on the integrated K-weighted loudness already computed in
//! [`crate::perceptual`] (ITU-R BS.1770-4 / EBU R128) and reuse its exact
//! K-weighting filter via [`crate::perceptual::k_weighted_signal`] — no
//! duplicated filter math.
//!
//! Provided:
//! - [`true_peak_db`] — true peak (dBTP) via 4x oversampling (BS.1770-4 Annex 2).
//! - [`replaygain_db`] — ReplayGain-style track gain to a -18 LUFS reference.
//! - [`loudness_curve`] — short-term loudness curve (3 s window, 1 s hop).
//! - [`loudness_momentary_max_db`] — maximum momentary (400 ms) loudness.
//! - [`loudness_range_lu`] — EBU R128 loudness range (LRA), in LU.
//!
//! All functions are pure and opt-in: the analysis pipeline only calls them when
//! the `"loudness"` feature is requested, so default modes pay nothing.

use ndarray::ArrayView1;

use crate::perceptual;
use crate::types::Float;

/// EBU R128 absolute gate (LUFS). Blocks quieter than this are treated as silence.
const ABS_GATE_LUFS: Float = -70.0;

/// Reference loudness for ReplayGain-style track gain, in LUFS.
///
/// -18 LUFS is a widely used replay-gain reference target for music playback.
pub const REPLAYGAIN_REFERENCE_LUFS: Float = -18.0;

// ============================================================
// ReplayGain-style track gain
// ============================================================

/// ReplayGain-style track gain in dB.
///
/// This is the gain a player should apply to bring the track to the
/// [`REPLAYGAIN_REFERENCE_LUFS`] (-18 LUFS) reference:
///
/// ```text
/// replaygain_db = REPLAYGAIN_REFERENCE_LUFS - integrated_loudness_lufs
///               = -18 - loudness_lufs
/// ```
///
/// A track measured at -9 LUFS (loud master) yields `-18 - (-9) = -9 dB`
/// (turn it down 9 dB); a quiet -24 LUFS track yields `+6 dB`. Pure arithmetic
/// on the already-computed integrated loudness — zero extra signal processing.
#[inline]
pub fn replaygain_db(loudness_lufs: Float) -> Float {
    REPLAYGAIN_REFERENCE_LUFS - loudness_lufs
}

// ============================================================
// True peak (dBTP)
// ============================================================

// 4x oversampling, windowed-sinc polyphase FIR interpolation.
//
// ITU-R BS.1770-4 Annex 2 specifies a >=4x oversampled peak measurement and
// gives a reference 48-tap polyphase coefficient table defined for 48 kHz. sonara
// works at arbitrary sample rates (22050 Hz by default), so instead of that
// fixed table we build an equivalent 4x interpolator on the fly: a 48-tap
// (12 taps/phase) sinc lowpass, Blackman-windowed, with each polyphase branch
// normalised to unity DC gain. Phase 0 reproduces the input samples exactly, so
// the oversampled peak is guaranteed >= the raw sample peak.
const TP_L: usize = 4; // oversampling factor
const TP_HALF: usize = 6; // taps-per-phase = 2 * TP_HALF = 12; total taps = TP_L * 12 = 48

#[inline]
fn sinc(x: Float) -> Float {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = std::f32::consts::PI * x;
        px.sin() / px
    }
}

#[inline]
fn blackman(i: usize, m: usize) -> Float {
    // Symmetric Blackman window over indices 0..m-1.
    let d = (m - 1) as Float;
    let a = 2.0 * std::f32::consts::PI * i as Float / d;
    0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos()
}

/// True peak of the signal in dBTP (decibels relative to full scale, true peak).
///
/// Computed via 4x oversampling per ITU-R BS.1770-4 Annex 2 (windowed-sinc
/// polyphase interpolation — see the module notes for why a windowed sinc is
/// used instead of the spec's 48 kHz-specific table). The returned value is
/// `20*log10(max |oversampled sample|)`; a full-scale sine is ~0 dBTP, half
/// scale ~-6.02 dBTP. Values above 0 dBTP indicate inter-sample overs that can
/// clip a downstream reconstruction filter / DAC.
///
/// Empty input returns [`ABS_GATE_LUFS`] as a finite floor (no peak to report).
pub fn true_peak_db(y: ArrayView1<Float>) -> Float {
    let n = y.len();
    if n == 0 {
        return ABS_GATE_LUFS;
    }

    // Fast contiguous access (only copies for the rare non-contiguous view).
    let owned: Option<Vec<Float>> = match y.as_slice() {
        Some(_) => None,
        None => Some(y.iter().copied().collect()),
    };
    let x: &[Float] = match owned {
        Some(ref v) => v.as_slice(),
        None => y.as_slice().unwrap(),
    };

    // Raw sample peak (this is exactly phase 0 of the interpolator).
    let mut peak: Float = 0.0;
    for &v in x {
        peak = peak.max(v.abs());
    }

    // Build the polyphase prototype: h[idx] = sinc((idx-center)/L) * blackman.
    let taps = 2 * TP_HALF; // per phase
    let m = TP_L * taps; // total prototype length
    let center = (TP_HALF * TP_L) as Float;
    let mut proto = vec![0.0 as Float; m];
    for idx in 0..m {
        let arg = (idx as Float - center) / TP_L as Float;
        proto[idx] = sinc(arg) * blackman(idx, m);
    }
    // Normalise each polyphase branch (phase p = taps p, p+L, p+2L, ...) to unit
    // DC gain, so a constant / low-frequency input is reconstructed at gain 1.
    for p in 0..TP_L {
        let mut s = 0.0 as Float;
        let mut t = 0;
        while t < taps {
            s += proto[t * TP_L + p];
            t += 1;
        }
        if s.abs() > 1e-12 {
            let mut t = 0;
            while t < taps {
                proto[t * TP_L + p] /= s;
                t += 1;
            }
        }
    }

    // Scan every inter-sample position. Phase 0 == raw samples (already in `peak`),
    // so only phases 1..L need reconstruction. The window centered on sample `i`
    // reads x[i+TP_HALF-t] for t in 0..taps, i.e. x[i-(TP_HALF-1)..=i+TP_HALF].
    //
    // Interior samples (where that range is fully in-bounds) take a tight loop
    // with no bounds handling; the few edge samples fall back to a clamped path.
    let lo_edge = TP_HALF.saturating_sub(1); // first fully-interior i
    let hi_edge = n.saturating_sub(TP_HALF); // one past last fully-interior i
    let ni = n as isize;

    // Interior fast path (no clamping).
    if hi_edge > lo_edge {
        for i in lo_edge..hi_edge {
            let base = i + TP_HALF; // x index for t = 0
            for p in 1..TP_L {
                let mut acc = 0.0 as Float;
                for t in 0..taps {
                    acc += proto[t * TP_L + p] * x[base - t];
                }
                let a = acc.abs();
                if a > peak {
                    peak = a;
                }
            }
        }
    }

    // Edge samples (clamped reflection at the boundaries).
    let edge_iter = (0..lo_edge.min(n)).chain(hi_edge.max(lo_edge)..n);
    for i in edge_iter {
        for p in 1..TP_L {
            let mut acc = 0.0 as Float;
            for t in 0..taps {
                let xi = i as isize + TP_HALF as isize - t as isize;
                let xc = xi.clamp(0, ni - 1) as usize;
                acc += proto[t * TP_L + p] * x[xc];
            }
            let a = acc.abs();
            if a > peak {
                peak = a;
            }
        }
    }

    if peak <= 1e-12 {
        ABS_GATE_LUFS
    } else {
        20.0 * peak.log10()
    }
}

// ============================================================
// Short-term / momentary loudness
// ============================================================

/// EBU R128 block loudness from a mean-square of K-weighted samples.
#[inline]
fn block_loudness(mean_sq: Float) -> Float {
    if mean_sq <= 1e-20 {
        ABS_GATE_LUFS
    } else {
        -0.691 + 10.0 * mean_sq.log10()
    }
}

/// f64 prefix sum of squares for numerically stable window energy over long signals.
fn prefix_sq(kw: &[Float]) -> Vec<f64> {
    let mut p = vec![0.0f64; kw.len() + 1];
    for (i, &v) in kw.iter().enumerate() {
        let v = v as f64;
        p[i + 1] = p[i] + v * v;
    }
    p
}

/// Short-term curve from a precomputed prefix-of-squares (win/hop in samples).
fn curve_from_prefix(prefix: &[f64], n: usize, win: usize, hop: usize) -> Vec<Float> {
    if win == 0 || n < win {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start + win <= n {
        let mean_sq = ((prefix[start + win] - prefix[start]) / win as f64) as Float;
        out.push(block_loudness(mean_sq));
        start += hop;
    }
    out
}

/// Max momentary loudness from a precomputed prefix-of-squares (win/hop in samples).
fn momentary_from_prefix(prefix: &[f64], n: usize, win: usize, hop: usize) -> Float {
    if n == 0 {
        return ABS_GATE_LUFS;
    }
    let win = win.max(1).min(n);
    let hop = hop.max(1);
    let mut max_l = ABS_GATE_LUFS;
    let mut start = 0;
    loop {
        let end = (start + win).min(n);
        let mean_sq = ((prefix[end] - prefix[start]) / (end - start) as f64) as Float;
        let l = block_loudness(mean_sq);
        if l > max_l {
            max_l = l;
        }
        if end >= n {
            break;
        }
        start += hop;
    }
    max_l
}

/// Bundle of extended loudness metrics computed from a single K-weighting pass.
pub struct LoudnessMetrics {
    /// Short-term loudness curve (3 s window, `hop_sec` hop), one LUFS per window.
    pub curve: Vec<Float>,
    /// Maximum momentary (400 ms) loudness, dB.
    pub momentary_max_db: Float,
    /// EBU R128 loudness range (LRA), LU.
    pub range_lu: Float,
}

/// Compute the short-term curve, momentary max and LRA in one pass.
///
/// This K-weights the signal a single time (instead of once per metric) and
/// derives all three block measurements from one shared prefix-sum, which is why
/// the analysis pipeline calls this rather than the individual functions.
pub fn loudness_metrics(
    y: ArrayView1<Float>,
    sr: u32,
    window_sec: Float,
    hop_sec: Float,
) -> LoudnessMetrics {
    let n = y.len();
    let hop = ((hop_sec * sr as Float).round() as usize).max(1);
    let st_win = (window_sec * sr as Float).round() as usize;
    let mom_win = ((0.4 * sr as Float).round() as usize).max(1);
    let mom_hop = ((0.1 * sr as Float).round() as usize).max(1);

    if n == 0 {
        return LoudnessMetrics {
            curve: Vec::new(),
            momentary_max_db: ABS_GATE_LUFS,
            range_lu: 0.0,
        };
    }
    let kw = perceptual::k_weighted_signal(y, sr);
    let prefix = prefix_sq(&kw);

    let curve = curve_from_prefix(&prefix, n, st_win, hop);
    let momentary_max_db = momentary_from_prefix(&prefix, n, mom_win, mom_hop);
    let range_lu = loudness_range_lu(&curve);
    LoudnessMetrics {
        curve,
        momentary_max_db,
        range_lu,
    }
}

/// Short-term loudness curve: one LUFS value per sliding window.
///
/// Per ITU-R BS.1770 / EBU R128 the short-term measurement uses a 3 s window;
/// this function slides it with a configurable hop (`hop_sec`). The analysis
/// pipeline calls it with `window_sec = 3.0`, `hop_sec = 1.0`.
///
/// Each value is the EBU R128 loudness (`-0.691 + 10*log10(mean_sq)`) of the
/// K-weighted signal over that window. The returned vector has
/// `floor((duration - window_sec) / hop_sec) + 1` entries when the track is at
/// least `window_sec` long, and is **empty** for tracks shorter than one window
/// (a documented simplification — there is no full short-term window to report).
/// Blocks below the absolute gate are reported as [`ABS_GATE_LUFS`].
pub fn loudness_curve(
    y: ArrayView1<Float>,
    sr: u32,
    window_sec: Float,
    hop_sec: Float,
) -> Vec<Float> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let win = (window_sec * sr as Float).round() as usize;
    let hop = ((hop_sec * sr as Float).round() as usize).max(1);
    if win == 0 || n < win {
        return Vec::new();
    }
    let kw = perceptual::k_weighted_signal(y, sr);
    let prefix = prefix_sq(&kw);
    curve_from_prefix(&prefix, n, win, hop)
}

/// Maximum momentary loudness (dB), using the EBU R128 400 ms momentary window.
///
/// Slides a 400 ms window (100 ms hop) over the K-weighted signal and returns the
/// loudest block's loudness. For tracks shorter than 400 ms the whole signal is
/// used as a single window. Silence / empty input returns [`ABS_GATE_LUFS`].
pub fn loudness_momentary_max_db(y: ArrayView1<Float>, sr: u32) -> Float {
    let n = y.len();
    if n == 0 {
        return ABS_GATE_LUFS;
    }
    let win = ((0.4 * sr as Float).round() as usize).max(1);
    let hop = ((0.1 * sr as Float).round() as usize).max(1);
    let kw = perceptual::k_weighted_signal(y, sr);
    let prefix = prefix_sq(&kw);
    momentary_from_prefix(&prefix, n, win, hop)
}

// ============================================================
// Loudness range (EBU R128 LRA)
// ============================================================

/// Linear-interpolated percentile of a pre-sorted ascending slice.
fn percentile_sorted(sorted: &[Float], p: Float) -> Float {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (n as Float - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as Float;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// EBU R128 loudness range (LRA) in LU, from a short-term loudness distribution.
///
/// Implements EBU Tech 3342: over the short-term (3 s) loudness values,
/// 1. discard blocks below the absolute gate (-70 LUFS);
/// 2. compute a relative threshold = (energy-mean of the remaining blocks) - 20 LU
///    and discard blocks below it;
/// 3. LRA = 95th percentile - 10th percentile of the gated distribution.
///
/// This is the standard the existing `dynamic_range_db` (a raw p95-p5 of RMS)
/// only approximates. A steady-loudness signal yields ~0 LU; material that
/// alternates loud and quiet yields a large positive value. Fewer than two
/// usable blocks (e.g. a track shorter than one short-term window) returns 0.0.
pub fn loudness_range_lu(short_term: &[Float]) -> Float {
    // Absolute gate.
    let abs_gated: Vec<Float> = short_term
        .iter()
        .copied()
        .filter(|&v| v > ABS_GATE_LUFS)
        .collect();
    if abs_gated.len() < 2 {
        return 0.0;
    }

    // Relative gate: energy mean of the absolute-gated blocks, minus 20 LU.
    let mean_pow: f64 = abs_gated
        .iter()
        .map(|&l| 10f64.powf((l as f64 + 0.691) / 10.0))
        .sum::<f64>()
        / abs_gated.len() as f64;
    let integrated = (-0.691 + 10.0 * mean_pow.log10()) as Float;
    let rel_thresh = integrated - 20.0;

    let mut gated: Vec<Float> = abs_gated.into_iter().filter(|&v| v >= rel_thresh).collect();
    if gated.len() < 2 {
        return 0.0;
    }
    gated.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p10 = percentile_sorted(&gated, 10.0);
    let p95 = percentile_sorted(&gated, 95.0);
    (p95 - p10).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::f32::consts::PI;

    fn sine(freq: Float, sr: u32, dur: Float, amp: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| {
            amp * (2.0 * PI * freq * i as Float / sr as Float).sin()
        })
    }

    #[test]
    fn test_true_peak_full_scale_sine() {
        // Full-scale 997 Hz sine -> ~0 dBTP.
        let y = sine(997.0, 48000, 1.0, 1.0);
        let tp = true_peak_db(y.view());
        assert!(tp.is_finite());
        assert!((tp - 0.0).abs() < 0.1, "expected ~0 dBTP, got {tp}");
    }

    #[test]
    fn test_true_peak_half_scale_sine() {
        // Half-scale -> ~-6.02 dBTP.
        let y = sine(997.0, 48000, 1.0, 0.5);
        let tp = true_peak_db(y.view());
        assert!(
            (tp - (-6.0206)).abs() < 0.1,
            "expected ~-6.02 dBTP, got {tp}"
        );
    }

    #[test]
    fn test_true_peak_intersample_over() {
        // A sine near fs/4 sampled so the raw samples straddle — but never land on —
        // the true crest. The oversampled true peak must exceed the raw sample peak.
        let sr = 48000u32;
        let f = sr as Float / 4.0; // fs/4
        let n = 4000usize;
        // Phase pi/4 -> samples at +/-sin(pi/4)=0.707 crests, true peak at 1.0.
        let phase = PI / 4.0;
        let y = Array1::from_shape_fn(n, |i| {
            (2.0 * PI * f * i as Float / sr as Float + phase).sin()
        });
        let sample_peak = y.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        let sample_peak_db = 20.0 * sample_peak.log10();
        let tp = true_peak_db(y.view());
        assert!(
            tp > sample_peak_db + 0.5,
            "true peak {tp} should exceed sample peak {sample_peak_db} for inter-sample over"
        );
        // True crest is at 0 dBTP; interpolation should recover close to it.
        assert!(
            tp > -0.5 && tp < 0.2,
            "inter-sample true peak {tp} not near 0 dBTP"
        );
    }

    #[test]
    fn test_replaygain_formula() {
        // Synthetic signal with a known integrated loudness.
        let y = sine(1000.0, 22050, 3.0, 0.5);
        let lufs = perceptual::loudness_lufs(y.view(), 22050);
        let rg = replaygain_db(lufs);
        assert!((rg - (-18.0 - lufs)).abs() < 1e-5);
    }

    #[test]
    fn test_loudness_curve_length() {
        // floor((dur - 3) / 1) + 1 windows.
        for dur in [4.0_f32, 6.0, 10.0, 12.0] {
            let y = sine(440.0, 22050, dur, 0.5);
            let curve = loudness_curve(y.view(), 22050, 3.0, 1.0);
            let expected = ((dur - 3.0).floor() as usize) + 1;
            assert_eq!(curve.len(), expected, "dur {dur}");
            assert!(curve.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_loudness_curve_short_track_empty() {
        // < 3 s -> empty curve.
        let y = sine(440.0, 22050, 2.0, 0.5);
        assert!(loudness_curve(y.view(), 22050, 3.0, 1.0).is_empty());
    }

    #[test]
    fn test_lra_constant_near_zero() {
        // Steady-loudness signal -> LRA ~0.
        let y = sine(440.0, 22050, 20.0, 0.5);
        let curve = loudness_curve(y.view(), 22050, 3.0, 1.0);
        let lra = loudness_range_lu(&curve);
        assert!(lra.is_finite());
        assert!(lra < 1.0, "constant signal LRA should be near 0, got {lra}");
    }

    #[test]
    fn test_lra_alternating_large() {
        // Alternate loud (amp 0.7) and quieter (amp 0.12, ~15 dB down) segments.
        // Segments are 6 s — longer than the 3 s short-term window — so some
        // windows sit fully inside a loud stretch and some fully inside a quiet
        // one. The ~15 LU gap stays inside the EBU -20 LU relative gate, so both
        // levels survive gating and the range comes out large and positive.
        let sr = 22050u32;
        let seg = 6.0;
        let seglen = (seg * sr as Float) as usize;
        let mut v = Vec::new();
        for k in 0..6 {
            let amp = if k % 2 == 0 { 0.7 } else { 0.12 };
            for i in 0..seglen {
                v.push(amp * (2.0 * PI * 440.0 * i as Float / sr as Float).sin());
            }
        }
        let y = Array1::from(v);
        let curve = loudness_curve(y.view(), sr, 3.0, 1.0);
        let lra = loudness_range_lu(&curve);
        assert!(
            lra > 5.0,
            "alternating loud/quiet LRA should be large, got {lra}"
        );
    }

    #[test]
    fn test_silence_and_short_all_finite() {
        let silence = Array1::<Float>::zeros(22050);
        assert!(true_peak_db(silence.view()).is_finite());
        assert!(loudness_momentary_max_db(silence.view(), 22050).is_finite());
        assert!(loudness_curve(silence.view(), 22050, 3.0, 1.0)
            .iter()
            .all(|v| v.is_finite()));
        assert_eq!(loudness_range_lu(&[]), 0.0);

        // Empty input.
        let empty = Array1::<Float>::zeros(0);
        assert!(true_peak_db(empty.view()).is_finite());
        assert!(loudness_momentary_max_db(empty.view(), 22050).is_finite());
        assert!(loudness_curve(empty.view(), 22050, 3.0, 1.0).is_empty());
    }

    #[test]
    fn test_momentary_ge_nothing_but_finite() {
        let y = sine(440.0, 22050, 5.0, 0.5);
        let mom = loudness_momentary_max_db(y.view(), 22050);
        let integrated = perceptual::loudness_lufs(y.view(), 22050);
        assert!(mom.is_finite());
        // Momentary max over sub-windows is >= integrated for a steady signal (within noise).
        assert!(mom >= integrated - 1.0, "mom {mom} integrated {integrated}");
    }
}
