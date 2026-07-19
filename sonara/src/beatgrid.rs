//! Beat grid and downbeat detection.
//!
//! Turns a raw list of tracked beat positions into a *grid*: where the first
//! beat falls in time, which beats are bar-starting downbeats, and how rigidly
//! the beats fit a constant-tempo lattice. This is the information DJ software
//! needs to warp, loop, and align tracks.
//!
//! The analysis is intentionally cheap — it consumes the beats and onset
//! envelope that the pipeline has already computed and adds only O(n_beats)
//! work. It never re-runs tempo estimation; the grid is purely additive.
//!
//! # Concepts
//!
//! - **grid offset**: the time (in seconds) of the first tracked beat. Together
//!   with the tempo this anchors the metric grid.
//! - **downbeats**: the subset of beats that begin a bar. Assuming a meter of
//!   `beats_per_bar` (4 by default), there are `beats_per_bar` possible phase
//!   alignments; the one whose beats carry the most onset/accent energy is
//!   chosen.
//! - **grid stability**: a 0..1 score describing how close the inter-beat
//!   intervals are to a single constant value (a rigid grid).

use ndarray::ArrayView1;

use crate::types::Float;

/// Default meter assumption when no time signature is supplied.
pub const DEFAULT_BEATS_PER_BAR: usize = 4;

/// Half-width (in frames) of the neighbourhood searched for accent energy
/// around each candidate downbeat. Beat frames from the DP tracker can sit a
/// frame or two away from the true onset peak, so a small window makes phase
/// scoring robust without smearing across neighbouring beats.
const ACCENT_WINDOW: usize = 2;

/// Result of beat-grid analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatGrid {
    /// Time (seconds) of the first tracked beat — the grid anchor.
    /// `0.0` when there are no beats.
    pub grid_offset_sec: Float,
    /// Frame indices of bar-starting beats. Always a subset of the input beats.
    pub downbeats: Vec<usize>,
    /// How rigidly the beats fit a constant-tempo grid, in `[0, 1]`.
    /// `1.0` is a perfectly regular grid; lower values indicate jitter.
    /// `0.0` when there are too few beats to judge.
    pub grid_stability: Float,
    /// Meter used for downbeat detection (beats per bar).
    pub beats_per_bar: usize,
}

/// Median of a slice of `Float`s. Returns `0.0` for an empty slice.
fn median(values: &[Float]) -> Float {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Time (seconds) of the first beat, i.e. the grid anchor.
///
/// Beat frame indices share the pipeline's frame convention, so the time of a
/// beat is simply `frame * hop_length / sr`.
pub fn grid_offset(beats: &[usize], sr: u32, hop_length: usize) -> Float {
    match beats.first() {
        Some(&f) => f as Float * hop_length as Float / sr as Float,
        None => 0.0,
    }
}

/// Grid stability in `[0, 1]`.
///
/// # Formula
///
/// Let `ibi_i = beats[i+1] - beats[i]` be the inter-beat intervals (in frames),
/// `m = median(ibi)`, and `MAD = median(|ibi_i - m|)` the median absolute
/// deviation. Then
///
/// ```text
/// stability = clamp(1 - MAD / m, 0, 1)
/// ```
///
/// A perfectly regular grid has `MAD = 0`, giving `stability = 1`. Jitter grows
/// `MAD`, monotonically lowering the score. The median/MAD pair is used (rather
/// than mean/std) so a few outlier intervals — e.g. a dropped or doubled beat —
/// do not dominate the estimate.
///
/// Returns `0.0` when there are fewer than two intervals (i.e. fewer than three
/// beats), or when the median interval is non-positive.
pub fn grid_stability(beats: &[usize]) -> Float {
    if beats.len() < 3 {
        return 0.0;
    }
    let ibis: Vec<Float> = beats
        .windows(2)
        .map(|w| w[1] as Float - w[0] as Float)
        .collect();
    let m = median(&ibis);
    if m <= 0.0 {
        return 0.0;
    }
    let devs: Vec<Float> = ibis.iter().map(|&x| (x - m).abs()).collect();
    let mad = median(&devs);
    (1.0 - mad / m).clamp(0.0, 1.0)
}

/// Peak accent energy in a small window around a beat frame.
fn accent_at(onset_env: ArrayView1<Float>, frame: usize) -> Float {
    let n = onset_env.len();
    if n == 0 {
        return 0.0;
    }
    let lo = frame.saturating_sub(ACCENT_WINDOW);
    let hi = (frame + ACCENT_WINDOW + 1).min(n);
    if lo >= hi {
        // Frame lies past the end of the envelope.
        return 0.0;
    }
    onset_env
        .slice(ndarray::s![lo..hi])
        .iter()
        .copied()
        .fold(0.0_f32, Float::max)
}

/// Detect downbeats by scoring each metric phase against onset accent energy.
///
/// For a meter of `beats_per_bar`, beat `p` starts the first bar for phase
/// `p in 0..beats_per_bar`; the bar-starting beats for that phase are
/// `beats[p], beats[p + beats_per_bar], beats[p + 2*beats_per_bar], ...`. The
/// phase whose downbeats carry the highest *mean* accent energy (onset strength
/// near the beat) is chosen — kicks and other bar-anchoring accents typically
/// land on beat one.
///
/// Returns the frame indices of the winning phase's downbeats (a subset of
/// `beats`). Empty when `beats` is empty.
pub fn detect_downbeats(
    beats: &[usize],
    onset_env: ArrayView1<Float>,
    beats_per_bar: usize,
) -> Vec<usize> {
    if beats.is_empty() {
        return vec![];
    }
    let bpb = beats_per_bar.max(1);
    if bpb == 1 {
        return beats.to_vec();
    }

    let n_phases = bpb.min(beats.len());
    let mut best_phase = 0usize;
    let mut best_score = Float::NEG_INFINITY;

    for phase in 0..n_phases {
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        let mut idx = phase;
        while idx < beats.len() {
            sum += accent_at(onset_env, beats[idx]);
            count += 1;
            idx += bpb;
        }
        // Mean accent: phases can differ in count by one, so normalising keeps
        // the comparison fair.
        let score = if count > 0 {
            sum / count as Float
        } else {
            Float::NEG_INFINITY
        };
        if score > best_score {
            best_score = score;
            best_phase = phase;
        }
    }

    let mut downbeats = Vec::new();
    let mut idx = best_phase;
    while idx < beats.len() {
        downbeats.push(beats[idx]);
        idx += bpb;
    }
    downbeats
}

/// Full beat-grid analysis: offset, downbeats, and stability in one call.
///
/// `beats` are frame indices (as returned by beat tracking), `onset_env` is the
/// onset-strength envelope in the same frame space, and `beats_per_bar` is the
/// meter numerator (pass 4 for 4/4). This is a standalone entry point; the
/// analysis pipeline calls it after beat tracking.
pub fn analyze_grid(
    beats: &[usize],
    onset_env: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    beats_per_bar: usize,
) -> BeatGrid {
    let bpb = beats_per_bar.max(1);
    BeatGrid {
        grid_offset_sec: grid_offset(beats, sr, hop_length),
        downbeats: detect_downbeats(beats, onset_env, bpb),
        grid_stability: grid_stability(beats),
        beats_per_bar: bpb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    const SR: u32 = 22050;
    const HOP: usize = 512;

    /// Frames per beat for a given BPM in the pipeline's frame space.
    fn frames_per_beat(bpm: Float) -> usize {
        (60.0 * SR as Float / HOP as Float / bpm).round() as usize
    }

    /// Build a synthetic kick-accented 4/4 onset envelope plus its beat list.
    ///
    /// `offset_frames` frames of silence are prepended (the known grid offset).
    /// The kick (strong accent) lands on every 4th beat starting at the offset;
    /// the other three beats get a weaker hat/click accent.
    fn synth_4_4(bpm: Float, n_bars: usize, offset_frames: usize) -> (Vec<usize>, Array1<Float>) {
        let fpb = frames_per_beat(bpm);
        let n_beats = n_bars * 4;
        let total = offset_frames + n_beats * fpb + fpb;
        let mut env = Array1::<Float>::zeros(total);
        let mut beats = Vec::with_capacity(n_beats);
        for b in 0..n_beats {
            let frame = offset_frames + b * fpb;
            beats.push(frame);
            // Kick on beat 1 of each bar (phase 0), weaker accent elsewhere.
            env[frame] = if b % 4 == 0 { 1.0 } else { 0.3 };
        }
        (beats, env)
    }

    #[test]
    fn grid_offset_matches_prepended_silence() {
        let offset_frames = 100usize;
        let (beats, env) = synth_4_4(120.0, 4, offset_frames);
        let grid = analyze_grid(&beats, env.view(), SR, HOP, 4);
        let expected = offset_frames as Float * HOP as Float / SR as Float;
        assert!(
            (grid.grid_offset_sec - expected).abs() < 1e-4,
            "grid_offset {} should match prepended silence {}",
            grid.grid_offset_sec,
            expected
        );
        assert!(grid.grid_offset_sec >= 0.0);
    }

    #[test]
    fn downbeats_land_on_kicks() {
        let offset_frames = 100usize;
        let (beats, env) = synth_4_4(120.0, 4, offset_frames);
        let fpb = frames_per_beat(120.0);
        let grid = analyze_grid(&beats, env.view(), SR, HOP, 4);

        // Every downbeat should be a kick frame (phase 0).
        let expected: Vec<usize> = (0..4).map(|bar| offset_frames + bar * 4 * fpb).collect();
        assert_eq!(
            grid.downbeats, expected,
            "downbeats should be the kick beats"
        );

        // Downbeats are a subset of beats and about one per bar.
        assert!(grid.downbeats.iter().all(|d| beats.contains(d)));
        assert_eq!(grid.downbeats.len(), beats.len() / 4);
    }

    #[test]
    fn downbeats_recover_shifted_phase() {
        // Put the kick on phase 2 instead of phase 0 and confirm detection.
        let fpb = frames_per_beat(120.0);
        let offset_frames = 50usize;
        let n_beats = 16usize;
        let total = offset_frames + n_beats * fpb + fpb;
        let mut env = Array1::<Float>::zeros(total);
        let mut beats = Vec::new();
        for b in 0..n_beats {
            let frame = offset_frames + b * fpb;
            beats.push(frame);
            env[frame] = if b % 4 == 2 { 1.0 } else { 0.3 };
        }
        let grid = analyze_grid(&beats, env.view(), SR, HOP, 4);
        let expected: Vec<usize> = (0..4).map(|k| offset_frames + (2 + k * 4) * fpb).collect();
        assert_eq!(grid.downbeats, expected, "should recover phase-2 downbeats");
    }

    #[test]
    fn perfectly_regular_grid_is_stable() {
        let fpb = frames_per_beat(128.0);
        let beats: Vec<usize> = (0..32).map(|i| 10 + i * fpb).collect();
        let s = grid_stability(&beats);
        assert!(s > 0.99, "regular grid stability {} should be ~1.0", s);
    }

    #[test]
    fn jitter_lowers_stability_monotonically() {
        let fpb = frames_per_beat(128.0) as i64;
        // Deterministic pseudo-random jitter in [-1, 1], scaled per level.
        let base: Vec<i64> = (0..40).collect();
        let jitter_unit = |i: usize| -> f32 {
            // Simple deterministic hash → [-1, 1].
            let x = ((i as u64).wrapping_mul(2654435761) >> 8) & 0xffff;
            (x as f32 / 32768.0) - 1.0
        };
        let make = |amp: f32| -> Vec<usize> {
            let mut pos = 10i64;
            let mut out = Vec::new();
            for (k, _) in base.iter().enumerate() {
                let j = (jitter_unit(k) * amp).round() as i64;
                let interval = (fpb + j).max(1);
                out.push(pos as usize);
                pos += interval;
            }
            out
        };
        let s0 = grid_stability(&make(0.0));
        let s_small = grid_stability(&make(fpb as f32 * 0.1));
        let s_large = grid_stability(&make(fpb as f32 * 0.4));
        assert!(s0 >= s_small - 1e-6, "s0 {} >= s_small {}", s0, s_small);
        assert!(
            s_small >= s_large - 1e-6,
            "s_small {} >= s_large {}",
            s_small,
            s_large
        );
        assert!(
            s0 > s_large,
            "more jitter must reduce stability: {} vs {}",
            s0,
            s_large
        );
    }

    #[test]
    fn empty_beats() {
        let env = Array1::<Float>::zeros(100);
        let grid = analyze_grid(&[], env.view(), SR, HOP, 4);
        assert_eq!(grid.grid_offset_sec, 0.0);
        assert!(grid.downbeats.is_empty());
        assert_eq!(grid.grid_stability, 0.0);
    }

    #[test]
    fn single_beat() {
        let mut env = Array1::<Float>::zeros(200);
        env[64] = 1.0;
        let grid = analyze_grid(&[64], env.view(), SR, HOP, 4);
        assert!((grid.grid_offset_sec - 64.0 * HOP as Float / SR as Float).abs() < 1e-4);
        assert_eq!(grid.downbeats, vec![64]);
        assert_eq!(grid.grid_stability, 0.0, "single beat cannot form a grid");
    }

    #[test]
    fn two_beats_no_crash() {
        let fpb = frames_per_beat(120.0);
        let env = Array1::<Float>::zeros(3 * fpb);
        let grid = analyze_grid(&[10, 10 + fpb], env.view(), SR, HOP, 4);
        // Fewer than 3 beats → stability undefined, reported as 0.0.
        assert_eq!(grid.grid_stability, 0.0);
        assert!(!grid.downbeats.is_empty());
    }

    #[test]
    fn silence_flat_envelope() {
        // Beats present but a flat (silent) envelope: no accent to distinguish
        // phases; must still return a valid grid and pick phase 0 by tie-break.
        let fpb = frames_per_beat(120.0);
        let beats: Vec<usize> = (0..16).map(|i| i * fpb).collect();
        let env = Array1::<Float>::zeros(17 * fpb);
        let grid = analyze_grid(&beats, env.view(), SR, HOP, 4);
        assert_eq!(grid.downbeats.first(), Some(&0));
        assert_eq!(grid.downbeats.len(), 4);
        // Perfectly regular spacing → high stability even without accents.
        assert!(grid.grid_stability > 0.99);
    }

    #[test]
    fn meter_three_four() {
        // Waltz: kick every 3rd beat.
        let fpb = frames_per_beat(120.0);
        let n_beats = 12usize;
        let mut env = Array1::<Float>::zeros((n_beats + 1) * fpb);
        let mut beats = Vec::new();
        for b in 0..n_beats {
            let frame = b * fpb;
            beats.push(frame);
            env[frame] = if b % 3 == 0 { 1.0 } else { 0.3 };
        }
        let grid = analyze_grid(&beats, env.view(), SR, HOP, 3);
        assert_eq!(grid.beats_per_bar, 3);
        assert_eq!(grid.downbeats.len(), 4);
        assert_eq!(grid.downbeats[0], 0);
    }
}
