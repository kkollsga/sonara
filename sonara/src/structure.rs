//! Structural segmentation and energy curve.
//!
//! Turns a track from "one number per feature" into "where things happen":
//! a time-resolved energy curve, novelty-based section boundaries, an
//! intro/outro estimate, and a coarse 1-10 energy level.
//!
//! Everything here is derived from arrays that the fused analysis pass already
//! computed (per-frame RMS, spectral centroid, spectral bandwidth, and the mel
//! dB spectrogram). No second decode or FFT pass is required.
//!
//! This module is **opt-in**: it is only run when the caller explicitly
//! requests `features=["structure"]`. It is never computed by the compact,
//! playlist, or full modes on their own, so it adds zero cost to the default
//! pipelines.
//!
//! ## Energy curve
//!
//! The track is chopped into overlapping windows (`WIN_SEC` = 1.0 s long,
//! `HOP_SEC` = 0.5 s apart). For each window we average the per-frame RMS,
//! centroid, and bandwidth and count the onsets that fall inside it, then feed
//! those into the same `perceptual::energy` 0-1 model used for the whole-track
//! energy scalar. The result is `energy_curve` (a `Vec` of 0-1 values), with
//! `energy_curve_hop_sec` = 0.5 so consumers can map index → time
//! (`t = i * energy_curve_hop_sec`).
//!
//! ## Segment boundaries (Foote novelty)
//!
//! Classical self-similarity novelty segmentation (J. Foote, 2000):
//!
//! 1. **Descriptor** — per window, a 13-dim timbral descriptor: the mean MFCC
//!    over the window's frames (MFCC = DCT of the mel dB spectrum). Timbre is a
//!    good cue for section changes (e.g. a breakdown vs a drop).
//! 2. **Self-similarity matrix (SSM)** — descriptors are mean-centred per
//!    dimension and L2-normalised, then `S[i,j]` = cosine similarity of windows
//!    `i` and `j`.
//! 3. **Checkerboard novelty** — a Gaussian-tapered checkerboard kernel
//!    (half-width `KERNEL_SEC` ≈ 4 s) is slid down the SSM diagonal. It responds
//!    strongly where within-block similarity is high but cross-block similarity
//!    is low — i.e. a boundary between two homogeneous sections.
//! 4. **Peak-pick** — the novelty curve is normalised, thresholded at
//!    `mean + NOVELTY_THRESH_STD·std` (floored at `NOVELTY_THRESH_FLOOR`),
//!    and peaks are greedily selected in
//!    novelty-descending order subject to a `MIN_SEGMENT_SEC` (8 s) minimum
//!    spacing, capped at `MAX_BOUNDARIES` interior boundaries. This keeps the
//!    count sane (typically 4-12 segments for a 5-minute track).
//!
//! `segments` is a list of [`SegmentEvent`]s, contiguous and
//! covering the whole track (first starts at 0, last ends at the duration).
//! A constant or silent track yields a single whole-track segment.
//!
//! ## Intro / outro (heuristic)
//!
//! Honestly a heuristic, not a trained model. We take the 10th and 90th
//! percentiles of the energy curve and set `thresh = p10 + 0.5·(p90 - p10)`.
//! `intro_end_sec` is the first window that reaches `thresh` (the end of the
//! initial low-energy / pre-first-drop region); `outro_start_sec` is just after
//! the last window that reaches it (the start of the final fade). Each is
//! snapped to a nearby segment boundary (within `SNAP_SEC`) when one exists, for
//! cleaner alignment. A flat track gives `intro_end = 0`, `outro_start = duration`.
//!
//! ## Energy level (1-10)
//!
//! `energy_level` maps the mean of the energy curve onto an integer 1-10. The
//! 0-1 `perceptual::energy` model runs through a sigmoid, so real-world music
//! clusters in a narrow band: measured over a 9,400-track commercial library
//! (pop/hip-hop/dance), mean track energy spans ~0.25 (quiet ballads) to ~0.62
//! (peak-time dance), median 0.45. Linearly mapping the raw 0-1 value would
//! bunch everything around 4-5; instead the observed 0.25-0.60 band is
//! stretched across the full 1-10 range so tracks actually spread out. The
//! mapping is monotonic: a louder/brighter track never gets a lower level.

use ndarray::ArrayView2;

use crate::perceptual;
use crate::types::Float;

// ---- Windowing ----
/// Energy/descriptor window length (seconds).
const WIN_SEC: Float = 1.0;
/// Energy/descriptor hop (seconds). Also `energy_curve_hop_sec`.
const HOP_SEC: Float = 0.5;

// ---- Segmentation ----
/// Checkerboard kernel half-width (seconds) — the novelty "look-around".
const KERNEL_SEC: Float = 4.0;
/// Minimum spacing between boundaries / minimum segment length (seconds).
const MIN_SEGMENT_SEC: Float = 8.0;
/// Maximum number of interior boundaries (→ at most this + 1 segments).
const MAX_BOUNDARIES: usize = 11;
/// Snap intro/outro to a segment boundary within this distance (seconds).
const SNAP_SEC: Float = 5.0;
/// Novelty peak threshold: `mean + this·std` of the normalised novelty curve.
/// Calibrated on a real commercial library (60-track random sample): at 0.5
/// the MAX_BOUNDARIES cap bound on most 3-4 minute pop tracks (median 12
/// segments, cap binding >50%); at 2.0 the distribution centres on ~8 segments
/// (p25=6, p75=9) with the cap binding <10% and no degenerate
/// under-segmentation, while the synthetic known-structure tests still recover
/// their boundaries. Tuning for boundary *accuracy* (vs. plausibility) needs
/// annotated structure data and is future work.
const NOVELTY_THRESH_STD: Float = 2.0;
/// Absolute floor for the normalised novelty threshold.
const NOVELTY_THRESH_FLOOR: Float = 0.25;

/// Number of MFCC coefficients used for the timbral descriptor.
const N_MFCC: usize = 13;

/// A structural section with its time span, in seconds.
///
/// Segments are contiguous, ordered, and cover the whole track: the first
/// starts at 0.0 and the last ends at the track duration.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentEvent {
    pub start_sec: Float,
    pub end_sec: Float,
    /// Mean perceptual energy (0-1) over the span.
    pub energy: Float,
}

/// Result of structural analysis.
pub struct StructureResult {
    /// Time-resolved perceptual energy (0-1), one value per window.
    pub energy_curve: Vec<Float>,
    /// Seconds between successive `energy_curve` samples.
    pub energy_curve_hop_sec: Float,
    /// Contiguous sections covering the track. See [`SegmentEvent`].
    pub segments: Vec<SegmentEvent>,
    /// End of the initial low-energy / pre-first-drop region (seconds).
    pub intro_end_sec: Float,
    /// Start of the final fade / low-energy region (seconds).
    pub outro_start_sec: Float,
    /// Coarse 1-10 energy level derived from mean energy.
    pub energy_level: u8,
}

/// Map a 0-1 perceptual energy value to an integer 1-10 level.
///
/// See the module docs: the 0.25-0.60 band of the sigmoid-shaped energy model
/// (measured over a large real-music library) is stretched across 1-10 so real
/// music spreads out instead of clustering. Monotonic non-decreasing in `e`.
pub fn energy_level_from_energy(e: Float) -> u8 {
    let t = ((e - 0.25) / (0.60 - 0.25)).clamp(0.0, 1.0);
    1 + (t * 9.0).round() as u8
}

/// Compute structural features from per-frame pipeline arrays.
///
/// `rms_frames`, `centroids`, `bandwidths` are per-frame (length `n_frames`);
/// `bandwidths` may be empty if not available (treated as 0). `s_db` is the mel
/// dB spectrogram `(n_mels, n_frames)`; `dct_matrix` is `(N_MFCC, n_mels)`.
/// `onset_frames` are onset positions in frame units. `fps` is frames/second
/// (`sr / hop_length`).
pub fn analyze_structure(
    rms_frames: &[Float],
    centroids: &[Float],
    bandwidths: &[Float],
    s_db: ArrayView2<Float>,
    dct_matrix: ArrayView2<Float>,
    onset_frames: &[usize],
    fps: Float,
    duration_sec: Float,
) -> StructureResult {
    let n_frames = rms_frames.len();
    let n_mels = s_db.shape()[0];
    let have_bw = bandwidths.len() == n_frames;

    let win_frames = ((WIN_SEC * fps).round() as usize).max(1);
    let hop_frames = ((HOP_SEC * fps).round() as usize).max(1);
    // Actual seconds between window starts (frame-quantized, ≈ HOP_SEC). This is
    // the true index→time factor: `win_start_sec[i] == i * hop_sec`.
    let hop_sec = hop_frames as Float / fps;

    // ---- Build per-window energy curve + timbral descriptors ----
    let mut energy_curve: Vec<Float> = Vec::new();
    let mut win_start_sec: Vec<Float> = Vec::new();
    let mut descriptors: Vec<[Float; N_MFCC]> = Vec::new();

    let mut start_frame = 0usize;
    while start_frame < n_frames.max(1) {
        let end_frame = (start_frame + win_frames).min(n_frames);
        let nf = end_frame.saturating_sub(start_frame).max(1);

        // Aggregate scalar features over the window.
        let mut sum_rms = 0.0;
        let mut sum_cent = 0.0;
        let mut sum_bw = 0.0;
        for t in start_frame..end_frame {
            sum_rms += rms_frames[t];
            sum_cent += centroids[t];
            if have_bw {
                sum_bw += bandwidths[t];
            }
        }
        let mean_rms = sum_rms / nf as Float;
        let mean_cent = sum_cent / nf as Float;
        let mean_bw = if have_bw { sum_bw / nf as Float } else { 0.0 };

        // Onsets landing in this window → local onset density (onsets/sec).
        let win_sec = nf as Float / fps;
        let n_onsets = onset_frames
            .iter()
            .filter(|&&f| f >= start_frame && f < end_frame)
            .count();
        let onset_density = n_onsets as Float / win_sec.max(1e-6);

        let e = perceptual::energy(mean_rms, mean_cent, onset_density, mean_bw);
        energy_curve.push(e);
        win_start_sec.push(start_frame as Float / fps);

        // Timbral descriptor: mean MFCC over the window.
        // MFCC[k] = sum_m dct[k,m] * s_db[m,t]; mean over frames == DCT of the
        // per-mel mean, so accumulate the mel sum once then apply the DCT.
        let mut mel_sum = vec![0.0_f32; n_mels];
        for t in start_frame..end_frame {
            for m in 0..n_mels {
                mel_sum[m] += s_db[(m, t)];
            }
        }
        let mut desc = [0.0_f32; N_MFCC];
        for k in 0..N_MFCC {
            let mut acc = 0.0;
            for m in 0..n_mels {
                acc += dct_matrix[(k, m)] * mel_sum[m];
            }
            desc[k] = acc / nf as Float;
        }
        descriptors.push(desc);

        if end_frame >= n_frames {
            break;
        }
        start_frame += hop_frames;
    }

    let n_win = energy_curve.len();
    let mean_energy = if n_win > 0 {
        energy_curve.iter().sum::<Float>() / n_win as Float
    } else {
        0.0
    };
    let energy_level = energy_level_from_energy(mean_energy);

    // ---- Segment boundaries via Foote checkerboard novelty ----
    let kernel_half = ((KERNEL_SEC / HOP_SEC).round() as usize).max(2);
    let min_gap = ((MIN_SEGMENT_SEC / HOP_SEC).round() as usize).max(1);

    let boundaries: Vec<Float> = if n_win >= 4 * kernel_half && has_variation(&descriptors) {
        let ssm = build_ssm(&descriptors);
        let novelty = checkerboard_novelty(&ssm, kernel_half);
        pick_boundaries(&novelty, &win_start_sec, min_gap)
    } else {
        Vec::new()
    };

    // ---- Assemble contiguous segments ----
    let mut cuts: Vec<Float> = Vec::with_capacity(boundaries.len() + 2);
    cuts.push(0.0);
    for &b in &boundaries {
        if b > MIN_SEGMENT_SEC * 0.5 && b < duration_sec - MIN_SEGMENT_SEC * 0.5 {
            cuts.push(b);
        }
    }
    cuts.push(duration_sec);
    cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-3);

    let mut segments: Vec<SegmentEvent> = Vec::new();
    for w in cuts.windows(2) {
        let (start, end) = (w[0], w[1]);
        if end - start < 1e-3 {
            continue;
        }
        let me = segment_mean_energy(&energy_curve, &win_start_sec, start, end, mean_energy);
        segments.push(SegmentEvent {
            start_sec: start,
            end_sec: end,
            energy: me,
        });
    }
    if segments.is_empty() {
        segments.push(SegmentEvent {
            start_sec: 0.0,
            end_sec: duration_sec.max(0.0),
            energy: mean_energy,
        });
    }

    // ---- Intro / outro heuristic ----
    let (intro_end_sec, outro_start_sec) = intro_outro(
        &energy_curve,
        &win_start_sec,
        &segments,
        duration_sec,
        hop_sec,
    );

    StructureResult {
        energy_curve,
        energy_curve_hop_sec: hop_sec,
        segments,
        intro_end_sec,
        outro_start_sec,
        energy_level,
    }
}

/// Mean descriptor across all windows (per dimension).
fn descriptor_mean(descriptors: &[[Float; N_MFCC]]) -> [Float; N_MFCC] {
    let mut mean = [0.0_f32; N_MFCC];
    for d in descriptors {
        for k in 0..N_MFCC {
            mean[k] += d[k];
        }
    }
    let n = descriptors.len().max(1) as Float;
    for k in 0..N_MFCC {
        mean[k] /= n;
    }
    mean
}

/// Whether the descriptors carry meaningful timbral variation.
///
/// A constant or silent track has descriptors that are all essentially equal;
/// the only differences are floating-point residuals (e.g. from trailing
/// partial windows). Running the SSM/novelty pipeline on those would fabricate
/// boundaries out of numerical noise, so we bail out when the largest deviation
/// from the mean is negligible relative to the descriptor scale.
fn has_variation(descriptors: &[[Float; N_MFCC]]) -> bool {
    if descriptors.len() < 2 {
        return false;
    }
    let mean = descriptor_mean(descriptors);
    let mut max_dev = 0.0_f32;
    let mut scale = 0.0_f32;
    for d in descriptors {
        let mut dev = 0.0;
        let mut raw = 0.0;
        for k in 0..N_MFCC {
            dev += (d[k] - mean[k]).powi(2);
            raw += d[k] * d[k];
        }
        max_dev = max_dev.max(dev.sqrt());
        scale = scale.max(raw.sqrt());
    }
    // Relative floor rejects FP-noise-only variation; absolute floor guards the
    // all-zero-descriptor (true silence) case.
    max_dev > 1e-2 * scale.max(1e-6) && max_dev > 1e-3
}

/// Build the cosine self-similarity matrix from per-window descriptors.
///
/// Descriptors are mean-centred per dimension (removes the constant timbral
/// offset so the SSM reflects *relative* change) then L2-normalised per window,
/// so `S[i,j]` is a cosine similarity in roughly [-1, 1].
fn build_ssm(descriptors: &[[Float; N_MFCC]]) -> Vec<Vec<Float>> {
    let n = descriptors.len();
    let mean = descriptor_mean(descriptors);
    // Centre + L2 normalise.
    let mut norm: Vec<[Float; N_MFCC]> = Vec::with_capacity(n);
    for d in descriptors {
        let mut v = [0.0_f32; N_MFCC];
        let mut mag = 0.0;
        for k in 0..N_MFCC {
            v[k] = d[k] - mean[k];
            mag += v[k] * v[k];
        }
        let mag = mag.sqrt();
        if mag > 1e-9 {
            for k in 0..N_MFCC {
                v[k] /= mag;
            }
        }
        norm.push(v);
    }
    let mut ssm = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0;
            for k in 0..N_MFCC {
                dot += norm[i][k] * norm[j][k];
            }
            ssm[i][j] = dot;
            ssm[j][i] = dot;
        }
    }
    ssm
}

/// Slide a Gaussian-tapered checkerboard kernel down the SSM diagonal.
fn checkerboard_novelty(ssm: &[Vec<Float>], l: usize) -> Vec<Float> {
    let n = ssm.len();
    let size = 2 * l;
    let sigma = l as Float / 2.0;
    let two_sig2 = 2.0 * sigma * sigma;

    // Kernel[a,b] = sign(da·db) · Gaussian(da,db), da/db centred on the corner.
    let mut kernel = vec![0.0_f32; size * size];
    for a in 0..size {
        for b in 0..size {
            let da = a as Float - l as Float + 0.5;
            let db = b as Float - l as Float + 0.5;
            let g = (-(da * da + db * db) / two_sig2).exp();
            let sign = (da * db).signum();
            kernel[a * size + b] = sign * g;
        }
    }

    let mut novelty = vec![0.0_f32; n];
    for i in 0..n {
        // Need indices i-l .. i+l-1 in range.
        if i < l || i + l > n {
            continue;
        }
        let mut acc = 0.0;
        for a in 0..size {
            let ii = i + a - l;
            let row = &ssm[ii];
            let krow = &kernel[a * size..a * size + size];
            for b in 0..size {
                acc += krow[b] * row[i + b - l];
            }
        }
        novelty[i] = acc;
    }
    novelty
}

/// Normalise novelty, threshold adaptively, and greedily pick peaks.
fn pick_boundaries(novelty: &[Float], win_start_sec: &[Float], min_gap: usize) -> Vec<Float> {
    let n = novelty.len();
    let max_nov = novelty.iter().copied().fold(0.0_f32, Float::max);
    if max_nov <= 1e-6 {
        return Vec::new();
    }
    // Normalise positive part to [0, 1].
    let norm: Vec<Float> = novelty.iter().map(|&v| (v / max_nov).max(0.0)).collect();

    // Adaptive threshold over the region where the kernel is defined (nonzero).
    let valid: Vec<Float> = norm.iter().copied().filter(|&v| v > 0.0).collect();
    let thresh = if valid.is_empty() {
        return Vec::new();
    } else {
        let mean = valid.iter().sum::<Float>() / valid.len() as Float;
        let var = valid.iter().map(|&v| (v - mean).powi(2)).sum::<Float>() / valid.len() as Float;
        (mean + NOVELTY_THRESH_STD * var.sqrt()).max(NOVELTY_THRESH_FLOOR)
    };

    // Candidate = strict local maximum above threshold.
    let mut candidates: Vec<usize> = Vec::new();
    for i in 1..n.saturating_sub(1) {
        if norm[i] >= thresh && norm[i] > norm[i - 1] && norm[i] >= norm[i + 1] {
            candidates.push(i);
        }
    }
    // Greedy selection in novelty-descending order with min-gap spacing.
    candidates.sort_by(|&a, &b| norm[b].partial_cmp(&norm[a]).unwrap());
    let mut chosen: Vec<usize> = Vec::new();
    for &c in &candidates {
        if chosen.len() >= MAX_BOUNDARIES {
            break;
        }
        if chosen
            .iter()
            .all(|&s| (s as isize - c as isize).unsigned_abs() >= min_gap)
        {
            chosen.push(c);
        }
    }
    chosen.sort_unstable();
    chosen.iter().map(|&i| win_start_sec[i]).collect()
}

/// Mean energy over windows whose start falls in `[start, end)`.
fn segment_mean_energy(
    energy_curve: &[Float],
    win_start_sec: &[Float],
    start: Float,
    end: Float,
    fallback: Float,
) -> Float {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (w, &t) in win_start_sec.iter().enumerate() {
        if t >= start && t < end {
            sum += energy_curve[w];
            count += 1;
        }
    }
    if count > 0 {
        sum / count as Float
    } else {
        fallback
    }
}

/// Percentile of a slice (linear, unsorted input).
fn percentile(values: &[Float], p: Float) -> Float {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (v.len() - 1) as Float).round() as usize;
    v[idx.min(v.len() - 1)]
}

/// Derive intro end / outro start from the energy curve, snapped to boundaries.
fn intro_outro(
    energy_curve: &[Float],
    win_start_sec: &[Float],
    segments: &[SegmentEvent],
    duration_sec: Float,
    hop_sec: Float,
) -> (Float, Float) {
    let n = energy_curve.len();
    if n == 0 {
        return (0.0, duration_sec);
    }
    let p10 = percentile(energy_curve, 10.0);
    let p90 = percentile(energy_curve, 90.0);
    let thresh = p10 + 0.5 * (p90 - p10);

    // First window reaching the threshold.
    let mut intro_end = 0.0;
    for w in 0..n {
        if energy_curve[w] >= thresh {
            intro_end = win_start_sec[w];
            break;
        }
    }
    // Just after the last window reaching the threshold.
    let mut outro_start = duration_sec;
    for w in (0..n).rev() {
        if energy_curve[w] >= thresh {
            outro_start = (win_start_sec[w] + hop_sec).min(duration_sec);
            break;
        }
    }

    // Snap to a nearby interior segment boundary for cleaner alignment.
    let interior: Vec<Float> = segments.iter().skip(1).map(|s| s.start_sec).collect();
    intro_end = snap(intro_end, &interior);
    outro_start = snap(outro_start, &interior);

    if !(intro_end < outro_start) {
        return (0.0, duration_sec);
    }
    (intro_end.max(0.0), outro_start.min(duration_sec))
}

/// Snap `t` to the nearest value in `bounds` within `SNAP_SEC`, else return `t`.
fn snap(t: Float, bounds: &[Float]) -> Float {
    let mut best = t;
    let mut best_d = SNAP_SEC;
    for &b in bounds {
        let d = (b - t).abs();
        if d < best_d {
            best_d = d;
            best = b;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Build synthetic per-frame arrays with a known low/high/low structure.
    // Returns (rms, centroid, bandwidth, s_db, dct, onset_frames, fps, duration).
    struct Synth {
        rms: Vec<Float>,
        cent: Vec<Float>,
        bw: Vec<Float>,
        s_db: Array2<Float>,
        dct: Array2<Float>,
        onsets: Vec<usize>,
        fps: Float,
        dur: Float,
    }

    fn synth_low_high_low() -> Synth {
        // 30s quiet -> 60s loud/broadband -> 30s quiet, at ~43 fps.
        let fps = 43.0;
        let n_mels = 16;
        let secs = [(30.0, false), (60.0, true), (30.0, false)];
        let mut rms = Vec::new();
        let mut cent = Vec::new();
        let mut bw = Vec::new();
        let mut onsets = Vec::new();
        let mut mel_cols: Vec<Vec<Float>> = Vec::new();
        let mut frame = 0usize;
        for &(dur, loud) in &secs {
            let nf = (dur * fps) as usize;
            for _ in 0..nf {
                if loud {
                    rms.push(0.35);
                    cent.push(3000.0);
                    bw.push(2500.0);
                    // broadband spectrum
                    mel_cols.push((0..n_mels).map(|m| -10.0 - m as Float).collect());
                    if frame % 10 == 0 {
                        onsets.push(frame);
                    }
                } else {
                    rms.push(0.03);
                    cent.push(500.0);
                    bw.push(400.0);
                    // low-frequency, narrowband spectrum
                    mel_cols.push(
                        (0..n_mels)
                            .map(|m| if m < 2 { -20.0 } else { -70.0 })
                            .collect(),
                    );
                    if frame % 40 == 0 {
                        onsets.push(frame);
                    }
                }
                frame += 1;
            }
        }
        let n_frames = mel_cols.len();
        let mut s_db = Array2::<Float>::zeros((n_mels, n_frames));
        for (t, col) in mel_cols.iter().enumerate() {
            for m in 0..n_mels {
                s_db[(m, t)] = col[m];
            }
        }
        // Simple DCT-II matrix (13 x n_mels).
        let dct = Array2::from_shape_fn((N_MFCC, n_mels), |(k, m)| {
            let norm = if k == 0 {
                (1.0 / n_mels as Float).sqrt()
            } else {
                (2.0 / n_mels as Float).sqrt()
            };
            norm * (std::f32::consts::PI * k as Float * (2 * m + 1) as Float
                / (2.0 * n_mels as Float))
                .cos()
        });
        let dur = n_frames as Float / fps;
        Synth {
            rms,
            cent,
            bw,
            s_db,
            dct,
            onsets,
            fps,
            dur,
        }
    }

    fn run(s: &Synth) -> StructureResult {
        analyze_structure(
            &s.rms,
            &s.cent,
            &s.bw,
            s.s_db.view(),
            s.dct.view(),
            &s.onsets,
            s.fps,
            s.dur,
        )
    }

    #[test]
    fn test_boundaries_near_truth() {
        let s = synth_low_high_low();
        let r = run(&s);
        // Expect a boundary near 30s and near 90s.
        let interior: Vec<Float> = r.segments.iter().skip(1).map(|seg| seg.start_sec).collect();
        let near = |target: Float| interior.iter().any(|&b| (b - target).abs() < 6.0);
        assert!(near(30.0), "expected boundary near 30s, got {:?}", interior);
        assert!(near(90.0), "expected boundary near 90s, got {:?}", interior);
        // Sane count.
        assert!(
            r.segments.len() >= 3 && r.segments.len() <= 12,
            "segment count {} out of range",
            r.segments.len()
        );
    }

    #[test]
    fn test_energy_curve_shape_and_length() {
        let s = synth_low_high_low();
        let r = run(&s);
        // Length ≈ duration / hop.
        let expected = (s.dur / r.energy_curve_hop_sec).round() as usize;
        assert!(
            (r.energy_curve.len() as isize - expected as isize).abs() <= 2,
            "curve len {} vs expected {}",
            r.energy_curve.len(),
            expected
        );
        // Low-high-low shape: middle window much higher than the ends.
        let n = r.energy_curve.len();
        let early = r.energy_curve[n / 10];
        let mid = r.energy_curve[n / 2];
        let late = r.energy_curve[n - n / 10 - 1];
        assert!(
            mid > early + 0.2,
            "mid {} should exceed early {}",
            mid,
            early
        );
        assert!(mid > late + 0.2, "mid {} should exceed late {}", mid, late);
    }

    #[test]
    fn test_intro_outro_sane() {
        let s = synth_low_high_low();
        let r = run(&s);
        // Intro ends somewhere in the first section, outro starts in the last.
        assert!(
            r.intro_end_sec > 10.0 && r.intro_end_sec < 45.0,
            "intro_end {} unexpected",
            r.intro_end_sec
        );
        assert!(
            r.outro_start_sec > 80.0 && r.outro_start_sec < s.dur,
            "outro_start {} unexpected",
            r.outro_start_sec
        );
        assert!(r.intro_end_sec < r.outro_start_sec);
    }

    #[test]
    fn test_segments_cover_and_order() {
        let s = synth_low_high_low();
        let r = run(&s);
        assert!(
            r.segments.first().unwrap().start_sec.abs() < 1e-3,
            "first must start at 0"
        );
        assert!(
            (r.segments.last().unwrap().end_sec - s.dur).abs() < 1e-2,
            "last must end at duration"
        );
        for w in r.segments.windows(2) {
            assert!(
                (w[0].end_sec - w[1].start_sec).abs() < 1e-3,
                "segments must be contiguous"
            );
            assert!(w[0].end_sec > w[0].start_sec, "segments must be ordered");
        }
    }

    #[test]
    fn test_energy_level_monotonic() {
        // Level never decreases as energy rises.
        let mut prev = 0u8;
        let mut e = 0.0;
        while e <= 1.0 {
            let lvl = energy_level_from_energy(e);
            assert!(lvl >= prev, "level dropped at e={}", e);
            assert!(lvl >= 1 && lvl <= 10, "level {} out of range", lvl);
            prev = lvl;
            e += 0.01;
        }
        assert_eq!(energy_level_from_energy(0.0), 1);
        assert_eq!(energy_level_from_energy(1.0), 10);
    }

    #[test]
    fn test_energy_level_louder_brighter_not_lower() {
        // Louder + brighter feeds a higher perceptual energy → level must not drop.
        let quiet = perceptual::energy(0.05, 600.0, 1.0, 500.0);
        let loud = perceptual::energy(0.35, 3000.0, 5.0, 2500.0);
        assert!(
            energy_level_from_energy(loud) >= energy_level_from_energy(quiet),
            "louder/brighter level {} < quieter level {}",
            energy_level_from_energy(loud),
            energy_level_from_energy(quiet)
        );
    }

    #[test]
    fn test_constant_signal_no_spurious_boundaries() {
        // Flat features → no interior boundaries (0 or 1 segment).
        let fps = 43.0;
        let n_mels = 16;
        let n_frames = (120.0 * fps) as usize;
        let rms = vec![0.2_f32; n_frames];
        let cent = vec![1500.0_f32; n_frames];
        let bw = vec![1200.0_f32; n_frames];
        let s_db = Array2::<Float>::from_elem((n_mels, n_frames), -20.0);
        let dct = Array2::from_shape_fn((N_MFCC, n_mels), |(k, m)| {
            let norm = if k == 0 {
                (1.0 / n_mels as Float).sqrt()
            } else {
                (2.0 / n_mels as Float).sqrt()
            };
            norm * (std::f32::consts::PI * k as Float * (2 * m + 1) as Float
                / (2.0 * n_mels as Float))
                .cos()
        });
        let dur = n_frames as Float / fps;
        let r = analyze_structure(&rms, &cent, &bw, s_db.view(), dct.view(), &[], fps, dur);
        assert!(
            r.segments.len() <= 1,
            "constant signal should not be split, got {} segments",
            r.segments.len()
        );
    }

    #[test]
    fn test_short_track() {
        // < 10s track: should not panic, single segment, curve non-empty.
        let fps = 43.0;
        let n_mels = 16;
        let n_frames = (5.0 * fps) as usize;
        let rms = vec![0.1_f32; n_frames];
        let cent = vec![1000.0_f32; n_frames];
        let bw = vec![800.0_f32; n_frames];
        let s_db = Array2::<Float>::from_elem((n_mels, n_frames), -30.0);
        let dct = Array2::from_shape_fn((N_MFCC, n_mels), |(k, m)| 0.1 * (k + m) as Float);
        let dur = n_frames as Float / fps;
        let r = analyze_structure(&rms, &cent, &bw, s_db.view(), dct.view(), &[], fps, dur);
        assert!(!r.energy_curve.is_empty());
        assert_eq!(r.segments.len(), 1);
        assert!(r.energy_level >= 1 && r.energy_level <= 10);
    }

    #[test]
    fn test_silence() {
        let fps = 43.0;
        let n_mels = 16;
        let n_frames = (60.0 * fps) as usize;
        let rms = vec![0.0_f32; n_frames];
        let cent = vec![0.0_f32; n_frames];
        let bw = vec![0.0_f32; n_frames];
        let s_db = Array2::<Float>::from_elem((n_mels, n_frames), -80.0);
        let dct = Array2::from_shape_fn((N_MFCC, n_mels), |(k, m)| 0.1 * (k + m) as Float);
        let dur = n_frames as Float / fps;
        let r = analyze_structure(&rms, &cent, &bw, s_db.view(), dct.view(), &[], fps, dur);
        assert!(r.segments.len() <= 1);
        assert_eq!(r.energy_level, 1, "silence should be level 1");
        assert!(r.intro_end_sec < r.outro_start_sec);
    }
}
