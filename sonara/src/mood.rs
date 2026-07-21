//! Mood heuristics isolated from the fused analysis orchestration.
//!
//! This module preserves heuristic v1 bit-for-bit. Future aggression research
//! can evolve here without assigning all of `analyze.rs` to one accuracy
//! domain or disturbing the other perceptual descriptors.

use crate::perceptual::{danceability_heuristic, energy, KeyResult};
use crate::types::Float;

/// Four rough mood affinities, each in `[0, 1]`. Heuristic v1 — NOT an ML
/// classifier. The four scores are correlated (a happy track tends to be
/// un-sad) but are computed independently and are **not** constrained to sum
/// to 1.
pub struct MoodScores {
    /// Bright/upbeat affinity: major mode + moderate-fast tempo + brightness + danceability.
    pub happy: Float,
    /// Intense/harsh affinity: energy + rhythmic density + dissonance + minor nudge.
    pub aggressive: Float,
    /// Calm affinity: low energy + slow tempo + sparse onsets + narrow dynamics.
    pub relaxed: Float,
    /// Melancholic affinity: minor mode + slow tempo + darkness + low energy.
    pub sad: Float,
}

/// Heuristic mood affinities (happy / aggressive / relaxed / sad), each `[0, 1]`.
///
/// **Heuristic v1, not ML.** These are rough hints derived from scalars the
/// fused pipeline already produced. No extra signal processing. Perceived mood
/// is subjective and context-dependent; treat these as coarse tags, not ground
/// truth.
///
/// The two composite drivers (`energy`, `danceability_heuristic`) are recomputed
/// internally from the same raw scalars, so a mood request does not depend on
/// whether the caller also asked for those fields.
///
/// Terms (each normalized to `[0, 1]` over empirical music ranges):
/// - `mode_major` / `mode_minor`: 1/0 for a confident major key, 0/1 for minor,
///   0.5/0.5 when key confidence `< 0.05`.
/// - `tempo` = `((bpm-60)/120)`; `slow` = `1-tempo`.
/// - `brightness` = `((centroid-1000)/3000)`; `darkness` = `1-brightness`.
/// - `onset` = `onset_density/8`; `low_onset` = `1-onset`.
/// - `diss` = clamped `dissonance` (0 when unavailable).
/// - `narrow_dyn` = `1 - dynamic_range_db/20`.
///
/// Weighted sums:
/// - happy      = 0.35·major + 0.25·tempo + 0.20·brightness + 0.20·dance
/// - aggressive = 0.35·energy + 0.30·onset + 0.20·diss + 0.15·minor
/// - relaxed    = 0.30·(1-energy) + 0.25·slow + 0.25·low_onset + 0.20·narrow_dyn
/// - sad        = 0.35·minor + 0.25·slow + 0.20·darkness + 0.20·(1-energy)
#[allow(clippy::too_many_arguments)]
pub fn mood_scores(
    key_result: Option<&KeyResult>,
    bpm: Float,
    rms_mean: Float,
    spectral_centroid_mean: Float,
    onset_density: Float,
    spectral_bandwidth_mean: Float,
    beats: &[usize],
    dissonance: Option<Float>,
    dynamic_range_db: Float,
) -> MoodScores {
    let energy_val = energy(
        rms_mean,
        spectral_centroid_mean,
        onset_density,
        spectral_bandwidth_mean,
    );
    let dance_val = danceability_heuristic(bpm, beats, onset_density);

    let (mode_major, mode_minor) = match key_result {
        Some(kr) if kr.confidence >= 0.05 => {
            if kr.mode == "major" {
                (1.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        _ => (0.5, 0.5),
    };

    let tempo = ((bpm - 60.0) / 120.0).clamp(0.0, 1.0);
    let slow = 1.0 - tempo;
    let brightness = ((spectral_centroid_mean - 1000.0) / 3000.0).clamp(0.0, 1.0);
    let darkness = 1.0 - brightness;
    let onset = (onset_density / 8.0).clamp(0.0, 1.0);
    let low_onset = 1.0 - onset;
    let diss = dissonance.map(|d| d.clamp(0.0, 1.0)).unwrap_or(0.0);
    let narrow_dyn = (1.0 - (dynamic_range_db / 20.0)).clamp(0.0, 1.0);

    let happy =
        (0.35 * mode_major + 0.25 * tempo + 0.20 * brightness + 0.20 * dance_val).clamp(0.0, 1.0);
    let aggressive =
        (0.35 * energy_val + 0.30 * onset + 0.20 * diss + 0.15 * mode_minor).clamp(0.0, 1.0);
    let relaxed = (0.30 * (1.0 - energy_val) + 0.25 * slow + 0.25 * low_onset + 0.20 * narrow_dyn)
        .clamp(0.0, 1.0);
    let sad = (0.35 * mode_minor + 0.25 * slow + 0.20 * darkness + 0.20 * (1.0 - energy_val))
        .clamp(0.0, 1.0);

    MoodScores {
        happy,
        aggressive,
        relaxed,
        sad,
    }
}
