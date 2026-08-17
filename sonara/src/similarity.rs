//! Hand-crafted similarity / embedding vector for nearest-neighbor search.
//!
//! This is the *classical* (non-ML) similarity layer. It assembles a fixed,
//! 48-dimensional feature vector from measurements the playlist pipeline already
//! computes (MFCC timbre, chroma harmony, spectral shape, rhythm, dynamics, and
//! tonal descriptors), applies **fixed, documented per-dimension normalization**,
//! and exposes a bounded distance / similarity metric. No ML dependency.
//!
//! ## Design goals
//!
//! - **Cross-track & cross-run comparability.** Every dimension is normalized
//!   with *constant* bounds/scales chosen from typical music ranges — never with
//!   per-track or per-batch adaptive statistics. Two vectors computed on
//!   different days, on different machines, in different library runs are
//!   directly comparable.
//! - **Bounded & finite.** Every dimension is squashed into `[0, 1]` (via a
//!   clamped linear map or a `tanh` sigmoid). Even for degenerate inputs
//!   (silence, white noise, pure sine, NaN-producing edge cases) the vector is
//!   finite and in range, so distances never blow up.
//! - **Future-proofing for ML.** The vector sits behind [`SIMILARITY_VERSION`].
//!   A learned (e.g. ONNX) embedding can later replace this hand-crafted layer:
//!   bump the version, keep the same `embedding` / `embedding_version` fields and
//!   the same [`distance`] / [`similarity`] API.
//!
//! ## Versioning
//!
//! [`SIMILARITY_VERSION`] identifies the exact layout + normalization constants.
//! **Bump rule:** ANY change that alters the meaning of a stored vector — adding,
//! removing or reordering a dimension, changing a normalization constant, or
//! changing the **default profile's** weight table ([`WEIGHTS`]) used by
//! [`distance`] — REQUIRES incrementing [`SIMILARITY_VERSION`]. Vectors
//! carrying different versions are not comparable and callers should refuse to
//! compare them.
//!
//! Weight tables for *named non-default profiles* (see [`SimilarityProfile`])
//! are versioned independently: changing [`WEIGHTS_TIMBRE`] bumps
//! [`SIMILARITY_PROFILE_TIMBRE_VERSION`] only, never [`SIMILARITY_VERSION`] —
//! stored vectors stay valid because profiles are applied at distance time.
//!
//! ## Distance metric
//!
//! We use a **weighted, normalized Euclidean (L2) distance**, not cosine.
//! Rationale: all dimensions are non-negative and live in `[0, 1]`, so cosine
//! similarity is biased toward 1 (vectors with no negative components are never
//! near-orthogonal), which compresses the useful dynamic range. Euclidean
//! distance on bounded dimensions is more discriminative and directly
//! interpretable. Per-dimension weights (see [`WEIGHTS`]) let perceptually
//! salient blocks (tempo, timbre, harmony) count for more than incidental ones
//! (absolute loudness). The weighted distance is normalized by the total weight,
//! so — because each dimension difference is in `[0, 1]` — the result is itself
//! in `[0, 1]`: `0.0` = identical, `1.0` = maximally far.
//!
//! Raw distances between real tracks occupy a narrow band: measured over a
//! 9,400-track commercial library, random pairs span ~0.08-0.27 with a median
//! of ~0.19, and same-artist neighbors sit near ~0.13. [`similarity`] therefore
//! applies a calibrated linear stretch, `1 - distance / SIMILARITY_SCALE`
//! (clamped to `[0, 1]`), chosen so a median random pair scores ~0.5, close
//! neighbors score 0.65+, and identical tracks score exactly 1.0. The stretch
//! is monotone in the raw distance, so nearest-neighbor rankings are identical
//! whether you sort by [`distance`] or by [`similarity`].

use crate::analyze::TrackAnalysis;
use crate::types::Float;

/// Version of the embedding layout + normalization constants.
///
/// See the module docs for the bump rule. Exposed alongside every vector as
/// `TrackAnalysis::embedding_version` so stored vectors can be validated before
/// comparison.
///
/// v2 (2026-07-17): the chroma input feeding embedding dims `[13..25]` changed
/// when the chroma filterbank gained librosa-parity octave-domain weighting,
/// so vectors built before this date are not comparable to v2 vectors.
pub const SIMILARITY_VERSION: u32 = 2;

/// Calibrated stretch for [`similarity`]: a raw [`distance`] of this magnitude
/// (or more) maps to similarity `0.0`. Chosen as 2x the median random-pair
/// distance (~0.19) measured over a large commercial music library, so an
/// unrelated pair scores ~0.5 and true neighbors score noticeably higher.
/// Changing this rescales `similarity` output — bump [`SIMILARITY_VERSION`]
/// if stored similarity *scores* (not vectors) must stay comparable.
pub const SIMILARITY_SCALE: Float = 0.38;

/// Fixed embedding dimensionality. Do not change without bumping
/// [`SIMILARITY_VERSION`].
pub const EMBEDDING_DIM: usize = 48;

// ============================================================
// Per-dimension weights (used by `distance`)
// ============================================================

/// Per-dimension weights for the weighted L2 distance.
///
/// Higher weight = the dimension contributes more to the distance. Chosen so
/// that timbre (MFCC), harmony (chroma / key) and tempo dominate, while
/// gain-dependent dimensions (absolute loudness) are down-weighted so that the
/// same track at a different gain still reads as highly similar.
///
/// Layout mirrors [`embed`] exactly (see the block comments there):
/// - `[0..13]`   MFCC timbre        (13)
/// - `[13..25]`  chroma harmony     (12)
/// - `[25..31]`  spectral contrast  (6)
/// - `[31..35]`  spectral scalars   (4)
/// - `[35..39]`  rhythm             (4)
/// - `[39..41]`  dynamics           (2)
/// - `[41..46]`  tonal              (5)
/// - `[46..48]`  perceptual         (2)
///
/// This is the **default profile's** table: changing any weight here REQUIRES
/// bumping [`SIMILARITY_VERSION`] (the default profile's version aliases it).
/// Non-default profile tables (e.g. [`WEIGHTS_TIMBRE`]) are governed by their
/// own version constants instead — see the module docs.
pub const WEIGHTS: [Float; EMBEDDING_DIM] = [
    // MFCC timbre (0..13): coeff 0 (log-energy-ish) low weight, shape coeffs high
    0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6,
    // Chroma harmony (13..25)
    0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
    // Spectral contrast bands (25..31)
    0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
    // Spectral scalars: centroid, bandwidth, rolloff, flatness (31..35)
    1.0, 0.8, 0.8, 0.8,
    // Rhythm: bpm_fold, onset_density, danceability, grid_regularity (35..39)
    2.0, 1.0, 1.0, 0.6, // Dynamics: lufs (gain-dependent, low), dynamic_range (39..41)
    0.3, 0.6,
    // Tonal: dissonance, chord_change_rate, key_cof_sin, key_cof_cos, key_mode (41..46)
    0.6, 0.6, 1.0, 1.0, 0.5, // Perceptual: energy, valence (46..48)
    1.0, 0.7,
];

/// Version of the **default** profile's weight table. Aliases
/// [`SIMILARITY_VERSION`]: the default profile is the historical metric, so its
/// weights are part of what `SIMILARITY_VERSION` has always identified, and a
/// change to [`WEIGHTS`] bumps `SIMILARITY_VERSION` itself. Profile versions
/// are otherwise independent — a change to a non-default profile's table bumps
/// THAT profile's version only (stored vectors are unaffected because profile
/// weights are applied at distance time, not at embed time).
pub const SIMILARITY_PROFILE_DEFAULT_VERSION: u32 = SIMILARITY_VERSION;

/// Version of the timbre profile's weight table ([`WEIGHTS_TIMBRE`]). Bump on
/// any change to that table. Independent of [`SIMILARITY_VERSION`]: timbre
/// weights only exist at distance time, so stored vectors never carry them.
pub const SIMILARITY_PROFILE_TIMBRE_VERSION: u32 = 1;

/// Per-dimension weights for the **timbre** profile — same layout as
/// [`WEIGHTS`], different emphasis.
///
/// Motivation: with the default table the largest single weight is `bpm_fold`
/// (2.0) and the vibe scalars (energy, valence, dynamics) carry full weight, so
/// nearest neighbors are structure-coherent (same tempo/energy) but can mix
/// style character across genres. This profile makes *what the track sounds
/// like* — spectral texture — dominate the metric, so neighbors share sonic
/// style even when tempo or energy differ.
///
/// Per-block rationale (v1 — an interpretable first cut, validated by the
/// labeled neighbor-quality gate; retune there, not by feel):
/// - **MFCC timbre `[0..13]`** — the primary style-character signal; raised to
///   dominate. c0 stays low (0.6): it tracks overall log-energy, which is
///   gain-adjacent, not texture. Low-order shape coeffs (c1-c5, broad spectral
///   envelope) get 2.0; mid (c6-c8) 1.6; high-order (c9-c12, fine/noisy
///   detail) 1.2.
/// - **Chroma harmony `[13..25]`** — kept moderate (0.5): harmony correlates
///   with style but is what the default profile already captures well; it must
///   not outvote texture here.
/// - **Spectral contrast `[25..31]`** — raised to 1.6: peak-to-valley band
///   structure separates dense/compressed textures from sparse/acoustic ones,
///   exactly the "vibe-mixed neighbor" failure reported.
/// - **Spectral scalars `[31..35]`** — brightness/spread/rolloff/flatness are
///   timbre-adjacent; kept near default (1.2, 1.0, 1.0, 1.0) so gross spectral
///   placement still counts.
/// - **Rhythm `[35..39]`** — demoted hard (bpm_fold 2.0 → 0.3; onset density
///   0.4; danceability 0.3; grid regularity 0.3): tempo/groove is the main
///   cross-genre puller in the default metric.
/// - **Dynamics `[39..41]`** — lufs 0.1 (gain-dependent), dynamic range 0.3.
/// - **Tonal `[41..46]`** — dissonance keeps 0.6 (it is a texture quality);
///   chord-change rate 0.4; key circle-of-fifths 0.4/0.4 and mode 0.3 (key is
///   irrelevant to sonic character).
/// - **Perceptual `[46..48]`** — energy 0.3, valence 0.2: these are the
///   vibe summaries whose dominance the profile exists to remove.
///
/// Resulting mass: timbre blocks (MFCC + contrast + spectral scalars) carry
/// ~77% of the total weight vs ~48% in the default table.
pub const WEIGHTS_TIMBRE: [Float; EMBEDDING_DIM] = [
    // MFCC timbre (0..13): c0 low (log-energy-ish), shape coeffs dominant
    0.6, 2.0, 2.0, 2.0, 2.0, 2.0, 1.6, 1.6, 1.6, 1.2, 1.2, 1.2, 1.2,
    // Chroma harmony (13..25): moderate
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    // Spectral contrast bands (25..31): raised
    1.6, 1.6, 1.6, 1.6, 1.6, 1.6,
    // Spectral scalars: centroid, bandwidth, rolloff, flatness (31..35)
    1.2, 1.0, 1.0, 1.0,
    // Rhythm: bpm_fold, onset_density, danceability, grid_regularity (35..39)
    0.3, 0.4, 0.3, 0.3, // Dynamics: lufs, dynamic_range (39..41)
    0.1, 0.3,
    // Tonal: dissonance, chord_change_rate, key_cof_sin, key_cof_cos, key_mode (41..46)
    0.6, 0.4, 0.4, 0.4, 0.3, // Perceptual: energy, valence (46..48)
    0.3, 0.2,
];

/// A named per-dimension weighting applied by [`distance_with_profile`] /
/// [`similarity_with_profile`] at comparison time.
///
/// Profiles never change the stored vector — the same
/// [`SIMILARITY_VERSION`]-stamped embedding serves every profile — they only
/// change which weight table the distance uses. Each profile's table carries
/// its own version constant (see the module docs for the bump rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimilarityProfile {
    /// The historical metric: [`WEIGHTS`], balanced across timbre, harmony and
    /// rhythm with tempo weighted highest.
    #[default]
    Default,
    /// Timbre-dominant metric: [`WEIGHTS_TIMBRE`], spectral texture dominates
    /// while tempo/energy/valence are demoted.
    Timbre,
}

impl SimilarityProfile {
    /// Parse a profile name (case-insensitive): `"default"` / `"timbre"`.
    /// Unknown names return `None` — callers must reject them explicitly, never
    /// fall back silently.
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("default") {
            Some(Self::Default)
        } else if name.eq_ignore_ascii_case("timbre") {
            Some(Self::Timbre)
        } else {
            None
        }
    }

    /// The canonical (lowercase) profile name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Timbre => "timbre",
        }
    }

    /// The version of this profile's weight table.
    pub fn version(self) -> u32 {
        match self {
            Self::Default => SIMILARITY_PROFILE_DEFAULT_VERSION,
            Self::Timbre => SIMILARITY_PROFILE_TIMBRE_VERSION,
        }
    }

    /// This profile's weight table.
    pub fn weights(self) -> &'static [Float; EMBEDDING_DIM] {
        match self {
            Self::Default => &WEIGHTS,
            Self::Timbre => &WEIGHTS_TIMBRE,
        }
    }

    /// All selectable profiles, in canonical order.
    pub const ALL: [SimilarityProfile; 2] = [Self::Default, Self::Timbre];
}

// ============================================================
// Normalization helpers — all outputs guaranteed finite & in [0, 1]
// ============================================================

/// Replace non-finite input with a neutral default (keeps vectors well-defined
/// for degenerate signals).
#[inline]
fn finite_or(x: Float, default: Float) -> Float {
    if x.is_finite() {
        x
    } else {
        default
    }
}

/// Clamped linear map of `x` from `[lo, hi]` onto `[0, 1]`.
#[inline]
fn lin01(x: Float, lo: Float, hi: Float) -> Float {
    finite_or(((x - lo) / (hi - lo)).clamp(0.0, 1.0), 0.0)
}

/// Smooth sigmoid squash onto `[0, 1]`, centered at `center` with half-width
/// `scale`. Robust for unbounded inputs (e.g. raw MFCC coefficients): always in
/// `(0, 1)`, saturates gracefully at extremes.
#[inline]
fn tanh01(x: Float, center: Float, scale: Float) -> Float {
    let z = (finite_or(x, center) - center) / scale;
    0.5 + 0.5 * z.tanh()
}

/// Fold a BPM into the `[60, 180)` octave, then log-normalize to `[0, 1)`.
///
/// Tempo is octave-ambiguous (a 75 BPM detection and its 150 BPM double describe
/// the same groove), so we fold onto a single octave before comparing. Within
/// the octave we use a *log* scale (perceived tempo distance is multiplicative):
/// `log2(bpm / 60) / log2(180 / 60)`.
fn fold_bpm(bpm: Float) -> Float {
    if !bpm.is_finite() || bpm <= 0.0 {
        return 0.0;
    }
    let mut b = bpm;
    while b < 60.0 {
        b *= 2.0;
    }
    while b >= 180.0 {
        b /= 2.0;
    }
    // b now in [60, 180)
    (b / 60.0).log2() / 3.0_f32.log2()
}

/// Beat-grid regularity in `[0, 1]` from beat frame positions.
///
/// `1 - CV` of inter-beat intervals (coefficient of variation). Steady grids
/// (drum machine) approach 1.0; irregular/rubato playing approaches 0.0. Cheap:
/// a single pass over the beat list we already have.
fn grid_regularity(beats: &[usize]) -> Float {
    if beats.len() < 3 {
        return 0.0;
    }
    let intervals: Vec<Float> = beats.windows(2).map(|w| (w[1] - w[0]) as Float).collect();
    let mean = intervals.iter().sum::<Float>() / intervals.len() as Float;
    if mean <= 0.0 {
        return 0.0;
    }
    let var =
        intervals.iter().map(|&i| (i - mean).powi(2)).sum::<Float>() / intervals.len() as Float;
    let cv = var.sqrt() / mean;
    (1.0 - cv).clamp(0.0, 1.0)
}

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Parse a key string ("C major", "A minor", "F# minor") into (pitch_class,
/// is_major). Returns `None` if unparseable.
fn parse_key(key: &str) -> Option<(usize, bool)> {
    let mut it = key.split_whitespace();
    let root = it.next()?;
    let pc = NOTE_NAMES.iter().position(|&n| n == root)?;
    let is_major = !it
        .next()
        .map(|m| m.eq_ignore_ascii_case("minor"))
        .unwrap_or(false);
    Some((pc, is_major))
}

/// Encode a pitch class on the circle of fifths as (sin, cos), each mapped to
/// `[0, 1]`. Adjacent keys on the circle of fifths (a perfect fifth apart, e.g.
/// C and G) are close in this 2-D encoding, so harmonically related tracks land
/// near each other. `pos = (pc * 7) mod 12` walks the circle of fifths.
fn key_circle_of_fifths(pitch_class: usize) -> (Float, Float) {
    let pos = (pitch_class * 7) % 12;
    let ang = 2.0 * std::f32::consts::PI * pos as Float / 12.0;
    (0.5 + 0.5 * ang.sin(), 0.5 + 0.5 * ang.cos())
}

// ============================================================
// Embedding construction
// ============================================================

/// Build the fixed [`EMBEDDING_DIM`]-dimensional similarity vector from a
/// completed [`TrackAnalysis`].
///
/// The analysis must have been produced with the underlying (playlist-level)
/// features available — MFCC, chroma, spectral contrast, key, energy, valence,
/// danceability, dissonance and chord change rate. When a field is absent the
/// dimension falls back to a documented neutral value (so the vector is always
/// full-length and finite), but for meaningful similarity request the
/// `"embedding"` feature, which pulls in every dependency automatically.
///
/// Every entry is finite and in `[0, 1]`. See inline comments for each
/// dimension's normalization constant.
pub fn embed(a: &TrackAnalysis) -> Vec<Float> {
    let mut v = Vec::with_capacity(EMBEDDING_DIM);

    // ---- MFCC timbre (dims 0..13) ----
    // Coefficient 0 tracks overall log-energy (large negative for typical music,
    // since it is a DCT of dB-scaled mel energies); higher coeffs encode timbre
    // shape and cluster near 0. tanh keeps unbounded coeffs in range.
    let mfcc = a.mfcc_mean.as_deref().unwrap_or(&[]);
    // c0: typical music ~ -250 .. -50; center -180, half-width 120.
    v.push(tanh01(
        mfcc.first().copied().unwrap_or(-180.0),
        -180.0,
        120.0,
    ));
    // c1..c12: shape coeffs, roughly -40..40; center 0, half-width 30.
    for k in 1..13 {
        v.push(tanh01(mfcc.get(k).copied().unwrap_or(0.0), 0.0, 30.0));
    }

    // ---- Chroma harmony (dims 13..25) ----
    // chroma_mean is already per-frame L-inf-normalized then averaged, so each
    // pitch class is a relative strength already in ~[0, 1]. Clamp defensively.
    let chroma = a.chroma_mean.as_deref().unwrap_or(&[]);
    for c in 0..12 {
        v.push(
            chroma
                .get(c)
                .copied()
                .map(|x| finite_or(x, 0.0).clamp(0.0, 1.0))
                .unwrap_or(0.0),
        );
    }

    // ---- Spectral contrast bands (dims 25..31) ----
    // spectral_contrast_mean is [6 bands + 1 mean-magnitude]; we take the 6
    // per-band peak-to-valley log10 ratios. Typical range 0..5 dB-decades.
    let contrast = a.spectral_contrast_mean.as_deref().unwrap_or(&[]);
    for b in 0..6 {
        v.push(lin01(contrast.get(b).copied().unwrap_or(0.0), 0.0, 5.0));
    }

    // ---- Spectral scalars (dims 31..35) ----
    // Centroid (brightness): music ~500..5000 Hz.
    v.push(lin01(a.spectral_centroid_mean, 500.0, 5000.0));
    // Bandwidth (spread): ~500..4000 Hz.
    v.push(lin01(
        a.spectral_bandwidth_mean.unwrap_or(0.0),
        500.0,
        4000.0,
    ));
    // Rolloff (85% energy freq): ~0..8000 Hz.
    v.push(lin01(a.spectral_rolloff_mean.unwrap_or(0.0), 0.0, 8000.0));
    // Flatness (tonal 0 .. noise-like): music ~0..0.25.
    v.push(lin01(a.spectral_flatness_mean.unwrap_or(0.0), 0.0, 0.25));

    // ---- Rhythm (dims 35..39) ----
    // Tempo: octave-folded log scale (see fold_bpm).
    v.push(fold_bpm(a.bpm));
    // Onset density: 0..12 onsets/sec covers most music.
    v.push(lin01(a.onset_density, 0.0, 12.0));
    // Danceability (already 0..1); neutral 0.5 if not computed.
    v.push(finite_or(a.danceability.unwrap_or(0.5), 0.5).clamp(0.0, 1.0));
    // Grid regularity from beats (cheap, always available).
    v.push(grid_regularity(&a.beats));

    // ---- Dynamics (dims 39..41) ----
    // Integrated loudness: music ~-40..0 LUFS. GAIN-DEPENDENT — down-weighted.
    v.push(lin01(a.loudness_lufs, -40.0, 0.0));
    // Dynamic range (p95-p5 RMS, dB): ~0..30 dB.
    v.push(lin01(a.dynamic_range_db, 0.0, 30.0));

    // ---- Tonal (dims 41..46) ----
    // Dissonance already 0..1; neutral 0 if not computed.
    v.push(finite_or(a.dissonance.unwrap_or(0.0), 0.0).clamp(0.0, 1.0));
    // Chord change rate: 0..4 changes/sec spans still .. busy.
    v.push(lin01(a.chord_change_rate.unwrap_or(0.0), 0.0, 4.0));
    // Key on circle of fifths (2-D) so harmonic neighbors are close; neutral
    // (0.5, 0.5) center when key is unknown/unparseable.
    let (cof_sin, cof_cos, is_major) = a
        .key
        .as_deref()
        .and_then(parse_key)
        .map(|(pc, maj)| {
            let (s, c) = key_circle_of_fifths(pc);
            (s, c, maj)
        })
        .unwrap_or((0.5, 0.5, true));
    v.push(cof_sin);
    v.push(cof_cos);
    // Mode: major = 1.0, minor = 0.0.
    v.push(if is_major { 1.0 } else { 0.0 });

    // ---- Perceptual (dims 46..48) ----
    // Energy & valence are compact 0..1 summaries; neutral 0.5 if absent.
    v.push(finite_or(a.energy.unwrap_or(0.5), 0.5).clamp(0.0, 1.0));
    v.push(finite_or(a.valence.unwrap_or(0.5), 0.5).clamp(0.0, 1.0));

    debug_assert_eq!(v.len(), EMBEDDING_DIM);
    v
}

// ============================================================
// Distance / similarity
// ============================================================

/// Weighted, normalized Euclidean distance between two embedding vectors.
///
/// Returns a value in `[0, 1]`: `0.0` = identical, `1.0` = maximally far. If the
/// vectors are equal length and of the canonical [`EMBEDDING_DIM`] length, the
/// per-dimension [`WEIGHTS`] are applied; otherwise a uniform weighting over the
/// shared prefix is used (so mismatched/legacy lengths degrade gracefully rather
/// than panicking). Empty / fully-mismatched input yields `1.0`.
pub fn distance(a: &[Float], b: &[Float]) -> Float {
    distance_with_profile(a, b, SimilarityProfile::Default)
}

/// [`distance`] under a named weighting profile.
///
/// Identical structure to [`distance`] — a weighted, normalized Euclidean
/// distance in `[0, 1]` — with the profile's weight table in place of the
/// default [`WEIGHTS`]. `SimilarityProfile::Default` is bit-identical to
/// [`distance`]. The uniform-weight fallback for non-canonical vector lengths
/// applies to every profile (it degrades on *length*, never on profile name —
/// unknown profile names must be rejected before reaching this function).
pub fn distance_with_profile(a: &[Float], b: &[Float], profile: SimilarityProfile) -> Float {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let use_weights = a.len() == EMBEDDING_DIM && b.len() == EMBEDDING_DIM;
    let weights = profile.weights();

    let mut wsum = 0.0f32;
    let mut acc = 0.0f32;
    for i in 0..n {
        let w = if use_weights { weights[i] } else { 1.0 };
        let d = finite_or(a[i], 0.0) - finite_or(b[i], 0.0);
        acc += w * d * d;
        wsum += w;
    }
    if wsum <= 0.0 {
        return 1.0;
    }
    // Each |a_i - b_i| <= 1 (vectors are in [0,1]), so acc/wsum <= 1 and the
    // square root is in [0, 1].
    (acc / wsum).clamp(0.0, 1.0).sqrt()
}

/// Similarity in `[0, 1]`, higher = more similar. Identical vectors → `1.0`.
///
/// Applies the calibrated stretch `1 - distance / SIMILARITY_SCALE` (clamped)
/// so scores spread usefully over real music instead of clustering near 0.85
/// (see the module docs). Monotone in [`distance`], so rankings are unchanged.
pub fn similarity(a: &[Float], b: &[Float]) -> Float {
    similarity_with_profile(a, b, SimilarityProfile::Default)
}

/// [`similarity`] under a named weighting profile.
///
/// Applies the same calibrated stretch as [`similarity`],
/// `1 - distance / SIMILARITY_SCALE` (clamped to `[0, 1]`). The distance is
/// weight-normalized under every profile, so it stays in `[0, 1]` regardless
/// of a table's total mass and [`SIMILARITY_SCALE`] is reused as-is for the
/// timbre profile; if the labeled neighbor-quality gate shows the timbre
/// distance distribution sits far from the default's ~0.19 random-pair median,
/// a per-profile scale is the follow-up — bumping that profile's version, not
/// [`SIMILARITY_VERSION`]. Monotone in [`distance_with_profile`], so rankings
/// under a profile are the same whether sorted by distance or similarity.
pub fn similarity_with_profile(a: &[Float], b: &[Float], profile: SimilarityProfile) -> Float {
    (1.0 - distance_with_profile(a, b, profile) / SIMILARITY_SCALE).clamp(0.0, 1.0)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::{analyze_signal, AnalysisConfig, AnalysisMode};
    use ndarray::Array1;
    use std::collections::HashSet;
    use std::f32::consts::PI;

    fn embed_config() -> AnalysisConfig {
        let feats: HashSet<String> = ["embedding"].iter().map(|s| s.to_string()).collect();
        AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: Some(feats),
            ..AnalysisConfig::default()
        }
    }

    fn sine(freq: Float, sr: u32, dur: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin())
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

    /// Kick-drum-like pattern at a given BPM.
    fn kick_pattern(bpm: Float, sr: u32, dur: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        let interval = (60.0 / bpm * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..(sr as usize / 20).min(n - pos) {
                let t = i as Float / sr as Float;
                // decaying low-frequency thump
                y[pos + i] = (2.0 * PI * 60.0 * t).sin() * (-30.0 * t).exp();
            }
            pos += interval;
        }
        y
    }

    fn embed_signal(y: &Array1<Float>, sr: u32) -> Vec<Float> {
        let r = analyze_signal(y.view(), sr, &embed_config()).unwrap();
        r.embedding
            .clone()
            .expect("embedding populated when requested")
    }

    #[test]
    fn test_version_exposed() {
        assert_eq!(SIMILARITY_VERSION, 2);
        assert_eq!(EMBEDDING_DIM, 48);
        assert_eq!(WEIGHTS.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_dimension_exact() {
        let y = sine(440.0, 22050, 3.0);
        let v = embed_signal(&y, 22050);
        assert_eq!(
            v.len(),
            EMBEDDING_DIM,
            "vector must be exactly {} dims",
            EMBEDDING_DIM
        );
    }

    #[test]
    fn test_bounds_extreme_inputs() {
        let sr = 22050u32;
        let cases = vec![
            ("silence", Array1::<Float>::zeros(3 * sr as usize)),
            ("white_noise", white_noise(3 * sr as usize)),
            ("pure_sine", sine(440.0, sr, 3.0)),
            ("loud_sine", sine(440.0, sr, 3.0).mapv(|x| x * 1000.0)),
            ("dc_offset", Array1::<Float>::ones(3 * sr as usize)),
        ];
        for (name, y) in cases {
            // Some degenerate signals (pure silence) may error in analysis; skip
            // those but assert bounds whenever a vector is produced.
            if let Ok(r) = analyze_signal(y.view(), sr, &embed_config()) {
                if let Some(v) = r.embedding {
                    assert_eq!(v.len(), EMBEDDING_DIM, "{name}: dim");
                    for (i, &x) in v.iter().enumerate() {
                        assert!(x.is_finite(), "{name}: dim {i} not finite: {x}");
                        assert!(
                            (0.0..=1.0).contains(&x),
                            "{name}: dim {i} out of [0,1]: {x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_identical_signal_zero_distance() {
        let y = sine(440.0, 22050, 3.0);
        let v = embed_signal(&y, 22050);
        assert_eq!(distance(&v, &v), 0.0, "identical → distance 0");
        assert_eq!(similarity(&v, &v), 1.0, "identical → similarity 1");
    }

    #[test]
    fn test_kick_tempo_neighbors_closer_than_noise() {
        let sr = 22050u32;
        let k120 = embed_signal(&kick_pattern(120.0, sr, 6.0), sr);
        let k125 = embed_signal(&kick_pattern(125.0, sr, 6.0), sr);
        let noise = embed_signal(&white_noise(6 * sr as usize), sr);

        let d_kick = distance(&k120, &k125);
        let d_noise = distance(&k120, &noise);
        assert!(
            d_kick < d_noise,
            "120bpm kick should be closer to 125bpm kick ({d_kick}) than to white noise ({d_noise})"
        );
    }

    #[test]
    fn test_gain_invariance_high_similarity() {
        let sr = 22050u32;
        let y = kick_pattern(120.0, sr, 6.0) + sine(220.0, sr, 6.0).mapv(|x| x * 0.3);
        let full = embed_signal(&y, sr);
        let half = embed_signal(&y.mapv(|x| x * 0.5), sr);
        let sim = similarity(&full, &half);
        // Loudness dims differ, but the bulk (timbre, harmony, tempo, key) is
        // gain-invariant, so overall similarity stays high.
        assert!(sim > 0.7, "0.5x gain should stay highly similar, got {sim}");
    }

    #[test]
    fn test_distance_symmetric_and_bounded() {
        let sr = 22050u32;
        let a = embed_signal(&sine(440.0, sr, 3.0), sr);
        let b = embed_signal(&white_noise(3 * sr as usize), sr);
        let dab = distance(&a, &b);
        let dba = distance(&b, &a);
        assert!((dab - dba).abs() < 1e-6, "distance must be symmetric");
        assert!((0.0..=1.0).contains(&dab), "distance in [0,1], got {dab}");
        assert!((0.0..=1.0).contains(&similarity(&a, &b)));
    }

    // ---- Similarity profiles ----

    /// Deterministic synthetic 48-dim vectors (no audio needed).
    fn pin_vec_a() -> Vec<Float> {
        (0..EMBEDDING_DIM)
            .map(|i| ((i * 37) % 97) as Float / 96.0)
            .collect()
    }

    fn pin_vec_b() -> Vec<Float> {
        (0..EMBEDDING_DIM)
            .map(|i| ((i * 53 + 11) % 89) as Float / 88.0)
            .collect()
    }

    /// pin_vec_a with a small deterministic perturbation (a "close" pair, so
    /// the similarity pin is not just exercising the clamp at 0).
    fn pin_vec_a_close() -> Vec<Float> {
        (0..EMBEDDING_DIM)
            .map(|i| (((i * 37) % 97) as Float / 96.0 + ((i % 5) as Float) * 0.03).min(1.0))
            .collect()
    }

    #[test]
    fn test_profile_versions_pinned() {
        assert_eq!(SIMILARITY_PROFILE_DEFAULT_VERSION, SIMILARITY_VERSION);
        assert_eq!(SIMILARITY_PROFILE_DEFAULT_VERSION, 2);
        assert_eq!(SIMILARITY_PROFILE_TIMBRE_VERSION, 1);
        assert_eq!(WEIGHTS_TIMBRE.len(), EMBEDDING_DIM);
        assert_eq!(SimilarityProfile::Default.version(), 2);
        assert_eq!(SimilarityProfile::Timbre.version(), 1);
    }

    /// No-movement proof: the default profile (and the plain `distance` /
    /// `similarity` functions that delegate to it) must reproduce the exact
    /// pre-profile results. Expected bits were computed from the code BEFORE
    /// `SimilarityProfile` existed (commit e723241) on these same vectors.
    #[test]
    fn test_default_profile_bit_identical_to_pre_profile_code() {
        let a = pin_vec_a();
        let b = pin_vec_b();
        let ac = pin_vec_a_close();
        // Pre-change pins: d(a,b)=0.44799155, s(a,b)=0.0 (clamped),
        // d(a,ac)=0.06737465, s(a,ac)=0.8226983.
        assert_eq!(distance(&a, &b).to_bits(), 0x3ee55f26);
        assert_eq!(similarity(&a, &b).to_bits(), 0x00000000);
        assert_eq!(distance(&a, &ac).to_bits(), 0x3d89fbb8);
        assert_eq!(similarity(&a, &ac).to_bits(), 0x3f529c5b);
        // The explicit-profile route is the same computation.
        assert_eq!(
            distance_with_profile(&a, &b, SimilarityProfile::Default).to_bits(),
            distance(&a, &b).to_bits()
        );
        assert_eq!(
            similarity_with_profile(&a, &ac, SimilarityProfile::Default).to_bits(),
            similarity(&a, &ac).to_bits()
        );
    }

    /// Designed triple: B shares A's timbre dims (MFCC + contrast) but moves
    /// the rhythm/dynamics/perceptual "vibe" dims; C is the reverse. The
    /// timbre profile must rank B closer to A than C; the default profile must
    /// rank C closer or equal (it is tempo/energy-dominated).
    #[test]
    fn test_timbre_profile_reorders_designed_triple() {
        let a = vec![0.5; EMBEDDING_DIM];
        // B: timbre identical, vibe dims moved by 0.4
        // (bpm_fold 35, lufs 39, dynamic_range 40, energy 46, valence 47).
        let mut b = a.clone();
        for i in [35, 39, 40, 46, 47] {
            b[i] = 0.9;
        }
        // C: vibe identical, all MFCC [0..13] + contrast [25..31] moved by 0.15.
        let mut c = a.clone();
        for v in c.iter_mut().take(13) {
            *v = 0.65;
        }
        for v in c.iter_mut().take(31).skip(25) {
            *v = 0.65;
        }

        let d_ab_def = distance(&a, &b);
        let d_ac_def = distance(&a, &c);
        let d_ab_tim = distance_with_profile(&a, &b, SimilarityProfile::Timbre);
        let d_ac_tim = distance_with_profile(&a, &c, SimilarityProfile::Timbre);
        assert!(
            d_ac_def <= d_ab_def,
            "default must rank the same-timbre/different-vibe pair farther: d(A,C)={d_ac_def} vs d(A,B)={d_ab_def}"
        );
        assert!(
            d_ab_tim < d_ac_tim,
            "timbre must rank the same-timbre pair closer: d(A,B)={d_ab_tim} vs d(A,C)={d_ac_tim}"
        );
    }

    #[test]
    fn test_profile_from_name() {
        assert_eq!(
            SimilarityProfile::from_name("default"),
            Some(SimilarityProfile::Default)
        );
        assert_eq!(
            SimilarityProfile::from_name("Timbre"),
            Some(SimilarityProfile::Timbre)
        );
        assert_eq!(
            SimilarityProfile::from_name("TIMBRE"),
            Some(SimilarityProfile::Timbre)
        );
        assert_eq!(SimilarityProfile::from_name("acoustic"), None);
        assert_eq!(SimilarityProfile::from_name(""), None);
        for p in SimilarityProfile::ALL {
            assert_eq!(SimilarityProfile::from_name(p.name()), Some(p));
        }
    }

    #[test]
    fn test_timbre_symmetric_self_zero_bounded() {
        let a = pin_vec_a();
        let b = pin_vec_b();
        let dab = distance_with_profile(&a, &b, SimilarityProfile::Timbre);
        let dba = distance_with_profile(&b, &a, SimilarityProfile::Timbre);
        assert_eq!(
            distance_with_profile(&a, &a, SimilarityProfile::Timbre),
            0.0,
            "identical → timbre distance 0"
        );
        assert_eq!(
            similarity_with_profile(&a, &a, SimilarityProfile::Timbre),
            1.0,
            "identical → timbre similarity 1"
        );
        assert!(
            (dab - dba).abs() < 1e-6,
            "timbre distance must be symmetric"
        );
        assert!(
            (0.0..=1.0).contains(&dab),
            "timbre distance in [0,1]: {dab}"
        );
        assert!((0.0..=1.0).contains(&similarity_with_profile(&a, &b, SimilarityProfile::Timbre)));
    }

    #[test]
    fn test_distance_mismatched_length_graceful() {
        // Should not panic; degrades to uniform weighting / max distance.
        assert_eq!(distance(&[], &[]), 1.0);
        let a = vec![0.5; EMBEDDING_DIM];
        let b = vec![0.5; 10];
        let d = distance(&a, &b);
        assert!((0.0..=1.0).contains(&d));
    }
}
