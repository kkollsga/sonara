//! Versioned perceptual aggression ranking.
//!
//! Audio analysis uses a compact fused-evidence model: a FerricML linear rank
//! plus a small random-forest ensemble. The older similarity-embedding
//! scorer remains available explicitly as the legacy v1 API.

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use ferricml::linear_model::LogisticRegression;
use ndarray::ArrayView1;
use sha2::{Digest, Sha256};

use crate::analyze::{self, AnalysisConfig};
use crate::error::{Result, SonaraError};
use crate::similarity::{EMBEDDING_DIM, SIMILARITY_VERSION};
use crate::types::Float;

/// Version of the bundled aggression model.
pub const AGGRESSION_MODEL_VERSION: u32 = 2;

/// Similarity embedding layout consumed by this model.
pub const AGGRESSION_EMBEDDING_VERSION: u32 = 2;

/// Stable identifier for the bundled model artifact.
pub const AGGRESSION_MODEL_ID: &str = "aggression-rank-v2";

/// Stable identifier of the retained 48D embedding scorer.
pub const LEGACY_AGGRESSION_MODEL_ID: &str = "aggression-logistic-v1";

/// Number of fused evidence values consumed by the current rank model.
pub const AGGRESSION_FEATURE_COUNT: usize = 39;

// Retained only to validate the bundled artifact format. Pairwise tie policy is
// deliberately outside Sonara's inference contract.
const ARTIFACT_TIE_BAND: Float = 0.07;

const LEGACY_FEATURE_SCHEMA_SHA256: [u8; 32] = [
    0xc5, 0x35, 0x88, 0x39, 0x3c, 0xeb, 0xf1, 0xd6, 0x09, 0xa3, 0xa6, 0x83, 0xdb, 0x61, 0x4e, 0xc9,
    0x28, 0x32, 0x16, 0x97, 0xbe, 0x15, 0x04, 0xae, 0xd8, 0x0d, 0x48, 0x49, 0x99, 0x37, 0x63, 0x44,
];

// Canonical FerricML v1 logistic artifact. Its envelope binds the estimator
// kind and feature schema and checks the payload checksum before use.
const LEGACY_MODEL_ARTIFACT: &[u8; 300] = &[
    0x46, 0x45, 0x52, 0x52, 0x49, 0x43, 0x4d, 0x4c, 0x01, 0x00, 0x01, 0x00, 0xc5, 0x35, 0x88, 0x39,
    0x3c, 0xeb, 0xf1, 0xd6, 0x09, 0xa3, 0xa6, 0x83, 0xdb, 0x61, 0x4e, 0xc9, 0x28, 0x32, 0x16, 0x97,
    0xbe, 0x15, 0x04, 0xae, 0xd8, 0x0d, 0x48, 0x49, 0x99, 0x37, 0x63, 0x44, 0x30, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x0a, 0xd7, 0x23, 0x3c, 0x64, 0x00, 0x00, 0x00, 0x17, 0xb7, 0xd1, 0x38,
    0x04, 0x00, 0x00, 0x00, 0xfc, 0x5a, 0x10, 0xc0, 0x30, 0x00, 0x00, 0x00, 0x97, 0x7a, 0x13, 0x3f,
    0xb6, 0x5d, 0xbd, 0xbc, 0x7a, 0x90, 0x57, 0xbe, 0xba, 0x31, 0xca, 0x3d, 0xea, 0x42, 0x18, 0xbd,
    0x9b, 0x1c, 0xba, 0x3d, 0x2e, 0x78, 0x65, 0x3d, 0xa0, 0xa9, 0x05, 0x3e, 0xd8, 0xfa, 0x88, 0x3d,
    0x55, 0xb5, 0xe1, 0x3d, 0xbc, 0x7c, 0x95, 0x3d, 0x7b, 0xe0, 0xc5, 0x3d, 0xa2, 0xdc, 0x4a, 0x3d,
    0x98, 0xb0, 0xaf, 0x3d, 0xa9, 0xd5, 0x0f, 0x3e, 0x65, 0xef, 0x05, 0x3e, 0xb3, 0xaf, 0x1a, 0x3e,
    0xdb, 0xe0, 0x08, 0x3e, 0x54, 0x17, 0x0b, 0x3e, 0x4a, 0xc6, 0x23, 0x3e, 0x8b, 0xbd, 0x06, 0x3e,
    0xd3, 0xdd, 0x27, 0x3e, 0x8c, 0xde, 0x19, 0x3e, 0x09, 0x27, 0x2a, 0x3e, 0x59, 0xda, 0x0e, 0x3e,
    0x21, 0xa3, 0xb3, 0xbd, 0x5a, 0xa8, 0xe7, 0xbd, 0xd9, 0x6b, 0xe3, 0xbd, 0x07, 0x33, 0xc6, 0xbd,
    0x5d, 0xfc, 0x7f, 0xbd, 0xa6, 0xd2, 0x39, 0xbd, 0x56, 0x9e, 0x96, 0x3e, 0xea, 0x29, 0x8a, 0x3e,
    0x2e, 0x36, 0xb7, 0x3e, 0x18, 0x3f, 0x0b, 0x3e, 0xf6, 0x18, 0xa5, 0x3c, 0xaf, 0x98, 0xbb, 0xba,
    0x98, 0x3a, 0x0d, 0x3d, 0x84, 0xdf, 0x37, 0xb9, 0xe9, 0x6b, 0x2f, 0x3e, 0xfc, 0x4f, 0x10, 0xbe,
    0xec, 0x17, 0x6f, 0xbb, 0xb1, 0x56, 0xd0, 0x3d, 0xc0, 0xb4, 0x86, 0x3b, 0x36, 0x0e, 0x86, 0xbd,
    0xe8, 0x1d, 0xea, 0xbd, 0xc4, 0xe2, 0x36, 0x3e, 0x39, 0xdf, 0xa4, 0x3d, 0x3f, 0x06, 0xda, 0x3f,
    0x18, 0xfc, 0x84, 0xcd, 0x42, 0xc9, 0xda, 0xa4, 0x61, 0xe9, 0xe9, 0xf2, 0xd3, 0x76, 0x52, 0x9c,
    0xee, 0x79, 0x4e, 0x3d, 0x23, 0x00, 0xa4, 0x2e, 0x33, 0xcd, 0x98, 0x3a,
];

static LEGACY_MODEL: OnceLock<std::result::Result<LogisticRegression, String>> = OnceLock::new();

/// Score a current Sonara embedding with the retained legacy-v1 model.
///
/// The input must contain exactly [`EMBEDDING_DIM`] finite values. Call
/// [`score_versioned`] when reading stored embeddings so layout drift is
/// rejected explicitly.
pub fn score(embedding: &[Float]) -> Result<Float> {
    score_versioned(embedding, SIMILARITY_VERSION)
}

/// Score an embedding after validating its layout version.
pub fn score_versioned(embedding: &[Float], embedding_version: u32) -> Result<Float> {
    if embedding_version != AGGRESSION_EMBEDDING_VERSION || embedding_version != SIMILARITY_VERSION
    {
        return Err(SonaraError::InvalidParameter {
            param: "embedding_version",
            reason: format!(
                "expected version {AGGRESSION_EMBEDDING_VERSION}, got {embedding_version}"
            ),
        });
    }
    if embedding.len() != EMBEDDING_DIM {
        return Err(SonaraError::ShapeMismatch {
            expected: format!("{EMBEDDING_DIM} embedding values"),
            got: format!("{} embedding values", embedding.len()),
        });
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err(SonaraError::InvalidParameter {
            param: "embedding",
            reason: "all values must be finite".into(),
        });
    }

    legacy_model()?
        .predict_positive_proba(embedding)
        .map_err(|error| SonaraError::ModelError(format!("aggression inference failed: {error}")))
}

fn legacy_model() -> Result<&'static LogisticRegression> {
    LEGACY_MODEL
        .get_or_init(|| {
            LogisticRegression::from_artifact(LEGACY_MODEL_ARTIFACT, LEGACY_FEATURE_SCHEMA_SHA256)
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| SonaraError::ModelError(format!("invalid aggression model: {error}")))
}

const RANK_FEATURE_SCHEMA_SHA256: [u8; 32] = [
    0x6a, 0xc0, 0x3f, 0x40, 0x12, 0x6b, 0x70, 0x25, 0xfd, 0x28, 0x90, 0xf8, 0xa8, 0xf4, 0x0c, 0x95,
    0x06, 0x53, 0x11, 0x7d, 0xb4, 0x7a, 0xdf, 0x13, 0xb6, 0xe9, 0xac, 0x37, 0x6d, 0xb4, 0xb7, 0x85,
];
const RANK_ARTIFACT: &[u8] = include_bytes!("aggression_model.bin");
const LINEAR_ARTIFACT: &[u8] = include_bytes!("aggression_linear.ferricml");
const RANK_MAGIC: &[u8; 8] = b"SNRAGGR2";
const RANK_ARTIFACT_VERSION: u32 = 2;
const MIN_CONTENT_SUPPORT: Float = 0.10;
const HARSHNESS_FEATURE_INDEX: usize = 26;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggressionAnalysis {
    /// Bounded perceptual rank. `None` means the audio did not contain enough
    /// supported musical evidence to make a responsible judgment.
    pub score: Option<Float>,
    /// Independent evidence support in `[0, 1]`; this is not rank certainty.
    pub confidence: Float,
    pub forcefulness: Float,
    pub harshness: Float,
    pub tension: Float,
    pub rhythm: Float,
}

/// Fused measurements assembled by `analyze_signal_inner` without a second
/// decode, FFT, onset, or embedding pass.
pub(crate) struct AggressionEvidence {
    pub crest_p50: Float,
    pub crest_p90: Float,
    pub dissonance_p50: Float,
    pub dissonance_p90: Float,
    pub mfcc_0: Float,
    pub mfcc_2: Float,
    pub contrast: [Float; 5],
    pub centroid: Float,
    pub bandwidth: Float,
    pub bpm: Float,
    pub onset_density_embedding: Float,
    pub danceability: Float,
    pub grid_regularity: Float,
    pub dynamic_range_db: Float,
    pub energy: Float,
    pub high_energy_p50: Float,
    pub high_energy_p90: Float,
    pub high_flatness_p50: Float,
    pub high_flux_p90: Float,
    pub onset_density: Float,
    pub onset_interval_cv: Float,
    pub onset_strength_p50: Float,
    pub onset_strength_p90: Float,
    pub rms_dynamic_ratio: Float,
    pub spectral_peak_ratio: Float,
    pub window_force_top2: Float,
    pub window_harshness_top2: Float,
    pub window_impact_persistence: Float,
    pub window_impact_top2: Float,
    pub content_support: Float,
}

impl AggressionEvidence {
    fn components(&self) -> (Float, Float, Float, Float) {
        let crest = unit(self.crest_p50, 20.0);
        let onset_density = unit(self.onset_density, 15.0);
        let onset_strength = unit(self.onset_strength_p50, 2.0);
        let force = ((1.0 - crest) + onset_density + onset_strength) / 3.0;
        let high_energy = unit(self.high_energy_p50, 0.35);
        let high_flatness = unit(self.high_flatness_p50, 1.0);
        let peak_ratio = unit(self.spectral_peak_ratio, 1.0);
        let harshness = (high_energy + high_flatness + (1.0 - peak_ratio)) / 3.0;
        let tension = unit(self.dissonance_p90, 0.15);
        let rhythm = (unit(self.onset_density_embedding, 12.0)
            + self.danceability.clamp(0.0, 1.0)
            + self.grid_regularity.clamp(0.0, 1.0))
            / 3.0;
        (force, harshness, tension, rhythm)
    }

    fn features(&self) -> [Float; AGGRESSION_FEATURE_COUNT] {
        let (force, harshness, tension, _) = self.components();
        let dance = self.danceability.clamp(0.0, 1.0);
        let regularity = self.grid_regularity.clamp(0.0, 1.0);
        [
            unit(self.crest_p50, 20.0),
            unit(self.crest_p90, 20.0),
            unit(self.dissonance_p50, 0.15),
            tension,
            tanh01(self.mfcc_0, -180.0, 120.0),
            tanh01(self.mfcc_2, 0.0, 30.0),
            unit(self.contrast[0], 5.0),
            unit(self.contrast[1], 5.0),
            unit(self.contrast[2], 5.0),
            unit(self.contrast[3], 5.0),
            unit(self.contrast[4], 5.0),
            lin01(self.centroid, 500.0, 5000.0),
            lin01(self.bandwidth, 500.0, 4000.0),
            fold_bpm(self.bpm),
            unit(self.onset_density_embedding, 12.0),
            dance,
            regularity,
            unit(self.dynamic_range_db, 30.0),
            self.energy.clamp(0.0, 1.0),
            unit(self.high_energy_p50, 0.35),
            unit(self.high_energy_p90, 0.35),
            unit(self.high_flatness_p50, 1.0),
            unit(self.high_flux_p90.max(0.0).ln_1p(), 11.0_f32.ln()),
            force,
            force * harshness,
            force * tension,
            harshness,
            dance * regularity * (1.0 - harshness),
            tension,
            unit(self.onset_density, 15.0),
            unit(self.onset_interval_cv, 2.0),
            unit(self.onset_strength_p50, 2.0),
            unit(self.onset_strength_p90, 4.0),
            unit(self.rms_dynamic_ratio.max(1.0).ln(), 10.0_f32.ln()),
            unit(self.spectral_peak_ratio, 1.0),
            self.window_force_top2.clamp(0.0, 1.0),
            self.window_harshness_top2.clamp(0.0, 1.0),
            self.window_impact_persistence.clamp(0.0, 1.0),
            self.window_impact_top2.clamp(0.0, 1.0),
        ]
    }
}

#[inline]
fn unit(value: Float, high: Float) -> Float {
    (value / high).clamp(0.0, 1.0)
}

#[inline]
fn lin01(value: Float, low: Float, high: Float) -> Float {
    ((value - low) / (high - low)).clamp(0.0, 1.0)
}

#[inline]
fn tanh01(value: Float, center: Float, scale: Float) -> Float {
    0.5 + 0.5 * ((value - center) / scale).tanh()
}

fn fold_bpm(mut bpm: Float) -> Float {
    if !bpm.is_finite() || bpm <= 0.0 {
        return 0.0;
    }
    while bpm < 60.0 {
        bpm *= 2.0;
    }
    while bpm >= 180.0 {
        bpm /= 2.0;
    }
    (bpm / 60.0).log2() / 3.0_f32.log2()
}

#[derive(Clone, Copy)]
struct RankNode {
    feature: u8,
    left: u16,
    right: u16,
    threshold: Float,
    value: Float,
    leaf: bool,
}

struct RankModel {
    linear: LogisticRegression,
    center: [Float; AGGRESSION_FEATURE_COUNT],
    calibration_slope: Float,
    calibration_intercept: Float,
    baseline: Float,
    linear_weight: Float,
    tree_weight: Float,
    harshness_correction: Float,
    trees: Vec<Vec<RankNode>>,
}

impl RankModel {
    #[inline]
    fn predict(&self, features: &[Float; AGGRESSION_FEATURE_COUNT]) -> Float {
        let raw = self
            .linear
            .coefficients()
            .iter()
            .zip(features)
            .zip(&self.center)
            .fold(self.linear.intercept(), |sum, ((weight, value), center)| {
                sum + weight * (value - center)
            });
        let linear = sigmoid(self.calibration_slope * raw + self.calibration_intercept);
        let mut tree_score = self.baseline;
        for tree in &self.trees {
            let mut index = 0_usize;
            loop {
                let node = tree[index];
                if node.leaf {
                    tree_score += node.value;
                    break;
                }
                index = if features[node.feature as usize] <= node.threshold {
                    node.left as usize
                } else {
                    node.right as usize
                };
            }
        }
        let blend = self.linear_weight * linear + self.tree_weight * tree_score.clamp(0.0, 1.0);
        (blend + self.harshness_correction * (features[HARSHNESS_FEATURE_INDEX] - 0.5))
            .clamp(0.0, 1.0)
    }
}

#[inline]
fn sigmoid(value: Float) -> Float {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, length: usize) -> std::result::Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| "truncated rank artifact".to_owned())?;
        let value = &self.bytes[self.position..end];
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self) -> std::result::Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> std::result::Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> std::result::Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> std::result::Result<Float, String> {
        let value = Float::from_le_bytes(self.take(4)?.try_into().unwrap());
        value
            .is_finite()
            .then_some(value)
            .ok_or_else(|| "non-finite rank artifact value".to_owned())
    }
}

fn decode_rank_model() -> std::result::Result<RankModel, String> {
    if RANK_ARTIFACT.len() < 32 {
        return Err("truncated rank artifact".to_owned());
    }
    let payload_length = RANK_ARTIFACT.len() - 32;
    let checksum = Sha256::digest(&RANK_ARTIFACT[..payload_length]);
    if checksum[..] != RANK_ARTIFACT[payload_length..] {
        return Err("rank artifact checksum mismatch".to_owned());
    }
    let mut cursor = Cursor {
        bytes: &RANK_ARTIFACT[..payload_length],
        position: 0,
    };
    if cursor.take(8)? != RANK_MAGIC {
        return Err("invalid rank artifact magic".to_owned());
    }
    if cursor.u32()? != RANK_ARTIFACT_VERSION || cursor.u32()? as usize != AGGRESSION_FEATURE_COUNT
    {
        return Err("unsupported rank artifact schema".to_owned());
    }
    let tree_count = cursor.u32()? as usize;
    if tree_count == 0 || tree_count > 128 || cursor.u32()? != AGGRESSION_MODEL_VERSION {
        return Err("invalid rank artifact model metadata".to_owned());
    }
    if cursor.take(32)? != RANK_FEATURE_SCHEMA_SHA256 {
        return Err("rank feature schema mismatch".to_owned());
    }
    let baseline = cursor.f32()?;
    let linear_weight = cursor.f32()?;
    let tree_weight = cursor.f32()?;
    let calibration_slope = cursor.f32()?;
    let calibration_intercept = cursor.f32()?;
    let tie_band = cursor.f32()?;
    let harshness_correction = cursor.f32()?;
    if linear_weight < 0.0
        || tree_weight < 0.0
        || (linear_weight + tree_weight - 1.0).abs() > 1.0e-6
        || calibration_slope <= 0.0
        || tie_band.to_bits() != ARTIFACT_TIE_BAND.to_bits()
        || !(0.0..=0.25).contains(&harshness_correction)
    {
        return Err("invalid rank artifact transforms".to_owned());
    }
    let mut center = [0.0; AGGRESSION_FEATURE_COUNT];
    for value in &mut center {
        *value = cursor.f32()?;
    }
    let mut trees = Vec::with_capacity(tree_count);
    for _ in 0..tree_count {
        let node_count = cursor.u16()? as usize;
        if node_count == 0 || node_count > 31 {
            return Err("invalid rank tree size".to_owned());
        }
        let mut tree = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let feature = cursor.u8()?;
            let leaf = match cursor.u8()? {
                0 => false,
                1 => true,
                _ => return Err("invalid rank node flag".to_owned()),
            };
            tree.push(RankNode {
                feature,
                left: cursor.u16()?,
                right: cursor.u16()?,
                threshold: cursor.f32()?,
                value: cursor.f32()?,
                leaf,
            });
        }
        for (index, node) in tree.iter().enumerate() {
            if node.leaf {
                if node.feature != u8::MAX {
                    return Err("invalid rank leaf".to_owned());
                }
            } else if node.feature as usize >= AGGRESSION_FEATURE_COUNT
                || node.left as usize <= index
                || node.right as usize <= index
                || node.left as usize >= tree.len()
                || node.right as usize >= tree.len()
            {
                return Err("invalid rank branch".to_owned());
            }
        }
        trees.push(tree);
    }
    if cursor.position != cursor.bytes.len() {
        return Err("rank artifact trailing bytes".to_owned());
    }
    let linear = LogisticRegression::from_artifact(LINEAR_ARTIFACT, RANK_FEATURE_SCHEMA_SHA256)
        .map_err(|error| error.to_string())?;
    if linear.n_features_in() != AGGRESSION_FEATURE_COUNT {
        return Err("rank linear width mismatch".to_owned());
    }
    Ok(RankModel {
        linear,
        center,
        calibration_slope,
        calibration_intercept,
        baseline,
        linear_weight,
        tree_weight,
        harshness_correction,
        trees,
    })
}

static RANK_MODEL: OnceLock<std::result::Result<RankModel, String>> = OnceLock::new();

fn rank_model() -> Result<&'static RankModel> {
    RANK_MODEL
        .get_or_init(decode_rank_model)
        .as_ref()
        .map_err(|error| SonaraError::ModelError(format!("invalid aggression rank model: {error}")))
}

pub(crate) fn score_evidence(evidence: &AggressionEvidence) -> Result<AggressionAnalysis> {
    let features = evidence.features();
    if features.iter().any(|value| !value.is_finite()) {
        return Err(SonaraError::ModelError(
            "aggression evidence contains non-finite values".to_owned(),
        ));
    }
    let confidence = evidence.content_support.clamp(0.0, 1.0);
    let (forcefulness, harshness, tension, rhythm) = evidence.components();
    let score = if confidence >= MIN_CONTENT_SUPPORT {
        Some(rank_model()?.predict(&features))
    } else {
        None
    };
    Ok(AggressionAnalysis {
        score,
        confidence,
        forcefulness,
        harshness,
        tension,
        rhythm,
    })
}

/// Analyze an audio file with the current fused rank model.
pub fn analyze_file(path: &Path, sample_rate: u32) -> Result<AggressionAnalysis> {
    let config = aggression_config();
    let result = analyze::analyze_file(path, sample_rate, &config)?;
    analysis_result(&result)
}

/// Analyze a mono signal with the current fused rank model.
pub fn analyze_signal(
    signal: ArrayView1<'_, Float>,
    sample_rate: u32,
) -> Result<AggressionAnalysis> {
    let config = aggression_config();
    let result = analyze::analyze_signal(signal, sample_rate, &config)?;
    analysis_result(&result)
}

/// Analyze audio files in parallel while isolating failures per path.
///
/// The returned vector preserves input order and has exactly one entry per
/// path, matching [`crate::analyze::analyze_batch`].
pub fn analyze_batch(paths: &[&Path], sample_rate: u32) -> Vec<Result<AggressionAnalysis>> {
    let config = aggression_config();
    analyze::analyze_batch(paths, sample_rate, &config)
        .into_iter()
        .map(|result| result.and_then(|analysis| analysis_result(&analysis)))
        .collect()
}

fn aggression_config() -> AnalysisConfig {
    AnalysisConfig {
        features: Some(HashSet::from(["aggression".to_owned()])),
        ..AnalysisConfig::default()
    }
}

fn analysis_result(analysis: &analyze::TrackAnalysis) -> Result<AggressionAnalysis> {
    Ok(AggressionAnalysis {
        score: analysis.aggression_score,
        confidence: analysis
            .aggression_confidence
            .ok_or_else(|| SonaraError::ModelError("aggression confidence missing".to_owned()))?,
        forcefulness: analysis
            .aggression_forcefulness
            .ok_or_else(|| SonaraError::ModelError("aggression forcefulness missing".to_owned()))?,
        harshness: analysis
            .aggression_harshness
            .ok_or_else(|| SonaraError::ModelError("aggression harshness missing".to_owned()))?,
        tension: analysis
            .aggression_tension
            .ok_or_else(|| SonaraError::ModelError("aggression tension missing".to_owned()))?,
        rhythm: analysis
            .aggression_rhythm
            .ok_or_else(|| SonaraError::ModelError("aggression rhythm missing".to_owned()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn model_metadata_and_artifacts_are_bound() {
        assert_eq!(AGGRESSION_MODEL_VERSION, 2);
        assert_eq!(AGGRESSION_EMBEDDING_VERSION, SIMILARITY_VERSION);
        assert_eq!(legacy_model().unwrap().n_features_in(), EMBEDDING_DIM);
        assert_eq!(LEGACY_MODEL_ARTIFACT.len(), 300);
        let current = rank_model().unwrap();
        assert_eq!(current.linear.n_features_in(), AGGRESSION_FEATURE_COUNT);
        assert_eq!(current.trees.len(), 16);
        assert_eq!(RANK_ARTIFACT.len(), 7_080);
    }

    #[test]
    fn rejects_wrong_dimension_version_and_non_finite_values() {
        assert!(matches!(
            score(&[0.0; EMBEDDING_DIM - 1]),
            Err(SonaraError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            score_versioned(&[0.0; EMBEDDING_DIM], SIMILARITY_VERSION + 1),
            Err(SonaraError::InvalidParameter {
                param: "embedding_version",
                ..
            })
        ));
        let mut invalid = [0.0; EMBEDDING_DIM];
        invalid[7] = Float::NAN;
        assert!(matches!(
            score(&invalid),
            Err(SonaraError::InvalidParameter {
                param: "embedding",
                ..
            })
        ));
        invalid[7] = Float::INFINITY;
        assert!(matches!(
            score(&invalid),
            Err(SonaraError::InvalidParameter {
                param: "embedding",
                ..
            })
        ));
    }

    #[test]
    fn score_is_bounded_and_deterministic() {
        let embedding = std::array::from_fn::<_, EMBEDDING_DIM, _>(|index| {
            index as Float / (EMBEDDING_DIM - 1) as Float
        });
        let first = score(&embedding).unwrap();
        assert!((0.0..=1.0).contains(&first));
        assert_eq!(first.to_bits(), score(&embedding).unwrap().to_bits());
    }

    #[test]
    fn scores_match_frozen_model_goldens() {
        let zero = [0.0; EMBEDDING_DIM];
        let one = [1.0; EMBEDDING_DIM];
        let ramp = std::array::from_fn::<_, EMBEDDING_DIM, _>(|index| {
            index as Float / (EMBEDDING_DIM - 1) as Float
        });
        assert_eq!(score(&zero).unwrap().to_bits(), 0x3dc2_4c01);
        assert_eq!(score(&one).unwrap().to_bits(), 0x3f49_eb2b);
        assert_eq!(score(&ramp).unwrap().to_bits(), 0x3e9e_f194);
    }

    #[test]
    fn audio_convenience_path_matches_fused_analysis() {
        let sample_rate = 22_050;
        let signal = Array1::from_iter((0..sample_rate).map(|index| {
            let time = index as Float / sample_rate as Float;
            0.3 * (2.0 * std::f32::consts::PI * 220.0 * time).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 660.0 * time).sin()
        }));
        let direct = analyze_signal(signal.view(), sample_rate).unwrap();
        let analysis = analyze::analyze_signal(
            signal.view(),
            sample_rate,
            &AnalysisConfig {
                features: Some(HashSet::from(["aggression".to_owned()])),
                ..AnalysisConfig::default()
            },
        )
        .unwrap();
        assert_eq!(direct.score, analysis.aggression_score);
        assert_eq!(direct.confidence, analysis.aggression_confidence.unwrap());
        assert_eq!(
            direct.forcefulness,
            analysis.aggression_forcefulness.unwrap()
        );
        assert!(analysis.embedding.is_none());
        assert_eq!(
            analysis.provenance.aggression_model_id.as_deref(),
            Some(AGGRESSION_MODEL_ID)
        );
    }

    #[test]
    fn silence_abstains_with_zero_support() {
        let silence = Array1::zeros(22_050);
        let result = analyze_signal(silence.view(), 22_050).unwrap();
        assert_eq!(result.score, None);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn empty_batch_preserves_cardinality() {
        assert!(analyze_batch(&[], 22_050).is_empty());
    }
}
