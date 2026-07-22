//! Versioned aggression scoring over Sonara's similarity embedding.
//!
//! The scorer is deliberately small: a fixed 48-term linear model followed by
//! a stable sigmoid. It performs no allocation and adds no runtime dependency.

use std::collections::HashSet;
use std::path::Path;

use ndarray::ArrayView1;

use crate::analyze::{self, AnalysisConfig};
use crate::error::{Result, SonaraError};
use crate::similarity::{EMBEDDING_DIM, SIMILARITY_VERSION};
use crate::types::Float;

/// Version of the bundled aggression model.
pub const AGGRESSION_MODEL_VERSION: u32 = 1;

/// Similarity embedding layout consumed by this model.
pub const AGGRESSION_EMBEDDING_VERSION: u32 = 2;

/// Stable identifier for the bundled coefficients and intercept.
pub const AGGRESSION_MODEL_ID: &str = "aggression-logistic-v1";

const INTERCEPT: Float = f32::from_bits(0xc010_5afc);

// Stored as exact IEEE-754 bit patterns so compiler decimal parsing cannot
// change the fitted artifact.
const COEFFICIENT_BITS: [u32; EMBEDDING_DIM] = [
    0x3f13_7a97,
    0xbcbd_5db6,
    0xbe57_907a,
    0x3dca_31ba,
    0xbd18_42ea,
    0x3dba_1c9b,
    0x3d65_782e,
    0x3e05_a9a0,
    0x3d88_fad8,
    0x3de1_b555,
    0x3d95_7cbc,
    0x3dc5_e07b,
    0x3d4a_dca2,
    0x3daf_b098,
    0x3e0f_d5a9,
    0x3e05_ef65,
    0x3e1a_afb3,
    0x3e08_e0db,
    0x3e0b_1754,
    0x3e23_c64a,
    0x3e06_bd8b,
    0x3e27_ddd3,
    0x3e19_de8c,
    0x3e2a_2709,
    0x3e0e_da59,
    0xbdb3_a321,
    0xbde7_a85a,
    0xbde3_6bd9,
    0xbdc6_3307,
    0xbd7f_fc5d,
    0xbd39_d2a6,
    0x3e96_9e56,
    0x3e8a_29ea,
    0x3eb7_362e,
    0x3e0b_3f18,
    0x3ca5_18f6,
    0xbabb_98af,
    0x3d0d_3a98,
    0xb937_df84,
    0x3e2f_6be9,
    0xbe10_4ffc,
    0xbb6f_17ec,
    0x3dd0_56b1,
    0x3b86_b4c0,
    0xbd86_0e36,
    0xbdea_1de8,
    0x3e36_e2c4,
    0x3da4_df39,
];

/// Score a current Sonara embedding in `[0, 1]`.
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

    let logit = embedding
        .iter()
        .zip(COEFFICIENT_BITS)
        .fold(INTERCEPT, |sum, (&value, bits)| {
            sum + value * Float::from_bits(bits)
        });
    Ok(stable_sigmoid(logit))
}

/// Analyze an audio file and return its aggression score.
pub fn analyze_file(path: &Path, sample_rate: u32) -> Result<Float> {
    let config = embedding_config();
    let result = analyze::analyze_file(path, sample_rate, &config)?;
    score_analysis_embedding(result.embedding, result.embedding_version)
}

/// Analyze a mono audio signal and return its aggression score.
pub fn analyze_signal(signal: ArrayView1<'_, Float>, sample_rate: u32) -> Result<Float> {
    let config = embedding_config();
    let result = analyze::analyze_signal(signal, sample_rate, &config)?;
    score_analysis_embedding(result.embedding, result.embedding_version)
}

/// Analyze audio files in parallel while isolating failures per path.
///
/// The returned vector preserves input order and has exactly one entry per
/// path, matching [`crate::analyze::analyze_batch`].
pub fn analyze_batch(paths: &[&Path], sample_rate: u32) -> Vec<Result<Float>> {
    let config = embedding_config();
    analyze::analyze_batch(paths, sample_rate, &config)
        .into_iter()
        .map(|result| {
            result.and_then(|analysis| {
                score_analysis_embedding(analysis.embedding, analysis.embedding_version)
            })
        })
        .collect()
}

fn embedding_config() -> AnalysisConfig {
    AnalysisConfig {
        features: Some(HashSet::from(["embedding".to_owned()])),
        ..AnalysisConfig::default()
    }
}

fn score_analysis_embedding(
    embedding: Option<Vec<Float>>,
    embedding_version: Option<u32>,
) -> Result<Float> {
    let embedding = embedding.ok_or_else(|| SonaraError::ModelError("embedding missing".into()))?;
    let version = embedding_version
        .ok_or_else(|| SonaraError::ModelError("embedding version missing".into()))?;
    score_versioned(&embedding, version)
}

#[inline]
fn stable_sigmoid(value: Float) -> Float {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn model_metadata_is_bound_to_current_embedding() {
        assert_eq!(AGGRESSION_MODEL_VERSION, 1);
        assert_eq!(AGGRESSION_EMBEDDING_VERSION, SIMILARITY_VERSION);
        assert_eq!(COEFFICIENT_BITS.len(), EMBEDDING_DIM);
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
    fn audio_convenience_path_matches_direct_embedding_score() {
        let sample_rate = 22_050;
        let signal = Array1::from_iter((0..sample_rate).map(|index| {
            let time = index as Float / sample_rate as Float;
            0.3 * (2.0 * std::f32::consts::PI * 220.0 * time).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 660.0 * time).sin()
        }));
        let direct = analyze_signal(signal.view(), sample_rate).unwrap();
        let analysis =
            analyze::analyze_signal(signal.view(), sample_rate, &embedding_config()).unwrap();
        let expected = score_versioned(
            analysis.embedding.as_deref().unwrap(),
            analysis.embedding_version.unwrap(),
        )
        .unwrap();
        assert_abs_diff_eq!(direct, expected, epsilon = Float::EPSILON);
    }

    #[test]
    fn empty_batch_preserves_cardinality() {
        assert!(analyze_batch(&[], 22_050).is_empty());
    }
}
