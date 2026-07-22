//! Versioned aggression scoring over Sonara's similarity embedding.
//!
//! FerricML validates the bundled model artifact once, then inference is one
//! allocation-free dot product and sigmoid per track.

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use ferricml::linear_model::LogisticRegression;
use ndarray::ArrayView1;

use crate::analyze::{self, AnalysisConfig};
use crate::error::{Result, SonaraError};
use crate::similarity::{EMBEDDING_DIM, SIMILARITY_VERSION};
use crate::types::Float;

/// Version of the bundled aggression model.
pub const AGGRESSION_MODEL_VERSION: u32 = 1;

/// Similarity embedding layout consumed by this model.
pub const AGGRESSION_EMBEDDING_VERSION: u32 = 2;

/// Stable identifier for the bundled model artifact.
pub const AGGRESSION_MODEL_ID: &str = "aggression-logistic-v1";

const FEATURE_SCHEMA_SHA256: [u8; 32] = [
    0xc5, 0x35, 0x88, 0x39, 0x3c, 0xeb, 0xf1, 0xd6, 0x09, 0xa3, 0xa6, 0x83, 0xdb, 0x61, 0x4e, 0xc9,
    0x28, 0x32, 0x16, 0x97, 0xbe, 0x15, 0x04, 0xae, 0xd8, 0x0d, 0x48, 0x49, 0x99, 0x37, 0x63, 0x44,
];

// Canonical FerricML v1 logistic artifact. Its envelope binds the estimator
// kind and feature schema and checks the payload checksum before use.
const MODEL_ARTIFACT: &[u8; 300] = &[
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

static MODEL: OnceLock<std::result::Result<LogisticRegression, String>> = OnceLock::new();

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

    model()?
        .predict_positive_proba(embedding)
        .map_err(|error| SonaraError::ModelError(format!("aggression inference failed: {error}")))
}

fn model() -> Result<&'static LogisticRegression> {
    MODEL
        .get_or_init(|| {
            LogisticRegression::from_artifact(MODEL_ARTIFACT, FEATURE_SCHEMA_SHA256)
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| SonaraError::ModelError(format!("invalid aggression model: {error}")))
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn model_metadata_is_bound_to_current_embedding() {
        assert_eq!(AGGRESSION_MODEL_VERSION, 1);
        assert_eq!(AGGRESSION_EMBEDDING_VERSION, SIMILARITY_VERSION);
        assert_eq!(model().unwrap().n_features_in(), EMBEDDING_DIM);
        assert_eq!(MODEL_ARTIFACT.len(), 300);
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
