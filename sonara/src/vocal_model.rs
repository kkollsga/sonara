//! Bring-your-own vocal-presence classification: a small feed-forward model
//! over the hand-crafted similarity embedding, replacing the built-in
//! `vocalness` heuristic when supplied.
//!
//! The built-in contrast-based heuristic (see `TrackAnalysis::vocalness`)
//! cannot separate clean solo voice from clean solo pitched instruments —
//! on a labeled real-music set it scores AUC ≈ 0.63 with a large clear-vocal
//! false-negative cluster. This module is the precision upgrade socket: a
//! trained classifier (same JSON MLP format as [`crate::genre`], trainer in
//! `python/sonara/vocal_model.py`) that overrides `vocalness` and
//! `instrumentalness` when set via
//! [`AnalysisConfig::vocalness_model`](crate::analyze::AnalysisConfig).
//!
//! ## Format
//!
//! The JSON layout is exactly the genre-model format
//! ([`crate::genre::GENRE_MODEL_FORMAT_VERSION`]) with three extra
//! requirements, validated at load:
//! - `id` is **required** (non-empty): the model identity is carried into
//!   [`AnalysisProvenance`](crate::analyze::AnalysisProvenance) as
//!   `vocalness_model_id`, so downstream caches can invalidate scores
//!   produced by a different (or no) model.
//! - exactly **two labels**, one of which must be `"vocal"`
//!   (case-insensitive); the other conventionally `"instrumental"`.
//! - the usual `embedding_version` match against
//!   [`crate::similarity::SIMILARITY_VERSION`] is enforced at use time.
//!
//! The reported score is the softmax probability of the `"vocal"` label —
//! a calibrated `[0, 1]` vocal-presence estimate, not an argmax.

use std::path::Path;

use crate::error::{Result, SonaraError};
use crate::genre::GenreModel;
use crate::types::Float;

/// A loaded, validated vocal-presence classifier over the similarity
/// embedding. Wraps the shared MLP machinery from [`crate::genre`].
#[derive(Debug, Clone)]
pub struct VocalnessModel {
    inner: GenreModel,
    /// Index of the `"vocal"` label in the softmax output.
    vocal_idx: usize,
}

impl VocalnessModel {
    /// The model identity string (required at load, never empty).
    pub fn id(&self) -> &str {
        self.inner.id.as_deref().unwrap_or("")
    }

    /// The `embedding_version` the model was trained against.
    pub fn embedding_version(&self) -> u32 {
        self.inner.embedding_version
    }

    /// P(vocal) for an embedding vector, in `[0, 1]`.
    pub fn predict_vocalness(&self, embedding: &[Float]) -> Float {
        self.try_predict_vocalness(embedding).unwrap_or(0.0)
    }

    /// Fallible P(vocal) inference for analysis paths that must surface
    /// malformed model state rather than emitting a non-finite value.
    pub fn try_predict_vocalness(&self, embedding: &[Float]) -> Result<Float> {
        let probs = self.inner.try_predict_probs(embedding)?;
        let probability = probs
            .get(self.vocal_idx)
            .copied()
            .ok_or_else(|| SonaraError::ModelError("vocalness output is missing".into()))?;
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(SonaraError::ModelError(
                "vocalness probability must be finite and in [0,1]".into(),
            ));
        }
        Ok(probability)
    }

    fn validate(inner: GenreModel) -> Result<Self> {
        let err = |msg: &str| SonaraError::ModelError(format!("vocalness model: {msg}"));
        match inner.id.as_deref() {
            Some(id) if !id.trim().is_empty() => {}
            _ => {
                return Err(err(
                    "`id` is required (non-empty) — it identifies the model in analysis provenance",
                ))
            }
        }
        if inner.labels.len() != 2 {
            return Err(err("exactly two labels required (vocal + instrumental)"));
        }
        let vocal_idx = inner
            .labels
            .iter()
            .position(|l| l.eq_ignore_ascii_case("vocal"))
            .ok_or_else(|| err("one label must be \"vocal\""))?;
        Ok(Self { inner, vocal_idx })
    }
}

/// The vocalness model bundled with the crate, as JSON
/// (`models/vocalness_v2.json`, embedded at compile time). Identical to the
/// artifact the Python package resolves for `vocalness_model="bundled"`.
const BUNDLED_JSON: &str = include_str!("../models/vocalness_v2.json");

/// Load the vocalness model bundled with the crate (`sonara-vocalness-v2`).
///
/// This is the Rust-native equivalent of the Python
/// `vocalness_model="bundled"` shorthand: the ~33 KB JSON artifact is embedded
/// in the crate at compile time (dead-stripped when unused), so in-process
/// consumers of the Rust core need no filesystem lookup or vendored copy.
/// Validation numbers and known limitations are documented on the release.
///
/// Fallible by signature for API stability across future model revisions;
/// with the shipped artifact it always succeeds (covered by tests).
pub fn bundled() -> Result<VocalnessModel> {
    from_json_str(BUNDLED_JSON)
}

/// Parse and validate a vocalness model from a JSON string.
pub fn from_json_str(s: &str) -> Result<VocalnessModel> {
    VocalnessModel::validate(crate::genre::from_json_str(s)?)
}

/// Load and validate a vocalness model from a JSON file on disk.
pub fn load(path: &Path) -> Result<VocalnessModel> {
    VocalnessModel::validate(crate::genre::load(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_json(id: &str, labels: &str) -> String {
        // 48 → 2 single-layer softmax; weights favor the second label a bit.
        let row0: Vec<String> = (0..48).map(|_| "0.0".to_string()).collect();
        let row1: Vec<String> = (0..48).map(|i| format!("{}", (i as f32) * 0.01)).collect();
        format!(
            r#"{{"format_version":1,"embedding_version":{ev},"id":{id},"labels":{labels},
                 "layers":[{{"weights":[[{r0}],[{r1}]],"bias":[0.0,0.1],"activation":"softmax"}}]}}"#,
            ev = crate::similarity::SIMILARITY_VERSION,
            r0 = row0.join(","),
            r1 = row1.join(","),
        )
    }

    #[test]
    fn test_valid_model_loads_and_predicts() {
        let m = from_json_str(&model_json(
            "\"vocal-test-v1\"",
            r#"["instrumental","vocal"]"#,
        ))
        .unwrap();
        assert_eq!(m.id(), "vocal-test-v1");
        let p_zero = m.predict_vocalness(&[0.0; 48]);
        assert!((0.0..=1.0).contains(&p_zero));
        let p_ones = m.predict_vocalness(&[1.0; 48]);
        assert!(p_ones > p_zero, "positive weights on vocal row must raise P");
        // Deterministic.
        assert_eq!(p_ones, m.predict_vocalness(&[1.0; 48]));
    }

    #[test]
    fn test_missing_id_rejected() {
        let json = model_json("null", r#"["instrumental","vocal"]"#);
        assert!(from_json_str(&json).is_err());
        let json = model_json("\"  \"", r#"["instrumental","vocal"]"#);
        assert!(from_json_str(&json).is_err());
    }

    #[test]
    fn test_label_shape_rejected() {
        // Three labels.
        let json = model_json("\"m\"", r#"["a","vocal","c"]"#);
        assert!(from_json_str(&json).is_err());
        // No "vocal" label. (Two labels, but neither is vocal.)
        let row: Vec<String> = (0..48).map(|_| "0.0".into()).collect();
        let json = format!(
            r#"{{"format_version":1,"embedding_version":{},"id":"m","labels":["a","b"],
                 "layers":[{{"weights":[[{r}],[{r}]],"bias":[0.0,0.0],"activation":"softmax"}}]}}"#,
            crate::similarity::SIMILARITY_VERSION,
            r = row.join(","),
        );
        assert!(from_json_str(&json).is_err());
    }

    #[test]
    fn test_vocal_label_position_respected() {
        // "vocal" first: P(vocal) must read index 0.
        let m = from_json_str(&model_json(
            "\"vocal-first\"",
            r#"["vocal","instrumental"]"#,
        ))
        .unwrap();
        // Row 1 (instrumental here) has the positive weights, so ones input
        // should LOWER P(vocal).
        let p_zero = m.predict_vocalness(&[0.0; 48]);
        let p_ones = m.predict_vocalness(&[1.0; 48]);
        assert!(p_ones < p_zero);
    }

    #[test]
    fn test_bundled_loads_with_stable_id() {
        let m = bundled().expect("bundled model must always load");
        assert_eq!(m.id(), "sonara-vocalness-v2");
        assert_eq!(m.embedding_version(), crate::similarity::SIMILARITY_VERSION);
        let p = m.predict_vocalness(&[0.0; 48]);
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_bundled_matches_python_package_artifact() {
        // The Python wheel ships its own copy as package data; the two must
        // stay byte-identical. Runs only in the repo layout (both files
        // present) — exactly where a divergence could be introduced.
        let py_copy = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../python/sonara/models/vocalness_v2.json"
        );
        match std::fs::read_to_string(py_copy) {
            Ok(py_json) => assert_eq!(
                py_json, BUNDLED_JSON,
                "sonara/models/vocalness_v2.json and \
                 python/sonara/models/vocalness_v2.json have diverged"
            ),
            Err(_) => {} // vendored-crate layout: python/ not present
        }
    }

    #[test]
    fn test_non_finite_embedding_is_safe() {
        let m = from_json_str(&model_json(
            "\"vocal-test-v1\"",
            r#"["instrumental","vocal"]"#,
        ))
        .unwrap();
        let mut emb = [1.0 as Float; 48];
        emb[3] = Float::NAN;
        emb[7] = Float::INFINITY;
        let p = m.predict_vocalness(&emb);
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_fallible_prediction_rejects_non_finite_model_output() {
        let mut model = from_json_str(&model_json(
            "\"vocal-test-v1\"",
            r#"["instrumental","vocal"]"#,
        ))
        .unwrap();
        let layer = model.inner.layers.last_mut().unwrap();
        layer.bias[0] = Float::MAX;
        layer.weights[0][0] = Float::MAX;

        let embedding = [2.0; 48];
        assert!(matches!(
            model.try_predict_vocalness(&embedding),
            Err(SonaraError::ModelError(_))
        ));
        assert!(model.predict_vocalness(&embedding).is_finite());
    }
}
