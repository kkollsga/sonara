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
        let probs = self.inner.predict_probs(embedding);
        probs
            .get(self.vocal_idx)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
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
}
