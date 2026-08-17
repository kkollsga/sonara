//! Python bindings for the hand-crafted similarity / embedding vector.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sonara::similarity as rs;
use sonara::types::Float;

/// Parse a profile name into a [`rs::SimilarityProfile`], raising `ValueError`
/// (naming the valid set) on an unknown name — never a silent fallback.
fn parse_profile(name: &str) -> PyResult<rs::SimilarityProfile> {
    rs::SimilarityProfile::from_name(name).ok_or_else(|| {
        let valid: Vec<&str> = rs::SimilarityProfile::ALL
            .iter()
            .map(|p| p.name())
            .collect();
        PyValueError::new_err(format!(
            "unknown similarity profile {name:?}; valid profiles: {}",
            valid.join(", ")
        ))
    })
}

/// Weighted cosine-free similarity between two embedding vectors, in `[0, 1]`
/// (higher = more similar). `profile` selects the weight table ("default" or
/// "timbre"); see `sonara::similarity` for the metric definition.
#[pyfunction]
#[pyo3(name = "similarity", signature = (a, b, *, profile = "default"))]
pub fn py_similarity(a: Vec<Float>, b: Vec<Float>, profile: &str) -> PyResult<Float> {
    Ok(rs::similarity_with_profile(&a, &b, parse_profile(profile)?))
}

/// Distance between two embedding vectors, in `[0, 1]` (0 = identical).
/// `profile` selects the weight table ("default" or "timbre").
#[pyfunction]
#[pyo3(name = "embedding_distance", signature = (a, b, *, profile = "default"))]
pub fn py_embedding_distance(a: Vec<Float>, b: Vec<Float>, profile: &str) -> PyResult<Float> {
    Ok(rs::distance_with_profile(&a, &b, parse_profile(profile)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expose the current embedding layout version + dimensionality so callers can
    // validate stored vectors before comparing.
    m.add("SIMILARITY_VERSION", rs::SIMILARITY_VERSION)?;
    m.add("EMBEDDING_DIM", rs::EMBEDDING_DIM)?;
    // Selectable distance-time weighting profiles: name -> weight-table version.
    // Profile versions are independent of SIMILARITY_VERSION (the default
    // profile's version aliases it); see sonara::similarity module docs.
    let profiles = PyDict::new(m.py());
    for p in rs::SimilarityProfile::ALL {
        profiles.set_item(p.name(), p.version())?;
    }
    m.add("SIMILARITY_PROFILES", profiles)?;
    m.add_function(wrap_pyfunction!(py_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_embedding_distance, m)?)?;
    Ok(())
}
