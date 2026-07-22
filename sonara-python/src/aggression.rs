//! Python bindings for the bundled aggression model.

use std::path::Path;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{error_kind, IntoPyResult};
use sonara::aggression as rs;
use sonara::types::Float;

/// Score a versioned Sonara embedding in `[0, 1]`.
#[pyfunction]
#[pyo3(name = "aggression_score", signature = (embedding, *, embedding_version=rs::AGGRESSION_EMBEDDING_VERSION))]
pub fn py_aggression_score(embedding: Vec<Float>, embedding_version: u32) -> PyResult<Float> {
    rs::score_versioned(&embedding, embedding_version).into_pyresult()
}

/// Analyze an audio file and return its aggression score.
#[pyfunction]
#[pyo3(name = "analyze_aggression_file", signature = (path, *, sr=22050))]
pub fn py_analyze_aggression_file(path: &str, sr: u32) -> PyResult<Float> {
    rs::analyze_file(Path::new(path), sr).into_pyresult()
}

/// Analyze a mono signal and return its aggression score.
#[pyfunction]
#[pyo3(name = "analyze_aggression_signal", signature = (y, *, sr=22050))]
pub fn py_analyze_aggression_signal(y: PyReadonlyArray1<'_, Float>, sr: u32) -> PyResult<Float> {
    rs::analyze_signal(y.as_array(), sr).into_pyresult()
}

/// Analyze audio files in parallel, preserving input order and isolating errors.
#[pyfunction]
#[pyo3(name = "analyze_aggression_batch", signature = (paths, *, sr=22050))]
pub fn py_analyze_aggression_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    sr: u32,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let path_refs = paths
        .iter()
        .map(|path| Path::new(path.as_str()))
        .collect::<Vec<_>>();
    rs::analyze_batch(&path_refs, sr)
        .into_iter()
        .zip(paths)
        .map(|(result, path)| {
            let item = PyDict::new(py);
            item.set_item("path", path)?;
            match result {
                Ok(score) => item.set_item("aggression_score", score)?,
                Err(error) => {
                    item.set_item("error", error.to_string())?;
                    item.set_item("error_kind", error_kind(&error))?;
                }
            }
            Ok(item)
        })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("AGGRESSION_MODEL_VERSION", rs::AGGRESSION_MODEL_VERSION)?;
    m.add(
        "AGGRESSION_EMBEDDING_VERSION",
        rs::AGGRESSION_EMBEDDING_VERSION,
    )?;
    m.add("AGGRESSION_MODEL_ID", rs::AGGRESSION_MODEL_ID)?;
    m.add_function(wrap_pyfunction!(py_aggression_score, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_aggression_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_aggression_signal, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_aggression_batch, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_score_matches_core() {
        let embedding = vec![0.5; sonara::similarity::EMBEDDING_DIM];
        let expected = rs::score(&embedding).unwrap();
        let actual = py_aggression_score(embedding, rs::AGGRESSION_EMBEDDING_VERSION).unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
