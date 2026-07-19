use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::error::IntoPyResult;
use sonara::onset as rs;

#[pyfunction]
#[pyo3(name = "onset_detect", signature = (*, y=None, onset_envelope=None, sr=22050, hop_length=512, backtrack=false, delta=0.07, wait=0))]
pub fn py_onset_detect(
    y: Option<PyReadonlyArray1<'_, f32>>,
    onset_envelope: Option<PyReadonlyArray1<'_, f32>>,
    sr: u32,
    hop_length: usize,
    backtrack: bool,
    delta: f32,
    wait: usize,
) -> PyResult<Vec<usize>> {
    let yv = y.as_ref().map(|a| a.as_array());
    let ev = onset_envelope.as_ref().map(|a| a.as_array());
    rs::onset_detect(yv, ev, sr, hop_length, backtrack, delta, wait).into_pyresult()
}

#[pyfunction]
#[pyo3(name = "onset_strength", signature = (y, *, sr=22050, hop_length=512))]
pub fn py_onset_strength<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    hop_length: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let result = rs::onset_strength(y.as_array(), sr, hop_length).into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "onset_strength_method", signature = (y, *, sr=22050, hop_length=512, method="spectral_flux"))]
pub fn py_onset_strength_method<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    hop_length: usize,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let m = rs::OnsetMethod::from_str(method).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid onset method '{}'. Valid: 'spectral_flux', 'energy', 'phase', 'complex'",
            method
        ))
    })?;
    let result = rs::onset_strength_method(y.as_array(), sr, hop_length, m).into_pyresult()?;
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_onset_detect, m)?)?;
    m.add_function(wrap_pyfunction!(py_onset_strength, m)?)?;
    m.add_function(wrap_pyfunction!(py_onset_strength_method, m)?)?;
    Ok(())
}
