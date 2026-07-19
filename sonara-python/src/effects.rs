use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::error::IntoPyResult;
use sonara::effects as rs;

#[pyfunction]
#[pyo3(name = "trim", signature = (y, *, top_db=60.0, frame_length=2048, hop_length=512))]
pub fn py_trim<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, (usize, usize))> {
    let (trimmed, bounds) =
        rs::trim(y.as_array(), top_db, frame_length, hop_length).into_pyresult()?;
    Ok((trimmed.into_pyarray(py), bounds))
}

#[pyfunction]
#[pyo3(name = "split", signature = (y, *, top_db=60.0, frame_length=2048, hop_length=512))]
pub fn py_split(
    y: PyReadonlyArray1<'_, f32>,
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
) -> PyResult<Vec<(usize, usize)>> {
    rs::split(y.as_array(), top_db, frame_length, hop_length).into_pyresult()
}

#[pyfunction]
#[pyo3(name = "split_with_constraints", signature = (
    y, *, sr=22050, top_db=60.0, frame_length=2048, hop_length=512,
    min_silence_duration=None, min_signal_duration=None
))]
pub fn py_split_with_constraints(
    y: PyReadonlyArray1<'_, f32>,
    sr: u32,
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
    min_silence_duration: Option<f32>,
    min_signal_duration: Option<f32>,
) -> PyResult<Vec<(usize, usize)>> {
    rs::split_with_constraints(
        y.as_array(),
        sr,
        top_db,
        frame_length,
        hop_length,
        min_silence_duration,
        min_signal_duration,
    )
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "melody_separate", signature = (
    y, *, sr=22050, fmin=65.0, fmax=2100.0, n_harmonics=10, n_fft=2048, hop_length=512
))]
pub fn py_melody_separate<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    fmin: f32,
    fmax: f32,
    n_harmonics: usize,
    n_fft: usize,
    hop_length: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let (melody, accomp) =
        rs::melody_separate(y.as_array(), sr, fmin, fmax, n_harmonics, n_fft, hop_length)
            .into_pyresult()?;
    Ok((melody.into_pyarray(py), accomp.into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new(m.py(), "effects")?;
    sub.add_function(wrap_pyfunction!(py_trim, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_split, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_split_with_constraints, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_melody_separate, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}
