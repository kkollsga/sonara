use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::IntoPyResult;
use sonara::feature::rhythm as rs;

#[pyfunction]
#[pyo3(name = "metrogram", signature = (*, y=None, onset_envelope=None, sr=22050, hop_length=512, win_length=384))]
pub fn py_metrogram<'py>(
    py: Python<'py>,
    y: Option<PyReadonlyArray1<'py, f32>>,
    onset_envelope: Option<PyReadonlyArray1<'py, f32>>,
    sr: u32,
    hop_length: usize,
    win_length: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let yv = y.as_ref().map(|a| a.as_array());
    let ev = onset_envelope.as_ref().map(|a| a.as_array());
    let result = rs::metrogram(yv, ev, sr, hop_length, win_length, None).into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "detect_time_signature")]
pub fn py_detect_time_signature(metrogram: PyReadonlyArray2<'_, f32>) -> (String, f32) {
    rs::detect_time_signature(metrogram.as_array(), None)
}

#[pyfunction]
#[pyo3(name = "tempo", signature = (*, y=None, onset_envelope=None, sr=22050, hop_length=512, start_bpm=120.0, max_tempo=320.0))]
pub fn py_tempo(
    y: Option<PyReadonlyArray1<'_, f32>>,
    onset_envelope: Option<PyReadonlyArray1<'_, f32>>,
    sr: u32,
    hop_length: usize,
    start_bpm: f32,
    max_tempo: f32,
) -> PyResult<f32> {
    let yv = y.as_ref().map(|a| a.as_array());
    let ev = onset_envelope.as_ref().map(|a| a.as_array());
    rs::tempo(yv, ev, sr, hop_length, start_bpm, max_tempo).into_pyresult()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_metrogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_time_signature, m)?)?;
    m.add_function(wrap_pyfunction!(py_tempo, m)?)?;
    Ok(())
}
