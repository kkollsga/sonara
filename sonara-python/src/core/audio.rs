use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::path::Path;

use crate::error::IntoPyResult;
use sonara::core::audio as rs;

#[pyfunction]
#[pyo3(name = "load", signature = (path, *, sr=22050, mono=true, offset=0.0, duration=0.0))]
pub fn py_load<'py>(
    py: Python<'py>,
    path: &str,
    sr: u32,
    mono: bool,
    offset: f32,
    duration: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, u32)> {
    let (y, sr_out) = rs::load(Path::new(path), sr, mono, offset, duration).into_pyresult()?;
    Ok((y.into_pyarray(py), sr_out))
}

#[pyfunction]
#[pyo3(name = "to_mono")]
pub fn py_to_mono<'py>(
    py: Python<'py>,
    y: numpy::PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray1<f32>> {
    rs::to_mono(y.as_array()).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "resample", signature = (y, *, orig_sr, target_sr))]
pub fn py_resample<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    orig_sr: u32,
    target_sr: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    rs::resample(y.as_array(), orig_sr, target_sr)
        .map(|r| r.into_pyarray(py))
        .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "get_duration")]
pub fn py_get_duration(path: &str) -> PyResult<f32> {
    rs::get_duration(Path::new(path)).into_pyresult()
}

#[pyfunction]
#[pyo3(name = "get_samplerate")]
pub fn py_get_samplerate(path: &str) -> PyResult<u32> {
    rs::get_samplerate(Path::new(path)).into_pyresult()
}

#[pyfunction]
#[pyo3(name = "autocorrelate", signature = (y, *, max_size=None))]
pub fn py_autocorrelate<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    max_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    rs::autocorrelate(y.as_array(), max_size)
        .map(|r| r.into_pyarray(py))
        .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "lpc", signature = (y, *, order))]
pub fn py_lpc<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    order: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    rs::lpc(y.as_array(), order)
        .map(|r| r.into_pyarray(py))
        .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "zero_crossings", signature = (y, *, threshold=0.0))]
pub fn py_zero_crossings<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    threshold: f32,
) -> Bound<'py, numpy::PyArray1<bool>> {
    rs::zero_crossings(y.as_array(), threshold).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "tone", signature = (frequency, *, sr=22050, length=22050))]
pub fn py_tone<'py>(
    py: Python<'py>,
    frequency: f32,
    sr: u32,
    length: usize,
) -> Bound<'py, PyArray1<f32>> {
    rs::tone(frequency, sr, length).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "chirp", signature = (*, fmin, fmax, sr=22050, length=22050))]
pub fn py_chirp<'py>(
    py: Python<'py>,
    fmin: f32,
    fmax: f32,
    sr: u32,
    length: usize,
) -> Bound<'py, PyArray1<f32>> {
    rs::chirp(fmin, fmax, sr, length).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "clicks", signature = (*, times, sr=22050, length=22050, click_freq=1000.0, click_duration=0.1))]
pub fn py_clicks<'py>(
    py: Python<'py>,
    times: Vec<f32>,
    sr: u32,
    length: usize,
    click_freq: f32,
    click_duration: f32,
) -> Bound<'py, PyArray1<f32>> {
    rs::clicks(&times, sr, length, click_freq, click_duration).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "mu_compress", signature = (x, *, mu=255.0))]
pub fn py_mu_compress<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    mu: f32,
) -> Bound<'py, PyArray1<f32>> {
    rs::mu_compress(x.as_array(), mu).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "mu_expand", signature = (y, *, mu=255.0))]
pub fn py_mu_expand<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    mu: f32,
) -> Bound<'py, PyArray1<f32>> {
    rs::mu_expand(y.as_array(), mu).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "stream_with_resample", signature = (
    path, *, block_length=64, frame_length=2048, hop_length=512, target_sr=22050, mono=true
))]
pub fn py_stream_with_resample<'py>(
    py: Python<'py>,
    path: &str,
    block_length: usize,
    frame_length: usize,
    hop_length: usize,
    target_sr: u32,
    mono: bool,
) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
    let blocks = rs::stream_with_resample(
        Path::new(path),
        block_length,
        frame_length,
        hop_length,
        target_sr,
        mono,
    )
    .into_pyresult()?;
    Ok(blocks.into_iter().map(|b| b.into_pyarray(py)).collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_load, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_mono, m)?)?;
    m.add_function(wrap_pyfunction!(py_resample, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_duration, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_samplerate, m)?)?;
    m.add_function(wrap_pyfunction!(py_autocorrelate, m)?)?;
    m.add_function(wrap_pyfunction!(py_lpc, m)?)?;
    m.add_function(wrap_pyfunction!(py_zero_crossings, m)?)?;
    m.add_function(wrap_pyfunction!(py_tone, m)?)?;
    m.add_function(wrap_pyfunction!(py_chirp, m)?)?;
    m.add_function(wrap_pyfunction!(py_clicks, m)?)?;
    m.add_function(wrap_pyfunction!(py_mu_compress, m)?)?;
    m.add_function(wrap_pyfunction!(py_mu_expand, m)?)?;
    m.add_function(wrap_pyfunction!(py_stream_with_resample, m)?)?;
    Ok(())
}
