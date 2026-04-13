#![allow(non_snake_case)]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use canora::feature::spectral as rs;
use crate::error::IntoPyResult;

#[pyfunction]
#[pyo3(name = "melspectrogram", signature = (*, y=None, S=None, sr=22050.0, n_fft=2048, hop_length=512, n_mels=128, fmin=0.0, fmax=0.0, power=2.0))]
pub fn py_melspectrogram<'py>(
    py: Python<'py>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    S: Option<PyReadonlyArray2<'py, f64>>,
    sr: f64, n_fft: usize, hop_length: usize,
    n_mels: usize, fmin: f64, fmax: f64, power: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y_view = y.as_ref().map(|a| a.as_array());
    let s_view = S.as_ref().map(|a| a.as_array());
    let result = rs::melspectrogram(y_view, s_view, sr, n_fft, hop_length, n_mels, fmin, fmax, power)
        .into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "mfcc", signature = (*, y=None, S=None, sr=22050.0, n_mfcc=20, n_fft=2048, hop_length=512, n_mels=128, fmin=0.0, fmax=0.0))]
pub fn py_mfcc<'py>(
    py: Python<'py>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    S: Option<PyReadonlyArray2<'py, f64>>,
    sr: f64, n_mfcc: usize, n_fft: usize, hop_length: usize,
    n_mels: usize, fmin: f64, fmax: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y_view = y.as_ref().map(|a| a.as_array());
    let s_view = S.as_ref().map(|a| a.as_array());
    let result = rs::mfcc(y_view, s_view, sr, n_mfcc, n_fft, hop_length, n_mels, fmin, fmax)
        .into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "chroma_stft", signature = (*, y=None, S=None, sr=22050.0, n_fft=2048, hop_length=512, n_chroma=12, tuning=0.0))]
pub fn py_chroma_stft<'py>(
    py: Python<'py>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    S: Option<PyReadonlyArray2<'py, f64>>,
    sr: f64, n_fft: usize, hop_length: usize, n_chroma: usize, tuning: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y_view = y.as_ref().map(|a| a.as_array());
    let s_view = S.as_ref().map(|a| a.as_array());
    let result = rs::chroma_stft(y_view, s_view, sr, n_fft, hop_length, n_chroma, tuning)
        .into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "spectral_centroid", signature = (*, y=None, S=None, sr=22050.0, n_fft=2048, hop_length=512))]
pub fn py_spectral_centroid<'py>(
    py: Python<'py>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    S: Option<PyReadonlyArray2<'py, f64>>,
    sr: f64, n_fft: usize, hop_length: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y_view = y.as_ref().map(|a| a.as_array());
    let s_view = S.as_ref().map(|a| a.as_array());
    let result = rs::spectral_centroid(y_view, s_view, sr, n_fft, hop_length)
        .into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "rms", signature = (*, y=None, S=None, frame_length=2048, hop_length=512))]
pub fn py_rms<'py>(
    py: Python<'py>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    S: Option<PyReadonlyArray2<'py, f64>>,
    frame_length: usize, hop_length: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y_view = y.as_ref().map(|a| a.as_array());
    let s_view = S.as_ref().map(|a| a.as_array());
    let result = rs::rms(y_view, s_view, frame_length, hop_length)
        .into_pyresult()?;
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_melspectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_mfcc, m)?)?;
    m.add_function(wrap_pyfunction!(py_chroma_stft, m)?)?;
    m.add_function(wrap_pyfunction!(py_spectral_centroid, m)?)?;
    m.add_function(wrap_pyfunction!(py_rms, m)?)?;
    Ok(())
}
