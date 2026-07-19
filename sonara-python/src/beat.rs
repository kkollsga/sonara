use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::error::IntoPyResult;
use sonara::beat as rs;

#[pyfunction]
#[pyo3(name = "beat_track", signature = (*, y=None, onset_envelope=None, sr=22050, hop_length=512, start_bpm=120.0, tightness=100.0, trim=true, bpm_min=None, bpm_max=None))]
pub fn py_beat_track(
    y: Option<PyReadonlyArray1<'_, f32>>,
    onset_envelope: Option<PyReadonlyArray1<'_, f32>>,
    sr: u32,
    hop_length: usize,
    start_bpm: f32,
    tightness: f32,
    trim: bool,
    bpm_min: Option<f32>,
    bpm_max: Option<f32>,
) -> PyResult<(f32, Vec<usize>)> {
    let yv = y.as_ref().map(|a| a.as_array());
    let ev = onset_envelope.as_ref().map(|a| a.as_array());
    rs::beat_track_with_bpm_range(
        yv, ev, sr, hop_length, start_bpm, tightness, trim, bpm_min, bpm_max,
    )
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "tempo_curve", signature = (beat_frames, *, sr=22050, hop_length=512, smooth=None))]
pub fn py_tempo_curve(
    beat_frames: Vec<usize>,
    sr: u32,
    hop_length: usize,
    smooth: Option<usize>,
) -> PyResult<Vec<f32>> {
    rs::tempo_curve(&beat_frames, sr, hop_length, smooth).into_pyresult()
}

#[pyfunction]
#[pyo3(name = "tempo_variability")]
pub fn py_tempo_variability(tempo_curve: Vec<f32>) -> f32 {
    rs::tempo_variability(&tempo_curve)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_beat_track, m)?)?;
    m.add_function(wrap_pyfunction!(py_tempo_curve, m)?)?;
    m.add_function(wrap_pyfunction!(py_tempo_variability, m)?)?;
    Ok(())
}
