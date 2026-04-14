use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use sonara::tonal as rs;
use sonara::types::Float;

#[pyfunction]
#[pyo3(name = "hpcp", signature = (power_spec, freqs, *, n_harmonics=4, min_freq=40.0, max_freq=5000.0, peak_threshold=0.0, max_peaks=50))]
pub fn py_hpcp<'py>(
    py: Python<'py>,
    power_spec: PyReadonlyArray2<'py, Float>,
    freqs: PyReadonlyArray1<'py, Float>,
    n_harmonics: usize,
    min_freq: Float,
    max_freq: Float,
    peak_threshold: Float,
    max_peaks: usize,
) -> PyResult<Bound<'py, PyArray2<Float>>> {
    let result = rs::hpcp(
        power_spec.as_array(),
        freqs.as_array(),
        n_harmonics,
        min_freq,
        max_freq,
        peak_threshold,
        max_peaks,
    );
    Ok(PyArray2::from_owned_array(py, result))
}

#[pyfunction]
#[pyo3(name = "chords_from_beats")]
pub fn py_chords_from_beats<'py>(
    _py: Python<'py>,
    hpcp: PyReadonlyArray2<'py, Float>,
    beats: Vec<usize>,
) -> PyResult<Vec<String>> {
    Ok(rs::chords_from_beats(hpcp.as_array(), &beats))
}

#[pyfunction]
#[pyo3(name = "chords_from_frames", signature = (hpcp, *, segment_frames=10))]
pub fn py_chords_from_frames<'py>(
    _py: Python<'py>,
    hpcp: PyReadonlyArray2<'py, Float>,
    segment_frames: usize,
) -> PyResult<Vec<String>> {
    Ok(rs::chords_from_frames(hpcp.as_array(), segment_frames))
}

#[pyfunction]
#[pyo3(name = "chord_descriptors")]
pub fn py_chord_descriptors<'py>(
    py: Python<'py>,
    chords: Vec<String>,
    duration_sec: Float,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let desc = rs::chord_descriptors(&chords, duration_sec);
    let d = pyo3::types::PyDict::new(py);
    d.set_item("predominant_chord", desc.predominant_chord)?;
    d.set_item("chord_change_rate", desc.change_rate)?;
    d.set_item("n_unique", desc.n_unique)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(name = "dissonance", signature = (power_spec, freqs, *, peak_threshold=0.0, max_peaks=50))]
pub fn py_dissonance<'py>(
    _py: Python<'py>,
    power_spec: PyReadonlyArray2<'py, Float>,
    freqs: PyReadonlyArray1<'py, Float>,
    peak_threshold: Float,
    max_peaks: usize,
) -> PyResult<Float> {
    Ok(rs::dissonance(
        power_spec.as_array(),
        freqs.as_array(),
        peak_threshold,
        max_peaks,
    ))
}

#[pyfunction]
#[pyo3(name = "dissonance_from_peaks")]
pub fn py_dissonance_from_peaks<'py>(
    _py: Python<'py>,
    peaks_freq: Vec<Float>,
    peaks_mag: Vec<Float>,
) -> PyResult<Float> {
    Ok(rs::dissonance_from_peaks(&peaks_freq, &peaks_mag))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_hpcp, m)?)?;
    m.add_function(wrap_pyfunction!(py_chords_from_beats, m)?)?;
    m.add_function(wrap_pyfunction!(py_chords_from_frames, m)?)?;
    m.add_function(wrap_pyfunction!(py_chord_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(py_dissonance, m)?)?;
    m.add_function(wrap_pyfunction!(py_dissonance_from_peaks, m)?)?;
    Ok(())
}
