use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::IntoPyResult;
use sonara::core::spectrum as rs;
use sonara::types::*;

#[pyfunction]
#[pyo3(name = "stft", signature = (y, *, n_fft=2048, hop_length=None, win_length=None, window="hann", center=true, pad_mode="constant"))]
pub fn py_stft<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: &str,
    center: bool,
    pad_mode: &str,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let win = WindowSpec::Named(window.to_string());
    let pm = PadMode::from_str(pad_mode).into_pyresult()?;
    let result = rs::stft(
        y.as_array(),
        n_fft,
        hop_length,
        win_length,
        &win,
        center,
        pm,
    )
    .into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "istft", signature = (stft_matrix, *, hop_length=None, win_length=None, window="hann", center=true, length=None))]
pub fn py_istft<'py>(
    py: Python<'py>,
    stft_matrix: PyReadonlyArray2<'py, Complex32>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: &str,
    center: bool,
    length: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let win = WindowSpec::Named(window.to_string());
    let result = rs::istft(
        stft_matrix.as_array(),
        hop_length,
        win_length,
        &win,
        center,
        length,
    )
    .into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "power_to_db", signature = (s, *, ref_power=1.0, amin=1e-10, top_db=80.0))]
pub fn py_power_to_db<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    ref_power: f32,
    amin: f32,
    top_db: f32,
) -> Bound<'py, PyArray2<f32>> {
    let top = if top_db > 0.0 { Some(top_db) } else { None };
    rs::power_to_db(s.as_array(), ref_power, amin, top).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "amplitude_to_db", signature = (s, *, ref_amplitude=1.0, amin=1e-5, top_db=80.0))]
pub fn py_amplitude_to_db<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    ref_amplitude: f32,
    amin: f32,
    top_db: f32,
) -> Bound<'py, PyArray2<f32>> {
    let top = if top_db > 0.0 { Some(top_db) } else { None };
    rs::amplitude_to_db(s.as_array(), ref_amplitude, amin, top).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "griffinlim", signature = (s_mag, *, n_iter=32, hop_length=None, win_length=None, window="hann"))]
pub fn py_griffinlim<'py>(
    py: Python<'py>,
    s_mag: PyReadonlyArray2<'py, f32>,
    n_iter: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let win = WindowSpec::Named(window.to_string());
    let result =
        rs::griffinlim(s_mag.as_array(), n_iter, hop_length, win_length, &win).into_pyresult()?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "db_to_power", signature = (s_db, *, ref_power=1.0))]
pub fn py_db_to_power<'py>(
    py: Python<'py>,
    s_db: PyReadonlyArray2<'py, f32>,
    ref_power: f32,
) -> Bound<'py, PyArray2<f32>> {
    rs::db_to_power(s_db.as_array(), ref_power).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "db_to_amplitude", signature = (s_db, *, ref_amplitude=1.0))]
pub fn py_db_to_amplitude<'py>(
    py: Python<'py>,
    s_db: PyReadonlyArray2<'py, f32>,
    ref_amplitude: f32,
) -> Bound<'py, PyArray2<f32>> {
    rs::db_to_amplitude(s_db.as_array(), ref_amplitude).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "magphase", signature = (d, *, power=1.0))]
pub fn py_magphase<'py>(
    py: Python<'py>,
    d: PyReadonlyArray2<'py, Complex32>,
    power: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<Complex32>>) {
    let (mag, phase) = rs::magphase(d.as_array(), power);
    (mag.into_pyarray(py), phase.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "phase_vocoder", signature = (d, *, rate, hop_length=None))]
pub fn py_phase_vocoder<'py>(
    py: Python<'py>,
    d: PyReadonlyArray2<'py, Complex32>,
    rate: f32,
    hop_length: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    rs::phase_vocoder(d.as_array(), rate, hop_length)
        .map(|r| r.into_pyarray(py))
        .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "pcen", signature = (s, *, sr=22050.0, hop_length=512, gain=0.98, bias=2.0, power=0.5, time_constant=0.06, eps=1e-6))]
pub fn py_pcen<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    sr: f32,
    hop_length: usize,
    gain: f32,
    bias: f32,
    power: f32,
    time_constant: f32,
    eps: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    rs::pcen(
        s.as_array(),
        sr,
        hop_length,
        gain,
        bias,
        power,
        time_constant,
        eps,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "perceptual_weighting", signature = (s, frequencies, *, kind="A"))]
pub fn py_perceptual_weighting<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    frequencies: PyReadonlyArray1<'py, f32>,
    kind: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    rs::perceptual_weighting(s.as_array(), frequencies.as_array(), kind)
        .map(|r| r.into_pyarray(py))
        .into_pyresult()
}

// CQT functions
#[pyfunction]
#[pyo3(name = "cqt", signature = (y, *, sr=22050, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, filter_scale=1.0))]
pub fn py_cqt<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    hop_length: usize,
    fmin: Option<f32>,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: f32,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    sonara::core::constantq::cqt(
        y.as_array(),
        sr,
        hop_length,
        fmin,
        n_bins,
        bins_per_octave,
        filter_scale,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "vqt", signature = (y, *, sr=22050, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, filter_scale=1.0, gamma=0.0))]
pub fn py_vqt<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    hop_length: usize,
    fmin: Option<f32>,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: f32,
    gamma: f32,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    sonara::core::constantq::vqt(
        y.as_array(),
        sr,
        hop_length,
        fmin,
        n_bins,
        bins_per_octave,
        filter_scale,
        gamma,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "hybrid_cqt", signature = (y, *, sr=22050, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, filter_scale=1.0))]
pub fn py_hybrid_cqt<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    hop_length: usize,
    fmin: Option<f32>,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: f32,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    sonara::core::constantq::hybrid_cqt(
        y.as_array(),
        sr,
        hop_length,
        fmin,
        n_bins,
        bins_per_octave,
        filter_scale,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "pseudo_cqt", signature = (y, *, sr=22050, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, filter_scale=1.0))]
pub fn py_pseudo_cqt<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    hop_length: usize,
    fmin: Option<f32>,
    n_bins: usize,
    bins_per_octave: usize,
    filter_scale: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    sonara::core::constantq::pseudo_cqt(
        y.as_array(),
        sr,
        hop_length,
        fmin,
        n_bins,
        bins_per_octave,
        filter_scale,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "icqt", signature = (cq, *, sr=22050, hop_length=512, fmin=None, bins_per_octave=12, filter_scale=1.0))]
pub fn py_icqt<'py>(
    py: Python<'py>,
    cq: PyReadonlyArray2<'py, Complex32>,
    sr: u32,
    hop_length: usize,
    fmin: Option<f32>,
    bins_per_octave: usize,
    filter_scale: f32,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    sonara::core::constantq::icqt(
        cq.as_array(),
        sr,
        hop_length,
        fmin,
        bins_per_octave,
        filter_scale,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "griffinlim_cqt", signature = (cq_mag, *, sr=22050, hop_length=512, fmin=None, bins_per_octave=12, n_iter=32))]
pub fn py_griffinlim_cqt<'py>(
    py: Python<'py>,
    cq_mag: PyReadonlyArray2<'py, f32>,
    sr: u32,
    hop_length: usize,
    fmin: Option<f32>,
    bins_per_octave: usize,
    n_iter: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    sonara::core::constantq::griffinlim_cqt(
        cq_mag.as_array(),
        sr,
        hop_length,
        fmin,
        bins_per_octave,
        n_iter,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

// Pitch functions
#[pyfunction]
#[pyo3(name = "yin", signature = (y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None, trough_threshold=0.1))]
pub fn py_yin<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    fmin: f32,
    fmax: f32,
    sr: u32,
    frame_length: usize,
    hop_length: Option<usize>,
    trough_threshold: f32,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    sonara::core::pitch::yin(
        y.as_array(),
        fmin,
        fmax,
        sr,
        frame_length,
        hop_length,
        trough_threshold,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "pyin", signature = (y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None))]
pub fn py_pyin<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    fmin: f32,
    fmax: f32,
    sr: u32,
    frame_length: usize,
    hop_length: Option<usize>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f32>>,
    Bound<'py, numpy::PyArray1<bool>>,
    Bound<'py, numpy::PyArray1<f32>>,
)> {
    let (f0, voiced, prob) =
        sonara::core::pitch::pyin(y.as_array(), fmin, fmax, sr, frame_length, hop_length)
            .into_pyresult()?;
    Ok((
        f0.into_pyarray(py),
        voiced.into_pyarray(py),
        prob.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(name = "estimate_tuning", signature = (*, y=None, sr=22050, n_fft=None, resolution=None, bins_per_octave=None))]
pub fn py_estimate_tuning(
    y: Option<PyReadonlyArray1<'_, f32>>,
    sr: u32,
    n_fft: Option<usize>,
    resolution: Option<f32>,
    bins_per_octave: Option<usize>,
) -> PyResult<f32> {
    sonara::core::pitch::estimate_tuning(
        y.as_ref().map(|a| a.as_array()),
        sr,
        n_fft,
        resolution,
        bins_per_octave,
    )
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "pitch_tuning", signature = (pitches, *, resolution=None, bins_per_octave=None))]
pub fn py_pitch_tuning(
    pitches: Vec<f32>,
    resolution: Option<f32>,
    bins_per_octave: Option<usize>,
) -> PyResult<f32> {
    sonara::core::pitch::pitch_tuning(&pitches, resolution, bins_per_octave).into_pyresult()
}

#[pyfunction]
#[pyo3(name = "piptrack", signature = (y, *, sr=22050, n_fft=2048, hop_length=None))]
pub fn py_piptrack<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    n_fft: usize,
    hop_length: Option<usize>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let (p, m) =
        sonara::core::pitch::piptrack(y.as_array(), sr, n_fft, hop_length).into_pyresult()?;
    Ok((p.into_pyarray(py), m.into_pyarray(py)))
}

// Harmonic functions
#[pyfunction]
#[pyo3(name = "salience", signature = (s, freqs, *, harmonics, weights=None, fill_value=0.0))]
pub fn py_salience<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freqs: PyReadonlyArray1<'py, f32>,
    harmonics: Vec<usize>,
    weights: Option<Vec<f32>>,
    fill_value: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    sonara::core::harmonic::salience(
        s.as_array(),
        freqs.as_array(),
        &harmonics,
        weights.as_deref(),
        fill_value,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "interp_harmonics", signature = (s, freqs, *, harmonics, fill_value=0.0))]
pub fn py_interp_harmonics<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freqs: PyReadonlyArray1<'py, f32>,
    harmonics: Vec<f32>,
    fill_value: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    sonara::core::harmonic::interp_harmonics(s.as_array(), freqs.as_array(), &harmonics, fill_value)
        .map(|r| r.into_pyarray(py))
        .into_pyresult()
}

#[pyfunction]
#[pyo3(name = "f0_harmonics", signature = (s, freqs, f0, *, harmonics, fill_value=0.0))]
pub fn py_f0_harmonics<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freqs: PyReadonlyArray1<'py, f32>,
    f0: PyReadonlyArray1<'py, f32>,
    harmonics: Vec<f32>,
    fill_value: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    sonara::core::harmonic::f0_harmonics(
        s.as_array(),
        freqs.as_array(),
        f0.as_array(),
        &harmonics,
        fill_value,
    )
    .map(|r| r.into_pyarray(py))
    .into_pyresult()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_stft, m)?)?;
    m.add_function(wrap_pyfunction!(py_istft, m)?)?;
    m.add_function(wrap_pyfunction!(py_power_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(py_amplitude_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(py_db_to_power, m)?)?;
    m.add_function(wrap_pyfunction!(py_db_to_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(py_magphase, m)?)?;
    m.add_function(wrap_pyfunction!(py_phase_vocoder, m)?)?;
    m.add_function(wrap_pyfunction!(py_pcen, m)?)?;
    m.add_function(wrap_pyfunction!(py_perceptual_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(py_griffinlim, m)?)?;
    // CQT
    m.add_function(wrap_pyfunction!(py_cqt, m)?)?;
    m.add_function(wrap_pyfunction!(py_vqt, m)?)?;
    m.add_function(wrap_pyfunction!(py_hybrid_cqt, m)?)?;
    m.add_function(wrap_pyfunction!(py_pseudo_cqt, m)?)?;
    m.add_function(wrap_pyfunction!(py_icqt, m)?)?;
    m.add_function(wrap_pyfunction!(py_griffinlim_cqt, m)?)?;
    // Pitch
    m.add_function(wrap_pyfunction!(py_yin, m)?)?;
    m.add_function(wrap_pyfunction!(py_pyin, m)?)?;
    m.add_function(wrap_pyfunction!(py_estimate_tuning, m)?)?;
    m.add_function(wrap_pyfunction!(py_pitch_tuning, m)?)?;
    m.add_function(wrap_pyfunction!(py_piptrack, m)?)?;
    // Harmonic
    m.add_function(wrap_pyfunction!(py_salience, m)?)?;
    m.add_function(wrap_pyfunction!(py_interp_harmonics, m)?)?;
    m.add_function(wrap_pyfunction!(py_f0_harmonics, m)?)?;
    Ok(())
}
