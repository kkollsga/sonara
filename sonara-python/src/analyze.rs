use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;
use std::path::Path;

use sonara::analyze as rs;
use crate::error::IntoPyResult;

fn result_to_dict<'py>(py: Python<'py>, r: &rs::TrackAnalysis) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    // Core (always present)
    d.set_item("duration_sec", r.duration_sec)?;
    d.set_item("bpm", r.bpm)?;
    d.set_item("n_beats", r.beats.len())?;
    d.set_item("beats", r.beats.clone())?;
    d.set_item("onset_frames", r.onset_frames.clone())?;
    d.set_item("rms_mean", r.rms_mean)?;
    d.set_item("rms_max", r.rms_max)?;
    d.set_item("loudness_lufs", r.loudness_lufs)?;
    d.set_item("dynamic_range_db", r.dynamic_range_db)?;
    d.set_item("spectral_centroid_mean", r.spectral_centroid_mean)?;
    d.set_item("zero_crossing_rate", r.zero_crossing_rate)?;
    d.set_item("onset_density", r.onset_density)?;

    // Spectral (playlist/full modes)
    if let Some(v) = r.spectral_bandwidth_mean { d.set_item("spectral_bandwidth_mean", v)?; }
    if let Some(v) = r.spectral_rolloff_mean { d.set_item("spectral_rolloff_mean", v)?; }
    if let Some(v) = r.spectral_flatness_mean { d.set_item("spectral_flatness_mean", v)?; }
    if let Some(ref v) = r.spectral_contrast_mean { d.set_item("spectral_contrast_mean", v.clone())?; }
    if let Some(ref v) = r.mfcc_mean { d.set_item("mfcc_mean", v.clone())?; }
    if let Some(ref v) = r.chroma_mean { d.set_item("chroma_mean", v.clone())?; }

    // Rhythm (playlist/full modes)
    if let Some(ref v) = r.tempo_curve { d.set_item("tempo_curve", v.clone())?; }
    if let Some(v) = r.tempo_variability { d.set_item("tempo_variability", v)?; }
    if let Some(ref v) = r.time_signature { d.set_item("time_signature", v.as_str())?; }
    if let Some(v) = r.time_signature_confidence { d.set_item("time_signature_confidence", v)?; }

    // Tonal (playlist/full modes)
    if let Some(ref v) = r.chord_sequence { d.set_item("chord_sequence", v.clone())?; }
    if let Some(v) = r.chord_change_rate { d.set_item("chord_change_rate", v)?; }
    if let Some(ref v) = r.predominant_chord { d.set_item("predominant_chord", v.as_str())?; }
    if let Some(v) = r.dissonance { d.set_item("dissonance", v)?; }

    // Perceptual (playlist/full modes)
    if let Some(v) = r.energy { d.set_item("energy", v)?; }
    if let Some(v) = r.danceability { d.set_item("danceability", v)?; }
    if let Some(ref v) = r.key { d.set_item("key", v.as_str())?; }
    if let Some(v) = r.key_confidence { d.set_item("key_confidence", v)?; }
    if let Some(v) = r.valence { d.set_item("valence", v)?; }
    if let Some(v) = r.acousticness { d.set_item("acousticness", v)?; }

    // Embedding (future)
    if let Some(ref v) = r.embedding { d.set_item("embedding", v.clone())?; }

    // Tier 3 placeholders (only included when not None)
    if let Some(v) = r.mood_happy { d.set_item("mood_happy", v)?; }
    if let Some(v) = r.mood_aggressive { d.set_item("mood_aggressive", v)?; }
    if let Some(v) = r.mood_relaxed { d.set_item("mood_relaxed", v)?; }
    if let Some(v) = r.mood_sad { d.set_item("mood_sad", v)?; }
    if let Some(v) = r.instrumentalness { d.set_item("instrumentalness", v)?; }
    if let Some(ref v) = r.genre { d.set_item("genre", v.as_str())?; }

    Ok(d)
}

fn parse_config(mode: &str, features: Option<Vec<String>>) -> PyResult<rs::AnalysisConfig> {
    let mode = rs::AnalysisMode::from_str(mode).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid mode '{}'. Valid modes: 'compact', 'playlist', 'full'", mode
        ))
    })?;
    let features = features.map(|f| f.into_iter().map(|s| s.to_lowercase()).collect::<HashSet<_>>());
    Ok(rs::AnalysisConfig { mode, features })
}

#[pyfunction]
#[pyo3(name = "analyze_file", signature = (path, *, sr=22050, mode="compact", features=None))]
pub fn py_analyze_file<'py>(
    py: Python<'py>,
    path: &str,
    sr: u32,
    mode: &str,
    features: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_config(mode, features)?;
    let result = rs::analyze_file(Path::new(path), sr, &config).into_pyresult()?;
    result_to_dict(py, &result)
}

#[pyfunction]
#[pyo3(name = "analyze_signal", signature = (y, *, sr=22050, mode="compact", features=None))]
pub fn py_analyze_signal<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    mode: &str,
    features: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_config(mode, features)?;
    let result = rs::analyze_signal(y.as_array(), sr, &config).into_pyresult()?;
    result_to_dict(py, &result)
}

#[pyfunction]
#[pyo3(name = "analyze_batch", signature = (paths, *, sr=22050, mode="compact", features=None))]
pub fn py_analyze_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    sr: u32,
    mode: &str,
    features: Option<Vec<String>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let config = parse_config(mode, features)?;
    let path_refs: Vec<&Path> = paths.iter().map(|p| Path::new(p.as_str())).collect();
    let results = rs::analyze_batch(&path_refs, sr, &config);
    results
        .into_iter()
        .map(|r| {
            let analysis = r.into_pyresult()?;
            result_to_dict(py, &analysis)
        })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_analyze_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_signal, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_batch, m)?)?;
    Ok(())
}
