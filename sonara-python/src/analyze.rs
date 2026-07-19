use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;
use std::path::Path;

use crate::error::{error_kind, IntoPyResult};
use sonara::analyze as rs;

fn result_to_dict<'py>(py: Python<'py>, r: &rs::TrackAnalysis) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    // Core (always present)
    // Provenance: schema version, effective sample rate, hop length,
    // mode/features — lets a consumer convert frame indices to seconds and
    // detect stale persisted records.
    let prov = PyDict::new(py);
    prov.set_item("schema_version", r.provenance.schema_version)?;
    prov.set_item("sample_rate", r.provenance.sample_rate)?;
    prov.set_item("hop_length", r.provenance.hop_length)?;
    prov.set_item("mode", r.provenance.mode.as_str())?;
    if let Some(ref v) = r.provenance.requested_features {
        prov.set_item("requested_features", v.clone())?;
    }
    // Model identities (absent when the built-in paths produced the fields).
    if let Some(ref id) = r.provenance.genre_model_id {
        prov.set_item("genre_model_id", id.clone())?;
    }
    if let Some(ref id) = r.provenance.vocalness_model_id {
        prov.set_item("vocalness_model_id", id.clone())?;
    }
    d.set_item("provenance", prov)?;
    d.set_item("duration_sec", r.duration_sec)?;
    d.set_item("bpm", r.bpm)?;
    d.set_item("bpm_raw", r.bpm_raw)?;
    d.set_item("bpm_confidence", r.bpm_confidence)?;
    // Top tempo candidates as [bpm, score] pairs, sorted by score descending.
    let bpm_candidates: Vec<(f32, f32)> = r.bpm_candidates.clone();
    d.set_item("bpm_candidates", bpm_candidates)?;
    d.set_item("n_beats", r.beats.len())?;
    d.set_item("beats", r.beats.clone())?;
    d.set_item("onset_frames", r.onset_frames.clone())?;
    d.set_item("rms_mean", r.rms_mean)?;
    d.set_item("rms_max", r.rms_max)?;
    d.set_item("loudness_lufs", r.loudness_lufs)?;
    d.set_item("dynamic_range_db", r.dynamic_range_db)?;

    // --- loudness ---
    // Extended loudness / gain metrics (opt-in via features=["loudness"]).
    if let Some(v) = r.true_peak_db {
        d.set_item("true_peak_db", v)?;
    }
    if let Some(v) = r.replaygain_db {
        d.set_item("replaygain_db", v)?;
    }
    if let Some(ref v) = r.loudness_curve {
        d.set_item("loudness_curve", v.clone())?;
    }
    if let Some(v) = r.loudness_momentary_max_db {
        d.set_item("loudness_momentary_max_db", v)?;
    }
    if let Some(v) = r.loudness_range_lu {
        d.set_item("loudness_range_lu", v)?;
    }
    // --- end loudness ---
    d.set_item("spectral_centroid_mean", r.spectral_centroid_mean)?;
    d.set_item("zero_crossing_rate", r.zero_crossing_rate)?;
    d.set_item("onset_density", r.onset_density)?;

    // Spectral (playlist/full modes)
    if let Some(v) = r.spectral_bandwidth_mean {
        d.set_item("spectral_bandwidth_mean", v)?;
    }
    if let Some(v) = r.spectral_rolloff_mean {
        d.set_item("spectral_rolloff_mean", v)?;
    }
    if let Some(v) = r.spectral_flatness_mean {
        d.set_item("spectral_flatness_mean", v)?;
    }
    if let Some(ref v) = r.spectral_contrast_mean {
        d.set_item("spectral_contrast_mean", v.clone())?;
    }
    if let Some(ref v) = r.mfcc_mean {
        d.set_item("mfcc_mean", v.clone())?;
    }
    if let Some(ref v) = r.chroma_mean {
        d.set_item("chroma_mean", v.clone())?;
    }

    // Rhythm (playlist/full modes)
    if let Some(ref v) = r.tempo_curve {
        d.set_item("tempo_curve", v.clone())?;
    }
    if let Some(v) = r.tempo_variability {
        d.set_item("tempo_variability", v)?;
    }
    if let Some(ref v) = r.time_signature {
        d.set_item("time_signature", v.as_str())?;
    }
    if let Some(v) = r.time_signature_confidence {
        d.set_item("time_signature_confidence", v)?;
    }

    // Tonal (playlist/full modes)
    if let Some(ref v) = r.chord_sequence {
        d.set_item("chord_sequence", v.clone())?;
    }
    // Time-spanned chord events (merged runs of chord_sequence), list of dicts
    // mirroring the segments shape: {"label", "start_sec", "end_sec"}.
    if let Some(ref events) = r.chord_events {
        let list = pyo3::types::PyList::empty(py);
        for e in events {
            let ed = PyDict::new(py);
            ed.set_item("label", e.label.as_str())?;
            ed.set_item("start_sec", e.start_sec)?;
            ed.set_item("end_sec", e.end_sec)?;
            list.append(ed)?;
        }
        d.set_item("chord_events", list)?;
    }
    if let Some(v) = r.chord_change_rate {
        d.set_item("chord_change_rate", v)?;
    }
    if let Some(ref v) = r.predominant_chord {
        d.set_item("predominant_chord", v.as_str())?;
    }
    if let Some(v) = r.dissonance {
        d.set_item("dissonance", v)?;
    }

    // Perceptual (playlist/full modes)
    if let Some(v) = r.energy {
        d.set_item("energy", v)?;
    }
    if let Some(v) = r.danceability {
        d.set_item("danceability", v)?;
    }
    if let Some(ref v) = r.key {
        d.set_item("key", v.as_str())?;
    }
    if let Some(v) = r.key_confidence {
        d.set_item("key_confidence", v)?;
    }
    if let Some(ref v) = r.key_camelot {
        d.set_item("key_camelot", v.as_str())?;
    }
    if let Some(v) = r.valence {
        d.set_item("valence", v)?;
    }
    if let Some(v) = r.acousticness {
        d.set_item("acousticness", v)?;
    }

    // Embedding (future)
    if let Some(ref v) = r.embedding {
        d.set_item("embedding", v.clone())?;
    }
    // --- similarity ---
    if let Some(v) = r.embedding_version {
        d.set_item("embedding_version", v)?;
    }

    // Tier 3 placeholders (only included when not None)
    if let Some(v) = r.mood_happy {
        d.set_item("mood_happy", v)?;
    }
    if let Some(v) = r.mood_aggressive {
        d.set_item("mood_aggressive", v)?;
    }
    if let Some(v) = r.mood_relaxed {
        d.set_item("mood_relaxed", v)?;
    }
    if let Some(v) = r.mood_sad {
        d.set_item("mood_sad", v)?;
    }
    if let Some(v) = r.instrumentalness {
        d.set_item("instrumentalness", v)?;
    }
    // Genre: populated only when a user-supplied genre model was passed.
    if let Some(ref v) = r.genre {
        d.set_item("genre", v.as_str())?;
    }
    if let Some(v) = r.genre_confidence {
        d.set_item("genre_confidence", v)?;
    }

    // --- beat grid ---
    // Opt-in (features=["beatgrid"]); keys absent by default.
    if let Some(v) = r.grid_offset_sec {
        d.set_item("grid_offset_sec", v)?;
    }
    if let Some(ref v) = r.downbeats {
        d.set_item("downbeats", v.clone())?;
    }
    if let Some(v) = r.grid_stability {
        d.set_item("grid_stability", v)?;
    }
    // --- structure --- (opt-in: features=["structure"])
    if let Some(ref v) = r.energy_curve {
        d.set_item("energy_curve", v.clone())?;
    }
    if let Some(v) = r.energy_curve_hop_sec {
        d.set_item("energy_curve_hop_sec", v)?;
    }
    if let Some(ref segs) = r.segments {
        let list = pyo3::types::PyList::empty(py);
        for s in segs {
            let sd = PyDict::new(py);
            sd.set_item("start_sec", s.start_sec)?;
            sd.set_item("end_sec", s.end_sec)?;
            sd.set_item("energy", s.energy)?;
            list.append(sd)?;
        }
        d.set_item("segments", list)?;
    }
    if let Some(v) = r.intro_end_sec {
        d.set_item("intro_end_sec", v)?;
    }
    if let Some(v) = r.outro_start_sec {
        d.set_item("outro_start_sec", v)?;
    }
    if let Some(v) = r.energy_level {
        d.set_item("energy_level", v)?;
    }
    // --- silence --- (opt-in via features=["silence"])
    if let Some(v) = r.leading_silence_sec {
        d.set_item("leading_silence_sec", v)?;
    }
    if let Some(v) = r.trailing_silence_sec {
        d.set_item("trailing_silence_sec", v)?;
    }

    // --- key candidates --- (opt-in via features=["key_candidates"])
    // List of (key string, camelot code, score) tuples, ranked best-first.
    if let Some(ref v) = r.key_candidates {
        let items: Vec<(String, String, f32)> = v.clone();
        d.set_item("key_candidates", items)?;
    }

    // --- vocalness --- (opt-in via features=["vocalness"])
    if let Some(v) = r.vocalness {
        d.set_item("vocalness", v)?;
    }

    // --- fingerprint ---
    // Opt-in acoustic fingerprint for duplicate detection. Serialized as a
    // compact base64 string plus an integer format version. Present only when
    // the "fingerprint" feature was requested.
    if let Some(ref fp) = r.fingerprint {
        d.set_item("fingerprint", sonara::fingerprint::encode_base64(fp))?;
        d.set_item(
            "fingerprint_version",
            sonara::fingerprint::FINGERPRINT_VERSION,
        )?;
    }

    // --- tags ---
    // Opt-in file metadata (features=["tags"], analyze_file/analyze_batch only).
    // Nested "tags" dict mirroring the provenance pattern; each key present only
    // when that tag was found in the file.
    if let Some(ref t) = r.tags {
        let td = PyDict::new(py);
        if let Some(ref v) = t.title {
            td.set_item("title", v.as_str())?;
        }
        if let Some(ref v) = t.artist {
            td.set_item("artist", v.as_str())?;
        }
        if let Some(ref v) = t.album {
            td.set_item("album", v.as_str())?;
        }
        if let Some(ref v) = t.genre {
            td.set_item("genre", v.as_str())?;
        }
        if let Some(v) = t.year {
            td.set_item("year", v)?;
        }
        if let Some(v) = t.original_year {
            td.set_item("original_year", v)?;
        }
        if let Some(v) = t.track_no {
            td.set_item("track_no", v)?;
        }
        d.set_item("tags", td)?;
    }

    Ok(d)
}

fn parse_config(
    mode: &str,
    features: Option<Vec<String>>,
    bpm_min: Option<f32>,
    bpm_max: Option<f32>,
    genre_model: Option<String>,
    vocalness_model: Option<String>,
) -> PyResult<rs::AnalysisConfig> {
    let mode = rs::AnalysisMode::from_str(mode).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid mode '{}'. Valid modes: 'compact', 'playlist', 'full'",
            mode
        ))
    })?;
    let features = features.map(|f| {
        f.into_iter()
            .map(|s| s.to_lowercase())
            .collect::<HashSet<_>>()
    });
    // Load the bring-your-own genre model once per call (path → validated model),
    // mapping a load/validation failure to the standard SonaraError → PyErr path.
    // The embedding_version match is enforced later, at analysis time.
    let genre_model = match genre_model {
        Some(path) => Some(std::sync::Arc::new(
            sonara::genre::load(Path::new(&path)).into_pyresult()?,
        )),
        None => None,
    };
    // Same one-load-per-call handling for the vocalness model.
    let vocalness_model = match vocalness_model {
        Some(path) => Some(std::sync::Arc::new(
            sonara::vocal_model::load(Path::new(&path)).into_pyresult()?,
        )),
        None => None,
    };
    Ok(rs::AnalysisConfig {
        mode,
        features,
        bpm_min,
        bpm_max,
        genre_model,
        vocalness_model,
    })
}

#[pyfunction]
#[pyo3(name = "analyze_file", signature = (path, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_analyze_file<'py>(
    py: Python<'py>,
    path: &str,
    sr: u32,
    mode: &str,
    features: Option<Vec<String>>,
    bpm_min: Option<f32>,
    bpm_max: Option<f32>,
    genre_model: Option<String>,
    vocalness_model: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_config(mode, features, bpm_min, bpm_max, genre_model, vocalness_model)?;
    let result = rs::analyze_file(Path::new(path), sr, &config).into_pyresult()?;
    result_to_dict(py, &result)
}

#[pyfunction]
#[pyo3(name = "analyze_signal", signature = (y, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_analyze_signal<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    sr: u32,
    mode: &str,
    features: Option<Vec<String>>,
    bpm_min: Option<f32>,
    bpm_max: Option<f32>,
    genre_model: Option<String>,
    vocalness_model: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_config(mode, features, bpm_min, bpm_max, genre_model, vocalness_model)?;
    let result = rs::analyze_signal(y.as_array(), sr, &config).into_pyresult()?;
    result_to_dict(py, &result)
}

/// Build a structured error entry for a file that failed to analyze.
///
/// Returns a dict with `path`, `error` (human-readable, includes container/
/// codec and underlying cause) and `error_kind` (short stable category).
fn error_to_dict<'py>(
    py: Python<'py>,
    path: &str,
    err: &sonara::SonaraError,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("path", path)?;
    d.set_item("error", err.to_string())?;
    d.set_item("error_kind", error_kind(err))?;
    Ok(d)
}

/// Turn per-file analysis `Result`s into the input-ordered list of dicts.
///
/// Shared by both the plain and progress-callback code paths so the mapping
/// (success → feature dict + `"path"`; failure → error dict) lives in one place.
fn batch_results_to_dicts<'py>(
    py: Python<'py>,
    results: Vec<Result<rs::TrackAnalysis, sonara::SonaraError>>,
    paths: &[String],
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    results
        .into_iter()
        .zip(paths.iter())
        .map(|(r, path)| match r {
            Ok(analysis) => {
                // Every batch entry carries its input path so consumers can
                // correlate results without zipping against the input list.
                let d = result_to_dict(py, &analysis)?;
                d.set_item("path", path)?;
                Ok(d)
            }
            Err(err) => error_to_dict(py, path, &err),
        })
        .collect()
}

/// Analyze many files in parallel, returning one entry per input path in order.
///
/// Unlike `analyze_file`, this never raises on a per-file decode/IO failure.
/// Every input path yields exactly one dict, in input order:
/// - success → the usual feature dict (unchanged);
/// - failure → `{ "path", "error", "error_kind" }`.
///
/// A single bad file therefore cannot abort analysis of a large library.
/// `ValueError` is still raised only for whole-call configuration errors
/// (e.g. an invalid `mode`), which apply to every path.
///
/// `progress`, if given, must be callable and is invoked as `progress(done,
/// total)` after **each** file finishes (success or failure), where `done`
/// counts completions in completion order (not input order) and `total ==
/// len(paths)`. A raising/broken callback never aborts the batch — its error is
/// swallowed (per-file isolation). `progress=None` (the default) takes exactly
/// the original code path with zero overhead.
#[pyfunction]
#[pyo3(name = "analyze_batch", signature = (paths, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None, progress=None, genre_model=None, vocalness_model=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_analyze_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    sr: u32,
    mode: &str,
    features: Option<Vec<String>>,
    bpm_min: Option<f32>,
    bpm_max: Option<f32>,
    progress: Option<Bound<'py, PyAny>>,
    genre_model: Option<String>,
    vocalness_model: Option<String>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    // Load the models once for the whole batch (parse_config validates them);
    // the Arcs are cheaply cloned per file inside the core.
    let config = parse_config(mode, features, bpm_min, bpm_max, genre_model, vocalness_model)?;
    let path_refs: Vec<&Path> = paths.iter().map(|p| Path::new(p.as_str())).collect();

    let results = match progress {
        // Fast path: no callback → exactly the original code, zero overhead.
        None => rs::analyze_batch(&path_refs, sr, &config),
        Some(cb) => {
            // Fail fast on a non-callable so a typo can't silently no-op.
            if !cb.is_callable() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "progress must be callable: progress(done: int, total: int) -> None",
                ));
            }
            // A thread-shareable (`Send`) handle to the callback for the workers.
            let cb_py: Py<PyAny> = cb.unbind();
            // Release the GIL around the parallel batch; workers re-attach only
            // to fire the callback. `config`/`path_refs` are plain Rust data.
            // The core owns the parallel map + completion counter; the closure
            // just re-attaches to Python and forwards (done, total).
            py.detach(|| {
                rs::analyze_batch_with(&path_refs, sr, &config, |n, total| {
                    // Per-file isolation: a raising callback must never abort
                    // the batch — drop its error (the `Err` carries + clears it).
                    Python::attach(|py| {
                        let _ = cb_py.call1(py, (n, total));
                    });
                })
            })
        }
    };

    batch_results_to_dicts(py, results, &paths)
}

// --- fingerprint ---
/// Pull a base64 fingerprint string out of a Python object that is either the
/// string itself or a mapping (TrackAnalysis/dict) carrying a `"fingerprint"` field.
fn extract_fp_string(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(item) = obj.get_item("fingerprint") {
        if let Ok(s) = item.extract::<String>() {
            return Ok(s);
        }
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "fingerprint_match expects base64 fingerprint strings or analysis dicts \
         containing a 'fingerprint' field (request it with features=['fingerprint'])",
    ))
}

/// Similarity in [0, 1] between two acoustic fingerprints for duplicate detection.
///
/// Each argument may be a base64 `fingerprint` string or a `TrackAnalysis`/dict
/// that contains one. A score above ~0.30 indicates the same recording (see the
/// Rust `fingerprint` module docs for the BER→score mapping and threshold).
#[pyfunction]
#[pyo3(name = "fingerprint_match")]
pub fn py_fingerprint_match(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f32> {
    let sa = extract_fp_string(a)?;
    let sb = extract_fp_string(b)?;
    let fa = sonara::fingerprint::decode_base64(&sa).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("invalid base64 fingerprint (first argument)")
    })?;
    let fb = sonara::fingerprint::decode_base64(&sb).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("invalid base64 fingerprint (second argument)")
    })?;
    Ok(sonara::fingerprint::match_score(&fa, &fb))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_analyze_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_signal, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_batch, m)?)?;
    // --- fingerprint ---
    m.add_function(wrap_pyfunction!(py_fingerprint_match, m)?)?;
    Ok(())
}
