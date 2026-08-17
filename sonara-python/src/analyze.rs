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
    // The octave-folding tempo range in effect at analysis time (absent when
    // unset) — without it, two results for the same audio can silently
    // diverge in `bpm` with no visible cause.
    if let Some(v) = r.provenance.bpm_min {
        prov.set_item("bpm_min", v)?;
    }
    if let Some(v) = r.provenance.bpm_max {
        prov.set_item("bpm_max", v)?;
    }
    // Model identities (absent when the built-in paths produced the fields).
    if let Some(ref id) = r.provenance.genre_model_id {
        prov.set_item("genre_model_id", id.clone())?;
    }
    if let Some(ref id) = r.provenance.vocalness_model_id {
        prov.set_item("vocalness_model_id", id.clone())?;
    }
    if let Some(ref id) = r.provenance.aggression_model_id {
        prov.set_item("aggression_model_id", id.clone())?;
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
    if let Some(v) = r.aggression_confidence {
        d.set_item("aggression_score", r.aggression_score)?;
        d.set_item("aggression_confidence", v)?;
        d.set_item("aggression_forcefulness", r.aggression_forcefulness)?;
        d.set_item("aggression_harshness", r.aggression_harshness)?;
        d.set_item("aggression_tension", r.aggression_tension)?;
        d.set_item("aggression_rhythm", r.aggression_rhythm)?;
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
    // Rust owns canonical feature-name validation and case normalization so
    // every binding and direct core caller observes the same contract.
    let features = features.map(|f| f.into_iter().collect::<HashSet<_>>());
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
    let config = parse_config(
        mode,
        features,
        bpm_min,
        bpm_max,
        genre_model,
        vocalness_model,
    )?;
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
    let config = parse_config(
        mode,
        features,
        bpm_min,
        bpm_max,
        genre_model,
        vocalness_model,
    )?;
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
    let config = parse_config(
        mode,
        features,
        bpm_min,
        bpm_max,
        genre_model,
        vocalness_model,
    )?;
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

// ============================================================
// Augment lane: dict → TrackAnalysis ingestion + bindings
// ============================================================

/// Error for a cached-analysis field that is present but of the wrong type.
fn bad_field(field: &str, expected: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "cached analysis field '{field}' is invalid: expected {expected}"
    ))
}

/// Error for a strictly required cached-analysis field that is absent.
fn missing_field(field: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "cached analysis is missing required field '{field}'"
    ))
}

/// Extract an optional field: absent key or Python `None` → `None`; a present
/// value that fails to extract is a hard error naming the field (`$label`).
macro_rules! opt_named {
    ($d:expr, $key:expr, $label:expr, $ty:ty, $expected:expr) => {
        match $d.get_item($key)? {
            None => None,
            Some(v) if v.is_none() => None,
            Some(v) => Some(
                v.extract::<$ty>()
                    .map_err(|_| bad_field($label, $expected))?,
            ),
        }
    };
}

/// `opt_named!` where the dict key doubles as the error label.
macro_rules! opt_field {
    ($d:expr, $key:expr, $ty:ty, $expected:expr) => {
        opt_named!($d, $key, $key, $ty, $expected)
    };
}

/// A core (always-emitted) field: absent → the given default (ingestion is
/// deliberately liberal — only `provenance.schema_version` is strictly
/// required); present-but-mistyped → hard error naming the field.
macro_rules! core_field {
    ($d:expr, $key:expr, $ty:ty, $expected:expr, $default:expr) => {
        opt_field!($d, $key, $ty, $expected).unwrap_or($default)
    };
}

/// A `(f32, f32)` pair from a Python tuple **or** list (JSON round-trips turn
/// tuples into lists; the inverse must accept both).
fn pair_f32(v: &Bound<'_, PyAny>) -> Option<(f32, f32)> {
    if let Ok(pair) = v.extract::<(f32, f32)>() {
        return Some(pair);
    }
    let items = v.extract::<Vec<f32>>().ok()?;
    (items.len() == 2).then(|| (items[0], items[1]))
}

/// A `(key, camelot, score)` triple from a Python tuple or list.
fn key_candidate(v: &Bound<'_, PyAny>) -> Option<(String, String, f32)> {
    if let Ok(triple) = v.extract::<(String, String, f32)>() {
        return Some(triple);
    }
    let items = v.extract::<Vec<Bound<'_, PyAny>>>().ok()?;
    if items.len() != 3 {
        return None;
    }
    Some((
        items[0].extract().ok()?,
        items[1].extract().ok()?,
        items[2].extract().ok()?,
    ))
}

/// Rebuild a [`rs::TrackAnalysis`] from the dict produced by
/// [`result_to_dict`] — the exact inverse, accepting the omissions
/// `result_to_dict` makes (absent optional fields → `None`).
///
/// Ingestion is liberal by contract: only `provenance` /
/// `provenance.schema_version` are strictly required (augmentation gates on
/// the schema version). Other always-emitted core fields default to zero /
/// empty when absent; every *present* field that fails to extract is a hard
/// `ValueError` naming the field. Unknown keys are ignored. Tuples serialized
/// through JSON as lists are accepted. A `fingerprint` in a different
/// `fingerprint_version` than this build's is rejected (re-emission would
/// silently restamp the current version onto foreign bytes).
fn analysis_from_dict(cached: &Bound<'_, PyDict>) -> PyResult<rs::TrackAnalysis> {
    // --- provenance (schema_version strictly required) ---
    let prov_any = cached
        .get_item("provenance")?
        .ok_or_else(|| missing_field("provenance"))?;
    let prov = prov_any
        .cast_into::<PyDict>()
        .map_err(|_| bad_field("provenance", "a dict"))?;
    let schema_version = opt_named!(
        prov,
        "schema_version",
        "provenance.schema_version",
        u32,
        "an int"
    )
    .ok_or_else(|| missing_field("provenance.schema_version"))?;
    let mode = match opt_named!(prov, "mode", "provenance.mode", String, "a string") {
        None => rs::AnalysisMode::Compact,
        Some(s) => rs::AnalysisMode::from_str(&s)
            .ok_or_else(|| bad_field("provenance.mode", "one of 'compact', 'playlist', 'full'"))?,
    };
    let provenance = rs::AnalysisProvenance {
        schema_version,
        sample_rate: opt_named!(prov, "sample_rate", "provenance.sample_rate", u32, "an int")
            .unwrap_or(22050),
        hop_length: opt_named!(prov, "hop_length", "provenance.hop_length", usize, "an int")
            .unwrap_or(512),
        mode,
        requested_features: opt_named!(
            prov,
            "requested_features",
            "provenance.requested_features",
            Vec<String>,
            "a list of strings"
        ),
        bpm_min: opt_named!(prov, "bpm_min", "provenance.bpm_min", f32, "a float"),
        bpm_max: opt_named!(prov, "bpm_max", "provenance.bpm_max", f32, "a float"),
        genre_model_id: opt_named!(
            prov,
            "genre_model_id",
            "provenance.genre_model_id",
            String,
            "a string"
        ),
        vocalness_model_id: opt_named!(
            prov,
            "vocalness_model_id",
            "provenance.vocalness_model_id",
            String,
            "a string"
        ),
        aggression_model_id: opt_named!(
            prov,
            "aggression_model_id",
            "provenance.aggression_model_id",
            String,
            "a string"
        ),
    };

    // --- structured sub-objects ---
    let chord_events = match cached.get_item("chord_events")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => {
            let expected = "a list of {label, start_sec, end_sec} dicts";
            let items = v
                .extract::<Vec<Bound<'_, PyAny>>>()
                .map_err(|_| bad_field("chord_events", expected))?;
            let mut events = Vec::with_capacity(items.len());
            for item in &items {
                let label = item.get_item("label").ok().and_then(|x| x.extract().ok());
                let start = item
                    .get_item("start_sec")
                    .ok()
                    .and_then(|x| x.extract().ok());
                let end = item.get_item("end_sec").ok().and_then(|x| x.extract().ok());
                match (label, start, end) {
                    (Some(label), Some(start_sec), Some(end_sec)) => events.push(rs::ChordEvent {
                        label,
                        start_sec,
                        end_sec,
                    }),
                    _ => return Err(bad_field("chord_events", expected)),
                }
            }
            Some(events)
        }
    };
    let segments = match cached.get_item("segments")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => {
            let expected = "a list of {start_sec, end_sec, energy} dicts";
            let items = v
                .extract::<Vec<Bound<'_, PyAny>>>()
                .map_err(|_| bad_field("segments", expected))?;
            let mut segs = Vec::with_capacity(items.len());
            for item in &items {
                let start = item
                    .get_item("start_sec")
                    .ok()
                    .and_then(|x| x.extract().ok());
                let end = item.get_item("end_sec").ok().and_then(|x| x.extract().ok());
                let energy = item.get_item("energy").ok().and_then(|x| x.extract().ok());
                match (start, end, energy) {
                    (Some(start_sec), Some(end_sec), Some(energy)) => segs.push(rs::SegmentEvent {
                        start_sec,
                        end_sec,
                        energy,
                    }),
                    _ => return Err(bad_field("segments", expected)),
                }
            }
            Some(segs)
        }
    };
    let tags = match cached.get_item("tags")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => {
            let td = v
                .cast_into::<PyDict>()
                .map_err(|_| bad_field("tags", "a dict"))?;
            Some(sonara::analyze::TrackTags {
                title: opt_named!(td, "title", "tags.title", String, "a string"),
                artist: opt_named!(td, "artist", "tags.artist", String, "a string"),
                album: opt_named!(td, "album", "tags.album", String, "a string"),
                genre: opt_named!(td, "genre", "tags.genre", String, "a string"),
                year: opt_named!(td, "year", "tags.year", u32, "an int"),
                original_year: opt_named!(td, "original_year", "tags.original_year", u32, "an int"),
                track_no: opt_named!(td, "track_no", "tags.track_no", u32, "an int"),
            })
        }
    };
    let bpm_candidates = match cached.get_item("bpm_candidates")? {
        None => Vec::new(),
        Some(v) if v.is_none() => Vec::new(),
        Some(v) => {
            let expected = "a list of (bpm, score) pairs";
            let items = v
                .extract::<Vec<Bound<'_, PyAny>>>()
                .map_err(|_| bad_field("bpm_candidates", expected))?;
            items
                .iter()
                .map(|item| pair_f32(item).ok_or_else(|| bad_field("bpm_candidates", expected)))
                .collect::<PyResult<Vec<_>>>()?
        }
    };
    let key_candidates = match cached.get_item("key_candidates")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => {
            let expected = "a list of (key, camelot, score) triples";
            let items = v
                .extract::<Vec<Bound<'_, PyAny>>>()
                .map_err(|_| bad_field("key_candidates", expected))?;
            Some(
                items
                    .iter()
                    .map(|item| {
                        key_candidate(item).ok_or_else(|| bad_field("key_candidates", expected))
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )
        }
    };
    // The dict stores the fingerprint as base64 + a format version; the struct
    // stores decoded sub-fingerprints in this build's format.
    let fingerprint = match opt_field!(cached, "fingerprint", String, "a base64 string") {
        None => None,
        Some(s) => Some(sonara::fingerprint::decode_base64(&s).ok_or_else(|| {
            bad_field("fingerprint", "a valid base64-encoded fingerprint string")
        })?),
    };
    if fingerprint.is_some() {
        if let Some(v) = opt_field!(cached, "fingerprint_version", u32, "an int") {
            if v != sonara::fingerprint::FINGERPRINT_VERSION {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cached analysis carries fingerprint_version {v}, but this build reads \
                     fingerprint format {}; drop the fingerprint field or re-analyze",
                    sonara::fingerprint::FINGERPRINT_VERSION
                )));
            }
        }
    }

    Ok(rs::TrackAnalysis {
        provenance,
        duration_sec: core_field!(cached, "duration_sec", f32, "a float", 0.0),
        bpm: core_field!(cached, "bpm", f32, "a float", 0.0),
        bpm_raw: core_field!(cached, "bpm_raw", f32, "a float", 0.0),
        bpm_confidence: core_field!(cached, "bpm_confidence", f32, "a float", 0.0),
        bpm_candidates,
        beats: core_field!(cached, "beats", Vec<usize>, "a list of ints", Vec::new()),
        onset_frames: core_field!(
            cached,
            "onset_frames",
            Vec<usize>,
            "a list of ints",
            Vec::new()
        ),
        rms_mean: core_field!(cached, "rms_mean", f32, "a float", 0.0),
        rms_max: core_field!(cached, "rms_max", f32, "a float", 0.0),
        loudness_lufs: core_field!(cached, "loudness_lufs", f32, "a float", 0.0),
        dynamic_range_db: core_field!(cached, "dynamic_range_db", f32, "a float", 0.0),
        true_peak_db: opt_field!(cached, "true_peak_db", f32, "a float"),
        replaygain_db: opt_field!(cached, "replaygain_db", f32, "a float"),
        loudness_curve: opt_field!(cached, "loudness_curve", Vec<f32>, "a list of floats"),
        loudness_momentary_max_db: opt_field!(cached, "loudness_momentary_max_db", f32, "a float"),
        loudness_range_lu: opt_field!(cached, "loudness_range_lu", f32, "a float"),
        spectral_centroid_mean: core_field!(cached, "spectral_centroid_mean", f32, "a float", 0.0),
        zero_crossing_rate: core_field!(cached, "zero_crossing_rate", f32, "a float", 0.0),
        onset_density: core_field!(cached, "onset_density", f32, "a float", 0.0),
        spectral_bandwidth_mean: opt_field!(cached, "spectral_bandwidth_mean", f32, "a float"),
        spectral_rolloff_mean: opt_field!(cached, "spectral_rolloff_mean", f32, "a float"),
        spectral_flatness_mean: opt_field!(cached, "spectral_flatness_mean", f32, "a float"),
        spectral_contrast_mean: opt_field!(
            cached,
            "spectral_contrast_mean",
            Vec<f32>,
            "a list of floats"
        ),
        mfcc_mean: opt_field!(cached, "mfcc_mean", Vec<f32>, "a list of floats"),
        chroma_mean: opt_field!(cached, "chroma_mean", Vec<f32>, "a list of floats"),
        tempo_curve: opt_field!(cached, "tempo_curve", Vec<f32>, "a list of floats"),
        tempo_variability: opt_field!(cached, "tempo_variability", f32, "a float"),
        time_signature: opt_field!(cached, "time_signature", String, "a string"),
        time_signature_confidence: opt_field!(cached, "time_signature_confidence", f32, "a float"),
        chord_sequence: opt_field!(cached, "chord_sequence", Vec<String>, "a list of strings"),
        chord_events,
        chord_change_rate: opt_field!(cached, "chord_change_rate", f32, "a float"),
        predominant_chord: opt_field!(cached, "predominant_chord", String, "a string"),
        dissonance: opt_field!(cached, "dissonance", f32, "a float"),
        energy: opt_field!(cached, "energy", f32, "a float"),
        danceability: opt_field!(cached, "danceability", f32, "a float"),
        key: opt_field!(cached, "key", String, "a string"),
        key_confidence: opt_field!(cached, "key_confidence", f32, "a float"),
        key_camelot: opt_field!(cached, "key_camelot", String, "a string"),
        valence: opt_field!(cached, "valence", f32, "a float"),
        acousticness: opt_field!(cached, "acousticness", f32, "a float"),
        embedding: opt_field!(cached, "embedding", Vec<f32>, "a list of floats"),
        // The aggression fields are emitted together, keyed on a present
        // confidence; `aggression_score` itself may be a stored None (abstain).
        aggression_score: opt_field!(cached, "aggression_score", f32, "a float or None"),
        aggression_confidence: opt_field!(cached, "aggression_confidence", f32, "a float"),
        aggression_forcefulness: opt_field!(cached, "aggression_forcefulness", f32, "a float"),
        aggression_harshness: opt_field!(cached, "aggression_harshness", f32, "a float"),
        aggression_tension: opt_field!(cached, "aggression_tension", f32, "a float"),
        aggression_rhythm: opt_field!(cached, "aggression_rhythm", f32, "a float"),
        mood_happy: opt_field!(cached, "mood_happy", f32, "a float"),
        mood_aggressive: opt_field!(cached, "mood_aggressive", f32, "a float"),
        mood_relaxed: opt_field!(cached, "mood_relaxed", f32, "a float"),
        mood_sad: opt_field!(cached, "mood_sad", f32, "a float"),
        instrumentalness: opt_field!(cached, "instrumentalness", f32, "a float"),
        genre: opt_field!(cached, "genre", String, "a string"),
        genre_confidence: opt_field!(cached, "genre_confidence", f32, "a float"),
        grid_offset_sec: opt_field!(cached, "grid_offset_sec", f32, "a float"),
        downbeats: opt_field!(cached, "downbeats", Vec<usize>, "a list of ints"),
        grid_stability: opt_field!(cached, "grid_stability", f32, "a float"),
        energy_curve: opt_field!(cached, "energy_curve", Vec<f32>, "a list of floats"),
        energy_curve_hop_sec: opt_field!(cached, "energy_curve_hop_sec", f32, "a float"),
        segments,
        intro_end_sec: opt_field!(cached, "intro_end_sec", f32, "a float"),
        outro_start_sec: opt_field!(cached, "outro_start_sec", f32, "a float"),
        energy_level: opt_field!(cached, "energy_level", u8, "an int"),
        leading_silence_sec: opt_field!(cached, "leading_silence_sec", f32, "a float"),
        trailing_silence_sec: opt_field!(cached, "trailing_silence_sec", f32, "a float"),
        key_candidates,
        vocalness: opt_field!(cached, "vocalness", f32, "a float"),
        fingerprint,
        embedding_version: opt_field!(cached, "embedding_version", u32, "an int"),
        tags,
    })
}

/// Stable lowercase name for a [`rs::DependencyClass`] (part of the Python
/// API: the `class` value in `feature_dependencies()` rows).
fn class_name(class: rs::DependencyClass) -> &'static str {
    match class {
        rs::DependencyClass::Audio => "audio",
        rs::DependencyClass::FrameCurves => "frame_curves",
        rs::DependencyClass::Scalars => "scalars",
        rs::DependencyClass::Embedding => "embedding",
    }
}

/// Stable descriptive string for an [`rs::AugmentBlocker`] (the Python
/// `augment_blocker` return value; a structured form can come later).
fn blocker_to_string(blocker: &rs::AugmentBlocker) -> String {
    match blocker {
        rs::AugmentBlocker::UnknownFeature => "unknown feature".to_string(),
        rs::AugmentBlocker::NeedsAudio(class) => {
            format!("needs audio ({}-class feature)", class_name(*class))
        }
        rs::AugmentBlocker::SchemaVersionMismatch { record, current } => {
            format!("schema version mismatch (record {record}, current {current})")
        }
        rs::AugmentBlocker::EmbeddingVersionMismatch { record, current } => {
            format!("embedding version mismatch (record {record}, current {current})")
        }
        rs::AugmentBlocker::MissingEvidence(fields) => {
            format!("missing evidence: {}", fields.join(", "))
        }
    }
}

/// Recompute the named features onto a copy of a cached analysis dict —
/// decode-free where the record's evidence allows, via one re-analysis of
/// `audio_path` (at the record's own sample rate) otherwise. See the Rust
/// `sonara::analyze::augment_analysis` docs for the full contract; the Python
/// wrapper in `python/sonara/__init__.py` documents the surface.
#[pyfunction]
#[pyo3(name = "augment_analysis", signature = (cached, features=None, *, audio_path=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_augment_analysis<'py>(
    py: Python<'py>,
    cached: &Bound<'py, PyDict>,
    features: Option<Vec<String>>,
    audio_path: Option<String>,
    bpm_min: Option<f32>,
    bpm_max: Option<f32>,
    genre_model: Option<String>,
    vocalness_model: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let record = analysis_from_dict(cached)?;
    // Reuse the analyzers' model-loading path; the mode is irrelevant here
    // (augment's audio fallback requests an explicit feature list, which
    // overrides any mode).
    let config = parse_config(
        "compact",
        None,
        bpm_min,
        bpm_max,
        genre_model,
        vocalness_model,
    )?;
    let names = features.unwrap_or_default();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let result = rs::augment_analysis(
        &record,
        &name_refs,
        audio_path.as_deref().map(Path::new),
        &config,
    )
    .into_pyresult()?;
    result_to_dict(py, &result)
}

/// Can `feature` be recomputed decode-free from this cached analysis dict?
/// `False` for unknown names; `augment_blocker` returns the reason.
#[pyfunction]
#[pyo3(name = "can_augment")]
pub fn py_can_augment(cached: &Bound<'_, PyDict>, feature: &str) -> PyResult<bool> {
    let record = analysis_from_dict(cached)?;
    Ok(rs::can_augment(&record, feature))
}

/// Why `feature` cannot be recomputed decode-free from this cached analysis
/// dict (a stable descriptive string), or `None` when it can.
#[pyfunction]
#[pyo3(name = "augment_blocker")]
pub fn py_augment_blocker(cached: &Bound<'_, PyDict>, feature: &str) -> PyResult<Option<String>> {
    let record = analysis_from_dict(cached)?;
    Ok(rs::augment_blocker(&record, feature)
        .as_ref()
        .map(blocker_to_string))
}

/// The declared per-feature dependency map, one dict per public feature in
/// canonical order: {name, class, required_evidence, needs_extended,
/// opt_in_only, full_only}. Consumers persisting analyses plan cache
/// freshness on this map.
#[pyfunction]
#[pyo3(name = "feature_dependencies")]
pub fn py_feature_dependencies(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    rs::feature_dependencies()
        .map(|dep| {
            let d = PyDict::new(py);
            d.set_item("name", dep.name)?;
            d.set_item("class", class_name(dep.class))?;
            d.set_item("required_evidence", dep.required_evidence.to_vec())?;
            d.set_item("needs_extended", dep.needs_extended)?;
            d.set_item("opt_in_only", dep.opt_in_only)?;
            d.set_item("full_only", dep.full_only)?;
            Ok(d)
        })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_analyze_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_signal, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_batch, m)?)?;
    // --- fingerprint ---
    m.add_function(wrap_pyfunction!(py_fingerprint_match, m)?)?;
    // --- augment lane ---
    m.add_function(wrap_pyfunction!(py_augment_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(py_can_augment, m)?)?;
    m.add_function(wrap_pyfunction!(py_augment_blocker, m)?)?;
    m.add_function(wrap_pyfunction!(py_feature_dependencies, m)?)?;
    Ok(())
}
