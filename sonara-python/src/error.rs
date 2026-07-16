use sonara::SonaraError;
use pyo3::exceptions;
use pyo3::PyErr;

/// Convert a SonaraError into a PyErr.
/// We use a function instead of `impl From` to avoid the orphan rule.
pub fn to_pyerr(err: SonaraError) -> PyErr {
    match &err {
        SonaraError::InvalidParameter { .. }
        | SonaraError::ShapeMismatch { .. }
        | SonaraError::InvalidAudio(_)
        | SonaraError::ModelError(_)
        | SonaraError::InsufficientData { .. } => {
            exceptions::PyValueError::new_err(err.to_string())
        }
        SonaraError::AudioFile(_) | SonaraError::Decode(_) => {
            exceptions::PyIOError::new_err(err.to_string())
        }
        SonaraError::UnsupportedFormat(_) => {
            exceptions::PyNotImplementedError::new_err(err.to_string())
        }
        _ => exceptions::PyRuntimeError::new_err(err.to_string()),
    }
}

/// A short, stable category string for a `SonaraError`.
///
/// Used by `analyze_batch` to attach a machine-readable `error_kind` to each
/// per-file failure so callers can branch on the failure type (skip, retry,
/// re-encode, …) without parsing human-readable messages. These strings are
/// part of the public API — keep them stable.
pub fn error_kind(err: &SonaraError) -> &'static str {
    match err {
        // File could not be opened/read (missing path, permissions, truncated I/O).
        SonaraError::AudioFile(_) => "io",
        // Bitstream/container recognized but could not be decoded.
        SonaraError::Decode(_) => "decode",
        // No registered demuxer/codec for this container or codec.
        SonaraError::UnsupportedFormat(_) => "unsupported_format",
        // Caller-supplied parameters or signal shape were invalid.
        SonaraError::InvalidParameter { .. }
        | SonaraError::ShapeMismatch { .. }
        | SonaraError::InvalidAudio(_) => "invalid_audio",
        // A supplied genre model failed to load/validate or was version-mismatched.
        SonaraError::ModelError(_) => "model",
        // Audio decoded but was too short for the requested analysis.
        SonaraError::InsufficientData { .. } => "insufficient_data",
        // Downstream numerical/DSP computation failure.
        SonaraError::Fft(_)
        | SonaraError::ConvergenceFailed { .. }
        | SonaraError::Numerical(_)
        | SonaraError::NoPitchDetected => "compute",
    }
}

/// Extension trait to convert sonara Result to PyResult.
pub trait IntoPyResult<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T>;
}

impl<T> IntoPyResult<T> for sonara::Result<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T> {
        self.map_err(to_pyerr)
    }
}
