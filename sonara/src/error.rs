use thiserror::Error;

/// All errors that sonara operations can produce.
#[derive(Error, Debug)]
pub enum SonaraError {
    // ---- Audio I/O ----
    #[error("Audio file error: {0}")]
    AudioFile(String),

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("Audio decoding error: {0}")]
    Decode(String),

    // ---- Parameter validation ----
    #[error("Invalid parameter `{param}`: {reason}")]
    InvalidParameter {
        param: &'static str,
        reason: String,
    },

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Audio validation failed: {0}")]
    InvalidAudio(String),

    // ---- Computation ----
    #[error("FFT error: {0}")]
    Fft(String),

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error("Numerical error: {0}")]
    Numerical(String),

    // ---- Model loading / inference (bring-your-own genre model) ----
    #[error("Genre model error: {0}")]
    ModelError(String),

    // ---- Feature extraction ----
    #[error("No pitch detected in signal")]
    NoPitchDetected,

    #[error("Insufficient data: need at least {needed} samples, got {got}")]
    InsufficientData { needed: usize, got: usize },
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, SonaraError>;
