#[cfg(feature = "accelerate")]
extern crate blas_src;

pub mod analyze;
#[cfg(feature = "aggression")]
pub mod aggression;
pub mod beat;
pub mod beatgrid;
pub mod core;
pub mod decompose;
pub mod dsp;
pub mod effects;
pub mod error;
pub mod feature;
pub mod filters;
pub mod fingerprint;
pub mod genre;
pub mod loudness_ext;
mod mood;
pub mod onset;
pub mod perceptual;
pub mod segment;
pub mod sequence;
pub mod similarity;
pub mod structure;
pub mod tonal;
pub mod types;
pub mod util;
#[cfg(feature = "validation")]
pub mod validation;
pub mod vocal;
pub mod vocal_model;

// Re-export commonly used items at crate root
pub use error::{Result, SonaraError};
pub use types::*;
