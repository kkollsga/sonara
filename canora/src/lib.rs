#[cfg(feature = "accelerate")]
extern crate blas_src;

pub mod analyze;
pub mod beat;
pub mod core;
pub mod decompose;
pub mod dsp;
pub mod effects;
pub mod error;
pub mod feature;
pub mod filters;
pub mod onset;
pub mod segment;
pub mod sequence;
pub mod types;
pub mod util;

// Re-export commonly used items at crate root
pub use error::{CanoraError, Result};
pub use types::*;
