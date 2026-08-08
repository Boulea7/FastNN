//! Checkpointing to disk.
//!
//! - [`checkpoint`] — weights, for a finished model
//! - [`training`] — weights plus optimizer state, for resuming a run
//! - [`safetensors`] — the interchange format the wider ecosystem shares,
//!   including reading `F16`/`BF16` checkpoints widened to `f32`
//! - [`half`] — the bit-exact 16-bit float conversions behind that
//!
//! The first two use the same `.fdl` container; a training checkpoint just
//! holds more.

pub mod checkpoint;
pub mod half;
pub mod safetensors;
pub mod training;

pub use checkpoint::{load, load_tensors, save, save_tensors, Checkpoint};
pub use safetensors::{load_safetensors, load_safetensors_renamed, save_safetensors, LoadReport};
pub use training::{load_training, save_training};
