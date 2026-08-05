//! Checkpointing to disk.
//!
//! - [`checkpoint`] — weights, for a finished model
//! - [`training`] — weights plus optimizer state, for resuming a run
//!
//! Both use the same `.fdl` container; a training checkpoint just holds more.

pub mod checkpoint;
pub mod training;

pub use checkpoint::{load, load_tensors, save, save_tensors, Checkpoint};
pub use training::{load_training, save_training};
