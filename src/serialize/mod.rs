//! Checkpointing models to disk.

pub mod checkpoint;

pub use checkpoint::{load, load_tensors, save, save_tensors, Checkpoint};
