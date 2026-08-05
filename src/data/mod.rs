//! Datasets and batching.
//!
//! Implement [`Dataset`] for your own data, then wrap it in a [`DataLoader`] to
//! get shuffling, batching, and device placement.

pub mod dataset;
pub mod loader;
pub mod mnist;

pub use dataset::{Dataset, TensorDataset};
pub use loader::{Batch, Batches, DataLoader};
pub use mnist::{Mnist, Split};
