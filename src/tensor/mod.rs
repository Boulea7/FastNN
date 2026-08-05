//! Tensors: the data structure everything else is built on.
//!
//! - [`Tensor`] — shape, storage, and its link into the autograd graph (`core.rs`)
//! - [`Device`] — CPU or a numbered GPU
//! - [`ops`] — every operation, grouped by what it does
//! - [`shape`] — strides, broadcasting, and index arithmetic
//! - [`storage`] — the bytes, on one device or the other
//!
//! ```
//! use fastnn::prelude::*;
//!
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
//! let y = x.matmul(&x).relu().sum();
//! assert_eq!(y.item(), 54.0);  // [[7, 10], [15, 22]] summed
//! ```

mod checked;
mod core;
mod device;
mod display;
mod init;
pub mod ops;
pub mod shape;
pub mod storage;

pub use core::Tensor;
pub use device::Device;
pub use ops::conv::Window;
pub use ops::norm::BatchStats;
