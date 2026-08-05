//! Optimizers and learning-rate schedules.
//!
//! An optimizer owns [`Param`] handles, which are shared slots rather than
//! copies. That is what lets the training loop read:
//!
//! ```
//! use fastnn::prelude::*;
//! # let model = Sequential::new().add(Linear::new(4, 2));
//! # let x = Tensor::randn(&[3, 4]);
//! let mut opt = Adam::new(model.parameters(), 1e-3);
//!
//! let loss = cross_entropy(&model.forward(&x), &[0, 1, 0]);
//! opt.zero_grad();
//! loss.backward();
//! opt.step();
//! ```
//!
//! with no borrow of the model at `step()` time and no parameter list threaded
//! through every call.
//!
//! Updates run on whichever device the parameters live on: they are written with
//! tensor ops, so a GPU model never round-trips to the host to take a step.

pub mod adam;
pub mod schedule;
pub mod sgd;

pub use adam::{Adam, AdamW};
pub use schedule::{CosineAnnealing, Constant, LrSchedule, OneCycle, StepDecay, Warmup};
pub use sgd::SGD;

use crate::nn::Param;

/// What every optimizer can do.
pub trait Optimizer {
    /// Apply one update using the gradients currently in each parameter.
    ///
    /// Parameters with no gradient are skipped, so a model with an unused branch
    /// still steps cleanly.
    fn step(&mut self);

    /// The parameters this optimizer updates.
    fn parameters(&self) -> &[Param];

    /// The current learning rate.
    fn lr(&self) -> f32;

    /// Set the learning rate — how a [`LrSchedule`] is applied.
    fn set_lr(&mut self, lr: f32);

    /// Clear every gradient. Call before `backward()`, or skip it to accumulate
    /// gradients across several micro-batches.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }
}

/// Scale gradients down so their combined L2 norm is at most `max_norm`.
///
/// Returns the norm *before* clipping, which is worth logging: a sudden spike is
/// usually the first sign of divergence. Call between `backward()` and `step()`.
pub fn clip_grad_norm(params: &[Param], max_norm: f32) -> f32 {
    let total: f32 = params
        .iter()
        .filter_map(|p| p.grad())
        .map(|g| g.square().sum().item())
        .sum::<f32>()
        .sqrt();

    if total > max_norm {
        // The epsilon keeps the scale finite if the norm underflows to zero.
        let scale = max_norm / (total + 1e-6);
        for param in params {
            if let Some(grad) = param.grad() {
                param.set_grad(grad.mul_scalar(scale));
            }
        }
    }
    total
}
