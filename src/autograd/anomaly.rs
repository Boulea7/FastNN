//! Catching a NaN at the op that produced it.
//!
//! A single non-finite gradient spreads: the optimizer writes it into a
//! parameter, the next forward pass turns every activation downstream into NaN,
//! and the loss only *prints* as NaN some steps later — by which point the
//! checkpoint on disk is poisoned too and the op actually at fault is long gone.
//!
//! Wrapping a step in [`detect_anomaly`] makes the reverse pass check each rule's
//! output and fail immediately, naming the rule.
//!
//! ```should_panic
//! # use fastnn::prelude::*;
//! # use fastnn::autograd::detect_anomaly;
//! let x = Param::new(Tensor::zeros(&[2, 2]));
//!
//! detect_anomaly(|| {
//!     // d/dx log(x) = 1/x, which is infinite at x = 0
//!     x.tensor().log().sum().backward();   // panics naming "Log"
//! });
//! ```
//!
//! It is a debugging tool, not something to leave on: the check reads every
//! gradient, which on a GPU means downloading each one to the host.

use std::cell::Cell;

use crate::tensor::Tensor;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Whether the reverse pass is checking gradients on this thread.
pub fn is_detecting() -> bool {
    ENABLED.with(|flag| flag.get())
}

/// Run `f` with gradient checking on, then restore the previous setting.
pub fn detect_anomaly<T>(f: impl FnOnce() -> T) -> T {
    let _guard = Enabled::new();
    f()
}

/// Panic if `grads` holds a non-finite value, naming `op` and which input it was.
///
/// Called by the engine after each rule while detection is on.
pub(crate) fn check(op: &str, grads: &[Tensor]) {
    for (index, grad) in grads.iter().enumerate() {
        if let Some(report) = first_bad(grad) {
            panic!("anomaly: {op} produced {report} in the gradient for input {index}");
        }
    }
}

/// Describe the first non-finite value in `t`, if there is one.
fn first_bad(t: &Tensor) -> Option<String> {
    let data = t.to_vec();
    let (slot, value) = data.iter().enumerate().find(|(_, v)| !v.is_finite())?;
    let kind = if value.is_nan() { "NaN" } else { "inf" };
    Some(format!("{kind} at element {slot} of {:?}", t.shape()))
}

/// Restores the previous setting on drop, so a panic inside `detect_anomaly`
/// cannot leave checking permanently on.
struct Enabled(bool);

impl Enabled {
    fn new() -> Self {
        Enabled(ENABLED.with(|flag| flag.replace(true)))
    }
}

impl Drop for Enabled {
    fn drop(&mut self) {
        ENABLED.with(|flag| flag.set(self.0));
    }
}

impl Tensor {
    /// Whether every element is finite — no NaN, no infinity.
    ///
    /// Reads the whole tensor, and downloads it first if it is on a GPU. Use it
    /// at checkpoints and guard rails, not inside a training loop.
    pub fn is_finite(&self) -> bool {
        self.to_vec().iter().all(|v| v.is_finite())
    }
}
