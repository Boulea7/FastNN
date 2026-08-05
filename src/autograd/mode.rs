//! Whether ops attach graph nodes to their outputs.
//!
//! Tracking is on by default. Wrap inference in [`no_grad`] to skip building a
//! graph you will never differentiate — it saves the memory the saved forward
//! values would otherwise hold.

use std::cell::Cell;

thread_local! {
    /// Nesting depth of active `no_grad` scopes. Zero means "record".
    static SUPPRESSED: Cell<usize> = const { Cell::new(0) };
}

/// Whether ops on this thread currently attach graph nodes.
pub fn is_enabled() -> bool {
    SUPPRESSED.with(|s| s.get() == 0)
}

/// Run `f` without building a graph, then restore the previous mode.
///
/// ```
/// # use fastnn::prelude::*;
/// # let model = Linear::new(4, 2);
/// # let x = Tensor::randn(&[1, 4]);
/// let logits = no_grad(|| model.forward(&x));
/// assert!(logits.grad_fn().is_none());
/// ```
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    let _guard = Suppress::new();
    f()
}

/// Restores the previous mode on drop, so a panic inside `no_grad` cannot leave
/// tracking permanently off.
struct Suppress;

impl Suppress {
    fn new() -> Self {
        SUPPRESSED.with(|s| s.set(s.get() + 1));
        Suppress
    }
}

impl Drop for Suppress {
    fn drop(&mut self) {
        SUPPRESSED.with(|s| s.set(s.get() - 1));
    }
}
