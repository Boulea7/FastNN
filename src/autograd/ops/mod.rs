//! Backward rules, mirroring the layout of [`tensor::ops`](crate::tensor::ops).
//!
//! A rule is a small struct holding what the derivative needs, plus one
//! [`Backward::backward`](super::Backward::backward) method. They run with
//! tracking suppressed, so they can use ordinary tensor ops freely.

pub mod activation;
pub mod arith;
pub mod conv;
pub mod index;
pub mod loss;
pub mod matmul;
pub mod norm;
pub mod pool;
pub mod reduce;
pub mod unary;
pub mod view;

use crate::tensor::Tensor;

/// Undo broadcasting: sum `grad` down to `target`.
///
/// A forward op that broadcast a `[1, n]` bias across a batch produced a
/// `[batch, n]` output, so its gradient arrives at `[batch, n]` and every row
/// contributed. Summing the stretched axes is the adjoint of stretching them.
pub(crate) fn reduce_to(grad: &Tensor, target: &[usize]) -> Tensor {
    if grad.shape() == target {
        return grad.clone();
    }

    let mut out = grad.clone();
    while out.ndim() > target.len() {
        out = out.sum_axis(0);
    }
    for (axis, (&have, &want)) in out.shape().to_vec().iter().zip(target).enumerate() {
        if want == 1 && have != 1 {
            out = out.sum_axis_keep(axis);
        }
    }

    let dims: Vec<i64> = target.iter().map(|&d| d as i64).collect();
    out.reshape(&dims)
}
