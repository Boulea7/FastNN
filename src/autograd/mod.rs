//! Reverse-mode automatic differentiation.
//!
//! There is no tape and no global state. Every tensor produced by a differentiable
//! op carries a [`Node`] naming the rule that made it and the tensors it consumed,
//! so the graph *is* the tensors. It is freed when the loss is dropped.
//!
//! ```text
//!   w (Leaf) ─┐
//!             ├─► matmul ──► add ──► relu ──► … ──► loss
//!   x (None) ─┘      ▲        ▲
//!                    │        └─ Node::Op { rule: AddBackward, inputs: [.., b] }
//!                    └────────── Node::Op { rule: MatmulBackward, inputs: [x, w] }
//! ```
//!
//! `loss.backward()` walks that structure in reverse and deposits gradients in
//! the [`GradSlot`] of each leaf, which is the same slot the optimizer reads.
//!
//! ## Adding an op
//!
//! 1. Write the forward in `tensor/ops/`.
//! 2. Write a `Backward` rule in [`ops`], saving only what the derivative needs.
//! 3. Attach it with [`Tensor::with_grad`](crate::tensor::Tensor::with_grad).
//! 4. Add a finite-difference check in `tests/gradcheck.rs`.

pub mod engine;
pub mod mode;
pub mod node;
pub mod ops;

pub use engine::backward;
pub use mode::{is_enabled, no_grad};
pub use node::{Backward, GradSlot, Node, Op};
