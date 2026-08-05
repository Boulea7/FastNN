//! Activations as layers, for use inside [`Sequential`](super::Sequential).
//!
//! Each one is a thin wrapper over the corresponding [`Tensor`] method — call
//! that directly when you are writing a `forward` by hand.

use crate::tensor::Tensor;

use super::module::Module;

/// Declare a stateless activation layer that forwards to one tensor method.
macro_rules! activation {
    ($(#[$doc:meta])* $name:ident => $method:ident) => {
        $(#[$doc])*
        pub struct $name;

        impl Module for $name {
            fn forward(&self, input: &Tensor) -> Tensor {
                input.$method()
            }
        }
    };
}

activation!(/// `max(0, x)`.
            ReLU => relu);
activation!(/// `1 / (1 + e^-x)`.
            Sigmoid => sigmoid);
activation!(/// Hyperbolic tangent.
            Tanh => tanh);
activation!(/// Exact GELU — the transformer default.
            GELU => gelu);
activation!(/// SiLU / swish, `x · sigmoid(x)`.
            SiLU => silu);
activation!(/// Softmax over the last dimension.
            ///
            /// Not needed before [`cross_entropy`](super::cross_entropy), which
            /// applies its own and is more stable for it.
            Softmax => softmax);

/// ReLU with a non-zero slope for negative inputs, so units cannot go fully dead.
pub struct LeakyReLU {
    pub slope: f32,
}

impl LeakyReLU {
    pub fn new(slope: f32) -> LeakyReLU {
        LeakyReLU { slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        LeakyReLU { slope: 0.01 }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.leaky_relu(self.slope)
    }
}
