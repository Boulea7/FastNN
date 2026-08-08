//! Derivative of the fused cross-entropy loss.

use crate::autograd::Backward;
use crate::tensor::Tensor;

/// `d/dlogits mean(-log softmax(logits)[target]) = (softmax - one_hot) / batch`.
///
/// Fusing softmax and the negative log likelihood is what makes this one
/// subtraction. Done separately, the same gradient would go through a log, a
/// division, and a gather, each amplifying the rounding error of the last.
pub struct CrossEntropyBackward {
    pub softmax: Tensor,
    pub targets: Vec<usize>,
    pub classes: usize,
}

impl Backward for CrossEntropyBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let batch = self.targets.len();
        let scale = grad.item() / batch as f32;
        let mut out = self.softmax.to_vec();

        for (row, &target) in self.targets.iter().enumerate() {
            out[row * self.classes + target] -= 1.0;
        }
        for value in &mut out {
            *value *= scale;
        }

        vec![Tensor::from_vec(out, &[batch, self.classes]).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "CrossEntropy"
    }
}

/// Gradient of the configurable cross-entropy.
///
/// The forward already computed `w·(p − q)` per row — softmax minus the
/// (possibly smoothed) target distribution, scaled by that row's class weight,
/// zero for ignored rows. All that remains is the reduction's scaling: the
/// weighted-mean denominator for `Mean`, nothing for `Sum`, and a per-row
/// upstream gradient for `None`.
pub struct WeightedCrossEntropyBackward {
    pub difference: Tensor,
    pub normalizer: f32,
    pub per_row: bool,
}

impl Backward for WeightedCrossEntropyBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let difference = self.difference.to(grad.device());
        let scaled = if self.per_row {
            // Upstream is one gradient per row; broadcast it across classes.
            let rows = grad.numel() as i64;
            difference.mul(&grad.reshape(&[rows, 1]))
        } else {
            difference.mul_scalar(grad.item() / self.normalizer)
        };
        vec![scaled]
    }
    fn name(&self) -> &'static str {
        "WeightedCrossEntropy"
    }
}
