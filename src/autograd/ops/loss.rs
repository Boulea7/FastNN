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
