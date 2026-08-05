//! Loss functions.
//!
//! Free functions, not layers — a loss has no parameters and no state, so
//! `cross_entropy(&logits, &targets)` says everything a struct would.

use crate::autograd::ops::loss::CrossEntropyBackward;
use crate::tensor::Tensor;

/// Mean cross-entropy between `logits` `[batch, classes]` and integer class targets.
///
/// Takes raw logits, not probabilities: softmax and the log are fused here, so
/// the numerically dangerous `log(exp(...))` never appears and a confidently
/// wrong prediction yields a large finite loss instead of infinity.
pub fn cross_entropy(logits: &Tensor, targets: &[usize]) -> Tensor {
    assert_eq!(logits.ndim(), 2, "cross_entropy expects [batch, classes], got {:?}", logits.shape());
    let (batch, classes) = (logits.dim(0), logits.dim(1));
    assert_eq!(targets.len(), batch, "cross_entropy: {} targets for {batch} rows", targets.len());

    let data = logits.to_vec();
    let mut softmax = vec![0.0f32; batch * classes];
    let mut total = 0.0f32;

    for (row, &target) in targets.iter().enumerate() {
        assert!(target < classes, "cross_entropy: target {target} outside 0..{classes}");
        let base = row * classes;
        let values = &data[base..base + classes];

        // Shift by the row maximum so exp() cannot overflow.
        let shift = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = values.iter().map(|&v| (v - shift).exp()).sum();
        for c in 0..classes {
            softmax[base + c] = (values[c] - shift).exp() / sum;
        }
        total -= values[target] - shift - sum.ln();
    }

    let saved = Tensor::from_vec(softmax, &[batch, classes]);
    let targets = targets.to_vec();
    Tensor::scalar(total / batch as f32)
        .to(logits.device())
        .with_grad(&[logits], || CrossEntropyBackward { softmax: saved, targets, classes })
}

/// Mean squared error.
pub fn mse(prediction: &Tensor, target: &Tensor) -> Tensor {
    prediction.sub(target).square().mean()
}

/// Mean absolute error. Less sensitive to outliers than [`mse`].
pub fn mae(prediction: &Tensor, target: &Tensor) -> Tensor {
    prediction.sub(target).abs().mean()
}

/// Binary cross-entropy over probabilities already in `[0, 1]`.
///
/// Predictions are clamped away from the endpoints, because `log(0)` is `-inf`
/// and one saturated output would poison the whole batch. Prefer
/// [`bce_with_logits`] when you have raw scores.
pub fn bce(prediction: &Tensor, target: &Tensor) -> Tensor {
    const EDGE: f32 = 1e-7;
    let p = prediction.clamp(EDGE, 1.0 - EDGE);
    let positive = target.mul(&p.log());
    let negative = target.neg().add_scalar(1.0).mul(&p.neg().add_scalar(1.0).log());
    positive.add(&negative).neg().mean()
}

/// Binary cross-entropy straight from logits.
///
/// Uses `max(x,0) - x·t + log(1 + e^-|x|)`, which is the same value as
/// `bce(sigmoid(x), t)` but never exponentiates a large positive number.
pub fn bce_with_logits(logits: &Tensor, target: &Tensor) -> Tensor {
    let floor = logits.clamp(0.0, f32::INFINITY);
    let stable_log = logits.abs().neg().exp().add_scalar(1.0).log();
    floor.sub(&logits.mul(target)).add(&stable_log).mean()
}
