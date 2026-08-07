//! The inference paths behind text generation: KV-cached decoding must match
//! the plain forward pass, and sampling must be reproducible and honor its
//! filters.

use fastnn::prelude::*;

/// A small causal stack in eval mode, so dropout is identity on both paths.
fn stack() -> TransformerStack {
    manual_seed(7);
    let stack = TransformerStack::causal(32, 4, 64, 2, 0.1);
    stack.set_training(false);
    stack
}

fn assert_close(a: &Tensor, b: &Tensor, what: &str) {
    let (a, b) = (a.to_vec(), b.to_vec());
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(
            (x - y).abs() < 1e-3,
            "{what}: element {i} differs, {x} vs {y}"
        );
    }
}

/// Token-by-token decoding through the cache must produce the same outputs as
/// one uncached forward over the whole sequence.
#[test]
fn cached_decoding_matches_full_forward() {
    let stack = stack();
    let input = Tensor::randn(&[1, 6, 32]);

    let full = no_grad(|| stack.forward(&input));

    let mut cache = stack.new_cache();
    for t in 0..6 {
        let step = no_grad(|| stack.forward_cached(&input.narrow(1, t, 1), &mut cache));
        assert_close(&step, &full.narrow(1, t, 1), &format!("position {t}"));
    }
    assert_eq!(cache.len(), 6);
}

/// A multi-token prefill followed by single-token steps — how a prompt is
/// actually consumed — must also match.
#[test]
fn prefill_then_single_steps_matches_full_forward() {
    let stack = stack();
    let input = Tensor::randn(&[2, 5, 32]);

    let full = no_grad(|| stack.forward(&input));

    let mut cache = stack.new_cache();
    let prefill = no_grad(|| stack.forward_cached(&input.narrow(1, 0, 3), &mut cache));
    assert_close(&prefill, &full.narrow(1, 0, 3), "prefill");

    for t in 3..5 {
        let step = no_grad(|| stack.forward_cached(&input.narrow(1, t, 1), &mut cache));
        assert_close(&step, &full.narrow(1, t, 1), &format!("position {t}"));
    }
}

/// Clearing the cache must give the same results as a fresh one.
#[test]
fn cleared_cache_starts_over() {
    let stack = stack();
    let input = Tensor::randn(&[1, 4, 32]);

    let mut cache = stack.new_cache();
    let first = no_grad(|| stack.forward_cached(&input, &mut cache));

    cache.clear();
    assert!(cache.is_empty());
    let second = no_grad(|| stack.forward_cached(&input, &mut cache));
    assert_close(&first, &second, "after clear");
}

#[test]
fn greedy_picks_the_largest_logit() {
    let logits = Tensor::from_vec(vec![0.1, 3.0, -2.0, 1.5], &[4]);
    assert_eq!(Sampler::greedy().sample(&logits), 1);
    assert_eq!(Sampler::new().temperature(0.0).sample(&logits), 1);
}

#[test]
fn top_k_of_one_is_greedy_at_any_temperature() {
    let logits = Tensor::from_vec(vec![0.1, 3.0, -2.0, 1.5], &[4]);
    let sampler = Sampler::new().temperature(5.0).top_k(1);
    for _ in 0..20 {
        assert_eq!(sampler.sample(&logits), 1);
    }
}

#[test]
fn tiny_top_p_is_greedy() {
    // The most likely token alone already exceeds p, so it is the whole nucleus.
    let logits = Tensor::from_vec(vec![0.1, 3.0, -2.0, 1.5], &[4]);
    let sampler = Sampler::new().top_p(0.01);
    for _ in 0..20 {
        assert_eq!(sampler.sample(&logits), 1);
    }
}

#[test]
fn sampling_is_reproducible_under_manual_seed() {
    let logits = Tensor::from_vec(vec![1.0, 1.1, 0.9, 1.05], &[4]);
    let sampler = Sampler::new().temperature(0.9).top_k(3);

    manual_seed(42);
    let first: Vec<usize> = (0..10).map(|_| sampler.sample(&logits)).collect();
    manual_seed(42);
    let second: Vec<usize> = (0..10).map(|_| sampler.sample(&logits)).collect();
    assert_eq!(first, second);
}

#[test]
fn sampling_respects_the_distribution() {
    // With one logit far above the rest, anything else is (2e-9)-unlikely.
    let logits = Tensor::from_vec(vec![10.0, -10.0, -10.0], &[3]);
    let sampler = Sampler::new();
    for _ in 0..100 {
        assert_eq!(sampler.sample(&logits), 0);
    }
}
