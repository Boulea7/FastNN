# FastNN

A deep learning library in Rust, with CUDA kernels for the parts that matter.

Tensors with autograd, the usual layers, optimizers, data loading, and
checkpoints — small enough to read end to end, and it actually trains.

```rust
use fastnn::prelude::*;
use fastnn::data::{Mnist, Split};

let train = Mnist::load(Split::Train)?;
let loader = DataLoader::new(&train, 128).shuffle(true);

let model = Sequential::new()
    .add(Flatten::new())
    .add(Linear::new(784, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

let mut opt = Adam::new(model.parameters(), 1e-3);

for batch in loader.iter() {
    let loss = cross_entropy(&model.forward(&batch.inputs), &batch.labels());

    opt.zero_grad();
    loss.backward();
    opt.step();
}

save(&model, "mnist.fdl")?;
```

No gradient-mode flags, no borrow dance around the optimizer. That is the whole
training loop.

## Getting started

```bash
# CPU only — no CUDA toolkit needed
cargo run --example simple_mlp --no-default-features --release

# With a GPU
cargo run --example mnist_mlp --release
```

Always use `--release`. Debug builds are roughly 50× slower.

| Example | What it shows |
|---|---|
| `simple_mlp` | The smallest complete training loop (XOR) |
| `mnist_mlp` | Dataset loading, batching, evaluation, checkpointing |
| `mnist_cnn` | Convolutions, batch norm, pooling, an LR schedule |
| `char_lm` | A GPT-style transformer, trained from scratch, that generates text |

`char_lm` takes a corpus path, or trains on a small embedded one:

```bash
curl -o shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cargo run --example char_lm --release -- shakespeare.txt
```

## How it fits together

```
  data      Dataset ─► DataLoader ─► Batch
                                       │
  nn        Module ──► forward ────────┤    layers hold Param handles
                                       ▼
  tensor    Tensor ops on CPU or CUDA ─┴─► loss
                                       │
  autograd  loss.backward() walks the graph the ops built,
            depositing gradients in each Param's slot
                                       │
  optim     opt.step() reads those slots and updates in place
```

Everything starts on the CPU. `model.to_device(device)` moves a model,
`DataLoader::to_device` moves its batches, and mixing the two panics rather than
copying behind your back.

```rust
let device = Device::best();       // GPU if there is one, else CPU
model.to_device(device);
let loader = DataLoader::new(&dataset, 64).to_device(device);
```

### Two ideas worth knowing

**The graph is the tensors.** There is no tape and no global state. A tensor
produced by a differentiable op carries a node naming the rule that made it and
the tensors it consumed, so `loss.backward()` just walks that structure. Drop the
loss and the graph frees itself. A tensor with no node is plain data, so there is
no `requires_grad` flag to keep in sync — wrap inference in `no_grad(|| ...)`
when you want to skip building a graph at all.

**Parameters are shared slots.** `Param` is a handle, not a value. The model and
the optimizer hold handles to the same weight and the same gradient, which is why
`opt.step()` needs no borrow of the model. It is also why weight tying works
without special support: use the same handle twice and both paths' gradients add
up on their own.

## Layout

Every file is one idea, and none are long. 67 files, ~7,000 lines.

```
src/
  tensor/          the array type and everything you can do to it
    core.rs          Tensor: shape, storage, graph link
    shape.rs         strides, broadcasting, index math
    storage.rs       the bytes, on one device or the other
    device.rs        Device::cuda(0) -> Result
    init.rs          zeros, randn, kaiming, xavier, ...
    ops/             one file per category
      arith  unary  activation  matmul  reduce  view  index
      conv  pool  norm

  autograd/        reverse-mode differentiation
    node.rs          Backward trait, graph nodes, gradient slots
    engine.rs        the reverse pass
    mode.rs          no_grad
    ops/             one backward rule per forward op, same file names

  nn/              layers, all implementing Module
    module.rs param.rs sequential.rs linear.rs conv.rs pooling.rs
    norm.rs activation.rs dropout.rs shape.rs embedding.rs
    attention.rs transformer.rs rnn.rs loss.rs

  optim/           sgd.rs  adam.rs  schedule.rs
  data/            dataset.rs  loader.rs  mnist.rs
  serialize/       checkpoint.rs
  cuda/            ffi.rs (raw bindings)  kernels.rs (safe wrappers)  buffer.rs
  rng.rs  error.rs  lib.rs

cuda/kernels.cu    all GPU kernels
```

`tensor/ops/` and `autograd/ops/` mirror each other file for file: the forward
for `relu` is in `tensor/ops/activation.rs`, its derivative in
`autograd/ops/activation.rs`.

## Extending it

**A new layer.** Implement `forward`, plus `named_parameters` if it has weights.
Device placement, parameter counting, and gradient zeroing come from those.

```rust
struct Residual { inner: Linear }

impl Module for Residual {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.add(&self.inner.forward(x).relu())
    }
    fn named_parameters(&self) -> Vec<(String, Param)> {
        scoped("inner", self.inner.named_parameters())
    }
}
```

**A new op.** Write the forward in `tensor/ops/`, the rule in `autograd/ops/`,
attach it with `with_grad`, and add a check to `tests/gradcheck.rs`.

```rust
// tensor/ops/unary.rs
pub fn softplus(&self) -> Tensor {
    unary_op(self, |x| x.exp().ln_1p(), ffi::fastnn_cuda_softplus)
        .with_grad(&[self], || SoftplusBackward { input: self.detach() })
}

// autograd/ops/unary.rs
impl Backward for SoftplusBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![elementwise(grad, &self.input, |x| 1.0 / (1.0 + (-x).exp()))]
    }
    fn name(&self) -> &'static str { "Softplus" }
}
```

The rule's gradients must come back in the same order as the inputs. Save
detached tensors — a saved value that still carries its own node would pin the
graph that produced it.

**A new dataset.** Implement `Dataset` and hand it to a `DataLoader`. Items are
written into caller-provided slices rather than returned as tensors, so batching
60,000 images does not build 60,000 throwaway ones.

## Errors

Tensor maths panics on shape and device mistakes. Those are bugs in the calling
code, like indexing past the end of a slice, and threading `Result` through every
`add` would bury the model in `?` for no safety gained.

`Result` is for what genuinely fails at runtime:

```rust
let device = Device::cuda(0)?;      // no GPU
let data   = Mnist::load(Split::Train)?;  // download or parse failed
load(&model, "model.fdl")?;          // missing file, or a shape that moved
```

## Testing

```bash
cargo test --no-default-features --test gradcheck   # every backward rule
cargo test --test cuda_parity -- --test-threads=1   # CPU vs GPU
cargo test --no-default-features                    # everything
cargo bench --no-default-features                   # throughput
```

`gradcheck` checks every backward rule against central finite differences of its
own forward. It is the reason the autograd layer can be trusted; a new rule
without a test there is not finished.

`cuda_parity` runs each op on both devices and compares. It skips itself with a
note when there is no GPU, so a CPU-only machine still gets a green run. It found
a real bug in the softmax kernel: the block reduction assumed a power-of-two
thread count and silently dropped the tail of every row whose width was not one —
which included MNIST's ten classes and any odd sequence length.

## CUDA

Needs the NVIDIA CUDA Toolkit; set `CUDA_PATH` or `CUDA_HOME` if it is not in the
default location. `build.rs` compiles `cuda/kernels.cu` with `nvcc` and links
`cudart`, `cublas`, and `curand`. Compute capabilities 7.5 through 9.0 (Turing
through Hopper).

`--no-default-features` skips all of that: `cuda/stubs.c` supplies the symbols,
`Device::cuda(0)` returns an error, and everything runs on the CPU.

Two things carry most of the GPU performance. Matrix multiplication goes through
cuBLAS, and the two transposed forms the backward pass needs (`matmul_nt`,
`matmul_tn`) use a transpose flag rather than building a transposed copy. GPU
allocations come from a size-keyed free list, so the many short-lived temporaries
a training step creates cost a hash lookup instead of a driver round trip.

## Limitations

- `f32` only.
- Convolution unfolds to a matrix multiply, and the unfold itself runs on the
  host. The multiply is on the GPU; the `im2col` around it is not.
- `LSTM` and `GRU` are built from ordinary differentiable ops, one graph node per
  gate per timestep. Correct, and fine for short sequences — use a transformer
  for long ones.
- Tensors are always contiguous. `permute` and `expand` write a new buffer rather
  than returning a strided view.

## License

MIT.
