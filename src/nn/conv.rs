//! 2-D convolution.

use crate::tensor::{Tensor, Window};

use super::module::Module;
use super::param::Param;

/// Convolution over `[N, C, H, W]`, implemented as unfold-then-multiply.
///
/// [`im2col`](Tensor::im2col) lays every sliding window out as a column, which
/// turns the convolution into one matrix multiply against a `[out_channels,
/// C·kh·kw]` weight. That is why this layer needs no gradient code of its own:
/// the derivatives of `im2col` and `matmul` already compose into the right thing.
pub struct Conv2d {
    pub weight: Param,
    pub bias: Option<Param>,
    window: Window,
    in_channels: usize,
    out_channels: usize,
}

impl Conv2d {
    /// A square kernel with the given stride and padding.
    pub fn new(in_channels: usize, out_channels: usize, kernel: usize, stride: usize, padding: usize) -> Conv2d {
        Conv2d::with_window(in_channels, out_channels, Window::square(kernel, stride, padding), true)
    }

    /// `kernel × kernel`, stride 1, padded to preserve the spatial size.
    pub fn same(in_channels: usize, out_channels: usize, kernel: usize) -> Conv2d {
        assert!(kernel % 2 == 1, "same-padding needs an odd kernel, got {kernel}");
        Conv2d::new(in_channels, out_channels, kernel, 1, kernel / 2)
    }

    /// Full control over the window and whether there is a bias.
    pub fn with_window(in_channels: usize, out_channels: usize, window: Window, bias: bool) -> Conv2d {
        let (kh, kw) = window.kernel;
        let fan_in = in_channels * kh * kw;
        Conv2d {
            weight: Param::new(Tensor::kaiming_uniform(&[out_channels, in_channels, kh, kw], fan_in)),
            bias: bias.then(|| {
                let bound = 1.0 / (fan_in as f32).sqrt();
                Param::new(Tensor::uniform(&[out_channels], -bound, bound))
            }),
            window,
            in_channels,
            out_channels,
        }
    }

    /// Output spatial size for an input of `(height, width)`.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        self.window.output_size(height, width)
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 4, "Conv2d expects [N, C, H, W], got {:?}", input.shape());
        assert_eq!(
            input.dim(1), self.in_channels,
            "Conv2d expects {} channels, got {:?}", self.in_channels, input.shape()
        );

        let (batch, height, width) = (input.dim(0), input.dim(2), input.dim(3));
        let (out_h, out_w) = self.window.output_size(height, width);
        let out_c = self.out_channels as i64;

        // [N, C·kh·kw, out_h·out_w] against [out_channels, C·kh·kw].
        let columns = input.im2col(self.window);
        let kernels = self.weight.tensor().reshape(&[out_c, -1]);
        let mut out = kernels.matmul(&columns);

        if let Some(bias) = &self.bias {
            out = out.add(&bias.tensor().reshape(&[1, out_c, 1]));
        }

        out.reshape(&[batch as i64, out_c, out_h as i64, out_w as i64])
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = vec![("weight".into(), self.weight.clone())];
        if let Some(bias) = &self.bias {
            params.push(("bias".into(), bias.clone()));
        }
        params
    }
}
