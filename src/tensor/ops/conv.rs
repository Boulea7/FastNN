//! `im2col`, the op that turns convolution into a matrix multiply.
//!
//! Unfolding each sliding window into a column lets [`Conv2d`](crate::nn::Conv2d)
//! be `weight · columns`, so it inherits cuBLAS on the GPU and a ready-made
//! derivative from [`matmul`](super::matmul). The only new gradient this needs is
//! `col2im`: scatter each column back to the pixels it was copied from.

use crate::autograd::ops::conv::Col2ImBackward;
use crate::tensor::Tensor;

/// The four numbers that describe a 2-D sliding window.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Window {
    /// A square window with the same value on both axes.
    pub fn square(kernel: usize, stride: usize, padding: usize) -> Window {
        Window {
            kernel: (kernel, kernel),
            stride: (stride, stride),
            padding: (padding, padding),
        }
    }

    /// Output height and width for an input of `(height, width)`.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        let out = |size: usize, k: usize, s: usize, p: usize| {
            assert!(size + 2 * p >= k, "window {k} does not fit in {size} with padding {p}");
            (size + 2 * p - k) / s + 1
        };
        (
            out(height, self.kernel.0, self.stride.0, self.padding.0),
            out(width, self.kernel.1, self.stride.1, self.padding.1),
        )
    }

    /// Source pixel for output position `out` and kernel offset `k`, or `None`
    /// if it falls in the padding.
    fn source(&self, out: usize, k: usize, stride: usize, pad: usize, limit: usize) -> Option<usize> {
        let pos = (out * stride + k) as isize - pad as isize;
        (pos >= 0 && (pos as usize) < limit).then_some(pos as usize)
    }
}

impl Tensor {
    /// Unfold `[N, C, H, W]` into `[N, C·kh·kw, out_h·out_w]`.
    ///
    /// Column `l` holds every input pixel that feeds output pixel `l`, so a
    /// `[out_channels, C·kh·kw]` weight matrix times these columns is the
    /// convolution.
    pub fn im2col(&self, window: Window) -> Tensor {
        assert_eq!(self.ndim(), 4, "im2col expects [N, C, H, W], got {:?}", self.shape());
        let (n, c, h, w) = (self.dim(0), self.dim(1), self.dim(2), self.dim(3));
        let (kh, kw) = window.kernel;
        let (out_h, out_w) = window.output_size(h, w);
        let (patch, positions) = (c * kh * kw, out_h * out_w);

        let src = self.to_vec();
        let mut cols = vec![0.0f32; n * patch * positions];

        for image in 0..n {
            for channel in 0..c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let row = (channel * kh + ki) * kw + kj;
                        let dst_base = (image * patch + row) * positions;
                        let src_base = (image * c + channel) * h * w;
                        for oh in 0..out_h {
                            let Some(ih) = window.source(oh, ki, window.stride.0, window.padding.0, h) else {
                                continue;
                            };
                            for ow in 0..out_w {
                                let Some(iw) = window.source(ow, kj, window.stride.1, window.padding.1, w) else {
                                    continue;
                                };
                                cols[dst_base + oh * out_w + ow] = src[src_base + ih * w + iw];
                            }
                        }
                    }
                }
            }
        }

        let input_shape = self.shape().to_vec();
        Tensor::from_vec(cols, &[n, patch, positions])
            .to(self.device())
            .with_grad(&[self], || Col2ImBackward { shape: input_shape, window })
    }
}
