//! Derivative of `im2col`.

use crate::autograd::Backward;
use crate::tensor::ops::conv::Window;
use crate::tensor::Tensor;

/// Fold columns back into an image, adding where windows overlapped.
///
/// A pixel covered by several windows was copied into several columns, so its
/// gradient is the sum of theirs. Pixels that only ever landed in padding get
/// nothing.
pub struct Col2ImBackward {
    pub shape: Vec<usize>,
    pub window: Window,
}

impl Backward for Col2ImBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (kh, kw) = self.window.kernel;
        let (sh, sw) = self.window.stride;
        let (ph, pw) = self.window.padding;
        let (out_h, out_w) = self.window.output_size(h, w);
        let (patch, positions) = (c * kh * kw, out_h * out_w);

        let cols = grad.to_vec();
        let mut image = vec![0.0f32; n * c * h * w];

        for index in 0..n {
            for channel in 0..c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let row = (channel * kh + ki) * kw + kj;
                        let src_base = (index * patch + row) * positions;
                        let dst_base = (index * c + channel) * h * w;
                        for oh in 0..out_h {
                            let Some(ih) = source(oh, ki, sh, ph, h) else { continue };
                            for ow in 0..out_w {
                                let Some(iw) = source(ow, kj, sw, pw, w) else { continue };
                                image[dst_base + ih * w + iw] += cols[src_base + oh * out_w + ow];
                            }
                        }
                    }
                }
            }
        }

        vec![Tensor::from_vec(image, &self.shape).to(grad.device())]
    }
    fn name(&self) -> &'static str {
        "Col2Im"
    }
}

fn source(out: usize, k: usize, stride: usize, pad: usize, limit: usize) -> Option<usize> {
    let pos = (out * stride + k) as isize - pad as isize;
    (pos >= 0 && (pos as usize) < limit).then_some(pos as usize)
}
