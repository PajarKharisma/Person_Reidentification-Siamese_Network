# References:
# [1] Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding, Fukui et al., https://arxiv.org/abs/1606.01847
# [2] Compact Bilinear Pooling, Gao et al., https://arxiv.org/abs/1511.06062
# [3] Fast and Scalable Polynomial Kernels via Explicit Feature Maps, Pham and Pagh, https://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf
# [4] Fastfood — Approximating Kernel Expansions in Loglinear Time, Le et al., https://arxiv.org/abs/1408.3060
# Original implementation in Caffe: https://github.com/gy20073/compact_bilinear_pooling

import torch

class CompactBilinearPooling(torch.nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, sum_pool = True):
        super().__init__()
        self.out_channels = out_channels
        self.sum_pool = sum_pool
        generate_tensor_sketch = lambda rand_h, rand_s, in_channels1, out_channels: torch.sparse.FloatTensor(torch.stack([torch.arange(in_features), rand_h]), rand_s, [in_channels1, out_channels]).to_dense()
        self.tenosr_sketch1 = torch.nn.Parameter(generate_tensor_sketch(torch.randint(out_channels, size = (in_channels1,)), 2 * torch.randint(2, size = (in_channels1,), dtype = torch.float32) - 1, in_channels1, out_channels), requires_grad = False)
        self.tensor_sketch2 = torch.nn.Parameter(generate_tensor_sketch(torch.randint(out_channels, size = (in_channels2,)), 2 * torch.randint(2, size = (in_channels2,), dtype = torch.float32) - 1, in_channels2, out_channels), requires_grad = False)

    def forward(self, x1, x2):
        fft1 = torch.rfft(x1.permute(0, 2, 3, 1).matmul(self.tensor_sketch1), signal_ndim = 1)
        fft2 = torch.rfft(x2.permute(0, 2, 3, 1).matmul(self.tensor_sketch2), signal_ndim = 1)
        # torch.rfft does not support yet torch.complex64 outputs, so we do complex product manually
        fft_complex_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        cbp = torch.irfft(fft_complex_product, signal_ndim = 1, signal_sizes = (self.out_channels, )) * self.out_channels
        return cbp.sum(dim = [1, 2]) if self.sum_pool else cbp.permute(0, 3, 1, 2)