# Copyright (c) 2026, Tri Dao.
"""Softmax used a fixed 128-bit vector (8 elements for 16-bit, 4 for fp32) and asserted
``N % vecsize == 0``, so N=100 in bf16 or N=7 in fp32 raised an AssertionError. It now
uses gcd(N, vecsize) like the norm kernels; the 16-bit odd-N copies rely on the sub-32-bit
cp.async fallback in copy_utils."""

import pytest
import torch

from quack.softmax import softmax


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [1, 2, 3, 4, 7, 12, 100, 1001, 1004, 3000, 65540])
@pytest.mark.parametrize("M", [1, 37])
def test_softmax_non_multiple_of_8_n(M, N, dtype):
    torch.random.manual_seed(0)
    x = (0.1 * torch.randn(M, N, device="cuda", dtype=dtype)).requires_grad_()
    y = softmax(x)
    dy = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, grad_outputs=dy)
    xr = x.detach().float().requires_grad_()
    yr = torch.softmax(xr, dim=-1)
    (dxr,) = torch.autograd.grad(yr, xr, grad_outputs=dy.float())
    atol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(y.float(), yr, atol=atol, rtol=1e-3)
    torch.testing.assert_close(dx.float(), dxr, atol=atol * 10, rtol=1e-2)
