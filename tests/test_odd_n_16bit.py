# Copyright (c) 2026, Tri Dao.
"""16-bit rows whose length is odd get a 1-element (16-bit) gmem->smem copy from the
gcd vector size. cp.async only supports 4/8/16-byte copies, so these shapes used to
fail IR verification inside the DSL ("'cp_size' only supports (32, 64, 128)") for every
kernel on the cp.async path: rmsnorm/layernorm fwd and bwd and cross-entropy. GPT-2's
vocabulary (50257) is the everyday case."""

import pytest
import torch

from quack.cross_entropy import cross_entropy
from quack.rmsnorm import layernorm_fwd, rmsnorm, rmsnorm_bwd


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N", [1, 7, 33, 1001, 4097, 50257])
def test_rmsnorm_layernorm_odd_n(N, dtype):
    torch.random.manual_seed(0)
    M = 64
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * w).to(dtype)
    torch.testing.assert_close(rmsnorm(x, w), ref, atol=2e-2, rtol=2e-2)
    ref = torch.nn.functional.layer_norm(xf, (N,), w, None, 1e-6).to(dtype)
    torch.testing.assert_close(layernorm_fwd(x, w), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N", [7, 1001, 4097])
def test_rmsnorm_bwd_odd_n(N, dtype):
    torch.random.manual_seed(0)
    M = 64
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
    dout = torch.randn(M, N, device="cuda", dtype=dtype)
    rstd = torch.rsqrt(x.detach().float().square().mean(-1) + 1e-6)
    dx, dw = rmsnorm_bwd(x.detach(), w.detach(), dout, rstd)[:2]
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * w
    dx_ref, dw_ref = torch.autograd.grad(y, (x, w), grad_outputs=dout.float())
    torch.testing.assert_close(dx.float(), dx_ref.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dw.float(), dw_ref.float(), atol=5e-2 * M**0.5, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N", [7, 1001, 50257])
def test_cross_entropy_odd_n(N, dtype):
    torch.random.manual_seed(0)
    M = 64
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    t = torch.randint(0, N, (M,), device="cuda")
    ref = torch.nn.functional.cross_entropy(x.float(), t, reduction="none")
    torch.testing.assert_close(cross_entropy(x, t, reduction="none"), ref, atol=3e-2, rtol=1e-3)
