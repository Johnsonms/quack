# Copyright (c) 2026, Tri Dao.
"""rmsnorm with a residual stages x and the residual side by side in smem, but the fwd
heuristic sized the cluster for one tile: fp32 rows from 32k up overflowed smem
(2 x 128 KB on a 227 KB budget) and failed to launch with CUDA_LAUNCH_INVALID_CONFIG."""

import pytest
import torch

from quack.rmsnorm import rmsnorm


@pytest.mark.parametrize(
    "dtype, N",
    [
        (torch.float32, 32768),
        (torch.float32, 65536),
        (torch.float32, 131072),
        (torch.bfloat16, 262144),
    ],
)
def test_rmsnorm_residual_wide_rows(dtype, N):
    torch.random.manual_seed(0)
    M = 16
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    out, res_out = rmsnorm(x, w, residual=res, prenorm=True)
    xr = x.float() + res.float()
    ref = (xr * torch.rsqrt(xr.square().mean(-1, keepdim=True) + 1e-6) * w).to(dtype)
    atol, rtol = (1e-4, 1e-5) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(res_out, xr.to(dtype), atol=atol, rtol=rtol)
