# Copyright (c) 2026, Tri Dao.
"""SM90 16-bit norm fwd with a bias: config checks (CPU) and numerics of the delayed W/B load
at the row sizes that used to spill the whole row to local memory."""

import dataclasses

import pytest
import torch

from quack.rmsnorm import layernorm_fwd, rmsnorm_fwd
from quack.rmsnorm_config import RmsNormFwdConfig


def _fwd(N, ln, bias=True, res=None, width=16, arch_major=9):
    return RmsNormFwdConfig.from_analytical_heuristic(
        N, width, arch_major=arch_major, is_layernorm=ln, has_bias=bias, res_dtype_width=res
    )


@pytest.mark.parametrize("N", [3072, 6144, 12288, 16384])
def test_hopper_bias_register_rows_delay(N):
    assert _fwd(N, ln=True).delay_w_load
    if N <= 8192:
        assert _fwd(N, ln=False).delay_w_load


@pytest.mark.parametrize("N", [32768, 65536, 262144])
def test_hopper_bias_cluster_rows_delay(N):
    assert _fwd(N, ln=True).delay_w_load and _fwd(N, ln=False).delay_w_load


def test_hopper_bias_residual_only_spilling_sizes():
    # >= 96 elements per thread with a residual: delay; below that it can cost (7168: 0.78x)
    assert _fwd(16384, ln=False, res=16).delay_w_load
    assert _fwd(6144, ln=True, res=16).delay_w_load
    assert not _fwd(7168, ln=False, res=16).delay_w_load
    assert not _fwd(262144, ln=False, res=16).delay_w_load


@pytest.mark.parametrize("N", [6144, 12288, 65536])
def test_unchanged_without_bias_or_off_sm90(N):
    for ln in (False, True):
        assert not _fwd(N, ln, bias=False).delay_w_load
        assert not _fwd(N, ln, width=32).delay_w_load
        assert not _fwd(N, ln, arch_major=10).delay_w_load
        assert not _fwd(N, ln, arch_major=12).delay_w_load
        # bias only toggles delay_w_load; the rest of the layout is unchanged
        assert dataclasses.replace(_fwd(N, ln), delay_w_load=False) == _fwd(N, ln, bias=False)


@pytest.mark.parametrize("N", [3072, 6144, 12288, 16384, 65536])
@pytest.mark.parametrize("is_layernorm", [False, True])
@pytest.mark.parametrize("has_res", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_bias_fwd_numerics(N, is_layernorm, has_res, dtype):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if is_layernorm and has_res:
        pytest.skip("layernorm_fwd takes no residual")
    M = 37
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    res = torch.randn(M, N, device="cuda", dtype=dtype) if has_res else None
    h = x.float() + res.float() if has_res else x.float()
    if is_layernorm:
        out = layernorm_fwd(x, w, b)
        ref = torch.nn.functional.layer_norm(h, (N,), w, b, eps=1e-6)
    else:
        out, res_out, _ = rmsnorm_fwd(x, w, b, res)
        ref = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6) * w + b
        if has_res:
            torch.testing.assert_close(res_out.float(), h.to(dtype).float())
    torch.testing.assert_close(out.float(), ref.to(dtype).float(), atol=5e-2, rtol=2e-2)
