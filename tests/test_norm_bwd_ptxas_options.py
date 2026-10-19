# Copyright (c) 2026, Tri Dao.
"""SM90 16-bit RMSNorm/LayerNorm bwd with dw + db at > 48 elements per thread: compiled with ptxas -O1
(at -O3 ptxas falls back to 32 registers and runs 4-6x slower). Option checks (CPU) and numerics."""

import pytest
import torch

from quack.rmsnorm import layernorm_bwd, rmsnorm_bwd
from quack.rmsnorm_config import RmsNormBwdConfig, bwd_ptxas_options


def _opts(N, num_acc, width=16, dout_width=None, arch_major=9):
    dout_width = dout_width or width
    c = RmsNormBwdConfig.from_analytical_heuristic(
        N, width, dout_width, arch_major=arch_major, num_acc=num_acc
    )
    return bwd_ptxas_options(c, N, width, dout_width, num_acc, arch_major=arch_major)


@pytest.mark.parametrize("N", [200704, 212992, 229376, 245760, 258048, 270336, 327680])
def test_db_window_uses_O1(N):
    assert _opts(N, num_acc=2) == " --ptxas-options=-O1"


@pytest.mark.parametrize("N", [131072, 196608, 262144])
def test_db_outside_window_default(N):
    # 48 and 64 elements per thread: -O1 is 0.74-0.99x there.
    assert _opts(N, num_acc=2) == ""


@pytest.mark.parametrize("N", [212992, 229376, 245760])
def test_other_paths_default(N):
    assert _opts(N, num_acc=1) == ""  # no db
    assert _opts(N, num_acc=2, dout_width=32) == ""  # mixed dtype
    assert _opts(N, num_acc=2, arch_major=10) == ""
    assert _opts(N, num_acc=2, arch_major=8) == ""


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="exercises the SM90 ptxas option",
)
@pytest.mark.parametrize("N", [212992, 229376, 245760, 270336])
@pytest.mark.parametrize("is_layernorm", [False, True])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_db_window_numerics(N, is_layernorm, has_residual, dtype):
    if is_layernorm and has_residual:
        pytest.skip("layernorm_bwd takes no residual")
    M = 37
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    dout = torch.randn(M, N, device="cuda", dtype=dtype)
    dres_out = torch.randn(M, N, device="cuda", dtype=dtype) if has_residual else None
    xf = x.float().requires_grad_()
    wf = w.clone().requires_grad_()
    bf = b.clone().requires_grad_()
    if is_layernorm:
        mean = xf.detach().mean(-1)
        rstd = torch.rsqrt(xf.detach().var(-1, unbiased=False) + 1e-6)
        ref = torch.nn.functional.layer_norm(xf, (N,), wf, bf, eps=1e-6)
        dx, dw, db = layernorm_bwd(x, w, dout, rstd, mean, has_bias=True)
    else:
        rstd = torch.rsqrt(xf.detach().square().mean(-1) + 1e-6)
        ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * wf + bf
        dx, dw, db, _ = rmsnorm_bwd(
            x, w, dout, rstd, dres_out, has_bias=True, has_residual=has_residual
        )
    ref.backward(dout.float())
    dx_ref = xf.grad + dres_out.float() if has_residual else xf.grad
    torch.testing.assert_close(dx.float(), dx_ref, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dw.float(), wf.grad, atol=5e-1, rtol=1e-2)
    torch.testing.assert_close(db.float(), bf.grad, atol=5e-1, rtol=1e-2)
