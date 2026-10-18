# Copyright (c) 2026, Tri Dao.
"""SM90 16-bit RMSNorm/LayerNorm bwd heuristic: config checks (CPU) and numerics at the
non-power-of-two widths the heuristic routes to cluster/thread layouts with a tail."""

import pytest
import torch

from quack.rmsnorm import layernorm_bwd, rmsnorm_bwd
from quack.rmsnorm_config import RmsNormBwdConfig, _for_hopper_bwd


def _bwd(N, num_acc, width=16, arch_major=9, has_dresidual=False):
    return RmsNormBwdConfig.from_analytical_heuristic(
        N, width, width, arch_major=arch_major, num_acc=num_acc, has_dresidual=has_dresidual
    )


@pytest.mark.parametrize("N", [8192, 12288, 16384])
def test_hopper_16bit_mid_rows_use_tma(N):
    c = _bwd(N, num_acc=1)
    assert c.use_tma and c.threads_per_row == 256 and c.cluster_n == (1 if N <= 8192 else 2)


@pytest.mark.parametrize("N", [20480, 32768, 49152, 65536])
@pytest.mark.parametrize("num_acc", [1, 2])
def test_hopper_16bit_wide_rows_128_threads(N, num_acc):
    # Smallest cluster keeping <= 32 elements per thread, never cluster 16 at 256 threads here.
    c = _bwd(N, num_acc)
    assert c.threads_per_row == 128 and N // c.cluster_n // 128 <= 32 and c.use_tma
    assert c.reload_x == ("smem" if num_acc >= 2 else None)


@pytest.mark.parametrize("N", [196608, 262144])
def test_hopper_16bit_widest_rows_skip_tma(N):
    # > 80 B per thread: TMA's fragments spill, keep the cp.async path.
    for num_acc in (1, 2):
        c = _bwd(N, num_acc)
        threads = 512 if num_acc == 2 and N > 196608 else 256
        assert c.cluster_n == 16 and c.threads_per_row == c.num_threads == threads
        assert not c.use_tma


@pytest.mark.parametrize("N", [262144, 327680])
def test_hopper_16bit_widest_rows_with_db_512_threads(N):
    # dw + db at 64 elements per thread spill; 512 threads only with the second accumulator.
    assert _bwd(N, num_acc=2).num_threads == 512
    assert _bwd(N, num_acc=1).num_threads == 256


@pytest.mark.parametrize("N", [196608, 229376])
def test_hopper_16bit_db_keeps_256_threads(N):
    # 196608: 48 elements/thread don't spill; 229376: 28 elements/thread at 512 is not whole
    # 8-element vectors and runs up to 0.43x.
    assert _bwd(N, num_acc=2).num_threads == 256


def test_hopper_layernorm_tma_cap():
    # 163840 is 80 B/thread: RMSNorm takes TMA + x reload, LayerNorm stays on cp.async.
    assert _bwd(163840, 1).use_tma and _bwd(163840, 1).reload_x == "smem"
    assert not _bwd(163840, 2).use_tma


@pytest.mark.parametrize("N", [8192, 12288, 16384, 98304, 131072, 163840])
@pytest.mark.parametrize("num_acc", [1, 2])
def test_hopper_dresidual_256_thread_rows_match_generic_ladder(N, num_acc):
    # With a dresidual_out load, TMA at 256 threads is 0.46-0.95x: keep the generic cp.async layout.
    c = _bwd(N, num_acc, has_dresidual=True)
    assert c == _for_hopper_bwd(N, 16, 9)
    assert c.cluster_n == (1 if N <= 8192 else 2 if N <= 16384 else 16)


@pytest.mark.parametrize("N", [20480, 32768, 65536])
def test_hopper_dresidual_keeps_128_thread_rows(N):
    for num_acc in (1, 2):
        assert _bwd(N, num_acc, has_dresidual=True) == _bwd(N, num_acc)


@pytest.mark.parametrize("N", [4096, 32768, 131072])
def test_other_paths_unchanged(N):
    assert not _bwd(N, 1, width=32).use_tma
    assert not _bwd(4096, 1).use_tma
    assert not _bwd(N, 1, arch_major=8).use_tma


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="exercises the SM90 heuristic",
)
@pytest.mark.parametrize("N", [12288, 20480, 40960, 81920, 163840])
@pytest.mark.parametrize("is_layernorm", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hopper_16bit_bwd_numerics(N, is_layernorm, dtype):
    M = 37
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    dout = torch.randn(M, N, device="cuda", dtype=dtype)
    xf = x.float().requires_grad_()
    wf = w.clone().requires_grad_()
    if is_layernorm:
        mean = xf.detach().mean(-1)
        rstd = torch.rsqrt(xf.detach().var(-1, unbiased=False) + 1e-6)
        ref = torch.nn.functional.layer_norm(xf, (N,), wf, None, eps=1e-6)
        out = layernorm_bwd(x, w, dout, rstd, mean)
    else:
        rstd = torch.rsqrt(xf.detach().square().mean(-1) + 1e-6)
        ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * wf
        out = rmsnorm_bwd(x, w, dout, rstd)
    ref.backward(dout.float())
    dx, dw = out[0], out[1]
    torch.testing.assert_close(dx.float(), xf.grad, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dw.float(), wf.grad, atol=5e-1, rtol=1e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="exercises the SM90 heuristic",
)
@pytest.mark.parametrize("N", [12288, 32768, 98304, 163840])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hopper_16bit_bwd_dresidual_numerics(N, has_bias, dtype):
    M = 37
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    dout = torch.randn(M, N, device="cuda", dtype=dtype)
    dres_out = torch.randn(M, N, device="cuda", dtype=dtype)
    xf = x.float().requires_grad_()
    wf = w.clone().requires_grad_()
    rstd = torch.rsqrt(xf.detach().square().mean(-1) + 1e-6)
    ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * wf
    ref.backward(dout.float())
    dx, dw, db, dres = rmsnorm_bwd(x, w, dout, rstd, dres_out, has_bias=has_bias, has_residual=True)
    torch.testing.assert_close(dx.float(), xf.grad + dres_out.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dres.float(), dx.float())
    torch.testing.assert_close(dw.float(), wf.grad, atol=5e-1, rtol=1e-2)
    if has_bias:
        torch.testing.assert_close(db.float(), dout.float().sum(0), atol=5e-1, rtol=1e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="exercises the SM90 heuristic",
)
@pytest.mark.parametrize("N", [196608, 229376, 262144])
@pytest.mark.parametrize("is_layernorm", [False, True])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hopper_16bit_bwd_bias_numerics(N, is_layernorm, has_residual, dtype):
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
        dx, dw, db, _ = rmsnorm_bwd(x, w, dout, rstd, dres_out, has_bias=True)
    ref.backward(dout.float())
    dx_ref = xf.grad + dres_out.float() if has_residual else xf.grad
    torch.testing.assert_close(dx.float(), dx_ref, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dw.float(), wf.grad, atol=5e-1, rtol=1e-2)
    torch.testing.assert_close(db.float(), bf.grad, atol=5e-1, rtol=1e-2)
