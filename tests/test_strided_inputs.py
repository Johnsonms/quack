# Views the kernels can't vectorize directly (transposed input, column slice with an odd row
# stride, pointer offset by a few elements) must be copied on the host instead of failing at
# launch with a CuTe-DSL ValueError ("Invalid strides" / "Misaligned Tensor data").
import pytest
import torch

from quack.cross_entropy import cross_entropy, cross_entropy_bwd
from quack.rmsnorm import layernorm_bwd, layernorm_fwd, rmsnorm, rmsnorm_bwd
from quack.rotary import apply_rotary
from quack.softmax import softmax, softmax_bwd
from quack.topk import topk


def _views(M, N, dtype, device="cuda"):
    """(name, view) pairs whose elements are random; each view is (M, N) with last stride 1 or not."""
    torch.manual_seed(0)
    flat = torch.randn(M * N + 8, device=device, dtype=dtype)
    yield "ptr_offset", flat[1 : 1 + M * N].view(M, N)
    yield "transposed", torch.randn(N, M, device=device, dtype=dtype).t()
    yield "odd_row_stride", torch.randn(M, N + 1, device=device, dtype=dtype)[:, :N]
    yield "row_step", torch.randn(2 * M, N, device=device, dtype=dtype)[::2]


def _check(out, ref, dtype):
    torch.testing.assert_close(out, ref, atol=0, rtol=0)  # same kernel, same values


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [1024, 3000])
def test_rmsnorm_layernorm_strided(N, dtype):
    M = 32
    w = torch.randn(N, device="cuda", dtype=dtype)
    w32 = torch.randn(N, device="cuda", dtype=torch.float32)
    b32 = torch.randn(N, device="cuda", dtype=torch.float32)
    for name, x in _views(M, N, dtype):
        xc = x.contiguous()
        _check(rmsnorm(x, w), rmsnorm(xc, w), dtype)
        _check(layernorm_fwd(x, w32, b32), layernorm_fwd(xc, w32, b32), dtype)
        rstd = torch.rand(M, device="cuda") + 0.5
        mean = torch.randn(M, device="cuda")
        for a, b in zip(rmsnorm_bwd(x, w, x, rstd), rmsnorm_bwd(xc, w, xc, rstd)):
            if a is not None:
                _check(a, b, dtype)
        for a, b in zip(
            layernorm_bwd(x, w32, x, rstd, mean), layernorm_bwd(xc, w32, xc, rstd, mean)
        ):
            if a is not None:
                _check(a, b, dtype)
        # autograd through a view
        xg = x.detach().clone().requires_grad_()
        xcg = xc.detach().clone().requires_grad_()
        rmsnorm(xg, w).sum().backward()
        rmsnorm(xcg, w).sum().backward()
        _check(xg.grad, xcg.grad, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [1024, 3000])
def test_softmax_strided(N, dtype):
    M = 32
    for name, x in _views(M, N, dtype):
        xc = x.contiguous()
        _check(softmax(x), softmax(xc), dtype)
        y = torch.softmax(xc.float(), dim=-1).to(dtype)
        _check(softmax_bwd(x, y), softmax_bwd(xc, y), dtype)
        _check(softmax_bwd(y, x), softmax_bwd(y, xc), dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [1024, 3000])
def test_cross_entropy_strided(N, dtype):
    M = 32
    target = torch.randint(0, N, (M,), device="cuda")
    for name, x in _views(M, N, dtype):
        xc = x.contiguous()
        _check(
            cross_entropy(x, target, reduction="none"),
            cross_entropy(xc, target, reduction="none"),
            dtype,
        )
        dloss = torch.randn(M, device="cuda")
        lse = torch.logsumexp(xc.float(), dim=-1)
        _check(
            cross_entropy_bwd(x, target, dloss, lse),
            cross_entropy_bwd(xc, target, dloss, lse),
            dtype,
        )
        xg = x.detach().clone().requires_grad_()
        xcg = xc.detach().clone().requires_grad_()
        cross_entropy(xg, target).backward()
        cross_entropy(xcg, target).backward()
        _check(xg.grad, xcg.grad, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [256, 2048])
def test_topk_strided(N, dtype):
    M = 32
    for name, x in _views(M, N, dtype):
        xc = x.contiguous()
        for a, b in zip(topk(x, 8), topk(xc, 8)):
            _check(a, b, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_strided(interleaved, dtype):
    batch, seqlen, nheads, headdim = 2, 61, 3, 80
    torch.manual_seed(0)
    cos = torch.randn(seqlen, headdim // 2, device="cuda", dtype=dtype)
    sin = torch.randn(seqlen, headdim // 2, device="cuda", dtype=dtype)
    flat = torch.randn(batch * seqlen * nheads * headdim + 8, device="cuda", dtype=dtype)
    padded = torch.randn(batch, seqlen, nheads, headdim + 2, device="cuda", dtype=dtype)
    cs_padded = torch.randn(seqlen, headdim // 2 + 1, device="cuda", dtype=dtype)
    for name, x in [
        (
            "ptr_offset",
            flat[1 : 1 + batch * seqlen * nheads * headdim].view(batch, seqlen, nheads, headdim),
        ),
        ("odd_head_stride", padded[..., :headdim]),
    ]:
        xc = x.contiguous()
        _check(
            apply_rotary(x, cos, sin, interleaved=interleaved),
            apply_rotary(xc, cos, sin, interleaved=interleaved),
            dtype,
        )
    # cos/sin as column slices with an odd row stride
    cos_v, sin_v = cs_padded[:, : headdim // 2], cs_padded[:, 1 : 1 + headdim // 2]
    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    _check(
        apply_rotary(x, cos_v, sin_v, interleaved=interleaved),
        apply_rotary(x, cos_v.contiguous(), sin_v.contiguous(), interleaved=interleaved),
        dtype,
    )
