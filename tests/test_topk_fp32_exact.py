# Copyright (c) 2026, Tri Dao.
"""The top-k kernel keeps the column index in the low log2(N) mantissa bits of the fp32
key. 16-bit inputs are exact (their mantissa is zero there), but fp32 inputs came back
with those bits overwritten: values != x[indices] (N=4096: up to 2e-3 off) and the
softmax variant was computed on the truncated values. topk_fwd now returns the exact
selected elements for fp32 (and the softmax of those)."""

import pytest
import torch

from quack.topk import topk, topk_fwd


@pytest.mark.parametrize("N, k", [(1024, 8), (4096, 32), (4096, 128)])
@pytest.mark.parametrize("M", [1, 37, 2048])
def test_topk_fp32_values_are_input_elements(M, N, k):
    torch.random.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    values, indices = topk_fwd(x, k)
    assert torch.equal(values, torch.gather(x, 1, indices.long()))
    assert torch.all(values[:, :-1] >= values[:, 1:])
    # Selection is on keys with log2(N) low mantissa bits replaced: identical to
    # torch.topk except for near-ties (relative 2^-(23 - log2 N)), so the selected
    # values must match torch's to that resolution even where the index sets differ.
    ref_vals, _ = torch.topk(x, k, dim=-1)
    torch.testing.assert_close(values, ref_vals, atol=0, rtol=2.0 ** -(23 - N.bit_length()))


@pytest.mark.parametrize("N, k", [(1024, 8), (4096, 32)])
def test_topk_fp32_softmax_matches_exact_values(N, k):
    torch.random.manual_seed(0)
    x = torch.randn(64, N, device="cuda", dtype=torch.float32)
    values, indices = topk(x, k, softmax=True)
    ref = torch.softmax(torch.gather(x, 1, indices.long()), dim=-1)
    torch.testing.assert_close(values, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_topk_16bit_values_unchanged(dtype):
    torch.random.manual_seed(0)
    x = torch.randn(64, 4096, device="cuda", dtype=dtype)
    values, indices = topk_fwd(x, 32)
    assert torch.equal(values, torch.gather(x, 1, indices.long()))
