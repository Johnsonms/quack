# Copyright (c) 2026, Tri Dao.
"""CPU-only checks of the RMSNorm/LayerNorm fwd analytical heuristic."""

import pytest

from quack.rmsnorm_config import RmsNormFwdConfig


def _fwd(N, width, ln):
    return RmsNormFwdConfig.from_analytical_heuristic(N, width, arch_major=10, is_layernorm=ln)


@pytest.mark.parametrize("N", [4096, 6144, 8192, 12288, 16384])
def test_layernorm_16bit_reloads_from_smem_above_2k(N):
    # 16-bit LayerNorm keeps the row in registers between its two passes and spills
    # from 8k on; the smem reload is 1.16-1.40x faster on B200 from 4k up.
    assert _fwd(N, 16, ln=True).reload_from == "smem", N


@pytest.mark.parametrize("N", [1024, 2048])
def test_layernorm_16bit_narrow_rows_keep_registers(N):
    assert _fwd(N, 16, ln=True).reload_from is None, N


@pytest.mark.parametrize("N", [4096, 8192, 16384])
def test_layernorm_fp32_threshold_unchanged(N):
    # fp32 LayerNorm loses 4-9% from the reload below 16k.
    assert _fwd(N, 32, ln=True).reload_from is None, N
    assert _fwd(32768, 32, ln=True).reload_from == "smem"


@pytest.mark.parametrize("N", [4096, 8192, 12288, 16384])
def test_rmsnorm_threshold_unchanged(N):
    assert _fwd(N, 16, ln=False).reload_from == (None if N <= 8192 else "smem"), N
