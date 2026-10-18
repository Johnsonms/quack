# Copyright (c) 2026, Tri Dao.
"""CPU-only checks of the RMSNorm/LayerNorm fwd analytical heuristic."""

import pytest

from quack.rmsnorm_config import RmsNormFwdConfig


def _fwd(N, width, ln, arch_major=10, **kw):
    return RmsNormFwdConfig.from_analytical_heuristic(
        N, width, arch_major=arch_major, is_layernorm=ln, **kw
    )


@pytest.mark.parametrize("N", [6144, 8192, 12288, 16384])
def test_layernorm_16bit_reloads_from_smem_above_4k(N):
    # 16-bit LayerNorm keeps the row in registers between its two passes and spills
    # from 8k on; the smem reload is 1.16-1.40x faster on B200 from 4k up, and on
    # GB300 (with the deferred W/B load) wins from 6k up.
    assert _fwd(N, 16, ln=True).reload_from == "smem", N


@pytest.mark.parametrize("N", [1024, 2048, 3072, 4096])
def test_layernorm_16bit_narrow_rows_keep_registers(N):
    assert _fwd(N, 16, ln=True).reload_from is None, N


@pytest.mark.parametrize("N", [2048, 4096, 8192, 10240, 16384])
def test_layernorm_16bit_hopper_reloads_above_8k(N):
    # On H100 rows up to 8k stay at ~167 registers with no spill and lose ~1% from
    # the reload; 10k/16k gain 4-5%.
    assert _fwd(N, 16, ln=True, arch_major=9).reload_from == (None if N <= 8192 else "smem"), N


@pytest.mark.parametrize("arch_major", [8, 12])
def test_layernorm_16bit_unmeasured_archs_unchanged(arch_major):
    assert _fwd(16384, 16, ln=True, arch_major=arch_major).reload_from is None
    assert _fwd(32768, 16, ln=True, arch_major=arch_major).reload_from == "smem"


@pytest.mark.parametrize("N", [4096, 8192, 16384])
def test_layernorm_fp32_threshold_unchanged(N):
    # fp32 LayerNorm loses 4-9% from the reload below 16k.
    assert _fwd(N, 32, ln=True).reload_from is None, N
    assert _fwd(32768, 32, ln=True).reload_from == "smem"


@pytest.mark.parametrize("N", [4096, 8192, 12288, 16384])
def test_rmsnorm_threshold_unchanged(N):
    assert _fwd(N, 16, ln=False).reload_from == (None if N <= 8192 else "smem"), N


@pytest.mark.parametrize("res_width", [16, 32])
@pytest.mark.parametrize("arch_major", [9, 10])
@pytest.mark.parametrize("N", [4096, 8192, 16384])
def test_layernorm_16bit_residual_keeps_registers(N, arch_major, res_width):
    # With a residual the reload is 0.83-1.00x on GB300: keep the 16k threshold.
    cfg = _fwd(N, 16, ln=True, arch_major=arch_major, res_dtype_width=res_width)
    assert cfg.reload_from is None, (N, arch_major, res_width)
    assert (
        _fwd(32768, 16, ln=True, arch_major=arch_major, res_dtype_width=res_width).reload_from
        == "smem"
    )


@pytest.mark.parametrize("N", [2048, 8192, 16384])
def test_blackwell_delay_with_residual(N):
    # Deferring W/B pays with a residual only for 16-bit rows with a bias, and with
    # an fp32 residual only from 16k up.
    assert _fwd(N, 16, ln=True, has_bias=True, res_dtype_width=16).delay_w_load
    assert not _fwd(N, 16, ln=True, has_bias=False, res_dtype_width=16).delay_w_load
    assert _fwd(N, 16, ln=True, has_bias=True, res_dtype_width=32).delay_w_load == (N >= 16384)
    assert not _fwd(N, 32, ln=True, has_bias=True, res_dtype_width=32).delay_w_load
    # Without a residual the deferred load stays on for 16-bit rows with a bias.
    assert _fwd(N, 16, ln=True, has_bias=True).delay_w_load


@pytest.mark.parametrize(
    "N, res_width, cluster_n",
    [
        (16384, None, 2),
        (65536, None, 8),
        (131072, None, 16),
        (16384, 16, 2),
        (65536, 16, 8),
        (131072, 16, 16),
        # fp32 residual: cn=2 only for 12k < N <= 32k; outside that the Hopper
        # ladder's cluster is kept (RMSNorm.__init__ then widens it only until x +
        # residual fit).
        (10240, 32, 1),
        (12288, 32, 1),
        (14336, 32, 2),
        (24576, 32, 2),
        (32768, 32, 2),
        (65536, 32, 2),
        (131072, 32, 4),
    ],
)
def test_blackwell_fp32_rmsnorm_cluster(N, res_width, cluster_n):
    assert _fwd(N, 32, ln=False, res_dtype_width=res_width).cluster_n == cluster_n
