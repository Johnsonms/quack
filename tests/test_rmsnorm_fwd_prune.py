"""CPU checks for the RMSNorm/LayerNorm fwd autotune pruner's smem guard.

The expected counts were measured on GB300 (sm_103a, 232448 B opt-in smem) by
launching every config: exactly these many of the 75 layouts launch, and the
pruner keeps exactly those. Each layout comes with delay_w_load False and True;
the pruner keeps the True half for LayerNorm only.
"""

from types import SimpleNamespace

import pytest
import torch

import quack.rmsnorm_config as rc
from quack.autotuner import AutotuneConfig
from quack.rmsnorm_config import RmsNormFwdConfig, prune_invalid_rmsnorm_fwd_configs

_SM100_SMEM = 232448


@pytest.fixture
def blackwell(monkeypatch):
    monkeypatch.setattr(rc, "_detect_arch_major", lambda: 10)
    monkeypatch.setattr(rc, "_max_dynamic_smem_bytes", lambda: _SM100_SMEM)


def _args(N, dtype, res_dtype=None, is_layernorm=False):
    x = torch.empty(4, N, dtype=dtype, device="meta")
    w = torch.empty(N, dtype=torch.float32, device="meta")
    res = None if res_dtype is None else torch.empty(4, N, dtype=res_dtype, device="meta")
    return dict(x=x, weight=w, residual=res, out=torch.empty_like(x), is_layernorm=is_layernorm)


def _kept(N, dtype, res_dtype=None, is_layernorm=False):
    configs = [AutotuneConfig(config=c) for c in rc.get_all_fwd_configs()]
    assert len(configs) == 150
    return prune_invalid_rmsnorm_fwd_configs(configs, _args(N, dtype, res_dtype, is_layernorm))


@pytest.mark.parametrize(
    "N, dtype, res_dtype, is_layernorm, n_launchable",
    [
        (262144, torch.bfloat16, None, False, 33),
        (262144, torch.bfloat16, None, True, 33),
        (131072, torch.float32, None, False, 33),
        (65536, torch.float32, torch.float32, False, 33),
        (131072, torch.bfloat16, torch.float32, False, 33),
        (65536, torch.float32, torch.float32, True, 33),
        (32768, torch.bfloat16, None, False, 72),
        (32768, torch.float32, torch.float32, False, 48),
    ],
)
def test_prune_keeps_exactly_the_launchable_configs(
    blackwell, N, dtype, res_dtype, is_layernorm, n_launchable
):
    kept = _kept(N, dtype, res_dtype, is_layernorm)
    assert len(kept) == n_launchable * (2 if is_layernorm else 1)
    assert any(ac.kwargs["config"].delay_w_load for ac in kept) == is_layernorm


def test_prune_keeps_everything_for_narrow_rows(blackwell):
    assert len(_kept(4096, torch.bfloat16, torch.float32)) == 75
    assert len(_kept(4096, torch.bfloat16, torch.float32, is_layernorm=True)) == 150


def test_prune_drops_a_full_row_tile_over_smem(blackwell):
    # 128 threads, one row per CTA, no cluster: a 262144-wide bf16 row is 512 KB.
    c = RmsNormFwdConfig(num_threads=128, threads_per_row=128, cluster_n=1, reload_from=None)
    kept = prune_invalid_rmsnorm_fwd_configs(
        [SimpleNamespace(kwargs={"config": c})], _args(262144, torch.bfloat16)
    )
    assert kept == []


def test_prune_without_smem_info_only_checks_shape(monkeypatch):
    monkeypatch.setattr(rc, "_detect_arch_major", lambda: 10)
    monkeypatch.setattr(rc, "_max_dynamic_smem_bytes", lambda: 0)
    assert len(_kept(262144, torch.bfloat16)) == 75
