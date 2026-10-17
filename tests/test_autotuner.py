import pytest

from quack.autotuner import AutotuneConfig


def test_autotune_config_supports_multi_kwarg_hash_and_equality():
    config_a = AutotuneConfig(block_m=128, num_warps=4)
    config_b = AutotuneConfig(block_m=128, num_warps=4)
    config_c = AutotuneConfig(block_m=64, num_warps=4)

    assert config_a == config_b
    assert hash(config_a) == hash(config_b)
    assert config_a != config_c

    timings = {config_a: 1.25, config_c: 2.5}
    assert timings[config_b] == 1.25
    assert len({config_a, config_b, config_c}) == 2


# ---------------------------------------------------------------------------
# Bench loop: defer-and-retry over configs via the async compile pool
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_bench_loop_defers_and_retries(monkeypatch):
    """A config whose kernel raises CompilePending is rotated to the back and
    retried once its sha polls done; all configs end up benchmarked exactly
    as if they had been warm.

    This encodes the compile-only rip-out contract: the autotuner no longer
    precompiles via fake tensors — the bench loop discovers cold keys with
    the real tensors in-process and overlaps compilation via the pool.
    """
    from quack.autotuner import Autotuner, AutotuneConfig
    from quack.cache import async_compile
    from quack.cache.async_compile import CompilePending

    class _StubPool:
        """poll() reports 'pending' once per sha, then 'done'."""

        def __init__(self):
            self.polls = {}

        def poll(self, sha):
            n = self.polls.get(sha, 0) + 1
            self.polls[sha] = n
            return ("pending" if n == 1 else "done"), None

    stub = _StubPool()
    monkeypatch.setattr(async_compile, "_active_pool", stub)

    bench_order = []
    raised_once = set()

    def kernel(x, block: int = 0):
        # config block=1 is "cold": its first invocation defers.
        if block == 1 and 1 not in raised_once:
            raised_once.add(1)
            raise CompilePending("f" * 64, "fake._compile_kernel")
        bench_order.append(block)

    def do_bench(fn, quantiles=None, **kw):
        fn()
        return [1.0 + bench_order[-1], 1.0, 1.0]  # block=0 fastest

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1, 2)],
        do_bench=do_bench,
    )
    import torch

    x = torch.empty(4, device="cuda")
    tuner(x)

    # Bench order: block=1 deferred, so it benched AFTER block 2 (exactly
    # once). The trailing 0 is __call__'s real invocation with the winner.
    assert bench_order == [0, 2, 1, 0], bench_order
    assert stub.polls == {"f" * 64: 2}  # one rotation, one release
    assert len(tuner.configs_timings) == 3
    best = tuner.cache[next(iter(tuner.cache))]
    assert best.kwargs["block"] == 0  # timings intact despite the deferral


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_wedged_pool_falls_back_in_process(monkeypatch):
    """A sha that never resolves must not hang the sweep: past the attempt
    cap the config is benched with the pool suppressed (in-process compile),
    so autotuning always terminates.
    """
    import quack.autotuner as at
    from quack.autotuner import Autotuner, AutotuneConfig
    from quack.cache import async_compile
    from quack.cache.async_compile import CompilePending, get_active_pool

    class _WedgedPool:
        def poll(self, sha):
            return "pending", None  # never completes

    monkeypatch.setattr(async_compile, "_active_pool", _WedgedPool())
    monkeypatch.setattr(at, "_POOL_WEDGE_TIMEOUT_S", 0.2)

    benched = []

    def kernel(x, block: int = 0):
        # Defer as long as a pool is visible; succeed once suppressed.
        if block == 1 and get_active_pool() is not None:
            raise CompilePending("e" * 64, "fake._compile_kernel")
        benched.append(block)

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1)],
        do_bench=lambda fn, quantiles=None, **kw: (fn(), [1.0, 1.0, 1.0])[1],
    )
    import torch

    tuner(torch.empty(4, device="cuda"))
    assert benched.count(1) == 1  # eventually ran, via suppress_pool
    assert len(tuner.configs_timings) == 2


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_skips_config_raising_dsl_launch_error():
    """A launch the DSL rejects must cost that config, not the whole sweep.

    cutlass-dsl >= 4.8 reports launch failures (e.g. the launch preflight's
    CUDA_LAUNCH_INVALID_CONFIG for a smem request over the per-block limit)
    as CUDADialectError, which derives from DSLCudaRuntimeError rather than
    RuntimeError. The L2-cold bench path's narrow catch let it escape, so a
    cold-cache rmsnorm_fwd_tuned call at N=262144 crashed instead of skipping
    the over-budget configs.
    """
    tvm_ffi_provider = pytest.importorskip("cutlass.cutlass_dsl.tvm_ffi_provider")
    CUDADialectError = getattr(tvm_ffi_provider, "CUDADialectError", None)
    if CUDADialectError is None:
        pytest.skip("cutlass-dsl without CUDADialectError")
    import torch

    from quack.autotuner import Autotuner, AutotuneConfig

    def kernel(x, block: int = 0):
        if block == 1:
            raise CUDADialectError("CUDA Error Code: 1")
        x.add_(1)

    tuner = Autotuner(kernel, key=[], configs=[AutotuneConfig(block=b) for b in (0, 1)])
    tuner(torch.zeros(1024, device="cuda"))  # no do_bench / hooks -> L2-cold path
    timings = {c.kwargs["block"]: t for c, t in tuner.configs_timings.items()}
    assert timings[1][0] == float("inf")
    assert timings[0][0] < float("inf")
    assert tuner.cache[next(iter(tuner.cache))].kwargs["block"] == 0


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_autotuning_does_not_pollute_selective_checkpoint_cache():
    import torch
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        checkpoint,
        create_selective_checkpoint_contexts,
    )

    from quack.autotuner import Autotuner

    def matmul(a, b, block=0):
        return a @ b

    tuner = Autotuner(
        matmul,
        key=[],
        configs=[AutotuneConfig(block=0), AutotuneConfig(block=1)],
    )
    saved_matmuls = []

    def policy(context, op, *args, **kwargs):
        if op == torch.ops.aten.mm.default:
            if not context.is_recompute:
                saved_matmuls.append(tuple(args[0].shape))
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    a = torch.randn(32, 32, device="cuda", requires_grad=True)
    b = torch.randn(32, 32, device="cuda", requires_grad=True)
    expected = a @ b
    expected_grads = torch.autograd.grad(expected.sum(), (a, b))
    actual = checkpoint(
        tuner,
        a,
        b,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
    )
    assert saved_matmuls == [(32, 32)]
    actual_grads = torch.autograd.grad(actual.sum(), (a, b))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
