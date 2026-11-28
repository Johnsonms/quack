# Copyright (c) 2026, Tri Dao.
"""Regression tests for the async compile pool (``--async-compile``).

Each test encodes a specific failure mode discovered while building the
feature — see quack/cache/async_compile.py and the defer-and-retry loop in
quack/testing/pytest_plugin.py.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from quack.cache import jit_cache
from quack.cache.async_compile import CompilePending, CompilePool, _flock_held_exclusively


def _inner_pytest(cwd, *args, timeout=300):
    """Run a pytest session with the async-compile pool in ``cwd``.
    Returns (exit status, output)."""
    env = dict(os.environ)
    env.pop("PYTEST_XDIST_WORKER", None)  # the inner session is not an xdist worker
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--async-compile=2"]
    result = subprocess.run(
        cmd + list(args), cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# CompilePending semantics
# ---------------------------------------------------------------------------


def test_compile_pending_is_base_exception():
    """CompilePending must NOT be catchable by ``except Exception``.

    Regression: test bodies with broad ``try/except Exception`` (or
    ``pytest.raises(Exception)``) around a kernel call would otherwise
    swallow the deferral signal and turn a never-executed test into a false
    pass.
    """
    assert issubclass(CompilePending, BaseException)
    assert not issubclass(CompilePending, Exception)

    with pytest.raises(BaseException):
        try:
            raise CompilePending("0" * 64, "some._compile_fn")
        except Exception:  # must NOT catch it
            pytest.fail("except Exception swallowed CompilePending")

    exc = CompilePending("ab" * 32, "quack.foo._compile_bar")
    assert exc.sha == "ab" * 32
    assert "quack.foo._compile_bar" in str(exc)


# ---------------------------------------------------------------------------
# Pool bookkeeping: poll states, external-compile detection
# ---------------------------------------------------------------------------


def test_flock_probe_detects_exclusive_holder(tmp_path):
    """The external-compile probe: exclusive flock held => True, free => False.

    This is what lets one xdist worker defer on a key that another worker's
    pool is already compiling, instead of burning a pool slot on a duplicate
    compile that would just block on the same flock.
    """
    import fcntl

    lock_path = tmp_path / "key.lock"
    assert not _flock_held_exclusively(str(lock_path))  # nonexistent -> free

    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert _flock_held_exclusively(str(lock_path))
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert not _flock_held_exclusively(str(lock_path))
    finally:
        os.close(fd)


def test_pool_poll_external_lifecycle(tmp_path):
    """poll() on an externally-compiled key: pending while locked, done once
    the .o appears, back to new if the external compiler died lockless.

    Regression: without the died-without-.o transition, a crashed foreign
    compiler would leave every waiter deferring forever (until the wedge
    deadline force-synced them one by one).
    """
    import fcntl

    pool = CompilePool.__new__(CompilePool)  # skip executor spawn: bookkeeping only
    pool._futures = {}
    pool._external = {}
    pool.n_submitted = 0

    sha = "e" * 64
    o_path = tmp_path / f"{sha}.o"
    lock_path = tmp_path / f"{sha}.lock"

    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        pool.mark_external(sha, str(o_path), str(lock_path))
        assert pool.poll(sha) == ("pending", None)  # locked, no .o yet

        o_path.write_bytes(b"fake object file")
        assert pool.poll(sha) == ("done", None)  # .o landed
        assert sha not in pool._external  # consumed

        # Crash case: locked then released without producing a .o.
        sha2 = "f" * 64
        o2 = tmp_path / f"{sha2}.o"
        pool.mark_external(sha2, str(o2), str(lock_path))
        assert pool.poll(sha2) == ("pending", None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert pool.poll(sha2) == ("new", None)  # forgotten -> resubmittable
        assert sha2 not in pool._external
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# End-to-end: defer-and-retry through a real pytest session
# ---------------------------------------------------------------------------


def test_deferral_releases_test_locals_without_gc(tmp_path):
    """Discarding CompilePending must not leave test tensors in traceback cycles."""
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["quack.testing.pytest_plugin"]\n')
    (tmp_path / "test_deferred_locals.py").write_text(
        textwrap.dedent(
            """
            import gc
            import weakref

            import torch
            from quack.cache.async_compile import CompilePending

            deferred_tensor = None

            def test_deferred_locals():
                global deferred_tensor
                if deferred_tensor is None:
                    gc.disable()  # Do not let cyclic GC hide retained traceback frames.
                    tensor = torch.empty(1)
                    deferred_tensor = weakref.ref(tensor)
                    raise CompilePending("a" * 64, "fake.compile")
                try:
                    assert deferred_tensor() is None, "deferred attempt still owns its tensor"
                finally:
                    gc.enable()
            """
        )
    )

    env = dict(os.environ)
    env.pop("PYTEST_XDIST_WORKER", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--async-compile=2"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "1 passed" in out, out
    assert "1 test deferrals" in out, out


class _FakeKernel:
    def export_to_c(self, object_file_path, function_name):
        Path(object_file_path).write_bytes(b"fake object file")


@jit_cache
def _alias_probe_compile(mode):
    return _FakeKernel()


def test_pool_worker_exports_keys_equal_in_memory_but_not_on_disk(tmp_path, monkeypatch):
    """Two requests whose arguments compare equal but pickle differently (an
    IntEnum and its int) are one in-memory jit-cache entry and two disk keys.
    A worker that had compiled the first answered the second from memory and
    exported nothing: the pool reported "compile succeeded but .o was not
    exported" (seen on a B300 CI-style run, for a kernel requested with
    ``RoundingMode.RN`` eagerly and ``0`` under torch.compile)."""
    import base64
    import pickle

    import quack.cache as cache_state
    from quack.cache import async_compile
    from quack.cache.jit import _compute_source_fingerprint, _key_to_hash
    from quack.rounding import RoundingMode

    monkeypatch.setattr(cache_state, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_state, "CACHE_ENABLED", True)
    monkeypatch.setattr(async_compile, "_active_pool", None)  # run the worker body right here
    fn = _alias_probe_compile
    assert (RoundingMode.RN,) == (0,) and hash((RoundingMode.RN,)) == hash((0,))
    for mode in (RoundingMode.RN, 0):
        sha = _key_to_hash((fn.__qualname__, mode))
        o_path = tmp_path / _compute_source_fingerprint() / f"{sha}.o"
        key_b64 = base64.b64encode(pickle.dumps(((mode,), {}))).decode("ascii")
        err = async_compile._pool_worker(fn.__module__, fn.__qualname__, key_b64, str(o_path))
        assert err is None, f"mode {mode!r}: {err}"
        assert o_path.exists()
    assert len(list((tmp_path / _compute_source_fingerprint()).glob("*.o"))) == 2


@jit_cache
def _worker_failed_probe_compile(ok):
    if not ok:
        raise ValueError("config rejected")
    return _FakeKernel()


class _FailedPool:
    """An active pool whose worker failed every request."""

    def __init__(self):
        self.noticed = set()

    def poll(self, sha):
        return "failed", "ValueError: rejected by the worker [worker: ...]"


@pytest.mark.parametrize("ok", [True, False], ids=["pool_missed", "rejected"])
def test_worker_failure_warns_only_if_in_process_compile_succeeds(tmp_path, monkeypatch, ok):
    """A key whose pool worker failed is recompiled in-process.

    pool_missed: the in-process compile succeeds, so the pool missed a kernel:
    warn and record a fallback. rejected: it fails too (a config the caller
    expects to be rejected, or an autotune candidate that doesn't fit), so the
    exception is the report, with the worker's error as a note. A warning there
    was noise: every H100 QUACK_ARCH=120 CI run warned twice for
    test_gemm_sm120_fp8.py::test_fp8_cluster_rejected, which passed."""
    import warnings

    import quack.cache as cache_state
    from quack.cache import async_compile

    monkeypatch.setattr(cache_state, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_state, "CACHE_ENABLED", True)
    monkeypatch.setattr(async_compile, "_active_pool", _FailedPool())
    monkeypatch.setattr(async_compile, "_fallbacks", [])
    _worker_failed_probe_compile.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if ok:
            _worker_failed_probe_compile(True)
        else:
            with pytest.raises(ValueError) as excinfo:
                _worker_failed_probe_compile(False)
    failed = [w for w in caught if "async compile failed" in str(w.message)]
    if ok:
        assert len(failed) == 1 and failed[0].filename == __file__, failed
        assert [f.reason for f in async_compile._fallbacks] == ["worker_failed"]
    else:
        assert not failed, [str(w.message) for w in failed]
        assert not async_compile._fallbacks
        # The local exception, not the note (pytest.raises(match=) searches both).
        assert str(excinfo.value) == "config rejected"
        assert any("rejected by the worker" in note for note in excinfo.value.__notes__)


_RETRY_WARNS_CONFTEST_SRC = textwrap.dedent(
    """
    pytest_plugins = ["quack.testing.pytest_plugin"]


    def pytest_configure(config):
        # test_deferred_warns's kernel is still pending at the first poll and
        # ready at the next: it is retried in the session loop's call for
        # test_b2, after test_b1 ran.
        from quack.cache import async_compile as ac

        poll, polls = ac.CompilePool.poll, []

        def fake_poll(self, sha):
            if sha != "a" * 64:
                return poll(self, sha)
            polls.append(sha)
            return ("pending" if len(polls) == 1 else "new"), None

        ac.CompilePool.poll = fake_poll
    """
)

_RETRY_WARNS_TEST_A_SRC = textwrap.dedent(
    """
    import warnings

    from quack.cache.async_compile import CompilePending

    attempts = []

    def test_deferred_warns():
        attempts.append(1)
        if len(attempts) == 1:
            raise CompilePending("a" * 64, "fake.compile")
        warnings.warn("emitted by the retry", UserWarning)
    """
)

_RETRY_WARNS_TEST_B_SRC = textwrap.dedent(
    """
    import warnings

    import pytest

    @pytest.fixture(scope="module")
    def modfix():
        yield
        warnings.warn("module teardown", UserWarning)

    {B1_MARK}
    def test_b1(modfix):
        pass

    # Its filters must not reach a retry run in xdist's call for it.
    @pytest.mark.filterwarnings("error:emitted by the retry")
    def test_b2():
        pass
    """
)


def _warnings_by_test(out: str) -> dict:
    """nodeid -> its warning lines, from pytest's warnings summary (which
    lists several nodeids above one warning when they share it)."""
    by_test, heads, in_block = {}, [], False
    for line in out.partition("warnings summary")[2].splitlines()[1:]:
        if line.startswith(("-- Docs", "=")):
            break
        if line.strip() and not line[0].isspace():
            if in_block:
                heads, in_block = [], False
            heads.append(line.strip())
        elif line.strip():
            in_block = True
            for head in heads:
                by_test[head] = by_test.get(head, "") + line + "\n"
    return by_test


@pytest.mark.parametrize("teardown_error", [False, True], ids=["warns", "raises"])
@pytest.mark.parametrize("xdist", [False, True], ids=["single", "xdist"])
def test_deferred_retry_warnings_name_their_own_test(tmp_path, xdist, teardown_error):
    """Every attempt reports its warnings under its own nodeid, filtered by
    its own marks.

    pytest captures warnings around the pytest_runtest_protocol hook call.
    xdist: a worker retried deferred tests inside the hook call for the next
    test, so their warnings were filed under that unrelated test (CI blamed
    test_epi_autotune.py::test_tuned_mod_gemm_cache_hit for compile warnings
    from test_gemm_sm120_fp8.py), and its filterwarnings marks applied to
    them (here: error, failing the retry). So did the forced teardown, before
    the retry, of the fixture test_b1 left: plain pytest runs finalizers in a
    teardown of the last test run, so its warning belongs to test_b1, and
    an error there (raises: test_b1 makes the warning one) is a teardown
    error of test_b1, not an INTERNALERROR. single: the loop bypassed the
    hook, so nothing captured a passing test's warnings."""
    b1_mark = '@pytest.mark.filterwarnings("error:module teardown")' if teardown_error else ""
    (tmp_path / "conftest.py").write_text(_RETRY_WARNS_CONFTEST_SRC)
    (tmp_path / "test_a.py").write_text(_RETRY_WARNS_TEST_A_SRC)
    (tmp_path / "test_b.py").write_text(_RETRY_WARNS_TEST_B_SRC.format(B1_MARK=b1_mark))
    returncode, out = _inner_pytest(tmp_path, *(["-n", "1"] if xdist else []))
    assert "INTERNALERROR" not in out, out
    by_test = _warnings_by_test(out)
    assert "emitted by the retry" in by_test.pop("test_a.py::test_deferred_warns"), out
    if teardown_error:
        assert returncode == 1 and "3 passed" in out and "1 error" in out, out
        assert "ERROR test_b.py::test_b1 - UserWarning: module teardown" in out, out
        assert not by_test, out
    else:
        assert returncode == 0 and "3 passed" in out, out
        assert set(by_test) == {"test_b.py::test_b1"}, out
        assert "module teardown" in by_test["test_b.py::test_b1"], out


_RECORDING_WRAPPER_CONFTEST_SRC = textwrap.dedent(
    """
    from pathlib import Path

    import pytest

    pytest_plugins = ["quack.testing.pytest_plugin"]


    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(item):
        # A valid wrapper: the hook's result is unused, so it needn't pass it on.
        with open(Path(__file__).parent / "calls.txt", "a") as f:
            f.write(item.name + "\\n")
        yield
    """
)

_RECORDING_WRAPPER_TEST_SRC = textwrap.dedent(
    """
    from quack.cache.async_compile import CompilePending

    attempts = []

    def test_deferred_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise CompilePending("a" * 64, "fake.compile")

    def test_warm():
        pass
    """
)


@pytest.mark.parametrize("xdist", [False, True], ids=["single", "xdist"])
def test_protocol_wrappers_see_each_attempt_once(tmp_path, xdist):
    """Other plugins' pytest_runtest_protocol wrappers see one hook call per
    attempt, and may drop the hook's result. The runner returned an attempt's
    reports through the hook (this wrapper made that an INTERNALERROR), and
    ran a test in a hook call nested in the loop's own call for it (wrappers
    saw a second, empty call)."""
    (tmp_path / "conftest.py").write_text(_RECORDING_WRAPPER_CONFTEST_SRC)
    (tmp_path / "test_calls.py").write_text(_RECORDING_WRAPPER_TEST_SRC)
    returncode, out = _inner_pytest(tmp_path, *(["-n", "1"] if xdist else []))
    assert returncode == 0 and "2 passed" in out, out
    calls = Counter((tmp_path / "calls.txt").read_text().split())
    assert calls == {"test_deferred_once": 2, "test_warm": 1}, calls


_DUPLICATE_TEST_SRC = textwrap.dedent(
    """
    from quack.cache.async_compile import CompilePending, get_active_pool

    SHA = "a" * 64
    calls = []

    def test_twice():
        # The first instance waits for a kernel and then passes without it;
        # the second, same nodeid, passes while that kernel is still pending.
        calls.append(1)
        if len(calls) == 1:
            get_active_pool()._requests[SHA] = ("fake._compile_kernel", "/nonexistent/a.o")
            raise CompilePending(SHA, "fake._compile_kernel")
    """
)

_DUPLICATE_CONFTEST_SRC = textwrap.dedent(
    """
    pytest_plugins = ["quack.testing.pytest_plugin"]
    failed = []


    def pytest_configure(config):
        from quack.cache import async_compile as ac

        poll = ac.CompilePool.poll

        def fake_poll(self, sha):
            if sha != "a" * 64:
                return poll(self, sha)
            return ("failed", "worker died") if failed else ("pending", None)

        ac.CompilePool.poll = fake_poll


    def pytest_runtest_logfinish(nodeid, location):
        failed.append(nodeid)  # the kernel's worker fails once the second instance passed
    """
)


def test_duplicate_nodeids_keep_their_own_awaited_kernels(tmp_path):
    """Per-test state is the item's, not its nodeid's: with --keep-duplicates
    two items share one. Keyed by nodeid, the second instance's pass dropped
    the kernel the first was waiting for, and a never_built miss went
    unreported under --async-compile-strict."""
    (tmp_path / "conftest.py").write_text(_DUPLICATE_CONFTEST_SRC)
    (tmp_path / "test_dup.py").write_text(_DUPLICATE_TEST_SRC)
    returncode, out = _inner_pytest(
        tmp_path, "--keep-duplicates", "--async-compile-strict", "test_dup.py", "test_dup.py"
    )
    assert "2 passed" in out, out
    assert returncode == 1 and "never_built: worker died" in out, out


_NEVER_BUILT_TEST_SRC = textwrap.dedent(
    """
    from quack.cache.async_compile import CompilePending, get_active_pool

    SHA = "a" * 64
    attempts = []

    def test_served_another_way():
        # First attempt: needs a kernel, the pool takes the request. The
        # retry passes without asking jit_cache for that kernel again.
        attempts.append(1)
        if len(attempts) == 1:
            get_active_pool()._requests[SHA] = ("fake._compile_kernel", "/nonexistent/aaa.o")
            raise CompilePending(SHA, "fake._compile_kernel")
        if "{RETRY}" == "noticed":  # what jit_cache does when a retry lands on a failed request
            get_active_pool().noticed.add(SHA)
        if "{RETRY}" == "skips":
            import pytest

            pytest.skip("not here")
    """
)

_NEVER_BUILT_CONFTEST_SRC = textwrap.dedent(
    """
    pytest_plugins = ["quack.testing.pytest_plugin"]


    def pytest_configure(config):
        # The request for sha "aaa..." fails in its worker and leaves no .o.
        from quack.cache import async_compile as ac

        poll = ac.CompilePool.poll
        failed = ("failed", "compile succeeded but .o was not exported")
        ac.CompilePool.poll = lambda self, s: failed if s == "a" * 64 else poll(self, s)
    """
)


@pytest.mark.parametrize(
    "retry,xdist",
    [("passes", False), ("passes", True), ("noticed", False), ("skips", False)],
    ids=["silent", "xdist", "seen", "skipped"],
)
def test_failed_request_nobody_retried_is_reported(tmp_path, retry, xdist):
    """A pool request that fails, for a kernel the waiting test then passes
    without, must not go unseen: the request and the retry disagree about
    which kernel is needed. This is how one kernel under two disk keys looked
    (``rounding_mode`` as an enum eagerly and an int under torch.compile): a
    worker answered the second key from memory and exported nothing, and the
    test's retry was served from the in-memory cache under the first key. Only
    a per-worker "failed" counter showed it.

    silent/xdist: reported as ``never_built`` and fails ``--async-compile-strict``,
    also when the test ran in an xdist worker. seen: a retry that does come
    back to jit_cache handles the failure itself (compiles in-process, or
    surfaces the expected error), so nothing is reported here. skipped: a
    retry that does not pass says nothing about the kernel either."""
    (tmp_path / "conftest.py").write_text(_NEVER_BUILT_CONFTEST_SRC)
    (tmp_path / "test_never_built.py").write_text(_NEVER_BUILT_TEST_SRC.format(RETRY=retry))
    env = dict(os.environ)
    env.pop("PYTEST_XDIST_WORKER", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--async-compile=2"]
    result = subprocess.run(
        cmd + ["--async-compile-strict"] + (["-n", "1"] if xdist else []),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = result.stdout + result.stderr
    assert ("1 skipped" if retry == "skips" else "1 passed") in out, out
    if retry != "passes":
        assert result.returncode == 0 and "never_built" not in out, out
    else:
        assert result.returncode == 1, out
        assert "1 kernels not compiled by the pool" in out, out
        assert "never_built: compile succeeded but .o was not exported" in out, out
        assert "test_never_built.py::test_served_another_way" in out, out


def test_strict_refuses_a_disabled_cache(tmp_path):
    """The pool hands kernels over as .o files in the disk cache. With the
    cache off every kernel compiles in-process and none of it is recorded,
    so a strict run would pass having checked nothing."""
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["quack.testing.pytest_plugin"]\n')
    (tmp_path / "test_nothing.py").write_text("def test_nothing():\n    pass\n")
    env = dict(os.environ, QUACK_CACHE_ENABLED="0")
    env.pop("PYTEST_XDIST_WORKER", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--async-compile=2"]
    strict = subprocess.run(
        cmd + ["--async-compile-strict"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert strict.returncode == 4 and "needs the kernel cache" in strict.stderr, (
        strict.stdout + strict.stderr
    )
    plain = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert plain.returncode == 0, plain.stdout + plain.stderr


@pytest.mark.parametrize("progressing", [True, False], ids=["busy", "stalled"])
def test_through_pool_waits_while_the_pool_progresses(through_pool, monkeypatch, progressing):
    """through_pool's kernels queue behind every compile submitted before
    them. On a cold full-suite run that wait outlasted its fixed 180 s budget
    (test_epi_mod_async_compile: "compile never completed"). busy: a pool that
    keeps finishing compiles is waited for however long it takes. stalled: one
    that finishes nothing for ``timeout`` seconds still fails the call."""
    import contextlib
    import time

    from quack.cache import async_compile

    class _Pool:
        n_submitted = done = 0

        def stats(self):  # busy: a compile finishes between any two polls
            self.done += progressing
            return {"done": self.done}

    monkeypatch.setattr(async_compile, "pool_scope", lambda: contextlib.nullcontext(_Pool()))
    t0 = time.monotonic()

    def call():  # busy: its kernel lands after 1 s, past the 0.4 s timeout
        if not progressing or time.monotonic() - t0 < 1.0:
            raise CompilePending("a" * 64, "fake.compile")
        return "ok"

    if progressing:
        assert through_pool(call, timeout=0.4) == ("ok", 0)
    else:
        with pytest.raises(AssertionError, match="finished no compile"):
            through_pool(call, timeout=0.4)


_STOP_TEST_SRC = textwrap.dedent(
    """
    from pathlib import Path

    from quack.cache.async_compile import CompilePending

    attempts = []

    def test_fails_on_retry():
        attempts.append(1)
        if len(attempts) == 1:
            raise CompilePending("a" * 64, "fake.compile")
        assert False

    def test_after_the_stop():
        Path(__file__).with_name("ran").touch()
    """
)


def test_maxfail_stop_on_a_retry(tmp_path):
    """-x stops the session at a failing retry, which runs just before the
    next test: that test must not run. Then the session ends with exit status
    1 (pytest's Session.Failed), as a finished run with failures does, and
    the integrity check must not take the tests the stop skipped for reports
    lost to a worker crash."""
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["quack.testing.pytest_plugin"]\n')
    (tmp_path / "test_stop.py").write_text(_STOP_TEST_SRC)
    returncode, out = _inner_pytest(tmp_path, "-x")
    assert returncode == 1 and "1 failed" in out, out
    assert "INTEGRITY ERROR" not in out, out
    assert not (tmp_path / "ran").exists(), out


_WEDGE_QUEUE_TEST_SRC = textwrap.dedent(
    """
    from quack.cache.async_compile import CompilePending, get_active_pool
    from quack.testing import pytest_plugin as tp

    attempts = []

    def test_deferred():
        attempts.append(get_active_pool() is not None)
        if len(attempts) == 1:
            raise CompilePending("a" * 64, "fake.compile")
        assert attempts[-1] == {POOL_ACTIVE_ON_RETRY}, attempts
        if len(attempts) == 2:
            # Defer again, late in a long final drain: this kernel's budget
            # starts now, not when the drain did.
            tp.time.sleep(10)
            raise CompilePending("b" * 64, "fake.compile")

    def test_slow_neighbour():
        tp.time.sleep(10)  # past the first kernel's deadline
    """
)

_WEDGE_QUEUE_CONFTEST_SRC = textwrap.dedent(
    """
    import types
    from collections import Counter

    pytest_plugins = ["quack.testing.pytest_plugin"]
    polls = Counter()


    def pytest_configure(config):
        from quack.cache import async_compile as ac
        from quack.testing import pytest_plugin as tp

        # The runner's clock, advanced only by sleeping (the runner's, and the
        # tests'): no real time window matters.
        now = [0.0]
        tp.time = types.SimpleNamespace(
            monotonic=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s)
        )
        tp._DeferAndRetry._WEDGE_TIMEOUT_S = 1.0

        def poll(self, sha):
            # wedged: kernels stay pending. healthy: each is pending at its
            # first poll and ready at the next.
            polls[sha] += 1
            return ("pending" if {WEDGED} or polls[sha] == 1 else "new"), None

        ac.CompilePool.poll = poll
    """
)


@pytest.mark.parametrize("xdist", [False, True], ids=["single", "xdist"])
@pytest.mark.parametrize("wedged", [False, True], ids=["healthy", "wedged"])
def test_wedge_timeout_ignores_queue_time(tmp_path, wedged, xdist):
    """The wedge timeout is about a kernel that stays pending, not about how
    long its deferred test waited for a retry; and each deferral has a budget
    of its own.

    A test is retried only between other tests, so in a long session its
    retry may come well after the timeout although its kernel was ready long
    before. Taking that for a wedged pool re-ran the test with the pool
    suppressed, and every further cold kernel it needed compiled in-process
    (seen as ``suppressed`` fallbacks in a 16 min single-process run). The
    xdist final drain had one deadline for the whole drain: a test deferring
    again late in a long drain was taken for wedged at once.

    healthy: the first kernel is pending until test_slow_neighbour has taken
    it past its deadline, then ready: the retry runs with the pool active, as
    does the one after the second kernel, deferred on late in the drain.
    wedged: kernels poll pending forever, so each retry escapes to an
    in-process compile once its kernel's deadline passes.
    """
    (tmp_path / "conftest.py").write_text(_WEDGE_QUEUE_CONFTEST_SRC.format(WEDGED=wedged))
    (tmp_path / "test_queue.py").write_text(
        _WEDGE_QUEUE_TEST_SRC.format(POOL_ACTIVE_ON_RETRY=not wedged)
    )
    returncode, out = _inner_pytest(tmp_path, *(["-n", "1"] if xdist else []))
    assert returncode == 0 and "2 passed" in out, out
    if not xdist:  # (an xdist worker's pool summary doesn't reach the output)
        assert "2 test deferrals" in out, out


_INNER_TEST_SRC = textwrap.dedent(
    """
    import torch
    from quack.rmsnorm import rmsnorm_fwd

    def _run(N):
        x = torch.randn(32, N, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, device="cuda", dtype=torch.float32)
        out, *_ = rmsnorm_fwd(x, w)
        ref = torch.nn.functional.rms_norm(x.float(), (N,), weight=w).to(x.dtype)
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=1e-2)

    def test_a():
        _run({N_A})

    def test_b():
        _run({N_B})
    """
)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_defer_and_retry_cold_session(tmp_path):
    """Cold keys + --async-compile: tests defer, retry, and verify numerics.

    Two test *modules* are used so the deferred retry crosses a module
    boundary — the regression that produced pytest's "previous item was not
    torn down properly" SetupState assert before the misprediction guard in
    ``_DeferAndRetry._attempt`` (fixture teardown is scoped to a *predicted* next item
    and deferral breaks the prediction chain).
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    # Unusual N values so the keys are cold in the throwaway cache dir.
    (suite / "test_defer_mod_a.py").write_text(_INNER_TEST_SRC.format(N_A=704, N_B=1216))
    (suite / "test_defer_mod_b.py").write_text(_INNER_TEST_SRC.format(N_A=1728, N_B=2240))
    (suite / "conftest.py").write_text('pytest_plugins = ["quack.testing.pytest_plugin"]\n')

    env = dict(os.environ)
    env["QUACK_CACHE_DIR"] = str(tmp_path / "cache")
    env.pop("PYTEST_XDIST_WORKER", None)  # inner session is single-proc
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--async-compile=4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=str(suite),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"inner pytest failed:\n{out}"
    assert "4 passed" in out, out
    # The session must actually have exercised the pool + deferral path.
    assert "keys submitted" in out, out
    n_submitted = int(out.split("async-compile: ")[1].split(" keys")[0])
    assert n_submitted >= 4, out  # one fwd key per distinct N


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_warm_session_with_flag_submits_nothing(tmp_path):
    """Warm cache + --async-compile must be a no-op (no submits, no defers).

    Guards the lazy-pool property: enabling the flag unconditionally (e.g.
    in an alias or CI default) must cost nothing when the cache is warm.
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_warm.py").write_text(_INNER_TEST_SRC.format(N_A=704, N_B=1216))
    (suite / "conftest.py").write_text('pytest_plugins = ["quack.testing.pytest_plugin"]\n')

    env = dict(os.environ)
    env["QUACK_CACHE_DIR"] = str(tmp_path / "cache")
    env.pop("PYTEST_XDIST_WORKER", None)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(suite),
        "-q",
        "-p",
        "no:cacheprovider",
        "--async-compile=4",
    ]
    kw = dict(capture_output=True, text=True, env=env, timeout=600, cwd=str(suite))
    first = subprocess.run(cmd, **kw)
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run(cmd, **kw)
    out = second.stdout + second.stderr
    assert second.returncode == 0, out
    assert "0 keys submitted" in out, out
    assert "0 test deferrals" in out, out


_WEDGED_POOL_CONFTEST_SRC = textwrap.dedent(
    """
    pytest_plugins = ["quack.testing.pytest_plugin"]


    def pytest_configure(config):
        # Simulate a wedged pool: submissions succeed but poll never resolves
        # — exactly what a hung pool worker (still holding the per-key flock)
        # looks like to every waiter. Shrink the wedge deadline so the escape
        # path runs quickly.
        from quack.cache import async_compile as ac
        from quack.testing import pytest_plugin as tp

        tp._DeferAndRetry._WEDGE_TIMEOUT_S = 2.0
        ac.CompilePool.poll = lambda self, sha: ("pending", None)
    """
)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_xdist_drain_escapes_wedged_pool(tmp_path):
    """A sha that polls "pending" forever must not deadlock the final drain.

    Regression (Aug 2026, run 31081200282): pool compiles froze mid-job and
    every xdist worker's end-of-session drain skipped its still-pending items
    forever — the drain never retried a pending sha, so the wedge escape's
    attempts bump never took effect and the CI leg died at
    ``timeout-minutes: 30``. With the fix, the wedge deadline forces the
    remaining items through in-process; without it, this inner session hangs
    (the outer ``timeout=`` converts that into a failure).
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    # Unusual N values so the keys are cold in the throwaway cache dir.
    (suite / "test_wedged.py").write_text(_INNER_TEST_SRC.format(N_A=2752, N_B=3264))
    (suite / "conftest.py").write_text(_WEDGED_POOL_CONFTEST_SRC)

    env = dict(os.environ)
    env["QUACK_CACHE_DIR"] = str(tmp_path / "cache")
    env.pop("PYTEST_XDIST_WORKER", None)  # the inner master must not be a worker
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--async-compile=4",
            "-n",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=str(suite),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"inner pytest failed:\n{out}"
    assert "2 passed" in out, out


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_oom_retry_compile_pending_defers_not_warns(tmp_path):
    """OOM-retry hookwrapper + CompilePending on the retry must defer cleanly.

    Regression: tests/conftest.py's OOM-retry hook re-invokes
    ``item.runtest()`` inside an old-style hookwrapper teardown. When the
    retry hit a cold jit_cache miss (--async-compile), the resulting
    ``CompilePending`` escaped the teardown: pluggy emitted
    ``PluggyTeardownRaisedWarning``, the plugin's defer machinery (which had
    already run for this phase) never saw it, and the test reported as a
    CompilePending *failure* instead of deferring (or, worse, passing from a
    half-run attempt). Full analysis: the ``pytest_runtest_call`` docstring
    in tests/conftest.py.

    Uses the real repo conftest so the actual hook code is exercised. The
    inner test scripts the exact sequence: attempt 1 raises a synthetic OOM,
    the in-teardown retry raises CompilePending, and the defer-loop retry
    verifies the deferred attempt released its tensor without cyclic GC. The
    sha is unknown to the pool, so it polls "new" and re-runs immediately.
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    shutil.copy(Path(__file__).parent / "conftest.py", suite / "conftest.py")
    (suite / "test_oom_retry.py").write_text(
        textwrap.dedent(
            """
            import gc
            import weakref

            import torch
            from quack.cache.async_compile import CompilePending

            CALLS = {"n": 0}
            deferred_tensor = None

            def test_oom_then_cold_compile():
                global deferred_tensor
                CALLS["n"] += 1
                if CALLS["n"] == 1:
                    raise torch.OutOfMemoryError("synthetic CUDA out of memory")
                if CALLS["n"] == 2:
                    # This is the conftest OOM-retry's item.runtest() call:
                    # simulate the retry hitting a cold kernel compile.
                    gc.disable()
                    tensor = torch.empty(1)
                    deferred_tensor = weakref.ref(tensor)
                    raise CompilePending("a" * 64, "fake._compile_rmsnorm_bwd")
                try:
                    assert CALLS["n"] == 3
                    assert deferred_tensor() is None, "OOM retry still owns its deferred tensor"
                finally:
                    gc.enable()
            """
        )
    )

    env = dict(os.environ)
    env["QUACK_CACHE_DIR"] = str(tmp_path / "cache")
    env.pop("PYTEST_XDIST_WORKER", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--async-compile=2",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=str(suite),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"inner pytest failed:\n{out}"
    assert "1 passed" in out, out
    assert "PluggyTeardownRaisedWarning" not in out, out
    assert "1 test deferrals" in out, out


def test_gpu_blind_device_attr_shim():
    """GPU-blind workers must answer the DSL's one trace-time driver query
    (max smem per SM, needed when a launch sets min_blocks_per_mp > 1) from
    the static arch table — cuDeviceGetAttribute raises NOT_INITIALIZED with
    CUDA_VISIBLE_DEVICES="" and would fail every occupancy-bound kernel key
    (e.g. the W4 small-N decode configs) over to in-process compiles."""
    code = textwrap.dedent(
        """
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["CUTE_DSL_ARCH"] = "sm_90a"
        from quack.cache.async_compile import _install_gpu_blind_device_attrs

        _install_gpu_blind_device_attrs()
        from cutlass.base_dsl.runtime import cuda as cuda_helpers

        attr = (
            cuda_helpers.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        )
        # sm_90 SM total: the DSL's per-CTA capacity + the 1 KiB reserved slice
        assert cuda_helpers.get_device_attribute(attr) == 233472
        print("shim-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0 and "shim-ok" in result.stdout, result.stderr


def test_pool_arch_pin_survives_early_dsl_construction():
    """The worker's ptxas target must land on a DSL singleton that was built
    BEFORE the pool's env pinning ran.

    Regression (cutlass-dsl 4.6.2 CI image): importing
    ``quack.cache._pool_preload`` executes the parent packages first, and
    ``quack/__init__`` pulls in cutlass — constructing the DSL env manager
    before ``CUTE_DSL_ARCH`` is set. 4.6.2 snapshots the env var at
    construction (4.6.0 read it lazily), so env pinning alone left the
    singleton unlatched; a GPU-blind worker's first ``.arch`` read then fell
    back to ``detect_gpu_arch()``'s sm_100a default, and every pool ``.o``
    on the B300 runner (sm_103a) failed to load with
    cudaErrorNoKernelImageForDevice — which the 4.6.2 runtime escalated into
    a permanent spinlock hang on the process's next launch. The explicit
    ``_pin_dsl_arch`` re-latch in ``_pool_initializer`` / ``_pool_preload``
    is what this test protects.
    """
    code = textwrap.dedent(
        """
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("CUTE_DSL_ARCH", None)
        # Simulate the parent-package import chain: cutlass (and the DSL
        # env-manager singleton) come up with no CUTE_DSL_ARCH in sight.
        import quack.cache  # noqa: F401

        # The pool's env pinning + re-latch, as _pool_initializer runs them.
        from quack.cache.async_compile import _pool_initializer

        _pool_initializer("90", "sm_90a")

        from cutlass.cutlass_dsl import CuTeDSL

        arch = CuTeDSL._get_dsl().envar.arch
        assert arch == "sm_90a", f"worker would compile for {arch}, not sm_90a"
        print("arch-pin-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0 and "arch-pin-ok" in result.stdout, result.stderr


def test_oom_retry_recognizes_cutlass_cuda_oom():
    """The OOM-retry hook must cover kernel-side OOMs, not only torch's.

    Regression: cutlass-dsl 4.8 raises ``CUDADialectError`` for a failed CUDA
    allocation at kernel load/launch. It derives from ``Exception`` rather than
    ``RuntimeError``, so the hook stopped retrying and a small test that ran
    after a memory-heavy one on the same xdist worker (torch still caching most
    of the GPU) failed instead of freeing the cache and retrying.
    """
    from conftest import _is_oom

    class CUDADialectError(Exception):  # same base as cutlass's class
        pass

    msg = "\nerror: cudaErrorMemoryAllocation (error code: 2) \nCUDA out of memory error.\n"
    assert _is_oom(CUDADialectError, CUDADialectError(msg))
    assert _is_oom(RuntimeError, RuntimeError("CUDA out of memory. Tried to allocate 1024.00 MiB"))
    assert not _is_oom(CUDADialectError, CUDADialectError("cudaErrorInvalidValue (error code: 1)"))
    assert not _is_oom(KeyboardInterrupt, KeyboardInterrupt("out of memory"))
    # Text alone is not enough: a failure merely quoting an OOM must still fail.
    assert not _is_oom(AssertionError, AssertionError("inner run printed: CUDA out of memory"))
