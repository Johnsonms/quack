# Copyright (c) 2026, Tri Dao.
"""Reusable pytest plugin for QuACK async kernel compilation.

Adds the ``--async-compile[=N]`` CLI flag: on a kernel-compile ``.o``-cache
miss, the compile is shipped to a pool of N CPU workers (forkserver sidecar,
see :mod:`quack.cache.async_compile`), the test is deferred while other tests
run, and it is retried once the ``.o`` is exported. Zero overhead when the
cache is warm. Works single-process and under xdist (both ``load`` and
``worksteal`` schedulers).

To opt in, add this line to your ``conftest.py``::

    pytest_plugins = ["quack.testing.pytest_plugin"]
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import time
import traceback
import warnings
from collections import Counter, defaultdict
from contextlib import nullcontext

import pytest
import torch
from _pytest.runner import get_reraise_exceptions, runtestprotocol
from _pytest.warnings import catch_warnings_for_item

from quack.cache.async_compile import (
    CompilePending,
    activate,
    deactivate,
    drain_fallbacks,
    get_active_pool,
    suppress_pool,
)


def pytest_addoption(parser):
    parser.addoption(
        "--async-compile",
        type=int,
        nargs="?",
        const=32,
        default=None,
        metavar="N",
        help=(
            "On a kernel-compile cache miss, submit the compile to a pool of "
            "N CPU workers, defer the test, and retry it once the .o is "
            "ready. Zero overhead when the cache is warm."
        ),
    )
    parser.addoption(
        "--async-compile-strict",
        action="store_true",
        help=(
            "Exit non-zero if any kernel was not compiled by the active pool: "
            "it compiled in-process instead, or its pool compile failed and "
            "the test passed without it (tests still run to completion and "
            "report their real results)."
        ),
    )


# --- pytest speedups (independent of --async-compile) --------------------------

# Saved originals so ``pytest_unconfigure`` can restore pytest internals we
# monkey-patched in ``pytest_configure``. Set to ``None`` when the
# corresponding patch was skipped (e.g. pytest internals didn't match what
# we expected, or the env-var opt-out was set).
_orig_compat_getfuncargnames = None
_orig_fixtures_getfuncargnames = None


def _install_getfuncargnames_cache() -> None:
    """Cache ``_pytest.compat.getfuncargnames`` by stable function identity.

    Performance background
    ----------------------
    Pytest's fixture resolution calls ``getfuncargnames`` ~20 times per test
    (once per active fixture / parametrize axis). Each call runs
    ``inspect.signature(function)`` from scratch — a deep AST/Signature build
    with no caching upstream. For a 2k-test file this is ~40k
    ``inspect.signature`` invocations and ~1 s of pure CPU per worker. See
    pytest-dev/pytest#11284 (open since 2023) for the underlying root cause:
    pytest probes the entire fixture closure for every test instead of just
    the test's ``initialnames``.

    Caching by ``id(function)`` does NOT work: the ``request`` fixture is
    rebuilt per test, producing thousands of distinct function objects with
    identical signatures. Cache by ``(qualname, co_filename, co_firstlineno)``
    instead — that triple is stable across the per-test wrappers and uniquely
    identifies a Python function definition.

    Deliberate tradeoffs
    --------------------
    This is a monkey-patch of pytest's private API (``_pytest.compat`` and
    ``_pytest.fixtures``). We accept this because (a) the upstream fix
    (#11284) has been stuck in deprecation cycles for years, and (b) the
    speedup is ~1 s wall per file for parametrize-heavy suites. To minimize
    risk we:

    * Validate that ``getfuncargnames`` has the expected ``(function, *, name,
      cls)`` signature before patching, and skip silently with a warning if
      pytest's internals have changed.
    * Stash the originals so ``pytest_unconfigure`` restores them.
    * Honor ``QUACK_PYTEST_NO_GETFUNCARGNAMES_CACHE=1`` to opt out at runtime.

    Subtlety: ``_pytest.fixtures`` does ``from .compat import getfuncargnames``
    at import time, so we must rebind the name on both modules.
    """
    global _orig_compat_getfuncargnames, _orig_fixtures_getfuncargnames

    if os.environ.get("QUACK_PYTEST_NO_GETFUNCARGNAMES_CACHE"):
        return

    try:
        import _pytest.compat as _compat
        import _pytest.fixtures as _fixtures
    except ImportError:
        return  # pytest internals not where we expect them

    orig = getattr(_compat, "getfuncargnames", None)
    if orig is None or getattr(_fixtures, "getfuncargnames", None) is not orig:
        warnings.warn(
            "quack.testing.pytest_plugin: skipping getfuncargnames cache; "
            "pytest internals (_pytest.compat / _pytest.fixtures) do not "
            "match expected shape. Tests still run, just without the "
            "~1 s/file fixture-resolution speedup.",
            stacklevel=2,
        )
        return

    # Verify the signature is still `(function, *, name, cls)`. If pytest bumps
    # the API we'd rather skip than silently miscache.
    try:
        sig = inspect.signature(orig)
        params = sig.parameters
        expected = ("function", "name", "cls")
        if not all(p in params for p in expected):
            raise ValueError(f"unexpected params {tuple(params)!r}")
    except (TypeError, ValueError) as e:
        warnings.warn(
            f"quack.testing.pytest_plugin: skipping getfuncargnames cache; "
            f"signature check failed ({e!r}). Tests still run normally.",
            stacklevel=2,
        )
        return

    cache: dict = {}

    def _identity_key(function):
        code = getattr(function, "__code__", None)
        if code is None:
            return ("__obj__", function)
        return (
            getattr(function, "__qualname__", None) or function.__name__,
            code.co_filename,
            code.co_firstlineno,
        )

    def _patched(function, *, name="", cls=None, **kw):
        # **kw forwards keyword params added by newer pytest (8.1: is_method)
        # and joins the cache key, since they can change the result
        try:
            key = (_identity_key(function), name, cls, tuple(sorted(kw.items())))
        except (AttributeError, TypeError):
            return orig(function, name=name, cls=cls, **kw)
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = orig(function, name=name, cls=cls, **kw)
        cache[key] = result
        return result

    _orig_compat_getfuncargnames = orig
    _orig_fixtures_getfuncargnames = _fixtures.getfuncargnames  # == orig
    _compat.getfuncargnames = _patched
    # Captured via `from .compat import getfuncargnames` at import time.
    _fixtures.getfuncargnames = _patched


def _restore_getfuncargnames_cache() -> None:
    """Undo ``_install_getfuncargnames_cache``. No-op if not installed."""
    global _orig_compat_getfuncargnames, _orig_fixtures_getfuncargnames
    if _orig_compat_getfuncargnames is None:
        return
    try:
        import _pytest.compat as _compat
        import _pytest.fixtures as _fixtures
    except ImportError:
        return
    _compat.getfuncargnames = _orig_compat_getfuncargnames
    _fixtures.getfuncargnames = _orig_fixtures_getfuncargnames
    _orig_compat_getfuncargnames = None
    _orig_fixtures_getfuncargnames = None


def _disable_unused_accelerator_lazy_call() -> None:
    """No-op ``_lazy_call`` for accelerators that aren't available.

    Every ``torch.manual_seed(seed)`` fans out across CUDA, MPS, XPU, MTIA, and
    any custom device. For each *uninitialized* backend, ``_lazy_call`` takes
    a slow path that calls ``traceback.format_stack()`` to record where the
    seed was queued from. Under pytest the call stack is deep, so each
    ``format_stack`` costs ~1 ms; across thousands of tests this is several
    seconds of pure CPU overhead per worker, even though no XPU/MTIA work is
    ever submitted.

    Subtlety: ``torch.xpu/random.py`` and ``torch.cuda/random.py`` do
    ``from . import _lazy_call`` at import time, so we must replace the name
    on the submodule too — patching only the package attribute is not enough.
    """
    nop = lambda callable, **kwargs: None  # noqa: E731
    if not torch.xpu.is_available():
        torch.xpu._lazy_call = nop
        torch.xpu.random._lazy_call = nop  # captured via `from . import _lazy_call`
    if not torch.mtia.is_available():
        torch.mtia._lazy_call = nop


# --- Session setup and teardown -----------------------------------------------


def pytest_configure(config):
    """Set up the async compile pool and pytest-internal speedup patches."""
    jobs = config.getoption("--async-compile", default=None)
    _fallbacks_by_test.clear()  # a second pytest.main() in one interpreter starts clean
    if jobs is not None:
        import quack.cache as cache_state

        if config.getoption("--async-compile-strict") and not cache_state.CACHE_ENABLED:
            # the pool hands kernels over as .o files in the disk cache: with
            # it off every kernel compiles in-process and nothing is recorded
            raise pytest.UsageError(
                "--async-compile-strict needs the kernel cache (QUACK_CACHE_ENABLED=0 is set)"
            )
        drain_fallbacks()  # anything recorded before this session is not a test's
        worker = os.environ.get("PYTEST_XDIST_WORKER")
        is_xdist_master = worker is None and getattr(config.option, "numprocesses", None)
        if not is_xdist_master:
            # Real test-running process (single-proc pytest or xdist worker).
            n_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
            pool = activate(max(2, jobs // n_workers))
            pool.prewarm()  # sidecar import overlaps collection, not the first miss
            config.pluginmanager.register(_DeferAndRetry(pool, config), "quack-defer")

    # Speed up manual_seed by short-circuiting the queued-seed path on
    # unavailable accelerators.
    _disable_unused_accelerator_lazy_call()

    # Cache inspect.signature() lookups behind _pytest.compat.getfuncargnames.
    # Pytest re-builds signatures ~20x per test during fixture resolution; on a
    # 2k-test file this is several seconds of pure CPU we can avoid.
    _install_getfuncargnames_cache()


def pytest_unconfigure(config):
    """Tear down the compile pool and undo any pytest-internal patches."""
    pool = get_active_pool()
    if pool is not None:
        stats = pool.stats()
        defer_plugin = config.pluginmanager.get_plugin("quack-defer")
        defers = defer_plugin.defer_count if defer_plugin else 0
        summary = (
            f"async-compile: {stats['submitted']} keys submitted, "
            f"{stats['failed']} failed, {defers} test deferrals"
        )
        detail = [f"  pool compile error [{sha[:12]}]: {err}" for sha, err in stats["errors"][:5]]
        tr = config.pluginmanager.get_plugin("terminalreporter")
        if stats["failed"] and tr is not None:
            # Failures degrade to in-process compiles (tests still pass), so
            # paint the summary red or it reads as routine bookkeeping.
            print()
            tr.write_line(summary, red=True, bold=True)
            for line in detail:
                tr.write_line(line, red=True)
        else:
            print(f"\n{summary}")
            for line in detail:
                print(line)
        deactivate()

    # Always undo the global monkey-patches we installed. This keeps the
    # process clean for downstream callers (e.g. notebook hosts that
    # pytest-main multiple times in the same interpreter).
    _restore_getfuncargnames_cache()


# --- CompilePending: defer the attempt ----------------------------------------


def _defer_compile_pending_call(item, exc: CompilePending) -> None:
    """Defer ``item`` on ``exc``, which its call phase raised and the caller
    swallows. The retry starts from scratch, so free this attempt's locals
    now. Swallowing the exception doesn't break this cycle:
      exception -> traceback -> pluggy frame -> exception
                       |
                       +-------> test frame -> GPU tensors
    so without clearing, the memory would wait for cyclic GC."""
    item._quack_pending_sha = exc.sha
    traceback.clear_frames(exc.__traceback__)
    exc.__traceback__ = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except CompilePending as exc:
        # Stays a setup failure, so pytest skips the body (its fixtures are
        # half built). The attempt's reports are discarded anyway.
        item._quack_pending_sha = exc.sha
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except CompilePending as exc:
        # Made a pass: the attempt's reports are discarded anyway, and a
        # failure would have pytest format a full traceback for every
        # deferral (~100 ms each, nearly doubling a cold run).
        _defer_compile_pending_call(item, exc)


# --- Defer and retry -----------------------------------------------------------
#
# With --async-compile, a jit_cache miss ships the compile to the CPU worker
# pool and raises CompilePending, which defers the attempt (hooks above).
# _DeferAndRetry discards a deferred attempt's reports and retries the test
# once its kernel is ready, while the session runs other tests meanwhile.
#
# The session's test loop (pytest's own, or an xdist worker's) calls
# pytest_runtest_protocol for each test in turn. _DeferAndRetry wraps that
# call: it first retries the deferred tests whose kernel is ready, then runs
# the test itself. Each attempt has a hook call of its own, never nested in
# another test's (the loop's call, or one a retry makes), so pytest's per-test
# wrappers apply to that test alone, and other plugins' wrappers see each
# attempt once. Under xdist, the master hears the protocol complete at
# once, deferred or not (its mark_test_complete is an order-independent
# .remove()), so it keeps the worker fed; a retried test's reports reach it
# later. Tests still deferred when the loop ends are retried after it,
# waiting on the pool (an xdist worker's channel is still open then).

# A test needing K distinct cold kernels defers K times (keys are discovered
# serially: each attempt stops at the first missing kernel). Attempts are
# cheap — an item is only re-run once its awaited sha resolves — so the cap
# just needs to exceed the realistic kernels-per-test count; past it, the
# final attempt compiles in-process (pool suppressed) so a wedged pool can't
# defer a test forever.
_MAX_ATTEMPTS = 20


def _warnings_of(item):
    """pytest's warnings capture for ``item``: its filters, its nodeid."""
    if item is None or not item.config.pluginmanager.has_plugin("warnings"):
        return nullcontext()
    return catch_warnings_for_item(item.config, item.ihook, "runtest", item)


def _stopped(session) -> bool:
    """A test stopped the session: -x, --maxfail (under xdist, the master's)."""
    return bool(session.shouldfail or session.shouldstop)


class _DeferAndRetry:
    """The --async-compile test runner, in a single process or an xdist worker."""

    #: How long a deferred test may wait on a kernel that stays "pending"
    #: before it is retried with the pool suppressed (in-process compile): a
    #: wedged worker, or a foreign flock holder that never produces the .o,
    #: must not hold the session forever. Counted from the deferral, and only
    #: while the kernel polls "pending": a retry may come long after its
    #: kernel is ready, and that says nothing about the pool. Class attribute
    #: so the regression tests can shrink it.
    _WEDGE_TIMEOUT_S = 600.0

    def __init__(self, pool, config):
        self.pool = pool
        self.config = config
        self.deferred = []  # (item, awaited sha, wedge deadline)
        self.attempts = Counter()  # item -> attempts run
        self.fallbacks = defaultdict(list)  # item -> kernels the pool did not compile, so far
        self.awaited = defaultdict(list)  # item -> shas its attempts deferred on
        self.defer_count = 0
        self.retrying = False  # in a retry's hook call, from _retry_ready
        self.wedged = False  # ... of a test whose kernel stayed pending too long
        self.predicted_next = None  # see _teardown_if_mispredicted
        self.last_run = None

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        if not self.retrying:
            # The session loop's call for its next test. Registered after
            # pytest's builtin plugins, this wrapper encloses their per-test
            # wrappers (warnings capture and filters, assertion hooks,
            # faulthandler timeout): retry the ready deferred tests before any
            # of those is active for this test, each in a hook call of its own.
            self._retry_ready(nextitem=item)
            self._teardown_if_mispredicted(item)
        return (yield)

    # (pytest only registers attributes named pytest_*, whatever specname says)
    @pytest.hookimpl(tryfirst=True, specname="pytest_runtest_protocol")
    def pytest_quack_attempt(self, item, nextitem):
        """One attempt of ``item``: log its reports, or defer it again.

        Wedged or past _MAX_ATTEMPTS, it runs with the pool suppressed: its
        kernels compile in-process, and it can't defer.
        """
        if _stopped(item.session):
            return True  # e.g. -x, and a retry just before this test failed
        self.attempts[item] += 1
        force_sync = self.wedged or self.attempts[item] > _MAX_ATTEMPTS
        self.predicted_next, self.last_run = nextitem, item
        item._quack_pending_sha = None
        with suppress_pool() if force_sync else nullcontext():
            reports = runtestprotocol(item, nextitem=nextitem, log=False)
        pending_sha = item._quack_pending_sha
        # Kept across deferred attempts (their reports are dropped) and
        # attached to the reports that are logged: user_properties travel
        # with a report, so they reach the xdist controller too.
        self.fallbacks[item] += [list(f) for f in drain_fallbacks()]
        if pending_sha:
            self.awaited[item].append(pending_sha)
            self.deferred.append((item, pending_sha, time.monotonic() + self._WEDGE_TIMEOUT_S))
            self.defer_count += 1
            # The attempt's teardown did run: an error there is a real bug
            # (leaked fixture state) that the retry may not reproduce.
            failed_teardown = [r for r in reports if r.when == "teardown" and r.failed]
            if failed_teardown:
                self._log(item, failed_teardown)
            return True
        fallbacks = self.fallbacks.pop(item)
        awaited = self.awaited.pop(item, [])
        if all(report.passed for report in reports):
            # Kernels this test waited for that failed in the pool and that it
            # then passed without asking for again: built by nobody. (A retry
            # that fails or skips says nothing about the kernel.)
            for sha in awaited:
                orphan = self.pool.orphaned(sha)
                if orphan is not None:
                    fallbacks.append([sha, orphan[0], "never_built", orphan[1], 0.0])
        if fallbacks:
            reports[-1].user_properties.append((_FALLBACK_PROP, fallbacks))
        self._log(item, reports)
        return True

    @pytest.hookimpl(wrapper=True)
    def pytest_runtestloop(self, session):
        result = yield  # an aborted loop (-x, ^C, collection errors) raises past this
        # Retry the tests still deferred (an xdist worker's channel is still
        # open). Not after a stop: an xdist worker's loop ends normally then,
        # and running more tests would only delay shutdown.
        while self.deferred and not _stopped(session):
            if not self._retry_ready(nextitem=None):
                time.sleep(0.2)  # everything left is waiting on the pool
        return result

    def _retry_ready(self, nextitem) -> bool:
        """Retry the deferred tests whose kernel is no longer pending, or has
        been pending past the wedge deadline. Returns whether any ran."""
        waiting, self.deferred = self.deferred, []
        ran = False
        for item, sha, deadline in waiting:
            pending = self.pool.poll(sha)[0] == "pending"
            if (pending and time.monotonic() <= deadline) or _stopped(item.session):
                self.deferred.append((item, sha, deadline))
                continue
            self._teardown_if_mispredicted(item)
            self.retrying, self.wedged = True, pending
            try:
                item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
            finally:
                self.retrying = self.wedged = False
            ran = True
        return ran

    def _teardown_if_mispredicted(self, item) -> None:
        """pytest scopes fixture teardown to the next item an attempt predicts
        (its nextitem; SetupState.teardown_exact), and asserts the prediction
        at the next setup ("previous item was not torn down properly"). A
        retry breaks the chain, so tear down fully before ``item`` runs, on
        behalf of the last test run: plain pytest runs these finalizers in its
        teardown, so their warnings and errors are reported under it. (Not
        through its teardown hooks: its protocol is over, and plugins' hooks
        may expect what that dropped, e.g. ``item.funcargs``.)"""
        ss = item.session._setupstate
        if self.predicted_next is item or not ss.stack:
            return
        last = self.last_run
        with _warnings_of(last):
            call = pytest.CallInfo.from_call(
                lambda: ss.teardown_exact(None),
                when="teardown",
                reraise=get_reraise_exceptions(last.config),
            )
        if call.excinfo is not None:
            report = pytest.TestReport.from_item_and_call(last, call)
            report.sections = []  # its captured output went with its own reports
            self._log(last, [report])

    @functools.cached_property
    def _xdist_worker(self):
        """xdist's WorkerInteractor (it registers itself as a plugin), or None."""
        plugins = self.config.pluginmanager.get_plugins()
        return next((p for p in plugins if type(p).__name__ == "WorkerInteractor"), None)

    def _log(self, item, reports) -> None:
        worker = self._xdist_worker
        if worker is not None:
            # WorkerInteractor.pytest_runtest_logreport asserts (and
            # serializes) session.items[item_index].nodeid == report.nodeid,
            # and a retried item's reports are logged while another is current.
            saved, worker.item_index = worker.item_index, item.session.items.index(item)
        try:
            item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
            for report in reports:
                item.ihook.pytest_runtest_logreport(report=report)
            item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
        finally:
            if worker is not None:
                worker.item_index = saved


# --- Session-end reporting ----------------------------------------------------
#
# Pool misses: the kernels the active pool did not compile, attached to a
# test's last report and gathered where reports are logged.
#
# Report integrity: the defer machinery reports an xdist item's protocol as complete *before*
# its reports are sent (that's what keeps the master streaming new items to
# the worker). The one integrity hole this opens: a worker that crashes with
# deferred items still stashed loses them SILENTLY — the master already
# counted them complete, so nothing else will notice the missing reports.
# This check closes the hole: at session end, any collected-but-unreported,
# non-deselected test flips the exit status to failure.

_reported_nodeids: set = set()
_deselected_nodeids: set = set()

_FALLBACK_PROP = "quack_async_fallback"
# nodeid -> [[sha, qualname, reason, detail, seconds], ...], filled where the
# reports are logged (single process, or the xdist controller).
_fallbacks_by_test: dict = {}


def pytest_deselected(items):
    for item in items:
        _deselected_nodeids.add(item.nodeid)


def pytest_runtest_logreport(report):
    _reported_nodeids.add(report.nodeid)
    for name, value in report.user_properties:
        if name == _FALLBACK_PROP:
            _fallbacks_by_test.setdefault(report.nodeid, []).extend(value)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _fallback_cause(reason: str, detail: str) -> str:
    """One line naming the cause: the exception line of a worker error,
    without its traceback tail."""
    detail = _ANSI.sub("", detail).split(" [worker:")[0]
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    return f"{reason}: {' '.join(lines[:2])[:240]}" if lines else reason


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Name every kernel the active pool should have compiled and did not:
    grouped by compile function and cause, with the tests that hit it."""
    if config.getoption("--async-compile", default=None) is None or not _fallbacks_by_test:
        return
    tr = terminalreporter
    # (kernel compile fn, cause) -> {sha: seconds}, tests that hit it
    groups: dict = {}
    for nodeid, events in _fallbacks_by_test.items():
        for sha, qualname, reason, detail, seconds in events:
            keys, tests = groups.setdefault(
                (qualname, _fallback_cause(reason, detail)), ({}, set())
            )
            keys[sha] = seconds
            tests.add(nodeid)
    n_keys = sum(len(keys) for keys, _ in groups.values())
    total = sum(sum(keys.values()) for keys, _ in groups.values())
    tr.write_sep(
        "=",
        f"async-compile: {n_keys} kernels not compiled by the pool "
        f"({total:.0f}s in-process) in {len(_fallbacks_by_test)} tests",
        red=True,
        bold=True,
    )
    for (qualname, cause), (keys, tests) in sorted(
        groups.items(), key=lambda kv: -sum(kv[1][0].values())
    ):
        tr.write_line(
            f"{qualname}: {len(keys)} kernels, {sum(keys.values()):.0f}s, {len(tests)} tests",
            bold=True,
        )
        tr.write_line(f"  {cause}")
        shown = sorted(tests)[: None if config.option.verbose > 0 else 8]
        for nodeid in shown:
            tr.write_line(f"    {nodeid}")
        if len(shown) < len(tests):
            tr.write_line(f"    ... {len(tests) - len(shown)} more (-v lists all)")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if config.getoption("--async-compile", default=None) is None:
        return  # integrity risk only exists with the defer machinery
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return  # workers see the full collection but run a subset
    if getattr(config.option, "collectonly", False):
        return
    if config.getoption("--async-compile-strict") and _fallbacks_by_test and exitstatus == 0:
        session.exitstatus = 1  # the terminal summary names the kernels and tests
    if exitstatus not in (0, 1) or session.shouldfail or session.shouldstop:
        return  # stopped early (-x, ^C, internal error): missing reports expected
    missing = {item.nodeid for item in session.items} - _reported_nodeids - _deselected_nodeids
    if not missing:
        return
    tr = config.pluginmanager.get_plugin("terminalreporter")
    lines = [
        f"async-compile INTEGRITY ERROR: {len(missing)} collected test(s) produced no "
        "report (deferred tests lost to a worker crash?):"
    ] + [f"  {nodeid}" for nodeid in sorted(missing)[:20]]
    for line in lines:
        (tr.write_line if tr else print)(line)
    session.exitstatus = 1
