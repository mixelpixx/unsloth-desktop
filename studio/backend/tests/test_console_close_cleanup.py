# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Closing the console window takes studio.pid, the per-port record and the startup marker back.

Windows raises no signal for a closed window, a logoff or a system shutdown, and ends the process about
five seconds later, so neither atexit nor the server thread's exit path can be counted on. The handler
was installed only by `python run.py`, and only once run_server() had returned: `unsloth studio` never
had one, and a window closed during the minute of imports skipped it either way.
"""

from __future__ import annotations

import ast
import os
import sys
import threading
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import run  # noqa: E402
from utils import cache_cleanup  # noqa: E402

IS_WINDOWS = sys.platform == "win32"
CTRL_C_EVENT = 0
CTRL_CLOSE_EVENT = 2


@pytest.fixture(autouse = True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_studio_root", lambda: tmp_path)
    monkeypatch.setattr(run, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(run, "_OWN_PID_FILE", None)
    monkeypatch.setattr(run, "_OWN_STARTUP_MARKERS", [])
    monkeypatch.setattr(run, "_pid_alive", lambda pid: pid == os.getpid())
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created_times = (): True)
    monkeypatch.setattr(cache_cleanup, "cache_coordination_dir", lambda: tmp_path)
    monkeypatch.setattr(run, "_server", None)
    monkeypatch.setattr(run, "_shutdown_event", None)
    # Never let a test leave its handler, or its target, behind for the rest of the session.
    monkeypatch.setattr(run, "_WINDOWS_CONSOLE_HANDLER", None)
    monkeypatch.setattr(run, "_console_shutdown_target", None)
    yield


def _records(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("studio"))


def _publish_records():
    run.write_startup_marker()
    run._write_pid_file(8901)


def test_records_are_published_for_these_tests(tmp_path):
    _publish_records()

    assert _records(tmp_path) == sorted(
        ["studio.pid", f"studio-8901-{os.getpid()}.pid", f"studio-starting-{os.getpid()}.marker"]
    )


def test_a_close_during_the_imports_drops_the_records_without_the_teardown(tmp_path, monkeypatch):
    # No server yet: the main thread is inside `from main import app`, and _graceful_shutdown would wait on
    # its import locks past the deadline.
    run.write_startup_marker()
    teardowns = []
    monkeypatch.setattr(run, "_graceful_shutdown", lambda server = None: teardowns.append(server))

    run._console_close_shutdown()

    assert teardowns == []
    assert _records(tmp_path) == []
    assert run._OWN_STARTUP_MARKERS == []


def test_a_close_while_serving_shuts_down_then_drops_the_records(tmp_path, monkeypatch):
    _publish_records()
    server = object()
    monkeypatch.setattr(run, "_server", server)
    teardowns = []
    monkeypatch.setattr(run, "_graceful_shutdown", lambda srv = None: teardowns.append(srv))
    seen_at_set = []

    class _Event(threading.Event):
        def set(self):
            seen_at_set.append(_records(tmp_path))
            super().set()

    event = _Event()
    monkeypatch.setattr(run, "_shutdown_event", event)

    run._console_close_shutdown()

    assert teardowns == [server]
    # _graceful_shutdown leaves the marker to the server thread, which Windows may never let finish.
    assert _records(tmp_path) == []
    # Set last: it lets the main thread finalize the interpreter under this one.
    assert event.is_set() and seen_at_set == [[]]


def test_a_stalled_teardown_does_not_cost_the_records(tmp_path, monkeypatch):
    _publish_records()
    monkeypatch.setattr(run, "_server", object())
    monkeypatch.setattr(run, "_CONSOLE_TEARDOWN_BUDGET", 0.2)
    release = threading.Event()
    monkeypatch.setattr(run, "_graceful_shutdown", lambda server = None: release.wait(10))
    event = threading.Event()
    monkeypatch.setattr(run, "_shutdown_event", event)

    started = time.monotonic()
    try:
        run._console_close_shutdown()
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 3.0, elapsed
    assert _records(tmp_path) == []
    assert event.is_set()


def test_the_teardown_budget_leaves_room_inside_windows_deadline():
    assert 0 < run._CONSOLE_TEARDOWN_BUDGET < run._CONSOLE_SHUTDOWN_BUDGET < 5.0


def test_off_windows_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    assert run.install_console_close_cleanup() is False
    assert run._WINDOWS_CONSOLE_HANDLER is None


class _FakeKernel32:
    """Stands in for kernel32, so no test registers a real handler with this console."""

    def __init__(self):
        self.registered = []

        def SetConsoleCtrlHandler(callback, add):
            self.registered.append((callback, add))
            return 1

        self.SetConsoleCtrlHandler = SetConsoleCtrlHandler


@pytest.fixture
def kernel32(monkeypatch):
    import ctypes

    fake = _FakeKernel32()
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error = False: fake)
    return fake


@pytest.mark.skipif(not IS_WINDOWS, reason = "WINFUNCTYPE callbacks exist only on Windows")
def test_a_close_event_runs_the_cleanup_and_ctrl_c_is_passed_on(tmp_path, kernel32):
    _publish_records()

    assert run.install_console_close_cleanup() is True
    assert len(kernel32.registered) == 1
    callback, add = kernel32.registered[0]
    assert add is True

    # Ctrl+C is Python's to deliver as a signal; the records stay.
    assert not callback(CTRL_C_EVENT)
    assert len(_records(tmp_path)) == 3

    assert callback(CTRL_CLOSE_EVENT)
    assert _records(tmp_path) == []


@pytest.mark.skipif(not IS_WINDOWS, reason = "WINFUNCTYPE callbacks exist only on Windows")
def test_a_second_install_registers_nothing_new(kernel32):
    # Two registrations would both stay live, and the first callback's only reference would be dropped.
    assert run.install_console_close_cleanup() is True
    first = run._WINDOWS_CONSOLE_HANDLER
    ran = []

    assert run._install_windows_console_handler(lambda: ran.append("replacement")) is True

    assert len(kernel32.registered) == 1
    assert run._WINDOWS_CONSOLE_HANDLER is first
    # A later caller's shutdown is the one a close runs.
    kernel32.registered[0][0](CTRL_CLOSE_EVENT)
    assert ran == ["replacement"]


def test_python_run_py_installs_it_before_run_server():
    """`python run.py` (the desktop app's backend and the CLI's out-of-venv child) installs it before the
    imports, too, rather than after run_server() returns."""
    tree = ast.parse(Path(run.__file__).read_text(encoding = "utf-8"))
    main_blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(main_blocks) == 1

    def first_call(name):
        lines = [
            node.lineno
            for node in ast.walk(main_blocks[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        ]
        return min(lines) if lines else None

    installed = first_call("install_console_close_cleanup")
    started = first_call("run_server")
    assert installed is not None and started is not None
    assert installed < started
