# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The orphan reaper must not take a reused parent PID for the server's owner.

Windows never reparents an orphan, so a dead Unsloth's llama-server keeps pointing at its
parent's old PID. Once anything else is given that PID, ``psutil.pid_exists(ppid)`` said
"alive" and the orphan was never reaped. ``Process.parent()`` rejects a process that
started after the child, so it cannot be mistaken for the parent.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

for _name, _attrs in (
    ("loggers", {"get_logger": lambda name: __import__("logging").getLogger(name)}),
    ("structlog", {"get_logger": lambda *a, **k: __import__("logging").getLogger("stub")}),
):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            _mod = types.ModuleType(_name)
            for _k, _v in _attrs.items():
                setattr(_mod, _k, _v)
            sys.modules[_name] = _mod

psutil = pytest.importorskip("psutil")

from core.inference.llama_cpp import LlamaCppBackend


class _Parent:
    def __init__(self, running = True):
        self._running = running

    def is_running(self):
        return self._running


def _fake_process(monkeypatch, *, ppid = 4321, parent = "alive", child_error = None):
    class _Proc:
        def __init__(self, pid):
            if child_error is not None:
                raise child_error
            self.pid = pid

        def ppid(self):
            return ppid

        def parent(self):
            if isinstance(parent, BaseException):
                raise parent
            if parent == "reused":
                # psutil: the process now at ppid started after the child -> None.
                return None
            return _Parent(running = parent == "alive")

    monkeypatch.setattr(psutil, "Process", _Proc)
    # The old check: a reused PID always "exists".
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)


def test_a_live_parent_owns_the_server(monkeypatch):
    _fake_process(monkeypatch)
    assert LlamaCppBackend._pid_parent_is_alive(100) is True


def test_a_reused_parent_pid_is_an_orphan(monkeypatch):
    _fake_process(monkeypatch, parent = "reused")
    assert LlamaCppBackend._pid_parent_is_alive(100) is False


def test_a_parent_that_has_exited_is_an_orphan(monkeypatch):
    _fake_process(monkeypatch, parent = "exited")
    assert LlamaCppBackend._pid_parent_is_alive(100) is False


@pytest.mark.parametrize("ppid", [0, 1])
def test_reparented_to_init_is_an_orphan(monkeypatch, ppid):
    # init is running, so parent() alone would call this owned.
    _fake_process(monkeypatch, ppid = ppid)
    assert LlamaCppBackend._pid_parent_is_alive(100) is False


def test_the_server_itself_gone(monkeypatch):
    _fake_process(monkeypatch, child_error = psutil.NoSuchProcess(100))
    assert LlamaCppBackend._pid_parent_is_alive(100) is False


def test_cannot_tell_is_never_a_kill(monkeypatch):
    # AccessDenied reading the parent's start time: keep the server.
    _fake_process(monkeypatch, parent = psutil.AccessDenied(4321))
    assert LlamaCppBackend._pid_parent_is_alive(100) is True


def test_this_process_really_is_its_childs_parent():
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert LlamaCppBackend._pid_parent_is_alive(child.pid) is True
    finally:
        child.kill()
        child.wait(timeout = 10)
