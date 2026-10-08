# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SIGTERM (docker stop through supervisord, `unsloth studio stop`) takes the same path as Ctrl+C.

`unsloth studio` and `unsloth start` run the server in-process and caught only
KeyboardInterrupt, so a SIGTERM died on Python's default action: no `_graceful_shutdown`,
no stop-and-save for a running training job, no child cleanup.
"""

import ast
import signal
from pathlib import Path

import pytest

from unsloth_cli.commands import studio as studio_mod

STUDIO_SRC = Path(studio_mod.__file__).read_text(encoding = "utf-8")


def _installs_before_waiting(fn: ast.FunctionDef) -> bool:
    waits = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.While)
        and any(
            isinstance(inner, ast.Attribute) and inner.attr == "_shutdown_event"
            for inner in ast.walk(node)
        )
    ]
    if not waits:
        return False
    return any(
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "_graceful_shutdown_on_sigterm"
        and stmt.lineno < min(waits)
        for stmt in fn.body
    )


_HANDLED = [signal.SIGTERM] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])


@pytest.fixture
def restore_sigterm():
    previous = {signum: signal.getsignal(signum) for signum in _HANDLED}
    yield
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def test_sigterm_becomes_a_keyboard_interrupt_once(restore_sigterm):
    studio_mod._graceful_shutdown_on_sigterm()
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler) and handler is not signal.SIG_DFL
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


@pytest.mark.skipif(not hasattr(signal, "SIGBREAK"), reason = "SIGBREAK is Windows-only")
def test_ctrl_break_takes_the_same_path(restore_sigterm):
    """Python installs no SIGBREAK handler, so Ctrl+Break ended `unsloth studio` on the spot with no
    cleanup, and run.py's console handler passes it on expecting one."""
    studio_mod._graceful_shutdown_on_sigterm()
    handler = signal.getsignal(signal.SIGBREAK)
    assert callable(handler) and handler is not signal.SIG_DFL
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGBREAK, None)
    # Only the signal that arrived goes back to the default; SIGTERM still shuts down cleanly.
    assert signal.getsignal(signal.SIGBREAK) is signal.SIG_DFL
    assert signal.getsignal(signal.SIGTERM) is handler


def test_both_server_wait_loops_install_the_handler():
    """Both in-process commands, matched on the parsed tree so a commented-out call fails."""
    functions = [n for n in ast.walk(ast.parse(STUDIO_SRC)) if isinstance(n, ast.FunctionDef)]
    installed = sorted(fn.name for fn in functions if _installs_before_waiting(fn))
    assert installed == ["run", "studio_default"], installed
