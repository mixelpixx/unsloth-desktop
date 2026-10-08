# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Closing the console window `unsloth studio` runs in shuts the server down the way `python run.py` does.

Windows raises no signal for a closed window, a logoff or a system shutdown. The in-process commands
installed only a SIGTERM handler, so each of those ended the process with studio.pid, the per-port
record and the startup marker left behind. run.py owns the handler and the cleanup; these pin that both
commands install it, and early enough to cover the imports run_server() spends its first minute on.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.commands import studio as studio_mod  # noqa: E402

STUDIO_SRC = Path(studio_mod.__file__).read_text(encoding = "utf-8")


def _first_call_line(fn: ast.FunctionDef, name: str) -> "int | None":
    lines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]
    return min(lines) if lines else None


def test_every_in_process_command_installs_it_before_run_server():
    """Matched on the parsed tree, so a commented-out call fails, and a new command that calls
    run_server() has to be added here, which is the point."""
    functions = {n.name: n for n in ast.walk(ast.parse(STUDIO_SRC)) if isinstance(n, ast.FunctionDef)}
    callers = sorted(name for name, fn in functions.items() if _first_call_line(fn, "run_server"))
    assert callers == ["run", "studio_default"], callers
    for name in callers:
        installed = _first_call_line(functions[name], "_install_console_close_cleanup")
        started = _first_call_line(functions[name], "run_server")
        assert installed is not None, f"{name} never installs the console-close cleanup"
        assert installed < started, f"{name} installs it only after run_server() (line {installed} vs {started})"


def test_the_helper_asks_run_py_once():
    calls = []
    backend = types.SimpleNamespace(install_console_close_cleanup = lambda: calls.append("install"))

    studio_mod._install_console_close_cleanup(backend)

    assert calls == ["install"]


def test_an_older_run_py_without_it_is_left_alone():
    # A mixed install mid-update can load a run.py from before this existed; it must still start.
    studio_mod._install_console_close_cleanup(types.SimpleNamespace())


def test_run_installs_it_before_the_server_starts(monkeypatch, tmp_path):
    """The real command, in-venv, with run.py stubbed: the handler has to be in place while run_server()
    is still importing, not only once it returns."""
    # A real directory: the launch gate creates STUDIO_HOME and locks inside it.
    fake_venv = tmp_path / "studio" / "venv" / "unsloth_studio"
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", fake_venv.parent)

    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(_tp_mod, "resolve_tool_policy", lambda host, flag, yes, silent: False)

    calls = []

    class _Stop(Exception):
        pass

    def _run_server(**_kwargs):
        calls.append("run_server")
        raise _Stop

    backend = types.ModuleType("studio.backend.run")
    backend.run_server = _run_server
    backend.install_console_close_cleanup = lambda: calls.append("install")
    backend._resolve_external_ip = lambda: "127.0.0.1"
    monkeypatch.setitem(sys.modules, "studio.backend.run", backend)
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", backend)

    state_mod = types.ModuleType("state")
    tp_mod = types.ModuleType("state.tool_policy")
    tp_mod.set_tool_policy = lambda *a, **k: None
    tp_mod.set_tool_policy_default = lambda *a, **k: None
    state_mod.tool_policy = tp_mod
    monkeypatch.setitem(sys.modules, "state", state_mod)
    monkeypatch.setitem(sys.modules, "state.tool_policy", tp_mod)

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, ["--model", "unsloth/Qwen3-1.7B-GGUF"], catch_exceptions = True)

    assert isinstance(result.exception, _Stop), result.output
    assert calls == ["install", "run_server"], calls
