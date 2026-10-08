# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On Windows the stop hint names every way to stop Unsloth, not only the console's Ctrl+C.

A Windows user who started Unsloth from a shortcut or a terminal was told to press Ctrl+C and to
mind the difference between Control and Command on macOS, and never that the web UI has a
Shutdown item or that `unsloth studio stop` works from any terminal. macOS and Linux keep the
wording they had. The banner block itself is pinned byte for byte, on both platforms, in
test_account_desktop_network.py.
"""

import logging

import pytest

import run


def _rewritten(monkeypatch, platform: str) -> str:
    monkeypatch.setattr(run.sys, "platform", platform)
    loggers = [logging.getLogger(name) for name in ("uvicorn", "uvicorn.error")]
    before = [list(logger.filters) for logger in loggers]
    try:
        run._install_uvicorn_startup_log_rewrite("127.0.0.1")
        record = logging.LogRecord(
            "uvicorn.error",
            logging.INFO,
            __file__,
            0,
            "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)",
            ("http", "127.0.0.1", 8888),
            None,
        )
        for f in loggers[0].filters[len(before[0]) :]:
            f.filter(record)
        return record.getMessage()
    finally:
        for logger, filters in zip(loggers, before):
            logger.filters[:] = filters


def test_windows_startup_line_names_every_way_to_stop(monkeypatch):
    line = _rewritten(monkeypatch, "win32")
    assert line.startswith("Unsloth Studio running on http://127.0.0.1:8888 ")
    assert "Shutdown" in line and "unsloth studio stop" in line and "Ctrl+C" in line
    assert "Command+C" not in line


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_other_platforms_keep_the_control_not_command_hint(monkeypatch, platform):
    line = _rewritten(monkeypatch, platform)
    assert line.endswith("(To stop: press Ctrl+C -- on macOS, Control+C not Command+C)")


def test_windows_stop_hint_block_keeps_its_shape(monkeypatch, capsys):
    import startup_banner

    monkeypatch.setattr(startup_banner, "stdout_supports_color", lambda: False)
    monkeypatch.setattr(run.sys, "platform", "win32")
    run.print_studio_stop_hint()
    windows = capsys.readouterr().out

    monkeypatch.setattr(run.sys, "platform", "linux")
    run.print_studio_stop_hint()
    other = capsys.readouterr().out

    # Same blank line, one hint line, divider and trailing blank line; only the hint differs.
    assert windows.splitlines()[0] == other.splitlines()[0] == ""
    assert windows.splitlines()[2:] == other.splitlines()[2:]
    assert windows.splitlines()[1] == run._WINDOWS_STOP_HINT
    assert "Command+C" not in windows
