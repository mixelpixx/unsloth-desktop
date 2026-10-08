# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local-program MCP server's stderr is masked on its way to logs/mcp/*.log.

The child used to be handed the log file itself as its stderr, so whatever it printed reached the disk
untouched: a server echoing its env or its request headers left its API key in a file the log viewer
and the log export then read. It now writes into a pipe that _MaskedStderrPump drains, masking each
line with the server's configured values (mcp_secret_values) and the viewer's credential shapes. The
servers below are tiny raw JSON-RPC scripts run with this interpreter, so the real SDK spawn path is
what is exercised.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference import mcp_client
from core.inference.mcp_client import McpStdioServerError, close_mcp_sessions
from storage import mcp_servers_db

ENV_SECRET = "s3cr3t-env-value-7f2a"
BEARER_TOKEN = "tok-9c41b7e2d0aa55"
ARGV_SECRET = "argv-pass-31d9e0"
HF_SHAPED = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWx"

LEAKY_SERVER = r'''
import json, os, sys

mode = sys.argv[1]
err = sys.stderr
err.write("starting with env API_KEY=" + os.environ.get("API_KEY", "") + "\n")
err.write("sending Authorization: " + os.environ.get("AUTH_HEADER", "") + "\n")
bare = os.environ.get("AUTH_HEADER", "").split(" ", 1)[-1]
err.write("token was " + bare + " (bare)\n")
err.write("argv password " + sys.argv[3] + "\n")
# Not a configured value (a bare positional argument): only the credential-shape rules can catch it.
err.write("found a shaped token " + sys.argv[4] + "\n")
err.flush()
if mode == "crash":
    err.write("fatal: cannot reach upstream\n")
    err.flush()
    sys.exit(3)

out = sys.stdout.buffer
for raw in sys.stdin.buffer:
    msg = json.loads(raw)
    if "id" not in msg:
        continue
    if msg["method"] == "initialize":
        result = {
            "protocolVersion": msg["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "leaky", "version": "1"},
        }
    elif msg["method"] == "tools/list":
        result = {"tools": [{"name": "probe", "inputSchema": {"type": "object"}}]}
    else:
        result = {"content": []}
    out.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}).encode() + b"\n")
    out.flush()
'''

SECRETS = (ENV_SECRET, BEARER_TOKEN, ARGV_SECRET, HF_SHAPED)


@pytest.fixture
def studio(tmp_path, monkeypatch):
    home = tmp_path / "studio"
    home.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    script = tmp_path / "leaky_server.py"
    script.write_text(LEAKY_SERVER, encoding = "utf-8")
    yield SimpleNamespace(home = home, script = script, tmp = tmp_path)
    close_mcp_sessions()


def _command(studio, mode: str) -> str:
    return mcp_client.join_stdio_command(
        [sys.executable, str(studio.script), mode, "--password", ARGV_SECRET, HF_SHAPED]
    )


def _headers() -> dict:
    return {"API_KEY": ENV_SECRET, "AUTH_HEADER": f"Bearer {BEARER_TOKEN}"}


def _log_text(studio) -> str:
    folder = studio.home / "logs" / "mcp"
    logs = sorted(folder.iterdir()) if folder.is_dir() else []
    assert len(logs) == 1, logs
    return logs[0].read_text(encoding = "utf-8")


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _launch(studio, env = None):
    return mcp_client._StdioLaunch("server.exe --x", ["server.exe", "--x"], env)


# ── the configured values ──────────────────────────────────────────────────────────────────────────


def test_secret_values_cover_env_scheme_credentials_urls_and_flags():
    url = mcp_client.join_stdio_command(
        [
            "server.exe",
            "--token",
            "flag-token-value",
            "--api-key=inline-key-value",
            "GITHUB_TOKEN=assigned-value",
            "--db=postgres://admin:db-password-1@localhost/app",
            "https://ghp_userpart1234@example.com/repo",
            "https://example.com/hook?apiKey=query-key-value&mode=fast",
            "--verbose",
            "plain-argument",
        ]
    )
    values = mcp_client.mcp_secret_values(
        url, {"OPENAI_API_KEY": "env-value-123", "AUTH": "Bearer header-cred-456", "LEVEL": "dev"}
    )
    for expected in (
        "env-value-123",
        "Bearer header-cred-456",
        "header-cred-456",
        "flag-token-value",
        "inline-key-value",
        "assigned-value",
        "db-password-1",
        "ghp_userpart1234",
        "query-key-value",
    ):
        assert expected in values
    # Too short to mask safely, and not secret-named.
    for kept in ("dev", "fast", "plain-argument", "--verbose", "server.exe"):
        assert kept not in values
    # Longest first, so a value containing another is masked whole.
    assert values == sorted(values, key = len, reverse = True)


def test_http_url_credentials_are_secret_values():
    values = mcp_client.mcp_secret_values(
        "https://user:hunter2-pass@mcp.example.com/sse?token=query-token-1&page=2",
        {"Authorization": "Bearer http-bearer-1"},
    )
    assert {"hunter2-pass", "query-token-1", "http-bearer-1"} <= set(values)
    assert "2" not in values


# ── on disk ─────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.timeout(60)
def test_a_crash_leaves_only_masks_on_disk_and_a_useful_tail(studio):
    with pytest.raises(McpStdioServerError) as exc:
        asyncio.run(mcp_client.list_tools_async(_command(studio, "crash"), _headers(), timeout = 30))
    message = str(exc.value)
    text = _log_text(studio)
    for secret in SECRETS:
        assert secret not in text, secret
        assert secret not in message, secret
    assert "starting with env API_KEY=***" in text
    assert "sending Authorization: ***" in text
    assert "token was *** (bare)" in text
    assert "argv password ***" in text
    assert "found a shaped token hf_<redacted>" in text
    # The tail still says what went wrong.
    assert message.startswith("The server exited during startup.\nLast output on stderr:\n")
    assert "fatal: cannot reach upstream" in message
    assert "starting with env API_KEY=***" in message


@pytest.mark.timeout(60)
def test_a_running_server_writes_masked_lines(studio):
    tools = asyncio.run(
        mcp_client.list_tools_async(_command(studio, "serve"), _headers(), timeout = 30)
    )
    assert [tool["name"] for tool in tools] == ["probe"]
    _wait_for(lambda: "found a shaped token" in _log_text(studio))
    text = _log_text(studio)
    for secret in SECRETS:
        assert secret not in text, secret
    assert "starting with env API_KEY=***" in text
    assert "found a shaped token hf_<redacted>" in text


def test_a_secret_split_across_writes_is_still_masked(studio):
    launch = _launch(studio, {"KEY": ENV_SECRET})
    writer = launch.open_log()
    writer.write("before " + ENV_SECRET[:6])
    writer.flush()
    time.sleep(0.2)  # let the pump read the first half on its own
    writer.write(ENV_SECRET[6:] + " after\n")
    writer.flush()
    launch.close_log()
    launch._pump.join(5)
    text = launch.log_path.read_text(encoding = "utf-8")
    assert ENV_SECRET not in text
    assert "before *** after" in text


def test_an_unterminated_prompt_is_quoted_masked_but_not_yet_written(studio):
    launch = _launch(studio, {"KEY": ENV_SECRET})
    writer = launch.open_log()
    writer.write(f"license {ENV_SECRET} rejected, enter another: ")
    writer.flush()
    _wait_for(lambda: launch._pump._pending)
    tail = launch.stderr_tail()
    assert tail == "license *** rejected, enter another:"
    assert "enter another" not in launch.log_path.read_text(encoding = "utf-8")
    launch.close_log()
    launch._pump.join(5)
    # EOF writes the held line, masked.
    assert "license *** rejected, enter another: \n" in launch.log_path.read_text(encoding = "utf-8")


def test_utf16_output_is_masked_too(studio):
    launch = _launch(studio, {"KEY": ENV_SECRET})
    writer = launch.open_log()
    os.write(writer.fileno(), f"key is {ENV_SECRET}\n".encode("utf-16-le"))
    launch.close_log()
    launch._pump.join(5)
    text = launch.log_path.read_text(encoding = "utf-8")
    assert ENV_SECRET not in text.replace("\x00", "")
    assert "key is ***" in text


def test_the_pump_ends_and_closes_the_file_when_the_program_exits(studio):
    launch = _launch(studio, {"KEY": ENV_SECRET})
    writer = launch.open_log()
    child = subprocess.Popen(
        [sys.executable, "-c", f"import sys; sys.stderr.write('k={ENV_SECRET}\\n'); sys.exit(1)"],
        stderr = writer,
    )
    launch.close_log()
    assert child.wait(30) == 1
    launch._pump.join(10)
    assert not launch._pump._thread.is_alive()
    assert launch._pump._sink.closed
    assert "k=***" in launch.log_path.read_text(encoding = "utf-8")


def test_a_failing_disk_never_blocks_the_program(studio):
    """The child blocks once its stderr pipe is full, and an MCP server blocked on stderr hangs every
    call, so the pump keeps reading even when it cannot write."""

    class BrokenSink:
        closed = False

        def write(self, _text):
            raise OSError(28, "No space left on device")

        def flush(self):
            raise OSError(28, "No space left on device")

        def close(self):
            self.closed = True

    read_fd, write_fd = os.pipe()
    sink = BrokenSink()
    pump = mcp_client._MaskedStderrPump(read_fd, sink, lambda text: text)
    done = threading.Event()

    def flood():
        with open(write_fd, "wb") as writer:
            for _ in range(64):
                writer.write(b"x" * 8191 + b"\n")
        done.set()

    threading.Thread(target = flood, daemon = True).start()
    assert done.wait(20), "the writer blocked: the pump stopped draining"
    pump.join(10)
    assert not pump._thread.is_alive()
    assert sink.closed


def test_a_line_with_no_newline_is_written_in_bounded_pieces(studio, monkeypatch):
    monkeypatch.setattr(mcp_client, "_PUMP_MAX_LINE_CHARS", 100)
    launch = _launch(studio, None)
    writer = launch.open_log()
    writer.write(("progress 10%\r" * 40))
    writer.flush()
    _wait_for(lambda: "progress" in launch.log_path.read_text(encoding = "utf-8"))
    launch.close_log()
    launch._pump.join(5)
    # Bytes: read_text would turn every \r into a line break and prove nothing.
    body = launch.log_path.read_bytes().decode("utf-8").split("=====", 2)[2]
    pieces = [line.rstrip("\r") for line in body.split("\n") if line.strip("\r")]
    assert len(pieces) > 4
    assert all(len(piece) <= 100 for piece in pieces)


def test_logs_of_removed_servers_are_pruned_and_the_live_one_kept(studio, monkeypatch):
    """One file per command, so without pruning every server ever removed leaves its log forever."""
    monkeypatch.setattr(mcp_client, "_STDIO_LOG_KEEP", 3)
    folder = studio.home / "logs" / "mcp"
    folder.mkdir(parents = True)
    for i in range(5):
        stale = folder / f"old-{i}.log"
        stale.write_text("x", encoding = "utf-8")
        os.utime(stale, (1_000_000 + i, 1_000_000 + i))
    launch = _launch(studio)
    writer = launch.open_log()
    try:
        names = sorted(path.name for path in folder.glob("*.log"))
        # The two newest stale files plus this spawn's own, which is protected.
        assert names == sorted(["old-3.log", "old-4.log", launch.log_path.name])
    finally:
        writer.close()
        launch.close_log()
