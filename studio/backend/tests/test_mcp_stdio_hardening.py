# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local-program (stdio) MCP servers as real-world Windows programs behave.

Each failure here used to reach the user as an opaque string or a silent hang: a missing program as
"[WinError 2] The system cannot find the file specified", a crash as "Connection closed", a hang as
"An internal error occurred"; a stray non-UTF-8 byte or a BOM on stdout hung the call until its
timeout; the child saw only 12 environment variables (no ProgramFiles, no proxy); there was no way to
set a working directory; every non-JSON stdout line logged a full traceback; and an unquoted spaced
path silently ran a different program. The servers below are tiny raw JSON-RPC scripts run with this
interpreter, so the tests exercise the real SDK reader and process plumbing without the scratch
fixtures a manual repro used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.inference import mcp_client
from core.inference.mcp_client import McpStdioServerError, call_tool_sync, close_mcp_sessions
from storage import mcp_servers_db

RAW_SERVER = r'''
import json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
out = sys.stdout.buffer

if mode == "crash":
    sys.stderr.write(
        "\x1b[31mfatal:\x1b[0m token=" + os.environ.get("SECRET_TOKEN", "") + " missing config\n"
    )
    sys.stderr.flush()
    sys.exit(3)
if mode == "hang":
    sys.stderr.write("waiting for a license key on stdin\n")
    sys.stderr.flush()
    for _ in sys.stdin.buffer:
        pass
    sys.exit(0)
if mode == "usage":
    out.write(b"usage: server --config FILE\n")
    out.flush()
    sys.exit(2)
if mode == "banner":
    for i in range(10):
        out.write(b"starting up %d\n" % i)
    out.flush()

first = True
for raw in sys.stdin.buffer:
    msg = json.loads(raw)
    if "id" not in msg:
        continue
    method = msg["method"]
    if method == "initialize":
        result = {
            "protocolVersion": msg["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "raw", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{"name": "probe", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        if mode == "badbyte":
            body = json.dumps(
                {"jsonrpc": "2.0", "id": msg["id"],
                 "result": {"content": [{"type": "text", "text": "caf@ ok"}]}}
            ).encode().replace(b"@", b"\x82")
            out.write(body + b"\n")
            out.flush()
            continue
        state = {"cwd": os.getcwd(), "proxy": os.environ.get("HTTPS_PROXY"), "keys": sorted(os.environ)}
        result = {"content": [{"type": "text", "text": json.dumps(state)}]}
    else:
        out.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                              "error": {"code": -32601, "message": "no"}}).encode() + b"\n")
        out.flush()
        continue
    prefix = b"\xef\xbb\xbf" if mode == "bom" and first else b""
    first = False
    out.write(prefix + json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}).encode() + b"\n")
    out.flush()
'''


@pytest.fixture
def studio(tmp_path, monkeypatch):
    home = tmp_path / "studio"
    home.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    script = tmp_path / "raw_server.py"
    script.write_text(RAW_SERVER, encoding = "utf-8")
    yield SimpleNamespace(home = home, script = script, tmp = tmp_path)
    close_mcp_sessions()


def _command(studio, mode: str = "normal") -> str:
    return mcp_client.join_stdio_command([sys.executable, str(studio.script), mode])


def _list_tools(url, headers = None, timeout = 30.0, cwd = None):
    return asyncio.run(mcp_client.list_tools_async(url, headers, timeout = timeout, cwd = cwd))


def _logs(studio) -> list:
    folder = studio.home / "logs" / "mcp"
    return sorted(folder.iterdir()) if folder.is_dir() else []


# ── 1. a failed start says why ──────────────────────────────────────────────────────────────────


@pytest.mark.timeout(60)
def test_missing_program_is_named_and_leaves_no_log(studio):
    missing = studio.tmp / "nope" / "missing-server.exe"
    url = mcp_client.join_stdio_command([str(missing), "--stdio"])
    with pytest.raises(McpStdioServerError) as exc:
        _list_tools(url)
    message = str(exc.value)
    assert message.startswith("Program not found: missing-server.exe. Check the program path.")
    # The file name only: the folder (and the user name in it) stays out of a message that can reach a model.
    assert str(missing.parent) not in message
    assert "WinError" not in message
    assert _logs(studio) == []


def test_bare_program_not_found_points_at_path():
    launch = mcp_client._StdioLaunch("no-such-mcp-server --x", ["no-such-mcp-server", "--x"], None)
    assert launch._not_found_message() == (
        "Program not found: no-such-mcp-server. Check that it is installed and on PATH, "
        "or enter its full path."
    )


def test_unquoted_spaced_path_gets_the_quoted_form(tmp_path):
    folder = tmp_path / "My Tools"
    folder.mkdir()
    program = folder / "server.exe"
    program.write_bytes(b"")
    url = f"{program} --stdio"  # unquoted, so it splits at the space
    launch = mcp_client._StdioLaunch(url, mcp_client.parse_stdio_command(url), None)
    assert launch._not_found_message() == (
        "Program not found: My. Check the program path. The path contains spaces — wrap it in "
        f'double quotes: "{program}"'
    )


@pytest.mark.timeout(60)
def test_unquoted_spaced_path_that_names_nothing_still_hints_at_quotes(studio):
    url = f"{studio.tmp / 'My Tools' / 'server.exe'} --stdio"
    with pytest.raises(McpStdioServerError) as exc:
        _list_tools(url)
    assert str(exc.value) == (
        "Program not found: My. Check the program path. If the program's path contains spaces, "
        "wrap it in double quotes."
    )


@pytest.mark.timeout(60)
def test_crash_quotes_stderr_tail_with_env_values_masked(studio):
    headers = {"SECRET_TOKEN": "sk-test-0123456789"}
    with pytest.raises(McpStdioServerError) as exc:
        _list_tools(_command(studio, "crash"), headers)
    message = str(exc.value)
    assert message.startswith("The server exited during startup.\nLast output on stderr:\n")
    assert "fatal: token=*** missing config" in message
    assert "sk-test-0123456789" not in message
    assert "\x1b" not in message
    # The full output is in this command's own log, named after the program and the address digest.
    logs = _logs(studio)
    assert [log.name for log in logs] == [mcp_client._stdio_log_name(_command(studio, "crash"), sys.executable)]
    assert f"logs/mcp/{logs[0].name}" in message
    assert "missing config" in logs[0].read_text(encoding = "utf-8")


@pytest.mark.timeout(60)
def test_exit_with_usage_on_stdout_quotes_it(studio):
    with pytest.raises(McpStdioServerError) as exc:
        _list_tools(_command(studio, "usage"))
    message = str(exc.value)
    assert message.startswith("The server exited during startup.")
    assert "usage: server --config FILE" in message


@pytest.mark.timeout(60)
def test_hang_reports_no_handshake_with_stderr(studio):
    with pytest.raises(McpStdioServerError) as exc:
        _list_tools(_command(studio, "hang"), timeout = 3)
    message = str(exc.value)
    assert message.startswith(
        "No MCP handshake within 3s — the program may be waiting for input, downloading on first "
        "run, or not an MCP stdio server."
    )
    assert "waiting for a license key on stdin" in message


@pytest.mark.timeout(60)
def test_test_route_returns_the_explanation_not_a_fallback(studio, monkeypatch):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerTestRequest

    monkeypatch.setattr(routes_mcp, "stdio_mcp_enabled", lambda: True)
    result = asyncio.run(
        routes_mcp.test_mcp_server(
            McpServerTestRequest(url = _command(studio, "crash")), current_subject = "u"
        )
    )
    assert result.ok is False
    assert result.error.startswith("The server exited during startup.")
    assert "missing config" in result.error


@pytest.mark.timeout(60)
def test_tool_call_reports_a_failed_start(studio):
    out = call_tool_sync(_command(studio, "crash"), None, "probe", {}, timeout = 30)
    assert out.startswith("Error: MCP tool 'probe' could not start its server: The server exited during startup.")
    assert "missing config" in out


@pytest.mark.timeout(60)
def test_tool_call_connect_timeout_is_explained(studio, monkeypatch):
    monkeypatch.setattr(mcp_client, "_STDIO_CONNECT_TIMEOUT", 2.0)
    out = call_tool_sync(_command(studio, "hang"), None, "probe", {}, timeout = 30, scope = "chat")
    assert out.startswith("Error: MCP tool 'probe' could not start its server: No MCP handshake within 2s")
    assert "waiting for a license key on stdin" in out


def test_log_is_truncated_at_spawn_once_oversized(studio, monkeypatch):
    monkeypatch.setattr(mcp_client, "_STDIO_LOG_MAX_BYTES", 1024)
    launch = mcp_client._StdioLaunch("server.exe --x", ["server.exe", "--x"], None)
    folder = studio.home / "logs" / "mcp"
    folder.mkdir(parents = True)
    (folder / launch.log_name).write_bytes(b"x" * 4096)
    assert launch.open_log() is not None
    launch.close_log()
    assert (folder / launch.log_name).stat().st_size < 200

    # Under the cap it is appended to, keeping earlier starts.
    second = mcp_client._StdioLaunch("server.exe --x", ["server.exe", "--x"], None)
    before = (folder / launch.log_name).stat().st_size
    second.open_log()
    second.close_log()
    assert (folder / launch.log_name).stat().st_size > before


def test_stderr_tail_is_bounded_cleaned_and_only_this_spawn(studio):
    launch = mcp_client._StdioLaunch("server.exe", ["server.exe"], {"KEY": "abcd", "SHORT": "ab"})
    handle = launch.open_log()
    handle.write("x" * 3000 + "\r\n\x1b[1mbold\x1b[0m key=abcd short=ab\r\n")
    handle.flush()
    launch.close_log()
    tail = launch.stderr_tail()
    assert tail.startswith("…")
    assert len(tail) <= mcp_client._STDIO_TAIL_CHARS + 1
    # Values of four or more characters are masked; shorter ones are too common to be worth it.
    assert tail.endswith("bold key=*** short=ab")

    # A later spawn of the same command reads only its own output.
    again = mcp_client._StdioLaunch("server.exe", ["server.exe"], None)
    again.open_log()
    again.close_log()
    assert again.stderr_tail() == ""


def test_child_processes_sharing_a_log_do_not_overwrite_each_other(studio):
    """Two chats on one server run two copies of the command into one log; on Windows a plain
    append-mode handle let the second overwrite the first."""
    import subprocess

    writer = studio.tmp / "writer.py"
    writer.write_text(
        "import sys, time\n"
        "for i in range(5):\n"
        "    sys.stderr.write(f'{sys.argv[1]}{i}\\n'); sys.stderr.flush(); time.sleep(0.02)\n",
        encoding = "utf-8",
    )
    path = studio.tmp / "shared.log"
    first = mcp_client._open_append_only(path)
    second = mcp_client._open_append_only(path)
    procs = [
        subprocess.Popen([sys.executable, str(writer), "A"], stderr = first),
        subprocess.Popen([sys.executable, str(writer), "B"], stderr = second),
    ]
    first.close()
    second.close()
    for proc in procs:
        assert proc.wait(30) == 0
    lines = path.read_text(encoding = "utf-8").split()
    assert sorted(lines) == [f"{who}{i}" for who in "AB" for i in range(5)]


def test_missing_working_directory_is_named_before_spawn(studio):
    gone = studio.tmp / "deleted-project"
    with pytest.raises(McpStdioServerError, match = "Working directory not found: deleted-project"):
        _list_tools(_command(studio), cwd = str(gone))


# ── 2. non-UTF-8 output and a BOM no longer hang ───────────────────────────────────────────────


def test_lenient_decoding_is_installed():
    import mcp.client.stdio as sdk_stdio

    assert getattr(sdk_stdio.TextReceiveStream, "unsloth_lenient", False) is True


@pytest.mark.timeout(60)
def test_bom_before_the_first_message_is_dropped(studio):
    tools = _list_tools(_command(studio, "bom"), timeout = 20)
    assert [tool["name"] for tool in tools] == ["probe"]


@pytest.mark.timeout(60)
def test_invalid_utf8_in_a_result_is_replaced_not_fatal(studio):
    out = call_tool_sync(_command(studio, "badbyte"), None, "probe", {}, timeout = 20)
    assert out == "caf\ufffd ok"


# ── 3. the child inherits what real programs need ──────────────────────────────────────────────


def _clear_inheritable(monkeypatch):
    for name in (
        mcp_client._INHERITED_WINDOWS_ENV
        + mcp_client._INHERITED_NETWORK_ENV
        + mcp_client._INHERITED_POSIX_PROXY_ENV
    ):
        monkeypatch.delenv(name, raising = False)


def test_inherited_env_is_an_allowlist_without_path_or_secrets(monkeypatch):
    _clear_inheritable(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/etc/corp-ca.pem")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("UNSLOTH_STUDIO_SECRET", "s")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    if mcp_client._IS_WINDOWS:
        monkeypatch.setenv("ProgramData", "C:\\ProgramData")
    env = mcp_client._inherited_stdio_env({"API_KEY": "mine"})
    assert env["API_KEY"] == "mine"
    assert env["HTTPS_PROXY"] == "http://proxy.corp:3128"
    assert env["NODE_EXTRA_CA_CERTS"] == "/etc/corp-ca.pem"
    if mcp_client._IS_WINDOWS:
        assert env["ProgramData"] == "C:\\ProgramData"
    upper = {key.upper() for key in env}
    assert "PATH" not in upper
    assert not upper & {"HF_TOKEN", "UNSLOTH_STUDIO_SECRET", "OPENAI_API_KEY"}


def test_configured_names_win_without_case(monkeypatch):
    _clear_inheritable(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://host-proxy:1")
    env = mcp_client._inherited_stdio_env({"https_proxy": ""})
    # The user's (empty, i.e. "no proxy") choice is the only spelling the child sees.
    assert env == {"https_proxy": ""}


def test_nothing_to_inherit_leaves_env_untouched(monkeypatch):
    _clear_inheritable(monkeypatch)
    assert mcp_client._inherited_stdio_env(None) is None
    configured = {"K": "v"}
    assert mcp_client._inherited_stdio_env(configured) is configured


def test_explicit_empty_path_sandbox_survives(studio, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    command = mcp_client.join_stdio_command([sys.executable, "-m", "some_server"])
    client = mcp_client._client(command, {"PATH": "", "K": "v"})
    env = client.transport.env
    assert env["PATH"] == ""
    assert env["K"] == "v"
    assert env["HTTPS_PROXY"] == "http://proxy.corp:3128"


@pytest.mark.timeout(60)
def test_spawned_server_sees_proxy_and_system_vars(studio, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    out = call_tool_sync(_command(studio), None, "probe", {}, timeout = 30)
    state = json.loads(out)
    assert state["proxy"] == "http://proxy.corp:3128"
    if mcp_client._IS_WINDOWS and os.environ.get("ProgramFiles"):
        assert "PROGRAMFILES" in {key.upper() for key in state["keys"]}


def test_sessions_stay_keyed_on_the_configured_env(monkeypatch):
    before = mcp_client._session_key("npx x", {"A": "1"}, "chat")
    monkeypatch.setenv("HTTPS_PROXY", "http://changed:1")
    assert mcp_client._session_key("npx x", {"A": "1"}, "chat") == before


# ── 4. a working directory, end to end ──────────────────────────────────────────────────────────


@pytest.mark.timeout(60)
def test_server_starts_in_its_working_directory(studio):
    project = studio.tmp / "project dir"
    project.mkdir()
    out = call_tool_sync(_command(studio), None, "probe", {}, timeout = 30, cwd = str(project))
    assert os.path.samefile(json.loads(out)["cwd"], project)
    # No cwd keeps the backend's own, as before.
    out = call_tool_sync(_command(studio), None, "probe", {}, timeout = 30)
    assert os.path.samefile(json.loads(out)["cwd"], os.getcwd())


def test_transport_receives_cwd(studio, monkeypatch):
    import fastmcp
    from fastmcp.client import transports

    captured = {}

    class CapturingStdioTransport:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(fastmcp, "Client", lambda transport: transport)
    monkeypatch.setattr(transports, "StdioTransport", CapturingStdioTransport)
    monkeypatch.setattr(mcp_client, "_stdio_argv", lambda parts, env: parts)
    mcp_client._client("python -m srv", None, cwd = str(studio.tmp))
    assert captured["cwd"] == str(studio.tmp)


class _CwdClient:
    instances: list["_CwdClient"] = []

    def __init__(self, url, cwd):
        self.url = url
        self.cwd = cwd
        self.connected = False
        self.exited = 0
        self.transport = SimpleNamespace(_is_session_dead = lambda: False)
        _CwdClient.instances.append(self)

    async def __aenter__(self):
        self.connected = True
        return self

    async def __aexit__(self, *exc):
        self.exited += 1
        self.connected = False

    def is_connected(self):
        return self.connected

    async def call_tool(self, name, args, raise_on_error = True):
        return SimpleNamespace(
            content = [SimpleNamespace(type = "text", text = self.cwd or "-")],
            is_error = False,
            structured_content = None,
        )


def test_sessions_are_keyed_and_closed_by_cwd(monkeypatch):
    _CwdClient.instances = []
    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False, cwd = None: _CwdClient(url, cwd),
    )
    try:
        assert call_tool_sync("npx srv", None, "t", {}, scope = "chat", cwd = "/a") == "/a"
        assert call_tool_sync("npx srv", None, "t", {}, scope = "chat", cwd = "/b") == "/b"
        assert call_tool_sync("npx srv", None, "t", {}, scope = "chat") == "-"
        assert call_tool_sync("npx srv", None, "t", {}, scope = "chat", cwd = "/a") == "/a"
        assert len(_CwdClient.instances) == 3

        # Closing one row's configuration leaves the same command run from another folder alone.
        close_mcp_sessions("npx srv", None, cwd = "/a")
        cwds = sorted(mcp_client._session_cwd(key) for key in mcp_client._mcp_sessions)
        assert cwds == ["", "/b"]
        # Without cwd, a row's env still closes every folder's process.
        close_mcp_sessions("npx srv", None)
        assert mcp_client._mcp_sessions == {}
    finally:
        close_mcp_sessions()


def _routes(monkeypatch, tmp_path):
    import routes.mcp_servers as routes_mcp

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    monkeypatch.setattr(routes_mcp, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: True)
    return routes_mcp


def test_schema_migration_adds_cwd_to_an_existing_table(tmp_path, monkeypatch):
    from utils.paths import studio_db_path

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    path = studio_db_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE mcp_servers (id TEXT NOT NULL PRIMARY KEY, display_name TEXT NOT NULL, "
        "url TEXT NOT NULL, headers_json TEXT, is_enabled INTEGER NOT NULL DEFAULT 1, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO mcp_servers VALUES ('old', 'Old', 'npx srv', NULL, 1, 'then', 'then')"
    )
    conn.commit()
    conn.close()
    assert mcp_servers_db.get_server("old")["cwd"] is None


@pytest.mark.parametrize(
    "bad,needle",
    [
        ("relative/dir", "absolute"),
        ("MISSING", "not found"),
        ("FILE", "not a folder"),
    ],
)
def test_create_validates_the_working_directory(tmp_path, monkeypatch, bad, needle):
    from models.mcp_servers import McpServerCreate

    routes_mcp = _routes(monkeypatch, tmp_path)
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    value = {"MISSING": str(tmp_path / "missing"), "FILE": str(a_file)}.get(bad, bad)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.create_mcp_server(
                McpServerCreate(display_name = "s", url = "npx srv", cwd = value),
                current_subject = "u",
            )
        )
    assert exc.value.status_code == 400
    assert needle in exc.value.detail
    assert mcp_servers_db.list_servers() == []


def test_cwd_is_refused_for_http_and_blank_means_none(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerCreate

    routes_mcp = _routes(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.create_mcp_server(
                McpServerCreate(display_name = "h", url = "https://example.com/mcp", cwd = str(tmp_path)),
                current_subject = "u",
            )
        )
    assert exc.value.status_code == 400
    assert "only applies to local programs" in exc.value.detail
    created = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "s", url = "npx srv", cwd = "   "), current_subject = "u"
        )
    )
    assert created.cwd is None


def test_api_keys_cannot_set_a_working_directory(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerCreate

    routes_mcp = _routes(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.create_mcp_server(
                McpServerCreate(display_name = "s", url = "npx srv", cwd = str(tmp_path / "missing")),
                current_subject = "u",
                via_api_key = True,
            )
        )
    # The gate answers before the filesystem is consulted, so a key cannot probe which folders exist.
    assert exc.value.status_code == 403


def test_create_update_clear_and_switch_to_http(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerCreate, McpServerUpdate

    routes_mcp = _routes(monkeypatch, tmp_path)
    closed = []
    monkeypatch.setattr(
        routes_mcp, "close_mcp_sessions", lambda *args, **kwargs: closed.append((args, kwargs))
    )
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()

    created = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "s", url = "npx srv", cwd = f'"{first}"'),
            current_subject = "u",
        )
    )
    # Explorer's "Copy as path" quotes are dropped, as for the program.
    assert created.cwd == str(first)
    assert routes_mcp.list_mcp_servers(current_subject = "u")[0].cwd == str(first)

    mcp_client.cache_tools(created.id, [{"name": "t"}])
    updated = asyncio.run(
        routes_mcp.update_mcp_server(created.id, McpServerUpdate(cwd = str(second)), current_subject = "u")
    )
    assert updated.cwd == str(second)
    # A new folder is a new process: cached tools go, and the old folder's sessions close, narrowed to it.
    assert mcp_client.get_cached_tools(created.id) is None
    assert closed == [(("npx srv", None), {"cwd": str(first)})]

    unchanged = asyncio.run(
        routes_mcp.update_mcp_server(created.id, McpServerUpdate(display_name = "renamed"), current_subject = "u")
    )
    assert unchanged.cwd == str(second)

    cleared = asyncio.run(
        routes_mcp.update_mcp_server(created.id, McpServerUpdate(cwd = None), current_subject = "u")
    )
    assert cleared.cwd is None

    asyncio.run(
        routes_mcp.update_mcp_server(created.id, McpServerUpdate(cwd = str(first)), current_subject = "u")
    )
    switched = asyncio.run(
        routes_mcp.update_mcp_server(
            created.id, McpServerUpdate(url = "https://example.com/mcp"), current_subject = "u"
        )
    )
    assert switched.cwd is None

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(created.id, McpServerUpdate(cwd = str(first)), current_subject = "u")
        )
    assert exc.value.status_code == 400


def test_test_and_refresh_probe_with_the_working_directory(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerCreate, McpServerTestRequest

    routes_mcp = _routes(monkeypatch, tmp_path)
    probes = []

    async def probe(**kwargs):
        probes.append(kwargs.get("cwd"))
        return []

    monkeypatch.setattr(routes_mcp, "list_tools_async", probe)
    asyncio.run(
        routes_mcp.test_mcp_server(
            McpServerTestRequest(url = "npx srv", cwd = str(tmp_path)), current_subject = "u"
        )
    )
    # Per chat: refresh probes a one-shot copy, in the row's folder.
    created = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(
                display_name = "s", url = "npx srv", cwd = str(tmp_path), process_mode = "per_chat"
            ),
            current_subject = "u",
        )
    )
    asyncio.run(routes_mcp.refresh_mcp_server_tools(created.id, current_subject = "u"))
    assert probes == [str(tmp_path), str(tmp_path)]

    # Shared (the default for a new local program): refresh asks the server's own process, started in that folder.
    session_probes = []
    monkeypatch.setattr(
        routes_mcp,
        "list_session_tools_sync",
        lambda url, headers, **kwargs: session_probes.append(kwargs["cwd"]) or [],
    )
    shared = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "t", url = "npx srv2", cwd = str(tmp_path)),
            current_subject = "u",
        )
    )
    asyncio.run(routes_mcp.refresh_mcp_server_tools(shared.id, current_subject = "u"))
    assert session_probes == [str(tmp_path)]
    assert probes == [str(tmp_path), str(tmp_path)]


def test_chat_discovery_and_calls_use_the_row_cwd(tmp_path, monkeypatch):
    from core.inference import tools as tools_mod

    _routes(monkeypatch, tmp_path)
    monkeypatch.setattr(tools_mod, "stdio_mcp_enabled", lambda: True)
    mcp_servers_db.create_server(
        id = "s1", display_name = "S", url = "npx srv", is_enabled = True, cwd = str(tmp_path)
    )
    mcp_client.invalidate_tool_cache()
    probed = []

    async def probe(**kwargs):
        probed.append(kwargs["cwd"])
        return [{"name": "t", "inputSchema": {"type": "object"}}]

    calls = []
    monkeypatch.setattr(tools_mod, "list_tools_async", probe)
    monkeypatch.setattr(tools_mod, "call_tool_sync", lambda **kwargs: calls.append(kwargs) or "ok")
    try:
        asyncio.run(tools_mod.get_enabled_mcp_tools())
        assert probed == [str(tmp_path)]
        assert tools_mod.execute_tool("mcp__s1__t", {}) == "ok"
        assert calls[0]["cwd"] == str(tmp_path)
        assert calls[0]["config_check"]() is True
        mcp_servers_db.update_server("s1", {"cwd": None})
        # A process started in the old folder must not be cached for the edited row.
        assert calls[0]["config_check"]() is False
    finally:
        mcp_client.invalidate_tool_cache()


def test_cwd_is_a_tool_cache_invalidating_field():
    assert "cwd" in mcp_client.TOOL_CACHE_INVALIDATING_FIELDS


# ── 5. one concise warning per process for non-JSON stdout ─────────────────────────────────────


def _parse_failure_record(line: str) -> logging.LogRecord:
    from pydantic import TypeAdapter, ValidationError

    try:
        TypeAdapter(dict).validate_json(line)
    except ValidationError:
        exc_info = sys.exc_info()
    return logging.LogRecord(
        "mcp.client.stdio", logging.ERROR, __file__, 1,
        "Failed to parse JSONRPC message from server", (), exc_info,
    )


def test_filter_keeps_one_warning_per_process_and_other_errors():
    sdk_filter = mcp_client._CollapseNonJsonStdout()
    state = mcp_client._StdioProcessLog("server.exe#abc", lambda text: text.replace("hunter22", "***"))
    token = mcp_client._stdio_process_log.set(state)
    try:
        first = _parse_failure_record("banner with hunter22")
        assert sdk_filter.filter(first) is True
        assert first.levelno == logging.WARNING
        assert first.exc_info is None
        message = first.getMessage()
        assert "server.exe#abc" in message and "banner with ***" in message
        assert "hunter22" not in message
        for _ in range(5):
            assert sdk_filter.filter(_parse_failure_record("more noise")) is False
        assert state.non_json_lines == 6
        other = logging.LogRecord("mcp.client.stdio", logging.ERROR, __file__, 1, "Something else", (), None)
        assert sdk_filter.filter(other) is True
        assert other.levelno == logging.ERROR
    finally:
        mcp_client._stdio_process_log.reset(token)

    # Outside a Studio spawn there is no process to attribute it to: at most one a minute.
    unscoped = mcp_client._CollapseNonJsonStdout()
    assert unscoped.filter(_parse_failure_record("x")) is True
    assert unscoped.filter(_parse_failure_record("y")) is False


@pytest.mark.timeout(60)
def test_a_chatty_server_logs_one_line_not_a_traceback_per_line(studio, caplog):
    with caplog.at_level(logging.WARNING, logger = "mcp.client.stdio"):
        tools = _list_tools(_command(studio, "banner"))
    assert [tool["name"] for tool in tools] == ["probe"]
    records = [r for r in caplog.records if r.name == "mcp.client.stdio"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "starting up 0" in records[0].getMessage()
    assert records[0].exc_info is None


# ── 6. an unquoted spaced program path is refused, not guessed ─────────────────────────────────


def test_unquoted_spaced_program_detection(tmp_path):
    folder = tmp_path / "My Tools"
    folder.mkdir()
    program = folder / "server.exe"
    program.write_bytes(b"")
    split = [str(tmp_path / "My"), "Tools" + os.sep + "server.exe", "--x"]
    assert mcp_client.unquoted_spaced_program(split) == str(program)
    # A planted My.exe next to the folder does not make the split command acceptable.
    (tmp_path / "My.exe").write_bytes(b"")
    assert mcp_client.unquoted_spaced_program(split) == str(program)
    # An exact existing program, a bare name, or nothing that joins up is left alone.
    assert mcp_client.unquoted_spaced_program([str(program), "--x"]) is None
    assert mcp_client.unquoted_spaced_program(["npx", "-y", "server"]) is None
    assert mcp_client.unquoted_spaced_program([str(tmp_path / "Other"), "thing"]) is None


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows guesses program paths; POSIX shells do not")
def test_routes_refuse_an_unquoted_spaced_program_on_windows(tmp_path, monkeypatch):
    from models.mcp_servers import McpStdioDecodeRequest

    routes_mcp = _routes(monkeypatch, tmp_path)
    folder = tmp_path / "My Tools"
    folder.mkdir()
    program = folder / "server.exe"
    program.write_bytes(b"")
    with pytest.raises(HTTPException) as exc:
        routes_mcp._validate_url(f"{program} --stdio")
    assert exc.value.status_code == 400
    assert exc.value.detail == (
        f'This program path contains spaces — wrap it in double quotes: "{program}"'
    )
    assert routes_mcp._validate_url(f'"{program}" --stdio') == f'"{program}" --stdio'
    # An unquoted row saved before the check still opens in the editor.
    decoded = routes_mcp.decode_stdio_command(
        McpStdioDecodeRequest(url = f"{program} --stdio"), current_subject = "u"
    )
    assert decoded.command == str(tmp_path / "My")
