# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP server lifecycle (shared vs per-chat processes, idle timeout, list_changed, discovery wait, restart/stop),
against a real stdio fixture server (tests/fixtures/mcp_lifecycle_server.py)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from core.inference import mcp_client
from core.inference import tools as tools_mod
from models.mcp_servers import (
    McpServerCreate,
    McpServerImportRequest,
    McpServerUpdate,
    McpStdioCommand,
)
from routes import mcp_servers as routes_mcp
from storage import mcp_servers_db
from utils.account_context import AccountContext, run_as
from utils.process_lifetime import pid_is_running


FIXTURE = Path(__file__).parent / "fixtures" / "mcp_lifecycle_server.py"


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    mcp_client.invalidate_tool_cache()
    mcp_client._server_errors.clear()
    yield tmp_path
    mcp_client.close_mcp_sessions(all_accounts = True)
    mcp_client.invalidate_tool_cache()
    mcp_client._server_errors.clear()


def _command() -> str:
    return routes_mcp.encode_stdio_command(
        McpStdioCommand(command = sys.executable, arguments = [str(FIXTURE)]),
        current_subject = "u",
    ).url


def _create(studio: Path, name: str = "board", **fields) -> tuple[str, Path]:
    launches = studio / f"{name}-launches.log"
    env = {"UNSLOTH_MCP_LIFECYCLE_LOG": str(launches), **fields.pop("env", {})}
    created = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = name, url = _command(), headers = env, **fields),
            current_subject = "u",
        )
    )
    return created.id, launches


def _launches(path: Path) -> list[int]:
    if not path.exists():
        return []
    return [int(line) for line in path.read_text(encoding = "utf-8").split()]


def _call(server_id: str, tool: str, thread: str | None, **arguments) -> str:
    return tools_mod.execute_tool(
        f"mcp__{server_id}__{tool}", arguments, session_id = "project", thread_id = thread, timeout = 30
    )


def _whoami(server_id: str, thread: str | None) -> dict:
    output = _call(server_id, "whoami", thread)
    assert not output.startswith("Error"), output
    return json.loads(output)


def _tool_names(specs: list[dict]) -> set[str]:
    return {spec["function"]["name"].split("__", 2)[-1] for spec in specs}


def _discover(thread: str | None = None) -> set[str]:
    return _tool_names(
        asyncio.run(tools_mod.get_enabled_mcp_tools(session_id = "project", thread_id = thread))
    )


def _wait_gone(pid: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while pid_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not pid_is_running(pid)


# ── storage: the migration and the defaults ─────────────────────────


def test_rows_saved_before_the_setting_stay_per_chat(studio):
    """An existing DB gains the columns in place; its rows keep the isolation they were set up with."""
    db_path = mcp_servers_db.studio_db_path()
    db_path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(db_path)
    # The table as it stood before this change.
    conn.execute(
        """
        CREATE TABLE mcp_servers (
            id TEXT NOT NULL PRIMARY KEY, display_name TEXT NOT NULL, url TEXT NOT NULL, headers_json TEXT,
            is_enabled INTEGER NOT NULL DEFAULT 1, use_oauth INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, builtin_id TEXT, builtin_config_json TEXT,
            image_input_mappings_json TEXT, oauth_client_id TEXT, oauth_client_secret TEXT, cwd TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO mcp_servers (id, display_name, url, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("old", "Konnect", "konnect.exe --stdio", "2026-01-01", "2026-01-01"),
    )
    conn.commit()
    conn.close()

    row = mcp_servers_db.get_server("old")
    assert row["process_mode"] == "per_chat"
    assert row["idle_timeout_seconds"] is None
    response = routes_mcp._row_to_response(row)
    assert (response.process_mode, response.idle_timeout_seconds) == ("per_chat", 300)
    lifecycle = mcp_client.server_lifecycle(row)
    assert lifecycle == mcp_client.McpLifecycle("old", False, 300.0)

    # Idempotent: a second schema pass over the migrated DB changes nothing.
    mcp_servers_db.reset_schema_state_for_tests()
    assert mcp_servers_db.get_server("old")["process_mode"] == "per_chat"


def test_new_and_imported_local_programs_default_to_shared(studio):
    shared_id, _ = _create(studio, "new")
    row = mcp_servers_db.get_server(shared_id)
    assert row["process_mode"] == "shared"
    assert routes_mcp._row_to_response(row).idle_timeout_seconds == 1800

    explicit_id, _ = _create(studio, "explicit", process_mode = "per_chat", idle_timeout_seconds = 60)
    explicit = routes_mcp._row_to_response(mcp_servers_db.get_server(explicit_id))
    assert (explicit.process_mode, explicit.idle_timeout_seconds) == ("per_chat", 60)

    http = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "remote", url = "https://mcp.example.test/mcp"),
            current_subject = "u",
        )
    )
    assert http.process_mode == "per_chat"
    assert mcp_client.server_lifecycle(mcp_servers_db.get_server(http.id)) is None

    imported = asyncio.run(
        routes_mcp.import_mcp_servers(
            McpServerImportRequest(
                config = {
                    "mcpServers": {
                        "imported": {"command": sys.executable, "args": [str(FIXTURE), "--x"]}
                    }
                }
            ),
            current_subject = "u",
        )
    )
    assert [server.process_mode for server in imported.created] == ["shared"]

    never = asyncio.run(
        routes_mcp.update_mcp_server(
            shared_id, McpServerUpdate(idle_timeout_seconds = 0), current_subject = "u"
        )
    )
    assert never.idle_timeout_seconds == 0
    assert mcp_client.server_lifecycle(mcp_servers_db.get_server(shared_id)).idle_ttl == float("inf")


# ── shared vs per chat ───────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_shared_server_is_one_process_for_every_chat_and_discovery(studio):
    server_id, launches = _create(studio)
    assert {"whoami", "load_toolset", "slow"} <= _discover("chat-a")
    # Discovery started the shared process instead of a throwaway copy...
    assert len(_launches(launches)) == 1
    first = _whoami(server_id, "chat-a")
    second = _whoami(server_id, "chat-b")
    third = _whoami(server_id, None)
    # ...and both chats (and a call with no chat) then talk to that same process, keeping its state.
    assert first["pid"] == second["pid"] == third["pid"] == _launches(launches)[0]
    assert [first["calls"], second["calls"], third["calls"]] == [1, 2, 3]
    assert len(_launches(launches)) == 1


@pytest.mark.timeout(120)
def test_per_chat_server_keeps_a_process_per_chat(studio):
    server_id, launches = _create(studio, process_mode = "per_chat")
    assert "whoami" in _discover("chat-a")
    a1 = _whoami(server_id, "chat-a")
    b1 = _whoami(server_id, "chat-b")
    a2 = _whoami(server_id, "chat-a")
    assert a1["pid"] == a2["pid"] != b1["pid"]
    assert [a1["calls"], b1["calls"], a2["calls"]] == [1, 1, 2]
    # The one-shot discovery probe, then one process per chat.
    assert len(_launches(launches)) == 3


@pytest.mark.timeout(120)
def test_switching_mode_ends_the_old_processes(studio):
    server_id, _ = _create(studio)
    assert "whoami" in _discover("chat-a")
    pid = _whoami(server_id, "chat-a")["pid"]
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            routes_mcp.update_mcp_server(
                server_id,
                McpServerUpdate(process_mode = "per_chat"),
                current_subject = "u",
                via_api_key = True,
            )
        )
    assert refused.value.status_code == 403
    assert pid_is_running(pid)

    updated = asyncio.run(
        routes_mcp.update_mcp_server(
            server_id, McpServerUpdate(process_mode = "per_chat"), current_subject = "u"
        )
    )
    assert (updated.process_mode, updated.idle_timeout_seconds) == ("per_chat", 300)
    assert _wait_gone(pid)
    assert mcp_client.get_cached_tools(server_id) is None
    assert _whoami(server_id, "chat-a")["pid"] != _whoami(server_id, "chat-b")["pid"]


@pytest.mark.timeout(120)
def test_testing_a_running_shared_server_asks_its_process(studio):
    server_id, launches = _create(studio)
    row = mcp_servers_db.get_server(server_id)
    pid = _whoami(server_id, "chat-a")["pid"]
    from models.mcp_servers import McpServerTestRequest

    probe = asyncio.run(
        routes_mcp.test_mcp_server(
            McpServerTestRequest(
                url = row["url"], headers = json.loads(row["headers_json"]), server_id = server_id
            ),
            current_subject = "u",
        )
    )
    assert (probe.ok, probe.tool_count) == (True, 3)
    # A second copy would have been launched (and, for a serial-port server, failed); this asked the running one.
    assert _launches(launches) == [pid]


# ── idle timeout ─────────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_idle_timeout_reaps_a_shared_process_and_never_means_never(studio):
    server_id, _ = _create(studio, idle_timeout_seconds = 60)
    pid = _whoami(server_id, "chat-a")["pid"]
    status = routes_mcp._status_response(mcp_servers_db.get_server(server_id))
    assert status.state == "idle" and status.processes == 1
    assert 0 < status.stops_in_seconds <= 60

    mcp_client._reap_idle_sessions(now = time.monotonic() + 30)
    assert pid_is_running(pid)
    mcp_client._reap_idle_sessions(now = time.monotonic() + 61)
    assert _wait_gone(pid)
    assert routes_mcp._status_response(mcp_servers_db.get_server(server_id)).state == "stopped"

    asyncio.run(
        routes_mcp.update_mcp_server(
            server_id, McpServerUpdate(idle_timeout_seconds = 0), current_subject = "u"
        )
    )
    pid = _whoami(server_id, "chat-a")["pid"]
    mcp_client._reap_idle_sessions(now = time.monotonic() + 10 * 365 * 86400)
    assert pid_is_running(pid)
    assert routes_mcp._status_response(mcp_servers_db.get_server(server_id)).stops_in_seconds is None

    # A new timeout reaches the running process without restarting it.
    asyncio.run(
        routes_mcp.update_mcp_server(
            server_id, McpServerUpdate(idle_timeout_seconds = 60), current_subject = "u"
        )
    )
    assert pid_is_running(pid)
    mcp_client._reap_idle_sessions(now = time.monotonic() + 61)
    assert _wait_gone(pid)


# ── notifications/tools/list_changed ─────────────────────────────────


@pytest.mark.timeout(120)
def test_list_changed_marks_shared_tools_stale_and_relists_over_the_same_process(studio):
    server_id, launches = _create(studio)
    assert "probe_board" not in _discover("chat-a")
    assert not mcp_client.tools_cache_stale(server_id)

    assert _call(server_id, "load_toolset", "chat-a", name = "probe_board") == "loaded probe_board"
    # The notification arrived on the shared session and invalidated the server's cached list...
    assert mcp_client.tools_cache_stale(server_id)
    # ...whose old entries are still served until the re-list lands.
    assert "whoami" in {tool["name"] for tool in mcp_client.get_cached_tools(server_id)}

    # The next send, from another chat, re-lists over that same process: no respawn, and the new tool is there.
    assert "probe_board" in _discover("chat-b")
    assert not mcp_client.tools_cache_stale(server_id)
    assert len(_launches(launches)) == 1
    loaded = json.loads(_call(server_id, "probe_board", "chat-b"))
    assert loaded == {"pid": _launches(launches)[0], "tool": "probe_board"}

    # A process that changed its tools takes them with it: once stopped, the list is stale again.
    mcp_client.stop_server_processes(mcp_servers_db.get_server(server_id))
    assert mcp_client.tools_cache_stale(server_id)


@pytest.mark.timeout(120)
def test_list_changed_on_a_per_chat_process_reaches_only_that_chat(studio):
    server_id, launches = _create(studio, process_mode = "per_chat")
    assert "probe_board" not in _discover("chat-a")
    chat_a_pid = _whoami(server_id, "chat-a")["pid"]
    _whoami(server_id, "chat-b")
    spawned = len(_launches(launches))

    assert _call(server_id, "load_toolset", "chat-a", name = "probe_board") == "loaded probe_board"
    # Chat A's own process re-lists for chat A; chat B's process and the server's cache are untouched.
    assert "probe_board" in _discover("chat-a")
    assert "probe_board" not in _discover("chat-b")
    assert not mcp_client.tools_cache_stale(server_id)
    assert len(_launches(launches)) == spawned
    assert json.loads(_call(server_id, "probe_board", "chat-a")) == {
        "pid": chat_a_pid,
        "tool": "probe_board",
    }


# ── discovery on a send ──────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_slow_discovery_does_not_hold_the_send_and_finishes_in_the_background(studio):
    server_id, launches = _create(studio, env = {"UNSLOTH_MCP_LIFECYCLE_START_DELAY": "4"})

    async def _sends():
        started = time.monotonic()
        first = await tools_mod.get_enabled_mcp_tools(thread_id = "chat-a")
        first_took = time.monotonic() - started
        deadline = time.monotonic() + 60
        while mcp_client.get_cached_tools(server_id) is None and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        second = await tools_mod.get_enabled_mcp_tools(thread_id = "chat-b")
        return first, first_took, second

    first, first_took, second = asyncio.run(_sends())
    assert first == []
    assert first_took < tools_mod._MCP_DISCOVERY_SEND_WAIT + 1.5
    assert "whoami" in _tool_names(second)
    # The background discovery started the shared process; the chat's call uses it rather than a new one.
    assert _whoami(server_id, "chat-b")["pid"] == _launches(launches)[0]
    assert len(_launches(launches)) == 1


def test_sends_during_one_discovery_wait_on_it_instead_of_probing_again(studio, monkeypatch):
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x.test/mcp")
    probes: list[str] = []

    async def slow_probe(url, headers = None, timeout = None, use_oauth = False, cwd = None):
        probes.append(url)
        await asyncio.sleep(2.5)
        return [{"name": "echo", "inputSchema": {"type": "object"}}]

    monkeypatch.setattr(tools_mod, "list_tools_async", slow_probe)
    monkeypatch.setattr(tools_mod, "_MCP_DISCOVERY_SEND_WAIT", 0.2)

    async def _sends():
        first = await tools_mod.get_enabled_mcp_tools()
        second = await tools_mod.get_enabled_mcp_tools()
        await asyncio.sleep(3.0)
        third = await tools_mod.get_enabled_mcp_tools()
        return first, second, third

    first, second, third = asyncio.run(_sends())
    assert (first, second) == ([], [])
    assert _tool_names(third) == {"echo"}
    assert probes == ["https://x.test/mcp"]


# ── status, restart, stop ────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_restart_and_stop_routes(studio):
    server_id, launches = _create(studio)
    statuses = {s.server_id: s for s in routes_mcp.list_mcp_server_status(current_subject = "u")}
    assert statuses[server_id].state == "stopped"

    before = _whoami(server_id, "chat-a")["pid"]
    status = asyncio.run(routes_mcp.restart_mcp_server(server_id, current_subject = "u"))
    assert status.state == "idle" and status.processes == 1 and status.uptime_seconds is not None
    assert status.log_path and Path(status.log_path).is_file()
    assert _wait_gone(before)
    after = _launches(launches)[-1]
    assert after != before and pid_is_running(after)
    # The restart re-read the tools over the new process.
    assert "whoami" in {tool["name"] for tool in mcp_client.get_cached_tools(server_id)}
    assert _whoami(server_id, "chat-b")["pid"] == after

    stopped = asyncio.run(routes_mcp.stop_mcp_server(server_id, current_subject = "u"))
    assert stopped.state == "stopped" and stopped.processes == 0
    assert _wait_gone(after)

    # Per chat: a restart ends every chat's process, and each starts again on its next call.
    per_chat_id, _ = _create(studio, "per-chat", process_mode = "per_chat")
    pid_a = _whoami(per_chat_id, "chat-a")["pid"]
    pid_b = _whoami(per_chat_id, "chat-b")["pid"]
    assert routes_mcp._status_response(mcp_servers_db.get_server(per_chat_id)).processes == 2
    restarted = asyncio.run(routes_mcp.restart_mcp_server(per_chat_id, current_subject = "u"))
    assert restarted.state == "stopped"
    assert _wait_gone(pid_a) and _wait_gone(pid_b)


@pytest.mark.timeout(120)
def test_a_failed_start_shows_as_failed_until_stopped(studio, tmp_path):
    broken = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(
                display_name = "broken",
                url = routes_mcp.encode_stdio_command(
                    McpStdioCommand(
                        command = sys.executable, arguments = ["-c", "import sys; sys.exit(3)"]
                    ),
                    current_subject = "u",
                ).url,
            ),
            current_subject = "u",
        )
    )
    assert _discover("chat-a") == set()
    status = routes_mcp._status_response(mcp_servers_db.get_server(broken.id))
    assert status.state == "failed"
    assert "exited during startup" in status.last_error
    assert asyncio.run(routes_mcp.stop_mcp_server(broken.id, current_subject = "u")).state == "stopped"


def test_lifecycle_routes_keep_the_local_program_gates(studio, monkeypatch):
    server_id, _ = _create(studio)
    http = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "remote", url = "https://mcp.example.test/mcp"),
            current_subject = "u",
        )
    )
    for route in (routes_mcp.restart_mcp_server, routes_mcp.stop_mcp_server):
        for caller in ({"via_api_key": True}, {"no_credential": True}):
            with pytest.raises(HTTPException) as refused:
                asyncio.run(route(server_id, current_subject = "u", **caller))
            assert refused.value.status_code == 403
        with pytest.raises(HTTPException) as not_local:
            asyncio.run(route(http.id, current_subject = "u"))
        assert not_local.value.status_code == 400
        with pytest.raises(HTTPException) as missing:
            asyncio.run(route("nope", current_subject = "u"))
        assert missing.value.status_code == 404
    assert routes_mcp.list_mcp_server_status(current_subject = "u", via_api_key = True) == []
    assert routes_mcp.list_mcp_server_status(current_subject = "u", no_credential = True) == []

    managed = AccountContext("alice", "alice")
    assert run_as(managed, routes_mcp.list_mcp_server_status, current_subject = "alice") == []
    with pytest.raises(HTTPException) as not_owner:
        run_as(
            managed,
            lambda: asyncio.run(routes_mcp.stop_mcp_server(server_id, current_subject = "alice")),
        )
    assert not_owner.value.status_code in (400, 403)

    # Remote Access (or --disable-tools) suspends local programs: nothing may start one, but stopping is fine.
    monkeypatch.setattr(routes_mcp, "stdio_mcp_enabled", lambda: False)
    monkeypatch.setattr(routes_mcp, "stdio_mcp_disabled_reason", lambda: "Remote Access is on")
    with pytest.raises(HTTPException) as suspended:
        asyncio.run(routes_mcp.restart_mcp_server(server_id, current_subject = "u"))
    assert (suspended.value.status_code, suspended.value.detail) == (400, "Remote Access is on")
    assert asyncio.run(routes_mcp.stop_mcp_server(server_id, current_subject = "u")).state == "stopped"


@pytest.mark.timeout(120)
def test_suspended_local_programs_are_reaped_even_before_their_timeout(studio, monkeypatch):
    server_id, _ = _create(studio, idle_timeout_seconds = 0)
    pid = _whoami(server_id, "chat-a")["pid"]
    mcp_client._reap_idle_sessions()
    assert pid_is_running(pid)
    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: False)
    mcp_client._reap_idle_sessions()
    assert _wait_gone(pid)


# ── a busy shared server ─────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_a_call_behind_another_chats_call_says_busy_instead_of_hanging(studio, monkeypatch):
    server_id, _ = _create(studio)
    monkeypatch.setattr(mcp_client, "_SHARED_BUSY_GRACE", 0.5)
    _whoami(server_id, "chat-a")
    outcome: dict = {}
    worker = threading.Thread(
        target = lambda: outcome.setdefault("slow", _call(server_id, "slow", "chat-a", seconds = 4))
    )
    worker.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = routes_mcp._status_response(mcp_servers_db.get_server(server_id))
        if status.busy_tool == "slow":
            break
        time.sleep(0.05)
    assert status.state == "running" and status.busy_tool == "slow"

    started = time.monotonic()
    busy = _call(server_id, "whoami", "chat-b")
    assert time.monotonic() - started < 3
    assert busy.startswith("Error:") and "busy with another chat" in busy and "'slow'" in busy
    worker.join(20)
    assert outcome["slow"] == "done"
    # Serialized, not refused: once the other chat's call is done this one runs on the same process (whose
    # counter only whoami bumps: chat A's one, then this).
    assert _whoami(server_id, "chat-b")["calls"] == 2


# ── shutdown ─────────────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_shutdown_ends_every_long_lived_process(studio):
    shared_id, _ = _create(studio, "shared")
    per_chat_id, _ = _create(studio, "per-chat", process_mode = "per_chat")
    pids = [
        _whoami(shared_id, "chat-a")["pid"],
        _whoami(per_chat_id, "chat-a")["pid"],
        _whoami(per_chat_id, "chat-b")["pid"],
    ]
    assert all(pid_is_running(pid) for pid in pids)
    mcp_client.shutdown_mcp_sessions()
    assert all(_wait_gone(pid) for pid in pids)
    assert mcp_client._mcp_sessions == {}
