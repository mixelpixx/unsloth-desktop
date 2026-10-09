# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-tool MCP settings: tools the owner turns off are never offered to a model and are refused if one calls them
anyway; tools set to ask first pause under "Approve for me"; the dialog's catalog prices each tool's schema."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from core.inference import mcp_client
from core.inference import tools as tools_mod
from models.mcp_servers import McpServerCreate, McpServerUpdate, McpStdioCommand, McpUiToolCallRequest
from routes import mcp_servers as routes_mcp
from storage import mcp_servers_db


LIFECYCLE_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_lifecycle_server.py"


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    mcp_client.invalidate_tool_cache()
    # Nothing is loaded in a test process, but never let a resident model's tokenizer leak into a count.
    monkeypatch.setattr(tools_mod, "_loaded_context_tokens", lambda: None)
    yield tmp_path
    mcp_client.close_mcp_sessions(all_accounts = True)
    mcp_client.invalidate_tool_cache()


def _tool(name: str, description: str = "", properties: dict | None = None, **extra) -> dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties or {}},
        **extra,
    }


BOARD_TOOLS = [
    _tool("list_items", "List the items on the board. Read only.", {"filter": {"type": "string"}}),
    _tool("ssh_exec", "Run a shell command on the remote host.", {"command": {"type": "string"}}),
    _tool("read_note", "Read one note.", {"id": {"type": "string"}}),
    _tool("widget_only", "Drawn by the server's widget.", _meta = {"ui": {"visibility": ["app"]}}),
]


def _http_server(server_id: str = "s1", name: str = "Board", **fields) -> dict:
    mcp_servers_db.create_server(
        id = server_id, display_name = name, url = f"https://{server_id}.example.test/mcp", **fields
    )
    mcp_client.cache_tools(server_id, [dict(tool) for tool in BOARD_TOOLS])
    return mcp_servers_db.get_server(server_id)


def _set(server_id: str, **payload) -> object:
    return asyncio.run(
        routes_mcp.update_mcp_server(server_id, McpServerUpdate(**payload), current_subject = "u")
    )


def _names(specs: list[dict]) -> set[str]:
    return {spec["function"]["name"].split("__", 2)[-1] for spec in specs}


# ── storage ─────────────────────────────────────────────────────────


def test_an_existing_database_gains_the_columns_and_keeps_every_tool_on(studio):
    db_path = mcp_servers_db.studio_db_path()
    db_path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(db_path)
    # The table as it stood before per-tool settings.
    conn.execute(
        """
        CREATE TABLE mcp_servers (
            id TEXT NOT NULL PRIMARY KEY, display_name TEXT NOT NULL, url TEXT NOT NULL, headers_json TEXT,
            is_enabled INTEGER NOT NULL DEFAULT 1, use_oauth INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, builtin_id TEXT, builtin_config_json TEXT,
            image_input_mappings_json TEXT, oauth_client_id TEXT, oauth_client_secret TEXT, cwd TEXT,
            process_mode TEXT NOT NULL DEFAULT 'per_chat', idle_timeout_seconds INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO mcp_servers (id, display_name, url, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("old", "Board", "https://old.example.test/mcp", "2026-01-01", "2026-01-01"),
    )
    conn.commit()
    conn.close()

    row = mcp_servers_db.get_server("old")
    assert row["disabled_tools_json"] is None and row["ask_tools_json"] is None
    assert mcp_client.server_disabled_tools(row) == frozenset()
    mcp_client.cache_tools("old", [dict(tool) for tool in BOARD_TOOLS])
    assert _names(asyncio.run(tools_mod.get_enabled_mcp_tools())) == {"list_items", "ssh_exec", "read_note"}
    response = routes_mcp.list_mcp_servers(current_subject = "u")[0]
    assert (response.disabled_tools, response.ask_tools) == ([], [])

    # Running the migration again is a no-op.
    mcp_servers_db.reset_schema_state_for_tests()
    assert mcp_servers_db.get_server("old")["disabled_tools_json"] is None


def test_unreadable_stored_names_mean_none(studio):
    for raw in (None, "", "not json", '{"a": 1}', '[1, "", null, "ok"]'):
        assert mcp_client.server_disabled_tools({"disabled_tools_json": raw}) <= {"ok"}
    assert mcp_client.tool_names_json(["b", "a", "b", ""]) == '["a", "b"]'
    assert mcp_client.tool_names_json([]) is None


# ── the update API ──────────────────────────────────────────────────


def test_update_stores_the_whole_set_and_leaves_the_tool_cache_and_processes_alone(studio, monkeypatch):
    _http_server()
    closed: list = []
    monkeypatch.setattr(routes_mcp, "close_mcp_sessions", lambda *a, **k: closed.append(a))

    updated = _set("s1", disabled_tools = ["ssh_exec", "read_note", "ssh_exec"], ask_tools = ["list_items"])
    assert updated.disabled_tools == ["read_note", "ssh_exec"]
    assert updated.ask_tools == ["list_items"]
    row = mcp_servers_db.get_server("s1")
    assert json.loads(row["disabled_tools_json"]) == ["read_note", "ssh_exec"]
    # The server lists the same tools either way: nothing to re-discover, no process to restart.
    assert mcp_client.get_cached_tools("s1") is not None
    assert closed == []

    # Absent leaves it; [] and null clear it.
    assert _set("s1", display_name = "Renamed").disabled_tools == ["read_note", "ssh_exec"]
    assert _set("s1", disabled_tools = []).disabled_tools == []
    assert _set("s1", ask_tools = None).ask_tools == []
    assert mcp_servers_db.get_server("s1")["ask_tools_json"] is None


@pytest.mark.parametrize(
    "bad",
    [["ok", ""], ["ok", 3], "ssh_exec", ["x" * 257], ["t"] * 2001],
)
def test_update_validates_tool_names(bad):
    with pytest.raises(ValidationError):
        McpServerUpdate(disabled_tools = bad)


def test_an_api_key_cannot_change_a_local_programs_tools_but_keeps_http(studio, monkeypatch):
    monkeypatch.setattr(routes_mcp, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: True)
    mcp_servers_db.create_server(id = "local", display_name = "SSH", url = "ssh-connect --stdio")
    _http_server("remote")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "local",
                McpServerUpdate(disabled_tools = ["ssh_exec"]),
                current_subject = "u",
                via_api_key = True,
            )
        )
    assert refused.value.status_code == 403
    assert mcp_servers_db.get_server("local")["disabled_tools_json"] is None
    # The local program's catalog is the signed-in window's alone, as list_mcp_servers hides its row.
    for caller in ({"via_api_key": True}, {"no_credential": True}):
        with pytest.raises(HTTPException) as hidden:
            routes_mcp.get_mcp_server_tool_catalog("local", current_subject = "u", **caller)
        assert hidden.value.status_code == 403
    assert routes_mcp.get_mcp_server_tool_catalog("local", current_subject = "u").cached is False

    updated = asyncio.run(
        routes_mcp.update_mcp_server(
            "remote",
            McpServerUpdate(disabled_tools = ["ssh_exec"]),
            current_subject = "u",
            via_api_key = True,
        )
    )
    assert updated.disabled_tools == ["ssh_exec"]
    assert routes_mcp.get_mcp_server_tool_catalog(
        "remote", current_subject = "u", via_api_key = True
    ).enabled_count == 2
    with pytest.raises(HTTPException) as missing:
        routes_mcp.get_mcp_server_tool_catalog("nope", current_subject = "u")
    assert missing.value.status_code == 404


def test_a_managed_integration_keeps_its_own_setup(studio):
    mcp_servers_db.create_server(
        id = "blend", display_name = "Blender", url = "", is_enabled = False, builtin_id = "blender",
        builtin_config_json = "{}",
    )
    with pytest.raises(HTTPException) as refused:
        _set("blend", disabled_tools = ["execute_code"])
    assert refused.value.status_code == 400


# ── every path that builds a tool list ──────────────────────────────


def test_turned_off_tools_leave_the_send_and_the_count(studio):
    _http_server()
    _set("s1", disabled_tools = ["ssh_exec"])
    assert _names(asyncio.run(tools_mod.get_enabled_mcp_tools())) == {"list_items", "read_note"}
    cached, complete = tools_mod.cached_mcp_tools()
    assert complete is True
    assert _names(cached) == {"list_items", "read_note"}

    # Every tool off: the server offers nothing, and the catalog stays complete (nothing left to discover).
    _set("s1", disabled_tools = ["ssh_exec", "list_items", "read_note"])
    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []
    assert tools_mod.cached_mcp_tools() == ([], True)


def test_turned_off_tools_leave_the_request_selector_every_backend_uses(studio, monkeypatch):
    """_select_request_tools builds the catalog for the GGUF, safetensors, MLX, external-provider, Codex and
    OpenAI-compatible (/v1/chat/completions) loops alike."""
    import routes.inference as routes_inference

    monkeypatch.setattr(routes_inference, "_thread_has_conversation_archive", lambda _tid: False)
    _http_server()
    _set("s1", disabled_tools = ["ssh_exec"])
    payload = types.SimpleNamespace(
        enabled_tools = None, rag_scope = None, thread_id = "t1", session_id = "p1", bypass_permissions = False
    )
    tools = asyncio.run(
        routes_inference._select_request_tools(payload, tools_on = False, mcp_allowed = True)
    )
    names = [tool["function"]["name"] for tool in tools]
    assert "mcp__s1__list_items" in names
    assert not any(name.endswith("ssh_exec") for name in names)


def test_a_chats_own_relisted_process_is_filtered_too(studio, monkeypatch):
    """A per-chat process that announced new tools offers that chat its own list (session_tool_overlay); a tool the
    owner turned off stays off there too."""
    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(tools_mod, "stdio_mcp_enabled", lambda: True)
    mcp_servers_db.create_server(
        id = "kon", display_name = "Konnect", url = "konnect --stdio", process_mode = "per_chat"
    )
    mcp_client.cache_tools("kon", [_tool("open_project")])
    _set("kon", disabled_tools = ["place_part"])
    overlay_tools = [_tool("open_project"), _tool("place_part"), _tool("route_trace")]
    monkeypatch.setattr(tools_mod, "session_tool_overlay", lambda *a, **k: (overlay_tools, False))

    specs = asyncio.run(tools_mod.get_enabled_mcp_tools(session_id = "p1", thread_id = "chat-a"))
    assert _names(specs) == {"open_project", "route_trace"}


def _stdio_command() -> str:
    return routes_mcp.encode_stdio_command(
        McpStdioCommand(command = sys.executable, arguments = [str(LIFECYCLE_FIXTURE)]),
        current_subject = "u",
    ).url


@pytest.mark.timeout(120)
@pytest.mark.parametrize("process_mode", ["shared", "per_chat"])
def test_a_tool_a_running_server_adds_later_is_filtered_by_name(studio, monkeypatch, process_mode):
    """Against a real stdio server that adds a tool at runtime and says so with notifications/tools/list_changed:
    the shared process's re-list (server cache) and the per-chat one (the chat's own list) both drop a turned-off
    name, and a tool nobody turned off arrives on, because the stored set is the tools turned OFF."""
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    created = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "board", url = _stdio_command(), process_mode = process_mode),
            current_subject = "u",
        )
    )
    _set(created.id, disabled_tools = ["probe_board", "slow"])

    def discover(expected: set[str]) -> set[str]:
        # A send waits only briefly for discovery and finishes it in the background, so ask again until it lands.
        deadline = time.monotonic() + 60
        while True:
            names = _names(
                asyncio.run(tools_mod.get_enabled_mcp_tools(session_id = "p1", thread_id = "chat-a"))
            )
            if names == expected or time.monotonic() > deadline:
                return names
            time.sleep(0.2)

    assert discover({"whoami", "load_toolset"}) == {"whoami", "load_toolset"}
    for loaded in ("probe_board", "fresh_board"):
        call = tools_mod.execute_tool(
            f"mcp__{created.id}__load_toolset",
            {"name": loaded},
            session_id = "p1",
            thread_id = "chat-a",
            timeout = 30,
        )
        assert call == f"loaded {loaded}"
    on = {"whoami", "load_toolset", "fresh_board"}
    assert discover(on) == on
    refused = tools_mod.execute_tool(
        f"mcp__{created.id}__probe_board", {}, session_id = "p1", thread_id = "chat-a", timeout = 30
    )
    assert refused.startswith("Error: MCP tool 'probe_board' on 'board' is turned off")
    assert json.loads(
        tools_mod.execute_tool(
            f"mcp__{created.id}__fresh_board", {}, session_id = "p1", thread_id = "chat-a", timeout = 30
        )
    )["tool"] == "fresh_board"


# ── a call to a turned-off tool ─────────────────────────────────────


def test_calling_a_turned_off_tool_is_refused_without_reaching_the_server(studio, monkeypatch):
    _http_server()
    # The run's tool list was built while the tool was still on...
    assert "ssh_exec" in _names(asyncio.run(tools_mod.get_enabled_mcp_tools()))
    calls: list[str] = []

    def fake_call(**kwargs):
        calls.append(kwargs["name"])
        return "ran"

    monkeypatch.setattr(tools_mod, "call_tool_sync", fake_call)
    # ...then the owner turns it off mid-run, and the model calls it (or names one it was never offered).
    _set("s1", disabled_tools = ["ssh_exec"])
    result = tools_mod.execute_tool("mcp__s1__ssh_exec", {"command": "rm -rf /"})
    assert result == (
        "Error: MCP tool 'ssh_exec' on 'Board' is turned off in Studio's MCP server settings, so it was not run. "
        "Do not call it again; use one of the tools you were given, or tell the user it is turned off."
    )
    assert calls == []
    # Read as a tool error by the loops (is_tool_error), not a result.
    from core.inference.tool_loop_controller import is_tool_error

    assert is_tool_error(result)
    # Its schema and image mapping go with it.
    assert tools_mod._mcp_tool_schema("mcp__s1__ssh_exec").startswith("Error: MCP tool 'ssh_exec'")
    assert tools_mod.mcp_tool_input_schema("mcp__s1__ssh_exec") is None
    # A tool still on runs as before.
    assert tools_mod.execute_tool("mcp__s1__list_items", {}) == "ran"
    assert calls == ["list_items"]


def test_an_image_mapping_on_a_turned_off_tool_is_inactive(studio):
    _http_server(
        image_input_mappings_json = json.dumps([{"tool": "read_note", "field": "id", "encoding": "base64"}])
    )
    assert routes_mcp._image_mappings_active(mcp_servers_db.get_server("s1")) is True
    assert tools_mod.mcp_catalog_takes_image(["mcp__s1__read_note"]) is True
    updated = _set("s1", disabled_tools = ["read_note"])
    assert updated.image_mappings_active is False
    assert tools_mod.mcp_catalog_takes_image(["mcp__s1__read_note"]) is False


def test_a_widget_cannot_call_a_turned_off_tool(studio):
    mcp_servers_db.create_server(id = "app", display_name = "App", url = "https://app.example.test/mcp")
    mcp_client.cache_tools("app", [_tool("refresh_board"), _tool("delete_board")])
    _set("app", disabled_tools = ["delete_board"])
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            routes_mcp.call_mcp_ui_tool(
                "app",
                McpUiToolCallRequest(tool_name = "delete_board", permission_mode = "full"),
                current_subject = "u",
            )
        )
    assert refused.value.status_code == 403
    assert "turned off" in refused.value.detail


# ── ask before running ──────────────────────────────────────────────


def test_an_ask_first_tool_pauses_under_approve_for_me_only(studio):
    from state.tool_policy import needs_tool_confirmation

    _http_server()
    name = "mcp__s1__list_items"
    assert tools_mod.is_high_risk_tool_call(name, {}) is False
    assert tools_mod.is_potentially_unsafe_tool_call(name, {}) is False

    _set("s1", ask_tools = ["list_items"])
    assert tools_mod.is_high_risk_tool_call(name, {}) is True
    assert tools_mod.is_potentially_unsafe_tool_call(name, {}) is True
    # The ordinary classifier still decides the other tools.
    assert tools_mod.is_high_risk_tool_call("mcp__s1__read_note", {}) is False

    def asks(mode: str, *, bypass: bool = False) -> bool:
        return needs_tool_confirmation(
            confirm_tool_calls = True,
            bypass_permissions = bypass,
            permission_mode = mode,
            name = name,
            arguments = {},
        )

    assert asks("auto") is True
    assert asks("ask") is True
    # "Run automatically" and Full access are the user's choice never to be asked.
    assert asks("off") is False
    assert asks("full", bypass = True) is False


def test_an_ask_first_read_failure_falls_back_to_the_classifier(studio, monkeypatch):
    def broken(_key):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(tools_mod.mcp_servers_db, "get_server_for_tool", broken)
    assert tools_mod.mcp_tool_always_asks("mcp__s1__list_items") is False
    assert tools_mod.is_high_risk_tool_call("mcp__s1__delete_everything", {}) is True
    assert tools_mod.mcp_tool_always_asks("python") is False


# ── the catalog the dialog shows ────────────────────────────────────


def test_the_catalog_lists_model_tools_with_their_state_and_cost(studio):
    _http_server()
    assert routes_mcp.get_mcp_server_tool_catalog("s1", current_subject = "u").cached is True
    _set("s1", disabled_tools = ["ssh_exec", "gone_tool"], ask_tools = ["read_note"])
    catalog = routes_mcp.get_mcp_server_tool_catalog("s1", current_subject = "u")

    assert [tool.name for tool in catalog.tools] == ["list_items", "ssh_exec", "read_note"]
    by_name = {tool.name: tool for tool in catalog.tools}
    assert (by_name["ssh_exec"].enabled, by_name["ssh_exec"].ask) == (False, False)
    assert (by_name["read_note"].enabled, by_name["read_note"].ask) == (True, True)
    assert by_name["list_items"].summary == "List the items on the board."
    assert by_name["list_items"].description == "List the items on the board. Read only."
    assert (catalog.enabled_count, catalog.total_count) == (2, 3)
    assert catalog.enabled_tokens == by_name["list_items"].tokens + by_name["read_note"].tokens
    assert catalog.unlisted_disabled == ["gone_tool"]
    assert catalog.tokens_measured is False

    # Priced with the listing's own estimator, on the spec exactly as it ships to the model.
    server = mcp_servers_db.get_server("s1")
    spec = tools_mod._mcp_specs_for_server(server, [BOARD_TOOLS[0]])[0]
    expected = tools_mod._text_token_estimate(json.dumps(spec, separators = (",", ":")))
    assert by_name["list_items"].tokens == round(expected)


def test_the_catalog_uses_the_loaded_models_tokenizer_when_there_is_one(studio, monkeypatch):
    _http_server()
    counted: list[str] = []

    def measure(text, ctx):
        counted.append(text)
        return len(text) / 2

    monkeypatch.setattr(tools_mod, "_loaded_context_tokens", lambda: 32768)
    monkeypatch.setattr(tools_mod, "_measured_text_tokens", measure)
    catalog = routes_mcp.get_mcp_server_tool_catalog("s1", current_subject = "u")
    assert catalog.tokens_measured is True
    assert catalog.context_tokens == 32768
    # One count for the server's whole listing, not one per tool.
    assert len(counted) == 1
    server = mcp_servers_db.get_server("s1")
    spec = tools_mod._mcp_specs_for_server(server, [BOARD_TOOLS[1]])[0]
    ssh = next(tool for tool in catalog.tools if tool.name == "ssh_exec")
    assert ssh.tokens == round(len(json.dumps(spec, separators = (",", ":"))) / 2)


def test_the_catalog_before_discovery_starts_nothing(studio, monkeypatch):
    mcp_servers_db.create_server(id = "cold", display_name = "Cold", url = "https://cold.example.test/mcp")

    async def no_probe(*a, **k):
        raise AssertionError("the catalog must not probe")

    monkeypatch.setattr(tools_mod, "list_tools_async", no_probe)
    _set("cold", disabled_tools = ["x"])
    catalog = routes_mcp.get_mcp_server_tool_catalog("cold", current_subject = "u")
    assert (catalog.cached, catalog.tools, catalog.unlisted_disabled) == (False, [], ["x"])


def test_the_text_token_cost_estimate_is_unchanged():
    """_text_token_cost keeps its doubled estimate; the catalog shows the undoubled one it is built on."""
    text = "abcd" * 10 + "é" * 3
    assert tools_mod._text_token_estimate(text) == 13
    assert tools_mod._text_token_cost(text, 0) == 13 / tools_mod._UNMEASURED_ROOM_MARGIN
