# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import MCP servers from other apps on this computer (Claude Desktop incl. the Microsoft Store
install, Claude Code, Cursor, VS Code, Windsurf): discovery of every config shape, JSONC/BOM/broken
files, ${...} resolution, masking (no secret value in any response), duplicates, the enable rules
and the owner/UI-session gates. Every config here is synthetic and lives under tmp_path; the
machine's real home directory is never read.

Run from studio/backend:  python -m pytest tests/test_mcp_import_sources.py -q
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from core.inference import mcp_import_sources as sources_mod
from core.inference.mcp_client import parse_stdio_command
from storage import mcp_servers_db

# Synthetic credentials. Each must stay out of every response the dialog receives.
ENV_SECRET = "env-secret-value-1234"
HEADER_SECRET = "hdr-secret-abcdef-123456"
FLAG_SECRET = "flag-secret-777"
EQ_SECRET = "eq-secret-4242"
POSITIONAL_SECRET = "pos-secret-0123456789ABCDEFGHIJKLMNOP"
VENDOR_SECRET = "sk-test-SYNTHETIC-0123456789abcdef"
QUERY_SECRET = "qs-secret-98765"
USERINFO_SECRET = "pw-secret-5555"
PATH_SECRET = "0123456789abcdef0123456789abcdefXYZ"
RESOLVED_SECRET = "resolved-secret-31337"
ALL_SECRETS = (
    ENV_SECRET,
    HEADER_SECRET,
    FLAG_SECRET,
    EQ_SECRET,
    POSITIONAL_SECRET,
    VENDOR_SECRET,
    QUERY_SECRET,
    USERINFO_SECRET,
    PATH_SECRET,
    RESOLVED_SECRET,
)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A fake Windows profile under tmp_path. discover_sources() reads only from here."""
    home = tmp_path / "home"
    appdata = home / "AppData" / "Roaming"
    local = home / "AppData" / "Local"
    for folder in (home, appdata, local):
        folder.mkdir(parents = True, exist_ok = True)
    environ = {"APPDATA": str(appdata), "LOCALAPPDATA": str(local)}
    monkeypatch.setattr(sources_mod, "_environment", lambda: ("win32", environ, home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    return home


def _write(path: Path, content, *, encoding = "utf-8") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding = encoding)
    return path


def _desktop(home: Path) -> Path:
    return home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"


def _store_desktop(home: Path) -> Path:
    return (
        home
        / "AppData"
        / "Local"
        / "Packages"
        / "Claude_pzs8sxrjxfjjc"
        / "LocalCache"
        / "Roaming"
        / "Claude"
        / "claude_desktop_config.json"
    )


def _program(tmp_path: Path, name = "server.exe") -> str:
    path = tmp_path / "tools" / name
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"")
    if sys.platform != "win32":
        path.chmod(0o755)
    return str(path)


def _by_id(found):
    return {source.source_id: source for source in found}


# ── discovery: every source shape ────────────────────────────────────


def test_discovers_every_source_shape(machine):
    _write(_store_desktop(machine), {"mcpServers": {"store-fs": {"command": "npx"}}})
    _write(_desktop(machine), {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "pkg"]}}})
    _write(
        machine / ".claude.json",
        {
            "oauthAccount": {"emailAddress": "someone@example.com"},
            "mcpServers": {"remote": {"type": "http", "url": "https://example.com/mcp"}},
            "projects": {
                "C:\\work\\alpha": {"mcpServers": {"local": {"type": "stdio", "command": "uvx"}}},
                "C:\\work\\empty": {"mcpServers": {}},
                "C:\\work\\none": {"allowedTools": []},
            },
        },
    )
    _write(machine / ".cursor" / "mcp.json", {"mcpServers": {"cur": {"url": "https://c.example/mcp"}}})
    _write(
        machine / "AppData" / "Roaming" / "Code" / "User" / "mcp.json",
        '{\n  // user servers\n  "servers": {"gh": {"type": "http", "url": "https://g.example/mcp",},},\n}',
    )
    _write(
        machine / "AppData" / "Roaming" / "Code" / "User" / "settings.json",
        '\ufeff{"editor.fontSize": 14, /* mcp */ "mcp": {"servers": {"set": {"command": "node"}}}}',
    )
    _write(
        machine / ".codeium" / "windsurf" / "mcp_config.json",
        {"mcpServers": {"wind": {"serverUrl": "https://w.example/mcp"}}},
    )

    found = sources_mod.discover_sources()
    ids = [source.source_id for source in found]
    assert ids[0].startswith("claude-desktop-store-")
    assert ids[1:3] == ["claude-desktop", "claude-code"]
    assert ids[3].startswith("claude-code-project-")
    assert ids[4:] == ["cursor", "vscode", "vscode-settings", "windsurf"]

    by_id = _by_id(found)
    store = found[0]
    assert (store.app, store.label) == ("Claude Desktop", "Microsoft Store install")
    assert [server.name for server in store.servers] == ["store-fs"]
    assert [server.name for server in by_id["claude-desktop"].servers] == ["fs"]
    assert [server.name for server in by_id["claude-code"].servers] == ["remote"]
    project = found[3]
    assert (project.app, project.label) == ("Claude Code", "project alpha")
    assert [server.name for server in project.servers] == ["local"]
    assert [server.name for server in by_id["vscode"].servers] == ["gh"]
    assert [server.name for server in by_id["vscode-settings"].servers] == ["set"]
    assert by_id["vscode-settings"].label == "settings.json"
    assert [server.name for server in by_id["windsurf"].servers] == ["wind"]
    assert all(source.error is None for source in found)
    # Nothing but the server map leaves the Claude Code file.
    assert "someone@example.com" not in repr(found)


def test_identical_store_and_classic_desktop_configs_are_listed_once(machine):
    config = {"mcpServers": {"fs": {"command": "npx", "args": ["pkg"]}}}
    _write(_store_desktop(machine), config)
    _write(_desktop(machine), config)
    found = sources_mod.discover_sources()
    assert len(found) == 1
    assert found[0].label == "Microsoft Store install"


def test_missing_and_empty_files_are_not_sources(machine):
    assert sources_mod.discover_sources() == []
    _write(machine / ".claude.json", {"numStartups": 3})
    _write(machine / "AppData" / "Roaming" / "Code" / "User" / "settings.json", {"a": 1})
    assert sources_mod.discover_sources() == []


def test_broken_files_are_reported_not_raised(machine):
    _write(machine / ".cursor" / "mcp.json", '{"mcpServers": {"x": ')
    _write(_desktop(machine), {"mcpServers": ["not", "a", "map"]})
    _write(machine / ".codeium" / "windsurf" / "mcp_config.json", "[1, 2]")
    found = _by_id(sources_mod.discover_sources())
    assert found["cursor"].error.startswith("Couldn't read this file: invalid JSON at line 1")
    assert found["cursor"].servers == []
    assert "isn't a JSON object" in found["claude-desktop"].error
    assert "doesn't hold a JSON object" in found["windsurf"].error


def test_utf16_config_from_windows_powershell_is_read(machine):
    _write(_desktop(machine), {"mcpServers": {"fs": {"command": "npx"}}}, encoding = "utf-16")
    (source,) = sources_mod.discover_sources()
    assert [server.name for server in source.servers] == ["fs"]


def test_jsonc_keeps_comment_lookalikes_inside_strings():
    text = (
        '\ufeff{"servers": {"a": {"url": "https://h.example/x//y/*z*/", '
        '"command": "C:\\\\tools\\\\a.exe", "args": ["a,}", ],}, }, // tail\n}'
    )
    assert sources_mod.load_jsonc(text) == {
        "servers": {
            "a": {
                "url": "https://h.example/x//y/*z*/",
                "command": "C:\\tools\\a.exe",
                "args": ["a,}"],
            }
        }
    }


def test_mac_and_linux_locations(tmp_path):
    home = tmp_path / "home"
    mac = _write(
        home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        {"mcpServers": {"mac": {"command": "npx"}}},
    )
    xdg = tmp_path / "xdg"
    _write(xdg / "Code" / "User" / "mcp.json", {"servers": {"lin": {"command": "npx"}}})
    on_mac = sources_mod.discover_sources(platform = "darwin", environ = {}, home = home)
    assert [(source.source_id, source.path) for source in on_mac] == [("claude-desktop", str(mac))]
    on_linux = sources_mod.discover_sources(
        platform = "linux", environ = {"XDG_CONFIG_HOME": str(xdg)}, home = home
    )
    assert [source.source_id for source in on_linux] == ["vscode"]


# ── ${...} resolution ────────────────────────────────────────────────


def test_resolve_references_from_studio_environment(tmp_path):
    environ = {"TOKEN": "t-value", "EMPTY": ""}
    resolved, missing = sources_mod.resolve_references(
        {
            "env": {
                "A": "${env:TOKEN}",
                "B": "Bearer ${TOKEN}",
                "C": "${MISSING:-fallback}",
                "D": "${EMPTY}",
                "E": "${env:NOT_SET}",
                "F": "${input:apiKey}",
            },
            "args": ["${userHome}/data", "${workspaceFolder}", "${ALSO_MISSING}"],
            "cwd": "${userHome}",
        },
        environ = environ,
        home = tmp_path,
    )
    assert resolved["env"]["A"] == "t-value"
    assert resolved["env"]["B"] == "Bearer t-value"
    assert resolved["env"]["C"] == "fallback"
    assert resolved["env"]["D"] == ""
    # Unresolvable references stay as written, never silently empty.
    assert resolved["env"]["E"] == "${env:NOT_SET}"
    assert resolved["env"]["F"] == "${input:apiKey}"
    assert resolved["args"] == [f"{tmp_path}/data", "${workspaceFolder}", "${ALSO_MISSING}"]
    assert resolved["cwd"] == str(tmp_path)
    assert missing == ["NOT_SET", "input:apiKey", "workspaceFolder", "ALSO_MISSING"]


# ── masking ──────────────────────────────────────────────────────────


def test_masked_command_hides_every_credential_shape():
    argv = [
        "node",
        "server.js",
        "--token",
        FLAG_SECRET,
        f"--api-key={EQ_SECRET}",
        POSITIONAL_SECRET,
        VENDOR_SECRET,
        f"--id={ENV_SECRET}",
        f"--db=postgres://user:{USERINFO_SECRET}@db.example/x?sslkey={QUERY_SECRET}",
        "--verbose",
    ]
    shown = sources_mod.masked_command(argv, {"SERVICE_ID": ENV_SECRET})
    for secret in ALL_SECRETS:
        assert secret not in shown
    assert "server.js" in shown and "--verbose" in shown and "--token" in shown


def test_mask_url_hides_userinfo_query_and_key_shaped_path():
    shown = sources_mod.mask_url(
        f"https://me:{USERINFO_SECRET}@mcp.example.com/s/{PATH_SECRET}/sse?k={QUERY_SECRET}&debug"
    )
    assert shown == "https://***@mcp.example.com/s/***/sse?k=***&debug"


def test_server_identity_ignores_case_of_host_and_trailing_slash():
    identity = sources_mod.server_identity
    assert identity("https://Example.com/mcp/") == identity("https://example.com/mcp")
    assert identity("https://example.com/a") != identity("https://example.com/b")
    assert identity("npx -y pkg") == identity("npx  -y  pkg")
    assert identity("npx -y pkg") != identity("npx -y other")


def test_find_program(tmp_path, monkeypatch):
    exe = _program(tmp_path)
    assert sources_mod.find_program(exe) is not None
    assert sources_mod.find_program(str(tmp_path / "missing.exe")) is None
    bin_dir = os.path.dirname(exe)
    assert sources_mod.find_program(os.path.basename(exe), {"PATH": bin_dir}) is not None
    monkeypatch.setattr("utils.node_runtime.managed_node_bin_dir", lambda: None)
    assert sources_mod.find_program("npx", {"PATH": str(tmp_path / "nothing")}) is None


# ── routes ───────────────────────────────────────────────────────────


@pytest.fixture
def routes(machine, monkeypatch):
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(routes_mcp, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr("utils.node_runtime.managed_node_bin_dir", lambda: None)
    return routes_mcp


def _secret_config(tmp_path: Path) -> dict:
    exe = _program(tmp_path)
    return {
        "mcpServers": {
            "local": {
                "command": exe,
                "args": [
                    "--token",
                    FLAG_SECRET,
                    f"--api-key={EQ_SECRET}",
                    POSITIONAL_SECRET,
                    VENDOR_SECRET,
                    f"--id={ENV_SECRET}",
                    "--key",
                    "${IMPORT_TEST_TOKEN}",
                ],
                "env": {"SERVICE_ID": ENV_SECRET, "RESOLVED": "${env:IMPORT_TEST_TOKEN}"},
            },
            "remote": {
                "url": f"https://me:{USERINFO_SECRET}@mcp.example.com/s/{PATH_SECRET}/mcp?key={QUERY_SECRET}",
                "headers": {"Authorization": f"Bearer {HEADER_SECRET}"},
            },
        }
    }


def _assert_no_secrets(payload) -> None:
    text = json.dumps(payload.model_dump())
    for secret in ALL_SECRETS:
        assert secret not in text, f"secret leaked: {secret[:6]}..."


def test_list_import_sources_masks_every_secret(routes, machine, tmp_path, monkeypatch):
    monkeypatch.setenv("IMPORT_TEST_TOKEN", RESOLVED_SECRET)
    path = _write(_desktop(machine), _secret_config(tmp_path))
    response = routes.list_import_sources(current_subject = "u")
    _assert_no_secrets(response)
    (source,) = response.sources
    assert (source.id, source.app, source.path) == ("claude-desktop", "Claude Desktop", str(path))
    local, remote = source.servers
    assert (local.name, local.transport, local.importable, local.note) == ("local", "stdio", True, None)
    assert local.env_keys == ["SERVICE_ID", "RESOLVED"]
    assert "***" in local.target
    assert (remote.transport, remote.header_keys) == ("http", ["Authorization"])
    assert remote.target == "https://***@mcp.example.com/s/***/mcp?key=***"
    assert not local.already_added and not remote.already_added


def test_list_reports_duplicates_unreadable_files_and_switched_off_notes(
    routes, machine, tmp_path
):
    exe = _program(tmp_path)
    mcp_servers_db.create_server(id = "seed", display_name = "seed", url = "https://example.com/mcp")
    _write(
        _desktop(machine),
        {
            "mcpServers": {
                "same": {"url": "https://EXAMPLE.com/mcp/"},
                "off": {"command": exe, "disabled": True},
                "needs": {"command": exe, "env": {"K": "${env:IMPORT_TEST_UNSET}"}},
                "gone": {"command": str(tmp_path / "missing.exe")},
                "bad": {"command": exe, "url": "https://x.example"},
            }
        },
    )
    _write(machine / ".cursor" / "mcp.json", "{oops")
    response = routes.list_import_sources(current_subject = "u")
    desktop, cursor = response.sources
    servers = {server.name: server for server in desktop.servers}
    assert servers["same"].already_added is True
    assert "switched off in Claude Desktop" in servers["off"].note
    assert "IMPORT_TEST_UNSET is not set" in servers["needs"].note
    assert "Program not found" in servers["gone"].note
    assert servers["bad"].importable is False
    assert "both 'command' and 'url'" in servers["bad"].note
    assert cursor.error.startswith("Couldn't read this file")
    assert cursor.servers == []


def test_list_marks_local_programs_unimportable_while_stdio_is_off(
    routes, machine, tmp_path, monkeypatch
):
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: False)
    monkeypatch.setattr(routes, "stdio_mcp_disabled_reason", lambda: "Local commands are off.")
    _write(_desktop(machine), _secret_config(tmp_path))
    local, remote = routes.list_import_sources(current_subject = "u").sources[0].servers
    assert (local.importable, local.note) == (False, "Local commands are off.")
    assert remote.importable is True


def _apply(routes, source_id, names, **kwargs):
    from models.mcp_servers import McpImportSourceApplyRequest

    return asyncio.run(
        routes.apply_import_source(
            McpImportSourceApplyRequest(source_id = source_id, server_names = names),
            current_subject = "u",
            **kwargs,
        )
    )


def test_apply_imports_with_resolved_env_cwd_and_headers(routes, machine, tmp_path, monkeypatch):
    monkeypatch.setenv("IMPORT_TEST_TOKEN", RESOLVED_SECRET)
    config = _secret_config(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    config["mcpServers"]["local"]["cwd"] = str(workdir)
    _write(_desktop(machine), config)

    result = _apply(routes, "claude-desktop", ["local", "remote"])
    _assert_no_secrets(result)
    assert [(item.name, item.status, item.detail) for item in result.results] == [
        ("local", "added", None),
        ("remote", "added", None),
    ]
    rows = {row["display_name"]: row for row in mcp_servers_db.list_servers()}
    local = rows["local"]
    argv = parse_stdio_command(local["url"])
    assert argv[0] == config["mcpServers"]["local"]["command"]
    assert argv[-2:] == ["--key", RESOLVED_SECRET]
    assert json.loads(local["headers_json"]) == {"SERVICE_ID": ENV_SECRET, "RESOLVED": RESOLVED_SECRET}
    assert local["cwd"] == str(workdir)
    assert local["is_enabled"] == 1
    remote = rows["remote"]
    assert remote["url"].endswith(f"/mcp?key={QUERY_SECRET}")
    assert json.loads(remote["headers_json"]) == {"Authorization": f"Bearer {HEADER_SECRET}"}
    assert remote["is_enabled"] == 1 and remote["use_oauth"] == 0


def test_apply_switches_off_what_cannot_run_as_written(routes, machine, tmp_path, monkeypatch):
    exe = _program(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    _write(
        _desktop(machine),
        {
            "mcpServers": {
                "npx-missing": {"command": "npx", "args": ["-y", "@scope/pkg"]},
                "unresolved": {"command": exe, "env": {"TOKEN": "${env:IMPORT_TEST_UNSET}"}},
                "off-at-source": {"command": exe, "args": ["--x"], "disabled": True},
                "remote-unresolved": {
                    "url": "https://h.example/mcp",
                    "headers": {"Authorization": "Bearer ${input:token}"},
                },
            }
        },
    )
    result = _apply(
        routes, "claude-desktop", ["npx-missing", "unresolved", "off-at-source", "remote-unresolved"]
    )
    outcomes = {item.name: item for item in result.results}
    assert {item.status for item in result.results} == {"added_disabled"}
    assert "Program not found: npx" in outcomes["npx-missing"].detail
    assert "IMPORT_TEST_UNSET is not set" in outcomes["unresolved"].detail
    assert "switched off in Claude Desktop" in outcomes["off-at-source"].detail
    assert "input:token" in outcomes["remote-unresolved"].detail
    rows = {row["display_name"]: row for row in mcp_servers_db.list_servers()}
    assert all(row["is_enabled"] == 0 for row in rows.values())
    # The reference is kept for the user to replace, not swapped for an empty string.
    assert json.loads(rows["unresolved"]["headers_json"]) == {"TOKEN": "${env:IMPORT_TEST_UNSET}"}


def test_apply_skips_duplicates_and_reports_each_server(routes, machine, tmp_path):
    exe = _program(tmp_path)
    mcp_servers_db.create_server(id = "seed", display_name = "old", url = "https://example.com/mcp")
    _write(
        _desktop(machine),
        {
            "mcpServers": {
                "dup-url": {"url": "https://example.com/mcp/"},
                "first": {"command": exe, "args": ["--a"]},
                "second": {"command": exe, "args": ["--a"], "env": {"OTHER": "x"}},
                "broken": {"command": 42},
            }
        },
    )
    result = _apply(
        routes, "claude-desktop", ["dup-url", "first", "second", "broken", "vanished", "first"]
    )
    assert [(item.name, item.status) for item in result.results] == [
        ("dup-url", "duplicate"),
        ("first", "added"),
        ("second", "duplicate"),
        ("broken", "error"),
        ("vanished", "error"),
    ]
    assert result.results[0].detail == "Studio already has a server with this address."
    assert result.results[2].detail == "Studio already has a server with this command."
    assert result.results[3].detail == "'command' must be a string."
    assert "no longer lists" in result.results[4].detail
    assert sorted(row["display_name"] for row in mcp_servers_db.list_servers()) == ["first", "old"]
    # Re-importing is idempotent.
    again = _apply(routes, "claude-desktop", ["first"])
    assert again.results[0].status == "duplicate"


def test_apply_refuses_local_programs_while_stdio_is_off(routes, machine, tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: False)
    monkeypatch.setattr(routes, "stdio_mcp_disabled_reason", lambda: "Local commands are off.")
    _write(_desktop(machine), _secret_config(tmp_path))
    result = _apply(routes, "claude-desktop", ["local", "remote"])
    assert [(item.name, item.status, item.detail) for item in result.results] == [
        ("local", "error", "Local commands are off."),
        ("remote", "added", None),
    ]
    assert [row["display_name"] for row in mcp_servers_db.list_servers()] == ["remote"]


def test_apply_unknown_or_unreadable_source(routes, machine):
    with pytest.raises(HTTPException) as missing:
        _apply(routes, "cursor", ["x"])
    assert missing.value.status_code == 404
    _write(machine / ".cursor" / "mcp.json", "{oops")
    with pytest.raises(HTTPException) as broken:
        _apply(routes, "cursor", ["x"])
    assert broken.value.status_code == 400
    assert mcp_servers_db.list_servers() == []


@pytest.mark.parametrize(
    "caller", [{"via_api_key": True}, {"no_credential": True}], ids = ["api-key", "keyless"]
)
def test_api_keys_and_keyless_callers_are_refused(routes, machine, tmp_path, caller):
    _write(_desktop(machine), _secret_config(tmp_path))
    with pytest.raises(HTTPException) as listed:
        routes.list_import_sources(current_subject = "u", **caller)
    assert listed.value.status_code == 403
    with pytest.raises(HTTPException) as applied:
        _apply(routes, "claude-desktop", ["remote", "local"], **caller)
    assert applied.value.status_code == 403
    assert mcp_servers_db.list_servers() == []


def test_managed_accounts_are_refused_and_never_see_the_path(routes, machine, tmp_path):
    from utils.account_context import AccountContext, bind_account, reset_account

    _write(_desktop(machine), _secret_config(tmp_path))
    token = bind_account(AccountContext("acct-1", "member", "user"))
    try:
        with pytest.raises(HTTPException) as listed:
            routes.list_import_sources(current_subject = "member")
        with pytest.raises(HTTPException) as applied:
            _apply(routes, "claude-desktop", ["remote"])
    finally:
        reset_account(token)
    assert listed.value.status_code == applied.value.status_code == 403
    assert "installation owner" in listed.value.detail
    # The owner's own session does get the path.
    assert routes.list_import_sources(current_subject = "u").sources[0].path == str(_desktop(machine))
