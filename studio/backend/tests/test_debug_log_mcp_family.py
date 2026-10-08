# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local-program MCP servers' stderr logs (logs/mcp/<program>-<digest>.log) as a Settings > Logs
family: listed under the server's name, opened only by opaque id, read and exported through the same
redaction as every other log plus the server's own configured values, which have no shape a pattern
could find. The conftest _isolate_studio_home fixture points UNSLOTH_STUDIO_HOME at a tmp dir."""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.settings as settings_route
from core.inference import mcp_client
from storage import mcp_servers_db
from utils import debug_log_export, debug_log_sources

COMMAND = mcp_client.join_stdio_command(["npx", "-y", "@example/github-mcp", "--token", "argv-token-1234"])
ENV = {"GITHUB_PERSONAL_ACCESS_TOKEN": "plain-env-secret-5678"}
# What an older build left on disk: stderr written raw.
LEGACY_BODY = (
    "starting with token plain-env-secret-5678\n"
    "cli args --token argv-token-1234\n"
    "loaded hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345\n"
    "ERROR: upstream refused the request\n"
)
RAW_SECRETS = ("plain-env-secret-5678", "argv-token-1234", "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345")


def _home() -> Path:
    return Path(os.environ["UNSLOTH_STUDIO_HOME"])


def _seed(name: str, body: str = "hello\n") -> Path:
    directory = _home() / "logs" / "mcp"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / name
    path.write_text(body, encoding = "utf-8")
    return path


def _owned_log_name() -> str:
    return mcp_client._stdio_log_name(COMMAND, mcp_client.parse_stdio_command(COMMAND)[0])


def _save_server() -> None:
    mcp_servers_db.create_server(
        id = "gh", display_name = "GitHub tools", url = COMMAND, headers_json = json.dumps(ENV)
    )


def _mcp_sources() -> list:
    return [s for s in debug_log_sources.list_sources() if s.family == "mcp"]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[settings_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings_route._require_ui_session] = lambda: None
    return TestClient(app, raise_server_exceptions = False)


# ── listing ─────────────────────────────────────────────────────────────────────────────────────────


def test_a_saved_servers_log_is_listed_under_its_name():
    _save_server()
    owned = _seed(_owned_log_name())
    stray = _seed("server-0123456789ab.log")
    by_label = {source.label: source for source in _mcp_sources()}
    assert by_label[owned.name].display_name == "GitHub tools"
    # A command nothing maps any more (edited, deleted, another account's) falls back to its stem.
    assert by_label[stray.name].display_name == "server-0123456789ab"
    assert not any(source.is_current for source in by_label.values())
    assert all(source.id.startswith("mcp:") for source in by_label.values())


def test_other_families_carry_no_display_name():
    directory = _home() / "logs" / "llama-server"
    directory.mkdir(parents = True)
    (directory / "llama-1765000000-port-8080.log").write_text("x\n", encoding = "utf-8")
    sources = debug_log_sources.list_sources()
    assert [source.display_name for source in sources] == [None]


def test_an_unreadable_server_list_still_lists_the_files(monkeypatch):
    monkeypatch.setattr(
        mcp_client, "stdio_log_owners", lambda: (_ for _ in ()).throw(RuntimeError("db gone"))
    )
    path = _seed(_owned_log_name())
    assert [source.display_name for source in _mcp_sources()] == [path.stem]


def test_the_newest_mcp_logs_are_listed_whatever_their_names():
    """MCP log names carry no time, so the name presort the other families use would keep the 30
    alphabetically LAST files and drop a fresh failure from the picker."""
    now = time.time()
    for index in range(40):
        path = _seed(f"zz-{index:02d}.log")
        os.utime(path, (now - 1000 - index, now - 1000 - index))
    fresh = _seed("aa-fresh.log")
    os.utime(fresh, (now, now))
    labels = [source.label for source in _mcp_sources()]
    assert labels[0] == "aa-fresh.log"
    assert len(labels) == debug_log_sources.MAX_SOURCES_PER_FAMILY


# ── allowlisting ────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hostile",
    [
        "mcp:../../../../etc/passwd",
        "mcp:..\\..\\studio.db",
        "mcp:" + "0" * 16,
        "mcp:" + "f" * 15,
        "mcp",
        "mcp:",
    ],
)
def test_a_hostile_mcp_id_resolves_to_nothing(hostile):
    _seed("server-0123456789ab.log")
    assert debug_log_sources.resolve_source_id(hostile) is None


def test_a_crafted_mcp_id_is_a_404_not_a_read(client):
    _seed("server-0123456789ab.log")
    response = client.get("/api/settings/debug/logs", params = {"source": "mcp:../../../studio.db"})
    assert response.status_code == 404


def test_a_symlink_out_of_the_mcp_dir_is_not_listed(tmp_path):
    secret = tmp_path / "id_rsa.log"
    secret.write_text("PRIVATE KEY\n", encoding = "utf-8")
    directory = _home() / "logs" / "mcp"
    directory.mkdir(parents = True)
    try:
        (directory / "evil.log").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this host")
    assert all("id_rsa" not in source.realpath for source in debug_log_sources.list_sources())


def test_only_dot_log_files_are_offered():
    _seed("server-0123456789ab.log")
    (_home() / "logs" / "mcp" / "notes.txt").write_text("not a log\n", encoding = "utf-8")
    assert [source.label for source in _mcp_sources()] == ["server-0123456789ab.log"]


# ── reads and export ────────────────────────────────────────────────────────────────────────────────


def test_a_read_masks_the_servers_own_values_and_credential_shapes(client):
    _save_server()
    _seed(_owned_log_name(), LEGACY_BODY)
    sources = client.get("/api/settings/debug/logs/sources").json()["sources"]
    source = next(source for source in sources if source["family"] == "mcp")
    assert source["display_name"] == "GitHub tools"
    body = client.get("/api/settings/debug/logs", params = {"source": source["id"]}).json()
    text = "\n".join(body["lines"])
    for secret in RAW_SECRETS:
        assert secret not in text
    assert "starting with token ***" in text
    assert "loaded hf_<redacted>" in text
    assert "ERROR: upstream refused the request" in text


def test_an_api_key_session_cannot_read_mcp_logs():
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[settings_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings_route.authenticated_via_api_key] = lambda: True
    api_client = TestClient(app, raise_server_exceptions = False)
    _seed("server-0123456789ab.log")
    assert api_client.get("/api/settings/debug/logs/sources").status_code == 403
    assert api_client.get("/api/settings/debug/logs/export").status_code == 403


def test_the_export_carries_mcp_logs_masked():
    _save_server()
    owned = _seed(_owned_log_name(), LEGACY_BODY)
    with debug_log_export.build_log_archive() as archive:
        with zipfile.ZipFile(io.BytesIO(archive.read())) as bundle:
            member = f"mcp/{owned.name}"
            assert member in bundle.namelist()
            text = bundle.read(member).decode("utf-8")
    for secret in RAW_SECRETS:
        assert secret not in text
    assert "starting with token ***" in text
    assert "ERROR: upstream refused the request" in text


def test_the_export_stays_within_its_byte_budget_with_mcp_logs(monkeypatch):
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 64)
    _seed("aa-0123456789ab.log", "x" * 50 + "\n")
    _seed("bb-0123456789ab.log", "y" * 50 + "\n")
    with debug_log_export.build_log_archive() as archive:
        with zipfile.ZipFile(io.BytesIO(archive.read())) as bundle:
            warnings = bundle.read(debug_log_export.WARNINGS_MEMBER).decode("utf-8")
    assert "export size budget reached" in warnings
