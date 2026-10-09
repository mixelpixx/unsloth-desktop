# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Logs > Diagnostics: ``GET /api/diagnostics`` and the diagnostics bundle.

The report is made to be handed to someone else, so what has to hold whatever the host looks
like: only the installation owner in a signed-in window can read it, no credential reaches it
(environment values, MCP commands, URLs, headers and env), and one section that breaks or hangs
costs that section only. The conftest _isolate_studio_home fixture points UNSLOTH_STUDIO_HOME at
a tmp dir, so the MCP rows and logs seeded here never touch a real install.
"""

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

import routes.diagnostics as diagnostics_route
import routes.settings as settings_route
from core.inference import mcp_client
from storage import mcp_servers_db
from utils import diagnostics
from utils.account_context import AccountContext, bind_account

HF_SECRET = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"
PLAIN_SECRET = "plain-env-secret-5678"
ARGV_SECRET = "argv-token-1234"
HEADER_SECRET = "header-bearer-9f8e7d6c5b4a"
URL_PASSWORD = "url-password-4242"
ENV_KEY_SECRET = "zz9-plural-z-alpha-0042"


def _fast_collectors(**overrides):
    collectors = {
        "studio": lambda: {"studio_version": "dev", "unsloth_version": "2026.1.1"},
        "os": lambda: {"system": "Windows", "name": "Windows 11", "memory": {"total_bytes": 2**34}},
        "gpus": lambda: {"driver_version": "999.1", "devices": []},
        "python": lambda: {"version": "3.12.0", "packages": {"torch": "2.9.0"}, "torch": {"cuda": "13.0"}},
        "llama_cpp": lambda: {"version": "b1", "update": {"state": "not_checked"}},
        "storage": lambda: {"locations": [], "hf_cache_size": {"state": "computing"}},
        "hardware_check": lambda: {"gpus": [], "findings": [], "settings": {}},
        "models": lambda: {"models": []},
        "mcp": lambda: {"servers": []},
        "environment": lambda: {"variables": []},
    }
    collectors.update(overrides)
    return collectors


@pytest.fixture
def stub_collectors(monkeypatch):
    monkeypatch.setattr(diagnostics, "default_collectors", lambda frontend_build_path = None: _fast_collectors())


def _diagnostics_app(overrides: dict | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(diagnostics_route.router, prefix = "/api/diagnostics")
    app.dependency_overrides[diagnostics_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[diagnostics_route._require_ui_session] = lambda: None
    app.dependency_overrides.update(overrides or {})
    return TestClient(app, raise_server_exceptions = False)


def _settings_app(overrides: dict | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[settings_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings_route._require_ui_session] = lambda: None
    app.dependency_overrides.update(overrides or {})
    return TestClient(app, raise_server_exceptions = False)


def _seed_server_log(body: str = "hello\n") -> Path:
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "server"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / f"server-20260813-120000-pid{os.getpid()}.log"
    path.write_text(body, encoding = "utf-8", newline = "")
    return path


def _seed_mcp_servers() -> None:
    command = mcp_client.join_stdio_command(
        ["npx", "-y", "@example/github-mcp", "--token", ARGV_SECRET]
    )
    mcp_servers_db.create_server(
        id = "gh",
        display_name = "GitHub tools",
        url = command,
        headers_json = json.dumps({"GITHUB_PERSONAL_ACCESS_TOKEN": PLAIN_SECRET}),
        process_mode = "shared",
    )
    mcp_servers_db.create_server(
        id = "remote",
        display_name = "Remote search",
        url = f"https://searcher:{URL_PASSWORD}@mcp.example.com/sse?api_key={HF_SECRET}",
        headers_json = json.dumps({"Authorization": f"Bearer {HEADER_SECRET}"}),
    )


ALL_SECRETS = (HF_SECRET, PLAIN_SECRET, ARGV_SECRET, HEADER_SECRET, URL_PASSWORD, ENV_KEY_SECRET)


# ── who may read it ─────────────────────────────────────────────────────────────────────────────


def test_the_owner_in_a_ui_session_gets_every_section_and_the_markdown(stub_collectors):
    response = _diagnostics_app().get("/api/diagnostics")
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == diagnostics.SCHEMA_VERSION
    assert list(body["sections"]) == list(diagnostics.SECTION_ORDER)
    assert all(section["status"] == "ok" for section in body["sections"].values())
    assert body["markdown"].startswith("## Unsloth Studio diagnostics")
    assert "no-store" in response.headers.get("cache-control", "")


def test_an_api_key_cannot_read_the_report(stub_collectors):
    app = FastAPI()
    app.include_router(diagnostics_route.router, prefix = "/api/diagnostics")
    app.dependency_overrides[diagnostics_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[diagnostics_route.authenticated_via_api_key] = lambda: True
    response = TestClient(app, raise_server_exceptions = False).get("/api/diagnostics")
    assert response.status_code == 403


def test_a_managed_account_cannot_read_the_report(stub_collectors):
    async def managed_subject() -> str:
        # Bound in an async dependency, so the owner check that follows sees it.
        bind_account(AccountContext("alice", "alice", "user"))
        return "alice"

    client = _diagnostics_app({diagnostics_route.get_current_subject: managed_subject})
    assert client.get("/api/diagnostics").status_code == 403


def test_the_route_carries_the_owner_and_ui_session_guards():
    route = next(r for r in diagnostics_route.router.routes if getattr(r, "path", None) == "")
    names = {getattr(dep.call, "__name__", "") for dep in route.dependant.dependencies}
    assert {"_require_installation_owner", "_require_ui_session"} <= names


# ── what may appear in it ───────────────────────────────────────────────────────────────────────


def test_secret_looking_environment_values_are_redacted():
    entries = diagnostics.environment_entries(
        {
            "HF_TOKEN": HF_SECRET,
            "UNSLOTH_SERVICE_KEY": ENV_KEY_SECRET,
            "HTTPS_PROXY": f"http://user:{URL_PASSWORD}@proxy:8080",
            "UNSLOTH_EXTRA_HEADER": f"Authorization: Bearer {HEADER_SECRET}",
            "UNSLOTH_WEBHOOK": f"https://hooks.example.com/x?token={PLAIN_SECRET}",
            "HF_HOME": r"D:\cache\huggingface",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "UNSLOTH_DISABLE_AUTH": "1",
        }
    )
    by_name = {entry["name"]: entry for entry in entries}
    for name in ("HF_TOKEN", "UNSLOTH_SERVICE_KEY", "HTTPS_PROXY", "UNSLOTH_EXTRA_HEADER", "UNSLOTH_WEBHOOK"):
        assert by_name[name]["redacted"] is True, name
        assert by_name[name]["value"] == "<redacted>"
    # Useful values survive, and a switch is not a credential.
    assert by_name["HF_HOME"]["value"] == r"D:\cache\huggingface"
    assert by_name["CUDA_VISIBLE_DEVICES"]["value"] == "0,1"
    assert by_name["UNSLOTH_DISABLE_AUTH"] == {"name": "UNSLOTH_DISABLE_AUTH", "value": "1", "redacted": False}
    text = json.dumps(entries)
    for secret in (HF_SECRET, ENV_KEY_SECRET, URL_PASSWORD, HEADER_SECRET, PLAIN_SECRET):
        assert secret not in text


def test_only_allowlisted_variables_are_listed():
    entries = diagnostics.environment_entries(
        {"PATH": "/usr/bin", "USERNAME": "someone", "AWS_SECRET_ACCESS_KEY": "x" * 30, "TORCH_HOME": "/t"}
    )
    assert [entry["name"] for entry in entries] == ["TORCH_HOME"]


def test_mcp_servers_are_summarised_without_commands_urls_headers_or_env():
    _seed_mcp_servers()
    data = diagnostics.collect_mcp()
    text = json.dumps(data)
    for secret in (HF_SECRET, PLAIN_SECRET, ARGV_SECRET, HEADER_SECRET, URL_PASSWORD):
        assert secret not in text, secret
    for needle in ("npx", "github-mcp", "mcp.example.com", "GITHUB_PERSONAL_ACCESS_TOKEN"):
        assert needle not in text, needle
    by_name = {server["name"]: server for server in data["servers"]}
    assert set(by_name) == {"GitHub tools", "Remote search"}
    assert by_name["GitHub tools"]["transport"] == "local"
    assert by_name["GitHub tools"]["process_mode"] == "shared"
    assert by_name["GitHub tools"]["state"] == "stopped"
    assert by_name["Remote search"]["transport"] == "remote"
    assert set(by_name["Remote search"]) == {
        "name", "builtin", "transport", "enabled", "oauth", "process_mode", "state"
    }


def test_the_real_report_carries_no_secret_from_env_or_mcp(monkeypatch):
    """Every real collector, end to end, with credentials planted where they live on a real host."""
    _seed_mcp_servers()
    monkeypatch.setenv("HF_TOKEN", HF_SECRET)
    monkeypatch.setenv("UNSLOTH_SERVICE_KEY", ENV_KEY_SECRET)
    monkeypatch.setenv("UNSLOTH_TEST_WEBHOOK", f"https://u:{URL_PASSWORD}@hooks.example.com/")
    # A walk of a real model cache has no place in a unit test.
    monkeypatch.setattr(diagnostics, "hf_cache_size", lambda path, **_: {"state": "computing"})
    response = _diagnostics_app().get("/api/diagnostics")
    assert response.status_code == 200
    for secret in ALL_SECRETS:
        assert secret not in response.text, secret
    body = response.json()
    names = [entry["name"] for entry in body["sections"]["environment"]["data"]["variables"]]
    assert {"HF_TOKEN", "UNSLOTH_SERVICE_KEY", "UNSLOTH_TEST_WEBHOOK"} <= set(names)
    assert body["sections"]["mcp"]["status"] == "ok"


def test_the_home_directory_is_written_as_a_tilde():
    import re

    pattern = re.compile(r"C:[\\/]+Users[\\/]+tester(?=[\\/]|$|[^A-Za-z0-9_.\-])", re.IGNORECASE)
    cleaned = diagnostics.sanitize(
        {"a": r"C:\Users\tester\AppData\Local\Temp", "b": r"C:\Users\testerOther\x", "c": [f"key {HF_SECRET}"]},
        _pattern = pattern,
    )
    assert cleaned["a"] == r"~\AppData\Local\Temp"
    assert cleaned["b"] == r"C:\Users\testerOther\x"
    assert HF_SECRET not in cleaned["c"][0]


# ── failing soft ────────────────────────────────────────────────────────────────────────────────


def test_a_failing_section_is_reported_and_the_rest_still_arrive():
    def broken():
        raise RuntimeError(f"driver exploded with {HF_SECRET}")

    report = diagnostics.collect_diagnostics(collectors = _fast_collectors(gpus = broken))
    gpus = report["sections"]["gpus"]
    assert gpus["status"] == "unavailable"
    assert gpus["reason"].startswith("RuntimeError")
    assert HF_SECRET not in gpus["reason"]
    assert report["sections"]["os"]["status"] == "ok"
    assert "_Unavailable: RuntimeError" in diagnostics.render_markdown(report)


def test_a_hung_section_times_out_without_holding_the_report():
    def hung():
        time.sleep(3)
        return {}

    started = time.monotonic()
    report = diagnostics.collect_diagnostics(
        collectors = _fast_collectors(storage = hung), deadline_s = 0.5
    )
    assert time.monotonic() - started < 2.5
    assert report["sections"]["storage"] == {"status": "unavailable", "reason": "timed out"}
    assert report["sections"]["studio"]["status"] == "ok"


def test_the_route_answers_when_a_real_section_breaks(monkeypatch):
    def broken():
        raise OSError("nvidia-smi went away")

    monkeypatch.setattr(diagnostics, "collect_gpus", broken)
    monkeypatch.setattr(diagnostics, "hf_cache_size", lambda path, **_: {"state": "computing"})
    response = _diagnostics_app().get("/api/diagnostics")
    assert response.status_code == 200
    assert response.json()["sections"]["gpus"]["status"] == "unavailable"


def test_the_llama_update_state_never_fetches(monkeypatch):
    from utils import llama_cpp_freshness as freshness

    def no_network(*_args, **_kwargs):
        raise AssertionError("diagnostics must not fetch release data")

    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", no_network)
    monkeypatch.setattr(freshness, "_load_disk_cache", lambda repo: None)
    monkeypatch.setattr(freshness, "_release_memo", {})
    marker = {"tag": "b100", "release_tag": "b100", "published_repo": "unslothai/llama.cpp"}
    assert diagnostics.llama_update_state(None, marker)["state"] == "not_checked"
    freshness._release_memo["unslothai/llama.cpp"] = (time.time(), "b200")
    state = diagnostics.llama_update_state(None, marker)
    assert state["state"] == "available" and state["latest"] == "b200"
    freshness._release_memo["unslothai/llama.cpp"] = (time.time(), "b100")
    assert diagnostics.llama_update_state(None, marker)["state"] == "up_to_date"


def test_nvidia_smi_rows_and_the_cuda_version_parse():
    rows = diagnostics.parse_gpu_rows(
        "0, 00000000:01:00.0, 616.92, 8.6, 24576, 3987, 20336, 4, 4, 16, 16, NVIDIA GeForce RTX 3090\n"
        "1, 00000000:02:00.0, 616.92, [N/A], 24576, 3316, 21007, [N/A], 4, 1, 16, Odd, Name, GPU\n",
        diagnostics._GPU_FIELDS,
    )
    assert [row["name"] for row in rows] == ["NVIDIA GeForce RTX 3090", "Odd, Name, GPU"]
    assert rows[0]["memory_total_bytes"] == 24576 * 1024 * 1024
    assert rows[0]["compute_capability"] == "8.6"
    assert rows[1]["compute_capability"] is None
    assert rows[1]["pcie_gen_current"] is None and rows[1]["pcie_width_current"] == 1
    assert diagnostics.parse_cuda_version("CUDA Version : 13.4 [Deprecated]\nCUDA UMD Version : 13.5\n") == "13.5"
    assert diagnostics.parse_cuda_version("Driver Version: 550.1   CUDA Version  : 12.4\n") is None
    assert diagnostics.parse_cuda_version("CUDA Version                : 12.4\n") == "12.4"


# ── the bundle ──────────────────────────────────────────────────────────────────────────────────


def _zip_members(response) -> dict[str, bytes]:
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.testzip() is None
    return {name: archive.read(name) for name in archive.namelist()}


def test_the_bundle_holds_the_report_and_the_redacted_logs(stub_collectors):
    _seed_server_log(f"loading with {HF_SECRET}\nfine\n")
    response = _settings_app().get("/api/settings/debug/logs/export", params = {"diagnostics": "true"})
    assert response.status_code == 200
    assert "unsloth-diagnostics-" in response.headers["content-disposition"]
    members = _zip_members(response)
    assert diagnostics.BUNDLE_JSON_MEMBER in members
    assert diagnostics.BUNDLE_MARKDOWN_MEMBER in members
    report = json.loads(members[diagnostics.BUNDLE_JSON_MEMBER])
    assert list(report["sections"]) == list(diagnostics.SECTION_ORDER)
    assert members[diagnostics.BUNDLE_MARKDOWN_MEMBER].decode().startswith("## Unsloth Studio diagnostics")
    logs = [name for name in members if name.startswith("server/")]
    assert logs, members.keys()
    assert all(HF_SECRET.encode() not in body for body in members.values())


def test_the_plain_log_export_is_unchanged(stub_collectors):
    _seed_server_log()
    response = _settings_app().get("/api/settings/debug/logs/export")
    assert response.status_code == 200
    assert "unsloth-logs-" in response.headers["content-disposition"]
    assert not any(name.startswith("diagnostics/") for name in _zip_members(response))


def test_an_api_key_cannot_download_the_bundle(stub_collectors):
    _seed_server_log()
    client = _settings_app()
    client.app.dependency_overrides.pop(settings_route._require_ui_session)
    client.app.dependency_overrides[settings_route.authenticated_via_api_key] = lambda: True
    response = client.get("/api/settings/debug/logs/export", params = {"diagnostics": "true"})
    assert response.status_code == 403
