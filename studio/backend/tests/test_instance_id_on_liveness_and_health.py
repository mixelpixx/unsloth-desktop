# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: /api/liveness and /api/health name the process behind the port, and agree on it.

studio_root_id is per install, so it is the same before and after a restart. The SPA's
connection monitor needs the other question answered: when a request that could not reach
the backend is followed by one that can, was that the same process coming back from a stall,
or a new one that has lost every model the old one had loaded? `instance_id` answers it. It
has to be the same on every reply from one process, or the monitor reports a restart on
every heartbeat, and the same on both routes, because liveness is the one the monitor polls
and health is the fallback for a launcher talking to an older backend.

No login on either: the monitor probes before a token exists, and on the sign-in page.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_INSTANCE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Both routes on a fresh app, against an isolated auth db and a settled hardware verdict."""
    from auth import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)

    import main as _main

    # About the id, not about startup: keep health off the detection wait and the payload
    # settled, as test_middleware.py's health fixture does.
    async def _settled(_budget):
        return True

    monkeypatch.setattr(_main, "_await_hardware_detection", _settled)
    monkeypatch.setattr(_main, "_hardware_snapshot", lambda: (False, None, None))
    app = FastAPI()
    app.add_api_route("/api/liveness", _main.liveness_check, methods = ["GET"])
    app.add_api_route("/api/health", _main.health_check, methods = ["GET"])
    return TestClient(app)


def test_liveness_carries_an_opaque_instance_id(client):
    body = client.get("/api/liveness").json()
    assert "instance_id" in body, (
        "/api/liveness dropped instance_id; the connection monitor can no longer tell a "
        "restarted backend from one that was briefly unreachable"
    )
    assert _INSTANCE_ID_RE.fullmatch(body["instance_id"]), body["instance_id"]


def test_the_id_is_stable_for_the_life_of_the_process(client):
    """A per-request id would read as a restart on every heartbeat."""
    ids = {client.get("/api/liveness").json()["instance_id"] for _ in range(3)}
    ids |= {client.get("/api/health").json()["instance_id"] for _ in range(3)}
    assert len(ids) == 1, f"one process served {len(ids)} instance ids: {sorted(ids)}"


def test_liveness_and_health_report_the_same_process(client):
    liveness = client.get("/api/liveness").json()
    health = client.get("/api/health").json()
    assert health["status"] == "healthy"
    assert "instance_id" in health, (
        "/api/health dropped instance_id; it is in lockstep with /api/liveness"
    )
    assert liveness["instance_id"] == health["instance_id"]


def test_the_id_needs_no_login(client):
    """The monitor probes on the sign-in page and before any token exists, so an id behind
    the bearer gate would be missing exactly when a restart has just logged nobody in."""
    health = client.get("/api/health", headers = {"Authorization": "Bearer not-a-real-token"})
    assert health.status_code == 200
    assert _INSTANCE_ID_RE.fullmatch(health.json()["instance_id"])
