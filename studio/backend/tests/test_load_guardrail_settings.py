# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The persisted load guardrail mode and its settings route.

Read on every /load, so the bar is that nothing stored or exported can make it raise or
hand back a mode the policy does not know: a corrupt value means "balanced", never "off".
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import utils.load_guardrail_settings as gs


@pytest.fixture
def store(monkeypatch):
    """An in-memory app_settings table behind the module's own read and write."""
    rows: dict[str, object] = {}
    monkeypatch.delenv(gs.LOAD_GUARDRAIL_ENV_VAR, raising = False)
    monkeypatch.setattr(gs, "_cached_setting", lambda key: rows.get(key))

    import storage.studio_db as studio_db

    monkeypatch.setattr(studio_db, "upsert_app_settings", lambda updates: rows.update(updates))
    return rows


@pytest.mark.parametrize("value", ["strict", "balanced", "relaxed", "off", " Strict ", "OFF"])
def test_known_modes_coerce(value):
    assert gs.coerce_mode(value) == value.strip().lower()


@pytest.mark.parametrize("value", [None, "", "loose", 1, True, [], {"mode": "off"}])
def test_unknown_modes_are_rejected(value):
    assert gs.coerce_mode(value) is None


def test_the_default_is_balanced(store):
    assert gs.get_load_guardrail_mode() == "balanced"
    assert gs.get_load_guardrail_state() == ("balanced", False)


def test_the_environment_is_a_startup_default(store, monkeypatch):
    monkeypatch.setenv(gs.LOAD_GUARDRAIL_ENV_VAR, "strict")
    assert gs.get_load_guardrail_state() == ("strict", False)


def test_a_stored_mode_wins_over_the_environment(store, monkeypatch):
    monkeypatch.setenv(gs.LOAD_GUARDRAIL_ENV_VAR, "strict")
    assert gs.set_load_guardrail_mode("off") == "off"
    assert gs.get_load_guardrail_state() == ("off", True)


def test_clearing_falls_back(store):
    gs.set_load_guardrail_mode("relaxed")
    assert gs.set_load_guardrail_mode(None) == "balanced"
    assert gs.get_load_guardrail_state() == ("balanced", False)


def test_a_corrupt_stored_value_reads_as_the_default(store):
    store[gs.LOAD_GUARDRAIL_SETTING_KEY] = "definitely-not-a-mode"
    assert gs.get_load_guardrail_mode() == "balanced"


def test_an_unknown_mode_is_refused_on_write(store):
    with pytest.raises(ValueError):
        gs.set_load_guardrail_mode("yolo")
    assert gs.LOAD_GUARDRAIL_SETTING_KEY not in store


def test_an_unreadable_database_does_not_fail_a_load(monkeypatch):
    monkeypatch.delenv(gs.LOAD_GUARDRAIL_ENV_VAR, raising = False)
    gs._invalidate(gs.LOAD_GUARDRAIL_SETTING_KEY)

    import storage.studio_db as studio_db

    def _boom(*_a, **_k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(studio_db, "get_app_setting", _boom)
    assert gs.get_load_guardrail_mode() == "balanced"


class TestRoute:
    def test_get_reports_the_mode_and_the_choices(self, store):
        import routes.settings as settings_route

        response = settings_route.get_load_guardrails(current_subject = "owner")
        assert response.mode == "balanced"
        assert response.is_stored is False
        assert response.default_mode == "balanced"
        assert response.modes == ["strict", "balanced", "relaxed", "off"]

    def test_put_stores_and_echoes(self, store):
        import routes.settings as settings_route

        response = settings_route.update_load_guardrails(
            settings_route.LoadGuardrailsPayload(mode = "strict"), current_subject = "owner"
        )
        assert (response.mode, response.is_stored) == ("strict", True)
        cleared = settings_route.update_load_guardrails(
            settings_route.LoadGuardrailsPayload(mode = None), current_subject = "owner"
        )
        assert (cleared.mode, cleared.is_stored) == ("balanced", False)

    def test_put_refuses_an_unknown_mode_with_400(self, store):
        import routes.settings as settings_route

        with pytest.raises(HTTPException) as excinfo:
            settings_route.update_load_guardrails(
                settings_route.LoadGuardrailsPayload(mode = "yolo"), current_subject = "owner"
            )
        assert excinfo.value.status_code == 400

    def test_the_route_is_owner_only(self):
        import routes.settings as settings_route

        paths = {route.path for route in settings_route._owner_settings_router.routes}
        assert "/load-guardrails" in paths
