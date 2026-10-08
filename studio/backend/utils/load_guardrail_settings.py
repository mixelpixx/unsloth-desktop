# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted load guardrail mode: which memory verdicts stop a load for an explicit "Load anyway".

``/load`` prices every GGUF load against the memory free right now
(``core.inference.load_verdict``) and, depending on this mode, refuses with 409 until the
caller retries with ``allow_memory_overcommit``:

* ``strict``   -- confirm ``likely_too_large`` AND ``disk_streaming``, the load that works
                  but runs from disk at a fraction of the speed.
* ``balanced`` -- the default. Confirm only ``likely_too_large``, the load that will crash.
* ``relaxed``  -- confirm only when the KV cache and buffers fit nowhere at all. A forced
                  GPU split that overflows is then the user's call, unasked.
* ``off``      -- never ask. The verdict is still computed and logged.

Stored like the VRAM budget: a stored value wins, ``UNSLOTH_LOAD_GUARDRAILS`` is a standalone
startup default, the constant is the last resort. Read on the load path, so a corrupt value
falls through to the default rather than failing a load.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

from utils.account_context import OWNER, run_as

LOAD_GUARDRAIL_SETTING_KEY = "load_guardrail_mode"

LOAD_GUARDRAIL_ENV_VAR = "UNSLOTH_LOAD_GUARDRAILS"

MODE_STRICT = "strict"
MODE_BALANCED = "balanced"
MODE_RELAXED = "relaxed"
MODE_OFF = "off"

# Display order, strictest first. Mirrored in load-guardrails.ts.
LOAD_GUARDRAIL_MODES = (MODE_STRICT, MODE_BALANCED, MODE_RELAXED, MODE_OFF)
LOAD_GUARDRAIL_DEFAULT = MODE_BALANCED

# Read on the load path, so memo briefly to spare SQLite, as vram_budget_settings.
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, Any]] = {}
# Bumped on every write: a read that began before it must not cache its stale value, or the new
# mode appears to revert for the rest of the TTL.
_generation: dict[tuple[str, str], int] = {}

# Retries converge; the bound only stops a write storm spinning here forever.
_MAX_REREADS = 3


def _cached_setting(key: str) -> Any:
    cache_key = (OWNER.account_id, key)
    stored = None
    for _attempt in range(_MAX_REREADS):
        with _cache_lock:
            hit = _cache.get(cache_key)
            if hit is not None and time.monotonic() - hit[0] < _CACHE_TTL_S:
                return hit[1]
            generation = _generation.get(cache_key, 0)
        try:
            from storage.studio_db import get_app_setting
            stored = run_as(OWNER, get_app_setting, key, None)
        except Exception:
            # An unreadable DB must not fail a load; fall back to the default.
            return None
        with _cache_lock:
            if _generation.get(cache_key, 0) == generation:
                _cache[cache_key] = (time.monotonic(), stored)
                return stored
        # A write landed mid-read, so `stored` predates it and must not be cached.
    return stored


def _invalidate(key: str) -> None:
    cache_key = (OWNER.account_id, key)
    with _cache_lock:
        _cache.pop(cache_key, None)
        _generation[cache_key] = _generation.get(cache_key, 0) + 1


def coerce_mode(value: Any) -> Optional[str]:
    """A known mode, else None. The stored JSON string and the raw environment value take the
    same path, so a value can never be legal in one and not the other."""
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    return mode if mode in LOAD_GUARDRAIL_MODES else None


def get_load_guardrail_mode() -> str:
    """The active mode. Never raises and never returns an unknown mode."""
    stored = coerce_mode(_cached_setting(LOAD_GUARDRAIL_SETTING_KEY))
    if stored is not None:
        return stored
    from_env = coerce_mode(os.environ.get(LOAD_GUARDRAIL_ENV_VAR))
    if from_env is not None:
        return from_env
    return LOAD_GUARDRAIL_DEFAULT


def get_load_guardrail_state() -> tuple[str, bool]:
    """``(mode, is_stored)`` for the settings route, as ``get_vram_budget_state``."""
    stored = coerce_mode(_cached_setting(LOAD_GUARDRAIL_SETTING_KEY))
    if stored is not None:
        return stored, True
    return get_load_guardrail_mode(), False


def set_load_guardrail_mode(mode: Any = None) -> str:
    """Store a mode, or clear it with ``None`` so env/default applies again."""
    from storage.studio_db import upsert_app_settings

    if mode is None:
        upsert_app_settings({LOAD_GUARDRAIL_SETTING_KEY: None})
        _invalidate(LOAD_GUARDRAIL_SETTING_KEY)
        return get_load_guardrail_mode()

    parsed = coerce_mode(mode)
    if parsed is None:
        raise ValueError(
            "Load guardrails must be one of: " + ", ".join(LOAD_GUARDRAIL_MODES) + "."
        )
    upsert_app_settings({LOAD_GUARDRAIL_SETTING_KEY: parsed})
    _invalidate(LOAD_GUARDRAIL_SETTING_KEY)
    return get_load_guardrail_mode()


def verdict_needs_confirmation(mode: str, level: str, reason: str) -> bool:
    """Whether ``mode`` stops a load with this verdict for an explicit "Load anyway".

    Takes the level and reason as strings so this module stays free of the inference
    package; ``core.inference.load_verdict`` defines them.
    """
    mode = coerce_mode(mode) or LOAD_GUARDRAIL_DEFAULT
    if mode == MODE_OFF:
        return False
    if mode == MODE_RELAXED:
        return level == "likely_too_large" and reason == "runtime_overflow"
    if mode == MODE_STRICT:
        return level in ("likely_too_large", "disk_streaming")
    return level == "likely_too_large"
