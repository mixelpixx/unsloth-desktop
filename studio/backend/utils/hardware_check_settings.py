# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted Settings > Resources > Hardware check options.

One JSON object under ``hardware_check_settings``:

* ``auto_run`` -- run the check once in the background when there is no result for the
  hardware that is installed now. On by default: the check only measures.
* ``prefer_fast_link`` -- automatic GPU placement (llama.cpp loads, the training and
  Transformers auto-selector) prefers the card with the faster measured host link.
* ``avoid_tensor_split`` -- an automatic choice never runs llama.cpp tensor parallelism across
  a slow link or a pair without direct GPU-to-GPU access; an explicit one is warned about.
* ``warn_training_slow_link`` -- the training fit panel warns when a run would use a slow-link
  GPU.

The three behaviour options default OFF: nothing changes until the user switches one on or
presses "Apply recommended". Read on the load path, so reads are memoised briefly and a corrupt
value falls through to the defaults rather than failing a load, as ``vram_budget_settings``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

from utils.account_context import OWNER, run_as

HARDWARE_CHECK_SETTING_KEY = "hardware_check_settings"

AUTO_RUN = "auto_run"
PREFER_FAST_LINK = "prefer_fast_link"
AVOID_TENSOR_SPLIT = "avoid_tensor_split"
WARN_TRAINING_SLOW_LINK = "warn_training_slow_link"

# The options "Apply recommended" may switch on, in display order. Mirrored in hardware-check.ts.
OPTION_KEYS = (PREFER_FAST_LINK, AVOID_TENSOR_SPLIT, WARN_TRAINING_SLOW_LINK)

DEFAULTS: dict[str, bool] = {
    AUTO_RUN: True,
    PREFER_FAST_LINK: False,
    AVOID_TENSOR_SPLIT: False,
    WARN_TRAINING_SLOW_LINK: False,
}

_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, Any]] = {}
# Bumped on every write: a read that began before it must not cache its stale value.
_generation: dict[tuple[str, str], int] = {}
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
            # An unreadable DB must not fail a load; the defaults apply.
            return None
        with _cache_lock:
            if _generation.get(cache_key, 0) == generation:
                _cache[cache_key] = (time.monotonic(), stored)
                return stored
    return stored


def _invalidate(key: str) -> None:
    cache_key = (OWNER.account_id, key)
    with _cache_lock:
        _cache.pop(cache_key, None)
        _generation[cache_key] = _generation.get(cache_key, 0) + 1


def coerce_settings(value: Any) -> dict[str, bool]:
    """The stored object over the defaults; unknown keys and non-boolean values are dropped."""
    out = dict(DEFAULTS)
    if isinstance(value, Mapping):
        for key in DEFAULTS:
            item = value.get(key)
            if isinstance(item, bool):
                out[key] = item
    return out


def get_hardware_check_settings() -> dict[str, bool]:
    """Every setting, defaults filled in. Never raises."""
    return coerce_settings(_cached_setting(HARDWARE_CHECK_SETTING_KEY))


def setting_enabled(key: str) -> bool:
    return bool(get_hardware_check_settings().get(key, False))


def update_hardware_check_settings(patch: Mapping[str, Any]) -> dict[str, bool]:
    """Merge ``patch`` into the stored settings. Unknown keys or non-boolean values raise."""
    unknown = [key for key in patch if key not in DEFAULTS]
    if unknown:
        raise ValueError("Unknown hardware check setting: " + ", ".join(sorted(unknown)) + ".")
    bad = [key for key, value in patch.items() if not isinstance(value, bool)]
    if bad:
        raise ValueError("Hardware check settings must be true or false: " + ", ".join(sorted(bad)) + ".")
    from storage.studio_db import get_app_setting, upsert_app_settings

    current = coerce_settings(run_as(OWNER, get_app_setting, HARDWARE_CHECK_SETTING_KEY, None))
    current.update({key: bool(value) for key, value in patch.items()})
    run_as(OWNER, upsert_app_settings, {HARDWARE_CHECK_SETTING_KEY: current})
    _invalidate(HARDWARE_CHECK_SETTING_KEY)
    return get_hardware_check_settings()
