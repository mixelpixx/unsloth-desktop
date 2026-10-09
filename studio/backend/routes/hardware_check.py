# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``/api/hardware-check``: Settings > Resources > Hardware check.

* ``GET  ""``                  -- the last result with its findings, whether it is current, the
                                  options and the run state (polled while a run is going).
* ``POST "/run"``              -- start a run; ``{"started": false, "reason": ...}`` while a
                                  training run or a model load is going on.
* ``PUT  "/settings"``         -- switch an option or the automatic run.
* ``POST "/apply-recommended"``-- switch on exactly the options the last result recommends.

Gated like Diagnostics: the installation owner, from a signed-in Studio window. The options
change how every account's loads are placed, and the result names the host's hardware and
paths, so neither an API key nor a managed account may read or change them.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from routes.provider_credentials import require_ui_session

router = APIRouter()


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


def _require_ui_session(via_api_key: bool = Depends(authenticated_via_api_key)) -> None:
    require_ui_session(via_api_key)


class HardwareCheckSettingsPayload(BaseModel):
    auto_run: Optional[bool] = None
    prefer_fast_link: Optional[bool] = None
    avoid_tensor_split: Optional[bool] = None
    warn_training_slow_link: Optional[bool] = None


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"


@router.get("")
def get_hardware_check(
    response: Response,
    current_subject: str = Depends(get_current_subject),
    _owner: None = Depends(_require_installation_owner),
    _ui_session: None = Depends(_require_ui_session),
) -> dict[str, Any]:
    """Sync on purpose: the "is it current" answer reads nvidia-smi (cached), off the loop."""
    from utils.hardware import hardware_check

    _no_store(response)
    return hardware_check.status_payload()


@router.post("/run")
def run_hardware_check(
    response: Response,
    current_subject: str = Depends(get_current_subject),
    _owner: None = Depends(_require_installation_owner),
    _ui_session: None = Depends(_require_ui_session),
) -> dict[str, Any]:
    from utils.hardware import hardware_check

    _no_store(response)
    outcome = hardware_check.start_run(hardware_check.TRIGGER_MANUAL)
    return {**outcome, "status": hardware_check.status_payload()}


@router.put("/settings")
def update_hardware_check_settings(
    payload: HardwareCheckSettingsPayload,
    response: Response,
    current_subject: str = Depends(get_current_subject),
    _owner: None = Depends(_require_installation_owner),
    _ui_session: None = Depends(_require_ui_session),
) -> dict[str, Any]:
    from utils import hardware_check_settings
    from utils.hardware import hardware_check

    patch = {key: value for key, value in payload.model_dump().items() if value is not None}
    if not patch:
        raise HTTPException(status_code = 400, detail = "Nothing to change.")
    try:
        hardware_check_settings.update_hardware_check_settings(patch)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    hardware_check._invalidate_preference()
    _no_store(response)
    return hardware_check.status_payload()


@router.post("/apply-recommended")
def apply_recommended(
    response: Response,
    current_subject: str = Depends(get_current_subject),
    _owner: None = Depends(_require_installation_owner),
    _ui_session: None = Depends(_require_ui_session),
) -> dict[str, Any]:
    """Switch on what the last result recommends; options it does not recommend are left as the
    user set them. 409 without a result for the hardware installed now."""
    from utils import hardware_check_settings
    from utils.hardware import hardware_check

    result = hardware_check.load_result()
    if not result or hardware_check.result_is_current(result) is False:
        raise HTTPException(
            status_code = 409,
            detail = "Run the hardware check first: there is no result for the GPUs installed now.",
        )
    recommended = hardware_check.analyze(result)["recommended"]
    patch = {key: True for key in hardware_check_settings.OPTION_KEYS if recommended.get(key)}
    if patch:
        hardware_check_settings.update_hardware_check_settings(patch)
        hardware_check._invalidate_preference()
    _no_store(response)
    return {"applied": sorted(patch), "status": hardware_check.status_payload()}
