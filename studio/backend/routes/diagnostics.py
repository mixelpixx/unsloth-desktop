# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``GET /api/diagnostics``: the environment report behind Settings > Logs > Diagnostics.

Gated like the logs routes it sits beside: the installation owner, from a signed-in Studio
window. An API key or keyless caller is refused, as is a managed account; the report names
the host's hardware, paths and configuration, which is operator material.

The shareable bundle is not a second archive builder: it is the logs export with
``?diagnostics=true`` (``routes/settings.py``), so the logs inside it are the same redacted
files "Download all logs" produces.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from routes.provider_credentials import require_ui_session

router = APIRouter()


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


def _require_ui_session(via_api_key: bool = Depends(authenticated_via_api_key)) -> None:
    require_ui_session(via_api_key)


def frontend_build_path(request: Request) -> Any:
    return getattr(request.app.state, "frontend_build_path", None)


@router.get("")
def get_diagnostics(
    request: Request,
    response: Response,
    current_subject: str = Depends(get_current_subject),
    _owner: None = Depends(_require_installation_owner),
    _ui_session: None = Depends(_require_ui_session),
) -> dict[str, Any]:
    """Every section of the report plus its Markdown rendering. Sync on purpose: the
    collectors shell out (nvidia-smi, git) and FastAPI runs a sync route in its threadpool."""
    from utils import diagnostics

    report = diagnostics.collect_diagnostics(frontend_build_path = frontend_build_path(request))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
    return {**report, "markdown": diagnostics.render_markdown(report)}
