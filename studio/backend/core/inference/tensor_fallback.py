# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tensor-parallel -> layer-split auto-fallback for GGUF loads.

Kept in its own module (no FastAPI / httpx deps) so the orchestration can be
unit-tested with a fake loader, without a GPU or a running llama-server.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from core.inference.llama_server_args import (
    _effective_tensor_parallel,
    strip_split_mode_only,
)

logger = logging.getLogger(__name__)


async def load_with_tensor_fallback(
    attempt_load: Callable[[bool, Optional[list[str]]], Awaitable[bool]],
    *,
    requested_tensor: bool,
    extra_args: Optional[list[str]],
    label: str = "",
    cancelled: Optional[Callable[[], bool]] = None,
    tensor_engaged: Optional[Callable[[], Optional[bool]]] = None,
    is_gpu_memory_failure: Optional[Callable[[BaseException], bool]] = None,
) -> bool:
    """Run a GGUF load with the tensor-parallel -> layer-split auto-fallback.

    ``attempt_load(tensor_parallel, extra_args)`` performs one load and returns
    True on success; it *raises* on a hard crash (llama-server aborts on some
    archs / older builds), which is treated the same as a False return.

    Tensor mode can be requested by the toggle, by a ``--split-mode tensor`` in
    ``extra_args`` (an allowed shadow flag), or by an inherited
    ``LLAMA_ARG_SPLIT_MODE=tensor`` env (load_model engages it the same way), so
    the retry is keyed on whether tensor mode is actually engaged, and it forces
    ``--split-mode layer`` on the retry so neither leftover extras nor the
    inherited tensor env can relaunch the same failing tensor load. A non-tensor
    load keeps its original contract and propagates exceptions.

    ``cancelled()`` distinguishes a real tensor-start failure from a user
    cancellation: ``attempt_load`` also returns False when the load was
    cancelled, so without this the helper would restart a load the user just
    cancelled.

    ``tensor_engaged()`` reports whether the failed attempt actually LAUNCHED
    tensor parallel (True / False), or None when it cannot say (nothing was
    spawned). A request is not a launch: the planner downgrades tensor to layer
    split when the pooled budget cannot hold it, and retrying that relaunched an
    identical layer command while blaming tensor parallelism. False skips the
    retry and lets the first failure stand.

    ``is_gpu_memory_failure(exc)`` marks a GPU allocation failure. It only decides
    when ``tensor_engaged`` cannot: a tensor launch that ran out of GPU memory still
    retries, since layer split is a different placement and lets llama.cpp's fitter
    move layers to system RAM, which tensor mode cannot.
    """
    tensor_requested = _effective_tensor_parallel(extra_args, requested_tensor)
    failure: Optional[BaseException] = None
    try:
        success = await attempt_load(requested_tensor, extra_args)
    except Exception as exc:
        if not tensor_requested:
            raise
        logger.warning("Tensor-parallel load raised for '%s': %s", label, exc)
        success = False
        failure = exc

    if success or not tensor_requested:
        return success

    # The first attempt returned False because the user cancelled, not because tensor mode is unsupported -- do not
    # relaunch the cancelled load.
    if cancelled is not None and cancelled():
        return success

    engaged: Optional[bool] = None
    if tensor_engaged is not None:
        try:
            engaged = tensor_engaged()
        except Exception:
            engaged = None
    if engaged is False:
        # The relaunch would differ only by an explicit --split-mode layer, the default the
        # failed launch already ran with, so it fails the same way and doubles the wait.
        logger.warning(
            "Tensor-parallel was requested for '%s' but this load ran layer split, so "
            "there is no layer-split retry to make; reporting the failure as is",
            label,
        )
        if failure is not None:
            raise failure
        return success
    if engaged is None and failure is not None and is_gpu_memory_failure is not None:
        try:
            out_of_gpu_memory = bool(is_gpu_memory_failure(failure))
        except Exception:
            out_of_gpu_memory = False
        if out_of_gpu_memory:
            logger.warning(
                "Load of '%s' ran out of GPU memory; not retrying with layer split, "
                "which cannot free that memory",
                label,
            )
            raise failure

    logger.warning(
        "Tensor-parallel load failed for '%s'; retrying with layer split "
        "(this model may not support tensor parallelism)",
        label,
    )
    # Force --split-mode layer (CLI wins over env) so neither leftover extras nor an inherited
    # LLAMA_ARG_SPLIT_MODE=tensor can re-engage tensor and re-crash the retry; load_model and the child both honor the
    # explicit layer override.
    layer_extras = strip_split_mode_only(extra_args) or []
    return await attempt_load(False, [*layer_extras, "--split-mode", "layer"])
