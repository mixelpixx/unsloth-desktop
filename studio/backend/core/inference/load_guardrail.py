# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The measuring half of the load guardrail: what is free right now, and where it goes.

``load_verdict`` decides; this gathers what it decides from, so the route's two callers
(``/estimate-memory`` for the panel headline, ``/load`` for the 409) read memory the same
way. Everything here fails soft: a probe that cannot answer yields ``None`` or an empty
list, which the verdict turns into ``unknown`` and never into a refusal.

Free memory comes from ``LlamaCppBackend._get_gpu_memory`` -- the very nvidia-smi /
amd-smi / Vulkan reading the loader's own placement and ``--fit-target`` margin are built
on. nvidia-smi's free is system-wide, so another program's memory (the LM Studio case) is
already out of it; the WDDM hidden-usage correction exists only because llama-server's OWN
probe cannot see that, and it is not needed here. Studio's resident model is the other
way round: it is IN the used figure but the load tears it down first, so its share is
credited back -- measured where llama-server logged it, flagged as uncertain where not.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

from loggers import get_logger

from core.inference.load_verdict import (
    PLACEMENT_AUTO,
    PLACEMENT_CPU,
    PLACEMENT_FIXED,
    GpuMemory,
    LoadVerdict,
    VerdictInputs,
    compute_load_verdict,
)

logger = get_logger(__name__)

__all__ = [
    "available_ram_bytes",
    "classify_placement",
    "describe_other_gpu_holders",
    "measure_gpu_memory",
    "resident_llama_gpu_bytes",
    "verdict_from_breakdown",
]

_MIB = 1024 * 1024

# Every device buffer llama-server sizes at startup, not only the weights: the whole child
# goes away with the swap, so its KV cache and compute buffers come back too. "CUDA_Host"
# is pinned HOST memory and does not match (the device id must follow the backend name).
_GPU_BUFFER_RE = re.compile(
    r"\b(?:CUDA|ROCm|ROCM|HIP|SYCL|Vulkan)(\d+)\s+(?:model|KV|compute|RS|output)\s+buffer size"
    r"\s*=\s*([0-9]+(?:\.[0-9]+)?)\s+MiB\b",
    re.IGNORECASE,
)
# The CUDA context and allocator slack each device carries beyond the logged buffers. Not
# logged anywhere, so it is credited as UNCERTAIN (may come back), never as freed.
_DEVICE_CONTEXT_ALLOWANCE_BYTES = 512 * _MIB

# What an idle card reports in use with nothing loaded (driver reservation, compositor):
# the same 5% / 1 GiB figure llama_cpp's WDDM correction subtracts, so "other programs are
# using N GB" here and in the loader's own hidden-usage note are the same N.
_IDLE_RESERVE_FRACTION = 0.05
_IDLE_RESERVE_MAX_BYTES = 1024 * _MIB

# /estimate-memory fires on every settings change; nvidia-smi costs ~100 ms on Windows and the
# Vulkan probe a subprocess. A /load always reads fresh.
_PROBE_TTL_S = 2.0
_probe_lock = threading.Lock()
_probe_cache: Optional[tuple[float, tuple, Any]] = None


def resident_llama_gpu_bytes(llama_backend: Any) -> Optional[dict[int, int]]:
    """``{physical_gpu: bytes}`` the running llama-server reported allocating, or None.

    None means "cannot say": no child, a child that never logged its buffers, or one whose
    device ordinals cannot be mapped to physical ids. Callers then treat whatever that child
    holds as uncertain rather than as freed.
    """
    try:
        process = getattr(llama_backend, "_process", None)
        if process is None or process.poll() is not None:
            return None
        physical_ids = list(getattr(llama_backend, "_child_gpu_physical_ids", None) or ())
        lines = list(getattr(llama_backend, "_stdout_lines", None) or ())
    except Exception:
        return None
    if not physical_ids or not lines:
        return None
    out: dict[int, int] = {}
    for line in lines:
        match = _GPU_BUFFER_RE.search(str(line))
        if match is None:
            continue
        ordinal = int(match.group(1))
        if ordinal >= len(physical_ids):
            return None
        physical = int(physical_ids[ordinal])
        out[physical] = out.get(physical, 0) + int(float(match.group(2)) * _MIB)
    return out or None


def _llama_child_is_running(llama_backend: Any) -> bool:
    try:
        process = getattr(llama_backend, "_process", None)
        return process is not None and process.poll() is None
    except Exception:
        return False


def _probe_rows(
    gpu_ids: Optional[Sequence[int]], gpu_memory_mode: Optional[str]
) -> Optional[list[tuple[int, int, int]]]:
    """``(index, free_mib, total_mib)`` for the cards this load may use, as the loader reads them."""
    from core.inference.llama_cpp import LlamaCppBackend

    binary = LlamaCppBackend._find_llama_server_binary()
    is_vulkan = bool(binary) and LlamaCppBackend._is_vulkan_backend(binary)
    # Same call the launch makes: the ROCm arch gate applies to automatic placement only.
    rows = list(LlamaCppBackend._get_gpu_memory(binary, for_llama_server = not gpu_ids) or [])
    if is_vulkan and not gpu_ids and gpu_memory_mode != "manual":
        rows = list(LlamaCppBackend._vulkan_auto_gpu_memory(rows))
    if gpu_ids:
        wanted = {int(i) for i in gpu_ids}
        picked = [row for row in rows if int(row[0]) in wanted]
        # Fail open like the loader (#7164): a stale pick keeps the whole pool.
        rows = picked or rows
    return rows


def measure_gpu_memory(
    *,
    llama_backend: Any,
    gpu_ids: Optional[Sequence[int]] = None,
    gpu_memory_mode: Optional[str] = None,
    other_studio_model_resident: bool = False,
    fresh: bool = True,
) -> Optional[list[GpuMemory]]:
    """Per-card memory for the verdict, crediting what the load evicts. None when unreadable.

    ``other_studio_model_resident`` -- a Transformers model or an Images/Video pipeline the
    load would unload or evict. Its memory is not attributed per card, so every byte in use
    on every card is flagged as possibly coming back, which is what keeps the verdict from
    calling anything too large while one is loaded.
    """
    global _probe_cache
    key = (tuple(sorted(int(i) for i in gpu_ids)) if gpu_ids else (), gpu_memory_mode)
    rows: Optional[list[tuple[int, int, int]]] = None
    if not fresh:
        with _probe_lock:
            cached = _probe_cache
        if cached is not None and cached[1] == key and time.monotonic() - cached[0] < _PROBE_TTL_S:
            rows = cached[2]
    if rows is None:
        try:
            rows = _probe_rows(gpu_ids, gpu_memory_mode)
        except Exception as exc:  # noqa: BLE001 -- a probe that failed is not a verdict
            logger.debug("Load guardrail GPU probe failed: %s", exc)
            return None
        with _probe_lock:
            _probe_cache = (time.monotonic(), key, rows)
    if not rows:
        return []

    llama_running = _llama_child_is_running(llama_backend)
    llama_bytes = resident_llama_gpu_bytes(llama_backend) if llama_running else None
    llama_ids: set[int] = set()
    if llama_running:
        try:
            llama_ids = {int(i) for i in getattr(llama_backend, "_child_gpu_physical_ids", None) or ()}
        except Exception:
            llama_ids = set()

    out: list[GpuMemory] = []
    for idx, free_mib, total_mib in rows:
        idx = int(idx)
        free = max(0, int(free_mib)) * _MIB
        total = max(0, int(total_mib)) * _MIB
        in_use = max(0, total - free) if total else 0
        credited = 0
        uncertain = 0
        attributed = True
        if llama_running:
            if llama_bytes is not None:
                credited = int(llama_bytes.get(idx, 0))
                if idx in llama_ids or credited:
                    uncertain += _DEVICE_CONTEXT_ALLOWANCE_BYTES
            elif not llama_ids or idx in llama_ids:
                # A child we cannot read: all of this card's used memory may be its.
                uncertain += in_use
                attributed = False
        if other_studio_model_resident:
            uncertain += in_use
            attributed = False
        # Never credit past what the card is actually using.
        if total:
            credited = min(credited, in_use)
            uncertain = min(uncertain, max(0, in_use - credited))
        other: Optional[int] = None
        if total and attributed:
            idle = int(min(_IDLE_RESERVE_MAX_BYTES, _IDLE_RESERVE_FRACTION * total))
            other = max(0, in_use - credited - idle)
        out.append(
            GpuMemory(
                index = idx,
                free_bytes = free + credited,
                total_bytes = total,
                uncertain_bytes = uncertain,
                other_bytes = other,
            )
        )
    return out


def classify_placement(
    gpu_memory_mode: Optional[str],
    gpu_layers: Optional[int],
    extras: Optional[Iterable[str]],
    offload_fraction: float,
) -> str:
    """Who decides how much lands on the GPU: Auto, the user (fixed), or nobody (CPU).

    Fixed means ``--fit off``: Manual with a layer count, or a layer count in the extras,
    which llama.cpp honours even under ``--fit on`` ("n_gpu_layers already set by user").
    Manual at -1 hands the count to ``--fit`` and stays Auto.
    """
    if offload_fraction <= 0.0:
        return PLACEMENT_CPU
    try:
        from core.inference.llama_server_args import parse_gpu_layers_override

        override = parse_gpu_layers_override(list(extras) if extras else None)
    except Exception:
        override = None
    if override is not None and override >= 0:
        return PLACEMENT_FIXED
    if gpu_memory_mode == "manual" and gpu_layers is not None and gpu_layers >= 0:
        return PLACEMENT_FIXED
    return PLACEMENT_AUTO


def verdict_from_breakdown(
    breakdown: Any,
    *,
    gpus: Optional[Sequence[GpuMemory]],
    placement: str,
    context_pinned: bool,
    tensor_split: Optional[Sequence[float]] = None,
    vram_fraction: Optional[float] = None,
    ram_available_bytes: Optional[int] = None,
    link_preference: Optional[Mapping[int, float]] = None,
    link_shared: Sequence[int] = (),
) -> LoadVerdict:
    """The verdict for a ``_GgufMemoryBreakdown`` (taken structurally) and a measurement.

    ``link_preference`` / ``link_shared``: the hardware check's measured host bandwidth per card
    and the cards another model runs on, when the loader will place with them (see
    ``load_verdict.VerdictInputs``)."""
    if vram_fraction is None:
        try:
            from core.inference.llama_cpp import _active_vram_fraction

            vram_fraction = _active_vram_fraction()
        except Exception:
            vram_fraction = 0.97
    gpu_bytes = getattr(breakdown, "gpu_bytes", None)
    inputs = VerdictInputs(
        weights_bytes = int(getattr(breakdown, "weights_bytes", 0) or 0),
        kv_bytes = int(getattr(breakdown, "kv_bytes", 0) or 0),
        compute_bytes = int(getattr(breakdown, "compute_bytes", 0) or 0),
        total_bytes = int(getattr(breakdown, "total_bytes", 0) or 0),
        gpu_bytes = None if gpu_bytes is None else int(gpu_bytes),
        kv_on_gpu = bool(getattr(breakdown, "kv_on_gpu", True)),
        kv_estimable = bool(getattr(breakdown, "kv_estimable", True)),
        sizes_are_floor = bool(
            getattr(breakdown, "drafter_kv_unsized", False)
            or getattr(breakdown, "adapters_unsized", False)
        ),
        placement = placement,
        context_pinned = context_pinned,
        n_ctx = int(getattr(breakdown, "n_ctx", 0) or 0),
        gpus = tuple(gpus or ()),
        tensor_split = tuple(tensor_split) if tensor_split else None,
        vram_fraction = float(vram_fraction),
        ram_available_bytes = ram_available_bytes,
        link_preference = dict(link_preference) if link_preference else None,
        link_shared = tuple(int(i) for i in link_shared),
    )
    # A failed probe (None) arrives as no cards, which compute_load_verdict reads as "no
    # reading" for any load that wants the GPU.
    return compute_load_verdict(inputs)


def describe_other_gpu_holders(gpu_indices: Sequence[int]) -> Optional[str]:
    """``"LM Studio.exe (PID 1372) ~13.8 GB on GPU 0"`` for the programs outside Studio that
    hold memory on these cards, or None. Windows only (the per-process counter), best effort.

    The same lookup the loader's out-of-memory message makes, read only when a refusal is
    about to be shown: the counter read spawns PowerShell, about a second.
    """
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        from utils.hardware import gpu_process_memory as _gpm

        samples = _gpm.query_gpu_process_memory()
        if not samples:
            return None
        rows = LlamaCppBackend._get_gpu_memory(LlamaCppBackend._find_llama_server_binary())
        used_by_index = {
            int(idx): max(0, int(total) - int(free)) * _MIB
            for idx, free, total in rows or ()
            if total and total > 0
        }
        return _gpm.describe_gpu_memory_holders(
            samples,
            used_by_index = used_by_index,
            gpu_indices = sorted(int(i) for i in gpu_indices) if gpu_indices else None,
            exclude_pids = _gpm.studio_process_ids(),
            nvidia_luids = _gpm.nvidia_adapter_luids(),
        )
    except Exception as exc:  # noqa: BLE001 -- naming the culprit is a courtesy
        logger.debug("GPU memory holder lookup failed: %s", exc)
        return None


def available_ram_bytes() -> Optional[int]:
    """Available host RAM, cgroup-capped, as the loader's own host preflight reads it."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        mib = LlamaCppBackend._available_system_memory_mib()
    except Exception:
        return None
    return None if mib is None else int(mib) * _MIB
