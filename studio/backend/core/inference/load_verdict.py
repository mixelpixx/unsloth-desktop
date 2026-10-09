# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One plain verdict per GGUF load: will it fit in the memory that is free RIGHT NOW.

The Load Model panel, the Hub picker and the model memory hook each grew their own
answer to that question, against different capacities (one of them the card's TOTAL),
so the same model could read "fits" on one screen and "too large" on the next. This is
the backend's answer, priced from the same GGUF estimate ``/estimate-memory`` returns
and the same nvidia-smi free reading the loader's own placement uses, and it is the one
``/load`` acts on.

Pure: no I/O, no probing. The route gathers the estimate and the per-GPU free memory
(already crediting whatever Studio model the load is about to evict); this only decides.

The levels, from best to worst:

* ``full_gpu``         -- everything fits on the GPU with room to spare.
* ``fits_barely``      -- it fits, but within the estimate's error: it may spill or OOM.
* ``partial_gpu``      -- Auto spills layers into system RAM. Slower, works.
* ``cpu``              -- no layers on the GPU; it runs from system RAM.
* ``disk_streaming``   -- larger than free GPU memory and RAM together, so llama.cpp
                          pages weights from disk through mmap. Upstream does this on
                          purpose and it works, very slowly, so it is not refused.
* ``likely_too_large`` -- the load will crash: a placement the USER fixed cannot fit
                          the free GPU memory, or the KV cache and compute buffers (which
                          cannot stream from disk) do not fit anywhere.
* ``unknown``          -- the inputs needed to answer are missing.

``likely_too_large`` is deliberately hard to reach. It needs a confident miss: the
shortfall has to exceed the estimation band (:func:`estimation_band_bytes`) even after
crediting every byte the load MIGHT reclaim. A miss inside that band reads
``fits_barely``, and one that hinges on memory nobody could attribute reads ``unknown``.
A false "won't fit" blocks a load that would have worked, which costs the user more than
an occasional crash the guardrail failed to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "AUTO_MIN_CONTEXT",
    "CPU",
    "DISK_STREAMING",
    "FITS_BARELY",
    "FULL_GPU",
    "GpuMemory",
    "LEVELS",
    "LIKELY_TOO_LARGE",
    "LoadVerdict",
    "PARTIAL_GPU",
    "PLACEMENT_AUTO",
    "PLACEMENT_CPU",
    "PLACEMENT_FIXED",
    "UNKNOWN",
    "VerdictInputs",
    "compute_load_verdict",
    "estimation_band_bytes",
]

FULL_GPU = "full_gpu"
FITS_BARELY = "fits_barely"
PARTIAL_GPU = "partial_gpu"
CPU = "cpu"
DISK_STREAMING = "disk_streaming"
LIKELY_TOO_LARGE = "likely_too_large"
UNKNOWN = "unknown"

LEVELS = (FULL_GPU, FITS_BARELY, PARTIAL_GPU, CPU, DISK_STREAMING, LIKELY_TOO_LARGE, UNKNOWN)

# Reason codes. The frontend keys its copy off these, so they are a wire contract: add, never rename.
REASON_FITS = "fits"
# Auto with no context named: the planner picks a window that keeps the model on the GPU.
REASON_CONTEXT_FITTED = "context_fitted"
REASON_TIGHT = "tight"
REASON_SPILLS_TO_RAM = "spills_to_ram"
REASON_CPU_ONLY = "cpu_only"
REASON_STREAMS_FROM_DISK = "streams_from_disk"
# A user-fixed GPU share (Manual layers, an -ngl in the extras, a fixed --tensor-split) that
# cannot fit: --fit is off, so llama.cpp allocates exactly what it was told and cudaMalloc fails.
REASON_FORCED_GPU_OVERFLOW = "forced_gpu_overflow"
# KV cache + compute buffers alone exceed free GPU memory and RAM together. Weights can page
# from disk; these cannot, so the allocation fails whatever the offload.
REASON_RUNTIME_OVERFLOW = "runtime_overflow"
REASON_NO_ESTIMATE = "no_estimate"
REASON_KV_UNSIZED = "kv_unsized"
REASON_NO_GPU_READING = "no_gpu_reading"
# A Studio model this load would unload holds memory nobody could measure, and the answer
# depends on how much of it comes back.
REASON_RESIDENT_UNATTRIBUTED = "resident_unattributed"

PLACEMENT_AUTO = "auto"
PLACEMENT_FIXED = "fixed"
PLACEMENT_CPU = "cpu"

_MIB = 1024 * 1024
_GIB = 1024 * _MIB

# The estimate reads exact weight sizes from the header and the KV cache from its dims, but
# the compute buffers are modelled and the per-device CUDA context (~300-500 MiB) is not in
# it at all, so the real footprint usually runs a little HIGH. The band covers the opposite
# error, an estimate that overshoots: a miss must clear it before anything is called too large.
_BAND_MIN_BYTES = 512 * _MIB
_BAND_FRACTION = 0.05

# What a card keeps back under the default VRAM budget, floor included: mirrors
# llama_cpp._vram_usable_mib so "fits comfortably" here means what it means to the planner.
_RESERVE_FLOOR_BYTES = 512 * _MIB
# Host RAM kept back for the OS and Studio itself before a spill is called comfortable.
_RAM_RESERVE_BYTES = 1 * _GIB

# The smallest window Auto is assumed to fall back to before it gives up GPU layers. Only
# used to decide whether an Auto load with NO context named can stay on the GPU, and never
# to call anything too large: the confidence gate refuses to rest a refusal on a guess.
AUTO_MIN_CONTEXT = 4096


@dataclass(frozen = True)
class GpuMemory:
    """One card this load may use, as the route measured it."""

    index: int
    # Free now, plus what the load provably frees by evicting Studio's resident model.
    free_bytes: int
    # 0 when the probe reports no total (an iGPU, a MIG / vGPU slice).
    total_bytes: int = 0
    # Held by a Studio model the load evicts that could not be measured: MAY come free too.
    uncertain_bytes: int = 0
    # Held by programs outside Studio's evictable models, when that is known.
    other_bytes: Optional[int] = None


@dataclass(frozen = True)
class VerdictInputs:
    """The estimate, the requested placement and the memory free now."""

    # The /estimate-memory figures: resident files, KV cache, compute buffers, the total,
    # and the GPU share under the requested offload (None when the planner never ran).
    weights_bytes: int
    kv_bytes: int
    compute_bytes: int
    total_bytes: int
    gpu_bytes: Optional[int]
    kv_on_gpu: bool = True
    kv_estimable: bool = True
    # A drafter cache or adapter the estimate could not size: the figures are a floor.
    sizes_are_floor: bool = False
    placement: str = PLACEMENT_AUTO
    # False when no context was named and the loader sizes its own window.
    context_pinned: bool = True
    n_ctx: int = 0
    # The cards this load may use: a pin's subset, or every visible card.
    gpus: Sequence[GpuMemory] = ()
    # A user-fixed per-card ratio, in the order of ``gpus``. Only read for PLACEMENT_FIXED.
    tensor_split: Optional[Sequence[float]] = None
    vram_fraction: float = 0.97
    ram_available_bytes: Optional[int] = None
    # Hardware check "Prefer fast-link GPUs": {gpu index: measured host GiB/s}. An Auto load
    # that fits one card alone is judged on the card the loader then picks (the faster-linked
    # one when it holds the load comfortably), not on the pool. None: the pool, as before.
    link_preference: Optional[Mapping[int, float]] = None
    # Cards another loaded model runs on, which the loader keeps a single-card load off.
    link_shared: tuple[int, ...] = ()


@dataclass(frozen = True)
class LoadVerdict:
    level: str
    reason: str
    # One sentence for logs and API callers. The UI builds its own copy from the numbers.
    message: str
    gpu_need_bytes: Optional[int] = None
    gpu_free_bytes: Optional[int] = None
    gpu_total_bytes: Optional[int] = None
    gpu_indices: tuple[int, ...] = ()
    other_apps_bytes: Optional[int] = None
    ram_need_bytes: Optional[int] = None
    ram_free_bytes: Optional[int] = None
    # KV cache + compute buffers + runtime state: what cannot page from disk.
    runtime_bytes: Optional[int] = None
    # How far the need is from the free figure it was compared to: positive is room.
    headroom_bytes: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "reason": self.reason,
            "message": self.message,
            "gpu_need_bytes": self.gpu_need_bytes,
            "gpu_free_bytes": self.gpu_free_bytes,
            "gpu_total_bytes": self.gpu_total_bytes,
            "gpu_indices": list(self.gpu_indices),
            "other_apps_bytes": self.other_apps_bytes,
            "ram_need_bytes": self.ram_need_bytes,
            "ram_free_bytes": self.ram_free_bytes,
            "runtime_bytes": self.runtime_bytes,
            "headroom_bytes": self.headroom_bytes,
        }


def estimation_band_bytes(need_bytes: int) -> int:
    """How far off the estimate may plausibly be for a footprint this size."""
    return int(max(_BAND_MIN_BYTES, _BAND_FRACTION * max(0, int(need_bytes))))


def _reserve_bytes(gpu: GpuMemory, vram_fraction: float) -> int:
    if gpu.total_bytes > 0:
        frac = min(max(float(vram_fraction), 0.0), 1.0)
        return int(
            max(
                (1.0 - frac) * gpu.total_bytes,
                min(_RESERVE_FLOOR_BYTES, (1.0 - 0.97) * gpu.total_bytes),
            )
        )
    return _RESERVE_FLOOR_BYTES


def _gb(num_bytes: Optional[int]) -> str:
    return f"{max(0, int(num_bytes or 0)) / _GIB:.1f} GiB"


def _gpu_label(indices: Sequence[int]) -> str:
    if not indices:
        return "the GPU"
    if len(indices) == 1:
        return f"GPU {indices[0]}"
    return "GPUs " + ", ".join(str(i) for i in indices)


def _other_sum(gpus: Sequence[GpuMemory]) -> Optional[int]:
    known = [g.other_bytes for g in gpus if g.other_bytes is not None]
    if not known:
        return None
    total = int(sum(known))
    # Under a gigabyte is a desktop and a browser, not a culprit worth a sentence.
    return total if total >= _GIB else None


def _other_note(other: Optional[int]) -> str:
    return f" Other programs are using {_gb(other)}." if other else ""


def _unknown(reason: str, message: str, **numbers: Any) -> LoadVerdict:
    return LoadVerdict(level = UNKNOWN, reason = reason, message = message, **numbers)


def compute_load_verdict(inputs: VerdictInputs) -> LoadVerdict:
    """The verdict for one prospective load. See the module docstring for the levels."""
    if inputs.total_bytes <= 0:
        return _unknown(REASON_NO_ESTIMATE, "Could not size this model, so it is not checked.")
    if not inputs.kv_estimable:
        # kv_bytes is 0 meaning UNKNOWN, and at a long context the cache is the biggest term.
        return _unknown(
            REASON_KV_UNSIZED,
            "The model file does not say how large its KV cache is, so the fit is not checked.",
        )

    weights = max(0, int(inputs.weights_bytes))
    kv = max(0, int(inputs.kv_bytes))
    total = max(0, int(inputs.total_bytes))
    gpu_need = max(0, int(inputs.gpu_bytes or 0)) if inputs.placement != PLACEMENT_CPU else 0
    gpu_need = min(gpu_need, total)

    # No context named: the loader sizes the window itself, so the cache priced at the
    # native context is not a requirement. Charge the smallest window Auto falls back to.
    kv_effective = kv
    if not inputs.context_pinned and inputs.n_ctx > AUTO_MIN_CONTEXT and kv > 0:
        kv_effective = int(kv * AUTO_MIN_CONTEXT / float(inputs.n_ctx))
    kv_saving = kv - kv_effective
    total_eff = max(0, total - kv_saving)
    gpu_need_eff = max(0, gpu_need - (kv_saving if inputs.kv_on_gpu else 0))
    # What cannot page from disk: every resident byte that is not a weight file.
    runtime_eff = max(0, total_eff - weights)

    ram_free = inputs.ram_available_bytes
    ram_usable = None if ram_free is None else max(0, int(ram_free) - _RAM_RESERVE_BYTES)

    gpus = list(inputs.gpus)
    if gpu_need_eff <= 0 or inputs.placement == PLACEMENT_CPU:
        return _cpu_verdict(inputs, total_eff, runtime_eff, ram_free, ram_usable)
    if not gpus:
        return _unknown(
            REASON_NO_GPU_READING,
            "Could not read free GPU memory, so the fit is not checked.",
            gpu_need_bytes = gpu_need_eff,
        )

    if inputs.placement == PLACEMENT_FIXED and inputs.tensor_split:
        split_verdict = _fixed_split_verdict(inputs, gpus, gpu_need_eff)
        if split_verdict is not None:
            return split_verdict

    if inputs.placement == PLACEMENT_AUTO and inputs.link_preference and len(gpus) > 1:
        gpus = _link_preferred_gpus(inputs, gpus, gpu_need, gpu_need_eff)

    indices = tuple(g.index for g in gpus)
    free = int(sum(max(0, g.free_bytes) for g in gpus))
    usable = int(sum(max(0, g.free_bytes - _reserve_bytes(g, inputs.vram_fraction)) for g in gpus))
    uncertain = int(sum(max(0, g.uncertain_bytes) for g in gpus))
    capacity = int(sum(max(0, g.total_bytes) for g in gpus)) or None
    capacity_known = all(g.total_bytes > 0 for g in gpus)
    other = _other_sum(gpus)
    band = estimation_band_bytes(gpu_need_eff)
    numbers = dict(
        gpu_need_bytes = gpu_need_eff,
        gpu_free_bytes = free,
        gpu_total_bytes = capacity,
        gpu_indices = indices,
        other_apps_bytes = other,
        ram_free_bytes = ram_free,
        runtime_bytes = runtime_eff,
    )

    if gpu_need_eff + band <= usable:
        context_fitted = gpu_need_eff < gpu_need and gpu_need + band > usable
        if inputs.sizes_are_floor:
            # Part of the footprint could not be sized, so "comfortably" is not ours to say.
            return LoadVerdict(
                level = FITS_BARELY,
                reason = REASON_TIGHT,
                message = f"Should fit on {_gpu_label(indices)}, but part of it could not be sized.",
                headroom_bytes = usable - gpu_need_eff,
                **numbers,
            )
        # When the whole native window fits, Auto loads all of it, so that is the size to quote; the
        # 4k floor charged above only decides the verdict when Auto would have to shrink the window.
        quoted_need = gpu_need_eff if context_fitted else gpu_need
        return LoadVerdict(
            level = FULL_GPU,
            reason = REASON_CONTEXT_FITTED if context_fitted else REASON_FITS,
            message = (
                f"Fits on {_gpu_label(indices)} with a shorter context: Auto picks one that leaves room."
                if context_fitted
                else f"Fits on {_gpu_label(indices)}: needs ~{_gb(quoted_need)}, {_gb(free)} free."
            ),
            headroom_bytes = usable - quoted_need,
            **{**numbers, "gpu_need_bytes": quoted_need},
        )

    if gpu_need_eff <= free + band:
        return LoadVerdict(
            level = FITS_BARELY,
            reason = REASON_TIGHT,
            message = (
                f"Tight fit: needs ~{_gb(gpu_need_eff)} on {_gpu_label(indices)}; "
                f"{_gb(free)} is free.{_other_note(other)}"
            ),
            headroom_bytes = free - gpu_need_eff,
            **numbers,
        )

    # It does not fit the GPU memory that is measurably free.
    runtime_verdict = _runtime_overflow(inputs, runtime_eff, usable + uncertain, ram_usable, numbers)
    if runtime_verdict is not None:
        return runtime_verdict

    if inputs.placement == PLACEMENT_FIXED:
        # A card with no total (an iGPU whose "VRAM" is host RAM, a MIG / vGPU slice) reports a
        # free figure nothing here can bound, so it never carries a refusal.
        if capacity_known and gpu_need_eff > free + uncertain + band:
            return LoadVerdict(
                level = LIKELY_TOO_LARGE,
                reason = REASON_FORCED_GPU_OVERFLOW,
                message = (
                    f"Needs ~{_gb(gpu_need_eff)} on {_gpu_label(indices)}; "
                    f"{_gb(free)} is free.{_other_note(other)}"
                ),
                headroom_bytes = free - gpu_need_eff,
                **numbers,
            )
        if not capacity_known:
            return _unknown(
                REASON_NO_GPU_READING,
                "This GPU does not report its capacity, so the fit is not checked.",
                headroom_bytes = free - gpu_need_eff,
                **numbers,
            )
        # Only the unmeasured share of an outgoing model stands between this and a crash.
        return _unknown(
            REASON_RESIDENT_UNATTRIBUTED,
            "The loaded model will be unloaded first, and how much memory that frees "
            "could not be measured, so the fit is not certain.",
            headroom_bytes = free - gpu_need_eff,
            **numbers,
        )

    # Auto: the planner (or llama.cpp's --fit) moves layers to system RAM.
    if uncertain and gpu_need_eff <= free + uncertain + band:
        return LoadVerdict(
            level = FITS_BARELY,
            reason = REASON_RESIDENT_UNATTRIBUTED,
            message = (
                "May fit once the loaded model is unloaded; how much memory that frees "
                "could not be measured."
            ),
            headroom_bytes = free - gpu_need_eff,
            **numbers,
        )
    spill = max(0, gpu_need_eff - max(0, usable))
    host_need = max(0, total_eff - gpu_need_eff) + spill
    numbers["ram_need_bytes"] = host_need
    if ram_usable is None or host_need <= ram_usable:
        on_gpu_at_all = usable > 0
        return LoadVerdict(
            level = PARTIAL_GPU if on_gpu_at_all else CPU,
            reason = REASON_SPILLS_TO_RAM if on_gpu_at_all else REASON_CPU_ONLY,
            message = (
                f"Doesn't fully fit on {_gpu_label(indices)}: about {_gb(spill)} runs from "
                f"system RAM, which is slower.{_other_note(other)}"
                if on_gpu_at_all
                else f"No room on {_gpu_label(indices)}: it runs from system RAM, which is slower."
                f"{_other_note(other)}"
            ),
            headroom_bytes = free - gpu_need_eff,
            **numbers,
        )
    return LoadVerdict(
        level = DISK_STREAMING,
        reason = REASON_STREAMS_FROM_DISK,
        message = (
            "Larger than free GPU memory and RAM together: parts of the model will be read "
            f"from disk while it runs, which is very slow.{_other_note(other)}"
        ),
        headroom_bytes = free - gpu_need_eff,
        **numbers,
    )


def _link_preferred_gpus(
    inputs: VerdictInputs, gpus: list[GpuMemory], need_full: int, need_floor: int
) -> list[GpuMemory]:
    """The one card the loader puts an Auto load on when "Prefer fast-link GPUs" moves it,
    else ``gpus`` unchanged. The same rule the loader applies
    (``utils.hardware.link_preference.faster_card``), from the same default: the roomiest card
    no other model runs on that holds the whole load, else the roomiest card."""
    from utils.hardware.link_preference import AUTO_CONTEXT_ROOM_TOLERANCE_MIB, faster_card

    usable = {
        g.index: max(0, g.free_bytes - _reserve_bytes(g, inputs.vram_fraction)) / _MIB for g in gpus
    }
    ranked = sorted(gpus, key = lambda g: usable[g.index], reverse = True)
    shared = set(inputs.link_shared)
    need_full_mib = need_full / _MIB
    default = next(
        (g.index for g in ranked if g.index not in shared and usable[g.index] >= need_full_mib),
        ranked[0].index,
    )
    auto_window = not inputs.context_pinned
    faster = faster_card(
        default,
        [(g.index, usable[g.index]) for g in ranked],
        inputs.link_preference,
        need_mib = need_full_mib,
        shared = shared,
        floor_need_mib = need_floor / _MIB if auto_window else None,
        room_tolerance_mib = AUTO_CONTEXT_ROOM_TOLERANCE_MIB if auto_window else 0.0,
    )
    if faster is None:
        return gpus
    return [g for g in gpus if g.index == faster]


def _runtime_overflow(
    inputs: VerdictInputs,
    runtime_bytes: int,
    gpu_room_bytes: int,
    ram_usable: Optional[int],
    numbers: dict[str, Any],
) -> Optional[LoadVerdict]:
    """``likely_too_large`` when the KV cache and buffers fit nowhere, else None.

    Generous on purpose: the runtime state is allowed every usable GPU byte AND all usable
    RAM at once, as though no weight needed either. Only a context the user named counts,
    since without one the loader shrinks the window and the cache with it.
    """
    if not inputs.context_pinned or ram_usable is None:
        return None
    room = max(0, gpu_room_bytes) + ram_usable
    if runtime_bytes <= room + estimation_band_bytes(runtime_bytes):
        return None
    return LoadVerdict(
        level = LIKELY_TOO_LARGE,
        reason = REASON_RUNTIME_OVERFLOW,
        message = (
            f"The KV cache and buffers alone need ~{_gb(runtime_bytes)}, more than free GPU "
            f"memory and RAM together ({_gb(room)}). Lower the context length.{_other_note(numbers.get('other_apps_bytes'))}"
        ),
        headroom_bytes = room - runtime_bytes,
        **numbers,
    )


def _cpu_verdict(
    inputs: VerdictInputs,
    total_eff: int,
    runtime_eff: int,
    ram_free: Optional[int],
    ram_usable: Optional[int],
) -> LoadVerdict:
    numbers = dict(
        gpu_need_bytes = 0,
        ram_need_bytes = total_eff,
        ram_free_bytes = ram_free,
        runtime_bytes = runtime_eff,
    )
    if ram_usable is None or total_eff <= ram_usable:
        return LoadVerdict(
            level = CPU,
            reason = REASON_CPU_ONLY,
            message = (
                f"Runs on the CPU from system RAM: needs ~{_gb(total_eff)}"
                + (f", {_gb(ram_free)} free." if ram_free is not None else ".")
            ),
            headroom_bytes = None if ram_usable is None else ram_usable - total_eff,
            **numbers,
        )
    if runtime_eff > ram_usable + estimation_band_bytes(runtime_eff):
        if inputs.context_pinned:
            return LoadVerdict(
                level = LIKELY_TOO_LARGE,
                reason = REASON_RUNTIME_OVERFLOW,
                message = (
                    f"The KV cache and buffers alone need ~{_gb(runtime_eff)}, more than the "
                    f"{_gb(ram_free)} of free RAM. Lower the context length."
                ),
                headroom_bytes = ram_usable - runtime_eff,
                **numbers,
            )
        return _unknown(
            REASON_RUNTIME_OVERFLOW,
            "Free RAM is short even for a small context; whether it loads depends on the "
            "context the loader picks.",
            headroom_bytes = ram_usable - runtime_eff,
            **numbers,
        )
    return LoadVerdict(
        level = DISK_STREAMING,
        reason = REASON_STREAMS_FROM_DISK,
        message = (
            f"Larger than free RAM ({_gb(ram_free)}): parts of the model will be read from "
            "disk while it runs, which is very slow."
        ),
        headroom_bytes = ram_usable - total_eff,
        **numbers,
    )


def _fixed_split_verdict(
    inputs: VerdictInputs, gpus: list[GpuMemory], gpu_need: int
) -> Optional[LoadVerdict]:
    """Per-card verdict for a user-fixed ``--tensor-split``, or None to fall back to the pool.

    A fixed ratio puts a fixed share on each card whatever each has free, so the pool can
    have room while one card overflows. None when the ratio does not line up with the cards
    (wrong length, no positive share): llama.cpp's own reading of such a list is not ours to
    model, and the pooled check is the conservative stand-in.
    """
    shares = [float(s) for s in (inputs.tensor_split or ())]
    if len(shares) != len(gpus) or any(s < 0 for s in shares) or sum(shares) <= 0:
        return None
    weight = sum(shares)
    worst: Optional[tuple[int, GpuMemory, int]] = None  # (shortfall, gpu, need on it)
    tight: Optional[tuple[GpuMemory, int]] = None
    for gpu, share in zip(gpus, shares):
        need = int(gpu_need * share / weight)
        if need <= 0:
            continue
        band = estimation_band_bytes(need)
        if need > gpu.free_bytes + band:
            shortfall = need - gpu.free_bytes
            if worst is None or shortfall > worst[0]:
                worst = (shortfall, gpu, need)
        elif need + band > gpu.free_bytes - _reserve_bytes(gpu, inputs.vram_fraction):
            if tight is None:
                tight = (gpu, need)
    if worst is None and tight is None:
        return None  # every card has room: the pooled check words it
    if worst is not None:
        _shortfall, gpu, need = worst
        numbers = dict(
            gpu_need_bytes = need,
            gpu_free_bytes = gpu.free_bytes,
            gpu_total_bytes = gpu.total_bytes or None,
            gpu_indices = (gpu.index,),
            other_apps_bytes = _other_sum([gpu]),
            ram_free_bytes = inputs.ram_available_bytes,
            headroom_bytes = gpu.free_bytes - need,
        )
        if gpu.total_bytes <= 0:
            return _unknown(
                REASON_NO_GPU_READING,
                "This GPU does not report its capacity, so the fit is not checked.",
                **numbers,
            )
        if need > gpu.free_bytes + gpu.uncertain_bytes + estimation_band_bytes(need):
            return LoadVerdict(
                level = LIKELY_TOO_LARGE,
                reason = REASON_FORCED_GPU_OVERFLOW,
                message = (
                    f"Needs ~{_gb(need)} on GPU {gpu.index}; {_gb(gpu.free_bytes)} is free."
                    f"{_other_note(numbers['other_apps_bytes'])}"
                ),
                **numbers,
            )
        return _unknown(
            REASON_RESIDENT_UNATTRIBUTED,
            "The loaded model will be unloaded first, and how much memory that frees "
            "could not be measured, so the fit is not certain.",
            **numbers,
        )
    gpu, need = tight  # type: ignore[misc]
    return LoadVerdict(
        level = FITS_BARELY,
        reason = REASON_TIGHT,
        message = f"Tight fit: needs ~{_gb(need)} on GPU {gpu.index}; {_gb(gpu.free_bytes)} is free.",
        gpu_need_bytes = need,
        gpu_free_bytes = gpu.free_bytes,
        gpu_total_bytes = gpu.total_bytes or None,
        gpu_indices = (gpu.index,),
        other_apps_bytes = _other_sum([gpu]),
        ram_free_bytes = inputs.ram_available_bytes,
        headroom_bytes = gpu.free_bytes - need,
    )
