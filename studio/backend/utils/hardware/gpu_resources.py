# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-GPU memory for the always-visible resources view, cheap enough to poll.

Two readings with very different costs, cached apart:

* Free and total per card: nvidia-smi through ``gpu_query``, with the argv the loader's own
  probe (``LlamaCppBackend._get_gpu_memory``) runs, so a reading either side takes is the
  other's cache entry. Held here for a second on top, so however many tabs poll, this view
  spawns nvidia-smi at most once a second.
* Who holds the used part: Windows' per-process GPU counter (``gpu_process_memory``), a
  PowerShell read of about a second. Refreshed on a daemon thread at most every 15 s, sooner
  (never more than every 2 s) once a load or unload changed what Studio holds. A request
  never waits on it: it reads the last answer.

Everything fails soft. No nvidia-smi (a Mac, an AMD or CPU host) is an empty card list, and a
counter that cannot answer leaves the Studio/other split to the caller's estimate.
"""

from __future__ import annotations

import os
import platform
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from loggers import get_logger

logger = get_logger(__name__)

_MIB = 1024 * 1024

READING_TTL_S = 1.0
# No nvidia-smi is the normal state of a Mac, AMD or CPU host, so it is asked again rarely.
_ABSENT_TTL_S = 60.0
# Display only: a reading this old may be served while a fresh one is taken.
_DISPLAY_MAX_STALE_S = 5.0

ATTRIBUTION_TTL_S = 15.0
ATTRIBUTION_MIN_INTERVAL_S = 2.0

# Every hardware-accelerated window holds some VRAM; below this a process is not listed by name.
MIN_LISTED_BYTES = 256 * _MIB
MAX_LISTED_PER_GPU = 5

_LIVE_QUERY = ("--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits")


@dataclass(frozen = True)
class GpuReading:
    index: int
    name: Optional[str]
    total_bytes: int
    free_bytes: int

    @property
    def used_bytes(self) -> int:
        return max(0, self.total_bytes - self.free_bytes)


@dataclass(frozen = True)
class HolderSamples:
    """One per-process counter read, with what is needed to split it per card."""

    taken_at: float
    generation: int
    # (pid, adapter_luid, bytes); None when the counter could not answer on this host.
    samples: Optional[tuple[tuple[int, int, int], ...]]
    studio_pids: frozenset = frozenset()
    luid_to_index: Mapping[int, int] = field(default_factory = dict)
    names: Mapping[int, str] = field(default_factory = dict)


def parse_memory_rows(stdout: str) -> list[tuple[int, int, int]]:
    """``(index, free_mib, total_mib)`` per ``index,memory.free,memory.total`` line.

    A row without a numeric total ("[N/A]" on MIG / vGPU) is skipped: it has no bar to draw.
    """
    rows: list[tuple[int, int, int]] = []
    for line in (stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    rows.sort()
    return rows


def _nvidia_smi() -> Optional[str]:
    """argv[0] for nvidia-smi, or None when this host has none.

    Bare when it is on PATH, since that is how the loader spells it and the cache key holds the
    whole argv.
    """
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    try:
        from utils.hardware.nvidia import _nvidia_smi_executable

        exe = _nvidia_smi_executable()
    except Exception:
        return None
    return exe if exe != "nvidia-smi" and os.path.isfile(exe) else None


def query_nvidia_memory(*, fresh: bool = False) -> Optional[list[tuple[int, int, int]]]:
    """Free and total MiB per NVIDIA card. None when there is no nvidia-smi at all, ``[]`` when
    it could not answer this time. ``fresh`` skips the shared cache (a load is running)."""
    exe = _nvidia_smi()
    if exe is None:
        return None
    try:
        from utils.hardware import gpu_query
        from utils.native_path_leases import child_env_without_native_path_secret
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        with gpu_query.display_reads(max_stale = _DISPLAY_MAX_STALE_S):
            result = gpu_query.run_nvidia_smi(
                [exe, *_LIVE_QUERY],
                cache = not fresh,
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **windows_hidden_subprocess_kwargs(),
            )
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 -- a view, never a failed request
        logger.debug("Resources view nvidia-smi read failed: %s", exc)
        return []
    if result.returncode != 0:
        return []
    return parse_memory_rows(result.stdout)


def query_gpu_names() -> dict[int, str]:
    """``{index: name}`` from the 60 s static inventory the health and system polls share."""
    try:
        from utils.hardware import nvidia

        rows = nvidia._query_gpu_inventory("resources view")
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    names: dict[int, str] = {}
    for row in rows:
        try:
            names[int(row["index"])] = str(row.get("name") or "").strip()
        except (KeyError, TypeError, ValueError):
            continue
    return names


def query_holder_samples() -> Optional[list[tuple[int, int, int]]]:
    """The per-process counter, NVIDIA adapters only when the registry can say which those are."""
    from utils.hardware import gpu_process_memory as gpm

    samples = gpm.query_gpu_process_memory()
    if samples is None:
        return None
    luids = gpm.nvidia_adapter_luids()
    if luids:
        samples = [s for s in samples if s[1] in luids]
    return list(samples)


def _process_name(pid: int) -> str:
    try:
        import psutil

        name = psutil.Process(pid).name()
        if name:
            return name
    except Exception:
        pass
    return f"PID {pid}"


def _studio_pids() -> frozenset:
    from utils.hardware.gpu_process_memory import studio_process_ids

    return frozenset(studio_process_ids())


def _generation() -> int:
    from utils import gpu_memory_events

    return gpu_memory_events.generation()


def _spawn(target: Callable[[], None]) -> None:
    threading.Thread(target = target, name = "resources-holders", daemon = True).start()


def _distinct_usage(used_by_index: Mapping[int, int]) -> bool:
    """Every card's in-use bytes differ from every other's by more than the matching slack, so
    the pairing could not have swapped two of them."""
    used = sorted(used_by_index.values())
    return all(b - a > max(512 * _MIB, b // 10) for a, b in zip(used, used[1:]))


def map_adapters(
    samples: Sequence[tuple[int, int, int]],
    readings: Sequence[GpuReading],
    previous: Mapping[int, int],
) -> tuple[dict[int, int], bool]:
    """``({luid: gpu_index}, unambiguous)`` for this read, else the last unambiguous pairing when
    it covers the same adapters.

    The pairing is by in-use bytes (see ``match_adapters_to_gpus``) and can fail on a read taken
    while a load moves memory; the adapters themselves do not move, so an earlier answer stands
    in. Only one that could not have swapped two cards: two idle 3090s read the same few hundred
    MiB, and a pairing learned then is a coin toss. One card and one adapter need no matching.
    """
    from utils.hardware.gpu_process_memory import match_adapters_to_gpus

    used_by_luid: dict[int, int] = {}
    for _pid, luid, used in samples:
        used_by_luid[luid] = used_by_luid.get(luid, 0) + used
    if not used_by_luid:
        return {}, False
    if len(readings) == 1 and len(used_by_luid) == 1:
        return {next(iter(used_by_luid)): readings[0].index}, True
    used_by_index = {r.index: r.used_bytes for r in readings if r.total_bytes > 0}
    mapping = match_adapters_to_gpus(used_by_luid, used_by_index)
    if mapping:
        return mapping, _distinct_usage(used_by_index)
    if previous and set(used_by_luid) <= set(previous):
        return dict(previous), True
    return {}, False


class ResourceReader:
    """Holds both caches. One process-wide instance below; tests build their own with stubs."""

    def __init__(
        self,
        *,
        memory_reader: Callable[..., Optional[list[tuple[int, int, int]]]] = query_nvidia_memory,
        name_reader: Callable[[], Mapping[int, str]] = query_gpu_names,
        holder_reader: Callable[[], Optional[list[tuple[int, int, int]]]] = query_holder_samples,
        studio_pids: Callable[[], frozenset] = _studio_pids,
        process_name: Callable[[int], str] = _process_name,
        generation: Callable[[], int] = _generation,
        spawn: Callable[[Callable[[], None]], None] = _spawn,
        clock: Callable[[], float] = time.monotonic,
        attribution_supported: Optional[bool] = None,
    ) -> None:
        self._memory_reader = memory_reader
        self._name_reader = name_reader
        self._holder_reader = holder_reader
        self._studio_pids = studio_pids
        self._process_name = process_name
        self._generation = generation
        self._spawn = spawn
        self._clock = clock
        self._attribution_supported = (
            platform.system() == "Windows"
            if attribution_supported is None
            else attribution_supported
        )
        # Two locks: a read stuck on a hung driver must not hold up the holder bookkeeping.
        self._read_lock = threading.Lock()
        self._lock = threading.Lock()
        self._readings: Optional[tuple[float, float, list[GpuReading]]] = None
        self._holders: Optional[HolderSamples] = None
        self._holders_running = False
        self._names: dict[int, str] = {}
        # The last pairing that could not have swapped two cards, for reads that cannot be paired.
        self._sure_mapping: dict[int, int] = {}

    # ── Free / total per card ─────────────────────────────────────

    def read_gpus(
        self, *, fresh: bool = False, visible: Optional[set[int]] = None
    ) -> list[GpuReading]:
        """The cards and their memory, at most ``READING_TTL_S`` old. Serialised, so callers
        arriving together share one nvidia-smi read instead of each starting one."""
        with self._read_lock:
            now = self._clock()
            cached = self._readings
            if cached is not None and now - cached[0] < cached[1]:
                readings = cached[2]
            else:
                rows = self._memory_reader(fresh = fresh)
                if rows is None:
                    readings, ttl = [], _ABSENT_TTL_S
                else:
                    names = self._name_reader() if rows else {}
                    readings = [
                        GpuReading(
                            index = int(idx),
                            name = names.get(int(idx)) or None,
                            total_bytes = max(0, int(total)) * _MIB,
                            free_bytes = max(0, int(free)) * _MIB,
                        )
                        for idx, free, total in rows
                        if int(total) > 0
                    ]
                    ttl = READING_TTL_S
                self._readings = (now, ttl, readings)
        if visible is not None:
            readings = [r for r in readings if r.index in visible]
        return list(readings)

    # ── Who holds it ──────────────────────────────────────────────

    def holders(self) -> Optional[HolderSamples]:
        """The last counter read, refreshing it in the background when due. Never blocks."""
        if not self._attribution_supported:
            return None
        now = self._clock()
        generation = self._generation()
        with self._lock:
            current = self._holders
            due = current is None or (
                now - current.taken_at >= ATTRIBUTION_TTL_S
                or (
                    current.generation != generation
                    and now - current.taken_at >= ATTRIBUTION_MIN_INTERVAL_S
                )
            )
            start = due and not self._holders_running
            if start:
                self._holders_running = True
        if start:
            try:
                self._spawn(lambda: self._refresh_holders(generation))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Resources holder refresh did not start: %s", exc)
                with self._lock:
                    self._holders_running = False
        with self._lock:
            return self._holders

    def _refresh_holders(self, generation: int) -> None:
        try:
            samples = self._holder_reader()
            studio = self._studio_pids() if samples else frozenset()
            # One attribute read, so no lock: the newest card reading, whichever it is.
            cached = self._readings
            readings = cached[2] if cached is not None else []
            mapping: dict[int, int] = {}
            names: dict[int, str] = {}
            if samples:
                mapping, sure = map_adapters(samples, readings, self._sure_mapping)
                if sure:
                    self._sure_mapping = dict(mapping)
                totals: dict[int, int] = {}
                for pid, _luid, used in samples:
                    if pid not in studio:
                        totals[pid] = totals.get(pid, 0) + used
                for pid, total in totals.items():
                    if total < MIN_LISTED_BYTES:
                        continue
                    # A pid keeps its name for its lifetime, so only new ones are looked up.
                    names[pid] = self._names.get(pid) or self._process_name(pid)
                self._names = dict(names)
            result = HolderSamples(
                taken_at = self._clock(),
                generation = generation,
                samples = None if samples is None else tuple(samples),
                studio_pids = studio,
                luid_to_index = mapping,
                names = names,
            )
            with self._lock:
                self._holders = result
        except Exception as exc:  # noqa: BLE001 -- naming holders is a courtesy
            logger.debug("Resources holder refresh failed: %s", exc)
            with self._lock:
                self._holders = HolderSamples(
                    taken_at = self._clock(), generation = generation, samples = None
                )
        finally:
            with self._lock:
                self._holders_running = False


def split_gpu_memory(
    readings: Sequence[GpuReading],
    holders: Optional[HolderSamples],
    *,
    estimate_by_gpu: Optional[Mapping[int, Optional[int]]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-card ``Studio · other · free`` and the named holders, plus holders no card claims.

    ``process``: the per-process counter, matched to cards. Studio is this process and its
    descendants (llama-server, the training worker, this process's own context); everything
    else in use, the driver's reservation included, is "other".
    ``estimate``: no usable counter, so Studio is what its runtimes logged (or planned) per card,
    ``estimate_by_gpu``. A card a runtime holds an unsized share of has no split (None).

    An empty counter answer is no answer: the counter's name is localised on non-English Windows
    and reads nothing there, which must not turn into "Studio holds nothing".
    """
    estimate_by_gpu = estimate_by_gpu or {}
    samples = holders.samples if holders is not None else None
    mapping = dict(holders.luid_to_index) if holders is not None else {}
    studio_pids = holders.studio_pids if holders is not None else frozenset()
    names = holders.names if holders is not None else {}
    known = {r.index for r in readings}
    placed = bool(samples) and bool(mapping) and set(mapping.values()) <= known

    studio_by_gpu: dict[int, int] = {}
    apps_by_gpu: dict[int, dict[int, int]] = {}
    unplaced: dict[int, int] = {}
    for pid, luid, used in samples or ():
        if not placed:
            # No card to put it on, so only another program's total is worth saying.
            if pid not in studio_pids:
                unplaced[pid] = unplaced.get(pid, 0) + used
            continue
        idx = mapping.get(luid)
        if idx is None:
            # An adapter the matching left out holds under its slack: nothing to report.
            continue
        if pid in studio_pids:
            studio_by_gpu[idx] = studio_by_gpu.get(idx, 0) + used
        else:
            bucket = apps_by_gpu.setdefault(idx, {})
            bucket[pid] = bucket.get(pid, 0) + used

    def named(holdings: Mapping[int, int]) -> list[dict[str, Any]]:
        rows = sorted(
            ((used, pid) for pid, used in holdings.items() if used >= MIN_LISTED_BYTES),
            key = lambda row: (-row[0], row[1]),
        )
        return [
            {"pid": pid, "name": names.get(pid) or f"PID {pid}", "bytes": used}
            for used, pid in rows[:MAX_LISTED_PER_GPU]
        ]

    out: list[dict[str, Any]] = []
    for reading in readings:
        used_total = reading.used_bytes
        if placed:
            attribution: Optional[str] = "process"
            studio: Optional[int] = studio_by_gpu.get(reading.index, 0)
        else:
            attribution = "estimate"
            studio = estimate_by_gpu.get(reading.index, 0)
        if studio is not None:
            studio = min(max(0, int(studio)), used_total)
        out.append(
            {
                "index": reading.index,
                "name": reading.name,
                "total_bytes": reading.total_bytes,
                "used_bytes": used_total,
                "free_bytes": reading.free_bytes,
                "studio_bytes": studio,
                "other_bytes": None if studio is None else used_total - studio,
                "attribution": attribution if studio is not None else None,
                "apps": named(apps_by_gpu.get(reading.index, {})) if placed else [],
            }
        )
    return out, named(unplaced)


_reader = ResourceReader()


def default_reader() -> ResourceReader:
    return _reader
