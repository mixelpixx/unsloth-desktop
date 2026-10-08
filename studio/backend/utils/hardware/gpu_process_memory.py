# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which processes hold GPU memory, best effort, so a load can name them.

Windows only. ``nvidia-smi --query-compute-apps`` reports ``used_memory`` as N/A under WDDM
("Windows KMD manages all the memory"), so the per-process figure comes from the
``\\GPU Process Memory(*)\\Dedicated Usage`` performance counter instead, the same
``Get-Counter`` route ``hardware._rocm_windows_perf_counter_vram_by_adapter`` already takes.

Purely advisory: every entry point returns None (or an empty answer) on any failure, and the
caller only ever appends the result to a message. Nothing here may fail or slow down a load
beyond the one short, bounded counter read.
"""

from __future__ import annotations

import itertools
import os
import platform
import re
import subprocess
from typing import Callable, Iterable, Mapping, Optional, Sequence

from utils.subprocess_compat import windows_hidden_subprocess_kwargs

_MIB = 1024 * 1024
_GIB = 1024 * _MIB

# Instances are named pid_<pid>_luid_0x<high>_0x<low>_phys_<n>. ``phys`` is the node inside a
# linked-adapter group, NOT the GPU index: two separate cards both read phys_0 and differ only
# by LUID (checked on a 2x RTX 3090 host), so the LUID is the adapter's identity here.
_INSTANCE_RE = re.compile(r"^pid_(\d+)_luid_0x([0-9a-f]+)_0x([0-9a-f]+)_phys_\d+", re.IGNORECASE)

# PCI vendor id the DirectX registry records NVIDIA adapters under.
_NVIDIA_PCI_VENDOR_ID = 0x10DE

# A process holding less than this is not worth naming: the desktop compositor and every
# hardware-accelerated window hold a few hundred MiB, and listing them buries the program
# that actually matters (another LLM app holding 13 GB).
_MIN_NAMED_BYTES = 512 * _MIB
_MAX_NAMED = 3
# Below this a holder's share of one card would print as "~0.0 GB on GPU 1".
_MIN_SLOT_BYTES = 64 * _MIB

# Matching adapters to nvidia-smi rows by how much each has in use: brute force over the
# assignments, bounded so a many-GPU host cannot turn a log line into a combinatorial search.
_MAX_MATCH_ROWS = 4
_MAX_MATCH_LUIDS = 6


def parse_gpu_process_memory(stdout: str) -> list[tuple[int, int, int]]:
    """``(pid, adapter_luid, bytes)`` per non-zero sample of the counter dump.

    ``stdout`` is ``<InstanceName>|<bytes>`` per line, as :func:`query_gpu_process_memory`
    prints it. Unparseable lines are skipped rather than voiding the read, and a pid with
    several samples on one adapter (a linked-adapter group) is summed.
    """
    totals: dict[tuple[int, int], int] = {}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        instance, _, raw = line.rpartition("|")
        match = _INSTANCE_RE.match(instance.strip())
        if match is None:
            continue
        try:
            used = int(float(raw.strip()))
        except (TypeError, ValueError):
            continue
        if used <= 0:
            continue
        pid = int(match.group(1))
        luid = (int(match.group(2), 16) << 32) | int(match.group(3), 16)
        totals[(pid, luid)] = totals.get((pid, luid), 0) + used
    return [(pid, luid, used) for (pid, luid), used in totals.items()]


def query_gpu_process_memory(timeout: float = 5.0) -> Optional[list[tuple[int, int, int]]]:
    """Read the per-process dedicated GPU memory counter. None off Windows or on any failure.

    Counter names are localised on non-English Windows, where this simply reads nothing.
    """
    if platform.system() != "Windows":
        return None
    try:
        ps = (
            "$s=(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage'"
            " -ErrorAction SilentlyContinue).CounterSamples;"
            "if($s){$s|Where-Object{$_.CookedValue -gt 0}|"
            "ForEach-Object{'{0}|{1}' -f $_.InstanceName,[int64]$_.CookedValue}}"
            "else{'__NONE__'}"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            **windows_hidden_subprocess_kwargs(),
        )
    except Exception:
        return None
    if result.returncode != 0 or not (result.stdout or "").strip():
        return None
    return parse_gpu_process_memory(result.stdout)


def nvidia_adapter_luids() -> Optional[set[int]]:
    """LUIDs the DirectX registry lists for NVIDIA adapters, or None when it cannot say.

    Lets the matcher ignore an iGPU whose usage happens to sit near an NVIDIA card's. The
    registry keeps stale records with a zero LUID, which never name a live adapter.
    """
    try:
        from utils.hardware.hardware import _windows_amd_adapter_records_or_none

        records = _windows_amd_adapter_records_or_none(_NVIDIA_PCI_VENDOR_ID)
    except Exception:
        return None
    if not records:
        return None
    luids = {int(luid) for luid in records if int(luid) != 0}
    return luids or None


def match_adapters_to_gpus(
    used_by_luid: Mapping[int, int],
    used_by_index: Mapping[int, int],
) -> dict[int, int]:
    """``{luid: gpu_index}`` pairing each nvidia-smi row with the adapter whose in-use bytes
    match it, or {} when no pairing is trustworthy.

    The counter names adapters by LUID and nvidia-smi by index, and nothing cheap on Windows
    maps one to the other, but both report how much each card has in use. The assignment with
    the smallest total difference wins; a row may also match "no adapter" (an idle card shows
    no process samples at all). Rejected unless every pair agrees to within the slack the two
    readings legitimately differ by (nvidia-smi counts the driver's own reservation, the
    counter does not). Two cards with near-identical usage can swap, which costs nothing: the
    sentence then quotes near-identical numbers either way.
    """
    rows = [(int(i), int(u)) for i, u in used_by_index.items()]
    luids = [(int(l), int(u)) for l, u in used_by_luid.items()]
    if not rows or len(rows) > _MAX_MATCH_ROWS or len(luids) > _MAX_MATCH_LUIDS:
        return {}
    # One "no adapter" slot per row, so every row can be left unmatched.
    candidates: list[tuple[Optional[int], int]] = [*luids, *([(None, 0)] * len(rows))]
    best: Optional[tuple[int, tuple]] = None
    for chosen in itertools.permutations(range(len(candidates)), len(rows)):
        cost = sum(abs(rows[k][1] - candidates[c][1]) for k, c in enumerate(chosen))
        if best is None or cost < best[0]:
            best = (cost, chosen)
    if best is None:
        return {}
    mapping: dict[int, int] = {}
    for k, c in enumerate(best[1]):
        idx, row_used = rows[k]
        luid, luid_used = candidates[c]
        if abs(row_used - luid_used) > max(512 * _MIB, row_used // 10):
            return {}
        if luid is not None:
            mapping[luid] = idx
    # A busy adapter left over means the readings disagree about where the memory is.
    for luid, used in luids:
        if luid not in mapping and used >= _MIN_NAMED_BYTES:
            return {}
    return mapping


def studio_process_ids() -> set[int]:
    """This process and its descendants: memory they hold is not "another program"."""
    own = {os.getpid()}
    try:
        import psutil

        own.update(child.pid for child in psutil.Process().children(recursive = True))
    except Exception:
        pass
    return own


def _process_name(pid: int) -> str:
    try:
        import psutil

        name = psutil.Process(pid).name()
        if name:
            return name
    except Exception:
        pass
    return "a process"


def _gb(num_bytes: int) -> str:
    return f"~{num_bytes / _GIB:.1f} GiB"


def describe_gpu_memory_holders(
    samples: Iterable[tuple[int, int, int]],
    *,
    used_by_index: Mapping[int, int],
    gpu_indices: Optional[Sequence[int]] = None,
    exclude_pids: Iterable[int] = (),
    nvidia_luids: Optional[set[int]] = None,
    process_name: Optional[Callable[[int], str]] = None,
) -> Optional[str]:
    """``"Bionic.exe (PID 1372) ~13.8 GB on GPU 0, ~13.4 GB on GPU 1"`` for the biggest holders.

    ``used_by_index`` is nvidia-smi's in-use bytes per GPU index (total minus free), the side
    the adapters are matched against. ``gpu_indices`` narrows the naming to the cards this load
    can use once that matching succeeded; without a match each holder is quoted in total.
    ``exclude_pids`` (Studio and its children) are never named. None when nobody qualifies.
    """
    process_name = process_name or _process_name
    samples = list(samples)
    if nvidia_luids:
        samples = [s for s in samples if s[1] in nvidia_luids]
    if not samples:
        return None
    used_by_luid: dict[int, int] = {}
    for _pid, luid, used in samples:
        used_by_luid[luid] = used_by_luid.get(luid, 0) + used
    mapping = match_adapters_to_gpus(used_by_luid, used_by_index)
    wanted = {int(i) for i in gpu_indices} if gpu_indices is not None else None
    excluded = {int(p) for p in exclude_pids}

    per_pid: dict[int, dict[Optional[int], int]] = {}
    for pid, luid, used in samples:
        if pid in excluded:
            continue
        idx = mapping.get(luid) if mapping else None
        if mapping and (idx is None or (wanted is not None and idx not in wanted)):
            # Matched, and this adapter is not one the load uses: not a culprit here.
            continue
        slots = per_pid.setdefault(pid, {})
        slots[idx] = slots.get(idx, 0) + used

    holders = [
        (sum(slots.values()), pid, slots)
        for pid, slots in per_pid.items()
        if sum(slots.values()) >= _MIN_NAMED_BYTES
    ]
    if not holders:
        return None
    holders.sort(key = lambda h: (-h[0], h[1]))
    parts: list[str] = []
    for total, pid, slots in holders[:_MAX_NAMED]:
        if None in slots:
            where = _gb(total)
        else:
            where = ", ".join(
                f"{_gb(slots[idx])} on GPU {idx}"
                for idx in sorted(slots)
                if slots[idx] >= _MIN_SLOT_BYTES
            )
        parts.append(f"{process_name(pid)} (PID {pid}) {where}")
    return "; ".join(parts)
