# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Resources > Hardware check: measure how each GPU and drive is attached, once.

What it measures, per NVIDIA GPU: the PCIe link (generation and width, current and maximum)
read WHILE a host-to-device copy runs -- an idle link downclocks, so a reading at rest says
nothing -- plus pinned host-to-device and device-to-host bandwidth; per pair of GPUs, whether
each can access the other directly (P2P) and the measured device-to-device copy speed. Per
storage location (Studio home, Hugging Face cache, temp): the bus (NVMe, SATA, USB) and the
media (SSD, HDD).

How: the GPU measurements run in a short-lived child (``hardware_check_probe.py``) on this
interpreter, because a CUDA context in the server process would pin ~300 MiB of VRAM per card
for good. The child copies a ``BUFFER_MIB`` buffer, skips a card without that much free plus a
context allowance, runs below normal priority and is killed at ``PROBE_TIMEOUT_S``. The link is
read with the nvidia-smi runner the resource strip uses (``gpu_query``). The storage query is
one PowerShell call on Windows (sysfs on Linux), run beside the GPU child.

When: once in the background after startup settles, when there is no result for the hardware
installed now (``auto_run``), and on demand from Settings. Both skip while a training run is
active or a model is loading, and say why. The result is JSON in the Studio home, stamped with
a fingerprint of the GPU UUIDs and PCI bus ids: a card moved to another slot reads "out of
date" and is checked again.

What it changes: nothing on its own. ``analyze`` turns a result into findings and the options
it recommends, and the options in ``utils.hardware_check_settings`` (all off until the user
switches them on) are read through ``active_link_preference``, ``tensor_split_concern`` and
``training_slow_link_warning``, each of which answers None unless its option is on AND the
stored result describes the hardware installed now.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from loggers import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
RESULT_FILENAME = "hardware_check.json"

# The child's copy buffer, and what a card must have free beyond it for the child's CUDA
# context and allocator slack. A card with less is skipped rather than pushed to OOM.
BUFFER_MIB = 256
CONTEXT_ALLOWANCE_MIB = 512
# Each timed copy loop runs about this long (an x1 link moves ~0.75 GiB in it).
TIMED_BUDGET_S = 0.5
# The whole child, torch import included. A cold import on a slow disk can take most of it.
PROBE_TIMEOUT_S = 60.0
STORAGE_TIMEOUT_S = 20.0
LINK_READ_TIMEOUT_S = 5.0
# A second link reading this long after a narrow first one, in case the link was still
# training up when the first was taken.
LINK_RECHECK_DELAY_S = 0.2

# A link is slow when it trained below half its width, or when it moves less than a quarter of
# what the best card moves from system memory.
SLOW_WIDTH_FRACTION = 0.5
SLOW_BANDWIDTH_FRACTION = 0.25

# Auto-run: how long after the startup warm it waits, how often it retries while busy, and when
# it gives up for this process.
AUTO_SETTLE_DELAY_S = 20.0
AUTO_RETRY_INTERVAL_S = 60.0
AUTO_GIVE_UP_S = 3600.0
AUTO_RUN_ENV_VAR = "UNSLOTH_HARDWARE_CHECK_AUTORUN"

# Reasons a run does not start. A wire contract: the UI keys its copy off these.
SKIP_RUNNING = "already_running"
SKIP_TRAINING = "training_active"
SKIP_LOADING = "model_loading"
SKIP_GENERATING = "generation_active"

TRIGGER_AUTO = "auto"
TRIGGER_MANUAL = "manual"

_PREFERENCE_TTL_S = 2.0

_MIB = 1024 * 1024
_GIB = 1024 * _MIB

_PROBE_SCRIPT = Path(__file__).with_name("hardware_check_probe.py")


# --------------------------------------------------------------------------------------------
# Small helpers


def _iso(timestamp: Optional[float] = None) -> str:
    moment = datetime.fromtimestamp(time.time() if timestamp is None else timestamp, tz = timezone.utc)
    return moment.replace(microsecond = 0).isoformat().replace("+00:00", "Z")


def _int_or_none(raw: Any) -> Optional[int]:
    try:
        text = str(raw).strip()
        if not text or text.startswith("["):
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _text_or_none(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    return None if not text or text.startswith("[") else text


def _hidden_kwargs() -> dict[str, Any]:
    try:
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        return windows_hidden_subprocess_kwargs()
    except Exception:  # noqa: BLE001
        return {}


def _child_env() -> dict[str, str]:
    try:
        from utils.native_path_leases import child_env_without_native_path_secret

        env = child_env_without_native_path_secret()
    except Exception:  # noqa: BLE001
        env = dict(os.environ)
    try:
        from utils.child_stdio import utf8_child_env

        env = utf8_child_env(env)
    except Exception:  # noqa: BLE001
        env["PYTHONIOENCODING"] = "utf-8"
    return env


def is_windows() -> bool:
    return platform.system() == "Windows"


# --------------------------------------------------------------------------------------------
# nvidia-smi: inventory, fingerprint and the link under load

_INVENTORY_FIELDS = (
    "index",
    "uuid",
    "pci.bus_id",
    "memory.total",
    "memory.free",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "name",
)
# All static to gpu_query, so this read is shared and cached for 60 s: cheap on the load path.
_FINGERPRINT_FIELDS = ("index", "uuid", "pci.bus_id", "name")
_LINK_FIELDS = (
    "index",
    "uuid",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
)


class NoNvidiaSmi(Exception):
    """This host has no nvidia-smi: no NVIDIA GPU to measure."""


def _smi_exe() -> Optional[str]:
    from utils.hardware.gpu_resources import _nvidia_smi

    return _nvidia_smi()


def _run_smi(fields: Sequence[str], *, fresh: bool, timeout: float = LINK_READ_TIMEOUT_S) -> str:
    """stdout of one ``--query-gpu`` read through the shared runner. Raises on failure."""
    exe = _smi_exe()
    if exe is None:
        raise NoNvidiaSmi()
    from utils.hardware import gpu_query

    result = gpu_query.run_nvidia_smi(
        [exe, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        timeout = timeout,
        cache = not fresh,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = _child_env(),
        **_hidden_kwargs(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi exited with code {result.returncode}")
    return result.stdout or ""


def parse_smi_rows(stdout: str, fields: Sequence[str]) -> list[dict[str, Any]]:
    """One dict per CSV row, keyed by field; ``name`` (last) rejoined if it held commas."""
    rows: list[dict[str, Any]] = []
    for line in (stdout or "").strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(fields):
            continue
        values = dict(zip(fields, parts))
        if fields[-1] == "name":
            values["name"] = ", ".join(parts[len(fields) - 1 :])
        index = _int_or_none(values.get("index"))
        if index is None:
            continue
        values["index"] = index
        rows.append(values)
    rows.sort(key = lambda row: row["index"])
    return rows


def _link_from_row(row: Mapping[str, Any]) -> dict[str, Optional[int]]:
    return {
        "gen_current": _int_or_none(row.get("pcie.link.gen.current")),
        "gen_max": _int_or_none(row.get("pcie.link.gen.max")),
        "width_current": _int_or_none(row.get("pcie.link.width.current")),
        "width_max": _int_or_none(row.get("pcie.link.width.max")),
    }


def read_inventory() -> list[dict[str, Any]]:
    """Every NVIDIA GPU: identity, memory and the link at rest. Raises NoNvidiaSmi / on failure."""
    rows = parse_smi_rows(_run_smi(_INVENTORY_FIELDS, fresh = True, timeout = 10.0), _INVENTORY_FIELDS)
    out = []
    for row in rows:
        total = _int_or_none(row.get("memory.total"))
        free = _int_or_none(row.get("memory.free"))
        out.append(
            {
                "index": row["index"],
                "uuid": _text_or_none(row.get("uuid")),
                "pci_bus_id": _text_or_none(row.get("pci.bus_id")),
                "name": _text_or_none(row.get("name")),
                "memory_total_bytes": total * _MIB if total is not None else None,
                "memory_free_bytes": free * _MIB if free is not None else None,
                "link_idle": _link_from_row(row),
            }
        )
    return out


def fingerprint_of(devices: Iterable[Mapping[str, Any]]) -> str:
    """Stable id for a set of GPUs: their UUIDs and PCI bus ids, order-independent."""
    keys = sorted(
        f"{str(d.get('uuid') or '').upper()}@{str(d.get('pci_bus_id') or '').upper()}" for d in devices
    )
    if not keys:
        return "no-nvidia-gpu"
    return hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()[:16]


def current_fingerprint() -> Optional[str]:
    """The fingerprint of the GPUs installed now, or None when nvidia-smi could not answer.

    Uses only static fields, so the read is gpu_query's 60 s cache on every call after the first.
    """
    try:
        rows = parse_smi_rows(_run_smi(_FINGERPRINT_FIELDS, fresh = False, timeout = 10.0), _FINGERPRINT_FIELDS)
    except NoNvidiaSmi:
        return fingerprint_of(())
    except Exception as exc:  # noqa: BLE001 -- unknown, never "changed"
        logger.debug("Hardware check: fingerprint read failed: %s", exc)
        return None
    return fingerprint_of(
        {"uuid": _text_or_none(r.get("uuid")), "pci_bus_id": _text_or_none(r.get("pci.bus_id"))}
        for r in rows
    )


def read_links_now() -> dict[str, dict[str, Optional[int]]]:
    """``{uuid: link}`` read fresh (a load is running on one of them). Raises on failure."""
    rows = parse_smi_rows(_run_smi(_LINK_FIELDS, fresh = True), _LINK_FIELDS)
    return {str(r.get("uuid")): _link_from_row(r) for r in rows if _text_or_none(r.get("uuid"))}


def read_link_under_load(uuid: str, *, reader: Callable[[], Mapping[str, Mapping[str, Any]]] = read_links_now,
                         sleep: Callable[[float], None] = time.sleep) -> Optional[dict[str, Optional[int]]]:
    """The link of one card while the child copies to it: the widest of two readings when the
    first is narrower than the card's maximum (the link may still have been training up)."""
    try:
        first = dict(reader().get(uuid) or {})
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hardware check: link read failed: %s", exc)
        return None
    if not first:
        return None
    width, width_max = first.get("width_current"), first.get("width_max")
    gen, gen_max = first.get("gen_current"), first.get("gen_max")
    if (width is not None and width_max is not None and width < width_max) or (
        gen is not None and gen_max is not None and gen < gen_max
    ):
        sleep(LINK_RECHECK_DELAY_S)
        try:
            second = reader().get(uuid) or {}
        except Exception:  # noqa: BLE001
            second = {}
        for key in ("gen_current", "width_current"):
            value = second.get(key)
            if value is not None and (first.get(key) is None or value > first[key]):
                first[key] = value
    return first


# --------------------------------------------------------------------------------------------
# The GPU child


def gpu_probe_argv(uuids: Sequence[str], *, python: Optional[str] = None) -> list[str]:
    return [
        python or sys.executable,
        "-I",
        str(_PROBE_SCRIPT),
        "--uuids",
        ",".join(uuids),
        "--buffer-mib",
        str(BUFFER_MIB),
        "--budget-s",
        str(TIMED_BUDGET_S),
    ]


def _probe_popen_kwargs() -> dict[str, Any]:
    kwargs = dict(_hidden_kwargs())
    if sys.platform == "win32":
        below = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
        kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | below
    env = _child_env()
    # The child names cards by UUID, so its ordinals never leave it; PCI order only keeps the
    # log lines readable.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    kwargs["env"] = env
    return kwargs


def run_gpu_probe(
    devices: Sequence[Mapping[str, Any]],
    *,
    link_reader: Callable[[str], Optional[Mapping[str, Any]]] = read_link_under_load,
    popen: Callable[..., Any] = subprocess.Popen,
    timeout_s: float = PROBE_TIMEOUT_S,
    progress: Callable[[str, float], None] = lambda phase, fraction: None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Measure ``devices`` (inventory rows) in the child. Never raises.

    Returns ``{"gpus": {uuid: {...}}, "pairs": [...], "error": Optional[str]}``. A card skipped
    for lack of free memory is not sent to the child at all.
    """
    out: dict[str, Any] = {"gpus": {}, "pairs": [], "error": None}
    need = (BUFFER_MIB + CONTEXT_ALLOWANCE_MIB) * _MIB
    targets: list[str] = []
    for device in devices:
        uuid = device.get("uuid")
        if not uuid:
            continue
        free = device.get("memory_free_bytes")
        if free is not None and free < need:
            out["gpus"][uuid] = {"status": "skipped", "reason": "low_free_memory"}
            continue
        targets.append(uuid)
        out["gpus"][uuid] = {"status": "pending"}
    if not targets:
        return out

    try:
        proc = popen(
            gpu_probe_argv(targets),
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            bufsize = 1,
            **_probe_popen_kwargs(),
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"could not start the measurement: {type(exc).__name__}"
        for uuid in targets:
            out["gpus"][uuid] = {"status": "failed", "reason": "probe_failed"}
        return out

    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    stderr_tail: list[str] = []

    def _pump_stdout() -> None:
        try:
            for line in proc.stdout:
                lines.put(line)
        except Exception:  # noqa: BLE001
            pass
        lines.put(None)

    def _pump_stderr() -> None:
        try:
            for line in proc.stderr:
                stderr_tail.append(line.rstrip())
                del stderr_tail[:-20]
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target = _pump_stdout, name = "hardware-check-out", daemon = True).start()
    if getattr(proc, "stderr", None) is not None:
        threading.Thread(target = _pump_stderr, name = "hardware-check-err", daemon = True).start()

    deadline = clock() + max(1.0, timeout_s)
    finished = False
    measured = 0
    try:
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                out["error"] = "the measurement timed out"
                break
            try:
                line = lines.get(timeout = min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                break
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("event")
            uuid = event.get("uuid")
            if kind == "load" and uuid in out["gpus"]:
                progress("gpus", measured / max(1, len(targets)))
                link = link_reader(uuid)
                if link:
                    out["gpus"][uuid]["link"] = dict(link)
                try:
                    proc.stdin.write(f"go {uuid}\n")
                    proc.stdin.flush()
                except Exception:  # noqa: BLE001 -- the child stops holding on its own
                    pass
            elif kind == "bandwidth" and uuid in out["gpus"]:
                measured += 1
                entry = out["gpus"][uuid]
                entry["status"] = "measured"
                entry["h2d_gibs"] = event.get("h2d_gibs")
                entry["d2h_gibs"] = event.get("d2h_gibs")
                progress("gpus", measured / max(1, len(targets)))
            elif kind == "device_error" and uuid in out["gpus"]:
                reason = str(event.get("reason") or "")
                out["gpus"][uuid].update(
                    status = "skipped" if "not visible" in reason else "failed",
                    reason = "not_visible" if "not visible" in reason else "probe_failed",
                    detail = reason[:200],
                )
            elif kind == "pair":
                out["pairs"].append(
                    {
                        "a": event.get("a"),
                        "b": event.get("b"),
                        "peer_ab": event.get("peer_ab"),
                        "peer_ba": event.get("peer_ba"),
                        "copy_ab_gibs": event.get("copy_ab_gibs"),
                        "copy_ba_gibs": event.get("copy_ba_gibs"),
                    }
                )
            elif kind == "error":
                out["error"] = str(event.get("reason") or "the measurement failed")[:240]
            elif kind == "done":
                finished = True
                break
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout = 5.0 if finished else 0.5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
                proc.wait(timeout = 5.0)
            except Exception:  # noqa: BLE001
                pass
    if not finished and out["error"] is None:
        code = getattr(proc, "returncode", None)
        tail = " ".join(stderr_tail[-3:])[:200]
        out["error"] = f"the measurement process exited early (code {code})" + (f": {tail}" if tail else "")
    for uuid in targets:
        entry = out["gpus"][uuid]
        if entry.get("status") == "pending":
            entry["status"] = "failed"
            entry["reason"] = "probe_failed"
    return out


# --------------------------------------------------------------------------------------------
# Storage

_DRIVE_LETTER_RE = re.compile(r"^([A-Za-z]):")

_STORAGE_PS = r"""
$ErrorActionPreference='SilentlyContinue'
$letters=@(__LETTERS__)
$phys=@{}
$disks=@()
foreach($p in @(Get-PhysicalDisk)){ $phys[[string]$p.DeviceId]=$p; $disks += [pscustomobject]@{id=[string]$p.DeviceId; bus=[string]$p.BusType; media=[string]$p.MediaType; model=[string]$p.FriendlyName} }
$drives=@()
foreach($l in $letters){
  $part=Get-Partition -DriveLetter $l | Select-Object -First 1
  if($part){
    $d=Get-Disk -Number $part.DiskNumber
    $p=$phys[[string]$part.DiskNumber]
    $drives += [pscustomobject]@{letter=$l; disk=[string]$part.DiskNumber; bus=[string]$d.BusType; media=[string]$p.MediaType; model=[string]$d.FriendlyName}
  } else { $drives += [pscustomobject]@{letter=$l; disk=$null} }
}
[pscustomobject]@{drives=$drives; disks=$disks} | ConvertTo-Json -Depth 4 -Compress
"""


def storage_locations() -> list[tuple[str, Path]]:
    """Where Studio keeps what it loads: its home, the Hugging Face hub cache and temp."""
    out: list[tuple[str, Path]] = []
    try:
        from utils.paths.storage_roots import studio_root

        out.append(("studio_home", Path(studio_root())))
    except Exception:  # noqa: BLE001
        pass
    try:
        from utils.hf_cache_settings import get_hf_cache_paths

        out.append(("hf_cache", Path(get_hf_cache_paths().hub_cache)))
    except Exception:  # noqa: BLE001
        pass
    out.append(("temp", Path(tempfile.gettempdir())))
    return out


def normalize_bus(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    upper = text.upper()
    for known in ("NVME", "SATA", "USB", "SAS", "RAID", "SCSI", "SD", "MMC", "ATA", "VIRTUAL"):
        if upper == known:
            return {"NVME": "NVMe", "VIRTUAL": "Virtual"}.get(known, known)
    if "STORAGE" in upper and "SPACE" in upper:
        return "Storage Spaces"
    if upper in ("FILE BACKED VIRTUAL", "FILEBACKEDVIRTUAL"):
        return "Virtual"
    return text


def normalize_media(raw: Any) -> Optional[str]:
    upper = str(raw or "").strip().upper()
    if upper in ("SSD", "HDD", "SCM"):
        return upper
    return None


def _windows_storage(letters: Sequence[str], *, run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    letters = sorted({letter.upper() for letter in letters if re.fullmatch(r"[A-Za-z]", letter)})
    script = _STORAGE_PS.replace("__LETTERS__", ",".join(f"'{letter}'" for letter in letters) or "''")
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = STORAGE_TIMEOUT_S,
        **_hidden_kwargs(),
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        raise RuntimeError(f"PowerShell exited with code {result.returncode}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    drives_raw = payload.get("drives") or []
    disks_raw = payload.get("disks") or []
    if isinstance(drives_raw, dict):
        drives_raw = [drives_raw]
    if isinstance(disks_raw, dict):
        disks_raw = [disks_raw]
    drives = {}
    for row in drives_raw:
        if not isinstance(row, dict) or not row.get("letter"):
            continue
        drives[str(row["letter"]).upper()] = {
            "bus_type": normalize_bus(row.get("bus")) if row.get("disk") is not None else None,
            "media_type": normalize_media(row.get("media")),
            "model": _text_or_none(row.get("model")),
        }
    disks = [
        {"bus_type": normalize_bus(d.get("bus")), "media_type": normalize_media(d.get("media"))}
        for d in disks_raw
        if isinstance(d, dict)
    ]
    return {"drives": drives, "disks": disks}


def _linux_block_info(path: Path) -> dict[str, Any]:
    """Bus and media for the block device behind ``path``, from sysfs."""
    st = os.stat(path)
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    node = Path(os.path.realpath(f"/sys/dev/block/{major}:{minor}"))
    disk = node
    if not (disk / "queue" / "rotational").exists() and (disk.parent / "queue" / "rotational").exists():
        disk = disk.parent
    name = disk.name
    rotational = None
    try:
        rotational = (disk / "queue" / "rotational").read_text().strip() == "1"
    except OSError:
        pass
    if name.startswith("nvme"):
        bus = "NVMe"
    elif "/usb" in str(disk):
        bus = "USB"
    elif name.startswith(("mmcblk",)):
        bus = "SD"
    elif name.startswith(("sd", "hd")):
        bus = "SATA"
    else:
        bus = None
    return {
        "bus_type": bus,
        "media_type": None if rotational is None else ("HDD" if rotational else "SSD"),
        "model": None,
    }


def probe_storage(
    locations: Optional[Sequence[tuple[str, Path]]] = None,
    *,
    windows_query: Callable[[Sequence[str]], dict[str, Any]] = _windows_storage,
    system: Optional[str] = None,
) -> dict[str, Any]:
    """``{"locations": [...], "nvme_present": bool|None, "error": str|None}``. Never raises."""
    locations = list(storage_locations() if locations is None else locations)
    system = system or platform.system()
    entries: list[dict[str, Any]] = []
    for key, path in locations:
        match = _DRIVE_LETTER_RE.match(str(path))
        entries.append(
            {
                "key": key,
                "path": str(path),
                "drive": f"{match.group(1).upper()}:" if match else None,
                "bus_type": None,
                "media_type": None,
                "model": None,
            }
        )
    out: dict[str, Any] = {"locations": entries, "nvme_present": None, "error": None}
    try:
        if system == "Windows":
            letters = [e["drive"][0] for e in entries if e["drive"]]
            for entry in entries:
                if entry["drive"] is None and str(entry["path"]).startswith(("\\\\", "//")):
                    entry["bus_type"] = "Network"
            info = windows_query(letters)
            for entry in entries:
                if entry["drive"]:
                    entry.update(info["drives"].get(entry["drive"][0], {}))
            disks = info.get("disks") or []
            out["nvme_present"] = any(d.get("bus_type") == "NVMe" for d in disks) if disks else None
        elif system == "Linux":
            for entry in entries:
                candidate = Path(entry["path"])
                while not candidate.exists() and candidate.parent != candidate:
                    candidate = candidate.parent
                try:
                    entry.update(_linux_block_info(candidate))
                except Exception as exc:  # noqa: BLE001
                    entry["error"] = type(exc).__name__
            try:
                out["nvme_present"] = any(name.startswith("nvme") for name in os.listdir("/sys/block"))
            except OSError:
                out["nvme_present"] = None
        else:
            out["error"] = "not supported on this system"
    except Exception as exc:  # noqa: BLE001 -- storage facts are a courtesy
        logger.debug("Hardware check: storage query failed: %s", exc)
        out["error"] = f"{type(exc).__name__}"
    return out


# --------------------------------------------------------------------------------------------
# One run


def run_check(
    trigger: str = TRIGGER_MANUAL,
    *,
    inventory_reader: Callable[[], list[dict[str, Any]]] = read_inventory,
    gpu_probe: Callable[..., dict[str, Any]] = run_gpu_probe,
    storage_probe: Callable[[], dict[str, Any]] = probe_storage,
    progress: Callable[[str, float], None] = lambda phase, fraction: None,
) -> dict[str, Any]:
    """Every measurement, as the JSON that is stored. Never raises: what fails is said inside."""
    started = time.time()
    t0 = time.monotonic()
    result: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "trigger": trigger,
        "started_at": _iso(started),
        "platform": platform.system(),
        "gpus": [],
        "pairs": [],
        "storage": [],
        "nvme_present": None,
        "gpu_error": None,
        "storage_error": None,
    }

    storage_box: dict[str, Any] = {}

    def _storage() -> None:
        storage_box["value"] = storage_probe()

    storage_thread = threading.Thread(target = _storage, name = "hardware-check-storage", daemon = True)
    storage_thread.start()

    progress("gpus", 0.0)
    devices: list[dict[str, Any]] = []
    try:
        devices = inventory_reader()
    except NoNvidiaSmi:
        devices = []
    except Exception as exc:  # noqa: BLE001
        result["gpu_error"] = f"could not read the GPUs: {type(exc).__name__}"
    result["fingerprint"] = fingerprint_of(devices)

    measured: dict[str, Any] = {"gpus": {}, "pairs": [], "error": None}
    if devices:
        measured = gpu_probe(devices, progress = progress)
        result["gpu_error"] = result["gpu_error"] or measured.get("error")
    by_uuid_index = {d.get("uuid"): d.get("index") for d in devices}
    for device in devices:
        entry = {
            "index": device.get("index"),
            "uuid": device.get("uuid"),
            "pci_bus_id": device.get("pci_bus_id"),
            "name": device.get("name"),
            "memory_total_bytes": device.get("memory_total_bytes"),
            "memory_free_bytes": device.get("memory_free_bytes"),
            "link_idle": device.get("link_idle"),
            "status": "failed",
            "reason": None,
            "link": None,
            "h2d_gibs": None,
            "d2h_gibs": None,
        }
        entry.update(measured.get("gpus", {}).get(device.get("uuid"), {}))
        entry.pop("detail", None)
        result["gpus"].append(entry)
    for pair in measured.get("pairs") or []:
        a, b = by_uuid_index.get(pair.get("a")), by_uuid_index.get(pair.get("b"))
        if a is None or b is None:
            continue
        result["pairs"].append(
            {
                "a": a,
                "b": b,
                "peer_ab": pair.get("peer_ab"),
                "peer_ba": pair.get("peer_ba"),
                "copy_ab_gibs": pair.get("copy_ab_gibs"),
                "copy_ba_gibs": pair.get("copy_ba_gibs"),
            }
        )

    progress("storage", 1.0)
    storage_thread.join(STORAGE_TIMEOUT_S + 5.0)
    storage = storage_box.get("value") or {"locations": [], "nvme_present": None, "error": "timed out"}
    result["storage"] = storage.get("locations") or []
    result["nvme_present"] = storage.get("nvme_present")
    result["storage_error"] = storage.get("error")
    result["finished_at"] = _iso()
    result["duration_ms"] = int((time.monotonic() - t0) * 1000)
    return result


# --------------------------------------------------------------------------------------------
# Findings (pure)


def _measured(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        g
        for g in result.get("gpus") or []
        if g.get("status") == "measured" and isinstance(g.get("h2d_gibs"), (int, float))
    ]


def slow_link_gpus(result: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """``{index: why}`` for every measured card whose link is slow: trained below half its width
    under load, or under a quarter of the best card's host-to-device bandwidth."""
    measured = _measured(result)
    if not measured:
        return {}
    best = max(measured, key = lambda g: float(g["h2d_gibs"]))
    best_gibs = float(best["h2d_gibs"])
    out: dict[int, dict[str, Any]] = {}
    for gpu in measured:
        link = gpu.get("link") or {}
        width, width_max = link.get("width_current"), link.get("width_max")
        narrow = (
            isinstance(width, int)
            and isinstance(width_max, int)
            and width_max > 0
            and width < width_max * SLOW_WIDTH_FRACTION
        )
        starved = (
            len(measured) > 1
            and gpu is not best
            and best_gibs > 0
            and float(gpu["h2d_gibs"]) < best_gibs * SLOW_BANDWIDTH_FRACTION
        )
        if narrow or starved:
            out[int(gpu["index"])] = {"narrow": narrow, "starved": starved}
    return out


def _round(value: Any, digits: int = 1) -> Optional[float]:
    return round(float(value), digits) if isinstance(value, (int, float)) else None


_LOCATION_TEXT = {
    "studio_home": "The Studio home",
    "hf_cache": "The Hugging Face cache",
    "temp": "The temp folder",
}


def analyze(result: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Findings and recommended options for a stored result. Pure.

    Each finding is ``{"id", "severity" ("warning" | "info" | "ok"), "values", "text"}``: the UI
    words it from ``id`` and ``values`` in the user's language, ``text`` is the English sentence
    for logs, API callers and the diagnostics report.
    """
    from utils.hardware_check_settings import (
        AVOID_TENSOR_SPLIT,
        PREFER_FAST_LINK,
        WARN_TRAINING_SLOW_LINK,
    )

    findings: list[dict[str, Any]] = []
    recommended = {PREFER_FAST_LINK: False, AVOID_TENSOR_SPLIT: False, WARN_TRAINING_SLOW_LINK: False}
    if not result:
        return {"findings": findings, "recommended": recommended, "slow_gpus": []}

    measured = _measured(result)
    slow = slow_link_gpus(result)
    best = max(measured, key = lambda g: float(g["h2d_gibs"])) if measured else None
    windows = result.get("platform") == "Windows"

    for gpu in measured:
        index = int(gpu["index"])
        if index not in slow:
            continue
        link = gpu.get("link") or {}
        h2d = float(gpu["h2d_gibs"])
        values: dict[str, Any] = {
            "gpu": index,
            "width": link.get("width_current"),
            "width_max": link.get("width_max"),
            "gen": link.get("gen_current"),
            "h2d_gibs": _round(h2d),
            "best_gpu": None,
            "best_h2d_gibs": None,
            "ratio": None,
        }
        width_text = (
            f"PCIe x{values['width']}" + (f" (of x{values['width_max']})" if values["width_max"] else "")
            if values["width"]
            else "a slow PCIe link"
        )
        if best is not None and best is not gpu and float(best["h2d_gibs"]) > h2d > 0:
            ratio = float(best["h2d_gibs"]) / h2d
            values.update(
                best_gpu = int(best["index"]),
                best_h2d_gibs = _round(best["h2d_gibs"]),
                ratio = int(round(ratio)),
            )
            text = (
                f"GPU {index} runs at {width_text}: {values['h2d_gibs']} GiB/s from system memory "
                f"vs {values['best_h2d_gibs']} GiB/s on GPU {values['best_gpu']}, so loading a model "
                f"onto it is ~{values['ratio']}x slower."
            )
        else:
            text = (
                f"GPU {index} runs at {width_text}: {values['h2d_gibs']} GiB/s from system memory, "
                "so loading a model onto it is slow."
            )
        findings.append({"id": "slow_link", "severity": "warning", "values": values, "text": text})

    measured_ids = {int(g["index"]) for g in measured}
    no_peer: list[tuple[int, int]] = []
    for pair in result.get("pairs") or []:
        a, b = pair.get("a"), pair.get("b")
        if a not in measured_ids or b not in measured_ids:
            continue
        if pair.get("peer_ab") is False or pair.get("peer_ba") is False:
            no_peer.append((int(a), int(b)))
            copies = [
                float(v)
                for v in (pair.get("copy_ab_gibs"), pair.get("copy_ba_gibs"))
                if isinstance(v, (int, float))
            ]
            copy = _round(min(copies)) if copies else None
            values = {"a": int(a), "b": int(b), "copy_gibs": copy, "windows": windows}
            text = (
                f"GPU {a} and GPU {b} cannot access each other directly"
                f"{' (Windows)' if windows else ''}: tensor parallelism moves every layer's results "
                "through system memory"
                + (f" ({copy} GiB/s between them)." if copy is not None else ".")
            )
            findings.append({"id": "no_peer_access", "severity": "warning", "values": values, "text": text})

    for gpu in result.get("gpus") or []:
        if gpu.get("status") == "measured":
            continue
        reason = gpu.get("reason") or "probe_failed"
        values = {
            "gpu": gpu.get("index"),
            "reason": reason,
            "free_gib": _round((gpu.get("memory_free_bytes") or 0) / _GIB),
        }
        if reason == "low_free_memory":
            text = (
                f"GPU {gpu.get('index')} was not measured: only {values['free_gib']} GiB was free. "
                "Unload what is on it and run the check again."
            )
        elif reason == "not_visible":
            text = f"GPU {gpu.get('index')} was not measured: Studio is set not to use it."
        else:
            text = f"GPU {gpu.get('index')} could not be measured."
        findings.append({"id": "gpu_skipped", "severity": "info", "values": values, "text": text})

    if result.get("gpu_error") and not measured:
        findings.append(
            {
                "id": "gpu_probe_failed",
                "severity": "warning",
                "values": {"reason": str(result["gpu_error"])},
                "text": f"The GPU measurement did not finish: {result['gpu_error']}.",
            }
        )

    nvme_present = result.get("nvme_present") is True
    for entry in result.get("storage") or []:
        key = entry.get("key")
        bus, media = entry.get("bus_type"), entry.get("media_type")
        kind = None
        if bus == "USB":
            kind = "usb"
        elif media == "HDD":
            kind = "hdd"
        elif key == "hf_cache" and bus in ("SATA", "SAS", "ATA") and nvme_present:
            kind = "sata"
        if kind is None or (key != "hf_cache" and kind == "sata"):
            continue
        where = _LOCATION_TEXT.get(key, "A Studio folder")
        drive = entry.get("drive")
        on = f" ({drive})" if drive else ""
        text = {
            "usb": f"{where}{on} is on a USB drive; an internal SSD loads models much faster.",
            "hdd": f"{where}{on} is on a hard disk; an SSD loads models much faster.",
            "sata": f"{where}{on} is on a SATA SSD; an NVMe drive loads models faster.",
        }[kind]
        findings.append(
            {
                "id": "storage_slow",
                "severity": "info" if kind == "sata" or key == "temp" else "warning",
                "values": {"location": key, "drive": drive, "kind": kind, "model": entry.get("model")},
                "text": text,
            }
        )

    multi = len(measured) >= 2
    recommended[PREFER_FAST_LINK] = multi and bool(slow)
    recommended[AVOID_TENSOR_SPLIT] = multi and (bool(slow) or bool(no_peer))
    recommended[WARN_TRAINING_SLOW_LINK] = bool(slow)

    if not any(f["severity"] in ("warning", "info") for f in findings):
        findings.append(
            {
                "id": "all_good",
                "severity": "ok",
                "values": {"gpus": len(measured)},
                "text": "All good: every measured GPU and drive is connected as fast as it can be.",
            }
        )
    return {"findings": findings, "recommended": recommended, "slow_gpus": sorted(slow)}


# --------------------------------------------------------------------------------------------
# Persistence


def result_path() -> Path:
    from utils.paths.storage_roots import studio_root

    return Path(studio_root()) / RESULT_FILENAME


_result_lock = threading.Lock()
_result_cache: Optional[tuple[str, float, Optional[dict[str, Any]]]] = None


def load_result() -> Optional[dict[str, Any]]:
    """The stored result, or None (none yet, unreadable, or another schema). Memoised on mtime."""
    global _result_cache
    path = result_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    with _result_lock:
        cached = _result_cache
        if cached is not None and cached[0] == str(path) and cached[1] == mtime:
            return cached[2]
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Hardware check: stored result unreadable: %s", exc)
        data = None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
        data = None
    with _result_lock:
        _result_cache = (str(path), mtime, data)
    return data


def save_result(result: Mapping[str, Any]) -> None:
    global _result_cache
    path = result_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent = 2) + "\n", encoding = "utf-8")
    os.replace(tmp, path)
    with _result_lock:
        _result_cache = None
    _invalidate_preference()


def result_is_current(
    result: Optional[Mapping[str, Any]] = None, fingerprint: Optional[str] = ...  # type: ignore[assignment]
) -> Optional[bool]:
    """True when the stored result describes the GPUs installed now, False when it does not (or
    there is none), None when the current hardware could not be read."""
    result = load_result() if result is None else result
    if not result:
        return False
    current = current_fingerprint() if fingerprint is ... else fingerprint
    if current is None:
        return None
    return result.get("fingerprint") == current


# --------------------------------------------------------------------------------------------
# What the options read


_preference_lock = threading.Lock()
_preference_cache: Optional[tuple[float, Optional[dict[str, Any]]]] = None


def _invalidate_preference() -> None:
    global _preference_cache
    with _preference_lock:
        _preference_cache = None


def _current_result() -> Optional[dict[str, Any]]:
    """The stored result when it describes the GPUs installed now, else None. Memoised briefly:
    the load path asks several times per load."""
    global _preference_cache
    now = time.monotonic()
    with _preference_lock:
        cached = _preference_cache
        if cached is not None and now - cached[0] < _PREFERENCE_TTL_S:
            return cached[1]
    result = load_result()
    current = result if result and result_is_current(result) is True else None
    with _preference_lock:
        _preference_cache = (now, current)
    return current


def _bandwidth_map(result: Mapping[str, Any]) -> dict[int, float]:
    return {int(g["index"]): float(g["h2d_gibs"]) for g in _measured(result)}


def active_link_preference() -> Optional[dict[int, float]]:
    """``{gpu index: measured host-to-device GiB/s}`` for automatic placement, or None.

    None unless "Prefer fast-link GPUs" is on and the stored result describes the GPUs installed
    now, so a caller handed None keeps its behaviour exactly. Never raises.
    """
    try:
        from utils.hardware_check_settings import PREFER_FAST_LINK, setting_enabled

        if not setting_enabled(PREFER_FAST_LINK):
            return None
        result = _current_result()
        if not result:
            return None
        bandwidth = _bandwidth_map(result)
        return bandwidth if len(bandwidth) >= 2 else None
    except Exception as exc:  # noqa: BLE001 -- placement must never fail over a preference
        logger.debug("Hardware check: link preference unavailable: %s", exc)
        return None


def tensor_split_concern(gpu_ids: Optional[Sequence[int]] = None, *, require_option: bool = True) -> Optional[dict[str, Any]]:
    """Why tensor parallelism across ``gpu_ids`` (every measured card when None) is a poor
    choice here, or None. ``{"slow_gpus": [...], "no_peer_pairs": [[a, b], ...]}``.

    With ``require_option`` (the default) None unless "Avoid tensor parallel on slow links" is
    on. Never raises.
    """
    try:
        if require_option:
            from utils.hardware_check_settings import AVOID_TENSOR_SPLIT, setting_enabled

            if not setting_enabled(AVOID_TENSOR_SPLIT):
                return None
        result = _current_result()
        if not result:
            return None
        measured = {int(g["index"]) for g in _measured(result)}
        span = measured if not gpu_ids else {int(i) for i in gpu_ids} & measured
        if len(span) < 2:
            return None
        slow = sorted(i for i in slow_link_gpus(result) if i in span)
        no_peer = sorted(
            [int(p["a"]), int(p["b"])]
            for p in result.get("pairs") or []
            if p.get("a") in span
            and p.get("b") in span
            and (p.get("peer_ab") is False or p.get("peer_ba") is False)
        )
        if not slow and not no_peer:
            return None
        return {"slow_gpus": slow, "no_peer_pairs": no_peer}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hardware check: tensor split concern unavailable: %s", exc)
        return None


def env_tensor_split_avoided(
    requested_tensor_parallel: bool,
    extra_args: Optional[Iterable[str]],
    gpu_ids: Optional[Sequence[int]],
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """True when tensor mode would come only from an inherited ``LLAMA_ARG_SPLIT_MODE=tensor``
    (no toggle, no ``--split-mode`` in the extras) and the option says to avoid it here.

    The one automatic way llama.cpp tensor parallelism is chosen; the toggle and an explicit
    ``--split-mode`` are the user's and are never overridden.
    """
    if requested_tensor_parallel:
        return False
    try:
        from core.inference.llama_server_args import (
            _env_split_mode_is_tensor,
            parse_split_mode_override,
        )

        if parse_split_mode_override(list(extra_args) if extra_args else None) is not None:
            return False
        if not _env_split_mode_is_tensor(env):
            return False
    except Exception:  # noqa: BLE001
        return False
    return tensor_split_concern(gpu_ids) is not None


def training_slow_link_warning(gpu_ids: Sequence[int], gradient_checkpointing: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The training fit panel's warning when the run would use a slow-link GPU, or None.

    None unless "Warn when training uses a slow-link GPU" is on and the result is current.
    """
    try:
        from utils.hardware_check_settings import WARN_TRAINING_SLOW_LINK, setting_enabled

        if not setting_enabled(WARN_TRAINING_SLOW_LINK):
            return None
        result = _current_result()
        if not result:
            return None
        slow = slow_link_gpus(result)
        hit = [int(i) for i in gpu_ids if int(i) in slow]
        if not hit:
            return None
        measured = _measured(result)
        by_index = {int(g["index"]): g for g in measured}
        best = max(measured, key = lambda g: float(g["h2d_gibs"]))
        gpus = []
        for index in hit:
            gpu = by_index[index]
            link = gpu.get("link") or {}
            gpus.append(
                {
                    "index": index,
                    "width": link.get("width_current"),
                    "width_max": link.get("width_max"),
                    "h2d_gibs": _round(gpu.get("h2d_gibs")),
                }
            )
        return {
            "gpus": gpus,
            "best_gpu": int(best["index"]),
            "best_h2d_gibs": _round(best["h2d_gibs"]),
            "offloaded_gradient_checkpointing": (gradient_checkpointing or "unsloth").strip().lower()
            == "unsloth",
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hardware check: training warning unavailable: %s", exc)
        return None


def resource_link_badges() -> dict[int, dict[str, Any]]:
    """``{gpu index: link facts}`` for the resource strip, for every card the current result
    measured (whatever the options). Empty without a current result. Never raises."""
    try:
        result = _current_result()
        if not result:
            return {}
        measured = _measured(result)
        if not measured:
            return {}
        slow = slow_link_gpus(result)
        best = max(measured, key = lambda g: float(g["h2d_gibs"]))
        out: dict[int, dict[str, Any]] = {}
        for gpu in measured:
            link = gpu.get("link") or {}
            out[int(gpu["index"])] = {
                "width": link.get("width_current"),
                "width_max": link.get("width_max"),
                "gen": link.get("gen_current"),
                "h2d_gibs": _round(gpu.get("h2d_gibs")),
                "best_gpu": int(best["index"]),
                "best_h2d_gibs": _round(best["h2d_gibs"]),
                "slow": int(gpu["index"]) in slow,
            }
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hardware check: link badges unavailable: %s", exc)
        return {}


# --------------------------------------------------------------------------------------------
# Gating: what a run must not compete with


def _training_active() -> bool:
    """Either trainer, read only when its module is already loaded (nothing loaded, nothing
    training): the check must not import the training stack to find out."""
    module = sys.modules.get("core.training.training")
    if module is not None:
        try:
            if module.get_training_backend().is_training_active():
                return True
        except Exception as exc:  # noqa: BLE001 -- unreadable is not evidence of idle
            logger.debug("Hardware check: training state unreadable: %s", exc)
            return True
    module = sys.modules.get("core.training.diffusion_training_service")
    if module is not None:
        try:
            if module.get_diffusion_training_service().is_active():
                return True
        except Exception:  # noqa: BLE001
            return True
    return False


def _model_loading() -> bool:
    """A load in flight anywhere, as the resource strip sees it."""
    resources = sys.modules.get("routes.resources")
    if resources is None:
        return False
    try:
        if any(model.loading for model in resources.collect_models()):
            return True
        return bool(resources._slots_loading())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hardware check: load state unreadable: %s", exc)
        return False


def _generation_active() -> bool:
    module = sys.modules.get("state.active_generations")
    if module is None:
        return False
    try:
        return module.count() > 0
    except Exception:  # noqa: BLE001
        return False


# Seams for tests.
_busy_checks: dict[str, Callable[[], bool]] = {
    SKIP_TRAINING: _training_active,
    SKIP_LOADING: _model_loading,
    SKIP_GENERATING: _generation_active,
}


def busy_reason(trigger: str) -> Optional[str]:
    """Why a run must not start now, or None. A manual run waits for nothing but training and
    loads; the background run also waits for generations to finish."""
    with _state_lock:
        if _state["running"]:
            return SKIP_RUNNING
    reasons = [SKIP_TRAINING, SKIP_LOADING]
    if trigger == TRIGGER_AUTO:
        reasons.append(SKIP_GENERATING)
    for reason in reasons:
        try:
            if _busy_checks[reason]():
                return reason
        except Exception:  # noqa: BLE001
            continue
    return None


# --------------------------------------------------------------------------------------------
# Running in the background, and the state the UI polls

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "trigger": None,
    "phase": None,
    "progress": 0.0,
    "started_at": None,
    "last_skip": None,
    "last_error": None,
    "auto": None,
}


def state_snapshot() -> dict[str, Any]:
    with _state_lock:
        return json.loads(json.dumps(_state))


def _set_state(**changes: Any) -> None:
    with _state_lock:
        _state.update(changes)


def _record_skip(trigger: str, reason: str) -> None:
    _set_state(last_skip = {"trigger": trigger, "reason": reason, "at": _iso()})


def _progress(phase: str, fraction: float) -> None:
    _set_state(phase = phase, progress = round(max(0.0, min(1.0, float(fraction))), 3))


# Seam for tests: what one run does.
_runner: Callable[..., dict[str, Any]] = run_check


def _execute(trigger: str) -> None:
    try:
        result = _runner(trigger, progress = _progress)
        save_result(result)
        _set_state(last_error = None)
        summary = analyze(result)
        logger.info(
            "Hardware check finished (%s) in %.1fs: %s",
            trigger,
            (result.get("duration_ms") or 0) / 1000.0,
            "; ".join(f["text"] for f in summary["findings"]),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hardware check failed: %s", exc)
        _set_state(last_error = f"{type(exc).__name__}: {exc}"[:240])
    finally:
        _set_state(running = False, phase = None, progress = 0.0, trigger = None, started_at = None)


def start_run(trigger: str = TRIGGER_MANUAL, *, wait: bool = False) -> dict[str, Any]:
    """Start one run unless something it must not compete with is going on.

    ``{"started": bool, "reason": Optional[str]}``; the run itself is on a daemon thread unless
    ``wait`` (the auto-runner, which is already on one).
    """
    reason = busy_reason(trigger)
    if reason is None:
        with _state_lock:
            if _state["running"]:
                reason = SKIP_RUNNING
            else:
                _state.update(
                    running = True,
                    trigger = trigger,
                    phase = "starting",
                    progress = 0.0,
                    started_at = _iso(),
                    last_skip = None,
                )
    if reason is not None:
        _record_skip(trigger, reason)
        return {"started": False, "reason": reason}
    if wait:
        _execute(trigger)
    else:
        threading.Thread(target = _execute, args = (trigger,), name = "hardware-check", daemon = True).start()
    return {"started": True, "reason": None}


def auto_run_wanted() -> bool:
    """Whether the background run should happen: the setting is on, the kill switch is not set,
    and there is no result for the GPUs installed now."""
    if os.environ.get(AUTO_RUN_ENV_VAR, "").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        from utils.hardware_check_settings import AUTO_RUN, setting_enabled

        if not setting_enabled(AUTO_RUN):
            return False
    except Exception:  # noqa: BLE001
        return False
    return result_is_current() is False


def auto_run_loop(
    alive: Callable[[], bool] = lambda: True,
    *,
    settle_s: float = AUTO_SETTLE_DELAY_S,
    retry_s: float = AUTO_RETRY_INTERVAL_S,
    give_up_s: float = AUTO_GIVE_UP_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Wait for startup to settle, then run once if wanted, retrying while Studio is busy.

    Returns what happened (``"ran"``, ``"not_needed"``, ``"stopped"``, ``"gave_up"``) for tests
    and the log. Called on its own daemon thread after the startup warm.
    """
    sleep(max(0.0, settle_s))
    started = clock()
    while True:
        if not alive():
            return "stopped"
        if not auto_run_wanted():
            _set_state(auto = "not_needed")
            return "not_needed"
        outcome = start_run(TRIGGER_AUTO, wait = True)
        if outcome["started"]:
            _set_state(auto = "ran")
            return "ran"
        _set_state(auto = f"waiting:{outcome['reason']}")
        if clock() - started >= give_up_s:
            _set_state(auto = "gave_up")
            return "gave_up"
        sleep(max(0.0, retry_s))


def start_auto_run(alive: Callable[[], bool] = lambda: True) -> Optional[threading.Thread]:
    """The background run's thread, or None when it is not wanted at all. Never raises."""
    try:
        if os.environ.get(AUTO_RUN_ENV_VAR, "").strip().lower() in ("0", "false", "no", "off"):
            return None
        thread = threading.Thread(
            target = auto_run_loop, args = (alive,), name = "hardware-check-auto", daemon = True
        )
        thread.start()
        return thread
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hardware check: auto-run did not start: %s", exc)
        return None


# --------------------------------------------------------------------------------------------
# What the route returns


def status_payload() -> dict[str, Any]:
    """Settings, the stored result with its findings, whether it is current, and the run state."""
    from utils.hardware_check_settings import get_hardware_check_settings

    result = load_result()
    payload: dict[str, Any] = {
        "settings": get_hardware_check_settings(),
        "result": None,
        "up_to_date": None,
        "state": state_snapshot(),
    }
    if result:
        payload["result"] = {**result, **analyze(result)}
        payload["up_to_date"] = result_is_current(result)
    else:
        payload["up_to_date"] = False
    return payload
