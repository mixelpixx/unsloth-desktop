# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Logs > Diagnostics: the whole environment in one snapshot that is safe to hand over.

Built from what the rest of Studio already knows rather than re-derived: the same nvidia-smi runner
the resource strip uses (bounded wait), package versions from ``importlib.metadata`` (torch is never
imported for this), the llama.cpp install marker and the freshness checker's CACHED answer (no
network call is made here), the cache walker Settings > Storage uses (in the background), the
resource view's model collectors and the MCP status the MCP dialog shows.

Every section fails on its own: a section that raises or runs past the deadline reports
``{"status": "unavailable", "reason": ...}`` and the rest of the report still arrives.

What leaves this module is sanitized twice over. Environment values whose name or value looks like a
credential are replaced by ``<redacted>`` (the names come from an allowlist to begin with), every
string in the report goes through the log redactor the logs viewer uses, and the user's home
directory is written as ``~``. MCP servers are summarised by name, transport and state only: no
command, argument, URL, header or environment value is ever read into the report.
"""

from __future__ import annotations

import contextvars
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from loggers import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1

# Sections run in parallel; one that has not answered by then is reported as timed out.
COLLECTION_DEADLINE_S = 10.0
NVIDIA_SMI_TIMEOUT_S = 5.0
_REASON_MAX_CHARS = 240
_ENV_VALUE_MAX_CHARS = 400

SECTION_ORDER = (
    "studio",
    "os",
    "gpus",
    "python",
    "llama_cpp",
    "storage",
    "hardware_check",
    "models",
    "mcp",
    "environment",
)

BUNDLE_JSON_MEMBER = "diagnostics/diagnostics.json"
BUNDLE_MARKDOWN_MEMBER = "diagnostics/diagnostics.md"

# Distributions worth naming in a bug report. Read from metadata only.
PACKAGES = (
    "unsloth",
    "unsloth_zoo",
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "xformers",
    "triton",
    "triton-windows",
    "flash-attn",
    "diffusers",
    "datasets",
    "huggingface_hub",
    "hf_xet",
    "safetensors",
    "tokenizers",
    "numpy",
    "fastapi",
    "uvicorn",
    "mcp",
)

# Environment variables that bear on how Studio, torch, CUDA or llama.cpp behave. Anything else in
# the environment is left out entirely rather than masked.
ENV_PREFIXES = (
    "UNSLOTH_",
    "STUDIO_",
    "HF_",
    "HUGGINGFACE_",
    "TRANSFORMERS_",
    "CUDA_",
    "NVIDIA_",
    "PYTORCH_",
    "TORCH_",
    "TORCHINDUCTOR_",
    "LLAMA_",
    "GGML_",
    "TRITON_",
    "XFORMERS_",
    "BNB_",
    "BITSANDBYTES_",
    "ACCELERATE_",
    "TOKENIZERS_",
    "OMP_",
    "MKL_",
    "NCCL_",
    "WANDB_",
    "VLLM_",
)
ENV_EXACT = frozenset(
    {
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "XDG_CACHE_HOME",
        "TMP",
        "TEMP",
        "TMPDIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
    }
)
# Names the shared child-env scrubber does not cover but that still carry a credential when set:
# a bare *_KEY, an auth header or cookie, a session or signature.
_EXTRA_SECRET_NAME = re.compile(
    r"(?i)(?:^|_)(?:KEY|AUTH|AUTHORIZATION|COOKIE|BEARER|SIGNATURE|SESSION|PAT)(?:_|$)"
)
# A switch is not a credential: masking UNSLOTH_DISABLE_AUTH=1 only hides how Studio is configured.
_FLAG_VALUES = frozenset({"", "0", "1", "true", "false", "yes", "no", "on", "off", "auto", "none"})


class Unavailable(Exception):
    """A section that has a plain-language reason for not answering."""


# --------------------------------------------------------------------------------------------
# Small helpers


def _iso(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    try:
        return (
            datetime.fromtimestamp(float(timestamp), tz = timezone.utc)
            .replace(microsecond = 0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OverflowError, OSError, ValueError):
        return None


def _reason(exc: BaseException) -> str:
    """Client-safe text for a failed section, never a traceback."""
    from utils.log_redaction import redact_log_text
    from utils.utils import safe_curated_detail

    if isinstance(exc, Unavailable):
        text = str(exc)
    elif isinstance(exc, subprocess.TimeoutExpired):
        text = "timed out"
    else:
        detail = safe_curated_detail(exc, fallback = "")
        text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    text = redact_log_text(" ".join(text.split()))
    if len(text) > _REASON_MAX_CHARS:
        text = text[: _REASON_MAX_CHARS - 3] + "..."
    return text or "unknown error"


def _run_section(collect: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"status": "ok", "data": collect()}
    except Exception as exc:  # noqa: BLE001 -- one section must never fail the report
        logger.debug("diagnostics: section %s failed: %s", getattr(collect, "__name__", collect), exc)
        return {"status": "unavailable", "reason": _reason(exc)}


def _hidden_subprocess_kwargs() -> dict[str, Any]:
    try:
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        return windows_hidden_subprocess_kwargs()
    except Exception:  # noqa: BLE001
        return {}


def _home_pattern() -> Optional[re.Pattern[str]]:
    """The user's home directory at a path boundary, in either separator (and any case on Windows)."""
    try:
        home = str(Path.home())
    except (RuntimeError, OSError):
        return None
    home = home.rstrip("\\/")
    # A home of "/" or "C:" would turn every path into "~".
    if len(home) < 4:
        return None
    parts = re.split(r"[\\/]+", home)
    body = r"[\\/]+".join(re.escape(part) for part in parts)
    flags = re.IGNORECASE if os.name == "nt" else 0
    return re.compile(body + r"(?=[\\/]|$|[^A-Za-z0-9_.\-])", flags)


def sanitize(value: Any, *, _pattern: Any = ...) -> Any:
    """Every string through the shared log redactor, and the home directory written as ``~``.

    Applied to the whole report so a credential that turns up somewhere unexpected (a model name,
    a path, a failure reason) is masked just as it would be in the logs viewer.
    """
    from utils.log_redaction import redact_log_text

    pattern = _home_pattern() if _pattern is ... else _pattern

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            text = redact_log_text(node)
            return pattern.sub("~", text) if pattern is not None else text
        if isinstance(node, Mapping):
            return {key: walk(item) for key, item in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(item) for item in node]
        return node

    return walk(value)


def _disk(path: Path) -> dict[str, Any]:
    """Drive, free and total for a path, climbing past parts that do not exist yet."""
    entry: dict[str, Any] = {"path": str(path), "exists": False, "drive": None}
    try:
        entry["exists"] = path.exists()
    except OSError:
        pass
    drive = os.path.splitdrive(str(path))[0]
    entry["drive"] = drive or None
    for candidate in (path, *path.parents):
        try:
            usage = shutil.disk_usage(candidate)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            entry["error"] = _reason(exc)
            break
        entry["free_bytes"] = int(usage.free)
        entry["total_bytes"] = int(usage.total)
        if not drive:
            entry["drive"] = _mount_point(candidate)
        break
    return entry


def _mount_point(path: Path) -> Optional[str]:
    current = path
    try:
        while not os.path.ismount(current) and current.parent != current:
            current = current.parent
    except OSError:
        return None
    return str(current)


# --------------------------------------------------------------------------------------------
# Studio


def _unsloth_version() -> Optional[str]:
    # main.py resolved it at startup (with a source-tree fallback); reuse that when it is loaded.
    main = sys.modules.get("main")
    resolved = getattr(main, "UNSLOTH_VERSION", None) if main is not None else None
    if isinstance(resolved, str) and resolved:
        return resolved
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("unsloth")
    except PackageNotFoundError:
        return None


def _frontend_build(frontend_build_path: Optional[Path]) -> dict[str, Any]:
    from utils.studio_version import _repo_root

    served = frontend_build_path is not None
    path = Path(frontend_build_path) if served else _repo_root() / "studio" / "frontend" / "dist"
    index = path / "index.html"
    try:
        built_at = _iso(index.stat().st_mtime)
    except OSError:
        built_at = None
    return {"path": str(path), "served": served, "built_at": built_at}


def collect_studio(frontend_build_path: Optional[Path] = None) -> dict[str, Any]:
    from utils.studio_version import get_source_checkout_info, get_studio_version
    from utils.update_status import detect_install_source

    data: dict[str, Any] = {
        "studio_version": get_studio_version(),
        "unsloth_version": _unsloth_version(),
        "install_source": detect_install_source(),
        "source_checkout": get_source_checkout_info(),
        "frontend_build": _frontend_build(frontend_build_path),
    }
    try:
        import psutil

        started = psutil.Process(os.getpid()).create_time()
        data["process"] = {
            "started_at": _iso(started),
            "uptime_seconds": max(0, round(time.time() - started)),
        }
    except Exception:  # noqa: BLE001
        data["process"] = None
    return data


# --------------------------------------------------------------------------------------------
# Operating system


def _windows_version() -> dict[str, Any]:
    release, build, _csd, _ptype = platform.win32_ver()
    info: dict[str, Any] = {"build": build or None}
    try:
        build_number = int((build or "0").split(".")[-1])
    except ValueError:
        build_number = 0
    # platform.release() says "10" on Windows 11 for older Pythons: the build number decides.
    info["name"] = "Windows 11" if build_number >= 22000 else f"Windows {release}".strip()
    try:
        info["edition"] = platform.win32_edition() or None
    except Exception:  # noqa: BLE001
        info["edition"] = None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        ) as key:
            for value_name, field in (("DisplayVersion", "display_version"), ("UBR", "ubr")):
                try:
                    info[field] = winreg.QueryValueEx(key, value_name)[0]
                except OSError:
                    pass
    except Exception:  # noqa: BLE001
        pass
    if info.get("ubr") is not None and build:
        info["build"] = f"{build}.{info['ubr']}"
    info.pop("ubr", None)
    return info


def _cpu_name() -> Optional[str]:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip() or None
        if system == "Linux":
            with open("/proc/cpuinfo", encoding = "utf-8", errors = "replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip() or None
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output = True,
                text = True,
                timeout = 2,
            )
            return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or None


def collect_os() -> dict[str, Any]:
    import psutil

    system = platform.system()
    data: dict[str, Any] = {
        "system": system,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if system == "Windows":
        data.update(_windows_version())
    elif system == "Darwin":
        data["name"] = f"macOS {platform.mac_ver()[0]}".strip()
    elif system == "Linux":
        try:
            data["name"] = platform.freedesktop_os_release().get("PRETTY_NAME")
        except OSError:
            data["name"] = f"Linux {platform.release()}"
        data["kernel"] = platform.release()
    memory = psutil.virtual_memory()
    data["cpu"] = {
        "name": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical = False),
        "logical_cores": psutil.cpu_count(logical = True),
    }
    data["memory"] = {"total_bytes": int(memory.total), "available_bytes": int(memory.available)}
    try:
        swap = psutil.swap_memory()
        data["swap"] = {"total_bytes": int(swap.total), "free_bytes": int(swap.free)}
    except Exception:  # noqa: BLE001
        data["swap"] = None
    return data


# --------------------------------------------------------------------------------------------
# GPUs

# `name` last, so a name holding commas is rejoined from whatever is left.
_GPU_FIELDS = (
    "index",
    "pci.bus_id",
    "driver_version",
    "compute_cap",
    "memory.total",
    "memory.used",
    "memory.free",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "name",
)
# What every nvidia-smi this side of a decade answers, for a driver that refuses a newer field.
_GPU_FIELDS_MINIMAL = ("index", "driver_version", "memory.total", "memory.used", "memory.free", "name")
_CUDA_VERSION_RE = re.compile(r"^\s*CUDA (UMD )?Version\s*:\s*([0-9][0-9.]*)", re.MULTILINE)


def _smi_number(raw: str) -> Optional[float]:
    from utils.hardware.nvidia import _parse_smi_value

    try:
        return _parse_smi_value(raw)
    except Exception:  # noqa: BLE001
        return None


def _smi_text(raw: str) -> Optional[str]:
    raw = raw.strip()
    return None if not raw or raw.startswith("[") else raw


def _run_smi(exe: str, args: list[str]) -> subprocess.CompletedProcess:
    from utils.hardware import gpu_query
    from utils.native_path_leases import child_env_without_native_path_secret

    return gpu_query.run_nvidia_smi(
        [exe, *args],
        timeout = NVIDIA_SMI_TIMEOUT_S,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = child_env_without_native_path_secret(),
        **_hidden_subprocess_kwargs(),
    )


def parse_gpu_rows(stdout: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mib = 1024 * 1024
    for line in stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(fields):
            continue
        head = parts[: len(fields) - 1]
        values = dict(zip(fields[:-1], head))
        values["name"] = ", ".join(parts[len(fields) - 1 :])
        try:
            index = int(values["index"])
        except (KeyError, ValueError):
            continue

        def mem(field: str) -> Optional[int]:
            number = _smi_number(values.get(field, ""))
            return int(number * mib) if number is not None else None

        def link(field: str) -> Optional[int]:
            number = _smi_number(values.get(field, ""))
            return int(number) if number is not None else None

        rows.append(
            {
                "index": index,
                "name": values["name"] or None,
                "pci_bus_id": _smi_text(values.get("pci.bus_id", "")),
                "driver_version": _smi_text(values.get("driver_version", "")),
                "compute_capability": _smi_text(values.get("compute_cap", "")),
                "memory_total_bytes": mem("memory.total"),
                "memory_used_bytes": mem("memory.used"),
                "memory_free_bytes": mem("memory.free"),
                "pcie_gen_current": link("pcie.link.gen.current"),
                "pcie_gen_max": link("pcie.link.gen.max"),
                "pcie_width_current": link("pcie.link.width.current"),
                "pcie_width_max": link("pcie.link.width.max"),
            }
        )
    rows.sort(key = lambda row: row["index"])
    return rows


def parse_cuda_version(stdout: str) -> Optional[str]:
    """The driver's CUDA version. Newer drivers print it as "CUDA UMD Version", which wins."""
    found = {bool(match.group(1)): match.group(2) for match in _CUDA_VERSION_RE.finditer(stdout or "")}
    return found.get(True) or found.get(False)


def collect_gpus() -> dict[str, Any]:
    from utils.hardware.gpu_resources import _nvidia_smi

    exe = _nvidia_smi()
    if exe is None:
        raise Unavailable("nvidia-smi was not found (no NVIDIA driver, or not an NVIDIA GPU)")
    fields = _GPU_FIELDS
    result = _run_smi(exe, [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if result.returncode != 0:
        # An older driver refuses the whole query over one field it does not know.
        fields = _GPU_FIELDS_MINIMAL
        result = _run_smi(exe, [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if result.returncode != 0:
        raise Unavailable(f"nvidia-smi exited with code {result.returncode}")
    devices = parse_gpu_rows(result.stdout or "", fields)
    cuda_version = None
    try:
        query = _run_smi(exe, ["-q", "-d", "COMPUTE"])
        if query.returncode == 0:
            cuda_version = parse_cuda_version(query.stdout or "")
    except Exception as exc:  # noqa: BLE001 -- the table above is still worth returning
        logger.debug("diagnostics: CUDA version query failed: %s", exc)
    driver = next((row["driver_version"] for row in devices if row.get("driver_version")), None)
    return {
        "driver_version": driver,
        "cuda_driver_version": cuda_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
    }


# --------------------------------------------------------------------------------------------
# Python and packages

_TORCH_CUDA_RE = re.compile(r"^cuda\s*(?::[^=\n]*)?=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
_TORCH_HIP_RE = re.compile(r"^hip\s*(?::[^=\n]*)?=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
_TORCH_LOCAL_CUDA_RE = re.compile(r"\+cu(\d+)(\d)$")


def torch_build_info(torch_version: Optional[str]) -> dict[str, Any]:
    """The CUDA (or ROCm) torch was built with, without importing torch.

    torch already imported: ask it. Otherwise read ``torch/version.py`` from the installed files,
    and fall back to the ``+cu130`` local version tag.
    """
    loaded = sys.modules.get("torch")
    if loaded is not None:
        version_module = getattr(loaded, "version", None)
        return {
            "cuda": getattr(version_module, "cuda", None),
            "hip": getattr(version_module, "hip", None),
            "imported": True,
        }
    info: dict[str, Any] = {"cuda": None, "hip": None, "imported": False}
    try:
        from importlib.metadata import distribution

        source = Path(str(distribution("torch").locate_file("torch/version.py"))).read_text(
            encoding = "utf-8", errors = "replace"
        )
        cuda = _TORCH_CUDA_RE.search(source)
        hip = _TORCH_HIP_RE.search(source)
        info["cuda"] = cuda.group(1) if cuda else None
        info["hip"] = hip.group(1) if hip else None
        return info
    except Exception:  # noqa: BLE001
        pass
    tag = _TORCH_LOCAL_CUDA_RE.search(torch_version or "")
    if tag:
        info["cuda"] = f"{tag.group(1)}.{tag.group(2)}"
    return info


def collect_python() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
        except Exception:  # noqa: BLE001
            packages[name] = None
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "torch": torch_build_info(packages.get("torch")),
        "packages": packages,
    }


# --------------------------------------------------------------------------------------------
# llama.cpp


def llama_update_state(binary: Optional[str], marker: Optional[dict]) -> dict[str, Any]:
    """Whether a newer llama.cpp exists, from the freshness checker's CACHE only.

    Never fetches: no cached answer (memory or the 24 h disk cache) is "not_checked".
    """
    from utils import llama_cpp_freshness as freshness
    from utils import llama_cpp_update as update
    from utils.update_status import update_checks_disabled

    if update_checks_disabled():
        return {"state": "disabled"}
    try:
        if update._studio_custom_path_active() or update._active_install_is_local_link(binary):
            return {"state": "user_managed"}
    except Exception:  # noqa: BLE001
        pass
    if not marker:
        return {"state": "not_checked", "detail": "source build or custom path"}
    repo = marker.get("published_repo") or update.DEFAULT_PUBLISHED_REPO
    cached = freshness._release_memo.get(repo) or freshness._load_disk_cache(repo)
    if not cached or not cached[1]:
        return {"state": "not_checked"}
    fetched_at, latest = cached
    installed = marker.get("release_tag") or marker.get("tag")
    behind = freshness.is_behind(installed, latest)
    return {
        "state": "available" if behind else "up_to_date",
        "latest": latest,
        "checked_at": _iso(fetched_at),
    }


def collect_llama_cpp() -> dict[str, Any]:
    from utils import llama_cpp_update as update
    from utils.llama_cpp_freshness import parse_base_build, read_install_marker
    from utils.prebuilt.llama_backend import marker_backend

    binary = update._find_binary()
    if not binary:
        raise Unavailable("no llama-server binary was found")
    marker = read_install_marker(binary)
    version = update.get_installed_llama_version()
    data: dict[str, Any] = {
        "version": version,
        "build": parse_base_build((marker or {}).get("tag") or version),
        "commit": (marker or {}).get("source_commit_short"),
        "backend": marker_backend(marker) if marker else None,
        "bundle": (marker or {}).get("bundle_profile"),
        "installed_at": (marker or {}).get("installed_at_utc"),
        "prebuilt": marker is not None,
        "binary": binary,
        "update": llama_update_state(binary, marker),
    }
    return data


# --------------------------------------------------------------------------------------------
# Storage

_HF_SIZE_TTL_S = 15 * 60.0
_HF_SIZE_FIRST_WAIT_S = 0.3
_hf_size_lock = threading.Lock()
# hub cache path -> (measured at, bytes, top-level entries)
_hf_sizes: dict[str, tuple[float, int, int]] = {}
_hf_size_running: set[str] = set()


def _measure_hf_cache(key: str) -> None:
    try:
        from utils.cache_inventory import _measure_root

        size, entries = _measure_root(Path(key))
        with _hf_size_lock:
            _hf_sizes[key] = (time.time(), int(size), int(entries))
    except Exception as exc:  # noqa: BLE001
        logger.debug("diagnostics: HF cache size failed: %s", exc)
    finally:
        with _hf_size_lock:
            _hf_size_running.discard(key)


def hf_cache_size(hub_cache: Path, *, wait: float = _HF_SIZE_FIRST_WAIT_S) -> dict[str, Any]:
    """The HF hub cache size, never by walking on the request thread.

    The walk is the one Settings > Storage uses, run on a daemon thread and remembered for
    ``_HF_SIZE_TTL_S``; a request waits ``wait`` seconds for it and otherwise says "computing"
    (or returns the previous figure while it refreshes).
    """
    key = str(hub_cache)
    try:
        if not hub_cache.is_dir():
            return {"state": "missing"}
    except OSError:
        return {"state": "missing"}
    with _hf_size_lock:
        cached = _hf_sizes.get(key)
        fresh = cached is not None and time.time() - cached[0] < _HF_SIZE_TTL_S
        start = not fresh and key not in _hf_size_running
        if start:
            _hf_size_running.add(key)
    if start:
        worker = threading.Thread(
            target = _measure_hf_cache, args = (key,), name = "diagnostics-hf-size", daemon = True
        )
        worker.start()
        worker.join(max(0.0, wait))
        with _hf_size_lock:
            cached = _hf_sizes.get(key)
    with _hf_size_lock:
        running = key in _hf_size_running
    if cached is None:
        return {"state": "computing"}
    return {
        "state": "refreshing" if running else "ready",
        "bytes": cached[1],
        "entries": cached[2],
        "measured_at": _iso(cached[0]),
    }


def collect_storage() -> dict[str, Any]:
    from utils.hf_cache_settings import get_hf_cache_paths
    from utils.paths.storage_roots import studio_root

    hf = get_hf_cache_paths()
    locations = (
        ("studio_home", Path(studio_root())),
        ("hf_home", Path(hf.cache_home)),
        ("hf_hub_cache", Path(hf.hub_cache)),
        ("temp", Path(tempfile.gettempdir())),
    )
    entries = []
    for key, path in locations:
        entry = _disk(path)
        entry["key"] = key
        entries.append(entry)
    return {
        "locations": entries,
        "hf_cache_source": hf.source,
        "hf_cache_size": hf_cache_size(Path(hf.hub_cache)),
    }


# --------------------------------------------------------------------------------------------
# Hardware check


def collect_hardware_check() -> dict[str, Any]:
    """The last Settings > Resources > Hardware check result, its findings and the options.

    Read from the stored JSON; nothing is measured here (a run takes ~10 s and touches every GPU).
    """
    from utils.hardware import hardware_check
    from utils.hardware_check_settings import get_hardware_check_settings

    result = hardware_check.load_result()
    if not result:
        raise Unavailable("the hardware check has not run yet")
    summary = hardware_check.analyze(result)
    return {
        "checked_at": result.get("finished_at"),
        "trigger": result.get("trigger"),
        "duration_ms": result.get("duration_ms"),
        "up_to_date": hardware_check.result_is_current(result),
        "settings": get_hardware_check_settings(),
        "gpus": [
            {
                "index": gpu.get("index"),
                "name": gpu.get("name"),
                "status": gpu.get("status"),
                "reason": gpu.get("reason"),
                "link": gpu.get("link"),
                "link_idle": gpu.get("link_idle"),
                "h2d_gibs": gpu.get("h2d_gibs"),
                "d2h_gibs": gpu.get("d2h_gibs"),
            }
            for gpu in result.get("gpus") or []
        ],
        "pairs": list(result.get("pairs") or []),
        "storage": [
            {
                "key": entry.get("key"),
                "drive": entry.get("drive"),
                "bus_type": entry.get("bus_type"),
                "media_type": entry.get("media_type"),
                "model": entry.get("model"),
            }
            for entry in result.get("storage") or []
        ],
        "gpu_error": result.get("gpu_error"),
        "storage_error": result.get("storage_error"),
        "findings": [
            {"id": f["id"], "severity": f["severity"], "text": f["text"]} for f in summary["findings"]
        ],
        "recommended": summary["recommended"],
    }


# --------------------------------------------------------------------------------------------
# Environment


def _secret_looking(name: str, value: str) -> bool:
    from utils.log_redaction import redact_log_text
    from utils.prebuilt.child_env import URL_USERINFO_RE, is_secret_env_name

    if value.strip().lower() in _FLAG_VALUES:
        return False
    if is_secret_env_name(name) or _EXTRA_SECRET_NAME.search(name):
        return True
    if URL_USERINFO_RE.search(value):
        return True
    # The logs viewer's own rules, on the value alone and as NAME=value.
    return redact_log_text(value) != value or redact_log_text(f"{name}={value}") != f"{name}={value}"


def env_allowed(name: str) -> bool:
    upper = name.upper()
    return upper in ENV_EXACT or upper.startswith(ENV_PREFIXES)


def environment_entries(environ: Mapping[str, str]) -> list[dict[str, Any]]:
    from utils.log_redaction import REDACTED

    entries = []
    for name in sorted(environ, key = str.upper):
        if not env_allowed(name):
            continue
        value = str(environ.get(name) or "")
        secret = _secret_looking(name, value)
        if not secret and len(value) > _ENV_VALUE_MAX_CHARS:
            value = value[: _ENV_VALUE_MAX_CHARS - 3] + "..."
        entries.append({"name": name, "value": REDACTED if secret else value, "redacted": secret})
    return entries


def collect_environment() -> dict[str, Any]:
    return {"variables": environment_entries(os.environ)}


# --------------------------------------------------------------------------------------------
# Loaded models and MCP servers


def collect_models() -> dict[str, Any]:
    from routes.resources import collect_models as resident_models

    models = []
    for model in resident_models():
        row = model.to_json()
        models.append(
            {
                "name": row.get("name"),
                "kind": row.get("kind"),
                "engine": row.get("source"),
                "variant": row.get("variant"),
                "gpu_ids": row.get("gpu_ids") or [],
                "device": row.get("device"),
                "layers_on_gpu": row.get("layers_on_gpu"),
                "layers_total": row.get("layers_total"),
                "context_length": row.get("context_length"),
                "vram_bytes": row.get("vram_bytes"),
                "loading": bool(row.get("loading")),
                "inactive": bool(row.get("inactive")),
            }
        )
    return {"models": models}


def collect_mcp() -> dict[str, Any]:
    """Name, transport, process mode and state per saved server. The command, its arguments, the
    URL, headers and env values are never copied out of the row."""
    from core.inference.mcp_client import is_stdio, server_process_mode, server_status
    from storage import mcp_servers_db

    servers = []
    for row in mcp_servers_db.list_servers():
        local = is_stdio(row.get("url") or "")
        entry: dict[str, Any] = {
            "name": str(row.get("display_name") or ""),
            "builtin": row.get("builtin_id") or None,
            "transport": "local" if local else "remote",
            "enabled": bool(row.get("is_enabled")),
            "oauth": bool(row.get("use_oauth")),
            "process_mode": None,
            "state": None,
        }
        if local:
            try:
                entry["process_mode"] = server_process_mode(row)
            except Exception:  # noqa: BLE001
                pass
            try:
                entry["state"] = server_status(row).get("state")
            except Exception:  # noqa: BLE001
                pass
        servers.append(entry)
    return {"servers": servers}


# --------------------------------------------------------------------------------------------
# The report


def default_collectors(frontend_build_path: Optional[Path] = None) -> dict[str, Callable[[], Any]]:
    return {
        "studio": lambda: collect_studio(frontend_build_path),
        "os": collect_os,
        "gpus": collect_gpus,
        "python": collect_python,
        "llama_cpp": collect_llama_cpp,
        "storage": collect_storage,
        "hardware_check": collect_hardware_check,
        "models": collect_models,
        "mcp": collect_mcp,
        "environment": collect_environment,
    }


def collect_diagnostics(
    *,
    frontend_build_path: Optional[Path] = None,
    collectors: Optional[Mapping[str, Callable[[], Any]]] = None,
    deadline_s: float = COLLECTION_DEADLINE_S,
) -> dict[str, Any]:
    """Every section, in parallel, each failing soft, then sanitized as a whole."""
    started = time.monotonic()
    collectors = dict(collectors if collectors is not None else default_collectors(frontend_build_path))
    results: dict[str, Any] = {}
    finished = {name: threading.Event() for name in collectors}

    def run(name: str, collect: Callable[[], Any]) -> None:
        results[name] = _run_section(collect)
        finished[name].set()

    for name, collect in collectors.items():
        # Daemon threads, not a pool: a probe stuck on a dead network drive or a wedged driver
        # must neither hold this request past the deadline nor keep the process from exiting.
        # One context copy each, so the account binding (MCP rows are per account) reaches it.
        threading.Thread(
            target = contextvars.copy_context().run,
            args = (run, name, collect),
            name = f"diagnostics-{name}",
            daemon = True,
        ).start()
    end = started + max(0.0, deadline_s)
    sections: dict[str, Any] = {}
    for name in [*SECTION_ORDER, *[key for key in collectors if key not in SECTION_ORDER]]:
        if name not in collectors:
            continue
        if finished[name].wait(max(0.0, end - time.monotonic())):
            sections[name] = results[name]
        else:
            sections[name] = {"status": "unavailable", "reason": "timed out"}
    report = {
        "schema": SCHEMA_VERSION,
        "generated_at": _iso(time.time()),
        "collection_ms": int((time.monotonic() - started) * 1000),
        "sections": sections,
    }
    return sanitize(report)


# --------------------------------------------------------------------------------------------
# Markdown


def _gib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "?"
    return f"{value / 1024**3:.1f} GiB"


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _code(value: Any) -> str:
    text = str(value).replace("`", "'")
    return f"`{text}`"


def _section_lines(report: Mapping[str, Any], name: str, title: str, render: Callable[[dict], list[str]]) -> list[str]:
    section = (report.get("sections") or {}).get(name)
    lines = [f"### {title}", ""]
    if not isinstance(section, Mapping):
        return [*lines, "_Not collected._", ""]
    if section.get("status") != "ok":
        return [*lines, f"_Unavailable: {section.get('reason') or 'unknown'}_", ""]
    try:
        body = render(section.get("data") or {})
    except Exception as exc:  # noqa: BLE001 -- a renderer bug must not lose the other sections
        body = [f"_Could not render: {type(exc).__name__}_"]
    return [*lines, *body, ""]


def _md_studio(data: dict) -> list[str]:
    lines = [
        f"- Studio: {_code(data.get('studio_version') or '?')}",
        f"- Unsloth package: {_code(data.get('unsloth_version') or '?')}",
        f"- Install type: {_code(data.get('install_source') or '?')}",
    ]
    checkout = data.get("source_checkout")
    if isinstance(checkout, Mapping):
        dirty = checkout.get("dirty")
        state = " (modified)" if dirty else (" (clean)" if dirty is False else "")
        lines.append(
            f"- Source checkout: branch {_code(checkout.get('branch') or '?')} "
            f"at {_code(checkout.get('commit') or '?')}{state}"
        )
    build = data.get("frontend_build")
    if isinstance(build, Mapping):
        lines.append(f"- Frontend build: {build.get('built_at') or 'unknown'}")
    process = data.get("process")
    if isinstance(process, Mapping) and process.get("uptime_seconds") is not None:
        lines.append(f"- Server uptime: {int(process['uptime_seconds']) // 60} min")
    return lines


def _md_os(data: dict) -> list[str]:
    name = data.get("name") or data.get("system") or "?"
    extra = [part for part in (data.get("edition"), data.get("display_version")) if part]
    build = f", build {data['build']}" if data.get("build") else ""
    lines = [f"- OS: {name}{' ' + ' '.join(extra) if extra else ''}{build} ({data.get('machine') or '?'})"]
    cpu = data.get("cpu") or {}
    lines.append(
        f"- CPU: {cpu.get('name') or '?'} ({cpu.get('physical_cores') or '?'} cores, "
        f"{cpu.get('logical_cores') or '?'} threads)"
    )
    memory = data.get("memory") or {}
    lines.append(
        f"- RAM: {_gib(memory.get('total_bytes'))} total, {_gib(memory.get('available_bytes'))} available"
    )
    swap = data.get("swap")
    if isinstance(swap, Mapping) and swap.get("total_bytes"):
        lines.append(f"- Swap / page file: {_gib(swap.get('total_bytes'))}, {_gib(swap.get('free_bytes'))} free")
    return lines


def _pcie(device: Mapping[str, Any]) -> str:
    gen, gen_max = device.get("pcie_gen_current"), device.get("pcie_gen_max")
    width, width_max = device.get("pcie_width_current"), device.get("pcie_width_max")
    if gen is None and width is None:
        return "-"
    return f"Gen{gen or '?'}/{gen_max or '?'} x{width or '?'}/{width_max or '?'}"


def _md_gpus(data: dict) -> list[str]:
    lines = [
        f"- Driver: {_code(data.get('driver_version') or '?')}, CUDA (driver): "
        f"{_code(data.get('cuda_driver_version') or '?')}",
    ]
    if data.get("cuda_visible_devices") is not None:
        lines.append(f"- CUDA_VISIBLE_DEVICES: {_code(data['cuda_visible_devices'])}")
    devices = data.get("devices") or []
    if not devices:
        return [*lines, "- No GPUs reported."]
    lines += [
        "",
        "| # | Name | VRAM total | Used | Free | Compute | PCIe (now/max) |",
        "|---|------|-----------|------|------|---------|----------------|",
    ]
    for device in devices:
        lines.append(
            f"| {_cell(device.get('index'))} | {_cell(device.get('name'))} | "
            f"{_gib(device.get('memory_total_bytes'))} | {_gib(device.get('memory_used_bytes'))} | "
            f"{_gib(device.get('memory_free_bytes'))} | {_cell(device.get('compute_capability'))} | "
            f"{_cell(_pcie(device))} |"
        )
    return lines


def _md_python(data: dict) -> list[str]:
    torch = data.get("torch") or {}
    lines = [
        f"- Python: {_code(data.get('version') or '?')} ({data.get('implementation') or '?'}"
        f"{', virtualenv' if data.get('virtualenv') else ''})",
    ]
    packages = data.get("packages") or {}
    if packages.get("torch"):
        if torch.get("cuda"):
            built = f"CUDA {torch['cuda']}"
        elif torch.get("hip"):
            built = f"ROCm {torch['hip']}"
        else:
            built = "CPU only"
        lines.append(f"- torch: {_code(packages['torch'])} (built with {built})")
    installed = [f"{name} {_code(version)}" for name, version in packages.items() if version and name != "torch"]
    if installed:
        lines.append("- Packages: " + ", ".join(installed))
    missing = [name for name, version in packages.items() if not version]
    if missing:
        lines.append("- Not installed: " + ", ".join(missing))
    return lines


_UPDATE_TEXT = {
    "available": "update available",
    "up_to_date": "up to date",
    "not_checked": "not checked",
    "disabled": "update checks disabled",
    "user_managed": "user-managed install",
}


def _md_llama(data: dict) -> list[str]:
    update = data.get("update") or {}
    state = _UPDATE_TEXT.get(update.get("state"), update.get("state") or "?")
    if update.get("latest"):
        state += f" (latest {update['latest']}, checked {update.get('checked_at') or '?'})"
    return [
        f"- Version: {_code(data.get('version') or '?')} (build {data.get('build') or '?'}, "
        f"commit {data.get('commit') or '?'})",
        f"- Backend: {_code(data.get('backend') or '?')}"
        f"{', bundle ' + _code(data['bundle']) if data.get('bundle') else ''}",
        f"- Binary: {_code(data.get('binary') or '?')}",
        f"- Update: {state}",
    ]


_LOCATION_LABELS = {
    "studio_home": "Studio home",
    "hf_home": "Hugging Face home",
    "hf_hub_cache": "Hugging Face hub cache",
    "hf_cache": "Hugging Face hub cache",
    "temp": "Temp",
}


def _md_storage(data: dict) -> list[str]:
    lines = ["| Location | Path | Drive | Free | Total |", "|----------|------|-------|------|-------|"]
    for entry in data.get("locations") or []:
        path = entry.get("path")
        if path and entry.get("exists") is False:
            path = f"{path} (missing)"
        lines.append(
            f"| {_LOCATION_LABELS.get(entry.get('key'), entry.get('key'))} | {_cell(path)} | "
            f"{_cell(entry.get('drive'))} | {_gib(entry.get('free_bytes'))} | {_gib(entry.get('total_bytes'))} |"
        )
    size = data.get("hf_cache_size") or {}
    if size.get("bytes") is not None:
        lines += ["", f"- Hugging Face hub cache size: {_gib(size['bytes'])} (measured {size.get('measured_at') or '?'})"]
    elif size.get("state") == "missing":
        lines += ["", "- Hugging Face hub cache size: the folder does not exist yet"]
    else:
        lines += ["", "- Hugging Face hub cache size: still being measured"]
    return lines


def _gibs(value: Any) -> str:
    return f"{value:.1f} GiB/s" if isinstance(value, (int, float)) else "-"


def _link_text(link: Any) -> str:
    if not isinstance(link, Mapping):
        return "-"
    gen, width, width_max = link.get("gen_current"), link.get("width_current"), link.get("width_max")
    if width is None and gen is None:
        return "-"
    return f"Gen{gen or '?'} x{width or '?'}" + (f" (max x{width_max})" if width_max else "")


_OPTION_TEXT = (
    ("prefer_fast_link", "prefer fast-link GPUs"),
    ("avoid_tensor_split", "avoid tensor parallel on slow links"),
    ("warn_training_slow_link", "warn when training uses a slow-link GPU"),
    ("auto_run", "check automatically when the hardware changes"),
)


def _md_hardware_check(data: dict) -> list[str]:
    current = data.get("up_to_date")
    state = "up to date" if current is True else ("out of date: the GPUs changed since" if current is False else "currency unknown")
    lines = [f"- Checked: {data.get('checked_at') or '?'} ({state})"]
    settings = data.get("settings") or {}
    lines.append(
        "- Options: "
        + ", ".join(f"{label} {'on' if settings.get(key) else 'off'}" for key, label in _OPTION_TEXT)
    )
    gpus = data.get("gpus") or []
    if gpus:
        lines += [
            "",
            "| # | GPU | Link under load | Link at rest | Host to GPU | GPU to host |",
            "|---|-----|-----------------|--------------|-------------|-------------|",
        ]
        for gpu in gpus:
            measured = gpu.get("status") == "measured"
            lines.append(
                f"| {_cell(gpu.get('index'))} | {_cell(gpu.get('name'))} | "
                f"{_cell(_link_text(gpu.get('link')) if measured else gpu.get('reason') or gpu.get('status'))} | "
                f"{_cell(_link_text(gpu.get('link_idle')))} | {_gibs(gpu.get('h2d_gibs'))} | {_gibs(gpu.get('d2h_gibs'))} |"
            )
    for pair in data.get("pairs") or []:
        peer = {True: "yes", False: "no"}
        lines.append(
            f"- GPU {pair.get('a')} and GPU {pair.get('b')}: peer access "
            f"{peer.get(pair.get('peer_ab'), '?')}/{peer.get(pair.get('peer_ba'), '?')}, copy "
            f"{_gibs(pair.get('copy_ab_gibs'))} / {_gibs(pair.get('copy_ba_gibs'))}"
        )
    storage = data.get("storage") or []
    if storage:
        lines += ["", "| Location | Drive | Bus | Media | Model |", "|----------|-------|-----|-------|-------|"]
        for entry in storage:
            lines.append(
                f"| {_LOCATION_LABELS.get(entry.get('key'), entry.get('key'))} | {_cell(entry.get('drive'))} | "
                f"{_cell(entry.get('bus_type'))} | {_cell(entry.get('media_type'))} | {_cell(entry.get('model'))} |"
            )
    findings = data.get("findings") or []
    if findings:
        lines += ["", "Findings:"]
        lines += [f"- [{f.get('severity')}] {f.get('text')}" for f in findings]
    return lines


def _md_models(data: dict) -> list[str]:
    models = data.get("models") or []
    if not models:
        return ["- No models loaded."]
    lines = ["| Model | Kind | Engine | GPUs | VRAM |", "|-------|------|--------|------|------|"]
    for model in models:
        gpus = ", ".join(str(gpu) for gpu in model.get("gpu_ids") or []) or model.get("device") or "-"
        name = model.get("name") or "?"
        if model.get("variant"):
            name = f"{name} ({model['variant']})"
        lines.append(
            f"| {_cell(name)} | {_cell(model.get('kind'))} | {_cell(model.get('engine'))} | {_cell(gpus)} | "
            f"{_gib(model.get('vram_bytes')) if model.get('vram_bytes') else '-'} |"
        )
    return lines


def _md_mcp(data: dict) -> list[str]:
    servers = data.get("servers") or []
    if not servers:
        return ["- No MCP servers configured."]
    lines = ["| Server | Transport | Process mode | State | Enabled |", "|--------|-----------|--------------|-------|---------|"]
    for server in servers:
        lines.append(
            f"| {_cell(server.get('name'))} | {_cell(server.get('transport'))} | {_cell(server.get('process_mode'))} | "
            f"{_cell(server.get('state'))} | {'yes' if server.get('enabled') else 'no'} |"
        )
    return lines


def _md_environment(data: dict) -> list[str]:
    variables = data.get("variables") or []
    if not variables:
        return ["- None of the listed variables are set."]
    body = "\n".join(f"{entry.get('name')}={entry.get('value')}" for entry in variables)
    body = body.replace("```", "'''")
    return [
        f"<details><summary>{len(variables)} variables (secret-looking values redacted)</summary>",
        "",
        "```",
        body,
        "```",
        "",
        "</details>",
    ]


_MARKDOWN_SECTIONS = (
    ("studio", "Studio", _md_studio),
    ("os", "System", _md_os),
    ("gpus", "GPUs", _md_gpus),
    ("python", "Python and packages", _md_python),
    ("llama_cpp", "llama.cpp", _md_llama),
    ("storage", "Storage", _md_storage),
    ("hardware_check", "Hardware check", _md_hardware_check),
    ("models", "Loaded models", _md_models),
    ("mcp", "MCP servers", _md_mcp),
    ("environment", "Environment", _md_environment),
)


def render_markdown(report: Mapping[str, Any]) -> str:
    """A GitHub-issue-ready summary, in English whatever the UI language."""
    lines = [
        "## Unsloth Studio diagnostics",
        "",
        f"Generated {report.get('generated_at') or '?'}. Secret-looking values are redacted and the "
        "home folder is shown as `~`.",
        "",
    ]
    for name, title, render in _MARKDOWN_SECTIONS:
        lines += _section_lines(report, name, title, render)
    return "\n".join(lines).rstrip() + "\n"


def bundle_members(report: Mapping[str, Any]) -> dict[str, bytes]:
    """The diagnostics files that ride along in the logs archive."""
    return {
        BUNDLE_JSON_MEMBER: (json.dumps(report, indent = 2, sort_keys = False) + "\n").encode("utf-8"),
        BUNDLE_MARKDOWN_MEMBER: render_markdown(report).encode("utf-8"),
    }
