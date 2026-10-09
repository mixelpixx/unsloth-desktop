# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``GET /api/resources``: every card's memory and every resident model, in one snapshot.

The sidebar strip polls this every 10 s, and every second while a load runs, so it reads only
what is already in memory: attributes of the runtimes, and the cached readings in
``utils.hardware.gpu_resources``. Nothing is constructed or imported for the answer: a runtime
whose module is not in ``sys.modules`` has never loaded anything, which is "nothing resident"
for free. Each source fails on its own, so one runtime that cannot answer drops only its rows.
"""

import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends

from auth.authentication import authenticated_via_api_key, get_current_subject
from loggers import get_logger

router = APIRouter()

logger = get_logger(__name__)

_MIB = 1024 * 1024
_CUDA_DEVICE_RE = re.compile(r"^cuda:(\d+)$", re.IGNORECASE)

# Rows in a fixed order, so the panel does not reshuffle between polls.
_KIND_ORDER = ("chat", "audio", "stt", "image", "video", "embedding")


@dataclass
class ResidentModel:
    kind: str
    # Which runtime's unload releases it; None when nothing in the UI can.
    source: Optional[str]
    name: str
    variant: Optional[str] = None
    gpu_ids: list = field(default_factory = list)
    device: Optional[str] = None
    layers_on_gpu: Optional[int] = None
    layers_total: Optional[int] = None
    context_length: Optional[int] = None
    cache_type_kv: Optional[str] = None
    # {physical gpu: bytes}; None when no runtime sized it.
    vram_by_gpu: Optional[dict] = None
    vram_approx: bool = False
    loading: bool = False
    # Held but not the one answering: a Transformers model cached behind the active one, or a
    # kept slot.
    inactive: bool = False
    stt_engine: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        vram = sum(self.vram_by_gpu.values()) if self.vram_by_gpu else None
        return {
            "id": f"{self.source or self.kind}:{self.stt_engine or self.name}",
            "kind": self.kind,
            "source": self.source,
            "name": self.name,
            "variant": self.variant,
            "gpu_ids": sorted(int(i) for i in self.gpu_ids),
            "device": self.device,
            "layers_on_gpu": self.layers_on_gpu,
            "layers_total": self.layers_total,
            "context_length": self.context_length,
            "cache_type_kv": self.cache_type_kv,
            "vram_bytes": vram,
            "vram_approx": bool(self.vram_approx) if vram is not None else False,
            "loading": self.loading,
            "inactive": self.inactive,
            "stt_engine": self.stt_engine,
        }


def _positive(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _cuda_visible_ids() -> Optional[list[int]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        return None


def _device_gpu_ids(device: Optional[str]) -> list[int]:
    """``cuda:N`` as a physical card. N is a torch ordinal, so a numeric mask is applied; a bare
    ``cuda`` names no card and is left unplaced rather than guessed."""
    match = _CUDA_DEVICE_RE.match(str(device or "").strip())
    if match is None:
        return []
    ordinal = int(match.group(1))
    mask = _cuda_visible_ids()
    if mask is None:
        return [ordinal]
    return [mask[ordinal]] if ordinal < len(mask) else []


def llama_model(llama: Any, name: str, *, inactive: bool = False) -> ResidentModel:
    """A running llama-server: its cards, layers, context and what it allocated.

    VRAM is what the child logged per device (weights, KV, compute), else the plan the launch
    sized it to, flagged approximate. A load that put no layer on a GPU is on no card.
    """
    from core.inference.load_guardrail import resident_llama_gpu_bytes

    by_gpu = resident_llama_gpu_bytes(llama)
    approx = False
    if not by_gpu:
        planned = getattr(llama, "_planned_vram_mib", None) or {}
        by_gpu = {int(k): int(v) * _MIB for k, v in planned.items() if v} or None
        approx = by_gpu is not None
    on_layers = getattr(llama, "offloaded_layers", None)
    on_gpu = getattr(llama, "_gpu_offload_active", None) is not False and on_layers != 0
    if not on_gpu:
        gpu_ids: list[int] = []
        by_gpu = None
    elif by_gpu and not approx:
        gpu_ids = sorted(by_gpu)
    else:
        gpu_ids = list(
            getattr(llama, "gpu_ids", None)
            or getattr(llama, "_child_gpu_physical_ids", None)
            or (sorted(by_gpu) if by_gpu else [])
        )
    return ResidentModel(
        kind = "chat",
        source = "chat",
        name = name,
        variant = getattr(llama, "hf_variant", None),
        gpu_ids = gpu_ids,
        layers_on_gpu = on_layers,
        layers_total = getattr(llama, "offload_total_layers", None),
        context_length = _positive(getattr(llama, "context_length", None)),
        cache_type_kv = getattr(llama, "cache_type_kv", None),
        vram_by_gpu = by_gpu,
        vram_approx = approx,
        inactive = inactive,
    )


def _orchestrator_kind(info: dict) -> str:
    """The loaded-models card's split: TTS speaks, whisper transcribes, audio_vlm chats."""
    audio_type = info.get("audio_type")
    if info.get("is_audio") and audio_type == "whisper":
        return "stt"
    if info.get("is_audio") and audio_type != "audio_vlm":
        return "audio"
    return "chat"


def _orchestrator_models(backend: Any, *, inactive_all: bool = False) -> list[ResidentModel]:
    models = getattr(backend, "models", None)
    if not isinstance(models, dict):
        return []
    active = getattr(backend, "active_model_name", None)
    rows: list[ResidentModel] = []
    for name, info in list(models.items()):
        if not isinstance(name, str):
            continue
        info = info if isinstance(info, dict) else {}
        rows.append(
            ResidentModel(
                kind = _orchestrator_kind(info),
                source = "chat",
                name = name,
                variant = info.get("gguf_variant"),
                gpu_ids = [int(i) for i in info.get("gpu_ids") or () if isinstance(i, int)],
                context_length = _positive(info.get("context_length")),
                inactive = inactive_all or name != active,
            )
        )
    return rows


def chat_models() -> list[ResidentModel]:
    """The chat runtime (primary llama-server or Transformers/MLX orchestrator), the kept slots,
    and chat loads in flight. Hidden from a managed account the way ``/status`` hides them."""
    from core.inference import model_slots
    from hub.services.models import account_access
    import routes.inference as inference

    rows: list[ResidentModel] = []
    hidden = account_access.resident_hidden("chat") or (
        account_access.managed_account()
        and account_access.resident_hidden("chat", inference._loaded_slot_ident())
    )
    if not hidden:
        llama = model_slots.in_slot(None, inference.get_llama_cpp_backend)
        if getattr(llama, "is_loaded", False):
            display, _ = inference._llama_status_model_ids(llama)
            if display:
                rows.append(llama_model(llama, display))
        backend = model_slots.in_slot(None, inference._peek_inference_backend)
        if backend is not None:
            rows.extend(_orchestrator_models(backend))
            for name in list(getattr(backend, "loading_models", None) or ()):
                public = inference._loading_public_id(str(name)) or str(name)
                rows.append(
                    ResidentModel(kind = "chat", source = "chat", name = public, loading = True)
                )
    for slot in model_slots.visible():
        try:
            if getattr(slot.llama, "is_loaded", False):
                public = inference._llama_public_model_id(slot.llama)
                if public:
                    rows.append(llama_model(slot.llama, public, inactive = True))
            active = getattr(slot.orchestrator, "active_model_name", None)
            if active:
                info = (getattr(slot.orchestrator, "models", None) or {}).get(active) or {}
                rows.append(
                    ResidentModel(
                        kind = _orchestrator_kind(info),
                        source = "chat",
                        name = str(active),
                        gpu_ids = [int(i) for i in info.get("gpu_ids") or () if isinstance(i, int)],
                        context_length = _positive(info.get("context_length")),
                        inactive = True,
                    )
                )
        except Exception as exc:  # noqa: BLE001 -- one slot must not drop the rest
            logger.debug("Resources: kept slot unreadable: %s", exc)
    for path in _visible_chat_loads(inference, model_slots, account_access):
        rows.append(ResidentModel(kind = "chat", source = "chat", name = path, loading = True))
    return rows


def _visible_chat_loads(inference: Any, model_slots: Any, account_access: Any) -> list[str]:
    """Public ids of the chat loads this caller may see, as ``/status`` reports them."""
    from utils.account_context import current_account_id

    managed = account_access.managed_account()
    me = current_account_id() if managed else None
    with inference._scoped_load_attempts_lock:
        attempts = [inference._running_load_attempt, *inference._pending_load_attempts.values()]
    paths = [
        a.model_path for a in attempts if a is not None and (not managed or a.subject == me)
    ]
    filling = model_slots.visible_loading()
    if filling is not None:
        paths.append(filling[1])
    out: list[str] = []
    for path in paths:
        public = inference._loading_public_id(path) or path
        if public and public not in out:
            out.append(public)
    return out


def _engine(module_name: str, attribute: str) -> Any:
    """A runtime singleton if its module is loaded and it was built; never imports or builds."""
    module = sys.modules.get(module_name)
    return getattr(module, attribute, None) if module is not None else None


def media_models() -> list[ResidentModel]:
    """Images (Diffusers and sd.cpp) and video: whatever pipeline is resident or loading."""
    from hub.services.models import account_access

    rows: list[ResidentModel] = []
    for kind, modality, module_name, attribute in (
        ("image", "diffusion", "core.inference.diffusion", "_diffusion_backend"),
        ("image", "diffusion", "core.inference.sd_cpp_backend", "_sd_cpp_backend"),
        ("video", "video", "core.inference.video", "_backend"),
    ):
        engine = _engine(module_name, attribute)
        if engine is None:
            continue
        try:
            if account_access.resident_hidden(modality):
                continue
            pending = getattr(engine, "_loading", None)
            if getattr(engine, "_state", None) is not None:
                status = engine.status()
                repo = status.get("repo_id")
                hidden = account_access.resident_hidden(modality, repo) if repo else True
                if status.get("loaded") and not hidden:
                    device = status.get("device")
                    rows.append(
                        ResidentModel(
                            kind = kind,
                            source = kind,
                            name = str(repo),
                            variant = status.get("gguf_variant") or status.get("transformer_quant"),
                            gpu_ids = _device_gpu_ids(device),
                            device = device,
                        )
                    )
            if pending is not None:
                name = getattr(pending, "repo_id", None)
                if name and not any(r.name == name for r in rows):
                    rows.append(
                        ResidentModel(kind = kind, source = kind, name = str(name), loading = True)
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Resources: %s status unreadable: %s", module_name, exc)
    return rows


_STT_ENGINES = (
    ("transformers", "core.inference.stt_sidecar"),
    ("mtmd", "core.inference.stt_mtmd_sidecar"),
    ("gguf", "core.inference.stt_ggml_sidecar"),
    ("audiocpp", "core.inference.stt_audiocpp_sidecar"),
)


def stt_models() -> list[ResidentModel]:
    """The dictation sidecars, released per engine as ``/audio/stt/unload`` does."""
    rows: list[ResidentModel] = []
    for engine_name, module_name in _STT_ENGINES:
        sidecar = _engine(module_name, "_sidecar")
        if sidecar is None:
            continue
        try:
            loaded = sidecar.loaded_model
            loading = bool(sidecar.is_loading())
            if not loaded and not loading:
                continue
            device = sidecar.device
            rows.append(
                ResidentModel(
                    kind = "stt",
                    source = "stt",
                    name = str(loaded or engine_name),
                    gpu_ids = _device_gpu_ids(device),
                    device = device,
                    loading = loading and not loaded,
                    stt_engine = engine_name,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Resources: %s sidecar unreadable: %s", engine_name, exc)
    return rows


def embedding_models() -> list[ResidentModel]:
    """The RAG embedder: a GGUF llama-server child or a sentence-transformers model in process."""
    embeddings = sys.modules.get("core.rag.embeddings")
    if embeddings is None:
        return []
    try:
        if not embeddings.backend_is_loaded():
            return []
        backend = getattr(embeddings, "_backend", None)
        name = getattr(backend, "_model_repo", None) or getattr(embeddings, "_name", None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Resources: embedder unreadable: %s", exc)
        return []
    label = str(name or "embedding")
    return [ResidentModel(kind = "embedding", source = "embedding", name = label)]


_COLLECTORS: tuple[Callable[[], list[ResidentModel]], ...] = (
    chat_models,
    media_models,
    stt_models,
    embedding_models,
)


def collect_models(collectors = _COLLECTORS) -> list[ResidentModel]:
    rows: list[ResidentModel] = []
    for collect in collectors:
        try:
            rows.extend(collect())
        except Exception as exc:  # noqa: BLE001 -- one runtime must not blank the others
            logger.debug("Resources: %s failed: %s", getattr(collect, "__name__", collect), exc)
    seen: set[tuple[str, str]] = set()
    unique: list[ResidentModel] = []
    for row in rows:
        key = (row.source or row.kind, row.stt_engine or row.name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    rank = {kind: i for i, kind in enumerate(_KIND_ORDER)}
    unique.sort(key = lambda r: rank.get(r.kind, len(_KIND_ORDER)))
    return unique


def studio_estimate(models: list[ResidentModel]) -> dict[int, Optional[int]]:
    """What Studio's runtimes say they hold per card, for when no counter can say. A card holding
    any resident model of unknown size has no estimate (None)."""
    out: dict[int, Optional[int]] = {}
    for model in models:
        if model.vram_by_gpu:
            for idx, used in model.vram_by_gpu.items():
                current = out.get(int(idx), 0)
                out[int(idx)] = None if current is None else current + int(used)
        else:
            for idx in model.gpu_ids:
                out[int(idx)] = None
    return out


def _visible_mask() -> Optional[set[int]]:
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        return LlamaCppBackend._visible_devices_mask("CUDA_VISIBLE_DEVICES")
    except Exception:
        return None


def build_snapshot(
    *,
    reader: Any = None,
    models: Optional[list[ResidentModel]] = None,
    loading: Optional[bool] = None,
    visible: Any = ...,
) -> dict[str, Any]:
    """The whole payload. Arguments are seams for tests; the route passes none."""
    from utils.hardware.gpu_resources import default_reader, split_gpu_memory

    reader = reader or default_reader()
    models = collect_models() if models is None else models
    if loading is None:
        loading = any(m.loading for m in models) or _slots_loading()
    readings = reader.read_gpus(
        fresh = bool(loading), visible = _visible_mask() if visible is ... else visible
    )
    holders = reader.holders()
    gpus, unplaced = split_gpu_memory(readings, holders, estimate_by_gpu = studio_estimate(models))
    return {
        "gpus": gpus,
        "models": [m.to_json() for m in models],
        "other_apps": unplaced,
        "loading": bool(loading),
    }


def _slots_loading() -> bool:
    try:
        from core.inference import model_slots

        return model_slots.any_loading()
    except Exception:
        return False


@router.get("")
async def get_resources(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> dict[str, Any]:
    """Per-GPU memory (Studio, other programs, free) and the models holding Studio's share."""
    from hub.utils.host_paths import redact_host_paths

    snapshot = await asyncio.to_thread(build_snapshot)
    snapshot = _without_host_process_names(snapshot)
    # Answered long after the loads that resolved these ids, like /status: no handle to restore.
    return redact_host_paths(snapshot, via_api_key = via_api_key)


def _without_host_process_names(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Which other programs run on the host is the owner's business: a managed account keeps the
    per-card "other programs" total and loses the names and PIDs behind it."""
    from hub.services.models import account_access

    if not account_access.managed_account():
        return snapshot
    for gpu in snapshot.get("gpus", []):
        gpu["apps"] = []
    snapshot["other_apps"] = []
    return snapshot
