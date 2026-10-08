# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory coordination between inference and training. Uses live free VRAM to keep resident chat and STT
models when they fit. STT is evicted before chat when training needs memory. Also prices a training config
against its target GPUs before Start (estimate_training_fit), with the same estimator Start sizes with.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from hub.utils.hf_tokens import HfTokenArg, normalize_token
from loggers import get_logger

logger = get_logger(__name__)

# keep iff usable_gb >= required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB. Conservative: the probe sees only the chat
# model's current footprint, so reserve headroom for estimate error + KV-cache growth (KEEP_FLOOR_GB ~= 2 GB load
# buffer + 2 GB chat).
SAFETY_MARGIN = 1.15
KEEP_FLOOR_GB = 4.0

# Each extra GPU contributes less than its raw free memory (sharding overhead).
_MULTI_GPU_OVERHEAD = 0.85


def _free_vram_by_index(devices: List[Dict[str, Any]]) -> Dict[int, float]:
    """Map GPU index -> free VRAM (GB) from a get_visible_gpu_utilization() device list."""
    free_by_index: Dict[int, float] = {}
    for device in devices:
        total_gb = device.get("vram_total_gb")
        used_gb = device.get("vram_used_gb")
        if total_gb is None or used_gb is None:
            continue
        free_by_index[device["index"]] = max(total_gb - used_gb, 0.0)
    return free_by_index


def summarize_resident_chat() -> Dict[str, Any]:
    """Report which chat models hold GPU memory (resident even while loading). Never raises."""
    hf_name: Optional[str] = None
    gguf_name: Optional[str] = None
    loading: bool = False

    try:
        from core.inference import get_inference_backend
        inf = get_inference_backend()
        # active_model_name is set only on success; a mid-load model sits in
        # loading_models while already holding VRAM -> both count as resident.
        if inf.active_model_name or inf.loading_models:
            hf_name = inf.active_model_name or next(iter(inf.loading_models), None)
            # Any in-flight load (incl. a replacement while the old model is still
            # active) can't be sized -> flag it so the caller frees instead of keeps.
            if inf.loading_models:
                loading = True
    except Exception as e:
        logger.warning("Could not inspect inference backend: %s", e)

    try:
        from routes.inference import get_llama_cpp_backend
        llama = get_llama_cpp_backend()
        # is_active (not is_loaded): a mid-start server already allocates VRAM.
        # A confirmed CPU-only server (_gpu_offload_active is False) holds no VRAM.
        if llama.is_active and getattr(llama, "_gpu_offload_active", None) is not False:
            gguf_name = llama.model_identifier or "gguf"
            if not getattr(llama, "is_loaded", False):
                loading = True
    except Exception as e:
        logger.warning("Could not inspect GGUF backend: %s", e)

    return {
        "hf": hf_name,
        "gguf": gguf_name,
        "loading": loading,
        "any": bool(hf_name or gguf_name),
    }


def summarize_resident_stt() -> Dict[str, Any]:
    """Report the resident dictation model (either engine). Never raises."""
    try:
        from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar
        from core.inference.stt_sidecar import get_stt_sidecar

        sidecar = get_stt_sidecar()
        model = sidecar.loaded_model
        device = sidecar.device
        loading = sidecar.is_loading()
        # whisper.cpp holds GPU memory via its subprocess. Both engines can be live at once (engine switch or direct
        # /audio/stt/load), so always fold the GGUF sidecar in: a resident Transformers model must not mask a GGUF
        # server still binding its backend, or admission lets training launch into that startup and OOM.
        ggml = get_ggml_stt_sidecar()
        if not model:
            model = ggml.loaded_model
            device = device or ggml.device
        loading = loading or ggml.is_loading()
    except Exception as e:
        logger.warning("Could not inspect STT sidecar: %s", e)
        return {"model": None, "device": None, "loading": False, "any": False}

    try:
        from core.inference.stt_mtmd_sidecar import get_mtmd_stt_sidecar

        mtmd = get_mtmd_stt_sidecar()
        if not model:
            model = mtmd.loaded_model
            device = device or mtmd.device
        loading = loading or mtmd.is_loading()
    except Exception as e:
        logger.warning("Could not inspect mtmd STT sidecar: %s", e)

    return {
        "model": model,
        "device": device,
        "loading": loading,
        "any": bool(model or loading),
    }


def can_keep_chat_during_training(
    *,
    model_name: str,
    hf_token: HfTokenArg,
    training_type: str,
    load_in_4bit: bool,
    batch_size: int,
    max_seq_length: int,
    lora_rank: int,
    target_modules: Optional[List[str]],
    gradient_checkpointing: str,
    optimizer: str,
    gpu_ids: Optional[List[int]],
) -> Tuple[bool, Dict[str, Any]]:
    """Decide if a resident chat model can coexist with training given free VRAM. Reuses training's own
    estimator/selector so the decision matches later placement. Default-deny: anything we can't size
    returns False (unload)."""
    try:
        from utils.hardware import (
            DeviceType,
            auto_select_gpu_ids,
            estimate_required_model_memory_gb,
            get_device,
            get_visible_gpu_utilization,
            resolve_requested_gpu_ids,
        )

        if get_device() not in (DeviceType.CUDA, DeviceType.XPU):
            return False, {"mode": "non_accelerator", "reason": "non_accelerator"}

        # Full finetuning runs in 16-bit, so ignore the 4-bit request or we under-count.
        effective_4bit = False if training_type == "Full Finetuning" else load_in_4bit

        est_kwargs = dict(
            hf_token = normalize_token(hf_token),
            training_type = training_type,
            load_in_4bit = effective_4bit,
            batch_size = batch_size,
            max_seq_length = max_seq_length,
            lora_rank = lora_rank,
            target_modules = target_modules,
            gradient_checkpointing = gradient_checkpointing,
            optimizer = optimizer,
        )

        if gpu_ids:
            try:
                resolved = resolve_requested_gpu_ids(gpu_ids)
            except ValueError:
                # Invalid ids -> start_training will 400 first, so don't unload.
                return True, {"mode": "explicit", "reason": "invalid_gpu_ids"}

            required_gb, est_meta = estimate_required_model_memory_gb(model_name, **est_kwargs)
            if required_gb is None:
                return False, {"mode": "explicit", "reason": "estimate_unavailable"}

            free_by_index = _free_vram_by_index(get_visible_gpu_utilization().get("devices", []))

            # A requested GPU missing from the device list contributes 0.
            free_vals = [free_by_index.get(i, 0.0) for i in resolved]
            ranked = sorted(free_vals, reverse = True)
            usable_gb = (
                ranked[0] + sum(f * _MULTI_GPU_OVERHEAD for f in ranked[1:]) if ranked else 0.0
            )
            aggregate_fits = usable_gb >= required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB

            # Activations don't shard: enforce a per-GPU floor so an uneven split (free [45, 10]) cannot be
            # kept into an OOM the aggregate misses.
            per_gpu_fits = True
            min_free_gb = min(free_vals) if free_vals else 0.0
            if len(resolved) > 1:
                min_per_gpu_gb = est_meta.get("vram_breakdown", {}).get(
                    f"min_per_gpu_{len(resolved)}"
                )
                if min_per_gpu_gb is not None:
                    per_gpu_fits = min_free_gb >= min_per_gpu_gb

            keep = aggregate_fits and per_gpu_fits
            return keep, {
                "mode": "explicit",
                "required_gb": required_gb,
                "usable_gb": round(usable_gb, 3),
                "min_free_gb": round(min_free_gb, 3),
            }

        # Auto: same call start_training makes later; reuse its sizing metadata.
        _selected, meta = auto_select_gpu_ids(model_name, **est_kwargs)
        mode = meta.get("selection_mode")
        required_gb = meta.get("required_gb")
        usable_gb = meta.get("usable_gb")
        keep = (
            mode == "auto"
            and required_gb is not None
            and usable_gb is not None
            and usable_gb >= required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB
        )
        return keep, {
            "mode": mode,
            "required_gb": required_gb,
            "usable_gb": usable_gb,
        }
    except Exception as e:
        # Never let a sizing failure keep a chat model loaded into a training OOM.
        logger.warning("Chat-coexistence probe failed; will unload: %s", e)
        return False, {"reason": "probe_error", "error": str(e)}


def can_load_chat_during_training(
    *,
    model_name: str,
    hf_token: Optional[str],
    load_in_4bit: bool,
    max_seq_length: int,
    requested_gpu_ids: Optional[List[int]],
    is_gguf: bool = False,
    gpu_ids_are_vulkan_ordinals: bool = False,
    vulkan_free_vram_gb: Optional[Dict[int, float]] = None,
    required_override_gb: Optional[float] = None,
    single_device_gpu: Optional[str] = None,
    post_handoff_free_gpu_vram_gb: Optional[Dict[int, float]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Decide if a NEW chat model can load without OOMing active training (inverse of
    can_keep_chat_during_training: training is already resident, so size the chat model against the free VRAM
    that remains). Sizes/places it the same way the loader will: HF auto reuses auto_select_gpu_ids; HF explicit
    requires an even-share per-GPU floor for device_map="balanced"; GGUF sizes from required_override_gb over
    the visible pool. ``single_device_gpu`` is the exact physical device token selected by a single-device
    runner. `load_in_4bit` must be effective (LoRA can flip 4-bit -> 16-bit). CPU/MLX allows the load;
    default-deny on any CUDA/XPU case it can't size, so a load never OOMs training."""
    try:
        from utils.hardware import (
            DeviceType,
            auto_select_gpu_ids,
            estimate_required_model_memory_gb,
            get_device,
            get_visible_gpu_utilization,
            resolve_requested_gpu_ids,
        )

        if get_device() not in (DeviceType.CUDA, DeviceType.XPU):
            return True, {"mode": "non_accelerator", "reason": "non_accelerator"}

        est_kwargs = dict(
            hf_token = hf_token or None,
            training_type = None,
            load_in_4bit = load_in_4bit,
            max_seq_length = max_seq_length or 2048,
        )

        # A native-audio switch's post-handoff snapshot already combines live free memory with the
        # outgoing Studio backend, so do not re-read and add those values here.
        if not requested_gpu_ids and not is_gguf and post_handoff_free_gpu_vram_gb is not None:
            required_gb = required_override_gb
            if required_gb is None:
                required_gb, _meta = estimate_required_model_memory_gb(model_name, **est_kwargs)
            free_vals = [
                max(float(effective), 0.0) for effective in post_handoff_free_gpu_vram_gb.values()
            ]
            usable_gb = max(free_vals) if free_vals else None
            needed_gb = (
                round(required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB, 3)
                if required_gb is not None
                else None
            )
            fits = required_gb is not None and usable_gb is not None and usable_gb >= needed_gb
            return fits, {
                "mode": "native_post_handoff",
                "required_gb": required_gb,
                "usable_gb": usable_gb,
                "needed_gb": needed_gb,
            }

        # HF auto: reuse the loader's selector; fits iff its pick clears the margin.
        if not requested_gpu_ids and not is_gguf:
            _selected, meta = auto_select_gpu_ids(
                model_name,
                required_override_gb = required_override_gb,
                **est_kwargs,
            )
            mode = meta.get("selection_mode")
            required_gb = meta.get("required_gb")
            usable_gb = meta.get("usable_gb")
            needed_gb = (
                round(required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB, 3)
                if required_gb is not None
                else None
            )
            fits = (
                mode == "auto"
                and required_gb is not None
                and usable_gb is not None
                and usable_gb >= needed_gb
            )
            return fits, {
                "mode": mode,
                "required_gb": required_gb,
                "usable_gb": usable_gb,
                "needed_gb": needed_gb,
            }

        # Explicit GPUs, or GGUF: size directly and check live free VRAM.
        uses_vulkan_memory = vulkan_free_vram_gb is not None
        if is_gguf and uses_vulkan_memory:
            mode = "gguf_vulkan"
        elif single_device_gpu is not None:
            mode = "single_device"
        elif is_gguf:
            mode = "gguf"
        else:
            mode = "explicit"
        required_gb = required_override_gb
        if required_gb is None:
            required_gb, _meta = estimate_required_model_memory_gb(model_name, **est_kwargs)
        if required_gb is None:
            return False, {"mode": mode, "reason": "estimate_unavailable"}

        free_by_index = (
            vulkan_free_vram_gb
            if uses_vulkan_memory
            else _free_vram_by_index(get_visible_gpu_utilization().get("devices", []))
        )
        if requested_gpu_ids and gpu_ids_are_vulkan_ordinals:
            free_vals = [free_by_index.get(int(gpu_id), 0.0) for gpu_id in requested_gpu_ids]
        elif single_device_gpu is not None:
            token = str(single_device_gpu).strip()
            if not token:
                # Empty token = a CPU-only single-device runner, for example a CPU diffusion GGUF: it uses no GPU VRAM, so it
                # never threatens active training and can always load.
                return True, {"mode": "single_device", "reason": "cpu_only"}
            try:
                selected_gpu = int(token)
                if selected_gpu < 0:
                    raise ValueError
            except (TypeError, ValueError):
                # A non-numeric device token (CUDA UUID / MIG handle) has no free-VRAM index, but the runner still drives
                # ONE device: size against the worst-case visible device, never the aggregate pool, or a single-device load
                # is OK'd on capacity it cannot use.
                free_vals = [min(free_by_index.values())] if free_by_index else []
            else:
                free_vals = [free_by_index.get(selected_gpu, 0.0)]
        elif requested_gpu_ids:
            # Invalid ids -> load_model 400s first, so don't block; missing id = 0.
            try:
                resolved = resolve_requested_gpu_ids(requested_gpu_ids)
            except ValueError:
                return True, {"mode": mode, "reason": "invalid_gpu_ids"}
            free_vals = [free_by_index.get(i, 0.0) for i in resolved]
        else:
            # GGUF self-placement / auto Vulkan (no requested ids): llama.cpp picks
            # the GPU(s), so any visible GPU is a candidate -> size the whole pool.
            free_vals = list(free_by_index.values())

        if not free_vals:
            return False, {"mode": mode, "reason": "no_visible_gpus"}

        ranked = sorted(free_vals, reverse = True)
        usable_gb = ranked[0] + sum(f * _MULTI_GPU_OVERHEAD for f in ranked[1:])
        needed_gb = required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB
        aggregate_fits = usable_gb >= needed_gb

        # Explicit HF placement uses balanced sharding across a known number of GPUs. GGUF pins are candidate pools:
        # llama.cpp may narrow an uneven pool to the smallest fitting subset, so only their aggregate matters.
        min_free_gb = min(free_vals)
        per_gpu_fits = True
        per_gpu_needed_gb = None
        if mode == "explicit" and len(free_vals) > 1:
            per_gpu_needed_gb = needed_gb / len(free_vals)
        if per_gpu_needed_gb is not None:
            per_gpu_fits = min_free_gb >= per_gpu_needed_gb

        info = {
            "mode": mode,
            "required_gb": round(required_gb, 3),
            "usable_gb": round(usable_gb, 3),
            "needed_gb": round(needed_gb, 3),
            "min_free_gb": round(min_free_gb, 3),
        }
        if per_gpu_needed_gb is not None:
            info["per_gpu_needed_gb"] = round(per_gpu_needed_gb, 3)
        return aggregate_fits and per_gpu_fits, info
    except Exception as e:
        # Never let a sizing failure load a chat model into a training OOM.
        logger.warning("Chat-load coexistence probe failed; will refuse: %s", e)
        return False, {"reason": "probe_error", "error": str(e)}


def free_chat_models_for_training(reason: str) -> List[str]:
    """Unload every resident chat model (HF/MLX orchestrator + GGUF server) to free
    VRAM for training. Each backend isolated. Returns labels of what was freed."""
    freed: List[str] = []

    try:
        from core.inference import get_inference_backend
        inf = get_inference_backend()
        # No CPU exemption here, unlike the GGUF branch and the STT sidecars: it would key off a marker the
        # orchestrator writes rather than the worker that masked, and a marker that disagreed is an OOM mid-training.
        # Freeing a model that held no VRAM only costs a reload.
        if inf.active_model_name or inf.loading_models:
            name = inf.active_model_name or next(iter(inf.loading_models), None)
            logger.info(
                "Unloading inference model '%s' to free GPU memory for training (%s)",
                name,
                reason,
            )
            inf._shutdown_subprocess()
            inf.active_model_name = None
            inf.models.clear()
            inf.loading_models.clear()
            freed.append(f"hf:{name}")
    except Exception as e:
        logger.warning("Could not unload inference model: %s", e)

    try:
        from routes.inference import get_llama_cpp_backend
        llama = get_llama_cpp_backend()
        # CPU-only GGUF holds no VRAM, so killing it can't help (see summarize).
        if llama.is_active and getattr(llama, "_gpu_offload_active", None) is not False:
            name = llama.model_identifier or "gguf"
            logger.info(
                "Unloading GGUF chat model '%s' to free GPU memory for training (%s)",
                name,
                reason,
            )
            llama.unload_model()
            freed.append(f"gguf:{name}")
    except Exception as e:
        logger.warning("Could not unload GGUF chat model: %s", e)

    return freed


def _stt_sidecar_holds_no_vram(sidecar) -> bool:
    """True only when the resident dictation model is provably in CPU RAM. Conservative on purpose: anything
    unreadable answers False and the sidecar is freed as before. Skipping one that does hold VRAM would starve
    the run this is making room for, which is far worse than a needless reload.
    """
    try:
        device = getattr(sidecar, "device", None)
        if isinstance(device, str) and device.strip().lower() == "cpu":
            return True
        # whisper.cpp and llama.cpp report a runtime name rather than a device, so read the flag each sets when it
        # started without the GPU. Prefer the fact over the wish: mtmd's _gpu_disabled is what the live server was
        # started with, while _forced_cpu is the standing preference, recorded even on the branch that does NOT restart
        # a server with a request in flight, so reading it reports a server still at -ngl 99 as holding no VRAM.
        gpu_disabled = getattr(sidecar, "_gpu_disabled", None)
        if gpu_disabled is not None:
            return gpu_disabled is True
        # whisper.cpp sets _forced_cpu only alongside a spawned --no-gpu and clears it
        # on release, so there it is the fact.
        return getattr(sidecar, "_forced_cpu", False) is True
    except Exception:  # noqa: BLE001 - a probe must never fail the release it precedes
        return False


def free_stt_model_for_training(reason: str) -> List[str]:
    """Unload the dictation model(s) before training. Never raises. The Transformers and GGUF sidecars
    are freed under independent exception boundaries so a failure unloading one backend never skips
    freeing the other (both can hold accelerator memory at once after an engine switch)."""
    freed: List[str] = []
    try:
        from core.inference.stt_sidecar import get_stt_sidecar
        sidecar = get_stt_sidecar()
        if sidecar.is_loading() and sidecar.cancel_pending_load():
            logger.info("Cancelling STT model load for training (%s)", reason)
            # The loader may still be in from_pretrained()/.to(device) holding
            # VRAM; wait for it to observe the cancel and release first.
            sidecar.wait_for_load_to_settle()
            # A load that finished before seeing the cancel leaves a resident
            # model; unload it so training gets the memory back.
            if sidecar.loaded_model:
                sidecar.unload()
            freed.append("stt:loading")
        else:
            model = sidecar.loaded_model
            if model and _stt_sidecar_holds_no_vram(sidecar):
                logger.info(
                    "Keeping CPU-placed STT model '%s' through training (%s)", model, reason
                )
            elif model:
                logger.info("Unloading STT model '%s' for training (%s)", model, reason)
                sidecar.unload()
                freed.append(f"stt:{model}")
    except Exception as e:
        logger.warning("Could not unload Transformers STT model: %s", e)

    # Check the GGUF sidecar even after a cancelled or failed Transformers unload; both engines can hold memory at once.
    try:
        from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar
        ggml = get_ggml_stt_sidecar()
        if ggml.is_loading() and ggml.cancel_pending_load():
            logger.info("Cancelling GGUF STT model load for training (%s)", reason)
            # whisper-server may still be binding its backend; wait for the cancelled startup to be killed and
            # reaped before training claims the memory (loaded_model stays unset until it is ready).
            ggml.wait_for_load_to_settle()
            if ggml.loaded_model:
                ggml.unload()
            freed.append("stt:gguf-loading")
        else:
            ggml_model = ggml.loaded_model
            if ggml_model and _stt_sidecar_holds_no_vram(ggml):
                logger.info(
                    "Keeping CPU-placed GGUF STT model '%s' through training (%s)",
                    ggml_model,
                    reason,
                )
            elif ggml_model:
                logger.info("Unloading GGUF STT model '%s' for training (%s)", ggml_model, reason)
                ggml.unload()
                freed.append(f"stt:{ggml_model}")
    except Exception as e:
        logger.warning("Could not unload GGUF STT model: %s", e)

    try:
        from core.inference.stt_mtmd_sidecar import get_mtmd_stt_sidecar
        mtmd = get_mtmd_stt_sidecar()
        if mtmd.is_loading() and mtmd.cancel_pending_load():
            logger.info("Cancelling mtmd STT model load for training (%s)", reason)
            # llama-server only becomes reachable through unload() once it is ready, so a startup has to be
            # cancelled and reaped instead. Wait for that before training claims the memory it is allocating.
            mtmd.wait_for_load_to_settle()
            if mtmd.loaded_model:
                mtmd.unload()
            freed.append("stt:mtmd-loading")
        else:
            mtmd_model = mtmd.loaded_model
            if mtmd_model and _stt_sidecar_holds_no_vram(mtmd):
                logger.info(
                    "Keeping CPU-placed mtmd STT model '%s' through training (%s)",
                    mtmd_model,
                    reason,
                )
            elif mtmd_model:
                logger.info("Unloading mtmd STT model '%s' for training (%s)", mtmd_model, reason)
                mtmd.unload()
                freed.append(f"stt:{mtmd_model}")
    except Exception as e:
        logger.warning("Could not unload mtmd STT model: %s", e)

    return freed


def coordinate_models_for_training(
    can_keep: Callable[[], Tuple[bool, Dict[str, Any]]],
) -> List[str]:
    """Keep resident models when they fit, evicting STT before chat."""
    resident_chat = summarize_resident_chat()
    resident_stt = summarize_resident_stt()
    if not resident_chat["any"] and not resident_stt["any"]:
        return []

    if resident_chat.get("loading"):
        freed = free_stt_model_for_training(reason = "chat model still loading")
        freed += free_chat_models_for_training(reason = "chat model still loading")
        return freed

    freed: List[str] = []
    if resident_stt.get("loading"):
        released_stt = free_stt_model_for_training(reason = "STT model still loading")
        freed += released_stt
        resident_stt = (
            {"model": None, "device": None, "loading": False, "any": False}
            if released_stt
            else summarize_resident_stt()
        )
        if not resident_chat["any"] and not resident_stt["any"]:
            return freed

    keep, info = can_keep()
    if keep:
        logger.info(
            "Keeping resident models loaded during training (free ~%s GB, needs ~%s GB): %s",
            info.get("usable_gb"),
            info.get("required_gb"),
            {"chat": resident_chat, "stt": resident_stt},
        )
        return freed

    if resident_stt["any"]:
        freed += free_stt_model_for_training(reason = "insufficient training memory")
        if not resident_chat["any"]:
            return freed
        keep, _info = can_keep()
        if keep:
            logger.info("Keeping chat model loaded after freeing STT: %s", resident_chat)
            return freed

    freed += free_chat_models_for_training(
        reason = "insufficient VRAM to run training alongside chat",
    )
    return freed


# A fit above this share of the usable memory reads as tight rather than clean. The same 0.85 the frontend's
# MEMORY_FIT_TIGHT_RATIO (src/lib/memory/thresholds.ts) draws the Load Model panel's line at, so the two
# surfaces cannot call one footprint differently.
FIT_TIGHT_RATIO = 0.85

# VramBreakdown.to_gb_dict()'s parts. Anything else in vram_breakdown (min_per_gpu_N) is not a part.
_BREAKDOWN_KEYS = (
    "model_weights_gb",
    "lora_adapters_gb",
    "optimizer_states_gb",
    "gradients_gb",
    "activations_gb",
    "cuda_overhead_gb",
    "total_gb",
)


def _usable_training_gb(free_vals: List[float]) -> float:
    """Usable memory across a GPU set, with auto_select_gpu_ids' arithmetic: the roomiest card counts in full and
    each extra one at _MULTI_GPU_OVERHEAD. Studio splits layers across cards in one process, so a second GPU
    adds room, not speed."""
    ranked = sorted(free_vals, reverse = True)
    return ranked[0] + sum(f * _MULTI_GPU_OVERHEAD for f in ranked[1:]) if ranked else 0.0


def classify_training_fit(
    required_gb: Optional[float],
    usable_gb: Optional[float],
    *,
    min_per_gpu_gb: Optional[float] = None,
    min_free_gb: Optional[float] = None,
) -> Tuple[str, Optional[str]]:
    """(verdict, reason) for a priced config against usable memory. The two checks auto-selection runs: the
    aggregate, then the per-GPU floor (activations do not shard, so every card in a split needs its own)."""
    if required_gb is None or usable_gb is None:
        return "unknown", "estimate_unavailable"
    # NaN and inf fail every comparison below and would fall through to a confident "fits".
    if not (math.isfinite(required_gb) and math.isfinite(usable_gb)) or required_gb <= 0:
        return "unknown", "estimate_unavailable"
    if usable_gb <= 0 or required_gb > usable_gb:
        return "exceeds", "insufficient_memory"
    if min_per_gpu_gb is not None and min_free_gb is not None and min_free_gb < min_per_gpu_gb:
        return "exceeds", "per_gpu_minimum"
    if required_gb / usable_gb > FIT_TIGHT_RATIO:
        return "tight", None
    return "fits", None


def _training_fit_on_gpus(
    required_gb: Optional[float],
    estimate_meta: Dict[str, Any],
    gpu_ids: List[int],
    free_by_index: Dict[int, float],
) -> Dict[str, Any]:
    """Verdict for a priced config on a concrete GPU set."""
    if not gpu_ids or any(gpu_id not in free_by_index for gpu_id in gpu_ids):
        # A card nvidia-smi could not read has no free figure, and 0 would read as "full".
        return {
            "verdict": "unknown",
            "reason": "no_gpu_telemetry",
            "usable_gb": None,
            "min_per_gpu_gb": None,
        }
    free_vals = [free_by_index[gpu_id] for gpu_id in gpu_ids]
    usable_gb = _usable_training_gb(free_vals)
    min_per_gpu_gb = None
    if len(gpu_ids) > 1:
        min_per_gpu_gb = (estimate_meta.get("vram_breakdown") or {}).get(
            f"min_per_gpu_{len(gpu_ids)}"
        )
    verdict, reason = classify_training_fit(
        required_gb,
        usable_gb,
        min_per_gpu_gb = min_per_gpu_gb,
        min_free_gb = min(free_vals),
    )
    return {
        "verdict": verdict,
        "reason": reason,
        "usable_gb": round(usable_gb, 3),
        "min_per_gpu_gb": min_per_gpu_gb,
    }


def _training_gpu_rows(devices: List[Dict[str, Any]], selected: List[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for device in devices:
        index = device.get("index")
        if not isinstance(index, int):
            continue
        total_gb = device.get("vram_total_gb")
        used_gb = device.get("vram_used_gb")
        rows.append(
            {
                "index": index,
                "name": device.get("name"),
                "total_gb": total_gb,
                "free_gb": (
                    round(max(total_gb - used_gb, 0.0), 3)
                    if total_gb is not None and used_gb is not None
                    else None
                ),
                "selected": index in selected,
            }
        )
    return rows


def estimate_training_fit(
    *,
    model_name: str,
    hf_token: HfTokenArg,
    training_type: str,
    load_in_4bit: bool,
    batch_size: int,
    max_seq_length: int,
    lora_rank: int,
    target_modules: Optional[List[str]],
    gradient_checkpointing: str,
    optimizer: str,
    gpu_ids: Optional[List[int]],
    four_bit_available: bool = True,
) -> Dict[str, Any]:
    """Price a training config against the GPUs it would run on, before Start.

    Sizes and places with the estimator and auto-selector Start itself uses, so the preview names the GPUs Start
    would pick. Reads only what those read (config.json / Hub metadata, nvidia-smi): nothing is downloaded and
    nothing is allocated on a GPU. Never raises: anything it cannot price is verdict "unknown" with a reason,
    because a failed preview must not stand between the user and Start."""
    explicit = bool(gpu_ids)
    result: Dict[str, Any] = {
        "verdict": "unknown",
        "reason": None,
        "required_gb": None,
        "estimation_mode": None,
        "breakdown": None,
        "selection_mode": "explicit" if explicit else "auto",
        "gpu_ids": [],
        "usable_gb": None,
        "min_per_gpu_gb": None,
        "gpus": [],
        "suggestion": None,
    }
    try:
        from utils.hardware import (
            DeviceType,
            auto_select_gpu_ids,
            estimate_required_model_memory_gb,
            get_device,
            get_visible_gpu_utilization,
            resolve_requested_gpu_ids,
        )
        from utils.hardware.hardware import (
            attention_preview_estimates,
            reject_gpu_ids_without_torch_kernels,
        )

        # Auto-selection's own gate: per-card free VRAM exists on CUDA/ROCm and XPU only. MLX trains out of
        # host memory, which is not a ceiling this verdict can be drawn against.
        if get_device() not in (DeviceType.CUDA, DeviceType.XPU):
            result["reason"] = "unsupported_device"
            return result

        # Full finetuning runs in 16-bit, so a 4-bit flag left over from QLoRA would under-count it.
        effective_4bit = False if training_type == "Full Finetuning" else load_in_4bit
        # Normalized the way /start hands them to the selector, so the two price the same config.
        est_kwargs: Dict[str, Any] = dict(
            hf_token = normalize_token(hf_token),
            training_type = training_type,
            load_in_4bit = effective_4bit,
            batch_size = batch_size,
            max_seq_length = max_seq_length,
            lora_rank = lora_rank,
            target_modules = target_modules or None,
            gradient_checkpointing = (gradient_checkpointing or "").strip() or "unsloth",
            optimizer = optimizer,
        )

        devices = get_visible_gpu_utilization().get("devices", []) or []
        free_by_index = _free_vram_by_index(devices)

        explicit_ids: List[int] = []
        if explicit:
            try:
                explicit_ids = resolve_requested_gpu_ids(gpu_ids)
                reject_gpu_ids_without_torch_kernels(explicit_ids)
            except ValueError as exc:
                # /start would 400 these; say so rather than price a target that cannot run.
                logger.info("Training fit estimate: gpu_ids %s rejected: %s", gpu_ids, exc)
                result["reason"] = "invalid_gpu_ids"
                result["gpus"] = _training_gpu_rows(devices, [])
                return result

        def _price(**overrides: Any) -> Tuple[Optional[float], Dict[str, Any], List[int]]:
            kwargs = {**est_kwargs, **overrides}
            # Without unsloth's import: a preview must not pin a CUDA context in the server process.
            if explicit:
                with attention_preview_estimates():
                    required, meta = estimate_required_model_memory_gb(model_name, **kwargs)
                return required, meta, list(explicit_ids)
            # One estimator call: the selector runs it and hands back its metadata, breakdown included.
            with attention_preview_estimates():
                selected, meta = auto_select_gpu_ids(model_name, **kwargs)
            # None means training inherits every parent-visible card (non-numeric visibility mask).
            ids = list(selected) if selected else sorted(free_by_index)
            return meta.get("required_gb"), meta, ids

        required_gb, meta, selected_ids = _price()
        result["gpu_ids"] = selected_ids
        result["gpus"] = _training_gpu_rows(devices, selected_ids)
        if required_gb is None:
            result["reason"] = "estimate_unavailable"
            return result

        result["required_gb"] = round(float(required_gb), 3)
        estimation_mode = meta.get("estimation_mode")
        if estimation_mode in ("detailed", "fallback"):
            result["estimation_mode"] = estimation_mode
        breakdown = meta.get("vram_breakdown")
        if isinstance(breakdown, dict) and all(
            isinstance(breakdown.get(key), (int, float)) for key in _BREAKDOWN_KEYS
        ):
            result["breakdown"] = {key: float(breakdown[key]) for key in _BREAKDOWN_KEYS}
        result.update(_training_fit_on_gpus(required_gb, meta, selected_ids, free_by_index))

        if result["verdict"] != "exceeds":
            return result

        # Cheaper settings the estimator can price, biggest lever first. QLoRA only for the methods it replaces
        # (CPT is its own method) and only when the run could actually load 4-bit.
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        if (
            four_bit_available
            and not effective_4bit
            and training_type in ("LoRA/QLoRA", "Full Finetuning")
        ):
            candidates.append(("qlora", {"training_type": "LoRA/QLoRA", "load_in_4bit": True}))
        if batch_size > 1:
            candidates.append(("batch_size", {"batch_size": 1}))
        for kind, overrides in candidates:
            alt_required, alt_meta, alt_ids = _price(**overrides)
            if alt_required is None:
                continue
            alt_fit = _training_fit_on_gpus(alt_required, alt_meta, alt_ids, free_by_index)
            if alt_fit["verdict"] in ("fits", "tight"):
                result["suggestion"] = {
                    "kind": kind,
                    "required_gb": round(float(alt_required), 3),
                    "verdict": alt_fit["verdict"],
                    "gpu_ids": alt_ids,
                    "batch_size": overrides.get("batch_size"),
                }
                break
        return result
    except Exception as e:
        logger.warning("Training fit estimate failed: %s", e)
        result.update(verdict = "unknown", reason = "estimate_failed", suggestion = None)
        return result
