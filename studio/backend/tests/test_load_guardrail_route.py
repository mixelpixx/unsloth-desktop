# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``/load``'s memory guardrail: a 409 the UI answers with "Load anyway".

Drives the real ``_load_model_impl`` GGUF path with the memory reading and the estimate
stubbed, so what is pinned is the route's behaviour: a refusal happens before the arbiter
handoff and before anything is unloaded, the override and ``off`` mode both load, and only
the explicit POST /load enforces it (auto-switch and preview keep loading as before).
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import core.inference.gpu_arbiter as arbiter
import core.inference.llama_keepwarm as keepwarm
import core.inference.load_guardrail as lg
import routes.inference as route
import utils.load_guardrail_settings as gs
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
from core.inference.load_verdict import GpuMemory
from models.inference import EstimateMemoryRequest, EstimateMemoryResponse, LoadRequest

GIB = 1024**3

_BREAKDOWN = SimpleNamespace(
    weights_bytes = int(18.0 * GIB),
    kv_bytes = int(1.0 * GIB),
    compute_bytes = int(0.6 * GIB),
    total_bytes = int(19.6 * GIB),
    gpu_bytes = int(19.6 * GIB),
    kv_on_gpu = True,
    kv_estimable = True,
    drafter_kv_unsized = False,
    adapters_unsized = False,
    n_ctx = 8192,
    layer_count = 40,
    gpu_layers = 41,
)
# One 24 GB card with another LLM app holding 17 GB of it.
_CARD = GpuMemory(0, int(6.1 * GIB), 24 * GIB, 0, 17 * GIB)
_HOLDERS = "LM Studio.exe (PID 1372) ~17.0 GB on GPU 0"


class _Harness:
    def __init__(self, monkeypatch):
        self.mode = "balanced"
        self.loads: list = []
        self.unloads: list = []
        self.acquired: list = []
        self.owner = [None]
        self.response = object()

        backend = LlamaCppBackend()
        backend.matches_load_source = lambda _intent: False
        backend.non_chat_gguf_refusal_for_intent = lambda _intent: None
        backend.host_offload_warning_for_intent = lambda _intent: None

        def _load(*, intent, load_cancel_event = None):
            self.loads.append(intent)
            return True

        backend.load_model = _load
        backend.unload_model = lambda *_a, **_k: self.unloads.append("llama-server")
        self.backend = backend
        # A resident Transformers model the GGUF path would unload once past the guard.
        self.unsloth_backend = SimpleNamespace(
            active_model_name = "resident/model",
            unload_model = lambda name: self.unloads.append(name),
        )
        config = SimpleNamespace(
            identifier = "owner/model.gguf",
            display_name = "model.gguf",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            is_audio = False,
            is_local = True,
            gguf_hf_repo = None,
            gguf_file = "/models/model.gguf",
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_variant = None,
        )
        self.config = config
        intent = GgufLoadIntent(model_identifier = config.identifier, gguf_path = config.gguf_file)

        async def _inline_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        async def _prepare_load_placement(*_args, **_kwargs):
            return route._LoadPlacement(None, None, False, False)

        async def _idle(**_kwargs):
            return None

        def _acquire(requested, register = None, **_kwargs):
            if register is not None:
                register()
            self.owner[0] = requested
            self.acquired.append(requested)

        def _release(requested):
            if self.owner[0] == requested:
                self.owner[0] = None

        m = monkeypatch
        m.setattr(route.asyncio, "to_thread", _inline_to_thread)
        m.setattr(
            route,
            "_resolve_model_identifier_for_request",
            lambda *_a, **_k: (config.identifier, config.identifier, False),
        )
        m.setattr(route, "resolve_effective_chat_template_override", lambda **_k: None)
        m.setattr(route, "get_inference_backend", lambda: self.unsloth_backend)
        m.setattr(route, "get_llama_cpp_backend", lambda: backend)
        m.setattr(route, "ModelConfig", SimpleNamespace(from_identifier = lambda **_k: config))
        m.setattr(route, "_hf_offline_if_unreachable_for", lambda *_a: nullcontext())
        m.setattr(route, "_resolve_inherited_extra_args", lambda *_a: None)
        m.setattr(route, "_prepare_load_placement", _prepare_load_placement)
        m.setattr(route, "_resolve_gguf_load_intent", lambda *_a, **_k: intent)
        m.setattr(route, "_guard_chat_load_against_training", lambda *_a, **_k: None)
        m.setattr(route, "_raise_if_sidecar_swap_in_progress", lambda: None)
        m.setattr(route, "_wait_for_model_switch_idle", _idle)
        m.setattr(route, "_close_load_event", lambda *_a, **_k: None)
        m.setattr(route, "_gguf_load_response", lambda *_a, **_k: self.response)
        m.setattr(route.api_monitor, "record_lifecycle", lambda **_k: object())
        m.setattr(route, "_request_used_api_key", lambda _request: False)
        m.setattr(keepwarm, "note_model_loaded", lambda _backend: None)
        m.setattr(arbiter, "acquire_for", _acquire)
        m.setattr(arbiter, "current_owner", lambda: self.owner[0])
        m.setattr(arbiter, "release", _release)

        # The guardrail's inputs: the estimate and the memory reading.
        m.setattr(route, "_local_gguf_main_path", lambda _config: config.gguf_file)
        m.setattr(route, "_localized_estimate_config", lambda cfg, _path: cfg)
        m.setattr(route, "_placement_priced_breakdown", lambda *_a, **_k: _BREAKDOWN)
        m.setattr(route, "_effective_parallel_slots", lambda n, **_k: n)
        m.setattr(lg, "measure_gpu_memory", lambda **_k: [_CARD])
        m.setattr(lg, "available_ram_bytes", lambda: 64 * GIB)
        m.setattr(lg, "describe_other_gpu_holders", lambda _indices: _HOLDERS)
        m.setattr(gs, "get_load_guardrail_mode", lambda: self.mode)

    def load(self, *, enforce = True, **request_fields):
        fields = dict(
            model_path = self.config.identifier,
            # A forced full offload at a named context: --fit off, so it allocates or dies.
            gpu_memory_mode = "manual",
            gpu_layers = 99,
            max_seq_length = 8192,
        )
        fields.update(request_fields)
        fastapi_request = SimpleNamespace(
            app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
        )
        kwargs = {"enforce_memory_guardrail": True} if enforce else {}
        return asyncio.run(
            route._load_model_impl(
                LoadRequest(**fields), fastapi_request, current_subject = "test-user", **kwargs
            )
        )


@pytest.fixture
def harness(monkeypatch):
    return _Harness(monkeypatch)


def test_a_load_that_will_crash_is_refused_with_a_structured_409(harness):
    with pytest.raises(HTTPException) as excinfo:
        harness.load()
    exc = excinfo.value
    assert exc.status_code == 409
    detail = exc.detail
    assert detail["code"] == "memory_overcommit"
    assert detail["error"] == "memory_overcommit"
    verdict = detail["verdict"]
    assert verdict["level"] == "likely_too_large"
    assert verdict["reason"] == "forced_gpu_overflow"
    assert verdict["needs_confirmation"] is True
    assert verdict["mode"] == "balanced"
    assert verdict["gpu_indices"] == [0]
    assert verdict["gpu_need_bytes"] == int(19.6 * GIB)
    assert verdict["gpu_free_bytes"] == int(6.1 * GIB)
    assert verdict["other_apps_note"] == _HOLDERS
    # An old client renders `message` from a dict detail: it must stand on its own.
    assert "allow_memory_overcommit" in detail["message"]
    assert _HOLDERS in detail["message"]


def test_a_refusal_leaves_the_resident_model_loaded(harness):
    with pytest.raises(HTTPException):
        harness.load()
    # Nothing past the guard ran: no GPU handoff (which evicts Images/Video), no unload of
    # the resident Transformers model, no llama-server launch or teardown.
    assert harness.acquired == []
    assert harness.unloads == []
    assert harness.loads == []
    assert harness.unsloth_backend.active_model_name == "resident/model"


def test_load_anyway_loads(harness):
    result = harness.load(allow_memory_overcommit = True)
    assert result is harness.response
    assert len(harness.loads) == 1
    assert harness.acquired == [arbiter.CHAT]


def test_mode_off_never_asks(harness):
    harness.mode = "off"
    assert harness.load() is harness.response
    assert len(harness.loads) == 1


def test_relaxed_lets_a_forced_split_through(harness):
    harness.mode = "relaxed"
    assert harness.load() is harness.response


def test_a_load_that_fits_is_not_asked_about(harness, monkeypatch):
    monkeypatch.setattr(lg, "measure_gpu_memory", lambda **_k: [GpuMemory(0, 23 * GIB, 24 * GIB)])
    assert harness.load() is harness.response


def test_auto_placement_spills_rather_than_asks(harness):
    # Auto moves layers to RAM itself; nothing here will crash.
    assert harness.load(gpu_memory_mode = "auto", gpu_layers = -1) is harness.response


def test_callers_that_do_not_enforce_load_as_before(harness):
    """Auto-switch and preview call the impl without the flag: unchanged behaviour."""
    assert harness.load(enforce = False) is harness.response
    assert len(harness.loads) == 1


def test_an_unpriceable_load_goes_ahead(harness, monkeypatch):
    # Not downloaded yet: nothing to read, so nothing to refuse on.
    monkeypatch.setattr(route, "_local_gguf_main_path", lambda _config: None)
    assert harness.load() is harness.response
    # And a failed memory probe is "unknown", never a refusal.
    monkeypatch.setattr(route, "_local_gguf_main_path", lambda config: config.gguf_file)
    monkeypatch.setattr(lg, "measure_gpu_memory", lambda **_k: None)
    assert harness.load() is harness.response


def test_only_the_explicit_load_route_enforces(monkeypatch):
    seen: list[dict] = []

    async def _record(_request, _fastapi_request, _subject, **kwargs):
        seen.append(kwargs)
        return None

    monkeypatch.setattr(route, "_load_model_impl", _record)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: SimpleNamespace())
    request = LoadRequest(model_path = "unsloth/A-GGUF")
    asyncio.run(route.load_model_gated(request, object(), "tester", user_initiated = True))
    asyncio.run(route.load_model_gated(request, object(), "tester"))
    assert seen[0].get("enforce_memory_guardrail") is True
    # Preview's call is byte-for-byte what it was: the flag is not even passed.
    assert "enforce_memory_guardrail" not in seen[1]


def test_the_estimate_carries_the_same_verdict(monkeypatch):
    monkeypatch.setattr(lg, "measure_gpu_memory", lambda **_k: [_CARD])
    monkeypatch.setattr(lg, "available_ram_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(gs, "get_load_guardrail_mode", lambda: "balanced")
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: SimpleNamespace(_process = None))
    request = EstimateMemoryRequest(
        model_path = "owner/model.gguf", n_ctx = 8192, gpu_memory_mode = "manual", gpu_layers = 99
    )
    info = route._estimate_route_verdict(_BREAKDOWN, request)
    assert info["level"] == "likely_too_large"
    assert info["needs_confirmation"] is True
    # The response model takes it as is, and an old client simply ignores the field.
    response = EstimateMemoryResponse(available = True, verdict = info)
    assert response.verdict is not None and response.verdict.level == "likely_too_large"
    assert EstimateMemoryResponse(available = True).verdict is None


def test_the_estimate_verdict_never_fails_the_estimate(monkeypatch):
    def _boom(**_k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(lg, "measure_gpu_memory", _boom)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: SimpleNamespace(_process = None))
    request = EstimateMemoryRequest(model_path = "owner/model.gguf")
    assert route._estimate_route_verdict(_BREAKDOWN, request) is None


@pytest.mark.parametrize(
    "n_ctx,extras,pinned",
    [
        (0, None, False),
        (None, None, False),
        (8192, None, True),
        (0, ["-c", "4096"], True),
        (8192, ["--ctx-size", "0"], False),  # -c 0 is "the model's own", last-wins
    ],
)
def test_what_counts_as_a_named_context(n_ctx, extras, pinned):
    assert route._context_is_pinned(n_ctx, extras) is pinned
