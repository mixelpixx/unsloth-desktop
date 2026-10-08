# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /api/train/estimate -- the training fit planner's route.

It must price a config with the estimator and auto-selector Start uses (so the preview names the
GPUs Start would pick), honour explicit gpu_ids, draw verdicts at the shared thresholds, degrade
to "unknown" instead of failing, and never answer an unauthenticated caller.
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.training as tr
import routes.training_vram as tv
from models.training import TrainingEstimateRequest
from utils.hardware import DeviceType

_BREAKDOWN = {
    "model_weights_gb": 14.9,
    "lora_adapters_gb": 0.08,
    "optimizer_states_gb": 0.16,
    "gradients_gb": 0.08,
    "activations_gb": 1.2,
    "cuda_overhead_gb": 1.4,
    "total_gb": 17.82,
    # Not a part: the selector's per-GPU floor for a two-card split.
    "min_per_gpu_2": 9.0,
}

# Two 24 GiB cards: GPU 0 has 20 free, GPU 1 has 23.5 free.
_DEVICES = [
    {"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 4.0},
    {"index": 1, "vram_total_gb": 24.0, "vram_used_gb": 0.5},
]


def _request(**overrides) -> TrainingEstimateRequest:
    values = dict(
        model_name = "unsloth/tiny-model",
        training_type = "LoRA/QLoRA",
        load_in_4bit = False,
        batch_size = 2,
        max_seq_length = 2048,
        lora_r = 16,
    )
    values.update(overrides)
    return TrainingEstimateRequest(**values)


@pytest.fixture
def hardware(monkeypatch):
    """Stub the hardware layer: a CUDA host with _DEVICES, an estimator returning `required`."""
    state = {
        "required": {False: 17.82, True: 7.5},  # keyed by load_in_4bit
        "estimate_calls": [],
        "auto_calls": [],
        "devices": list(_DEVICES),
        "device": DeviceType.CUDA,
    }

    def _estimate(model_name, **kwargs):
        state["estimate_calls"].append(kwargs)
        required = state["required"][kwargs["load_in_4bit"]]
        if required is None:
            return None, {"model_size_source": "unavailable", "required_gb": None}
        breakdown = dict(_BREAKDOWN, total_gb = required)
        return required, {
            "required_gb": required,
            "estimation_mode": "detailed",
            "vram_breakdown": breakdown,
        }

    def _auto(model_name, **kwargs):
        state["auto_calls"].append(kwargs)
        required, meta = _estimate(model_name, **kwargs)
        meta = dict(meta, selection_mode = "auto")
        if required is None:
            return [0, 1], dict(meta, selection_mode = "fallback_all")
        # Roomiest card first, as the real selector ranks them.
        if required <= 23.5:
            return [1], meta
        return [0, 1], dict(meta, selection_mode = "fallback_all")

    def _resolve(gpu_ids, **_kwargs):
        if any(gpu_id not in (0, 1) for gpu_id in gpu_ids):
            raise ValueError(f"Invalid gpu_ids {gpu_ids}")
        return list(gpu_ids)

    monkeypatch.setattr("utils.hardware.get_device", lambda: state["device"])
    monkeypatch.setattr(
        "utils.hardware.get_visible_gpu_utilization", lambda: {"devices": state["devices"]}
    )
    monkeypatch.setattr("utils.hardware.estimate_required_model_memory_gb", _estimate)
    monkeypatch.setattr("utils.hardware.auto_select_gpu_ids", _auto)
    monkeypatch.setattr("utils.hardware.resolve_requested_gpu_ids", _resolve)
    monkeypatch.setattr(
        "utils.hardware.hardware.reject_gpu_ids_without_torch_kernels", lambda _ids: None
    )
    monkeypatch.setattr("utils.hardware.ensure_hardware_detected", lambda: None)
    return state


def _post(request: TrainingEstimateRequest, *, via_api_key = False):
    return asyncio.run(
        tr.estimate_training_memory(
            request = request,
            current_subject = "test-user",
            via_api_key = via_api_key,
        )
    )


def test_route_returns_the_estimators_breakdown_and_auto_selection(hardware):
    response = _post(_request())

    assert response.selection_mode == "auto"
    assert response.gpu_ids == [1]
    assert response.required_gb == pytest.approx(17.82)
    assert response.estimation_mode == "detailed"
    # The estimator's parts, verbatim; the selector's min_per_gpu_N is not a part.
    assert response.breakdown is not None
    assert response.breakdown.model_weights_gb == pytest.approx(14.9)
    assert response.breakdown.activations_gb == pytest.approx(1.2)
    assert response.breakdown.total_gb == pytest.approx(17.82)
    assert "min_per_gpu_2" not in response.breakdown.model_dump()
    # Every visible card, with its free memory, and which ones the run would use.
    assert [(g.index, g.total_gb, g.free_gb, g.selected) for g in response.gpus] == [
        (0, 24.0, 20.0, False),
        (1, 24.0, 23.5, True),
    ]
    assert response.usable_gb == pytest.approx(23.5)
    # 17.82 / 23.5 = 0.76: under the tight ratio.
    assert response.verdict == "fits"
    assert response.suggestion is None
    # Priced once, through the selector Start uses, not a second estimator call.
    assert len(hardware["auto_calls"]) == 1
    assert hardware["estimate_calls"] == hardware["auto_calls"]


def test_route_prices_what_start_would_size(hardware):
    _post(
        _request(
            training_type = "Full Finetuning",
            load_in_4bit = True,
            target_modules = [],
            gradient_checkpointing = "  ",
        )
    )
    kwargs = hardware["auto_calls"][0]
    # Full finetuning trains 16-bit whatever the flag says; blank settings take Start's defaults.
    assert kwargs["load_in_4bit"] is False
    assert kwargs["target_modules"] is None
    assert kwargs["gradient_checkpointing"] == "unsloth"
    assert kwargs["training_type"] == "Full Finetuning"
    # A UI session's token is entitled to ambient credentials, as /start treats it.
    assert kwargs["hf_token"] is None


def test_route_honours_explicit_gpu_ids(hardware):
    response = _post(_request(gpu_ids = [0, 1]))

    assert response.selection_mode == "explicit"
    assert response.gpu_ids == [0, 1]
    # Never re-ranked by the auto-selector: the user's pick is the target.
    assert hardware["auto_calls"] == []
    assert len(hardware["estimate_calls"]) == 1
    # Same arithmetic as the selector: roomiest card in full, the other at 0.85.
    assert response.usable_gb == pytest.approx(23.5 + 20.0 * 0.85)
    assert response.min_per_gpu_gb == pytest.approx(9.0)
    assert [g.selected for g in response.gpus] == [True, True]
    assert response.verdict == "fits"


def test_single_pinned_gpu_is_measured_on_its_own_free_memory(hardware):
    response = _post(_request(gpu_ids = [0]))
    # 17.82 of 20 free = 0.89: tight on GPU 0, though GPU 1 alone would fit cleanly.
    assert response.gpu_ids == [0]
    assert response.usable_gb == pytest.approx(20.0)
    assert response.verdict == "tight"


def test_invalid_gpu_ids_are_unknown_not_an_error(hardware):
    response = _post(_request(gpu_ids = [7]))
    assert response.verdict == "unknown"
    assert response.reason == "invalid_gpu_ids"
    assert hardware["estimate_calls"] == []


def test_exceeds_suggests_qlora_when_4bit_fits(hardware):
    # 45 overflows even both cards (23.5 + 20 * 0.85 = 40.5); 4-bit fits GPU 1 alone.
    hardware["required"] = {False: 45.0, True: 12.0}
    response = _post(_request())

    assert response.verdict == "exceeds"
    assert response.reason == "insufficient_memory"
    assert response.suggestion is not None
    assert response.suggestion.kind == "qlora"
    assert response.suggestion.required_gb == pytest.approx(12.0)
    assert response.suggestion.verdict == "fits"
    # Priced as QLoRA, not merely guessed from the 16-bit figure.
    assert hardware["auto_calls"][-1]["load_in_4bit"] is True


def test_no_qlora_suggestion_when_the_run_cannot_load_4bit(hardware):
    hardware["required"] = {False: 45.0, True: 12.0}
    response = _post(_request(four_bit_available = False, batch_size = 1))
    assert response.verdict == "exceeds"
    assert response.suggestion is None


def test_exceeds_falls_back_to_a_smaller_batch_the_estimator_can_price(hardware, monkeypatch):
    def _estimate(model_name, **kwargs):
        required = 45.0 if kwargs["batch_size"] > 1 else 22.0
        return required, {"required_gb": required, "estimation_mode": "fallback"}

    def _auto(model_name, **kwargs):
        required, meta = _estimate(model_name, **kwargs)
        return ([1] if required <= 23.5 else [0, 1]), meta

    monkeypatch.setattr("utils.hardware.estimate_required_model_memory_gb", _estimate)
    monkeypatch.setattr("utils.hardware.auto_select_gpu_ids", _auto)
    # Already 4-bit, so QLoRA is no remedy; batch 1 is.
    response = _post(_request(load_in_4bit = True, batch_size = 8))

    assert response.verdict == "exceeds"
    # A fallback estimate has no breakdown to show, and the route does not invent one.
    assert response.estimation_mode == "fallback"
    assert response.breakdown is None
    assert response.suggestion is not None
    assert response.suggestion.kind == "batch_size"
    assert response.suggestion.batch_size == 1
    assert response.suggestion.verdict == "tight"


def test_unpriceable_model_is_unknown(hardware):
    hardware["required"] = {False: None, True: None}
    response = _post(_request())
    assert response.verdict == "unknown"
    assert response.reason == "estimate_unavailable"
    assert response.required_gb is None
    assert response.breakdown is None


def test_non_accelerator_host_is_unknown(hardware):
    hardware["device"] = DeviceType.MLX
    response = _post(_request())
    assert response.verdict == "unknown"
    assert response.reason == "unsupported_device"
    assert hardware["estimate_calls"] == []


def test_missing_telemetry_is_unknown_not_full(hardware):
    hardware["devices"] = [{"index": 1, "vram_total_gb": 24.0, "vram_used_gb": None}]
    response = _post(_request(gpu_ids = [1]))
    assert response.verdict == "unknown"
    assert response.reason == "no_gpu_telemetry"


def test_estimator_crash_is_unknown_not_a_500(hardware, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("config.json unreachable")

    monkeypatch.setattr("utils.hardware.auto_select_gpu_ids", _boom)
    response = _post(_request())
    assert response.verdict == "unknown"
    assert response.reason == "estimate_failed"


@pytest.mark.parametrize(
    ("required", "usable", "expected"),
    [
        (10.0, 20.0, ("fits", None)),
        (17.0, 20.0, ("fits", None)),  # exactly 0.85 is still clean
        (17.2, 20.0, ("tight", None)),
        (20.0, 20.0, ("tight", None)),
        (20.1, 20.0, ("exceeds", "insufficient_memory")),
        (5.0, 0.0, ("exceeds", "insufficient_memory")),
        (None, 20.0, ("unknown", "estimate_unavailable")),
        (float("nan"), 20.0, ("unknown", "estimate_unavailable")),
        (10.0, float("inf"), ("unknown", "estimate_unavailable")),
    ],
)
def test_verdict_thresholds(required, usable, expected):
    assert tv.classify_training_fit(required, usable) == expected


def test_per_gpu_floor_turns_an_aggregate_fit_into_exceeds():
    # 30 of 40 usable fits in aggregate, but activations need 12 on a card that has 10 free.
    assert tv.classify_training_fit(30.0, 40.0, min_per_gpu_gb = 12.0, min_free_gb = 10.0) == (
        "exceeds",
        "per_gpu_minimum",
    )
    assert tv.FIT_TIGHT_RATIO == 0.85


def test_estimate_route_requires_authentication(monkeypatch):
    route = next(r for r in tr.router.routes if getattr(r, "path", None) == "/estimate")
    assert "POST" in route.methods
    calls = [dep.call for dep in route.dependant.dependencies]
    assert tr.get_current_subject in calls

    # And over the wire: no credential, no keyless admission -> refused before the handler runs.
    monkeypatch.setattr(
        "utils.keyless_api_access.keyless_request_allowed", lambda *_args, **_kwargs: False
    )
    reached = []
    monkeypatch.setattr(
        tv, "estimate_training_fit", lambda **kwargs: reached.append(kwargs) or {}
    )
    app = FastAPI()
    app.include_router(tr.router, prefix = "/api/train")
    with TestClient(app) as client:
        response = client.post(
            "/api/train/estimate",
            json = {"model_name": "unsloth/tiny-model", "training_type": "LoRA/QLoRA"},
        )
    assert response.status_code in (401, 403)
    assert reached == []


def test_the_preview_prices_without_importing_unsloth(hardware, monkeypatch):
    """Importing unsloth in the server process pins a CUDA context on GPU 0 for the life of the server, so
    the planner, which re-prices on every edit, must price inside attention_preview_estimates."""
    import utils.hardware as hardware_pkg
    import utils.hardware.hardware as hw

    seen = []
    # The fixture's stubs, wrapped to record whether each call ran inside the preview.
    original_auto = hardware_pkg.auto_select_gpu_ids
    original_estimate = hardware_pkg.estimate_required_model_memory_gb

    def _auto(model_name, **kwargs):
        seen.append(("auto", hw._ATTENTION_PREVIEW.get()))
        return original_auto(model_name, **kwargs)

    def _estimate(model_name, **kwargs):
        seen.append(("explicit", hw._ATTENTION_PREVIEW.get()))
        return original_estimate(model_name, **kwargs)

    monkeypatch.setattr("utils.hardware.auto_select_gpu_ids", _auto)
    monkeypatch.setattr("utils.hardware.estimate_required_model_memory_gb", _estimate)

    _post(_request())
    _post(_request(gpu_ids = [0]))
    assert seen == [("auto", True), ("explicit", True)]
    # Scoped to the preview: nothing after it inherits the shortcut.
    assert hw._ATTENTION_PREVIEW.get() is False


class _SdpaModel:
    _supports_sdpa = True


class _FlashOnlyModel:
    _supports_sdpa = False
    _supports_flash_attn = True


class _PlainModel:
    _supports_sdpa = False


@pytest.mark.parametrize(
    "model_class, expected",
    [(_SdpaModel, "sdpa"), (_FlashOnlyModel, "flash_attention_2"), (_PlainModel, "eager"), (None, "eager")],
)
def test_preview_attention_reads_the_class_flags_without_unsloth(monkeypatch, model_class, expected):
    import sys

    import utils.hardware.hardware as hw

    monkeypatch.delitem(sys.modules, "unsloth", raising = False)
    # A None entry makes any import of it raise, so reaching the exact resolver would fail the test.
    monkeypatch.setitem(sys.modules, "unsloth.models._utils", None)
    monkeypatch.setattr(hw, "_model_class_for_gpu_estimate", lambda _config: model_class)
    with hw.attention_preview_estimates():
        assert hw._determine_attention_impl_for_gpu_estimate(object()) == expected
    assert "unsloth" not in sys.modules


def test_preview_keeps_the_exact_resolver_once_unsloth_is_loaded(monkeypatch):
    """No context to save once unsloth is imported (a training run already did it), so stay exact."""
    import sys
    import types

    import utils.hardware.hardware as hw

    monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
    monkeypatch.setattr(
        hw, "_approximate_attention_impl_for_gpu_estimate", lambda _config: pytest.fail("approximated")
    )
    utils_module = types.ModuleType("unsloth.models._utils")
    utils_module.resolve_attention_implementation = lambda _cls, _config: "flex_attention"
    monkeypatch.setitem(sys.modules, "unsloth.models", types.ModuleType("unsloth.models"))
    monkeypatch.setitem(sys.modules, "unsloth.models._utils", utils_module)

    class _Config:
        model_type = "not-a-registered-type"

    with hw.attention_preview_estimates():
        assert hw._determine_attention_impl_for_gpu_estimate(_Config()) == "flex_attention"
