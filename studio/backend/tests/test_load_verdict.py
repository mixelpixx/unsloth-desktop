# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The load guardrail's verdict: one plain answer per GGUF load, against memory free now.

The bar is asymmetric. A verdict that says "fits" and is wrong costs a crash the user
would have had anyway; one that says ``likely_too_large`` and is wrong blocks a load that
works. So most of these cases are about what must NOT be called too large: a miss inside
the estimation band, a miss that hinges on memory a resident model may hand back, an Auto
load with no context named, missing inputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import load_guardrail as lg
from core.inference.load_verdict import (
    CPU,
    DISK_STREAMING,
    FITS_BARELY,
    FULL_GPU,
    LIKELY_TOO_LARGE,
    PARTIAL_GPU,
    PLACEMENT_AUTO,
    PLACEMENT_CPU,
    PLACEMENT_FIXED,
    UNKNOWN,
    GpuMemory,
    VerdictInputs,
    compute_load_verdict,
    estimation_band_bytes,
)
from utils.load_guardrail_settings import verdict_needs_confirmation

MIB = 1024**2
GIB = 1024**3


def _gb(value: float) -> int:
    return int(value * GIB)


def _inputs(**overrides) -> VerdictInputs:
    """A 19.6 GB load at a pinned 8k context on one 24 GB card with 6.1 GB free: the
    field case, another LLM app holding the rest."""
    base = dict(
        weights_bytes = _gb(18.0),
        kv_bytes = _gb(1.0),
        compute_bytes = _gb(0.6),
        total_bytes = _gb(19.6),
        gpu_bytes = _gb(19.6),
        placement = PLACEMENT_FIXED,
        context_pinned = True,
        n_ctx = 8192,
        gpus = (GpuMemory(0, _gb(6.1), _gb(24), 0, _gb(17)),),
        ram_available_bytes = _gb(64),
    )
    base.update(overrides)
    return VerdictInputs(**base)


def _card(free: float, total: float = 24, uncertain: float = 0, other = None, index: int = 0):
    return GpuMemory(
        index,
        _gb(free),
        _gb(total),
        _gb(uncertain),
        None if other is None else _gb(other),
    )


# (id, overrides, expected level, expected reason)
CASES = [
    (
        "fits-comfortably",
        dict(placement = PLACEMENT_AUTO, gpus = (_card(23.0),)),
        FULL_GPU,
        "fits",
    ),
    (
        "fits-but-inside-the-reserve-is-tight",
        dict(placement = PLACEMENT_AUTO, gpus = (_card(20.2),)),
        FITS_BARELY,
        "tight",
    ),
    (
        "forced-miss-inside-the-band-is-tight-not-refused",
        # 0.3 GB short of free, the band is ~1 GB: within estimation error.
        dict(gpus = (_card(19.3),)),
        FITS_BARELY,
        "tight",
    ),
    (
        "forced-full-offload-that-cannot-fit",
        dict(),
        LIKELY_TOO_LARGE,
        "forced_gpu_overflow",
    ),
    (
        "forced-miss-the-outgoing-model-might-cover-is-unknown",
        dict(gpus = (_card(6.1, uncertain = 17.0),)),
        UNKNOWN,
        "resident_unattributed",
    ),
    (
        "auto-spills-into-ram",
        dict(placement = PLACEMENT_AUTO),
        PARTIAL_GPU,
        "spills_to_ram",
    ),
    (
        "auto-larger-than-gpu-and-ram-streams-from-disk",
        dict(placement = PLACEMENT_AUTO, ram_available_bytes = _gb(8)),
        DISK_STREAMING,
        "streams_from_disk",
    ),
    (
        "auto-whose-kv-and-buffers-fit-nowhere",
        dict(
            placement = PLACEMENT_AUTO,
            kv_bytes = _gb(40),
            total_bytes = _gb(58.6),
            gpu_bytes = _gb(58.6),
            ram_available_bytes = _gb(16),
        ),
        LIKELY_TOO_LARGE,
        "runtime_overflow",
    ),
    (
        "cpu-placement-that-fits-ram",
        dict(placement = PLACEMENT_CPU, gpu_bytes = 0),
        CPU,
        "cpu_only",
    ),
    (
        "cpu-placement-larger-than-ram-streams",
        dict(placement = PLACEMENT_CPU, gpu_bytes = 0, ram_available_bytes = _gb(12)),
        DISK_STREAMING,
        "streams_from_disk",
    ),
    (
        "cpu-placement-whose-cache-exceeds-ram",
        dict(
            placement = PLACEMENT_CPU,
            gpu_bytes = 0,
            kv_bytes = _gb(30),
            total_bytes = _gb(48.6),
            ram_available_bytes = _gb(16),
        ),
        LIKELY_TOO_LARGE,
        "runtime_overflow",
    ),
    (
        "unsizable-cache-is-unknown",
        dict(kv_estimable = False),
        UNKNOWN,
        "kv_unsized",
    ),
    (
        "no-estimate-is-unknown",
        dict(total_bytes = 0, gpu_bytes = None),
        UNKNOWN,
        "no_estimate",
    ),
    (
        "no-gpu-reading-is-unknown",
        dict(gpus = ()),
        UNKNOWN,
        "no_gpu_reading",
    ),
    (
        "a-fit-with-unsized-parts-is-only-tight",
        dict(placement = PLACEMENT_AUTO, gpus = (_card(23.5),), sizes_are_floor = True),
        FITS_BARELY,
        "tight",
    ),
    (
        "auto-with-no-context-named-fits-at-a-shorter-one",
        # Native 128k cache would not fit; Auto picks a window that does.
        dict(
            placement = PLACEMENT_AUTO,
            context_pinned = False,
            n_ctx = 131072,
            kv_bytes = _gb(16),
            total_bytes = _gb(34.6),
            gpu_bytes = _gb(34.6),
            gpus = (_card(23.0),),
        ),
        FULL_GPU,
        "context_fitted",
    ),
    (
        "two-cards-pooled",
        dict(
            placement = PLACEMENT_AUTO,
            gpus = (_card(12.0, index = 0), _card(12.0, index = 1)),
        ),
        FULL_GPU,
        "fits",
    ),
    (
        "a-card-with-no-total-never-carries-a-refusal",
        # An iGPU's "VRAM" is host RAM and a MIG slice reports none: nothing bounds the free figure.
        dict(gpus = (GpuMemory(0, _gb(6.1), 0),)),
        UNKNOWN,
        "no_gpu_reading",
    ),
    (
        "an-outgoing-model-that-may-cover-the-miss-is-tight-under-auto",
        dict(placement = PLACEMENT_AUTO, gpus = (_card(6.1, uncertain = 17.0),)),
        FITS_BARELY,
        "resident_unattributed",
    ),
]


@pytest.mark.parametrize(
    "overrides,level,reason", [c[1:] for c in CASES], ids = [c[0] for c in CASES]
)
def test_verdict_table(overrides, level, reason):
    verdict = compute_load_verdict(_inputs(**overrides))
    assert (verdict.level, verdict.reason) == (level, reason), verdict.message


def test_the_refusal_carries_the_numbers_the_dialog_prints():
    verdict = compute_load_verdict(_inputs())
    assert verdict.level == LIKELY_TOO_LARGE
    assert verdict.gpu_need_bytes == _gb(19.6)
    assert verdict.gpu_free_bytes == _gb(6.1)
    assert verdict.gpu_indices == (0,)
    assert verdict.other_apps_bytes == _gb(17)
    assert verdict.message == (
        "Needs ~19.6 GiB on GPU 0; 6.1 GiB is free. Other programs are using 17.0 GiB."
    )
    wire = verdict.as_dict()
    assert wire["gpu_indices"] == [0] and wire["level"] == LIKELY_TOO_LARGE


def test_a_fixed_split_overflowing_one_card_names_that_card():
    # 3:1 puts ~14.7 GB on GPU 0, which has 8 GB; the pool (8 + 20) would have room.
    verdict = compute_load_verdict(
        _inputs(
            gpus = (_card(8.0, index = 0), _card(20.0, index = 1)),
            tensor_split = (3.0, 1.0),
        )
    )
    assert verdict.level == LIKELY_TOO_LARGE
    assert verdict.gpu_indices == (0,)
    assert verdict.gpu_need_bytes == int(_gb(19.6) * 0.75)


def test_a_split_that_does_not_line_up_falls_back_to_the_pool():
    verdict = compute_load_verdict(
        _inputs(gpus = (_card(12.0, index = 0), _card(12.0, index = 1)), tensor_split = (1.0,))
    )
    assert verdict.gpu_indices == (0, 1)


def test_the_band_scales_with_the_footprint():
    assert estimation_band_bytes(0) == 512 * MIB
    assert estimation_band_bytes(_gb(40)) == int(0.05 * _gb(40))


@pytest.mark.parametrize("free", [0.5, 2.0, 6.1, 12.0, 19.0, 22.0])
@pytest.mark.parametrize("ram", [None, 2.0, 16.0, 64.0])
def test_auto_with_no_context_named_is_never_called_too_large(free, ram):
    """The loader shrinks the window, so a refusal would rest on a guess at its size."""
    verdict = compute_load_verdict(
        _inputs(
            placement = PLACEMENT_AUTO,
            context_pinned = False,
            n_ctx = 262144,
            kv_bytes = _gb(60),
            total_bytes = _gb(78.6),
            gpu_bytes = _gb(78.6),
            gpus = (_card(free),),
            ram_available_bytes = None if ram is None else _gb(ram),
        )
    )
    assert verdict.level != LIKELY_TOO_LARGE


@pytest.mark.parametrize("shortfall_gb", [0.0, 0.2, 0.5, 0.9])
def test_a_forced_miss_inside_the_band_is_never_refused(shortfall_gb):
    need = 19.6
    verdict = compute_load_verdict(_inputs(gpus = (_card(need - shortfall_gb),)))
    assert verdict.level == FITS_BARELY


# (mode, level, reason, confirms)
CONFIRMATION = [
    ("balanced", LIKELY_TOO_LARGE, "forced_gpu_overflow", True),
    ("balanced", LIKELY_TOO_LARGE, "runtime_overflow", True),
    ("balanced", DISK_STREAMING, "streams_from_disk", False),
    ("balanced", PARTIAL_GPU, "spills_to_ram", False),
    ("balanced", UNKNOWN, "resident_unattributed", False),
    ("strict", LIKELY_TOO_LARGE, "forced_gpu_overflow", True),
    ("strict", DISK_STREAMING, "streams_from_disk", True),
    ("strict", PARTIAL_GPU, "spills_to_ram", False),
    ("strict", FITS_BARELY, "tight", False),
    ("relaxed", LIKELY_TOO_LARGE, "forced_gpu_overflow", False),
    ("relaxed", LIKELY_TOO_LARGE, "runtime_overflow", True),
    ("relaxed", DISK_STREAMING, "streams_from_disk", False),
    ("off", LIKELY_TOO_LARGE, "runtime_overflow", False),
    ("off", LIKELY_TOO_LARGE, "forced_gpu_overflow", False),
    # An unreadable mode falls back to the default, not to "never ask".
    ("bogus", LIKELY_TOO_LARGE, "forced_gpu_overflow", True),
]


@pytest.mark.parametrize("mode,level,reason,confirms", CONFIRMATION)
def test_which_modes_ask(mode, level, reason, confirms):
    assert verdict_needs_confirmation(mode, level, reason) is confirms


# (gpu_memory_mode, gpu_layers, extras, offload fraction, expected)
PLACEMENTS = [
    ("auto", -1, None, 1.0, PLACEMENT_AUTO),
    ("manual", -1, None, 1.0, PLACEMENT_AUTO),  # Manual at Auto layers hands it to --fit
    ("manual", 20, None, 0.5, PLACEMENT_FIXED),
    ("manual", 99, None, 1.0, PLACEMENT_FIXED),
    ("auto", -1, ["-ngl", "99"], 1.0, PLACEMENT_FIXED),  # honoured even under --fit on
    ("manual", 0, None, 0.0, PLACEMENT_CPU),
    ("auto", -1, ["--device", "none"], 0.0, PLACEMENT_CPU),
]


@pytest.mark.parametrize("mode,layers,extras,fraction,expected", PLACEMENTS)
def test_placement_classification(mode, layers, extras, fraction, expected):
    assert lg.classify_placement(mode, layers, extras, fraction) == expected


class _Live:
    def poll(self):
        return None


class _Dead:
    def poll(self):
        return 0


def _llama(lines, physical_ids = (0,), process = None):
    return SimpleNamespace(
        _process = process or _Live(),
        _child_gpu_physical_ids = tuple(physical_ids),
        _stdout_lines = list(lines),
    )


class TestResidentLlamaBytes:
    LINES = [
        "load_tensors:        CUDA0 model buffer size =  1024.00 MiB",
        "llama_kv_cache:      CUDA1 KV buffer size =   512.00 MiB",
        "llama_context:  CUDA_Host  output buffer size =     0.58 MiB",
        "sched_reserve:       CUDA0 compute buffer size =   256.00 MiB",
    ]

    def test_every_device_buffer_is_counted_per_physical_card(self):
        # Ordinal 0 is physical card 1: the child's mask reorders them.
        out = lg.resident_llama_gpu_bytes(_llama(self.LINES, physical_ids = (1, 0)))
        assert out == {1: 1280 * MIB, 0: 512 * MIB}

    def test_an_ordinal_past_the_mask_cannot_be_mapped(self):
        assert lg.resident_llama_gpu_bytes(_llama(self.LINES, physical_ids = (0,))) is None

    def test_a_dead_child_holds_nothing_to_credit(self):
        assert lg.resident_llama_gpu_bytes(_llama(self.LINES, process = _Dead())) is None

    def test_no_child(self):
        assert lg.resident_llama_gpu_bytes(SimpleNamespace(_process = None)) is None


class TestMeasureGpuMemory:
    @pytest.fixture(autouse = True)
    def _rows(self, monkeypatch):
        # 24 GiB card, 6 GiB free.
        self.rows = [(0, 6 * 1024, 24 * 1024)]
        monkeypatch.setattr(lg, "_probe_rows", lambda *_a: list(self.rows))

    def test_nothing_resident_reports_other_programs(self):
        (gpu,) = lg.measure_gpu_memory(llama_backend = SimpleNamespace(_process = None))
        assert gpu.free_bytes == 6 * GIB
        assert gpu.uncertain_bytes == 0
        # In use minus the idle reservation an empty card shows (5%, capped at 1 GiB).
        assert gpu.other_bytes == 18 * GIB - 1 * GIB

    def test_a_resident_llama_server_is_credited_back(self):
        llama = _llama(["CUDA0 model buffer size = 4096.00 MiB"])
        (gpu,) = lg.measure_gpu_memory(llama_backend = llama)
        assert gpu.free_bytes == 10 * GIB
        # The CUDA context is not logged: it may come back, it is not counted as freed.
        assert gpu.uncertain_bytes == 512 * MIB
        assert gpu.other_bytes == 18 * GIB - 4 * GIB - 1 * GIB

    def test_an_unreadable_resident_child_makes_the_card_uncertain(self):
        llama = _llama(["nothing useful"])
        (gpu,) = lg.measure_gpu_memory(llama_backend = llama)
        assert gpu.free_bytes == 6 * GIB
        assert gpu.uncertain_bytes == 18 * GIB
        assert gpu.other_bytes is None

    def test_another_studio_model_makes_the_card_uncertain(self):
        (gpu,) = lg.measure_gpu_memory(
            llama_backend = SimpleNamespace(_process = None),
            other_studio_model_resident = True,
        )
        assert gpu.uncertain_bytes == 18 * GIB
        assert gpu.other_bytes is None

    def test_a_failed_probe_is_none_and_an_empty_one_is_no_cards(self, monkeypatch):
        def _boom(*_a):
            raise RuntimeError("nvidia-smi hung")

        monkeypatch.setattr(lg, "_probe_rows", _boom)
        assert lg.measure_gpu_memory(llama_backend = None) is None
        monkeypatch.setattr(lg, "_probe_rows", lambda *_a: [])
        assert lg.measure_gpu_memory(llama_backend = None) == []

    def test_the_estimate_path_reuses_a_recent_probe(self, monkeypatch):
        calls = []

        def _count(*_a):
            calls.append(1)
            return list(self.rows)

        monkeypatch.setattr(lg, "_probe_cache", None)
        monkeypatch.setattr(lg, "_probe_rows", _count)
        lg.measure_gpu_memory(llama_backend = None, fresh = False)
        lg.measure_gpu_memory(llama_backend = None, fresh = False)
        assert len(calls) == 1
        # A /load always reads fresh.
        lg.measure_gpu_memory(llama_backend = None, fresh = True)
        assert len(calls) == 2


def test_the_measured_card_feeds_the_verdict_end_to_end(monkeypatch):
    """The LM Studio case: 18 GiB held elsewhere, a forced full offload of 19.6 GB."""
    monkeypatch.setattr(lg, "_probe_rows", lambda *_a: [(0, 6 * 1024, 24 * 1024)])
    gpus = lg.measure_gpu_memory(llama_backend = SimpleNamespace(_process = None))
    breakdown = SimpleNamespace(
        weights_bytes = _gb(18),
        kv_bytes = _gb(1),
        compute_bytes = _gb(0.6),
        total_bytes = _gb(19.6),
        gpu_bytes = _gb(19.6),
        kv_on_gpu = True,
        kv_estimable = True,
        n_ctx = 8192,
    )
    verdict = lg.verdict_from_breakdown(
        breakdown,
        gpus = gpus,
        placement = PLACEMENT_FIXED,
        context_pinned = True,
        vram_fraction = 0.97,
        ram_available_bytes = _gb(64),
    )
    assert verdict.level == LIKELY_TOO_LARGE
    assert verdict.other_apps_bytes == 17 * GIB


def test_a_comfortable_auto_fit_quotes_the_window_auto_will_load():
    """Two 24 GB cards and a 262k model whose whole cache fits: Auto loads the native window, so the
    headline must quote that (~38 GB), not the 4k floor the verdict charges to decide (~20 GB)."""
    verdict = compute_load_verdict(
        _inputs(
            placement = PLACEMENT_AUTO,
            context_pinned = False,
            n_ctx = 262144,
            weights_bytes = _gb(16.0),
            kv_bytes = _gb(18.0),
            compute_bytes = _gb(4.0),
            total_bytes = _gb(38.0),
            gpu_bytes = _gb(38.0),
            gpus = (_card(23.5, index = 0), _card(23.5, index = 1)),
        )
    )
    assert (verdict.level, verdict.reason) == (FULL_GPU, "fits")
    assert verdict.gpu_need_bytes == _gb(38.0)
    assert "needs ~38.0 GiB" in verdict.message
