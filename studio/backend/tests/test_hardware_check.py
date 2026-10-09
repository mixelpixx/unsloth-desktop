# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Resources > Hardware check: the measurements, the findings, and the options.

What has to hold: the findings say what was measured (two RTX 3090s, one on a chipset x1 slot,
Windows without peer access) and say "all good" when nothing is off; the options change
placement only when switched on AND the stored result describes the GPUs installed now; an
explicit GPU pick is never touched; the guardrail judges the card the loader then uses; a run
never competes with training or a load; and a child that hangs or dies costs only its result.
The conftest isolates UNSLOTH_STUDIO_HOME per test, so the stored result never touches a real
install.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.hardware_check as hardware_check_route
import utils.hardware_check_settings as hcs
from utils.account_context import AccountContext, bind_account
from utils.hardware import hardware_check as hc
from utils.hardware import link_preference as lp

GIB = 1024**3
MIB = 1024**2


def _gpu(index, *, h2d, d2h = None, width = 16, width_max = 16, gen = 4, status = "measured", free = 24 * GIB, reason = None):
    return {
        "index": index,
        "uuid": f"GPU-{index:04d}",
        "pci_bus_id": f"00000000:{index + 1:02X}:00.0",
        "name": "NVIDIA GeForce RTX 3090",
        "memory_total_bytes": 24 * GIB,
        "memory_free_bytes": free,
        "link_idle": {"gen_current": 1, "gen_max": gen, "width_current": width, "width_max": width_max},
        "status": status,
        "reason": reason,
        "link": {"gen_current": gen, "gen_max": gen, "width_current": width, "width_max": width_max}
        if status == "measured"
        else None,
        "h2d_gibs": h2d if status == "measured" else None,
        "d2h_gibs": (d2h if d2h is not None else h2d) if status == "measured" else None,
    }


def this_machine(**overrides):
    """Measured on the reference box: GPU 1 in a chipset slot wired x1, no P2P on Windows,
    the Hugging Face cache on a SATA SSD with an NVMe drive present."""
    result = {
        "schema": hc.SCHEMA_VERSION,
        "trigger": "manual",
        "platform": "Windows",
        "fingerprint": "fp-two-3090s",
        "finished_at": "2026-10-09T12:00:00Z",
        "duration_ms": 6200,
        "gpus": [_gpu(0, h2d = 24.9, d2h = 24.4), _gpu(1, h2d = 1.5, d2h = 1.53, width = 1)],
        "pairs": [
            {"a": 0, "b": 1, "peer_ab": False, "peer_ba": False, "copy_ab_gibs": 1.47, "copy_ba_gibs": 1.52}
        ],
        "storage": [
            {"key": "studio_home", "path": "D:\\apps\\unsloth-studio", "drive": "D:", "bus_type": "SATA", "media_type": "SSD", "model": "M500DC"},
            {"key": "hf_cache", "path": "D:\\hf\\hub", "drive": "D:", "bus_type": "SATA", "media_type": "SSD", "model": "M500DC"},
            {"key": "temp", "path": "C:\\Temp", "drive": "C:", "bus_type": "SATA", "media_type": "SSD", "model": "MTFDDAK"},
        ],
        "nvme_present": True,
        "gpu_error": None,
        "storage_error": None,
    }
    result.update(overrides)
    return result


def all_good_machine():
    return this_machine(
        platform = "Linux",
        gpus = [_gpu(0, h2d = 24.9), _gpu(1, h2d = 24.6)],
        pairs = [{"a": 0, "b": 1, "peer_ab": True, "peer_ba": True, "copy_ab_gibs": 22.0, "copy_ba_gibs": 22.1}],
        storage = [{"key": "hf_cache", "path": "/nvme/hf", "drive": None, "bus_type": "NVMe", "media_type": "SSD", "model": None}],
    )


@pytest.fixture(autouse = True)
def _fresh_module_state(monkeypatch):
    """Module-level memos and the run state belong to one test."""
    monkeypatch.setattr(hc, "_preference_cache", None)
    monkeypatch.setattr(hc, "_result_cache", None)
    monkeypatch.setattr(hc, "_state", dict(hc._state, running = False, last_skip = None, last_error = None, auto = None))
    monkeypatch.setattr(hcs, "_cache", {})
    yield


@pytest.fixture
def settings(monkeypatch):
    """An in-memory settings row behind the module's own read and write."""
    rows: dict[str, object] = {}
    monkeypatch.setattr(hcs, "_cached_setting", lambda key: rows.get(key))
    import storage.studio_db as studio_db

    monkeypatch.setattr(studio_db, "get_app_setting", lambda key, fallback = None: rows.get(key, fallback))
    monkeypatch.setattr(studio_db, "upsert_app_settings", lambda updates, **_kw: rows.update(updates))

    def set_options(**values):
        rows[hcs.HARDWARE_CHECK_SETTING_KEY] = {**hcs.DEFAULTS, **values}
        hc._invalidate_preference()

    return set_options


@pytest.fixture
def stored(monkeypatch):
    """Store a result and pin what the GPUs installed now fingerprint as."""
    current = {"fingerprint": "fp-two-3090s"}
    monkeypatch.setattr(hc, "current_fingerprint", lambda: current["fingerprint"])

    def store(result):
        hc.save_result(result)
        return current

    return store


# ── Findings ────────────────────────────────────────────────────────────────────────────────────


def test_this_machine_the_x1_card_no_p2p_and_the_sata_cache_are_found():
    summary = hc.analyze(this_machine())
    ids = [f["id"] for f in summary["findings"]]
    assert ids == ["slow_link", "no_peer_access", "storage_slow"]
    slow = summary["findings"][0]
    assert slow["severity"] == "warning"
    assert slow["values"] == {
        "gpu": 1,
        "width": 1,
        "width_max": 16,
        "gen": 4,
        "h2d_gibs": 1.5,
        "best_gpu": 0,
        "best_h2d_gibs": 24.9,
        "ratio": 17,
    }
    assert "GPU 1 runs at PCIe x1 (of x16): 1.5 GiB/s" in slow["text"]
    assert "24.9 GiB/s on GPU 0" in slow["text"] and "~17x slower" in slow["text"]
    peer = summary["findings"][1]
    assert peer["values"] == {"a": 0, "b": 1, "copy_gibs": 1.5, "windows": True}
    assert "(Windows)" in peer["text"] and "through system memory" in peer["text"]
    storage = summary["findings"][2]
    assert storage["severity"] == "info"
    assert storage["values"]["location"] == "hf_cache" and storage["values"]["kind"] == "sata"
    assert summary["slow_gpus"] == [1]
    assert summary["recommended"] == {
        "prefer_fast_link": True,
        "avoid_tensor_split": True,
        "warn_training_slow_link": True,
    }


def test_an_all_good_machine_says_so_and_recommends_nothing():
    summary = hc.analyze(all_good_machine())
    assert [f["id"] for f in summary["findings"]] == ["all_good"]
    assert summary["findings"][0]["severity"] == "ok"
    assert not any(summary["recommended"].values())
    assert summary["slow_gpus"] == []


def test_a_sata_cache_is_only_flagged_when_an_nvme_drive_exists():
    summary = hc.analyze(this_machine(nvme_present = False))
    assert "storage_slow" not in [f["id"] for f in summary["findings"]]


@pytest.mark.parametrize(
    ("bus", "media", "kind", "severity"),
    [("USB", "SSD", "usb", "warning"), ("SATA", "HDD", "hdd", "warning")],
)
def test_a_cache_on_usb_or_a_hard_disk_is_a_warning(bus, media, kind, severity):
    result = this_machine(
        storage = [{"key": "hf_cache", "path": "E:\\hf", "drive": "E:", "bus_type": bus, "media_type": media, "model": None}],
        nvme_present = False,
    )
    finding = next(f for f in hc.analyze(result)["findings"] if f["id"] == "storage_slow")
    assert finding["values"]["kind"] == kind and finding["severity"] == severity


def test_slow_means_under_half_the_width_or_a_quarter_of_the_best_bandwidth():
    narrow = this_machine(gpus = [_gpu(0, h2d = 24.9), _gpu(1, h2d = 20.0, width = 4)])
    assert hc.slow_link_gpus(narrow) == {1: {"narrow": True, "starved": False}}
    starved = this_machine(gpus = [_gpu(0, h2d = 24.9), _gpu(1, h2d = 5.0, width = 16)])
    assert hc.slow_link_gpus(starved) == {1: {"narrow": False, "starved": True}}
    x8 = this_machine(gpus = [_gpu(0, h2d = 24.9), _gpu(1, h2d = 12.4, width = 8)])
    assert hc.slow_link_gpus(x8) == {}, "x8 of x16 is half, not under half; 12.4 is over a quarter"


def test_one_gpu_on_a_narrow_link_is_slow_but_placement_has_nothing_to_prefer():
    result = this_machine(gpus = [_gpu(0, h2d = 3.1, width = 4)], pairs = [])
    summary = hc.analyze(result)
    finding = next(f for f in summary["findings"] if f["id"] == "slow_link")
    assert finding["values"]["best_gpu"] is None and "is slow" in finding["text"]
    assert summary["recommended"] == {
        "prefer_fast_link": False,
        "avoid_tensor_split": False,
        "warn_training_slow_link": True,
    }


def test_a_card_skipped_for_memory_is_named_and_a_failed_probe_is_a_warning():
    skipped = this_machine(gpus = [_gpu(0, h2d = 24.9), _gpu(1, h2d = None, status = "skipped", reason = "low_free_memory", free = int(0.3 * GIB))], pairs = [])
    finding = next(f for f in hc.analyze(skipped)["findings"] if f["id"] == "gpu_skipped")
    assert finding["values"] == {"gpu": 1, "reason": "low_free_memory", "free_gib": 0.3}
    failed = this_machine(gpus = [_gpu(0, h2d = None, status = "failed", reason = "probe_failed")], pairs = [], gpu_error = "the measurement timed out")
    ids = [f["id"] for f in hc.analyze(failed)["findings"]]
    assert "gpu_probe_failed" in ids


# ── Persistence and the fingerprint ─────────────────────────────────────────────────────────────


def test_the_result_round_trips_in_the_studio_home(stored):
    stored(this_machine())
    assert hc.result_path().parent == Path(hc.result_path()).parent
    assert hc.result_path().is_file()
    loaded = hc.load_result()
    assert loaded["gpus"][1]["h2d_gibs"] == 1.5
    assert hc.result_is_current() is True


def test_a_changed_fingerprint_makes_the_result_out_of_date(stored):
    current = stored(this_machine())
    current["fingerprint"] = "fp-card-moved"
    assert hc.result_is_current() is False
    current["fingerprint"] = None
    assert hc.result_is_current() is None, "unreadable hardware is unknown, not changed"


def test_another_schema_or_a_corrupt_file_reads_as_no_result(stored):
    stored(this_machine(schema = 999))
    assert hc.load_result() is None
    hc.result_path().write_text("{not json", encoding = "utf-8")
    assert hc.load_result() is None


def test_the_fingerprint_covers_uuids_and_bus_ids_in_any_order():
    a = [{"uuid": "GPU-a", "pci_bus_id": "00000000:01:00.0"}, {"uuid": "GPU-b", "pci_bus_id": "00000000:08:00.0"}]
    assert hc.fingerprint_of(a) == hc.fingerprint_of(list(reversed(a)))
    moved = [a[0], {"uuid": "GPU-b", "pci_bus_id": "00000000:09:00.0"}]
    assert hc.fingerprint_of(moved) != hc.fingerprint_of(a)
    assert hc.fingerprint_of([]) == "no-nvidia-gpu"


# ── The options read nothing unless on and current ──────────────────────────────────────────────


def test_the_link_preference_is_none_while_the_option_is_off(settings, stored):
    stored(this_machine())
    settings(prefer_fast_link = False)
    assert hc.active_link_preference() is None


def test_the_link_preference_is_none_without_a_result(settings, stored):
    settings(prefer_fast_link = True)
    assert hc.active_link_preference() is None


def test_the_link_preference_is_none_once_the_hardware_changed(settings, stored):
    current = stored(this_machine())
    settings(prefer_fast_link = True)
    current["fingerprint"] = "fp-other"
    assert hc.active_link_preference() is None


def test_the_link_preference_maps_each_measured_card_when_on_and_current(settings, stored):
    stored(this_machine())
    settings(prefer_fast_link = True)
    assert hc.active_link_preference() == {0: 24.9, 1: 1.5}


# ── The placement rule ──────────────────────────────────────────────────────────────────────────

FAST_SLOW = {0: 24.9, 1: 1.5}


def test_a_model_that_fits_one_card_moves_to_the_fast_link():
    # GPU 1 has more room (the display sits on GPU 0), so it is the default pick.
    pick = lp.faster_card(1, [(1, 23_000.0), (0, 22_400.0)], FAST_SLOW, need_mib = 17_000.0)
    assert pick == 0


def test_a_fast_card_that_only_fits_tightly_is_not_preferred():
    # 17 GiB needed, 17.3 GiB usable on GPU 0: under the 5% / 512 MiB comfort margin.
    pick = lp.faster_card(1, [(1, 23_000.0), (0, 17_700.0)], FAST_SLOW, need_mib = 17_408.0)
    assert pick is None


def test_two_fast_cards_are_a_tie_and_keep_the_existing_order():
    assert lp.faster_card(1, [(1, 23_000.0), (0, 22_000.0)], {0: 24.9, 1: 24.1}, need_mib = 10_000.0) is None


def test_an_unmeasured_default_is_never_moved():
    assert lp.faster_card(1, [(1, 23_000.0), (0, 22_000.0)], {0: 24.9}, need_mib = 10_000.0) is None


def test_an_unshared_card_is_never_traded_for_one_another_model_runs_on():
    assert lp.faster_card(1, [(1, 23_000.0), (0, 22_000.0)], FAST_SLOW, need_mib = 10_000.0, shared = {0}) is None


def test_auto_context_takes_the_fast_card_within_the_room_tolerance():
    # Neither holds the native window; Auto sizes it. GPU 0 has 600 MiB less room: within 1 GiB.
    kwargs = dict(need_mib = 60_000.0, floor_need_mib = 18_000.0, room_tolerance_mib = lp.AUTO_CONTEXT_ROOM_TOLERANCE_MIB)
    assert lp.faster_card(1, [(1, 23_000.0), (0, 22_400.0)], FAST_SLOW, **kwargs) == 0
    # 2 GiB less room is more context than the option trades away.
    assert lp.faster_card(1, [(1, 23_000.0), (0, 21_000.0)], FAST_SLOW, **kwargs) is None


def test_a_split_puts_the_fast_card_first_and_otherwise_keeps_the_order():
    assert lp.fast_first_order([0, 1], {0: 1.5, 1: 24.9}) == [1, 0]
    assert lp.fast_first_order([0, 1], FAST_SLOW) == [0, 1]
    assert lp.fast_first_order([0, 1], None) == [0, 1]
    assert lp.fast_first_order([0, 1, 2], {0: 24.0, 1: 24.9, 2: 1.5}) == [0, 1, 2], "a tie keeps device 0"


def test_select_gpus_places_a_one_card_model_on_the_fast_link():
    from core.inference.llama_cpp import LlamaCppBackend

    gpus = [(0, 23_000), (1, 24_000)]  # MiB free: GPU 1 roomier
    totals = {0: 24_576, 1: 24_576}
    size = 17 * GIB
    assert LlamaCppBackend._select_gpus(size, gpus, usable_fraction = 0.97, total_by_idx = totals) == ([1], False)
    assert LlamaCppBackend._select_gpus(
        size, gpus, usable_fraction = 0.97, total_by_idx = totals, prefer_bandwidth = FAST_SLOW
    ) == ([0], False)


def test_select_gpus_leaves_a_split_and_an_absent_preference_alone():
    from core.inference.llama_cpp import LlamaCppBackend

    gpus = [(0, 23_000), (1, 24_000)]
    totals = {0: 24_576, 1: 24_576}
    split = LlamaCppBackend._select_gpus(30 * GIB, gpus, usable_fraction = 0.97, total_by_idx = totals, prefer_bandwidth = FAST_SLOW)
    assert split == ([0, 1], False)
    for preference in (None, {}):
        assert LlamaCppBackend._select_gpus(
            17 * GIB, gpus, usable_fraction = 0.97, total_by_idx = totals, prefer_bandwidth = preference
        ) == ([1], False)


def test_select_gpus_split_aware_passes_the_preference_through():
    from core.inference.llama_cpp import LlamaCppBackend

    gpus = [(0, 23_000), (1, 24_000)]
    totals = {0: 24_576, 1: 24_576}
    assert LlamaCppBackend._select_gpus_split_aware(
        17 * GIB, gpus, usable_fraction = 0.97, total_by_idx = totals, split_extra_bytes = 256 * MIB, prefer_bandwidth = FAST_SLOW
    ) == ([0], False)


def test_the_loader_never_applies_the_preference_to_an_explicit_pick_or_vulkan():
    """load_model is 10k lines of launch logic; the gate itself is one expression, read here."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    source = inspect.getsource(LlamaCppBackend.load_model)
    assert "if (gpu_ids or is_vulkan_backend)\n" in source
    assert "else _hardware_check_link_preference()" in source
    # Device order: automatic placement only, nothing inherited, no positional user ratio.
    assert source.count("and not gpu_ids\n                        and _inherited_order is None") == 1


def test_the_loader_preference_needs_nvidia_smi_indices(monkeypatch, settings, stored):
    from core.inference import llama_cpp

    stored(this_machine())
    settings(prefer_fast_link = True)
    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    assert llama_cpp._hardware_check_link_preference() == {0: 24.9, 1: 1.5}
    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", False)
    assert llama_cpp._hardware_check_link_preference() is None, "torch ordinals are another index space"


# ── The guardrail judges the card the loader uses ───────────────────────────────────────────────


def _verdict(gpus, *, need, preference = None, pinned = True, shared = ()):
    from core.inference.load_verdict import PLACEMENT_AUTO, VerdictInputs, compute_load_verdict

    return compute_load_verdict(
        VerdictInputs(
            weights_bytes = int(need * 0.8),
            kv_bytes = int(need * 0.15),
            compute_bytes = int(need * 0.05),
            total_bytes = need,
            gpu_bytes = need,
            placement = PLACEMENT_AUTO,
            context_pinned = pinned,
            n_ctx = 8192,
            gpus = gpus,
            ram_available_bytes = 28 * GIB,
            link_preference = preference,
            link_shared = tuple(shared),
        )
    )


def test_the_verdict_names_the_fast_card_the_loader_picks():
    from core.inference.llama_cpp import LlamaCppBackend
    from core.inference.load_verdict import FULL_GPU, GpuMemory

    free = {0: int(22.5 * GIB), 1: int(23.5 * GIB)}
    gpus = [GpuMemory(index = i, free_bytes = f, total_bytes = 24 * GIB) for i, f in free.items()]
    need = 17 * GIB
    without = _verdict(gpus, need = need)
    with_pref = _verdict(gpus, need = need, preference = FAST_SLOW)
    assert without.level == with_pref.level == FULL_GPU
    assert without.gpu_indices == (0, 1)
    assert with_pref.gpu_indices == (0,), "the verdict is drawn against the card the load lands on"
    rows = [(i, f // MIB) for i, f in free.items()]
    totals = {0: 24_576, 1: 24_576}
    picked, _ = LlamaCppBackend._select_gpus(need, rows, usable_fraction = 0.97, total_by_idx = totals, prefer_bandwidth = FAST_SLOW)
    assert tuple(picked) == with_pref.gpu_indices


def test_when_the_fast_card_cannot_hold_it_both_keep_the_old_placement():
    from core.inference.llama_cpp import LlamaCppBackend
    from core.inference.load_verdict import GpuMemory

    free = {0: int(17.5 * GIB), 1: int(23.5 * GIB)}
    gpus = [GpuMemory(index = i, free_bytes = f, total_bytes = 24 * GIB) for i, f in free.items()]
    need = 17 * GIB
    verdict = _verdict(gpus, need = need, preference = FAST_SLOW)
    assert verdict.gpu_indices == (0, 1), "pooled, exactly as without the option"
    rows = [(i, f // MIB) for i, f in free.items()]
    picked, _ = LlamaCppBackend._select_gpus(need, rows, usable_fraction = 0.97, total_by_idx = {0: 24_576, 1: 24_576}, prefer_bandwidth = FAST_SLOW)
    assert picked == [1]


def test_the_route_hands_the_verdict_no_preference_for_a_pick_or_a_tensor_split(monkeypatch):
    import routes.inference as inference
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_hardware_check_link_preference", lambda: dict(FAST_SLOW))
    monkeypatch.setattr(llama_cpp, "_hardware_check_avoids_env_tensor", lambda *a, **k: False)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: SimpleNamespace(_other_planned_vram_mib = lambda: {}))
    assert inference._guardrail_link_preference(gpu_ids = [1], llama_extra_args = None, tensor_parallel = False) == (None, ())
    assert inference._guardrail_link_preference(gpu_ids = None, llama_extra_args = None, tensor_parallel = True) == (None, ())
    assert inference._guardrail_link_preference(
        gpu_ids = None, llama_extra_args = None, tensor_parallel = False, gpu_memory_mode = "manual"
    ) == (None, ()), "Manual memory hands every card to llama.cpp, no planner pick"
    assert inference._guardrail_link_preference(gpu_ids = None, llama_extra_args = None, tensor_parallel = False) == (FAST_SLOW, ())


def test_training_auto_selection_prefers_the_fast_card(monkeypatch, settings, stored):
    from utils.hardware import hardware as hw

    stored(this_machine())
    settings(prefer_fast_link = True)
    ranked = [{"index": 1, "free_gb": 23.5}, {"index": 0, "free_gb": 22.4}]
    devices = [{"index": 0, "index_kind": "physical"}, {"index": 1, "index_kind": "physical"}]
    assert hw._fast_link_single_gpu(ranked, devices, 14.0) == 0
    assert hw._fast_link_single_gpu(ranked, devices, 23.0) is None, "too big for GPU 0 with margin"
    settings(prefer_fast_link = False)
    assert hw._fast_link_single_gpu(ranked, devices, 14.0) is None


# ── Tensor split ────────────────────────────────────────────────────────────────────────────────


def test_tensor_split_concern_names_the_slow_link_and_the_pair(settings, stored):
    stored(this_machine())
    settings(avoid_tensor_split = True)
    assert hc.tensor_split_concern(None) == {"slow_gpus": [1], "no_peer_pairs": [[0, 1]]}
    assert hc.tensor_split_concern([0]) is None, "one card spans no link"
    settings(avoid_tensor_split = False)
    assert hc.tensor_split_concern(None) is None


def test_only_an_inherited_tensor_mode_is_declined(monkeypatch, settings, stored):
    stored(this_machine())
    settings(avoid_tensor_split = True)
    monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "tensor")
    assert hc.env_tensor_split_avoided(False, None, None) is True
    assert hc.env_tensor_split_avoided(True, None, None) is False, "the toggle is the user's"
    assert hc.env_tensor_split_avoided(False, ["--split-mode", "tensor"], None) is False, "so are extras"
    settings(avoid_tensor_split = False)
    assert hc.env_tensor_split_avoided(False, None, None) is False
    settings(avoid_tensor_split = True)
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE")
    assert hc.env_tensor_split_avoided(False, None, None) is False


def test_the_estimate_prices_the_split_mode_the_loader_launches(monkeypatch):
    """_placement_pricer appends --split-mode layer exactly when the loader declines the env."""
    import inspect

    import routes.inference as inference
    from core.inference import llama_cpp

    source = inspect.getsource(inference._placement_pricer)
    assert "_hardware_check_avoids_env_tensor(tensor_parallel, llama_extra_args, selected_gpu_ids)" in source
    loader = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    assert "_hardware_check_avoids_env_tensor(\n                    tensor_parallel, extra_args, gpu_ids\n" in loader


# ── Training warning ────────────────────────────────────────────────────────────────────────────


def test_the_training_warning_names_the_slow_card_and_offloaded_checkpointing(settings, stored):
    stored(this_machine())
    settings(warn_training_slow_link = True)
    warning = hc.training_slow_link_warning([1], "unsloth")
    assert warning == {
        "gpus": [{"index": 1, "width": 1, "width_max": 16, "h2d_gibs": 1.5}],
        "best_gpu": 0,
        "best_h2d_gibs": 24.9,
        "offloaded_gradient_checkpointing": True,
    }
    assert hc.training_slow_link_warning([0], "unsloth") is None
    assert hc.training_slow_link_warning([1], "true")["offloaded_gradient_checkpointing"] is False
    settings(warn_training_slow_link = False)
    assert hc.training_slow_link_warning([1], "unsloth") is None


def test_the_training_estimate_carries_the_warning(monkeypatch):
    import routes.training_vram as tv

    monkeypatch.setattr(hc, "training_slow_link_warning", lambda ids, gc: {"gpus": [{"index": ids[0]}], "gc": gc})
    assert tv._slow_link_warning([1], "unsloth") == {"gpus": [{"index": 1}], "gc": "unsloth"}

    def boom(*_a, **_k):
        raise RuntimeError("x")

    monkeypatch.setattr(hc, "training_slow_link_warning", boom)
    assert tv._slow_link_warning([1], "unsloth") is None


def test_the_training_estimate_response_accepts_the_warning():
    from models.training import TrainingEstimateResponse

    body = TrainingEstimateResponse(
        verdict = "fits",
        slow_link = {"gpus": [{"index": 1, "width": 1, "width_max": 16, "h2d_gibs": 1.5}], "best_gpu": 0, "best_h2d_gibs": 24.9, "offloaded_gradient_checkpointing": True},
    )
    assert body.model_dump()["slow_link"]["gpus"][0]["index"] == 1


# ── The resource strip badge ────────────────────────────────────────────────────────────────────


def test_the_strip_badges_every_measured_card_whatever_the_options(settings, stored):
    stored(this_machine())
    settings()
    badges = hc.resource_link_badges()
    assert badges[1] == {"width": 1, "width_max": 16, "gen": 4, "h2d_gibs": 1.5, "best_gpu": 0, "best_h2d_gibs": 24.9, "slow": True}
    assert badges[0]["slow"] is False


def test_the_resources_snapshot_carries_the_link_only_when_measured():
    import routes.resources as resources
    from utils.hardware.gpu_resources import GpuReading

    class Reader:
        def read_gpus(self, **_kw):
            return [GpuReading(0, "3090", 24 * GIB, 20 * GIB), GpuReading(1, "3090", 24 * GIB, 23 * GIB)]

        def holders(self):
            return None

    snap = resources.build_snapshot(reader = Reader(), models = [], loading = False, visible = None, link_badges = {1: {"width": 1, "slow": True}})
    assert "link" not in snap["gpus"][0]
    assert snap["gpus"][1]["link"] == {"width": 1, "slow": True}
    plain = resources.build_snapshot(reader = Reader(), models = [], loading = False, visible = None)
    assert all("link" not in gpu for gpu in plain["gpus"]), "no stored result in this home"


# ── Running: gating, the child, the auto-run ────────────────────────────────────────────────────


@pytest.fixture
def busy(monkeypatch):
    flags = {hc.SKIP_TRAINING: False, hc.SKIP_LOADING: False, hc.SKIP_GENERATING: False}
    monkeypatch.setattr(hc, "_busy_checks", {key: (lambda key = key: flags[key]) for key in flags})
    return flags


@pytest.fixture
def fake_runner(monkeypatch):
    calls = []

    def runner(trigger, progress = None):
        calls.append(trigger)
        return this_machine(trigger = trigger)

    monkeypatch.setattr(hc, "_runner", runner)
    return calls


def test_a_manual_run_is_refused_during_training_and_loads(busy, fake_runner):
    busy[hc.SKIP_TRAINING] = True
    assert hc.start_run(hc.TRIGGER_MANUAL, wait = True) == {"started": False, "reason": "training_active"}
    busy[hc.SKIP_TRAINING] = False
    busy[hc.SKIP_LOADING] = True
    assert hc.start_run(hc.TRIGGER_MANUAL, wait = True) == {"started": False, "reason": "model_loading"}
    assert hc.state_snapshot()["last_skip"]["reason"] == "model_loading"
    assert fake_runner == []


def test_a_manual_run_does_not_wait_for_a_generation_but_the_auto_run_does(busy, fake_runner):
    busy[hc.SKIP_GENERATING] = True
    assert hc.start_run(hc.TRIGGER_AUTO, wait = True)["reason"] == "generation_active"
    assert hc.start_run(hc.TRIGGER_MANUAL, wait = True) == {"started": True, "reason": None}
    assert fake_runner == ["manual"]
    assert hc.load_result()["trigger"] == "manual"
    assert hc.state_snapshot()["running"] is False


def test_a_second_run_is_refused_while_one_is_going(busy, fake_runner, monkeypatch):
    monkeypatch.setitem(hc._state, "running", True)
    assert hc.start_run(hc.TRIGGER_MANUAL, wait = True)["reason"] == "already_running"


def test_the_auto_run_waits_out_training_then_runs_once(busy, fake_runner, settings, monkeypatch):
    settings()
    monkeypatch.setattr(hc, "current_fingerprint", lambda: "fp-two-3090s")
    busy[hc.SKIP_TRAINING] = True
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            busy[hc.SKIP_TRAINING] = False

    clock = iter(range(0, 10_000, 1))
    outcome = hc.auto_run_loop(settle_s = 20, retry_s = 60, give_up_s = 3600, sleep = sleep, clock = lambda: next(clock))
    assert outcome == "ran"
    assert fake_runner == ["auto"]
    assert sleeps == [20, 60, 60], "settle first, then retry while training"


def test_the_auto_run_is_not_needed_with_a_current_result_or_the_setting_off(busy, fake_runner, settings, stored):
    settings()
    stored(this_machine())
    assert hc.auto_run_loop(settle_s = 0, sleep = lambda s: None) == "not_needed"
    hc.result_path().unlink()
    hc._result_cache = None
    settings(auto_run = False)
    assert hc.auto_run_loop(settle_s = 0, sleep = lambda s: None) == "not_needed"
    assert fake_runner == []


def test_the_auto_run_gives_up_and_stands_down(busy, fake_runner, settings, monkeypatch):
    settings()
    monkeypatch.setattr(hc, "current_fingerprint", lambda: "fp-new")
    busy[hc.SKIP_LOADING] = True
    clock = iter(range(0, 100_000, 1000))
    assert hc.auto_run_loop(settle_s = 0, retry_s = 0, give_up_s = 3000, sleep = lambda s: None, clock = lambda: next(clock)) == "gave_up"
    assert hc.auto_run_loop(lambda: False, settle_s = 0, sleep = lambda s: None) == "stopped"
    assert fake_runner == []


def test_the_kill_switch_stops_the_auto_run(monkeypatch):
    monkeypatch.setenv(hc.AUTO_RUN_ENV_VAR, "0")
    assert hc.start_auto_run() is None
    assert hc.auto_run_wanted() is False


class FakeChild:
    """A probe child: emits the given lines, optionally hangs, records what the parent wrote."""

    def __init__(self, lines, *, hang = False, returncode = 0):
        self._lines = list(lines)
        self._hang = hang
        self.returncode = None
        self._final = returncode
        self.killed = False
        self.written = []
        self._release = threading.Event()
        parent = self

        class Stdin:
            def write(self, text):
                parent.written.append(text)

            def flush(self):
                pass

            def close(self):
                pass

        self.stdin = Stdin()
        self.stdout = self._stdout()
        self.stderr = io.StringIO("Traceback: boom\n")

    def _stdout(self):
        for line in self._lines:
            yield line + "\n"
        if self._hang:
            self._release.wait(30)

    def wait(self, timeout = None):
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired("probe", timeout)
        self.returncode = self._final
        return self.returncode

    def kill(self):
        self.killed = True
        self._release.set()
        self.returncode = -9


def _devices():
    return [
        {"index": 0, "uuid": "GPU-0000", "memory_free_bytes": 20 * GIB},
        {"index": 1, "uuid": "GPU-0001", "memory_free_bytes": 23 * GIB},
    ]


def test_the_child_protocol_reads_the_link_under_load_and_releases_each_card():
    events = [
        {"event": "devices", "devices": []},
        {"event": "load", "uuid": "GPU-0000"},
        {"event": "bandwidth", "uuid": "GPU-0000", "h2d_gibs": 24.9, "d2h_gibs": 24.4},
        {"event": "load", "uuid": "GPU-0001"},
        {"event": "bandwidth", "uuid": "GPU-0001", "h2d_gibs": 1.48, "d2h_gibs": 1.53},
        {"event": "pair", "a": "GPU-0000", "b": "GPU-0001", "peer_ab": False, "peer_ba": False, "copy_ab_gibs": 1.47, "copy_ba_gibs": 1.51},
        {"event": "done"},
    ]
    child = FakeChild([json.dumps(e) for e in events])
    links = {"GPU-0000": {"gen_current": 4, "gen_max": 4, "width_current": 16, "width_max": 16},
             "GPU-0001": {"gen_current": 4, "gen_max": 4, "width_current": 1, "width_max": 16}}
    out = hc.run_gpu_probe(_devices(), link_reader = lambda uuid: links[uuid], popen = lambda *a, **k: child)
    assert out["error"] is None
    assert out["gpus"]["GPU-0001"] == {"status": "measured", "link": links["GPU-0001"], "h2d_gibs": 1.48, "d2h_gibs": 1.53}
    assert child.written == ["go GPU-0000\n", "go GPU-0001\n"]
    assert out["pairs"][0]["peer_ab"] is False


def test_a_card_without_room_for_the_buffer_is_skipped_without_a_child():
    started = []
    devices = [{"index": 0, "uuid": "GPU-0000", "memory_free_bytes": 300 * MIB}]
    out = hc.run_gpu_probe(devices, popen = lambda *a, **k: started.append(a) or None)
    assert started == []
    assert out["gpus"]["GPU-0000"] == {"status": "skipped", "reason": "low_free_memory"}


def test_a_hung_child_is_killed_at_the_deadline_and_the_run_still_returns():
    child = FakeChild([json.dumps({"event": "devices", "devices": []})], hang = True)
    started = time.monotonic()
    out = hc.run_gpu_probe(_devices(), link_reader = lambda uuid: None, popen = lambda *a, **k: child, timeout_s = 1.5)
    assert time.monotonic() - started < 15
    assert child.killed is True
    assert out["error"] == "the measurement timed out"
    assert {g["status"] for g in out["gpus"].values()} == {"failed"}


def test_a_child_that_dies_early_is_reported_with_its_code():
    child = FakeChild([json.dumps({"event": "devices", "devices": []})], returncode = 3)
    out = hc.run_gpu_probe(_devices(), link_reader = lambda uuid: None, popen = lambda *a, **k: child)
    assert out["error"].startswith("the measurement process exited early (code 3)")


def test_a_child_that_cannot_start_fails_soft():
    def refuse(*_a, **_k):
        raise OSError("no python")

    out = hc.run_gpu_probe(_devices(), popen = refuse)
    assert out["error"].startswith("could not start the measurement")
    assert all(g["status"] == "failed" for g in out["gpus"].values())


def test_run_check_survives_every_part_failing():
    def no_inventory():
        raise RuntimeError("nvidia-smi hung")

    result = hc.run_check(
        inventory_reader = no_inventory,
        storage_probe = lambda: {"locations": [], "nvme_present": None, "error": "TimeoutExpired"},
    )
    assert result["gpus"] == [] and result["gpu_error"].startswith("could not read the GPUs")
    assert result["storage_error"] == "TimeoutExpired"
    assert result["schema"] == hc.SCHEMA_VERSION


def test_the_link_under_load_takes_the_wider_of_two_readings():
    readings = iter([
        {"GPU-1": {"gen_current": 1, "gen_max": 4, "width_current": 1, "width_max": 16}},
        {"GPU-1": {"gen_current": 4, "gen_max": 4, "width_current": 1, "width_max": 16}},
    ])
    link = hc.read_link_under_load("GPU-1", reader = lambda: next(readings), sleep = lambda s: None)
    assert link == {"gen_current": 4, "gen_max": 4, "width_current": 1, "width_max": 16}


def test_the_child_never_imports_torch_into_the_server():
    """The probe runs as `python -I <file>`; the backend module itself must not touch torch."""
    argv = hc.gpu_probe_argv(["GPU-a"], python = "py")
    assert argv[:3] == ["py", "-I", str(Path(hc.__file__).with_name("hardware_check_probe.py"))]
    source = Path(hc.__file__).read_text(encoding = "utf-8")
    assert "import torch" not in source


def test_windows_storage_is_parsed_from_one_powershell_answer():
    def run(argv, **_kw):
        assert "-EncodedCommand" in argv
        payload = {
            "drives": [{"letter": "D", "disk": "1", "bus": "SATA", "media": "SSD", "model": "M500DC"}],
            "disks": [{"id": "2", "bus": "NVMe", "media": "SSD", "model": "Samsung"}, {"id": "1", "bus": "SATA", "media": "SSD"}],
        }
        return SimpleNamespace(returncode = 0, stdout = json.dumps(payload))

    out = hc.probe_storage(
        [("hf_cache", Path("D:/hf/hub")), ("temp", Path("Z:/tmp"))],
        windows_query = lambda letters: hc._windows_storage(letters, run = run),
        system = "Windows",
    )
    hf, temp = out["locations"]
    assert (hf["drive"], hf["bus_type"], hf["media_type"], hf["model"]) == ("D:", "SATA", "SSD", "M500DC")
    assert temp["bus_type"] is None
    assert out["nvme_present"] is True


def test_a_storage_query_that_fails_is_reported_not_raised():
    def boom(_letters):
        raise subprocess.TimeoutExpired("powershell", 20)

    out = hc.probe_storage([("hf_cache", Path("D:/hf"))], windows_query = boom, system = "Windows")
    assert out["error"] == "TimeoutExpired"
    assert out["locations"][0]["bus_type"] is None


# ── Settings ────────────────────────────────────────────────────────────────────────────────────


def test_settings_default_to_auto_run_on_and_every_option_off():
    assert hcs.coerce_settings(None) == {
        "auto_run": True,
        "prefer_fast_link": False,
        "avoid_tensor_split": False,
        "warn_training_slow_link": False,
    }
    assert hcs.coerce_settings({"prefer_fast_link": "yes", "bogus": True})["prefer_fast_link"] is False


def test_settings_persist_in_the_studio_database():
    hcs.update_hardware_check_settings({"prefer_fast_link": True})
    hcs._cache.clear()
    assert hcs.get_hardware_check_settings()["prefer_fast_link"] is True
    with pytest.raises(ValueError):
        hcs.update_hardware_check_settings({"nonsense": True})
    with pytest.raises(ValueError):
        hcs.update_hardware_check_settings({"auto_run": "off"})


# ── The routes ──────────────────────────────────────────────────────────────────────────────────


def _app(overrides = None) -> TestClient:
    app = FastAPI()
    app.include_router(hardware_check_route.router, prefix = "/api/hardware-check")
    app.dependency_overrides[hardware_check_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[hardware_check_route._require_ui_session] = lambda: None
    app.dependency_overrides.update(overrides or {})
    return TestClient(app, raise_server_exceptions = False)


def test_every_route_carries_the_owner_and_ui_session_guards():
    for route in hardware_check_route.router.routes:
        names = {getattr(dep.call, "__name__", "") for dep in route.dependant.dependencies}
        assert {"_require_installation_owner", "_require_ui_session"} <= names, route.path


def test_an_api_key_cannot_read_or_run_the_check():
    app = FastAPI()
    app.include_router(hardware_check_route.router, prefix = "/api/hardware-check")
    app.dependency_overrides[hardware_check_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[hardware_check_route.authenticated_via_api_key] = lambda: True
    client = TestClient(app, raise_server_exceptions = False)
    assert client.get("/api/hardware-check").status_code == 403
    assert client.post("/api/hardware-check/run").status_code == 403
    assert client.put("/api/hardware-check/settings", json = {"prefer_fast_link": True}).status_code == 403


def test_a_managed_account_cannot_read_or_change_the_check():
    async def managed_subject() -> str:
        bind_account(AccountContext("alice", "alice", "user"))
        return "alice"

    client = _app({hardware_check_route.get_current_subject: managed_subject})
    assert client.get("/api/hardware-check").status_code == 403
    assert client.post("/api/hardware-check/apply-recommended").status_code == 403


def test_the_owner_reads_the_result_with_findings(settings, stored):
    stored(this_machine())
    body = _app().get("/api/hardware-check").json()
    assert body["up_to_date"] is True
    assert body["settings"]["auto_run"] is True and body["settings"]["prefer_fast_link"] is False
    assert [f["id"] for f in body["result"]["findings"]][:2] == ["slow_link", "no_peer_access"]
    assert body["result"]["recommended"]["prefer_fast_link"] is True
    assert body["state"]["running"] is False


def test_the_run_route_reports_why_it_did_not_start(busy, fake_runner, settings):
    busy[hc.SKIP_TRAINING] = True
    body = _app().post("/api/hardware-check/run").json()
    assert body["started"] is False and body["reason"] == "training_active"
    assert body["status"]["state"]["last_skip"]["reason"] == "training_active"


def test_apply_recommended_switches_on_exactly_what_is_recommended(settings, stored):
    stored(this_machine(gpus = [_gpu(0, h2d = 24.9), _gpu(1, h2d = 24.6)], pairs = [{"a": 0, "b": 1, "peer_ab": False, "peer_ba": False, "copy_ab_gibs": 1.5, "copy_ba_gibs": 1.5}]))
    body = _app().post("/api/hardware-check/apply-recommended").json()
    assert body["applied"] == ["avoid_tensor_split"], "no slow link: only the P2P recommendation"
    assert body["status"]["settings"] == {
        "auto_run": True,
        "prefer_fast_link": False,
        "avoid_tensor_split": True,
        "warn_training_slow_link": False,
    }


def test_apply_recommended_needs_a_current_result(settings, stored):
    assert _app().post("/api/hardware-check/apply-recommended").status_code == 409
    current = stored(this_machine())
    current["fingerprint"] = "fp-moved"
    assert _app().post("/api/hardware-check/apply-recommended").status_code == 409


def test_the_settings_route_switches_one_option(settings):
    client = _app()
    body = client.put("/api/hardware-check/settings", json = {"warn_training_slow_link": True}).json()
    assert body["settings"]["warn_training_slow_link"] is True
    assert client.put("/api/hardware-check/settings", json = {}).status_code == 400


# ── Diagnostics ─────────────────────────────────────────────────────────────────────────────────


def test_the_diagnostics_report_carries_the_check(settings, stored):
    from utils import diagnostics

    with pytest.raises(diagnostics.Unavailable):
        diagnostics.collect_hardware_check()
    stored(this_machine())
    data = diagnostics.collect_hardware_check()
    assert data["up_to_date"] is True
    assert data["gpus"][1]["h2d_gibs"] == 1.5
    report = {"sections": {"hardware_check": {"status": "ok", "data": data}}, "generated_at": "now"}
    markdown = diagnostics.render_markdown(report)
    assert "### Hardware check" in markdown
    assert "| 1 | NVIDIA GeForce RTX 3090 | Gen4 x1 (max x16) | Gen1 x1 (max x16) | 1.5 GiB/s | 1.5 GiB/s |" in markdown
    assert "GPU 0 and GPU 1: peer access no/no" in markdown
    assert "- [warning] GPU 1 runs at PCIe x1" in markdown
