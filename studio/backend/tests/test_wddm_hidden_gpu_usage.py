# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPU memory other programs hold, which a Windows CUDA llama-server cannot see.

Field case: Windows 11, 2x RTX 3090 24 GB. LM Studio's Bionic.exe held ~13.5 GB on each
card. Studio's nvidia-smi probe saw 10571 / 10961 MiB free and warned the model would not
fit, then launched ``--fit on`` with nothing else. llama-server's own probe reported 23332
MiB free on both cards (the figure it reports for an empty card too), so its fitter planned
a full load and cudaMalloc failed on GPU 0. The fix widens the ``--fit-target`` margin by
what the child cannot see, and names the program in the warning and the OOM message.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _stub_module(name: str, **attrs):
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_module("loggers", get_logger = lambda name: __import__("logging").getLogger(name))
_stub_module("structlog", get_logger = lambda *a, **k: __import__("logging").getLogger("stub"))

import core.inference.llama_cpp as lc
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
from utils.hardware import gpu_process_memory as gpm

MIB = 1024 * 1024
GIB = 1024 * MIB

# nvidia-smi's rows in the field log: (index, free MiB, total MiB).
FIELD_ROWS = [(0, 10571, 24576), (1, 10961, 24576)]


# ── the arithmetic ─────────────────────────────────────────────────────────


class TestHiddenUsageArithmetic:
    def test_field_numbers(self):
        hidden = LlamaCppBackend._gpu_memory_outside_child_mib(FIELD_ROWS)
        # In use (total - free) less the 1 GiB a WDDM child already sees as taken.
        assert hidden == {0: 24576 - 1024 - 10571, 1: 24576 - 1024 - 10961}
        # llama-server's own reading put the blind spot at 23332 - 10571 = 12761 MiB on
        # GPU 0. The estimate may sit above that (more held back), never far below it.
        assert 12761 <= hidden[0] <= 12761 + 512

    def test_an_idle_card_hides_nothing(self):
        # nvidia-smi's own idle reading on the same host: 254 MiB reserved, nothing in use.
        assert LlamaCppBackend._gpu_memory_outside_child_mib([(1, 24322, 24576)]) == {1: 0.0}

    def test_a_small_card_reserves_its_share_not_a_flat_gib(self):
        # 5% of 4 GiB is 204.8 MiB; a flat 1 GiB would hide 820 MiB of real usage.
        hidden = LlamaCppBackend._gpu_memory_outside_child_mib([(0, 2048, 4096)])
        assert hidden[0] == pytest.approx(4096 - 204.8 - 2048)

    def test_rows_without_a_total_and_unselected_rows_are_left_out(self):
        rows = [(0, 1000, 0), (1, 2000, 24576), (2, 3000, 24576)]
        assert set(LlamaCppBackend._gpu_memory_outside_child_mib(rows, [1, 0])) == {1}


class TestGate:
    @pytest.fixture
    def windows_cuda(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend,
            "_child_vram_probe_is_process_local",
            staticmethod(lambda binary, is_vulkan_backend: not is_vulkan_backend),
        )
        monkeypatch.setattr(
            LlamaCppBackend,
            "_nvidia_driver_models",
            staticmethod(lambda: {0: "WDDM", 1: "WDDM"}),
        )

    def test_field_case_is_reported(self, windows_cuda):
        hidden = LlamaCppBackend._wddm_hidden_gpu_usage_mib(
            "x", FIELD_ROWS, None, is_vulkan_backend = False
        )
        assert set(hidden) == {0, 1}

    def test_below_a_gib_everywhere_is_noise(self, windows_cuda):
        rows = [(0, 23000, 24576), (1, 24322, 24576)]  # 552 MiB on GPU 0
        hidden = LlamaCppBackend._wddm_hidden_gpu_usage_mib(
            "x", rows, None, is_vulkan_backend = False
        )
        assert hidden == {}

    def test_one_busy_card_is_enough(self, windows_cuda):
        rows = [(0, 10571, 24576), (1, 24322, 24576)]
        hidden = LlamaCppBackend._wddm_hidden_gpu_usage_mib(
            "x", rows, None, is_vulkan_backend = False
        )
        assert hidden[0] > 12000 and hidden[1] == 0.0

    def test_a_tcc_card_sees_other_processes_already(self, windows_cuda, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_nvidia_driver_models", staticmethod(lambda: {0: "TCC", 1: "WDDM"})
        )
        hidden = LlamaCppBackend._wddm_hidden_gpu_usage_mib(
            "x", FIELD_ROWS, None, is_vulkan_backend = False
        )
        assert set(hidden) == {1}

    def test_an_unreadable_driver_model_counts_as_wddm(self, windows_cuda, monkeypatch):
        monkeypatch.setattr(LlamaCppBackend, "_nvidia_driver_models", staticmethod(lambda: None))
        hidden = LlamaCppBackend._wddm_hidden_gpu_usage_mib(
            "x", FIELD_ROWS, None, is_vulkan_backend = False
        )
        assert set(hidden) == {0, 1}

    def test_the_driver_model_is_only_asked_when_something_is_hidden(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend,
            "_child_vram_probe_is_process_local",
            staticmethod(lambda binary, is_vulkan_backend: True),
        )

        def _boom():
            raise AssertionError("queried nvidia-smi for an idle host")

        monkeypatch.setattr(LlamaCppBackend, "_nvidia_driver_models", staticmethod(_boom))
        assert LlamaCppBackend._wddm_hidden_gpu_usage_mib(
            "x", [(0, 24322, 24576)], None, is_vulkan_backend = False
        ) == {}

    def test_vulkan_is_out_of_scope(self, windows_cuda):
        assert LlamaCppBackend._wddm_hidden_gpu_usage_mib(
            "x", FIELD_ROWS, None, is_vulkan_backend = True
        ) == {}


def _process_local() -> bool:
    return LlamaCppBackend._child_vram_probe_is_process_local("x", is_vulkan_backend = False)


class TestPlatformGate:
    """The real gate: Windows, a CUDA build, nvidia-smi rows, not opted out."""

    @pytest.fixture
    def blind_host(self, monkeypatch):
        monkeypatch.setattr(lc.sys, "platform", "win32")
        monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
        monkeypatch.setattr(
            LlamaCppBackend, "_sysmem_fallback_risk", staticmethod(lambda binary = None: True)
        )
        monkeypatch.delenv("UNSLOTH_FIT_TARGET_OUTSIDE_USAGE", raising = False)

    def test_windows_cuda(self, blind_host):
        assert _process_local()

    @pytest.mark.parametrize("platform", ["linux", "darwin"])
    def test_other_platforms_see_every_process(self, blind_host, monkeypatch, platform):
        # Linux cudaMemGetInfo is system-wide: adding the usage again would hold it back twice.
        monkeypatch.setattr(lc.sys, "platform", platform)
        assert not _process_local()

    def test_a_non_cuda_build(self, blind_host, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_sysmem_fallback_risk", staticmethod(lambda binary = None: False)
        )
        assert not _process_local()

    def test_torch_fallback_rows_are_just_as_blind(self, blind_host, monkeypatch):
        monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", False)
        assert not _process_local()

    def test_opt_out(self, blind_host, monkeypatch):
        monkeypatch.setenv("UNSLOTH_FIT_TARGET_OUTSIDE_USAGE", "0")
        assert not _process_local()


# ── composing the margin ──────────────────────────────────────────────────


class TestFitTargetComposition:
    def test_field_case_broadcasts_the_fuller_card(self):
        hidden = LlamaCppBackend._gpu_memory_outside_child_mib(FIELD_ROWS)
        value = LlamaCppBackend._outside_usage_fit_target(
            hidden, base_margin_mib = 1024.0, device_order = None
        )
        # llama.cpp's 1024 default the legacy path relied on, plus GPU 0's hidden 12981.
        assert value == "14005"

    def test_per_device_in_the_child_order(self):
        value = LlamaCppBackend._outside_usage_fit_target(
            {0: 12981.0, 1: 0.0}, base_margin_mib = 1024.0, device_order = [1, 0]
        )
        assert value == "1024,14005"

    def test_a_card_not_measured_keeps_the_base(self):
        value = LlamaCppBackend._outside_usage_fit_target(
            {2: 4096.0}, base_margin_mib = 512.0, device_order = [0, 2]
        )
        assert value == "512,4608"

    def test_equal_margins_are_said_once(self):
        assert (
            LlamaCppBackend._outside_usage_fit_target(
                {0: 2000.0, 1: 2000.0}, base_margin_mib = 1024.0, device_order = [0, 1]
            )
            == "3024"
        )

    def test_rounds_up(self):
        assert (
            LlamaCppBackend._outside_usage_fit_target(
                {0: 1500.2}, base_margin_mib = 512.0, device_order = None
            )
            == "2013"
        )

    def test_nothing_hidden(self):
        assert (
            LlamaCppBackend._outside_usage_fit_target(
                {}, base_margin_mib = 1024.0, device_order = None
            )
            is None
        )

    def test_composes_with_the_vram_budget_rather_than_replacing_it(self):
        caps = {"supports_fit_target": True}
        # Manual + Auto at a budget lowered by 2 GiB: the flags already ask for 512 + 2048.
        flags = LlamaCppBackend._ctx_integrity_flags(
            1, True, True, 0, 0, caps, fit_target_delta_mib = 2048.0
        )
        assert flags[-2:] == ["--fit-target", "2560"]
        base = LlamaCppBackend._fit_target_margin_mib(
            auto_fit = True, fit_target_delta_mib = 2048.0
        )
        value = LlamaCppBackend._outside_usage_fit_target(
            {0: 12981.0, 1: 12591.0}, base_margin_mib = base, device_order = [0, 1]
        )
        composed = LlamaCppBackend._with_fit_target(flags, value)
        assert composed.count("--fit-target") == 1
        assert composed[-2:] == ["--fit-target", f"{2560 + 12981},{2560 + 12591}"]
        # Everything else in the flags is untouched.
        assert composed[:-2] == flags[:-2]

    def test_the_legacy_path_gains_a_margin_it_did_not_emit(self):
        caps = {"supports_fit_target": True, "supports_fit_ctx": True}
        flags = LlamaCppBackend._ctx_integrity_flags(1, True, False, 8192, 8192, caps)
        assert "--fit-target" not in flags  # llama.cpp's default, left implicit
        composed = LlamaCppBackend._with_fit_target(flags, "14005")
        assert composed == [*flags, "--fit-target", "14005"]


class TestDeviceOrder:
    @pytest.fixture(autouse = True)
    def _pci(self, monkeypatch):
        monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)

    def _order(self, gpu_indices, gpu_ids = None, visible = (0, 1), extra = None, **env):
        return LlamaCppBackend._fit_target_device_order(
            gpu_indices,
            gpu_ids = gpu_ids,
            visible_ids = list(visible),
            extra_args = extra,
            env = env,
        )

    def test_an_explicit_pick_is_pinned_in_its_own_order(self):
        # The launch writes CUDA_VISIBLE_DEVICES=1,0 under PCI_BUS_ID for this pick.
        assert self._order([1, 0], gpu_ids = [1, 0]) == [1, 0]

    def test_an_auto_subset_without_an_inherited_mask(self):
        assert self._order([1]) == [1]

    def test_an_inherited_mask_keeps_cudas_order_not_ours(self):
        assert self._order([0, 1], CUDA_VISIBLE_DEVICES = "1,0") is None

    def test_unpinned_follows_pci_order_only_when_the_env_says_so(self):
        # The field case: nothing pinned, nothing inherited -> CUDA's FASTEST_FIRST.
        assert self._order(None) is None
        assert self._order(None, CUDA_DEVICE_ORDER = "PCI_BUS_ID") == [0, 1]

    def test_one_visible_card_has_one_order(self):
        assert self._order(None, visible = (3,)) == [3]

    def test_a_pass_through_device_reindexes_the_list(self):
        pci = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        assert self._order(None, extra = ["--device", "CUDA1"], **pci) is None
        assert self._order(None, LLAMA_ARG_DEVICE = "CUDA1", **pci) is None
        # A pick strips the pass-through, so the pin decides again.
        assert self._order([0, 1], gpu_ids = [0, 1], extra = ["--device", "CUDA1"]) == [0, 1]

    def test_torch_ordinals_are_not_pci_ids(self, monkeypatch):
        monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", False)
        assert self._order([0, 1], gpu_ids = [0, 1]) is None


class TestWithFitTarget:
    def test_replaces_the_existing_pair(self):
        assert LlamaCppBackend._with_fit_target(
            ["--kv-unified", "--fit-target", "512", "--fit-ctx", "8192"], "9000"
        ) == ["--kv-unified", "--fit-ctx", "8192", "--fit-target", "9000"]

    def test_appends_when_absent(self):
        assert LlamaCppBackend._with_fit_target([], "9000") == ["--fit-target", "9000"]


# ── naming the culprit ────────────────────────────────────────────────────


# Get-Counter '\GPU Process Memory(*)\Dedicated Usage' as the module prints it. phys_0 on
# BOTH cards: phys is the node inside a linked-adapter group, the LUID is the card.
_LUID0 = "luid_0x00000000_0x00018d33_phys_0"
_LUID1 = "luid_0x00000000_0x00019e72_phys_0"
_COUNTER_DUMP = "\n".join(
    [
        f"pid_1372_{_LUID0}|{13800 * MIB}",
        f"pid_1372_{_LUID1}|{13400 * MIB}",
        f"pid_9552_{_LUID0}|{300 * MIB}",  # the desktop: below the naming floor
        f"pid_2820_{_LUID0}|{200 * MIB}",  # Studio itself
        "pid_77_luid_0x00000000_0x0001746f_phys_0|0",  # idle samples are dropped
        "__NONE__",
        "garbage line",
        "pid_x_luid_0x0_0x1_phys_0|5",
    ]
)
_LUID0_INT = 0x18D33
_LUID1_INT = 0x19E72


class TestCounterParser:
    def test_parses_pid_luid_and_bytes(self):
        rows = sorted(gpm.parse_gpu_process_memory(_COUNTER_DUMP))
        assert rows == sorted(
            [
                (1372, _LUID0_INT, 13800 * MIB),
                (1372, _LUID1_INT, 13400 * MIB),
                (9552, _LUID0_INT, 300 * MIB),
                (2820, _LUID0_INT, 200 * MIB),
            ]
        )

    def test_the_high_luid_half_is_kept(self):
        rows = gpm.parse_gpu_process_memory("pid_5_luid_0x00000001_0x00000002_phys_0|7")
        assert rows == [(5, (1 << 32) | 2, 7)]

    def test_a_linked_adapter_group_sums_per_pid(self):
        dump = "pid_5_luid_0x0_0x2_phys_0|100\npid_5_luid_0x0_0x2_phys_1|50"
        assert gpm.parse_gpu_process_memory(dump) == [(5, 2, 150)]

    def test_empty_and_none(self):
        assert gpm.parse_gpu_process_memory("") == []
        assert gpm.parse_gpu_process_memory(None) == []

    def test_the_query_is_windows_only(self, monkeypatch):
        monkeypatch.setattr(gpm.platform, "system", lambda: "Linux")
        assert gpm.query_gpu_process_memory() is None

    def test_a_failed_query_is_none(self, monkeypatch):
        monkeypatch.setattr(gpm.platform, "system", lambda: "Windows")

        def _raise(*a, **k):
            raise subprocess.TimeoutExpired("powershell", 5)

        monkeypatch.setattr(gpm.subprocess, "run", _raise)
        assert gpm.query_gpu_process_memory() is None


def _used_by_index(rows):
    return {idx: (total - free) * MIB for idx, free, total in rows}


class TestAdapterMatching:
    def test_matches_by_usage(self):
        used_by_luid = {_LUID0_INT: 14000 * MIB, _LUID1_INT: 13400 * MIB}
        assert gpm.match_adapters_to_gpus(used_by_luid, _used_by_index(FIELD_ROWS)) == {
            _LUID0_INT: 0,
            _LUID1_INT: 1,
        }

    def test_an_idle_card_has_no_adapter_samples(self):
        rows = [(0, 10571, 24576), (1, 24322, 24576)]
        used_by_luid = {_LUID0_INT: 14000 * MIB}
        assert gpm.match_adapters_to_gpus(used_by_luid, _used_by_index(rows)) == {_LUID0_INT: 0}

    def test_readings_that_disagree_give_no_mapping(self):
        used_by_luid = {_LUID0_INT: 4000 * MIB}
        assert gpm.match_adapters_to_gpus(used_by_luid, _used_by_index(FIELD_ROWS)) == {}

    def test_too_many_cards_is_not_searched(self):
        rows = {i: GIB for i in range(5)}
        assert gpm.match_adapters_to_gpus({1: GIB}, rows) == {}


class TestDescribeHolders:
    def _describe(self, samples = None, **kw):
        kw.setdefault("used_by_index", _used_by_index(FIELD_ROWS))
        kw.setdefault("exclude_pids", {2820})
        kw.setdefault("process_name", lambda pid: {1372: "Bionic.exe"}.get(pid, f"p{pid}"))
        return gpm.describe_gpu_memory_holders(
            gpm.parse_gpu_process_memory(_COUNTER_DUMP) if samples is None else samples, **kw
        )

    def test_field_case(self):
        assert self._describe() == "Bionic.exe (PID 1372) ~13.5 GiB on GPU 0, ~13.1 GiB on GPU 1"

    def test_studio_is_never_named(self):
        samples = [(2820, _LUID0_INT, 14000 * MIB), (2820, _LUID1_INT, 13400 * MIB)]
        assert self._describe(samples) is None

    def test_only_the_cards_this_load_uses(self):
        assert self._describe(gpu_indices = [1]) == "Bionic.exe (PID 1372) ~13.1 GiB on GPU 1"

    def test_without_a_mapping_the_total_is_quoted(self):
        rows = [(0, 20000, 24576), (1, 20000, 24576)]  # disagrees with the counters
        assert (
            self._describe(used_by_index = _used_by_index(rows))
            == "Bionic.exe (PID 1372) ~26.6 GiB"
        )

    def test_the_biggest_holders_first_and_at_most_three(self):
        samples = [(p, _LUID0_INT, (p + 1) * GIB) for p in range(5)]
        text = self._describe(samples, used_by_index = {0: 15 * GIB})
        assert text.split("; ")[0].startswith("p4 (PID 4)")
        assert text.count("PID") == 3

    def test_a_non_nvidia_adapter_is_ignored(self):
        samples = [(1372, 0x1746F, 2 * GIB)]  # the AMD iGPU on the field host
        assert self._describe(samples, nvidia_luids = {_LUID0_INT, _LUID1_INT}) is None


class TestHolderNoteIsLazy:
    def _backend(self, state):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._hidden_gpu_usage = state
        return backend

    def test_nothing_hidden_never_queries(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("spawned PowerShell with nothing hidden")

        monkeypatch.setattr(gpm, "query_gpu_process_memory", _boom)
        assert self._backend(None)._hidden_gpu_usage_note() is None

    def test_looked_up_once_and_worded_for_the_message(self, monkeypatch):
        calls = []

        def _query(*a, **k):
            calls.append(1)
            return gpm.parse_gpu_process_memory(_COUNTER_DUMP)

        monkeypatch.setattr(gpm, "query_gpu_process_memory", _query)
        monkeypatch.setattr(gpm, "studio_process_ids", lambda: {2820})
        monkeypatch.setattr(gpm, "nvidia_adapter_luids", lambda: {_LUID0_INT, _LUID1_INT})
        monkeypatch.setattr(gpm, "_process_name", lambda pid: "Bionic.exe")
        backend = self._backend({"gpu_mem": FIELD_ROWS, "mib": {0: 12981.0, 1: 12591.0}})
        note = backend._hidden_gpu_usage_note()
        assert note == (
            "GPU memory in use by other programs: Bionic.exe (PID 1372) ~13.5 GiB on GPU 0, "
            "~13.1 GiB on GPU 1."
        )
        assert backend._hidden_gpu_usage_note() == note
        assert calls == [1]

    def test_a_failed_lookup_is_silent(self, monkeypatch):
        monkeypatch.setattr(gpm, "query_gpu_process_memory", lambda *a, **k: None)
        backend = self._backend({"gpu_mem": FIELD_ROWS, "mib": {0: 12981.0}})
        assert backend._hidden_gpu_usage_note() is None



# ── the out-of-memory message ─────────────────────────────────────────────


_OOM_TAIL = "\n".join(
    [
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12730.50 MiB on device 0: "
        "cudaMalloc failed: out of memory",
        "llama_model_load: error loading model: unable to allocate CUDA0 buffer",
    ]
)
_NOTE = (
    "GPU memory in use by other programs: Bionic.exe (PID 1372) ~13.5 GB on GPU 0, "
    "~13.1 GB on GPU 1."
)


class TestOutOfMemoryMessageNamesTheProgram:
    def _classify(self, out, note):
        return LlamaCppBackend._classify_llama_start_failure(
            out, "/m/big.gguf", "local/big", 1, gpu_memory_note = note
        )

    def test_the_note_replaces_the_guess(self):
        msg = self._classify(_OOM_TAIL, _NOTE)
        head = msg.split("\n\n", 1)[0]
        assert head.startswith(
            "Not enough GPU memory to load this model: llama.cpp could not allocate a GPU "
            "buffer. GPU memory in use by other programs: Bionic.exe (PID 1372)"
        )
        assert "Close or unload it there, then retry." in head
        assert "may be using GPU memory" not in head
        assert head.index("Bionic.exe") < head.index("lower the context length")

    def test_a_callable_is_only_asked_for_a_gpu_allocation_failure(self):
        calls = []

        def _note():
            calls.append(1)
            return _NOTE

        self._classify("error: unknown model architecture: 'foo'", _note)
        assert calls == []
        assert "Bionic.exe" in self._classify(_OOM_TAIL, _note)
        assert calls == [1]

    def test_no_note_keeps_the_generic_wording(self):
        head = self._classify(_OOM_TAIL, lambda: None).split("\n\n", 1)[0]
        assert "Another program (another LLM app, a game, a training run) may be using" in head

    def test_a_note_that_raises_never_breaks_the_message(self):
        def _boom():
            raise RuntimeError("counter read blew up")

        assert self._classify(_OOM_TAIL, _boom).startswith("Not enough GPU memory")


# ── end to end: the argv the field load would now get ─────────────────────


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _field_backend(tmp_path: Path, memory = FIELD_ROWS, size_gb: int = 27):
    backend = LlamaCppBackend()
    gguf = _write_gguf(tmp_path / "model.gguf")
    backend._get_gpu_memory = lambda _binary = None, **_kw: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [
        (index, free) for index, free, _total in memory
    ]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    # 27 GB of weights against the field's free memory: no GPU subset holds it, so the
    # launch hands placement to --fit on.
    backend._get_gguf_size_bytes = lambda _path: size_gb * GIB
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    backend._launch_host_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout, **_kw: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_fit_target": True,
        "supports_fit_ctx": True,
    }
    backend._planned_tensor_spill = lambda *a, **k: None
    return backend, gguf


def _launch(backend, gguf, **load_kwargs):
    captured = {}
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", **load_kwargs)
        )
    return captured["cmd"]


@pytest.fixture
def blind_child(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend,
        "_child_vram_probe_is_process_local",
        staticmethod(lambda binary, is_vulkan_backend: not is_vulkan_backend),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_nvidia_driver_models", staticmethod(lambda: {0: "WDDM", 1: "WDDM"})
    )
    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(LlamaCppBackend, "_hidden_gpu_usage_note", lambda self: _NOTE)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "LLAMA_ARG_FIT_TARGET",
        "LLAMA_ARG_DEVICE",
    ):
        monkeypatch.delenv(name, raising = False)


def _record_warnings(monkeypatch) -> list[str]:
    """The module logger's warnings, formatted. structlog does not reach caplog here."""
    seen: list[str] = []
    real = lc.logger

    class _Recorder:
        def __getattr__(self, name):
            return getattr(real, name)

        def warning(self, msg, *args, **kwargs):
            seen.append(str(msg) % args if args else str(msg))

    monkeypatch.setattr(lc, "logger", _Recorder())
    return seen


def _fit_target(cmd):
    assert cmd.count("--fit-target") == 1, cmd
    return cmd[cmd.index("--fit-target") + 1]


class TestFieldLaunch:
    def test_the_fitter_keeps_what_other_programs_hold(self, tmp_path, blind_child, monkeypatch):
        warnings = _record_warnings(monkeypatch)
        backend, gguf = _field_backend(tmp_path)
        cmd = _launch(backend, gguf, n_ctx = 8192)
        assert cmd[cmd.index("--fit") + 1] == "on"
        # Unpinned, CUDA's own device order: one value, sized for the fuller card.
        assert _fit_target(cmd) == "14005"
        # The pre-load warning says how much, what was passed, and who holds it.
        (line,) = [w for w in warnings if "Other processes hold GPU memory" in w]
        assert "~12.7 GB on GPU 0, ~12.3 GB on GPU 1" in line
        assert "--fit-target 14005" in line
        assert "Bionic.exe (PID 1372)" in line

    def test_an_explicit_pick_gets_a_per_device_list(self, tmp_path, blind_child):
        backend, gguf = _field_backend(tmp_path, memory = [(0, 10571, 24576), (1, 24322, 24576)])
        cmd = _launch(backend, gguf, n_ctx = 8192, gpu_ids = [0, 1])
        assert cmd[cmd.index("--fit") + 1] == "on"
        assert _fit_target(cmd) == "14005,1024"

    def test_a_pass_through_margin_is_left_alone(self, tmp_path, blind_child):
        backend, gguf = _field_backend(tmp_path)
        cmd = _launch(backend, gguf, n_ctx = 8192, extra_args = ["--fit-target", "2048"])
        assert _fit_target(cmd) == "2048"

    def test_nothing_hidden_leaves_the_command_alone(self, tmp_path, blind_child):
        backend, gguf = _field_backend(
            tmp_path, memory = [(0, 24322, 24576), (1, 24322, 24576)], size_gb = 60
        )
        cmd = _launch(backend, gguf, n_ctx = 8192)
        assert cmd[cmd.index("--fit") + 1] == "on"
        assert "--fit-target" not in cmd

    def test_off_windows_nothing_changes(self, tmp_path, blind_child, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend,
            "_child_vram_probe_is_process_local",
            staticmethod(lambda binary, is_vulkan_backend: False),
        )
        backend, gguf = _field_backend(tmp_path)
        cmd = _launch(backend, gguf, n_ctx = 8192)
        assert "--fit-target" not in cmd


def test_retries_that_turn_the_fitter_on_are_covered():
    """The full-offload --fit on retry and the --flash-attn off respawn flip a pinned
    launch's fitter back on with no margin; the per-attempt hook is what covers them."""
    import inspect

    source = inspect.getsource(LlamaCppBackend.load_model)
    spawn = source[source.index("def _spawn_and_wait(") :]
    hook = spawn.index("run_cmd = _with_hidden_fit_margin(run_cmd)")
    assert hook < spawn.index("_last_spawn_cmd = list(run_cmd)") < spawn.index("subprocess.Popen(")
    # The budget record is read off the budget's own flags, before the hidden margin joins.
    assert source.index('_fit_target_priced = "--fit-target" in _integrity_flags') < source.index(
        "_integrity_flags = self._with_fit_target("
    )
