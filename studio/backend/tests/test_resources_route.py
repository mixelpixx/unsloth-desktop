# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``GET /api/resources``: the sidebar strip's one snapshot, read from stubbed readings.

Pinned here: the payload shape, the Studio-versus-other split on the per-process counter, the
estimate when there is no counter, the two caches (one nvidia-smi read a second, the ~1 s
holder read every 15 s and never on the request path), and that a host with no GPU answers an
empty list rather than failing.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.resources as resources
from auth.authentication import authenticated_via_api_key, get_current_subject
from utils.hardware import gpu_resources as gr

GIB = 1024**3
MIB = 1024**2

# Two 24 GiB cards. GPU 0 holds Studio's model, GPU 1 another LLM app.
ROWS = [(0, 3 * 1024, 24 * 1024), (1, int(9.5 * 1024), 24 * 1024)]
NAMES = {0: "NVIDIA GeForce RTX 3090", 1: "NVIDIA GeForce RTX 3090"}
LUID_A, LUID_B = 0x1111, 0x2222
BACKEND_PID, LLAMA_PID, LM_STUDIO_PID, BROWSER_PID = 101, 202, 1372, 4040
SAMPLES = [
    (LLAMA_PID, LUID_A, int(20.0 * GIB)),
    (BACKEND_PID, LUID_A, int(0.5 * GIB)),
    (LM_STUDIO_PID, LUID_B, int(13.5 * GIB)),
    # Under the listing floor: a hardware-accelerated window, not worth naming.
    (BROWSER_PID, LUID_B, 100 * MIB),
]
STUDIO = frozenset({BACKEND_PID, LLAMA_PID})


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Calls:
    """Counts each stubbed reading, so the caches can be pinned by how often they ask."""

    def __init__(self, rows = ROWS, samples = SAMPLES) -> None:
        self.rows = rows
        self.samples = samples
        self.memory = 0
        self.fresh: list[bool] = []
        self.names = 0
        self.holders = 0
        self.process_names: list[int] = []
        self.generation = 0

    def memory_reader(self, *, fresh: bool = False):
        self.memory += 1
        self.fresh.append(fresh)
        return None if self.rows is None else list(self.rows)

    def name_reader(self):
        self.names += 1
        return dict(NAMES)

    def holder_reader(self):
        self.holders += 1
        return None if self.samples is None else list(self.samples)

    def process_name(self, pid: int) -> str:
        self.process_names.append(pid)
        return {LM_STUDIO_PID: "LM Studio.exe"}.get(pid, f"proc{pid}.exe")


def make_reader(calls: Calls, clock: Clock, *, supported: bool = True, spawn = None):
    return gr.ResourceReader(
        memory_reader = calls.memory_reader,
        name_reader = calls.name_reader,
        holder_reader = calls.holder_reader,
        studio_pids = lambda: STUDIO,
        process_name = calls.process_name,
        generation = lambda: calls.generation,
        # Synchronous, so a refresh lands before the assertion that reads it.
        spawn = spawn or (lambda target: target()),
        clock = clock,
        attribution_supported = supported,
    )


def _primed(calls: Calls, clock: Clock, **kwargs):
    reader = make_reader(calls, clock, **kwargs)
    reader.read_gpus()
    reader.holders()
    return reader


def _snapshot(reader, models):
    return resources.build_snapshot(
        reader = reader, models = models, loading = False, visible = None
    )


# ── Shape ─────────────────────────────────────────────────────────


def test_snapshot_shape_lists_every_card_and_model():
    calls, clock = Calls(), Clock()
    reader = _primed(calls, clock)
    llama = resources.ResidentModel(
        kind = "chat",
        source = "chat",
        name = "unsloth/Qwen3-27B-GGUF",
        variant = "Q4_K_M",
        gpu_ids = [0],
        layers_on_gpu = 66,
        layers_total = 66,
        context_length = 8192,
        cache_type_kv = "q8_0",
        vram_by_gpu = {0: int(20.7 * GIB)},
    )
    snap = _snapshot(reader, [llama])

    assert set(snap) == {"gpus", "models", "other_apps", "loading"}
    assert [g["index"] for g in snap["gpus"]] == [0, 1]
    gpu = snap["gpus"][0]
    assert set(gpu) == {
        "index",
        "name",
        "total_bytes",
        "used_bytes",
        "free_bytes",
        "studio_bytes",
        "other_bytes",
        "attribution",
        "apps",
    }
    assert gpu["name"] == "NVIDIA GeForce RTX 3090"
    assert gpu["total_bytes"] == 24 * GIB
    assert gpu["free_bytes"] == 3 * GIB
    assert gpu["used_bytes"] == 21 * GIB
    # Studio + other + free is the whole card: the bar's three segments add up.
    assert gpu["studio_bytes"] + gpu["other_bytes"] + gpu["free_bytes"] == gpu["total_bytes"]

    (model,) = snap["models"]
    assert model == {
        "id": "chat:unsloth/Qwen3-27B-GGUF",
        "kind": "chat",
        "source": "chat",
        "name": "unsloth/Qwen3-27B-GGUF",
        "variant": "Q4_K_M",
        "gpu_ids": [0],
        "device": None,
        "layers_on_gpu": 66,
        "layers_total": 66,
        "context_length": 8192,
        "cache_type_kv": "q8_0",
        "vram_bytes": int(20.7 * GIB),
        "vram_approx": False,
        "loading": False,
        "inactive": False,
        "stt_engine": None,
    }
    assert snap["loading"] is False


def test_route_is_authenticated_and_serves_the_snapshot(monkeypatch):
    route = next(r for r in resources.router.routes if getattr(r, "path", None) == "")
    calls_made = {dep.call for dep in route.dependant.dependencies}
    assert get_current_subject in calls_made
    assert authenticated_via_api_key in calls_made

    empty = {"gpus": [], "models": [], "other_apps": [], "loading": False}
    monkeypatch.setattr(resources, "build_snapshot", lambda: dict(empty))
    app = FastAPI()
    app.include_router(resources.router, prefix = "/api/resources")
    app.dependency_overrides = {
        get_current_subject: lambda: "owner",
        authenticated_via_api_key: lambda: False,
    }
    response = TestClient(app).get("/api/resources")
    assert response.status_code == 200
    assert response.json() == empty


def test_main_mounts_the_router_under_api_resources():
    from pathlib import Path

    main_py = Path(resources.__file__).resolve().parents[1] / "main.py"
    source = main_py.read_text(encoding = "utf-8")
    assert "from routes.resources import router as resources_router" in source
    assert 'app.include_router(resources_router, prefix = "/api/resources"' in source


# ── Studio versus other programs ──────────────────────────────────


def test_counter_splits_studio_from_other_apps_per_card():
    calls, clock = Calls(), Clock()
    reader = _primed(calls, clock)
    gpus, unplaced = gr.split_gpu_memory(reader.read_gpus(), reader.holders())
    gpu0, gpu1 = gpus

    # This process and its llama-server child are Studio; the rest of the card's use is not.
    assert gpu0["attribution"] == "process"
    assert gpu0["studio_bytes"] == int(20.5 * GIB)
    assert gpu0["other_bytes"] == int(0.5 * GIB)
    assert gpu0["apps"] == []

    assert gpu1["attribution"] == "process"
    assert gpu1["studio_bytes"] == 0
    assert gpu1["other_bytes"] == int(14.5 * GIB)
    # Named, biggest first, the small window left out.
    assert gpu1["apps"] == [
        {"pid": LM_STUDIO_PID, "name": "LM Studio.exe", "bytes": int(13.5 * GIB)}
    ]
    assert unplaced == []


def test_unmatched_adapters_list_holders_without_a_card():
    # The counter disagrees with nvidia-smi by far more than the slack: no pairing is trusted.
    samples = [(LLAMA_PID, LUID_A, 2 * GIB), (LM_STUDIO_PID, LUID_B, 2 * GIB)]
    calls, clock = Calls(samples = samples), Clock()
    reader = _primed(calls, clock)
    gpus, unplaced = gr.split_gpu_memory(
        reader.read_gpus(), reader.holders(), estimate_by_gpu = {0: int(20 * GIB)}
    )
    assert [g["attribution"] for g in gpus] == ["estimate", "estimate"]
    assert gpus[0]["studio_bytes"] == 20 * GIB
    assert all(g["apps"] == [] for g in gpus)
    assert unplaced == [{"pid": LM_STUDIO_PID, "name": "LM Studio.exe", "bytes": 2 * GIB}]


def _unpairable(calls: Calls, clock: Clock, reader) -> None:
    """A read taken mid-load: the counter and nvidia-smi disagree by far more than the slack."""
    calls.rows = ROWS
    calls.samples = [(LLAMA_PID, LUID_A, 2 * GIB), (LM_STUDIO_PID, LUID_B, 2 * GIB)]
    clock.now += gr.ATTRIBUTION_TTL_S
    reader.read_gpus()
    reader.holders()


def test_an_unpairable_read_falls_back_to_the_last_unambiguous_pairing():
    calls, clock = Calls(), Clock()
    reader = _primed(calls, clock)  # 21 GiB against 14.5 GiB in use: no way to swap them
    _unpairable(calls, clock, reader)
    gpus, _ = gr.split_gpu_memory(reader.read_gpus(), reader.holders())
    assert [g["attribution"] for g in gpus] == ["process", "process"]
    assert gpus[0]["studio_bytes"] == 2 * GIB


def test_a_pairing_learned_on_two_identical_idle_cards_is_never_reused():
    # Two idle 3090s read the same few hundred MiB: the pairing then is a coin toss.
    idle_rows = [(0, 24 * 1024 - 256, 24 * 1024), (1, 24 * 1024 - 256, 24 * 1024)]
    idle_samples = [(BACKEND_PID, LUID_A, 200 * MIB), (BROWSER_PID, LUID_B, 200 * MIB)]
    calls, clock = Calls(rows = idle_rows, samples = idle_samples), Clock()
    reader = _primed(calls, clock)
    assert reader.holders().luid_to_index, "paired for this read"
    _unpairable(calls, clock, reader)
    gpus, unplaced = gr.split_gpu_memory(reader.read_gpus(), reader.holders())
    assert [g["attribution"] for g in gpus] == ["estimate", "estimate"]
    assert [app["name"] for app in unplaced] == ["LM Studio.exe"]


def test_without_a_counter_studio_is_what_its_runtimes_logged():
    # Linux, or a non-English Windows whose counter reads nothing: never "Studio holds nothing".
    for samples in (None, []):
        calls, clock = Calls(samples = samples), Clock()
        reader = _primed(calls, clock)
        llama = resources.ResidentModel(
            kind = "chat", source = "chat", name = "m", gpu_ids = [0], vram_by_gpu = {0: 18 * GIB}
        )
        snap = _snapshot(reader, [llama])
        gpu0, gpu1 = snap["gpus"]
        assert gpu0["attribution"] == "estimate"
        assert gpu0["studio_bytes"] == 18 * GIB
        assert gpu0["other_bytes"] == 3 * GIB
        assert gpu1["studio_bytes"] == 0


def test_a_card_holding_an_unsized_model_has_no_split():
    calls, clock = Calls(samples = None), Clock()
    reader = _primed(calls, clock)
    transformers = resources.ResidentModel(
        kind = "chat", source = "chat", name = "t", gpu_ids = [1]
    )
    snap = _snapshot(reader, [transformers])
    gpu1 = snap["gpus"][1]
    assert gpu1["studio_bytes"] is None
    assert gpu1["other_bytes"] is None
    assert gpu1["attribution"] is None
    assert gpu1["used_bytes"] == int(14.5 * GIB)


def test_studio_share_never_exceeds_what_the_card_uses():
    # A holder read from before an unload can name more than nvidia-smi now sees in use.
    samples = [(LLAMA_PID, LUID_A, 23 * GIB)]
    rows = [(0, 3 * 1024, 24 * 1024)]
    calls, clock = Calls(rows = rows, samples = samples), Clock()
    reader = _primed(calls, clock)
    (gpu,), _ = gr.split_gpu_memory(reader.read_gpus(), reader.holders())
    assert gpu["studio_bytes"] == 21 * GIB
    assert gpu["other_bytes"] == 0


# ── Caches ────────────────────────────────────────────────────────


def test_nvidia_smi_is_read_at_most_once_a_second():
    calls, clock = Calls(), Clock()
    reader = make_reader(calls, clock)
    for _ in range(5):
        assert len(reader.read_gpus()) == 2
    assert calls.memory == 1
    clock.now += 0.5
    reader.read_gpus(fresh = True)
    assert calls.memory == 1, "a fresh ask inside the second still shares the reading"
    clock.now += 0.6
    reader.read_gpus(fresh = True)
    assert calls.memory == 2
    assert calls.fresh == [False, True]


def test_concurrent_pollers_share_one_reading():
    calls, clock = Calls(), Clock()
    gate = threading.Event()

    def slow_reader(*, fresh: bool = False):
        gate.wait(5)
        return calls.memory_reader(fresh = fresh)

    reader = make_reader(calls, clock)
    reader._memory_reader = slow_reader
    threads = [threading.Thread(target = reader.read_gpus) for _ in range(4)]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join(5)
    assert calls.memory == 1


def test_holder_names_refresh_every_15_seconds_not_per_request():
    calls, clock = Calls(), Clock()
    reader = _primed(calls, clock)
    assert calls.holders == 1
    assert calls.process_names == [LM_STUDIO_PID]

    for step in range(14):
        clock.now += 1.0
        reader.read_gpus()
        reader.holders()
    assert calls.holders == 1, "the ~1 s counter read must not run per poll"

    clock.now += 1.0
    reader.holders()
    assert calls.holders == 2
    # Same pid, same process: its name is not looked up again.
    assert calls.process_names == [LM_STUDIO_PID]


def test_a_load_or_unload_refreshes_holders_sooner_but_not_in_a_burst():
    calls, clock = Calls(), Clock()
    reader = _primed(calls, clock)
    calls.generation += 1
    clock.now += 1.0
    reader.holders()
    assert calls.holders == 1, "under the 2 s floor"
    clock.now += 1.0
    reader.holders()
    assert calls.holders == 2
    clock.now += 1.0
    reader.holders()
    assert calls.holders == 2, "the generation was consumed by that refresh"


def test_the_request_never_waits_on_the_holder_read():
    calls, clock = Calls(), Clock()
    started: list = []
    reader = make_reader(calls, clock, spawn = started.append)
    reader.read_gpus()
    assert reader.holders() is None, "no answer yet, and none waited for"
    assert len(started) == 1
    assert reader.holders() is None
    assert len(started) == 1, "one refresh in flight at a time"
    started[0]()
    assert reader.holders() is not None


def test_a_failed_holder_read_is_remembered_not_retried_every_poll():
    calls, clock = Calls(), Clock()

    def broken():
        calls.holders += 1
        raise RuntimeError("Get-Counter blew up")

    reader = make_reader(calls, clock)
    reader._holder_reader = broken
    reader.read_gpus()
    held = reader.holders()
    assert held is not None and held.samples is None
    clock.now += 5
    reader.holders()
    assert calls.holders == 1


# ── No GPU ────────────────────────────────────────────────────────


def test_a_host_without_nvidia_smi_answers_an_empty_card_list():
    calls, clock = Calls(rows = None, samples = None), Clock()
    reader = make_reader(calls, clock, supported = False)
    model = resources.ResidentModel(kind = "chat", source = "chat", name = "mlx-community/Qwen3-4B")
    snap = _snapshot(reader, [model])
    assert snap["gpus"] == []
    assert snap["other_apps"] == []
    assert [m["name"] for m in snap["models"]] == ["mlx-community/Qwen3-4B"]
    # Nor asked again on the next poll: no nvidia-smi is not a transient failure.
    clock.now += 30
    reader.read_gpus()
    assert calls.memory == 1
    assert calls.holders == 0, "no counter off Windows"


def test_an_unanswered_read_is_an_empty_list_and_retried():
    calls, clock = Calls(rows = []), Clock()
    reader = make_reader(calls, clock)
    assert reader.read_gpus() == []
    clock.now += 1.1
    reader.read_gpus()
    assert calls.memory == 2


def test_visible_mask_hides_cards_studio_cannot_use():
    calls, clock = Calls(), Clock()
    reader = make_reader(calls, clock)
    assert [g.index for g in reader.read_gpus(visible = {1})] == [1]


def test_parse_memory_rows_skips_unreadable_lines():
    stdout = "1, 9000, 24576\n0, 3000, 24576\ngarbage\n2, 100, [N/A]\n"
    assert gr.parse_memory_rows(stdout) == [(0, 3000, 24576), (1, 9000, 24576)]


# ── Model rows ────────────────────────────────────────────────────


class _Proc:
    def poll(self):
        return None


def _llama(**overrides):
    base = dict(
        _process = _Proc(),
        _child_gpu_physical_ids = [0, 1],
        _stdout_lines = [
            "load_tensors: CUDA0 model buffer size = 16000.00 MiB",
            "llama_kv_cache: CUDA0 KV buffer size = 2048.00 MiB",
            "llama_context: CUDA0 compute buffer size = 512.00 MiB",
            "load_tensors: CUDA_Host model buffer size = 400.00 MiB",
        ],
        _planned_vram_mib = {0: 19000},
        _gpu_offload_active = True,
        offloaded_layers = 66,
        offload_total_layers = 66,
        context_length = 8192,
        cache_type_kv = "q8_0",
        hf_variant = "Q4_K_M",
        gpu_ids = None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_llama_row_reports_logged_buffers_as_measured():
    row = resources.llama_model(_llama(), "unsloth/Qwen3-27B-GGUF").to_json()
    assert row["gpu_ids"] == [0], "only the card the child logged buffers on"
    assert row["vram_bytes"] == (16000 + 2048 + 512) * MIB
    assert row["vram_approx"] is False
    assert (row["layers_on_gpu"], row["layers_total"]) == (66, 66)
    assert row["context_length"] == 8192
    assert row["variant"] == "Q4_K_M"


def test_llama_row_falls_back_to_the_plan_and_says_approx():
    row = resources.llama_model(_llama(_stdout_lines = []), "m").to_json()
    assert row["vram_bytes"] == 19000 * MIB
    assert row["vram_approx"] is True
    assert row["gpu_ids"] == [0, 1]


def test_llama_row_on_the_cpu_holds_no_card():
    row = resources.llama_model(
        _llama(_stdout_lines = [], _planned_vram_mib = {}, offloaded_layers = 0), "m"
    ).to_json()
    assert row["gpu_ids"] == []
    assert row["vram_bytes"] is None
    assert row["vram_approx"] is False


def test_orchestrator_rows_split_speech_dictation_and_cached_models():
    backend = SimpleNamespace(
        active_model_name = "unsloth/orpheus-3b",
        models = {
            "unsloth/orpheus-3b": {"is_audio": True, "audio_type": "snac", "gpu_ids": [1]},
            "openai/whisper-large-v3": {"is_audio": True, "audio_type": "whisper"},
            "unsloth/Qwen3-4B": {"context_length": 4096},
        },
    )
    rows = {r.name: r for r in resources._orchestrator_models(backend)}
    assert rows["unsloth/orpheus-3b"].kind == "audio"
    assert rows["unsloth/orpheus-3b"].gpu_ids == [1]
    assert rows["unsloth/orpheus-3b"].inactive is False
    assert rows["openai/whisper-large-v3"].kind == "stt"
    assert rows["unsloth/Qwen3-4B"].kind == "chat"
    assert rows["unsloth/Qwen3-4B"].inactive is True


def test_one_runtime_failing_does_not_blank_the_others():
    def broken():
        raise RuntimeError("video status exploded")

    def chat():
        return [resources.ResidentModel(kind = "chat", source = "chat", name = "a")]

    def image():
        return [resources.ResidentModel(kind = "image", source = "image", name = "b")]

    # Out of order on purpose: rows come back in the fixed kind order.
    rows = resources.collect_models((image, broken, chat, chat))
    assert [(r.kind, r.name) for r in rows] == [("chat", "a"), ("image", "b")]


def test_media_and_dictation_are_read_without_building_a_runtime(monkeypatch):
    import sys

    for name in (
        "core.inference.diffusion",
        "core.inference.sd_cpp_backend",
        "core.inference.video",
        "core.inference.stt_sidecar",
        "core.inference.stt_mtmd_sidecar",
        "core.inference.stt_ggml_sidecar",
        "core.inference.stt_audiocpp_sidecar",
        "core.rag.embeddings",
    ):
        monkeypatch.delitem(sys.modules, name, raising = False)
    assert resources.media_models() == []
    assert resources.stt_models() == []
    assert resources.embedding_models() == []
    assert "core.inference.video" not in sys.modules


def test_dictation_rows_name_their_engine(monkeypatch):
    import sys
    import types

    module = types.ModuleType("core.inference.stt_ggml_sidecar")
    module._sidecar = SimpleNamespace(
        loaded_model = "ggml-large-v3-turbo", is_loading = lambda: False, device = "whisper.cpp"
    )
    monkeypatch.setitem(sys.modules, "core.inference.stt_ggml_sidecar", module)
    for name in (
        "core.inference.stt_sidecar",
        "core.inference.stt_mtmd_sidecar",
        "core.inference.stt_audiocpp_sidecar",
    ):
        monkeypatch.delitem(sys.modules, name, raising = False)
    (row,) = [r.to_json() for r in resources.stt_models()]
    assert row["id"] == "stt:gguf"
    assert row["stt_engine"] == "gguf"
    assert row["name"] == "ggml-large-v3-turbo"


def test_cuda_device_strings_map_through_the_visibility_mask(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    assert resources._device_gpu_ids("cuda:1") == [1]
    assert resources._device_gpu_ids("cuda") == []
    assert resources._device_gpu_ids("cpu") == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert resources._device_gpu_ids("cuda:0") == [1]
    assert resources._device_gpu_ids("cuda:5") == []


def test_real_collectors_read_an_idle_backend_without_failing(monkeypatch):
    # The collectors reach into routes.inference by name: a renamed private there must fail
    # here, not turn into a silent empty list on the strip.
    failures: list = []
    monkeypatch.setattr(resources.logger, "debug", lambda *args, **kwargs: failures.append(args))
    rows = resources.chat_models()
    assert rows == []
    assert resources.collect_models() == []
    assert failures == []


def test_loading_reads_the_cards_fresh():
    calls, clock = Calls(), Clock()
    reader = make_reader(calls, clock)
    pending = resources.ResidentModel(kind = "chat", source = "chat", name = "m", loading = True)
    snap = resources.build_snapshot(reader = reader, models = [pending], visible = None)
    assert snap["loading"] is True
    assert calls.fresh == [True]


def test_a_managed_account_sees_other_programs_as_a_total_not_by_name(monkeypatch):
    from hub.services.models import account_access
    from routes import resources as route

    snapshot = {
        "gpus": [
            {
                "index": 0,
                "other_bytes": 13 * 1024**3,
                "apps": [{"pid": 1372, "name": "Bionic.exe", "bytes": 13 * 1024**3}],
            }
        ],
        "other_apps": [{"pid": 77, "name": "game.exe", "bytes": 2 * 1024**3}],
        "models": [],
        "loading": False,
    }
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    hidden = route._without_host_process_names(json.loads(json.dumps(snapshot)))
    assert hidden["gpus"][0]["apps"] == [] and hidden["other_apps"] == []
    assert hidden["gpus"][0]["other_bytes"] == 13 * 1024**3

    monkeypatch.setattr(account_access, "managed_account", lambda: False)
    assert route._without_host_process_names(json.loads(json.dumps(snapshot))) == snapshot
