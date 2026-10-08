# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /load tensor -> layer fallback retries only a launch that actually ran tensor.

Field log: tensor parallel was requested, the planner downgraded it to layer split
("the pooled VRAM budget cannot hold ..."), the layer launch ran out of GPU memory, and
the fallback relaunched the same command plus ``--split-mode layer`` -- the default it
already had -- under "this model may not support tensor parallelism". Twice the wait,
the same failure, and a false diagnosis.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

for _name, _attrs in (
    ("loggers", {"get_logger": lambda name: __import__("logging").getLogger(name)}),
    ("structlog", {"get_logger": lambda *a, **k: __import__("logging").getLogger("stub")}),
):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            _mod = types.ModuleType(_name)
            for _k, _v in _attrs.items():
                setattr(_mod, _k, _v)
            sys.modules[_name] = _mod

from core.inference.llama_cpp import LlamaCppBackend
from core.inference.tensor_fallback import load_with_tensor_fallback

_OOM = RuntimeError(
    "Not enough GPU memory to load this model: llama.cpp could not allocate a GPU buffer.\n\n"
    "llama-server output:\n"
    "llama_model_load: error loading model: unable to allocate CUDA0 buffer"
)


class _Loader:
    def __init__(self, outcome):
        self.calls: list[tuple] = []
        self._outcome = outcome

    async def __call__(self, tensor_parallel, extra_args):
        self.calls.append((tensor_parallel, list(extra_args) if extra_args else extra_args))
        outcome = self._outcome(len(self.calls))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _run(loader, **kwargs):
    kwargs.setdefault("requested_tensor", True)
    kwargs.setdefault("extra_args", None)
    kwargs.setdefault("label", "m")
    return asyncio.run(load_with_tensor_fallback(loader, **kwargs))


class TestEngagement:
    def test_a_downgraded_request_is_not_retried_and_keeps_its_error(self):
        loader = _Loader(lambda n: _OOM)
        with pytest.raises(RuntimeError) as err:
            _run(loader, tensor_engaged = lambda: False)
        # The first failure, with its own diagnosis, not a second identical launch.
        assert err.value is _OOM
        assert len(loader.calls) == 1

    def test_a_downgraded_request_returning_false_is_not_retried(self):
        loader = _Loader(lambda n: False)
        assert _run(loader, tensor_engaged = lambda: False) is False
        assert len(loader.calls) == 1

    def test_a_tensor_launch_still_falls_back(self):
        loader = _Loader(lambda n: RuntimeError("GGML_ASSERT tensor split") if n == 1 else True)
        assert _run(loader, tensor_engaged = lambda: True) is True
        assert loader.calls == [(True, None), (False, ["--split-mode", "layer"])]

    def test_a_tensor_launch_that_ran_out_of_gpu_memory_still_falls_back(self):
        # Layer split is a different placement, and it hands llama.cpp's fitter the
        # option of moving layers to system RAM, which tensor mode has no way to do.
        loader = _Loader(lambda n: _OOM if n == 1 else True)
        assert _run(loader, tensor_engaged = lambda: True, is_gpu_memory_failure = lambda e: True)
        assert len(loader.calls) == 2

    def test_unknown_engagement_keeps_the_old_retry(self):
        loader = _Loader(lambda n: RuntimeError("crash") if n == 1 else True)
        assert _run(loader, tensor_engaged = lambda: None) is True
        assert len(loader.calls) == 2

    def test_unknown_engagement_and_out_of_gpu_memory_is_not_retried(self):
        loader = _Loader(lambda n: _OOM)
        with pytest.raises(RuntimeError) as err:
            _run(
                loader,
                tensor_engaged = lambda: None,
                is_gpu_memory_failure = lambda e: (
                    LlamaCppBackend._is_gpu_memory_start_failure(str(e))
                ),
            )
        assert err.value is _OOM
        assert len(loader.calls) == 1

    def test_a_probe_that_raises_reads_as_unknown(self):
        def _boom():
            raise AttributeError("no backend")

        loader = _Loader(lambda n: RuntimeError("crash") if n == 1 else True)
        assert _run(loader, tensor_engaged = _boom) is True
        assert len(loader.calls) == 2

    def test_a_cancel_still_wins(self):
        loader = _Loader(lambda n: False)
        assert _run(loader, tensor_engaged = lambda: True, cancelled = lambda: True) is False
        assert len(loader.calls) == 1

    def test_without_the_new_callables_nothing_changes(self):
        loader = _Loader(lambda n: _OOM if n == 1 else True)
        assert _run(loader) is True
        assert len(loader.calls) == 2


class TestLaunchedSplitModeRecord:
    def _backend(self):
        return LlamaCppBackend.__new__(LlamaCppBackend)

    def test_a_layer_launch(self):
        b = self._backend()
        b._launched_tensor_parallel = None
        b._note_launched_split_mode(["llama-server", "-m", "x.gguf", "--fit", "on"], {})
        assert b._launched_tensor_parallel is False

    def test_a_tensor_launch(self):
        b = self._backend()
        b._launched_tensor_parallel = None
        b._note_launched_split_mode(["llama-server", "--split-mode", "tensor"], {})
        assert b._launched_tensor_parallel is True

    def test_last_wins_on_the_argv(self):
        b = self._backend()
        b._launched_tensor_parallel = None
        b._note_launched_split_mode(["s", "--split-mode", "tensor", "-sm", "layer"], {})
        assert b._launched_tensor_parallel is False

    def test_the_env_counts_when_the_argv_is_silent(self):
        b = self._backend()
        b._launched_tensor_parallel = None
        b._note_launched_split_mode(["s"], {"LLAMA_ARG_SPLIT_MODE": "tensor"})
        assert b._launched_tensor_parallel is True

    def test_sticky_across_the_loads_spawns(self):
        # Tensor first, then an internal layer retry: the load DID try tensor.
        b = self._backend()
        b._launched_tensor_parallel = None
        b._note_launched_split_mode(["s", "--split-mode", "tensor"], {})
        b._note_launched_split_mode(["s", "--split-mode", "layer"], {})
        assert b._launched_tensor_parallel is True

    def test_a_double_built_without_init(self):
        b = self._backend()
        b._note_launched_split_mode(["s"], {})
        assert b._launched_tensor_parallel is False


def test_every_spawn_records_and_each_load_starts_clean():
    start = inspect.getsource(LlamaCppBackend._start_llama_process)
    note = start.index("self._note_launched_split_mode(cmd, env)")
    assert note < start.index("subprocess.Popen(")
    load = inspect.getsource(LlamaCppBackend.load_model)
    spawn = load[load.index("def _spawn_and_wait(") :]
    assert spawn.index("self._note_launched_split_mode(run_cmd, env)") < spawn.index(
        "subprocess.Popen("
    )
    assert "self._launched_tensor_parallel = None" in load


def test_the_route_passes_what_was_launched():
    import routes.inference as routes

    source = inspect.getsource(routes)
    call = source[source.index("success = await load_with_tensor_fallback(") :]
    call = call[: call.index("except Exception:")]
    assert '"_launched_tensor_parallel"' in call
    assert "LlamaCppBackend._is_gpu_memory_start_failure(" in call
    assert "is_gpu_memory_failure = " in call
