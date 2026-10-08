// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A model load that ran out of GPU memory names the fix, not just the runner's text. */

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import {
  GPU_OUT_OF_MEMORY_HINT,
  modelLoadErrorHint,
} from "../src/features/chat/utils/model-load-error-hint.ts";

test("the hint names the settings that free GPU memory", () => {
  assert.equal(
    GPU_OUT_OF_MEMORY_HINT,
    "Not enough GPU memory for these settings. If another app (another LLM app, a game) is using the GPU, close or unload it first. Otherwise try a smaller quant (e.g. Q4_K_M), a lower Context Length, or fewer GPU layers in the model's load settings.",
  );
});

test("a failure the backend already explained gets no second hint", () => {
  const explained = [
    "Not enough GPU memory to load this model: llama.cpp could not allocate a GPU buffer. Another program may be using GPU memory.",
    "",
    "llama-server output:",
    "  llama_model_load: error loading model: unable to allocate CUDA0 buffer",
  ].join("\n");
  assert.equal(modelLoadErrorHint(explained), null);
});

test("each GPU backend's allocation failure is recognised", () => {
  for (const text of [
    "llama_model_load: error loading model: unable to allocate CUDA0 buffer",
    "alloc_tensor_range: failed to allocate CUDA1 buffer of size 4294967296",
    "ggml_gallocr_reserve_n: failed to allocate ROCm0 buffer of size 123",
    "unable to allocate Vulkan0 buffer",
    "failed to allocate Metal buffer",
    "failed to allocate SYCL0 buffer of size 99",
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8192.00 MiB on device 0: cudaMalloc failed: out of memory",
    "CUDA error: out of memory",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "HIP out of memory",
    "vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
    "hipMalloc returned hipErrorOutOfMemory",
  ]) {
    assert.equal(modelLoadErrorHint(text), GPU_OUT_OF_MEMORY_HINT, text);
  }
});

test("the marker is found anywhere in a multi-line diagnostic, in any case", () => {
  const diagnostic = [
    "llama-server failed to start. Check that the GGUF file is valid and you have enough memory.",
    "",
    "llama-server output:",
    "  llama_model_load: error loading model: UNABLE TO ALLOCATE cuda0 BUFFER",
    "",
    "Full log: C:\\Users\\u\\.unsloth\\studio\\logs\\llama-server\\llama-1765000000-port-8080.log",
  ].join("\n");
  assert.equal(modelLoadErrorHint(diagnostic), GPU_OUT_OF_MEMORY_HINT);
});

test("failures that are not GPU memory get no hint", () => {
  for (const text of [
    "",
    "Failed to load model",
    "llama-server failed to start. Check that the GGUF file is valid and you have enough memory.",
    "llama_model_load: error loading model: unknown model architecture: 'foo'",
    // Pinned host memory is system RAM, not a device buffer.
    "unable to allocate CUDA_Host buffer",
    // The OS killing the runner for system RAM: fewer GPU layers would make this worse.
    "llama-server was stopped by the operating system (signal 9), most likely out of memory. Try a smaller or more quantized GGUF, lower the context length, or free memory (on WSL, raise the memory limit in .wslconfig).",
    "llama-server was stopped by macOS (signal 9) before it started. This is most often out of memory, so try a smaller or more quantized GGUF, or lower the context length.",
  ]) {
    assert.equal(modelLoadErrorHint(text), null, text);
  }
});

test("the load-failure toast and the header chip both carry the hint", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(
    runtime,
    /const hint = modelLoadErrorHint\(message\);\s*const detail = \[hint, rest\.join\("\\n"\)\.trim\(\)\]/,
    "the load-failure toast no longer leads its description with the hint",
  );
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /modelLoadErrorHint\(modelsError\)/);
  assert.match(
    page,
    /<CopyableErrorChip\s+message=\{\s*modelsErrorHint\s*\?/,
    "the header chip no longer shows the hint",
  );
  assert.match(
    page,
    /onClick=\{modelsErrorLogsAction\.onClick\}/,
    "the header chip no longer offers View logs",
  );
});
