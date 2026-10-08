// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const GPU_OUT_OF_MEMORY_HINT =
  "Not enough GPU memory for these settings. If another app (another LLM app, a game) is using the GPU, close or unload it first. Otherwise try a smaller quant (e.g. Q4_K_M), a lower Context Length, or fewer GPU layers in the model's load settings.";

// The backend's own wording for a classified GPU allocation failure, which already carries the same
// advice; repeating it in front of the message reads as two different problems.
const BACKEND_EXPLAINED = /^\s*Not enough GPU memory to load this model\b/i;

// Runner wording for a device allocation that did not fit: llama.cpp's "unable to allocate CUDA0
// buffer" / "failed to allocate ROCm0 buffer", a failed cudaMalloc, Vulkan's ErrorOutOfDeviceMemory
// and HIP's hipErrorOutOfMemory. A bare "out of memory" counts only beside a GPU backend on the same
// line: the backend's own signal-9 message ("most likely out of memory") is the OS killing the
// process for system RAM, where fewer GPU layers would make it worse. "CUDA_Host" (pinned host
// memory) is not a device buffer, so it is left out too.
const GPU_OUT_OF_MEMORY =
  /(unable|failed) to allocate (CUDA|ROCm|Vulkan|Metal|SYCL)\d* buffer|cudaMalloc failed|\b(CUDA|ROCm|HIP|Vulkan|Metal|SYCL|GPU)\b[^\n]*\bout of memory|ErrorOutOfDeviceMemory|hipErrorOutOfMemory/i;

/** A plain-language next step for a model-load failure the raw runner text does not explain. */
export function modelLoadErrorHint(text: string): string | null {
  if (!text || BACKEND_EXPLAINED.test(text)) return null;
  return GPU_OUT_OF_MEMORY.test(text) ? GPU_OUT_OF_MEMORY_HINT : null;
}
