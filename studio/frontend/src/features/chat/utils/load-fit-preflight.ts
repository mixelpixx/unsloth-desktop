// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoadVerdict } from "@/lib/load-verdict";
import { requestMemoryOvercommitConsent } from "../memory-overcommit-consent.ts";

/**
 * The load guardrail's question, asked before a model switch unloads the resident model.
 *
 * /load asks it too, ahead of its own eviction, but a switch unloads the outgoing model first, so
 * by the time /load refused there was nothing left to protect: Cancel then reloaded the model it
 * had just unloaded. The estimate route carries the same verdict, priced by the same code against
 * the same memory reading (resident buffers credited back), so it can be answered up front.
 *
 * Resolves true when the user chose "Load anyway" (the load then sends allow_memory_overcommit so
 * /load does not ask twice), false when nothing needed asking or the estimate could not say, and
 * throws the user-cancelled marker on Cancel. Fails open: /load still guards whatever this misses.
 */
export async function confirmFitBeforeUnload(
  estimate: () => Promise<{ verdict?: LoadVerdict | null }>,
  modelLabel: string,
  options: {
    signal?: AbortSignal;
    /** Told when the question opens and closes, so the load toast can stop claiming progress. */
    onAsking?: (asking: boolean) => void;
  } = {},
): Promise<boolean> {
  let verdict: LoadVerdict | null | undefined;
  try {
    verdict = (await estimate()).verdict;
  } catch {
    if (options.signal?.aborted) {
      throw options.signal.reason ?? new DOMException("Aborted", "AbortError");
    }
    return false;
  }
  if (!verdict?.needsConfirmation) return false;
  options.onAsking?.(true);
  let decision: Awaited<ReturnType<typeof requestMemoryOvercommitConsent>>;
  try {
    decision = await requestMemoryOvercommitConsent({ modelLabel, verdict }, options.signal);
  } finally {
    options.onAsking?.(false);
  }
  if (options.signal?.aborted) {
    throw options.signal.reason ?? new DOMException("Aborted", "AbortError");
  }
  if (decision !== "load") {
    // The marker every load caller already reads as "the user said no", not a failure.
    throw Object.assign(new Error("Model load cancelled."), { unslothUserCancelled: true });
  }
  return true;
}
