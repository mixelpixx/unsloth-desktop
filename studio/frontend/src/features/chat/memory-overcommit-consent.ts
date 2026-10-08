// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import type { LoadVerdict } from "@/lib/load-verdict";

// The "Load anyway" question for a load the backend's memory guardrail refused (409
// memory_overcommit). Asked from inside loadModel, so every surface that loads a model gets it
// without wiring its own dialog; answered by the one MemoryOvercommitDialog mounted at the app
// root. Same shape as auto-load-consent: a queue, a host count, and "cancel" whenever nobody
// can answer.

export type MemoryOvercommitDecision = "load" | "cancel";

export interface MemoryOvercommitRequest {
  /** What the user picked, as they would recognise it: "unsloth/Qwen3-27B-GGUF". */
  modelLabel: string;
  verdict: LoadVerdict;
}

type PendingOvercommit = MemoryOvercommitRequest & {
  id: number;
  resolve: (decision: MemoryOvercommitDecision) => void;
};

type MemoryOvercommitState = {
  /** First in line is on screen; Compare's two panes can each be refused while one dialog shows. */
  pending: PendingOvercommit[];
  /** Mounted dialogs. With none, asking would wait forever, so the load is declined. */
  hosts: number;
};

export const useMemoryOvercommitStore = create<MemoryOvercommitState>(() => ({
  pending: [],
  hosts: 0,
}));

let nextOvercommitId = 1;

/** Register a mounted dialog; returns its unregister. */
export function registerMemoryOvercommitHost(): () => void {
  useMemoryOvercommitStore.setState((state) => ({ hosts: state.hosts + 1 }));
  let registered = true;
  return () => {
    if (!registered) return;
    registered = false;
    const state = useMemoryOvercommitStore.getState();
    const hosts = Math.max(0, state.hosts - 1);
    useMemoryOvercommitStore.setState({ hosts });
    // The last dialog went away with questions open: nobody can answer them now. Settled outside
    // any updater, since each resolve removes its own entry with a setState of its own.
    if (hosts === 0) {
      for (const entry of [...state.pending]) entry.resolve("cancel");
    }
  };
}

/**
 * Ask whether to load past the guardrail. Resolves "cancel" at once with no dialog mounted, and
 * on abort, so a load torn down mid-question never leaves a dialog asking about it.
 */
export function requestMemoryOvercommitConsent(
  request: MemoryOvercommitRequest,
  signal?: AbortSignal,
): Promise<MemoryOvercommitDecision> {
  if (signal?.aborted || useMemoryOvercommitStore.getState().hosts === 0) {
    return Promise.resolve("cancel");
  }
  return new Promise<MemoryOvercommitDecision>((resolve) => {
    const id = nextOvercommitId++;
    let settled = false;
    const finish = (decision: MemoryOvercommitDecision): void => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener("abort", onAbort);
      useMemoryOvercommitStore.setState((state) => ({
        pending: state.pending.filter((entry) => entry.id !== id),
      }));
      resolve(decision);
    };
    const onAbort = (): void => finish("cancel");
    signal?.addEventListener("abort", onAbort, { once: true });
    useMemoryOvercommitStore.setState((state) => ({
      pending: [...state.pending, { ...request, id, resolve: finish }],
    }));
  });
}

/** Answer the request on screen. */
export function answerMemoryOvercommitConsent(
  id: number,
  decision: MemoryOvercommitDecision,
): void {
  const entry = useMemoryOvercommitStore
    .getState()
    .pending.find((candidate) => candidate.id === id);
  entry?.resolve(decision);
}
