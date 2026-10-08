// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** Why the send picked this model, which decides how the dialog explains it. */
export type AutoLoadConsentReason = "last-used" | "smallest" | "default-download";

export type AutoLoadConsentDecision = "load" | "choose" | "cancel";

export interface AutoLoadConsentRequest {
  reason: AutoLoadConsentReason;
  /** "unsloth/Qwen3-27B-GGUF (Q4_K_M)". */
  modelLabel: string;
  /** On-disk size, or the download size for a starter model; 0 when unknown. */
  sizeBytes: number;
}

type PendingConsent = AutoLoadConsentRequest & {
  id: number;
  resolve: (decision: AutoLoadConsentDecision) => void;
};

type AutoLoadConsentState = {
  /** First in line is on screen; Compare's two panes can each ask while one dialog shows. */
  pending: PendingConsent[];
  /** Mounted dialogs. With none, asking would wait forever, so the request is declined. */
  hosts: number;
};

export const useAutoLoadConsentStore = create<AutoLoadConsentState>(() => ({
  pending: [],
  hosts: 0,
}));

let nextConsentId = 1;

export const AUTO_LOAD_LAST_MODEL_KEY = "unsloth_chat_auto_load_last_model";

/** Opt-in, and only for the model the user loaded last: anything else is a model they never picked. */
export function autoLoadsLastModelWithoutAsking(): boolean {
  try {
    return localStorage.getItem(AUTO_LOAD_LAST_MODEL_KEY) === "1";
  } catch {
    return false;
  }
}

export function setAutoLoadsLastModelWithoutAsking(enabled: boolean): void {
  try {
    if (enabled) localStorage.setItem(AUTO_LOAD_LAST_MODEL_KEY, "1");
    else localStorage.removeItem(AUTO_LOAD_LAST_MODEL_KEY);
  } catch {
    // Storage blocked: the dialog simply asks again next time.
  }
}

/** Register a mounted dialog; returns its unregister. */
export function registerAutoLoadConsentHost(): () => void {
  useAutoLoadConsentStore.setState((state) => ({ hosts: state.hosts + 1 }));
  let registered = true;
  return () => {
    if (!registered) return;
    registered = false;
    const state = useAutoLoadConsentStore.getState();
    const hosts = Math.max(0, state.hosts - 1);
    useAutoLoadConsentStore.setState({ hosts });
    // The last dialog went away with questions still open: nobody can answer them now. Settled
    // outside any updater, since each resolve removes its own entry with a setState of its own.
    if (hosts === 0) {
      for (const entry of [...state.pending]) entry.resolve("cancel");
    }
  };
}

/**
 * Ask before a send loads (or downloads) a model the user did not pick for it.
 *
 * Sending with nothing loaded used to load the smallest model on disk silently, which on a
 * machine holding one 27B model meant 22 GB of VRAM for a "hello". An abort (Stop, leaving the
 * chat) answers "cancel"; so does having no dialog mounted to ask with.
 */
export function requestAutoLoadConsent(
  request: AutoLoadConsentRequest,
  signal?: AbortSignal,
): Promise<AutoLoadConsentDecision> {
  if (request.reason === "last-used" && autoLoadsLastModelWithoutAsking()) {
    return Promise.resolve("load");
  }
  if (signal?.aborted || useAutoLoadConsentStore.getState().hosts === 0) {
    return Promise.resolve("cancel");
  }
  return new Promise<AutoLoadConsentDecision>((resolve) => {
    const id = nextConsentId++;
    let settled = false;
    const finish = (decision: AutoLoadConsentDecision): void => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener("abort", onAbort);
      useAutoLoadConsentStore.setState((state) => ({
        pending: state.pending.filter((entry) => entry.id !== id),
      }));
      resolve(decision);
    };
    const onAbort = (): void => finish("cancel");
    signal?.addEventListener("abort", onAbort, { once: true });
    useAutoLoadConsentStore.setState((state) => ({
      pending: [...state.pending, { ...request, id, resolve: finish }],
    }));
  });
}

/** Answer the request on screen. */
export function answerAutoLoadConsent(
  id: number,
  decision: AutoLoadConsentDecision,
): void {
  const entry = useAutoLoadConsentStore
    .getState()
    .pending.find((candidate) => candidate.id === id);
  entry?.resolve(decision);
}

export function formatAutoLoadSize(bytes: number): string | null {
  if (!(bytes > 0)) return null;
  const gb = bytes / 1000 ** 3;
  return gb >= 1
    ? `${gb.toFixed(1)} GB`
    : `${Math.max(1, Math.round(bytes / 1000 ** 2))} MB`;
}
