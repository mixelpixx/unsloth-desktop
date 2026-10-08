// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Cadence for the chat page's mount observer, which watches /status for a load started outside
 * this tab (the CLI, the API, another tab). A flat 500 ms kept an idle chat page issuing two
 * /status requests a second for its whole idle window, on a connection budget every tab shares.
 * It now starts quick, when a load started elsewhere is most likely to be in flight, and backs
 * off; once a load is seen a steady second keeps its completion prompt.
 */
export const SERVER_MODEL_POLL_MIN_MS = 500;
export const SERVER_MODEL_POLL_LOADING_MS = 1_000;
export const SERVER_MODEL_POLL_MAX_MS = 5_000;
const SERVER_MODEL_POLL_BACKOFF = 1.6;

export function nextServerModelPollDelay(
  previousMs: number,
  state: { loading: boolean; hidden: boolean },
): number {
  if (state.loading) return SERVER_MODEL_POLL_LOADING_MS;
  // Nobody is watching a hidden tab, and the visible tab's own observer covers the same server.
  if (state.hidden) return SERVER_MODEL_POLL_MAX_MS;
  return Math.min(
    SERVER_MODEL_POLL_MAX_MS,
    Math.max(SERVER_MODEL_POLL_MIN_MS, Math.round(previousMs * SERVER_MODEL_POLL_BACKOFF)),
  );
}

/** Resolves after `ms`, or as soon as `signal` aborts: a 5 s wait must not outlive the unmount. */
export function sleepUnlessAborted(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", done);
      resolve();
    };
    const timer = setTimeout(done, ms);
    signal?.addEventListener("abort", done, { once: true });
  });
}
