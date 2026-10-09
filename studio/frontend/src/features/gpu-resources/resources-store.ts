// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One shared reading of /api/resources. The sidebar strip is its first reader; anything else can
// subscribe to the store without starting a poll of its own. Polling runs while at least one
// component holds useResourcesPolling(true), at the cadence resourcesPollIntervalMs picks, and
// stops in a hidden tab. Timers, not a stream: no connection stays open.

import { useEffect } from "react";
import { create } from "zustand";
import { authFetch, hasAuthToken, mustChangePassword } from "@/features/auth";
import { isTauri } from "@/lib/api-base";
import {
  type ModelRuntime,
  subscribeModelLifecycle,
} from "@/lib/model-lifecycle-events";
import {
  type ResourceSnapshot,
  normalizeResourceSnapshot,
  resourcesPollIntervalMs,
} from "./resources-model";

// Well past a cold read (the holder lookup never runs on the request), so only a hang trips it.
const READ_TIMEOUT_MS = 10_000;

type ResourcesState = {
  snapshot: ResourceSnapshot | null;
  /** The last read failed. `snapshot` is the last good one, kept rather than blanked. */
  stale: boolean;
  panelOpen: boolean;
  /** Loads this tab announced (withModelLoadNotice) that have not settled yet. */
  announcedLoads: readonly ModelRuntime[];
  setPanelOpen: (open: boolean) => void;
  refresh: () => Promise<void>;
};

export const useResourcesStore = create<ResourcesState>()((set) => ({
  snapshot: null,
  stale: false,
  panelOpen: false,
  announcedLoads: [],
  setPanelOpen: (panelOpen) => {
    set({ panelOpen });
    // Opening wants current numbers now, not at the end of a 10 s wait.
    if (panelOpen) void refreshThenSchedule();
    else schedule();
  },
  refresh: () => refreshThenSchedule(),
}));

/** Whether any load is running, by the backend's word or this tab's own announcement. */
export function resourcesLoading(state: ResourcesState): boolean {
  return state.snapshot?.loading === true || state.announcedLoads.length > 0;
}

// Nothing to read before there is a session: every request would be a 401 and a redirect.
function canPoll(): boolean {
  return isTauri || (hasAuthToken() && !mustChangePassword());
}

let consumers = 0;
let timer: ReturnType<typeof setTimeout> | null = null;
let inFlight: Promise<void> | null = null;
let detachLifecycle: (() => void) | null = null;

async function read(): Promise<void> {
  if (inFlight) return inFlight;
  if (!canPoll()) return;
  const controller = new AbortController();
  const abort = setTimeout(() => controller.abort(), READ_TIMEOUT_MS);
  inFlight = (async () => {
    try {
      const response = await authFetch("/api/resources", {
        signal: controller.signal,
      });
      if (!response.ok) {
        // A backend older than the route answers 404: no snapshot, so no strip.
        useResourcesStore.setState(
          response.status === 404
            ? { snapshot: null, stale: false }
            : { stale: true },
        );
        return;
      }
      const snapshot = normalizeResourceSnapshot(await response.json());
      useResourcesStore.setState({ snapshot, stale: false });
    } catch {
      useResourcesStore.setState({ stale: true });
    } finally {
      clearTimeout(abort);
      inFlight = null;
    }
  })();
  return inFlight;
}

function schedule(): void {
  if (timer !== null) {
    clearTimeout(timer);
    timer = null;
  }
  if (consumers === 0) return;
  const state = useResourcesStore.getState();
  const delay = resourcesPollIntervalMs({
    hidden: typeof document !== "undefined" && document.hidden,
    loading: resourcesLoading(state),
    panelOpen: state.panelOpen,
  });
  if (delay === null) return;
  timer = setTimeout(() => {
    timer = null;
    void refreshThenSchedule();
  }, delay);
}

async function refreshThenSchedule(): Promise<void> {
  try {
    await read();
  } finally {
    schedule();
  }
}

function onVisibilityChange(): void {
  // Back on screen: the reading is as old as the time away, so take one now.
  if (!document.hidden) void refreshThenSchedule();
  else schedule();
}

function onLifecycle(runtime: ModelRuntime, loading: boolean): void {
  const announced = useResourcesStore.getState().announcedLoads;
  const next = loading
    ? [...announced.filter((r) => r !== runtime), runtime]
    : announced.filter((r) => r !== runtime);
  useResourcesStore.setState({ announcedLoads: next });
  // A settled load is read at once; a started one just moves the cadence to every second.
  if (loading) schedule();
  else void refreshThenSchedule();
}

function start(): void {
  document.addEventListener("visibilitychange", onVisibilityChange);
  detachLifecycle = subscribeModelLifecycle(({ runtime, loading }) =>
    onLifecycle(runtime, loading),
  );
  void refreshThenSchedule();
}

function stop(): void {
  document.removeEventListener("visibilitychange", onVisibilityChange);
  detachLifecycle?.();
  detachLifecycle = null;
  if (timer !== null) {
    clearTimeout(timer);
    timer = null;
  }
}

/** Keep the shared poll running while this component is mounted and `enabled`. */
export function useResourcesPolling(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    consumers += 1;
    if (consumers === 1) start();
    return () => {
      consumers -= 1;
      if (consumers === 0) stop();
    };
  }, [enabled]);
}
