// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Feeds the activity store from what each subsystem already tracks. No requests and no timers of
// its own: every source is a store another part of the app keeps current (and polls, where it
// polls), or an event the API calls raise. Mounted once, from the always-rendered sidebar.

import { useNavigate } from "@tanstack/react-router";
import { useEffect, useRef } from "react";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import {
  selectExportProgressPercent,
  useExportRuntimeStore,
} from "@/features/export";
import { presentedProgress, useDownloadManagerStore } from "@/features/hub";
// The store module alone: the recipe-studio barrel re-exports its page, which this always-loaded
// sidebar would pull out of its code split.
// eslint-disable-next-line no-restricted-imports
import { useRecipeExecutionsStore } from "@/features/recipe-studio/stores/recipe-executions";
import {
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import {
  type ActivityEntry,
  reportActivity,
  resetActivity,
  settleActivity,
  subscribeActivityFinished,
  useActivityStore,
} from "@/lib/activity-store";
import { isTauri } from "@/lib/api-base";
import {
  type ModelRuntime,
  canCancelModelLoad,
  subscribeModelLifecycle,
  subscribeModelLoadCancels,
} from "@/lib/model-lifecycle-events";
import { notifyJobFinished } from "./activity-notifications";
import { openActivityEntry } from "./activity-open";
import {
  downloadSnapshots,
  exportSnapshots,
  modelLoadRoute,
  modelLoadSettlement,
  modelLoadTitle,
  recipeSnapshots,
  tracksModelLoad,
  trainingSnapshots,
} from "./activity-sources";

function activeEntry(id: string): ActivityEntry | undefined {
  return useActivityStore
    .getState()
    .entries.find((entry) => entry.id === id && entry.state === "active");
}

function startDownloadFeed(): () => void {
  const sync = () =>
    reportActivity(
      downloadSnapshots(
        Object.values(useDownloadManagerStore.getState().jobs),
        presentedProgress,
      ),
      // A job the manager dropped while running was found gone, which it reports as cancelled.
      { prefix: "download:", missing: "cancel" },
    );
  sync();
  return useDownloadManagerStore.subscribe((state, previous) => {
    if (state.jobs !== previous.jobs) sync();
  });
}

function startTrainingFeed(): () => void {
  const sync = () =>
    reportActivity(
      trainingSnapshots(
        useTrainingRuntimeStore.getState(),
        useTrainingConfigStore.getState().selectedModel,
        Date.now(),
      ),
      // The store moved to another job or back to idle without saying how this one ended.
      { prefix: "training:", missing: "cancel" },
    );
  sync();
  return useTrainingRuntimeStore.subscribe(sync);
}

function startExportFeed(): () => void {
  // The store changes on every log line; only these fields move the row.
  let lastKey = "";
  const sync = () => {
    const state = useExportRuntimeStore.getState();
    const key = JSON.stringify([
      state.phase,
      state.isExporting,
      state.startedAt,
      state.error,
      state.stage,
      state.quantIndex,
      state.quantTotal,
      state.summary?.baseModelName,
    ]);
    if (key === lastKey) return;
    lastKey = key;
    reportActivity(
      exportSnapshots(state, selectExportProgressPercent(state)),
      // A recovered run that turns out to have been a checkpoint load was never an export.
      { prefix: "export:", missing: "drop" },
    );
  };
  sync();
  return useExportRuntimeStore.subscribe(sync);
}

function startRecipeFeed(): () => void {
  const signatures = new Map<string, string>();
  const sessionStart = Date.now();
  const sync = () => {
    const activeIds = new Set(
      useActivityStore
        .getState()
        .entries.filter(
          (entry) => entry.kind === "recipe" && entry.state === "active",
        )
        .map((entry) => entry.id),
    );
    reportActivity(
      recipeSnapshots(
        useRecipeExecutionsStore.getState().executions,
        signatures,
        activeIds,
        sessionStart,
      ),
      // Opening another recipe empties the store while this run's tracker keeps going; its next
      // update puts the record back, so the row waits rather than settling on a guess.
      { prefix: "recipe:", missing: "keep" },
    );
  };
  sync();
  return useRecipeExecutionsStore.subscribe((state, previous) => {
    if (state.executions !== previous.executions) sync();
  });
}

function startModelLoadFeed(): () => void {
  // The row for each runtime's load in flight. Background loads announce "loading" more than once.
  const current = new Map<ModelRuntime, string>();
  const report = (
    runtime: ModelRuntime,
    id: string,
    title: string,
    startedAt: number,
  ) =>
    reportActivity(
      [
        {
          id,
          kind: "model-load",
          title,
          state: "active",
          startedAt,
          route: modelLoadRoute(runtime),
          ref: runtime,
          cancellable: canCancelModelLoad(runtime),
        },
      ],
      { prefix: id, missing: "keep" },
    );

  const stopLifecycle = subscribeModelLifecycle(
    ({ runtime, loading, model, outcome }) => {
      if (!tracksModelLoad(runtime)) return;
      const now = Date.now();
      const known = current.get(runtime);
      const existing = known ? activeEntry(known) : undefined;
      if (loading) {
        const id = existing?.id ?? `model-load:${runtime}:${now}`;
        current.set(runtime, id);
        report(
          runtime,
          id,
          modelLoadTitle(model) || existing?.title || "",
          existing?.startedAt ?? now,
        );
        return;
      }
      current.delete(runtime);
      if (!existing) return;
      settleActivity(
        existing.id,
        modelLoadSettlement(outcome, existing.startedAt, now),
        undefined,
        now,
      );
    },
  );
  // A page that can stop the load mounted or went away: the row's Cancel follows it.
  const stopCancels = subscribeModelLoadCancels(() => {
    for (const [runtime, id] of current) {
      const entry = activeEntry(id);
      if (entry) report(runtime, id, entry.title, entry.startedAt);
    }
  });
  return () => {
    stopLifecycle();
    stopCancels();
    // Nobody will hear these settle now.
    for (const id of current.values()) settleActivity(id, "drop");
    current.clear();
  };
}

/** Keep the activity store fed for as long as the caller is mounted. Call once, from the shell. */
export function useActivityFeeds(): void {
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  useEffect(() => {
    navigateRef.current = navigate;
  }, [navigate]);

  useEffect(() => {
    const stops = [
      startDownloadFeed(),
      startTrainingFeed(),
      startExportFeed(),
      startRecipeFeed(),
      startModelLoadFeed(),
      subscribeActivityFinished((entry) =>
        notifyJobFinished(entry, (opened) =>
          openActivityEntry(opened, navigateRef.current),
        ),
      ),
    ];
    return () => {
      for (const stop of stops) stop();
    };
  }, []);
}

// A signed-out web session's errors and jobs are not the next account's to read. The desktop app
// keeps them, as its download manager keeps its jobs.
function resetForNewSession(): void {
  if (!isTauri) resetActivity();
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, resetForNewSession);
  if (import.meta.hot) {
    import.meta.hot.dispose(() => {
      window.removeEventListener(
        AUTH_SESSION_CLEARED_EVENT,
        resetForNewSession,
      );
    });
  }
}
