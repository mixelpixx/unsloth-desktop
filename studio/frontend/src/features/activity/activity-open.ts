// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useNavigate } from "@tanstack/react-router";
import { useTrainingRuntimeStore } from "@/features/training";
import type { ActivityEntry } from "@/lib/activity-store";
import { RECIPE_ROUTE } from "./activity-sources";

type Navigate = ReturnType<typeof useNavigate>;

const PAGE_ROUTES = [
  "/hub",
  "/studio",
  "/export",
  "/chat",
  "/images",
  "/video",
  "/audio",
] as const;

type PageRoute = (typeof PAGE_ROUTES)[number];

function isPageRoute(route: string | null): route is PageRoute {
  return PAGE_ROUTES.some((page) => page === route);
}

/** Open: the page that owns the job. Nothing is started again from here. */
export function openActivityEntry(
  entry: ActivityEntry,
  navigate: Navigate,
): void {
  if (entry.kind === "training") {
    // A running job is the Current Run view; a finished one is its history entry.
    useTrainingRuntimeStore
      .getState()
      .setSelectedHistoryRunId(entry.state === "active" ? null : entry.ref);
  }
  if (entry.route === RECIPE_ROUTE && entry.ref) {
    void navigate({ to: RECIPE_ROUTE, params: { recipeId: entry.ref } });
    return;
  }
  if (isPageRoute(entry.route)) void navigate({ to: entry.route });
}
