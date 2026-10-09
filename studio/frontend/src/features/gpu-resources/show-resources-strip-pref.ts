// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the sidebar's GPU memory strip shows (Settings -> Appearance). On by default; only an
// explicit "false" turns it off, the same tri-state the loaded models card uses, so a pre-update
// tab reading an absent key and this one agree. Imports nothing but React: general-tab reads the
// key at module scope, through the barrel that exports this module first.

import { useSyncExternalStore } from "react";

export const RESOURCES_STRIP_PREFERENCE_KEY = "unsloth_show_resources_strip";

const listeners = new Set<() => void>();

export function getShowResourcesStrip(): boolean {
  try {
    return localStorage.getItem(RESOURCES_STRIP_PREFERENCE_KEY) !== "false";
  } catch {
    return true;
  }
}

export function setShowResourcesStrip(show: boolean): void {
  try {
    localStorage.setItem(RESOURCES_STRIP_PREFERENCE_KEY, show ? "true" : "false");
  } catch {
    // storage unavailable
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Another tab's toggle, and "Reset all local preferences" (a cleared store reads as on).
  const onStorage = (event: StorageEvent) => {
    if (event.key === RESOURCES_STRIP_PREFERENCE_KEY || event.key === null) {
      listener();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function useShowResourcesStrip(): boolean {
  return useSyncExternalStore(subscribe, getShowResourcesStrip);
}
