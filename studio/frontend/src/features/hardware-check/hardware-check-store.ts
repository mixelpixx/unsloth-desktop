// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One shared reading of /api/hardware-check (owner only). Settings > Resources > Hardware check
// drives it, and the Tensor Parallelism control reads it for its warning. Polled only while a run
// is going; when one finishes the resource strip is refreshed so its link badges follow.

import { useEffect } from "react";
import { create } from "zustand";
import { useIsAccountOwner } from "@/features/auth";
import { useResourcesStore } from "@/features/gpu-resources";
import {
  HardwareCheckAbsentError,
  applyHardwareCheckRecommended,
  loadHardwareCheck,
  runHardwareCheck,
  updateHardwareCheckSettings,
} from "./hardware-check-api";
import {
  type HardwareCheckSettingKey,
  type HardwareCheckStatus,
  hardwareCheckPollMs,
} from "./hardware-check-model";

type HardwareCheckState = {
  status: HardwareCheckStatus | null;
  /** The first read has answered (successfully or not). */
  loaded: boolean;
  error: string | null;
  /** The backend predates the hardware check: there is nothing to show. */
  absent: boolean;
  /** A settings write or a run request is in flight. */
  busy: boolean;
  refresh: () => Promise<void>;
  run: () => Promise<{ started: boolean; reason: string | null } | null>;
  setSetting: (key: HardwareCheckSettingKey, value: boolean) => Promise<boolean>;
  applyRecommended: () => Promise<string[] | null>;
};

let pollTimer: ReturnType<typeof setTimeout> | null = null;

function message(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function adopt(status: HardwareCheckStatus): void {
  const previous = useHardwareCheckStore.getState().status;
  useHardwareCheckStore.setState({ status, error: null, loaded: true });
  if (previous?.state.running && !status.state.running) {
    // A run just finished: the strip's link badges come from the stored result.
    void useResourcesStore.getState().refresh();
  }
  schedule();
}

function schedule(): void {
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  const delay = hardwareCheckPollMs(useHardwareCheckStore.getState().status);
  if (delay === null) return;
  pollTimer = setTimeout(() => {
    pollTimer = null;
    void useHardwareCheckStore.getState().refresh();
  }, delay);
}

export const useHardwareCheckStore = create<HardwareCheckState>()((set, get) => ({
  status: null,
  loaded: false,
  error: null,
  absent: false,
  busy: false,
  refresh: async () => {
    try {
      adopt(await loadHardwareCheck());
    } catch (error) {
      if (error instanceof HardwareCheckAbsentError) {
        set({ loaded: true, absent: true, error: null });
        return;
      }
      set({ loaded: true, error: message(error, "Could not read the hardware check.") });
    }
  },
  run: async () => {
    if (get().busy) return null;
    set({ busy: true });
    try {
      const outcome = await runHardwareCheck();
      adopt(outcome.status);
      return { started: outcome.started, reason: outcome.reason };
    } catch (error) {
      set({ error: message(error, "Could not start the hardware check.") });
      return null;
    } finally {
      set({ busy: false });
    }
  },
  setSetting: async (key, value) => {
    set({ busy: true });
    try {
      adopt(await updateHardwareCheckSettings({ [key]: value }));
      return true;
    } catch (error) {
      set({ error: message(error, "Could not save the hardware check setting.") });
      return false;
    } finally {
      set({ busy: false });
    }
  },
  applyRecommended: async () => {
    set({ busy: true });
    try {
      const { applied, status } = await applyHardwareCheckRecommended();
      adopt(status);
      return applied;
    } catch (error) {
      set({ error: message(error, "Could not apply the recommended options.") });
      return null;
    } finally {
      set({ busy: false });
    }
  },
}));

/** Read the hardware check once (owner only) for a component that shows it. */
export function useHardwareCheck(): HardwareCheckStatus | null {
  const isOwner = useIsAccountOwner();
  const status = useHardwareCheckStore((s) => s.status);
  const loaded = useHardwareCheckStore((s) => s.loaded);
  useEffect(() => {
    if (isOwner && !loaded) {
      void useHardwareCheckStore.getState().refresh();
    }
  }, [isOwner, loaded]);
  return isOwner ? status : null;
}
