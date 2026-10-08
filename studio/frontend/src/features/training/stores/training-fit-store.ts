// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import {
  type TrainingFitEstimate,
  type TrainingGpuDevice,
  type TrainingGpuTarget,
  gpuIdsForTrainingTarget,
  isTrainingGpuTarget,
} from "../lib/training-fit";
import { TRAINING_GPU_TARGET_STORAGE_KEY } from "../lib/training-ui-preferences";

/** localStorage that cannot throw (private browsing, blocked storage, opaque webview origins). */
const safeStorage = {
  getItem: (name: string): string | null => {
    try {
      return window.localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name: string, value: string): void => {
    try {
      window.localStorage.setItem(name, value);
    } catch {
      // Denied or over quota: the pick lasts the session.
    }
  },
  removeItem: (name: string): void => {
    try {
      window.localStorage.removeItem(name);
    } catch {
      // Same.
    }
  },
};

export type TrainingFitStatus = "idle" | "loading" | "ready" | "error";

interface TrainingFitState {
  /** Persisted. Resolved against the live inventory before use, so a pick naming a GPU this
   *  host no longer has reads as Auto instead of pinning the run to nothing. */
  gpuTarget: TrainingGpuTarget;
  /** The GPUs the target select offers, published by the fit panel so Start maps the target to
   *  gpu_ids against the same list the user chose from. Empty means "no choice": Start sends
   *  no gpu_ids and the backend auto-selects. */
  pinnableGpus: TrainingGpuDevice[];
  status: TrainingFitStatus;
  estimate: TrainingFitEstimate | null;
  setGpuTarget: (gpuTarget: TrainingGpuTarget) => void;
  setPinnableGpus: (pinnableGpus: TrainingGpuDevice[]) => void;
  setEstimate: (status: TrainingFitStatus, estimate: TrainingFitEstimate | null) => void;
}

/** Only the target persists; an estimate is about the free memory of the moment. */
export const useTrainingFitStore = create<TrainingFitState>()(
  persist(
    (set) => ({
      gpuTarget: "auto",
      pinnableGpus: [],
      status: "idle",
      estimate: null,
      setGpuTarget: (gpuTarget) => set({ gpuTarget }),
      setPinnableGpus: (pinnableGpus) => set({ pinnableGpus }),
      setEstimate: (status, estimate) => set({ status, estimate }),
    }),
    {
      name: TRAINING_GPU_TARGET_STORAGE_KEY,
      version: 1,
      storage: createJSONStorage(() => safeStorage),
      partialize: (state) => ({ gpuTarget: state.gpuTarget }),
      migrate: (persisted) => persisted,
      // A hand-edited or future payload must not put an unparseable target in front of Start.
      merge: (persisted, current) => {
        const stored = (persisted as { gpuTarget?: unknown } | null)?.gpuTarget;
        return {
          ...current,
          gpuTarget: isTrainingGpuTarget(stored) ? stored : current.gpuTarget,
        };
      },
    },
  ),
);

/** gpu_ids for the run about to start: the visible target mapped onto the GPUs the select
 *  offered, or null (backend auto-selection) when there was nothing to choose. */
export function selectedTrainingGpuIds(): number[] | null {
  const { gpuTarget, pinnableGpus } = useTrainingFitStore.getState();
  return gpuIdsForTrainingTarget(gpuTarget, pinnableGpus);
}
