// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hfApiToken, useHfTokenStore } from "@/features/hub";
import { type SystemGpuDevice, useGpuDevices } from "@/hooks/use-gpu-info";
import { useEffect, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { buildTrainingEstimatePayload } from "../api/mappers";
import { estimateTrainingFit } from "../api/train-api";
import {
  type TrainingGpuDevice,
  type TrainingGpuTarget,
  gpuIdsForTrainingTarget,
  resolveTrainingGpuTarget,
} from "../lib/training-fit";
import { useTrainingConfigStore } from "../stores/training-config-store";
import { useTrainingFitStore } from "../stores/training-fit-store";

// Re-price once edits settle, not per keystroke in the batch or context fields.
const ESTIMATE_DEBOUNCE_MS = 400;

function toTrainingGpuDevice(device: SystemGpuDevice): TrainingGpuDevice {
  return {
    index: device.index,
    name: device.name,
    memoryTotalGb: device.memoryTotalGb,
  };
}

/**
 * Keep the fit store's estimate in step with the Configure form and the GPU target.
 *
 * Mount once (the run preview does). Failures land as status "error" and the panel shows its
 * unknown state; nothing here can hold up Start.
 *
 * `fourBitUnavailable` comes from the caller's useTrainingTransformersUpgradeNotice() rather
 * than a second call here: that hook fetches on mount with no in-flight dedupe, so two
 * consumers would ask the backend the same question twice per model.
 */
export function useTrainingFitEstimate(fourBitUnavailable: boolean): {
  /** GPUs a run can be pinned to (the target select's options). */
  devices: TrainingGpuDevice[];
  /** Every training-visible GPU, for describing the hardware. */
  allDevices: TrainingGpuDevice[];
  target: TrainingGpuTarget;
} {
  const config = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      trainingMethod: s.trainingMethod,
      contextLength: s.contextLength,
      batchSize: s.batchSize,
      loraRank: s.loraRank,
      targetModules: s.targetModules,
      gradientCheckpointing: s.gradientCheckpointing,
      optimizerType: s.optimizerType,
      isLoadingModelDefaults: s.isLoadingModelDefaults,
    })),
  );
  const hfToken = useHfTokenStore((s) => s.token);
  const gpuTarget = useTrainingFitStore((s) => s.gpuTarget);
  const setPinnableGpus = useTrainingFitStore((s) => s.setPinnableGpus);
  const setEstimate = useTrainingFitStore((s) => s.setEstimate);

  // The torch inventory, narrowed to physical CUDA/ROCm ids: the index space training's gpu_ids
  // speak. An XPU tile handle or a Vulkan ordinal is not one, so those hosts get no choice and
  // keep backend auto-selection.
  const systemDevices = useGpuDevices(true);
  const allDevices = useMemo(
    () => systemDevices.map(toTrainingGpuDevice),
    [systemDevices],
  );
  const devices = useMemo(
    () =>
      systemDevices
        .filter((device) => device.diffusionPinnable)
        .map(toTrainingGpuDevice),
    [systemDevices],
  );
  useEffect(() => {
    setPinnableGpus(devices);
  }, [devices, setPinnableGpus]);

  const target = resolveTrainingGpuTarget(gpuTarget, devices);
  const gpuIds = gpuIdsForTrainingTarget(target, devices);
  // A string, so the effect re-runs on a changed value rather than on every fresh object.
  const payloadKey = config.selectedModel
    ? JSON.stringify(
        buildTrainingEstimatePayload(config, {
          hfToken: hfApiToken(hfToken) ?? null,
          gpuIds,
          fourBitAvailable: !fourBitUnavailable,
        }),
      )
    : null;
  const waitingForDefaults = config.isLoadingModelDefaults;

  useEffect(() => {
    if (payloadKey === null) {
      setEstimate("idle", null);
      return;
    }
    // The previous estimate stays on screen, dimmed, until this one lands; Start only acts on a
    // "ready" one, so a stale verdict can never prompt the confirm.
    setEstimate("loading", useTrainingFitStore.getState().estimate);
    // Model defaults are still landing (method, rank, context): price the settled form instead.
    if (waitingForDefaults) {
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      estimateTrainingFit(JSON.parse(payloadKey), controller.signal)
        .then((estimate) => {
          if (!controller.signal.aborted) {
            setEstimate("ready", estimate);
          }
        })
        .catch(() => {
          if (!controller.signal.aborted) {
            setEstimate("error", null);
          }
        });
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [payloadKey, waitingForDefaults, setEstimate]);

  return { devices, allDevices, target };
}
