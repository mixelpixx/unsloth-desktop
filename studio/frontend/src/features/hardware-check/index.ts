// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Resources > Hardware check: the shared client. Settings draws the section; the model
// settings page reads it for the Tensor Parallelism warning.
export {
  HARDWARE_CHECK_OPTIONS,
  type FindingMessage,
  type HardwareCheckFinding,
  type HardwareCheckGpu,
  type HardwareCheckOption,
  type HardwareCheckPair,
  type HardwareCheckSettingKey,
  type HardwareCheckStatus,
  type HardwareCheckStorage,
  findingMessage,
  formatGibs,
  formatLink,
  formatLinkMax,
  locationKey,
  optionsToApply,
  peerAccess,
  skipReasonKey,
  tensorSplitConcern,
} from "./hardware-check-model";
export { HardwareCheckAbsentError } from "./hardware-check-api";
export { useHardwareCheck, useHardwareCheckStore } from "./hardware-check-store";
export { TensorParallelLinkWarning } from "./tensor-parallel-link-warning";
