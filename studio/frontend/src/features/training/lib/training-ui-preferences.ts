// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const TRAINING_MODEL_PICKER_TAB_STORAGE_KEY =
  "unsloth.studio.train.modelPickerTab";
export const TRAINING_DATASET_PICKER_TAB_STORAGE_KEY =
  "unsloth.studio.train.datasetPickerTab";
export const TRAINING_PARAM_MODE_STORAGE_KEY = "unsloth.studio.train.paramMode";
export const LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY =
  "unsloth_train_param_mode";
// The GPU a run is pinned to. A preference of this browser on this host, not part of the run
// config: it names physical GPU indices, which mean nothing in an exported YAML on another box.
export const TRAINING_GPU_TARGET_STORAGE_KEY =
  "unsloth.studio.train.gpuTarget";

export const TRAINING_UI_PREFERENCE_KEYS = [
  TRAINING_MODEL_PICKER_TAB_STORAGE_KEY,
  TRAINING_DATASET_PICKER_TAB_STORAGE_KEY,
  TRAINING_PARAM_MODE_STORAGE_KEY,
  LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY,
  TRAINING_GPU_TARGET_STORAGE_KEY,
] as const;
