// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The preference module comes FIRST, for the reason loaded-models/index.ts gives: general-tab
// reads RESOURCES_STRIP_PREFERENCE_KEY at module scope, and the strip reaches back into the
// settings barrel through the loaded-models card. Evaluating the key before the strip keeps that
// cycle harmless whichever side is entered first.
export {
  RESOURCES_STRIP_PREFERENCE_KEY,
  getShowResourcesStrip,
  setShowResourcesStrip,
  useShowResourcesStrip,
} from "./show-resources-strip-pref";
export { ResourcesStrip } from "./resources-strip";
export {
  resourcesLoading,
  useResourcesPolling,
  useResourcesStore,
} from "./resources-store";
export type {
  ResourceApp,
  ResourceGpu,
  ResourceModel,
  ResourceSnapshot,
} from "./resources-model";
