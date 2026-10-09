// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Ejects from the resources panel. Chat, images, video and dictation go through the loaded
// models card's own eject, which re-reads the runtime first and leaves alone a model that took
// the row's place; the embedder is released the way Settings -> Documents releases it.

import { authFetch } from "@/features/auth";
import { type EjectOutcome, ejectLoadedModel } from "@/features/loaded-models";
import { type ResourceModel, ejectEntryFor } from "./resources-model";

async function readErrorDetail(response: Response): Promise<string | null> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // non-JSON error body
  }
  return null;
}

/** Release one model. Throws with the backend's reason when the unload was refused. */
export async function ejectResourceModel(
  model: ResourceModel,
): Promise<EjectOutcome> {
  if (model.source === "embedding") {
    const response = await authFetch("/api/settings/embedding-model/unload", {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(
        (await readErrorDetail(response)) ??
          `Request failed (${response.status})`,
      );
    }
    return { status: "ejected" };
  }
  const entry = ejectEntryFor(model);
  if (!entry) throw new Error("This model cannot be ejected from here.");
  return ejectLoadedModel(entry);
}
