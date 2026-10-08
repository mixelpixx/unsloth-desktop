// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { SettingsRouteAbsentError } from "./settings-route-absent";

/** Which memory verdicts stop a load for "Load anyway". Mirrors LOAD_GUARDRAIL_MODES in
 *  studio/backend/utils/load_guardrail_settings.py, strictest first. */
export const LOAD_GUARDRAIL_MODES = [
  "strict",
  "balanced",
  "relaxed",
  "off",
] as const;

export type LoadGuardrailMode = (typeof LOAD_GUARDRAIL_MODES)[number];

export type LoadGuardrailSettings = {
  mode: LoadGuardrailMode;
  /** False when inherited from UNSLOTH_LOAD_GUARDRAILS or the default. */
  isStored: boolean;
  defaultMode: LoadGuardrailMode;
};

type ApiLoadGuardrailSettings = {
  mode: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  is_stored: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_mode: string;
};

export function isLoadGuardrailMode(value: unknown): value is LoadGuardrailMode {
  return (
    typeof value === "string" &&
    (LOAD_GUARDRAIL_MODES as readonly string[]).includes(value)
  );
}

function fromApi(settings: ApiLoadGuardrailSettings): LoadGuardrailSettings {
  // A mode a newer backend added reads as the default rather than as a blank select.
  const defaultMode = isLoadGuardrailMode(settings.default_mode)
    ? settings.default_mode
    : "balanced";
  return {
    mode: isLoadGuardrailMode(settings.mode) ? settings.mode : defaultMode,
    isStored: settings.is_stored === true,
    defaultMode,
  };
}

export async function loadLoadGuardrailSettings(): Promise<LoadGuardrailSettings> {
  const res = await authFetch("/api/settings/load-guardrails");
  if (res.status === 404) {
    throw new SettingsRouteAbsentError("/api/settings/load-guardrails");
  }
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load the load guardrail setting"),
    );
  }
  return fromApi(await res.json());
}

export async function updateLoadGuardrailMode(
  mode: LoadGuardrailMode,
): Promise<LoadGuardrailSettings> {
  const res = await authFetch("/api/settings/load-guardrails", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save the load guardrail setting"),
    );
  }
  return fromApi(await res.json());
}
