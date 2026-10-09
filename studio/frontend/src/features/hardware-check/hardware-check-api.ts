// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import {
  HARDWARE_CHECK_ENDPOINT,
  type HardwareCheckSettingKey,
  type HardwareCheckStatus,
  parseHardwareCheckStatus,
} from "./hardware-check-model";

/** The running backend predates the hardware check: there is nothing to show. */
export class HardwareCheckAbsentError extends Error {
  constructor() {
    super("This Studio backend has no hardware check.");
    this.name = "HardwareCheckAbsentError";
  }
}

export async function loadHardwareCheck(): Promise<HardwareCheckStatus> {
  const res = await authFetch(HARDWARE_CHECK_ENDPOINT);
  if (res.status === 404) {
    throw new HardwareCheckAbsentError();
  }
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Could not read the hardware check."));
  }
  return parseHardwareCheckStatus(await res.json());
}

export type HardwareCheckRunOutcome = {
  started: boolean;
  /** Why it did not start (training_active, model_loading, already_running). */
  reason: string | null;
  status: HardwareCheckStatus;
};

export async function runHardwareCheck(): Promise<HardwareCheckRunOutcome> {
  const res = await authFetch(`${HARDWARE_CHECK_ENDPOINT}/run`, { method: "POST" });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Could not start the hardware check."));
  }
  const body = (await res.json()) as Record<string, unknown>;
  return {
    started: body.started === true,
    reason: typeof body.reason === "string" ? body.reason : null,
    status: parseHardwareCheckStatus(body.status),
  };
}

export async function updateHardwareCheckSettings(
  patch: Partial<Record<HardwareCheckSettingKey, boolean>>,
): Promise<HardwareCheckStatus> {
  const res = await authFetch(`${HARDWARE_CHECK_ENDPOINT}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Could not save the hardware check setting."));
  }
  return parseHardwareCheckStatus(await res.json());
}

export async function applyHardwareCheckRecommended(): Promise<{
  applied: string[];
  status: HardwareCheckStatus;
}> {
  const res = await authFetch(`${HARDWARE_CHECK_ENDPOINT}/apply-recommended`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Could not apply the recommended options."));
  }
  const body = (await res.json()) as Record<string, unknown>;
  return {
    applied: Array.isArray(body.applied)
      ? body.applied.filter((v): v is string => typeof v === "string")
      : [],
    status: parseHardwareCheckStatus(body.status),
  };
}
