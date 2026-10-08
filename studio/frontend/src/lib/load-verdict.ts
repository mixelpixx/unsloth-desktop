// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend's load guardrail verdict: one plain answer to "will this load fit in the memory
// free right now", computed by core/inference/load_verdict.py. /estimate-memory carries it for
// the load page headline and /load refuses with it (409, code "memory_overcommit") when the
// guardrail mode says to ask first. In lib/ because both of those surfaces are different
// features. Import-free at runtime so the node test runner can load it directly.

import type { TranslationKey } from "@/i18n";

export type LoadVerdictLevel =
  | "full_gpu"
  | "fits_barely"
  | "partial_gpu"
  | "cpu"
  | "disk_streaming"
  | "likely_too_large"
  | "unknown";

const LEVELS: readonly LoadVerdictLevel[] = [
  "full_gpu",
  "fits_barely",
  "partial_gpu",
  "cpu",
  "disk_streaming",
  "likely_too_large",
  "unknown",
];

export interface LoadVerdict {
  level: LoadVerdictLevel;
  /** Stable code the copy is keyed off ("forced_gpu_overflow", "spills_to_ram", ...). */
  reason: string;
  /** The backend's English sentence: the fallback for a reason this bundle has no copy for. */
  message: string;
  gpuNeedBytes: number | null;
  gpuFreeBytes: number | null;
  gpuTotalBytes: number | null;
  gpuIndices: number[];
  otherAppsBytes: number | null;
  /** "LM Studio.exe (PID 1372) ~13.8 GB on GPU 0"; only on a /load refusal, Windows only. */
  otherAppsNote: string | null;
  ramNeedBytes: number | null;
  ramFreeBytes: number | null;
  /** KV cache, compute buffers and runtime state: what cannot page from disk. */
  runtimeBytes: number | null;
  headroomBytes: number | null;
  /** Whether the current guardrail mode stops /load for "Load anyway". */
  needsConfirmation: boolean;
  mode: string | null;
}

function bytes(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

function signedBytes(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

/** A verdict from the wire, or null when the shape is not one. A level this bundle does not
 *  know is not a verdict: rendering it would need copy that does not exist. */
export function parseLoadVerdict(raw: unknown): LoadVerdict | null {
  if (!raw || typeof raw !== "object") return null;
  const body = raw as Record<string, unknown>;
  const level = body.level;
  if (typeof level !== "string" || !LEVELS.includes(level as LoadVerdictLevel)) {
    return null;
  }
  return {
    level: level as LoadVerdictLevel,
    reason: typeof body.reason === "string" ? body.reason : "",
    message: typeof body.message === "string" ? body.message : "",
    gpuNeedBytes: bytes(body.gpu_need_bytes),
    gpuFreeBytes: bytes(body.gpu_free_bytes),
    gpuTotalBytes: bytes(body.gpu_total_bytes),
    gpuIndices: Array.isArray(body.gpu_indices)
      ? body.gpu_indices.filter(
          (index): index is number =>
            typeof index === "number" && Number.isInteger(index) && index >= 0,
        )
      : [],
    otherAppsBytes: bytes(body.other_apps_bytes),
    otherAppsNote: text(body.other_apps_note),
    ramNeedBytes: bytes(body.ram_need_bytes),
    ramFreeBytes: bytes(body.ram_free_bytes),
    runtimeBytes: bytes(body.runtime_bytes),
    headroomBytes: signedBytes(body.headroom_bytes),
    needsConfirmation: body.needs_confirmation === true,
    mode: typeof body.mode === "string" ? body.mode : null,
  };
}

export const MEMORY_OVERCOMMIT_CODE = "memory_overcommit";

/**
 * The verdict a /load refused on, or null when this response is anything else.
 *
 * Two shapes carry it. A refusal inside the padding window is a real 409 with the structured
 * detail; one that lands after the tunnel padding committed a 200 arrives as `_deferred_error`
 * in the body (see `_tunnel_safe_json`). The guardrail runs before anything slow in practice,
 * but a slow config resolve can push it past the window, and both must ask the same question.
 */
export function memoryOvercommitVerdict(
  status: number,
  body: unknown,
): LoadVerdict | null {
  if (!body || typeof body !== "object") return null;
  let detail: unknown;
  if (status === 409) {
    detail = (body as { detail?: unknown }).detail;
  } else {
    const deferred = (body as { _deferred_error?: unknown })._deferred_error;
    if (!deferred || typeof deferred !== "object") return null;
    if ((deferred as { status_code?: unknown }).status_code !== 409) return null;
    detail = (deferred as { detail?: unknown }).detail;
  }
  if (!detail || typeof detail !== "object") return null;
  const tagged = detail as { code?: unknown; error?: unknown; verdict?: unknown };
  if (
    tagged.code !== MEMORY_OVERCOMMIT_CODE &&
    tagged.error !== MEMORY_OVERCOMMIT_CODE
  ) {
    return null;
  }
  return parseLoadVerdict(tagged.verdict);
}

const GIB = 1024 ** 3;

/** "19.6 GiB": one decimal, labelled as the memory rows beside it and the backend's messages are. */
export function formatVerdictGb(value: number | null): string {
  return `${Math.max(0, (value ?? 0) / GIB).toFixed(1)} GiB`;
}

type LoadVerdictKey = Extract<TranslationKey, `loadVerdict.${string}`>;
type Translate = (
  key: LoadVerdictKey,
  values?: Record<string, string | number>,
) => string;

function gpuLabel(t: Translate, indices: number[]): string {
  if (indices.length === 0) return t("loadVerdict.gpuAny");
  if (indices.length === 1) return t("loadVerdict.gpuOne", { index: indices[0] });
  return t("loadVerdict.gpuMany", { indices: indices.join(", ") });
}

const LEVEL_KEYS: Record<LoadVerdictLevel, LoadVerdictKey> = {
  full_gpu: "loadVerdict.level.fullGpu",
  fits_barely: "loadVerdict.level.fitsBarely",
  partial_gpu: "loadVerdict.level.partialGpu",
  cpu: "loadVerdict.level.cpu",
  disk_streaming: "loadVerdict.level.diskStreaming",
  likely_too_large: "loadVerdict.level.likelyTooLarge",
  unknown: "loadVerdict.level.unknown",
};

/** "Probably won't fit": the short label a headline leads with. */
export function loadVerdictLabel(verdict: LoadVerdict, t: Translate): string {
  return t(LEVEL_KEYS[verdict.level]);
}

/** One sentence with the numbers: "Needs ~19.6 GB on GPU 0; 6.1 GB is free." */
export function loadVerdictSentence(verdict: LoadVerdict, t: Translate): string {
  const gpus = gpuLabel(t, verdict.gpuIndices);
  const need = formatVerdictGb(verdict.gpuNeedBytes);
  const free = formatVerdictGb(verdict.gpuFreeBytes);
  switch (verdict.reason) {
    case "fits":
    case "forced_gpu_overflow":
      return t("loadVerdict.needsOnGpu", { need, gpus, free });
    case "tight":
      return t("loadVerdict.tight", { need, gpus, free });
    case "context_fitted":
      return t("loadVerdict.contextFitted", { gpus });
    case "spills_to_ram":
      return t("loadVerdict.spillsToRam", {
        spill: formatVerdictGb(
          Math.max(0, (verdict.gpuNeedBytes ?? 0) - (verdict.gpuFreeBytes ?? 0)),
        ),
        gpus,
      });
    case "cpu_only":
      return t("loadVerdict.cpuOnly", {
        need: formatVerdictGb(verdict.ramNeedBytes),
      });
    case "streams_from_disk":
      return t("loadVerdict.streamsFromDisk");
    case "runtime_overflow":
      return t("loadVerdict.runtimeOverflow", {
        need: formatVerdictGb(verdict.runtimeBytes),
      });
    case "resident_unattributed":
      return t("loadVerdict.residentUnattributed");
    case "kv_unsized":
    case "no_estimate":
    case "no_gpu_reading":
      return t("loadVerdict.unknownFit");
    default:
      // A reason added on the backend after this bundle shipped: its own sentence is English,
      // which beats saying nothing.
      return verdict.message || t("loadVerdict.unknownFit");
  }
}

/** "Another app is using 17.0 GB." -- named when the backend could name it. */
export function loadVerdictOtherApps(
  verdict: LoadVerdict,
  t: Translate,
): string | null {
  if (verdict.otherAppsNote) {
    return t("loadVerdict.otherAppsNamed", { holders: verdict.otherAppsNote });
  }
  if (verdict.otherAppsBytes) {
    return t("loadVerdict.otherApps", {
      other: formatVerdictGb(verdict.otherAppsBytes),
    });
  }
  return null;
}

/** Tone for a headline: the three levels that cost the user something get color. */
export function loadVerdictTone(
  level: LoadVerdictLevel,
): "ok" | "warn" | "danger" | "muted" {
  switch (level) {
    case "full_gpu":
      return "ok";
    case "fits_barely":
    case "partial_gpu":
    case "disk_streaming":
      return "warn";
    case "likely_too_large":
      return "danger";
    default:
      return "muted";
  }
}
