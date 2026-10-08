// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The training fit planner's pure half: GPU target <-> gpu_ids, the /api/train/estimate
 *  parser, the fit bar's geometry and the one-line verdict. Kept free of `@/` alias imports so
 *  `tests/` can load it under `node --experimental-strip-types` (see lib/memory/format.ts);
 *  the relative imports carry explicit extensions for the same reason. */

import { formatGiB } from "../../../lib/memory/format.ts";

export type TrainingFitVerdict = "fits" | "tight" | "exceeds" | "unknown";

/** Auto, one physical GPU, or every pinnable GPU at once (a layer split, not data parallel). */
export type TrainingGpuTarget = "auto" | "all" | `gpu:${number}`;

const GPU_TARGET_RE = /^gpu:(\d{1,4})$/;

/** The parts the backend estimator itemizes, in the order the bar stacks them. Nothing here
 *  the estimator does not price: a fallback estimate carries no breakdown at all. */
export const TRAINING_FIT_PARTS = [
  "modelWeights",
  "loraAdapters",
  "optimizerStates",
  "gradients",
  "activations",
  "cudaOverhead",
] as const;

export type TrainingFitPart = (typeof TRAINING_FIT_PARTS)[number];

export type TrainingFitBreakdown = Record<TrainingFitPart, number> & {
  total: number;
};

export interface TrainingFitGpu {
  index: number;
  name: string | null;
  totalGb: number | null;
  freeGb: number | null;
  selected: boolean;
}

export interface TrainingFitSuggestion {
  kind: "qlora" | "batch_size";
  requiredGb: number;
  verdict: "fits" | "tight";
  gpuIds: number[];
  batchSize: number | null;
}

export interface TrainingFitEstimate {
  verdict: TrainingFitVerdict;
  /** Why the verdict is unknown, or which check an "exceeds" failed. */
  reason: string | null;
  requiredGb: number | null;
  breakdown: TrainingFitBreakdown | null;
  selectionMode: "auto" | "explicit";
  gpuIds: number[];
  usableGb: number | null;
  minPerGpuGb: number | null;
  gpus: TrainingFitGpu[];
  suggestion: TrainingFitSuggestion | null;
}

/** A GPU the run can be pinned to, as the GPU inventory reports it. */
export interface TrainingGpuDevice {
  index: number;
  name: string;
  memoryTotalGb: number;
}

export function isTrainingGpuTarget(value: unknown): value is TrainingGpuTarget {
  return (
    value === "auto" ||
    value === "all" ||
    (typeof value === "string" && GPU_TARGET_RE.test(value))
  );
}

/** Auto, each GPU, then all of them. Empty with fewer than two GPUs: one card is no choice. */
export function trainingGpuTargetOptions(
  devices: readonly TrainingGpuDevice[],
): TrainingGpuTarget[] {
  if (devices.length < 2) {
    return [];
  }
  return [
    "auto",
    ...devices.map((device): TrainingGpuTarget => `gpu:${device.index}`),
    "all",
  ];
}

/** The remembered target, or Auto when it names a GPU this host no longer offers. */
export function resolveTrainingGpuTarget(
  target: unknown,
  devices: readonly TrainingGpuDevice[],
): TrainingGpuTarget {
  if (!isTrainingGpuTarget(target)) {
    return "auto";
  }
  return trainingGpuTargetOptions(devices).includes(target) ? target : "auto";
}

/** gpu_ids for /start and /estimate: null leaves placement to the backend's auto-selection. */
export function gpuIdsForTrainingTarget(
  target: unknown,
  devices: readonly TrainingGpuDevice[],
): number[] | null {
  const resolved = resolveTrainingGpuTarget(target, devices);
  if (resolved === "auto") {
    return null;
  }
  if (resolved === "all") {
    return devices.map((device) => device.index);
  }
  return [Number(resolved.slice("gpu:".length))];
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function intList(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is number => typeof item === "number" && Number.isInteger(item),
      )
    : [];
}

function parseBreakdown(raw: unknown): TrainingFitBreakdown | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const source = raw as Record<string, unknown>;
  const read = (key: string): number | null => {
    const value = finiteOrNull(source[key]);
    return value !== null && value >= 0 ? value : null;
  };
  const parts = {
    modelWeights: read("model_weights_gb"),
    loraAdapters: read("lora_adapters_gb"),
    optimizerStates: read("optimizer_states_gb"),
    gradients: read("gradients_gb"),
    activations: read("activations_gb"),
    cudaOverhead: read("cuda_overhead_gb"),
    total: read("total_gb"),
  };
  // All or nothing: a bar with a missing part would draw a total that does not add up.
  if (Object.values(parts).some((value) => value === null)) {
    return null;
  }
  return parts as TrainingFitBreakdown;
}

const UNKNOWN_ESTIMATE: TrainingFitEstimate = {
  verdict: "unknown",
  reason: "estimate_failed",
  requiredGb: null,
  breakdown: null,
  selectionMode: "auto",
  gpuIds: [],
  usableGb: null,
  minPerGpuGb: null,
  gpus: [],
  suggestion: null,
};

/**
 * The /api/train/estimate body, checked rather than trusted.
 *
 * A sized verdict with no size behind it is downgraded to unknown: NaN and Infinity fail every
 * comparison, and `1e999` parses to Infinity, so a malformed body would otherwise print a
 * confident "Fits" (the same trap lib/memory/verdict.ts guards against).
 */
export function parseTrainingFitEstimate(raw: unknown): TrainingFitEstimate {
  if (!raw || typeof raw !== "object") {
    return UNKNOWN_ESTIMATE;
  }
  const body = raw as Record<string, unknown>;
  const requiredGb = finiteOrNull(body.required_gb);
  const usableGb = finiteOrNull(body.usable_gb);
  let verdict: TrainingFitVerdict =
    body.verdict === "fits" ||
    body.verdict === "tight" ||
    body.verdict === "exceeds"
      ? body.verdict
      : "unknown";
  let reason = typeof body.reason === "string" ? body.reason : null;
  if (verdict !== "unknown" && (requiredGb === null || usableGb === null)) {
    verdict = "unknown";
    reason = "estimate_unavailable";
  }
  const gpus = Array.isArray(body.gpus)
    ? body.gpus.flatMap((item): TrainingFitGpu[] => {
        if (!item || typeof item !== "object") return [];
        const gpu = item as Record<string, unknown>;
        if (typeof gpu.index !== "number" || !Number.isInteger(gpu.index)) {
          return [];
        }
        return [
          {
            index: gpu.index,
            name: typeof gpu.name === "string" ? gpu.name : null,
            totalGb: finiteOrNull(gpu.total_gb),
            freeGb: finiteOrNull(gpu.free_gb),
            selected: gpu.selected === true,
          },
        ];
      })
    : [];
  const rawSuggestion = body.suggestion as Record<string, unknown> | null;
  const suggestionRequired = finiteOrNull(rawSuggestion?.required_gb);
  const suggestion: TrainingFitSuggestion | null =
    verdict === "exceeds" &&
    rawSuggestion &&
    (rawSuggestion.kind === "qlora" || rawSuggestion.kind === "batch_size") &&
    (rawSuggestion.verdict === "fits" || rawSuggestion.verdict === "tight") &&
    suggestionRequired !== null
      ? {
          kind: rawSuggestion.kind,
          requiredGb: suggestionRequired,
          verdict: rawSuggestion.verdict,
          gpuIds: intList(rawSuggestion.gpu_ids),
          batchSize: finiteOrNull(rawSuggestion.batch_size),
        }
      : null;
  return {
    verdict,
    reason,
    requiredGb,
    breakdown: parseBreakdown(body.breakdown),
    selectionMode: body.selection_mode === "explicit" ? "explicit" : "auto",
    gpuIds: intList(body.gpu_ids),
    usableGb,
    minPerGpuGb: finiteOrNull(body.min_per_gpu_gb),
    gpus,
    suggestion,
  };
}

/** One decimal, binary units: every figure here is `bytes / 1024**3` (see lib/memory/format.ts). */
export function formatFitGiB(gib: number): string {
  if (!Number.isFinite(gib) || gib <= 0) return "0 GiB";
  return `${gib.toFixed(1)} GiB`;
}

export interface TrainingFitBarSegment {
  part: TrainingFitPart;
  gb: number;
  /** Share of the bar's full width, 0-100. */
  pct: number;
}

export interface TrainingFitBarGeometry {
  segments: TrainingFitBarSegment[];
  /** Where the free-memory line sits, 0-100. Below 100 when the run overflows it. */
  capacityPct: number;
}

/**
 * Lay the breakdown out against the target's usable memory. The bar spans whichever is larger,
 * so an overflow shows as parts running past the capacity line rather than being clipped.
 */
export function trainingFitBarGeometry(
  breakdown: TrainingFitBreakdown,
  capacityGb: number,
): TrainingFitBarGeometry {
  const capacity = Number.isFinite(capacityGb) && capacityGb > 0 ? capacityGb : 0;
  const scale = Math.max(breakdown.total, capacity);
  if (!(scale > 0)) {
    return { segments: [], capacityPct: 100 };
  }
  return {
    segments: TRAINING_FIT_PARTS.filter((part) => breakdown[part] > 0).map(
      (part) => ({
        part,
        gb: breakdown[part],
        pct: (breakdown[part] / scale) * 100,
      }),
    ),
    capacityPct: capacity > 0 ? (capacity / scale) * 100 : 100,
  };
}

export type TrainingFitLineKey =
  | "trainingFit.fits"
  | "trainingFit.tight"
  | "trainingFit.exceeds"
  | "trainingFit.exceedsPerGpu"
  | "trainingFit.unknownEstimate"
  | "trainingFit.unknownGpus"
  | "trainingFit.unknownTelemetry"
  | "trainingFit.unknownFailed";

export interface TrainingFitLine {
  key: TrainingFitLineKey;
  params: Record<string, string>;
}

/**
 * The verdict as one sentence. `formatTarget` names a GPU set ("GPU 0", "GPU 0 + GPU 1") so the
 * caller owns its translation; this owns the arithmetic.
 */
export function trainingFitLine(
  estimate: TrainingFitEstimate,
  formatTarget: (gpuIds: number[]) => string,
): TrainingFitLine {
  const target = formatTarget(estimate.gpuIds);
  const { requiredGb, usableGb } = estimate;
  if (estimate.verdict === "unknown" || requiredGb === null || usableGb === null) {
    switch (estimate.reason) {
      case "estimate_unavailable":
        return { key: "trainingFit.unknownEstimate", params: {} };
      case "invalid_gpu_ids":
        return { key: "trainingFit.unknownGpus", params: { target } };
      case "no_gpu_telemetry":
        return { key: "trainingFit.unknownTelemetry", params: {} };
      default:
        return { key: "trainingFit.unknownFailed", params: {} };
    }
  }
  if (estimate.verdict === "exceeds") {
    if (estimate.reason === "per_gpu_minimum" && estimate.minPerGpuGb !== null) {
      // Activations do not shard, so the fullest card in the split is the one that says no.
      const fullest = estimate.gpus
        .filter((gpu) => gpu.selected && gpu.freeGb !== null)
        .sort((a, b) => (a.freeGb ?? 0) - (b.freeGb ?? 0))[0];
      if (fullest) {
        return {
          key: "trainingFit.exceedsPerGpu",
          params: {
            perGpu: formatFitGiB(estimate.minPerGpuGb),
            target: formatTarget([fullest.index]),
            free: formatFitGiB(fullest.freeGb ?? 0),
          },
        };
      }
    }
    return {
      key: "trainingFit.exceeds",
      params: {
        required: formatFitGiB(requiredGb),
        target,
        free: formatFitGiB(usableGb),
      },
    };
  }
  return {
    key: estimate.verdict === "tight" ? "trainingFit.tight" : "trainingFit.fits",
    params: {
      target,
      used: formatFitGiB(requiredGb),
      free: formatFitGiB(usableGb),
    },
  };
}

/** Start asks first only when the estimate positively says it will not fit. A failed or
 *  missing estimate never stands between the user and Start. */
export function trainingStartNeedsFitConfirm(
  estimate: TrainingFitEstimate | null,
): boolean {
  return estimate?.verdict === "exceeds";
}

export type TrainingHardwareSummary =
  | { key: "trainingFit.hardwareSingle"; params: { name: string; memory: string } }
  | {
      key: "trainingFit.hardwarePinned";
      params: { index: string; name: string; memory: string };
    }
  | {
      key: "trainingFit.hardwareIdentical";
      params: { count: string; name: string; memory: string };
    }
  | { key: null; text: string };

/**
 * The run preview's Hardware row. Several GPUs used to print as one card with their summed
 * memory ("RTX 3090 · 48 GiB"), which reads as a 48 GiB card no model on this host can use
 * whole: layers split across cards, and activations need room on every one of them.
 */
export function trainingHardwareSummary(
  devices: readonly TrainingGpuDevice[],
  target: TrainingGpuTarget,
): TrainingHardwareSummary | null {
  if (devices.length === 0) {
    return null;
  }
  if (target.startsWith("gpu:")) {
    const index = Number(target.slice("gpu:".length));
    const device = devices.find((candidate) => candidate.index === index);
    if (device) {
      return {
        key: "trainingFit.hardwarePinned",
        params: {
          index: String(device.index),
          name: device.name,
          memory: formatGiB(device.memoryTotalGb),
        },
      };
    }
  }
  if (devices.length === 1) {
    return {
      key: "trainingFit.hardwareSingle",
      params: {
        name: devices[0].name,
        memory: formatGiB(devices[0].memoryTotalGb),
      },
    };
  }
  const first = devices[0];
  const identical = devices.every(
    (device) =>
      device.name === first.name &&
      formatGiB(device.memoryTotalGb) === formatGiB(first.memoryTotalGb),
  );
  if (identical) {
    return {
      key: "trainingFit.hardwareIdentical",
      params: {
        count: String(devices.length),
        name: first.name,
        memory: formatGiB(first.memoryTotalGb),
      },
    };
  }
  // Mixed cards: name each one rather than a sum no single card has.
  return {
    key: null,
    text: devices
      .map((device) => `${device.name} · ${formatGiB(device.memoryTotalGb)}`)
      .join(" + "),
  };
}
