// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What each subsystem's own store says, as activity snapshots (lib/activity-store.ts). Pure and
// React-free, with no `@/` imports outside type positions, so node --test imports it as is. The
// inputs are the few fields read, typed structurally, so a test needs no store to build one.

import type {
  ActivityDownloadRetry,
  ActivityEntry,
  ActivityLogTarget,
  ActivitySnapshot,
  ActivityState,
} from "@/lib/activity-store";
import type {
  ModelLoadOutcome,
  ModelRuntime,
} from "@/lib/model-lifecycle-events";

/** Training, exports, downloads and recipes write their failures to the server log. */
export const SERVER_LOGS: ActivityLogTarget = {
  family: "server",
  sourcePath: null,
};

/** A finished download this large is a long one: worth a native notification. */
export const LONG_DOWNLOAD_BYTES = 1024 ** 3;
/** A load that settled faster than this is not history worth keeping: picking a cached model in
 *  chat would otherwise fill Recent with one row per pick. Failures are always kept. */
export const SHORT_MODEL_LOAD_MS = 10_000;

export const RECIPE_ROUTE = "/data-recipes/$recipeId";

// ── Downloads (the Hub download manager store) ─────────────────────

export type DownloadJobLike = {
  key: string;
  kind: "model" | "dataset";
  repoId: string;
  variant: string | null;
  inventoryKind?: "model" | "gguf";
  state: "idle" | "running" | "complete" | "error" | "cancelled" | "cancelling";
  downloadedBytes: number;
  completedBytes: number;
  expectedBytes: number;
  error: string | null;
  startedAt: number;
  external?: boolean;
  scopedFiles?: string[];
  checkpoint?: boolean;
  presentation?: { label: string };
  activity?: string;
};

export type DownloadProgressLike = {
  expectedBytes: number;
  downloadedBytes: number;
  fraction: number;
};

/** One id per transfer, not per repo: downloading the same repo again is a new row. */
export function downloadEntryId(
  job: Pick<DownloadJobLike, "key" | "startedAt">,
): string {
  return `download:${job.key}@${job.startedAt}`;
}

export function downloadTitle(
  job: Pick<DownloadJobLike, "repoId" | "variant" | "presentation">,
): string {
  if (job.presentation?.label)
    return `${job.presentation.label} · ${job.repoId}`;
  // "@scope" is a staging slot, not a name anyone picked.
  return job.variant && !job.variant.startsWith("@")
    ? `${job.repoId} · ${job.variant}`
    : job.repoId;
}

/**
 * The request that starts this download again, or null when there is none to make: a dictation
 * model is fetched by its sidecar rather than the Hub API, and a scoped job persisted without its
 * file list cannot say what it was fetching.
 */
export function downloadRetryFor(
  job: DownloadJobLike,
): ActivityDownloadRetry | null {
  if (job.external) return null;
  const base: ActivityDownloadRetry = {
    kind: job.kind,
    repoId: job.repoId,
    variant: job.variant,
    expectedBytes: Math.max(0, job.expectedBytes),
    ...(job.inventoryKind ? { inventoryKind: job.inventoryKind } : {}),
  };
  if (!job.variant?.startsWith("@")) return base;
  if (!job.scopedFiles?.length) return null;
  return {
    ...base,
    scopeId: job.variant.slice(1),
    files: [...job.scopedFiles],
    ...(job.checkpoint !== undefined ? { checkpoint: job.checkpoint } : {}),
  };
}

const DOWNLOAD_STATE: Record<DownloadJobLike["state"], ActivityState> = {
  idle: "active",
  running: "active",
  cancelling: "active",
  complete: "done",
  error: "failed",
  cancelled: "cancelled",
};

export function downloadSnapshots<Job extends DownloadJobLike>(
  jobs: readonly Job[],
  progressOf: (job: Job) => DownloadProgressLike,
): ActivitySnapshot[] {
  return jobs.map((job) => {
    const progress = progressOf(job);
    const state = DOWNLOAD_STATE[job.state];
    const total = Math.max(0, progress.expectedBytes);
    return {
      id: downloadEntryId(job),
      kind: "download",
      title: downloadTitle(job),
      state,
      startedAt: job.startedAt,
      detail: state === "failed" ? (job.error ?? null) : (job.activity ?? null),
      progress: total > 0 ? progress.fraction : null,
      meter:
        total > 0
          ? {
              unit: "bytes",
              done: Math.max(0, progress.downloadedBytes),
              total,
            }
          : null,
      route: "/hub",
      logs: SERVER_LOGS,
      ref: job.key,
      retry: downloadRetryFor(job),
      bytes: Math.max(
        job.downloadedBytes,
        job.completedBytes,
        job.expectedBytes,
        0,
      ),
      // Cancelling is already under way; a second request would only race the first.
      cancellable: job.state === "running",
    } satisfies ActivitySnapshot;
  });
}

// ── Training (the training runtime store) ──────────────────────────

export type TrainingLike = {
  jobId: string | null;
  phase: string;
  isTrainingRunning: boolean;
  message: string;
  error: string | null;
  currentStep: number;
  totalSteps: number;
  progressPercent: number;
  startModelName: string | null;
};

const TRAINING_ACTIVE_PHASES = new Set([
  "downloading_model",
  "downloading_dataset",
  "loading_model",
  "loading_dataset",
  "configuring",
  "training",
  // Steps done, worker still saving: 100% is not success yet.
  "finalizing",
]);

const TRAINING_FINISHED: Record<string, ActivityState> = {
  completed: "done",
  error: "failed",
  stopped: "cancelled",
};

/** The current run, if there is one. `fallbackModel` is the picker's model, for a run started
 *  before this tab recorded what it was training. */
export function trainingSnapshots(
  state: TrainingLike,
  fallbackModel: string | null,
  now: number,
): ActivitySnapshot[] {
  if (!state.jobId) return [];
  const finished = TRAINING_FINISHED[state.phase];
  const active =
    !finished &&
    (state.isTrainingRunning || TRAINING_ACTIVE_PHASES.has(state.phase));
  if (!active && !finished) return [];
  const counting = state.phase === "training" && state.totalSteps > 0;
  return [
    {
      id: `training:${state.jobId}`,
      kind: "training",
      title: state.startModelName ?? fallbackModel ?? "",
      state: finished ?? "active",
      // A run's own start time is not in the store; the first report stamps it.
      startedAt: now,
      detail:
        finished === "failed"
          ? (state.error ?? state.message ?? null)
          : counting
            ? null
            : state.message || null,
      progress: counting ? state.progressPercent / 100 : null,
      meter: counting
        ? { unit: "steps", done: state.currentStep, total: state.totalSteps }
        : null,
      route: "/studio",
      logs: SERVER_LOGS,
      ref: state.jobId,
    },
  ];
}

// ── Export (the export runtime store) ──────────────────────────────

export type ExportLike = {
  phase: string;
  isExporting: boolean;
  startedAt: number | null;
  summary: { baseModelName: string; methodLabel: string } | null;
  error: string | null;
  stage: string | null;
};

const EXPORT_STATE: Record<string, ActivityState> = {
  loading: "active",
  exporting: "active",
  success: "done",
  error: "failed",
  canceled: "cancelled",
};

export function exportSnapshots(
  state: ExportLike,
  progressPercent: number,
): ActivitySnapshot[] {
  const mapped = EXPORT_STATE[state.phase];
  if (!mapped || state.startedAt === null) return [];
  return [
    {
      id: `export:${state.startedAt}`,
      kind: "export",
      title: state.summary
        ? `${state.summary.baseModelName} · ${state.summary.methodLabel}`
        : "",
      state: mapped,
      startedAt: state.startedAt,
      detail: mapped === "failed" ? state.error : state.stage,
      progress: progressPercent / 100,
      route: "/export",
      logs: SERVER_LOGS,
    },
  ];
}

// ── Data recipes (the recipe executions store) ─────────────────────

export type RecipeRecordLike = {
  id: string;
  recipeId: string;
  kind: "preview" | "full";
  // biome-ignore lint/style/useNamingConvention: recipe execution schema
  run_name: string | null;
  status: string;
  createdAt: number;
  stage: string | null;
  error: string | null;
  lastEventId: number | null;
  progress: {
    done?: number | null;
    total?: number | null;
    percent?: number | null;
  } | null;
};

const RECIPE_STATE: Record<string, ActivityState> = {
  pending: "active",
  running: "active",
  active: "active",
  cancelling: "active",
  completed: "done",
  error: "failed",
  cancelled: "cancelled",
};

function recipeSignature(record: RecipeRecordLike): string {
  return JSON.stringify([
    record.status,
    record.lastEventId,
    record.stage,
    record.progress,
  ]);
}

/**
 * The recipe runs worth a row. The store also holds the recipe's saved history, which can keep a
 * run marked running from a tab that closed mid-run with nobody tracking it, so an in-progress
 * record counts only once it shows life: started this session, moving since the last look, or
 * already a row. `signatures` is what the last call saw, updated in place.
 */
export function recipeSnapshots(
  records: readonly RecipeRecordLike[],
  signatures: Map<string, string>,
  activeIds: ReadonlySet<string>,
  sessionStart: number,
): ActivitySnapshot[] {
  const snapshots: ActivitySnapshot[] = [];
  for (const record of records) {
    const state = RECIPE_STATE[record.status];
    const signature = recipeSignature(record);
    const previous = signatures.get(record.id);
    signatures.set(record.id, signature);
    if (!state) continue;
    const id = `recipe:${record.id}`;
    if (
      state === "active" &&
      !activeIds.has(id) &&
      record.createdAt < sessionStart &&
      (previous === undefined || previous === signature)
    ) {
      continue;
    }
    const done = record.progress?.done ?? null;
    const total = record.progress?.total ?? null;
    const percent = record.progress?.percent ?? null;
    snapshots.push({
      id,
      kind: "recipe",
      title: record.run_name ?? "",
      state,
      startedAt: record.createdAt,
      detail: state === "failed" ? record.error : record.stage,
      progress:
        percent !== null
          ? percent / 100
          : done !== null && total
            ? done / total
            : null,
      meter: done !== null && total ? { unit: "rows", done, total } : null,
      route: RECIPE_ROUTE,
      logs: SERVER_LOGS,
      ref: record.recipeId,
    });
  }
  return snapshots;
}

// ── Model loads (lib/model-lifecycle-events.ts) ────────────────────

const MODEL_LOAD_ROUTES: Record<ModelRuntime, string> = {
  chat: "/chat",
  tts: "/audio",
  image: "/images",
  video: "/video",
  stt: "/chat",
};

/** Dictation loads and releases on its own whenever the mic is used: not a job anyone started. */
export function tracksModelLoad(runtime: ModelRuntime): boolean {
  return runtime !== "stt";
}

export function modelLoadRoute(runtime: ModelRuntime): string {
  return MODEL_LOAD_ROUTES[runtime];
}

/** A local path reads as its file or folder name; a Hub id stays whole. */
export function modelLoadTitle(model: string | null): string {
  const raw = model?.trim() ?? "";
  if (!/^(?:[A-Za-z]:[\\/]|\\\\|\/|~[\\/])/.test(raw)) return raw;
  return raw.split(/[\\/]/).filter(Boolean).at(-1) ?? raw;
}

/** How a settled load is kept. "drop" when it says nothing worth a row: a quick success, or an
 *  end the call could not vouch for. */
export function modelLoadSettlement(
  outcome: ModelLoadOutcome | undefined,
  startedAt: number,
  now: number,
): Exclude<ActivityState, "active"> | "drop" {
  if (outcome === "failed") return "failed";
  if (outcome === "cancelled") return "cancelled";
  if (outcome === "loaded") {
    return now - startedAt >= SHORT_MODEL_LOAD_MS ? "done" : "drop";
  }
  return "drop";
}

// ── Native notifications ───────────────────────────────────────────

export type FinishedNotice =
  | "trainingDone"
  | "trainingFailed"
  | "exportDone"
  | "downloadDone";

/** Which finished jobs are long enough that someone may have walked away from them. */
export function finishedNotice(
  entry: Pick<ActivityEntry, "kind" | "state" | "bytes">,
): FinishedNotice | null {
  if (entry.kind === "training") {
    if (entry.state === "done") return "trainingDone";
    if (entry.state === "failed") return "trainingFailed";
    return null;
  }
  if (entry.kind === "export" && entry.state === "done") return "exportDone";
  if (
    entry.kind === "download" &&
    entry.state === "done" &&
    (entry.bytes ?? 0) > LONG_DOWNLOAD_BYTES
  ) {
    return "downloadDone";
  }
  return null;
}
