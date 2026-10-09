// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type {
  CheckpointBestBasis,
  TrainingRunCheckpoint,
} from "../types/checkpoints";
import type { TrainingRunSummary } from "../types/history";
import type { TrainingPhase } from "../types/runtime";

type BestCandidate = Pick<
  TrainingRunCheckpoint,
  "id" | "step" | "is_final" | "train_loss" | "eval_loss"
>;

export type BestCheckpoint = {
  id: string;
  basis: CheckpointBestBasis;
};

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Mirrors the server's pick: lowest eval loss when at least two checkpoints were evaluated at
 * their save step, else lowest training loss. Ties go to the final save, then the earlier step.
 * The basis comes back with the pick so the label never claims eval loss it did not use. */
export function pickBestCheckpoint(
  checkpoints: readonly BestCandidate[],
): BestCheckpoint | null {
  for (const basis of ["eval_loss", "train_loss"] as const) {
    const candidates = checkpoints.filter((checkpoint) =>
      finite(checkpoint[basis]),
    );
    if (candidates.length < 2) continue;
    const best = candidates.reduce((current, next) => {
      const a = current[basis] as number;
      const b = next[basis] as number;
      if (b !== a) return b < a ? next : current;
      if (current.is_final !== next.is_final) {
        return current.is_final ? current : next;
      }
      return (next.step ?? 0) < (current.step ?? 0) ? next : current;
    });
    return { id: best.id, basis };
  }
  return null;
}

export const BEST_CHECKPOINT_LABEL_KEYS = {
  // biome-ignore lint/style/useNamingConvention: API schema
  eval_loss: "trainingRuns.checkpoints.bestEval",
  // biome-ignore lint/style/useNamingConvention: API schema
  train_loss: "trainingRuns.checkpoints.bestTrain",
} as const satisfies Record<CheckpointBestBasis, TranslationKey>;

const RESUME_BLOCKED_KEYS: Record<string, TranslationKey> = {
  // biome-ignore lint/style/useNamingConvention: API schema
  no_trainer_state: "trainingRuns.checkpoints.blocked.noTrainerState",
  finished: "trainingRuns.checkpoints.blocked.finished",
  // biome-ignore lint/style/useNamingConvention: API schema
  s3_dataset: "trainingRuns.checkpoints.blocked.s3Dataset",
  provenance: "trainingRuns.checkpoints.blocked.provenance",
  // biome-ignore lint/style/useNamingConvention: API schema
  run_active: "trainingRuns.checkpoints.blocked.runActive",
  // biome-ignore lint/style/useNamingConvention: API schema
  training_active: "trainingRuns.checkpoints.blocked.trainingActive",
};

export type CheckpointResumeAction =
  | { kind: "in_place" | "fork" }
  | { kind: "disabled"; reasonKey: TranslationKey; reason: string | null };

/** What "Resume from here" does for a row. Server-written reasons (provenance) are shown verbatim;
 * otherwise the code maps to a localized sentence. */
export function checkpointResumeAction(
  checkpoint: Pick<
    TrainingRunCheckpoint,
    "resume_mode" | "resume_blocked_code" | "resume_blocked_reason"
  >,
  trainingBusy: boolean,
): CheckpointResumeAction {
  if (checkpoint.resume_mode && trainingBusy) {
    return {
      kind: "disabled",
      reasonKey: "trainingRuns.checkpoints.blocked.trainingActive",
      reason: null,
    };
  }
  if (checkpoint.resume_mode) return { kind: checkpoint.resume_mode };
  const code = checkpoint.resume_blocked_code ?? "no_trainer_state";
  return {
    kind: "disabled",
    reasonKey:
      RESUME_BLOCKED_KEYS[code] ??
      "trainingRuns.checkpoints.blocked.noTrainerState",
    reason: checkpoint.resume_blocked_reason?.trim() || null,
  };
}

/** The Export page names a run by its output folder and a checkpoint by its folder, with the
 * run's own final save listed under the run name itself. */
export function checkpointExportSearch(
  runName: string,
  checkpoint: Pick<TrainingRunCheckpoint, "id" | "is_final">,
): { run: string; checkpoint: string } {
  return {
    run: runName,
    checkpoint: checkpoint.is_final ? runName : checkpoint.id,
  };
}

/** Only a run that is no longer training has a stable set of checkpoints to act on. Takes a
 * history status or a live phase; both name a finished run the same way. */
export function showRunCheckpoints(
  status: TrainingRunSummary["status"] | TrainingPhase | null | undefined,
  isTrainingRunning: boolean,
): boolean {
  if (isTrainingRunning) return false;
  return status === "completed" || status === "stopped" || status === "error";
}

export function formatCheckpointLoss(
  value: number | null | undefined,
  locale: string,
): string {
  if (!finite(value)) return "--";
  return value.toLocaleString(locale, {
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  });
}

export function formatCheckpointEpoch(
  value: number | null | undefined,
  locale: string,
): string {
  if (!finite(value)) return "--";
  return value.toLocaleString(locale, { maximumFractionDigits: 2 });
}

export function formatCheckpointSavedAt(
  value: string | null | undefined,
  locale: string,
): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleString(locale, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/** How a run row was seeded from another run's checkpoint, read from its stored config. */
export function forkedFromStep(config: Record<string, unknown> | null): number | null {
  const marker = config?.forked_from;
  if (!marker || typeof marker !== "object") return null;
  const step = (marker as { step?: unknown }).step;
  return typeof step === "number" && Number.isInteger(step) ? step : null;
}
