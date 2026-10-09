// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type CheckpointBestBasis = "eval_loss" | "train_loss";

/** in_place: the run's own resume continues from here. fork: a new run folder starts from a copy. */
export type CheckpointResumeMode = "in_place" | "fork";

export interface TrainingRunCheckpoint {
  /** "checkpoint-<step>", or "final" for the run's own save. */
  id: string;
  step: number | null;
  epoch: number | null;
  /** The last training loss logged at or before the save, and the step it was logged at. */
  train_loss: number | null;
  train_loss_step: number | null;
  /** Only an evaluation at the save step itself. */
  eval_loss: number | null;
  saved_at: string | null;
  size_bytes: number | null;
  is_final: boolean;
  is_adapter: boolean;
  path: string;
  resume_mode: CheckpointResumeMode | null;
  resume_blocked_code: string | null;
  resume_blocked_reason: string | null;
}

export interface TrainingRunCheckpointsResponse {
  run_id: string;
  /** The output folder's name: the run name the Export page knows it by. */
  run_name: string | null;
  checkpoints: TrainingRunCheckpoint[];
  best_checkpoint_id: string | null;
  best_basis: CheckpointBestBasis | null;
  total_size_bytes: number | null;
}

export interface TrainingRunCheckpointDeleteResponse {
  status: "deleted";
  checkpoint_id: string;
  freed_bytes: number | null;
}

export interface TrainingRunCheckpointForkResponse {
  run_id: string;
  output_dir_name: string;
  step: number;
}
