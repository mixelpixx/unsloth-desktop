// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  BEST_CHECKPOINT_LABEL_KEYS,
  checkpointExportSearch,
  checkpointResumeAction,
  forkedFromStep,
  formatCheckpointEpoch,
  formatCheckpointLoss,
  formatCheckpointSavedAt,
  pickBestCheckpoint,
  showRunCheckpoints,
} from "../src/features/training/lib/run-checkpoints.ts";

function source(relative: string): string {
  return readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");
}

function row(
  id: string,
  step: number | null,
  trainLoss: number | null,
  evalLoss: number | null = null,
  isFinal = false,
) {
  return {
    id,
    step,
    // biome-ignore lint/style/useNamingConvention: API schema
    is_final: isFinal,
    // biome-ignore lint/style/useNamingConvention: API schema
    train_loss: trainLoss,
    // biome-ignore lint/style/useNamingConvention: API schema
    eval_loss: evalLoss,
  };
}

function resumeRow(
  mode: "in_place" | "fork" | null,
  code: string | null = null,
  reason: string | null = null,
) {
  return {
    // biome-ignore lint/style/useNamingConvention: API schema
    resume_mode: mode,
    // biome-ignore lint/style/useNamingConvention: API schema
    resume_blocked_code: code,
    // biome-ignore lint/style/useNamingConvention: API schema
    resume_blocked_reason: reason,
  };
}

test("best checkpoint uses eval loss when at least two checkpoints were evaluated", () => {
  const best = pickBestCheckpoint([
    row("checkpoint-30", 30, 0.4, 1.3),
    row("checkpoint-20", 20, 0.6, 1.1),
    row("checkpoint-10", 10, 0.9, 1.4),
  ]);
  assert.deepEqual(best, { id: "checkpoint-20", basis: "eval_loss" });
  assert.equal(
    BEST_CHECKPOINT_LABEL_KEYS[best.basis],
    "trainingRuns.checkpoints.bestEval",
  );
});

test("best checkpoint falls back to training loss and says so", () => {
  const best = pickBestCheckpoint([
    row("checkpoint-30", 30, 0.7, 0.8),
    row("checkpoint-20", 20, 0.5, Number.NaN),
    row("checkpoint-10", 10, 0.9),
  ]);
  // One finite eval loss is no comparison, and NaN never counts.
  assert.deepEqual(best, { id: "checkpoint-20", basis: "train_loss" });
  assert.equal(
    BEST_CHECKPOINT_LABEL_KEYS.train_loss,
    "trainingRuns.checkpoints.bestTrain",
  );
});

test("best checkpoint ties go to the final save, then the earlier step", () => {
  assert.deepEqual(
    pickBestCheckpoint([
      row("checkpoint-30", 30, 0.5),
      row("final", 30, 0.5, null, true),
      row("checkpoint-20", 20, 0.6),
    ]),
    { id: "final", basis: "train_loss" },
  );
  assert.deepEqual(
    pickBestCheckpoint([row("checkpoint-30", 30, 0.5), row("checkpoint-10", 10, 0.5)]),
    { id: "checkpoint-10", basis: "train_loss" },
  );
  assert.equal(pickBestCheckpoint([row("checkpoint-10", 10, 0.5)]), null);
  assert.equal(pickBestCheckpoint([]), null);
});

test("resume action follows the server's mode and explains a refusal", () => {
  assert.deepEqual(checkpointResumeAction(resumeRow("in_place"), false), {
    kind: "in_place",
  });
  assert.deepEqual(checkpointResumeAction(resumeRow("fork"), false), {
    kind: "fork",
  });
  assert.deepEqual(checkpointResumeAction(resumeRow("fork"), true), {
    kind: "disabled",
    reasonKey: "trainingRuns.checkpoints.blocked.trainingActive",
    reason: null,
  });
  assert.deepEqual(checkpointResumeAction(resumeRow(null, "finished"), false), {
    kind: "disabled",
    reasonKey: "trainingRuns.checkpoints.blocked.finished",
    reason: null,
  });
  // The provenance gate words its own reason; it is shown as written.
  assert.deepEqual(
    checkpointResumeAction(
      resumeRow(null, "provenance", "  The exact model snapshot is gone.  "),
      false,
    ),
    {
      kind: "disabled",
      reasonKey: "trainingRuns.checkpoints.blocked.provenance",
      reason: "The exact model snapshot is gone.",
    },
  );
  assert.equal(
    (checkpointResumeAction(resumeRow(null, "something_new"), false) as {
      reasonKey: string;
    }).reasonKey,
    "trainingRuns.checkpoints.blocked.noTrainerState",
  );
});

test("export deep link names the final save by the run and others by folder", () => {
  // biome-ignore lint/style/useNamingConvention: API schema
  assert.deepEqual(checkpointExportSearch("run_a", { id: "final", is_final: true }), {
    run: "run_a",
    checkpoint: "run_a",
  });
  assert.deepEqual(
    // biome-ignore lint/style/useNamingConvention: API schema
    checkpointExportSearch("run_a", { id: "checkpoint-500", is_final: false }),
    { run: "run_a", checkpoint: "checkpoint-500" },
  );
});

test("checkpoints show only for a run that is no longer training", () => {
  for (const status of ["completed", "stopped", "error"] as const) {
    assert.equal(showRunCheckpoints(status, false), true);
    assert.equal(showRunCheckpoints(status, true), false);
  }
  for (const status of ["running", "training", "idle", "finalizing", null] as const) {
    assert.equal(showRunCheckpoints(status, false), false);
  }
});

test("formatting keeps blanks honest and follows the locale", () => {
  assert.equal(formatCheckpointLoss(null, "en"), "--");
  assert.equal(formatCheckpointLoss(Number.NaN, "en"), "--");
  assert.equal(formatCheckpointLoss(1.23456, "en"), "1.2346");
  assert.equal(formatCheckpointLoss(1.5, "de"), "1,5000");
  assert.equal(formatCheckpointEpoch(1.256, "en"), "1.26");
  assert.equal(formatCheckpointEpoch(null, "en"), "--");
  assert.equal(formatCheckpointSavedAt(null, "en"), "--");
  assert.equal(formatCheckpointSavedAt("not a date", "en"), "--");
  assert.notEqual(formatCheckpointSavedAt("2026-10-09T12:30:00+00:00", "en"), "--");
});

test("a forked run reads its source step from the stored config", () => {
  // biome-ignore lint/style/useNamingConvention: API schema
  assert.equal(forkedFromStep({ forked_from: { step: 500, run_id: "job_a" } }), 500);
  // biome-ignore lint/style/useNamingConvention: API schema
  assert.equal(forkedFromStep({ forked_from: { step: "500" } }), null);
  assert.equal(forkedFromStep({}), null);
  assert.equal(forkedFromStep(null), null);
});

test("checkpoint actions address checkpoints by id, never by path", () => {
  const api = source("../src/features/training/api/history-api.ts");
  assert.match(
    api,
    /checkpoints\/\$\{encodeURIComponent\(checkpointId\)\}/,
  );
  const fork = api.slice(api.indexOf("export async function forkTrainingRunCheckpoint"));
  const forkBody = fork.slice(0, fork.indexOf("\n}\n"));
  assert.match(forkBody, /\/fork`/);
  assert.doesNotMatch(forkBody, /body:/);
  assert.match(api, /\?confirm_final=true/);
});

test("the checkpoints table hands off to Chat, Export and a fork-then-resume", () => {
  const section = source("../src/features/studio/sections/run-checkpoints-section.tsx");
  // Chat: the Library's model hand-off, loading the checkpoint folder onto its base model.
  assert.match(section, /requestModelConfigHandoff\(\{/);
  assert.match(section, /navigate\(\{ to: "\/chat", search: \{ new: requestId \} \}\)/);
  assert.match(section, /to: "\/export",\s*search: checkpointExportSearch\(runName, checkpoint\)/);
  // An older checkpoint is copied into a new run first; only that new run is resumed.
  const fork = section.indexOf("forkTrainingRunCheckpoint(runId, checkpoint.id)");
  const resumeFork = section.indexOf("resumeTrainingRunFromHistory(fork.run_id)");
  assert.ok(fork > 0 && resumeFork > fork);
  // The final adapter needs the acknowledged checkbox before the destructive action enables.
  assert.match(section, /disabled=\{deleteTarget\?\.is_final === true && !finalAcknowledged\}/);
  assert.match(section, /confirmFinal: target\.is_final/);
  // The existing reveal helper decides whether "Open folder" is offered; copying the path always is.
  assert.match(section, /useRevealLabel\(\)/);
  assert.match(section, /copyToClipboard\(checkpoint\.path\)/);
});

test("history and the stopped current run both show the table", () => {
  const history = source("../src/features/studio/historical-training-view.tsx");
  assert.match(history, /showRunCheckpoints\(detail\.run\.status, false\) && \(\s*<RunCheckpointsSection/);
  const live = source("../src/features/studio/live-training-view.tsx");
  assert.match(
    live,
    /showRunCheckpoints\(runtime\.phase, runtime\.isTrainingRunning\) && \(/,
  );
});

test("export accepts a checkpoint deep link alongside the run", () => {
  const route = source("../src/app/routes/export.tsx");
  assert.match(route, /checkpoint:\s*typeof search\.checkpoint === "string"/);
  const page = source("../src/features/export/export-page.tsx");
  assert.match(page, /checkpoint: preselectCheckpoint/);
  assert.match(page, /pending\.run === selectedModelIdx/);
});

test("trainingRuns sits right after trainingFit in every locale", () => {
  for (const locale of [
    "en", "ar", "de", "es", "fr", "he", "hi", "it", "ja", "ko", "pt-br", "ru", "sv", "zh-CN",
  ]) {
    const text = source(`../src/i18n/locales/${locale}.ts`);
    const fit = text.indexOf("\n  trainingFit: {");
    const runs = text.indexOf("\n  trainingRuns: {");
    assert.ok(fit > 0 && runs > fit, locale);
    const between = text.slice(fit + 1, runs);
    // Nothing but trainingFit's own body before trainingRuns starts.
    assert.equal(between.match(/\n {2}[a-zA-Z]+: \{/g), null, locale);
  }
});
