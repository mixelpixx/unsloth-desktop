// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The training fit planner: GPU target <-> gpu_ids, the /api/train/estimate parser, the bar,
// the verdict sentence, the Hardware row, and the wiring that keeps a failed estimate from
// ever standing between the user and Start.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type TrainingFitEstimate,
  type TrainingGpuDevice,
  formatFitGiB,
  gpuIdsForTrainingTarget,
  isTrainingGpuTarget,
  parseTrainingFitEstimate,
  resolveTrainingGpuTarget,
  trainingFitBarGeometry,
  trainingFitLine,
  trainingGpuTargetOptions,
  trainingHardwareSummary,
  trainingStartNeedsFitConfirm,
} from "../src/features/training/lib/training-fit.ts";
import type { TrainingConfigState } from "../src/features/training/types/config.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { buildTrainingEstimatePayload, buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);

const TWO_3090S: TrainingGpuDevice[] = [
  { index: 0, name: "NVIDIA GeForce RTX 3090", memoryTotalGb: 24 },
  { index: 1, name: "NVIDIA GeForce RTX 3090", memoryTotalGb: 24 },
];

const formatTarget = (ids: number[]) =>
  ids.length ? ids.map((id) => `GPU ${id}`).join(" + ") : "Auto";

function estimate(overrides: Record<string, unknown> = {}): TrainingFitEstimate {
  return parseTrainingFitEstimate({
    verdict: "fits",
    reason: null,
    required_gb: 14.2,
    estimation_mode: "detailed",
    breakdown: {
      model_weights_gb: 10,
      lora_adapters_gb: 0.2,
      optimizer_states_gb: 0.4,
      gradients_gb: 0.2,
      activations_gb: 2,
      cuda_overhead_gb: 1.4,
      total_gb: 14.2,
    },
    selection_mode: "auto",
    gpu_ids: [0],
    usable_gb: 23.6,
    min_per_gpu_gb: null,
    gpus: [
      { index: 0, name: null, total_gb: 24, free_gb: 23.6, selected: true },
      { index: 1, name: null, total_gb: 24, free_gb: 23.9, selected: false },
    ],
    suggestion: null,
    ...overrides,
  });
}

test("GPU targets: Auto, each GPU, then all of them; nothing to pick on one card", () => {
  assert.deepEqual(trainingGpuTargetOptions(TWO_3090S), [
    "auto",
    "gpu:0",
    "gpu:1",
    "all",
  ]);
  assert.deepEqual(trainingGpuTargetOptions(TWO_3090S.slice(0, 1)), []);
  assert.deepEqual(trainingGpuTargetOptions([]), []);
});

test("a GPU target maps to gpu_ids, and Auto sends none", () => {
  assert.equal(gpuIdsForTrainingTarget("auto", TWO_3090S), null);
  assert.deepEqual(gpuIdsForTrainingTarget("gpu:1", TWO_3090S), [1]);
  assert.deepEqual(gpuIdsForTrainingTarget("all", TWO_3090S), [0, 1]);
  // A remembered pick naming a card this host no longer has falls back to Auto, never to [].
  assert.equal(gpuIdsForTrainingTarget("gpu:3", TWO_3090S), null);
  assert.equal(gpuIdsForTrainingTarget("all", TWO_3090S.slice(0, 1)), null);
  assert.equal(gpuIdsForTrainingTarget("gpu:0", []), null);
  assert.equal(resolveTrainingGpuTarget("gpu:3", TWO_3090S), "auto");
  assert.equal(resolveTrainingGpuTarget(42, TWO_3090S), "auto");
  assert.equal(isTrainingGpuTarget("gpu:-1"), false);
  assert.equal(isTrainingGpuTarget("gpu:1"), true);
});

test("the estimate parser downgrades a sized verdict with no size behind it", () => {
  assert.equal(estimate().verdict, "fits");
  // 1e999 parses to Infinity: that is no measurement, so no confident "fits".
  const bogus = estimate({ required_gb: Number.POSITIVE_INFINITY });
  assert.equal(bogus.verdict, "unknown");
  assert.equal(bogus.reason, "estimate_unavailable");
  assert.equal(parseTrainingFitEstimate(null).verdict, "unknown");
  assert.equal(parseTrainingFitEstimate({ verdict: "great" }).verdict, "unknown");
  // A breakdown with a missing part is dropped whole rather than drawn short.
  assert.equal(estimate({ breakdown: { model_weights_gb: 1 } }).breakdown, null);
  // A suggestion only rides along with an "exceeds".
  const suggestion = {
    kind: "qlora",
    required_gb: 9,
    verdict: "fits",
    gpu_ids: [0],
    batch_size: null,
  };
  assert.equal(estimate({ suggestion }).suggestion, null);
  assert.equal(
    estimate({ verdict: "exceeds", suggestion }).suggestion?.kind,
    "qlora",
  );
});

test("the verdict line names the target and its free memory", () => {
  const fits = trainingFitLine(estimate(), formatTarget);
  assert.equal(fits.key, "trainingFit.fits");
  assert.deepEqual(fits.params, {
    target: "GPU 0",
    used: "14.2 GiB",
    free: "23.6 GiB",
  });
  assert.equal(
    trainingFitLine(estimate({ verdict: "tight" }), formatTarget).key,
    "trainingFit.tight",
  );
  const exceeds = trainingFitLine(
    estimate({ verdict: "exceeds", required_gb: 31 }),
    formatTarget,
  );
  assert.equal(exceeds.key, "trainingFit.exceeds");
  assert.deepEqual(exceeds.params, {
    required: "31.0 GiB",
    target: "GPU 0",
    free: "23.6 GiB",
  });
});

test("a per-GPU overflow names the fullest card in the split", () => {
  const line = trainingFitLine(
    estimate({
      verdict: "exceeds",
      reason: "per_gpu_minimum",
      gpu_ids: [0, 1],
      usable_gb: 40,
      min_per_gpu_gb: 12,
      gpus: [
        { index: 0, total_gb: 24, free_gb: 20, selected: true },
        { index: 1, total_gb: 24, free_gb: 8, selected: true },
      ],
    }),
    formatTarget,
  );
  assert.equal(line.key, "trainingFit.exceedsPerGpu");
  assert.deepEqual(line.params, {
    perGpu: "12.0 GiB",
    target: "GPU 1",
    free: "8.0 GiB",
  });
});

test("every unknown reason says Start still works, in its own words", () => {
  const keyFor = (reason: string | null) =>
    trainingFitLine(
      estimate({ verdict: "unknown", reason, required_gb: null }),
      formatTarget,
    ).key;
  assert.equal(keyFor("estimate_unavailable"), "trainingFit.unknownEstimate");
  assert.equal(keyFor("invalid_gpu_ids"), "trainingFit.unknownGpus");
  assert.equal(keyFor("no_gpu_telemetry"), "trainingFit.unknownTelemetry");
  assert.equal(keyFor("estimate_failed"), "trainingFit.unknownFailed");
  assert.equal(keyFor(null), "trainingFit.unknownFailed");
});

test("the bar stacks the estimator's parts and shows an overflow past the free line", () => {
  const fits = estimate();
  assert.ok(fits.breakdown);
  const geometry = trainingFitBarGeometry(fits.breakdown, 23.6);
  assert.deepEqual(
    geometry.segments.map((segment) => segment.part),
    [
      "modelWeights",
      "loraAdapters",
      "optimizerStates",
      "gradients",
      "activations",
      "cudaOverhead",
    ],
  );
  // Fits: the bar spans the free memory, so the parts fill 14.2 / 23.6 of it.
  const filled = geometry.segments.reduce((sum, segment) => sum + segment.pct, 0);
  assert.ok(Math.abs(filled - (14.2 / 23.6) * 100) < 1e-9);
  assert.equal(geometry.capacityPct, 100);

  // Exceeds: the bar spans the requirement and the free line sits inside it.
  const over = trainingFitBarGeometry({ ...fits.breakdown, total: 31 }, 23.6);
  assert.ok(Math.abs(over.capacityPct - (23.6 / 31) * 100) < 1e-9);
  // A zero part is left out rather than drawn as a sliver.
  const noOptimizer = trainingFitBarGeometry(
    { ...fits.breakdown, optimizerStates: 0 },
    23.6,
  );
  assert.equal(
    noOptimizer.segments.some((segment) => segment.part === "optimizerStates"),
    false,
  );
});

test("Start asks first only when the estimate positively says exceeds", () => {
  assert.equal(trainingStartNeedsFitConfirm(null), false);
  assert.equal(trainingStartNeedsFitConfirm(estimate()), false);
  assert.equal(trainingStartNeedsFitConfirm(estimate({ verdict: "tight" })), false);
  assert.equal(
    trainingStartNeedsFitConfirm(estimate({ verdict: "unknown", required_gb: null })),
    false,
  );
  assert.equal(
    trainingStartNeedsFitConfirm(estimate({ verdict: "exceeds", required_gb: 40 })),
    true,
  );
});

test("two GPUs read as two cards, not one with their summed memory", () => {
  assert.deepEqual(trainingHardwareSummary(TWO_3090S, "auto"), {
    name: "2 × NVIDIA GeForce RTX 3090",
    pinnedIndex: null,
    memory: "2 × 24 GiB",
    memoryTitle: "2 × 24 GiB",
  });
  assert.deepEqual(trainingHardwareSummary(TWO_3090S, "gpu:1"), {
    name: "NVIDIA GeForce RTX 3090",
    pinnedIndex: 1,
    memory: "24 GiB",
    memoryTitle: "24 GiB",
  });
  assert.deepEqual(trainingHardwareSummary(TWO_3090S.slice(0, 1), "auto"), {
    name: "NVIDIA GeForce RTX 3090",
    pinnedIndex: null,
    memory: "24 GiB",
    memoryTitle: "24 GiB",
  });
  const mixed = trainingHardwareSummary(
    [TWO_3090S[0], { index: 1, name: "RTX 4060", memoryTotalGb: 8 }],
    "all",
  );
  assert.deepEqual(mixed, {
    name: "NVIDIA GeForce RTX 3090 + RTX 4060",
    pinnedIndex: null,
    memory: "24 GiB + 8.0 GiB",
    memoryTitle: "24 GiB + 8 GiB",
  });
  assert.equal(trainingHardwareSummary([], "auto"), null);
  assert.equal(formatFitGiB(Number.NaN), "0 GiB");
});

const CONFIG: TrainingConfigState = {
  ...initialTrainingConfigState,
  selectedModel: "unsloth/Llama-3.1-8B",
  trainingMethod: "lora",
  contextLength: 4096,
  batchSize: 4,
  loraRank: 32,
  datasetSource: "huggingface",
  dataset: "unsloth/test",
};

test("the estimate prices the fields Start sends, and the run the backend will load", () => {
  const payload = buildTrainingEstimatePayload(CONFIG, {
    hfToken: null,
    gpuIds: [1],
    fourBitAvailable: true,
  });
  const start = buildTrainingStartPayload(CONFIG, null);
  for (const key of [
    "model_name",
    "training_type",
    "load_in_4bit",
    "max_seq_length",
    "batch_size",
    "lora_r",
    "target_modules",
    "gradient_checkpointing",
    "optim",
  ] as const) {
    assert.deepEqual(payload[key], start[key], key);
  }
  assert.deepEqual(payload.gpu_ids, [1]);
  // The latest-transformers sidecar trains 16-bit whatever the method says.
  const sidecar = buildTrainingEstimatePayload(
    { ...CONFIG, trainingMethod: "qlora" },
    { hfToken: null, gpuIds: null, fourBitAvailable: false },
  );
  assert.equal(sidecar.load_in_4bit, false);
  assert.equal(sidecar.four_bit_available, false);
});

test("Start sends gpu_ids only for an explicit target", () => {
  assert.equal("gpu_ids" in buildTrainingStartPayload(CONFIG, null), false);
  assert.equal("gpu_ids" in buildTrainingStartPayload(CONFIG, null, null), false);
  assert.equal("gpu_ids" in buildTrainingStartPayload(CONFIG, null, []), false);
  assert.deepEqual(buildTrainingStartPayload(CONFIG, null, [0, 1]).gpu_ids, [0, 1]);
});

test("the start flow reads the visible target, and Start never waits on the estimate", () => {
  const start = readSrc("features/training/lib/start-fresh-training-run.ts");
  assert.match(
    start,
    /buildTrainingStartPayload\(\s*attempt\.config,\s*hfToken,\s*selectedTrainingGpuIds\(\),\s*\)/,
  );
  const cta = readSrc("features/studio/wizard/start-training-cta.tsx");
  // Only a settled "exceeds" asks; a loading or failed estimate starts straight away.
  assert.match(
    cta,
    /fitStatus === "ready" && trainingStartNeedsFitConfirm\(fitEstimate\)/,
  );
  assert.match(cta, /<AlertDialog open=\{confirmOverflow\}/);
  assert.match(cta, /t\("trainingFit\.startAnyway"\)/);
  assert.equal(cta.match(/\bstart\(\);/g)?.length, 2, "start from the click or the confirm only");
  assert.doesNotMatch(cta, /disabled=\{[^}]*fit/i, "the estimate must never disable Start");
});

test("the run preview describes cards, not their sum", () => {
  const card = readSrc("features/studio/wizard/run-preview-card.tsx");
  assert.match(card, /trainingHardwareSummary\(fit\.allDevices, fit\.target\)/);
  // Upstream's two rows: names on Hardware, per-card memory on VRAM, the sum only as a labelled fallback.
  assert.match(card, /const vramLabel = hardwareSummary\s*\?\s*hardwareSummary\.memory/);
  assert.match(card, /title=\{vramTitle\}\s*value=\{vramLabel\}/);
  // One upgrade-notice fetch shared by the notice and the estimate.
  assert.equal(card.match(/useTrainingTransformersUpgradeNotice\(\)/g)?.length, 1);
});

test("the panel labels its select and bar, and says Both is not faster", () => {
  const panel = readSrc("features/studio/wizard/training-fit-panel.tsx");
  assert.match(panel, /<label\s+htmlFor=\{selectId\}/);
  assert.match(panel, /<SelectTrigger\s+id=\{selectId\}/);
  assert.match(panel, /role="img"\s+aria-label=\{t\("trainingFit\.barLabel", \{ verdict: label \}\)\}/);
  assert.match(panel, /target === "all" \?[\s\S]{0,200}trainingFit\.targetAllHint/);
  const en = readSrc("i18n/locales/en.ts");
  assert.match(en, /targetAllHint: "Fits bigger models by splitting layers across GPUs\. Not faster\."/);
});

test("no method change happens silently: each comes with an Undo", () => {
  const panel = readSrc("features/studio/wizard/training-fit-panel.tsx");
  // The suggestion is a button; the switch happens in its handler, with an Undo toast.
  assert.match(
    panel,
    /setTrainingMethod\("qlora"\);\s*toast\.success\(t\("trainingFit\.switchedToQlora"\), \{\s*action: \{\s*label: undoLabel/,
  );
  assert.match(panel, /onClick=\{switchToQlora\}/);
  // The model-selection heuristic still picks LoRA or QLoRA, but now says so and offers it back.
  const store = readSrc("features/training/stores/training-config-store.ts");
  assert.match(
    store,
    /if \(method !== previousMethod\) \{\s*notifyAutoTrainingMethodSwitch\(method,/,
  );
  assert.match(
    store,
    /action: \{ label: translate\("shell\.sections\.undo"\), onClick: undo \}/,
  );
});
