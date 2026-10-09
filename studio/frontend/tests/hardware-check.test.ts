// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Resources > Hardware check: the payload parser, how each finding is worded, what
// "Apply recommended" switches on, the Tensor Parallelism warning, the GPU strip's link badge and
// the training warning. The pure halves are imported directly; the wiring (where each one mounts)
// is read from the shipped source, since the node suite has no DOM.

import assert from "node:assert/strict";
import test from "node:test";

import {
  findingMessage,
  locationKey,
  optionsToApply,
  parseHardwareCheckStatus,
  skipReasonKey,
  tensorSplitConcern,
} from "../src/features/hardware-check/hardware-check-model.ts";
import {
  linkBadge,
  normalizeResourceSnapshot,
} from "../src/features/gpu-resources/resources-model.ts";
import { parseTrainingFitEstimate } from "../src/features/training/lib/training-fit.ts";
import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

type Tree = { readonly [key: string]: string | Tree };

function message(path: string): string | undefined {
  let node: string | Tree | undefined = en as unknown as Tree;
  for (const part of path.split(".")) {
    if (!node || typeof node === "string") return undefined;
    node = node[part];
  }
  return typeof node === "string" ? node : undefined;
}

function placeholders(text: string): string[] {
  return [...text.matchAll(/\{([a-zA-Z0-9_]+)\}/g)].map((m) => m[1]).sort();
}

/** The payload GET /api/hardware-check returned on the reference box (two RTX 3090s, one wired
 *  x1 through the chipset, Windows without peer access). */
function thisMachine(overrides: Record<string, unknown> = {}) {
  return {
    settings: {
      auto_run: true,
      prefer_fast_link: false,
      avoid_tensor_split: true,
      warn_training_slow_link: false,
    },
    up_to_date: true,
    state: { running: false, phase: null, progress: 0, trigger: null, last_skip: null, last_error: null },
    result: {
      schema: 1,
      finished_at: "2026-10-09T16:04:13Z",
      trigger: "manual",
      duration_ms: 6234,
      gpus: [
        {
          index: 0,
          name: "NVIDIA GeForce RTX 3090",
          status: "measured",
          reason: null,
          link: { gen_current: 4, gen_max: 4, width_current: 16, width_max: 16 },
          link_idle: { gen_current: 4, gen_max: 4, width_current: 16, width_max: 16 },
          h2d_gibs: 24.848,
          d2h_gibs: 24.431,
        },
        {
          index: 1,
          name: "NVIDIA GeForce RTX 3090",
          status: "measured",
          reason: null,
          link: { gen_current: 4, gen_max: 4, width_current: 1, width_max: 16 },
          link_idle: { gen_current: 1, gen_max: 4, width_current: 1, width_max: 16 },
          h2d_gibs: 1.483,
          d2h_gibs: 1.534,
        },
      ],
      pairs: [{ a: 0, b: 1, peer_ab: false, peer_ba: false, copy_ab_gibs: 1.467, copy_ba_gibs: 1.515 }],
      storage: [
        { key: "hf_cache", path: "D:\\hf", drive: "D:", bus_type: "SATA", media_type: "SSD", model: "M500DC" },
      ],
      findings: [
        {
          id: "slow_link",
          severity: "warning",
          values: { gpu: 1, width: 1, width_max: 16, gen: 4, h2d_gibs: 1.5, best_gpu: 0, best_h2d_gibs: 24.8, ratio: 17 },
          text: "GPU 1 runs at PCIe x1 (of x16): ...",
        },
        {
          id: "no_peer_access",
          severity: "warning",
          values: { a: 0, b: 1, copy_gibs: 1.5, windows: true },
          text: "GPU 0 and GPU 1 cannot access each other directly (Windows): ...",
        },
        {
          id: "storage_slow",
          severity: "info",
          values: { location: "hf_cache", drive: "D:", kind: "sata", model: "M500DC" },
          text: "The Hugging Face cache (D:) is on a SATA SSD; ...",
        },
      ],
      recommended: { prefer_fast_link: true, avoid_tensor_split: true, warn_training_slow_link: true },
      slow_gpus: [1],
    },
    ...overrides,
  };
}

// ── Parsing ──────────────────────────────────────────────────────

test("the payload parses into the section's model, malformed rows dropped", () => {
  const status = parseHardwareCheckStatus(thisMachine());
  assert.equal(status.upToDate, true);
  assert.equal(status.settings.avoid_tensor_split, true);
  assert.equal(status.result?.gpus.length, 2);
  assert.deepEqual(status.result?.gpus[1].link, { genCurrent: 4, genMax: 4, widthCurrent: 1, widthMax: 16 });
  assert.equal(status.result?.gpus[1].h2dGibs, 1.483);
  assert.deepEqual(status.result?.slowGpus, [1]);
  assert.equal(status.result?.pairs[0].peerAb, false);
  assert.equal(status.result?.findings.length, 3);

  const junk = parseHardwareCheckStatus({
    settings: { prefer_fast_link: "yes", auto_run: false },
    up_to_date: "maybe",
    state: { running: true, progress: 7 },
    result: { gpus: [{ name: "no index" }, "x"], findings: [{ severity: "warning" }] },
  });
  assert.equal(junk.settings.prefer_fast_link, false, "only a boolean switches an option");
  assert.equal(junk.settings.auto_run, false);
  assert.equal(junk.upToDate, null);
  assert.equal(junk.state.progress, 1, "progress is clamped");
  assert.deepEqual(junk.result?.gpus, []);
  assert.deepEqual(junk.result?.findings, []);
  assert.equal(parseHardwareCheckStatus(null).result, null);
});

// ── Wording ──────────────────────────────────────────────────────

test("every finding is worded by an English message whose placeholders the values fill", () => {
  const findings = [
    ...(parseHardwareCheckStatus(thisMachine()).result?.findings ?? []),
    { id: "slow_link", severity: "warning" as const, values: { gpu: 1, width: null, h2d_gibs: 1.5, best_gpu: 0, best_h2d_gibs: 24.9, ratio: 17 }, text: "" },
    { id: "slow_link", severity: "warning" as const, values: { gpu: 0, width: 4, width_max: 16, h2d_gibs: 3.1 }, text: "" },
    { id: "no_peer_access", severity: "warning" as const, values: { a: 0, b: 1, copy_gibs: 1.5, windows: false }, text: "" },
    { id: "storage_slow", severity: "warning" as const, values: { location: "temp", drive: "E:", kind: "usb" }, text: "" },
    { id: "storage_slow", severity: "warning" as const, values: { location: "studio_home", drive: "E:", kind: "hdd" }, text: "" },
    { id: "gpu_skipped", severity: "info" as const, values: { gpu: 1, reason: "low_free_memory", free_gib: 0.3 }, text: "" },
    { id: "gpu_skipped", severity: "info" as const, values: { gpu: 1, reason: "not_visible" }, text: "" },
    { id: "gpu_skipped", severity: "info" as const, values: { gpu: 1, reason: "probe_failed" }, text: "" },
    { id: "gpu_probe_failed", severity: "warning" as const, values: { reason: "timed out" }, text: "" },
    { id: "all_good", severity: "ok" as const, values: { gpus: 2 }, text: "" },
  ];
  const keys = new Set<string>();
  for (const finding of findings) {
    const worded = findingMessage(finding);
    assert.ok(worded, finding.id);
    const text = message(`hardwareCheck.${worded.key}`);
    assert.ok(text, `hardwareCheck.${worded.key} is in en`);
    assert.deepEqual(placeholders(text), Object.keys(worded.values).sort(), worded.key);
    keys.add(worded.key);
  }
  assert.equal(keys.size, 13, "every finding variant has its own message");
  assert.equal(findingMessage({ id: "from_a_newer_backend", severity: "info", values: {}, text: "x" }), null);
});

test("the reference box reads as the measured numbers", () => {
  const [slow, peer, storage] = parseHardwareCheckStatus(thisMachine()).result?.findings ?? [];
  assert.deepEqual(findingMessage(slow), {
    key: "findings.slowLink",
    values: { gpu: 1, width: 1, widthMax: 16, h2d: "1.5", best: "24.8", bestGpu: 0, ratio: 17 },
  });
  assert.deepEqual(findingMessage(peer), {
    key: "findings.noPeerAccessWindows",
    values: { a: 0, b: 1, copy: "1.5" },
  });
  assert.deepEqual(findingMessage(storage), {
    key: "findings.storageSata",
    values: { location: "hf_cache", drive: "D:" },
  });
  for (const key of ["studio_home", "hf_cache", "temp", "something_new"]) {
    assert.ok(message(`hardwareCheck.${locationKey(key)}`), key);
  }
});

test("every skip reason has a line", () => {
  for (const reason of ["already_running", "training_active", "model_loading", "generation_active", "new"]) {
    const key = skipReasonKey(reason);
    assert.ok(key && message(`hardwareCheck.${key}`), reason);
  }
  assert.equal(skipReasonKey(null), null);
});

// ── Apply recommended ────────────────────────────────────────────

test("apply recommended offers exactly the recommended options still off", () => {
  assert.deepEqual(optionsToApply(parseHardwareCheckStatus(thisMachine())), [
    "prefer_fast_link",
    "warn_training_slow_link",
  ]);
  assert.deepEqual(optionsToApply(parseHardwareCheckStatus(thisMachine({ up_to_date: false }))), []);
  assert.deepEqual(optionsToApply(null), []);
});

// ── The Tensor Parallelism warning ───────────────────────────────

test("tensor parallelism is warned about with the measured numbers, only when the option is on", () => {
  const status = parseHardwareCheckStatus(thisMachine());
  assert.deepEqual(tensorSplitConcern(status, null), {
    slow: [{ gpu: 1, width: 1, h2d: "1.5", best: "24.8", bestGpu: 0 }],
    noPeer: [{ a: 0, b: 1, copy: "1.5" }],
  });
  assert.equal(tensorSplitConcern(status, [0]), null, "one GPU spans no link");
  const off = parseHardwareCheckStatus(
    thisMachine({ settings: { auto_run: true, avoid_tensor_split: false } }),
  );
  assert.equal(tensorSplitConcern(off, null), null);
  assert.equal(tensorSplitConcern(parseHardwareCheckStatus(thisMachine({ up_to_date: false })), null), null);
  assert.equal(tensorSplitConcern(null, null), null);
});

test("the warning sits under the Tensor Parallelism switch and only shows when it is on", () => {
  const page = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(page, /import \{ TensorParallelLinkWarning \} from "@\/features\/hardware-check";/);
  assert.match(
    page,
    /\{!isDiffusion && config\.tensorParallel \? \(\s*<TensorParallelLinkWarning gpuIds=\{config\.selectedGpuIds \?\? null\} \/>/,
  );
  const warning = readSrc("features/hardware-check/tensor-parallel-link-warning.tsx");
  assert.match(warning, /role="note"/);
  assert.doesNotMatch(warning, /disabled|onCheckedChange/, "a warning, never a block");
});

// ── The GPU strip badge ──────────────────────────────────────────

test("the strip badges a slow card with its measured link, and nothing else", () => {
  const snapshot = normalizeResourceSnapshot({
    gpus: [
      {
        index: 0, total_bytes: 24, used_bytes: 1, free_bytes: 23,
        link: { width: 16, width_max: 16, gen: 4, h2d_gibs: 24.8, best_gpu: 0, best_h2d_gibs: 24.8, slow: false },
      },
      {
        index: 1, total_bytes: 24, used_bytes: 1, free_bytes: 23,
        link: { width: 1, width_max: 16, gen: 4, h2d_gibs: 1.5, best_gpu: 0, best_h2d_gibs: 24.8, slow: true },
      },
      { index: 2, total_bytes: 24, used_bytes: 1, free_bytes: 23 },
    ],
  });
  const [fast, slow, unmeasured] = snapshot.gpus;
  assert.equal(linkBadge(fast), null);
  assert.deepEqual(linkBadge(slow), { width: 1, widthMax: 16, h2d: "1.5", best: "24.8", bestGpu: 0 });
  assert.equal("link" in unmeasured, false, "no link until the check measured the card");
  assert.equal(linkBadge(unmeasured), null);
  const tooltip = message("hardwareCheck.badge.tooltip") ?? "";
  assert.deepEqual(placeholders(tooltip), Object.keys(linkBadge(slow) ?? {}).sort());
  const strip = readSrc("features/gpu-resources/resources-strip.tsx");
  assert.equal(strip.match(/<LinkBadge gpu=\{gpu\} t=\{t\} \/>/g)?.length, 2, "in the strip and the panel");
});

// ── Training ─────────────────────────────────────────────────────

test("the training estimate carries the slow-link warning, malformed ones dropped", () => {
  const estimate = parseTrainingFitEstimate({
    verdict: "fits",
    required_gb: 14,
    usable_gb: 23,
    gpu_ids: [1],
    slow_link: {
      gpus: [{ index: 1, width: 1, width_max: 16, h2d_gibs: 1.5 }, { width: 4 }],
      best_gpu: 0,
      best_h2d_gibs: 24.9,
      offloaded_gradient_checkpointing: true,
    },
  });
  assert.deepEqual(estimate.slowLink, {
    gpus: [{ index: 1, width: 1, widthMax: 16, h2dGibs: 1.5 }],
    bestGpu: 0,
    bestH2dGibs: 24.9,
    offloadedGradientCheckpointing: true,
  });
  assert.equal(parseTrainingFitEstimate({ verdict: "fits", slow_link: { gpus: [] } }).slowLink, null);
  assert.equal(parseTrainingFitEstimate({ verdict: "fits" }).slowLink, null);
  const panel = readSrc("features/studio/wizard/training-fit-panel.tsx");
  assert.match(panel, /<SlowLinkWarning estimate=\{estimate\} devices=\{devices\} \/>/);
  for (const key of ["slowLink", "offloaded", "pickFast"]) {
    assert.ok(message(`hardwareCheck.training.${key}`), key);
  }
});

// ── Settings wiring ──────────────────────────────────────────────

test("the section lives in Resources, is searchable, and every key it uses exists", () => {
  const tab = readSrc("features/settings/tabs/resources-tab.tsx");
  assert.match(tab, /<HardwareCheckSection \/>/);
  const search = readSrc("features/settings/settings-search.ts");
  for (const key of [
    "hardwareCheck.title",
    "hardwareCheck.options.preferFastLink.label",
    "hardwareCheck.options.avoidTensorSplit.label",
    "hardwareCheck.options.warnTraining.label",
    "hardwareCheck.autoRun.label",
  ]) {
    assert.ok(search.includes(`"${key}"`), key);
  }
  assert.ok(search.includes('"hardwareCheck.title": "hardwareCheck.keywords"'));
  const section = readSrc("features/settings/components/hardware-check-section.tsx");
  const used = [...section.matchAll(/"(hardwareCheck\.[A-Za-z.]+)"/g)].map((m) => m[1]);
  assert.ok(used.length > 30);
  for (const key of used) {
    assert.ok(message(key), key);
  }
  // Defaults off: the switches read the stored settings, never a hard-coded true.
  assert.match(section, /checked=\{status\?\.settings\[key\] \?\? false\}/);
});
