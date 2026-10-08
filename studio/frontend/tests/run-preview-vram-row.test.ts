// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Run preview Hardware row truncates, so a long GPU name used to cut off the
 * VRAM figure appended to it ("AMD Radeon AI PRO R9700 · 31.86 …"). VRAM gets its
 * own row; this pins that it is not folded back into the name.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/studio/wizard/run-preview-card.tsx", import.meta.url),
  "utf8",
);

function metaRow(labelKey: string): string {
  const start = source.indexOf(`label={t("${labelKey}")}`);
  assert.ok(start >= 0, `no MetaRow labelled ${labelKey}`);
  return source.slice(start, source.indexOf("/>", start));
}

test("Hardware row carries the GPU name only, and wraps", () => {
  const row = metaRow("studio.preview.hardware");
  assert.doesNotMatch(row, /memoryTotalGb/);
  assert.match(row, /\bwrap\b/);
});

test("VRAM has its own rounded row with the exact value on hover", () => {
  const row = metaRow("studio.preview.vram");
  // Per card from the training fit summary ("2 × 24 GiB"): memoryTotalGb sums every GPU, and a
  // lone "48 GiB" reads as one card no run on a two-card host can use whole.
  assert.match(row, /title=\{vramTitle\}/);
  assert.match(row, /value=\{vramLabel\}/);
  assert.match(source, /: `\$\{Math\.round\(gpu\.memoryTotalGb\)\} GiB`;/);
  assert.match(source, /: `\$\{gpu\.memoryTotalGb\} GiB`;/);
});
