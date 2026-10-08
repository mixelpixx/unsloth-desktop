// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pins for actions that lose work: each one needs a confirm, a guard, or a visible error.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the training stop dialog lands on Stop and Save, and the discard button says so", () => {
  const source = readSrc("features/studio/sections/progress-section.tsx");
  assert.match(source, /onOpenAutoFocus=\{\(event\) => \{\s*event\.preventDefault\(\);\s*stopAndSaveRef\.current\?\.focus/);
  assert.match(source, /ref=\{stopAndSaveRef\}\s*onClick=\{\(\) => onRequestStop\(true\)\}/);
  assert.match(source, /onRequestStop\(false\)\}\s*>\s*\{t\("studio\.training\.stopWithoutSaving"\)\}/);
});

test("gallery Delete opens a confirm instead of deleting", () => {
  const source = readSrc("components/gallery-item-menu.tsx");
  assert.match(source, /variant="destructive" onClick=\{\(\) => setConfirmDelete\(true\)\}/);
  assert.match(source, /<AlertDialogAction variant="destructive" onClick=\{onDelete\}>/);
  assert.equal(source.match(/onClick=\{onDelete\}/g)?.length, 1, "onDelete must only fire from the confirm");
});

test("deleting a recipe asks first and reports a failure", () => {
  const source = readSrc("features/data-recipes/pages/data-recipes-page.tsx");
  assert.doesNotMatch(source, /handleDeleteRecipe\([^)]*\)\.catch\(\(\) => undefined\)/);
  assert.match(source, /<AlertDialog open=\{deleteDialogOpen\}/);
  assert.match(source, /toastError\(\s*`Couldn't delete/);
});

test("a failed save or load stops recipe autosave from looping or overwriting", () => {
  const source = readSrc("features/recipe-studio/hooks/use-recipe-persistence.ts");
  assert.match(source, /loadError !== null \|\|\s*failedSignature === currentSignature\s*\) \{\s*return;/);
  assert.match(source, /setFailedSignature\(buildSignature\(nextName, currentPayload\)\)/);
});

test("a GGUF export from a checkpoint names the run, not just the checkpoint", () => {
  const source = readSrc("features/export/export-page.tsx");
  assert.match(source, /checkpoint !== selectedModelIdx\s*\? `\$\{selectedModelIdx\}-\$\{checkpoint\}`/);
});

test("Push to Hub cannot start without a valid username and repo name", () => {
  const source = readSrc("features/export/components/export-run-panel.tsx");
  assert.match(source, /disabled=\{startBlockedReason !== null\}/);
  assert.match(source, /destination === "hub" \? hubRepoIssue\(hfUsername, modelName\) : null/);
});
