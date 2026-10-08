// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A model switch asks "Load anyway?" before it unloads the model it would have to put back. */

import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import {
  answerMemoryOvercommitConsent,
  registerMemoryOvercommitHost,
  useMemoryOvercommitStore,
} from "../src/features/chat/memory-overcommit-consent.ts";
import { confirmFitBeforeUnload } from "../src/features/chat/utils/load-fit-preflight.ts";
import type { LoadVerdict } from "../src/lib/load-verdict.ts";
import { readSrc } from "./helpers/kit.ts";

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const CHAT_API = readSrc("features/chat/api/chat-api.ts");

const TOO_LARGE: LoadVerdict = {
  level: "likely_too_large",
  reason: "forced_gpu_overflow",
  message: "Needs ~38.0 GiB on GPU 0; 23.5 GiB is free.",
  gpuNeedBytes: 38 * 1024 ** 3,
  gpuFreeBytes: 23.5 * 1024 ** 3,
  gpuTotalBytes: 24 * 1024 ** 3,
  gpuIndices: [0],
  otherAppsBytes: null,
  otherAppsNote: null,
  ramNeedBytes: null,
  ramFreeBytes: null,
  runtimeBytes: null,
  headroomBytes: null,
  needsConfirmation: true,
  mode: "balanced",
};

let unregister: (() => void) | null = null;
afterEach(() => {
  unregister?.();
  unregister = null;
  useMemoryOvercommitStore.setState({ pending: [], hosts: 0 });
});

async function nextTick(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test("a load that fits, or an estimate with no verdict, asks nothing", async () => {
  unregister = registerMemoryOvercommitHost();
  assert.equal(await confirmFitBeforeUnload(async () => ({ verdict: null }), "m"), false);
  assert.equal(
    await confirmFitBeforeUnload(
      async () => ({ verdict: { ...TOO_LARGE, level: "full_gpu", needsConfirmation: false } }),
      "m",
    ),
    false,
  );
  assert.deepEqual(useMemoryOvercommitStore.getState().pending, []);
});

test("an estimate that fails does not block the switch: /load still guards", async () => {
  unregister = registerMemoryOvercommitHost();
  const allowed = await confirmFitBeforeUnload(async () => {
    throw new Error("estimate route down");
  }, "m");
  assert.equal(allowed, false);
});

test("Load anyway lets the switch go ahead with the overcommit allowed", async () => {
  unregister = registerMemoryOvercommitHost();
  const asking: boolean[] = [];
  const answer = confirmFitBeforeUnload(async () => ({ verdict: TOO_LARGE }), "big model", {
    onAsking: (value) => asking.push(value),
  });
  await nextTick();
  const [pending] = useMemoryOvercommitStore.getState().pending;
  assert.equal(pending?.modelLabel, "big model");
  answerMemoryOvercommitConsent(pending.id, "load");
  assert.equal(await answer, true);
  // The toast says it is waiting while the dialog is up, and goes back after.
  assert.deepEqual(asking, [true, false]);
});

test("Cancel stops the switch before anything is unloaded, as a user cancel", async () => {
  unregister = registerMemoryOvercommitHost();
  const answer = confirmFitBeforeUnload(async () => ({ verdict: TOO_LARGE }), "big model");
  await nextTick();
  answerMemoryOvercommitConsent(useMemoryOvercommitStore.getState().pending[0].id, "cancel");
  await assert.rejects(answer, (error: unknown) => {
    assert.equal((error as { unslothUserCancelled?: boolean }).unslothUserCancelled, true);
    return true;
  });
});

test("the question is asked before the switch unloads the outgoing model", () => {
  const ask = RUNTIME.indexOf("await confirmFitBeforeUnload(");
  const unload = RUNTIME.indexOf("await unloadModel({ model_path: currentCheckpoint });");
  const reset = RUNTIME.indexOf("speculativeType: persistedSpeculativeType,");
  const load = RUNTIME.indexOf("const loadResponse = await loadModel({");
  assert.ok(ask > 0 && unload > ask, "ask before the preliminary unload");
  // The per-model store reset belongs to the new model, so a Cancel must find it untouched.
  assert.ok(reset > unload && load > reset);
  assert.match(
    RUNTIME,
    /\.\.\.\(allowMemoryOvercommit \? \{ allow_memory_overcommit: true \} : \{\}\),/,
  );
  // Only where the switch itself unloads first; otherwise /load refuses before its own eviction.
  assert.match(
    RUNTIME,
    /currentCheckpoint && !keepsOthers && !forceCancelActive && !touchesOnlySelected/,
  );
});

test("the load toast stops claiming progress while either question is open", () => {
  assert.match(RUNTIME, /onAsking: \(asking\) => loadToastAskingRef\.current\?\.\(asking\)/);
  assert.match(
    RUNTIME,
    /onMemoryOvercommitQuestion: \(asking\) => loadToastAskingRef\.current\?\.\(asking\)/,
  );
  assert.match(RUNTIME, /"Waiting for your answer…"/);
  assert.match(
    CHAT_API,
    /options\?\.onMemoryOvercommitQuestion\?\.\(true\);[\s\S]*?finally \{\s*options\?\.onMemoryOvercommitQuestion\?\.\(false\);/,
  );
});
