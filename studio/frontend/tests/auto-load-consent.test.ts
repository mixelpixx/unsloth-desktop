// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A send with no model loaded asks before loading or downloading one the user did not pick. */

import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import {
  AUTO_LOAD_LAST_MODEL_KEY,
  answerAutoLoadConsent,
  formatAutoLoadSize,
  registerAutoLoadConsentHost,
  requestAutoLoadConsent,
  setAutoLoadsLastModelWithoutAsking,
  useAutoLoadConsentStore,
} from "../src/features/chat/auto-load-consent.ts";
import { readSrc } from "./helpers/kit.ts";

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const CHAT_PAGE = readSrc("features/chat/chat-page.tsx");

const memory = new Map<string, string>();
(globalThis as { localStorage?: unknown }).localStorage = {
  getItem: (key: string) => memory.get(key) ?? null,
  setItem: (key: string, value: string) => void memory.set(key, value),
  removeItem: (key: string) => void memory.delete(key),
};

const smallest = {
  reason: "smallest" as const,
  modelLabel: "unsloth/Qwen3-27B-GGUF (Q4_K_M)",
  sizeBytes: 16_500_000_000,
};

let unregister: (() => void) | null = null;
afterEach(() => {
  unregister?.();
  unregister = null;
  memory.clear();
  useAutoLoadConsentStore.setState({ pending: [], hosts: 0 });
});

function pendingIds(): number[] {
  return useAutoLoadConsentStore.getState().pending.map((entry) => entry.id);
}

test("with no dialog mounted a send declines instead of loading silently", async () => {
  assert.equal(await requestAutoLoadConsent(smallest), "cancel");
  assert.deepEqual(pendingIds(), []);
});

test("the answer on screen resolves the waiting send", async () => {
  unregister = registerAutoLoadConsentHost();
  const answer = requestAutoLoadConsent(smallest);
  const [id] = pendingIds();
  assert.equal(useAutoLoadConsentStore.getState().pending[0]?.modelLabel, smallest.modelLabel);
  answerAutoLoadConsent(id, "load");
  assert.equal(await answer, "load");
  assert.deepEqual(pendingIds(), []);
});

test("Stop while the question is open cancels it and clears the dialog", async () => {
  unregister = registerAutoLoadConsentHost();
  const controller = new AbortController();
  const answer = requestAutoLoadConsent(smallest, controller.signal);
  assert.equal(pendingIds().length, 1);
  controller.abort();
  assert.equal(await answer, "cancel");
  assert.deepEqual(pendingIds(), []);
});

test("an already-stopped send never shows the dialog", async () => {
  unregister = registerAutoLoadConsentHost();
  const controller = new AbortController();
  controller.abort();
  assert.equal(await requestAutoLoadConsent(smallest, controller.signal), "cancel");
  assert.deepEqual(pendingIds(), []);
});

test("two panes asking at once are answered in order, one dialog at a time", async () => {
  unregister = registerAutoLoadConsentHost();
  const first = requestAutoLoadConsent(smallest);
  const second = requestAutoLoadConsent({ ...smallest, modelLabel: "other" });
  const [a, b] = pendingIds();
  answerAutoLoadConsent(a, "choose");
  assert.deepEqual(pendingIds(), [b]);
  answerAutoLoadConsent(b, "load");
  assert.equal(await first, "choose");
  assert.equal(await second, "load");
});

test("a second answer to the same question is ignored", async () => {
  unregister = registerAutoLoadConsentHost();
  const answer = requestAutoLoadConsent(smallest);
  const [id] = pendingIds();
  answerAutoLoadConsent(id, "load");
  answerAutoLoadConsent(id, "cancel");
  assert.equal(await answer, "load");
});

test("unmounting the last dialog cancels questions nobody can answer now", async () => {
  unregister = registerAutoLoadConsentHost();
  const answer = requestAutoLoadConsent(smallest);
  unregister();
  unregister = null;
  assert.equal(await answer, "cancel");
  assert.deepEqual(pendingIds(), []);
});

test("the opt-in skips the question only for the model used last", async () => {
  unregister = registerAutoLoadConsentHost();
  setAutoLoadsLastModelWithoutAsking(true);
  assert.equal(memory.get(AUTO_LOAD_LAST_MODEL_KEY), "1");
  assert.equal(
    await requestAutoLoadConsent({ ...smallest, reason: "last-used" }),
    "load",
  );
  assert.deepEqual(pendingIds(), []);

  // A model the user never chose still asks, opt-in or not.
  const answer = requestAutoLoadConsent(smallest);
  assert.equal(pendingIds().length, 1);
  answerAutoLoadConsent(pendingIds()[0], "cancel");
  assert.equal(await answer, "cancel");

  setAutoLoadsLastModelWithoutAsking(false);
  assert.equal(memory.has(AUTO_LOAD_LAST_MODEL_KEY), false);
});

test("sizes read as GB or MB, and unknown sizes are left out", () => {
  assert.equal(formatAutoLoadSize(16_500_000_000), "16.5 GB");
  assert.equal(formatAutoLoadSize(420_000_000), "420 MB");
  assert.equal(formatAutoLoadSize(0), null);
});

function bodyOf(source: string, header: string): string {
  const start = source.indexOf(header);
  assert.ok(start >= 0, `${header} is no longer defined`);
  let depth = 0;
  for (let i = source.indexOf("{", start + header.length - 1); i < source.length; i += 1) {
    if (source[i] === "{") depth += 1;
    else if (source[i] === "}" && --depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`unbalanced ${header}`);
}

test("a cached candidate is asked about after the guard passes it and before /load", () => {
  const body = bodyOf(ADAPTER, "async function loadAutoLoadCandidate(");
  const guard = body.indexOf("canAutoLoadRecordingFailures(");
  const ask = body.indexOf("confirmAutoLoad({");
  const load = body.indexOf("await loadModel({");
  assert.ok(guard >= 0 && ask > guard, "ask only about a model the guard accepts");
  assert.ok(load > ask, "nothing may load before the user agrees");
  assert.match(body, /reason: remembered \? "last-used" : "smallest"/);
  assert.match(ADAPTER, /loadAutoLoadCandidate\(candidate, isRemembered\)/);
});

test("the starter model is asked about before any byte is downloaded", () => {
  const preflight = ADAPTER.indexOf("model_path: DEFAULT_CHAT_MODEL_REPO,");
  const ask = ADAPTER.indexOf("confirmAutoLoad({", preflight);
  const download = ADAPTER.indexOf("await ensureDefaultModelDownloaded(", preflight);
  assert.ok(preflight >= 0 && ask > preflight && download > ask);
  assert.match(ADAPTER.slice(ask, download), /"default-download"/);
});

test("a declined load ends the sweep without an error toast or a starter download", () => {
  const confirm = bodyOf(ADAPTER, "async function confirmAutoLoad(");
  assert.match(confirm, /options\?\.abortSignal\?\.throwIfAborted\(\)/);
  assert.match(confirm, /autoLoadCancelled = true/);
  const declined = ADAPTER.indexOf("if (consent.declined) {");
  const capTail = ADAPTER.indexOf("// The cap gates the default download too");
  assert.ok(declined >= 0 && declined < capTail, "handled before the failure toast and starter path");
});

test("Stop cancels an auto-load already POSTed, scoped to that attempt", () => {
  const helper = bodyOf(ADAPTER, "function cancelAutoLoadOnAbort(");
  assert.match(helper, /cancel_load_request_id: loadRequestId/);
  assert.equal(
    ADAPTER.match(/(?<!cancel_)load_request_id: (?:loadRequestId|defaultLoadRequestId),/g)?.length,
    2,
  );
  assert.match(ADAPTER, /\.finally\(releaseLoadCancel\)/);
  assert.match(ADAPTER, /\.finally\(releaseDefaultLoadCancel\)/);
});

test("the chat page mounts the question", () => {
  assert.match(
    CHAT_PAGE,
    /active && \(\s*<AutoLoadConsentDialog onChooseModel=\{chooseModelInsteadOfAutoLoad\} \/>/,
  );
});
