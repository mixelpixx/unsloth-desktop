// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The memory guardrail's "Load anyway": a 409 from /load becomes one question, asked inside
 *  loadModel so every surface gets it, answered by one dialog at the app root. */

import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import {
  answerMemoryOvercommitConsent,
  registerMemoryOvercommitHost,
  requestMemoryOvercommitConsent,
  useMemoryOvercommitStore,
} from "../src/features/chat/memory-overcommit-consent.ts";
import {
  formatVerdictGb,
  loadVerdictOtherApps,
  loadVerdictSentence,
  loadVerdictTone,
  memoryOvercommitVerdict,
  parseLoadVerdict,
} from "../src/lib/load-verdict.ts";
import { readSrc } from "./helpers/kit.ts";

const CHAT_API = readSrc("features/chat/api/chat-api.ts");
const ROOT = readSrc("app/routes/__root.tsx");
const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const ESTIMATE_ROW = readSrc("features/model-picker/components/memory-estimate-row.tsx");
const MEMORY_ESTIMATE = readSrc("features/model-picker/api/memory-estimate.ts");
const MODEL_MEMORY_SECTION = readSrc("features/settings/components/model-memory-section.tsx");

const GIB = 1024 ** 3;

/** The /load 409 detail for the field case: another LLM app holding 17 GB of a 24 GB card. */
const WIRE_VERDICT = {
  level: "likely_too_large",
  reason: "forced_gpu_overflow",
  message: "Needs ~19.6 GiB on GPU 0; 6.1 GiB is free. Other programs are using 17.0 GiB.",
  gpu_need_bytes: 19.6 * GIB,
  gpu_free_bytes: 6.1 * GIB,
  gpu_total_bytes: 24 * GIB,
  gpu_indices: [0],
  other_apps_bytes: 17 * GIB,
  other_apps_note: null,
  ram_need_bytes: null,
  ram_free_bytes: 64 * GIB,
  runtime_bytes: 1.6 * GIB,
  headroom_bytes: -13.5 * GIB,
  needs_confirmation: true,
  mode: "balanced",
};
const DETAIL = {
  error: "memory_overcommit",
  code: "memory_overcommit",
  message: "This model probably won't fit: ...",
  verdict: WIRE_VERDICT,
};

const verdict = parseLoadVerdict(WIRE_VERDICT);
assert.ok(verdict);

let unregister: (() => void) | null = null;
afterEach(() => {
  unregister?.();
  unregister = null;
  useMemoryOvercommitStore.setState({ pending: [], hosts: 0 });
});

function pendingIds(): number[] {
  return useMemoryOvercommitStore.getState().pending.map((entry) => entry.id);
}

const request = { modelLabel: "unsloth/Qwen3-32B-GGUF", verdict };

test("with no dialog mounted the question is declined, not left hanging", async () => {
  assert.equal(await requestMemoryOvercommitConsent(request), "cancel");
  assert.deepEqual(pendingIds(), []);
});

test("the answer on screen resolves the waiting load", async () => {
  unregister = registerMemoryOvercommitHost();
  const answer = requestMemoryOvercommitConsent(request);
  const [id] = pendingIds();
  assert.equal(useMemoryOvercommitStore.getState().pending[0]?.modelLabel, request.modelLabel);
  answerMemoryOvercommitConsent(id, "load");
  assert.equal(await answer, "load");
  assert.deepEqual(pendingIds(), []);
});

test("an abort cancels the open question and clears the dialog", async () => {
  unregister = registerMemoryOvercommitHost();
  const controller = new AbortController();
  const answer = requestMemoryOvercommitConsent(request, controller.signal);
  assert.equal(pendingIds().length, 1);
  controller.abort();
  assert.equal(await answer, "cancel");
  assert.deepEqual(pendingIds(), []);
});

test("an already-aborted load never shows the dialog", async () => {
  unregister = registerMemoryOvercommitHost();
  const controller = new AbortController();
  controller.abort();
  assert.equal(await requestMemoryOvercommitConsent(request, controller.signal), "cancel");
  assert.deepEqual(pendingIds(), []);
});

test("two refusals at once are asked in order, one dialog at a time", async () => {
  unregister = registerMemoryOvercommitHost();
  const first = requestMemoryOvercommitConsent(request);
  const second = requestMemoryOvercommitConsent({ ...request, modelLabel: "other" });
  const [a, b] = pendingIds();
  answerMemoryOvercommitConsent(a, "cancel");
  assert.deepEqual(pendingIds(), [b]);
  answerMemoryOvercommitConsent(b, "load");
  assert.equal(await first, "cancel");
  assert.equal(await second, "load");
});

test("unmounting the last dialog cancels questions nobody can answer now", async () => {
  unregister = registerMemoryOvercommitHost();
  const answer = requestMemoryOvercommitConsent(request);
  unregister();
  unregister = null;
  assert.equal(await answer, "cancel");
});

test("the refusal is recognised as a real 409 and as a deferred one after the padding", () => {
  assert.equal(memoryOvercommitVerdict(409, { detail: DETAIL })?.level, "likely_too_large");
  const deferred = { _deferred_error: { status_code: 409, detail: DETAIL } };
  assert.equal(memoryOvercommitVerdict(200, deferred)?.reason, "forced_gpu_overflow");
  // Anything else is not this refusal: active generations, a cancelled load, a 500.
  assert.equal(
    memoryOvercommitVerdict(409, { detail: { error: "active_generations", message: "x" } }),
    null,
  );
  assert.equal(memoryOvercommitVerdict(409, { detail: "Model load cancelled" }), null);
  assert.equal(
    memoryOvercommitVerdict(200, { _deferred_error: { status_code: 500, detail: DETAIL } }),
    null,
  );
  assert.equal(memoryOvercommitVerdict(200, { model: "x" }), null);
});

test("a verdict with a level this bundle does not know is no verdict", () => {
  assert.equal(parseLoadVerdict({ ...WIRE_VERDICT, level: "maybe" }), null);
  assert.equal(parseLoadVerdict(null), null);
});

test("the dialog body reads the backend's numbers", () => {
  const t = (key: string, values?: Record<string, string | number>) =>
    `${key}${values ? JSON.stringify(values) : ""}`;
  assert.equal(formatVerdictGb(19.6 * GIB), "19.6 GiB");
  assert.equal(
    loadVerdictSentence(verdict, t),
    'loadVerdict.needsOnGpu{"need":"19.6 GiB","gpus":"loadVerdict.gpuOne{\\"index\\":0}","free":"6.1 GiB"}',
  );
  assert.equal(loadVerdictOtherApps(verdict, t), 'loadVerdict.otherApps{"other":"17.0 GiB"}');
  const named = parseLoadVerdict({ ...WIRE_VERDICT, other_apps_note: "LM Studio.exe (PID 1) ~17.0 GB on GPU 0" });
  assert.ok(named);
  assert.match(loadVerdictOtherApps(named, t) ?? "", /otherAppsNamed.*LM Studio\.exe/);
  // An unknown reason falls back to the backend's own sentence rather than to nothing.
  const novel = parseLoadVerdict({ ...WIRE_VERDICT, reason: "brand_new_reason" });
  assert.ok(novel);
  assert.equal(loadVerdictSentence(novel, t), WIRE_VERDICT.message);
  assert.equal(loadVerdictTone("likely_too_large"), "danger");
  assert.equal(loadVerdictTone("full_gpu"), "ok");
});

function bodyOf(source: string, header: string): string {
  const start = source.indexOf(header);
  assert.ok(start >= 0, `${header} is no longer defined`);
  let depth = 0;
  for (let i = source.indexOf("{", source.indexOf(")", start)); i < source.length; i += 1) {
    if (source[i] === "{") depth += 1;
    else if (source[i] === "}" && --depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`unbalanced ${header}`);
}

test("the question lives inside loadModel, so every surface gets it", () => {
  const body = bodyOf(CHAT_API, "export async function loadModel(");
  const detect = body.indexOf("memoryOvercommitVerdict(response.status, body)");
  const ask = body.indexOf("requestMemoryOvercommitConsent(");
  const retry = body.indexOf("response = await postLoad(true)");
  assert.ok(detect >= 0 && ask > detect && retry > ask, "detect, ask, then retry");
  // The retry is the same request with the override set, and only after a yes.
  assert.match(body, /\.\.\.\(allowMemoryOvercommit \? \{ allow_memory_overcommit: true \} : \{\}\)/);
  assert.match(body.slice(ask, retry), /if \(decision !== "load"\)/);
  // A No is the cancellation marker every caller already reads, not a failure.
  assert.match(body.slice(ask, retry), /unslothUserCancelled: true/);
  // The retried body still goes through the padded-response check.
  assert.match(body, /parsedBodyOrThrow<LoadModelResponse>\(response, body, "Model load"\)/);
});

test("the dialog is mounted once, at the app root", () => {
  assert.match(ROOT, /<MemoryOvercommitDialog \/>/);
  assert.equal(ROOT.match(/<MemoryOvercommitDialog/g)?.length, 1);
});

test("declining is not reported as a failed load", () => {
  assert.match(RUNTIME, /if \(isUserCancelledLoad\(err\)\) \{\s*\/\/[^\n]*\n[^\n]*\n\s*toast\.dismiss\(toastId\);/);
  assert.match(RUNTIME, /if \(isUserCancelledLoad\(error\)\) \{/);
});

test("the load page prefers the backend verdict and keeps its own reading as the fallback", () => {
  assert.match(MEMORY_ESTIMATE, /verdict: parseLoadVerdict\(body\.verdict\)/);
  assert.match(ESTIMATE_ROW, /const verdict = estimate\.verdict \?\? null;/);
  assert.match(ESTIMATE_ROW, /const shownAdvisory = verdict && !precisionNote \? null : advisory;/);
});

test("the guardrail setting sits with the model memory settings", () => {
  assert.match(MODEL_MEMORY_SECTION, /<LoadGuardrailsRow \/>/);
});

test("the dialog only tells the user to close another app when one is named", () => {
  const dialog = readSrc("features/chat/components/memory-overcommit-dialog.tsx");
  assert.match(
    dialog,
    /: otherApps\s*\?\s*t\("loadVerdict\.dialogHint"\)\s*:[\s\S]*?t\("loadVerdict\.dialogHintNoOtherApps"\)/,
  );
});
