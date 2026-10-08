// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** An idle chat page backs off its /status observer instead of polling twice a second. */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  nextServerModelPollDelay,
  SERVER_MODEL_POLL_LOADING_MS,
  SERVER_MODEL_POLL_MAX_MS,
  SERVER_MODEL_POLL_MIN_MS,
  sleepUnlessAborted,
} from "../src/features/chat/utils/server-model-poll.ts";
import { readSrc } from "./helpers/kit.ts";

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");

const idle = { loading: false, hidden: false };

test("idle polling backs off from 500 ms to a 5 s ceiling", () => {
  const delays: number[] = [];
  let delay = SERVER_MODEL_POLL_MIN_MS;
  for (let i = 0; i < 10; i += 1) {
    delay = nextServerModelPollDelay(delay, idle);
    delays.push(delay);
  }
  for (let i = 1; i < delays.length; i += 1) assert.ok(delays[i] >= delays[i - 1]);
  assert.equal(delays.at(-1), SERVER_MODEL_POLL_MAX_MS);

  // The 60 s idle window used to cost ~120 requests; now it costs a couple of dozen at most.
  let elapsed = 0;
  let requests = 0;
  delay = SERVER_MODEL_POLL_MIN_MS;
  while (elapsed < 60_000) {
    requests += 1;
    delay = nextServerModelPollDelay(delay, idle);
    elapsed += delay;
  }
  assert.ok(requests <= 20, `${requests} idle requests`);
});

test("a load in progress is watched every second, and a hidden tab waits the longest", () => {
  assert.equal(
    nextServerModelPollDelay(SERVER_MODEL_POLL_MAX_MS, { loading: true, hidden: false }),
    SERVER_MODEL_POLL_LOADING_MS,
  );
  assert.equal(
    nextServerModelPollDelay(SERVER_MODEL_POLL_MIN_MS, { loading: false, hidden: true }),
    SERVER_MODEL_POLL_MAX_MS,
  );
});

test("the wait ends early when the page unmounts", async () => {
  const controller = new AbortController();
  const started = Date.now();
  const waiting = sleepUnlessAborted(10_000, controller.signal);
  controller.abort();
  await waiting;
  assert.ok(Date.now() - started < 1_000);
  await sleepUnlessAborted(10_000, controller.signal);
});

test("the mount observer uses the backoff, not a flat 500 ms", () => {
  const wait = RUNTIME.slice(
    RUNTIME.indexOf("async function waitForServerModel("),
    RUNTIME.indexOf("function parseTrailingEpoch("),
  );
  assert.doesNotMatch(wait, /setTimeout\(resolve, 500\)/);
  assert.match(wait, /delayMs = nextServerModelPollDelay\(delayMs, \{/);
  assert.match(wait, /await sleepUnlessAborted\(delayMs, signal\);/);
});
