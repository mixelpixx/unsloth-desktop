// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  LAUNCH_SESSION_PARAM,
  extractLaunchSession,
} from "../src/features/auth/launcher-handoff-parse.ts";

const TOKEN =
  "AbCdEf0123456789_-AbCdEf0123456789_-AbCdEf0123456789_-AbCdEf01234";

test("takes the token out of the hash and leaves nothing behind", () => {
  const out = extractLaunchSession(`#${LAUNCH_SESSION_PARAM}=${TOKEN}`);
  assert.equal(out.token, TOKEN);
  assert.equal(out.remainingHash, "");
});

test("keeps unrelated hash parameters", () => {
  const out = extractLaunchSession(`#a=1&${LAUNCH_SESSION_PARAM}=${TOKEN}&b=2`);
  assert.equal(out.token, TOKEN);
  assert.equal(out.remainingHash, "#a=1&b=2");
});

test("a hash without the parameter is returned untouched", () => {
  assert.deepEqual(extractLaunchSession(""), {
    token: null,
    remainingHash: "",
  });
  assert.deepEqual(extractLaunchSession("#section"), {
    token: null,
    remainingHash: "#section",
  });
});

test("a malformed token is rejected but still removed from the address", () => {
  for (const bad of ["short", `${TOKEN}!!`, "a b c".repeat(20), ""]) {
    const out = extractLaunchSession(
      `#${LAUNCH_SESSION_PARAM}=${encodeURIComponent(bad)}`,
    );
    assert.equal(out.token, null, bad);
    assert.equal(out.remainingHash, "", bad);
  }
});

test("a token longer than any real one is rejected", () => {
  const out = extractLaunchSession(
    `#${LAUNCH_SESSION_PARAM}=${"a".repeat(500)}`,
  );
  assert.equal(out.token, null);
});
