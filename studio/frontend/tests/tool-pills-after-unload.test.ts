// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Unloading a model returns the tool pills to the user's saved choice, as a fresh page does. */

import assert from "node:assert/strict";
import { test } from "node:test";

import { readSrc } from "./helpers/kit.ts";

const STORE = readSrc("features/chat/stores/chat-runtime-store.ts");

function body(header: string): string {
  const start = STORE.indexOf(header);
  assert.ok(start >= 0, `${header} is gone`);
  let depth = 0;
  for (let i = STORE.indexOf("{", start); i < STORE.length; i += 1) {
    if (STORE[i] === "{") depth += 1;
    else if (STORE[i] === "}" && --depth === 0) return STORE.slice(start, i + 1);
  }
  throw new Error(`unbalanced ${header}`);
}

test("clearCheckpoint puts the saved pills back instead of switching them all off", () => {
  const clear = body("  clearCheckpoint: () => {");
  assert.match(clear, /\.\.\.persistedToolPills\(\),/);
  for (const pill of [
    "toolsEnabled",
    "codeToolsEnabled",
    "imageToolsEnabled",
    "deepResearchEnabled",
    "mcpEnabledForChat",
    "webFetchToolsEnabled",
  ]) {
    assert.doesNotMatch(clear, new RegExp(`\\b${pill}: false,`), `${pill} is forced off again`);
  }
});

test("the saved pills are read from the same keys a fresh page starts from", () => {
  const helper = body("function persistedToolPills()");
  for (const [pill, key] of [
    ["toolsEnabled", "CHAT_TOOLS_ENABLED_KEY"],
    ["codeToolsEnabled", "CHAT_CODE_TOOLS_ENABLED_KEY"],
    ["imageToolsEnabled", "CHAT_IMAGE_TOOLS_ENABLED_KEY"],
    ["deepResearchEnabled", "CHAT_DEEP_RESEARCH_ENABLED_KEY"],
    ["mcpEnabledForChat", "CHAT_MCP_ENABLED_KEY"],
    ["webFetchToolsEnabled", "CHAT_WEB_FETCH_TOOLS_ENABLED_KEY"],
  ]) {
    const read = `${pill}: loadBool(${key}, false)`;
    assert.ok(helper.includes(read), `helper: ${read}`);
    assert.ok(STORE.split(read).length >= 3, `initial state no longer reads ${key} the same way`);
  }
});
