// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { McpToolEntry } from "../src/features/chat/api/mcp-servers-api.ts";
import {
  createLatestSaver,
  filterMcpTools,
  formatTokenCount,
  mcpToolCostNote,
  mcpToolSummary,
  mcpToolTotals,
  sortedNames,
  withNames,
} from "../src/features/chat/utils/mcp-tool-settings.ts";

import { readSrc, readText } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/mcp-servers-api.ts");
const DIALOG = readSrc("features/chat/chat-mcp-servers-dialog.tsx");
const TOOLS = readSrc("features/chat/mcp-server-tools.tsx");
const MCP_CLIENT = readText("../../backend/core/inference/mcp_client.py");
const MODELS = readText("../../backend/models/mcp_servers.py");
const ROUTES = readText("../../backend/routes/mcp_servers.py");

function tool(name: string, overrides: Partial<McpToolEntry> = {}): McpToolEntry {
  return {
    name,
    title: null,
    summary: "",
    description: "",
    enabled: true,
    ask: false,
    tokens: 100,
    ...overrides,
  };
}

const SSH_TOOLS = [
  tool("ssh_exec", { summary: "Run a command over SSH.", tokens: 320 }),
  tool("ssh_upload_file", { summary: "Upload a file with SFTP.", tokens: 410 }),
  tool("vps_logs", { title: "Server logs", description: "Tail journald or nginx logs.", tokens: 270 }),
  tool("list_hosts", { tokens: 90 }),
];

test("token counts read coarsely, as the estimates they are", () => {
  assert.equal(formatTokenCount(0), "0");
  assert.equal(formatTokenCount(312.4), "312");
  assert.equal(formatTokenCount(999), "999");
  assert.equal(formatTokenCount(4000), "4K");
  assert.equal(formatTokenCount(4050), "4.1K");
  assert.equal(formatTokenCount(9960), "10K");
  assert.equal(formatTokenCount(41_300), "41K");
  assert.equal(formatTokenCount(-5), "0");
});

test("the filter matches every word against name, title and description, ignoring case", () => {
  assert.deepEqual(
    filterMcpTools(SSH_TOOLS, "").map((t) => t.name),
    SSH_TOOLS.map((t) => t.name),
  );
  assert.deepEqual(filterMcpTools(SSH_TOOLS, "SSH").map((t) => t.name), [
    "ssh_exec",
    "ssh_upload_file",
  ]);
  assert.deepEqual(filterMcpTools(SSH_TOOLS, "  logs   nginx ").map((t) => t.name), [
    "vps_logs",
  ]);
  assert.deepEqual(filterMcpTools(SSH_TOOLS, "server").map((t) => t.name), ["vps_logs"]);
  assert.deepEqual(filterMcpTools(SSH_TOOLS, "ssh nothing"), []);
});

test("the header counts the tools still on and what they cost each request", () => {
  const off = new Set(["ssh_upload_file", "gone_tool"]);
  const totals = mcpToolTotals(SSH_TOOLS, off);
  assert.deepEqual(totals, { on: 3, total: 4, tokens: 320 + 270 + 90 });
  assert.equal(mcpToolSummary(totals), "3 of 4 tools on · ~680 tokens per request");
  assert.equal(
    mcpToolSummary(mcpToolTotals(SSH_TOOLS, new Set(SSH_TOOLS.map((t) => t.name)))),
    "0 of 4 tools on · adds nothing to requests",
  );
  assert.equal(
    mcpToolSummary(mcpToolTotals([tool("one", { tokens: 5200 })], new Set())),
    "1 of 1 tool on · ~5.2K tokens per request",
  );
});

test("the cost note says how the figures were made and their share of the window", () => {
  assert.equal(mcpToolCostNote(680, false, null), "Estimated from each tool's schema.");
  assert.equal(
    mcpToolCostNote(4096, true, 32768),
    "Counted with the loaded model's tokenizer: 13% of the loaded model's 33K context.",
  );
  assert.equal(
    mcpToolCostNote(100, false, 131072),
    "Estimated from each tool's schema: <1% of the loaded model's 131K context.",
  );
  assert.equal(mcpToolCostNote(0, true, 8192), "Counted with the loaded model's tokenizer.");
});

test("name sets change as new sets and save sorted", () => {
  const base = new Set(["b", "a"]);
  const added = withNames(base, ["c", "a"], true);
  const removed = withNames(added, ["a", "zz"], false);
  assert.deepEqual(sortedNames(base), ["a", "b"]);
  assert.deepEqual(sortedNames(added), ["a", "b", "c"]);
  assert.deepEqual(sortedNames(removed), ["b", "c"]);
  // The original is never mutated: a pending switch state and the server's row stay apart.
  assert.deepEqual(sortedNames(base), ["a", "b"]);
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

test("saves run one at a time and a burst of flips collapses into its newest state", async () => {
  const writes: { value: string; done: ReturnType<typeof deferred<string>> }[] = [];
  const saver = createLatestSaver((value: string) => {
    const done = deferred<string>();
    writes.push({ value, done });
    return done.promise;
  });
  assert.equal(saver.busy(), false);

  const first = saver.save("a");
  const second = saver.save("b");
  const third = saver.save("c");
  assert.equal(saver.busy(), true);
  // Only the first went out; b was replaced by c while it waited.
  assert.deepEqual(writes.map((w) => w.value), ["a"]);

  writes[0].done.resolve("saved a");
  assert.equal(await first, "saved a");
  assert.deepEqual(writes.map((w) => w.value), ["a", "c"]);
  assert.equal(saver.busy(), true);

  writes[1].done.resolve("saved c");
  // Both waiters that were queued get the save that carried the newest value.
  assert.equal(await second, "saved c");
  assert.equal(await third, "saved c");
  assert.equal(saver.busy(), false);
});

test("a failed save rejects its callers and the next one still goes out", async () => {
  const outcomes: ReturnType<typeof deferred<number>>[] = [];
  const saver = createLatestSaver((value: number) => {
    const done = deferred<number>();
    outcomes.push(done);
    void value;
    return done.promise;
  });
  const failing = saver.save(1);
  const later = saver.save(2);
  outcomes[0].reject(new Error("offline"));
  await assert.rejects(failing, /offline/);
  outcomes[1].resolve(2);
  assert.equal(await later, 2);

  const throwing = createLatestSaver((): Promise<number> => {
    throw new Error("sync failure");
  });
  await assert.rejects(throwing.save(1), /sync failure/);
  assert.equal(throwing.busy(), false);
});

test("the dialog lists each server's tools and saves the whole sets on the update route", () => {
  assert.match(DIALOG, /import \{ McpServerTools \} from "\.\/mcp-server-tools";/);
  assert.match(DIALOG, /<McpServerTools\s/);
  assert.match(API, /body\.disabled_tools = payload\.disabledTools;/);
  assert.match(API, /body\.ask_tools = payload\.askTools;/);
  assert.match(API, /\/tool-catalog`/);
  assert.match(TOOLS, /disabledTools: sortedNames\(value\.disabled\)/);
  assert.match(TOOLS, /askTools: sortedNames\(value\.ask\)/);
  // The dialog is English-only, like the rest of it, and so is this section.
  assert.doesNotMatch(TOOLS, /useI18n|i18n/);
});

test("third-party tool text is rendered as text, never as HTML", () => {
  assert.doesNotMatch(TOOLS, /dangerouslySetInnerHTML|innerHTML/);
  assert.match(TOOLS, /\{tool\.summary\}/);
  assert.match(TOOLS, /\{tool\.name\}/);
});

test("the frontend and backend agree on the field names and the size limit", () => {
  assert.match(MODELS, /disabled_tools: Optional\[McpToolNames\] = None/);
  assert.match(MODELS, /ask_tools: Optional\[McpToolNames\] = None/);
  assert.match(MODELS, /McpToolNames = Annotated\[list\[McpToolName\], Field\(max_length = 2000\)\]/);
  assert.match(MCP_CLIENT, /MAX_TOOL_SETTING_NAMES = 2000/);
  assert.match(ROUTES, /@router\.get\("\/\{server_id\}\/tool-catalog", response_model = McpToolCatalog\)/);
  for (const field of [
    "cached",
    "stale",
    "tools",
    "enabled_count",
    "total_count",
    "enabled_tokens",
    "tokens_measured",
    "context_tokens",
    "unlisted_disabled",
  ]) {
    assert.match(MODELS, new RegExp(`\\n    ${field}: `), field);
    assert.match(API, new RegExp(`\\n  ${field}: `), field);
  }
});
