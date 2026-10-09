// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  McpImportSource,
  McpImportSourceServer,
} from "../src/features/chat/api/mcp-servers-api.ts";
import {
  canSelectImportServer,
  countImportSelection,
  defaultImportSelection,
  importOutcomeLabel,
  importRequests,
  importSourceTitle,
  summarizeImportReport,
  toggleImportSelection,
} from "../src/features/chat/utils/mcp-import-sources.ts";

import { readSrc } from "./helpers/kit.ts";

const MCP_SERVERS_API = readSrc("features/chat/api/mcp-servers-api.ts");
const IMPORT_PANEL = readSrc("features/chat/mcp-import-from-apps.tsx");
const DIALOG = readSrc("features/chat/chat-mcp-servers-dialog.tsx");

function server(
  name: string,
  overrides: Partial<McpImportSourceServer> = {},
): McpImportSourceServer {
  return {
    name,
    transport: "stdio",
    target: "npx -y pkg",
    env_keys: [],
    header_keys: [],
    already_added: false,
    importable: true,
    note: null,
    ...overrides,
  };
}

function source(
  id: string,
  servers: McpImportSourceServer[],
  overrides: Partial<McpImportSource> = {},
): McpImportSource {
  return {
    id,
    app: "Claude Desktop",
    label: null,
    path: null,
    error: null,
    servers,
    ...overrides,
  };
}

const SOURCES = [
  source("claude-desktop", [
    server("fs"),
    server("old", { already_added: true }),
    server("blocked", { importable: false, note: "Local commands are off." }),
  ]),
  source("cursor", [server("web", { transport: "http" })], { app: "Cursor" }),
  source("broken", [], { app: "Windsurf", error: "Couldn't read this file" }),
];

test("source titles add the label only when there is one", () => {
  assert.equal(importSourceTitle({ app: "Cursor", label: null }), "Cursor");
  assert.equal(
    importSourceTitle({ app: "Claude Desktop", label: "Microsoft Store install" }),
    "Claude Desktop · Microsoft Store install",
  );
});

test("only new, importable servers are selectable and preselected", () => {
  assert.equal(canSelectImportServer(server("x")), true);
  assert.equal(canSelectImportServer(server("x", { already_added: true })), false);
  assert.equal(canSelectImportServer(server("x", { importable: false })), false);
  assert.deepEqual(defaultImportSelection(SOURCES), {
    "claude-desktop": ["fs"],
    cursor: ["web"],
  });
});

test("toggling keeps other sources and drops an emptied one", () => {
  const start = defaultImportSelection(SOURCES);
  const off = toggleImportSelection(start, "cursor", "web", false);
  assert.deepEqual(off, { "claude-desktop": ["fs"] });
  const on = toggleImportSelection(off, "cursor", "web", true);
  assert.deepEqual(on, { "claude-desktop": ["fs"], cursor: ["web"] });
  assert.deepEqual(toggleImportSelection(on, "cursor", "web", true), on);
  assert.deepEqual(start, { "claude-desktop": ["fs"], cursor: ["web"] });
});

test("requests follow source order and never carry an unselectable server", () => {
  const selection = {
    cursor: ["web"],
    "claude-desktop": ["fs", "old", "blocked", "vanished"],
  };
  assert.deepEqual(importRequests(SOURCES, selection), [
    { sourceId: "claude-desktop", serverNames: ["fs"] },
    { sourceId: "cursor", serverNames: ["web"] },
  ]);
  assert.equal(countImportSelection(SOURCES, selection), 2);
  assert.deepEqual(importRequests(SOURCES, {}), []);
});

test("the report headline counts every outcome", () => {
  const row = (name: string, status: "added" | "added_disabled" | "duplicate" | "error") => ({
    name,
    status,
    detail: null,
    server_id: null,
    source: "Claude Desktop",
  });
  assert.equal(
    summarizeImportReport([
      row("a", "added"),
      row("b", "added_disabled"),
      row("c", "added"),
      row("d", "duplicate"),
      row("e", "error"),
    ]),
    "Imported 3 servers (1 switched off), 1 already added, 1 not imported",
  );
  assert.equal(summarizeImportReport([row("a", "added")]), "Imported 1 server");
  assert.equal(summarizeImportReport([row("d", "duplicate")]), "Imported 0 servers, 1 already added");
  assert.equal(importOutcomeLabel("added_disabled"), "Added, switched off");
  assert.equal(importOutcomeLabel("error"), "Not imported");
});

test("the API reads and imports by source id, tracked as a mutation", () => {
  assert.match(MCP_SERVERS_API, /mcpRequest\("\/import-sources"\)/);
  assert.match(
    MCP_SERVERS_API,
    /return trackMcpServerMutation\(\s*mcpRequest\("\/import-sources\/apply", \{\s*method: "POST",\s*body: \{ source_id: sourceId, server_names: serverNames \},/,
  );
  // Secrets stay on the server: the client types carry key names, never values.
  const serverType = MCP_SERVERS_API.slice(
    MCP_SERVERS_API.indexOf("export interface McpImportSourceServer"),
    MCP_SERVERS_API.indexOf("export interface McpImportSource {"),
  );
  assert.match(serverType, /env_keys: string\[\];/);
  assert.match(serverType, /header_keys: string\[\];/);
  assert.doesNotMatch(serverType, /\b(env|headers): /);
});

test("the dialog offers app import first and keeps the file import as a secondary option", () => {
  assert.match(DIALOG, /import \{ McpImportFromApps \} from "\.\/mcp-import-from-apps";/);
  const fromApp = DIALOG.indexOf("Import from app\n");
  const fromFile = DIALOG.lastIndexOf("Import config\n");
  assert.ok(fromApp !== -1 && fromFile !== -1 && fromApp < fromFile);
  assert.match(
    DIALOG,
    /<McpImportFromApps\s+onClose=\{\(\) => setAppImportOpen\(false\)\}\s+disabled=\{importing\}\s+onBusyChange=\{setAppImporting\}/,
  );
  assert.match(DIALOG, /disabled=\{importing \|\| appImporting\}/);
  // The panel closes on every reopen with the rest of the transient state.
  assert.match(
    DIALOG,
    /setConfirmingDelete\(null\);\s*setAppImportOpen\(false\);/,
  );
});

test("the panel shows the per-server report inline, not as a toast", () => {
  assert.doesNotMatch(IMPORT_PANEL, /toast/);
  assert.match(IMPORT_PANEL, /<AlertTitle>\{summarizeImportReport\(report\)\}<\/AlertTitle>/);
  assert.match(IMPORT_PANEL, /importOutcomeLabel\(row\.status\)/);
  // A failed batch still reports each of its servers.
  assert.match(
    IMPORT_PANEL,
    /request\.serverNames\.map\(\(name\) => \(\{\s*name,\s*status: "error" as const,/,
  );
  assert.match(IMPORT_PANEL, /disabled=\{!selectable \|\| busy\}/);
  assert.match(IMPORT_PANEL, /<span className=\{CHIP\}>Already added<\/span>/);
});
