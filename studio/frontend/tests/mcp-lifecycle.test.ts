// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { McpServerStatus } from "../src/features/chat/api/mcp-servers-api.ts";
import {
  MCP_IDLE_TIMEOUT_OPTIONS,
  MCP_PROCESS_MODE_OPTIONS,
  defaultIdleTimeout,
  formatMcpDuration,
  idleTimeoutAfterModeChange,
  idleTimeoutLabel,
  mcpCanStop,
  mcpStatusChip,
} from "../src/features/chat/utils/mcp-lifecycle.ts";

import { readSrc, readText } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/mcp-servers-api.ts");
const DIALOG = readSrc("features/chat/chat-mcp-servers-dialog.tsx");
const MCP_CLIENT = readText("../../backend/core/inference/mcp_client.py");
const MODELS = readText("../../backend/models/mcp_servers.py");

function status(overrides: Partial<McpServerStatus> = {}): McpServerStatus {
  return {
    server_id: "s1",
    state: "stopped",
    process_mode: "shared",
    idle_timeout_seconds: 1800,
    processes: 0,
    started_at: null,
    uptime_seconds: null,
    idle_seconds: null,
    stops_in_seconds: null,
    busy_tool: null,
    busy_seconds: null,
    last_error: null,
    last_error_at: null,
    log_path: null,
    ...overrides,
  };
}

test("the idle timeout choices are the backend's, in the order the select lists them", () => {
  assert.deepEqual(
    MCP_IDLE_TIMEOUT_OPTIONS.map((option) => option.value),
    [60, 300, 1800, 7200, 0],
  );
  assert.deepEqual(
    MCP_IDLE_TIMEOUT_OPTIONS.map((option) => option.label),
    ["1 min", "5 min", "30 min", "2 h", "Never"],
  );
  assert.match(MCP_CLIENT, /IDLE_TIMEOUT_CHOICES = \(60, 300, 1800, 7200, 0\)/);
  assert.match(MODELS, /McpIdleTimeout = Literal\[60, 300, 1800, 7200, 0\]/);
  assert.match(
    MCP_CLIENT,
    /DEFAULT_IDLE_TIMEOUT = \{PROCESS_MODE_SHARED: 1800, PROCESS_MODE_PER_CHAT: 300\}/,
  );
  assert.deepEqual(
    MCP_PROCESS_MODE_OPTIONS.map((option) => option.value),
    ["shared", "per_chat"],
  );
});

test("a mode switch carries the default timeout over but keeps one the user picked", () => {
  assert.equal(defaultIdleTimeout("shared"), 1800);
  assert.equal(defaultIdleTimeout("per_chat"), 300);
  assert.equal(idleTimeoutAfterModeChange("shared", "per_chat", 1800), 300);
  assert.equal(idleTimeoutAfterModeChange("per_chat", "shared", 300), 1800);
  assert.equal(idleTimeoutAfterModeChange("shared", "per_chat", 0), 0);
  assert.equal(idleTimeoutAfterModeChange("per_chat", "shared", 7200), 7200);
  assert.equal(idleTimeoutLabel(0), "Never");
  assert.equal(idleTimeoutLabel(1800), "30 min");
});

test("durations read coarsely", () => {
  assert.equal(formatMcpDuration(-3), "0s");
  assert.equal(formatMcpDuration(42.9), "42s");
  assert.equal(formatMcpDuration(60), "1 min");
  assert.equal(formatMcpDuration(3599), "59 min");
  assert.equal(formatMcpDuration(3600), "1 h");
  assert.equal(formatMcpDuration(3900), "1 h 5 min");
  assert.equal(formatMcpDuration(86400 * 2 + 3600 * 3), "2 d 3 h");
});

test("the status chip says what the process is doing", () => {
  assert.deepEqual(mcpStatusChip(status()), {
    label: "Stopped",
    tone: "stopped",
    detail: "Starts when a chat uses it",
  });
  assert.deepEqual(
    mcpStatusChip(
      status({
        state: "running",
        processes: 1,
        uptime_seconds: 300,
        busy_tool: "run_drc",
        busy_seconds: 12,
      }),
    ),
    {
      label: "Running",
      tone: "running",
      detail: "running run_drc for 12s · up 5 min",
    },
  );
  assert.deepEqual(
    mcpStatusChip(
      status({
        state: "idle",
        processes: 1,
        uptime_seconds: 120,
        stops_in_seconds: 1500,
      }),
    ),
    {
      label: "Idle",
      tone: "idle",
      detail: "up 2 min · stops in 25 min if unused",
    },
  );
  assert.equal(
    mcpStatusChip(
      status({
        state: "idle",
        processes: 3,
        uptime_seconds: 60,
        stops_in_seconds: null,
      }),
    ).detail,
    "3 chats · up 1 min · stays running",
  );
  assert.deepEqual(
    mcpStatusChip(
      status({ state: "failed", last_error: "Program not found: konnect.exe" }),
    ),
    {
      label: "Failed",
      tone: "failed",
      detail: "Program not found: konnect.exe",
    },
  );
});

test("Stop is offered while something runs or a failure is showing", () => {
  assert.equal(mcpCanStop(undefined), false);
  assert.equal(mcpCanStop(status()), false);
  assert.equal(mcpCanStop(status({ state: "idle", processes: 1 })), true);
  assert.equal(mcpCanStop(status({ state: "failed" })), true);
});

test("the API module carries the lifecycle fields and routes", () => {
  assert.match(API, /export type McpProcessMode = "shared" \| "per_chat";/);
  assert.match(API, /process_mode\?: McpProcessMode;/);
  assert.match(API, /mcpRequest<McpServerStatus\[\]>\("\/status"\)/);
  assert.match(API, /`\/\$\{serverId\}\/restart`, \{ method: "POST" \}/);
  assert.match(API, /`\/\$\{serverId\}\/stop`, \{ method: "POST" \}/);
  assert.match(API, /body\.process_mode = payload\.processMode/);
  assert.match(API, /body\.idle_timeout_seconds = payload\.idleTimeoutSeconds/);
});

test("the dialog edits the lifecycle of local programs only and defaults new ones to shared", () => {
  assert.match(
    DIALOG,
    /processMode: "shared",\n\s+idleTimeoutSeconds: defaultIdleTimeout\("shared"\)/,
  );
  assert.match(
    DIALOG,
    /function lifecyclePayload\(form: FormState\) \{\n\s+return form\.transport === "stdio"/,
  );
  assert.match(DIALOG, /processMode: server\.process_mode \?\? "per_chat"/);
  // Both saves send it; the selects sit with the other local-program fields.
  assert.equal(DIALOG.match(/\.\.\.lifecyclePayload\(form\),/g)?.length, 2);
  assert.match(
    DIALOG,
    /\{addressIsCommand && \(\n\s+<div className="grid gap-x-3 gap-y-2 sm:grid-cols-2">/,
  );
});

test("the list polls process status and opens the server's own log in Settings > Logs", () => {
  assert.match(DIALOG, /const STATUS_POLL_MS = 3000;/);
  assert.match(DIALOG, /if \(!open \|\| view\.kind !== "list"\) return;/);
  assert.match(
    DIALOG,
    /useSettingsDialogStore\.getState\(\)\.openLogs\("mcp", status\.log_path\)/,
  );
  // Only the installation owner gets the log button, as with every other View logs action.
  assert.match(
    DIALOG,
    /isAccountOwner\(\)\n\s+\? \(\) => viewServerLog\(statuses\[server\.id\]\)/,
  );
  // Restart is refused while local programs are off or the server is switched off; Stop never is.
  assert.match(
    DIALOG,
    /stdioBlocked\n\s+\? \(capabilities\?\.stdio_disabled_reason/,
  );
  assert.match(
    DIALOG,
    /disabled=\{pending !== null \|\| !mcpCanStop\(status\)\}/,
  );
});
