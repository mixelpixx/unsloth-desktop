// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { McpProcessMode, McpServerStatus } from "../api/mcp-servers-api";

// A local program's lifecycle: backend core/inference/mcp_client.py (PROCESS_MODES, IDLE_TIMEOUT_CHOICES,
// DEFAULT_IDLE_TIMEOUT) and the McpIdleTimeout literal in models/mcp_servers.py.

export const MCP_PROCESS_MODE_OPTIONS: readonly {
  value: McpProcessMode;
  label: string;
  hint: string;
}[] = [
  {
    value: "shared",
    label: "Shared",
    hint: "One process for all chats, kept running between them. Needed for servers that hold a device, a session or loaded toolsets.",
  },
  {
    value: "per_chat",
    label: "Per chat",
    hint: "A separate process for each chat, so nothing a chat changes on the server reaches another chat.",
  },
];

/** Seconds; 0 = never. */
export const MCP_IDLE_TIMEOUT_OPTIONS: readonly {
  value: number;
  label: string;
}[] = [
  { value: 60, label: "1 min" },
  { value: 300, label: "5 min" },
  { value: 1800, label: "30 min" },
  { value: 7200, label: "2 h" },
  { value: 0, label: "Never" },
];

export function defaultIdleTimeout(mode: McpProcessMode): number {
  return mode === "shared" ? 1800 : 300;
}

/** Switching mode carries the old mode's default over to the new one's; a timeout the user picked stays. */
export function idleTimeoutAfterModeChange(
  previous: McpProcessMode,
  next: McpProcessMode,
  idleTimeoutSeconds: number,
): number {
  return idleTimeoutSeconds === defaultIdleTimeout(previous)
    ? defaultIdleTimeout(next)
    : idleTimeoutSeconds;
}

export function idleTimeoutLabel(seconds: number): string {
  return (
    MCP_IDLE_TIMEOUT_OPTIONS.find((option) => option.value === seconds)
      ?.label ?? formatMcpDuration(seconds)
  );
}

/** "45s", "12 min", "1 h 5 min", "2 d 3 h": coarse on purpose, the list polls every few seconds. */
export function formatMcpDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const rest = minutes % 60;
    return rest ? `${hours} h ${rest} min` : `${hours} h`;
  }
  const days = Math.floor(hours / 24);
  const rest = hours % 24;
  return rest ? `${days} d ${rest} h` : `${days} d`;
}

export type McpStatusTone = McpServerStatus["state"];

export interface McpStatusChip {
  label: string;
  tone: McpStatusTone;
  /** One line under the server's address. */
  detail: string;
}

export function mcpStatusChip(status: McpServerStatus): McpStatusChip {
  const parts: string[] = [];
  if (status.processes > 1) parts.push(`${status.processes} chats`);
  const uptime =
    status.uptime_seconds != null
      ? `up ${formatMcpDuration(status.uptime_seconds)}`
      : null;
  switch (status.state) {
    case "running":
      if (status.busy_tool) {
        parts.push(
          `running ${status.busy_tool}` +
            (status.busy_seconds != null
              ? ` for ${formatMcpDuration(status.busy_seconds)}`
              : ""),
        );
      }
      if (uptime) parts.push(uptime);
      return { label: "Running", tone: "running", detail: parts.join(" · ") };
    case "idle":
      if (uptime) parts.push(uptime);
      parts.push(
        status.stops_in_seconds == null
          ? "stays running"
          : `stops in ${formatMcpDuration(status.stops_in_seconds)} if unused`,
      );
      return { label: "Idle", tone: "idle", detail: parts.join(" · ") };
    case "failed":
      return {
        label: "Failed",
        tone: "failed",
        detail: status.last_error ?? "The server failed to start.",
      };
    default:
      return {
        label: "Stopped",
        tone: "stopped",
        detail: "Starts when a chat uses it",
      };
  }
}

/** Whether Stop does anything: there is a process, or a failure to clear. */
export function mcpCanStop(status: McpServerStatus | undefined): boolean {
  return (
    status !== undefined && (status.processes > 0 || status.state === "failed")
  );
}
