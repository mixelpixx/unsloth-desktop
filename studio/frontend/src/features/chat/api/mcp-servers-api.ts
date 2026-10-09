// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";

import {
  getMcpServerMutationEpoch,
  readAfterPendingMcpServerMutations,
  readMcpServerMutationSnapshot,
  trackMcpServerMutation,
} from "./mcp-server-mutation-tracker";

export type McpImageInputMapping = {
  tool: string;
  field: string;
  encoding: "base64" | "data_url";
};

/** A local program's processes: one shared by every chat, or one per chat. HTTP servers ignore it. */
export type McpProcessMode = "shared" | "per_chat";

export interface McpServerConfig {
  id: string;
  builtin_id: string | null;
  display_name: string;
  url: string;
  headers: Record<string, string>;
  is_enabled: boolean;
  use_oauth: boolean;
  // Local programs only: the folder the program starts in; null = the backend's own.
  cwd?: string | null;
  oauth_client_id?: string | null;
  has_oauth_client_secret?: boolean;
  image_input_mappings?: McpImageInputMapping[];
  image_mappings_active?: boolean;
  // Optional: absent on a backend older than the lifecycle setting.
  process_mode?: McpProcessMode;
  /** Seconds a process may sit unused before it is stopped; 0 = never. */
  idle_timeout_seconds?: number;
  created_at: string;
  updated_at: string;
}

/** What a local program's processes are doing now (GET /api/mcp/servers/status). */
export interface McpServerStatus {
  server_id: string;
  state: "running" | "idle" | "stopped" | "failed";
  process_mode: McpProcessMode;
  idle_timeout_seconds: number;
  processes: number;
  started_at: number | null;
  uptime_seconds: number | null;
  idle_seconds: number | null;
  stops_in_seconds: number | null;
  busy_tool: string | null;
  busy_seconds: number | null;
  last_error: string | null;
  last_error_at: number | null;
  /** The program's stderr log, for Settings > Logs; null when it has never written one. */
  log_path: string | null;
}

export interface McpServerProbeResult {
  ok: boolean;
  tool_count: number;
  error: string | null;
}

export interface McpBuiltinConfig {
  builtin_id: "blender";
  display_name: string;
  server_id: string | null;
  is_enabled: boolean;
  available: boolean;
  unavailable_reason: string | null;
  port: number;
  blender_path: string;
  min_blender_version: string;
}

export interface BlenderMcpSettings {
  port: number;
  blender_path: string;
}

export function listMcpBuiltins(
  waitForPendingMutations = true,
): Promise<McpBuiltinConfig[]> {
  const read = () => mcpRequest<McpBuiltinConfig[]>("/builtins");
  return waitForPendingMutations
    ? readAfterPendingMcpServerMutations(read)
    : readMcpServerMutationSnapshot(read);
}

export function updateBlenderMcp(
  payload: BlenderMcpSettings & { is_enabled: boolean; consent: boolean },
): Promise<McpBuiltinConfig> {
  return trackMcpServerMutation(
    mcpRequest("/builtins/blender", { method: "PUT", body: payload }),
  );
}

export function testBlenderMcp(
  payload: BlenderMcpSettings & { consent: boolean },
): Promise<
  McpServerProbeResult & {
    blender_ready?: boolean;
    blender_error?: string | null;
  }
> {
  return mcpRequest("/builtins/blender/test", {
    method: "POST",
    body: payload,
  });
}

export interface McpServerImportResult {
  created: McpServerConfig[];
  skipped: string[];
  errors: string[];
}

export interface McpStdioCommand {
  command: string;
  arguments: string[];
}

export interface McpCapabilities {
  stdio_enabled: boolean;
  stdio_disabled_reason: string | null;
}

// Whether local programs (.exe, npx, uvx) may be added here, so the dialog can explain a closed gate
// before the user fills in an executable rather than after Save.
export function getMcpCapabilities(): Promise<McpCapabilities> {
  return mcpRequest<McpCapabilities>("/capabilities");
}

let mcpServerListRequest: Promise<McpServerConfig[]> | null = null;
let mcpServerSettlementListRequest: {
  minimumEpoch: number;
  promise: Promise<McpServerConfig[]>;
} | null = null;

function parseErrorText(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const { detail, message } = body as { detail?: unknown; message?: unknown };
    const formatted = formatFastApiDetail(detail);
    if (formatted) return formatted;
    if (typeof message === "string" && message) return message;
  }
  return `Request failed (${status})`;
}

async function mcpRequest<T>(
  path: string,
  init?: { method?: string; body?: object },
): Promise<T> {
  const response = await authFetch(`/api/mcp/servers${path}`, {
    method: init?.method,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    body: init?.body ? JSON.stringify(init.body) : undefined,
  });
  // 204 No Content (DELETE) has no body — calling .json() would throw.
  if (response.status === 204) return undefined as T;
  const json = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseErrorText(response.status, json));
  return json as T;
}

export function listMcpServers({
  waitForPendingMutations = true,
  minimumMutationEpoch,
}: {
  waitForPendingMutations?: boolean;
  minimumMutationEpoch?: number;
} = {}): Promise<McpServerConfig[]> {
  if (!waitForPendingMutations) {
    const requestedEpoch = minimumMutationEpoch ?? getMcpServerMutationEpoch();
    if (
      mcpServerSettlementListRequest &&
      mcpServerSettlementListRequest.minimumEpoch >= requestedEpoch
    ) {
      return mcpServerSettlementListRequest.promise;
    }
    const request = readMcpServerMutationSnapshot(() =>
      mcpRequest<McpServerConfig[]>("/"),
    );
    const slot = { minimumEpoch: requestedEpoch, promise: request };
    mcpServerSettlementListRequest = slot;
    void request.then(
      () => {
        if (mcpServerSettlementListRequest === slot) {
          mcpServerSettlementListRequest = null;
        }
      },
      () => {
        if (mcpServerSettlementListRequest === slot) {
          mcpServerSettlementListRequest = null;
        }
      },
    );
    return request;
  }

  if (mcpServerListRequest) return mcpServerListRequest;

  const request = readAfterPendingMcpServerMutations(() =>
    mcpRequest<McpServerConfig[]>("/"),
  );
  mcpServerListRequest = request;
  void request.then(
    () => {
      if (mcpServerListRequest === request) mcpServerListRequest = null;
    },
    () => {
      if (mcpServerListRequest === request) mcpServerListRequest = null;
    },
  );
  return request;
}

export function createMcpServer(payload: {
  displayName: string;
  url: string;
  headers?: Record<string, string>;
  isEnabled?: boolean;
  useOauth?: boolean;
  cwd?: string | null;
  oauthClientId?: string | null;
  oauthClientSecret?: string;
  imageInputMappings?: McpImageInputMapping[];
  /** Local programs only; omit for the backend's default (shared). */
  processMode?: McpProcessMode;
  idleTimeoutSeconds?: number;
}): Promise<McpServerConfig> {
  return trackMcpServerMutation(
    mcpRequest("/", {
      method: "POST",
      body: {
        display_name: payload.displayName,
        url: payload.url,
        headers: payload.headers ?? null,
        is_enabled: payload.isEnabled ?? true,
        use_oauth: payload.useOauth ?? false,
        cwd: payload.cwd ?? null,
        oauth_client_id: payload.oauthClientId ?? null,
        oauth_client_secret: payload.oauthClientSecret ?? null,
        image_input_mappings: payload.imageInputMappings ?? [],
        ...(payload.processMode !== undefined
          ? { process_mode: payload.processMode }
          : {}),
        ...(payload.idleTimeoutSeconds !== undefined
          ? { idle_timeout_seconds: payload.idleTimeoutSeconds }
          : {}),
      },
    }),
  );
}

export function updateMcpServer(
  serverId: string,
  payload: {
    displayName?: string;
    url?: string;
    /** null = drop stored headers; omit to leave as-is */
    headers?: Record<string, string> | null;
    isEnabled?: boolean;
    useOauth?: boolean;
    /** null = clear the working directory; omit to leave as-is */
    cwd?: string | null;
    oauthClientId?: string | null;
    /** omit to keep the stored secret */
    oauthClientSecret?: string;
    imageInputMappings?: McpImageInputMapping[];
    /** A new mode ends the processes of the old one. */
    processMode?: McpProcessMode;
    /** A new idle timeout reaches a running process without restarting it. */
    idleTimeoutSeconds?: number;
  },
): Promise<McpServerConfig> {
  const body: Record<string, unknown> = {};
  if (payload.displayName !== undefined)
    body.display_name = payload.displayName;
  if (payload.url !== undefined) body.url = payload.url;
  if (payload.headers !== undefined) body.headers = payload.headers;
  if (payload.isEnabled !== undefined) body.is_enabled = payload.isEnabled;
  if (payload.useOauth !== undefined) body.use_oauth = payload.useOauth;
  if (payload.cwd !== undefined) body.cwd = payload.cwd;
  if (payload.oauthClientId !== undefined)
    body.oauth_client_id = payload.oauthClientId;
  if (payload.oauthClientSecret !== undefined)
    body.oauth_client_secret = payload.oauthClientSecret;
  if (payload.imageInputMappings !== undefined)
    body.image_input_mappings = payload.imageInputMappings;
  if (payload.processMode !== undefined)
    body.process_mode = payload.processMode;
  if (payload.idleTimeoutSeconds !== undefined)
    body.idle_timeout_seconds = payload.idleTimeoutSeconds;
  return trackMcpServerMutation(
    mcpRequest(`/${serverId}`, { method: "PUT", body }),
  );
}

/** Every local program's process status. Empty for a caller that cannot run local programs. */
export function listMcpServerStatus(): Promise<McpServerStatus[]> {
  return mcpRequest<McpServerStatus[]>("/status");
}

/** End the server's processes; a shared one starts again at once and re-reads its tools. */
export function restartMcpServer(serverId: string): Promise<McpServerStatus> {
  return mcpRequest(`/${serverId}/restart`, { method: "POST" });
}

/** End the server's processes now; the next chat that needs it starts it again. */
export function stopMcpServer(serverId: string): Promise<McpServerStatus> {
  return mcpRequest(`/${serverId}/stop`, { method: "POST" });
}

export function deleteMcpServer(serverId: string): Promise<void> {
  return trackMcpServerMutation(
    mcpRequest(`/${serverId}`, { method: "DELETE" }),
  );
}

export function refreshMcpServerTools(
  serverId: string,
): Promise<McpServerProbeResult> {
  return mcpRequest(`/${serverId}/refresh`, { method: "POST" });
}

export function listMcpServerTools(
  serverId: string,
): Promise<{ name: string; inputSchema?: unknown }[]> {
  return mcpRequest(`/${serverId}/tools`);
}

export function testMcpServer(payload: {
  url: string;
  headers?: Record<string, string>;
  useOauth?: boolean;
  cwd?: string | null;
  oauthClientId?: string | null;
  oauthClientSecret?: string;
  serverId?: string;
}): Promise<McpServerProbeResult> {
  return mcpRequest("/test", {
    method: "POST",
    body: {
      url: payload.url,
      headers: payload.headers ?? null,
      use_oauth: payload.useOauth ?? false,
      cwd: payload.cwd ?? null,
      oauth_client_id: payload.oauthClientId ?? null,
      oauth_client_secret: payload.oauthClientSecret ?? null,
      server_id: payload.serverId ?? null,
    },
  });
}

export function decodeMcpStdioCommand(url: string): Promise<McpStdioCommand> {
  return mcpRequest("/stdio/decode", {
    method: "POST",
    body: { url },
  });
}

export function encodeMcpStdioCommand(payload: McpStdioCommand): Promise<{
  url: string;
}> {
  return mcpRequest("/stdio/encode", {
    method: "POST",
    body: payload,
  });
}

// Bulk-import servers from a standard mcpServers JSON config (Claude Desktop, Cursor, Cline, VS
// Code). The backend skips duplicates and reports per-entry errors instead of failing the batch.
export function importMcpServers(
  config: unknown,
): Promise<McpServerImportResult> {
  return trackMcpServerMutation(
    mcpRequest("/import", { method: "POST", body: { config } }),
  );
}

// A server another app on this computer has configured. The backend reads the app's file itself, so
// env and header values never reach the browser: only their names, and the command or URL masked.
export interface McpImportSourceServer {
  name: string;
  transport: "stdio" | "http";
  target: string;
  env_keys: string[];
  header_keys: string[];
  already_added: boolean;
  importable: boolean;
  note: string | null;
}

export interface McpImportSource {
  id: string;
  app: string;
  label: string | null;
  path: string | null;
  error: string | null;
  servers: McpImportSourceServer[];
}

export type McpImportOutcomeStatus =
  "added" | "added_disabled" | "duplicate" | "error";

export interface McpImportServerOutcome {
  name: string;
  status: McpImportOutcomeStatus;
  detail: string | null;
  server_id: string | null;
}

export function listMcpImportSources(): Promise<{
  sources: McpImportSource[];
}> {
  return mcpRequest("/import-sources");
}

// Import the chosen servers of one discovered app config, by source id: the backend re-reads the
// file, so no path or secret is sent from here.
export function importFromMcpSource(
  sourceId: string,
  serverNames: string[],
): Promise<{ results: McpImportServerOutcome[] }> {
  return trackMcpServerMutation(
    mcpRequest("/import-sources/apply", {
      method: "POST",
      body: { source_id: sourceId, server_names: serverNames },
    }),
  );
}

export type McpUiCspField =
  "connectDomains" | "resourceDomains" | "frameDomains" | "baseUriDomains";

export interface McpUiResource {
  uri: string;
  mime_type: string;
  text: string;
  blob?: string | null;
  ui: { csp?: Partial<Record<McpUiCspField, string[]>> };
  contents?: { uri: string; mimeType?: string; text?: string; blob?: string }[];
}

export interface McpUiToolCallResult {
  content: Record<string, unknown>[];
  structured_content: Record<string, unknown> | null;
  is_error: boolean;
  meta: Record<string, unknown> | null;
}

export function readMcpUiResource(
  serverId: string,
  uri: string,
  scope: { threadId?: string; sessionId?: string },
): Promise<McpUiResource> {
  const query = new URLSearchParams({ uri });
  if (scope.threadId) query.set("thread_id", scope.threadId);
  if (scope.sessionId) query.set("session_id", scope.sessionId);
  return mcpRequest(`/${serverId}/ui-resource?${query}`);
}

/** `serverId` comes from the tool part that drew the frame, never the widget. A 409 rejects with
 *  Error("approval_required"). */
export function callMcpUiTool(
  serverId: string,
  body: {
    tool_name: string;
    arguments: Record<string, unknown>;
    thread_id: string | null;
    session_id: string | null;
    permission_mode: string;
    approved: boolean;
  },
): Promise<McpUiToolCallResult> {
  return mcpRequest(`/${serverId}/ui-tool-call`, { method: "POST", body });
}
