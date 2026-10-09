// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Logs > Diagnostics: the report's shape and the pure helpers around it.
// No imports on purpose, so the node tests can load this file as it ships.

export const DIAGNOSTICS_ENDPOINT = "/api/diagnostics";
/** The logs export with the report added: one archive builder, one set of guards. */
export const DIAGNOSTICS_BUNDLE_ENDPOINT =
  "/api/settings/debug/logs/export?diagnostics=true";
export const ISSUE_TRACKER_NEW_URL =
  "https://github.com/mixelpixx/unsloth-desktop/issues/new";
/** The only thing ever put in the issue URL. Diagnostics go in the body, pasted by the user. */
export const ISSUE_TITLE = "[Studio] ";

export const DIAGNOSTICS_SECTIONS = [
  "studio",
  "os",
  "gpus",
  "python",
  "llama_cpp",
  "storage",
  "models",
  "mcp",
  "environment",
] as const;

export type DiagnosticsSectionName = (typeof DIAGNOSTICS_SECTIONS)[number];

export type DiagnosticsSection<T> =
  | { status: "ok"; data: T }
  | { status: "unavailable"; reason: string };

export interface StudioInfo {
  studio_version?: string | null;
  unsloth_version?: string | null;
  install_source?: string | null;
  source_checkout?: {
    branch?: string | null;
    commit?: string | null;
    dirty?: boolean | null;
  } | null;
  frontend_build?: {
    path?: string | null;
    served?: boolean;
    built_at?: string | null;
  } | null;
}

export interface OsInfo {
  system?: string | null;
  name?: string | null;
  edition?: string | null;
  display_version?: string | null;
  build?: string | null;
  machine?: string | null;
  platform?: string | null;
  cpu?: {
    name?: string | null;
    physical_cores?: number | null;
    logical_cores?: number | null;
  } | null;
  memory?: { total_bytes?: number | null; available_bytes?: number | null } | null;
}

export interface GpuDevice {
  index: number;
  name?: string | null;
  compute_capability?: string | null;
  memory_total_bytes?: number | null;
  memory_used_bytes?: number | null;
  memory_free_bytes?: number | null;
  pcie_gen_current?: number | null;
  pcie_gen_max?: number | null;
  pcie_width_current?: number | null;
  pcie_width_max?: number | null;
}

export interface GpuInfo {
  driver_version?: string | null;
  cuda_driver_version?: string | null;
  cuda_visible_devices?: string | null;
  devices?: GpuDevice[];
}

export interface PythonInfo {
  version?: string | null;
  implementation?: string | null;
  executable?: string | null;
  torch?: { cuda?: string | null; hip?: string | null } | null;
  packages?: Record<string, string | null>;
}

export interface LlamaCppInfo {
  version?: string | null;
  build?: number | null;
  commit?: string | null;
  backend?: string | null;
  bundle?: string | null;
  binary?: string | null;
  update?: {
    state?: string | null;
    latest?: string | null;
    checked_at?: string | null;
  } | null;
}

export interface StorageLocation {
  key: string;
  path?: string | null;
  exists?: boolean;
  drive?: string | null;
  free_bytes?: number | null;
  total_bytes?: number | null;
}

export interface StorageInfo {
  locations?: StorageLocation[];
  hf_cache_size?: {
    state?: string | null;
    bytes?: number | null;
    measured_at?: string | null;
  } | null;
}

export interface LoadedModel {
  name?: string | null;
  kind?: string | null;
  engine?: string | null;
  variant?: string | null;
  gpu_ids?: number[];
  device?: string | null;
  vram_bytes?: number | null;
  loading?: boolean;
}

export interface McpServerSummary {
  name?: string | null;
  transport?: string | null;
  enabled?: boolean;
  process_mode?: string | null;
  state?: string | null;
}

export interface EnvironmentVariable {
  name: string;
  value: string;
  redacted: boolean;
}

export interface DiagnosticsSections {
  studio: DiagnosticsSection<StudioInfo>;
  os: DiagnosticsSection<OsInfo>;
  gpus: DiagnosticsSection<GpuInfo>;
  python: DiagnosticsSection<PythonInfo>;
  llama_cpp: DiagnosticsSection<LlamaCppInfo>;
  storage: DiagnosticsSection<StorageInfo>;
  models: DiagnosticsSection<{ models?: LoadedModel[] }>;
  mcp: DiagnosticsSection<{ servers?: McpServerSummary[] }>;
  environment: DiagnosticsSection<{ variables?: EnvironmentVariable[] }>;
}

export interface DiagnosticsReport {
  generatedAt: string | null;
  sections: DiagnosticsSections;
  /** The backend's English Markdown, the same text the bundle carries. */
  markdown: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseSection(value: unknown): DiagnosticsSection<never> {
  if (isRecord(value) && value.status === "ok" && isRecord(value.data)) {
    return { status: "ok", data: value.data as never };
  }
  const reason =
    isRecord(value) && typeof value.reason === "string" && value.reason
      ? value.reason
      : "not reported";
  return { status: "unavailable", reason };
}

/** Every known section, each either ok with an object or unavailable with a reason. A backend
 * that leaves one out (older, or a section it dropped) reads as unavailable, never as a crash. */
export function parseDiagnosticsReport(body: unknown): DiagnosticsReport {
  const root = isRecord(body) ? body : {};
  const raw = isRecord(root.sections) ? root.sections : {};
  const sections = Object.fromEntries(
    DIAGNOSTICS_SECTIONS.map((name) => [name, parseSection(raw[name])]),
  ) as unknown as DiagnosticsSections;
  return {
    generatedAt:
      typeof root.generated_at === "string" ? root.generated_at : null,
    sections,
    markdown: typeof root.markdown === "string" ? root.markdown : "",
  };
}

/** The new-issue page with a fixed title and nothing else: the report is never put in a URL. */
export function buildIssueUrl(): string {
  const url = new URL(ISSUE_TRACKER_NEW_URL);
  url.searchParams.set("title", ISSUE_TITLE);
  return url.toString();
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

/** `unsloth-diagnostics-<YYYYmmdd-HHMMSS>.zip`, local time like the logs archive. */
export function diagnosticsArchiveFilename(now: Date = new Date()): string {
  const day = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}`;
  const time = `${pad2(now.getHours())}${pad2(now.getMinutes())}${pad2(now.getSeconds())}`;
  return `unsloth-diagnostics-${day}-${time}.zip`;
}

/** Binary gigabytes with one decimal, the unit nvidia-smi and the Markdown use. */
export function formatGiB(bytes: number | null | undefined): string | null {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) {
    return null;
  }
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

export type DiagnosticsFailure = "outdated" | "forbidden" | "failed";

export function diagnosticsFailureForStatus(status: number): DiagnosticsFailure {
  if (status === 404) return "outdated";
  if (status === 403 || status === 401) return "forbidden";
  return "failed";
}

export class DiagnosticsRequestError extends Error {
  readonly failure: DiagnosticsFailure;

  constructor(failure: DiagnosticsFailure, message: string) {
    super(message);
    this.name = "DiagnosticsRequestError";
    this.failure = failure;
  }
}

/** Installed packages, in the backend's order, minus the ones the caller shows on their own. */
export function installedPackages(
  packages: Record<string, string | null> | undefined,
  skip: readonly string[] = [],
): { name: string; version: string }[] {
  return Object.entries(packages ?? {})
    .filter(
      (entry): entry is [string, string] =>
        typeof entry[1] === "string" && entry[1] !== "" && !skip.includes(entry[0]),
    )
    .map(([name, version]) => ({ name, version }));
}

export function missingPackages(
  packages: Record<string, string | null> | undefined,
): string[] {
  return Object.entries(packages ?? {})
    .filter(([, version]) => !version)
    .map(([name]) => name);
}
