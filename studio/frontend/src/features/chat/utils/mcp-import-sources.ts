// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  McpImportOutcomeStatus,
  McpImportServerOutcome,
  McpImportSource,
  McpImportSourceServer,
} from "../api/mcp-servers-api";

/** Server names chosen per source id. */
export type McpImportSelection = Readonly<Record<string, readonly string[]>>;

export type McpImportReportRow = McpImportServerOutcome & { source: string };

export function importSourceTitle(
  source: Pick<McpImportSource, "app" | "label">,
): string {
  return source.label ? `${source.app} · ${source.label}` : source.app;
}

/** Only a server the backend can import and Studio doesn't already have gets a live checkbox. */
export function canSelectImportServer(server: McpImportSourceServer): boolean {
  return server.importable && !server.already_added;
}

/** Everything new and importable starts checked, so a first import is one click. */
export function defaultImportSelection(
  sources: readonly McpImportSource[],
): McpImportSelection {
  const selection: Record<string, string[]> = {};
  for (const source of sources) {
    const names = source.servers
      .filter(canSelectImportServer)
      .map((server) => server.name);
    if (names.length > 0) selection[source.id] = names;
  }
  return selection;
}

export function toggleImportSelection(
  selection: McpImportSelection,
  sourceId: string,
  name: string,
  checked: boolean,
): McpImportSelection {
  const current = selection[sourceId] ?? [];
  const names = checked
    ? current.includes(name)
      ? current
      : [...current, name]
    : current.filter((existing) => existing !== name);
  const next: Record<string, readonly string[]> = { ...selection };
  if (names.length > 0) next[sourceId] = names;
  else delete next[sourceId];
  return next;
}

/** One request per source, in the order the sources are listed, keeping only names still selectable. */
export function importRequests(
  sources: readonly McpImportSource[],
  selection: McpImportSelection,
): { sourceId: string; serverNames: string[] }[] {
  const requests: { sourceId: string; serverNames: string[] }[] = [];
  for (const source of sources) {
    const chosen = new Set(selection[source.id] ?? []);
    const serverNames = source.servers
      .filter((server) => chosen.has(server.name) && canSelectImportServer(server))
      .map((server) => server.name);
    if (serverNames.length > 0) requests.push({ sourceId: source.id, serverNames });
  }
  return requests;
}

export function countImportSelection(
  sources: readonly McpImportSource[],
  selection: McpImportSelection,
): number {
  return importRequests(sources, selection).reduce(
    (total, request) => total + request.serverNames.length,
    0,
  );
}

const STATUS_LABELS: Record<McpImportOutcomeStatus, string> = {
  added: "Added",
  added_disabled: "Added, switched off",
  duplicate: "Already added",
  error: "Not imported",
};

export function importOutcomeLabel(status: McpImportOutcomeStatus): string {
  return STATUS_LABELS[status];
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/** The report's headline, e.g. "Imported 3 servers (1 switched off), 1 already added, 1 not imported". */
export function summarizeImportReport(
  rows: readonly McpImportReportRow[],
): string {
  const count = (status: McpImportOutcomeStatus) =>
    rows.filter((row) => row.status === status).length;
  const switchedOff = count("added_disabled");
  const added = count("added") + switchedOff;
  const parts = [
    `Imported ${plural(added, "server")}${switchedOff ? ` (${switchedOff} switched off)` : ""}`,
  ];
  const duplicates = count("duplicate");
  if (duplicates) parts.push(`${duplicates} already added`);
  const errors = count("error");
  if (errors) parts.push(`${errors} not imported`);
  return parts.join(", ");
}
