// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ImportIcon, XIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Spinner } from "@/components/ui/spinner";
import {
  type McpImportSource,
  importFromMcpSource,
  listMcpImportSources,
} from "./api/mcp-servers-api";
import {
  type McpImportReportRow,
  type McpImportSelection,
  canSelectImportServer,
  countImportSelection,
  defaultImportSelection,
  importOutcomeLabel,
  importRequests,
  importSourceTitle,
  summarizeImportReport,
  toggleImportSelection,
} from "./utils/mcp-import-sources";

const CHIP =
  "shrink-0 rounded-sm bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground";

export interface McpImportFromAppsProps {
  onClose: () => void;
  // A picked config file is importing: don't start a second batch beside it.
  disabled?: boolean;
  onBusyChange?: (busy: boolean) => void;
}

/** Lists the MCP servers other apps on this computer have configured and imports the checked ones.
 *  The backend reads those apps' files itself, so what arrives here is masked: names, the command
 *  or URL with credentials hidden, and env/header names without their values. */
export function McpImportFromApps({
  onClose,
  disabled = false,
  onBusyChange,
}: McpImportFromAppsProps) {
  const [sources, setSources] = useState<McpImportSource[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selection, setSelection] = useState<McpImportSelection>({});
  const [importing, setImporting] = useState(false);
  const [report, setReport] = useState<McpImportReportRow[] | null>(null);
  const mountedRef = useRef(true);
  const loadGenerationRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const scan = useCallback(() => {
    const generation = loadGenerationRef.current + 1;
    loadGenerationRef.current = generation;
    const current = () =>
      mountedRef.current && loadGenerationRef.current === generation;
    return listMcpImportSources().then(
      ({ sources: next }) => {
        if (!current()) return;
        setSources(next);
        setLoadError(null);
        // Fresh each scan: a server imported a moment ago is now "Already added" and must drop out.
        setSelection(defaultImportSelection(next));
        setLoading(false);
      },
      (err: unknown) => {
        if (!current()) return;
        setLoadError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      },
    );
  }, []);

  useEffect(() => {
    void scan();
  }, [scan]);

  function rescan() {
    setLoading(true);
    void scan();
  }

  async function runImport() {
    if (!sources || importing) return;
    const requests = importRequests(sources, selection);
    if (requests.length === 0) return;
    setImporting(true);
    setReport(null);
    onBusyChange?.(true);
    const rows: McpImportReportRow[] = [];
    try {
      for (const request of requests) {
        const source = sources.find((item) => item.id === request.sourceId);
        const title = source ? importSourceTitle(source) : request.sourceId;
        try {
          const { results } = await importFromMcpSource(
            request.sourceId,
            request.serverNames,
          );
          rows.push(...results.map((result) => ({ ...result, source: title })));
        } catch (err) {
          // A refused or failed batch still names every server it held, so none goes unreported.
          const detail = err instanceof Error ? err.message : String(err);
          rows.push(
            ...request.serverNames.map((name) => ({
              name,
              status: "error" as const,
              detail,
              server_id: null,
              source: title,
            })),
          );
        }
      }
    } finally {
      onBusyChange?.(false);
      if (mountedRef.current) {
        setImporting(false);
        setReport(rows);
      }
    }
    if (mountedRef.current) rescan();
  }

  const busy = importing || disabled;
  const selectedCount = sources ? countImportSelection(sources, selection) : 0;
  const reportSpansSources = new Set(report?.map((row) => row.source)).size > 1;

  return (
    <section
      aria-labelledby="mcp-import-apps-title"
      className="flex min-w-0 flex-col gap-3 rounded-md border p-3"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 id="mcp-import-apps-title" className="text-sm font-medium">
            Import from another app
          </h3>
          <p className="text-xs text-muted-foreground">
            MCP servers set up in Claude Desktop, Claude Code, Cursor, VS Code
            or Windsurf on this computer. Studio reads their config files
            itself: keys and tokens are copied over without being shown here.
          </p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onClose}
          aria-label="Close import from another app"
        >
          <XIcon className="size-3.5" />
        </Button>
      </div>

      {report && (
        <Alert
          variant={
            report.some((row) => row.status === "error")
              ? "destructive"
              : "default"
          }
          role="status"
        >
          <AlertTitle>{summarizeImportReport(report)}</AlertTitle>
          <AlertDescription>
            <ul className="flex flex-col gap-0.5">
              {report.map((row, index) => (
                <li key={`${row.source}-${row.name}-${index}`}>
                  <span className="font-medium">{row.name}</span>
                  {reportSpansSources ? ` (${row.source})` : ""}:{" "}
                  {importOutcomeLabel(row.status)}
                  {row.detail ? ` — ${row.detail}` : ""}
                </li>
              ))}
            </ul>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="mt-1 h-auto px-0"
              onClick={() => setReport(null)}
            >
              Dismiss
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {loading && sources === null ? (
        <div className="flex justify-center py-4">
          <Spinner />
        </div>
      ) : loadError ? (
        <p className="text-xs text-destructive">{loadError}</p>
      ) : !sources || sources.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No MCP servers found in Claude Desktop, Claude Code, Cursor, VS Code
          or Windsurf on this computer.
        </p>
      ) : (
        <div className="flex min-w-0 flex-col gap-3">
          {sources.map((source) => (
            <div key={source.id} className="min-w-0">
              <div className="flex min-w-0 items-baseline justify-between gap-2">
                <span className="truncate text-sm font-medium">
                  {importSourceTitle(source)}
                </span>
                {!source.error && (
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {source.servers.length} server
                    {source.servers.length === 1 ? "" : "s"}
                  </span>
                )}
              </div>
              {source.path && (
                <div
                  className="truncate font-mono text-[11px] text-muted-foreground"
                  title={source.path}
                >
                  {source.path}
                </div>
              )}
              {source.error ? (
                <p className="mt-1 text-xs text-destructive">{source.error}</p>
              ) : (
                <ul className="mt-1 flex flex-col divide-y rounded-md border">
                  {source.servers.map((server, index) => {
                    const selectable = canSelectImportServer(server);
                    const checked =
                      selectable &&
                      (selection[source.id] ?? []).includes(server.name);
                    const checkboxId = `mcp-import-${source.id}-${index}`;
                    const keys =
                      server.transport === "stdio"
                        ? server.env_keys
                        : server.header_keys;
                    return (
                      <li
                        key={server.name}
                        className="flex min-w-0 items-start gap-2 px-3 py-2"
                      >
                        <Checkbox
                          id={checkboxId}
                          className="mt-0.5"
                          checked={checked}
                          disabled={!selectable || busy}
                          onCheckedChange={(next) =>
                            setSelection((prev) =>
                              toggleImportSelection(
                                prev,
                                source.id,
                                server.name,
                                next === true,
                              ),
                            )
                          }
                        />
                        <div className="min-w-0 flex-1">
                          <div className="flex min-w-0 items-center gap-2">
                            <label
                              htmlFor={checkboxId}
                              className="truncate text-sm font-medium"
                            >
                              {server.name}
                            </label>
                            <span className={CHIP}>
                              {server.transport === "stdio"
                                ? "Local program"
                                : "Remote"}
                            </span>
                            {server.already_added && (
                              <span className={CHIP}>Already added</span>
                            )}
                          </div>
                          {server.target && (
                            <div
                              className="truncate font-mono text-xs text-muted-foreground"
                              title={server.target}
                            >
                              {server.target}
                            </div>
                          )}
                          {keys.length > 0 && (
                            <div className="truncate text-xs text-muted-foreground">
                              {server.transport === "stdio"
                                ? "Environment"
                                : "Headers"}
                              : {keys.join(", ")}
                            </div>
                          )}
                          {server.note && (
                            <div
                              className={
                                server.importable
                                  ? "text-xs text-amber-700 dark:text-amber-400"
                                  : "text-xs text-destructive"
                              }
                            >
                              {server.note}
                            </div>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between gap-2">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={rescan}
          disabled={busy || loading}
        >
          {loading && sources !== null ? <Spinner /> : null}
          Rescan
        </Button>
        <Button
          type="button"
          size="sm"
          onClick={() => void runImport()}
          disabled={busy || selectedCount === 0}
        >
          {importing ? <Spinner /> : <ImportIcon className="size-3.5" />}
          {selectedCount > 0
            ? `Import ${selectedCount} server${selectedCount === 1 ? "" : "s"}`
            : "Import"}
        </Button>
      </div>
    </section>
  );
}
