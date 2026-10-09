// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronRightIcon, HandIcon, SearchIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Toggle } from "@/components/ui/toggle";
import { RefreshGlyph } from "@/lib/refresh-icon";

import {
  type McpServerConfig,
  type McpToolCatalog,
  getMcpServerToolCatalog,
  refreshMcpServerTools,
  updateMcpServer,
} from "./api/mcp-servers-api";
import {
  type LatestSaver,
  createLatestSaver,
  filterMcpTools,
  formatTokenCount,
  mcpToolCostNote,
  mcpToolSummary,
  mcpToolTotals,
  sortedNames,
  withNames,
} from "./utils/mcp-tool-settings";

type ToolSets = { disabled: ReadonlySet<string>; ask: ReadonlySet<string> };

// Past this many tools the list gets a filter box; below it one is clutter.
const FILTER_FROM = 6;

const ASK_HINT =
  "Ask before running: pauses this tool for your approval under Approve for me, even when it looks routine. Run automatically and Full access never ask.";

function serverSets(server: McpServerConfig): ToolSets {
  return {
    disabled: new Set(server.disabled_tools ?? []),
    ask: new Set(server.ask_tools ?? []),
  };
}

/**
 * A server's "Tools" section in the MCP Servers list: every tool it offers the model, each with an on/off
 * switch, an ask-first toggle and what its schema adds to every request. Local models lose tool-calling
 * accuracy and context as the list grows, so this is where the list gets trimmed. Changes save as they are
 * made, one request at a time with the newest state winning. Tool names and descriptions come from the
 * server and are rendered as text only.
 */
export function McpServerTools({
  server,
  disabled,
  refreshBlockedReason,
  onServerChange,
}: {
  server: McpServerConfig;
  /** The dialog is mid-import or similar: keep the switches still. */
  disabled: boolean;
  /** Why the tools cannot be re-read now (local programs paused), or null. */
  refreshBlockedReason: string | null;
  /** The server row as the backend saved it, so the list shows it without waiting for a reload. */
  onServerChange: (server: McpServerConfig) => void;
}) {
  const [open, setOpen] = useState(false);
  const [catalog, setCatalog] = useState<McpToolCatalog | null>(null);
  const [loading, setLoading] = useState(false);
  const [rereading, setRereading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  // The switches as the user left them while a save is pending; null = the server row's own sets.
  const [pendingSets, setPendingSets] = useState<ToolSets | null>(null);
  const loadGenerationRef = useRef(0);
  const mountedRef = useRef(true);
  const saverRef = useRef<LatestSaver<ToolSets, McpServerConfig> | null>(null);
  const onServerChangeRef = useRef(onServerChange);

  useEffect(() => {
    onServerChangeRef.current = onServerChange;
  }, [onServerChange]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      loadGenerationRef.current += 1;
    };
  }, []);

  const sets = pendingSets ?? serverSets(server);
  const tools = catalog?.tools ?? [];
  const totals = mcpToolTotals(tools, sets.disabled);
  const shown = filterMcpTools(tools, query);
  const filtering = query.trim().length > 0;

  async function load() {
    const generation = loadGenerationRef.current + 1;
    loadGenerationRef.current = generation;
    setLoading(true);
    setLoadError(null);
    try {
      const next = await getMcpServerToolCatalog(server.id);
      if (!mountedRef.current || loadGenerationRef.current !== generation)
        return;
      setCatalog(next);
    } catch (err) {
      if (!mountedRef.current || loadGenerationRef.current !== generation)
        return;
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      if (mountedRef.current && loadGenerationRef.current === generation)
        setLoading(false);
    }
  }

  function handleOpenChange(next: boolean) {
    setOpen(next);
    if (next) void load();
  }

  // Asks the server for its tools now (starting a local program if it has to), then shows them.
  async function reread() {
    setRereading(true);
    try {
      const result = await refreshMcpServerTools(server.id);
      if (!mountedRef.current) return;
      if (!result.ok) {
        toast.error(`Could not read the tools of "${server.display_name}"`, {
          description: result.error ?? "Unknown error",
        });
      }
      await load();
    } catch (err) {
      if (!mountedRef.current) return;
      toast.error(`Could not read the tools of "${server.display_name}"`, {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (mountedRef.current) setRereading(false);
    }
  }

  function saver(): LatestSaver<ToolSets, McpServerConfig> {
    if (saverRef.current === null) {
      const serverId = server.id;
      saverRef.current = createLatestSaver((value: ToolSets) =>
        updateMcpServer(serverId, {
          disabledTools: sortedNames(value.disabled),
          askTools: sortedNames(value.ask),
        }),
      );
    }
    return saverRef.current;
  }

  function apply(next: ToolSets) {
    setPendingSets(next);
    const queue = saver();
    queue.save(next).then(
      (saved) => {
        if (!mountedRef.current) return;
        onServerChangeRef.current(saved);
        // A newer flip is still on its way: keep showing it rather than this older save.
        if (!queue.busy()) setPendingSets(null);
      },
      (err: unknown) => {
        if (!mountedRef.current || queue.busy()) return;
        setPendingSets(null);
        toast.error(`Could not save the tools of "${server.display_name}"`, {
          description: err instanceof Error ? err.message : String(err),
        });
      },
    );
  }

  function setEnabled(names: readonly string[], enabled: boolean) {
    apply({ ...sets, disabled: withNames(sets.disabled, names, !enabled) });
  }

  function setAsk(name: string, ask: boolean) {
    apply({ ...sets, ask: withNames(sets.ask, [name], ask) });
  }

  const shownNames = shown.map((tool) => tool.name);
  const offCount = sets.disabled.size;
  const triggerDetail = catalog?.cached
    ? `${totals.on} of ${totals.total} on`
    : offCount > 0
      ? `${offCount} off`
      : null;

  return (
    <Collapsible open={open} onOpenChange={handleOpenChange} className="mt-1">
      <CollapsibleTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          className="-ml-2 text-muted-foreground"
          aria-label={`Tools of ${server.display_name}${triggerDetail ? `, ${triggerDetail}` : ""}`}
        >
          <ChevronRightIcon
            aria-hidden="true"
            className={`size-3 transition-transform ${open ? "rotate-90" : ""}`}
          />
          Tools
          {triggerDetail ? (
            <span className="font-normal">· {triggerDetail}</span>
          ) : null}
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-1 flex flex-col gap-2 rounded-md border bg-muted/30 p-2">
          {loading && catalog === null ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner />
              Loading tools…
            </div>
          ) : loadError && catalog === null ? (
            <div className="flex items-center justify-between gap-2 text-xs text-destructive">
              <span role="alert">Could not load the tools: {loadError}</span>
              <Button
                type="button"
                size="xs"
                variant="outline"
                onClick={() => void load()}
              >
                Retry
              </Button>
            </div>
          ) : catalog && !catalog.cached ? (
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
              <span>
                Studio has not read this server's tools yet. A chat that uses it
                reads them, or read them now.
              </span>
              <Button
                type="button"
                size="xs"
                variant="outline"
                onClick={() => void reread()}
                disabled={rereading || refreshBlockedReason !== null}
                title={refreshBlockedReason ?? undefined}
              >
                {rereading ? <Spinner /> : <RefreshGlyph className="size-3" />}
                Read tools
              </Button>
            </div>
          ) : catalog ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
                <div className="min-w-0">
                  <div
                    className="text-xs font-medium"
                    role="status"
                    aria-live="polite"
                  >
                    {mcpToolSummary(totals)}
                  </div>
                  <div className="text-[11px] text-muted-foreground">
                    {mcpToolCostNote(
                      totals.tokens,
                      catalog.tokens_measured,
                      catalog.context_tokens,
                    )}
                    {server.is_enabled
                      ? ""
                      : " The server is off; these apply when it is on."}
                  </div>
                </div>
                <div className="flex shrink-0 items-center">
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    disabled={disabled || shownNames.length === 0}
                    onClick={() => setEnabled(shownNames, true)}
                    title={
                      filtering
                        ? `Turn on the ${shownNames.length} tools shown`
                        : "Offer every tool to the model"
                    }
                  >
                    All on
                  </Button>
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    disabled={disabled || shownNames.length === 0}
                    onClick={() => setEnabled(shownNames, false)}
                    title={
                      filtering
                        ? `Turn off the ${shownNames.length} tools shown`
                        : "Offer none of this server's tools to the model"
                    }
                  >
                    All off
                  </Button>
                  <Button
                    type="button"
                    size="icon-xs"
                    variant="ghost"
                    onClick={() => void reread()}
                    disabled={rereading || refreshBlockedReason !== null}
                    title={
                      refreshBlockedReason ?? "Read the tool list from the server again"
                    }
                    aria-label={`Read the tools of ${server.display_name} again`}
                  >
                    {rereading ? (
                      <Spinner />
                    ) : (
                      <RefreshGlyph className="size-3" />
                    )}
                  </Button>
                </div>
              </div>
              {catalog.stale ? (
                <p className="text-[11px] text-muted-foreground">
                  The server says its tools changed since this list was read.
                  Read them again to see the new list.
                </p>
              ) : null}
              {tools.length >= FILTER_FROM ? (
                <div className="relative">
                  <SearchIcon
                    aria-hidden="true"
                    className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground"
                  />
                  <Input
                    type="search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Filter tools by name or description"
                    aria-label={`Filter the tools of ${server.display_name}`}
                    className="h-8 pl-8 text-xs md:text-xs"
                  />
                </div>
              ) : null}
              {tools.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  This server lists no tools for the model.
                </p>
              ) : shown.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  No tools match “{query.trim()}”.
                </p>
              ) : (
                <ul
                  className="max-h-72 divide-y overflow-y-auto rounded-md border bg-background"
                  aria-label={`Tools of ${server.display_name}`}
                >
                  {shown.map((tool) => {
                    const on = !sets.disabled.has(tool.name);
                    const asks = sets.ask.has(tool.name);
                    return (
                      <li
                        key={tool.name}
                        className="flex items-start gap-2 px-2 py-1.5"
                      >
                        <Switch
                          size="sm"
                          className="mt-0.5"
                          checked={on}
                          disabled={disabled}
                          onCheckedChange={(next) =>
                            setEnabled([tool.name], next)
                          }
                          aria-label={`Offer ${tool.name} to the model`}
                        />
                        <div
                          className={`min-w-0 flex-1 ${on ? "" : "opacity-60"}`}
                        >
                          <div className="flex min-w-0 items-baseline gap-2">
                            <span
                              className="truncate font-mono text-xs"
                              title={tool.name}
                            >
                              {tool.name}
                            </span>
                            {tool.title && tool.title !== tool.name ? (
                              <span className="truncate text-[11px] text-muted-foreground">
                                {tool.title}
                              </span>
                            ) : null}
                          </div>
                          {tool.summary ? (
                            <p
                              className="line-clamp-2 text-[11px] text-muted-foreground"
                              title={
                                tool.description.length > tool.summary.length
                                  ? tool.description
                                  : undefined
                              }
                            >
                              {tool.summary}
                            </p>
                          ) : null}
                        </div>
                        <span
                          className={`mt-0.5 shrink-0 text-[11px] tabular-nums text-muted-foreground ${on ? "" : "line-through opacity-60"}`}
                          title={`Its schema adds about ${tool.tokens} tokens to every request while it is on`}
                        >
                          ~{formatTokenCount(tool.tokens)}
                        </span>
                        <Toggle
                          size="sm"
                          className="h-6 min-w-6 px-1.5"
                          pressed={asks}
                          disabled={disabled}
                          onPressedChange={(next) => setAsk(tool.name, next)}
                          aria-label={`Ask before running ${tool.name}`}
                          title={ASK_HINT}
                        >
                          <HandIcon
                            className={`size-3.5 ${asks ? "text-foreground" : "text-muted-foreground/60"}`}
                          />
                        </Toggle>
                      </li>
                    );
                  })}
                </ul>
              )}
              <p className="text-[11px] text-muted-foreground">
                Tools turned off are never offered to the model, and a call to
                one is refused. Tools the server adds later start on.{" "}
                <HandIcon
                  aria-hidden="true"
                  className="inline size-3 align-[-2px]"
                />{" "}
                asks before running under Approve for me.
                {catalog.unlisted_disabled.length > 0
                  ? ` ${catalog.unlisted_disabled.length} turned-off ${catalog.unlisted_disabled.length === 1 ? "tool is" : "tools are"} not listed by the server now; ${catalog.unlisted_disabled.length === 1 ? "it stays" : "they stay"} off if it lists them again.`
                  : ""}
              </p>
            </>
          ) : null}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
