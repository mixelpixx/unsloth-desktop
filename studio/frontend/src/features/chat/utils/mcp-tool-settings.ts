// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { McpToolEntry } from "../api/mcp-servers-api";

// Per-tool MCP settings: backend core/inference/mcp_client.py (server_disabled_tools, server_ask_tools) and
// core/inference/tools.py (mcp_tool_catalog). The server stores the tools turned OFF, so a tool it adds later
// arrives on.

/** "312", "4.1K", "41K": the dialog's token figures are estimates, so coarse on purpose. */
export function formatTokenCount(tokens: number): string {
  const value = Math.max(0, Math.round(tokens));
  if (value < 1000) return String(value);
  if (value < 10_000) {
    const thousands = Math.round(value / 100) / 10;
    return `${Number.isInteger(thousands) ? thousands.toFixed(0) : thousands}K`;
  }
  return `${Math.round(value / 1000)}K`;
}

/** Tools whose name, title or description contains every word of `query`, case-insensitively. */
export function filterMcpTools(
  tools: readonly McpToolEntry[],
  query: string,
): McpToolEntry[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [...tools];
  return tools.filter((tool) => {
    const haystack = [tool.name, tool.title ?? "", tool.summary, tool.description]
      .join("\n")
      .toLowerCase();
    return terms.every((term) => haystack.includes(term));
  });
}

export interface McpToolTotals {
  on: number;
  total: number;
  tokens: number;
}

/** What the tools still on cost, under the switches as they are now (which may be ahead of the server). */
export function mcpToolTotals(
  tools: readonly McpToolEntry[],
  disabled: ReadonlySet<string>,
): McpToolTotals {
  let on = 0;
  let tokens = 0;
  for (const tool of tools) {
    if (disabled.has(tool.name)) continue;
    on += 1;
    tokens += tool.tokens;
  }
  return { on, total: tools.length, tokens };
}

/** "12 of 38 tools on · ~4.1K tokens per request". */
export function mcpToolSummary(totals: McpToolTotals): string {
  const noun = totals.total === 1 ? "tool" : "tools";
  const head = `${totals.on} of ${totals.total} ${noun} on`;
  return totals.on === 0
    ? `${head} · adds nothing to requests`
    : `${head} · ~${formatTokenCount(totals.tokens)} tokens per request`;
}

/** How the figures were made, and their share of the loaded model's window when one is known. */
export function mcpToolCostNote(
  tokens: number,
  measured: boolean,
  contextTokens: number | null,
): string {
  const how = measured
    ? "Counted with the loaded model's tokenizer"
    : "Estimated from each tool's schema";
  if (!contextTokens || contextTokens <= 0 || tokens <= 0) return `${how}.`;
  const share = (tokens / contextTokens) * 100;
  const percent = share < 1 ? "<1" : String(Math.round(share));
  return `${how}: ${percent}% of the loaded model's ${formatTokenCount(contextTokens)} context.`;
}

/** `names` added to (`include`) or removed from a name set, as a new set. */
export function withNames(
  set: ReadonlySet<string>,
  names: Iterable<string>,
  include: boolean,
): Set<string> {
  const next = new Set(set);
  for (const name of names) {
    if (include) next.add(name);
    else next.delete(name);
  }
  return next;
}

/** The stored form: sorted, so equal sets compare and save equal. */
export function sortedNames(set: ReadonlySet<string>): string[] {
  return [...set].sort();
}

export interface LatestSaver<T, R> {
  /** Save `value`, after any save already in flight. Values queued meanwhile collapse into the newest, so a
   *  burst of switch flips sends at most two requests and the last one carries the final state. Resolves with
   *  the response of the save that carried this value or a newer one. */
  save(value: T): Promise<R>;
  /** A save is in flight or queued. */
  busy(): boolean;
}

/** One write at a time, newest value wins. Each request carries the whole set, so two in flight could land in
 *  either order and leave the older one saved; serializing them here keeps the last flip the one that sticks. */
export function createLatestSaver<T, R>(
  write: (value: T) => Promise<R>,
): LatestSaver<T, R> {
  type Waiter = { resolve: (result: R) => void; reject: (error: unknown) => void };
  let inFlight = false;
  let queued: { value: T; waiters: Waiter[] } | null = null;

  function pump(): void {
    if (inFlight || queued === null) return;
    const { value, waiters } = queued;
    queued = null;
    inFlight = true;
    let outcome: Promise<R>;
    try {
      outcome = write(value);
    } catch (error) {
      outcome = Promise.reject(error);
    }
    outcome.then(
      (result) => {
        inFlight = false;
        for (const waiter of waiters) waiter.resolve(result);
        pump();
      },
      (error: unknown) => {
        inFlight = false;
        for (const waiter of waiters) waiter.reject(error);
        pump();
      },
    );
  }

  return {
    save(value: T): Promise<R> {
      return new Promise<R>((resolve, reject) => {
        if (queued) {
          queued.value = value;
          queued.waiters.push({ resolve, reject });
        } else {
          queued = { value, waiters: [{ resolve, reject }] };
        }
        pump();
      });
    },
    busy(): boolean {
      return inFlight || queued !== null;
    },
  };
}
