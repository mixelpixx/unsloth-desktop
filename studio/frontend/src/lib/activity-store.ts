// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The list behind the sidebar's activity bell (features/activity). An aggregator, not a job
// manager: every subsystem already tracks its own work, on the server and in its own store (Hub
// downloads, the training run and its SSE, export status, recipe executions, model loads), and the
// adapters in features/activity copy what those already know into entries here. The toast override
// in components/ui/sonner.tsx records every error toast it shows, so an error outlives its 15 s
// toast until it is dismissed. In lib/, not a feature, since the toaster records, the features
// feed, the shell renders and general-tab lists the storage keys. Imports nothing but zustand so
// tests can drive it directly.

import { create } from "zustand";

export type ActivityKind =
  | "download"
  | "training"
  | "export"
  | "recipe"
  | "model-load"
  | "error";

export type ActivityState = "active" | "done" | "failed" | "cancelled";

export type ActivityAction = "open" | "logs" | "cancel" | "retry" | "dismiss";

/** The Settings > Logs families a row can open, as view-logs-action names them. */
export const ACTIVITY_LOG_FAMILIES = [
  "server",
  "llama-server",
  "diffusion-server",
] as const;

export type ActivityLogFamily = (typeof ACTIVITY_LOG_FAMILIES)[number];

/** Which log View logs opens: a family, and the file when the failure named one. */
export interface ActivityLogTarget {
  family: ActivityLogFamily;
  sourcePath: string | null;
}

/** Counted progress the row spells out ("1.2 GB of 4 GB", "Step 40 of 500"). */
export interface ActivityMeter {
  unit: "bytes" | "steps" | "rows";
  done: number;
  total: number;
}

/** What a download Retry starts again: the request the download manager ran it with. Plain data,
 *  so a failed download read back after a reload can still be retried. */
export interface ActivityDownloadRetry {
  kind: "model" | "dataset";
  repoId: string;
  variant: string | null;
  expectedBytes: number;
  inventoryKind?: "model" | "gguf";
  scopeId?: string;
  files?: string[];
  checkpoint?: boolean;
}

export interface ActivityEntry {
  id: string;
  kind: ActivityKind;
  /** What it is about: a model, a repo, a run name, or an error's own message. Never UI copy, so
   *  the row reads right after a language switch; an empty title gets the kind's name. */
  title: string;
  detail: string | null;
  state: ActivityState;
  /** 0..1 while active, null when the source cannot tell. */
  progress: number | null;
  meter: ActivityMeter | null;
  startedAt: number;
  finishedAt: number | null;
  actions: ActivityAction[];
  /** The page Open goes to. */
  route: string | null;
  logs: ActivityLogTarget | null;
  /** The source's own handle: a download job key, a model runtime, a training job id. */
  ref: string | null;
  retry: ActivityDownloadRetry | null;
  /** Bytes moved, for a download: the native notification's test for a long one. */
  bytes: number | null;
  /** How many identical errors this entry stands for. */
  count: number;
}

/** What an adapter reports for one job of its source. Errors come in through recordActivityError. */
export interface ActivitySnapshot {
  id: string;
  kind: Exclude<ActivityKind, "error">;
  title: string;
  state: ActivityState;
  startedAt: number;
  detail?: string | null;
  progress?: number | null;
  meter?: ActivityMeter | null;
  route?: string | null;
  logs?: ActivityLogTarget | null;
  ref?: string | null;
  retry?: ActivityDownloadRetry | null;
  bytes?: number | null;
  /** The source can stop it right now. */
  cancellable?: boolean;
}

export interface ReconcileOptions {
  /** The ids this source owns, e.g. "download:". */
  prefix: string;
  /** What becomes of an active entry the source stopped reporting: left alone (the source may
   *  just be out of view), settled as cancelled, or removed as never having been a job at all. */
  missing: "keep" | "cancel" | "drop";
}

export interface ActivityErrorInput {
  title: string;
  detail?: string | null;
  logs?: ActivityLogTarget | null;
}

export const ACTIVITY_MAX_ENTRIES = 100;
export const ACTIVITY_HISTORY_LIMIT = 50;
/** Identical errors closer together than this are one entry with a count. */
export const ACTIVITY_ERROR_DEDUPE_MS = 10_000;
export const ACTIVITY_HISTORY_KEY = "unsloth_activity_history";
export const ACTIVITY_NOTIFY_KEY = "unsloth_activity_notify";
/** Every key this store writes, for "Reset all local preferences". */
export const ACTIVITY_PREFERENCE_KEYS = [
  ACTIVITY_HISTORY_KEY,
  ACTIVITY_NOTIFY_KEY,
] as const;

const KINDS: readonly ActivityKind[] = [
  "download",
  "training",
  "export",
  "recipe",
  "model-load",
  "error",
];
const FINISHED_STATES: readonly ActivityState[] = [
  "done",
  "failed",
  "cancelled",
];
const METER_UNITS: readonly ActivityMeter["unit"][] = [
  "bytes",
  "steps",
  "rows",
];
const HISTORY_VERSION = 1;

/** The controls a row offers, from what the entry knows. Training, export and recipes are never
 *  re-run from here: Open takes the user to the screen that owns them. */
export function actionsFor(
  entry: Pick<ActivityEntry, "kind" | "state" | "route" | "logs" | "retry">,
  cancellable = false,
): ActivityAction[] {
  const actions: ActivityAction[] = [];
  if (entry.route) actions.push("open");
  if (entry.logs && entry.state === "failed") actions.push("logs");
  if (entry.state === "active" && cancellable) actions.push("cancel");
  if (
    entry.kind === "download" &&
    entry.retry &&
    (entry.state === "failed" || entry.state === "cancelled")
  ) {
    actions.push("retry");
  }
  if (entry.state !== "active") actions.push("dismiss");
  return actions;
}

function clampProgress(value: number | null | undefined): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return Math.min(1, Math.max(0, value));
}

function entryFromSnapshot(snapshot: ActivitySnapshot): ActivityEntry {
  const entry: ActivityEntry = {
    id: snapshot.id,
    kind: snapshot.kind,
    title: snapshot.title,
    detail: snapshot.detail ?? null,
    state: "active",
    progress: clampProgress(snapshot.progress),
    meter: snapshot.meter ?? null,
    startedAt: snapshot.startedAt,
    finishedAt: null,
    actions: [],
    route: snapshot.route ?? null,
    logs: snapshot.logs ?? null,
    ref: snapshot.ref ?? null,
    retry: snapshot.retry ?? null,
    bytes: snapshot.bytes ?? null,
    count: 1,
  };
  entry.actions = actionsFor(entry, snapshot.cancellable === true);
  return entry;
}

function settle(
  current: ActivityEntry,
  state: Exclude<ActivityState, "active">,
  now: number,
  snapshot?: ActivitySnapshot,
): ActivityEntry {
  const next: ActivityEntry = {
    ...current,
    state,
    finishedAt: now,
    detail: snapshot?.detail !== undefined ? snapshot.detail : current.detail,
    progress:
      state === "done"
        ? 1
        : snapshot?.progress !== undefined
          ? clampProgress(snapshot.progress)
          : current.progress,
    meter: snapshot?.meter !== undefined ? snapshot.meter : current.meter,
    logs: snapshot?.logs !== undefined ? snapshot.logs : current.logs,
    retry: snapshot?.retry !== undefined ? snapshot.retry : current.retry,
    bytes: snapshot?.bytes !== undefined ? snapshot.bytes : current.bytes,
  };
  next.actions = actionsFor(next);
  return next;
}

/** Cheap identity of what a row draws, so a report that changed nothing leaves the store alone. */
function entrySignature(entry: ActivityEntry): string {
  return JSON.stringify([
    entry.title,
    entry.detail,
    entry.state,
    entry.progress,
    entry.meter,
    entry.startedAt,
    entry.finishedAt,
    entry.actions,
    entry.route,
    entry.logs,
    entry.ref,
    entry.retry,
    entry.bytes,
    entry.count,
  ]);
}

/**
 * Fold one source's report into the list. An active snapshot creates or refreshes its entry; a
 * finished one settles an entry that was seen running, and nothing else: a run that ended before
 * the bell saw it (hydrated history, a job lingering in its store) is not news. `finished` lists
 * the entries that settled in this call, for the native notification.
 */
export function reconcileEntries(
  entries: readonly ActivityEntry[],
  snapshots: readonly ActivitySnapshot[],
  options: ReconcileOptions,
  now: number,
): { entries: ActivityEntry[]; finished: ActivityEntry[]; changed: boolean } {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const replaced = new Map<string, ActivityEntry | null>();
  const added: ActivityEntry[] = [];
  const finished: ActivityEntry[] = [];
  const reported = new Set<string>();

  for (const snapshot of snapshots) {
    reported.add(snapshot.id);
    const current = byId.get(snapshot.id);
    if (snapshot.state === "active") {
      const fresh = entryFromSnapshot(snapshot);
      if (!current) {
        added.push(fresh);
        byId.set(fresh.id, fresh);
      } else if (current.state === "active") {
        const merged = { ...fresh, startedAt: current.startedAt };
        if (entrySignature(merged) !== entrySignature(current)) {
          replaced.set(current.id, merged);
        }
      } else {
        // Running again under the same id (a stream glitch read as a failure): the newer word wins.
        replaced.set(current.id, fresh);
      }
      continue;
    }
    if (current?.state !== "active") continue;
    const settled = settle(current, snapshot.state, now, snapshot);
    replaced.set(current.id, settled);
    finished.push(settled);
  }

  if (options.missing !== "keep") {
    for (const entry of entries) {
      if (
        entry.state !== "active" ||
        !entry.id.startsWith(options.prefix) ||
        reported.has(entry.id)
      ) {
        continue;
      }
      if (options.missing === "drop") {
        replaced.set(entry.id, null);
      } else {
        const settled = settle(entry, "cancelled", now);
        replaced.set(entry.id, settled);
        finished.push(settled);
      }
    }
  }

  if (replaced.size === 0 && added.length === 0) {
    return { entries: entries as ActivityEntry[], finished, changed: false };
  }
  const next: ActivityEntry[] = [];
  for (const entry of entries) {
    if (!replaced.has(entry.id)) {
      next.push(entry);
      continue;
    }
    const replacement = replaced.get(entry.id);
    if (replacement) next.push(replacement);
  }
  next.push(...added);
  return { entries: next, finished, changed: true };
}

/** Settle one active entry directly, for a source that announces the end rather than a state. */
export function settleEntry(
  entries: readonly ActivityEntry[],
  id: string,
  state: Exclude<ActivityState, "active"> | "drop",
  now: number,
  detail?: string | null,
): { entries: ActivityEntry[]; finished: ActivityEntry | null } {
  const index = entries.findIndex((entry) => entry.id === id);
  const current = entries[index];
  if (!current || current.state !== "active") {
    return { entries: entries as ActivityEntry[], finished: null };
  }
  const next = entries.slice();
  if (state === "drop") {
    next.splice(index, 1);
    return { entries: next, finished: null };
  }
  const settled = settle(
    current,
    state,
    now,
    detail !== undefined ? ({ detail } as ActivitySnapshot) : undefined,
  );
  next[index] = settled;
  return { entries: next, finished: settled };
}

/**
 * Add an error, or count it against the identical one still within the dedupe window: a request
 * failing on every poll is one row that says how often, not a screenful.
 */
export function addErrorEntry(
  entries: readonly ActivityEntry[],
  input: ActivityErrorInput,
  now: number,
  id: string,
): { entries: ActivityEntry[]; entry: ActivityEntry; deduped: boolean } {
  const detail = input.detail ?? null;
  const index = entries.findIndex(
    (entry) =>
      entry.kind === "error" &&
      entry.title === input.title &&
      entry.detail === detail &&
      now - (entry.finishedAt ?? entry.startedAt) < ACTIVITY_ERROR_DEDUPE_MS,
  );
  if (index !== -1) {
    const current = entries[index];
    const entry: ActivityEntry = {
      ...current,
      count: current.count + 1,
      finishedAt: now,
      logs: input.logs ?? current.logs,
    };
    entry.actions = actionsFor(entry);
    const next = entries.slice();
    next[index] = entry;
    return { entries: next, entry, deduped: true };
  }
  const entry: ActivityEntry = {
    id,
    kind: "error",
    title: input.title,
    detail,
    state: "failed",
    progress: null,
    meter: null,
    startedAt: now,
    finishedAt: now,
    actions: [],
    route: null,
    logs: input.logs ?? null,
    ref: null,
    retry: null,
    bytes: null,
    count: 1,
  };
  entry.actions = actionsFor(entry);
  return { entries: [...entries, entry], entry, deduped: false };
}

function lastTouched(entry: ActivityEntry): number {
  return entry.finishedAt ?? entry.startedAt;
}

/** At most `max` entries. Running jobs always stay; the oldest finished ones go first. */
export function capEntries(
  entries: readonly ActivityEntry[],
  max: number = ACTIVITY_MAX_ENTRIES,
): ActivityEntry[] {
  if (entries.length <= max) return entries as ActivityEntry[];
  const active = entries.filter((entry) => entry.state === "active").length;
  const room = Math.max(0, max - active);
  const keep = new Set(
    entries
      .filter((entry) => entry.state !== "active")
      .sort((a, b) => lastTouched(b) - lastTouched(a))
      .slice(0, room)
      .map((entry) => entry.id),
  );
  return entries.filter(
    (entry) => entry.state === "active" || keep.has(entry.id),
  );
}

export function selectActiveEntries(
  entries: readonly ActivityEntry[],
): ActivityEntry[] {
  return entries
    .filter((entry) => entry.state === "active")
    .sort((a, b) => b.startedAt - a.startedAt);
}

/** Finished jobs, newest first. Errors have their own tab. */
export function selectRecentEntries(
  entries: readonly ActivityEntry[],
): ActivityEntry[] {
  return entries
    .filter((entry) => entry.state !== "active" && entry.kind !== "error")
    .sort((a, b) => lastTouched(b) - lastTouched(a));
}

/** Every recorded error and every job that failed, newest first. */
export function selectErrorEntries(
  entries: readonly ActivityEntry[],
): ActivityEntry[] {
  return entries
    .filter((entry) => entry.kind === "error" || entry.state === "failed")
    .sort((a, b) => lastTouched(b) - lastTouched(a));
}

export function countActiveEntries(entries: readonly ActivityEntry[]): number {
  let count = 0;
  for (const entry of entries) if (entry.state === "active") count += 1;
  return count;
}

export function countUnseenErrors(
  entries: readonly ActivityEntry[],
  seenAt: number,
): number {
  let count = 0;
  for (const entry of entries) {
    if (
      (entry.kind === "error" || entry.state === "failed") &&
      lastTouched(entry) > seenAt
    ) {
      count += 1;
    }
  }
  return count;
}

/** The finished entries worth a reload, newest first: running ones are re-reported by their source. */
export function historyEntries(
  entries: readonly ActivityEntry[],
  limit: number = ACTIVITY_HISTORY_LIMIT,
): ActivityEntry[] {
  return entries
    .filter((entry) => entry.state !== "active")
    .sort((a, b) => lastTouched(b) - lastTouched(a))
    .slice(0, limit);
}

export function serializeActivityHistory(
  entries: readonly ActivityEntry[],
  seenErrorsAt: number,
): string {
  return JSON.stringify({
    v: HISTORY_VERSION,
    seenErrorsAt,
    entries: historyEntries(entries),
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteOr<T>(value: unknown, fallback: T): number | T {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringOr<T>(value: unknown, fallback: T): string | T {
  return typeof value === "string" ? value : fallback;
}

function parseLogs(value: unknown): ActivityLogTarget | null {
  if (!isRecord(value)) return null;
  const family = ACTIVITY_LOG_FAMILIES.find((name) => name === value.family);
  if (!family) return null;
  return { family, sourcePath: stringOr(value.sourcePath, null) };
}

function parseMeter(value: unknown): ActivityMeter | null {
  if (!isRecord(value)) return null;
  const unit = METER_UNITS.find((name) => name === value.unit);
  const done = finiteOr(value.done, null);
  const total = finiteOr(value.total, null);
  if (!unit || done === null || total === null) return null;
  return { unit, done, total };
}

function parseRetry(value: unknown): ActivityDownloadRetry | null {
  if (!isRecord(value)) return null;
  const kind =
    value.kind === "model" || value.kind === "dataset" ? value.kind : null;
  const repoId = stringOr(value.repoId, null);
  if (!kind || !repoId) return null;
  const retry: ActivityDownloadRetry = {
    kind,
    repoId,
    variant: stringOr(value.variant, null),
    expectedBytes: finiteOr(value.expectedBytes, 0),
  };
  if (value.inventoryKind === "model" || value.inventoryKind === "gguf") {
    retry.inventoryKind = value.inventoryKind;
  }
  if (typeof value.scopeId === "string") retry.scopeId = value.scopeId;
  if (Array.isArray(value.files)) {
    retry.files = value.files.filter(
      (file): file is string => typeof file === "string",
    );
  }
  if (typeof value.checkpoint === "boolean")
    retry.checkpoint = value.checkpoint;
  return retry;
}

function parseEntry(value: unknown): ActivityEntry | null {
  if (!isRecord(value)) return null;
  const id = stringOr(value.id, null);
  const kind = KINDS.find((name) => name === value.kind);
  // Only finished entries are written, and a running one read back would never settle.
  const state = FINISHED_STATES.find((name) => name === value.state);
  const startedAt = finiteOr(value.startedAt, null);
  if (!id || !kind || !state || startedAt === null) return null;
  const entry: ActivityEntry = {
    id,
    kind,
    title: stringOr(value.title, ""),
    detail: stringOr(value.detail, null),
    state,
    progress: clampProgress(finiteOr(value.progress, null)),
    meter: parseMeter(value.meter),
    startedAt,
    finishedAt: finiteOr(value.finishedAt, startedAt),
    actions: [],
    route: stringOr(value.route, null),
    logs: parseLogs(value.logs),
    ref: stringOr(value.ref, null),
    retry: parseRetry(value.retry),
    bytes: finiteOr(value.bytes, null),
    count: Math.max(1, Math.floor(finiteOr(value.count, 1))),
  };
  entry.actions = actionsFor(entry);
  return entry;
}

/** The history a reload starts from. Anything malformed is dropped rather than trusted. */
export function parseActivityHistory(raw: string | null): {
  entries: ActivityEntry[];
  seenErrorsAt: number;
} {
  if (!raw) return { entries: [], seenErrorsAt: 0 };
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { entries: [], seenErrorsAt: 0 };
  }
  if (!isRecord(parsed) || parsed.v !== HISTORY_VERSION) {
    return { entries: [], seenErrorsAt: 0 };
  }
  const seen = new Set<string>();
  const entries = (Array.isArray(parsed.entries) ? parsed.entries : [])
    .map(parseEntry)
    .filter((entry): entry is ActivityEntry => {
      if (!entry || seen.has(entry.id)) return false;
      seen.add(entry.id);
      return true;
    });
  return {
    entries: historyEntries(entries),
    seenErrorsAt: finiteOr(parsed.seenErrorsAt, 0),
  };
}

// ── The store ───────────────────────────────────────────────────────

export interface ActivityStoreState {
  entries: ActivityEntry[];
  /** Errors newer than this light the bell's red dot. */
  seenErrorsAt: number;
  /** "Notify me when long jobs finish". Off until the user turns it on, which asks permission. */
  notifyOnFinish: boolean;
}

function storage(): Storage | null {
  try {
    return typeof globalThis.localStorage === "undefined"
      ? null
      : globalThis.localStorage;
  } catch {
    // A sandboxed frame throws on the getter itself.
    return null;
  }
}

function readStoredState(): ActivityStoreState {
  const store = storage();
  let history: string | null = null;
  let notify: string | null = null;
  try {
    history = store?.getItem(ACTIVITY_HISTORY_KEY) ?? null;
    notify = store?.getItem(ACTIVITY_NOTIFY_KEY) ?? null;
  } catch {
    // Blocked storage: start empty, which is what a first run looks like anyway.
  }
  const { entries, seenErrorsAt } = parseActivityHistory(history);
  return { entries, seenErrorsAt, notifyOnFinish: notify === "1" };
}

export const useActivityStore = create<ActivityStoreState>(() =>
  readStoredState(),
);

let persistQueued = false;

function writeHistory(): void {
  const store = storage();
  if (!store) return;
  const { entries, seenErrorsAt } = useActivityStore.getState();
  try {
    store.setItem(
      ACTIVITY_HISTORY_KEY,
      serializeActivityHistory(entries, seenErrorsAt),
    );
  } catch {
    // Full or blocked storage: the history just does not survive a reload.
  }
}

/** One write per burst: a page of errors in one tick is one setItem. */
function persistSoon(): void {
  if (persistQueued) return;
  persistQueued = true;
  queueMicrotask(() => {
    persistQueued = false;
    writeHistory();
  });
}

type FinishedListener = (entry: ActivityEntry) => void;
const finishedListeners = new Set<FinishedListener>();

/** Calls back for each job that settles from here on (not for recorded errors). */
export function subscribeActivityFinished(
  listener: FinishedListener,
): () => void {
  finishedListeners.add(listener);
  return () => finishedListeners.delete(listener);
}

function announce(finished: readonly ActivityEntry[]): void {
  for (const entry of finished) {
    for (const listener of finishedListeners) {
      try {
        listener(entry);
      } catch {
        // A listener's failure must not stop the others, nor the store.
      }
    }
  }
}

function commit(
  entries: ActivityEntry[],
  finished: readonly ActivityEntry[],
): void {
  useActivityStore.setState({ entries: capEntries(entries) });
  if (finished.length > 0) {
    persistSoon();
    announce(finished);
  }
}

/** One source's current jobs. See reconcileEntries. */
export function reportActivity(
  snapshots: readonly ActivitySnapshot[],
  options: ReconcileOptions,
  now: number = Date.now(),
): ActivityEntry[] {
  const result = reconcileEntries(
    useActivityStore.getState().entries,
    snapshots,
    options,
    now,
  );
  if (result.changed) commit(result.entries, result.finished);
  return result.finished;
}

/** Settle one running entry; "drop" removes it, for an end nobody can vouch for. */
export function settleActivity(
  id: string,
  state: Exclude<ActivityState, "active"> | "drop",
  detail?: string | null,
  now: number = Date.now(),
): ActivityEntry | null {
  const current = useActivityStore.getState().entries;
  const result = settleEntry(current, id, state, now, detail);
  if (result.entries === current) return null;
  commit(result.entries, result.finished ? [result.finished] : []);
  return result.finished;
}

let errorSequence = 0;

export function recordActivityError(
  input: ActivityErrorInput,
  now: number = Date.now(),
): ActivityEntry {
  errorSequence += 1;
  const result = addErrorEntry(
    useActivityStore.getState().entries,
    input,
    now,
    `error:${now}:${errorSequence}`,
  );
  useActivityStore.setState({ entries: capEntries(result.entries) });
  persistSoon();
  return result.entry;
}

function toastText(value: unknown): string | null {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value !== "string") return null;
  const text = value.trim();
  return text.length > 0 ? text : null;
}

/** The log a toast's View logs action opens (features/settings/lib/view-logs-action.ts tags it). */
export function toastLogTarget(action: unknown): ActivityLogTarget | null {
  if (!isRecord(action)) return null;
  return parseLogs(action.logTarget);
}

/**
 * Record an error toast as it is shown. Text only: a React node has no text to keep, so a toast
 * that is nothing else is still recorded, with an empty title the row fills in.
 */
export function recordToastError(
  message: unknown,
  description: unknown,
  action: unknown,
  now: number = Date.now(),
): ActivityEntry {
  const title = toastText(message);
  const detail = toastText(description);
  return recordActivityError(
    {
      title: title ?? detail ?? "",
      detail: title ? detail : null,
      logs: toastLogTarget(action),
    },
    now,
  );
}

export function dismissActivity(id: string): void {
  const { entries } = useActivityStore.getState();
  const next = entries.filter(
    (entry) => entry.id !== id || entry.state === "active",
  );
  if (next.length === entries.length) return;
  useActivityStore.setState({ entries: next });
  persistSoon();
}

/** Empty one tab. Running jobs are never cleared: their source would only put them back. */
export function clearActivity(tab: "recent" | "errors"): void {
  const { entries } = useActivityStore.getState();
  const cleared = new Set(
    (tab === "recent"
      ? selectRecentEntries(entries)
      : selectErrorEntries(entries)
    ).map((entry) => entry.id),
  );
  if (cleared.size === 0) return;
  useActivityStore.setState({
    entries: entries.filter((entry) => !cleared.has(entry.id)),
  });
  persistSoon();
}

export function markActivityErrorsSeen(now: number = Date.now()): void {
  const { entries, seenErrorsAt } = useActivityStore.getState();
  if (countUnseenErrors(entries, seenErrorsAt) === 0) return;
  useActivityStore.setState({ seenErrorsAt: now });
  persistSoon();
}

export function setActivityNotifyOnFinish(on: boolean): void {
  useActivityStore.setState({ notifyOnFinish: on });
  try {
    storage()?.setItem(ACTIVITY_NOTIFY_KEY, on ? "1" : "0");
  } catch {
    // The toggle still holds for this session.
  }
}

/** Forget everything, stored history included: a signed-out session's errors are not the next one's. */
export function resetActivity(): void {
  useActivityStore.setState({ entries: [], seenErrorsAt: 0 });
  try {
    storage()?.removeItem(ACTIVITY_HISTORY_KEY);
  } catch {
    // Nothing stored to forget.
  }
}
