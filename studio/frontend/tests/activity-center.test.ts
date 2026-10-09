// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The jobs and notification center: the store in lib/activity-store.ts, the source mappings in
// features/activity/activity-sources.ts, the outcome and cancel hooks in model-lifecycle-events,
// and the wiring (toast override, sidebar header, reset list) read from the shipped source.

import assert from "node:assert/strict";
import test from "node:test";

import type { DownloadJobLike } from "../src/features/activity/activity-sources.ts";
import type {
  ActivityEntry,
  ActivitySnapshot,
} from "../src/lib/activity-store.ts";
import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

// The store reads its history from localStorage as it is created, and the lifecycle events
// dispatch on `window`, so both stand up before either module is imported.
const backing = new Map<string, string>();
const fakeStorage = {
  getItem: (key: string) => backing.get(key) ?? null,
  setItem: (key: string, value: string) => void backing.set(key, value),
  removeItem: (key: string) => void backing.delete(key),
};
backing.set(
  "unsloth_activity_history",
  JSON.stringify({
    v: 1,
    seenErrorsAt: 5,
    entries: [
      {
        id: "download:a@1",
        kind: "download",
        title: "unsloth/a",
        state: "failed",
        startedAt: 1,
        finishedAt: 10,
        retry: {
          kind: "model",
          repoId: "unsloth/a",
          variant: null,
          expectedBytes: 3,
        },
      },
      // Running entries are never written; one read back would never settle.
      {
        id: "training:x",
        kind: "training",
        title: "m",
        state: "active",
        startedAt: 2,
      },
      { id: "bad", kind: "nope", state: "done", startedAt: 3 },
    ],
  }),
);
backing.set("unsloth_activity_notify", "1");
(globalThis as { localStorage?: unknown }).localStorage = fakeStorage;
(globalThis as { window?: unknown }).window ??= new EventTarget();

const store = await import("../src/lib/activity-store.ts");
const sources = await import("../src/features/activity/activity-sources.ts");
const lifecycle = await import("../src/lib/model-lifecycle-events.ts");

const {
  ACTIVITY_ERROR_DEDUPE_MS,
  ACTIVITY_HISTORY_KEY,
  ACTIVITY_HISTORY_LIMIT,
  ACTIVITY_MAX_ENTRIES,
  ACTIVITY_NOTIFY_KEY,
  ACTIVITY_PREFERENCE_KEYS,
  actionsFor,
  addErrorEntry,
  capEntries,
  countUnseenErrors,
  parseActivityHistory,
  reconcileEntries,
  serializeActivityHistory,
  settleEntry,
  useActivityStore,
} = store;

const flushMicrotasks = () =>
  new Promise<void>((resolve) => setImmediate(resolve));

function snapshot(overrides: Partial<ActivitySnapshot> = {}): ActivitySnapshot {
  return {
    id: "download:k@1",
    kind: "download",
    title: "unsloth/Qwen3-8B-GGUF",
    state: "active",
    startedAt: 1_000,
    route: "/hub",
    ...overrides,
  };
}

function finished(
  id: string,
  at: number,
  overrides: Partial<ActivityEntry> = {},
): ActivityEntry {
  return {
    id,
    kind: "download",
    title: id,
    detail: null,
    state: "done",
    progress: 1,
    meter: null,
    startedAt: at - 1,
    finishedAt: at,
    actions: ["dismiss"],
    route: null,
    logs: null,
    ref: null,
    retry: null,
    bytes: null,
    count: 1,
    ...overrides,
  };
}

const OPTIONS = { prefix: "download:", missing: "cancel" } as const;

// ── Reading back a reload ──────────────────────────────────────────

test("a reload starts from the stored finished history and the notify choice", () => {
  const state = useActivityStore.getState();
  assert.deepEqual(
    state.entries.map((entry) => entry.id),
    ["download:a@1"],
    "running and malformed entries are dropped",
  );
  assert.equal(state.seenErrorsAt, 5);
  assert.equal(state.notifyOnFinish, true);
  // Actions are recomputed, not trusted: a failed download with its request can be retried.
  assert.deepEqual(state.entries[0]?.actions, ["retry", "dismiss"]);
});

// ── State transitions ──────────────────────────────────────────────

test("an active report creates a row, and a later one refreshes it in place", () => {
  const first = reconcileEntries(
    [],
    [snapshot({ progress: 0.25, cancellable: true })],
    OPTIONS,
    2_000,
  );
  assert.equal(first.changed, true);
  assert.equal(first.entries.length, 1);
  const row = first.entries[0];
  assert.equal(row.state, "active");
  assert.equal(row.progress, 0.25);
  assert.deepEqual(row.actions, ["open", "cancel"]);

  const second = reconcileEntries(
    first.entries,
    [snapshot({ progress: 0.5, startedAt: 9_999, cancellable: true })],
    OPTIONS,
    3_000,
  );
  assert.equal(second.entries[0].progress, 0.5);
  assert.equal(
    second.entries[0].startedAt,
    1_000,
    "the first report's start time is kept",
  );

  const same = reconcileEntries(
    second.entries,
    [snapshot({ progress: 0.5, cancellable: true })],
    OPTIONS,
    4_000,
  );
  assert.equal(same.changed, false);
  assert.equal(
    same.entries,
    second.entries,
    "a report that moved nothing leaves the list alone",
  );
});

test("a finished report settles a row that was seen running, and nothing else", () => {
  const running = reconcileEntries([], [snapshot()], OPTIONS, 2_000).entries;
  const done = reconcileEntries(
    running,
    [snapshot({ state: "done", bytes: 5 })],
    OPTIONS,
    5_000,
  );
  assert.equal(done.finished.length, 1);
  const row = done.entries[0];
  assert.equal(row.state, "done");
  assert.equal(row.finishedAt, 5_000);
  assert.equal(row.progress, 1);
  assert.deepEqual(row.actions, ["open", "dismiss"]);

  // A job that ended before the bell saw it run is history, not news.
  const unseen = reconcileEntries(
    [],
    [snapshot({ state: "failed" })],
    OPTIONS,
    5_000,
  );
  assert.equal(unseen.changed, false);
  assert.deepEqual(unseen.entries, []);

  // Settled once: a second terminal report is not a second finish.
  const again = reconcileEntries(
    done.entries,
    [snapshot({ state: "done" })],
    OPTIONS,
    6_000,
  );
  assert.equal(again.finished.length, 0);
});

test("a row its source stopped reporting is kept, cancelled or dropped as the source says", () => {
  const running = reconcileEntries([], [snapshot()], OPTIONS, 2_000).entries;
  const cancelled = reconcileEntries(running, [], OPTIONS, 3_000);
  assert.equal(cancelled.entries[0].state, "cancelled");
  assert.equal(cancelled.finished.length, 1);

  const kept = reconcileEntries(
    running,
    [],
    { prefix: "download:", missing: "keep" },
    3_000,
  );
  assert.equal(kept.changed, false);

  const dropped = reconcileEntries(
    running,
    [],
    { prefix: "download:", missing: "drop" },
    3_000,
  );
  assert.deepEqual(dropped.entries, []);

  // Another source's rows are not this one's to settle.
  const other = reconcileEntries(
    running,
    [],
    { prefix: "export:", missing: "cancel" },
    3_000,
  );
  assert.equal(other.changed, false);
});

test("settleEntry ends one running row, or drops it when nobody can vouch for the end", () => {
  const running = reconcileEntries([], [snapshot()], OPTIONS, 2_000).entries;
  const failed = settleEntry(
    running,
    "download:k@1",
    "failed",
    3_000,
    "disk full",
  );
  assert.equal(failed.finished?.state, "failed");
  assert.equal(failed.finished?.detail, "disk full");
  assert.equal(
    settleEntry(failed.entries, "download:k@1", "done", 4_000).finished,
    null,
  );
  assert.deepEqual(
    settleEntry(running, "download:k@1", "drop", 3_000).entries,
    [],
  );
});

test("controls follow the rules: retry for downloads only, cancel only while running", () => {
  const base = {
    route: "/studio",
    logs: { family: "server", sourcePath: null },
    retry: null,
  } as const;
  assert.deepEqual(actionsFor({ ...base, kind: "training", state: "failed" }), [
    "open",
    "logs",
    "dismiss",
  ]);
  assert.deepEqual(
    actionsFor({ ...base, kind: "training", state: "active" }, true),
    ["open", "cancel"],
  );
  assert.deepEqual(actionsFor({ ...base, kind: "training", state: "done" }), [
    "open",
    "dismiss",
  ]);
  const retry = {
    kind: "model",
    repoId: "r",
    variant: null,
    expectedBytes: 0,
  } as const;
  assert.deepEqual(
    actionsFor({ ...base, retry, kind: "download", state: "cancelled" }),
    ["open", "retry", "dismiss"],
  );
  // A training run is never re-run from the bell, whatever it carries.
  assert.ok(
    !actionsFor({ ...base, retry, kind: "training", state: "failed" }).includes(
      "retry",
    ),
  );
});

// ── Errors ─────────────────────────────────────────────────────────

test("identical errors inside the dedupe window are one row with a count", () => {
  const first = addErrorEntry(
    [],
    { title: "Load failed", detail: "OOM" },
    1_000,
    "e1",
  );
  assert.equal(first.deduped, false);
  const second = addErrorEntry(
    first.entries,
    { title: "Load failed", detail: "OOM" },
    1_000 + ACTIVITY_ERROR_DEDUPE_MS - 1,
    "e2",
  );
  assert.equal(second.deduped, true);
  assert.equal(second.entries.length, 1);
  assert.equal(second.entry.count, 2);

  const different = addErrorEntry(
    second.entries,
    { title: "Load failed", detail: "disk" },
    12_000,
    "e3",
  );
  assert.equal(different.entries.length, 2);

  // Measured from the last repeat, so a steady failure stays one row.
  const later = addErrorEntry(
    second.entries,
    { title: "Load failed", detail: "OOM" },
    second.entry.finishedAt! + ACTIVITY_ERROR_DEDUPE_MS,
    "e4",
  );
  assert.equal(later.deduped, false);
  assert.equal(later.entries.length, 2);
});

test("an error toast is recorded with its description and its View logs target", () => {
  const entry = store.recordToastError(
    "Failed to load model",
    "llama-server exited",
    {
      label: "View logs",
      onClick: () => {},
      logTarget: { family: "llama-server", sourcePath: "/logs/llama-1.log" },
    },
    50_000,
  );
  assert.equal(entry.kind, "error");
  assert.equal(entry.state, "failed");
  assert.equal(entry.title, "Failed to load model");
  assert.equal(entry.detail, "llama-server exited");
  assert.deepEqual(entry.logs, {
    family: "llama-server",
    sourcePath: "/logs/llama-1.log",
  });
  assert.deepEqual(entry.actions, ["logs", "dismiss"]);

  // A node with no text still lands, with its description as the title when it has one.
  const nodeOnly = store.recordToastError(
    { type: "span" },
    "only text",
    undefined,
    90_000,
  );
  assert.equal(nodeOnly.title, "only text");
  assert.equal(nodeOnly.detail, null);
  // An action that is not a View logs action opens nothing.
  assert.equal(
    store.toastLogTarget({ label: "Retry", onClick: () => {} }),
    null,
  );
  assert.equal(store.toastLogTarget({ logTarget: { family: "nope" } }), null);
  store.dismissActivity(entry.id);
  store.dismissActivity(nodeOnly.id);
});

test("unseen errors light the bell until they are looked at", () => {
  const entry = store.recordActivityError({ title: "boom" }, 100_000);
  const state = () => useActivityStore.getState();
  assert.ok(countUnseenErrors(state().entries, state().seenErrorsAt) >= 1);
  store.markActivityErrorsSeen(100_001);
  assert.equal(countUnseenErrors(state().entries, state().seenErrorsAt), 0);
  store.dismissActivity(entry.id);
  assert.ok(!state().entries.some((row) => row.id === entry.id));
});

// ── Caps and persistence ───────────────────────────────────────────

test("the list keeps at most 100 rows, running jobs first, then the newest finished", () => {
  const running: ActivityEntry[] = [
    finished("running-1", 1, {
      state: "active",
      finishedAt: null,
      actions: [],
    }),
    finished("running-2", 2, {
      state: "active",
      finishedAt: null,
      actions: [],
    }),
  ];
  const old = Array.from({ length: 120 }, (_, index) =>
    finished(`old-${index}`, 10 + index),
  );
  const capped = capEntries([...running, ...old]);
  assert.equal(capped.length, ACTIVITY_MAX_ENTRIES);
  assert.ok(capped.some((entry) => entry.id === "running-1"));
  assert.ok(
    capped.some((entry) => entry.id === "old-119"),
    "the newest finished row survives",
  );
  assert.ok(
    !capped.some((entry) => entry.id === "old-0"),
    "the oldest goes first",
  );
});

test("only the newest 50 finished rows are written, and a bad record reads back empty", () => {
  const rows = [
    finished("live", 999, { state: "active", finishedAt: null }),
    ...Array.from({ length: 70 }, (_, index) =>
      finished(`row-${index}`, 100 + index),
    ),
  ];
  const raw = serializeActivityHistory(rows, 42);
  const parsed = parseActivityHistory(raw);
  assert.equal(parsed.entries.length, ACTIVITY_HISTORY_LIMIT);
  assert.equal(parsed.entries[0]?.id, "row-69", "newest first");
  assert.ok(!parsed.entries.some((entry) => entry.id === "live"));
  assert.equal(parsed.seenErrorsAt, 42);

  assert.deepEqual(parseActivityHistory("{not json").entries, []);
  assert.deepEqual(
    parseActivityHistory(JSON.stringify({ v: 99, entries: rows })).entries,
    [],
  );
  assert.deepEqual(parseActivityHistory(null).entries, []);
});

test("a settled job is written to storage once per burst and announced to listeners", async () => {
  const heard: string[] = [];
  const stop = store.subscribeActivityFinished((entry) => heard.push(entry.id));
  store.reportActivity(
    [snapshot({ id: "download:p@7", startedAt: 7 })],
    OPTIONS,
    200_000,
  );
  store.reportActivity(
    [snapshot({ id: "download:p@7", startedAt: 7, state: "done" })],
    OPTIONS,
    200_100,
  );
  stop();
  assert.deepEqual(heard, ["download:p@7"]);
  await flushMicrotasks();
  const written = parseActivityHistory(
    backing.get(ACTIVITY_HISTORY_KEY) ?? null,
  );
  assert.ok(written.entries.some((entry) => entry.id === "download:p@7"));

  store.clearActivity("recent");
  assert.ok(
    !useActivityStore
      .getState()
      .entries.some((entry) => entry.id === "download:p@7"),
  );
  store.setActivityNotifyOnFinish(false);
  assert.equal(backing.get(ACTIVITY_NOTIFY_KEY), "0");
  store.resetActivity();
  assert.equal(backing.has(ACTIVITY_HISTORY_KEY), false);
  assert.deepEqual(useActivityStore.getState().entries, []);
});

// ── Source mappings ────────────────────────────────────────────────

const job: DownloadJobLike = {
  key: "model:unsloth/qwen#q4_k_m",
  kind: "model",
  repoId: "unsloth/Qwen",
  variant: "Q4_K_M",
  state: "running",
  downloadedBytes: 512,
  completedBytes: 0,
  expectedBytes: 2048,
  error: null,
  startedAt: 77,
};
const progressOf = (row: DownloadJobLike) => ({
  expectedBytes: row.expectedBytes,
  downloadedBytes: row.downloadedBytes,
  fraction: row.downloadedBytes / row.expectedBytes,
});

test("a download maps to a row with bytes, Cancel while running, and a Retry request", () => {
  const [row] = sources.downloadSnapshots([job], progressOf);
  assert.equal(row.id, "download:model:unsloth/qwen#q4_k_m@77");
  assert.equal(row.title, "unsloth/Qwen · Q4_K_M");
  assert.equal(row.state, "active");
  assert.equal(row.progress, 0.25);
  assert.deepEqual(row.meter, { unit: "bytes", done: 512, total: 2048 });
  assert.equal(row.cancellable, true);
  assert.deepEqual(row.retry, {
    kind: "model",
    repoId: "unsloth/Qwen",
    variant: "Q4_K_M",
    expectedBytes: 2048,
  });

  const [failed] = sources.downloadSnapshots(
    [{ ...job, state: "error", error: "403" }],
    progressOf,
  );
  assert.equal(failed.state, "failed");
  assert.equal(failed.detail, "403");
  const [stopping] = sources.downloadSnapshots(
    [{ ...job, state: "cancelling" }],
    progressOf,
  );
  assert.equal(
    stopping.cancellable,
    false,
    "a cancel already under way is not offered twice",
  );
});

test("a scoped download retries with its scope and files; one without them cannot", () => {
  const scoped = {
    ...job,
    variant: "@hub-assets",
    scopedFiles: ["model.gguf"],
    checkpoint: true,
  };
  assert.equal(sources.downloadTitle(scoped), "unsloth/Qwen");
  assert.deepEqual(sources.downloadRetryFor(scoped), {
    kind: "model",
    repoId: "unsloth/Qwen",
    variant: "@hub-assets",
    expectedBytes: 2048,
    scopeId: "hub-assets",
    files: ["model.gguf"],
    checkpoint: true,
  });
  assert.equal(
    sources.downloadRetryFor({ ...scoped, scopedFiles: undefined }),
    null,
  );
  assert.equal(sources.downloadRetryFor({ ...job, external: true }), null);
});

const training = {
  jobId: "job-1",
  phase: "training",
  isTrainingRunning: true,
  message: "Training...",
  error: null,
  currentStep: 40,
  totalSteps: 200,
  progressPercent: 20,
  startModelName: "unsloth/Llama-3.2-1B",
};

test("a training run maps by phase, counting steps once it trains", () => {
  const [row] = sources.trainingSnapshots(training, null, 5);
  assert.equal(row.id, "training:job-1");
  assert.equal(row.state, "active");
  assert.equal(row.progress, 0.2);
  assert.deepEqual(row.meter, { unit: "steps", done: 40, total: 200 });

  const [loading] = sources.trainingSnapshots(
    { ...training, phase: "downloading_model", isTrainingRunning: false },
    null,
    5,
  );
  assert.equal(loading.state, "active");
  assert.equal(loading.progress, null);
  assert.equal(loading.detail, "Training...");

  assert.equal(
    sources.trainingSnapshots({ ...training, phase: "completed" }, null, 5)[0]
      .state,
    "done",
  );
  const [failed] = sources.trainingSnapshots(
    { ...training, phase: "error", error: "CUDA OOM" },
    null,
    5,
  );
  assert.equal(failed.state, "failed");
  assert.equal(failed.detail, "CUDA OOM");
  assert.equal(
    sources.trainingSnapshots({ ...training, phase: "stopped" }, null, 5)[0]
      .state,
    "cancelled",
  );
  assert.deepEqual(
    sources.trainingSnapshots(
      { ...training, phase: "idle", isTrainingRunning: false },
      null,
      5,
    ),
    [],
  );
  assert.deepEqual(
    sources.trainingSnapshots({ ...training, jobId: null }, null, 5),
    [],
  );
  assert.equal(
    sources.trainingSnapshots(
      { ...training, startModelName: null },
      "picked/model",
      5,
    )[0].title,
    "picked/model",
  );
});

test("an export maps by phase, one row per run", () => {
  const state = {
    phase: "exporting",
    isExporting: true,
    startedAt: 123,
    summary: { baseModelName: "Qwen3-8B", methodLabel: "GGUF" },
    error: null,
    stage: "Quantizing Q4_K_M",
  };
  const [row] = sources.exportSnapshots(state, 50);
  assert.equal(row.id, "export:123");
  assert.equal(row.title, "Qwen3-8B · GGUF");
  assert.equal(row.progress, 0.5);
  assert.equal(row.detail, "Quantizing Q4_K_M");
  assert.equal(
    sources.exportSnapshots({ ...state, phase: "success" }, 100)[0].state,
    "done",
  );
  assert.equal(
    sources.exportSnapshots({ ...state, phase: "canceled" }, 40)[0].state,
    "cancelled",
  );
  assert.deepEqual(sources.exportSnapshots({ ...state, phase: "idle" }, 0), []);
});

test("a recipe run counts once it shows life, never from stale saved history", () => {
  const record = {
    id: "exec-1",
    recipeId: "recipe-9",
    kind: "full" as const,
    run_name: "nightly",
    status: "running",
    createdAt: 10,
    stage: "generating",
    error: null,
    lastEventId: 3,
    progress: { done: 25, total: 100, percent: 25 },
  };
  const signatures = new Map<string, string>();
  const none = new Set<string>();
  // Saved before this session and not moving: a tab that closed mid-run left it marked running.
  assert.deepEqual(
    sources.recipeSnapshots([record], signatures, none, 1_000),
    [],
  );
  // Its tracker resumed and it moved.
  const [moving] = sources.recipeSnapshots(
    [{ ...record, lastEventId: 4 }],
    signatures,
    none,
    1_000,
  );
  assert.equal(moving.id, "recipe:exec-1");
  assert.equal(moving.progress, 0.25);
  assert.deepEqual(moving.meter, { unit: "rows", done: 25, total: 100 });
  assert.equal(moving.ref, "recipe-9");
  // Started this session.
  assert.equal(
    sources.recipeSnapshots(
      [{ ...record, id: "exec-2", createdAt: 2_000 }],
      new Map(),
      none,
      1_000,
    ).length,
    1,
  );
  // Finished records are always reported; the store only settles rows it saw running.
  assert.equal(
    sources.recipeSnapshots(
      [{ ...record, status: "completed" }],
      new Map(),
      none,
      1_000,
    )[0].state,
    "done",
  );
});

test("model loads: dictation is not tracked, quick successes are not kept, paths read short", () => {
  assert.equal(sources.tracksModelLoad("stt"), false);
  assert.equal(sources.tracksModelLoad("chat"), true);
  assert.equal(sources.modelLoadRoute("image"), "/images");
  assert.equal(
    sources.modelLoadSettlement("loaded", 0, sources.SHORT_MODEL_LOAD_MS - 1),
    "drop",
  );
  assert.equal(
    sources.modelLoadSettlement("loaded", 0, sources.SHORT_MODEL_LOAD_MS),
    "done",
  );
  assert.equal(sources.modelLoadSettlement("failed", 0, 1), "failed");
  assert.equal(sources.modelLoadSettlement("cancelled", 0, 1), "cancelled");
  assert.equal(sources.modelLoadSettlement(undefined, 0, 99_999), "drop");
  assert.equal(
    sources.modelLoadTitle("D:\\models\\qwen\\Qwen3-8B-Q4_K_M.gguf"),
    "Qwen3-8B-Q4_K_M.gguf",
  );
  assert.equal(sources.modelLoadTitle("/home/u/models/llama"), "llama");
  assert.equal(
    sources.modelLoadTitle("unsloth/Qwen3-8B-GGUF"),
    "unsloth/Qwen3-8B-GGUF",
  );
});

test("a native notice is for long jobs only: training, a finished export, a download over 1 GiB", () => {
  const notice = (
    kind: ActivityEntry["kind"],
    state: ActivityEntry["state"],
    bytes: number | null = null,
  ) => sources.finishedNotice({ kind, state, bytes });
  assert.equal(notice("training", "done"), "trainingDone");
  assert.equal(notice("training", "failed"), "trainingFailed");
  assert.equal(notice("training", "cancelled"), null);
  assert.equal(notice("export", "done"), "exportDone");
  assert.equal(notice("export", "failed"), null);
  assert.equal(notice("download", "done", sources.LONG_DOWNLOAD_BYTES), null);
  assert.equal(
    notice("download", "done", sources.LONG_DOWNLOAD_BYTES + 1),
    "downloadDone",
  );
  assert.equal(notice("model-load", "done"), null);
  assert.equal(notice("error", "failed"), null);
});

// ── Model load outcome and cancel ──────────────────────────────────

test("a load's settling notice says how it ended", async () => {
  const outcomes: (string | undefined)[] = [];
  const stop = lifecycle.subscribeModelLifecycle((detail) => {
    if (!detail.loading) outcomes.push(detail.outcome);
  });
  await lifecycle.withModelLoadNotice("chat", "m", async () => "ok");
  await assert.rejects(
    lifecycle.withModelLoadNotice("chat", "m", async () => {
      throw new Error("OOM");
    }),
  );
  await assert.rejects(
    lifecycle.withModelLoadNotice("chat", "m", async () => {
      throw new DOMException("Aborted", "AbortError");
    }),
  );
  stop();
  assert.deepEqual(outcomes, ["loaded", "failed", "cancelled"]);
});

test("Cancel reaches every page that registered for the runtime, and only while registered", () => {
  let changes = 0;
  const stopListening = lifecycle.subscribeModelLoadCancels(() => {
    changes += 1;
  });
  const calls: string[] = [];
  assert.equal(lifecycle.canCancelModelLoad("video"), false);
  const offA = lifecycle.registerModelLoadCancel("video", () =>
    calls.push("a"),
  );
  const offB = lifecycle.registerModelLoadCancel("video", () =>
    calls.push("b"),
  );
  assert.equal(lifecycle.canCancelModelLoad("video"), true);
  assert.equal(lifecycle.cancelModelLoad("video"), true);
  assert.deepEqual(calls, ["a", "b"]);
  offA();
  offB();
  offB();
  assert.equal(lifecycle.canCancelModelLoad("video"), false);
  assert.equal(lifecycle.cancelModelLoad("video"), false);
  assert.equal(
    changes,
    4,
    "two registrations and two removals; a repeat removal is silent",
  );
  stopListening();
});

// ── Wiring ─────────────────────────────────────────────────────────

test("the toast override records the error where it shows it, after the banner's hold-back", () => {
  const sonner = readSrc("components/ui/sonner.tsx");
  const override = sonner.slice(sonner.indexOf("toast.error = "));
  const show = override.slice(
    override.indexOf("const show = () => {"),
    override.indexOf("if (\n"),
  );
  // Inside `show`, so a transport toast held back below is never recorded and a replayed one is.
  assert.match(
    show,
    /recordToastError\(message, data\?\.description, data\?\.action\)/,
  );
  assert.ok(
    show.indexOf("recordToastError(") < show.indexOf("return showErrorToast("),
    "recorded as the toast shows",
  );
  // The banner's suppression still hands the same `show` over.
  assert.match(
    override,
    /data\?\.id === undefined &&\s*suppressTransportErrorToast\(\[message, data\?\.description\], show\)/,
  );
  // The View logs action carries its target as data, which is what the bell keeps.
  const action = readSrc("features/settings/lib/view-logs-action.ts");
  assert.match(
    action,
    /logTarget: \{ family, sourcePath: sourcePath \?\? null \}/,
  );
});

test("the bell sits in the sidebar header, and its feeds are mounted once from the sidebar", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  assert.match(
    sidebar,
    /import \{ ActivityBell, useActivityFeeds \} from "@\/features\/activity";/,
  );
  assert.equal(sidebar.split("<ActivityBell").length - 1, 1);
  assert.equal(sidebar.split("useActivityFeeds();").length - 1, 1);
  const header = sidebar.slice(
    sidebar.indexOf("<SidebarHeader"),
    sidebar.indexOf("</SidebarHeader>"),
  );
  assert.ok(header.includes("<ActivityBell />"), "the bell is in the header");
  assert.ok(
    header.indexOf("<ActivityBell />") <
      header.indexOf("useChatSearchStore.getState().open()"),
    "next to search, in the header's button group",
  );
});

test("the bell names its counts, and its rows are labelled", () => {
  const bell = readSrc("features/activity/activity-bell.tsx");
  assert.match(
    bell,
    /t\("activity\.bellLabel", \{\s*active: activeCount,\s*errors: unseen,?\s*\}\)/,
  );
  assert.match(bell, /aria-label=\{label\}/);
  assert.match(bell, /aria-labelledby=\{`\$\{titleId\} \$\{statusId\}`\}/);
  // Every key the bell asks for exists in the English catalogue.
  const keys = [...bell.matchAll(/"(activity\.[A-Za-z.]+)"/g)].map(
    (match) => match[1],
  );
  assert.ok(keys.length > 20);
  for (const key of keys) {
    const value = key
      .split(".")
      .reduce<unknown>(
        (node, part) => (node as Record<string, unknown> | undefined)?.[part],
        en,
      );
    assert.equal(typeof value, "string", key);
  }
  for (const notice of [
    "trainingDone",
    "trainingFailed",
    "exportDone",
    "downloadDone",
  ]) {
    assert.equal(
      typeof en.activity.notify[notice as keyof typeof en.activity.notify],
      "string",
    );
  }
});

test("the feeds start no timers and make no requests of their own", () => {
  const feeds = readSrc("features/activity/use-activity-feeds.ts");
  const sourcesText = readSrc("features/activity/activity-sources.ts");
  for (const source of [feeds, sourcesText]) {
    assert.doesNotMatch(source, /setInterval|setTimeout|authFetch|fetch\(/);
  }
  // Permission is only ever asked from the toggle, never on its own.
  const notifications = readSrc("features/activity/activity-notifications.ts");
  assert.equal(notifications.split("requestPermission()").length - 1, 1);
  assert.match(notifications, /requestPermission: false/);
  const bell = readSrc("features/activity/activity-bell.tsx");
  assert.equal(bell.split("requestJobNotifications()").length - 1, 1);
});

test("Reset all local preferences clears the history and the notify choice", () => {
  assert.deepEqual(
    [...ACTIVITY_PREFERENCE_KEYS],
    [ACTIVITY_HISTORY_KEY, ACTIVITY_NOTIFY_KEY],
  );
  const general = readSrc("features/settings/tabs/general-tab.tsx");
  const list = general.slice(
    general.indexOf("const PREFS_KEYS"),
    general.indexOf("];", general.indexOf("const PREFS_KEYS")),
  );
  assert.match(list, /\.\.\.ACTIVITY_PREFERENCE_KEYS,/);
  assert.match(
    general,
    /import \{ ACTIVITY_PREFERENCE_KEYS \} from "@\/lib\/activity-store";/,
  );
});
