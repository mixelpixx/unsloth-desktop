// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One banner instead of a toast per failed request, and a notice when the backend that comes back is a new process. */

import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import {
  backoffDelayMs,
  createLivenessProbe,
  HEARTBEAT_MS,
  isTransportFailureText,
  type LivenessProbeResult,
  reportTransportFailure,
  retryConnectionNow,
  RETRY_MAX_MS,
  startConnectionMonitor,
  stopConnectionMonitor,
  suppressTransportErrorToast,
  useConnectionStore,
} from "../src/lib/connection-monitor.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const NOT_RUNNING = "Unsloth isn't running -- please relaunch it.";
const A = "a".repeat(32);
const B = "b".repeat(32);

/** Settle the probe's promise chain: `probe().catch().then()` is a few microtasks deep. */
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

/** The monitor with a hand-cranked clock, timers, visibility and probe. */
function harness() {
  let now = 1_700_000_000_000;
  let hidden = false;
  let nextTimerId = 1;
  const timers = new Map<number, { at: number; run: () => void }>();
  const pending: Array<(result: LivenessProbeResult) => void> = [];
  let wake: (() => void) | null = null;

  const stop = startConnectionMonitor({
    probe: () =>
      new Promise<LivenessProbeResult>((resolve) => pending.push(resolve)),
    now: () => now,
    setTimer: (run, ms) => {
      const id = nextTimerId++;
      timers.set(id, { at: now + ms, run });
      return id;
    },
    clearTimer: (id) => void timers.delete(id as number),
    isHidden: () => hidden,
    subscribeWake: (onWake) => {
      wake = onWake;
      return () => {
        wake = null;
      };
    },
  });

  return {
    stop,
    get now() {
      return now;
    },
    /** Probes sent and not yet answered. */
    get probesInFlight() {
      return pending.length;
    },
    /** Delay of the one timer armed, relative to now; null with none. */
    get armedIn(): number | null {
      const [timer] = [...timers.values()];
      return timer ? timer.at - now : null;
    },
    setHidden(value: boolean) {
      hidden = value;
    },
    wake() {
      wake?.();
    },
    async answer(result: LivenessProbeResult) {
      const resolve = pending.shift();
      assert.ok(resolve, "no probe in flight to answer");
      resolve(result);
      await flush();
    },
    /** Move the clock and run every timer that comes due, in order. */
    advance(ms: number) {
      const until = now + ms;
      for (;;) {
        const due = [...timers.entries()]
          .filter(([, timer]) => timer.at <= until)
          .sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        timers.delete(due[0]);
        now = due[1].at;
        due[1].run();
      }
      now = until;
    },
  };
}

const reachable = (instanceId: string | null): LivenessProbeResult => ({
  kind: "reachable",
  instanceId,
});
const unreachable: LivenessProbeResult = { kind: "unreachable" };
const inconclusive: LivenessProbeResult = { kind: "inconclusive" };

/** Started, and the first probe has told it which process is behind the port. */
async function online(instanceId = A) {
  const h = harness();
  assert.equal(h.probesInFlight, 1, "the monitor learns the instance id on start");
  await h.answer(reachable(instanceId));
  assert.equal(useConnectionStore.getState().lastInstanceId, instanceId);
  return h;
}

afterEach(() => {
  stopConnectionMonitor();
});

test("a transport failure flips to reconnecting at once and probes", async () => {
  const h = await online();
  reportTransportFailure(NOT_RUNNING);
  const state = useConnectionStore.getState();
  assert.equal(state.status, "reconnecting");
  assert.equal(state.disconnectedSince, h.now);
  // Not confirmed yet, so the banner (which waits for failedProbes > 0) is still down.
  assert.equal(state.failedProbes, 0);
  assert.equal(h.probesInFlight, 1);
  // A second failure in the same outage starts nothing new.
  reportTransportFailure(NOT_RUNNING);
  assert.equal(h.probesInFlight, 1);
});

test("the same process answering again goes online with no restart notice", async () => {
  const h = await online();
  reportTransportFailure(NOT_RUNNING);
  await h.answer(unreachable);
  assert.equal(useConnectionStore.getState().failedProbes, 1);
  assert.equal(useConnectionStore.getState().nextRetryAt, h.now + 1_000);
  h.advance(1_000);
  await h.answer(reachable(A));
  const state = useConnectionStore.getState();
  assert.equal(state.status, "online");
  assert.equal(state.disconnectedSince, null);
  assert.equal(state.restartedAt, null);
  assert.equal(state.reconnectedAt, h.now, "a confirmed outage that ended says so once");
  assert.equal(h.armedIn, HEARTBEAT_MS, "back on the heartbeat");
});

test("a different process answering raises the restart notice", async () => {
  const h = await online(A);
  reportTransportFailure(NOT_RUNNING);
  await h.answer(unreachable);
  h.advance(1_000);
  await h.answer(reachable(B));
  const state = useConnectionStore.getState();
  assert.equal(state.status, "online");
  assert.equal(state.restartedAt, h.now);
  assert.equal(state.lastInstanceId, B);
  assert.equal(state.reconnectedAt, null, "the restart notice stands in for the reconnected one");
});

test("an idle app notices a restart on the heartbeat, with no outage seen", async () => {
  const h = await online(A);
  h.advance(HEARTBEAT_MS);
  assert.equal(h.probesInFlight, 1);
  await h.answer(reachable(B));
  assert.equal(useConnectionStore.getState().status, "online");
  assert.equal(useConnectionStore.getState().restartedAt, h.now);
});

test("a reply that names no process is not read as a restart", async () => {
  const h = await online(A);
  h.advance(HEARTBEAT_MS);
  await h.answer(reachable(null));
  assert.equal(useConnectionStore.getState().restartedAt, null);
  assert.equal(useConnectionStore.getState().lastInstanceId, A);
});

test("the retry schedule doubles from 1s and caps at 30s", async () => {
  assert.deepEqual(
    [1, 2, 3, 4, 5, 6, 7, 50, 5_000].map(backoffDelayMs),
    [1_000, 2_000, 4_000, 8_000, 16_000, RETRY_MAX_MS, RETRY_MAX_MS, RETRY_MAX_MS, RETRY_MAX_MS],
  );
  const h = await online();
  reportTransportFailure(NOT_RUNNING);
  const delays: number[] = [];
  for (let i = 0; i < 7; i++) {
    // A timeout while reconnecting counts as a failure too: only an answer ends the outage.
    await h.answer(i % 2 === 0 ? unreachable : inconclusive);
    const { nextRetryAt } = useConnectionStore.getState();
    assert.ok(nextRetryAt !== null);
    delays.push(nextRetryAt - h.now);
    h.advance(nextRetryAt - h.now);
  }
  assert.deepEqual(delays, [1_000, 2_000, 4_000, 8_000, 16_000, 30_000, 30_000]);
});

test("transport toasts are held back only while reconnecting, and other errors never", async () => {
  const h = await online();
  let shown = 0;
  const show = () => void shown++;
  assert.equal(suppressTransportErrorToast([NOT_RUNNING], show), false, "online: nothing to stand in for it");

  reportTransportFailure(NOT_RUNNING);
  await h.answer(unreachable);
  assert.equal(suppressTransportErrorToast([NOT_RUNNING], show), true);
  assert.equal(
    suppressTransportErrorToast(["Couldn't load chats", `Failed to load: ${NOT_RUNNING}`], show),
    true,
    "a description carrying the news counts, and so does a prefixed message",
  );
  assert.equal(suppressTransportErrorToast(["Failed to fetch"], show), true);
  assert.equal(suppressTransportErrorToast(["Model is too large for this GPU"], show), false);
  assert.equal(suppressTransportErrorToast(["Model load failed: out of memory"], show), false);

  h.advance(1_000);
  await h.answer(reachable(A));
  assert.equal(shown, 0, "a confirmed outage was the banner's to report; nothing is replayed");
  assert.equal(suppressTransportErrorToast([NOT_RUNNING], show), false);
});

test("a failure the first probe disproves gets its toast after all, and no banner", async () => {
  const h = await online();
  let shown = 0;
  reportTransportFailure(NOT_RUNNING);
  assert.equal(suppressTransportErrorToast([NOT_RUNNING], () => void shown++), true);
  assert.equal(suppressTransportErrorToast([NOT_RUNNING], () => void shown++), true);
  await h.answer(reachable(A));
  const state = useConnectionStore.getState();
  assert.equal(state.status, "online");
  assert.equal(state.reconnectedAt, null, "nothing was down, so nothing reconnected");
  assert.equal(shown, 1, "the first held-back toast is shown, once");
});

test("a heartbeat that finds nothing listening confirms the outage itself", async () => {
  const h = await online();
  h.advance(HEARTBEAT_MS);
  await h.answer(unreachable);
  const state = useConnectionStore.getState();
  assert.equal(state.status, "reconnecting");
  assert.equal(state.failedProbes, 1);
  assert.equal(state.disconnectedSince, h.now);
  assert.equal(h.armedIn, 1_000);
});

test("a heartbeat that times out is a busy backend, not a lost one", async () => {
  const h = await online();
  h.advance(HEARTBEAT_MS);
  await h.answer(inconclusive);
  assert.equal(useConnectionStore.getState().status, "online");
  assert.equal(h.armedIn, HEARTBEAT_MS);
});

test("a hidden tab sends no heartbeat, and checks in when it is back", async () => {
  const h = await online();
  h.setHidden(true);
  h.advance(HEARTBEAT_MS * 4);
  assert.equal(h.probesInFlight, 0);
  assert.equal(h.armedIn, null, "no timer left ticking while hidden");
  h.setHidden(false);
  h.wake();
  assert.equal(h.probesInFlight, 1);
  await h.answer(reachable(A));
  // Focus again right away: a beat has not gone by, so window switching costs nothing.
  h.wake();
  assert.equal(h.probesInFlight, 0);
});

test("retries pause while hidden and run the moment the tab is back", async () => {
  const h = await online();
  reportTransportFailure(NOT_RUNNING);
  await h.answer(unreachable);
  h.setHidden(true);
  h.advance(60_000);
  assert.equal(h.probesInFlight, 0);
  assert.equal(useConnectionStore.getState().nextRetryAt, null);
  h.setHidden(false);
  h.wake();
  assert.equal(h.probesInFlight, 1);
});

test("Retry now probes at once, and not on top of a probe already out", async () => {
  const h = await online();
  reportTransportFailure(NOT_RUNNING);
  await h.answer(unreachable);
  assert.equal(h.armedIn, 1_000);
  retryConnectionNow();
  assert.equal(h.probesInFlight, 1);
  assert.equal(h.armedIn, null, "the scheduled retry is replaced, not doubled");
  retryConnectionNow();
  assert.equal(h.probesInFlight, 1);
});

test("with no monitor running a report changes nothing and holds back no toast", () => {
  reportTransportFailure(NOT_RUNNING);
  assert.equal(useConnectionStore.getState().status, "online");
  assert.equal(suppressTransportErrorToast([NOT_RUNNING], () => {}), false);
});

test("stopping forgets the process, so a remounted shell does not report a restart it showed", async () => {
  const h = await online(A);
  h.stop();
  assert.equal(useConnectionStore.getState().lastInstanceId, null);
  const next = harness();
  await next.answer(reachable(B));
  assert.equal(useConnectionStore.getState().restartedAt, null);
});

test("a probe answered after a stop answers for nobody", async () => {
  const h = harness();
  stopConnectionMonitor();
  await h.answer(reachable(A));
  assert.equal(useConnectionStore.getState().lastInstanceId, null);
});

test("transport text is recognised without catching real errors that mention loading", () => {
  assert.equal(isTransportFailureText("Load failed"), true);
  assert.equal(isTransportFailureText("Couldn't save: Load failed"), true);
  assert.equal(isTransportFailureText("Model load failed: out of memory"), false);
  assert.equal(isTransportFailureText("NetworkError when attempting to fetch resource."), true);
  assert.equal(isTransportFailureText(undefined), false);
});

test("the liveness probe tells an answer from silence and from a stall", async () => {
  const originalFetch = globalThis.fetch;
  const probe = createLivenessProbe(() => "http://127.0.0.1:1/api/liveness", 20);
  try {
    globalThis.fetch = async () => Response.json({ status: "alive", instance_id: A });
    assert.deepEqual(await probe(), { kind: "reachable", instanceId: A });

    // Any status the backend itself sends means it is up, whatever it thought of the request.
    globalThis.fetch = async () => new Response("not found", { status: 404 });
    assert.deepEqual(await probe(), { kind: "reachable", instanceId: null });

    // A proxy or tunnel with nothing behind it.
    globalThis.fetch = async () => new Response("bad gateway", { status: 502 });
    assert.deepEqual(await probe(), { kind: "unreachable" });

    globalThis.fetch = async () => {
      throw new TypeError("Failed to fetch");
    };
    assert.deepEqual(await probe(), { kind: "unreachable" });

    globalThis.fetch = (_input, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () =>
          reject(new DOMException("aborted", "AbortError")),
        );
      });
    assert.deepEqual(await probe(), { kind: "inconclusive" });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("authFetch reports a request that never reached the backend, and not one it rejected", async () => {
  const originalFetch = globalThis.fetch;
  const reported: Array<string | undefined> = [];
  const authApi = loadWithStubs<{
    authFetch: (input: string) => Promise<Response>;
    BACKEND_NOT_RUNNING_MESSAGE: string;
  }>(new URL("../src/features/auth/api.ts", import.meta.url), {
    "@/lib/api-base": {
      apiUrl: (path: string) => path,
      getApiPort: () => null,
      isTauri: false,
    },
    "@/lib/account-transition": { accountTransitionPending: () => false },
    "@/lib/connection-monitor": {
      reportTransportFailure: (message?: string) => void reported.push(message),
    },
    "./session": {
      clearAuthTokens: () => {},
      getAuthToken: () => "access-token",
      getRefreshToken: () => null,
      mustChangePassword: () => false,
      setMustChangePassword: () => {},
      storeAuthTokens: () => {},
    },
  });
  try {
    globalThis.fetch = async () => {
      throw new TypeError("Failed to fetch");
    };
    await assert.rejects(authApi.authFetch("/api/inference/status"), {
      message: authApi.BACKEND_NOT_RUNNING_MESSAGE,
    });
    assert.deepEqual(reported, [authApi.BACKEND_NOT_RUNNING_MESSAGE]);

    globalThis.fetch = async () => new Response("boom", { status: 500 });
    const response = await authApi.authFetch("/api/inference/status");
    assert.equal(response.status, 500);
    assert.equal(reported.length, 1, "an HTTP error is an answer, not a lost backend");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("the toast override asks the monitor, and lets an update to an existing toast through", () => {
  const sonner = readSrc("components/ui/sonner.tsx");
  const override = sonner.slice(sonner.indexOf("toast.error = "));
  assert.match(
    override,
    /data\?\.id === undefined &&\s*suppressTransportErrorToast\(\[message, data\?\.description\], show\)/,
  );
});

test("the banner is mounted once at the root, on every route, and announces politely", () => {
  const root = readSrc("app/routes/__root.tsx");
  assert.equal(root.split("<ConnectionBanner ").length - 1, 1);
  assert.match(root, /\n\s*<ConnectionBanner authFlow=\{isAuthFlowRoute\} \/>/);
  const banner = readSrc("components/connection-banner.tsx");
  assert.match(banner, /role="status"\s*aria-live="polite"/);
  // No query string: it would split the access log's dedup bucket for the route.
  assert.ok(banner.includes('apiUrl("/api/liveness")'));
});

test("a restart with no model picked leaves the composer's tool toggles alone", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  const resync = runtime.slice(
    runtime.indexOf("export async function resyncInferenceStatusAfterServerModelChange("),
    runtime.indexOf("function pickOf("),
  );
  assert.match(resync, /if \(checkpoint && !isExternalModelId\(checkpoint\)\) \{\s*useChatRuntimeStore\.getState\(\)\.clearCheckpoint\(\);/);
});
