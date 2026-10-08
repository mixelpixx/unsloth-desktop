// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the backend can be reached, and whether the one answering is the one we had. Without this,
// a backend that goes away (a crash, a restart, the console closed, the laptop asleep) fails every
// in-flight and later request on its own, so the user gets a toast per request and no statement of
// what happened; and after a restart the UI keeps believing whatever the old process told it, such
// as which model is loaded.
//
// Fed from one place: `asTransportFailure` in features/auth/api.ts, which every authFetch rejection
// passes through, reports here. A failure is a claim, not a verdict, so it is checked against
// /api/liveness before the banner goes up, and the toast override in components/ui/sonner.tsx asks
// here before showing a transport toast. In lib/, not a feature, since auth reports, the toaster
// asks and the shell renders. Imports nothing but zustand so tests can drive it directly.

import { create } from "zustand";

export type ConnectionStatus = "online" | "reconnecting";

export interface ConnectionState {
  status: ConnectionStatus;
  /** When this outage started (ms since epoch); null while online. */
  disconnectedSince: number | null;
  /** When the next automatic probe is due; null while online, while one is in flight, and while
   *  the tab is hidden, since probing is paused then. */
  nextRetryAt: number | null;
  probing: boolean;
  /** Probes that failed in this outage. The banner waits for the first: a report is only a claim,
   *  and one probe against a backend that answers at once is cheaper than a banner that flashes. */
  failedProbes: number;
  /** `instance_id` of the last backend that answered; null until one has. */
  lastInstanceId: string | null;
  /** When a reconnect found a different process. An event stamp the shell turns into a notice. */
  restartedAt: number | null;
  /** When a confirmed outage ended with the same process answering. Same kind of stamp. */
  reconnectedAt: number | null;
}

/** What one liveness probe found. Only `unreachable` is evidence the backend is gone. */
export type LivenessProbeResult =
  | { kind: "reachable"; instanceId: string | null }
  | { kind: "unreachable" }
  | { kind: "inconclusive" };

export interface ConnectionMonitorDeps {
  probe: () => Promise<LivenessProbeResult>;
  now?: () => number;
  setTimer?: (run: () => void, ms: number) => unknown;
  clearTimer?: (handle: unknown) => void;
  isHidden?: () => boolean;
  /** Calls `wake` when the page comes back to the foreground. Returns an unsubscriber. */
  subscribeWake?: (wake: () => void) => () => void;
}

// One short GET per visible tab, so it stays well inside the six connections per origin that every
// tab shares on HTTP/1.1. /api/liveness is declared in the log budget and dedups to a line a minute.
export const HEARTBEAT_MS = 25_000;
export const RETRY_BASE_MS = 1_000;
export const RETRY_MAX_MS = 30_000;
// Under the desktop watchdog's 10s probe budget: a liveness reply that slow is a stalled loop, which
// the monitor reports as inconclusive rather than as gone.
export const PROBE_TIMEOUT_MS = 5_000;

// What a reverse proxy or tunnel answers when nothing is listening behind it. Any other status came
// from the backend itself, so it is up, whatever it thought of the request.
const GATEWAY_STATUSES = new Set([502, 503, 504]);

// TypeError messages fetch rejects with when it never got an answer (Chromium, Firefox), matched
// anywhere in a toast because callers prefix them. Safari's "Load failed" is matched only whole or
// as a suffix: "Model load failed: ..." is a real error, not a transport one.
const FETCH_FAILURE_FRAGMENTS = [
  "Failed to fetch",
  "NetworkError when attempting to fetch resource",
];
const FETCH_FAILURE_EXACT = ["Load failed"];
// The copy asTransportFailure attached, remembered as it is reported so the two cannot drift. Capped:
// it holds a handful of fixed strings, and a cap keeps a bug from growing it without bound.
const MAX_TRANSPORT_MESSAGES = 8;
const transportMessages = new Set<string>();

const INITIAL_STATE: ConnectionState = {
  status: "online",
  disconnectedSince: null,
  nextRetryAt: null,
  probing: false,
  failedProbes: 0,
  lastInstanceId: null,
  restartedAt: null,
  reconnectedAt: null,
};

export const useConnectionStore = create<ConnectionState>(() => ({
  ...INITIAL_STATE,
}));

type Runtime = {
  deps: Required<ConnectionMonitorDeps>;
  timer: unknown;
  /** When the last probe settled, so a focus storm does not become a probe storm. */
  lastProbeAt: number;
  /** The first transport toast swallowed in this outage, shown after all if no outage is confirmed. */
  replaySuppressed: (() => void) | null;
  stopWake: () => void;
};

let runtime: Runtime | null = null;

/** 1s, 2s, 4s, 8s, 16s, then every 30s. `failedProbes` counts the probe that just failed. */
export function backoffDelayMs(failedProbes: number): number {
  const exponent = Math.max(0, failedProbes - 1);
  return Math.min(RETRY_BASE_MS * 2 ** exponent, RETRY_MAX_MS);
}

function defaultSubscribeWake(wake: () => void): () => void {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return () => {};
  }
  const onVisibility = () => {
    if (!document.hidden) wake();
  };
  document.addEventListener("visibilitychange", onVisibility);
  window.addEventListener("focus", wake);
  window.addEventListener("online", wake);
  return () => {
    document.removeEventListener("visibilitychange", onVisibility);
    window.removeEventListener("focus", wake);
    window.removeEventListener("online", wake);
  };
}

function resolveDeps(deps: ConnectionMonitorDeps): Required<ConnectionMonitorDeps> {
  return {
    probe: deps.probe,
    now: deps.now ?? (() => Date.now()),
    setTimer: deps.setTimer ?? ((run, ms) => setTimeout(run, ms)),
    clearTimer:
      deps.clearTimer ??
      ((handle) => clearTimeout(handle as ReturnType<typeof setTimeout>)),
    isHidden:
      deps.isHidden ??
      (() => typeof document !== "undefined" && document.hidden),
    subscribeWake: deps.subscribeWake ?? defaultSubscribeWake,
  };
}

function clearRuntimeTimer(rt: Runtime): void {
  if (rt.timer === null) return;
  rt.deps.clearTimer(rt.timer);
  rt.timer = null;
}

function runProbe(rt: Runtime): void {
  if (useConnectionStore.getState().probing) return;
  clearRuntimeTimer(rt);
  useConnectionStore.setState({ probing: true, nextRetryAt: null });
  void rt.deps
    .probe()
    .catch((): LivenessProbeResult => ({ kind: "unreachable" }))
    .then((result) => {
      // Stopped, or replaced by a fresh start, while this was in flight: it answers for nobody.
      if (runtime !== rt) return;
      settleProbe(rt, result);
    });
}

function scheduleHeartbeat(rt: Runtime): void {
  clearRuntimeTimer(rt);
  rt.timer = rt.deps.setTimer(() => {
    rt.timer = null;
    if (runtime !== rt) return;
    // Hidden: no request, and no timer either. The wake listener probes when the tab is back,
    // since by then this beat is stale.
    if (rt.deps.isHidden()) return;
    runProbe(rt);
  }, HEARTBEAT_MS);
}

function scheduleRetry(rt: Runtime): void {
  clearRuntimeTimer(rt);
  const delay = backoffDelayMs(useConnectionStore.getState().failedProbes);
  useConnectionStore.setState({ nextRetryAt: rt.deps.now() + delay });
  rt.timer = rt.deps.setTimer(() => {
    rt.timer = null;
    if (runtime !== rt) return;
    if (rt.deps.isHidden()) {
      // Paused, not dropped: the wake listener retries the moment the tab is visible again.
      useConnectionStore.setState({ nextRetryAt: null });
      return;
    }
    runProbe(rt);
  }, delay);
}

function settleProbe(rt: Runtime, result: LivenessProbeResult): void {
  const now = rt.deps.now();
  rt.lastProbeAt = now;
  const state = useConnectionStore.getState();

  if (result.kind === "reachable") {
    const id = result.instanceId;
    // Needs an id on both sides: a reply that could not be parsed says the port answers, not who.
    const restarted =
      id !== null && state.lastInstanceId !== null && id !== state.lastInstanceId;
    const next: Partial<ConnectionState> = {
      probing: false,
      lastInstanceId: id ?? state.lastInstanceId,
    };
    if (restarted) next.restartedAt = now;
    // Unconfirmed: the report was the only evidence, and the first probe got an answer. Whatever
    // failed was not an outage, so its toast was not redundant after all.
    const unconfirmed = state.status === "reconnecting" && state.failedProbes === 0;
    if (state.status === "reconnecting") {
      next.status = "online";
      next.disconnectedSince = null;
      next.nextRetryAt = null;
      next.failedProbes = 0;
      if (!restarted && !unconfirmed) next.reconnectedAt = now;
    }
    const replay = unconfirmed && !restarted ? rt.replaySuppressed : null;
    rt.replaySuppressed = null;
    useConnectionStore.setState(next);
    replay?.();
    scheduleHeartbeat(rt);
    return;
  }

  if (state.status === "online") {
    // A heartbeat that timed out met a stalled loop, not an empty port: a host generating on every
    // slot answers late and is still the backend the user is waiting on.
    if (result.kind === "inconclusive") {
      useConnectionStore.setState({ probing: false });
      scheduleHeartbeat(rt);
      return;
    }
    // The heartbeat itself found nothing listening, so this outage is confirmed from the start.
    useConnectionStore.setState({
      probing: false,
      status: "reconnecting",
      disconnectedSince: now,
      failedProbes: 1,
      reconnectedAt: null,
    });
  } else {
    // While reconnecting a timeout counts too: the banner is already up, and only an answer ends it.
    useConnectionStore.setState({
      probing: false,
      failedProbes: state.failedProbes + 1,
    });
  }
  scheduleRetry(rt);
}

function wake(rt: Runtime): void {
  if (runtime !== rt || rt.deps.isHidden()) return;
  const state = useConnectionStore.getState();
  if (state.probing) return;
  if (state.status === "reconnecting") {
    runProbe(rt);
    return;
  }
  // Online: a tab back from the background (or a laptop back from sleep) may have missed a
  // restart. Only once a beat has gone by, so switching windows back and forth costs nothing.
  if (rt.deps.now() - rt.lastProbeAt >= HEARTBEAT_MS) runProbe(rt);
}

/**
 * Start watching. One monitor at a time: starting again replaces the last, and stopping resets the
 * state, so a shell that remounts (the desktop app does, after its own startup screen) does not
 * take the next backend for a restart it already showed the user.
 */
export function startConnectionMonitor(deps: ConnectionMonitorDeps): () => void {
  stopConnectionMonitor();
  const rt: Runtime = {
    deps: resolveDeps(deps),
    timer: null,
    lastProbeAt: 0,
    replaySuppressed: null,
    stopWake: () => {},
  };
  runtime = rt;
  rt.stopWake = rt.deps.subscribeWake(() => wake(rt));
  // Learn the instance id now, so a restart before the first heartbeat still reads as one.
  runProbe(rt);
  return () => {
    if (runtime === rt) stopConnectionMonitor();
  };
}

export function stopConnectionMonitor(): void {
  const rt = runtime;
  if (!rt) return;
  runtime = null;
  clearRuntimeTimer(rt);
  rt.stopWake();
  useConnectionStore.setState({ ...INITIAL_STATE });
}

/**
 * A request never reached the backend. `message` is the copy the caller will show for it, kept so
 * the toast override can recognise it. A no-op with no monitor running: nothing would ever probe,
 * so the reconnecting state, and the toasts it holds back, would never end.
 */
export function reportTransportFailure(message?: string): void {
  if (message && !transportMessages.has(message)) {
    if (transportMessages.size >= MAX_TRANSPORT_MESSAGES) {
      const oldest = transportMessages.values().next().value;
      if (oldest !== undefined) transportMessages.delete(oldest);
    }
    transportMessages.add(message);
  }
  const rt = runtime;
  if (!rt) return;
  const state = useConnectionStore.getState();
  if (state.status === "reconnecting") return;
  // Synchronous, before the caller's toast: the toast for this very failure is the first one the
  // banner replaces.
  useConnectionStore.setState({
    status: "reconnecting",
    disconnectedSince: rt.deps.now(),
    nextRetryAt: null,
    failedProbes: 0,
    reconnectedAt: null,
  });
  // A heartbeat already in flight settles into the reconnecting branch on its own.
  runProbe(rt);
}

/** "Retry now". Ignored while online or while a probe is already out. */
export function retryConnectionNow(): void {
  const rt = runtime;
  if (!rt) return;
  if (useConnectionStore.getState().status !== "reconnecting") return;
  runProbe(rt);
}

/** True when `text` is the news that a request never reached the backend. */
export function isTransportFailureText(text: unknown): boolean {
  if (typeof text !== "string" || text.length === 0) return false;
  for (const message of transportMessages) {
    if (text.includes(message)) return true;
  }
  if (FETCH_FAILURE_FRAGMENTS.some((fragment) => text.includes(fragment))) {
    return true;
  }
  return FETCH_FAILURE_EXACT.some(
    (exact) => text === exact || text.endsWith(`: ${exact}`),
  );
}

/**
 * Whether an error toast should be held back because the banner already says it. Only transport
 * failures, and only while reconnecting: anything else is news the banner does not carry. `replay`
 * shows the toast after all, and is kept for the first one in case the outage is never confirmed.
 */
export function suppressTransportErrorToast(
  texts: readonly unknown[],
  replay: () => void,
): boolean {
  const rt = runtime;
  if (!rt || useConnectionStore.getState().status !== "reconnecting") {
    return false;
  }
  if (!texts.some(isTransportFailureText)) return false;
  rt.replaySuppressed ??= replay;
  return true;
}

/** A probe of /api/liveness. No login, no retry ladder, and no query string, which would split the
 *  access log's dedup bucket for the route. */
export function createLivenessProbe(
  resolveUrl: () => string,
  timeoutMs: number = PROBE_TIMEOUT_MS,
): () => Promise<LivenessProbeResult> {
  return async () => {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
    try {
      const response = await fetch(resolveUrl(), {
        cache: "no-store",
        signal: controller.signal,
      });
      if (GATEWAY_STATUSES.has(response.status)) return { kind: "unreachable" };
      let instanceId: string | null = null;
      if (response.ok) {
        try {
          const body = (await response.json()) as { instance_id?: unknown };
          if (typeof body?.instance_id === "string" && body.instance_id) {
            instanceId = body.instance_id;
          }
        } catch {
          // Answered, so reachable; just not able to say which process it is.
        }
      }
      return { kind: "reachable", instanceId };
    } catch {
      return timedOut ? { kind: "inconclusive" } : { kind: "unreachable" };
    } finally {
      clearTimeout(timer);
    }
  };
}
