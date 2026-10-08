// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { resyncInferenceStatusAfterServerModelChange } from "@/features/chat";
import { formatRelativeTime, type Locale, useLocale, useT } from "@/i18n";
import { apiUrl } from "@/lib/api-base";
import {
  createLivenessProbe,
  retryConnectionNow,
  startConnectionMonitor,
  useConnectionStore,
} from "@/lib/connection-monitor";
import { notifyModelEjected } from "@/lib/model-lifecycle-events";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Z_LAYER } from "@/lib/z-layers";
import { Alert02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useId, useState } from "react";

const RESTARTED_TOAST_ID = "connection-restarted";
const RECONNECTED_TOAST_ID = "connection-reconnected";
const RECONNECTED_TOAST_MS = 4_000;

function formatClockTime(locale: Locale, at: number): string {
  return new Date(at).toLocaleTimeString(locale, {
    hour: "numeric",
    minute: "2-digit",
  });
}

/** The time, refreshed every second while `active`, for the retry countdown. */
function useSecondTicker(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const tick = () => setNow(Date.now());
    // Once straight away: `now` is from whenever the banner last ticked, so an outage an hour
    // into the session would otherwise count down from an hour ago for its first second.
    const first = window.setTimeout(tick, 0);
    const interval = window.setInterval(tick, 1_000);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(interval);
    };
  }, [active]);
  return now;
}

/**
 * Runs the connection monitor for as long as the app shell is mounted, and says what it finds: one
 * banner while the backend cannot be reached, in place of a toast per failed request, and a notice
 * when the backend that comes back is a new process.
 *
 * Mounted inside the app tree on purpose. On desktop that tree is unmounted whenever the launcher
 * puts up its own startup screen ("Server stopped unexpectedly", a restart, an update), so the two
 * never show at once, and the monitor starts over with the next backend instead of reporting a
 * restart the user just watched happen. The banner covers the gap before the launcher notices, and
 * the browser build, which has no launcher at all.
 *
 * `authFlow`: the sign-in pages still get the banner, since signing in fails the same way, but not
 * the restart notice, which is about models a signed-out page knows nothing of.
 */
export function ConnectionBanner({ authFlow }: { authFlow: boolean }) {
  const t = useT();
  const locale = useLocale();
  const statusId = useId();
  const status = useConnectionStore((s) => s.status);
  const failedProbes = useConnectionStore((s) => s.failedProbes);
  const disconnectedSince = useConnectionStore((s) => s.disconnectedSince);
  const nextRetryAt = useConnectionStore((s) => s.nextRetryAt);
  const probing = useConnectionStore((s) => s.probing);
  // Not on the report alone: the first probe confirms it, so a one-off failure never flashes it.
  const visible =
    status === "reconnecting" && failedProbes > 0 && disconnectedSince !== null;
  const now = useSecondTicker(visible);

  useEffect(
    () =>
      startConnectionMonitor({
        probe: createLivenessProbe(() => apiUrl("/api/liveness")),
      }),
    [],
  );

  // Notices fire on the transition between two store states, not on a render that sees a stamp, so
  // a dismissed notice cannot come back with a re-render, and re-subscribing on a locale change
  // replays nothing.
  useEffect(
    () =>
      useConnectionStore.subscribe((state, prev) => {
        if (state.restartedAt !== null && state.restartedAt !== prev.restartedAt) {
          if (authFlow) return;
          toast.info(
            t("connection.restarted.title", {
              time: formatClockTime(locale, state.restartedAt),
            }),
            {
              id: RESTARTED_TOAST_ID,
              description: t("connection.restarted.description"),
              // Until dismissed: it explains state the user would otherwise trip over later.
              duration: Infinity,
            },
          );
          // The new process holds nothing. The same reconciliation the llama.cpp update banner runs
          // when the server unloads a model under the UI, plus the Images and Video pages' own eject
          // hooks, which drop their resident state and re-read status.
          void resyncInferenceStatusAfterServerModelChange().catch(() => undefined);
          notifyModelEjected("image");
          notifyModelEjected("video");
          return;
        }
        if (
          state.reconnectedAt !== null &&
          state.reconnectedAt !== prev.reconnectedAt
        ) {
          toast.success(t("connection.reconnected"), {
            id: RECONNECTED_TOAST_ID,
            duration: RECONNECTED_TOAST_MS,
          });
        }
      }),
    [t, locale, authFlow],
  );

  const secondsLeft =
    nextRetryAt === null ? 0 : Math.ceil((nextRetryAt - now) / 1_000);
  const retryText =
    probing || secondsLeft <= 0
      ? t("connection.banner.retrying")
      : t("connection.banner.retryingIn", {
          countdown: formatRelativeTime(locale, secondsLeft, "second"),
        });

  return (
    <div
      className="pointer-events-none fixed inset-x-0 flex justify-center px-4"
      style={{
        top: "calc(var(--studio-window-chrome-top, 0px) + 0.5rem)",
        zIndex: Z_LAYER.CONNECTION_BANNER,
      }}
    >
      {/* The live region stays mounted, empty while online: one that appears together with its
          text is not reliably announced. */}
      <div
        className={cn(
          visible
            ? "pointer-events-auto flex max-w-xl flex-wrap items-center gap-x-3 gap-y-1.5 rounded-2xl border border-amber-500/40 bg-popover px-4 py-2 text-sm text-popover-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-[0_8px_28px_-6px_var(--background)]"
            : "sr-only",
        )}
      >
        <p
          id={statusId}
          role="status"
          aria-live="polite"
          className="flex min-w-0 items-center gap-2 font-medium"
        >
          {visible && disconnectedSince !== null && (
            <>
              <HugeiconsIcon
                icon={Alert02Icon}
                strokeWidth={2}
                aria-hidden="true"
                className="size-4 shrink-0 text-amber-600 dark:text-amber-400"
              />
              {t("connection.banner.unreachable", {
                time: formatClockTime(locale, disconnectedSince),
              })}
            </>
          )}
        </p>
        {visible && (
          <>
            {/* Outside the announcement: a countdown read out every second is noise. */}
            <span aria-hidden="true" className="text-muted-foreground tabular-nums">
              {retryText}
            </span>
            <Button
              size="xs"
              variant="outline"
              disabled={probing}
              aria-describedby={statusId}
              onClick={retryConnectionNow}
            >
              {t("connection.banner.retryNow")}
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
