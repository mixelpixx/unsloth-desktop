// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Notify me when long jobs finish": a system notification for a finished training run, export or
// large download, sent only while the window is not in front and only once the user has turned it
// on from the bell, which is where permission is asked. Never prompted on its own.

import { translate } from "@/i18n";
import { type ActivityEntry, useActivityStore } from "@/lib/activity-store";
import { isTauri } from "@/lib/api-base";
import {
  notifyNative,
  primeNativeNotificationPermission,
  safeNotificationLabel,
  sanitizeNotificationBody,
} from "@/lib/native-notifications";
import { finishedNotice } from "./activity-sources";

export type JobNotificationPermission =
  | "granted"
  | "denied"
  | "default"
  | "unsupported";

function browserNotifications(): typeof Notification | null {
  if (typeof window === "undefined" || !("Notification" in window)) return null;
  // Browsers refuse it outside a secure context (a LAN address over plain http).
  return window.isSecureContext ? window.Notification : null;
}

export function jobNotificationPermission(): JobNotificationPermission {
  // The desktop shell asks through its own plugin, whose answer is not readable up front.
  if (isTauri) return "granted";
  return browserNotifications()?.permission ?? "unsupported";
}

/** Ask, from the toggle's click: the one place a permission prompt may come from. */
export async function requestJobNotifications(): Promise<JobNotificationPermission> {
  if (isTauri) {
    await primeNativeNotificationPermission().catch(() => undefined);
    return "granted";
  }
  const api = browserNotifications();
  if (!api) return "unsupported";
  if (api.permission !== "default") return api.permission;
  try {
    return await api.requestPermission();
  } catch {
    return "denied";
  }
}

function windowInFront(): boolean {
  if (typeof document === "undefined") return true;
  return document.visibilityState === "visible" && document.hasFocus();
}

/** Send the notice for `entry` if it is one, the user asked for them and is looking elsewhere. */
export function notifyJobFinished(
  entry: ActivityEntry,
  onOpen: (entry: ActivityEntry) => void,
): void {
  const notice = finishedNotice(entry);
  if (!notice || !useActivityStore.getState().notifyOnFinish) return;
  if (windowInFront()) return;
  const title = translate(`activity.notify.${notice}`);
  const name = safeNotificationLabel(
    entry.title,
    translate(
      entry.kind === "download"
        ? "activity.kind.download"
        : entry.kind === "export"
          ? "activity.kind.export"
          : "activity.kind.training",
    ),
  );
  const body =
    notice === "trainingFailed"
      ? sanitizeNotificationBody(entry.detail, name)
      : name;
  if (isTauri) {
    // The training lifecycle already sends its own native "finished" / "failed" notice.
    if (entry.kind === "training") return;
    void notifyNative({
      key: `activity:${entry.id}`,
      title,
      body,
      requestPermission: false,
    });
    return;
  }
  const api = browserNotifications();
  if (api?.permission !== "granted") return;
  try {
    const notification = new api(title, { body, tag: entry.id });
    notification.onclick = () => {
      window.focus();
      onOpen(entry);
      notification.close();
    };
  } catch {
    // Some browsers only allow notifications from a service worker; nothing else to try.
  }
}
