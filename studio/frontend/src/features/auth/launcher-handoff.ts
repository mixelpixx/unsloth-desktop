// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { refreshSession } from "./api";
import { extractLaunchSession } from "./launcher-handoff-parse";
import {
  AUTH_REFRESH_TOKEN_KEY,
  hasAuthToken,
  hasRefreshToken,
  setMustChangePassword,
} from "./session";

/**
 * Sign in without a password when the local launcher opened this window with a one-shot token in the
 * URL hash (see launcher-handoff-parse.ts). Returns null when there is nothing to do, so a normal
 * start pays no delay; otherwise a promise that never rejects and resolves once the session (if any)
 * is stored, so the first route guard sees it. A failure simply leaves the regular login page.
 */
export function consumeLaunchSession(): Promise<void> | null {
  if (typeof window === "undefined") return null;
  const { token, remainingHash } = extractLaunchSession(window.location.hash);
  if (token === null && remainingHash === window.location.hash) return null;

  // Drop the token from the address immediately, whatever happens next.
  try {
    window.history.replaceState(
      window.history.state,
      "",
      `${window.location.pathname}${window.location.search}${remainingHash}`,
    );
  } catch {
    // Without history access the token is single-use anyway.
  }
  if (token === null) return null;

  return (async () => {
    try {
      // A session that still refreshes wins: do not replace a working login (or another account's).
      if (hasAuthToken() && hasRefreshToken() && (await refreshSession()))
        return;
      localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, token);
      if (await refreshSession()) setMustChangePassword(false);
    } catch {
      // Fall through to the login page.
    }
  })();
}
