// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Passwordless hand-off from the local launcher. The launcher exchanges the machine's desktop secret
 * (the same file the Tauri shell uses) for a refresh token and opens the app window at
 * `/#unsloth-launch-session=<refresh token>`. A refresh token is single-use: the first page load
 * exchanges it for a fresh pair, so what remains in a command line or browser history is dead.
 *
 * Pure parsing lives here, free of DOM and app imports, so it can be tested on its own.
 */

export const LAUNCH_SESSION_PARAM = "unsloth-launch-session";

// secrets.token_urlsafe(48) is 64 characters of [A-Za-z0-9_-]; accept a little around that and nothing else.
const TOKEN_SHAPE = /^[A-Za-z0-9_-]{32,200}$/;

export type LaunchHandoff = {
  /** The refresh token, or null when the hash carries none (or a malformed one). */
  token: string | null;
  /** The hash with the hand-off parameter removed ("" or "#other=1"), for history.replaceState. */
  remainingHash: string;
};

export function extractLaunchSession(hash: string): LaunchHandoff {
  const body = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!body.includes(LAUNCH_SESSION_PARAM)) {
    return { token: null, remainingHash: hash };
  }
  const params = new URLSearchParams(body);
  const raw = params.get(LAUNCH_SESSION_PARAM);
  params.delete(LAUNCH_SESSION_PARAM);
  const rest = params.toString();
  return {
    token: raw !== null && TOKEN_SHAPE.test(raw) ? raw : null,
    remainingHash: rest ? `#${rest}` : "",
  };
}
