// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import { App } from "./app/app";
import {
  applyMathBlockContainment,
  watchMathBlockContainmentOverride,
} from "./components/assistant-ui/math-block-containment";
import { fetchDeviceType } from "./config/env";
import { refreshSession } from "./features/auth/api";
import {
  applyInterfaceScaleBeforeFirstPaint,
  useInterfaceScaleStore,
} from "./features/settings/stores/interface-scale-store";
import { initializeLocale } from "./i18n";
import { isTauri } from "./lib/api-base";
import { setHubSessionRefresh } from "./lib/hf-endpoint";
import { watchInputModality } from "./lib/input-modality";
import { watchOverlayScrollbarGutter } from "./lib/overlay-scrollbar";

setHubSessionRefresh(refreshSession);

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}
const root = createRoot(rootElement);

if (isTauri) {
  document.documentElement.classList.add("tauri");
}

// Rasterization follows the browser OS, not the potentially remote server.
// This adjustment is calibrated for desktop Linux, so exclude Android.
const uaLower = navigator.userAgent.toLowerCase();
if (uaLower.includes("linux") && !uaLower.includes("android")) {
  document.documentElement.classList.add("render-linux");
}

// index.css keys off this to restore ::-webkit-scrollbar styling on Windows.
if (uaLower.includes("windows")) {
  document.documentElement.classList.add("client-windows");
}

// Whether off-screen maths takes containment. ON by default, subject to a feature detect for the
// engine's find-in-page, so on a recent engine this normally SETS the attribute and arms the rule;
// on an older one it removes an attribute that was never there. Before the first render, because
// the rule it arms is a rendering rule and arming it late would relayout the first thread that
// mounts.
applyMathBlockContainment();
// And keep watching, so a devtools flip of `__UNSLOTH_MATH_BLOCK_CONTAINMENT__` reapplies instead of
// leaving the session measuring the arm it was already in.
watchMathBlockContainmentOverride();

// Keep right-edge controls clear of overlay scrollbars.
watchOverlayScrollbarGutter(window);
watchInputModality(window);

// An update or rebuild replaces the hashed chunks this page was built against, so its next lazy
// import 404s. Reload once to pick up the new build: route chunks already get that from the
// router's lazyRouteComponent, this covers every other lazy import and stylesheet. The import
// still rejects, so nothing renders half-loaded before the reload lands. The timestamp stops a
// loop when the chunk is missing for some other reason: a second failure right after the reload
// is left to the router's error screen.
const PRELOAD_RELOAD_KEY = "unsloth:preload-error-reload-at";
const PRELOAD_RELOAD_WINDOW_MS = 10_000;
window.addEventListener("vite:preloadError", () => {
  try {
    const lastReload = Number(sessionStorage.getItem(PRELOAD_RELOAD_KEY));
    if (lastReload && Date.now() - lastReload < PRELOAD_RELOAD_WINDOW_MS) {
      return;
    }
    sessionStorage.setItem(PRELOAD_RELOAD_KEY, String(Date.now()));
  } catch {
    // Without storage there is no loop guard, so leave the error to the error screen.
    return;
  }
  window.location.reload();
});

function renderApp(): void {
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

const localeInitialization = initializeLocale();
const interfaceScaleInitialization = applyInterfaceScaleBeforeFirstPaint(
  useInterfaceScaleStore.getState().scale,
);
if (typeof localeInitialization !== "string" || isTauri) {
  Promise.all([localeInitialization, interfaceScaleInitialization]).then(
    renderApp,
  );
} else {
  renderApp();
}

fetchDeviceType().catch(() => undefined);
