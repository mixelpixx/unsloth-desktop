// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Alert02Icon,
  CheckmarkCircle02Icon,
  InformationCircleIcon,
  MultiplicationSignCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useTheme } from "@/features/settings/stores/theme-store";
import { suppressTransportErrorToast } from "@/lib/connection-monitor";
import { createLoadingToastIcon } from "@/lib/toast";
import { Toaster as Sonner, type ToasterProps, toast } from "sonner";

// Sonner's Toaster takes one duration for every type, and an error needs longer than 5 s to be
// read. So errors get their own default on the shared `toast` instance, which also reaches the
// call sites that import "sonner" directly rather than "@/lib/toast". A caller's own duration
// still wins.
const ERROR_TOAST_DURATION_MS = 15_000;
// Returned for a toast held back below; no toast has it, so a later dismiss() by it is a no-op.
const SUPPRESSED_TOAST_ID = "connection-suppressed";
const showErrorToast = toast.error;
toast.error = (message, data) => {
  const show = () =>
    showErrorToast(message, {
      ...data,
      duration: data?.duration ?? ERROR_TOAST_DURATION_MS,
    });
  // While the connection banner says the backend cannot be reached, a toast per failed request is
  // the same news again, once per request in flight. Only that news is held back; any other error
  // still shows. A toast with an id is updating one already on screen, a loading toast more often
  // than not, so it always goes through rather than leave that one spinning.
  if (
    data?.id === undefined &&
    suppressTransportErrorToast([message, data?.description], show)
  ) {
    return SUPPRESSED_TOAST_ID;
  }
  return show();
};

// Make toast text selectable. Sonner's onPointerDown calls setPointerCapture(), which steals the
// drag and blocks text selection. dismissible:false would stop it but also kills the close button.
// So we swallow pointerdown on toast text (never on its buttons) before sonner sees it.
const handleToastPointerDownCapture = (
  event: React.PointerEvent<HTMLDivElement>,
) => {
  // closest() lives on Element, so this also covers SVG icon targets; guard
  // against non-Element targets defensively.
  const target = event.target as Element | null;
  if (typeof target?.closest !== "function") return;
  if (!target.closest("[data-sonner-toast]")) return;
  if (
    target.closest("button,[data-button],[data-close-button],[data-cancel]")
  ) {
    return;
  }
  event.stopPropagation();
};

const Toaster = ({ ...props }: ToasterProps) => {
  // Use the resolved mode so sonner's data-sonner-theme always matches the
  // class the theme store puts on <html>.
  const { resolved } = useTheme();

  return (
    // display:contents adds no box; only carries the selection-fix handler.
    // biome-ignore lint/a11y/noStaticElementInteractions: capture-only guard, not interactive
    <div
      style={{ display: "contents" }}
      onPointerDownCapture={handleToastPointerDownCapture}
    >
      <Sonner
        theme={resolved}
        className="toaster group"
        duration={5000}
        icons={{
          success: (
            <HugeiconsIcon
              icon={CheckmarkCircle02Icon}
              strokeWidth={2}
              className="size-4"
            />
          ),
          info: (
            <HugeiconsIcon
              icon={InformationCircleIcon}
              strokeWidth={2}
              className="size-4"
            />
          ),
          warning: (
            <HugeiconsIcon
              icon={Alert02Icon}
              strokeWidth={2}
              className="size-4"
            />
          ),
          error: (
            <HugeiconsIcon
              icon={MultiplicationSignCircleIcon}
              strokeWidth={2}
              className="size-4"
            />
          ),
          // App-wide arc spinner so loading toasts match the "Downloading model" toast.
          loading: createLoadingToastIcon(),
        }}
        style={
          {
            "--normal-bg": "var(--popover)",
            "--normal-text": "var(--popover-foreground)",
            // No border line; elevation comes from the shadow in index.css.
            "--normal-border": "transparent",
            // Rounder than cards, a step below the composer's 28px.
            "--border-radius": "calc(var(--radius) + 8px)",
            // Pin the close button inside the toast's top-right corner.
            // Sonner defaults to the left/outside edge, so keep the horizontal
            // override here and the top offset in index.css.
            "--toast-close-button-start": "auto",
            "--toast-close-button-end": "12px",
            "--toast-close-button-transform": "none",
          } as React.CSSProperties
        }
        // No swipe gestures; text selection handled by the wrapper above.
        swipeDirections={[]}
        toastOptions={{
          classNames: {
            // an open modal dialog sets pointer-events:none on body, which toasts would otherwise inherit.
            toast: "cn-toast pointer-events-auto",
            description: "!text-muted-foreground",
          },
        }}
        {...props}
      />
    </div>
  );
};

export { Toaster };
