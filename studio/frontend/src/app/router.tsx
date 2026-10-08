// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ErrorComponentProps,
  Link,
  createRouter,
  useRouter,
  useRouterState,
} from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { MascotImg } from "@/components/mascot-img";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Route as rootRoute } from "./routes/__root";
import { Route as apiMonitorRoute } from "./routes/api";
import { Route as dataRecipesRoute } from "./routes/data-recipes";
import { Route as dataRecipeRoute } from "./routes/data-recipes.$recipeId";
import { Route as chatRoute } from "./routes/chat";
import { Route as exportRoute } from "./routes/export";
import { Route as imagesRoute } from "./routes/images";
import { Route as videoRoute } from "./routes/video";
import { Route as audioRoute } from "./routes/audio";
import { Route as indexRoute } from "./routes/index";
import { Route as loginRoute } from "./routes/login";
import { Route as hubRoute } from "./routes/hub";
import { Route as projectsRoute } from "./routes/projects";
import { Route as libraryRoute } from "./routes/library";
import { Route as changePasswordRoute } from "./routes/change-password";
import { Route as settingsRoute } from "./routes/settings";
import { Route as studioRoute } from "./routes/studio";

const routeTree = rootRoute.addChildren([
  indexRoute,
  loginRoute,
  changePasswordRoute,
  hubRoute,
  settingsRoute,
  studioRoute,
  chatRoute,
  projectsRoute,
  libraryRoute,
  exportRoute,
  imagesRoute,
  videoRoute,
  audioRoute,
  dataRecipesRoute,
  dataRecipeRoute,
  apiMonitorRoute,
]);

function DefaultNotFound() {
  const t = useT();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
      <MascotImg src="Sloth emojis/sloth shy large.png" className="size-24" />
      <div className="flex flex-col items-center gap-1">
        <h1 className="font-heading font-semibold text-2xl tracking-tight">
          {t("shell.notFound.title")}
        </h1>
        <p className="text-muted-foreground text-sm break-all">
          {t("shell.notFound.description", { path: pathname })}
        </p>
      </div>
      <Button asChild>
        <Link to="/chat">{t("shell.notFound.backToChat")}</Link>
      </Button>
    </div>
  );
}

// How Chromium/Firefox and Safari word a lazy chunk the server no longer has: the
// page outlived an update or rebuild, so only a reload fixes it.
const STALE_CHUNK_ERROR =
  /dynamically imported module|Importing a module script failed/i;

function appErrorDetails(error: unknown, componentStack?: string): string {
  const lines = [
    error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    `URL: ${window.location.href}`,
    `Time: ${new Date().toISOString()}`,
    `User agent: ${navigator.userAgent}`,
  ];
  if (error instanceof Error && error.stack) lines.push("", error.stack);
  if (componentStack) lines.push("", "Component stack:", componentStack.trim());
  return lines.join("\n");
}

// Plain English, not useT: this has to render when anything else, the locale
// layer included, is what failed.
function AppError({ error, info, reset }: ErrorComponentProps) {
  const router = useRouter();
  const [copied, setCopied] = useState<boolean | null>(null);
  const message = error instanceof Error ? error.message : String(error);
  const staleChunk = STALE_CHUNK_ERROR.test(message);

  useEffect(() => {
    if (copied === null) return;
    const timer = window.setTimeout(() => setCopied(null), 2000);
    return () => window.clearTimeout(timer);
  }, [copied]);

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
      <div role="alert" className="flex max-w-md flex-col items-center gap-1">
        <h1 className="font-heading font-semibold text-2xl tracking-tight">
          {staleChunk ? "Unsloth was updated" : "Something went wrong"}
        </h1>
        <p className="text-muted-foreground text-sm [overflow-wrap:anywhere]">
          {staleChunk
            ? "Reload to continue with the new version."
            : message || "An unexpected error stopped this page."}
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button type="button" onClick={() => window.location.reload()}>
          Reload
        </Button>
        {/* Retrying would fetch the same missing chunk. */}
        {staleChunk ? null : (
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              // reset clears a render error; invalidate re-runs a failed loader.
              reset();
              void router.invalidate();
            }}
          >
            Try again
          </Button>
        )}
        <Button
          type="button"
          variant="ghost"
          onClick={async () => {
            setCopied(
              await copyToClipboard(
                appErrorDetails(error, info?.componentStack),
              ),
            );
          }}
        >
          {copied === null ? "Copy details" : copied ? "Copied" : "Copy failed"}
        </Button>
      </div>
    </div>
  );
}

export const router = createRouter({
  routeTree,
  defaultNotFoundComponent: DefaultNotFound,
  defaultErrorComponent: AppError,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
