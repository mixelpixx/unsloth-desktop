// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The bell in the sidebar header: what is running, what finished, and the errors that outlived
// their toasts. A badge counts running jobs and a red dot marks errors not yet looked at. Rows act
// only through controls their owners already have: Open goes to the job's page, View logs is the
// failure toasts' own action, Cancel only where a cancel exists (downloads, model loads), Retry
// only for downloads. Training, export and recipes are never re-run from here.

import {
  Alert02Icon,
  Cancel01Icon,
  ChefHatIcon,
  CpuIcon,
  Download01Icon,
  FolderExportIcon,
  Notification03Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { type ReactElement, useEffect, useId, useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { downloadManager, formatBytes } from "@/features/hub";
import { viewLogsAction } from "@/features/settings";
import {
  type Locale,
  type TranslationKey,
  formatRelativeTime,
  useLocale,
  useT,
} from "@/i18n";
import {
  type ActivityEntry,
  type ActivityKind,
  type ActivityState,
  clearActivity,
  countActiveEntries,
  countUnseenErrors,
  dismissActivity,
  markActivityErrorsSeen,
  selectActiveEntries,
  selectErrorEntries,
  selectRecentEntries,
  setActivityNotifyOnFinish,
  useActivityStore,
} from "@/lib/activity-store";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import {
  type ModelRuntime,
  cancelModelLoad,
} from "@/lib/model-lifecycle-events";
import { cn } from "@/lib/utils";
import {
  type JobNotificationPermission,
  jobNotificationPermission,
  requestJobNotifications,
} from "./activity-notifications";
import { openActivityEntry } from "./activity-open";

type Translate = ReturnType<typeof useT>;
type ActivityTab = "active" | "recent" | "errors";

const KIND_ICON: Record<ActivityKind, IconSvgElement> = {
  download: Download01Icon,
  training: TestTubeOutlineIcon,
  export: FolderExportIcon,
  recipe: ChefHatIcon,
  "model-load": CpuIcon,
  error: Alert02Icon,
};

const KIND_LABEL: Record<ActivityKind, TranslationKey> = {
  download: "activity.kind.download",
  training: "activity.kind.training",
  export: "activity.kind.export",
  recipe: "activity.kind.recipe",
  "model-load": "activity.kind.modelLoad",
  error: "activity.kind.error",
};

const STATE_LABEL: Record<ActivityState, TranslationKey> = {
  active: "activity.state.active",
  done: "activity.state.done",
  failed: "activity.state.failed",
  cancelled: "activity.state.cancelled",
};

const ICON_TONE: Record<ActivityState, string> = {
  active: "text-foreground",
  done: "text-status-success",
  failed: "text-destructive",
  cancelled: "text-muted-foreground",
};

const SOFT_HOVER =
  "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground";

function entryTitle(t: Translate, entry: ActivityEntry): string {
  if (entry.title) return entry.title;
  return entry.kind === "error"
    ? t("activity.untitledError")
    : t(KIND_LABEL[entry.kind]);
}

function meterText(t: Translate, entry: ActivityEntry): string | null {
  const meter = entry.meter;
  if (!meter || meter.total <= 0) return null;
  if (meter.unit === "bytes") {
    return t("activity.meter.bytes", {
      done: formatBytes(meter.done),
      total: formatBytes(meter.total),
    });
  }
  return t(
    meter.unit === "steps" ? "activity.meter.steps" : "activity.meter.rows",
    { done: meter.done, total: meter.total },
  );
}

/** "just now", "5 min. ago", "3 hr. ago", "2 days ago", in the UI's language. */
function timeAgo(t: Translate, locale: Locale, at: number): string {
  const minutes = Math.floor((Date.now() - at) / 60_000);
  if (minutes < 1) return t("activity.justNow");
  if (minutes < 60) return formatRelativeTime(locale, -minutes, "minute");
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return formatRelativeTime(locale, -hours, "hour");
  return formatRelativeTime(locale, -Math.floor(hours / 24), "day");
}

/** One line under the title: what kind of job, where it stands, and when. */
function statusLine(
  t: Translate,
  locale: Locale,
  entry: ActivityEntry,
): string {
  const parts = [t(KIND_LABEL[entry.kind])];
  if (entry.state === "active") {
    parts.push(meterText(t, entry) ?? entry.detail ?? t(STATE_LABEL.active));
    parts.push(timeAgo(t, locale, entry.startedAt));
  } else {
    if (entry.kind !== "error") parts.push(t(STATE_LABEL[entry.state]));
    parts.push(timeAgo(t, locale, entry.finishedAt ?? entry.startedAt));
  }
  return parts.join(" · ");
}

function ActionButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex h-6 shrink-0 cursor-pointer items-center rounded-full border border-border/70 px-2.5 text-ui-11 font-medium text-muted-foreground transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        SOFT_HOVER,
      )}
    >
      {label}
    </button>
  );
}

function ActivityRow({
  entry,
  t,
  locale,
  onOpen,
  onLogs,
}: {
  entry: ActivityEntry;
  t: Translate;
  locale: Locale;
  onOpen: (entry: ActivityEntry) => void;
  onLogs: () => void;
}) {
  const title = entryTitle(t, entry);
  const status = statusLine(t, locale, entry);
  const titleId = useId();
  const statusId = useId();
  const logsAction =
    entry.actions.includes("logs") && entry.logs
      ? viewLogsAction(entry.logs.family, entry.logs.sourcePath)
      : undefined;
  const has = (action: ActivityEntry["actions"][number]) =>
    entry.actions.includes(action);
  const showDetail = entry.state !== "active" && entry.detail;
  const progressLabel = meterText(t, entry) ?? t(STATE_LABEL.active);

  const retry = () => {
    const spec = entry.retry;
    if (!spec) return;
    dismissActivity(entry.id);
    void downloadManager
      .requestStart({
        kind: spec.kind,
        repoId: spec.repoId,
        variant: spec.variant,
        expectedBytes: spec.expectedBytes,
        inventoryKind: spec.inventoryKind,
        scopeId: spec.scopeId,
        files: spec.files,
        checkpoint: spec.checkpoint,
      })
      .then((outcome) => {
        // An earlier partial on the other transport: the Hub is where that choice is made.
        if (outcome === "conflict") onOpen(entry);
      });
  };

  const cancel = () => {
    if (!entry.ref) return;
    if (entry.kind === "download") void downloadManager.cancel(entry.ref);
    else if (entry.kind === "model-load")
      cancelModelLoad(entry.ref as ModelRuntime);
  };

  return (
    <li
      aria-labelledby={`${titleId} ${statusId}`}
      className="flex flex-col gap-1.5 rounded-[14px] px-2 py-2"
    >
      <div className="flex items-start gap-2.5">
        <HugeiconsIcon
          icon={KIND_ICON[entry.kind]}
          strokeWidth={1.75}
          aria-hidden="true"
          className={cn("mt-0.5 size-4 shrink-0", ICON_TONE[entry.state])}
        />
        <div className="min-w-0 flex-1">
          <span
            id={titleId}
            title={title}
            className="block truncate text-ui-12p5 font-medium text-foreground"
          >
            {title}
            {entry.count > 1 && (
              <span className="ml-1.5 text-ui-11 font-normal text-muted-foreground tabular-nums">
                {t("activity.repeated", { count: entry.count })}
              </span>
            )}
          </span>
          <span
            id={statusId}
            className="block truncate text-ui-11 text-muted-foreground tabular-nums"
          >
            {status}
          </span>
          {showDetail && (
            <p className="mt-0.5 line-clamp-3 select-text break-words text-ui-11 text-muted-foreground">
              {entry.detail}
            </p>
          )}
        </div>
        {has("dismiss") && (
          <button
            type="button"
            aria-label={t("activity.actions.dismiss", { title })}
            title={t("activity.actions.dismiss", { title })}
            onClick={() => dismissActivity(entry.id)}
            className={cn(
              "inline-flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              SOFT_HOVER,
            )}
          >
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={1.75}
              className="size-3.5"
            />
          </button>
        )}
      </div>
      {entry.state === "active" && (
        <Progress
          aria-label={progressLabel}
          value={entry.progress === null ? undefined : entry.progress * 100}
          indeterminate={entry.progress === null}
          className="ml-6.5 h-1 w-auto"
        />
      )}
      {(has("open") || logsAction || has("cancel") || has("retry")) && (
        <div className="ml-6.5 flex flex-wrap gap-1.5">
          {has("open") && (
            <ActionButton
              label={t("activity.actions.open")}
              onClick={() => onOpen(entry)}
            />
          )}
          {logsAction && (
            <ActionButton
              label={logsAction.label}
              onClick={() => {
                onLogs();
                logsAction.onClick();
              }}
            />
          )}
          {has("cancel") && (
            <ActionButton
              label={t("activity.actions.cancel")}
              onClick={cancel}
            />
          )}
          {has("retry") && (
            <ActionButton label={t("activity.actions.retry")} onClick={retry} />
          )}
        </div>
      )}
    </li>
  );
}

function NotifyToggle({ t }: { t: Translate }) {
  const on = useActivityStore((state) => state.notifyOnFinish);
  const [permission, setPermission] = useState<JobNotificationPermission>(
    jobNotificationPermission,
  );
  const [asking, setAsking] = useState(false);
  const switchId = useId();

  const toggle = async (next: boolean) => {
    if (!next) {
      setActivityNotifyOnFinish(false);
      return;
    }
    setAsking(true);
    try {
      const answer = await requestJobNotifications();
      setPermission(answer);
      setActivityNotifyOnFinish(answer === "granted");
    } finally {
      setAsking(false);
    }
  };

  const hint =
    permission === "unsupported"
      ? t("activity.notify.unsupported")
      : permission === "denied"
        ? t("activity.notify.blocked")
        : null;

  return (
    <div className="flex flex-col gap-1 border-t border-border/60 px-1 pt-2.5">
      <div className="flex items-center gap-2.5">
        <label
          htmlFor={switchId}
          className="min-w-0 flex-1 cursor-pointer text-ui-12 text-foreground"
        >
          {t("activity.notify.toggle")}
        </label>
        <Switch
          id={switchId}
          size="sm"
          checked={on && permission === "granted"}
          disabled={asking || permission === "unsupported"}
          onCheckedChange={(next) => void toggle(next)}
        />
      </div>
      {hint && <p className="text-ui-11 text-muted-foreground">{hint}</p>}
    </div>
  );
}

function ActivityPanel({
  initialTab,
  onClose,
}: {
  initialTab: ActivityTab;
  onClose: () => void;
}) {
  const t = useT();
  const locale = useLocale();
  const navigate = useNavigate();
  const entries = useActivityStore((state) => state.entries);
  const seenErrorsAt = useActivityStore((state) => state.seenErrorsAt);
  const [tab, setTab] = useState<ActivityTab>(initialTab);
  const lists = useMemo(
    () => ({
      active: selectActiveEntries(entries),
      recent: selectRecentEntries(entries),
      errors: selectErrorEntries(entries),
    }),
    [entries],
  );
  const unseen = countUnseenErrors(entries, seenErrorsAt);

  // Looked at once the Errors tab is on screen, including errors that arrive while it is.
  useEffect(() => {
    if (tab === "errors" && unseen > 0) markActivityErrorsSeen();
  }, [tab, unseen]);

  const open = (entry: ActivityEntry) => {
    onClose();
    openActivityEntry(entry, navigate);
  };

  const empty: Record<ActivityTab, TranslationKey> = {
    active: "activity.empty.active",
    recent: "activity.empty.recent",
    errors: "activity.empty.errors",
  };
  const tabs: { value: ActivityTab; label: TranslationKey; count: number }[] = [
    {
      value: "active",
      label: "activity.tabs.active",
      count: lists.active.length,
    },
    { value: "recent", label: "activity.tabs.recent", count: 0 },
    { value: "errors", label: "activity.tabs.errors", count: unseen },
  ];

  return (
    <Tabs
      value={tab}
      onValueChange={(value) => setTab(value as ActivityTab)}
      className="gap-2 font-heading"
    >
      <div className="flex items-center gap-2 px-1">
        <h2 className="min-w-0 flex-1 truncate text-ui-13p5 font-semibold text-foreground">
          {t("activity.title")}
        </h2>
        {tab !== "active" && lists[tab].length > 0 && (
          <button
            type="button"
            onClick={() => clearActivity(tab)}
            className={cn(
              "flex h-7 shrink-0 cursor-pointer items-center rounded-full px-2.5 text-ui-12 text-muted-foreground transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              SOFT_HOVER,
            )}
          >
            {t(
              tab === "recent"
                ? "activity.clearRecent"
                : "activity.clearErrors",
            )}
          </button>
        )}
      </div>
      <TabsList className="w-full">
        {tabs.map(({ value, label, count }) => (
          <TabsTrigger key={value} value={value} className="text-ui-12">
            {t(label)}
            {count > 0 && (
              <span
                className={cn(
                  "ml-1.5 inline-flex min-w-4 items-center justify-center rounded-full px-1 text-ui-10 tabular-nums",
                  value === "errors"
                    ? "bg-destructive text-white"
                    : "bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)]",
                )}
              >
                {count}
              </span>
            )}
          </TabsTrigger>
        ))}
      </TabsList>
      {tabs.map(({ value }) => (
        <TabsContent key={value} value={value} className="min-h-0">
          {lists[value].length === 0 ? (
            <p className="px-2 py-6 text-center text-ui-12 text-muted-foreground">
              {t(empty[value])}
            </p>
          ) : (
            <ul className="flex max-h-[min(60vh,calc(420px*var(--ui-space-scale,1)))] flex-col gap-0.5 overflow-y-auto">
              {lists[value].map((entry) => (
                <ActivityRow
                  key={entry.id}
                  entry={entry}
                  t={t}
                  locale={locale}
                  onOpen={open}
                  onLogs={onClose}
                />
              ))}
            </ul>
          )}
        </TabsContent>
      ))}
      <NotifyToggle t={t} />
    </Tabs>
  );
}

export function ActivityBell({
  className,
  side = "bottom",
}: {
  className?: string;
  /** Where the tooltip and panel open: below the expanded header, right of the icon rail. */
  side?: "bottom" | "right";
}): ReactElement {
  const t = useT();
  const activeCount = useActivityStore((state) =>
    countActiveEntries(state.entries),
  );
  const unseen = useActivityStore((state) =>
    countUnseenErrors(state.entries, state.seenErrorsAt),
  );
  const [open, setOpen] = useState(false);
  const [initialTab, setInitialTab] = useState<ActivityTab>("active");
  const label = t("activity.bellLabel", {
    active: activeCount,
    errors: unseen,
  });

  const onOpenChange = (next: boolean) => {
    // Open on what is news: unseen errors first, then what is running, else the history.
    if (next) {
      setInitialTab(
        unseen > 0 ? "errors" : activeCount > 0 ? "active" : "recent",
      );
    }
    setOpen(next);
  };

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <Tooltip>
        <TooltipPrimitive.Trigger asChild={true}>
          <PopoverTrigger asChild={true}>
            <button
              type="button"
              aria-label={label}
              data-activity-bell=""
              className={cn(
                "relative top-px inline-flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                className,
              )}
            >
              <HugeiconsIcon
                icon={Notification03Icon}
                strokeWidth={1.75}
                className="size-4"
              />
              {activeCount > 0 && (
                <span
                  aria-hidden="true"
                  className="absolute right-0 bottom-0 inline-flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-control-accent px-[3px] text-[9px] font-semibold leading-none text-control-accent-foreground tabular-nums"
                >
                  {activeCount > 9 ? "9+" : activeCount}
                </span>
              )}
              {unseen > 0 && (
                <span
                  aria-hidden="true"
                  className="absolute top-1 right-1 size-2 rounded-full bg-destructive"
                />
              )}
            </button>
          </PopoverTrigger>
        </TooltipPrimitive.Trigger>
        <TooltipContent
          side={side}
          sideOffset={side === "right" ? 8 : 6}
          className="tooltip-compact"
        >
          {t("activity.title")}
        </TooltipContent>
      </Tooltip>
      <PopoverContent
        side={side}
        align="start"
        sideOffset={side === "right" ? 8 : 6}
        aria-label={t("activity.title")}
        className="menu-soft-surface w-[calc(360px*var(--ui-space-scale,1))] gap-2 rounded-[20px] p-3"
      >
        <ActivityPanel initialTab={initialTab} onClose={() => setOpen(false)} />
      </PopoverContent>
    </Popover>
  );
}
