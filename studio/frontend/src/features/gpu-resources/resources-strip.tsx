// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sidebar's always-visible GPU memory: one thin bar per card, split into Studio, other apps
// and free, so memory another program holds (LM Studio, a game) is on screen before a load runs
// into it. Collapsed to the icon rail it is one short column per card. Clicking opens a panel
// naming every resident model and every other app, with Eject per model and for all.

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useSidebar } from "@/components/ui/sidebar";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { RemoveCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { ejectResourceModel } from "./resources-eject";
import {
  type ResourceApp,
  type ResourceGpu,
  type ResourceModel,
  type ResourceSnapshot,
  barSegments,
  canEjectModel,
  formatResourceGiB,
  gpuFigures,
  modelFacts,
  shortResourceName,
} from "./resources-model";
import { useResourcesPolling, useResourcesStore } from "./resources-store";
import { useShowResourcesStrip } from "./show-resources-strip-pref";

type Translate = ReturnType<typeof useT>;

const STUDIO_FILL = "bg-control-accent";
const OTHER_FILL = "bg-amber-500 dark:bg-amber-400";
const USED_FILL = "bg-muted-foreground/60";
const TRACK = "bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)]";

function barLabel(t: Translate, gpu: ResourceGpu): string {
  const figures = gpuFigures(gpu);
  return figures.studio !== null && figures.other !== null
    ? t("resources.barLabel", {
        index: figures.index,
        studio: figures.studio,
        other: figures.other,
        free: figures.free,
        total: figures.total,
      })
    : t("resources.barLabelUnsplit", {
        index: figures.index,
        used: figures.used,
        free: figures.free,
        total: figures.total,
      });
}

/** One card's bar, horizontal in the strip and the panel, vertical on the collapsed rail. */
function GpuBar({
  gpu,
  label,
  vertical = false,
  meter = false,
  className,
}: {
  gpu: ResourceGpu;
  label: string;
  vertical?: boolean;
  /** In the panel the bar is its own element, so it is a meter with values; inside the strip's
   *  button it is presentational and the button's name carries the numbers. */
  meter?: boolean;
  className?: string;
}) {
  const segments = barSegments(gpu);
  const size = (percent: number) =>
    vertical ? { height: `${percent}%` } : { width: `${percent}%` };
  return (
    <span
      role={meter ? "meter" : "img"}
      aria-label={label}
      {...(meter
        ? {
            "aria-valuemin": 0,
            "aria-valuemax": 100,
            "aria-valuenow": Math.round(segments.used),
            "aria-valuetext": label,
          }
        : {})}
      className={cn(
        "flex shrink-0 overflow-hidden rounded-full",
        vertical ? "w-[calc(5px*var(--ui-space-scale,1))] flex-col-reverse" : "h-1.5 w-full",
        TRACK,
        className,
      )}
    >
      {segments.split ? (
        <>
          <span
            className={cn("shrink-0", STUDIO_FILL, vertical ? "w-full" : "h-full")}
            style={size(segments.studio)}
          />
          <span
            className={cn("shrink-0", OTHER_FILL, vertical ? "w-full" : "h-full")}
            style={size(segments.other)}
          />
        </>
      ) : (
        <span
          className={cn("shrink-0", USED_FILL, vertical ? "w-full" : "h-full")}
          style={size(segments.used)}
        />
      )}
    </span>
  );
}

function LegendDot({ className }: { className: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn("inline-block size-2 shrink-0 rounded-full", className)}
    />
  );
}

function AppRows({ apps }: { apps: ResourceApp[] }) {
  if (apps.length === 0) return null;
  return (
    <ul className="flex flex-col gap-0.5 pl-3.5">
      {apps.map((app) => (
        <li
          key={app.pid}
          className="flex items-center gap-2 text-ui-11 text-muted-foreground"
        >
          <span className="min-w-0 flex-1 truncate" title={`${app.name} (PID ${app.pid})`}>
            {app.name}
          </span>
          <span className="shrink-0 tabular-nums">
            {formatResourceGiB(app.bytes)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function GpuSection({ gpu, t }: { gpu: ResourceGpu; t: Translate }) {
  const figures = gpuFigures(gpu);
  return (
    <section className="flex flex-col gap-1.5">
      <div className="flex min-w-0 items-baseline gap-1.5">
        <span className="shrink-0 text-ui-12p5 font-semibold text-foreground">
          {t("resources.gpuHeading", { index: gpu.index })}
        </span>
        {gpu.name && (
          <span className="min-w-0 truncate text-ui-11 text-muted-foreground">
            {gpu.name}
          </span>
        )}
      </div>
      <GpuBar gpu={gpu} label={barLabel(t, gpu)} meter={true} />
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-ui-11 text-muted-foreground tabular-nums">
        {figures.studio !== null && figures.other !== null ? (
          <>
            <span className="flex items-center gap-1">
              <LegendDot className={STUDIO_FILL} />
              {t("resources.legendStudio", { size: figures.studio })}
            </span>
            <span className="flex items-center gap-1">
              <LegendDot className={OTHER_FILL} />
              {t("resources.legendOther", { size: figures.other })}
            </span>
          </>
        ) : (
          <span className="flex items-center gap-1">
            <LegendDot className={USED_FILL} />
            {t("resources.legendInUse", { size: figures.used })}
          </span>
        )}
        <span className="flex items-center gap-1">
          <LegendDot className={TRACK} />
          {t("resources.legendFree", { free: figures.free, total: figures.total })}
        </span>
      </div>
      <AppRows apps={gpu.apps} />
    </section>
  );
}

/** "Chat · Qwen3-27B-GGUF Q4_K_M" and "GPU 0 · 66/66 layers · 8K ctx · 20.7 GiB". */
function modelLines(
  t: Translate,
  model: ResourceModel,
): { title: string; detail: string } {
  const facts = modelFacts(model);
  const name = shortResourceName(model.name);
  const title = [
    t(`resources.kind.${model.kind}`),
    model.variant ? `${name} ${model.variant}` : name,
  ].join(" · ");
  const detail: string[] = [];
  if (model.loading) detail.push(t("resources.model.loading"));
  if (facts.gpus) detail.push(t("resources.model.gpus", { ids: facts.gpus }));
  else if (facts.cpu) detail.push(t("resources.model.cpu"));
  if (facts.layers) detail.push(t("resources.model.layers", facts.layers));
  if (facts.context) detail.push(t("resources.model.context", { size: facts.context }));
  if (facts.vram) {
    detail.push(
      facts.vramApprox
        ? t("resources.model.vramApprox", { size: facts.vram })
        : facts.vram,
    );
  }
  if (model.inactive && !model.loading) detail.push(t("resources.model.cached"));
  return { title, detail: detail.join(" · ") };
}

function ModelRow({
  model,
  t,
  ejecting,
  onEject,
}: {
  model: ResourceModel;
  t: Translate;
  ejecting: boolean;
  onEject: () => void;
}) {
  const { title, detail } = modelLines(t, model);
  const label = shortResourceName(model.name);
  return (
    <li className="flex items-center gap-2">
      <div className="min-w-0 flex-1" title={model.name}>
        <span className="block truncate text-ui-12p5 font-medium text-foreground">
          {title}
        </span>
        {detail && (
          <span className="block truncate text-ui-11 text-muted-foreground tabular-nums">
            {detail}
          </span>
        )}
      </div>
      {model.loading ? (
        <span className="flex size-6 shrink-0 items-center justify-center">
          <Spinner className="size-3.5" label={t("resources.model.loading")} />
        </span>
      ) : canEjectModel(model) ? (
        <button
          type="button"
          aria-label={t("resources.ejectModel", { model: label })}
          title={t("resources.ejectModel", { model: label })}
          disabled={ejecting}
          onClick={onEject}
          className="flex size-6 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-60"
        >
          {ejecting ? (
            <Spinner className="size-3.5" label={t("resources.ejectModel", { model: label })} />
          ) : (
            <HugeiconsIcon icon={RemoveCircleIcon} strokeWidth={1.75} className="size-3.5" />
          )}
        </button>
      ) : null}
    </li>
  );
}

function reportEject(
  t: Translate,
  model: ResourceModel,
  outcome: Awaited<ReturnType<typeof ejectResourceModel>>,
  quiet: boolean,
): void {
  const name = shortResourceName(model.name);
  switch (outcome.status) {
    case "ejected":
      if (!quiet) toast.success(t("resources.toast.ejected", { model: name }));
      return;
    case "alreadyFree":
      if (!quiet) toast.info(t("resources.toast.alreadyFree", { model: name }));
      return;
    case "replaced":
      toast.info(
        t("resources.toast.replaced", {
          model: name,
          resident: shortResourceName(outcome.resident),
        }),
      );
      return;
    case "stillResident":
      toast.warning(t("resources.toast.stillResident", { model: name }));
      return;
    case "unverified":
      toast.warning(t("resources.toast.unverified", { model: name }));
      return;
  }
}

function ResourcesPanel({
  snapshot,
  t,
}: {
  snapshot: ResourceSnapshot;
  t: Translate;
}) {
  const [ejecting, setEjecting] = useState<ReadonlySet<string>>(() => new Set());
  const [ejectingAll, setEjectingAll] = useState(false);
  const refresh = useResourcesStore((s) => s.refresh);
  const ejectable = snapshot.models.filter(canEjectModel);
  const estimated = snapshot.gpus.some((gpu) => gpu.attribution !== "process");

  const eject = async (model: ResourceModel, quiet = false): Promise<boolean> => {
    setEjecting((prev) => new Set(prev).add(model.id));
    try {
      const outcome = await ejectResourceModel(model);
      reportEject(t, model, outcome, quiet);
      return outcome.status === "ejected" || outcome.status === "alreadyFree";
    } catch (error: unknown) {
      const reason = error instanceof Error && error.message ? ` ${error.message}` : "";
      toast.error(
        `${t("resources.toast.failed", { model: shortResourceName(model.name) })}${reason}`,
      );
      return false;
    } finally {
      setEjecting((prev) => {
        const next = new Set(prev);
        next.delete(model.id);
        return next;
      });
    }
  };

  const ejectAll = async () => {
    setEjectingAll(true);
    let released = true;
    try {
      // One at a time: the unloads share runtimes' locks, and a failure names its own model.
      for (const model of ejectable) {
        released = (await eject(model, true)) && released;
      }
      if (released) toast.success(t("resources.toast.ejectedAll"));
    } finally {
      setEjectingAll(false);
      void refresh();
    }
  };

  return (
    <div className="flex flex-col gap-3 font-heading">
      <div className="flex items-center gap-2">
        <h2 className="min-w-0 flex-1 truncate text-ui-13p5 font-semibold text-foreground">
          {t("resources.title")}
        </h2>
        {ejectable.length > 0 && (
          <button
            type="button"
            disabled={ejectingAll || ejecting.size > 0}
            onClick={() => void ejectAll()}
            className="flex h-7 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-ui-12 text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-60"
          >
            {ejectingAll ? (
              <Spinner className="size-3.5" label={t("resources.ejectAll")} />
            ) : (
              <HugeiconsIcon icon={RemoveCircleIcon} strokeWidth={1.75} className="size-3.5" />
            )}
            {t("resources.ejectAll")}
          </button>
        )}
      </div>
      {snapshot.gpus.map((gpu) => (
        <GpuSection key={gpu.index} gpu={gpu} t={t} />
      ))}
      <section className="flex flex-col gap-1.5 border-t border-border/60 pt-3">
        <h3 className="text-ui-12 font-semibold text-muted-foreground">
          {t("resources.modelsHeading")}
        </h3>
        {snapshot.models.length === 0 ? (
          <p className="text-ui-12 text-muted-foreground">{t("resources.noModels")}</p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {snapshot.models.map((model) => (
              <ModelRow
                key={model.id}
                model={model}
                t={t}
                ejecting={ejectingAll || ejecting.has(model.id)}
                onEject={() =>
                  void eject(model).finally(() => void refresh())
                }
              />
            ))}
          </ul>
        )}
      </section>
      {snapshot.other_apps.length > 0 && (
        <section className="flex flex-col gap-1.5 border-t border-border/60 pt-3">
          <h3 className="text-ui-12 font-semibold text-muted-foreground">
            {t("resources.otherAppsHeading")}
          </h3>
          <AppRows apps={snapshot.other_apps} />
        </section>
      )}
      {estimated && (
        <p className="text-ui-11 text-muted-foreground">{t("resources.estimated")}</p>
      )}
    </div>
  );
}

export function ResourcesStrip() {
  const show = useShowResourcesStrip();
  useResourcesPolling(show);
  const snapshot = useResourcesStore((s) => s.snapshot);
  const panelOpen = useResourcesStore((s) => s.panelOpen);
  const setPanelOpen = useResourcesStore((s) => s.setPanelOpen);
  const { isMobile } = useSidebar();
  const t = useT();

  // No card to draw (a Mac, a CPU host, a backend without the route): nothing to show.
  if (!show || !snapshot || snapshot.gpus.length === 0) return null;
  const gpus = snapshot.gpus;
  const summary = gpus
    .map((gpu) => {
      const figures = gpuFigures(gpu);
      return t("resources.gpuSummary", {
        index: figures.index,
        free: figures.free,
        total: figures.total,
      });
    })
    .join("; ");

  return (
    <Popover open={panelOpen} onOpenChange={setPanelOpen}>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          aria-label={t("resources.stripLabel", { summary })}
          data-resources-strip=""
          className="flex w-full flex-col gap-1.5 rounded-[12px] px-2 py-1.5 text-left transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring group-data-[collapsible=icon]:mx-auto group-data-[collapsible=icon]:h-[calc(34px*var(--ui-space-scale,1))] group-data-[collapsible=icon]:w-[calc(34px*var(--ui-space-scale,1))] group-data-[collapsible=icon]:flex-row group-data-[collapsible=icon]:items-center group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:gap-[3px] group-data-[collapsible=icon]:rounded-full group-data-[collapsible=icon]:p-0"
        >
          {gpus.map((gpu) => {
            const label = barLabel(t, gpu);
            return (
              <span key={gpu.index} className="contents">
                {/* Expanded: a label over a thin bar. */}
                <span className="flex flex-col gap-1 group-data-[collapsible=icon]:hidden">
                  <span className="truncate text-ui-11 tabular-nums text-muted-foreground">
                    {t("resources.gpuFree", {
                      index: gpu.index,
                      free: formatResourceGiB(gpu.free_bytes),
                    })}
                  </span>
                  <GpuBar gpu={gpu} label={label} />
                </span>
                {/* Collapsed rail: one short column per card. */}
                <GpuBar
                  gpu={gpu}
                  label={label}
                  vertical={true}
                  className="hidden h-[calc(18px*var(--ui-space-scale,1))] group-data-[collapsible=icon]:flex"
                />
              </span>
            );
          })}
        </button>
      </PopoverTrigger>
      <PopoverContent
        side={isMobile ? "top" : "right"}
        align="end"
        sideOffset={10}
        className="menu-soft-surface w-[calc(320px*var(--ui-space-scale,1))] rounded-[20px] p-3.5"
      >
        <ResourcesPanel snapshot={snapshot} t={t} />
      </PopoverContent>
    </Popover>
  );
}
