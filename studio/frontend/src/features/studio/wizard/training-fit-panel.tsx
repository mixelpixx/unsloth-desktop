// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  type TrainingFitEstimate,
  type TrainingFitPart,
  type TrainingGpuDevice,
  type TrainingGpuTarget,
  formatFitGiB,
  trainingFitBarGeometry,
  trainingGpuTargetOptions,
  useTrainingConfigStore,
  useTrainingFitLineText,
  useTrainingFitStore,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { type ReactElement, useId } from "react";
import { useShallow } from "zustand/react/shallow";

/**
 * One hue family for the run's own state (weights, adapters, optimizer) and the foreground
 * for what training adds on top (gradients, activations, overhead), stepped in luminance so the
 * parts stay distinguishable without a palette of their own. Same tokens as the Hub memory bar,
 * so both re-theme with the Appearance accent.
 */
const PART_COLORS: Record<TrainingFitPart, string> = {
  modelWeights: "var(--primary)",
  loraAdapters: "color-mix(in oklab, var(--primary) 62%, black)",
  optimizerStates: "color-mix(in oklab, var(--primary) 45%, var(--foreground))",
  gradients: "color-mix(in oklab, var(--foreground) 70%, transparent)",
  activations: "color-mix(in oklab, var(--foreground) 45%, transparent)",
  cudaOverhead: "color-mix(in oklab, var(--foreground) 25%, transparent)",
};

const PART_LABEL_KEYS: Record<TrainingFitPart, TranslationKey> = {
  modelWeights: "trainingFit.partWeights",
  loraAdapters: "trainingFit.partAdapters",
  optimizerStates: "trainingFit.partOptimizer",
  gradients: "trainingFit.partGradients",
  activations: "trainingFit.partActivations",
  cudaOverhead: "trainingFit.partOverhead",
};

// A part that is not zero never draws thinner than this, or a 0.1 GiB adapter on a 24 GiB card
// reads as absent. Presentational only; the widths behind it stay exact.
const MIN_SEGMENT_PX = 2;

function GpuTargetSelect({
  devices,
  target,
}: {
  devices: TrainingGpuDevice[];
  target: TrainingGpuTarget;
}): ReactElement | null {
  const t = useT();
  const selectId = useId();
  const setGpuTarget = useTrainingFitStore((s) => s.setGpuTarget);
  const options = trainingGpuTargetOptions(devices);
  if (options.length === 0) {
    return null;
  }
  const optionLabel = (option: TrainingGpuTarget): string => {
    if (option === "auto") return t("trainingFit.targetAuto");
    if (option === "all") {
      return devices.length === 2
        ? t("trainingFit.targetBoth")
        : t("trainingFit.targetAll", { count: String(devices.length) });
    }
    const index = Number(option.slice("gpu:".length));
    const device = devices.find((candidate) => candidate.index === index);
    return t("trainingFit.targetGpu", {
      index: String(index),
      name: device?.name ?? "",
    });
  };
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-3">
        <label
          htmlFor={selectId}
          className="shrink-0 text-ui-11p5 text-muted-foreground/85"
        >
          {t("trainingFit.targetLabel")}
        </label>
        <Select
          value={target}
          onValueChange={(value) => setGpuTarget(value as TrainingGpuTarget)}
        >
          <SelectTrigger
            id={selectId}
            size="sm"
            className="h-7 min-w-0 max-w-[65%] text-ui-12"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map((option) => (
              <SelectItem key={option} value={option}>
                {optionLabel(option)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {/* Studio splits layers across cards in one process (no data parallelism, and no NCCL on
          Windows), so a second GPU buys room, not speed. Said where the choice is made. */}
      {target === "all" ? (
        <p className="text-ui-10p5 leading-relaxed text-muted-foreground/75">
          {t("trainingFit.targetAllHint")}
        </p>
      ) : null}
    </div>
  );
}

function FitBar({
  estimate,
  label,
}: {
  estimate: TrainingFitEstimate;
  label: string;
}): ReactElement | null {
  const t = useT();
  const capacityGb = estimate.usableGb ?? 0;
  // A fallback estimate has a total and no itemized parts; it draws as one undivided bar
  // rather than inventing a split the estimator never made.
  const geometry = estimate.breakdown
    ? trainingFitBarGeometry(estimate.breakdown, capacityGb)
    : null;
  const total = estimate.requiredGb ?? 0;
  const scale = Math.max(total, capacityGb);
  const segments = geometry?.segments ?? [];
  const capacityPct = geometry
    ? geometry.capacityPct
    : scale > 0
      ? (capacityGb / scale) * 100
      : 100;
  return (
    <div className="flex flex-col gap-2">
      <div
        role="img"
        aria-label={t("trainingFit.barLabel", { verdict: label })}
        className="relative h-2 w-full"
      >
        <div className="flex h-full w-full overflow-hidden rounded-full bg-muted">
          {geometry ? (
            segments.map((segment) => (
              <div
                key={segment.part}
                data-part={segment.part}
                className="h-full"
                style={{
                  width: `${segment.pct}%`,
                  minWidth: MIN_SEGMENT_PX,
                  backgroundColor: PART_COLORS[segment.part],
                }}
              />
            ))
          ) : (
            <div
              className="h-full"
              style={{
                width: `${scale > 0 ? (total / scale) * 100 : 0}%`,
                backgroundColor: PART_COLORS.modelWeights,
              }}
            />
          )}
        </div>
        {/* The free-memory line, drawn only when the run overflows it: everything right of it
            is memory the target does not have. */}
        {capacityPct < 100 ? (
          <div
            aria-hidden="true"
            className="absolute -inset-y-0.5 w-0.5 rounded-full bg-destructive"
            style={{ left: `${capacityPct}%` }}
          />
        ) : null}
      </div>
      {segments.length > 0 ? (
        <ul className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-ui-10p5 text-muted-foreground/80">
          {segments.map((segment) => (
            <li key={segment.part} className="flex min-w-0 items-center gap-1.5">
              <span
                aria-hidden="true"
                className="size-1.5 shrink-0 rounded-full"
                style={{ backgroundColor: PART_COLORS[segment.part] }}
              />
              <span className="truncate">{t(PART_LABEL_KEYS[segment.part])}</span>
              <span className="ml-auto shrink-0 font-mono">
                {formatFitGiB(segment.gb)}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function FitSuggestion({
  estimate,
}: {
  estimate: TrainingFitEstimate;
}): ReactElement | null {
  const t = useT();
  const { trainingMethod, batchSize, setTrainingMethod, setBatchSize } =
    useTrainingConfigStore(
      useShallow((s) => ({
        trainingMethod: s.trainingMethod,
        batchSize: s.batchSize,
        setTrainingMethod: s.setTrainingMethod,
        setBatchSize: s.setBatchSize,
      })),
    );
  const suggestion = estimate.suggestion;
  if (!suggestion) {
    return null;
  }
  const undoLabel = t("shell.sections.undo");

  // Never switched for the user: the change is a button press, and the toast puts it back.
  if (suggestion.kind === "qlora") {
    if (trainingMethod === "qlora") return null;
    const switchToQlora = () => {
      const previous = trainingMethod;
      setTrainingMethod("qlora");
      toast.success(t("trainingFit.switchedToQlora"), {
        action: {
          label: undoLabel,
          onClick: () => {
            const state = useTrainingConfigStore.getState();
            if (state.trainingMethod === "qlora") {
              state.setTrainingMethod(previous);
            }
          },
        },
      });
    };
    return (
      <div className="flex items-center justify-between gap-2">
        <p className="min-w-0 text-ui-10p5 leading-relaxed text-foreground/80">
          {t("trainingFit.suggestQlora", {
            required: formatFitGiB(suggestion.requiredGb),
          })}
        </p>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="h-7 shrink-0 px-2.5 text-ui-11p5"
          onClick={switchToQlora}
        >
          {t("trainingFit.switchToQlora")}
        </Button>
      </div>
    );
  }

  if (batchSize <= 1) return null;
  const applyBatchOne = () => {
    const previous = batchSize;
    setBatchSize(1);
    toast.success(t("trainingFit.batchSetToOne"), {
      action: {
        label: undoLabel,
        onClick: () => {
          const state = useTrainingConfigStore.getState();
          if (state.batchSize === 1) {
            state.setBatchSize(previous);
          }
        },
      },
    });
  };
  return (
    <div className="flex items-center justify-between gap-2">
      <p className="min-w-0 text-ui-10p5 leading-relaxed text-foreground/80">
        {t("trainingFit.suggestBatch", {
          required: formatFitGiB(suggestion.requiredGb),
        })}
      </p>
      <Button
        type="button"
        size="sm"
        variant="outline"
        className="h-7 shrink-0 px-2.5 text-ui-11p5"
        onClick={applyBatchOne}
      >
        {t("trainingFit.useBatchOne")}
      </Button>
    </div>
  );
}

/**
 * The run preview's memory plan: which GPUs the run targets, how the estimate stacks up against
 * their free memory, and the one cheaper setting that would fit when it does not.
 */
export function TrainingFitPanel({
  devices,
  target,
}: {
  devices: TrainingGpuDevice[];
  target: TrainingGpuTarget;
}): ReactElement | null {
  const t = useT();
  const lineText = useTrainingFitLineText();
  const { status, estimate } = useTrainingFitStore(
    useShallow((s) => ({ status: s.status, estimate: s.estimate })),
  );

  const select = <GpuTargetSelect devices={devices} target={target} />;
  // Nothing to price yet, or a host (MLX, CPU) whose memory is not a VRAM ceiling.
  if (status === "idle" || estimate?.reason === "unsupported_device") {
    return select;
  }

  let body: ReactElement;
  if (status === "error" || (status !== "loading" && !estimate)) {
    body = (
      <p className="text-ui-10p5 leading-relaxed text-muted-foreground/75">
        {t("trainingFit.unknownFailed")}
      </p>
    );
  } else if (!estimate) {
    body = (
      <div className="flex flex-col gap-2" aria-busy="true">
        <Skeleton className="h-2 w-full rounded-full" />
        <p className="text-ui-10p5 text-muted-foreground/75">
          {t("trainingFit.estimating")}
        </p>
      </div>
    );
  } else {
    const text = lineText(estimate);
    const sized = estimate.verdict !== "unknown";
    body = (
      <div
        className={cn(
          "flex flex-col gap-2 transition-opacity",
          status === "loading" && "opacity-60",
        )}
        aria-busy={status === "loading" || undefined}
      >
        {sized ? <FitBar estimate={estimate} label={text} /> : null}
        <p
          role="status"
          className={cn(
            "text-ui-10p5 leading-relaxed",
            estimate.verdict === "exceeds"
              ? "text-destructive"
              : estimate.verdict === "tight"
                ? "text-status-warning"
                : sized
                  ? "text-foreground/80"
                  : "text-muted-foreground/75",
          )}
        >
          {text}
        </p>
        {status === "ready" ? <FitSuggestion estimate={estimate} /> : null}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {select}
      {body}
    </div>
  );
}
