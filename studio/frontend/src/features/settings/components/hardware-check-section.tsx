// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import {
  HARDWARE_CHECK_OPTIONS,
  type HardwareCheckFinding,
  type HardwareCheckOption,
  type HardwareCheckStatus,
  findingMessage,
  formatGibs,
  formatLink,
  formatLinkMax,
  locationKey,
  optionsToApply,
  peerAccess,
  skipReasonKey,
  useHardwareCheckStore,
} from "@/features/hardware-check";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useEffect } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsGroupDivider, SettingsSection } from "./settings-section";

type Translate = ReturnType<typeof useT>;

const OPTION_KEYS: Record<HardwareCheckOption, { label: TranslationKey; description: TranslationKey }> = {
  prefer_fast_link: {
    label: "hardwareCheck.options.preferFastLink.label",
    description: "hardwareCheck.options.preferFastLink.description",
  },
  avoid_tensor_split: {
    label: "hardwareCheck.options.avoidTensorSplit.label",
    description: "hardwareCheck.options.avoidTensorSplit.description",
  },
  warn_training_slow_link: {
    label: "hardwareCheck.options.warnTraining.label",
    description: "hardwareCheck.options.warnTraining.description",
  },
};

const PHASE_KEYS: Record<string, TranslationKey> = {
  starting: "hardwareCheck.phase.starting",
  gpus: "hardwareCheck.phase.gpus",
  storage: "hardwareCheck.phase.storage",
};

const SEVERITY_CLASS: Record<HardwareCheckFinding["severity"], string> = {
  warning: "bg-amber-500",
  info: "bg-sky-500",
  ok: "bg-emerald-500",
};

/** `hardwareCheck.<key>` for a key the pure model returns relative to the section. */
function hcKey(key: string): TranslationKey {
  return `hardwareCheck.${key}` as TranslationKey;
}

function formatTime(iso: string | null): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function findingText(t: Translate, finding: HardwareCheckFinding): string {
  const message = findingMessage(finding);
  if (!message) return finding.text;
  const values: Record<string, string | number> = { ...message.values };
  if (typeof values.location === "string") {
    values.location = t(hcKey(locationKey(values.location)));
  }
  return t(hcKey(message.key), values);
}

function StatusLine({ status, t }: { status: HardwareCheckStatus | null; t: Translate }) {
  if (!status) return null;
  const { state, result, upToDate } = status;
  if (state.running) {
    return (
      <div className="flex flex-col gap-1.5 py-2" aria-live="polite">
        <span className="text-xs text-muted-foreground">
          {t(PHASE_KEYS[state.phase ?? ""] ?? "hardwareCheck.running")}
        </span>
        <Progress
          value={Math.round(state.progress * 100)}
          aria-label={t("hardwareCheck.running")}
          className="h-1.5 w-full max-w-[calc(392px*var(--ui-space-scale,1))] rounded-full bg-muted"
        />
      </div>
    );
  }
  const skipKey = skipReasonKey(state.lastSkip?.reason);
  return (
    <div className="flex flex-col gap-0.5 py-2 text-xs" aria-live="polite">
      <span className="text-muted-foreground">
        {result?.finishedAt
          ? t("hardwareCheck.lastChecked", { time: formatTime(result.finishedAt) ?? "" })
          : t("hardwareCheck.neverChecked")}
      </span>
      {result && upToDate === false ? (
        <span className="text-amber-700 dark:text-amber-400">{t("hardwareCheck.outOfDate")}</span>
      ) : null}
      {result && upToDate === null ? (
        <span className="text-muted-foreground">{t("hardwareCheck.currencyUnknown")}</span>
      ) : null}
      {skipKey ? <span className="text-muted-foreground">{t(hcKey(skipKey))}</span> : null}
      {state.lastError ? (
        <span className="text-destructive">
          {t("hardwareCheck.runError", { error: state.lastError })}
        </span>
      ) : null}
    </div>
  );
}

function GpuTable({ status, t }: { status: HardwareCheckStatus; t: Translate }) {
  const result = status.result;
  if (!result || result.gpus.length === 0) return null;
  const slow = new Set(result.slowGpus);
  return (
    <div className="flex flex-col gap-1 py-2">
      <h3 className="text-xs font-semibold text-foreground">{t("hardwareCheck.gpus.heading")}</h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[calc(30rem*var(--ui-space-scale,1))] text-left text-xs tabular-nums">
          <thead className="text-muted-foreground">
            <tr>
              <th className="py-1 pr-3 font-medium">{t("hardwareCheck.gpus.gpu")}</th>
              <th className="py-1 pr-3 font-medium">{t("hardwareCheck.gpus.linkNow")}</th>
              <th className="py-1 pr-3 font-medium">{t("hardwareCheck.gpus.linkMax")}</th>
              <th className="py-1 pr-3 font-medium">{t("hardwareCheck.gpus.toGpu")}</th>
              <th className="py-1 font-medium">{t("hardwareCheck.gpus.fromGpu")}</th>
            </tr>
          </thead>
          <tbody>
            {result.gpus.map((gpu) => {
              const measured = gpu.status === "measured";
              return (
                <tr key={gpu.index} className="border-t border-border/50">
                  <td className="py-1 pr-3">
                    <span className="font-medium text-foreground">{gpu.index}</span>
                    {gpu.name ? (
                      <span className="ml-1.5 text-muted-foreground">{gpu.name}</span>
                    ) : null}
                    {slow.has(gpu.index) ? (
                      <span className="ml-1.5 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-ui-10 font-semibold text-amber-700 dark:text-amber-400">
                        {t("hardwareCheck.gpus.slow")}
                      </span>
                    ) : null}
                  </td>
                  <td className="py-1 pr-3 font-mono">
                    {measured ? formatLink(gpu.link) : t("hardwareCheck.gpus.notMeasured")}
                  </td>
                  <td className="py-1 pr-3 font-mono">{formatLinkMax(gpu.link ?? gpu.linkIdle)}</td>
                  <td className="py-1 pr-3 font-mono">
                    {measured ? t("hardwareCheck.gpus.gibs", { value: formatGibs(gpu.h2dGibs) }) : "—"}
                  </td>
                  <td className="py-1 font-mono">
                    {measured ? t("hardwareCheck.gpus.gibs", { value: formatGibs(gpu.d2hGibs) }) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {result.pairs.length > 0 ? (
        <ul className="flex flex-col gap-0.5 pt-1 text-xs text-muted-foreground">
          {result.pairs.map((pair) => {
            const access = peerAccess(pair);
            const copies = [pair.copyAbGibs, pair.copyBaGibs].filter(
              (v): v is number => typeof v === "number",
            );
            return (
              <li key={`${pair.a}-${pair.b}`} className="tabular-nums">
                <span className="text-foreground">
                  {t("hardwareCheck.pairs.pair", { a: pair.a, b: pair.b })}
                </span>
                {" · "}
                {t("hardwareCheck.pairs.direct")}:{" "}
                {t(
                  access === "yes"
                    ? "hardwareCheck.pairs.yes"
                    : access === "no"
                      ? "hardwareCheck.pairs.no"
                      : "hardwareCheck.pairs.unknown",
                )}
                {copies.length > 0 ? (
                  <>
                    {" · "}
                    {t("hardwareCheck.pairs.copy")}:{" "}
                    {t("hardwareCheck.gpus.gibs", { value: formatGibs(Math.min(...copies)) })}
                  </>
                ) : null}
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}

function StorageTable({ status, t }: { status: HardwareCheckStatus; t: Translate }) {
  const storage = status.result?.storage ?? [];
  if (storage.length === 0) return null;
  const unknown = t("hardwareCheck.storage.unknown");
  return (
    <div className="flex flex-col gap-1 py-2">
      <h3 className="text-xs font-semibold text-foreground">{t("hardwareCheck.storage.heading")}</h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[calc(24rem*var(--ui-space-scale,1))] text-left text-xs">
          <thead className="text-muted-foreground">
            <tr>
              <th className="py-1 pr-3 font-medium">{t("hardwareCheck.storage.location")}</th>
              <th className="py-1 pr-3 font-medium">{t("hardwareCheck.storage.drive")}</th>
              <th className="py-1 pr-3 font-medium">{t("hardwareCheck.storage.bus")}</th>
              <th className="py-1 font-medium">{t("hardwareCheck.storage.media")}</th>
            </tr>
          </thead>
          <tbody>
            {storage.map((entry) => (
              <tr key={entry.key} className="border-t border-border/50">
                <td className="py-1 pr-3 text-foreground">{t(hcKey(locationKey(entry.key)))}</td>
                <td className="py-1 pr-3 font-mono">{entry.drive ?? "—"}</td>
                <td className="py-1 pr-3" title={entry.model ?? undefined}>
                  {entry.busType ?? unknown}
                </td>
                <td className="py-1">{entry.mediaType ?? unknown}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Findings({ status, t }: { status: HardwareCheckStatus; t: Translate }) {
  const findings = status.result?.findings ?? [];
  if (findings.length === 0) return null;
  return (
    <div className="flex flex-col gap-1 py-2">
      <h3 className="text-xs font-semibold text-foreground">{t("hardwareCheck.findings.heading")}</h3>
      <ul className="flex flex-col gap-1">
        {findings.map((finding, index) => (
          <li
            key={`${finding.id}-${index}`}
            data-finding={finding.id}
            className="flex items-start gap-2 text-xs leading-snug text-foreground/85"
          >
            <span
              aria-hidden="true"
              className={cn("mt-1 size-2 shrink-0 rounded-full", SEVERITY_CLASS[finding.severity])}
            />
            <span>{findingText(t, finding)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Settings > Resources > Hardware check: the last result, its findings, the options it
 * recommends (all off until switched on) and a Run button. Owner only, like the tab.
 */
export function HardwareCheckSection() {
  const t = useT();
  const status = useHardwareCheckStore((s) => s.status);
  const loaded = useHardwareCheckStore((s) => s.loaded);
  const absent = useHardwareCheckStore((s) => s.absent);
  const error = useHardwareCheckStore((s) => s.error);
  const busy = useHardwareCheckStore((s) => s.busy);

  useEffect(() => {
    void useHardwareCheckStore.getState().refresh();
  }, []);

  if (absent) return null;

  const running = status?.state.running === true;
  const toApply = optionsToApply(status);
  const recommended = status?.result?.recommended;

  const run = async () => {
    const outcome = await useHardwareCheckStore.getState().run();
    if (outcome && !outcome.started) {
      const key = skipReasonKey(outcome.reason);
      toast.info(t(hcKey(key ?? "skip.other")));
    }
  };

  const apply = async () => {
    const applied = await useHardwareCheckStore.getState().applyRecommended();
    if (applied === null) {
      toast.error(t("hardwareCheck.options.saveError"));
    } else if (applied.length > 0) {
      toast.success(t("hardwareCheck.options.applied"));
    } else {
      toast.info(t("hardwareCheck.options.nothingToApply"));
    }
  };

  const setSetting = async (key: HardwareCheckOption | "auto_run", value: boolean) => {
    const saved = await useHardwareCheckStore.getState().setSetting(key, value);
    if (!saved) toast.error(t("hardwareCheck.options.saveError"));
  };

  return (
    <SettingsSection
      title={t("hardwareCheck.title")}
      description={t("hardwareCheck.description")}
      action={
        <Button
          variant="outline"
          size="sm"
          className="h-8 shrink-0"
          disabled={!loaded || running || busy}
          onClick={() => void run()}
        >
          {running ? t("hardwareCheck.running") : t("hardwareCheck.run")}
        </Button>
      }
    >
      {!status && loaded && error ? (
        <p className="py-2 text-xs text-destructive">{t("hardwareCheck.loadError")}</p>
      ) : null}
      <StatusLine status={status} t={t} />
      {status ? (
        <>
          <Findings status={status} t={t} />
          <GpuTable status={status} t={t} />
          <StorageTable status={status} t={t} />
        </>
      ) : null}
      <SettingsGroupDivider />
      <div className="flex items-center justify-between gap-3 pt-2">
        <h3 className="text-xs font-semibold text-foreground">{t("hardwareCheck.options.heading")}</h3>
        <Button
          variant="outline"
          size="sm"
          className="h-7 shrink-0 text-ui-11p5"
          disabled={!status || busy || toApply.length === 0}
          onClick={() => void apply()}
        >
          {t("hardwareCheck.options.applyRecommended")}
        </Button>
      </div>
      {HARDWARE_CHECK_OPTIONS.map((key) => (
        <SettingsRow
          key={key}
          label={t(OPTION_KEYS[key].label)}
          description={
            <>
              {t(OPTION_KEYS[key].description)}
              {recommended?.[key] ? (
                <span className="ml-1.5 rounded-full bg-control-accent/10 px-1.5 py-0.5 text-ui-10 font-semibold text-control-accent">
                  {t("hardwareCheck.options.recommended")}
                </span>
              ) : null}
            </>
          }
        >
          <Switch
            aria-label={t(OPTION_KEYS[key].label)}
            checked={status?.settings[key] ?? false}
            disabled={!status || busy}
            onCheckedChange={(value) => void setSetting(key, value)}
          />
        </SettingsRow>
      ))}
      <SettingsRow
        label={t("hardwareCheck.autoRun.label")}
        description={t("hardwareCheck.autoRun.description")}
      >
        <Switch
          aria-label={t("hardwareCheck.autoRun.label")}
          checked={status?.settings.auto_run ?? true}
          disabled={!status || busy}
          onCheckedChange={(value) => void setSetting("auto_run", value)}
        />
      </SettingsRow>
    </SettingsSection>
  );
}
