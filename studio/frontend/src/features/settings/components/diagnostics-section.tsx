// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { type TranslationKey, useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import {
  ArrowUpRight01Icon,
  Copy01Icon,
  Download01Icon,
  MessageNotification01Icon,
  Refresh01Icon,
  Shield01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useCallback, useEffect, useState } from "react";
import { loadDiagnostics, saveDiagnosticsBundle } from "../api/diagnostics";
import {
  type DiagnosticsFailure,
  type DiagnosticsReport,
  type DiagnosticsSection as Section,
  DiagnosticsRequestError,
  type GpuDevice,
  type GpuInfo,
  type LlamaCppInfo,
  type LoadedModel,
  type McpServerSummary,
  type OsInfo,
  type PythonInfo,
  type StorageInfo,
  type StudioInfo,
  buildIssueUrl,
  formatGiB,
  installedPackages,
  missingPackages,
} from "../lib/diagnostics";
import { SettingsSection } from "./settings-section";

type Translate = ReturnType<typeof useT>;

const FAILURE_MESSAGE: Record<DiagnosticsFailure, TranslationKey> = {
  outdated: "settings.diagnostics.outdated",
  forbidden: "settings.diagnostics.forbidden",
  failed: "settings.diagnostics.loadFailed",
};

const LOCATION_LABEL: Record<string, TranslationKey> = {
  studio_home: "settings.diagnostics.studioHome",
  hf_home: "settings.diagnostics.hfHome",
  hf_hub_cache: "settings.diagnostics.hfHubCache",
  temp: "settings.diagnostics.tempDir",
};

const UPDATE_STATE_LABEL: Record<string, TranslationKey> = {
  up_to_date: "settings.diagnostics.upToDate",
  not_checked: "settings.diagnostics.notChecked",
  disabled: "settings.diagnostics.updateChecksOff",
  user_managed: "settings.diagnostics.userManaged",
};

const MCP_STATE_LABEL: Record<string, TranslationKey> = {
  running: "settings.diagnostics.state.running",
  idle: "settings.diagnostics.state.idle",
  stopped: "settings.diagnostics.state.stopped",
  failed: "settings.diagnostics.state.failed",
};

function failureOf(error: unknown): DiagnosticsFailure {
  return error instanceof DiagnosticsRequestError ? error.failure : "failed";
}

function formatTime(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function Mono({ children }: { children: ReactNode }) {
  return (
    <code className="break-all font-mono text-ui-12 text-foreground">
      {children}
    </code>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] gap-x-3 py-0.5 text-xs">
      <dt className="text-muted-foreground [overflow-wrap:anywhere]">{label}</dt>
      <dd className="min-w-0 text-foreground">{children ?? "—"}</dd>
    </div>
  );
}

function Group<T>({
  title,
  section,
  wide,
  render,
}: {
  title: string;
  section: Section<T>;
  wide?: boolean;
  render: (data: T) => ReactNode;
}) {
  const t = useT();
  return (
    <div
      data-testid="diagnostics-group"
      className={`flex min-w-0 flex-col gap-1 rounded-xl border border-border/60 p-3 ${wide ? "sm:col-span-2" : ""}`}
    >
      <h3 className="text-xs font-semibold text-foreground">{title}</h3>
      {section.status === "ok" ? (
        <dl className="flex flex-col">{render(section.data)}</dl>
      ) : (
        <p className="text-xs text-muted-foreground">
          {t("settings.diagnostics.unavailable", { reason: section.reason })}
        </p>
      )}
    </div>
  );
}

function StudioFields({ data, t }: { data: StudioInfo; t: Translate }) {
  const checkout = data.source_checkout;
  return (
    <>
      <Field label={t("settings.diagnostics.studioVersion")}>
        <Mono>{data.studio_version ?? "—"}</Mono>
      </Field>
      <Field label={t("settings.diagnostics.unslothPackage")}>
        <Mono>{data.unsloth_version ?? "—"}</Mono>
      </Field>
      <Field label={t("settings.diagnostics.installType")}>
        <Mono>{data.install_source ?? "—"}</Mono>
      </Field>
      {checkout ? (
        <Field label={t("settings.diagnostics.sourceCheckout")}>
          <Mono>
            {checkout.branch ?? "?"} @ {checkout.commit ?? "?"}
          </Mono>
          {checkout.dirty ? (
            <span className="ml-1.5 text-muted-foreground">
              ({t("settings.diagnostics.modified")})
            </span>
          ) : null}
        </Field>
      ) : null}
      {data.frontend_build?.built_at ? (
        <Field label={t("settings.diagnostics.frontendBuild")}>
          {formatTime(data.frontend_build.built_at)}
        </Field>
      ) : null}
    </>
  );
}

function SystemFields({ data, t }: { data: OsInfo; t: Translate }) {
  const name = [data.name ?? data.system, data.edition, data.display_version]
    .filter(Boolean)
    .join(" ");
  const total = formatGiB(data.memory?.total_bytes);
  const available = formatGiB(data.memory?.available_bytes);
  return (
    <>
      <Field label={t("settings.diagnostics.os")}>
        {name || "—"}
        {data.build ? (
          <span className="text-muted-foreground"> · {data.build}</span>
        ) : null}
      </Field>
      <Field label={t("settings.diagnostics.cpu")}>
        {data.cpu?.name ?? "—"}
        {data.cpu?.physical_cores ? (
          <span className="text-muted-foreground">
            {" · "}
            {t("settings.diagnostics.cpuCores", {
              cores: String(data.cpu.physical_cores),
              threads: String(data.cpu.logical_cores ?? "?"),
            })}
          </span>
        ) : null}
      </Field>
      <Field label={t("settings.diagnostics.memory")}>
        {total && available
          ? t("settings.diagnostics.freeOfTotal", { free: available, total })
          : "—"}
      </Field>
    </>
  );
}

function pcieText(device: GpuDevice, t: Translate): string | null {
  if (device.pcie_gen_current == null && device.pcie_width_current == null) {
    return null;
  }
  return t("settings.diagnostics.pcieValue", {
    gen: String(device.pcie_gen_current ?? "?"),
    width: String(device.pcie_width_current ?? "?"),
    maxGen: String(device.pcie_gen_max ?? "?"),
    maxWidth: String(device.pcie_width_max ?? "?"),
  });
}

function GpuFields({ data, t }: { data: GpuInfo; t: Translate }) {
  const devices = data.devices ?? [];
  return (
    <>
      <Field label={t("settings.diagnostics.driver")}>
        <Mono>{data.driver_version ?? "—"}</Mono>
      </Field>
      <Field label={t("settings.diagnostics.cudaDriver")}>
        <Mono>{data.cuda_driver_version ?? "—"}</Mono>
      </Field>
      {data.cuda_visible_devices != null ? (
        <Field label="CUDA_VISIBLE_DEVICES">
          <Mono>{data.cuda_visible_devices}</Mono>
        </Field>
      ) : null}
      {devices.map((device) => {
        const used = formatGiB(device.memory_used_bytes);
        const free = formatGiB(device.memory_free_bytes);
        const total = formatGiB(device.memory_total_bytes);
        const pcie = pcieText(device, t);
        return (
          <Field
            key={device.index}
            label={t("settings.diagnostics.gpu", { index: String(device.index) })}
          >
            <span className="flex flex-col">
              <span>{device.name ?? "—"}</span>
              {used && free && total ? (
                <span className="text-muted-foreground">
                  {t("settings.diagnostics.vramValue", { used, free, total })}
                </span>
              ) : null}
              <span className="text-muted-foreground">
                {[
                  device.compute_capability
                    ? t("settings.diagnostics.computeCapability", {
                        value: device.compute_capability,
                      })
                    : null,
                  pcie,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            </span>
          </Field>
        );
      })}
    </>
  );
}

function PythonFields({ data, t }: { data: PythonInfo; t: Translate }) {
  const torchVersion = data.packages?.torch;
  const runtime = data.torch?.cuda
    ? `CUDA ${data.torch.cuda}`
    : data.torch?.hip
      ? `ROCm ${data.torch.hip}`
      : t("settings.diagnostics.cpuOnly");
  const packages = installedPackages(data.packages, ["torch"]);
  const missing = missingPackages(data.packages);
  return (
    <>
      <Field label={t("settings.diagnostics.python")}>
        <Mono>{data.version ?? "—"}</Mono>
      </Field>
      <Field label={t("settings.diagnostics.torch")}>
        {torchVersion ? (
          <>
            <Mono>{torchVersion}</Mono>
            <span className="text-muted-foreground">
              {" · "}
              {t("settings.diagnostics.builtWith", { runtime })}
            </span>
          </>
        ) : (
          "—"
        )}
      </Field>
      <Field label={t("settings.diagnostics.packages")}>
        <span className="flex flex-wrap gap-1">
          {packages.map((pkg) => (
            <span
              key={pkg.name}
              className="rounded-md bg-muted px-1.5 py-px font-mono text-ui-11"
            >
              {pkg.name} {pkg.version}
            </span>
          ))}
        </span>
        {missing.length > 0 ? (
          <span className="mt-1 block text-muted-foreground">
            {t("settings.diagnostics.notInstalled", {
              names: missing.join(", "),
            })}
          </span>
        ) : null}
      </Field>
    </>
  );
}

function LlamaFields({ data, t }: { data: LlamaCppInfo; t: Translate }) {
  const update = data.update;
  const updateLabel =
    update?.state === "available"
      ? t("settings.diagnostics.updateAvailable", {
          latest: update.latest ?? "?",
        })
      : update?.state && UPDATE_STATE_LABEL[update.state]
        ? t(UPDATE_STATE_LABEL[update.state])
        : (update?.state ?? "—");
  return (
    <>
      <Field label={t("settings.diagnostics.version")}>
        <Mono>{data.version ?? "—"}</Mono>
        {data.commit ? (
          <span className="text-muted-foreground"> · {data.commit}</span>
        ) : null}
      </Field>
      <Field label={t("settings.diagnostics.backend")}>
        <Mono>
          {[data.backend, data.bundle].filter(Boolean).join(" · ") || "—"}
        </Mono>
      </Field>
      <Field label={t("settings.diagnostics.binary")}>
        <Mono>{data.binary ?? "—"}</Mono>
      </Field>
      <Field label={t("settings.diagnostics.update")}>{updateLabel}</Field>
    </>
  );
}

function StorageFields({ data, t }: { data: StorageInfo; t: Translate }) {
  const size = data.hf_cache_size;
  const sizeText =
    typeof size?.bytes === "number"
      ? formatGiB(size.bytes)
      : size?.state === "missing"
        ? t("settings.diagnostics.missingFolder")
        : t("settings.diagnostics.measuring");
  return (
    <>
      {(data.locations ?? []).map((location) => {
        const free = formatGiB(location.free_bytes);
        const total = formatGiB(location.total_bytes);
        const label = LOCATION_LABEL[location.key];
        return (
          <Field
            key={location.key}
            label={label ? t(label) : location.key}
          >
            <span className="flex flex-col">
              <Mono>{location.path ?? "—"}</Mono>
              <span className="text-muted-foreground">
                {[
                  location.drive,
                  location.exists === false
                    ? t("settings.diagnostics.missingFolder")
                    : null,
                  free && total
                    ? t("settings.diagnostics.freeOfTotal", { free, total })
                    : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            </span>
          </Field>
        );
      })}
      <Field label={t("settings.diagnostics.hfCacheSize")}>{sizeText}</Field>
    </>
  );
}

function ModelRows({ models, t }: { models: LoadedModel[]; t: Translate }) {
  if (models.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {t("settings.diagnostics.noModels")}
      </p>
    );
  }
  return (
    <>
      {models.map((model, index) => {
        const gpus = (model.gpu_ids ?? []).join(", ");
        const placement = gpus
          ? t("settings.diagnostics.gpu", { index: gpus })
          : (model.device ?? null);
        const vram = model.vram_bytes ? formatGiB(model.vram_bytes) : null;
        return (
          <Field
            // biome-ignore lint/suspicious/noArrayIndexKey: model rows have no stable id here.
            key={index}
            label={[model.kind, model.engine].filter(Boolean).join(" · ")}
          >
            <Mono>
              {model.name ?? "—"}
              {model.variant ? ` (${model.variant})` : ""}
            </Mono>
            {placement || vram ? (
              <span className="text-muted-foreground">
                {" · "}
                {[placement, vram].filter(Boolean).join(" · ")}
              </span>
            ) : null}
          </Field>
        );
      })}
    </>
  );
}

function McpRows({ servers, t }: { servers: McpServerSummary[]; t: Translate }) {
  if (servers.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {t("settings.diagnostics.noMcp")}
      </p>
    );
  }
  return (
    <>
      {servers.map((server, index) => {
        const details = [
          server.transport === "local"
            ? t("settings.diagnostics.localProgram")
            : t("settings.diagnostics.remoteServer"),
          server.process_mode === "shared"
            ? t("settings.diagnostics.sharedProcess")
            : server.process_mode === "per_chat"
              ? t("settings.diagnostics.perChatProcess")
              : null,
          server.state && MCP_STATE_LABEL[server.state]
            ? t(MCP_STATE_LABEL[server.state])
            : null,
          server.enabled === false ? t("settings.diagnostics.disabled") : null,
        ];
        return (
          // biome-ignore lint/suspicious/noArrayIndexKey: the summary carries no server id on purpose.
          <Field key={index} label={server.name || "—"}>
            {details.filter(Boolean).join(" · ")}
          </Field>
        );
      })}
    </>
  );
}

function EnvironmentRows({
  variables,
  t,
}: {
  variables: { name: string; value: string; redacted: boolean }[];
  t: Translate;
}) {
  if (variables.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {t("settings.diagnostics.noEnvironment")}
      </p>
    );
  }
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-muted-foreground">
        {t("settings.diagnostics.variableCount", {
          count: String(variables.length),
        })}
      </summary>
      <div className="mt-1 flex flex-col">
        {variables.map((variable) => (
          <Field key={variable.name} label={variable.name}>
            {variable.redacted ? (
              <span className="italic text-muted-foreground">
                {t("settings.diagnostics.redacted")}
              </span>
            ) : (
              <Mono>{variable.value}</Mono>
            )}
          </Field>
        ))}
      </div>
    </details>
  );
}

/** The grouped report. Exported for its test; the section below is the only caller. */
export function ReportGroups({
  report,
  t,
}: {
  report: DiagnosticsReport;
  t: Translate;
}) {
  const { sections } = report;
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      <Group
        title={t("settings.diagnostics.groups.studio")}
        section={sections.studio}
        render={(data) => <StudioFields data={data} t={t} />}
      />
      <Group
        title={t("settings.diagnostics.groups.system")}
        section={sections.os}
        render={(data) => <SystemFields data={data} t={t} />}
      />
      <Group
        title={t("settings.diagnostics.groups.gpus")}
        section={sections.gpus}
        render={(data) => <GpuFields data={data} t={t} />}
      />
      <Group
        title={t("settings.diagnostics.groups.llamaCpp")}
        section={sections.llama_cpp}
        render={(data) => <LlamaFields data={data} t={t} />}
      />
      <Group
        wide={true}
        title={t("settings.diagnostics.groups.python")}
        section={sections.python}
        render={(data) => <PythonFields data={data} t={t} />}
      />
      <Group
        wide={true}
        title={t("settings.diagnostics.groups.storage")}
        section={sections.storage}
        render={(data) => <StorageFields data={data} t={t} />}
      />
      <Group
        title={t("settings.diagnostics.groups.models")}
        section={sections.models}
        render={(data) => <ModelRows models={data.models ?? []} t={t} />}
      />
      <Group
        title={t("settings.diagnostics.groups.mcp")}
        section={sections.mcp}
        render={(data) => <McpRows servers={data.servers ?? []} t={t} />}
      />
      <Group
        wide={true}
        title={t("settings.diagnostics.groups.environment")}
        section={sections.environment}
        render={(data) => (
          <EnvironmentRows variables={data.variables ?? []} t={t} />
        )}
      />
    </div>
  );
}

/** Settings > Logs > Diagnostics: the environment report, its Markdown, and the bundle. */
export function DiagnosticsSection() {
  const t = useT();
  const [report, setReport] = useState<DiagnosticsReport | null>(null);
  const [failure, setFailure] = useState<DiagnosticsFailure | null>(null);
  const [loading, setLoading] = useState(true);
  const [reloadKey, setReloadKey] = useState(0);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    loadDiagnostics(controller.signal).then(
      (next) => {
        setReport(next);
        setFailure(null);
        setLoading(false);
      },
      (error: unknown) => {
        if (controller.signal.aborted) return;
        setFailure(failureOf(error));
        setLoading(false);
      },
    );
    return () => controller.abort();
  }, [reloadKey]);

  const refresh = useCallback(() => {
    setLoading(true);
    setReloadKey((key) => key + 1);
  }, []);

  const copyForIssue = useCallback(async () => {
    if (!report?.markdown) return;
    if (await copyToClipboard(report.markdown)) {
      toast.success(t("settings.diagnostics.copied"));
    } else {
      toast.error(t("settings.diagnostics.copyFailed"));
    }
  }, [report, t]);

  const saveBundle = useCallback(async () => {
    setSaving(true);
    try {
      if (await saveDiagnosticsBundle()) {
        toast.success(t("settings.diagnostics.bundleSaved"));
      }
    } catch (error) {
      const reason = failureOf(error);
      toast.error(
        t(
          reason === "failed"
            ? "settings.diagnostics.bundleFailed"
            : FAILURE_MESSAGE[reason],
        ),
        reason === "failed"
          ? { description: (error as Error).message }
          : undefined,
      );
    } finally {
      setSaving(false);
    }
  }, [t]);

  return (
    <SettingsSection
      title={t("settings.diagnostics.title")}
      description={t("settings.diagnostics.description")}
      action={
        <Button
          size="sm"
          variant="ghost"
          data-testid="diagnostics-refresh"
          aria-busy={loading}
          disabled={loading}
          onClick={refresh}
        >
          <HugeiconsIcon strokeWidth={1.75} icon={Refresh01Icon} />
          {t("settings.diagnostics.refresh")}
        </Button>
      }
    >
      <div className="flex flex-col gap-3 py-2">
        {report ? (
          <>
            <ReportGroups report={report} t={t} />
            {report.generatedAt ? (
              <p className="text-ui-11 text-muted-foreground">
                {t("settings.diagnostics.collectedAt", {
                  time: formatTime(report.generatedAt) ?? report.generatedAt,
                })}
              </p>
            ) : null}
          </>
        ) : loading ? (
          <p className="text-xs text-muted-foreground">
            {t("settings.diagnostics.loading")}
          </p>
        ) : null}
        {failure && !loading ? (
          <p
            role="alert"
            className="text-xs text-destructive"
            data-testid="diagnostics-error"
          >
            {t(FAILURE_MESSAGE[failure])}
          </p>
        ) : null}

        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            data-testid="diagnostics-copy"
            disabled={!report?.markdown}
            onClick={() => void copyForIssue()}
          >
            <HugeiconsIcon strokeWidth={1.75} icon={Copy01Icon} />
            {t("settings.diagnostics.copyForIssue")}
          </Button>
          <Button
            size="sm"
            variant="outline"
            data-testid="diagnostics-save-bundle"
            aria-busy={saving}
            disabled={saving}
            onClick={() => void saveBundle()}
          >
            <HugeiconsIcon strokeWidth={1.75} icon={Download01Icon} />
            {saving
              ? t("settings.diagnostics.savingBundle")
              : t("settings.diagnostics.saveBundle")}
          </Button>
          <Button size="sm" variant="ghost" asChild={true}>
            <a
              href={buildIssueUrl()}
              target="_blank"
              rel="noopener noreferrer"
              data-testid="diagnostics-report-issue"
              title={t("settings.diagnostics.reportIssueHint")}
            >
              <HugeiconsIcon
                strokeWidth={1.75}
                icon={MessageNotification01Icon}
              />
              {t("settings.diagnostics.reportIssue")}
              <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
            </a>
          </Button>
        </div>

        <div className="flex gap-2 text-xs text-muted-foreground">
          <HugeiconsIcon
            strokeWidth={1.75}
            icon={Shield01Icon}
            className="mt-0.5 size-3.5 shrink-0"
          />
          <div className="flex flex-col gap-1">
            <p>{t("settings.diagnostics.privacyIncluded")}</p>
            <p>{t("settings.diagnostics.privacyExcluded")}</p>
            <p>{t("settings.diagnostics.reportIssueHint")}</p>
          </div>
        </div>
      </div>
    </SettingsSection>
  );
}
