// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Logs > Diagnostics: the report view, the bundle and the "Report an issue" link.
//
// The report is made to be handed to someone else, so the things asserted here are the
// ones that keep that safe: the issue URL carries a fixed title and nothing from the
// report, a redacted environment value is never rendered, and a section the backend could
// not answer reads as unavailable rather than breaking the view.

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";

import type * as DiagnosticsApi from "../src/features/settings/api/diagnostics.ts";
import type * as DiagnosticsSectionModule from "../src/features/settings/components/diagnostics-section.tsx";
import * as diagnosticsLib from "../src/features/settings/lib/diagnostics.ts";
import { en } from "../src/i18n/locales/en.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const API_URL = new URL(
  "../src/features/settings/api/diagnostics.ts",
  import.meta.url,
);
const SECTION_URL = new URL(
  "../src/features/settings/components/diagnostics-section.tsx",
  import.meta.url,
);
const ARCHIVE_NAME = /^unsloth-diagnostics-\d{8}-\d{6}\.zip$/;
const SECRET = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345";

function t(
  key: string,
  values?: Record<string, string | number | boolean | null | undefined>,
): string {
  const message = key
    .split(".")
    .reduce<unknown>(
      (node, part) => (node as Record<string, unknown> | undefined)?.[part],
      en as unknown,
    );
  assert.equal(typeof message, "string", `no English message for "${key}"`);
  return String(message).replace(/\{(\w+)\}/g, (match, name: string) =>
    values && name in values ? String(values[name]) : match,
  );
}

const SAMPLE_BODY = {
  schema: 1,
  generated_at: "2026-10-09T15:13:27Z",
  markdown: "## Unsloth Studio diagnostics\n",
  sections: {
    studio: {
      status: "ok",
      data: {
        studio_version: "GitHub studio-ux-mcp-fixes",
        unsloth_version: "2026.9.12",
        install_source: "editable",
        source_checkout: { branch: "studio-ux-mcp-fixes", commit: "ad3d0a0923", dirty: true },
      },
    },
    os: {
      status: "ok",
      data: {
        name: "Windows 11",
        edition: "Professional",
        build: "10.0.26220.9606",
        cpu: { name: "AMD Ryzen 5 9600X", physical_cores: 6, logical_cores: 12 },
        memory: { total_bytes: 32 * 1024 ** 3, available_bytes: 8 * 1024 ** 3 },
      },
    },
    gpus: {
      status: "ok",
      data: {
        driver_version: "616.92",
        cuda_driver_version: "13.4",
        devices: [
          {
            index: 0,
            name: "NVIDIA GeForce RTX 3090",
            compute_capability: "8.6",
            memory_total_bytes: 24 * 1024 ** 3,
            memory_used_bytes: 4 * 1024 ** 3,
            memory_free_bytes: 20 * 1024 ** 3,
            pcie_gen_current: 4,
            pcie_gen_max: 4,
            pcie_width_current: 1,
            pcie_width_max: 16,
          },
        ],
      },
    },
    python: {
      status: "ok",
      data: {
        version: "3.12.10",
        torch: { cuda: "13.0" },
        packages: { torch: "2.11.0+cu130", transformers: "5.5.0", xformers: null },
      },
    },
    llama_cpp: { status: "unavailable", reason: "no llama-server binary was found" },
    storage: {
      status: "ok",
      data: {
        locations: [
          {
            key: "hf_hub_cache",
            path: "F:\\huggingface\\hub",
            exists: true,
            drive: "F:",
            free_bytes: 100 * 1024 ** 3,
            total_bytes: 1000 * 1024 ** 3,
          },
        ],
        hf_cache_size: { state: "computing" },
      },
    },
    models: { status: "ok", data: { models: [] } },
    mcp: {
      status: "ok",
      data: {
        servers: [
          { name: "GitHub tools", transport: "local", enabled: true, process_mode: "shared", state: "idle" },
        ],
      },
    },
    environment: {
      status: "ok",
      data: {
        variables: [
          { name: "HF_HOME", value: "F:\\huggingface", redacted: false },
          { name: "HF_TOKEN", value: "<redacted>", redacted: true },
        ],
      },
    },
  },
};

// ── pure helpers ───────────────────────────────────────────────────────────────

test("the issue link carries a fixed title and nothing from the report", () => {
  const url = new URL(diagnosticsLib.buildIssueUrl());
  assert.equal(url.origin, "https://github.com");
  assert.equal(url.pathname, "/mixelpixx/unsloth-desktop/issues/new");
  assert.deepEqual([...url.searchParams.keys()], ["title"]);
  assert.equal(url.searchParams.get("title"), diagnosticsLib.ISSUE_TITLE);
  assert.equal(url.hash, "");
});

test("a report missing or mangling a section reads it as unavailable", () => {
  const parsed = diagnosticsLib.parseDiagnosticsReport({
    generated_at: "2026-10-09T15:13:27Z",
    sections: { studio: { status: "ok", data: { studio_version: "dev" } }, os: { status: "ok" } },
  });
  assert.deepEqual(parsed.sections.studio, { status: "ok", data: { studio_version: "dev" } });
  assert.deepEqual(parsed.sections.os, { status: "unavailable", reason: "not reported" });
  assert.deepEqual(parsed.sections.gpus, { status: "unavailable", reason: "not reported" });
  assert.deepEqual(Object.keys(parsed.sections), [...diagnosticsLib.DIAGNOSTICS_SECTIONS]);
  assert.equal(parsed.markdown, "");
  assert.deepEqual(diagnosticsLib.parseDiagnosticsReport(null).sections.mcp, {
    status: "unavailable",
    reason: "not reported",
  });
});

test("archive names, sizes, failures and package lists format as the view expects", () => {
  assert.match(diagnosticsLib.diagnosticsArchiveFilename(), ARCHIVE_NAME);
  assert.equal(
    diagnosticsLib.diagnosticsArchiveFilename(new Date(2026, 0, 2, 3, 4, 5)),
    "unsloth-diagnostics-20260102-030405.zip",
  );
  assert.equal(diagnosticsLib.formatGiB(24 * 1024 ** 3), "24.0 GiB");
  assert.equal(diagnosticsLib.formatGiB(null), null);
  assert.equal(diagnosticsLib.formatGiB(-1), null);
  assert.equal(diagnosticsLib.diagnosticsFailureForStatus(404), "outdated");
  assert.equal(diagnosticsLib.diagnosticsFailureForStatus(403), "forbidden");
  assert.equal(diagnosticsLib.diagnosticsFailureForStatus(500), "failed");
  const packages = { torch: "2.9", transformers: "5.5", xformers: null, trl: "" };
  assert.deepEqual(diagnosticsLib.installedPackages(packages, ["torch"]), [
    { name: "transformers", version: "5.5" },
  ]);
  assert.deepEqual(diagnosticsLib.missingPackages(packages), ["xformers", "trl"]);
});

// ── the API ────────────────────────────────────────────────────────────────────

function loadApi(options: {
  respond: (path: string) => Response;
  cancelSave?: boolean;
}) {
  const requests: string[] = [];
  const saves: { filename: string; size: number }[] = [];
  class Cancelled extends Error {}
  const api = loadWithStubs<typeof DiagnosticsApi>(API_URL, {
    "@/features/auth": {
      authFetch: async (path: string) => {
        requests.push(path);
        return options.respond(path);
      },
    },
    "@/lib/format-fastapi-error": formatFastApiError,
    "@/lib/native-files": {
      downloadBlobStreaming: async (blob: Blob, filename: string) => {
        if (options.cancelSave) throw new Cancelled();
        saves.push({ filename, size: blob.size });
      },
      isDownloadCancelled: (error: unknown) => error instanceof Cancelled,
    },
    "../lib/diagnostics": diagnosticsLib,
  });
  return { api, requests, saves };
}

test("the report is read from the diagnostics route and parsed", async () => {
  const { api, requests } = loadApi({
    respond: () => Response.json(SAMPLE_BODY),
  });
  const report = await api.loadDiagnostics();
  assert.deepEqual(requests, ["/api/diagnostics"]);
  assert.equal(report.sections.gpus.status, "ok");
  assert.equal(report.markdown, SAMPLE_BODY.markdown);
});

test("a refused or missing route says which", async () => {
  for (const [status, failure] of [
    [403, "forbidden"],
    [404, "outdated"],
    [500, "failed"],
  ] as const) {
    const { api } = loadApi({
      respond: () => Response.json({ detail: "nope" }, { status }),
    });
    await assert.rejects(api.loadDiagnostics(), (error: unknown) => {
      assert.ok(error instanceof diagnosticsLib.DiagnosticsRequestError);
      assert.equal(error.failure, failure);
      return true;
    });
  }
});

test("the bundle is the logs export with the report, saved under a diagnostics name", async () => {
  const { api, requests, saves } = loadApi({
    respond: () => new Response(new Blob(["PK"])),
  });
  assert.equal(await api.saveDiagnosticsBundle(), true);
  assert.deepEqual(requests, ["/api/settings/debug/logs/export?diagnostics=true"]);
  assert.equal(saves.length, 1);
  assert.match(saves[0].filename, ARCHIVE_NAME);
});

test("cancelling the save dialog is not an error", async () => {
  const { api } = loadApi({
    respond: () => new Response(new Blob(["PK"])),
    cancelSave: true,
  });
  assert.equal(await api.saveDiagnosticsBundle(), false);
});

// ── the view ───────────────────────────────────────────────────────────────────

type Props = Record<string, unknown>;

function loadSection(options: {
  save?: () => Promise<boolean>;
  copyOk?: boolean;
}) {
  const buttons: Props[] = [];
  const toasts: { kind: string; title: string }[] = [];
  const copies: string[] = [];
  const passthrough = (props: { children?: React.ReactNode }) =>
    React.createElement(React.Fragment, null, props.children);
  const module = loadWithStubs<typeof DiagnosticsSectionModule>(SECTION_URL, {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/components/ui/button": {
      Button: (props: Props) => {
        buttons.push(props);
        return props.asChild
          ? passthrough(props as { children?: React.ReactNode })
          : React.createElement(
              "button",
              { type: "button", disabled: props.disabled, "data-testid": props["data-testid"] },
              props.children as React.ReactNode,
            );
      },
    },
    "@/i18n": { useT: () => t },
    "@/lib/copy-to-clipboard": {
      copyToClipboard: async (text: string) => {
        copies.push(text);
        return options.copyOk ?? true;
      },
    },
    "@/lib/toast": {
      toast: {
        success: (title: string) => toasts.push({ kind: "success", title }),
        error: (title: string) => toasts.push({ kind: "error", title }),
      },
    },
    "@hugeicons/core-free-icons": new Proxy({}, { get: () => ({}) }),
    "@hugeicons/react": { HugeiconsIcon: () => null },
    "../api/diagnostics": {
      loadDiagnostics: () => new Promise(() => {}),
      saveDiagnosticsBundle: options.save ?? (async () => true),
    },
    "../lib/diagnostics": diagnosticsLib,
    "./settings-section": {
      SettingsSection: (props: { title: string; action?: React.ReactNode; children?: React.ReactNode }) =>
        React.createElement("section", null, props.title, props.action, props.children),
    },
  });
  return { module, buttons, toasts, copies };
}

async function flush(): Promise<void> {
  for (let index = 0; index < 5; index += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

test("the section offers copy, the bundle, and an issue link with only a title", () => {
  const { module, buttons } = loadSection({});
  const html = renderToStaticMarkup(React.createElement(module.DiagnosticsSection));
  assert.ok(html.includes(t("settings.diagnostics.title")));
  assert.ok(html.includes(t("settings.diagnostics.loading")));
  assert.ok(html.includes(t("settings.diagnostics.privacyIncluded")));
  assert.ok(html.includes(t("settings.diagnostics.privacyExcluded").replace("<", "&lt;").replace(">", "&gt;")));
  const href = html.match(/href="([^"]+)"[^>]*data-testid="diagnostics-report-issue"/)?.[1];
  assert.equal(href?.replaceAll("&amp;", "&"), diagnosticsLib.buildIssueUrl());
  const copy = buttons.find((props) => props["data-testid"] === "diagnostics-copy");
  // Nothing to copy until the report arrives.
  assert.equal(copy?.disabled, true);
});

test("saving the bundle reports success, and a refusal names the session", async () => {
  const saved = loadSection({});
  renderToStaticMarkup(React.createElement(saved.module.DiagnosticsSection));
  const save = saved.buttons.find((props) => props["data-testid"] === "diagnostics-save-bundle");
  (save?.onClick as () => void)();
  await flush();
  assert.deepEqual(saved.toasts, [{ kind: "success", title: t("settings.diagnostics.bundleSaved") }]);

  const refused = loadSection({
    save: async () => {
      throw new diagnosticsLib.DiagnosticsRequestError("forbidden", "UI session required");
    },
  });
  renderToStaticMarkup(React.createElement(refused.module.DiagnosticsSection));
  const refusedSave = refused.buttons.find((props) => props["data-testid"] === "diagnostics-save-bundle");
  (refusedSave?.onClick as () => void)();
  await flush();
  assert.deepEqual(refused.toasts, [{ kind: "error", title: t("settings.diagnostics.forbidden") }]);

  const cancelled = loadSection({ save: async () => false });
  renderToStaticMarkup(React.createElement(cancelled.module.DiagnosticsSection));
  const cancelledSave = cancelled.buttons.find((props) => props["data-testid"] === "diagnostics-save-bundle");
  (cancelledSave?.onClick as () => void)();
  await flush();
  assert.deepEqual(cancelled.toasts, []);
});

test("the report renders every group, never a redacted value, and names what is unavailable", () => {
  const { module } = loadSection({});
  const report = diagnosticsLib.parseDiagnosticsReport({
    ...SAMPLE_BODY,
    sections: {
      ...SAMPLE_BODY.sections,
      // Even if a backend ever sent the real value beside the flag, it is not shown.
      environment: {
        status: "ok",
        data: { variables: [{ name: "HF_TOKEN", value: SECRET, redacted: true }] },
      },
    },
  });
  const html = renderToStaticMarkup(
    React.createElement(module.ReportGroups, { report, t }),
  );
  for (const key of ["studio", "system", "gpus", "python", "llamaCpp", "storage", "models", "mcp", "environment"]) {
    assert.ok(html.includes(t(`settings.diagnostics.groups.${key}`)), key);
  }
  assert.ok(html.includes("NVIDIA GeForce RTX 3090"));
  assert.ok(html.includes(t("settings.diagnostics.vramValue", { used: "4.0 GiB", free: "20.0 GiB", total: "24.0 GiB" })));
  assert.ok(html.includes(t("settings.diagnostics.pcieValue", { gen: "4", width: "1", maxGen: "4", maxWidth: "16" })));
  assert.ok(html.includes(t("settings.diagnostics.unavailable", { reason: "no llama-server binary was found" })));
  assert.ok(html.includes(t("settings.diagnostics.measuring")));
  assert.ok(html.includes(t("settings.diagnostics.state.idle")));
  assert.ok(html.includes("HF_TOKEN"));
  assert.ok(html.includes(t("settings.diagnostics.redacted")));
  assert.ok(!html.includes(SECRET));
});
