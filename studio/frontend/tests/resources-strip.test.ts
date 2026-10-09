// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sidebar's GPU memory strip (features/gpu-resources). The arithmetic and labels are pure and
// imported directly; the wiring (where it mounts, how it polls, what it hands a screen reader)
// is read from the shipped source, since the node suite has no DOM.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type ResourceGpu,
  type ResourceModel,
  RESOURCES_POLL_IDLE_MS,
  RESOURCES_POLL_LOADING_MS,
  RESOURCES_POLL_PANEL_MS,
  barSegments,
  canEjectModel,
  ejectEntryFor,
  formatContextLength,
  formatGpuIds,
  formatResourceGiB,
  gpuFigures,
  modelFacts,
  normalizeResourceSnapshot,
  resourcesPollIntervalMs,
  shortResourceName,
} from "../src/features/gpu-resources/resources-model.ts";
import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

const GIB = 1024 ** 3;

function gpu(overrides: Partial<ResourceGpu> = {}): ResourceGpu {
  return {
    index: 0,
    name: "NVIDIA GeForce RTX 3090",
    total_bytes: 24 * GIB,
    used_bytes: 21 * GIB,
    free_bytes: 3 * GIB,
    studio_bytes: 8 * GIB,
    other_bytes: 13 * GIB,
    attribution: "process",
    apps: [],
    ...overrides,
  };
}

function model(overrides: Partial<ResourceModel> = {}): ResourceModel {
  return {
    id: "chat:unsloth/Qwen3-27B-GGUF",
    kind: "chat",
    source: "chat",
    name: "unsloth/Qwen3-27B-GGUF",
    variant: "Q4_K_M",
    gpu_ids: [0],
    device: null,
    layers_on_gpu: 66,
    layers_total: 66,
    context_length: 8192,
    cache_type_kv: null,
    vram_bytes: 20.7 * GIB,
    vram_approx: false,
    loading: false,
    inactive: false,
    stt_engine: null,
    ...overrides,
  };
}

// ── The bar ───────────────────────────────────────────────────────

test("a split card draws Studio, other apps and free in proportion", () => {
  const segments = barSegments(gpu());
  assert.equal(segments.split, true);
  assert.equal(segments.studio, (8 / 24) * 100);
  assert.equal(segments.other, (13 / 24) * 100);
  assert.equal(segments.free, (3 / 24) * 100);
  assert.ok(Math.abs(segments.studio + segments.other + segments.free - 100) < 1e-9);
});

test("a card with no split draws everything in use as one segment", () => {
  const segments = barSegments(gpu({ studio_bytes: null, other_bytes: null }));
  assert.equal(segments.split, false);
  assert.equal(segments.studio, 0);
  assert.equal(segments.other, 0);
  assert.equal(segments.used, (21 / 24) * 100);
});

test("a reading that races an unload never draws past the end", () => {
  // Studio's share from before the unload, free from after it.
  const segments = barSegments(
    gpu({ studio_bytes: 20 * GIB, other_bytes: 10 * GIB, free_bytes: 10 * GIB }),
  );
  assert.ok(segments.studio + segments.other + segments.free <= 100 + 1e-9);
  assert.ok(segments.other >= 0);
});

test("a zero or missing total draws an empty bar, not NaN", () => {
  const segments = barSegments(gpu({ total_bytes: 0 }));
  for (const value of [segments.studio, segments.other, segments.used, segments.free]) {
    assert.equal(value, 0);
  }
});

// ── Labels ────────────────────────────────────────────────────────

test("memory reads in GiB to one decimal, like the load verdict", () => {
  assert.equal(formatResourceGiB(10.34 * GIB), "10.3 GiB");
  assert.equal(formatResourceGiB(24 * GIB), "24.0 GiB");
  assert.equal(formatResourceGiB(0), "0.0 GiB");
  assert.equal(formatResourceGiB(Number.NaN), "0.0 GiB");
  assert.equal(formatResourceGiB(-5), "0.0 GiB");
  assert.equal(formatResourceGiB(141 * GIB), "141 GiB");
  assert.doesNotMatch(formatResourceGiB(GIB), / GB$/);
});

test("context lengths read as K", () => {
  assert.equal(formatContextLength(8192), "8K");
  assert.equal(formatContextLength(131072), "128K");
  assert.equal(formatContextLength(40000), "39.1K");
  assert.equal(formatContextLength(512), "512");
  assert.equal(formatContextLength(0), "0");
});

test("a model on two cards names both", () => {
  assert.equal(formatGpuIds([1, 0]), "0+1");
  assert.equal(formatGpuIds([0]), "0");
});

test("gpu figures carry every number the bar's accessible name needs", () => {
  assert.deepEqual(gpuFigures(gpu()), {
    index: 0,
    free: "3.0 GiB",
    total: "24.0 GiB",
    used: "21.0 GiB",
    studio: "8.0 GiB",
    other: "13.0 GiB",
  });
  const unsplit = gpuFigures(gpu({ studio_bytes: null, other_bytes: null }));
  assert.equal(unsplit.studio, null);
  assert.equal(unsplit.other, null);
});

test("the bar's accessible names carry every figure", () => {
  const { barLabel, barLabelUnsplit, gpuFree, gpuSummary } = en.resources;
  for (const placeholder of ["{index}", "{studio}", "{other}", "{free}", "{total}"]) {
    assert.ok(barLabel.includes(placeholder), `barLabel lacks ${placeholder}`);
  }
  for (const placeholder of ["{index}", "{used}", "{free}", "{total}"]) {
    assert.ok(barLabelUnsplit.includes(placeholder), `barLabelUnsplit lacks ${placeholder}`);
  }
  assert.equal(gpuFree, "GPU {index} · {free} free");
  assert.match(gpuSummary, /\{free\}.*\{total\}/);
});

test("a model row says where it runs, how much of it and how big", () => {
  assert.deepEqual(modelFacts(model()), {
    gpus: "0",
    cpu: false,
    layers: { on: 66, total: 66 },
    context: "8K",
    vram: "20.7 GiB",
    vramApprox: false,
  });
  const planned = modelFacts(model({ vram_approx: true }));
  assert.equal(planned.vramApprox, true);
  const cpu = modelFacts(model({ gpu_ids: [], layers_on_gpu: 0, vram_bytes: null }));
  assert.equal(cpu.cpu, true);
  assert.equal(cpu.gpus, null);
  assert.equal(cpu.vram, null);
  // An unsized model never claims to be approximate about a number it does not have.
  assert.equal(modelFacts(model({ vram_bytes: null, vram_approx: true })).vramApprox, false);
});

test("a path keeps its last two segments, a repo id stays whole", () => {
  assert.equal(shortResourceName("unsloth/Qwen3-27B-GGUF"), "unsloth/Qwen3-27B-GGUF");
  assert.equal(
    shortResourceName("C:\\Users\\me\\models\\qwen\\Qwen3-27B-Q4_K_M.gguf"),
    "qwen/Qwen3-27B-Q4_K_M.gguf",
  );
});

// ── The payload ───────────────────────────────────────────────────

test("a malformed payload is dropped piece by piece, never trusted", () => {
  const snapshot = normalizeResourceSnapshot({
    gpus: [
      { index: 0, total_bytes: 24 * GIB, free_bytes: 3 * GIB, used_bytes: 21 * GIB, studio_bytes: 8 * GIB, other_bytes: 13 * GIB, attribution: "process", apps: [{ pid: 1372, name: "LM Studio.exe", bytes: 13 * GIB }, { name: "no pid" }] },
      { index: 1, total_bytes: 0 },
      { name: "no index", total_bytes: 24 * GIB },
      "garbage",
    ],
    models: [
      { kind: "chat", name: "a", source: "chat", gpu_ids: [0, -1, "x"] },
      { kind: "spaceship", name: "b" },
      { kind: "image" },
    ],
    other_apps: "not a list",
    loading: "yes",
    future_field: 1,
  });
  assert.equal(snapshot.gpus.length, 1);
  assert.deepEqual(snapshot.gpus[0].apps, [{ pid: 1372, name: "LM Studio.exe", bytes: 13 * GIB }]);
  assert.equal(snapshot.models.length, 1);
  assert.deepEqual(snapshot.models[0].gpu_ids, [0]);
  assert.equal(snapshot.models[0].id, "chat:a");
  assert.deepEqual(snapshot.other_apps, []);
  assert.equal(snapshot.loading, false, "only a literal true means a load is running");
  assert.deepEqual(normalizeResourceSnapshot(null), {
    gpus: [],
    models: [],
    other_apps: [],
    loading: false,
  });
});

// ── Eject ─────────────────────────────────────────────────────────

test("each row ejects through the loaded models card's guarded path", () => {
  assert.deepEqual(ejectEntryFor(model()), {
    id: "chat:unsloth/Qwen3-27B-GGUF",
    kind: "text",
    source: "chat",
    name: "unsloth/Qwen3-27B-GGUF",
    detail: "",
    inactive: false,
  });
  // A kept slot or a cached Transformers model is not the active one: the card's cached path.
  assert.equal(ejectEntryFor(model({ inactive: true }))?.inactive, true);
  assert.equal(ejectEntryFor(model({ kind: "audio" }))?.kind, "tts");
  assert.equal(ejectEntryFor(model({ kind: "image", source: "image" }))?.source, "image");
  assert.equal(
    ejectEntryFor(model({ kind: "stt", source: "stt", stt_engine: "gguf" }))?.sttEngine,
    "gguf",
  );
});

test("nothing ejects a load in flight or a row without a runtime to release it", () => {
  assert.equal(canEjectModel(model({ loading: true })), false);
  assert.equal(ejectEntryFor(model({ loading: true })), null);
  assert.equal(canEjectModel(model({ source: null })), false);
  assert.equal(canEjectModel(model({ kind: "stt", source: "stt", stt_engine: null })), false);
  // Released through Settings -> Documents, not the card.
  assert.equal(ejectEntryFor(model({ kind: "embedding", source: "embedding" })), null);
  assert.equal(canEjectModel(model({ kind: "embedding", source: "embedding" })), true);
});

// ── Polling ───────────────────────────────────────────────────────

test("the poll is every second during a load, 3 s with the panel open, else 10 s", () => {
  assert.equal(RESOURCES_POLL_LOADING_MS, 1000);
  assert.equal(RESOURCES_POLL_PANEL_MS, 3000);
  assert.equal(RESOURCES_POLL_IDLE_MS, 10_000);
  const at = (hidden: boolean, loading: boolean, panelOpen: boolean) =>
    resourcesPollIntervalMs({ hidden, loading, panelOpen });
  assert.equal(at(false, true, false), 1000);
  assert.equal(at(false, true, true), 1000, "a load outranks the open panel");
  assert.equal(at(false, false, true), 3000);
  assert.equal(at(false, false, false), 10_000);
  for (const loading of [true, false]) {
    for (const panelOpen of [true, false]) {
      assert.equal(at(true, loading, panelOpen), null, "a hidden tab never polls");
    }
  }
});

const STORE = readSrc("features/gpu-resources/resources-store.ts");

test("one shared zustand store polls /api/resources on timers, never a stream", () => {
  assert.match(STORE, /from "zustand"/);
  assert.match(STORE, /export const useResourcesStore = create</);
  assert.match(STORE, /authFetch\("\/api\/resources"/);
  assert.match(STORE, /resourcesPollIntervalMs\(/);
  assert.match(STORE, /hidden: typeof document !== "undefined" && document\.hidden/);
  assert.doesNotMatch(STORE, /EventSource|WebSocket|setInterval/);
  // Stops when nothing on screen wants it, and resumes on return to the tab.
  assert.match(STORE, /if \(consumers === 0\) stop\(\)/);
  assert.match(STORE, /addEventListener\("visibilitychange"/);
  // A load this tab started moves the cadence before the backend reports it.
  assert.match(STORE, /subscribeModelLifecycle\(/);
});

// ── Wiring ────────────────────────────────────────────────────────

const SIDEBAR = readSrc("components/app-sidebar.tsx");
const STRIP = readSrc("features/gpu-resources/resources-strip.tsx");

test("the strip sits at the foot of the sidebar, above the profile", () => {
  assert.match(SIDEBAR, /import \{ ResourcesStrip \} from "@\/features\/gpu-resources";/);
  const footer = SIDEBAR.indexOf("<SidebarFooter");
  const strip = SIDEBAR.indexOf("<ResourcesStrip />");
  const profileMenu = SIDEBAR.indexOf("<SidebarMenu", strip);
  const footerEnd = SIDEBAR.indexOf("</SidebarFooter>");
  assert.ok(footer !== -1 && strip > footer && strip < footerEnd, "inside the footer");
  assert.ok(profileMenu !== -1 && profileMenu < footerEnd, "before the profile menu");
});

test("the strip follows the collapsed rail with a minimal per-card column", () => {
  assert.match(STRIP, /group-data-\[collapsible=icon\]:hidden/);
  assert.match(STRIP, /vertical=\{true\}/);
  assert.match(
    STRIP,
    /hidden h-\[calc\(18px\*var\(--ui-space-scale,1\)\)\] group-data-\[collapsible=icon\]:flex/,
  );
});

test("bars carry their numbers to a screen reader, and the panel is keyboard reachable", () => {
  // Every bar has an accessible name built from barLabel / barLabelUnsplit.
  assert.match(STRIP, /aria-label=\{label\}/);
  assert.match(STRIP, /t\("resources\.barLabel"/);
  assert.match(STRIP, /t\("resources\.barLabelUnsplit"/);
  // In the panel a bar is a meter with values; in the strip the button names every card.
  assert.match(STRIP, /role=\{meter \? "meter" : "img"\}/);
  assert.match(STRIP, /"aria-valuenow": Math\.round\(segments\.used\)/);
  assert.match(STRIP, /aria-label=\{t\("resources\.stripLabel", \{ summary \}\)\}/);
  // A real button behind a Radix popover trigger: Tab, Enter and Escape for free.
  assert.match(STRIP, /<PopoverTrigger asChild=\{true\}>\s*<button\s+type="button"/);
  assert.match(STRIP, /focus-visible:ring-1/);
  assert.match(STRIP, /aria-label=\{t\("resources\.ejectModel", \{ model: label \}\)\}/);
});

test("nothing is drawn on a host with no GPU or with the strip switched off", () => {
  assert.match(
    STRIP,
    /if \(!show \|\| !snapshot \|\| snapshot\.gpus\.length === 0\) return null;/,
  );
  assert.match(STRIP, /useResourcesPolling\(show\)/);
});

test("Eject reuses the loaded models card's eject, and Eject all goes one at a time", () => {
  const eject = readSrc("features/gpu-resources/resources-eject.ts");
  assert.match(eject, /import \{ type EjectOutcome, ejectLoadedModel \} from "@\/features\/loaded-models";/);
  assert.match(eject, /"\/api\/settings\/embedding-model\/unload"/);
  assert.match(STRIP, /for \(const model of ejectable\) \{\s*released = \(await eject\(model, true\)\) && released;/);
  const barrel = readSrc("features/loaded-models/index.ts");
  assert.ok(
    barrel.indexOf("./show-loaded-models-pref") < barrel.indexOf("./loaded-models-api"),
    "the new export keeps the preference module first",
  );
});

// ── The setting ───────────────────────────────────────────────────

test("Appearance has the switch, and Reset all local preferences clears it", () => {
  const appearance = readSrc("features/settings/tabs/appearance-tab.tsx");
  assert.match(appearance, /checked=\{showResourcesStrip\}/);
  assert.match(appearance, /onCheckedChange=\{setShowResourcesStrip\}/);
  const general = readSrc("features/settings/tabs/general-tab.tsx");
  const listStart = general.indexOf("const PREFS_KEYS: string[] = [");
  const listEnd = general.indexOf("];", listStart);
  assert.ok(
    general.slice(listStart, listEnd).includes("RESOURCES_STRIP_PREFERENCE_KEY,"),
    "the key is in the reset inventory",
  );
  const pref = readSrc("features/gpu-resources/show-resources-strip-pref.ts");
  assert.match(pref, /RESOURCES_STRIP_PREFERENCE_KEY = "unsloth_show_resources_strip"/);
  const search = readSrc("features/settings/settings-search.ts");
  assert.match(search, /"resources\.settings\.showStrip"/);
});

test("the barrel evaluates the preference before the strip", () => {
  // general-tab reads the key at module scope, and the strip reaches the settings barrel
  // through the loaded models card: the same cycle loaded-models-barrel-order pins.
  const barrel = readSrc("features/gpu-resources/index.ts");
  const pref = barrel.indexOf("./show-resources-strip-pref");
  const strip = barrel.indexOf("./resources-strip");
  assert.ok(pref !== -1 && strip !== -1 && pref < strip);
});

test("every locale puts the resources namespace right after common", () => {
  for (const locale of [
    "en", "ar", "de", "es", "fr", "he", "hi", "it", "ja", "ko", "pt-br", "ru", "sv", "zh-CN",
  ]) {
    const source = readSrc(`i18n/locales/${locale}.ts`);
    const common = source.indexOf("\n  common: {");
    const commonEnd = source.indexOf("\n  },\n", common);
    const resources = source.indexOf("\n  resources: {");
    assert.ok(common !== -1 && resources !== -1, `${locale}: both namespaces`);
    // Only an optional comment line between the end of common and resources.
    const between = source.slice(commonEnd + "\n  },".length, resources);
    assert.match(between, /^(\n\s*\/\/[^\n]*)*$/, `${locale}: resources follows common`);
  }
});
