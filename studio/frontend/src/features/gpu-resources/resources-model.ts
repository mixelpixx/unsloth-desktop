// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The /api/resources payload and the arithmetic the sidebar strip draws from it. Pure and
// React-free, with no `@/` imports outside type positions, so node --test imports it as is.

import type { LoadedModelEntry } from "@/features/loaded-models";

const BYTES_PER_GIB = 1024 ** 3;

export type ResourceAttribution = "process" | "estimate";

export type ResourceApp = { pid: number; name: string; bytes: number };

/** The hardware check's measured link for a card (Settings > Resources > Hardware check). Only
 *  on a card the stored result measured, and only while it describes the GPUs installed now. */
export type ResourceGpuLink = {
  width: number | null;
  width_max: number | null;
  gen: number | null;
  h2d_gibs: number | null;
  best_gpu: number | null;
  best_h2d_gibs: number | null;
  slow: boolean;
};

export type ResourceGpu = {
  index: number;
  name: string | null;
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
  /** Null when Studio's share of this card is not known (a model of unknown size is on it). */
  studio_bytes: number | null;
  other_bytes: number | null;
  /** "process": Windows' per-process counter. "estimate": what Studio's runtimes logged. */
  attribution: ResourceAttribution | null;
  apps: ResourceApp[];
  /** Absent until the hardware check has measured this card. */
  link?: ResourceGpuLink;
};

export type ResourceModelKind =
  | "chat"
  | "audio"
  | "stt"
  | "image"
  | "video"
  | "embedding";

export type ResourceModel = {
  id: string;
  kind: ResourceModelKind;
  /** Which runtime's unload releases it; null when nothing in the UI can. */
  source: "chat" | "image" | "video" | "stt" | "embedding" | null;
  name: string;
  variant: string | null;
  gpu_ids: number[];
  device: string | null;
  layers_on_gpu: number | null;
  layers_total: number | null;
  context_length: number | null;
  cache_type_kv: string | null;
  vram_bytes: number | null;
  vram_approx: boolean;
  loading: boolean;
  inactive: boolean;
  stt_engine: string | null;
};

export type ResourceSnapshot = {
  gpus: ResourceGpu[];
  models: ResourceModel[];
  /** Other programs' memory the backend could not place on a card. */
  other_apps: ResourceApp[];
  /** A load is running somewhere, so the poll runs at its fastest. */
  loading: boolean;
};

const KINDS: readonly ResourceModelKind[] = [
  "chat",
  "audio",
  "stt",
  "image",
  "video",
  "embedding",
];
const SOURCES = ["chat", "image", "video", "stt", "embedding"] as const;

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nonNegative(value: unknown): number {
  const number = finite(value);
  return number === null ? 0 : Math.max(0, number);
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function gpuLink(value: unknown): ResourceGpuLink | null {
  if (!value || typeof value !== "object") return null;
  const link = value as Record<string, unknown>;
  const int = (v: unknown) =>
    typeof v === "number" && Number.isInteger(v) ? v : null;
  return {
    width: int(link.width),
    width_max: int(link.width_max),
    gen: int(link.gen),
    h2d_gibs: finite(link.h2d_gibs),
    best_gpu: int(link.best_gpu),
    best_h2d_gibs: finite(link.best_h2d_gibs),
    slow: link.slow === true,
  };
}

function apps(value: unknown): ResourceApp[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((row) => {
    if (!row || typeof row !== "object") return [];
    const app = row as Record<string, unknown>;
    const pid = finite(app.pid);
    if (pid === null) return [];
    return [{ pid, name: text(app.name) ?? `PID ${pid}`, bytes: nonNegative(app.bytes) }];
  });
}

/**
 * The payload as sent, with anything malformed dropped rather than trusted. A backend older
 * than the route answers 404 and never reaches here; a newer one may add fields, which pass
 * through untouched by being ignored.
 */
export function normalizeResourceSnapshot(raw: unknown): ResourceSnapshot {
  const body = (raw && typeof raw === "object" ? raw : {}) as Record<
    string,
    unknown
  >;
  const gpus = (Array.isArray(body.gpus) ? body.gpus : []).flatMap((row) => {
    if (!row || typeof row !== "object") return [];
    const gpu = row as Record<string, unknown>;
    const index = finite(gpu.index);
    const total = nonNegative(gpu.total_bytes);
    if (index === null || total <= 0) return [];
    const attribution =
      gpu.attribution === "process" || gpu.attribution === "estimate"
        ? gpu.attribution
        : null;
    const link = gpuLink(gpu.link);
    return [
      {
        index,
        name: text(gpu.name),
        total_bytes: total,
        used_bytes: nonNegative(gpu.used_bytes),
        free_bytes: nonNegative(gpu.free_bytes),
        studio_bytes: finite(gpu.studio_bytes),
        other_bytes: finite(gpu.other_bytes),
        attribution,
        apps: apps(gpu.apps),
        ...(link ? { link } : {}),
      } satisfies ResourceGpu,
    ];
  });
  const models = (Array.isArray(body.models) ? body.models : []).flatMap(
    (row) => {
      if (!row || typeof row !== "object") return [];
      const model = row as Record<string, unknown>;
      const name = text(model.name);
      const kind = KINDS.find((k) => k === model.kind);
      if (!name || !kind) return [];
      const source =
        SOURCES.find((s) => s === model.source) ?? null;
      return [
        {
          id: text(model.id) ?? `${source ?? kind}:${name}`,
          kind,
          source,
          name,
          variant: text(model.variant),
          gpu_ids: Array.isArray(model.gpu_ids)
            ? model.gpu_ids.filter(
                (id): id is number => typeof id === "number" && id >= 0,
              )
            : [],
          device: text(model.device),
          layers_on_gpu: finite(model.layers_on_gpu),
          layers_total: finite(model.layers_total),
          context_length: finite(model.context_length),
          cache_type_kv: text(model.cache_type_kv),
          vram_bytes: finite(model.vram_bytes),
          vram_approx: model.vram_approx === true,
          loading: model.loading === true,
          inactive: model.inactive === true,
          stt_engine: text(model.stt_engine),
        } satisfies ResourceModel,
      ];
    },
  );
  return {
    gpus,
    models,
    other_apps: apps(body.other_apps),
    loading: body.loading === true,
  };
}

// ── Formatting ────────────────────────────────────────────────────

/**
 * A memory figure in GiB to one decimal: `"10.3 GiB"`. Binary, and labelled so, like every
 * other memory readout (lib/memory/format.ts) and the load verdict's own sentences, which print
 * `{:.1f} GiB`. One decimal, not formatGiB's adaptive rounding: on a 24 GiB card "10 GiB free"
 * and "10.4 GiB free" are different answers to "will this fit".
 */
export function formatResourceGiB(bytes: number): string {
  const gib = Number.isFinite(bytes) && bytes > 0 ? bytes / BYTES_PER_GIB : 0;
  return `${gib >= 100 ? Math.round(gib) : gib.toFixed(1)} GiB`;
}

/** `8192` → `"8K"`, `131072` → `"128K"`, `40000` → `"39.1K"`; under 1024 as is. */
export function formatContextLength(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens <= 0) return "0";
  if (tokens < 1024) return String(Math.round(tokens));
  const k = tokens / 1024;
  return `${Number.isInteger(k) ? k : k.toFixed(1).replace(/\.0$/, "")}K`;
}

/** `[0]` → `"0"`, `[0, 1]` → `"0+1"`: the cards a model spans, for "GPU 0+1". */
export function formatGpuIds(ids: readonly number[]): string {
  return [...ids].sort((a, b) => a - b).join("+");
}

// ── The bar ───────────────────────────────────────────────────────

export type BarSegments = {
  /** Whether Studio and other programs could be told apart on this card. */
  split: boolean;
  /** Percent of the card, 0-100, each. Studio + other + free (or used + free) ≤ 100. */
  studio: number;
  other: number;
  /** Only when not split: everything in use, as one segment. */
  used: number;
  free: number;
};

function percent(part: number, total: number): number {
  if (!(total > 0) || !(part > 0)) return 0;
  return Math.min(100, (part / total) * 100);
}

/**
 * Studio · other apps · free, as percentages of the card. Clamped so a reading that races an
 * unload (Studio's share from before it, the total from after) never draws past the end.
 */
export function barSegments(gpu: ResourceGpu): BarSegments {
  const total = gpu.total_bytes;
  const free = percent(gpu.free_bytes, total);
  const room = Math.max(0, 100 - free);
  if (gpu.studio_bytes === null || gpu.other_bytes === null) {
    return {
      split: false,
      studio: 0,
      other: 0,
      used: Math.min(room, percent(gpu.used_bytes, total)),
      free,
    };
  }
  const studio = Math.min(room, percent(gpu.studio_bytes, total));
  const other = Math.min(room - studio, percent(gpu.other_bytes, total));
  return { split: true, studio, other, used: studio + other, free };
}

/** The sizes the bar's accessible name and the panel legend print, already formatted. */
export function gpuFigures(gpu: ResourceGpu): {
  index: number;
  free: string;
  total: string;
  used: string;
  studio: string | null;
  other: string | null;
} {
  return {
    index: gpu.index,
    free: formatResourceGiB(gpu.free_bytes),
    total: formatResourceGiB(gpu.total_bytes),
    used: formatResourceGiB(gpu.used_bytes),
    studio:
      gpu.studio_bytes === null ? null : formatResourceGiB(gpu.studio_bytes),
    other: gpu.other_bytes === null ? null : formatResourceGiB(gpu.other_bytes),
  };
}

// ── The link badge ────────────────────────────────────────────────

/** What the strip's "x1" badge says, or null: only a card the hardware check measured as slow,
 *  with the width it trained at under load. The tooltip's numbers come along, formatted. */
export function linkBadge(gpu: ResourceGpu): {
  width: number;
  widthMax: number | string;
  h2d: string;
  best: string;
  bestGpu: number | string;
} | null {
  const link = gpu.link;
  if (!link?.slow || link.width === null) return null;
  const gibs = (value: number | null) =>
    value === null ? "?" : value.toFixed(1);
  return {
    width: link.width,
    widthMax: link.width_max ?? "?",
    h2d: gibs(link.h2d_gibs),
    best: gibs(link.best_h2d_gibs),
    bestGpu: link.best_gpu ?? "?",
  };
}

// ── A model row ───────────────────────────────────────────────────

/** A path's last two segments; a repo id is already short. The loaded-models card's rule. */
export function shortResourceName(name: string): string {
  const trimmed = name.replace(/[\\/]+$/, "");
  const segments = trimmed.split(/[\\/]+/).filter(Boolean);
  return segments.length <= 2 ? trimmed : segments.slice(-2).join("/");
}

export type ModelFacts = {
  /** "GPU 0+1" ids, or null when on no known card. */
  gpus: string | null;
  /** Ran with no layer on a GPU. */
  cpu: boolean;
  layers: { on: number; total: number } | null;
  context: string | null;
  vram: string | null;
  vramApprox: boolean;
};

/** What the row says after the name, as parts the component translates. */
export function modelFacts(model: ResourceModel): ModelFacts {
  const cpu =
    model.layers_on_gpu === 0 ||
    (model.gpu_ids.length === 0 && model.device?.toLowerCase() === "cpu");
  return {
    gpus: model.gpu_ids.length > 0 ? formatGpuIds(model.gpu_ids) : null,
    cpu,
    layers:
      model.layers_on_gpu !== null && model.layers_total !== null
        ? { on: model.layers_on_gpu, total: model.layers_total }
        : null,
    context:
      model.context_length !== null
        ? formatContextLength(model.context_length)
        : null,
    vram: model.vram_bytes !== null ? formatResourceGiB(model.vram_bytes) : null,
    vramApprox: model.vram_bytes !== null && model.vram_approx,
  };
}

/**
 * The loaded-models card's row for this model, so its eject (which re-reads the runtime and
 * refuses to release a model that replaced this one) is the one used. Null for a row nothing
 * there can release: a load still in flight, an embedder (released through Settings), or a
 * dictation row without its engine.
 */
export function ejectEntryFor(model: ResourceModel): LoadedModelEntry | null {
  if (model.loading) return null;
  switch (model.source) {
    case "chat":
      return {
        id: model.id,
        kind:
          model.kind === "audio" ? "tts" : model.kind === "stt" ? "stt" : "text",
        source: "chat",
        name: model.name,
        detail: "",
        inactive: model.inactive,
      };
    case "image":
    case "video":
      return {
        id: model.id,
        kind: model.source,
        source: model.source,
        name: model.name,
        detail: "",
      };
    case "stt": {
      const engine = model.stt_engine;
      if (
        engine !== "transformers" &&
        engine !== "mtmd" &&
        engine !== "gguf" &&
        engine !== "audiocpp"
      ) {
        return null;
      }
      return {
        id: model.id,
        kind: "stt",
        source: "stt",
        name: model.name,
        detail: "",
        sttEngine: engine,
      };
    }
    default:
      return null;
  }
}

export function canEjectModel(model: ResourceModel): boolean {
  if (model.loading) return false;
  return model.source === "embedding" || ejectEntryFor(model) !== null;
}

// ── Polling ───────────────────────────────────────────────────────

export const RESOURCES_POLL_LOADING_MS = 1000;
export const RESOURCES_POLL_PANEL_MS = 3000;
export const RESOURCES_POLL_IDLE_MS = 10_000;

/**
 * How long until the next read, or null for none. Every second while a load runs (memory is
 * moving and the bar is the progress), every 3 s while the panel is open, otherwise every 10 s,
 * and not at all in a hidden tab: the strip is always on screen, so it is the one poll that
 * would otherwise never stop.
 */
export function resourcesPollIntervalMs({
  hidden,
  loading,
  panelOpen,
}: {
  hidden: boolean;
  loading: boolean;
  panelOpen: boolean;
}): number | null {
  if (hidden) return null;
  if (loading) return RESOURCES_POLL_LOADING_MS;
  if (panelOpen) return RESOURCES_POLL_PANEL_MS;
  return RESOURCES_POLL_IDLE_MS;
}
