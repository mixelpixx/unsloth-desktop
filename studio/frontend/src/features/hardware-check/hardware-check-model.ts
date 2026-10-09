// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Settings > Resources > Hardware check: the /api/hardware-check payload and the pure half of
 *  the section, the GPU strip badge, the Tensor Parallelism warning and the training warning.
 *  Free of `@/` imports so `tests/` loads it under `node --experimental-strip-types`. */

export const HARDWARE_CHECK_ENDPOINT = "/api/hardware-check";

/** The options "Apply recommended" may switch on, in display order. Mirrors OPTION_KEYS in
 *  studio/backend/utils/hardware_check_settings.py. */
export const HARDWARE_CHECK_OPTIONS = [
  "prefer_fast_link",
  "avoid_tensor_split",
  "warn_training_slow_link",
] as const;

export type HardwareCheckOption = (typeof HARDWARE_CHECK_OPTIONS)[number];

export type HardwareCheckSettingKey = HardwareCheckOption | "auto_run";

export type HardwareCheckSettings = Record<HardwareCheckSettingKey, boolean>;

export const DEFAULT_HARDWARE_CHECK_SETTINGS: HardwareCheckSettings = {
  auto_run: true,
  prefer_fast_link: false,
  avoid_tensor_split: false,
  warn_training_slow_link: false,
};

export type PcieLink = {
  genCurrent: number | null;
  genMax: number | null;
  widthCurrent: number | null;
  widthMax: number | null;
};

export type HardwareCheckGpu = {
  index: number;
  name: string | null;
  status: "measured" | "skipped" | "failed";
  reason: string | null;
  link: PcieLink | null;
  linkIdle: PcieLink | null;
  h2dGibs: number | null;
  d2hGibs: number | null;
};

export type HardwareCheckPair = {
  a: number;
  b: number;
  peerAb: boolean | null;
  peerBa: boolean | null;
  copyAbGibs: number | null;
  copyBaGibs: number | null;
};

export type HardwareCheckStorage = {
  key: string;
  drive: string | null;
  busType: string | null;
  mediaType: string | null;
  model: string | null;
};

export type FindingSeverity = "warning" | "info" | "ok";

export type FindingValue = string | number | boolean | null;

export type HardwareCheckFinding = {
  id: string;
  severity: FindingSeverity;
  values: Record<string, FindingValue>;
  /** The backend's English sentence, used when this build does not know the id. */
  text: string;
};

export type HardwareCheckResult = {
  finishedAt: string | null;
  trigger: string | null;
  durationMs: number | null;
  gpus: HardwareCheckGpu[];
  pairs: HardwareCheckPair[];
  storage: HardwareCheckStorage[];
  findings: HardwareCheckFinding[];
  recommended: Record<HardwareCheckOption, boolean>;
  slowGpus: number[];
};

export type HardwareCheckSkip = { reason: string; trigger: string | null; at: string | null };

export type HardwareCheckRunState = {
  running: boolean;
  phase: string | null;
  progress: number;
  trigger: string | null;
  lastSkip: HardwareCheckSkip | null;
  lastError: string | null;
};

export type HardwareCheckStatus = {
  settings: HardwareCheckSettings;
  result: HardwareCheckResult | null;
  /** null when the GPUs installed now could not be read. */
  upToDate: boolean | null;
  state: HardwareCheckRunState;
};

// ── Parsing ──────────────────────────────────────────────────────

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function int(value: unknown): number | null {
  const n = num(value);
  return n !== null && Number.isInteger(n) ? n : null;
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function bool(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function parseLink(raw: unknown): PcieLink | null {
  if (!raw || typeof raw !== "object") return null;
  const link = record(raw);
  const parsed: PcieLink = {
    genCurrent: int(link.gen_current),
    genMax: int(link.gen_max),
    widthCurrent: int(link.width_current),
    widthMax: int(link.width_max),
  };
  return parsed.genCurrent === null && parsed.widthCurrent === null ? null : parsed;
}

export function parseHardwareCheckSettings(raw: unknown): HardwareCheckSettings {
  const body = record(raw);
  const out = { ...DEFAULT_HARDWARE_CHECK_SETTINGS };
  for (const key of Object.keys(out) as HardwareCheckSettingKey[]) {
    const value = bool(body[key]);
    if (value !== null) out[key] = value;
  }
  return out;
}

function parseFinding(raw: unknown): HardwareCheckFinding[] {
  const finding = record(raw);
  const id = str(finding.id);
  if (!id) return [];
  const severity: FindingSeverity =
    finding.severity === "warning" || finding.severity === "ok" ? finding.severity : "info";
  const values: Record<string, FindingValue> = {};
  for (const [key, value] of Object.entries(record(finding.values))) {
    if (
      value === null ||
      typeof value === "string" ||
      typeof value === "boolean" ||
      (typeof value === "number" && Number.isFinite(value))
    ) {
      values[key] = value;
    }
  }
  return [{ id, severity, values, text: str(finding.text) ?? id }];
}

function parseResult(raw: unknown): HardwareCheckResult | null {
  if (!raw || typeof raw !== "object") return null;
  const body = record(raw);
  const gpus = list(body.gpus).flatMap((row): HardwareCheckGpu[] => {
    const gpu = record(row);
    const index = int(gpu.index);
    if (index === null) return [];
    const status =
      gpu.status === "measured" || gpu.status === "skipped" ? gpu.status : "failed";
    return [
      {
        index,
        name: str(gpu.name),
        status,
        reason: str(gpu.reason),
        link: parseLink(gpu.link),
        linkIdle: parseLink(gpu.link_idle),
        h2dGibs: num(gpu.h2d_gibs),
        d2hGibs: num(gpu.d2h_gibs),
      },
    ];
  });
  const pairs = list(body.pairs).flatMap((row): HardwareCheckPair[] => {
    const pair = record(row);
    const a = int(pair.a);
    const b = int(pair.b);
    if (a === null || b === null) return [];
    return [
      {
        a,
        b,
        peerAb: bool(pair.peer_ab),
        peerBa: bool(pair.peer_ba),
        copyAbGibs: num(pair.copy_ab_gibs),
        copyBaGibs: num(pair.copy_ba_gibs),
      },
    ];
  });
  const storage = list(body.storage).flatMap((row): HardwareCheckStorage[] => {
    const entry = record(row);
    const key = str(entry.key);
    if (!key) return [];
    return [
      {
        key,
        drive: str(entry.drive),
        busType: str(entry.bus_type),
        mediaType: str(entry.media_type),
        model: str(entry.model),
      },
    ];
  });
  const recommendedRaw = record(body.recommended);
  const recommended = Object.fromEntries(
    HARDWARE_CHECK_OPTIONS.map((key) => [key, recommendedRaw[key] === true]),
  ) as Record<HardwareCheckOption, boolean>;
  return {
    finishedAt: str(body.finished_at),
    trigger: str(body.trigger),
    durationMs: num(body.duration_ms),
    gpus,
    pairs,
    storage,
    findings: list(body.findings).flatMap(parseFinding),
    recommended,
    slowGpus: list(body.slow_gpus).flatMap((v) => (int(v) === null ? [] : [int(v) as number])),
  };
}

function parseState(raw: unknown): HardwareCheckRunState {
  const state = record(raw);
  const skip = record(state.last_skip);
  const reason = str(skip.reason);
  return {
    running: state.running === true,
    phase: str(state.phase),
    progress: Math.max(0, Math.min(1, num(state.progress) ?? 0)),
    trigger: str(state.trigger),
    lastSkip: reason ? { reason, trigger: str(skip.trigger), at: str(skip.at) } : null,
    lastError: str(state.last_error),
  };
}

/** The payload as sent, anything malformed dropped rather than trusted. */
export function parseHardwareCheckStatus(raw: unknown): HardwareCheckStatus {
  const body = record(raw);
  return {
    settings: parseHardwareCheckSettings(body.settings),
    result: parseResult(body.result),
    upToDate: bool(body.up_to_date),
    state: parseState(body.state),
  };
}

// ── Wording ──────────────────────────────────────────────────────

/** One decimal, as the backend rounds: `24.9`. */
export function formatGibs(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) : "?";
}

/** `"Gen4 x16"`, `"x1"` without a generation, `"?"` when nothing was read. */
export function formatLink(link: PcieLink | null | undefined): string {
  if (!link || (link.widthCurrent === null && link.genCurrent === null)) return "?";
  const width = link.widthCurrent === null ? "x?" : `x${link.widthCurrent}`;
  return link.genCurrent === null ? width : `Gen${link.genCurrent} ${width}`;
}

/** `"Gen4 x16"` for the link's maximum, `"?"` when unknown. */
export function formatLinkMax(link: PcieLink | null | undefined): string {
  if (!link || (link.widthMax === null && link.genMax === null)) return "?";
  const width = link.widthMax === null ? "x?" : `x${link.widthMax}`;
  return link.genMax === null ? width : `Gen${link.genMax} ${width}`;
}

/** A translation key under `hardwareCheck.` plus its values, for one finding. */
export type FindingMessage = {
  key: string;
  values: Record<string, string | number>;
};

const LOCATION_KEYS: Record<string, string> = {
  studio_home: "locations.studioHome",
  hf_cache: "locations.hfCache",
  temp: "locations.temp",
};

/** The translation key (relative to `hardwareCheck.`) a location is named by. */
export function locationKey(key: string): string {
  return LOCATION_KEYS[key] ?? "locations.other";
}

function val(value: FindingValue | undefined): string | number {
  if (typeof value === "number") return value;
  if (typeof value === "string") return value;
  return "?";
}

/**
 * How the UI words a finding: a key relative to `hardwareCheck.` and its values. Null for an id
 * this build does not know, which the caller shows as the backend's English `text`.
 * `values.location` of a storage finding is a location KEY: the caller translates it.
 */
export function findingMessage(finding: HardwareCheckFinding): FindingMessage | null {
  const v = finding.values;
  switch (finding.id) {
    case "slow_link": {
      const hasBest = typeof v.best_gpu === "number" && typeof v.ratio === "number";
      const hasWidth = typeof v.width === "number";
      const h2d = formatGibs(typeof v.h2d_gibs === "number" ? v.h2d_gibs : null);
      const width = { width: val(v.width), widthMax: val(v.width_max) };
      if (hasBest) {
        const best = {
          best: formatGibs(typeof v.best_h2d_gibs === "number" ? v.best_h2d_gibs : null),
          bestGpu: val(v.best_gpu),
          ratio: val(v.ratio),
        };
        return hasWidth
          ? { key: "findings.slowLink", values: { gpu: val(v.gpu), ...width, h2d, ...best } }
          : { key: "findings.slowLinkBandwidth", values: { gpu: val(v.gpu), h2d, ...best } };
      }
      return { key: "findings.slowLinkAlone", values: { gpu: val(v.gpu), ...width, h2d } };
    }
    case "no_peer_access":
      return {
        key: v.windows === true ? "findings.noPeerAccessWindows" : "findings.noPeerAccess",
        values: {
          a: val(v.a),
          b: val(v.b),
          copy: formatGibs(typeof v.copy_gibs === "number" ? v.copy_gibs : null),
        },
      };
    case "storage_slow": {
      const kind = v.kind === "hdd" || v.kind === "usb" ? v.kind : "sata";
      const key = {
        sata: "findings.storageSata",
        hdd: "findings.storageHdd",
        usb: "findings.storageUsb",
      }[kind];
      return {
        key,
        values: {
          location: typeof v.location === "string" ? v.location : "",
          drive: typeof v.drive === "string" ? v.drive : "?",
        },
      };
    }
    case "gpu_skipped": {
      if (v.reason === "low_free_memory") {
        return {
          key: "findings.gpuSkippedLowMemory",
          values: {
            gpu: val(v.gpu),
            free: formatGibs(typeof v.free_gib === "number" ? v.free_gib : null),
          },
        };
      }
      return {
        key: v.reason === "not_visible" ? "findings.gpuSkippedHidden" : "findings.gpuFailed",
        values: { gpu: val(v.gpu) },
      };
    }
    case "gpu_probe_failed":
      return { key: "findings.probeFailed", values: { reason: val(v.reason) } };
    case "all_good":
      return { key: "findings.allGood", values: {} };
    default:
      return null;
  }
}

/** The options the result recommends that are still off: what "Apply recommended" changes. */
export function optionsToApply(status: HardwareCheckStatus | null): HardwareCheckOption[] {
  if (!status?.result || status.upToDate === false) return [];
  return HARDWARE_CHECK_OPTIONS.filter(
    (key) => status.result?.recommended[key] === true && !status.settings[key],
  );
}

/** Whether a peer-access answer is "yes" both ways, "no" either way, or unknown. */
export function peerAccess(pair: HardwareCheckPair): "yes" | "no" | "unknown" {
  if (pair.peerAb === false || pair.peerBa === false) return "no";
  if (pair.peerAb === true && pair.peerBa === true) return "yes";
  return "unknown";
}

// ── Tensor Parallelism warning ───────────────────────────────────

export type TensorSplitConcern = {
  slow: { gpu: number; width: number | null; h2d: string; best: string; bestGpu: number }[];
  noPeer: { a: number; b: number; copy: string }[];
};

/**
 * What to warn about when Tensor Parallelism is switched on by hand, across `gpuIds` (every
 * measured card when null): measured slow links and pairs without direct access. Null when there
 * is nothing to say, when "Avoid tensor parallel on slow links" is off, or when the result does
 * not describe the GPUs installed now. Mirrors hardware_check.tensor_split_concern.
 */
export function tensorSplitConcern(
  status: HardwareCheckStatus | null,
  gpuIds: readonly number[] | null,
): TensorSplitConcern | null {
  if (!status?.result || !status.settings.avoid_tensor_split || status.upToDate !== true) {
    return null;
  }
  const measured = status.result.gpus.filter(
    (gpu) => gpu.status === "measured" && gpu.h2dGibs !== null,
  );
  const ids = new Set(measured.map((gpu) => gpu.index));
  const span = gpuIds && gpuIds.length > 0 ? gpuIds.filter((id) => ids.has(id)) : [...ids];
  if (span.length < 2) return null;
  const inSpan = new Set(span);
  const best = measured.reduce<HardwareCheckGpu | null>(
    (top, gpu) => (top === null || (gpu.h2dGibs ?? 0) > (top.h2dGibs ?? 0) ? gpu : top),
    null,
  );
  const slow = measured
    .filter((gpu) => inSpan.has(gpu.index) && status.result?.slowGpus.includes(gpu.index))
    .map((gpu) => ({
      gpu: gpu.index,
      width: gpu.link?.widthCurrent ?? null,
      h2d: formatGibs(gpu.h2dGibs),
      best: formatGibs(best?.h2dGibs ?? null),
      bestGpu: best?.index ?? gpu.index,
    }));
  const noPeer = status.result.pairs
    .filter((pair) => inSpan.has(pair.a) && inSpan.has(pair.b) && peerAccess(pair) === "no")
    .map((pair) => {
      const copies = [pair.copyAbGibs, pair.copyBaGibs].filter(
        (v): v is number => typeof v === "number",
      );
      return {
        a: pair.a,
        b: pair.b,
        copy: formatGibs(copies.length ? Math.min(...copies) : null),
      };
    });
  return slow.length || noPeer.length ? { slow, noPeer } : null;
}

// ── Run state ────────────────────────────────────────────────────

/** Known reasons a run did not start; anything else is shown as the generic line. */
export const SKIP_REASONS = [
  "already_running",
  "training_active",
  "model_loading",
  "generation_active",
] as const;

export type SkipReason = (typeof SKIP_REASONS)[number];

const SKIP_KEYS: Record<SkipReason, string> = {
  already_running: "skip.alreadyRunning",
  training_active: "skip.training",
  model_loading: "skip.loading",
  generation_active: "skip.generating",
};

/** The translation key (relative to `hardwareCheck.`) for why a run did not start. */
export function skipReasonKey(reason: string | null | undefined): string | null {
  if (!reason) return null;
  return (SKIP_REASONS as readonly string[]).includes(reason)
    ? SKIP_KEYS[reason as SkipReason]
    : "skip.other";
}

/** How often to poll: quickly while a run is going, never otherwise. */
export function hardwareCheckPollMs(status: HardwareCheckStatus | null): number | null {
  return status?.state.running ? 1000 : null;
}
