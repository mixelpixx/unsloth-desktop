// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Severity of each line in the Settings > Logs viewer, read from the formats the logs
 * actually use. Kept free of React, like debug-log-buffer, so node:test can import it.
 *
 * The logs share no format: the server log is structlog JSON records with plain
 * tracebacks echoed under them ("| " prefixed), the llama runner prints llama.cpp's
 * "<time> W msg", uvicorn and Python's logging print "ERROR:", npm prints "npm ERR!",
 * and an MCP server's stderr is whatever its author chose. So a line is read by the
 * first rule that recognises it, and a line nothing recognises takes the level of the
 * entry it continues when it is plainly part of it (a traceback frame), else "info".
 */

export type LogLevel = "error" | "warning" | "info";
export type LevelFilter = "all" | "warnings" | "errors";

export const LEVEL_FILTERS: LevelFilter[] = ["all", "warnings", "errors"];

const ERROR_NAMES = new Set([
  "err",
  "error",
  "fatal",
  "critical",
  "crit",
  "panic",
  "severe",
  "alert",
  "emerg",
  "exception",
]);
const WARNING_NAMES = new Set(["warn", "warning"]);

function levelFromName(name: string): LogLevel {
  const lowered = name.toLowerCase();
  if (ERROR_NAMES.has(lowered)) return "error";
  if (WARNING_NAMES.has(lowered)) return "warning";
  return "info";
}

// The start of a multi-line error, bare or echoed under a JSON record.
const TRACEBACK_START = /^(?:\| )?Traceback \(most recent call last\):/;
const FATAL_MARKERS =
  /^(?:Fatal Python error:|Segmentation fault|thread '.*' panicked at)/;
// structlog's "level" field, matched rather than parsed: a record can be cut at
// the server's line limit and still name its level near the start.
const JSON_LEVEL = /"level"\s*:\s*"([A-Za-z]+)"/;
// llama.cpp's common_log with timestamps: "0.00.146.554 W srv  llama_server: ...".
const LLAMA_LEVEL = /^\d+\.\d{2}\.\d{3}\.\d{3} ([EWID]) /;
const BRACKETED_LEVEL =
  /\[(error|err|fatal|critical|crit|panic|warn|warning)\]/i;
const LOGFMT_LEVEL = /(?:^|\s)level=["']?([A-Za-z]+)/i;
const NPM_LEVEL = /^npm (ERR!|error|WARN|warn)\b/;
// A CLI's own prefix, any case: "error: ...", "fatal: ...", "warning: ...", "error[E0382]: ...".
// Anchored at the start with its colon: a sentence rarely opens that way.
const PREFIX_LEVEL =
  /^(error|fatal|panic|critical|warning|warn)(?:\[[\w-]+\])?:\s/i;
// An upper-case level word among the first tokens: uvicorn's "ERROR:    ...",
// logging's "WARNING:root:...", "2026-10-08 07:40:25,123 - name - ERROR - ...".
// Upper case only, because "error" in a sentence is not a level, and the FIRST one
// wins, so "INFO retrying after ERROR" stays info.
const LEVEL_WORD =
  /(?:^|[\s[|:(-])(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERR|ERROR|CRITICAL|FATAL|PANIC|SEVERE)(?=$|[\s\]|:!)-])/;
const LEVEL_WORD_WINDOW = 80;
// The line that names an exception: Python's last traceback line, Node's thrown error.
const EXCEPTION_LINE =
  /^(?:Uncaught )?[A-Za-z_][\w.]*(?:Error|Exception)(?:\s*\[[\w-]+\])?:\s/;
// Python's warnings module: "path.py:12: DeprecationWarning: ...".
const PYTHON_WARNING = /(?:^|:\d+: )[A-Z]\w*Warning: /;
// One frame of a stack printed under an error line (Node, Java, Python's echo).
const STACK_FRAME = /^\s+(?:at |File "|\.\.\. \d+ more)/;
const TRACEBACK_JOINER = /^(?:During handling of|The above exception was)/;

/** The level a line states itself, or null when it states none. */
export function lineLevel(line: string): LogLevel | null {
  if (TRACEBACK_START.test(line) || FATAL_MARKERS.test(line)) return "error";
  if (line.trimStart().startsWith("{")) {
    const match = JSON_LEVEL.exec(line);
    return match ? levelFromName(match[1]) : null;
  }
  const llama = LLAMA_LEVEL.exec(line);
  if (llama) {
    return llama[1] === "E" ? "error" : llama[1] === "W" ? "warning" : "info";
  }
  const bracketed = BRACKETED_LEVEL.exec(line);
  if (bracketed) return levelFromName(bracketed[1]);
  const logfmt = LOGFMT_LEVEL.exec(line);
  if (logfmt) return levelFromName(logfmt[1]);
  const npm = NPM_LEVEL.exec(line);
  if (npm) return npm[1].toLowerCase().startsWith("err") ? "error" : "warning";
  const prefix = PREFIX_LEVEL.exec(line);
  if (prefix) return levelFromName(prefix[1]);
  const word = LEVEL_WORD.exec(line.slice(0, LEVEL_WORD_WINDOW));
  if (word) return levelFromName(word[1]);
  if (EXCEPTION_LINE.test(line)) return "error";
  if (PYTHON_WARNING.test(line)) return "warning";
  return null;
}

export interface ClassifiedLines {
  /** One level per input line. */
  levels: LogLevel[];
  /** Indices of the lines that START an error, ascending: what Previous / Next
   * error steps between, so a 30-line traceback is one stop, not thirty. */
  anchors: number[];
  /** Per line, the anchor of the error it belongs to (itself for an anchor), else null. */
  entries: (number | null)[];
}

// The unindented line ending a plain traceback names the exception, with or without
// "Error" in its name ("KeyboardInterrupt", "SystemExit: 2").
const EXCEPTION_NAME = /^[A-Za-z_][\w.]*(?::\s|$)/;

/** Every line's level, with tracebacks and stack frames joined to the error they belong to. */
export function classifyLogLines(lines: readonly string[]): ClassifiedLines {
  const levels: LogLevel[] = new Array(lines.length);
  const entries: (number | null)[] = new Array(lines.length);
  const anchors: number[] = [];
  // Inside a traceback: "echo" lines all carry the "| " prefix, a plain one runs
  // through its indented frames to the unindented line naming the exception.
  let traceback: "echo" | "plain" | null = null;
  let previous: LogLevel = "info";
  let entry: number | null = null;
  const join = (index: number) => {
    levels[index] = "error";
    entries[index] = entry;
  };
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (traceback) {
      const continues =
        traceback === "echo"
          ? line.startsWith("|")
          : line === "" || /^\s/.test(line) || TRACEBACK_JOINER.test(line);
      if (continues) {
        join(index);
        continue;
      }
      const wasPlain = traceback === "plain";
      traceback = null;
      if (
        wasPlain &&
        (EXCEPTION_LINE.test(line) ||
          (lineLevel(line) === null && EXCEPTION_NAME.test(line)))
      ) {
        join(index);
        continue;
      }
    }
    const own = lineLevel(line);
    if (own === null) {
      if (previous === "error" && STACK_FRAME.test(line)) {
        join(index);
        continue;
      }
      levels[index] = "info";
      entries[index] = null;
      previous = "info";
      entry = null;
      continue;
    }
    levels[index] = own;
    const startsTraceback = TRACEBACK_START.test(line);
    if (startsTraceback) traceback = line.startsWith("|") ? "echo" : "plain";
    if (own === "error") {
      // A traceback straight after an error line is that error's (a JSON record and
      // its echo, uvicorn's "Exception in ASGI application" and its stack).
      if (!(startsTraceback && previous === "error")) {
        anchors.push(index);
        entry = index;
      }
      entries[index] = entry;
    } else {
      entries[index] = null;
      entry = null;
    }
    previous = own;
  }
  return { levels, anchors, entries };
}

export function passesLevelFilter(level: LogLevel, filter: LevelFilter): boolean {
  if (filter === "errors") return level === "error";
  if (filter === "warnings") return level !== "info";
  return true;
}

/** The anchor before or after `current` (a line index, or null for none chosen
 * yet), or null at either end. With nothing chosen, Previous starts from the
 * newest error, since the pane opens following the bottom, and Next from the oldest. */
export function adjacentAnchor(
  anchors: readonly number[],
  current: number | null,
  direction: 1 | -1,
): number | null {
  if (anchors.length === 0) return null;
  if (current === null) {
    return direction === 1 ? anchors[0] : anchors[anchors.length - 1];
  }
  if (direction === 1) {
    for (const anchor of anchors) if (anchor > current) return anchor;
    return null;
  }
  for (let index = anchors.length - 1; index >= 0; index -= 1) {
    if (anchors[index] < current) return anchors[index];
  }
  return null;
}

export interface LogSegment {
  /** Index of the segment's first line in the buffer. */
  start: number;
  level: LogLevel;
  text: string;
  /** The anchor of the error entry this segment belongs to, for highlighting it whole. */
  entry: number | null;
}

/** The visible lines as runs of one level, an anchor always starting a new run.
 *
 * The pane stays a few text nodes rather than one element per line (a thousand
 * nodes repainted every poll is what makes a log pane feel broken), yet every
 * error still has an element of its own to scroll to and highlight.
 */
export function segmentLines(
  lines: readonly string[],
  visible: readonly number[],
  classified: ClassifiedLines,
): LogSegment[] {
  const segments: LogSegment[] = [];
  let parts: string[] = [];
  let head: Omit<LogSegment, "text"> | null = null;
  for (const index of visible) {
    const level = classified.levels[index] ?? "info";
    const entry = classified.entries[index] ?? null;
    // An anchor is its own entry, so it always differs from the run before it.
    if (!head || head.level !== level || head.entry !== entry) {
      if (head) segments.push({ ...head, text: parts.join("\n") });
      head = { start: index, level, entry };
      parts = [];
    }
    parts.push(lines[index]);
  }
  if (head) segments.push({ ...head, text: parts.join("\n") });
  return segments;
}
