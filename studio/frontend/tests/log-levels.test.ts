// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Logs: the level filter, the level colouring and Previous / Next error
// all read a line's severity from lib/log-levels. The lines below are the shapes the
// shipped logs actually contain (server JSON records and their echoed tracebacks,
// llama.cpp, uvicorn, Python logging, npm and MCP servers' own stderr).

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  adjacentAnchor,
  classifyLogLines,
  lineLevel,
  passesLevelFilter,
  segmentLines,
} from "../src/features/settings/lib/log-levels.ts";

test("each log format states its level", () => {
  const cases: [string, ReturnType<typeof lineLevel>][] = [
    ['{"timestamp": "2026-10-08T07:44:12Z", "level": "error", "event": "llama-server exited"}', "error"],
    ['{"timestamp": "2026-10-08T07:44:12Z", "level": "warning", "event": "slow"}', "warning"],
    ['{"timestamp": "2026-10-08T07:44:12Z", "level": "info", "event": "ERROR in name only"}', "info"],
    ['{"timestamp": "2026-10-08T07:44:12Z", "level": "critical", "event": "x"}', "error"],
    ["0.00.146.554 W srv  llama_server: security: no API key is set", "warning"],
    ["0.43.635.730 E main: failed to load model", "error"],
    ["0.00.501.925 I model loaded", "info"],
    ["ERROR:    Exception in ASGI application", "error"],
    ["WARNING:root:deprecated option", "warning"],
    ["2026-10-08 07:40:25,123 - mcp.server - ERROR - handler crashed", "error"],
    ["INFO:     127.0.0.1:5000 - \"GET /ERROR HTTP/1.1\" 200 OK", "info"],
    ["WARNING: llama.cpp prebuilt is 8 days behind", "warning"],
    ["[2026-10-08][07:40:25][unsloth][ERROR] backend exited", "error"],
    ["[warn] retrying connection", "warning"],
    ["time=2026-10-08T07:40:25Z level=error msg=\"dial failed\"", "error"],
    ["npm ERR! code ENOENT", "error"],
    ["npm warn deprecated inflight@1.0.6", "warning"],
    ["error: unexpected argument '--stdio' found", "error"],
    ["fatal: cannot reach upstream", "error"],
    ["warning: config file not found, using defaults", "warning"],
    ["Error: Cannot find module '@example/server'", "error"],
    ["TypeError [ERR_INVALID_ARG_TYPE]: The path must be a string", "error"],
    ["requests.exceptions.ConnectionError: HTTPSConnectionPool", "error"],
    ["/app/server.py:12: DeprecationWarning: use run() instead", "warning"],
    ["Traceback (most recent call last):", "error"],
    ["| Traceback (most recent call last):", "error"],
    ["Fatal Python error: Segmentation fault", "error"],
    ["thread 'main' panicked at src/main.rs:4:5:", "error"],
    // Prose is not a level.
    ["no error was found in the cache", null],
    ["Error loading model, retrying", null],
    ["===== 2026-10-08T08:34:52 start python.exe#6abc5b798c4d =====", null],
    ["  - loading PyTorch, Unsloth and Transformers...", null],
  ];
  for (const [line, expected] of cases) {
    assert.equal(lineLevel(line), expected, line);
  }
});

test("a traceback is one error entry, frames and exception line included", () => {
  const lines = [
    "starting",
    "Traceback (most recent call last):",
    '  File "server.py", line 3, in <module>',
    "    main()",
    "",
    "During handling of the above exception, another exception occurred:",
    '  File "server.py", line 9, in main',
    "KeyboardInterrupt",
    "after",
  ];
  const { levels, anchors, entries } = classifyLogLines(lines);
  assert.deepEqual(levels, [
    "info",
    "error",
    "error",
    "error",
    "error",
    "error",
    "error",
    "error",
    "info",
  ]);
  assert.deepEqual(anchors, [1]);
  assert.deepEqual(entries, [null, 1, 1, 1, 1, 1, 1, 1, null]);
});

test("a JSON error record and its echoed traceback are one entry", () => {
  const lines = [
    '{"level": "info", "event": "ok"}',
    '{"level": "error", "event": "Error loading model", "exception": "Traceback ..."}',
    "| Traceback (most recent call last):",
    '|   File "x.py", line 1, in <module>',
    "| ValueError: bad gguf",
    '{"level": "error", "event": "second failure"}',
    '{"level": "info", "event": "recovered"}',
  ];
  const { levels, anchors } = classifyLogLines(lines);
  assert.deepEqual(anchors, [1, 5]);
  assert.deepEqual(levels.slice(1, 6), ["error", "error", "error", "error", "error"]);
  assert.equal(levels[6], "info");
});

test("a Node stack under an error joins it, and an unrelated line ends it", () => {
  const lines = [
    "Error: connect ECONNREFUSED 127.0.0.1:5432",
    "    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1555:16)",
    "    at process.processTicksAndRejections (node:internal:1:1)",
    "listening on stdio",
    "    at not a frame of anything",
  ];
  const { levels, anchors } = classifyLogLines(lines);
  assert.deepEqual(levels, ["error", "error", "error", "info", "info"]);
  assert.deepEqual(anchors, [0]);
});

test("the level filter keeps warnings and above, or errors only", () => {
  assert.equal(passesLevelFilter("info", "all"), true);
  assert.equal(passesLevelFilter("info", "warnings"), false);
  assert.equal(passesLevelFilter("warning", "warnings"), true);
  assert.equal(passesLevelFilter("error", "warnings"), true);
  assert.equal(passesLevelFilter("warning", "errors"), false);
  assert.equal(passesLevelFilter("error", "errors"), true);
});

test("Previous / Next error step between anchors and stop at the ends", () => {
  const anchors = [3, 10, 42];
  // Nothing chosen yet: Previous starts from the newest, Next from the oldest.
  assert.equal(adjacentAnchor(anchors, null, -1), 42);
  assert.equal(adjacentAnchor(anchors, null, 1), 3);
  assert.equal(adjacentAnchor(anchors, 10, 1), 42);
  assert.equal(adjacentAnchor(anchors, 10, -1), 3);
  assert.equal(adjacentAnchor(anchors, 42, 1), null);
  assert.equal(adjacentAnchor(anchors, 3, -1), null);
  // From a line that is not an anchor (the chosen one was filtered or trimmed away).
  assert.equal(adjacentAnchor(anchors, 20, -1), 10);
  assert.equal(adjacentAnchor([], null, 1), null);
});

test("segments group runs of one level and give every error entry its own", () => {
  const lines = [
    "a",
    "b",
    "ERROR: first",
    "    at frame",
    "ERROR: second",
    "WARNING: w1",
    "WARNING: w2",
    "c",
  ];
  const classified = classifyLogLines(lines);
  const visible = lines.map((_, index) => index);
  const segments = segmentLines(lines, visible, classified);
  assert.deepEqual(
    segments.map(({ start, level, text, entry }) => [start, level, text, entry]),
    [
      [0, "info", "a\nb", null],
      [2, "error", "ERROR: first\n    at frame", 2],
      [4, "error", "ERROR: second", 4],
      [5, "warning", "WARNING: w1\nWARNING: w2", null],
      [7, "info", "c", null],
    ],
  );
  // Filtered to errors only: two entries, still separate.
  const errors = visible.filter((index) => classified.levels[index] === "error");
  assert.deepEqual(
    segmentLines(lines, errors, classified).map((segment) => segment.start),
    [2, 4],
  );
});

test("the viewer wires the level filter and error navigation", async () => {
  const tab = await readFile(
    new URL("../src/features/settings/tabs/debugging-tab.tsx", import.meta.url),
    "utf8",
  );
  for (const needle of [
    'from "../lib/log-levels"',
    "classifyLogLines(plainLines)",
    "passesLevelFilter(",
    "segmentLines(",
    "adjacentAnchor(",
    'data-testid="debug-log-level-filter"',
    'data-testid="debug-log-previous-error"',
    'data-testid="debug-log-next-error"',
    "data-log-line=",
    // Navigating away from the bottom must stop Live mode yanking the view back.
    "pinnedRef.current = false;",
    // An MCP log is shown under its server's name.
    "source.displayName ?? source.label",
  ]) {
    assert.ok(tab.includes(needle), `debugging-tab.tsx lost: ${needle}`);
  }
  // Still one surface for the copy button: the visible lines, not the segments.
  assert.match(tab, /copy\(text\)/);
});
