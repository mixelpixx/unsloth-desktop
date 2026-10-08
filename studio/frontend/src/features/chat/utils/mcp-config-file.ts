// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Each pass walks the text once and copies string literals verbatim, so "//", "/*" or ",}" inside a
// value (a URL, a Windows path, an argument) is never touched.
function forEachOutsideStrings(
  text: string,
  visit: (index: number) => number | null,
): string {
  let out = "";
  let i = 0;
  while (i < text.length) {
    if (text[i] === '"') {
      const start = i++;
      while (i < text.length && text[i] !== '"') i += text[i] === "\\" ? 2 : 1;
      i += 1;
      out += text.slice(start, i);
      continue;
    }
    const skipTo = visit(i);
    if (skipTo !== null) {
      i = skipTo;
      continue;
    }
    out += text[i];
    i += 1;
  }
  return out;
}

function stripComments(text: string): string {
  return forEachOutsideStrings(text, (i) => {
    if (text[i] !== "/") return null;
    if (text[i + 1] === "/") {
      const end = text.indexOf("\n", i);
      return end === -1 ? text.length : end;
    }
    if (text[i + 1] === "*") {
      const end = text.indexOf("*/", i + 2);
      return end === -1 ? text.length : end + 2;
    }
    return null;
  });
}

function stripTrailingCommas(text: string): string {
  return forEachOutsideStrings(text, (i) => {
    if (text[i] !== ",") return null;
    let j = i + 1;
    while (j < text.length && /\s/.test(text[j])) j += 1;
    return text[j] === "}" || text[j] === "]" ? i + 1 : null;
  });
}

/** Parse an MCP config file the way editors write them, for the Import config button.
 *
 * VS Code's mcp.json and settings.json are JSONC (comments and trailing commas are legal), Notepad
 * saves UTF-8 with a BOM, and settings.json nests servers under "mcp": { "servers": ... }. Plain
 * JSON.parse rejects all three with "Invalid JSON", although the backend parser accepts the
 * servers they hold. Throws the JSON.parse error (which names the position) for real syntax errors.
 */
export function parseMcpConfigFile(text: string): unknown {
  const body = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const parsed: unknown = JSON.parse(stripTrailingCommas(stripComments(body)));
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const record = parsed as Record<string, unknown>;
    const nested = record.mcp as Record<string, unknown> | undefined;
    if (
      !("mcpServers" in record) &&
      !("servers" in record) &&
      nested &&
      typeof nested === "object" &&
      nested.servers &&
      typeof nested.servers === "object"
    ) {
      return { servers: nested.servers };
    }
  }
  return parsed;
}
