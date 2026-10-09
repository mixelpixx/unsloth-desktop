// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// vite replaces `import.meta.env` at build time; bare node leaves it undefined, so a src
// module that reads it throws on import. Seed it instead, so a test can drive the real
// module rather than assert on its source. Register before importing any such module.
// Only the app's own src/: matching "/src/" anywhere also caught node_modules when the
// checkout itself sits under a src folder (D:\dev\src\...), and a CommonJS module there
// has a null source.
const APP_SRC = new URL("../../src/", import.meta.url).href;

export async function load(url, context, next) {
  const result = await next(url, context);
  if (!url.startsWith(APP_SRC) || result.source == null) return result;
  const source =
    typeof result.source === "string"
      ? result.source
      : Buffer.from(result.source).toString("utf8");
  if (!source.includes("import.meta.env")) return result;
  const seed =
    'import.meta.env = import.meta.env ?? { MODE: "test", DEV: false, PROD: false, BASE_URL: "/" };\n';
  return { ...result, source: seed + source };
}
