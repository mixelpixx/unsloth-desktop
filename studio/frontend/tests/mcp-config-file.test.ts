import assert from "node:assert/strict";
import test from "node:test";

import { parseMcpConfigFile } from "../src/features/chat/utils/mcp-config-file.ts";

test("plain Claude Desktop JSON parses unchanged", () => {
  const text = JSON.stringify({
    mcpServers: { fs: { command: "C:\\tools\\server.exe", args: ["--stdio"] } },
  });
  assert.deepEqual(parseMcpConfigFile(text), JSON.parse(text));
});

test("JSONC comments and trailing commas are accepted", () => {
  const text = `{
    // VS Code mcp.json
    "servers": {
      /* local program */
      "fs": { "command": "npx", "args": ["-y", "@scope/server",], },
    },
  }`;
  assert.deepEqual(parseMcpConfigFile(text), {
    servers: { fs: { command: "npx", args: ["-y", "@scope/server"] } },
  });
});

test("comment and comma look-alikes inside strings are preserved", () => {
  const text = `{"servers": {"web": {"url": "https://example.com/mcp//x/*y*/", "headers": {"X": "a,}"}}}}`;
  assert.deepEqual(parseMcpConfigFile(text), {
    servers: {
      web: { url: "https://example.com/mcp//x/*y*/", headers: { X: "a,}" } },
    },
  });
});

test("escaped quotes and backslash paths survive", () => {
  const text = String.raw`{"mcpServers": {"x": {"command": "C:\\Program Files\\x\\server.exe", "args": ["say \"hi\" // not a comment"]}}}`;
  const parsed = parseMcpConfigFile(text) as {
    mcpServers: { x: { command: string; args: string[] } };
  };
  assert.equal(parsed.mcpServers.x.command, "C:\\Program Files\\x\\server.exe");
  assert.equal(parsed.mcpServers.x.args[0], 'say "hi" // not a comment');
});

test("a UTF-8 BOM from Notepad is ignored", () => {
  assert.deepEqual(parseMcpConfigFile('\uFEFF{"mcpServers": {}}'), {
    mcpServers: {},
  });
});

test("VS Code settings.json nesting is unwrapped", () => {
  const text = `{
    "editor.fontSize": 14,
    "mcp": { "servers": { "gh": { "url": "https://api.example.com/mcp" } } },
  }`;
  assert.deepEqual(parseMcpConfigFile(text), {
    servers: { gh: { url: "https://api.example.com/mcp" } },
  });
});

test("a top-level server map wins over nested settings", () => {
  const text = `{"mcpServers": {"a": {"url": "https://a/mcp"}}, "mcp": {"servers": {"b": {}}}}`;
  assert.deepEqual(Object.keys((parseMcpConfigFile(text) as { mcpServers: object }).mcpServers), ["a"]);
});

test("real syntax errors still throw", () => {
  assert.throws(() => parseMcpConfigFile('{"mcpServers": {'));
});
