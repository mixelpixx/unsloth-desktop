# MCP in Unsloth Studio

## Add a local program (.exe, npx, uvx)

Open **Manage MCP servers → Add server** and enter the program in **URL or executable**,
for example `C:\tools\my-mcp\server.exe`, `npx` or `uvx`. Put each argument in its own
**Arguments** row, or paste a whole command line such as
`npx -y @modelcontextprotocol/server-filesystem C:\notes` and it is split for you. A path
copied with Explorer's **Copy as path** (with quotes) works as-is. Environment variables go
in the rows below. **Test connection** starts the program once and lists its tools.

Local programs run with your account's access, so they are only allowed when Unsloth is
on this computer alone. The default `unsloth studio` (which binds `127.0.0.1`) and the
desktop app allow them. A network bind such as `unsloth studio -H 0.0.0.0` turns them off,
and the dialog says so before you fill anything in. To use them anyway on a network bind,
set `UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1` before starting Unsloth.

You can also import an existing `mcpServers` JSON config (Claude Desktop, Cursor, VS Code)
with **Import config**. An entry's `cwd` is imported too.

**Working directory** (optional) is the folder the program starts in, for servers that read
a config or data file next to themselves. It must be the full path to an existing folder.
Left empty, the program starts in Unsloth's own working directory, as before.

The program gets the environment variables you set, plus the system ones programs commonly
need: `ProgramFiles`, `ProgramData`, `ComSpec`, `windir`, `TMP` and similar on Windows, and
the proxy and certificate variables (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`, ...) everywhere. A variable you set wins over the
inherited one. Unsloth's own settings and keys are never passed on.

### When a local program won't start

**Test connection** says why: the program was not found, it exited during startup (with the
last lines it printed), or it never answered the MCP handshake (it may be waiting for input,
downloading on first run, or not be an MCP stdio server). Values of your environment
variables are masked as `***` in that output.

Each program's error output (stderr) is kept in `logs/mcp/<program>-<id>.log` in the Unsloth
Studio folder (for example `~/.unsloth/studio`). A log is cleared when it grows past 2 MB.
It is masked as it is written: your environment variable values (4+ characters), credentials in
the command (after `--token`-style flags, in URLs) and recognisable tokens (`hf_...`, `sk-...`,
`Bearer ...`) are replaced before a line reaches the file. Settings › Logs lists these logs under
the server's name.

A stdio server must print only MCP messages on stdout. Other text there (a banner, progress)
is skipped and logged once per process.

When you enter a command line through the API or a config file, wrap a program path that
contains spaces in double quotes: `"C:\My Tools\server.exe" --stdio`. Without them, Windows
would guess which part is the program, so Unsloth refuses it and shows the quoted form.

## Connect Blender MCP

Blender MCP is **disabled by default**. Unsloth Studio downloads a pinned, checksum-verified
runtime on first enable/test and caches it on the backend machine. No commands,
Git or pip installs are needed. Subsequent starts use the cache without internet.
The Blender add-on is installed separately. No MCP archive or source is shipped
in Unsloth Studio's Python package or desktop build.

1. Open **Manage MCP servers → Blender**, approve the execution warning and choose
   **Enable Blender MCP**. Use a tool-capable model with MCP enabled for the chat.
2. Open **Setup help → Download Blender add-on** for
   [Blender's official page](https://www.blender.org/lab/mcp-server/).
3. In Blender 5.1+, enable **Preferences → System → Network → Allow Online Access**.
   Drag the website's install button into Blender twice: first to add the Blender
   Lab repository, then to install MCP. Alternatively, search **MCP** in
   **Get Extensions** after adding the repository.
4. Enable and start the add-on bridge, keep Blender open, then **Test connection**.

A green dot means Blender is connected; amber means only the MCP server is connected.
Setup help stays in the same Blender entry. Advanced settings configure the bridge
port (default `9876`) and optional Blender executable. This port is not an HTTP URL.

The bridge uses loopback on the **Unsloth Studio backend machine**, not a remote browser.
Unsloth Studio does not install or launch Blender during setup. Approved tools can run
Python, write files and launch background Blender. Existing tool permissions apply;
external model providers receive tool results. Keep the unauthenticated bridge local.

The downloaded runtime excludes the large API/manual reference corpus and its three
offline documentation tools. The official source is
https://projects.blender.org/lab/blender_mcp (GPL-3.0-or-later).
The pinned revision and SHA-256 are in `backend/integrations/blender/runtime.py`.
Downloads are staged and verified before activation; failures leave the server
disabled and can be retried with **Enable Blender MCP**. Merely opening the dialog
or launching Unsloth Studio does not download anything.

## Large tool catalogs

A server can expose dozens of tools with long descriptions and deeply nested
parameter schemas: Notion's catalog alone is about 65,000 tokens. Every tool is
listed in full whenever the catalog fits the loaded local model's context window,
so a model that can hold the full listing always gets it.

When the full listing would take more than three quarters of the window, which
would otherwise get even a short prompt refused, the largest tools (only those
whose description and schema together exceed about 1,500 characters) are listed
in a compact form, largest first, until the listing fits: a compact tool shows its
first sentence plus its top-level parameters with their types, required flags and
short enums. Every other tool keeps its full schema. The model then also gets `mcp_tool_schema`, which returns a tool's
full description and JSON Schema on demand, in pages when it is longer than the
room left for a tool result. A compact tool called without one of its required
arguments, or whose call the server rejects, answers with that schema so the model
can correct the call. Arguments to a compact tool are still typed against its full
schema. External providers always get the full listing.

## Unsloth Decisions MCP

When the Decision API is on (**Settings → API**), the chat's MCP menu lists
**Unsloth Decisions**. Enable it and a tool-capable chat model can call `decide`,
which asks the local Laya model the same typed questions `POST /v1/systemone`
answers (`noul`, `choice` and `score`), with the model chosen in Settings.

Other MCP clients reach the same tool at `http://127.0.0.1:8888/mcp/decisions/`
(use the actual Unsloth port). It takes the credentials `/v1/systemone` takes, so
send an Unsloth API key as `Authorization: Bearer sk-unsloth-...`.

<a id="studios-own-mcp-server"></a>

## Unsloth Studio's own MCP server

Unsloth can expose a local MCP server so an MCP client can inspect models and
GPU state, validate recipes, start or stop training, inspect recipe output, and
export a loaded model.

The server is disabled by default. Enable it for a local Unsloth process with:

```bash
UNSLOTH_STUDIO_ENABLE_MCP=1 \
UNSLOTH_STUDIO_MCP_TOKEN='use-a-local-secret' \
unsloth studio
```

The endpoint is `http://127.0.0.1:8888/mcp/` when Unsloth uses its default port
(a request to `/mcp` redirects to the canonical `/mcp/`). Use the actual Unsloth
port when it is configured differently.

The high-impact tools are:

- `studio_status` and `list_local_models` for discovery
- `get_training_status`, `start_training`, `stop_training`, and `list_training_runs`
- `validate_recipe`, `get_recipe_job_status`, and `get_recipe_job_dataset`
- `load_checkpoint` and `export_gguf`

`start_training` accepts the same fields as the Unsloth `TrainingStartRequest`.
The request is validated by the existing Pydantic model before a subprocess is
started. Export paths use the existing Unsloth validation as well.

The endpoint always requires `UNSLOTH_STUDIO_MCP_TOKEN` and checks an exact
Bearer token for both HTTP and WebSocket connections. Keep it on localhost
unless the deployment has an authenticated reverse proxy. The MCP endpoint is
intentionally opt-in because tools can consume GPU memory, write model
artifacts, and stop active work.