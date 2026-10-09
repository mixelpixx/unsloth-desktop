# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, StrictStr


class McpImageInputMapping(BaseModel):
    """A top-level string field of ``tool`` that receives the user's approved image."""

    tool: StrictStr = Field(min_length = 1, max_length = 256)
    field: StrictStr = Field(min_length = 1, max_length = 256)
    encoding: Literal["base64", "data_url"] = "base64"


# A local program's lifecycle (core.inference.mcp_client: PROCESS_MODES, IDLE_TIMEOUT_CHOICES). Seconds; 0 = never.
McpProcessMode = Literal["shared", "per_chat"]
McpIdleTimeout = Literal[60, 300, 1800, 7200, 0]

# Raw MCP tool names as the server lists them (core.inference.mcp_client: MAX_TOOL_SETTING_NAMES).
McpToolName = Annotated[StrictStr, Field(min_length = 1, max_length = 256)]
McpToolNames = Annotated[list[McpToolName], Field(max_length = 2000)]


class McpServerCreate(BaseModel):
    display_name: str
    url: str
    headers: Optional[dict[str, str]] = None
    is_enabled: bool = True
    use_oauth: bool = False
    # Local programs only: the absolute folder the program starts in. None = the backend's own.
    cwd: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    image_input_mappings: list[McpImageInputMapping] = Field(default_factory = list, max_length = 64)
    # Local programs only. None: "shared" for a local program; the idle timeout then defaults per mode.
    process_mode: Optional[McpProcessMode] = None
    idle_timeout_seconds: Optional[McpIdleTimeout] = None


class McpServerUpdate(BaseModel):
    display_name: Optional[str] = None
    url: Optional[str] = None
    # Absent in request body = leave as-is; null = drop all headers; dict = set.
    headers: Optional[dict[str, str]] = None
    is_enabled: Optional[bool] = None
    use_oauth: Optional[bool] = None
    # Absent = leave as-is; null or blank = clear.
    cwd: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    image_input_mappings: Optional[list[McpImageInputMapping]] = Field(None, max_length = 64)
    # Absent = leave as-is. A new mode ends the processes of the old one; a new idle timeout reaches a running
    # process on its next use.
    process_mode: Optional[McpProcessMode] = None
    idle_timeout_seconds: Optional[McpIdleTimeout] = None
    # Absent = leave as-is; null or [] = none. The whole set each time: the tools never offered to a model (and
    # refused if one calls them anyway), and the tools that always ask first under "Approve for me".
    disabled_tools: Optional[McpToolNames] = None
    ask_tools: Optional[McpToolNames] = None


class McpServerResponse(BaseModel):
    id: str
    builtin_id: Optional[str] = None
    display_name: str
    url: str
    headers: dict[str, str] = Field(default_factory = dict)
    is_enabled: bool = True
    use_oauth: bool = False
    cwd: Optional[str] = None
    oauth_client_id: Optional[str] = None
    has_oauth_client_secret: bool = False
    image_input_mappings: list[McpImageInputMapping] = Field(default_factory = list)
    # False when no mapping matches a cached tool schema any more; true while the tools are unknown.
    image_mappings_active: bool = False
    # Resolved: a row that never chose an idle timeout reports its mode's default. HTTP servers ignore both.
    process_mode: McpProcessMode = "per_chat"
    idle_timeout_seconds: int = 300
    # Raw tool names, sorted. Every tool not listed in disabled_tools is on, including ones the server adds later.
    disabled_tools: list[str] = Field(default_factory = list)
    ask_tools: list[str] = Field(default_factory = list)
    created_at: str
    updated_at: str


class McpToolEntry(BaseModel):
    """One tool a server lists for the model, as the per-tool controls show it."""

    name: str
    title: Optional[str] = None
    # Plain text from a third-party server: first sentence, and the full text capped for a tooltip.
    summary: str = ""
    description: str = ""
    enabled: bool = True
    ask: bool = False
    # What this tool's schema adds to every request while it is on.
    tokens: int = 0


class McpToolCatalog(BaseModel):
    server_id: str
    # False until the server's tools have been discovered (a chat used it, or Refresh); tools is then empty.
    cached: bool = False
    # The server announced a tool-list change since this list was read; the next chat or Refresh re-reads it.
    stale: bool = False
    tools: list[McpToolEntry] = Field(default_factory = list)
    enabled_count: int = 0
    total_count: int = 0
    enabled_tokens: int = 0
    # Counted with the loaded model's tokenizer; otherwise estimated from characters.
    tokens_measured: bool = False
    # The loaded model's context window, when one is known.
    context_tokens: Optional[int] = None
    # Turned-off names the server does not list (any more); kept, so the tool stays off if it comes back.
    unlisted_disabled: list[str] = Field(default_factory = list)


class McpServerStatus(BaseModel):
    """A local program's processes now (core.inference.mcp_client.server_status)."""

    server_id: str
    state: Literal["running", "idle", "stopped", "failed"]
    process_mode: McpProcessMode
    idle_timeout_seconds: int
    # Live processes: 0 or 1 when shared, one per chat with a process when per chat.
    processes: int = 0
    started_at: Optional[float] = None
    uptime_seconds: Optional[float] = None
    idle_seconds: Optional[float] = None
    # Until the idle timeout stops it; None while busy, stopped, or set to never.
    stops_in_seconds: Optional[float] = None
    busy_tool: Optional[str] = None
    busy_seconds: Optional[float] = None
    last_error: Optional[str] = None
    last_error_at: Optional[float] = None
    # The program's stderr log, for Settings > Logs. Only the installation owner's UI session gets a local path.
    log_path: Optional[str] = None


class McpServerTestRequest(BaseModel):
    url: str
    headers: Optional[dict[str, str]] = None
    use_oauth: bool = False
    cwd: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    # Edit form: reuse this server's stored secret when the secret field is left blank.
    server_id: Optional[str] = None


class BlenderSettings(BaseModel):
    model_config = {"extra": "forbid"}

    port: int = Field(default = 9876, ge = 1, le = 65535, strict = True)
    blender_path: StrictStr = Field(default = "", pattern = r"^[^\x00]*$")


class BlenderTest(BlenderSettings):
    consent: bool = Field(default = False, strict = True)


class BlenderSetup(BlenderSettings):
    is_enabled: bool
    consent: bool = False


class McpBuiltinResponse(BlenderSettings):
    builtin_id: str = "blender"
    display_name: str = "Blender"
    server_id: Optional[str] = None
    is_enabled: bool = False
    available: bool
    unavailable_reason: Optional[str] = None
    min_blender_version: str


class McpStdioDecodeRequest(BaseModel):
    url: StrictStr


class McpStdioCommand(BaseModel):
    command: StrictStr
    arguments: list[StrictStr] = Field(default_factory = list)


class McpStdioEncodeResponse(BaseModel):
    url: str


class McpCapabilities(BaseModel):
    # Whether this caller may add local-program (stdio) servers right now, and why not, so the dialog can
    # say so before the user fills in an executable instead of after Save.
    stdio_enabled: bool
    stdio_disabled_reason: Optional[str] = None


class McpServerProbeResult(BaseModel):
    ok: bool
    tool_count: int = 0
    error: Optional[str] = None
    blender_ready: Optional[bool] = None
    blender_error: Optional[str] = None


class McpServerImportRequest(BaseModel):
    # A standard mcpServers JSON config (Claude Desktop / Cursor / Cline / VS Code).
    config: dict


class McpServerImportResult(BaseModel):
    created: list[McpServerResponse] = Field(default_factory = list)
    skipped: list[str] = Field(default_factory = list)
    errors: list[str] = Field(default_factory = list)


class McpImportSourceServer(BaseModel):
    """One server another app has configured. Never a value from its env or headers: those are read
    and stored on the server, and the dialog gets their names only."""

    name: str
    transport: Literal["stdio", "http"]
    # The command line or URL as the app wrote it, credentials masked.
    target: str
    env_keys: list[str] = Field(default_factory = list)
    header_keys: list[str] = Field(default_factory = list)
    # Studio already has a server with this command line or URL.
    already_added: bool = False
    importable: bool = True
    # Why it can't be imported, or why it will arrive switched off.
    note: Optional[str] = None


class McpImportSource(BaseModel):
    id: str
    app: str
    label: Optional[str] = None
    # The config file, for the installation owner's own UI session only.
    path: Optional[str] = None
    # The file exists but couldn't be read; servers is then empty.
    error: Optional[str] = None
    servers: list[McpImportSourceServer] = Field(default_factory = list)


class McpImportSourcesResponse(BaseModel):
    sources: list[McpImportSource] = Field(default_factory = list)


class McpImportSourceApplyRequest(BaseModel):
    source_id: StrictStr = Field(min_length = 1, max_length = 256)
    server_names: list[StrictStr] = Field(min_length = 1, max_length = 500)


class McpImportServerOutcome(BaseModel):
    name: str
    # added_disabled: created switched off, detail says why. duplicate: Studio already has it.
    status: Literal["added", "added_disabled", "duplicate", "error"]
    detail: Optional[str] = None
    server_id: Optional[str] = None


class McpImportSourceApplyResult(BaseModel):
    results: list[McpImportServerOutcome] = Field(default_factory = list)


class McpUiResourceResponse(BaseModel):
    uri: str
    mime_type: str
    text: str
    # Base64, only for a resource that is not UTF-8 text.
    blob: Optional[str] = None
    ui: dict = Field(default_factory = dict)
    contents: list[dict] = Field(default_factory = list)


class McpUiToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory = dict)
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    permission_mode: Optional[str] = None
    approved: bool = False


class McpUiToolCallResult(BaseModel):
    content: list[dict] = Field(default_factory = list)
    structured_content: Optional[dict] = None
    is_error: bool = False
    meta: Optional[dict] = None
