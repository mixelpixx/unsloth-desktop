# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Literal, Optional

from pydantic import BaseModel, Field, StrictStr


class McpImageInputMapping(BaseModel):
    """A top-level string field of ``tool`` that receives the user's approved image."""

    tool: StrictStr = Field(min_length = 1, max_length = 256)
    field: StrictStr = Field(min_length = 1, max_length = 256)
    encoding: Literal["base64", "data_url"] = "base64"


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
    created_at: str
    updated_at: str


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
