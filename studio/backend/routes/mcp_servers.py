# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import os
import re
import sys
import uuid
from typing import Annotated, Optional
from urllib.parse import urlparse

import structlog
from fastapi import APIRouter, Depends, HTTPException
from integrations.blender import service as blender
from models.mcp_servers import BlenderSettings, BlenderSetup, McpBuiltinResponse

from auth.authentication import (
    authenticated_via_api_key,
    get_current_subject,
    request_admitted_without_credential,
    require_ui_session_for_local_commands,
)
from core.inference.mcp_client import (
    PROCESS_MODE_PER_CHAT,
    PROCESS_MODE_SHARED,
    TOOL_CACHE_INVALIDATING_FIELDS,
    McpServerBusy,
    McpStdioServerError,
    UI_RESOURCE_SCHEME,
    cache_tools,
    call_tool_structured_sync,
    clear_server_failure,
    get_cached_tools,
    in_failure_cooloff,
    clear_oauth_tokens_async,
    close_mcp_sessions,
    invalidate_tool_cache,
    is_stdio,
    join_stdio_command,
    list_session_tools_sync,
    list_tools_async,
    oauth_client_kwargs,
    parse_server_headers,
    parse_stdio_command,
    probe_timeout,
    read_resource_sync,
    record_probe_failure,
    record_server_failure,
    serialize_mcp_server_mutation,
    server_idle_timeout,
    server_lifecycle,
    server_process_mode,
    server_status,
    shared_session_scope,
    stdio_mcp_disabled_reason,
    stdio_mcp_enabled,
    retime_server_processes,
    stop_server_processes,
    tool_visible_to,
    tools_cache_epoch,
    unquoted_spaced_program,
)
from core.inference import mcp_import_sources
from core.inference.mcp_config_import import parse_mcp_config
from core.inference.mcp_image import image_input_mappings, image_mapping
from models.mcp_servers import (
    BlenderTest,
    McpCapabilities,
    McpImportServerOutcome,
    McpImportSource,
    McpImportSourceApplyRequest,
    McpImportSourceApplyResult,
    McpImportSourceServer,
    McpImportSourcesResponse,
    McpServerCreate,
    McpServerImportRequest,
    McpServerImportResult,
    McpServerProbeResult,
    McpServerResponse,
    McpServerStatus,
    McpServerTestRequest,
    McpServerUpdate,
    McpStdioCommand,
    McpStdioDecodeRequest,
    McpStdioEncodeResponse,
    McpUiResourceResponse,
    McpUiToolCallRequest,
    McpUiToolCallResult,
)
from storage import mcp_servers_db
from utils.account_context import is_owner_context
from utils.utils import safe_curated_detail, log_and_http_error

logger = structlog.get_logger(__name__)


router = APIRouter(dependencies = [Depends(get_current_subject)])

# Only a UI session may define a local command; API keys keep http(s) MCP. Annotated, not a Depends default:
# these routes are also called directly by the tests, where a Depends object is truthy and would read as "API key".
ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]
WithoutCredential = Annotated[bool, Depends(request_admitted_without_credential)]


_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
# No ".com": it is far likelier a scheme-less domain than an MS-DOS program.
_PROGRAM_SUFFIXES = (".exe", ".cmd", ".bat", ".ps1", ".py", ".js", ".mjs", ".cjs", ".sh")


def _looks_like_command(value: str) -> bool:
    """Only picks which error to show when local commands are off, so a wrong guess costs a message, never
    access. Whitespace is a one-way signal: a URL can't hold an unencoded space. A lone token is a command
    when it is a path (urlparse reads ``D:\\mcp\\server.exe`` as scheme ``d``, which used to answer an .exe
    with "must start with http://"), quoted, or a program name; otherwise it may be a scheme-less URL."""
    if any(ch.isspace() for ch in value):
        return True
    if "://" in value:
        return False
    return (
        _WINDOWS_DRIVE_PATH.match(value) is not None
        or value.startswith(("\\", "/", "./", "../", "~", '"', "'"))
        or "\\" in value
        or value.lower().endswith(_PROGRAM_SUFFIXES)
    )


def _normalize_stdio_command(url: str, *, reject_unquoted_spaced_program: bool = True) -> str:
    raw = url or ""
    trimmed = raw.strip()
    if not trimmed:
        raise HTTPException(status_code = 400, detail = "command must not be empty")
    # Leading whitespace is executable-field padding. At the other end, only
    # space/tab delimit arguments on Windows. POSIX quoting protects whitespace.
    normalized = raw.lstrip().rstrip(" \t") if sys.platform == "win32" else trimmed
    try:
        parts = parse_stdio_command(normalized)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            "Invalid command. Check quoting and try again.",
            event = "mcp_servers.invalid_command",
            log = logger,
        )
    if not parts or not parts[0].strip():
        raise HTTPException(status_code = 400, detail = "command must not be empty")
    if any("\x00" in part for part in parts):
        raise HTTPException(
            status_code = 400,
            detail = "command and arguments must not contain NUL characters",
        )
    if "://" in parts[0]:
        raise HTTPException(
            status_code = 400,
            detail = "Enter an http(s):// URL, or a local command whose "
            "first token is an executable (not a URL).",
        )
    # The dialog quotes the program itself (/stdio/encode), but the raw API and config import take a command line
    # as written, and ``C:\My Tools\server.exe --x`` then runs ``C:\My`` -- or a planted ``C:\My.exe``, since Windows
    # guesses the extension. Refuse it while the intended path is still visible. POSIX shells taught everyone to
    # quote; there is no extension guessing there either.
    if reject_unquoted_spaced_program and sys.platform == "win32":
        spaced = unquoted_spaced_program(parts)
        if spaced is not None:
            raise HTTPException(
                status_code = 400,
                detail = f'This program path contains spaces — wrap it in double quotes: "{spaced}"',
            )
    return normalized


def _strip_wrapping_quotes(command: str) -> str:
    """Explorer's "Copy as path" wraps the path in double quotes. The executable field holds one argv entry,
    so those quotes would otherwise be escaped into the program name and the spawn fails with "file not
    found". Windows file names cannot contain '"', so one matching outer pair is never part of the name."""
    if len(command) >= 2 and command[0] == command[-1] and command[0] in "\"'":
        return command[1:-1].strip()
    return command


def _validate_cwd(cwd: Optional[str], url: str) -> Optional[str]:
    """A local program's working directory: None or blank clears it, anything else must be an existing
    absolute folder. Checked when saved so a typo shows up in the dialog, not as a server that never
    starts; checked again at spawn because folders get deleted. Callers gate local commands to a UI
    session first, so an API key cannot use this to probe which folders exist."""
    if cwd is None:
        return None
    value = _strip_wrapping_quotes(cwd.strip())
    if not value:
        return None
    if not is_stdio(url):
        raise HTTPException(
            status_code = 400,
            detail = "A working directory only applies to local programs, not http(s) servers.",
        )
    if "\x00" in value:
        raise HTTPException(
            status_code = 400, detail = "The working directory must not contain NUL characters."
        )
    if not os.path.isabs(value):
        raise HTTPException(
            status_code = 400,
            detail = "The working directory must be an absolute path to an existing folder.",
        )
    if not os.path.isdir(value):
        raise HTTPException(
            status_code = 400,
            detail = f"Working directory not found, or not a folder: {value}",
        )
    return value


def _validate_url(url: str) -> str:
    raw = url or ""
    trimmed = raw.strip()
    if not trimmed:
        raise HTTPException(status_code = 400, detail = "url must not be empty")
    # Non-HTTP values reuse the URL field for local commands. Syntax validation
    # is policy-free, but persistence and execution stay behind the stdio gate.
    if stdio_mcp_enabled() and is_stdio(trimmed):
        return _normalize_stdio_command(raw)
    parsed = urlparse(trimmed)
    if parsed.scheme not in ("http", "https"):
        if _looks_like_command(trimmed):
            detail = stdio_mcp_disabled_reason()
        else:
            detail = (
                "MCP server address must start with http:// or https:// "
                "(for example https://example.com/mcp)."
            )
        raise HTTPException(status_code = 400, detail = detail)
    if not parsed.netloc:
        raise HTTPException(status_code = 400, detail = "url is missing a host")
    return trimmed


def _normalize_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Trim header names, drop empties, coerce values to str; None if empty."""
    if not headers:
        return None
    out: dict[str, str] = {}
    for raw_key, value in headers.items():
        key = str(raw_key).strip()
        if key:
            normalized_value = str(value)
            if "\x00" in key or "\x00" in normalized_value:
                raise HTTPException(
                    status_code = 400,
                    detail = "headers and environment variables must not contain NUL characters",
                )
            if "=" in key:
                raise HTTPException(
                    status_code = 400,
                    detail = "header and environment variable names must not contain '='",
                )
            out[key] = normalized_value
    return out or None


def _image_mappings_active(row: dict) -> bool:
    from core.inference.tools import _enabled_mcp_servers

    # Same servers the model's MCP catalog keeps; a mapping elsewhere could never receive the image.
    if not image_input_mappings(row) or not _enabled_mcp_servers([row]):
        return False
    if is_stdio(row["url"]) and not stdio_mcp_enabled():
        return False
    tools = get_cached_tools(row["id"])
    return tools is None or any(
        image_mapping(row, tool) for tool in tools if tool_visible_to(tool, "model")
    )


def _oauth_client(
    client_id: str | None, client_secret: str | None
) -> tuple[str | None, str | None]:
    client_id = (client_id or "").strip() or None
    if client_secret and not client_id:
        raise HTTPException(status_code = 400, detail = "oauth_client_secret requires oauth_client_id")
    return client_id, client_secret or None


def _row_to_response(row: dict, *, include_headers: bool = True) -> McpServerResponse:
    return McpServerResponse(
        id = row["id"],
        builtin_id = row.get("builtin_id"),
        display_name = row["display_name"],
        url = row["url"],
        headers = (parse_server_headers(row) or {}) if include_headers else {},
        is_enabled = bool(row["is_enabled"]),
        use_oauth = bool(row.get("use_oauth")),
        cwd = row.get("cwd"),
        oauth_client_id = row.get("oauth_client_id"),
        has_oauth_client_secret = bool(row.get("oauth_client_secret")),
        image_input_mappings = image_input_mappings(row),
        image_mappings_active = _image_mappings_active(row),
        process_mode = server_process_mode(row),
        idle_timeout_seconds = server_idle_timeout(row),
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


def _new_process_mode(url: str, requested: Optional[str]) -> str:
    """A new local program shares one process across chats unless asked otherwise, like the apps servers are
    imported from: a server holding a serial port or SSH sessions cannot run twice, and one that loads toolsets
    must keep them between chats. Rows saved before the setting existed stay per chat (see mcp_servers_db). An
    HTTP server has no process; it is stored per chat, the behaviour it has always had."""
    if not is_stdio(url):
        return PROCESS_MODE_PER_CHAT
    return requested or PROCESS_MODE_SHARED


def _blender_row():
    return next(
        (row for row in mcp_servers_db.list_servers() if row.get("builtin_id") == "blender"), None
    )


def _require_managed_access(
    via_api_key,
    no_credential,
    *,
    executes = False,
):
    require_ui_session_for_local_commands(via_api_key or no_credential)
    if executes and not stdio_mcp_enabled():
        raise HTTPException(status_code = 400, detail = stdio_mcp_disabled_reason())


@router.get("/capabilities", response_model = McpCapabilities)
def get_mcp_capabilities(
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    if via_api_key or no_credential:
        return McpCapabilities(
            stdio_enabled = False,
            stdio_disabled_reason = "Local programs can only be added from a signed-in Unsloth Studio window.",
        )
    if stdio_mcp_enabled():
        return McpCapabilities(stdio_enabled = True)
    return McpCapabilities(stdio_enabled = False, stdio_disabled_reason = stdio_mcp_disabled_reason())


@router.get("/builtins", response_model = list[McpBuiltinResponse])
def list_builtins(
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    if via_api_key or no_credential:
        item = blender.catalog_item()
        item.available = False
        item.unavailable_reason = (
            "An authenticated Unsloth Studio UI session is required for Blender MCP."
        )
        return [item]
    return [blender.catalog_item(_blender_row())]


@router.post("/builtins/blender/test", response_model = McpServerProbeResult)
@serialize_mcp_server_mutation
async def test_blender(
    payload: BlenderTest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    _require_managed_access(via_api_key, no_credential, executes = True)
    row = _blender_row()
    config = json.loads(row.get("builtin_config_json") or "{}") if row else {}
    if not (config.get("consent") or payload.consent):
        raise HTTPException(
            status_code = 400, detail = "Explicit consent is required before testing Blender MCP."
        )
    settings = BlenderSettings(port = payload.port, blender_path = payload.blender_path)
    on_tools = None
    if row and blender.settings_for(row) == settings:
        on_tools = lambda tools: cache_tools(row["id"], tools)
    return await blender.probe(settings, on_tools = on_tools)


@router.put("/builtins/blender", response_model = McpBuiltinResponse)
@serialize_mcp_server_mutation
async def setup_blender(
    payload: BlenderSetup,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    _require_managed_access(via_api_key, no_credential, executes = payload.is_enabled)
    old = _blender_row()
    config = json.loads(old.get("builtin_config_json") or "{}") if old else {}
    if payload.is_enabled and not (config.get("consent") or payload.consent):
        raise HTTPException(
            status_code = 400, detail = "Explicit consent is required before enabling Blender MCP."
        )
    settings = BlenderSettings(port = payload.port, blender_path = payload.blender_path)
    config = {**settings.model_dump(), "consent": bool(config.get("consent") or payload.consent)}
    server_id = old["id"] if old else uuid.uuid4().hex[:16]
    if old:
        mcp_servers_db.update_server(
            server_id, {"builtin_config_json": json.dumps(config), "is_enabled": False}
        )
    else:
        mcp_servers_db.create_server(
            server_id,
            "Blender",
            "",
            is_enabled = False,
            builtin_id = "blender",
            builtin_config_json = json.dumps(config),
        )
    invalidate_tool_cache(server_id)
    if old:
        await asyncio.to_thread(close_mcp_sessions, old["url"], parse_server_headers(old))
    if payload.is_enabled:
        result = await blender.probe(
            settings, check_bridge = False, on_tools = lambda tools: cache_tools(server_id, tools)
        )
        if not result.ok:
            raise HTTPException(status_code = 400, detail = result.error)
        mcp_servers_db.update_server(server_id, {"is_enabled": True})
    return blender.catalog_item(mcp_servers_db.get_server(server_id))


@router.post("/stdio/decode", response_model = McpStdioCommand)
def decode_stdio_command(
    payload: McpStdioDecodeRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    require_ui_session_for_local_commands(via_api_key)
    if not is_stdio(payload.url.strip()):
        raise HTTPException(status_code = 400, detail = "HTTP(S) MCP servers do not have arguments")
    # A row saved unquoted before that check existed must still open in the editor, where re-saving encodes it
    # properly.
    url = _normalize_stdio_command(payload.url, reject_unquoted_spaced_program = False)
    parts = parse_stdio_command(url)
    return McpStdioCommand(command = parts[0], arguments = parts[1:])


@router.post("/stdio/encode", response_model = McpStdioEncodeResponse)
def encode_stdio_command(
    payload: McpStdioCommand,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    require_ui_session_for_local_commands(via_api_key)
    command = _strip_wrapping_quotes(payload.command.strip())
    if not command:
        raise HTTPException(status_code = 400, detail = "command must not be empty")
    if "://" in command:
        raise HTTPException(
            status_code = 400,
            detail = "command must be a local executable, not a URL",
        )
    url = join_stdio_command([command, *payload.arguments])
    _normalize_stdio_command(url)
    return McpStdioEncodeResponse(url = url)


# FastAPI offloads sync reads; mutations stay on-loop to preserve atomic sequences.
@router.get("/", response_model = list[McpServerResponse])
def list_mcp_servers(
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    rows = mcp_servers_db.list_servers()
    if via_api_key or no_credential:
        # Drop the row, not just its fields: `url` is the argv (carries credentials), `headers` is the subprocess
        # env, and a blanked url would round-trip into update as a bogus command.
        rows = [row for row in rows if not is_stdio(row["url"])]
    return [_row_to_response(row, include_headers = not no_credential) for row in rows]


@router.post("/", response_model = McpServerResponse, status_code = 201)
async def create_mcp_server(
    payload: McpServerCreate,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    display_name = (payload.display_name or "").strip()
    if not display_name:
        raise HTTPException(status_code = 400, detail = "display_name must not be empty")
    url = _validate_url(payload.url)
    if is_stdio(url):
        require_ui_session_for_local_commands(via_api_key)
    cwd = _validate_cwd(payload.cwd, url)
    headers = _normalize_headers(payload.headers)
    # OAuth is HTTP-only; force it off for stdio commands so a stale flag can't
    # push the probe onto the 305s OAuth timeout. Backend enforces this.
    use_oauth = payload.use_oauth and not is_stdio(url)
    client_id, client_secret = (
        _oauth_client(payload.oauth_client_id, payload.oauth_client_secret)
        if use_oauth
        else (None, None)
    )

    server_id = uuid.uuid4().hex[:16]
    mcp_servers_db.create_server(
        id = server_id,
        display_name = display_name,
        url = url,
        headers_json = json.dumps(headers) if headers else None,
        is_enabled = payload.is_enabled,
        use_oauth = use_oauth,
        image_input_mappings_json = _mappings_json(payload.image_input_mappings),
        oauth_client_id = client_id,
        oauth_client_secret = client_secret,
        cwd = cwd,
        process_mode = _new_process_mode(url, payload.process_mode),
        idle_timeout_seconds = payload.idle_timeout_seconds,
    )
    return _row_to_response(mcp_servers_db.get_server(server_id))


def _mappings_json(mappings) -> str:
    return json.dumps([mapping.model_dump() for mapping in mappings or []])


def _changes_from_payload(payload: McpServerUpdate) -> dict:
    sent = payload.model_fields_set
    changes: dict = {}

    if "display_name" in sent:
        name = (payload.display_name or "").strip()
        if not name:
            raise HTTPException(status_code = 400, detail = "display_name must not be empty")
        changes["display_name"] = name
    if "url" in sent:
        changes["url"] = _validate_url(payload.url or "")
    if "headers" in sent:
        headers = _normalize_headers(payload.headers)
        changes["headers_json"] = json.dumps(headers) if headers else None
    if "is_enabled" in sent:
        if payload.is_enabled is None:
            raise HTTPException(status_code = 400, detail = "is_enabled must be true or false")
        changes["is_enabled"] = payload.is_enabled
    if "use_oauth" in sent:
        if payload.use_oauth is None:
            raise HTTPException(status_code = 400, detail = "use_oauth must be true or false")
        changes["use_oauth"] = payload.use_oauth
    if "image_input_mappings" in sent:
        changes["image_input_mappings_json"] = _mappings_json(payload.image_input_mappings)
    if "oauth_client_id" in sent:
        changes["oauth_client_id"] = (payload.oauth_client_id or "").strip() or None
    if "oauth_client_secret" in sent:
        changes["oauth_client_secret"] = payload.oauth_client_secret or None
    if "process_mode" in sent:
        if payload.process_mode is None:
            raise HTTPException(status_code = 400, detail = "process_mode must be shared or per_chat")
        changes["process_mode"] = payload.process_mode
    if "idle_timeout_seconds" in sent:
        # null = back to the mode's default.
        changes["idle_timeout_seconds"] = payload.idle_timeout_seconds
    # stdio is OAuth-less: drop a stale OAuth flag when switching to a command.
    if "url" in changes and is_stdio(changes["url"]):
        changes["use_oauth"] = False
    if changes.get("use_oauth") is False:
        changes["oauth_client_id"] = changes["oauth_client_secret"] = None
    return changes


@router.put("/{server_id}", response_model = McpServerResponse)
@serialize_mcp_server_mutation
async def update_mcp_server(
    server_id: str,
    payload: McpServerUpdate,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    old = mcp_servers_db.get_server(server_id)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    changes = _changes_from_payload(payload)
    if old.get("builtin_id"):
        _require_managed_access(via_api_key, no_credential)
        if payload.model_fields_set != {"is_enabled"} or payload.is_enabled is not False:
            raise HTTPException(
                status_code = 400,
                detail = "Use the managed integration setup to configure or enable this server.",
            )
    client_id = changes.get("oauth_client_id", old.get("oauth_client_id"))
    # A secret belongs to one client at one origin: a new client ID or URL drops it unless replaced.
    if "oauth_client_secret" not in changes and (
        client_id != old.get("oauth_client_id") or changes.get("url", old["url"]) != old["url"]
    ):
        changes["oauth_client_secret"] = None
    _oauth_client(client_id, changes.get("oauth_client_secret", old.get("oauth_client_secret")))
    cwd_sent = "cwd" in payload.model_fields_set
    if not changes and not cwd_sent:
        raise HTTPException(status_code = 400, detail = "No fields to update")
    # Both directions, so an API key can neither repoint an http row at a command nor edit a stdio row's
    # env/name/enabled flag. Before every side effect, so a refusal leaves the row, its OAuth tokens, cache and
    # sessions untouched.
    if is_stdio(old["url"]) or is_stdio(changes.get("url", old["url"])):
        require_ui_session_for_local_commands(via_api_key)
    # Validated against the address the row will have, after the gate (the check touches the filesystem). A switch
    # to http drops a stored working directory, as it drops env vars below.
    if cwd_sent:
        changes["cwd"] = _validate_cwd(payload.cwd, changes.get("url", old["url"]))
    elif "url" in changes and not is_stdio(changes["url"]) and old.get("cwd"):
        changes["cwd"] = None
    # headers == HTTP headers (remote) or env vars (stdio). On a transport-type switch with no new headers, drop
    # the old ones so env secrets aren't re-sent as HTTP headers (or vice versa).
    if (
        "url" in changes
        and is_stdio(changes["url"]) != is_stdio(old["url"])
        and "headers_json" not in changes
    ):
        changes["headers_json"] = None
    # Clear persisted OAuth tokens when the URL, the OAuth flag or the client changes
    if bool(old.get("use_oauth")) and (
        ("url" in changes and changes["url"] != old["url"])
        or changes.get("use_oauth") is False
        or any(
            changes.get(k, old.get(k)) != old.get(k)
            for k in ("oauth_client_id", "oauth_client_secret")
        )
    ):
        await clear_oauth_tokens_async(old["url"])
        # That await hands the loop to other requests.
        current = mcp_servers_db.get_server(server_id)
        if current is not None and (
            is_stdio(current["url"]) or is_stdio(changes.get("url", current["url"]))
        ):
            require_ui_session_for_local_commands(via_api_key)
    # A new endpoint/auth makes cached tools wrong and disabling makes them unreachable.
    invalidates_tools = any(
        changes[k] != old.get(k) for k in changes.keys() & TOOL_CACHE_INVALIDATING_FIELDS
    )
    mcp_servers_db.update_server(server_id, changes)
    if invalidates_tools:
        invalidate_tool_cache(server_id)
    if invalidates_tools:
        # Narrow to this row's env and working directory: another server row sharing the command but not those
        # keeps its live sessions.
        await asyncio.to_thread(
            lambda: close_mcp_sessions(old["url"], parse_server_headers(old), cwd = old.get("cwd"))
        )
    elif "idle_timeout_seconds" in changes:
        # A running process keeps going under the new timeout rather than being restarted for it.
        retime_server_processes(mcp_servers_db.get_server(server_id) or old)
    return _row_to_response(mcp_servers_db.get_server(server_id), include_headers = not no_credential)


@router.delete("/{server_id}", status_code = 204)
@serialize_mcp_server_mutation
async def delete_mcp_server(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    old = mcp_servers_db.get_server(server_id)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if old.get("builtin_id"):
        raise HTTPException(
            status_code = 400, detail = "Managed integrations cannot be deleted; disable them instead."
        )
    # Same rule as update: an API key cannot touch a stdio row.
    if is_stdio(old["url"]):
        require_ui_session_for_local_commands(via_api_key)
    if old.get("use_oauth"):
        await clear_oauth_tokens_async(old["url"])
    mcp_servers_db.delete_server(server_id)
    invalidate_tool_cache(server_id)
    await asyncio.to_thread(
        lambda: close_mcp_sessions(old["url"], parse_server_headers(old), cwd = old.get("cwd"))
    )


@router.get("/{server_id}/tools")
def list_mcp_server_tools(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    """Cached tool names and input schemas, for choosing an image input mapping."""
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if is_stdio(server["url"]):
        require_ui_session_for_local_commands(via_api_key)
    tools = get_cached_tools(server_id)
    if tools is None:
        raise HTTPException(status_code = 409, detail = "Refresh this server's tools first")
    # App-only tools never reach the model, so a mapping on one could never be used.
    return [
        {"name": tool["name"], "inputSchema": tool.get("inputSchema") or tool.get("input_schema")}
        for tool in tools
        if isinstance(tool, dict)
        and isinstance(tool.get("name"), str)
        and tool_visible_to(tool, "model")
    ]


@router.post("/{server_id}/refresh", response_model = McpServerProbeResult)
async def refresh_mcp_server_tools(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if server.get("builtin_id"):
        raise HTTPException(
            status_code = 400,
            detail = "Use the managed integration Test action to check Blender readiness.",
        )
    # Refresh uses the stored address.
    if is_stdio(server["url"]):
        require_ui_session_for_local_commands(via_api_key)
        if not stdio_mcp_enabled():
            raise HTTPException(status_code = 400, detail = stdio_mcp_disabled_reason())

    use_oauth = bool(server.get("use_oauth"))
    epoch = tools_cache_epoch(server_id)
    try:
        if _shares_process(server):
            tools = await _list_shared_process_tools(server)
        else:
            tools = await list_tools_async(
                url = server["url"],
                headers = parse_server_headers(server),
                timeout = probe_timeout(server["url"], use_oauth),
                use_oauth = use_oauth,
                cwd = server.get("cwd"),
                **oauth_client_kwargs(server),
            )
    except McpServerBusy as exc:
        # The server is fine, just running another chat's call: no cool-off.
        return McpServerProbeResult(ok = False, error = f"Could not refresh: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface transport+timeout errors to UI
        _log_probe_failure("mcp_servers.refresh_failed", exc, server_id = server_id)
        current = mcp_servers_db.get_server(server_id)
        if current is not None and not any(
            current.get(k) != server.get(k) for k in TOOL_CACHE_INVALIDATING_FIELDS
        ):
            # Start the cool-off so the next chat send does not re-hang on this server's timeout. If the row changed
            # while the probe was awaiting, the FAILURE belongs to the old config and must not park the newly edited
            # server.
            record_probe_failure(server_id, use_oauth)
            record_server_failure(server, exc)
        return McpServerProbeResult(ok = False, error = safe_curated_detail(exc))

    current = mcp_servers_db.get_server(server_id)
    if current is not None and not any(
        current.get(k) != server.get(k) for k in TOOL_CACHE_INVALIDATING_FIELDS
    ):
        cache_tools(server_id, tools, epoch = epoch)
    return McpServerProbeResult(ok = True, tool_count = len(tools))


def _shares_process(server: dict) -> bool:
    lifecycle = server_lifecycle(server)
    return lifecycle is not None and lifecycle.shared


async def _list_shared_process_tools(server: dict) -> list[dict]:
    """A shared local program's tools, asked over its one process (started if it is not running, then kept for the
    chats). Probing a second copy instead would fail for a server that holds a serial port or a session, and would
    list the tools a fresh process starts with rather than the ones the chats are using."""
    lifecycle = server_lifecycle(server)
    return await asyncio.to_thread(
        list_session_tools_sync,
        server["url"],
        parse_server_headers(server),
        scope = shared_session_scope(lifecycle.server_id),
        timeout = probe_timeout(server["url"], False),
        cwd = server.get("cwd"),
        lifecycle = lifecycle,
        config_check = lambda: _row_still_matches(lifecycle.server_id, server),
    )


def _status_response(row: dict) -> McpServerStatus:
    return McpServerStatus(
        server_id = row["id"],
        process_mode = server_process_mode(row),
        idle_timeout_seconds = server_idle_timeout(row),
        **server_status(row),
    )


def _local_program_or_error(
    server_id: str, via_api_key: bool, no_credential: bool, *, executes: bool
) -> dict:
    """A saved local program whose processes this caller may manage: the installation owner, from a signed-in
    Studio window (never an API key), with local programs allowed when ``executes`` would start one. Checked
    before the row is read, so a refused caller learns nothing about which servers exist."""
    _require_managed_access(via_api_key, no_credential, executes = executes)
    if not is_owner_context():
        raise HTTPException(
            status_code = 403, detail = "Only the installation owner runs local programs."
        )
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if not is_stdio(server["url"]):
        raise HTTPException(
            status_code = 400, detail = "Only local programs have a process to stop or restart."
        )
    return server


@router.get("/status", response_model = list[McpServerStatus])
def list_mcp_server_status(
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    """What each saved local program's processes are doing: running, idle, stopped or failed. Empty for an API
    key, a keyless caller or a managed account, none of which can run local programs (list_mcp_servers hides
    those rows from them too)."""
    if via_api_key or no_credential or not is_owner_context():
        return []
    return [
        _status_response(row)
        for row in mcp_servers_db.list_servers()
        if is_stdio(row.get("url") or "")
    ]


@router.post("/{server_id}/restart", response_model = McpServerStatus)
async def restart_mcp_server(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    """End this server's processes and, for a shared one, start it again now and re-read its tools. A per-chat
    server's processes start again in each chat on its next call."""
    server = _local_program_or_error(server_id, via_api_key, no_credential, executes = True)
    if not server.get("is_enabled"):
        raise HTTPException(status_code = 400, detail = "Turn this server on before restarting it.")
    await asyncio.to_thread(stop_server_processes, server)
    clear_server_failure(server)
    if _shares_process(server):
        # After the stop: closing a process whose tools changed marks the list stale, and this re-read replaces it.
        epoch = tools_cache_epoch(server_id)
        try:
            tools = await _list_shared_process_tools(server)
        except Exception as exc:  # noqa: BLE001 - the failure is what the status reports
            _log_probe_failure("mcp_servers.restart_failed", exc, server_id = server_id)
            record_server_failure(server, exc)
        else:
            if _row_still_matches(server_id, server):
                cache_tools(server_id, tools, epoch = epoch)
    return _status_response(mcp_servers_db.get_server(server_id) or server)


@router.post("/{server_id}/stop", response_model = McpServerStatus)
async def stop_mcp_server(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    """End this server's processes now (the shared one, or every chat's). Allowed while local programs are
    suspended: stopping one starts nothing. The next call that needs it starts it again; switch the server off
    to keep it stopped."""
    server = _local_program_or_error(server_id, via_api_key, no_credential, executes = False)
    await asyncio.to_thread(stop_server_processes, server)
    clear_server_failure(server)
    return _status_response(server)


@router.post("/import", response_model = McpServerImportResult)
async def import_mcp_servers(
    payload: McpServerImportRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    """Bulk-register servers from a standard mcpServers JSON config (issue
    #5936). Each entry rides the existing create path: _validate_url applies
    the same stdio gate (a stdio entry becomes a per-entry error when stdio is
    off; http still imports), and entries whose url already exists are skipped
    so re-importing the same file is idempotent. One bad entry never 400s the
    whole batch -- failures are reported per entry."""
    entries, errors = parse_mcp_config(payload.config)
    created: list[McpServerResponse] = []
    skipped: list[str] = []
    seen_urls = {row["url"] for row in mcp_servers_db.list_servers()}

    for entry in entries:
        try:
            url = _validate_url(entry.url)
            # Per entry, so an API-key import of a mixed config still creates its
            # http entries and reports the stdio ones.
            if is_stdio(url):
                require_ui_session_for_local_commands(via_api_key)
            cwd = _validate_cwd(entry.cwd, url)
            headers = _normalize_headers(entry.headers)
        except HTTPException as exc:
            errors.append(f"{entry.display_name}: {exc.detail}")
            continue
        if url in seen_urls:
            skipped.append(entry.display_name)
            continue
        server_id = uuid.uuid4().hex[:16]
        mcp_servers_db.create_server(
            id = server_id,
            display_name = entry.display_name,
            url = url,
            headers_json = json.dumps(headers) if headers else None,
            is_enabled = entry.is_enabled,
            use_oauth = entry.use_oauth and not is_stdio(url),
            cwd = cwd,
            # As in the app the config came from: one process per server.
            process_mode = _new_process_mode(url, None),
        )
        seen_urls.add(url)
        created.append(_row_to_response(mcp_servers_db.get_server(server_id)))

    return McpServerImportResult(created = created, skipped = skipped, errors = errors)


def _require_import_source_access(via_api_key: bool, no_credential: bool) -> None:
    """Import from another app reads this computer's config files and the API keys in them. Only the
    installation owner, from a signed-in Studio window: an API key could otherwise import a server
    and read its stored headers straight back, and a managed account would receive the owner's
    credentials."""
    if not is_owner_context():
        raise HTTPException(
            status_code = 403,
            detail = "Only the installation owner can import servers from other apps on this computer.",
        )
    if via_api_key or no_credential:
        raise HTTPException(
            status_code = 403,
            detail = "Importing from other apps reads this computer's config files, so it is only "
            "available from a signed-in Unsloth Studio window.",
        )


def _existing_server_identities() -> set:
    identities = set()
    for row in mcp_servers_db.list_servers():
        if row.get("builtin_id") or not row.get("url"):
            continue
        identity = mcp_import_sources.server_identity(row["url"])
        if identity is not None:
            identities.add(identity)
    return identities


def _switched_off_reasons(
    prepared: mcp_import_sources.PreparedServer, url: str, app: str
) -> list[str]:
    """Why an importable server should arrive switched off: off in the app it came from, a
    ``${...}`` nothing here fills in, or a program that isn't on this machine. Each is something the
    user fixes in the editor before turning it on, rather than a first chat that fails."""
    entry = prepared.entry
    reasons: list[str] = []
    if entry is not None and not entry.is_enabled:
        reasons.append(f"It is switched off in {app}.")
    if prepared.unresolved:
        names = ", ".join(prepared.unresolved)
        reasons.append(
            f"{names} {'is' if len(prepared.unresolved) == 1 else 'are'} not set in Studio's "
            "environment; edit the server to fill in the value."
        )
    if entry is not None and is_stdio(url):
        try:
            program = parse_stdio_command(url)[0]
        except (ValueError, IndexError):
            program = ""
        if program and mcp_import_sources.find_program(program, entry.headers, entry.cwd) is None:
            reasons.append(f"Program not found: {program}")
    return reasons


def _import_source_preview(
    prepared: mcp_import_sources.PreparedServer,
    app: str,
    existing: set,
    stdio_disabled_reason: Optional[str],
) -> McpImportSourceServer:
    entry = prepared.entry
    note = prepared.error
    importable = entry is not None
    if entry is not None and entry.is_stdio and stdio_disabled_reason:
        importable = False
        note = stdio_disabled_reason
    elif entry is not None:
        reasons = _switched_off_reasons(prepared, entry.url, app)
        note = f"Will be added switched off. {' '.join(reasons)}" if reasons else None
    identity = mcp_import_sources.server_identity(entry.url) if entry is not None else None
    return McpImportSourceServer(
        name = prepared.name,
        transport = prepared.transport,
        target = prepared.target,
        env_keys = prepared.env_keys,
        header_keys = prepared.header_keys,
        already_added = identity is not None and identity in existing,
        importable = importable,
        note = note,
    )


@router.get("/import-sources", response_model = McpImportSourcesResponse)
def list_import_sources(
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    """The MCP configs of other apps on this computer (Claude Desktop, Claude Code, Cursor, VS Code,
    Windsurf) and the servers in them, masked: names, transports, commands and URLs with credentials
    hidden, and env/header names without values."""
    _require_import_source_access(via_api_key, no_credential)
    stdio_disabled_reason = None if stdio_mcp_enabled() else stdio_mcp_disabled_reason()
    existing = _existing_server_identities()
    sources = []
    for source in mcp_import_sources.discover_sources():
        servers = [
            _import_source_preview(
                mcp_import_sources.prepare_server(server),
                source.app,
                existing,
                stdio_disabled_reason,
            )
            for server in source.servers
        ]
        sources.append(
            McpImportSource(
                id = source.source_id,
                app = source.app,
                label = source.label,
                path = source.path if is_owner_context() else None,
                error = source.error,
                servers = servers,
            )
        )
    return McpImportSourcesResponse(sources = sources)


def _import_from_source(
    server: mcp_import_sources.SourceServer, app: str, existing: set, via_api_key: bool
) -> McpImportServerOutcome:
    prepared = mcp_import_sources.prepare_server(server)
    entry = prepared.entry
    if entry is None:
        return McpImportServerOutcome(name = server.name, status = "error", detail = prepared.error)
    # Named here: _validate_url can't tell a bare program name ("konnect") from a scheme-less URL.
    if entry.is_stdio and not stdio_mcp_enabled():
        return McpImportServerOutcome(
            name = server.name, status = "error", detail = stdio_mcp_disabled_reason()
        )
    try:
        url = _validate_url(entry.url)
        if is_stdio(url):
            require_ui_session_for_local_commands(via_api_key)
        cwd = _validate_cwd(entry.cwd, url)
        headers = _normalize_headers(entry.headers)
    except HTTPException as exc:
        return McpImportServerOutcome(name = server.name, status = "error", detail = str(exc.detail))
    identity = mcp_import_sources.server_identity(url)
    if identity is not None and identity in existing:
        return McpImportServerOutcome(
            name = server.name,
            status = "duplicate",
            detail = "Studio already has a server with this "
            + ("command." if is_stdio(url) else "address."),
        )
    reasons = _switched_off_reasons(prepared, url, app)
    server_id = uuid.uuid4().hex[:16]
    mcp_servers_db.create_server(
        id = server_id,
        display_name = entry.display_name,
        url = url,
        headers_json = json.dumps(headers) if headers else None,
        is_enabled = not reasons,
        use_oauth = entry.use_oauth and not is_stdio(url),
        cwd = cwd,
        # As in the app it came from: one process per server.
        process_mode = _new_process_mode(url, None),
    )
    if identity is not None:
        existing.add(identity)
    return McpImportServerOutcome(
        name = server.name,
        status = "added_disabled" if reasons else "added",
        detail = " ".join(reasons) or None,
        server_id = server_id,
    )


@router.post("/import-sources/apply", response_model = McpImportSourceApplyResult)
@serialize_mcp_server_mutation
async def apply_import_source(
    payload: McpImportSourceApplyRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    """Import the chosen servers of one discovered app config. The file is re-read here by source
    id, so neither a path nor a secret ever comes from the browser. A server arrives switched on only
    when it can run as written; duplicates are skipped and every server gets its own outcome."""
    _require_import_source_access(via_api_key, no_credential)
    sources = await asyncio.to_thread(mcp_import_sources.discover_sources)
    source = next((item for item in sources if item.source_id == payload.source_id), None)
    if source is None:
        raise HTTPException(
            status_code = 404,
            detail = "That app's config file is no longer there. Reopen the list and try again.",
        )
    if source.error:
        raise HTTPException(status_code = 400, detail = source.error)
    by_name = {server.name: server for server in source.servers}
    existing = _existing_server_identities()
    results: list[McpImportServerOutcome] = []
    for name in dict.fromkeys(payload.server_names):
        server = by_name.get(name)
        if server is None:
            results.append(
                McpImportServerOutcome(
                    name = name,
                    status = "error",
                    detail = f"{source.app} no longer lists this server.",
                )
            )
            continue
        results.append(_import_from_source(server, source.app, existing, via_api_key))
    return McpImportSourceApplyResult(results = results)


@router.post("/test", response_model = McpServerProbeResult)
async def test_mcp_server(
    payload: McpServerTestRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    # URL/header validation must surface as 400 like create/update so the frontend's create-form pre-flight gets the
    # same error semantics as the save call. Only catch transport/timeout errors below.
    url = _validate_url(payload.url)
    # Caller-supplied and unstored, so the gate has to land before
    # list_tools_async -- after it the process has already started.
    if is_stdio(url):
        require_ui_session_for_local_commands(via_api_key)
    cwd = _validate_cwd(payload.cwd, url)
    headers = _normalize_headers(payload.headers)
    use_oauth = payload.use_oauth and not is_stdio(url)
    client_id, client_secret = _oauth_client(payload.oauth_client_id, payload.oauth_client_secret)
    if use_oauth and payload.server_id and client_id and not client_secret:
        stored = mcp_servers_db.get_server(payload.server_id) or {}
        if stored.get("url") == url and stored.get("oauth_client_id") == client_id:
            client_secret = stored.get("oauth_client_secret")
    # Testing the saved settings of a running shared server asks that process: a second copy could not open the
    # serial port or session the first one holds, and would report a working server as broken.
    running_as_saved = None
    if payload.server_id and is_stdio(url):
        stored = mcp_servers_db.get_server(payload.server_id)
        if (
            stored is not None
            and stored.get("is_enabled")
            and _shares_process(stored)
            and stored["url"] == url
            and parse_server_headers(stored) == headers
            and (stored.get("cwd") or None) == cwd
        ):
            running_as_saved = stored
    try:
        if running_as_saved is not None:
            tools = await _list_shared_process_tools(running_as_saved)
            return McpServerProbeResult(ok = True, tool_count = len(tools))
        tools = await list_tools_async(
            url = url,
            headers = headers,
            timeout = probe_timeout(url, use_oauth),
            use_oauth = use_oauth,
            cwd = cwd,
            **oauth_client_kwargs(
                {"oauth_client_id": client_id, "oauth_client_secret": client_secret}
                if use_oauth
                else {}
            ),
        )
    except McpServerBusy as exc:
        return McpServerProbeResult(ok = False, error = f"Could not test: {exc}")
    except Exception as exc:  # noqa: BLE001
        _log_probe_failure("mcp_servers.test_failed", exc)
        return McpServerProbeResult(ok = False, error = safe_curated_detail(exc))

    return McpServerProbeResult(ok = True, tool_count = len(tools))


def _log_probe_failure(event: str, exc: Exception, **fields) -> None:
    """An explained local-program failure (missing program, crash, no handshake) is the user's
    configuration talking, not a backend fault: one warning line with its first line only, since the
    rest quotes the program's own output, which stays in its log file. Anything else keeps the full
    traceback."""
    if isinstance(exc, McpStdioServerError):
        logger.warning(event, error = exc.summary, **fields)
    else:
        logger.error(event, error = str(exc), exc_info = True, **fields)


_UI_TIMEOUT = 60.0
UI_TOOL_APPROVAL_REQUIRED = "approval_required"


def _ui_server_or_404(server_id: str, via_api_key: bool) -> dict:
    """Re-read per request: a stale widget must not keep a removed server reachable."""
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if not server.get("is_enabled"):
        raise HTTPException(status_code = 400, detail = "MCP server is disabled")
    if is_stdio(server["url"]):
        require_ui_session_for_local_commands(via_api_key)
        if not stdio_mcp_enabled():
            raise HTTPException(status_code = 400, detail = stdio_mcp_disabled_reason())
    return server


def _row_still_matches(server_id: str, server: dict) -> bool:
    current = mcp_servers_db.get_server(server_id)
    return current is not None and all(
        current.get(k) == server.get(k) for k in TOOL_CACHE_INVALIDATING_FIELDS
    )


def _cwd_kwargs(row: dict) -> dict:
    # A local program's working directory, passed only when set (as oauth_client_kwargs), so an HTTP server's
    # call keeps the exact shape it had before working directories existed.
    return {"cwd": row["cwd"]} if row.get("cwd") else {}


# One discovery per server at a time: reopening a chat mounts every widget at once, each probing a cold cache.
_discovery_locks: dict = {}


async def _warm_tool_cache(server: dict) -> None:
    """Rediscover once on a cold cache: a chat reopened after a restart never ran the chat path, and widget calls read the cache."""
    server_id = server["id"]
    async with _discovery_locks.setdefault(server_id, asyncio.Lock()):
        tools = get_cached_tools(server_id)
        if tools is None and not in_failure_cooloff(server_id):
            use_oauth = bool(server.get("use_oauth"))
            url = server["url"]
            try:
                if _shares_process(server):
                    tools = await _list_shared_process_tools(server)
                else:
                    tools = await list_tools_async(
                        url = url,
                        headers = parse_server_headers(server),
                        timeout = probe_timeout(url, use_oauth),
                        use_oauth = use_oauth,
                        **_cwd_kwargs(server),
                        **oauth_client_kwargs(server),
                    )
            except McpServerBusy:
                return  # not a failure: the next widget request asks again
            except Exception:  # noqa: BLE001 - a probe failure reads as "nothing declared"
                tools = None
            # A row edited mid-probe: the old endpoint's answer must neither authorize a read nor be cached.
            if not _row_still_matches(server_id, server):
                tools = None
            elif tools is None:
                record_probe_failure(server_id, use_oauth)
            else:
                cache_tools(server_id, tools)


def _ui_call_kwargs(server_id: str, server: dict, thread_id, session_id) -> dict:
    from core.inference.tools import mcp_call_scope, mcp_session_scope

    lifecycle = server_lifecycle(server)
    return {
        "url": server["url"],
        "headers": parse_server_headers(server),
        "timeout": _UI_TIMEOUT,
        "use_oauth": bool(server.get("use_oauth")),
        **oauth_client_kwargs(server),
        # execute_tool's key (scope and working directory), so a widget reaches the process the chat's calls use:
        # the server's shared one, or the chat's own.
        "scope": mcp_call_scope(server, session_id, thread_id),
        **_cwd_kwargs(server),
        "config_check": lambda: _row_still_matches(server_id, server),
        **(
            {"lifecycle": lifecycle, "caller": mcp_session_scope(session_id, thread_id)}
            if lifecycle is not None
            else {}
        ),
    }


@router.get("/{server_id}/ui-resource", response_model = McpUiResourceResponse)
async def read_mcp_ui_resource(
    server_id: str,
    uri: str,
    thread_id: Optional[str] = None,
    session_id: Optional[str] = None,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    server = _ui_server_or_404(server_id, via_api_key)
    uri = (uri or "").strip()
    # Any ui:// resource (widgets read their own assets), no other scheme: a filesystem server maps file:// onto the host.
    if not uri.startswith(UI_RESOURCE_SCHEME):
        raise HTTPException(status_code = 400, detail = "uri must be a ui:// resource")
    from core.inference.tools import (
        _STUDIO_CREDENTIAL_BLOCKED,
        _mcp_arguments_reference_studio_credential,
    )

    if _mcp_arguments_reference_studio_credential({"uri": uri}):
        raise HTTPException(status_code = 403, detail = _STUDIO_CREDENTIAL_BLOCKED)
    await _warm_tool_cache(server)
    try:
        contents = await asyncio.to_thread(
            read_resource_sync, uri = uri, **_ui_call_kwargs(server_id, server, thread_id, session_id)
        )
    except McpServerBusy as exc:
        raise HTTPException(status_code = 409, detail = f"MCP server: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise log_and_http_error(
            exc,
            502,
            "Could not load this MCP app's interface.",
            event = "mcp_servers.ui_resource_failed",
            log = logger,
        )
    return McpUiResourceResponse(**contents)


@router.post("/{server_id}/ui-tool-call", response_model = McpUiToolCallResult)
async def call_mcp_ui_tool(
    server_id: str,
    payload: McpUiToolCallRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    """Widget is untrusted: server_id comes from the host frame, the tool must be discovered with "app" visibility, and it passes the confirm gate."""
    from core.inference.tools import (
        _STUDIO_CREDENTIAL_BLOCKED,
        MCP_TOOL_PREFIX,
        _mcp_arguments_reference_studio_credential,
        is_potentially_unsafe_tool_call,
        mcp_tool_definition,
    )
    from state.tool_policy import get_tool_policy

    if get_tool_policy() is False:
        raise HTTPException(status_code = 403, detail = "Tools are disabled on this server")
    server = _ui_server_or_404(server_id, via_api_key)
    tool_name = (payload.tool_name or "").strip()
    if not tool_name:
        raise HTTPException(status_code = 400, detail = "tool_name must not be empty")
    # A mounted widget outlives the cache: an edit or off/on toggle of the server empties it.
    await _warm_tool_cache(server)
    tool = mcp_tool_definition(server_id, tool_name)
    if tool is None:
        raise HTTPException(
            status_code = 404, detail = f"MCP server has no discovered tool named '{tool_name}'"
        )
    if not tool_visible_to(tool, "app"):
        raise HTTPException(
            status_code = 403, detail = f"Tool '{tool_name}' is not callable by an MCP app"
        )
    arguments = payload.arguments or {}
    if _mcp_arguments_reference_studio_credential(arguments):
        raise HTTPException(status_code = 403, detail = _STUDIO_CREDENTIAL_BLOCKED)
    mode = payload.permission_mode
    # An unstated or unknown mode asks; "auto" asks only for what the model's call would be asked for.
    needs_approval = mode not in ("off", "full") and (
        mode != "auto"
        or is_potentially_unsafe_tool_call(f"{MCP_TOOL_PREFIX}{server_id}__{tool_name}", arguments)
    )
    if needs_approval and not payload.approved:
        raise HTTPException(status_code = 409, detail = UI_TOOL_APPROVAL_REQUIRED)
    try:
        result = await asyncio.to_thread(
            call_tool_structured_sync,
            name = tool_name,
            args = arguments,
            **_ui_call_kwargs(server_id, server, payload.thread_id, payload.session_id),
        )
    except McpServerBusy as exc:
        # The widget asks the user only on the exact "approval_required" detail; this one reads as an error.
        raise HTTPException(status_code = 409, detail = f"MCP server: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise log_and_http_error(
            exc,
            502,
            "The MCP app's tool call failed.",
            event = "mcp_servers.ui_tool_call_failed",
            log = logger,
        )
    return McpUiToolCallResult(**result)
