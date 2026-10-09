# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextvars
import hashlib
import importlib
import ipaddress
import json
import logging
import math
import mimetypes
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, contextmanager, nullcontext
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, NamedTuple, Optional, get_type_hints
from urllib.parse import urlsplit, urlunsplit
from weakref import WeakKeyDictionary, ref as weak_ref

from loggers import get_logger
from utils.log_retention import prune_log_dir
from utils.account_context import (
    OWNER_ACCOUNT_ID,
    account_thread,
    current_account_id,
    is_owner_context,
)
from utils.log_redaction import redact_log_text

from core.inference import mcp_images

logger = get_logger(__name__)

MCP_TOOL_PREFIX = "mcp__"
_WINDOWS_BATCH_ALWAYS_UNSAFE_ARGUMENT_CHARS = frozenset('%!"\r\n')
_WINDOWS_BATCH_UNQUOTED_UNSAFE_ARGUMENT_CHARS = frozenset("&|<>^()")

# A failed probe isn't cached (a recovered server must come back), but it's recorded so a down server isn't re-probed
# -- and the chat send re-hung for the full timeout -- on every message. Cool off for this long after a failure; much
# longer for OAuth, whose probe can hang up to _OAUTH_PROBE_TIMEOUT, so that hang doesn't recur every minute.
FAILED_PROBE_COOLOFF_SECONDS = 60.0
OAUTH_FAILED_PROBE_COOLOFF_SECONDS = 300.0

_oauth_token_store = None
_account_oauth_token_stores: dict[str, Any] = {}


def _managed_mcp_restricted() -> bool:
    return not is_owner_context()


def _account_key(value):
    if is_owner_context():
        return value
    return current_account_id(), value


def _public_mcp_address(url: str) -> str:
    from fastapi import HTTPException
    detail = "Managed accounts may only use public-network HTTP MCP servers."
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("HTTP URL required")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Use MCP headers for credentials")
        host = parsed.hostname
        if "%" in host:
            raise ValueError("Scoped addresses are not public")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(host, port, type = socket.SOCK_STREAM)
        if not addresses:
            raise ValueError("No addresses")
        for *_, sockaddr in addresses:
            address = ipaddress.ip_address(sockaddr[0])
            address = getattr(address, "ipv4_mapped", None) or address
            if not address.is_global or address.is_multicast or address.is_reserved:
                raise ValueError("Non-public address")
        return addresses[0][4][0]
    except (OSError, ValueError, UnicodeError) as exc:
        raise HTTPException(status_code = 400, detail = detail) from exc


def validate_mcp_address(url: str) -> None:
    # Studio's own Decisions server is answered in process; with a local model no request leaves this machine.
    if not _managed_mcp_restricted() or (is_studio_decisions(url) and _local_decisions()):
        return
    if is_stdio(url):
        from fastapi import HTTPException
        raise HTTPException(
            status_code = 400,
            detail = "Only the installation owner may register local-command MCP servers.",
        )
    _public_mcp_address(url)


def _public_http_client_factory(**kwargs):
    """Pin public destinations, including redirects and OAuth discovery; SNI keeps the original hostname and proxy env vars are ignored so they cannot reach the LAN."""
    from mcp.shared._httpx_utils import create_mcp_http_client

    client_type = get_type_hints(create_mcp_http_client)["return"]
    http = importlib.import_module(client_type.__module__.split(".")[0])

    class PublicTransport(http.AsyncBaseTransport):
        def __init__(self):
            # Separate pools retain TLS identity when two names resolve to one IP.
            self.transports = {}

        async def handle_async_request(self, request):
            address = await asyncio.to_thread(_public_mcp_address, str(request.url))
            origin = (request.url.scheme, request.url.host, request.url.port)
            if origin not in self.transports:
                self.transports[origin] = http.AsyncHTTPTransport(trust_env = False)
            pinned = http.Request(
                method = request.method,
                url = request.url.copy_with(host = address),
                headers = request.headers,
                stream = request.stream,
                extensions = {**request.extensions, "sni_hostname": request.url.host},
            )
            return await self.transports[origin].handle_async_request(pinned)

        async def aclose(self):
            for transport in self.transports.values():
                await transport.aclose()

    kwargs["trust_env"] = False
    kwargs["transport"] = PublicTransport()
    kwargs.setdefault("follow_redirects", True)
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = http.Timeout(30.0, read = 300.0)
    return client_type(**kwargs)


def is_stdio(address: str) -> bool:
    """A non-HTTP address is a local stdio command, e.g. 'npx -y
    @modelcontextprotocol/server-filesystem /path'."""
    return not address.strip().lower().startswith(("http://", "https://"))


def _split_windows_command_line(address: str) -> list[str]:
    """Parse a Windows command line using the same backslash/quote rules that
    subprocess.list2cmdline() writes. This keeps trailing backslashes before a closing quote from
    being doubled in the resulting argv."""
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    backslashes = 0
    arg_started = False
    i = 0

    while i < len(address):
        ch = address[i]
        if ch == "\\":
            backslashes += 1
            i += 1
            continue
        if ch == '"':
            current.extend("\\" * (backslashes // 2))
            if backslashes % 2:
                current.append('"')
            else:
                in_quotes = not in_quotes
            arg_started = True
            backslashes = 0
            i += 1
            continue
        # subprocess.list2cmdline() implements the MS C runtime grammar: only space and tab delimit arguments. Other
        # Unicode/control whitespace is ordinary argument data and must not be split here.
        if ch in (" ", "\t") and not in_quotes:
            if backslashes:
                current.extend("\\" * backslashes)
                arg_started = True
                backslashes = 0
            if arg_started or current:
                parts.append("".join(current))
                current = []
                arg_started = False
            i += 1
            while i < len(address) and address[i] in (" ", "\t"):
                i += 1
            continue
        if backslashes:
            current.extend("\\" * backslashes)
            arg_started = True
            backslashes = 0
        current.append(ch)
        arg_started = True
        i += 1

    if backslashes:
        current.extend("\\" * backslashes)
        arg_started = True
    if in_quotes:
        raise ValueError("No closing quotation")
    if arg_started or current:
        parts.append("".join(current))
    return parts


def parse_stdio_command(address: str) -> list[str]:
    """Split a stdio command line into argv. Shared by route validation and the transport so both
    agree on quoting (notably Windows backslash paths)."""
    posix = sys.platform != "win32"
    if posix:
        return shlex.split(address, posix = posix)
    if address.lstrip().startswith("'"):
        raise ValueError("Single-quoted executables are not supported on Windows")
    return _split_windows_command_line(address)


def join_stdio_command(parts: list[str]) -> str:
    """Inverse of parse_stdio_command: join argv into a single command string that
    parse_stdio_command() splits back into ``parts`` on this platform. Config files (issue #5936)
    carry structured command + args; storage holds one string in the url field. Windows uses
    list2cmdline so spaced/backslash paths round-trip through the posix=False quote-strip; posix
    uses shlex."""
    if sys.platform == "win32":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def _windows_batch_argument_is_unsafe(argument: str) -> bool:
    if _WINDOWS_BATCH_ALWAYS_UNSAFE_ARGUMENT_CHARS.intersection(argument):
        return True
    serialized = subprocess.list2cmdline([argument])
    is_quoted = serialized.startswith('"') and serialized.endswith('"')
    return not is_quoted and bool(
        _WINDOWS_BATCH_UNQUOTED_UNSAFE_ARGUMENT_CHARS.intersection(argument)
    )


def _session_log_id(url: str) -> str:
    """A non-secret label for logs. stdio commands can embed credentials in argv (e.g. ``npx server
    --token sk-...``) and HTTP URLs in their query string, so never log the raw address; use the
    executable basename (or the host) plus a short digest of the full address instead."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    if not is_stdio(url):
        try:
            label = urlsplit(url).hostname or "<url>"
        except Exception:  # noqa: BLE001
            label = "<invalid>"
        return f"{label}#{digest}"
    try:
        parts = parse_stdio_command(url)
        exe = os.path.basename(parts[0]) if parts else "<empty>"
    except Exception:  # noqa: BLE001
        exe = "<invalid>"
    return f"{exe}#{digest}"


def stdio_mcp_enabled() -> bool:
    """stdio MCP servers spawn local processes as the backend user (bypassing the sandbox), so allowed
    only when the host is the user's own machine. On startup a loopback bind defaults
    UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 (see utils.host_policy.apply_stdio_mcp_loopback_default, called
    from run.py); the Tauri app does the same. Off for Colab and any network (0.0.0.0) bind unless
    an operator sets the var out-of-band; set it to 0 to force-disable.

    When stdio is on only because of that loopback auto-default, an explicit `unsloth studio run
    --disable-tools` turns it back off (a local stdio command is server-side code execution). An
    explicit operator opt-in via the env var still wins, including the documented `=1` network
    opt-in, where the process tool policy is False merely by the external-host default, not by
    choice.
    """
    if _managed_mcp_restricted():
        return False
    if os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP") != "1":
        return False
    from state.tool_policy import get_tool_policy
    from utils.host_policy import loopback_default_active, remote_connector_active

    if loopback_default_active() and (remote_connector_active() or get_tool_policy() is False):
        return False
    return True


def stdio_mcp_disabled_reason() -> str:
    """User-facing reason local commands are off, mirroring stdio_mcp_enabled().

    Telling a user whose gate is suspended by an active tunnel to set
    UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 would re-enable local command execution on
    a published API, so the suspended cases must name their actual cause."""
    if _managed_mcp_restricted():
        return "Only the installation owner may register local-command MCP servers."
    from state.tool_policy import get_tool_policy
    from utils.host_policy import (
        loopback_default_active,
        remote_connector_active,
        stdio_mcp_withheld_reason,
    )

    if os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP") == "1" and loopback_default_active():
        if remote_connector_active():
            return (
                "Local commands are disabled while Remote Access is on, because the "
                "server is reachable from outside this machine. Turn off Remote Access "
                "to use local MCP servers, or use an http:// or https:// URL instead."
            )
        if get_tool_policy() is False:
            return (
                "Local commands are disabled because tools are disabled for this "
                "server. Restart without --disable-tools, or use an http:// or "
                "https:// URL instead."
            )

    # Name the bind as the cause: a bare "set the env var" reads like a bug to someone who followed the
    # README's `-H 0.0.0.0` LAN example and then tried to add an .exe server.
    withheld = stdio_mcp_withheld_reason()
    if withheld == "network":
        return (
            "Local commands (an .exe, npx, uvx or other local program) are turned off because Unsloth "
            "is listening on your network (started with -H 0.0.0.0 or another non-local address), "
            "and a local program runs with your account's full access. To use them, restart with "
            "`unsloth studio -H 127.0.0.1` to keep Unsloth on this computer, or set "
            "UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 before starting to allow them on the network anyway. "
            "Remote http:// and https:// servers still work."
        )
    if withheld == "colab":
        return (
            "Local commands (stdio MCP servers) are turned off on Colab. "
            "Use an http:// or https:// MCP server URL instead."
        )
    return (
        "Local commands (an .exe, npx, uvx or other local program) aren't enabled on this server. "
        "To allow them, set UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 and restart Unsloth, or use an "
        "http:// or https:// URL instead."
    )


# Probe timeouts for discovering a server's tool list. OAuth needs minutes for first-connect/expired-token browser
# sign-in; stdio allows for first-run package download (e.g. `npx -y ...`); HTTP fails fast.
_HTTP_PROBE_TIMEOUT = 8.0
_OAUTH_PROBE_TIMEOUT = 305.0
_STDIO_PROBE_TIMEOUT = 60.0


def probe_timeout(address: str, use_oauth: bool) -> float:
    if use_oauth:
        return _OAUTH_PROBE_TIMEOUT
    return _STDIO_PROBE_TIMEOUT if is_stdio(address) else _HTTP_PROBE_TIMEOUT


def parse_server_headers(server: dict) -> Optional[dict]:
    """Parsed headers_json. For stdio servers this dict is the process env instead of HTTP headers
    (see _client)."""
    raw = server.get("headers_json")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _oauth_store():
    global _oauth_token_store
    if not is_owner_context():
        from key_value.aio._utils.sanitization import AlwaysHashStrategy
        from key_value.aio.stores.filetree import FileTreeStore
        from utils.paths.storage_roots import account_path, ensure_dir

        account_id = current_account_id()
        if account_id not in _account_oauth_token_stores:
            _account_oauth_token_stores[account_id] = FileTreeStore(
                data_directory = ensure_dir(account_path("mcp-oauth-tokens")),
                key_sanitization_strategy = AlwaysHashStrategy(),
                collection_sanitization_strategy = AlwaysHashStrategy(),
            )
        return _account_oauth_token_stores[account_id]
    if _oauth_token_store is None:
        from key_value.aio._utils.sanitization import AlwaysHashStrategy
        from key_value.aio.stores.filetree import FileTreeStore
        from utils.paths.storage_roots import ensure_dir, studio_root

        # Hash keys/collections: fastmcp uses raw URLs as keys, and FileTreeStore would treat the "://" as nested
        # directories.
        _oauth_token_store = FileTreeStore(
            data_directory = ensure_dir(studio_root() / "mcp-oauth-tokens"),
            key_sanitization_strategy = AlwaysHashStrategy(),
            collection_sanitization_strategy = AlwaysHashStrategy(),
        )
    return _oauth_token_store


def _strip_client_id_under_basic_auth(auth) -> None:
    """The SDK leaves client_id in the body under Basic auth; strict servers (Notion) read that as
    two auth methods -- RFC 6749 2.3 -- and 401. Safe to drop: 3.2.1 makes client_id a MAY once
    authenticated, and an unauthenticated client sends no header, so it keeps it."""
    context = getattr(auth, "context", None)
    prepare = getattr(context, "prepare_token_auth", None)
    if prepare is None:
        logger.warning("MCP OAuth: prepare_token_auth missing; client_id fixup skipped")
        return

    def prepare_token_auth(data, headers = None):
        data, headers = prepare(data, headers)
        if "Authorization" in headers:
            data = {k: v for k, v in data.items() if k != "client_id"}
        return data, headers

    try:
        context.prepare_token_auth = prepare_token_auth
    except Exception as exc:  # noqa: BLE001
        # The SDK reach-in must never break OAuth: without it only strict servers fail, as before.
        logger.warning("MCP OAuth: client_id fixup could not be applied: %s", exc)


def _oauth(
    url: str,
    oauth_client_id: Optional[str] = None,
    oauth_client_secret: Optional[str] = None,
):
    from fastmcp.client.auth import OAuth

    # A pre-registered client skips Dynamic Client Registration (Google's MCP servers have no /register).
    auth = OAuth(
        mcp_url = url,
        token_storage = _oauth_store(),
        client_id = oauth_client_id,
        client_secret = oauth_client_secret,
    )
    _strip_client_id_under_basic_auth(auth)
    return auth


async def clear_oauth_tokens_async(url: str) -> None:
    """Drop any persisted OAuth tokens for ``url``. fastmcp keys tokens by MCP URL, so on server
    delete / URL change / OAuth disable we must clear them, else re-registering the same URL
    reuses the old account's token. Best-effort: store / OAuth failures must not 500 the delete /
    update route."""
    try:
        await _oauth(url).token_storage_adapter.clear()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to clear OAuth tokens for %s: %s", url, exc)


_IS_WINDOWS = os.name == "nt"
_NODE_COMMANDS = frozenset({"node", "npm", "npx"})
_WINDOWS_LAUNCHER_SUFFIXES = (".cmd", ".exe", ".bat", ".ps1")


def _launcher_name(command: str) -> str:
    """argv[0] reduced to its bare launcher name, Windows suffix stripped."""
    name = os.path.basename(command).lower()
    for suffix in _WINDOWS_LAUNCHER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_node_command(command: str) -> bool:
    """Whether argv[0] is a Node launcher. Only those need the managed runtime, so a Python or other
    stdio server keeps the toolchain its own env pinned."""
    return _launcher_name(command) in _NODE_COMMANDS


def _command_selects_runtime(command: Optional[str]) -> bool:
    """A path to a ``node`` launcher picks the runtime explicitly and runs regardless of PATH, so
    handing its children a different Node would only split the two."""
    return (
        command is not None and bool(os.path.dirname(command)) and _launcher_name(command) == "node"
    )


def _runtime_requirements(command: Optional[str]) -> tuple[bool, bool]:
    """``(needs npm, needs npx)`` for argv[0]. Each launcher asks only for what it runs, so an
    unrelated missing launcher cannot shadow a good runtime: node needs neither, npm needs npm,
    and npx needs npx alone -- npx never shells out to an ``npm`` executable, its npx-cli.js
    delegates in-process to the npm library it ships with, so a PATH exposing node and npx
    without a separate npm runs it fine and must be left alone. A pathed npm/npx is already
    located and only needs a node for its shebang, so it does not require a second copy of itself
    on PATH either."""
    name = _launcher_name(command) if command is not None else None
    if name == "node":
        return False, False
    if command is not None and os.path.dirname(command):
        return False, False
    if name == "npm":
        return True, False
    if name == "npx":
        return False, True
    return True, True


def _path_key(env: dict) -> str:
    """The key holding PATH. Windows env names are case-insensitive, so a config may spell it
    ``Path``; on POSIX only the exact name counts."""
    if _IS_WINDOWS:
        for key in env:
            if key.upper() == "PATH":
                return key
    return "PATH"


def _stdio_env(headers: Optional[dict], command: Optional[str] = None) -> Optional[dict]:
    """Process env for a stdio server: its own vars, plus the managed Node bin dir on PATH so ``npx
    ...`` servers spawn on hosts with no usable system Node."""
    env = dict(headers or {})
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("stdio environment names and values must be strings")
        if "\x00" in key or "\x00" in value:
            raise ValueError("stdio environment must not contain NUL characters")
        if "=" in key:
            raise ValueError("stdio environment variable names must not contain '='")
    key = _path_key(env)
    base = env.get(key)
    if isinstance(base, str) and not base:
        # An explicitly empty PATH is a deliberate sandbox: hand it over untouched.
        return env
    if command is not None and not _is_node_command(command):
        return env or None
    if _command_selects_runtime(command):
        return env or None
    if not isinstance(base, str):
        base = os.environ.get("PATH", "")
    try:
        from utils.node_runtime import path_with_managed_node
        require_npm, require_npx = _runtime_requirements(command)
        patched = path_with_managed_node(base, require_npm = require_npm, require_npx = require_npx)
    except (ImportError, OSError, ValueError):
        patched = base
    if patched and patched != env.get(key):
        env[key] = patched
    return env or None


# The SDK hands a stdio child only get_default_environment() (12 names on Windows) under the server's own vars. Real
# programs need more: ProgramFiles/ProgramData to find their own installs (Playwright locating Chrome), ComSpec and
# windir for wrapper scripts, TMP for temp files, and the proxy/CA variables without which every outbound request
# fails behind a corporate proxy or TLS inspection. An allowlist rather than the whole environment, so none of
# Studio's own secrets (HF_TOKEN, provider keys, the UNSLOTH_* settings) reach a third-party program.
_INHERITED_WINDOWS_ENV = (
    "ALLUSERSPROFILE",
    "CommonProgramFiles",
    "CommonProgramFiles(x86)",
    "CommonProgramW6432",
    "COMPUTERNAME",
    "ComSpec",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL",
    "PROCESSOR_REVISION",
    "ProgramData",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "PUBLIC",
    "TMP",
    "USERDOMAIN",
    "windir",
)
_INHERITED_NETWORK_ENV = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)
# POSIX names are case-sensitive and curl, wget and most libraries read the lowercase spelling first.
_INHERITED_POSIX_PROXY_ENV = ("http_proxy", "https_proxy", "no_proxy", "all_proxy")


def _inherited_stdio_env(env: Optional[dict]) -> Optional[dict]:
    """The spawn env: the allowlisted host variables underneath ``env`` (the result of _stdio_env).
    Never PATH: _stdio_argv treats a PATH in the env as the user's explicit, authoritative choice, and
    the SDK's default env already carries the host's. Names the user configured win, compared without
    case even on POSIX, so a configured HTTPS_PROXY is not overridden by an inherited https_proxy
    that curl would read first. Only the spawn sees the result: sessions stay keyed on the
    configured env alone."""
    names = _INHERITED_NETWORK_ENV + (
        _INHERITED_WINDOWS_ENV if _IS_WINDOWS else _INHERITED_POSIX_PROXY_ENV
    )
    configured = {key.upper() for key in (env or {})}
    inherited = {}
    for name in names:
        value = os.environ.get(name)
        if value and name.upper() not in configured:
            inherited[name] = value
    if not inherited:
        return env
    return {**inherited, **(env or {})}


def _stdio_argv(parts: list, env: Optional[dict]) -> list:
    """argv with argv[0] resolved against the child's PATH. Windows resolves the command against the
    parent environment before ``env`` applies, so a managed-only ``npx`` has to be handed over as
    a full path."""
    path_key = _path_key(env or {})
    explicit_path = env is not None and path_key in env
    path = (env or {}).get(path_key)
    if not isinstance(path, str):
        path = os.environ.get("PATH", "")
    try:
        resolved = shutil.which(parts[0], path = path)
    except OSError:
        resolved = None
    if _IS_WINDOWS and resolved is None and explicit_path and not os.path.dirname(parts[0]):
        raise ValueError(f"Cannot find {parts[0]!r} on the MCP server's configured PATH")
    executable = resolved or parts[0]
    if _IS_WINDOWS:
        suffix = os.path.splitext(executable)[1].lower()
        if suffix in {".cmd", ".bat"} and _launcher_name(executable) in {"npm", "npx"}:
            launcher_dir = os.path.dirname(executable)
            cli = os.path.join(
                launcher_dir,
                "node_modules",
                "npm",
                "bin",
                f"{_launcher_name(executable)}-cli.js",
            )
            sibling_node = os.path.join(launcher_dir, "node.exe")
            try:
                node = (
                    sibling_node
                    if os.path.isfile(sibling_node)
                    else shutil.which("node", path = path)
                )
                cli_exists = os.path.isfile(cli)
            except OSError:
                node = None
                cli_exists = False
            if node and cli_exists:
                # bypass cmd.exe so shell metacharacters remain literal argv.
                return [node, cli, *parts[1:]]
            raise ValueError(
                f"Cannot launch {executable!r} without its Node executable and npm CLI script"
            )
        if suffix in {".cmd", ".bat"} and any(
            _windows_batch_argument_is_unsafe(argument) for argument in parts[1:]
        ):
            raise ValueError(
                "Windows batch launchers cannot safely preserve these MCP command arguments; "
                "invoke the executable directly, or use node.exe with the JavaScript entry point"
            )
    return [executable, *parts[1:]]


def _looks_like_path(token: str) -> bool:
    return bool(os.path.dirname(token)) or re.match(r"^[A-Za-z]:", token) is not None


def unquoted_spaced_program(parts: list[str]) -> Optional[str]:
    """The program a command meant when it is a spaced path typed without quotes. ``C:\\My
    Tools\\server.exe --x`` splits into ``C:\\My`` + ``Tools\\server.exe``; when argv[0] is not a
    file but argv[0..k] joined with spaces names one, return that joined path. argv[0] counts only
    as an exact file: CreateProcess-style extension guessing would resolve ``C:\\My`` to a planted
    ``C:\\My.exe``, which is exactly the ambiguity this exists to refuse."""
    if len(parts) < 2 or not _looks_like_path(parts[0]):
        return None
    try:
        if os.path.isfile(parts[0]):
            return None
        for end in range(2, len(parts) + 1):
            candidate = " ".join(parts[:end])
            if os.path.isfile(candidate) or (
                _looks_like_path(candidate) and shutil.which(candidate) is not None
            ):
                return candidate
    except (OSError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Local-program (stdio) start-up diagnostics
#
# Without these, every way a local program fails to start reached the user as one of three opaque strings: a missing
# EXE as "[WinError 2] The system cannot find the file specified" (no file name), a crash or missing env var as
# "Connection closed", a hang as "An internal error occurred" (a timeout carries no message). Its stderr went to the
# backend's raw stderr, outside the session log. Each spawn now writes stderr to a per-command log, and a failed start
# is described from the exception plus the tail of what that spawn printed.
# ---------------------------------------------------------------------------------------------------------------------

_STDIO_LOG_MAX_BYTES = 2 * 1024 * 1024
# More than one account's worth of servers (32 sessions each), so pruning only reaches logs of removed ones.
_STDIO_LOG_KEEP = 64
_STDIO_TAIL_CHARS = 800
# Read at most this much of a spawn's stderr back: the tail is all that is shown, and a chatty server can write
# megabytes before it fails.
_STDIO_TAIL_READ_BYTES = 64 * 1024
# Shorter values ("1", "true", "dev") are too likely to appear in ordinary output to be worth masking.
_REDACT_MIN_CHARS = 4
_SECRET_MASK = "***"
# Argument and query names whose value is a credential. Broader than log_redaction's key list on purpose: there a
# false positive blanks ordinary log text, here it only adds one configured value to the list masked by exact match.
_SECRET_ARG_NAME = re.compile(
    r"(?i)token|secret|passw|api[-_]?key|apikey|auth|credential|private[-_]?key|access[-_]?key|"
    r"session|cookie|signature"
)
_AUTH_SCHEMES = ("bearer", "basic", "digest", "token", "apikey")
# The pump that masks a stdio server's stderr on its way to disk (_MaskedStderrPump).
_PUMP_READ_BYTES = 64 * 1024
# A line with no newline is held for masking until it ends; past this it is written in pieces, cut at a space or
# carriage return where there is one (a progress bar redraws with \r and may never print \n).
_PUMP_MAX_LINE_CHARS = 64 * 1024
# How long a failed start waits for the child's last output to reach the file before quoting it. Short: the
# wait runs on whichever event loop is starting the server, and usually ends at EOF or idle within ~50 ms.
_PUMP_SETTLE_SECONDS = 0.5
# A reader that has sat in an empty read this long has everything written so far: a pipe read returns as soon
# as any byte is there.
_PUMP_IDLE_SECONDS = 0.05
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_CONNECTION_CLOSED_CODE = -32000  # mcp.types.CONNECTION_CLOSED
# Windows CreateFile access and disposition values for _open_append_only.
_FILE_APPEND_DATA = 0x0004
_FILE_READ_ATTRIBUTES = 0x0080
_SYNCHRONIZE = 0x00100000
_FILE_SHARE_ALL = 0x0007
_OPEN_ALWAYS = 4
_FILE_ATTRIBUTE_NORMAL = 0x0080


class McpStdioServerError(RuntimeError):
    """A local MCP program failed in a way Studio could explain. The message is written for the person
    who configured the server: it names the program by its file name only and may quote the tail of
    its output, with configured env values masked. Every line after the first is that quoted output,
    so log the summary and show the whole message."""

    @property
    def summary(self) -> str:
        return (str(self).splitlines() or [""])[0]


def _open_append_only(path: Path):
    """Open ``path`` so every write lands at the current end of the file, including writes by the
    child processes that inherit the handle as their stderr. Two chats using one server run two
    copies of the same command, which share this log. POSIX O_APPEND already appends atomically, but
    Windows only emulates append in the C runtime: a child writes at its handle's own position, so
    the second process overwrites the first one's output (and a truncation leaves the other writing
    past a hole). A handle opened with FILE_APPEND_DATA and no FILE_WRITE_DATA makes the kernel do
    the appending for every holder."""
    if _IS_WINDOWS:
        try:
            import _winapi
            import msvcrt

            handle = _winapi.CreateFile(
                str(path),
                _FILE_APPEND_DATA | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
                _FILE_SHARE_ALL,
                _winapi.NULL,
                _OPEN_ALWAYS,
                _FILE_ATTRIBUTE_NORMAL,
                _winapi.NULL,
            )
            try:
                fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_APPEND)
            except BaseException:
                _winapi.CloseHandle(handle)
                raise
            try:
                return open(fd, "a", encoding = "utf-8", errors = "replace")
            except BaseException:
                os.close(fd)
                raise
        except (ImportError, AttributeError, OSError) as exc:
            logger.debug("Append-only MCP log handle unavailable, using a plain one: %s", exc)
    return open(path, "a", encoding = "utf-8", errors = "replace")


def _program_label(argv0: str) -> str:
    """argv[0] by file name only: messages may reach a model or a shared log, and a full path names
    the user's home directory."""
    return os.path.basename(argv0.rstrip("\\/")) or argv0


def _is_secret_name(name: str) -> bool:
    bare = name.lstrip("-").lower()
    return bool(_SECRET_ARG_NAME.search(bare)) or bare in ("key", "pat")


def _url_secrets(token: str) -> list[str]:
    """Credentials inside one URL: the password of ``user:pass@`` (or the user part alone, the shape
    of ``https://<token>@host``) and the value of every secret-named query parameter."""
    from urllib.parse import parse_qsl

    # --db=postgres://user:pw@host: the URL starts after the flag.
    head = token.partition("://")[0]
    if "=" in head:
        token = token[head.index("=") + 1 :]
    try:
        parts = urlsplit(token)
        found = []
        if parts.password:
            found.append(parts.password)
        elif parts.username:
            found.append(parts.username)
        found.extend(
            value
            for name, value in parse_qsl(parts.query, keep_blank_values = True)
            if _is_secret_name(name)
        )
        return found
    except ValueError:
        return []


def mcp_secret_values(url: str, headers: Optional[dict]) -> list[str]:
    """Every configured value that must not appear in what Studio writes or shows about this server,
    longest first so a value that contains another is masked whole. The one notion of "secret" shared
    by the start-up error tail, the per-command stderr log on disk and the log viewer:

    - every env var or header value (a local program's env rides the headers field). By value, not
      name: Studio cannot tell GITHUB_TOKEN from MY_SERVICE_ID, and a masked id costs less than a
      missed key;
    - the credential behind an auth scheme, so ``Bearer abc...`` also masks a bare ``abc...``;
    - URL credentials and secret-named query values, in the address or in any argument;
    - the value after a secret-named flag (``--token X``, ``--api-key=X``) or in ``NAME=value``.

    Values shorter than _REDACT_MIN_CHARS are left out: "1" or "dev" would blank ordinary output."""
    values: list[str] = [
        value for value in (headers or {}).values() if isinstance(value, str)
    ]
    try:
        tokens = parse_stdio_command(url) if is_stdio(url) else [url]
    except ValueError:
        tokens = []
    for index, token in enumerate(tokens):
        if "://" in token:
            values.extend(_url_secrets(token))
            continue
        name, sep, value = token.partition("=")
        if sep and _is_secret_name(name):
            values.append(value)
        elif (
            token.startswith("-")
            and _is_secret_name(token)
            and index + 1 < len(tokens)
            and not tokens[index + 1].startswith("-")
        ):
            values.append(tokens[index + 1])
    for value in list(values):
        scheme, sep, credential = value.strip().partition(" ")
        if sep and scheme.lower() in _AUTH_SCHEMES:
            values.append(credential.strip())
    unique = {value for value in values if len(value) >= _REDACT_MIN_CHARS}
    return sorted(unique, key = len, reverse = True)


def mask_secret_values(text: str, secrets) -> str:
    for secret in secrets:
        text = text.replace(secret, _SECRET_MASK)
    return text


def stdio_log_owners() -> dict[str, tuple[str, list[str]]]:
    """Per-command stderr log file name -> (the server's display name, its secret values), for every
    saved local-program server of the current account. The log viewer names a file by the server
    that wrote it and masks that server's values on read, which also covers a file written before
    stderr was masked on its way to disk. A command edited since keeps its old file under a digest
    nothing maps any more; that file falls back to its program name and the shape-based redaction."""
    from storage import mcp_servers_db

    owners: dict[str, tuple[str, list[str]]] = {}
    for row in mcp_servers_db.list_servers():
        url = row.get("url") or ""
        if not is_stdio(url):
            continue
        try:
            parts = parse_stdio_command(url)
            headers = json.loads(row.get("headers_json") or "null")
        except (ValueError, TypeError):
            continue
        if not parts:
            continue
        name = str(row.get("display_name") or "").strip() or _program_label(parts[0])
        owners[_stdio_log_name(url, parts[0])] = (
            name,
            mcp_secret_values(url, headers if isinstance(headers, dict) else None),
        )
    return owners


def _stdio_log_name(url: str, argv0: str) -> str:
    """``<program>-<digest>.log``. The digest is _session_log_id's, so a backend log line naming
    ``server.exe#3f2a...`` leads straight to its file, and two commands that share a program
    (``npx a`` / ``npx b``) keep separate logs."""
    stem = os.path.basename(argv0.rstrip("\\/"))
    if stem.lower().endswith(_WINDOWS_LAUNCHER_SUFFIXES):
        stem = os.path.splitext(stem)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:60] or "server"
    return f"{stem}-{hashlib.sha256(url.encode()).hexdigest()[:12]}.log"


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """exc, its causes/contexts, and the members of any exception group on the way (anyio task groups
    raise those), so a FileNotFoundError wrapped as "Client failed to connect: ..." is still found."""
    seen: list[BaseException] = []
    pending = [exc]
    while pending and len(seen) < 32:
        current = pending.pop()
        if current is None or any(current is s for s in seen):
            continue
        seen.append(current)
        pending.extend(getattr(current, "exceptions", None) or ())
        pending.append(current.__cause__)
        pending.append(current.__context__)
    return seen


def _is_connection_closed(chain: list[BaseException]) -> bool:
    """The child's stdout ended before the handshake did, which in practice means it exited."""
    import anyio

    for exc in chain:
        error = getattr(exc, "error", None)
        if (
            getattr(error, "code", None) == _CONNECTION_CLOSED_CODE
            and "closed" in str(getattr(error, "message", "")).lower()
        ):
            return True
        if isinstance(
            exc,
            (
                anyio.ClosedResourceError,
                anyio.BrokenResourceError,
                anyio.EndOfStream,
                BrokenPipeError,
                ConnectionResetError,
            ),
        ):
            return True
        if isinstance(exc, RuntimeError) and "closed unexpectedly" in str(exc):
            return True
    return False


class _StdioProcessLog:
    """Per-spawn state that the SDK's stdout reader can reach. The reader runs in a task created
    (indirectly) by the task that entered the client, so it inherits the context variable set around
    that entry; nothing here has to thread through fastmcp or the SDK."""

    def __init__(self, label: str, redact):
        self.label = label
        self.redact = redact
        self.non_json_lines = 0
        self.recent_non_json: deque[str] = deque(maxlen = 5)


_stdio_process_log: contextvars.ContextVar[Optional[_StdioProcessLog]] = contextvars.ContextVar(
    "unsloth_mcp_stdio_process_log", default = None
)


class _MaskedStderrPump:
    """Stands between a local program's stderr and its log file, so what reaches the disk is masked.

    fastmcp hands ``log_file`` to the SDK, which passes it to the spawn as ``stderr=``: handed the file
    itself, the child wrote to it directly and Studio never saw the bytes, so a server echoing its
    env or request headers (plenty do with --debug) put its API key on disk in the clear, for the log
    viewer and the log export to read later. The child now gets the write end of a pipe; this thread
    reads the other end, masks every line (the configured values first, then the log viewer's
    credential shapes) and appends it to the file.

    Lines, not reads: a read can end in the middle of a secret. An unterminated line is held until
    its newline, EOF or _PUMP_MAX_LINE_CHARS, and offered to the error tail masked, so a program
    stopped at a prompt that printed no newline still shows it.

    The loop never stops reading before EOF, whatever fails: a child whose stderr pipe fills (4 KiB
    on Windows) blocks, and an MCP server blocked on stderr hangs every call. EOF arrives when the
    last holder of the write end exits, which is the server plus any child it started with inherited
    stderr (npx -> node), so the thread and the file handle live exactly as long as that output can."""

    def __init__(self, read_fd: int, sink, mask) -> None:
        self._fd = read_fd
        self._sink = sink
        self._mask = mask
        self._lock = threading.Lock()
        self._pending = ""
        # When the reader entered the read it is blocked in, None while it is processing a chunk.
        self._blocked_since: Optional[float] = None
        self._thread = threading.Thread(
            target = self._run, name = "mcp-stderr-log", daemon = True
        )
        self._thread.start()

    def _clean(self, line: str) -> str:
        line = line.rstrip("\r")
        if "\x00" in line:
            # UTF-16 output (a PowerShell host) arrives with a NUL between ASCII characters, and no
            # mask can match through those. Dropping them leaves the same text.
            line = line.replace("\x00", "")
        return redact_log_text(self._mask(line))

    def _feed(self, text: str, final: bool = False) -> None:
        with self._lock:
            *lines, pending = (self._pending + text).split("\n")
            if final and pending:
                lines.append(pending)
                pending = ""
            while len(pending) > _PUMP_MAX_LINE_CHARS:
                cut = max(
                    pending.rfind(" ", 0, _PUMP_MAX_LINE_CHARS),
                    pending.rfind("\r", 0, _PUMP_MAX_LINE_CHARS),
                )
                if cut <= 0:
                    cut = _PUMP_MAX_LINE_CHARS
                lines.append(pending[:cut])
                pending = pending[cut:]
            self._pending = pending
            if not lines:
                return
            # Under the lock, so the file plus the held line is always everything read so far.
            try:
                self._sink.write("".join(self._clean(line) + "\n" for line in lines))
                self._sink.flush()
            except (OSError, ValueError):
                pass  # a full disk loses log lines, never the server: keep draining

    def _run(self) -> None:
        import codecs

        decoder = codecs.getincrementaldecoder("utf-8")(errors = "replace")
        try:
            while True:
                self._blocked_since = time.monotonic()
                try:
                    chunk = os.read(self._fd, _PUMP_READ_BYTES)
                except OSError:
                    chunk = b""
                self._blocked_since = None
                if not chunk:
                    break
                try:
                    self._feed(decoder.decode(chunk))
                except Exception:  # noqa: BLE001
                    pass  # see the class docstring: whatever fails, keep reading
            self._feed(decoder.decode(b"", final = True), final = True)
        except Exception:  # noqa: BLE001
            pass
        finally:
            for close in (lambda: os.close(self._fd), self._sink.close):
                try:
                    close()
                except (OSError, ValueError):
                    pass

    @contextmanager
    def held_line(self):
        """Pause the pump between lines and yield the unterminated line read so far, masked like the
        file. Read the file inside it: outside, a line finishing between the two reads shows twice
        or not at all."""
        with self._lock:
            yield self._clean(self._pending) if self._pending else ""

    def settle(self, timeout: float = _PUMP_SETTLE_SECONDS) -> None:
        """Wait, briefly, until what the child has written so far is in the file: EOF, or the
        reader idle in an empty read. Sleeps first, so a reader the child's last write just woke gets
        the GIL before this looks: after a crash the output is in the pipe but maybe not yet read."""
        deadline = time.monotonic() + timeout
        while self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
            blocked = self._blocked_since
            if blocked is not None and time.monotonic() - blocked >= _PUMP_IDLE_SECONDS:
                return

    def join(self, timeout: float = _PUMP_SETTLE_SECONDS) -> None:
        self._thread.join(timeout)


class _StdioLaunch:
    """Everything a failed start needs to explain itself, built with the client and attached to its
    transport: the program's file name, where this spawn's stderr goes, and which configured values
    (mcp_secret_values) to mask in that log and in anything quoted back."""

    def __init__(self, url: str, parts: list[str], env: Optional[dict]):
        self.url = url
        self.parts = list(parts)
        self.program = _program_label(parts[0])
        self.label = _session_log_id(url)
        self.secrets = mcp_secret_values(url, env)
        self.log_name = _stdio_log_name(url, parts[0])
        self.log_path: Optional[Path] = None
        self.log_offset = 0
        self._log_created = False
        self._log_handle = None
        self._pump: Optional[_MaskedStderrPump] = None
        self.process_log = _StdioProcessLog(self.label, self.redact)
        self.handshake_done = False

    def redact(self, text: str) -> str:
        return mask_secret_values(text, self.secrets)

    def open_log(self):
        """Open the per-command stderr log for this spawn and return the handle for the transport: the
        write end of a pipe whose output _MaskedStderrPump masks into the file. None falls back to the
        backend's own stderr (the old behaviour) when either cannot be opened. Bounded by truncating
        at spawn once past _STDIO_LOG_MAX_BYTES: the newest start-up is the one worth keeping, and
        rotating files under processes that still hold them gains nothing."""
        from utils.paths.storage_roots import ensure_dir, studio_root

        try:
            path = ensure_dir(Path(studio_root()) / "logs" / "mcp") / self.log_name
            created = False
            try:
                if path.stat().st_size > _STDIO_LOG_MAX_BYTES:
                    with open(path, "r+b") as oversized:
                        oversized.truncate(0)
            except FileNotFoundError:
                created = True
            handle = _open_append_only(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open the stderr log for MCP server %s: %s", self.label, exc)
            return None
        # One file per command, so a server removed from the list left its log behind for good.
        # After the open, so this spawn's file is the protected one; a file another live server
        # still holds open fails to unlink on Windows and is retried next spawn.
        prune_log_dir(path.parent, "*.log", keep = _STDIO_LOG_KEEP, protect = path)
        try:
            # The label, never the command: argv and env can carry credentials.
            handle.write(
                f"\n===== {datetime.now().isoformat(timespec = 'seconds')} start {self.label} =====\n"
            )
            handle.flush()
            self.log_offset = os.fstat(handle.fileno()).st_size
        except OSError:
            self.log_offset = 0
        try:
            read_fd, write_fd = os.pipe()
        except OSError as exc:
            handle.close()
            logger.warning("Could not open the stderr pipe for MCP server %s: %s", self.label, exc)
            return None
        try:
            writer = open(write_fd, "w", encoding = "utf-8", errors = "replace")
        except Exception:  # noqa: BLE001
            os.close(write_fd)
            os.close(read_fd)
            handle.close()
            return None
        self.log_path = path
        self._log_created = created
        self._log_handle = writer
        self._pump = _MaskedStderrPump(read_fd, handle, self.redact)
        return writer

    def close_log(self) -> None:
        """Drop Studio's copy of the pipe's write end once the spawn has happened (or failed): the
        child keeps its own inherited one for as long as it runs, and the pump reads until the last
        copy closes."""
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def discard_unused_log(self) -> None:
        """After a failed start: remove the log this spawn created if nothing but its header reached it,
        so every mistyped path tried in the dialog does not leave an empty file behind."""
        path = self.log_path
        if path is None or not self._log_created:
            return
        if self._pump is not None:
            # No child holds the pipe after a failed spawn, so this is EOF at once; it closes the file,
            # which Windows will not unlink while a plain (non-sharing) handle has it open.
            self._pump.join()
        try:
            if path.stat().st_size == self.log_offset:
                path.unlink()
                self.log_path = None
        except OSError:
            pass

    def stderr_tail(self) -> str:
        """What this spawn printed to stderr, last _STDIO_TAIL_CHARS characters, cleaned for display:
        decoded leniently, ANSI colour and cursor codes removed, configured values masked (again: the
        file already is, unless an older build wrote it). Masked before it is cut, so a secret
        straddling the cut cannot leave half of itself behind. Includes the line the pump still holds,
        so a prompt printed with no newline is quoted."""
        if self.log_path is None:
            return ""
        pump = self._pump
        if pump is not None:
            pump.settle()
        try:
            with pump.held_line() if pump is not None else nullcontext("") as held:
                with open(self.log_path, "rb") as log:
                    size = log.seek(0, os.SEEK_END)
                    start = max(self.log_offset, size - _STDIO_TAIL_READ_BYTES)
                    raw = b""
                    if start < size:
                        log.seek(start)
                        raw = log.read(size - start)
        except OSError:
            return ""
        if not raw and not held:
            return ""
        text = _ANSI_ESCAPE.sub("", raw.decode("utf-8", errors = "replace") + held)
        text = self.redact(text.replace("\r\n", "\n").replace("\r", "\n"))
        text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
        if len(text) > _STDIO_TAIL_CHARS:
            text = "…" + text[-_STDIO_TAIL_CHARS:].lstrip()
        return text

    def output_section(self) -> str:
        tail = self.stderr_tail()
        if tail:
            return (
                f"\nLast output on stderr:\n{tail}"
                f"\n(Full log: logs/mcp/{self.log_name} in the Unsloth Studio folder.)"
            )
        if self.process_log.recent_non_json:
            # Plenty of CLI tools print their usage or error text to stdout, which a stdio server must
            # reserve for MCP messages; the reader saw those lines even though stderr is empty.
            return "\nIt printed this on stdout instead of MCP messages:\n" + "\n".join(
                self.process_log.recent_non_json
            )
        return ""

    def _not_found_message(self) -> str:
        if os.path.dirname(self.parts[0]):
            message = f"Program not found: {self.program}. Check the program path."
        else:
            message = (
                f"Program not found: {self.program}. Check that it is installed and on PATH, "
                "or enter its full path."
            )
        spaced = unquoted_spaced_program(self.parts)
        if spaced is not None:
            message += f' The path contains spaces — wrap it in double quotes: "{spaced}"'
        elif (
            len(self.parts) > 1
            and _looks_like_path(self.parts[0])
            and not self.url.lstrip().startswith(('"', "'"))
        ):
            message += " If the program's path contains spaces, wrap it in double quotes."
        return message

    def startup_error(self, exc: BaseException) -> Optional[McpStdioServerError]:
        """The explained form of a failed start, or None to let ``exc`` through unchanged (nothing to
        add: no recognisable cause and no output)."""
        chain = _exception_chain(exc)
        if any(isinstance(item, McpStdioServerError) for item in chain):
            return None
        if any(isinstance(item, FileNotFoundError) for item in chain):
            return McpStdioServerError(self._not_found_message())
        for item in chain:
            if isinstance(item, OSError) and getattr(item, "winerror", None) == 193:
                return McpStdioServerError(
                    f"{self.program} is not a program Windows can run directly (ERROR_BAD_EXE_FORMAT). "
                    "A script needs its interpreter as the program, for example node or python."
                )
            if isinstance(item, PermissionError):
                return McpStdioServerError(f"Permission denied starting {self.program}.")
        if _is_connection_closed(chain):
            section = self.output_section()
            if not section:
                return McpStdioServerError(
                    "The server exited during startup without printing an error."
                )
            return McpStdioServerError(f"The server exited during startup.{section}")
        section = self.output_section()
        if section:
            return McpStdioServerError(f"{str(exc).strip() or type(exc).__name__}{section}")
        return None

    def timeout_error(self, seconds: Optional[float]) -> McpStdioServerError:
        within = f" within {seconds:g}s" if seconds is not None else ""
        if self.handshake_done:
            head = f"The server started but did not list its tools{within}."
        else:
            head = (
                f"No MCP handshake{within} — the program may be waiting for input, downloading on "
                "first run, or not an MCP stdio server."
            )
        return McpStdioServerError(head + self.output_section())


def _stdio_launch_of(client) -> Optional[_StdioLaunch]:
    launch = getattr(getattr(client, "transport", None), "unsloth_stdio_launch", None)
    return launch if isinstance(launch, _StdioLaunch) else None


async def _start_stdio_client(client, launch: _StdioLaunch):
    """client.__aenter__() for a local program: open this spawn's stderr log just before the spawn
    (building a client must stay free of side effects; plenty are built and never entered), expose
    the per-spawn log state to the SDK's reader, and turn a failed start into an explained one."""
    # fastmcp reads log_file when it spawns, not when the transport is built.
    client.transport.log_file = launch.open_log()
    token = _stdio_process_log.set(launch.process_log)
    try:
        entered = await client.__aenter__()
    except Exception as exc:
        launch.close_log()
        explained = launch.startup_error(exc)
        launch.discard_unused_log()
        if explained is None:
            raise
        raise explained from exc
    finally:
        _stdio_process_log.reset(token)
        launch.close_log()
    launch.handshake_done = True
    return entered


@asynccontextmanager
async def _connected(client):
    """``async with client`` with the stdio start-up diagnostics when the client is a local program."""
    launch = _stdio_launch_of(client)
    if launch is None:
        async with client as entered:
            yield entered
        return
    entered = await _start_stdio_client(client, launch)
    try:
        yield entered
    finally:
        await client.__aexit__(None, None, None)


class _CollapseNonJsonStdout(logging.Filter):
    """The SDK logs a full ERROR traceback for every stdout line that is not JSON-RPC (a start-up
    banner, progress text, a debug print), so one test of a chatty program buried the log under
    dozens of identical tracebacks. Keep the first one per server process as a one-line warning
    naming the server and quoting the line, and drop the rest. Anything else the SDK logs, including
    its other errors, passes untouched. Outside a Studio spawn (no per-process state) it rate-limits
    instead."""

    _MESSAGE = "Failed to parse JSONRPC message"
    _UNSCOPED_INTERVAL = 60.0
    unsloth_collapses_non_json = True

    def __init__(self) -> None:
        super().__init__()
        self._unscoped_next = 0.0

    @staticmethod
    def _offending_line(record: logging.LogRecord) -> Optional[str]:
        exc = record.exc_info[1] if isinstance(record.exc_info, tuple) else None
        errors = getattr(exc, "errors", None)
        if not callable(errors):
            return None
        try:
            value = errors()[0].get("input")
        except Exception:  # noqa: BLE001
            return None
        if value is None:
            return None
        text = _ANSI_ESCAPE.sub("", value if isinstance(value, str) else str(value)).strip()
        return text[:200] + ("…" if len(text) > 200 else "")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if not (isinstance(record.msg, str) and record.msg.startswith(self._MESSAGE)):
                return True
            state = _stdio_process_log.get()
            if state is not None:
                line = self._offending_line(record)
                if line:
                    state.recent_non_json.append(state.redact(line))
                state.non_json_lines += 1
                if state.non_json_lines > 1:
                    return False
                record.msg = (
                    "MCP server %s wrote a line that is not JSON-RPC to stdout; it was skipped. "
                    "Further such lines from this process are not logged. Line: %s"
                )
                record.args = (state.label, state.redact(line) if line else "<unreadable>")
            else:
                now = time.monotonic()
                if now < self._unscoped_next:
                    return False
                self._unscoped_next = now + self._UNSCOPED_INTERVAL
                record.msg = (
                    "An MCP stdio server wrote a line that is not JSON-RPC to stdout; it was skipped. "
                    "Repeats are logged at most once a minute."
                )
                record.args = ()
            record.levelno = logging.WARNING
            record.levelname = "WARNING"
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            return True
        except Exception:  # noqa: BLE001
            return True


def _install_stdio_log_filter() -> None:
    sdk_logger = logging.getLogger("mcp.client.stdio")
    if not any(getattr(f, "unsloth_collapses_non_json", False) for f in sdk_logger.filters):
        sdk_logger.addFilter(_CollapseNonJsonStdout())


def _install_lenient_stdio_decoding() -> None:
    """The SDK decodes a stdio server's stdout strictly (anyio's TextReceiveStream, errors="strict").
    One byte that is not UTF-8 -- a cp1252 "é" from a Windows program printing through the ANSI code
    page -- raises inside the reader task and ends it, and every pending request then waits out its
    full timeout with nothing logged. A UTF-8 BOM, which .NET Framework and PowerShell hosts emit
    with UTF-8 console output, glues itself to the first line, so ``initialize`` never parses and the
    handshake hangs the same way.

    fastmcp builds StdioServerParameters itself and never exposes its encoding_error_handler, so
    replace the class the SDK's reader looks up by name when it runs: invalid bytes become U+FFFD
    (inside a JSON string that keeps the message; anywhere else it fails that one line, which the
    reader already skips) and a leading BOM is dropped. Installed once, and only over the exact class
    expected, so an SDK that restructures its reader keeps its own behaviour instead of breaking
    this import."""
    try:
        import mcp.client.stdio as sdk_stdio
        from anyio.streams.text import TextReceiveStream
    except Exception:  # noqa: BLE001
        return
    current = getattr(sdk_stdio, "TextReceiveStream", None)
    if getattr(current, "unsloth_lenient", False):
        return
    if current is not TextReceiveStream:
        logger.warning(
            "mcp.client.stdio no longer reads stdout through anyio's TextReceiveStream; stdio MCP "
            "servers that print invalid UTF-8 or a BOM may hang until their timeout"
        )
        return

    class _LenientTextReceiveStream(TextReceiveStream):
        unsloth_lenient = True

        def __post_init__(self, encoding: str, errors: str) -> None:
            super().__post_init__(encoding, "replace" if errors == "strict" else errors)
            self._bom_pending = True

        async def receive(self) -> str:
            while True:
                text = await super().receive()
                if self._bom_pending:
                    self._bom_pending = False
                    text = text[1:] if text.startswith("\ufeff") else text
                if text:
                    return text

    sdk_stdio.TextReceiveStream = _LenientTextReceiveStream


_install_stdio_log_filter()
_install_lenient_stdio_decoding()


def _local_decisions() -> bool:
    from core.systemone.catalog import parse_connection
    from utils.systemone_settings import get_model
    return parse_connection(get_model()) is None


def is_studio_decisions(url: str) -> bool:
    from routes.systemone import MCP_PATH
    from utils.host_policy import is_loopback_host

    parts = urlsplit(url)
    return (
        parts.scheme == "http"
        and is_loopback_host(parts.hostname or "")
        and parts.path.rstrip("/") == MCP_PATH
    )


def _client(
    url: str,
    headers: Optional[dict],
    use_oauth: bool = False,
    cwd: Optional[str] = None,
    **oauth,
):
    validate_mcp_address(url)
    from fastmcp import Client

    if is_studio_decisions(url):
        from routes.systemone import decisions_mcp
        return Client(decisions_mcp)

    if is_stdio(url):
        # Belt-and-suspenders: never spawn unless stdio is enabled on this host.
        if not stdio_mcp_enabled():
            raise PermissionError("stdio MCP servers are disabled on this host")
        from fastmcp.client.transports import StdioTransport

        parts = parse_stdio_command(url)
        if not parts:
            raise ValueError(f"Empty stdio command: {url!r}")
        # Validated when saved, but a folder can be deleted later; without this the spawn fails as "file not
        # found" and blames the program.
        if cwd is not None and not os.path.isdir(cwd):
            raise McpStdioServerError(
                f"Working directory not found: {_program_label(cwd)}. "
                "Check the server's Working directory setting."
            )
        # env vars ride the headers field (merged over the SDK default env and the inherited allowlist).
        # keep_alive=False tears the subprocess down so a one-shot call leaves no orphan. No cwd means the
        # backend's own, as before.
        env = _stdio_env(headers, parts[0])
        argv = _stdio_argv(parts, env)
        transport = StdioTransport(
            command = argv[0],
            args = argv[1:],
            env = _inherited_stdio_env(env),
            cwd = cwd,
            keep_alive = False,
        )
        # On the transport it describes, so whichever path enters the client can explain a failed start.
        transport.unsloth_stdio_launch = _StdioLaunch(url, parts, headers)
        return Client(transport)

    from fastmcp.client.transports import SSETransport, StreamableHttpTransport
    from fastmcp.mcp_config import infer_transport_type_from_url

    auth = _oauth(url, **oauth) if use_oauth else None

    transport_cls = (
        SSETransport if infer_transport_type_from_url(url) == "sse" else StreamableHttpTransport
    )
    kwargs = {}
    if _managed_mcp_restricted():
        kwargs["httpx_client_factory"] = _public_http_client_factory
        if auth is not None:
            auth.httpx_client_factory = _public_http_client_factory
    return Client(transport_cls(url = url, headers = headers or None, auth = auth, **kwargs))


def oauth_client_kwargs(row: dict) -> dict:
    # Empty unless configured, so test doubles of _client without these kwargs keep working.
    if not row.get("oauth_client_id"):
        return {}
    return {
        "oauth_client_id": row["oauth_client_id"],
        "oauth_client_secret": row.get("oauth_client_secret"),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Per-server lifecycle of a local program
#
# "shared": one long-lived process serves every chat of the account, as in Claude Desktop. A server that holds a
# serial port, keeps SSH sessions open or loads toolsets at runtime can only work this way: started once per chat, the
# second copy cannot open the port and each chat sees its own toolsets. Calls stay serialized on its one stdio stream.
# "per_chat": a process per conversation, the isolation every server had before this setting existed. HTTP servers
# have no process and ignore both settings.
# ---------------------------------------------------------------------------------------------------------------------

PROCESS_MODE_SHARED = "shared"
PROCESS_MODE_PER_CHAT = "per_chat"
PROCESS_MODES = (PROCESS_MODE_SHARED, PROCESS_MODE_PER_CHAT)
# Seconds a process may sit unused before it is stopped; 0 = never.
IDLE_TIMEOUT_CHOICES = (60, 300, 1800, 7200, 0)
# per_chat keeps the 300 s every chat process had before the setting existed.
DEFAULT_IDLE_TIMEOUT = {PROCESS_MODE_SHARED: 1800, PROCESS_MODE_PER_CHAT: 300}
# How long a call to a shared server waits behind another chat's call before it says so instead of waiting on.
_SHARED_BUSY_GRACE = 10.0


class McpLifecycle(NamedTuple):
    """One saved local program's process settings. idle_ttl is math.inf for never."""

    server_id: str
    shared: bool
    idle_ttl: float


def server_process_mode(row: dict) -> str:
    mode = row.get("process_mode")
    return mode if mode in PROCESS_MODES else PROCESS_MODE_PER_CHAT


def server_idle_timeout(row: dict) -> int:
    """The row's idle timeout in seconds, 0 for never; a row that never chose one gets its mode's default."""
    value = row.get("idle_timeout_seconds")
    if isinstance(value, int) and not isinstance(value, bool) and value in IDLE_TIMEOUT_CHOICES:
        return value
    return DEFAULT_IDLE_TIMEOUT[server_process_mode(row)]


def server_lifecycle(row: Optional[dict]) -> Optional[McpLifecycle]:
    """How a saved local program's processes live, or None for anything without a process (HTTP, unsaved)."""
    if not row or not row.get("id") or not is_stdio(row.get("url") or ""):
        return None
    idle = server_idle_timeout(row)
    return McpLifecycle(
        str(row["id"]),
        server_process_mode(row) == PROCESS_MODE_SHARED,
        math.inf if idle == 0 else float(idle),
    )


def shared_session_scope(server_id: str) -> str:
    """The session scope of a shared server's one process. Chat scopes start "s=" and ephemeral ones "request-",
    so this can collide with neither."""
    return f"server={server_id}"


class McpServerBusy(RuntimeError):
    """A shared local program is running another chat's call. Calls on its one stdio stream stay serialized, so this
    one would only wait; it reports instead of hanging silently."""

    def __init__(self, tool: Optional[str], seconds: float):
        self.tool = tool
        self.seconds = seconds
        running = f" (running '{tool}' for {int(seconds)}s)" if tool else ""
        super().__init__(
            f"the server is busy with another chat{running}. A shared server runs one call at a time; "
            "try again when that call finishes."
        )


_SESSION_IDLE_TTL = 300.0
_SESSION_REAP_INTERVAL = 30.0
_STDIO_CONNECT_TIMEOUT = 60.0  # allows first-run `npx -y ...` package download
_SESSION_CLOSE_TIMEOUT = 10.0
_SESSION_WEDGE_MARGIN = 15.0
_SESSION_LIVENESS_TIMEOUT = 5.0
_CANCEL_UNWIND_TIMEOUT = 2.0
# An HTTP session idle this long is re-proved with tools/list before the next dispatch: the server may have expired it
# (MCP says a server MAY terminate a session at any time), and no HTTP transport exposes a liveness probe we can ask
# instead -- see _transport_dead.
_HTTP_IDLE_RECHECK = 30.0
_DEFAULT_MAX_SESSIONS = 32


def _max_sessions_from_env() -> int:
    # The cap covers stdio and HTTP sessions alike now, so the stdio-specific name is wrong; keep honouring it so
    # deployments that already set it do not silently jump back to the default.
    raw = os.environ.get("UNSLOTH_STUDIO_MAX_MCP_SESSIONS")
    if raw is None:
        raw = os.environ.get("UNSLOTH_STUDIO_MAX_STDIO_MCP_SESSIONS")
    if raw is None:
        return _DEFAULT_MAX_SESSIONS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MAX_SESSIONS


_MAX_SESSIONS = _max_sessions_from_env()


def _connect_window(url: str, timeout: Optional[float]) -> Optional[float]:
    """How long connecting may take, out of the caller's remaining budget. stdio keeps the
    cold-start cap: a first run may download a package before the server says anything. HTTP has
    no such phase, and capping it would reject connections the caller explicitly allowed time
    for, which is what the one-shot path it replaced always did."""
    if timeout is None or not is_stdio(url):
        return timeout
    return min(timeout, _STDIO_CONNECT_TIMEOUT)


class _ConnectTimeout(asyncio.TimeoutError):
    """Ran out of time before the transport was up. Carries the window that actually expired, which
    is not the caller's timeout when stdio's cold-start cap is the tighter bound, and for a local
    program that was spawned, the explained form (see _StdioLaunch.timeout_error)."""

    def __init__(self, window: Optional[float], detail: Optional[str] = None):
        super().__init__()
        self.window = window
        self.detail = detail


def _is_tool_error(exc: BaseException) -> bool:
    """A tool-level failure (the tool ran and errored) leaves the transport alive, so the session is
    kept; fastmcp raises ToolError for these. Anything else from call_tool is transport-level.
    Version-safe (fastmcp 3.0.2 has no dead probe)."""
    try:
        from fastmcp.exceptions import ToolError
    except Exception:  # noqa: BLE001
        return False
    return isinstance(exc, ToolError)


def _is_protocol_error(exc: BaseException) -> bool:
    """A JSON-RPC error response, as opposed to a broken connection. A FastMCP server answers an
    unknown tool or bad arguments with a result carrying is_error, but the MCP spec also lets a
    server report those as a protocol error, and plenty of non-FastMCP servers do. fastmcp
    surfaces that as MCPError (McpError before the rename) carrying the ErrorData the server
    sent. Receiving it proves the connection is working, so retiring the session over it would
    throw away the chat's server-side state for a mistyped tool name. The caller still marks the
    session for a probe before reuse."""
    for module, name in (
        ("mcp.shared.exceptions", "MCPError"),
        ("mcp.shared.exceptions", "McpError"),
    ):
        try:
            cls = getattr(__import__(module, fromlist = [name]), name, None)
        except Exception:  # noqa: BLE001
            continue
        if cls is not None and isinstance(exc, cls):
            # Only when the server actually sent an error object; a synthetic MCPError with nothing behind it stays
            # transport-level.
            return getattr(exc, "error", None) is not None
    return False


def _transport_dead(session) -> bool:
    """Best-effort, version-adaptive liveness probe for a cached client. ``Client.is_connected()``
    only checks a session object exists, not that the subprocess (or the server's HTTP session)
    is alive, so it is never used here. Returns True only when the transport is positively gone;
    unknown returns False (the call surfaces it). Only the stdio transport answers:
    ``_is_session_dead``/``_connect_task`` are ``StdioTransport`` internals, and neither
    ``StreamableHttpTransport`` nor ``SSETransport`` has ever carried them (checked on fastmcp
    3.0.2 and 4.0.0). For HTTP this returns "unknown" every time, which is why an idle HTTP
    session is re-proved with tools/list instead -- see _needs_idle_recheck."""
    client = getattr(session, "client", None)
    if client is None:
        return True
    transport = getattr(client, "transport", None)
    probe = getattr(transport, "_is_session_dead", None)
    if callable(probe):
        try:
            if probe():
                return True
        except Exception:  # noqa: BLE001
            pass
    connect_task = getattr(transport, "_connect_task", None)
    if connect_task is not None:
        try:
            if connect_task.done():
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _needs_idle_recheck(session, idle_for: float, remaining: Optional[float]) -> bool:
    """Whether an idle HTTP session must prove itself before the next dispatch. A server MAY drop an
    HTTP session whenever it likes, and the client only learns on the next request, which would
    surface as a failed tool call the user has to retry by hand. stdio is exempt: _transport_dead
    answers there, and a live subprocess does not expire on its own. Skipped when the caller
    cannot afford it: the probe exists to save someone a failed call, so spending their whole
    budget on it (tool_call_timeout goes down to 1s) would cause the very failure it is meant to
    avoid."""
    if is_stdio(session.url):
        return False
    # Negative means this borrower connected the session itself, so the handshake it just completed is proof enough.
    if idle_for < 0.0 or idle_for < _HTTP_IDLE_RECHECK:
        return False
    return remaining is None or remaining > _SESSION_LIVENESS_TIMEOUT * 2


def _session_responsive(
    session,
    budget: Optional[float] = None,
    cancel_event = None,
    timeout_is_fatal: bool = True,
) -> bool:
    """Whether a session left dirty by an abandoned call can be reused: the server must answer inside
    ``budget`` (the caller's remaining deadline). Proves the server is alive, not that the abandoned
    call finished -- MCP requests are concurrent. Probes with a raw single-page tools/list: ping
    answers "Method not found" on a modern-era connection, and list_tools() auto-paginates up to 250
    pages.

    ``timeout_is_fatal`` separates the two callers. A dirty session is under suspicion, so silence
    within the window condemns it. An idle one is only being spot-checked: a slow answer says
    nothing about whether the transport is gone, and retiring it there would throw away the very
    state this cache exists to keep. Only a definite failure retires that one.
    """
    client = session.client
    if client is None:
        return False
    window = _SESSION_LIVENESS_TIMEOUT if budget is None else min(_SESSION_LIVENESS_TIMEOUT, budget)
    if window <= 0:
        # No budget left to ask in; that is not evidence either way.
        return not timeout_is_fatal
    probe = getattr(client, "list_tools_mcp", None) or client.list_tools
    try:
        # margin=0: a wedged loop must fail inside the window, not 15s past it.
        session.run(_race_tool_call(probe(), window, cancel_event), window, margin = 0.0)
    except _MCPCancelled:
        raise
    except (asyncio.TimeoutError, _SessionWedged):
        return not timeout_is_fatal
    except Exception as exc:  # noqa: BLE001
        # A JSON-RPC error (a rate limit, a permission rule on tools/list) is the server answering on this very
        # session, exactly as it is for call_tool. Reconnecting would throw away the chat's state over a reply that
        # proves the transport works.
        if not _is_protocol_error(exc):
            return False
    session.dirty = False
    session.proved_at = time.monotonic()
    return True


class _SessionWedged(Exception):
    pass


class _SessionClosed(Exception):
    """The session was closed (server update/delete/shutdown) mid-call."""


def _abort_future(future) -> None:
    # Let the cancelled coroutine unwind before its loop is stopped.
    future.cancel()
    try:
        future.result(1.0)
    except BaseException:  # noqa: BLE001
        pass


def _new_client(
    url: str,
    headers: Optional[dict],
    use_oauth: bool = False,
    cwd: Optional[str] = None,
    **oauth,
):
    # cwd is passed only when there is one: an HTTP server never has one, and most local programs don't either, so
    # every existing stand-in for _client (the session test doubles) keeps fitting. The pre-registered OAuth client
    # kwargs are likewise empty unless configured (see oauth_client_kwargs).
    if cwd is None:
        return _client(url, headers, use_oauth, **oauth)
    return _client(url, headers, use_oauth, cwd = cwd, **oauth)


def _is_tool_list_changed(message: Any) -> bool:
    root = getattr(message, "root", message)
    return getattr(root, "method", None) == "notifications/tools/list_changed"


def _watch_tool_list(client, session: "_McpSession") -> None:
    """Route the server's notifications/tools/list_changed to ``session``, ahead of fastmcp's own handler. fastmcp
    takes the handler at construction only and _client() builds clients for every path, so it is wrapped in place;
    a client without that slot (a test double, a future fastmcp) simply never reports a change. The SDK awaits the
    handler inside its read loop, so it only flips flags."""
    kwargs = getattr(client, "_session_kwargs", None)
    if not isinstance(kwargs, dict):
        return
    inner = kwargs.get("message_handler")
    # Weak: the client is the session's, and must not keep a closed session alive.
    owner = weak_ref(session)

    async def handler(message) -> None:
        if _is_tool_list_changed(message):
            target = owner()
            if target is not None:
                target.note_tools_changed()
        if inner is not None:
            await inner(message)

    kwargs["message_handler"] = handler


class _McpSession:
    def __init__(
        self,
        url: str,
        headers: Optional[dict],
        use_oauth: bool = False,
        cwd: Optional[str] = None,
        lifecycle: Optional[McpLifecycle] = None,
    ):
        # A cached session is built by _client(url, headers) with no auth, so an OAuth server must never reach here.
        # call_tool_sync already routes it to the one-shot path; this makes a future routing slip fail loudly rather
        # than quietly talk to an OAuth server unauthenticated.
        if use_oauth and not is_stdio(url):
            raise ValueError("OAuth MCP servers cannot use a shared session")
        self.url = url
        self.account_id = current_account_id()
        self.headers = headers
        self.cwd = cwd
        self.client = None
        # Set by connect() for a local program, so a connect timeout can say what the program printed.
        self.launch: Optional[_StdioLaunch] = None
        self.closed = threading.Event()
        self.defunct = False  # discarded; close once in_flight drains (see _retire)
        self.dirty = False  # a call was abandoned on it; ping before reuse
        self._close_lock = threading.Lock()
        self.call_lock = threading.Lock()
        # One stdio subprocess is one ordered byte stream, so overlapping calls must not interleave on it. HTTP has no
        # such constraint: every JSON-RPC message is its own POST and the spec lets a client keep several streams open
        # at once, so serializing it would only undo the parallelism the one-shot path had.
        self.serialize_calls = is_stdio(url)
        self.last_used = time.monotonic()
        # When the transport was last shown to be alive, as opposed to merely borrowed. Checkout refreshes last_used
        # immediately, so the idle gap the recheck needs has to be measured from here or a second borrower arriving
        # during the first one's probe would see no gap at all.
        self.proved_at = self.last_used
        self.in_flight = 0  # guarded by _mcp_sessions_lock
        # The saved server this process belongs to, and how long it may idle (None: _SESSION_IDLE_TTL, read at reap
        # time). A shared process is one per server, so it is left out of the per-account LRU cap.
        self.server_id = lifecycle.server_id if lifecycle is not None else None
        self.shared = bool(lifecycle is not None and lifecycle.shared)
        self.idle_ttl: Optional[float] = lifecycle.idle_ttl if lifecycle is not None else None
        self.started_at: Optional[float] = None  # wall clock, once connected
        self.started_mono: Optional[float] = None
        # (caller, tool, since) of the call holding call_lock, so a waiter can say what it is waiting behind.
        self.call_holder: Optional[tuple[Optional[str], str, float]] = None
        # notifications/tools/list_changed: bumped per notification. A shared process's change goes to the server's
        # tool cache; a per-chat one's only to this chat, which re-reads the list over this session into `tools`.
        self.tools_epoch = 0
        self.tools_changed = False
        self.tools_stale = False
        self.tools: Optional[list[dict]] = None
        # On Windows a bare new_event_loop() can be a SelectorEventLoop (if any component set that policy), which
        # cannot spawn subprocesses natively; force a ProactorEventLoop so the stdio transport always works.
        if sys.platform == "win32":
            self.loop = asyncio.ProactorEventLoop()
        else:
            self.loop = asyncio.new_event_loop()
        self._thread = account_thread(target = self._run_loop, name = "mcp-session", daemon = True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def note_tools_changed(self) -> None:
        """The server sent notifications/tools/list_changed on this session. Runs on the loop thread."""
        self.tools_epoch += 1
        self.tools_changed = True
        self.tools_stale = True
        if self.shared and self.server_id:
            _mark_tools_stale(self.account_id, self.server_id)

    def connect(self, timeout: Optional[float], cancel_event) -> None:
        async def _open():
            client = _new_client(self.url, self.headers, cwd = self.cwd)
            _watch_tool_list(client, self)
            launch = _stdio_launch_of(client)
            self.launch = launch
            if launch is None:
                await client.__aenter__()
            else:
                await _start_stdio_client(client, launch)
            # Publish on the loop thread with no await in between: if an abort races a just-completed connect, close()
            # still sees the client and __aexit__s it instead of orphaning the subprocess.
            self.client = client
            self.started_at = time.time()
            self.started_mono = time.monotonic()
            return client

        future = asyncio.run_coroutine_threadsafe(_open(), self.loop)
        # timeout=None means unlimited (no connect deadline); a finite caller timeout bounds connect by
        # _connect_window(), which caps stdio at its cold-start limit and hands HTTP the whole remaining budget.
        window = _connect_window(self.url, timeout)
        deadline = None if window is None else time.monotonic() + window
        while True:
            if cancel_event is not None and cancel_event.is_set():
                _abort_future(future)
                raise _MCPCancelled
            try:
                future.result(0.05)
                return
            except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
                if future.done():
                    raise  # the connect itself failed fast; don't wait out the window
                if deadline is not None and time.monotonic() >= deadline:
                    _abort_future(future)
                    launch = self.launch
                    detail = None if launch is None else str(launch.timeout_error(window))
                    raise _ConnectTimeout(window, detail)

    def is_connected(self) -> bool:
        client = self.client
        if client is None:
            return False
        probe = getattr(client, "is_connected", None)
        try:
            return bool(probe()) if callable(probe) else True
        except Exception:
            return False

    def run(
        self,
        coro,
        timeout: Optional[float],
        margin: float = _SESSION_WEDGE_MARGIN,
    ):
        self.last_used = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        # The coroutine enforces the tool timeout; the margin only catches a wedged loop. No deadline at all when the
        # caller set none -- but poll so a session closed under us (server update/delete) can't hang the request
        # thread forever on a stopped loop. Callers whose whole budget is the timeout (the liveness probe) pass
        # margin=0.
        deadline = None if timeout is None else time.monotonic() + timeout + margin
        try:
            while True:
                try:
                    return future.result(0.25)
                except concurrent.futures.CancelledError:
                    # Only close() cancels in-flight tasks (in _shutdown).
                    raise _SessionClosed
                except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
                    if future.done():
                        raise  # the call's own timeout; the session stays usable
                    if self.closed.is_set():
                        future.cancel()
                        raise _SessionClosed
                    if deadline is not None and time.monotonic() >= deadline:
                        future.cancel()
                        raise _SessionWedged
        finally:
            self.last_used = time.monotonic()

    def close(self) -> None:
        # Setting `closed` first also unblocks run() waiters (they poll it).
        with self._close_lock:
            if self.closed.is_set():
                return
            self.closed.set()
        loop = getattr(self, "loop", None)
        loop_alive = loop is not None and not loop.is_closed()
        if loop_alive:

            async def _shutdown() -> None:
                # Runs on the loop thread, so it serializes with an aborted connect() that finished anyway and just
                # published its client.
                client, self.client = self.client, None
                if client is not None:
                    await client.__aexit__(None, None, None)
                # Cancel in-flight calls so they unwind before loop.stop (their run() waiters have already been
                # released via `closed`).
                for task in asyncio.all_tasks():
                    if task is not asyncio.current_task():
                        task.cancel()

            try:
                asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(_SESSION_CLOSE_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MCP session close failed for %s: %s",
                    _session_log_id(getattr(self, "url", "")),
                    exc,
                )
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        else:
            self.client = None
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout = 5.0)
        if getattr(self, "shared", False) and getattr(self, "tools_changed", False) and self.server_id:
            # The next process starts with its initial tool list, not the one this process changed into (a loaded
            # toolset is gone with it), so the next send re-lists instead of offering tools nothing serves.
            _mark_tools_stale(self.account_id, self.server_id)


_mcp_sessions: dict[tuple, _McpSession] = {}


# Per-key locks so a slow connect/close never blocks unrelated servers; the global lock only guards the dicts.
class _McpKeyLock:
    """A per-key lock that can be removed once nobody references it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0  # guarded by _mcp_sessions_lock


_mcp_key_locks: dict[tuple, _McpKeyLock] = {}
_mcp_sessions_lock = threading.Lock()
_mcp_reaper_started = False
# Sessions discarded while somebody else was mid-call, closed by one worker off the request path. See _close_detached.
_mcp_cleanup_lock = threading.Lock()
_mcp_cleanup_queue: list = []
# Depth past which an evicting caller closes the overflow itself. See _close_detached.
_MAX_PENDING_CLOSES = 8
# How wide close_mcp_sessions fans out. See _close_all.
_MAX_CLOSE_THREADS = 16
_mcp_cleanup_worker: Optional[threading.Thread] = None
# close_mcp_sessions() can only close sessions already published in _mcp_sessions; one still inside connect() would be
# missed and then cached already stale. Bump a generation on every close so that connect discards its session instead
# of publishing it. Guarded by _mcp_sessions_lock.
_mcp_close_all_gen = 0
_mcp_account_close_gen: dict[str, int] = {}
_mcp_url_close_gen: dict[str | tuple[str, str], int] = {}
_mcp_cfg_close_gen: dict[str | tuple[str, str], int] = {}
_mcp_cwd_close_gen: dict[str | tuple[str, str], int] = {}
_mcp_connects_in_flight = 0

_ANY_HEADERS = object()
_ANY_CWD = object()


def _headers_key(headers: Optional[dict]) -> tuple:
    return tuple(sorted((headers or {}).items()))


def _url_close_key(url: str) -> str | tuple[str, str]:
    # Commands/URLs (token args, embedded credentials) and env values can hold
    # secrets and these maps are never pruned; key by digest so closed/edited
    # configs don't retain them in memory forever.
    return _account_key(hashlib.sha256(url.encode()).hexdigest())


def _cfg_close_key(url: str, headers: Optional[dict]) -> str | tuple[str, str]:
    return _account_key(hashlib.sha256(repr((url, _headers_key(headers))).encode()).hexdigest())


def _cwd_close_key(url: str, headers: Optional[dict], cwd: Optional[str]) -> str | tuple[str, str]:
    return _account_key(
        hashlib.sha256(repr((url, _headers_key(headers), cwd or "")).encode()).hexdigest()
    )


def _mcp_close_generation(
    url: str, headers: Optional[dict], cwd: Optional[str] = None
) -> tuple[tuple[int, int], int, int, int]:
    return (
        (_mcp_close_all_gen, _mcp_account_close_gen.get(current_account_id(), 0)),
        _mcp_url_close_gen.get(_url_close_key(url), 0),
        _mcp_cfg_close_gen.get(_cfg_close_key(url, headers), 0),
        _mcp_cwd_close_gen.get(_cwd_close_key(url, headers, cwd), 0),
    )


def _session_key(
    url: str, headers: Optional[dict], scope: Optional[str], cwd: Optional[str] = None
) -> tuple:
    # The working directory is part of what the subprocess is, like its env: two rows running the same command from
    # different folders must not share one process.
    return (url, _headers_key(headers), scope or "", current_account_id(), cwd or "")


def _session_cwd(key: tuple) -> str:
    return key[4] if len(key) > 4 else ""


def _checkout_session(key: tuple) -> tuple[Optional[_McpSession], float]:
    """Returns the session and how long it had been unused, measured before last_used is refreshed.
    The idle gap is returned rather than stored on the session because HTTP borrowers run
    concurrently: a second checkout would otherwise overwrite the first one's gap with a
    near-zero value and talk it out of proving a session that really had gone stale."""
    session = _mcp_sessions.get(key)
    if session is not None and session.is_connected():
        now = time.monotonic()
        idle_for = now - session.proved_at
        session.last_used = now
        session.in_flight += 1
        return session, idle_for
    return None, -1.0


def _borrow_key_lock(key: tuple) -> _McpKeyLock:
    """Return a stable per-key lock while a caller waits for/connects it."""
    key_lock = _mcp_key_locks.setdefault(key, _McpKeyLock())
    key_lock.users += 1
    return key_lock


def _discard_key_lock(key: tuple) -> None:
    key_lock = _mcp_key_locks.get(key)
    if key_lock is not None and key_lock.users == 0 and key not in _mcp_sessions:
        _mcp_key_locks.pop(key, None)


def _return_key_lock(key: tuple, key_lock: _McpKeyLock) -> None:
    with _mcp_sessions_lock:
        key_lock.users -= 1
        _discard_key_lock(key)


@contextmanager
def _connect_slot(url: str, headers: Optional[dict], cwd: Optional[str] = None):
    global _mcp_connects_in_flight
    with _mcp_sessions_lock:
        generation = _mcp_close_generation(url, headers, cwd)
        _mcp_connects_in_flight += 1
    try:
        yield generation
    finally:
        with _mcp_sessions_lock:
            _mcp_connects_in_flight -= 1


class _NoLiveSession(Exception):
    """A caller that must not start a process (spawn=False) found none running."""


def _get_session(
    url: str,
    headers: Optional[dict],
    scope: Optional[str],
    deadline,
    cancel_event,
    config_check,
    use_oauth: bool = False,
    cwd: Optional[str] = None,
    lifecycle: Optional[McpLifecycle] = None,
    spawn: bool = True,
) -> tuple[_McpSession, float]:
    """``deadline`` is the caller's absolute monotonic budget (None = no limit): the key-lock wait
    and the connect share it, so a slow startup can't stack full timeout windows (see
    _call_session_tool). Returns the session and this borrower's idle gap (negative when we
    connected it ourselves), which only the borrower may act on -- see _checkout_session.
    ``lifecycle`` binds the session to its saved server (an edited idle timeout reaches a running
    process on its next use); ``spawn=False`` raises _NoLiveSession rather than start one."""
    global _mcp_reaper_started
    key = _session_key(url, headers, scope, cwd)
    with _mcp_sessions_lock:
        session, idle_for = _checkout_session(key)
        if session is not None:
            if lifecycle is not None:
                session.idle_ttl = lifecycle.idle_ttl
            return session, idle_for
        if not spawn:
            raise _NoLiveSession
        key_lock = _borrow_key_lock(key)
    try:
        # Poll the acquire with connect()'s deadline/cancel semantics: a second same-scope call must not block
        # uncancellably behind another caller's slow startup (e.g. a first-run npx download).
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        # timeout=None means no key-lock deadline (only cancel unblocks it). The wait is bounded exactly like the
        # connect it is queueing behind, so an HTTP caller is not cut off at stdio's cold-start cap here either.
        window = _connect_window(url, remaining)
        lock_deadline = None if window is None else time.monotonic() + window
        while not key_lock.lock.acquire(timeout = 0.05):
            if cancel_event is not None and cancel_event.is_set():
                raise _MCPCancelled
            if lock_deadline is not None and time.monotonic() >= lock_deadline:
                raise _ConnectTimeout(window)
        try:
            stale = None
            with _mcp_sessions_lock:
                session, idle_for = _checkout_session(key)
                if session is not None:
                    if lifecycle is not None:
                        session.idle_ttl = lifecycle.idle_ttl
                    return session, idle_for
                if key in _mcp_sessions:
                    stale = _mcp_sessions.pop(key)
            if stale is not None:
                _retire_session(stale)
            with _connect_slot(url, headers, cwd) as generation:
                session = (
                    _McpSession(url, headers, use_oauth, cwd)
                    if lifecycle is None
                    else _McpSession(url, headers, use_oauth, cwd, lifecycle = lifecycle)
                )
                try:
                    session.connect(
                        None if deadline is None else max(0.0, deadline - time.monotonic()),
                        cancel_event,
                    )
                except _MCPCancelled:
                    session.close()
                    raise
                except Exception as exc:
                    _record_server_error(url, headers, cwd, exc)
                    session.close()
                    raise
                _clear_server_error(url, headers, cwd)
                if config_check is not None:
                    try:
                        current = bool(config_check())
                    except Exception:  # noqa: BLE001
                        current = False
                    if not current:
                        session.close()
                        raise RuntimeError("MCP server was updated or removed while connecting")
                evicted: list = []
                with _mcp_sessions_lock:
                    closed_while_connecting = _mcp_close_generation(url, headers, cwd) != generation
                    if not closed_while_connecting:
                        session.in_flight = 1
                        evicted = _evict_lru_locked()  # bound the cache (LRU idle)
                        _mcp_sessions[key] = session
                        if not _mcp_reaper_started:
                            _mcp_reaper_started = True
                            threading.Thread(
                                target = _session_reaper, name = "mcp-session-reaper", daemon = True
                            ).start()
                            atexit.register(close_mcp_sessions, all_accounts = True)
                for victim in evicted:
                    logger.info("Evicting LRU idle MCP session: %s", _session_log_id(victim.url))
                if evicted:
                    # Detached: these belong to other scopes and nobody is waiting on them, but an unresponsive
                    # transport costs _SESSION_CLOSE_TIMEOUT each. Charging that to this caller would spend the
                    # deadline meant for their tool call.
                    _close_detached(evicted)
                if closed_while_connecting:
                    session.close()
                    raise RuntimeError("MCP server was updated or removed while connecting")
                return session, -1.0
        finally:
            key_lock.lock.release()
    finally:
        _return_key_lock(key, key_lock)


def _release_session(session: _McpSession, defer_close: bool = False) -> None:
    victims: list = []
    with _mcp_sessions_lock:
        session.in_flight = max(0, session.in_flight - 1)
        session.last_used = time.monotonic()
        close_now = session.defunct and session.in_flight == 0
        # Re-enforce the cap once a burst's sessions go idle. Insert-time eviction only trims idle sessions, so it can
        # overshoot while every cached session is busy; reclaim that overshoot here instead of waiting for the idle
        # reaper. Never evict the session we just used (its last_used is newest).
        # The cap is per account: finishing a call must not close a cached session of an account under its own limit.
        account_id = getattr(session, "account_id", None) or current_account_id()
        # A shared process is one per saved server, already bounded, and holds the state shared mode exists to keep;
        # it neither counts toward the cap nor is evicted by it.
        mine = {
            k: s
            for k, s in _mcp_sessions.items()
            if (k[3] if len(k) > 3 else OWNER_ACCOUNT_ID) == account_id
            and not getattr(s, "shared", False)
        }
        while len(mine) > _MAX_SESSIONS:
            idle = [
                (s.last_used, k) for k, s in mine.items() if s.in_flight == 0 and s is not session
            ]
            if not idle:
                break
            _, oldest = min(idle, key = lambda item: item[0])
            victims.append(_mcp_sessions.pop(oldest))
            mine.pop(oldest)
            _discard_key_lock(oldest)
    if close_now and defer_close:
        # This borrower was the last one on a session that has been discarded, either by its own failure or by a
        # sibling's. Either way the caller is mid-request -- it may still have a reconnect and retry to do on the same
        # deadline -- and a transport that is being discarded because it stopped answering is exactly the one whose
        # close runs long. Only the unscoped one-shot path closes inline (defer_close is False there), because there
        # the teardown is the call's own work and the caller expects the subprocess gone by the time it returns.
        victims.append(session)
    elif close_now:
        session.close()
    if victims:
        _close_detached(victims)


def _retire_session(session: _McpSession) -> None:
    """Close a discarded session, but only once no other borrower is mid-call on it: overlapping
    same-scope calls share one client, and one call's timeout must not kill another's in-flight
    request. The last borrower's _release_session() performs the deferred close."""
    with _mcp_sessions_lock:
        session.defunct = True
        busy = session.in_flight > 0
    if not busy:
        session.close()


def _drop_session(key: tuple, session: _McpSession) -> None:
    with _mcp_sessions_lock:
        if _mcp_sessions.get(key) is session:
            _mcp_sessions.pop(key)
        _discard_key_lock(key)
    _retire_session(session)


def _evict_lru_locked() -> list:
    """Caller holds _mcp_sessions_lock. Evict least-recently-used *idle* sessions until the cache is
    under the cap. Returns the evicted sessions so the caller can close them OUTSIDE the lock. If
    every session is busy the cache may transiently overshoot rather than kill an in-flight call."""
    victims: list = []
    if len(_mcp_sessions) < _MAX_SESSIONS:
        return victims
    account_id = current_account_id()
    # Shared processes are outside the cap, as in _release_session.
    candidates = {
        key: session
        for key, session in _mcp_sessions.items()
        if (key[3] if len(key) > 3 else OWNER_ACCOUNT_ID) == account_id
        and not getattr(session, "shared", False)
    }
    while len(candidates) >= _MAX_SESSIONS:
        idle = [(s.last_used, k) for k, s in candidates.items() if s.in_flight == 0]
        if not idle:
            break
        _, oldest = min(idle, key = lambda item: item[0])
        victims.append(_mcp_sessions.pop(oldest))
        candidates.pop(oldest)
        _discard_key_lock(oldest)
    return victims


def close_mcp_sessions(
    url: Optional[str] = None,
    headers = _ANY_HEADERS,
    *,
    cwd = _ANY_CWD,
    all_accounts: bool = False,
) -> None:
    """Close cached sessions: all of them, one URL/command's, or (with ``headers``, and optionally
    ``cwd``) exactly one server row's configuration, so another row sharing the command but not its
    env or working directory keeps its live processes. ``cwd`` only narrows a call that also names
    ``headers``."""
    global _mcp_close_all_gen
    hk = None if headers is _ANY_HEADERS else _headers_key(headers)
    any_cwd = cwd is _ANY_CWD or hk is None
    account_id = current_account_id()
    with _mcp_sessions_lock:
        keys = [
            k
            for k in _mcp_sessions
            if (url is None or k[0] == url)
            and (hk is None or k[1] == hk)
            and (any_cwd or _session_cwd(k) == (cwd or ""))
            and (all_accounts or (k[3] if len(k) > 3 else OWNER_ACCOUNT_ID) == account_id)
        ]
        sessions = [_mcp_sessions.pop(k) for k in keys]
        for key in keys:
            _discard_key_lock(key)
        if sessions or _mcp_connects_in_flight:
            if url is None:
                if all_accounts:
                    _mcp_close_all_gen += 1
                else:
                    _mcp_account_close_gen[account_id] = (
                        _mcp_account_close_gen.get(account_id, 0) + 1
                    )
            elif hk is None:
                uk = _url_close_key(url)
                _mcp_url_close_gen[uk] = _mcp_url_close_gen.get(uk, 0) + 1
            elif any_cwd:
                cfg = _cfg_close_key(url, headers)
                _mcp_cfg_close_gen[cfg] = _mcp_cfg_close_gen.get(cfg, 0) + 1
            else:
                cfg = _cwd_close_key(url, headers, cwd)
                _mcp_cwd_close_gen[cfg] = _mcp_cwd_close_gen.get(cfg, 0) + 1
    pending, worker = _drain_cleanup_queue()
    _close_all(sessions + pending)
    if worker is not None and worker is not threading.current_thread():
        # Draining the queue does not recall the session the worker had already popped, and this function promises its
        # caller (a server edit, or atexit) that the teardown has happened. The worker stops as soon as the queue is
        # empty, so this waits for that one close and no longer.
        worker.join(_SESSION_CLOSE_TIMEOUT + 5.0)


def _close_all(sessions: list) -> None:
    """Close sessions in parallel.

    Serially, each unresponsive transport can burn _SESSION_CLOSE_TIMEOUT plus the thread join
    before the next one starts. This runs on the request thread when a server is edited or deleted,
    and a popular HTTP server now holds a session per chat rather than one overall, so a serial
    close could stall that route for minutes.

    Fanned out _MAX_CLOSE_THREADS wide rather than one thread per session. The cache is allowed to
    overshoot _MAX_SESSIONS while every session in it is busy, so the list handed here has no fixed
    length, and a shutdown is the worst moment to ask the process for an unbounded number of
    threads.
    """
    if not sessions:
        return
    if len(sessions) == 1:
        sessions[0].close()
        return

    pending = list(sessions)
    pending_lock = threading.Lock()

    def _drain() -> None:
        while True:
            with pending_lock:
                if not pending:
                    return
                session = pending.pop()
            _close_quietly(session)

    # Bare threads, not a ThreadPoolExecutor: this also runs as the atexit handler, and Python shuts the executor
    # machinery down before normal atexit callbacks, so submitting there raises ("can't register atexit after
    # shutdown") and the whole cleanup aborts with stdio subprocesses still up.
    width = min(len(pending), _MAX_CLOSE_THREADS)
    threads = [threading.Thread(target = _drain, name = "mcp-close", daemon = True) for _ in range(width)]
    for thread in threads:
        thread.start()
    # Each worker may take several sessions in turn, so the wait scales with the rounds it has to make rather than
    # with a single close.
    rounds = -(-len(sessions) // width)
    for thread in threads:
        thread.join(rounds * (_SESSION_CLOSE_TIMEOUT + 5.0))


def _close_detached(sessions: list) -> None:
    """Hand sessions nobody is waiting on to the cleanup worker.

    Used for LRU victims and for the deferred close of a retired session: those belong to another
    scope or to a call that has already ended, while the thread holding them is in the middle of
    serving a tool call on its own deadline and an unresponsive transport costs
    _SESSION_CLOSE_TIMEOUT to shut down.

    One worker rather than a thread per session: nobody is waiting on these, so closing them one at
    a time is fine, and a run of new chat scopes against a server that hangs on shutdown then cannot
    spawn threads without bound. close_mcp_sessions stays synchronous and drains this queue, because
    its caller (a server edit, or atexit) does need the teardown to have happened.

    Past _MAX_PENDING_CLOSES the overflow is closed on the caller instead. A queue that keeps
    growing means the server is shutting down slower than chats are opening, and every waiting
    session still holds its own loop thread and connection, so the queue has to be bounded as well
    as the cache. Making the caller wait is the backpressure that stops it: unpleasant, but the same
    thing that happened before any of this was deferred, and only once the deferral has already
    failed to keep up.
    """
    global _mcp_cleanup_worker
    if not sessions:
        return
    with _mcp_cleanup_lock:
        room = max(0, _MAX_PENDING_CLOSES - len(_mcp_cleanup_queue))
        _mcp_cleanup_queue.extend(sessions[:room])
        overflow = sessions[room:]
        if _mcp_cleanup_queue and _mcp_cleanup_worker is None:
            _mcp_cleanup_worker = threading.Thread(
                target = _cleanup_worker, name = "mcp-cleanup", daemon = True
            )
            _mcp_cleanup_worker.start()
    _close_all(overflow)


def _cleanup_worker() -> None:
    global _mcp_cleanup_worker
    while True:
        with _mcp_cleanup_lock:
            if not _mcp_cleanup_queue:
                _mcp_cleanup_worker = None  # _close_detached starts the next one
                return
            session = _mcp_cleanup_queue.pop(0)
        _close_quietly(session)


def _drain_cleanup_queue() -> tuple[list, Optional[threading.Thread]]:
    """Take back whatever the worker has not started on yet, plus the worker itself so the caller
    can wait out the one close already under way."""
    with _mcp_cleanup_lock:
        pending = list(_mcp_cleanup_queue)
        _mcp_cleanup_queue.clear()
        return pending, _mcp_cleanup_worker


def _close_quietly(session) -> None:
    try:
        session.close()
    except Exception:  # noqa: BLE001
        logger.exception("Closing a discarded MCP session failed")


# The cache stopped being stdio-only, but an in-place upgrade can leave a caller holding the old name; it costs two
# lines to keep it working.
close_stdio_sessions = close_mcp_sessions


def _reset_after_fork() -> None:
    """Drop everything the child inherited from the parent's cache. Only the forking thread survives
    a fork, so every session's loop thread is gone while its client still reports connected. A
    child that checked one out would wait on a loop that will never run, and _transport_dead
    cannot see it for HTTP. Nothing here is closed: those objects belong to the parent, which is
    still using them."""
    global _mcp_reaper_started, _mcp_connects_in_flight, _mcp_sessions_lock
    global _mcp_cleanup_lock, _mcp_cleanup_worker
    # Replaced, not just cleared: a lock the fork caught held belongs to a thread that no longer exists here, so the
    # child would block on it forever.
    _mcp_sessions_lock = threading.Lock()
    _mcp_sessions.clear()
    _mcp_key_locks.clear()
    _mcp_connects_in_flight = 0
    _mcp_reaper_started = False
    # The cleanup worker did not survive the fork either, and its queue holds the parent's sessions.
    _mcp_cleanup_lock = threading.Lock()
    _mcp_cleanup_queue.clear()
    _mcp_cleanup_worker = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child = _reset_after_fork)


def _session_idle_ttl(session) -> float:
    ttl = getattr(session, "idle_ttl", None)
    return _SESSION_IDLE_TTL if ttl is None else ttl


def _stdio_suspended() -> bool:
    """Local programs were allowed when these processes started and no longer are (Remote Access turned on, or
    tools switched off). Every call is refused then anyway; a long-lived process must not keep running, holding a
    serial port or an SSH session, behind a gate that is closed."""
    if os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP") != "1":
        return False
    from utils.account_context import OWNER, run_as

    try:
        # As the owner: only the owner can run local programs, and the reaper thread binds no account.
        return not run_as(OWNER, stdio_mcp_enabled)
    except Exception:  # noqa: BLE001
        return False


def _reap_idle_sessions(now: Optional[float] = None) -> None:
    now = time.monotonic() if now is None else now
    suspended = _stdio_suspended()
    with _mcp_sessions_lock:
        expired = [
            key
            for key, session in _mcp_sessions.items()
            if session.in_flight == 0
            and (
                now - session.last_used >= _session_idle_ttl(session)
                or (suspended and is_stdio(session.url))
            )
        ]
        sessions = [_mcp_sessions.pop(key) for key in expired]
        for key in expired:
            _discard_key_lock(key)
    for session in sessions:
        logger.info("Closing idle MCP session: %s", _session_log_id(session.url))
        session.close()


def _session_reaper() -> None:
    while True:
        time.sleep(_SESSION_REAP_INTERVAL)
        try:
            _reap_idle_sessions()
        except Exception as exc:  # noqa: BLE001
            logger.debug("MCP session reaper iteration failed: %s", exc)


# ---------------------------------------------------------------------------------------------------------------------
# Lifecycle: status, stop, re-listing over a live process, shutdown
# ---------------------------------------------------------------------------------------------------------------------

_LIFECYCLE_ERROR_CHARS = 400
# _cwd_close_key(url, headers, cwd) -> (wall time, message): how a configuration last failed to start or died, kept
# until it next starts. Keyed by digest like the close generations, so no command or env value is held here.
_server_errors: dict = {}


def _lifecycle_error_text(url: str, headers: Optional[dict], exc: BaseException) -> str:
    if isinstance(exc, McpStdioServerError):
        text = exc.summary
    elif isinstance(exc, _ConnectTimeout):
        if exc.detail:
            text = exc.detail.splitlines()[0]
        else:
            text = "Timed out while starting" + (
                f" (after {exc.window:g}s)" if exc.window is not None else ""
            )
    elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        text = "Timed out"
    else:
        text = (str(exc).strip() or type(exc).__name__).splitlines()[0]
    # The first line only, masked like the stderr log: an exception can quote the argv or an env value.
    text = redact_log_text(mask_secret_values(text, mcp_secret_values(url, headers)))
    return text[:_LIFECYCLE_ERROR_CHARS]


def _record_server_error(url: str, headers: Optional[dict], cwd: Optional[str], exc) -> None:
    if not is_stdio(url):
        return
    try:
        message = _lifecycle_error_text(url, headers, exc)
    except Exception:  # noqa: BLE001
        message = type(exc).__name__
    _server_errors[_cwd_close_key(url, headers, cwd)] = (time.time(), message)


def _clear_server_error(url: str, headers: Optional[dict], cwd: Optional[str]) -> None:
    _server_errors.pop(_cwd_close_key(url, headers, cwd), None)


def record_server_failure(row: dict, exc: BaseException) -> None:
    """A probe of this saved server failed; its status reports it until a process starts."""
    _record_server_error(row.get("url") or "", parse_server_headers(row), row.get("cwd"), exc)


def clear_server_failure(row: dict) -> None:
    """The owner stopped or restarted the server: an earlier failure no longer describes it."""
    _clear_server_error(row.get("url") or "", parse_server_headers(row), row.get("cwd"))


def _server_sessions(url: str, headers: Optional[dict], cwd: Optional[str]) -> list:
    """This account's live sessions of one configuration, any scope: a shared process, or each chat's own."""
    hk = _headers_key(headers)
    account_id = current_account_id()
    with _mcp_sessions_lock:
        return [
            session
            for key, session in _mcp_sessions.items()
            if key[0] == url
            and key[1] == hk
            and _session_cwd(key) == (cwd or "")
            and (key[3] if len(key) > 3 else OWNER_ACCOUNT_ID) == account_id
            and not session.defunct
            and not session.closed.is_set()
        ]


def _stdio_log_path(url: str) -> Optional[str]:
    from utils.paths.storage_roots import studio_root

    try:
        parts = parse_stdio_command(url)
        if not parts:
            return None
        path = Path(studio_root()) / "logs" / "mcp" / _stdio_log_name(url, parts[0])
        return str(path) if path.is_file() else None
    except (OSError, ValueError):
        return None


def server_status(row: dict) -> dict:
    """What a saved local program is doing now. ``state``: running (a call is in flight), idle (up, waiting; stopped
    after its idle timeout), stopped, or failed (its last start or its process failed; ``last_error`` says how).
    Times are seconds; ``started_at``/``last_error_at`` are epoch seconds."""
    url = row.get("url") or ""
    headers = parse_server_headers(row)
    cwd = row.get("cwd")
    now = time.monotonic()
    status: dict = {
        "state": "stopped",
        "processes": 0,
        "started_at": None,
        "uptime_seconds": None,
        "idle_seconds": None,
        "stops_in_seconds": None,
        "busy_tool": None,
        "busy_seconds": None,
        "last_error": None,
        "last_error_at": None,
        "log_path": _stdio_log_path(url) if is_stdio(url) else None,
    }
    sessions = [s for s in _server_sessions(url, headers, cwd) if s.client is not None]
    alive = []
    for session in sessions:
        if _transport_dead(session):
            # Exited on its own: the next call reconnects, but until then this is what happened.
            recorded = _server_errors.get(_cwd_close_key(url, headers, cwd))
            if recorded is None or recorded[0] < (session.started_at or 0.0):
                _record_server_error(
                    url, headers, cwd, RuntimeError("The server process exited unexpectedly.")
                )
        else:
            alive.append(session)
    error = _server_errors.get(_cwd_close_key(url, headers, cwd))
    if error is not None:
        status["last_error_at"], status["last_error"] = error
    if not alive:
        status["state"] = "failed" if error is not None else "stopped"
        return status
    started = [s for s in alive if s.started_mono is not None]
    if started:
        oldest = min(started, key = lambda s: s.started_mono)
        status["uptime_seconds"] = max(0.0, now - oldest.started_mono)
        status["started_at"] = oldest.started_at
    status["processes"] = len(alive)
    holders = [s.call_holder for s in alive if s.call_holder is not None]
    if holders or any(s.in_flight > 0 for s in alive):
        status["state"] = "running"
        if holders:
            holder = min(holders, key = lambda h: h[2])
            status["busy_tool"] = holder[1]
            status["busy_seconds"] = max(0.0, now - holder[2])
        return status
    status["state"] = "idle"
    latest = max(alive, key = lambda s: s.last_used)
    idle = max(0.0, now - latest.last_used)
    status["idle_seconds"] = idle
    ttl = _session_idle_ttl(latest)
    if math.isfinite(ttl):
        status["stops_in_seconds"] = max(0.0, ttl - idle)
    return status


def retime_server_processes(row: dict) -> None:
    """An edited idle timeout reaches this server's running processes now, not only on their next call."""
    lifecycle = server_lifecycle(row)
    if lifecycle is None:
        return
    for session in _server_sessions(row.get("url") or "", parse_server_headers(row), row.get("cwd")):
        session.idle_ttl = lifecycle.idle_ttl


def stop_server_processes(row: dict) -> int:
    """End every process of this saved server (the shared one, or each chat's), now. The next call that needs it
    starts it again. Returns how many were running."""
    url = row.get("url") or ""
    headers = parse_server_headers(row)
    cwd = row.get("cwd")
    running = len(_server_sessions(url, headers, cwd))
    close_mcp_sessions(url, headers, cwd = cwd)
    return running


def list_session_tools_sync(
    url: str,
    headers: Optional[dict],
    *,
    scope: str,
    timeout: Optional[float],
    cwd: Optional[str] = None,
    lifecycle: Optional[McpLifecycle] = None,
    config_check = None,
    spawn: bool = True,
) -> list[dict]:
    """tools/list over the cached session for ``scope`` -- a shared server's one process, or a chat's own -- starting
    it unless ``spawn`` is False. Discovery for a shared server goes through here, so a stateful program is never
    started a second time just to be asked for its tools, and the process it starts is the one the chat's calls
    then use. Raises like list_tools_async (McpStdioServerError for an explained failed start)."""
    try:
        tools = _call_session_tool(
            url,
            headers,
            "tools/list",
            {},
            timeout,
            None,
            scope,
            config_check,
            False,
            cwd,
            dispatch = lambda client: client.list_tools(),
            lifecycle = lifecycle,
            spawn = spawn,
        )
    except _ConnectTimeout as exc:
        if exc.detail:
            raise McpStdioServerError(exc.detail) from exc
        raise
    return [tool.model_dump(exclude_none = True) for tool in tools]


def session_tool_overlay(
    url: str, headers: Optional[dict], scope: Optional[str], cwd: Optional[str] = None
) -> Optional[tuple[Optional[list[dict]], bool]]:
    """A chat's own process that has announced a tool-list change: (its re-read tools, or None before the first
    re-read; whether a re-read is due). None when the chat has no such process, so the server's cached list
    applies. Per-chat mode only: a shared process's change goes to the server's cache instead."""
    if not scope:
        return None
    with _mcp_sessions_lock:
        session = _mcp_sessions.get(_session_key(url, headers, scope, cwd))
    if session is None or session.defunct or session.closed.is_set() or session.client is None:
        return None
    if not getattr(session, "tools_changed", False):
        return None
    return session.tools, session.tools_stale


def refresh_session_tools_sync(
    url: str,
    headers: Optional[dict],
    *,
    scope: str,
    timeout: Optional[float],
    cwd: Optional[str] = None,
    lifecycle: Optional[McpLifecycle] = None,
    config_check = None,
) -> Optional[list[dict]]:
    """Re-read a chat's own process's tools after it announced a change, over that same process; never starts one.
    The list is kept on the session for session_tool_overlay. None when the process is gone."""
    with _mcp_sessions_lock:
        session = _mcp_sessions.get(_session_key(url, headers, scope, cwd))
    if session is None:
        return None
    epoch = session.tools_epoch
    try:
        tools = list_session_tools_sync(
            url,
            headers,
            scope = scope,
            timeout = timeout,
            cwd = cwd,
            lifecycle = lifecycle,
            config_check = config_check,
            spawn = False,
        )
    except _NoLiveSession:
        return None
    if session.closed.is_set():
        return None
    session.tools = tools
    if session.tools_epoch == epoch:
        session.tools_stale = False
    return tools


def shutdown_mcp_sessions() -> None:
    """Studio is stopping: end every MCP process, every account's, before the interpreter does. The atexit hook
    does the same but runs late and not at all on some exits; on Windows a process that escapes both still dies
    with Studio's kill-on-close job (utils.process_lifetime) and the SDK's per-child job."""
    close_mcp_sessions(all_accounts = True)


async def list_tools_async(
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 5.0,
    use_oauth: bool = False,
    cwd: Optional[str] = None,
    **oauth,
) -> list[dict]:
    """Connect, list the server's tools, disconnect. A local program that fails to start raises
    McpStdioServerError saying why (missing program, exited, no handshake) with the tail of its
    stderr, instead of a bare "Connection closed" or an empty timeout."""
    client = _new_client(url, headers, use_oauth, cwd, **oauth)
    launch = _stdio_launch_of(client)

    async def _fetch() -> list[dict]:
        async with _connected(client) as connected:
            tools = await connected.list_tools()
        return [t.model_dump(exclude_none = True) for t in tools]

    try:
        return await asyncio.wait_for(_fetch(), timeout = timeout)
    except asyncio.TimeoutError as exc:
        if launch is None:
            raise
        raise launch.timeout_error(timeout) from exc


# Discovered-tool cache, keyed by MCP server id. get_enabled_mcp_tools() probes a server only
# on a cache miss, keeping MCP discovery off the chat send's critical path -- tool schemas are
# stable within a session. The /refresh route warms it; a URL/header/OAuth change or a delete
# evicts it. Successful probes are cached indefinitely.
_tool_cache: dict[str | tuple[str, str], list[dict]] = {}

# server_id -> monotonic time before which a failed server must not be
# re-probed (see record_probe_failure). Cleared on a successful probe or
# eviction.
_probe_cooloff_until: dict[str | tuple[str, str], float] = {}

# Coordinate off-loop token-count snapshots with row and schema-cache mutations.
_mcp_server_snapshot_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    WeakKeyDictionary()
)


def mcp_server_snapshot_guard() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    return _mcp_server_snapshot_locks.setdefault(loop, asyncio.Lock())


def serialize_mcp_server_mutation(handler):
    """Run an MCP mutation from validation through row/cache commit as one snapshot."""

    @wraps(handler)
    async def _serialized(*args, **kwargs):
        async with mcp_server_snapshot_guard():
            return await handler(*args, **kwargs)

    return _serialized


# MCP server fields whose change invalidates a server's discovered tools: the endpoint/auth used to probe it (url,
# headers, oauth), where a local program runs (cwd: a server can expose different tools per project folder), whether
# it's used at all (is_enabled), or which process answers (process_mode: a shared process and a chat's own can list
# different tools, and the switch must end the processes of the old mode). A rename or a new idle timeout does not.
# The update route's eviction and get_enabled_mcp_tools' mid-probe guard both key off this so they can't drift.
TOOL_CACHE_INVALIDATING_FIELDS = frozenset(
    {
        "url",
        "headers_json",
        "use_oauth",
        "oauth_client_id",
        "oauth_client_secret",
        "is_enabled",
        "cwd",
        "process_mode",
    }
)


def get_cached_tools(server_id: str) -> Optional[list[dict]]:
    return _tool_cache.get(_account_key(server_id))


# notifications/tools/list_changed from a shared process. The list it replaces stays cached and served (a send that
# cannot wait for the re-list is better off with the old tools than with none) but is marked stale, so the next send
# re-lists it over that same process. The epoch counts notifications, so a re-list that raced one is cached still
# stale rather than passed off as current.
_tool_cache_stale: set = set()
_tool_cache_epoch: dict = {}


def _owner_key(account_id: str, server_id: str):
    return server_id if account_id == OWNER_ACCOUNT_ID else (account_id, server_id)


def _mark_tools_stale(account_id: str, server_id: str) -> None:
    key = _owner_key(account_id, server_id)
    _tool_cache_epoch[key] = _tool_cache_epoch.get(key, 0) + 1
    if key in _tool_cache:
        _tool_cache_stale.add(key)


def mark_tools_stale(server_id: str) -> None:
    _mark_tools_stale(current_account_id(), server_id)


def tools_cache_stale(server_id: str) -> bool:
    return _account_key(server_id) in _tool_cache_stale


def tools_cache_epoch(server_id: str) -> int:
    return _tool_cache_epoch.get(_account_key(server_id), 0)


def cache_tools(server_id: str, tools: list[dict], epoch: Optional[int] = None) -> None:
    """``epoch``: tools_cache_epoch() read before the probe began; a list_changed since leaves the entry stale."""
    key = _account_key(server_id)
    _tool_cache[key] = tools
    _probe_cooloff_until.pop(key, None)
    if epoch is not None and epoch != _tool_cache_epoch.get(key, 0):
        _tool_cache_stale.add(key)
    else:
        _tool_cache_stale.discard(key)


def record_probe_failure(server_id: str, use_oauth: bool = False) -> None:
    cooloff = OAUTH_FAILED_PROBE_COOLOFF_SECONDS if use_oauth else FAILED_PROBE_COOLOFF_SECONDS
    _probe_cooloff_until[_account_key(server_id)] = time.monotonic() + cooloff


def in_failure_cooloff(server_id: str) -> bool:
    return _probe_cooloff_until.get(_account_key(server_id), 0.0) > time.monotonic()


def invalidate_tool_cache(server_id: Optional[str] = None) -> None:
    """Evict one server's cached tools, or every entry when server_id is None."""
    if server_id is None:
        account_id = current_account_id()
        for cache in (_tool_cache, _probe_cooloff_until):
            for key in list(cache):
                owner = key[0] if isinstance(key, tuple) else OWNER_ACCOUNT_ID
                if owner == account_id:
                    cache.pop(key, None)
        for key in list(_tool_cache_stale):
            if (key[0] if isinstance(key, tuple) else OWNER_ACCOUNT_ID) == account_id:
                _tool_cache_stale.discard(key)
    else:
        _tool_cache.pop(_account_key(server_id), None)
        _probe_cooloff_until.pop(_account_key(server_id), None)
        _tool_cache_stale.discard(_account_key(server_id))


UI_RESOURCE_SCHEME = "ui://"
MAX_UI_RESOURCE_CHARS = 5_000_000


def _ui_meta_field(tool: Any, field: str):
    """`ui.<field>`, else the flat `ui/<field>`, from the SDK's `meta` or the wire `_meta`."""
    metas = [tool.get(k) for k in ("meta", "_meta")] if isinstance(tool, dict) else []
    metas = [m for m in metas if isinstance(m, dict)]
    for ui in (m.get("ui") for m in metas):
        if isinstance(ui, dict) and ui.get(field) is not None:
            return ui[field]
    return next((m[f"ui/{field}"] for m in metas if m.get(f"ui/{field}") is not None), None)


def tool_ui_resource_uri(tool: Any) -> Optional[str]:
    """Only ui:// is honoured: the host fetches this URI."""
    uri = _ui_meta_field(tool, "resourceUri")
    uri = uri.strip() if isinstance(uri, str) else ""
    return (
        uri if uri.startswith(UI_RESOURCE_SCHEME) and len(uri) > len(UI_RESOURCE_SCHEME) else None
    )


def tool_visible_to(tool: Any, audience: str) -> bool:
    """`audience` is "model" or "app"; an undeclared visibility means both."""
    visibility = _ui_meta_field(tool, "visibility")
    return audience in visibility if isinstance(visibility, (list, tuple)) else True


MCP_IMAGES_SENTINEL = mcp_images.SENTINEL
MAX_IMAGE_PAYLOAD_CHARS = 12_000_000

# Emitted BEFORE the image envelope, whose parse reads to end of string.
MCP_UI_SENTINEL = "__MCP_UI__:"
MAX_UI_STRUCTURED_CHARS = 1_000_000


def _json_within(value: Any, limit: int) -> Optional[str]:
    try:
        line = json.dumps(value)
    except (TypeError, ValueError, RecursionError):
        return None
    return line if len(line) <= limit else None


def _ui_envelope(result: Any, ui_resource_uri: str, seed: list) -> str:
    payload: dict = {"resourceUri": ui_resource_uri}
    meta = getattr(result, "meta", None)
    if isinstance(meta, dict) and meta:
        payload["_meta"] = meta
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        payload["structuredContent"] = structured
    if seed:
        payload["content"] = [
            _seeded_image_block(b, m) if m else _content_block_json(b) for b, m in seed
        ]
    line = _json_within(payload, MAX_UI_STRUCTURED_CHARS)
    if line is None:
        # Shed structuredContent, then content, then _meta: the view still gets its template.
        reduced = {"resourceUri": ui_resource_uri, "structuredContentOmitted": True}
        keys = [k for k in ("content", "_meta") if k in payload]
        for kept in (keys, keys[:1], keys[1:]):
            line = _json_within(
                {**reduced, **{k: payload[k] for k in kept}}, MAX_UI_STRUCTURED_CHARS
            )
            if line:
                break
        else:
            line = json.dumps(reduced)
    return "\n" + MCP_UI_SENTINEL + line


def _is_ui_envelope_line(line: str) -> bool:
    if not line.startswith(MCP_UI_SENTINEL):
        return False
    try:
        payload = json.loads(line[len(MCP_UI_SENTINEL) :])
    except (ValueError, RecursionError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("resourceUri"), str)


def _drop_forged_ui_sentinels(body: str) -> str:
    """Readers take the last marker, so a tool-written one could summon a widget with forged seed text."""
    if MCP_UI_SENTINEL not in body:
        return body
    return "\n".join(line for line in body.split("\n") if not _is_ui_envelope_line(line))


def _block_text(block: Any) -> Optional[str]:
    text = getattr(block, "text", None)
    if text:
        return str(text)
    resource = getattr(block, "resource", None)
    if resource is not None:
        text = getattr(resource, "text", None)
        return str(text) if text else None
    return None


def _block_link(block: Any) -> Optional[str]:
    # keep host-generated link text from suppressing structured_content
    uri = getattr(block, "uri", None)
    if uri and getattr(block, "type", None) == "resource_link":
        name = getattr(block, "name", None)
        return f"[resource: {name} <{uri}>]" if name else f"[resource: <{uri}>]"
    return None


# fastmcp File(data=..., format=...) labels payloads as application/<format>
_IMAGE_SUBTYPES = {
    "apng": "image/apng",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "avif": "image/avif",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "ico": "image/vnd.microsoft.icon",
    "svg": "image/svg+xml",
    "svg+xml": "image/svg+xml",
}


# What tool-fallback.tsx may interpolate into data:<type>;base64,... : an RFC 9110 8.3.1 token subtype, minus "*",
# which names a range and never a payload.
_MEDIA_TYPE = re.compile(r"^image/[a-z0-9][a-z0-9!#$%&'^_`|~.+-]*$")


def _uri_mime(uri: Any) -> Optional[str]:
    """Guess a media type from the part of a URI that names the resource. mimetypes only stopped
    reading the query and fragment in 3.11.9 / 3.12.3 / 3.13 (CPython gh-117217), and on older
    supported interpreters 'gen.png?download=1' guessed nothing while 'download?name=gen.png'
    guessed image/png. Dropping both keeps every interpreter in agreement. The scheme stays so a
    data: URI still resolves; a bare host goes, since a host name is not a file name."""
    split = urlsplit(str(uri))
    cleaned = urlunsplit((split.scheme, split.netloc if split.path else "", split.path, "", ""))
    return mimetypes.guess_type(cleaned, strict = False)[0]


def _image_mime(mime: Any) -> Optional[str]:
    if not isinstance(mime, str):
        return None
    # media type names are case-insensitive; data urls need only the essence
    essence = mime.partition(";")[0].strip().lower()
    if essence.startswith("image/"):
        resolved = essence
    else:
        subtype = essence[len("application/") :] if essence.startswith("application/") else ""
        resolved = _IMAGE_SUBTYPES.get(subtype) or _uri_mime(f"file:///image.{subtype}")
    # one gate for every branch. Lowercased again because a registry answer carries the host's spelling: Windows
    # returns image/JXL for .jxl, Linux and macOS image/jxl.
    resolved = resolved.lower() if resolved else ""
    return resolved if _MEDIA_TYPE.match(resolved) else None


def _resource_mime(obj: Any) -> Any:
    # mcp 2.x renames mimeType to mime_type, keeping camelCase only as an alias
    mime = getattr(obj, "mimeType", None)
    return mime if mime is not None else getattr(obj, "mime_type", None)


def _block_image(block: Any) -> Optional[tuple[str, str]]:
    # embedded resources keep binary data on resource.blob
    data = getattr(block, "data", None)
    mime = _resource_mime(block)
    if not data:
        resource = getattr(block, "resource", None)
        if resource is None:
            return None
        data = getattr(resource, "blob", None)
        mime = _resource_mime(resource)
        if not mime:
            uri = getattr(resource, "uri", None)
            mime = _uri_mime(uri) if uri else None
    mime = _image_mime(mime)
    if data and mime:
        return str(data), mime
    return None


def _block_attachment(block: Any) -> Optional[tuple[str, str]]:
    # presence, not truthiness: a zero-byte file arrives as ""
    data = getattr(block, "data", None)
    if data is not None:
        kind = getattr(block, "type", "binary")
        mime = _resource_mime(block)
        uri = None
    else:
        resource = getattr(block, "resource", None)
        data = getattr(resource, "blob", None) if resource is not None else None
        if data is None:
            return None
        kind = "file"
        mime = _resource_mime(resource)
        uri = getattr(resource, "uri", None)
    label = f"{kind} attachment"
    if mime:
        label += f" ({mime})"
    if uri and not str(uri).lower().startswith("data:"):
        label += f" <{uri}>"
    return f"{label} not shown to the model", str(data)


_MIRRORED = object()
# fields MCP defines on image/audio and embedded resource blocks; anything else is the tool's own
_MEDIA_FIELDS = frozenset({"type", "data", "mimeType", "mime_type", "annotations", "_meta"})
_RESOURCE_BLOCK_FIELDS = frozenset({"type", "resource", "annotations", "_meta"})
_RESOURCE_FIELDS = frozenset({"uri", "blob", "mimeType", "mime_type", "_meta"})


def _is_payload(value: Any, payloads: set[str]) -> bool:
    return isinstance(value, str) and value in payloads


def _mirrored_extras(value: dict, payloads: set[str]) -> Optional[dict]:
    # the tool's own fields on a content block copied from result.content; None if it is not one
    kind = value.get("type")
    if kind in ("image", "audio") and _is_payload(value.get("data"), payloads):
        return {k: v for k, v in value.items() if k not in _MEDIA_FIELDS}
    resource = value.get("resource")
    if (
        kind == "resource"
        and isinstance(resource, dict)
        and _is_payload(resource.get("blob"), payloads)
    ):
        extras = {k: v for k, v in value.items() if k not in _RESOURCE_BLOCK_FIELDS}
        inner = {k: v for k, v in resource.items() if k not in _RESOURCE_FIELDS}
        if inner:
            extras["resource"] = inner
        return extras
    return None


def _strip_payloads(value: Any, payloads: set[str]) -> Any:
    # drop mirrored payloads and the MCP fields of the blocks carrying them, then containers left empty
    if isinstance(value, str):
        return _MIRRORED if value in payloads else value
    if isinstance(value, dict):
        extras = _mirrored_extras(value, payloads)
        if extras is not None:
            if not extras:
                return _MIRRORED
            value = extras
        kept = {}
        for key, item in value.items():
            item = _strip_payloads(item, payloads)
            if item is not _MIRRORED:
                kept[key] = item
    elif isinstance(value, (list, tuple)):
        kept = [
            s for s in (_strip_payloads(item, payloads) for item in value) if s is not _MIRRORED
        ]
    else:
        return value
    return _MIRRORED if value and not kept else kept


# The frontend refills bytes positionally: `data` on an image block, `resource.blob` on an embedded one.
def _seeded_image_block(block: Any, mime: str) -> dict:
    out = {k: v for k, v in _content_block_json(block).items() if k != "data"}
    if isinstance(out.get("resource"), dict):
        out["resource"] = {k: v for k, v in out["resource"].items() if k != "blob"}
        return out
    return {**out, "type": "image", "mimeType": mime}


def _flatten_result(result: Any, ui_resource_uri: Optional[str] = None) -> str:
    parts = []
    images = []
    seed = []
    unshown = []
    payloads = set()
    omitted = 0
    has_text = False
    budget = MAX_IMAGE_PAYLOAD_CHARS
    for block in getattr(result, "content", None) or []:
        text = _block_text(block)
        if text:
            parts.append(text)
            has_text = True
            seed.append((block, None))
            continue
        link = _block_link(block)
        if link:
            parts.append(link)
            seed.append((block, None))
            continue
        image = _block_image(block)
        if image is not None:
            data, mime = image
            payloads.add(data)
            if len(data) > budget:
                omitted += 1
                continue
            budget -= len(data)
            images.append({"data": data, "mimeType": mime})
            seed.append((block, mime))
            continue
        seed.append((block, None))
        attachment = _block_attachment(block)
        if attachment is not None:
            note, data = attachment
            unshown.append(note)
            if data:
                payloads.add(data)
    body = "\n".join(parts)
    # the filesystem server mirrors binary blocks in structured_content; keep everything else
    structured = None if has_text else getattr(result, "structured_content", None)
    if structured is not None and payloads:
        structured = _strip_payloads(structured, payloads)
    if structured is not None and structured is not _MIRRORED:
        body = f"{structured}\n{body}" if body else str(structured)
    if images or omitted or unshown:
        notes = []
        if images:
            n = len(images)
            notes.append(f"{n} image{'s' if n > 1 else ''} returned")
        if omitted:
            notes.append(f"{omitted} image{'s' if omitted > 1 else ''} omitted (too large)")
        notes.extend(unshown)
        note = f"[{'; '.join(notes)}]"
        body = f"{body}\n{note}" if body else note

    if getattr(result, "is_error", False):
        # "Error: " prefix triggers tool_call_parser's TOOL_ERROR_PREFIXES nudge.
        body = f"Error: {body}" if body else "Error: tool returned no content"
    body = _drop_forged_ui_sentinels(body)
    if ui_resource_uri and not getattr(result, "is_error", False):
        body += _ui_envelope(result, ui_resource_uri, seed)
    if images:
        body += "\n" + MCP_IMAGES_SENTINEL + json.dumps(images)
    return body


def _unwind_budget(unwind_timeout: float, timeout: Optional[float], elapsed: float) -> float:
    """How long a cancelled call may take to unwind: whatever is left of the caller's window, so the
    wait is never charged on top of an expired deadline."""
    if not unwind_timeout:
        return 0.0
    if timeout is None:
        return unwind_timeout
    return min(unwind_timeout, max(0.0, timeout - elapsed))


async def _race_tool_call(
    call_coro,
    timeout: Optional[float],
    cancel_event,
    unwind_timeout: float = 0.0,
) -> Any:
    """Await ``call_coro`` under ``timeout``, polling ``cancel_event`` so a /cancel POST interrupts
    even mid-network-read. ``unwind_timeout`` waits up to that long for a cancelled call to
    finish unwinding; only callers that hand the client back to a cache need it (one-shot clients
    are discarded anyway)."""

    async def _watch_cancel() -> None:
        while cancel_event is not None and not cancel_event.is_set():
            await asyncio.sleep(0.05)

    if cancel_event is not None and cancel_event.is_set():
        call_coro.close()
        raise _MCPCancelled
    started = time.monotonic()
    call_task = asyncio.create_task(call_coro)
    if cancel_event is None:
        return await asyncio.wait_for(call_task, timeout = timeout)
    watch_task = asyncio.create_task(_watch_cancel())
    try:
        done, pending = await asyncio.wait(
            {call_task, watch_task},
            timeout = timeout,
            return_when = asyncio.FIRST_COMPLETED,
        )
    finally:
        for t in (call_task, watch_task):
            if not t.done():
                t.cancel()
        # Let a cancelled call unwind before its session is reused, out of the caller's remaining budget. Outlasting
        # it just leaves the session dirty.
        left = _unwind_budget(unwind_timeout, timeout, time.monotonic() - started)
        if left:
            await asyncio.wait({call_task, watch_task}, timeout = left)
    if not done:
        raise asyncio.TimeoutError
    if call_task in done:
        return call_task.result()
    raise _MCPCancelled


def _call_session_tool(
    url: str,
    headers: Optional[dict],
    name: str,
    args: dict,
    timeout,
    cancel_event,
    scope: Optional[str],
    config_check,
    use_oauth: bool = False,
    cwd: Optional[str] = None,
    dispatch = None,
    lifecycle: Optional[McpLifecycle] = None,
    caller: Optional[str] = None,
    spawn: bool = True,
) -> Any:
    """``caller`` names the chat a call on a shared process comes from, so a call queued behind another chat's
    reports McpServerBusy after _SHARED_BUSY_GRACE instead of waiting out its whole budget in silence. Calls with
    no caller (discovery, a restart) wait, and only say busy if their budget runs out behind a chat."""
    if cancel_event is not None and cancel_event.is_set():
        raise _MCPCancelled
    # One deadline covers the key-lock wait, connect, call-lock wait, and the call itself, matching the one-shot path
    # where the caller's timeout wrapped connect plus call in a single window.
    deadline = None if timeout is None else time.monotonic() + timeout

    def _remaining() -> Optional[float]:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    # Callers without an Unsloth session id must retain the former one-shot behavior: no browser/cookie/tool state can
    # leak into another request. Use an ephemeral key (and close it below) rather than the shared empty scope that the
    # persistent-session cache used previously.
    def _config_ok() -> bool:
        if config_check is None:
            return True
        try:
            return bool(config_check())
        except Exception:  # noqa: BLE001
            return False

    ephemeral = not scope
    if ephemeral:
        scope = f"request-{uuid.uuid4().hex}"
    key = _session_key(url, headers, scope, cwd)
    # attempt 0 may find the cached session stale/dead *before* dispatch and reconnect once (safe); attempt 1 is a
    # freshly connected session.
    for attempt in (0, 1):
        session, idle_for = (
            _get_session(url, headers, scope, deadline, cancel_event, config_check, use_oauth, cwd)
            if lifecycle is None and spawn
            else _get_session(
                url,
                headers,
                scope,
                deadline,
                cancel_event,
                config_check,
                use_oauth,
                cwd,
                lifecycle = lifecycle,
                spawn = spawn,
            )
        )
        locked = False
        try:
            # Serialize calls per session where the transport demands it: overlapping same-scope calls must not
            # interleave operations on one stateful stdio server (browser, REPL). HTTP multiplexes by request id, so
            # its calls run in parallel as they did one-shot.
            if session.serialize_calls:
                queued_at = time.monotonic()
                while not session.call_lock.acquire(timeout = 0.05):
                    if cancel_event is not None and cancel_event.is_set():
                        raise _MCPCancelled
                    holder = session.call_holder
                    # Behind another chat's call on a shared process: say so rather than hang.
                    other_chat = (
                        getattr(session, "shared", False)
                        and holder is not None
                        and holder[0] is not None
                        and holder[0] != caller
                    )
                    rem = _remaining()
                    if rem is not None and rem <= 0:
                        if other_chat:
                            raise McpServerBusy(holder[1], time.monotonic() - holder[2])
                        raise asyncio.TimeoutError
                    if (
                        other_chat
                        and caller is not None
                        and time.monotonic() - queued_at >= _SHARED_BUSY_GRACE
                    ):
                        raise McpServerBusy(holder[1], time.monotonic() - holder[2])
                locked = True
                session.call_holder = (caller, name, time.monotonic())
        except BaseException:
            # Never touched the transport: keep the session for its borrower.
            _release_session(session)
            if ephemeral:
                _drop_session(key, session)
            raise
        discard_session = ephemeral
        retry = False
        try:
            # We may have waited on the call lock while another caller's timeout retired this session, a server
            # update/delete invalidated it, or a reused subprocess died. Re-check all three before dispatch so we
            # never run on a retired/dead client or a stale config.
            if session.closed.is_set():
                # Intentional close (server update/delete/shutdown): don't retry on stale config.
                discard_session = True
                raise RuntimeError("MCP server was updated or removed during the call")
            elif session.defunct:
                # A concurrent same-scope caller's timeout retired this session; move to a fresh one instead of
                # reusing the retired client.
                discard_session = True
                if attempt == 0:
                    retry = True
                else:
                    raise RuntimeError("MCP server session was retired during the call")
            elif not _config_ok():
                discard_session = True
                raise RuntimeError("MCP server was updated or removed during the call")
            elif _transport_dead(session):
                # Dead BEFORE dispatch: no request was sent, so reconnect + retry.
                discard_session = True
                if attempt == 0:
                    retry = True
                else:
                    raise RuntimeError("MCP server connection is not available")
            elif (
                session.dirty or _needs_idle_recheck(session, idle_for, _remaining())
            ) and not _session_responsive(
                session, _remaining(), cancel_event, timeout_is_fatal = session.dirty
            ):
                # Dirty: still stuck on the abandoned call. Idle HTTP: the server may have expired the session while
                # nothing was using it, and no HTTP transport lets us ask. Either way it failed to answer, so
                # reconnect BEFORE dispatch rather than losing the user's call.
                discard_session = True
                if attempt == 0:
                    retry = True
                else:
                    raise RuntimeError("MCP server is not responding")
            else:
                rem = _remaining()
                # raise_on_error=False for the same reason as the one-shot path.
                coro = _race_tool_call(
                    dispatch(session.client)
                    if dispatch is not None
                    else session.client.call_tool(name, args, raise_on_error = False),
                    rem,
                    cancel_event,
                    # Only a cached session is worth waiting on.
                    0.0 if ephemeral else _CANCEL_UNWIND_TIMEOUT,
                )
                out = session.run(coro, rem)
                # A completed round trip proves the transport better than any probe could, so it resets the idle clock
                # the recheck reads.
                session.proved_at = time.monotonic()
                return out
        except (_MCPCancelled, asyncio.TimeoutError):
            # Keep the session so a Stop doesn't destroy the server's state; the SDK drops the abandoned reply, and
            # reuse is gated on a live probe.
            session.dirty = True
            raise
        except _SessionWedged:
            discard_session = True
            raise asyncio.TimeoutError
        except _SessionClosed:
            # close_mcp_sessions() shut this session mid-call (server update/delete/shutdown); don't retry on the
            # stale config.
            discard_session = True
            raise RuntimeError("MCP server was updated or removed during the call")
        except Exception as exc:
            if session.closed.is_set():
                # An intentional close (server update/delete) can surface as a plain transport error or AttributeError
                # instead of _SessionClosed; don't mistake it for a crash.
                discard_session = True
                raise RuntimeError("MCP server was updated or removed during the call")
            # A ToolError or a JSON-RPC error response means the server answered, so the transport is alive: keep the
            # session and its state. Anything else is transport-level (dead subprocess, broken pipe, dropped HTTP
            # stream): evict so it can't poison the scope, but DO NOT replay (the tool may already have run); the next
            # call opens a fresh session.
            if _is_protocol_error(exc):
                # The server replied, so the transport is fine and the session's state is worth keeping. Probe it
                # before the next call anyway, in case the error was the server telling us the session is no longer
                # one it recognises.
                session.dirty = True
            elif not _is_tool_error(exc):
                discard_session = True
                # The process died or its stream broke: what the server's status shows until it starts again.
                _record_server_error(url, headers, cwd, exc)
            raise
        finally:
            # Remove from the cache and mark defunct BEFORE giving up the borrow, so no other caller can check this
            # session out after it failed. The order matters more now that HTTP callers do not queue on call_lock:
            # released first, a concurrent same-scope call could check out the broken transport while it was still
            # cached, and _release_session can sit closing LRU victims for seconds first. in_flight is still held
            # here, so the close defers to the release below.
            if discard_session:
                _drop_session(key, session)
            _release_session(session, defer_close = not ephemeral)
            if locked:
                session.call_holder = None
                session.call_lock.release()
        if not retry:
            break
    raise RuntimeError("unreachable")


def call_tool_sync(
    url: str,
    headers: Optional[dict],
    name: str,
    args: dict,
    timeout: Optional[float] = 300.0,
    use_oauth: bool = False,
    cancel_event = None,
    scope: Optional[str] = None,
    config_check = None,
    cwd: Optional[str] = None,
    ui_resource_uri: Optional[str] = None,
    lifecycle: Optional[McpLifecycle] = None,
    caller: Optional[str] = None,
    **oauth,
) -> str:
    """Call one MCP tool and return its flattened text/image result. Never raises: every failure comes
    back as an "Error: ..." string for the model.

    Which transport path runs depends on ``scope`` (an opaque per-chat key) and ``use_oauth``. stdio
    always goes through the session machinery, and without a scope it still gets a private ephemeral
    session that is closed afterwards, so no browser or cookie state leaks between requests. HTTP
    reuses a cached session only when it has a scope AND OAuth is off, so a server that keeps state
    between calls keeps it for the whole chat. OAuth HTTP, and HTTP without a scope, connect once
    and disconnect, exactly as before sessions were shared: refreshing a token on a long-lived
    shared connection is not something this code can do safely yet.

    ``timeout`` is one budget covering connect and call together. ``cancel_event`` aborts an
    in-flight call. ``config_check`` re-reads the server row so a call that raced an edit or delete
    cannot dispatch on the stale configuration. ``cwd`` is a local program's working directory
    (None: the backend's own). ``ui_resource_uri`` appends the frontend-only __MCP_UI__ envelope.
    ``lifecycle`` (server_lifecycle of the row) binds the process to its saved server, and ``caller`` is the
    chat's own scope when ``scope`` is a shared process's (see _call_session_tool).
    """

    async def _one_shot() -> Any:
        async with _client(url, headers, use_oauth, **oauth) as client:
            # Connecting (OAuth included) can outlast an edit or delete of the server row.
            if config_check is not None and not config_check():
                raise RuntimeError("MCP server was updated or removed while connecting")
            # raise_on_error=False lets an is_error result (which may still carry image content) reach _flatten_result
            # instead of FastMCP raising ToolError and dropping the images. Transport failures still raise (handled
            # below).
            return await client.call_tool(name, args, raise_on_error = False)

    try:
        if is_stdio(url) or (scope and not use_oauth):
            result = (
                _call_session_tool(
                    url, headers, name, args, timeout, cancel_event, scope, config_check, use_oauth, cwd
                )
                if lifecycle is None and caller is None
                else _call_session_tool(
                    url,
                    headers,
                    name,
                    args,
                    timeout,
                    cancel_event,
                    scope,
                    config_check,
                    use_oauth,
                    cwd,
                    lifecycle = lifecycle,
                    caller = caller,
                )
            )
        else:
            result = asyncio.run(_race_tool_call(_one_shot(), timeout, cancel_event))
    except _MCPCancelled:
        return f"Error: MCP tool '{name}' cancelled"
    except McpServerBusy as exc:
        return f"Error: MCP tool '{name}' did not run: {exc}"
    except _ConnectTimeout as exc:
        if exc.detail:
            return f"Error: MCP tool '{name}' could not start its server: {exc.detail}"
        # Report the window that actually expired: for stdio that is the cold-start cap, not the (larger) caller
        # timeout.
        suffix = f" after {round(exc.window, 1):g}s" if exc.window is not None else ""
        return f"Error: MCP tool '{name}' timed out connecting{suffix}"
    except asyncio.TimeoutError:
        suffix = f" after {timeout:g}s" if timeout is not None else ""
        return f"Error: MCP tool '{name}' timed out{suffix}"
    except McpStdioServerError as exc:
        # An explained start-up failure, not a bug: one line, no traceback. The quoted output stays out of the
        # backend log; the server's own log file has it.
        logger.warning("MCP tool %s: %s: %s", name, _session_log_id(url), exc.summary)
        return f"Error: MCP tool '{name}' could not start its server: {exc}"
    except Exception as exc:
        logger.exception("MCP call_tool failed for %s: %s", name, exc)
        return f"Error: MCP tool '{name}' failed: {exc}"

    return _flatten_result(result, ui_resource_uri)


MAX_UI_TOOL_RESULT_CHARS = 4_000_000


def _content_block_json(block: Any) -> dict:
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            # by_alias, or the SDK's `meta` reaches the widget instead of `_meta`.
            return dump(mode = "json", exclude_none = True, by_alias = True)
        except Exception:  # noqa: BLE001
            pass
    out = {"type": getattr(block, "type", "text")}
    for field in ("text", "data", "mimeType", "uri", "name"):
        value = getattr(block, field, None)
        if value is not None:
            out[field] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return out


def _structured_result(result: Any) -> dict:
    out: dict = {
        "content": [_content_block_json(b) for b in getattr(result, "content", None) or []],
        "is_error": bool(getattr(result, "is_error", False)),
    }
    if getattr(result, "structured_content", None) is not None:
        out["structured_content"] = result.structured_content
    meta = getattr(result, "meta", None)
    if isinstance(meta, dict) and meta:
        out["meta"] = meta
    if _json_within(out, MAX_UI_TOOL_RESULT_CHARS) is None:
        raise ValueError(
            f"tool result is not JSON-serialisable or over {MAX_UI_TOOL_RESULT_CHARS} chars"
        )
    return out


def _ui_request_sync(
    url,
    headers,
    label,
    dispatch,
    *,
    timeout,
    use_oauth = False,
    cancel_event = None,
    scope = None,
    config_check = None,
    cwd = None,
    lifecycle = None,
    caller = None,
    **oauth,
) -> Any:
    """``dispatch(client)`` on the transport call_tool_sync would pick for this scope. ``cwd`` keys a local
    program's session as call_tool_sync does, so a widget reaches the same process the chat's tool calls do.
    A shared process busy with another chat raises McpServerBusy."""

    async def _one_shot() -> Any:
        # As the session branch: an edit during discovery must not reach the old endpoint.
        if config_check is not None and not config_check():
            raise RuntimeError("MCP server was updated or removed during the call")
        async with _client(url, headers, use_oauth, **oauth) as client:
            return await dispatch(client)

    session_args = (
        url,
        headers,
        label,
        {},
        timeout,
        cancel_event,
        scope,
        config_check,
        use_oauth,
        cwd,
    )
    try:
        if is_stdio(url) or (scope and not use_oauth):
            if lifecycle is None and caller is None:
                return _call_session_tool(*session_args, dispatch = dispatch)
            return _call_session_tool(
                *session_args, dispatch = dispatch, lifecycle = lifecycle, caller = caller
            )
        return asyncio.run(_race_tool_call(_one_shot(), timeout, cancel_event))
    except _MCPCancelled as exc:
        raise TimeoutError(f"{label} was cancelled") from exc


def call_tool_structured_sync(
    url: str, headers: Optional[dict], name: str, args: dict, **kw
) -> dict:
    dispatch = lambda client: client.call_tool(name, args, raise_on_error = False)  # noqa: E731
    return _structured_result(_ui_request_sync(url, headers, f"MCP tool '{name}'", dispatch, **kw))


def read_resource_sync(url: str, headers: Optional[dict], uri: str, **kw) -> dict:
    dispatch = lambda client: client.read_resource(uri)  # noqa: E731
    return _resource_contents(
        _ui_request_sync(url, headers, f"MCP resource '{uri}'", dispatch, **kw), uri
    )


def _resource_contents(blocks: Any, uri: str) -> dict:
    """Keyed as McpUiResourceResponse. Of several contents the one matching ``uri`` wins, else the first."""
    import base64

    items = list(blocks or [])
    if not items:
        raise ValueError("resource is empty")
    chosen = next((b for b in items if str(getattr(b, "uri", "")) == uri), items[0])
    text, blob = getattr(chosen, "text", None), None
    if text is None:
        if getattr(chosen, "blob", None) is None:
            raise ValueError("resource carries neither text nor blob content")
        blob = str(chosen.blob)
        try:
            raw = base64.b64decode(blob, validate = True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"resource blob is not base64: {exc}") from exc
        try:
            # Decoded for the host to render a template; the widget still gets the server's blob.
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
    text = str(text)
    # Every block for a widget's own multi-block read; a single block is the fields above.
    contents = []
    for item in items if len(items) > 1 else ():
        entry = {"uri": str(getattr(item, "uri", "") or uri)}
        if _resource_mime(item):
            entry["mimeType"] = str(_resource_mime(item))
        if getattr(item, "text", None) is not None:
            entry["text"] = str(item.text)
        elif getattr(item, "blob", None) is not None:
            entry["blob"] = str(item.blob)
        else:
            continue
        contents.append(entry)
    size = (
        len(text)
        + len(blob or "")
        + sum(len(c.get("text") or c.get("blob") or "") for c in contents)
    )
    if size > MAX_UI_RESOURCE_CHARS:
        raise ValueError(f"resource is {size} chars, over the {MAX_UI_RESOURCE_CHARS} limit")
    # _meta.ui on the contents, not the tool: the template's CSP declaration.
    metas = (getattr(chosen, "meta", None), getattr(chosen, "_meta", None))
    ui = next((m["ui"] for m in metas if isinstance(m, dict) and isinstance(m.get("ui"), dict)), {})
    mime = str(_resource_mime(chosen) or "")
    return {
        "uri": uri,
        "mime_type": mime,
        "text": text,
        "blob": blob,
        "ui": ui,
        "contents": contents,
    }


class _MCPCancelled(Exception):
    """Internal sentinel raised when cancel_event fires before the tool returns."""
