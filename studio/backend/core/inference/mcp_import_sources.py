# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Find the MCP servers other apps on this computer already run (Claude Desktop, Claude Code,
Cursor, VS Code, Windsurf) so the dialog can import them in one click.

Read here, on the server, so an env block full of API keys never passes through the browser: the
dialog sees each server's name, transport, command or URL with credentials masked, and the NAMES
of its env vars or headers. The import route re-reads the file by source id, so the browser never
names a path either. A missing file is skipped; an unreadable one is reported, never raised.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from core.inference.mcp_client import (
    is_stdio,
    join_stdio_command,
    mask_secret_values,
    mcp_secret_values,
    parse_stdio_command,
)
from core.inference.mcp_config_import import ParsedMcpEntry, parse_mcp_entry
from utils.log_redaction import REDACTED, redact_log_text

# ~/.claude.json carries Claude Code's caches and history besides its servers and runs to megabytes;
# anything this large is not a config file.
_MAX_CONFIG_BYTES = 32 * 1024 * 1024
_MAX_SERVERS_PER_SOURCE = 200
MASK = "***"

# ${env:NAME} (VS Code, Cursor), ${NAME} and ${NAME:-default} (Claude Code), ${userHome} (VS Code).
_REFERENCE = re.compile(r"\$\{([^}]*)\}")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# VS Code's own variables. They name editor state, not environment variables, so they never resolve here.
_EDITOR_VARIABLES = frozenset(
    {
        "workspaceFolder",
        "workspaceFolderBasename",
        "workspaceRoot",
        "file",
        "fileBasename",
        "fileBasenameNoExtension",
        "fileDirname",
        "fileExtname",
        "fileWorkspaceFolder",
        "relativeFile",
        "relativeFileDirname",
        "cwd",
        "execPath",
        "lineNumber",
        "selectedText",
        "defaultBuildTask",
    }
)

# Display-only masking, on top of mcp_secret_values: a bare positional argument or a URL path segment
# can be a key too (``server.js sk-...``, ``https://host/s/<token>/sse``), and nothing names it.
_TOKEN_SHAPE = re.compile(r"^[A-Za-z0-9_\-+=]{32,}$")
_KNOWN_KEY_PREFIX = re.compile(
    r"^(?:sk-|sk_|pk_|rk_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|xox[abprs]-|xapp-|AKIA|ASIA|"
    r"AIza|ya29\.|hf_|npm_|shpat_|lin_api_|ntn_|secret_|eyJ)[A-Za-z0-9_\-.+/=]{8,}$"
)
_NODE_LAUNCHERS = frozenset({"node", "npm", "npx"})


@dataclass(frozen = True)
class _Location:
    source_id: str
    app: str
    label: Optional[str]
    path: Path
    # Where the server map lives: "mcpServers", "vscode-mcp" (servers), "vscode-settings"
    # (mcp.servers) or "claude-code" (mcpServers at the top level and per project).
    shape: str


@dataclass
class SourceServer:
    name: str
    # As written in the app's file: references unresolved, values unmasked. Never sent to a client.
    spec: object


@dataclass
class ImportSource:
    source_id: str
    app: str
    label: Optional[str]
    path: str
    servers: list[SourceServer] = field(default_factory = list)
    # Set when the file exists but could not be read; servers is then empty.
    error: Optional[str] = None


@dataclass
class PreparedServer:
    name: str
    transport: str
    # The command line or URL as written, credentials masked. The only form a client sees.
    target: str
    env_keys: list[str]
    header_keys: list[str]
    # What the import route creates, references resolved; None when the entry cannot be imported.
    entry: Optional[ParsedMcpEntry]
    error: Optional[str] = None
    # References no value was found for. Left in place, so the server is added switched off.
    unresolved: list[str] = field(default_factory = list)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:10]


def _config_dir(platform: str, environ: Mapping[str, str], home: Path) -> Path:
    """Where desktop apps keep per-user settings: %APPDATA%, Application Support, or XDG config."""
    if platform == "win32":
        return Path(environ.get("APPDATA") or home / "AppData" / "Roaming")
    if platform == "darwin":
        return home / "Library" / "Application Support"
    return Path(environ.get("XDG_CONFIG_HOME") or home / ".config")


def _config_locations(platform: str, environ: Mapping[str, str], home: Path) -> list[_Location]:
    config_dir = _config_dir(platform, environ, home)
    locations: list[_Location] = []
    if platform == "win32":
        local_app_data = Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
        # A Microsoft Store (MSIX) install has its %APPDATA% writes redirected into the package's
        # LocalCache and reads that copy first, so when one exists it is the live config.
        pattern = os.path.join(
            glob.escape(str(local_app_data)),
            "Packages",
            "Claude_*",
            "LocalCache",
            "Roaming",
            "Claude",
            "claude_desktop_config.json",
        )
        for match in sorted(glob.glob(pattern)):
            locations.append(
                _Location(
                    f"claude-desktop-store-{_digest(match)}",
                    "Claude Desktop",
                    "Microsoft Store install",
                    Path(match),
                    "mcpServers",
                )
            )
    locations.append(
        _Location(
            "claude-desktop",
            "Claude Desktop",
            None,
            config_dir / "Claude" / "claude_desktop_config.json",
            "mcpServers",
        )
    )
    claude_code_dir = environ.get("CLAUDE_CONFIG_DIR")
    locations.append(
        _Location(
            "claude-code",
            "Claude Code",
            None,
            Path(claude_code_dir) / ".claude.json" if claude_code_dir else home / ".claude.json",
            "claude-code",
        )
    )
    locations.append(
        _Location("cursor", "Cursor", None, home / ".cursor" / "mcp.json", "mcpServers")
    )
    for folder, source_id, label in (
        ("Code", "vscode", None),
        ("Code - Insiders", "vscode-insiders", "Insiders"),
    ):
        user_dir = config_dir / folder / "User"
        locations.append(
            _Location(source_id, "VS Code", label, user_dir / "mcp.json", "vscode-mcp")
        )
        locations.append(
            _Location(
                f"{source_id}-settings",
                "VS Code",
                f"{label} settings.json" if label else "settings.json",
                user_dir / "settings.json",
                "vscode-settings",
            )
        )
    locations.append(
        _Location(
            "windsurf",
            "Windsurf",
            None,
            home / ".codeium" / "windsurf" / "mcp_config.json",
            "mcpServers",
        )
    )
    return locations


def _outside_strings(text: str, visit) -> str:
    """Walk ``text`` once, copying string literals verbatim, so a "//" in a URL or a Windows path is
    never taken for a comment. ``visit(i)`` returns the index to skip to, or None to keep the char."""
    out: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i] == '"':
            start = i
            i += 1
            while i < length and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append(text[start:i])
            continue
        skip_to = visit(i)
        if skip_to is not None:
            i = skip_to
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def load_jsonc(text: str) -> object:
    """JSON as editors write it: VS Code's files are JSONC (comments and trailing commas are legal)
    and Notepad saves a BOM. Raises ValueError (json.JSONDecodeError) on a real syntax error."""
    if text.startswith("﻿"):
        text = text[1:]

    def comment(i: int) -> Optional[int]:
        if text[i] != "/" or i + 1 >= len(text):
            return None
        if text[i + 1] == "/":
            end = text.find("\n", i)
            return len(text) if end == -1 else end
        if text[i + 1] == "*":
            end = text.find("*/", i + 2)
            return len(text) if end == -1 else end + 2
        return None

    text = _outside_strings(text, comment)

    def trailing_comma(i: int) -> Optional[int]:
        if text[i] != ",":
            return None
        j = i + 1
        while j < len(text) and text[j].isspace():
            j += 1
        return i + 1 if j < len(text) and text[j] in "}]" else None

    return json.loads(_outside_strings(text, trailing_comma))


def _read_config(path: Path) -> object:
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("The file is too large to be an MCP config.")
    raw = path.read_bytes()
    # Windows PowerShell 5's Out-File and "Unicode" in Notepad write UTF-16 with a BOM.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return load_jsonc(raw.decode("utf-16"))
    return load_jsonc(raw.decode("utf-8-sig"))


def _unreadable(exc: Exception) -> str:
    """Why a file could not be read, without quoting it: a JSON error names a position, not content."""
    if isinstance(exc, json.JSONDecodeError):
        return f"Couldn't read this file: invalid JSON at line {exc.lineno}, column {exc.colno}."
    if isinstance(exc, UnicodeDecodeError):
        return "Couldn't read this file: it isn't UTF-8 text."
    if isinstance(exc, PermissionError):
        return "Couldn't read this file: permission denied."
    if isinstance(exc, OSError):
        return f"Couldn't read this file: {exc.strerror or type(exc).__name__}."
    return f"Couldn't read this file: {exc}"


def _server_map(container: object, shape: str) -> tuple[Optional[dict], Optional[str]]:
    if not isinstance(container, dict):
        return None, "Couldn't read this file: it doesn't hold a JSON object."
    if shape == "vscode-settings":
        mcp = container.get("mcp")
        servers = mcp.get("servers") if isinstance(mcp, dict) else None
    elif shape == "vscode-mcp":
        servers = container.get("servers", container.get("mcpServers"))
    else:
        servers = container.get("mcpServers")
    if servers is None:
        return {}, None
    if not isinstance(servers, dict):
        return None, "Couldn't read this file: its server list isn't a JSON object."
    return servers, None


def _source(
    location: _Location,
    servers: Optional[dict],
    error: Optional[str],
    *,
    source_id: Optional[str] = None,
    label: Optional[str] = None,
) -> ImportSource:
    items = list((servers or {}).items())[:_MAX_SERVERS_PER_SOURCE]
    return ImportSource(
        source_id = source_id or location.source_id,
        app = location.app,
        label = label if label is not None else location.label,
        path = str(location.path),
        servers = [SourceServer(str(name), spec) for name, spec in items],
        error = error,
    )


def _sources_at(location: _Location) -> list[ImportSource]:
    try:
        if not location.path.is_file():
            return []
    except OSError:
        return []
    try:
        data = _read_config(location.path)
    except (OSError, ValueError) as exc:
        return [_source(location, None, _unreadable(exc))]
    servers, error = _server_map(data, location.shape)
    found = [_source(location, servers, error)]
    if location.shape == "claude-code" and isinstance(data, dict):
        # `claude mcp add` without --scope stores the server under the project it ran in.
        projects = data.get("projects")
        for project, settings in (projects.items() if isinstance(projects, dict) else ()):
            project_servers, project_error = _server_map(settings, "mcpServers")
            if not project_servers and not project_error:
                continue
            # Either separator: the file may come from the other OS (WSL, a synced home folder).
            name = re.split(r"[\\/]", str(project).rstrip("\\/"))[-1] or str(project)
            found.append(
                _source(
                    location,
                    project_servers,
                    project_error,
                    source_id = f"claude-code-project-{_digest(str(project))}",
                    label = f"project {name}",
                )
            )
    # A shared file with no servers in it (VS Code settings, Claude Code state) is not a source.
    return [source for source in found if source.servers or source.error]


def _environment() -> tuple[str, Mapping[str, str], Path]:
    return sys.platform, os.environ, Path(os.path.expanduser("~"))


def discover_sources(
    *,
    platform: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> list[ImportSource]:
    """Every config on this machine that holds MCP servers, in a stable order. Two files of one app
    holding the same servers (a Store install next to a classic one) are listed once."""
    default_platform, default_environ, default_home = _environment()
    platform = platform or default_platform
    environ = default_environ if environ is None else environ
    home = home or default_home
    sources: list[ImportSource] = []
    seen: set[tuple[str, str]] = set()
    for location in _config_locations(platform, environ, home):
        for source in _sources_at(location):
            if source.servers:
                fingerprint = json.dumps(
                    [[server.name, server.spec] for server in source.servers],
                    sort_keys = True,
                    default = str,
                )
                if (source.app, fingerprint) in seen:
                    continue
                seen.add((source.app, fingerprint))
            sources.append(source)
    return sources


def resolve_references(
    value: object,
    *,
    environ: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    unresolved: Optional[list[str]] = None,
) -> tuple[object, list[str]]:
    """``value`` with every ``${env:NAME}``, ``${NAME}``, ``${NAME:-default}`` and ``${userHome}``
    filled in from Studio's own environment, as the app would at launch. A reference with no value
    (and no default) stays as written and is returned by name, so the caller can say what is
    missing instead of passing an empty string to the server."""
    environ = os.environ if environ is None else environ
    home = home or Path(os.path.expanduser("~"))
    missing: list[str] = [] if unresolved is None else unresolved

    def replace(match: re.Match[str]) -> str:
        body = match.group(1)
        if body == "userHome":
            return str(home)
        if body in ("pathSeparator", "/"):
            return os.sep
        if body.startswith("env:"):
            name, default = body[4:], None
        else:
            name, sep, default = body.partition(":-")
            if not sep:
                default = None
            if name in _EDITOR_VARIABLES or not _ENV_NAME.match(name):
                missing.append(body)
                return match.group(0)
        found = environ.get(name) if name else None
        if found is None:
            found = default
        if found is None:
            missing.append(name or body)
            return match.group(0)
        return found

    def walk(item: object) -> object:
        if isinstance(item, str):
            return _REFERENCE.sub(replace, item)
        if isinstance(item, list):
            return [walk(element) for element in item]
        if isinstance(item, dict):
            return {key: walk(element) for key, element in item.items()}
        return item

    resolved = walk(value)
    return resolved, list(dict.fromkeys(missing))


def _looks_like_key(value: str) -> bool:
    if _KNOWN_KEY_PREFIX.match(value):
        return True
    return (
        bool(_TOKEN_SHAPE.match(value))
        and any(char.isdigit() for char in value)
        and any(char.isalpha() for char in value)
    )


def mask_url(url: str) -> str:
    """A URL for display: user info, every query value and any key-shaped path segment masked."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return MASK
    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"{MASK}@{netloc.rpartition('@')[2]}"
    path = "/".join(
        MASK if _looks_like_key(segment) else segment for segment in parts.path.split("/")
    )
    query = "&".join(
        f"{name}={MASK}" if value else name
        for name, value in parse_qsl(parts.query, keep_blank_values = True)
    )
    return urlunsplit((parts.scheme, netloc, path, query, ""))


def _mask_argument(argument: str, secrets: list[str]) -> str:
    if "://" in argument:
        head, sep, rest = argument.partition("=")
        if sep and "://" not in head:
            return f"{head}={mask_url(rest)}"
        return mask_url(argument)
    masked = mask_secret_values(argument, secrets)
    head, sep, value = masked.partition("=")
    if sep and head and _looks_like_key(value):
        masked = f"{head}={MASK}"
    elif _looks_like_key(masked):
        masked = MASK
    return redact_log_text(masked).replace(REDACTED, MASK)


def masked_command(argv: list[str], env: Optional[Mapping[str, str]]) -> str:
    """A command line for display. Masks every configured env value wherever it appears, the value
    after a secret-named flag, URL credentials, and any argument shaped like a key."""
    headers = {str(key): str(value) for key, value in (env or {}).items()}
    try:
        secrets = mcp_secret_values(join_stdio_command(argv), headers)
    except ValueError:
        secrets = []
    return join_stdio_command([_mask_argument(argument, secrets) for argument in argv])


def _strip_label(name: str, error: str) -> str:
    prefix = f"{name.strip()}: "
    return error[len(prefix) :] if error.startswith(prefix) else error


def _normalized_spec(spec: object) -> object:
    if not isinstance(spec, dict):
        return spec
    spec = dict(spec)
    # Windsurf names a remote server's address serverUrl.
    if "url" not in spec and isinstance(spec.get("serverUrl"), str):
        spec["url"] = spec.pop("serverUrl")
    return spec


def prepare_server(
    server: SourceServer,
    *,
    environ: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> PreparedServer:
    """What one entry looks like to the dialog (masked, key names only) and what importing it would
    create (references resolved)."""
    spec = _normalized_spec(server.spec)
    is_command = isinstance(spec, dict) and bool(spec.get("command"))
    transport = "stdio" if is_command else "http"
    target = ""
    env_keys: list[str] = []
    header_keys: list[str] = []
    if isinstance(spec, dict):
        if is_command:
            env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
            args = spec.get("args") if isinstance(spec.get("args"), list) else []
            argv = [str(spec["command"]), *(str(arg) for arg in args)]
            target = masked_command(argv, env)
            env_keys = [str(key) for key in env]
        elif isinstance(spec.get("url"), str):
            target = mask_url(spec["url"])
            headers = spec.get("headers") if isinstance(spec.get("headers"), dict) else {}
            header_keys = [str(key) for key in headers]

    resolved, unresolved = resolve_references(spec, environ = environ, home = home)
    entry, error = parse_mcp_entry(server.name, resolved, allow_variable_references = True)
    if error:
        return PreparedServer(
            server.name,
            transport,
            target,
            env_keys,
            header_keys,
            None,
            _strip_label(server.name, error),
        )
    if entry is not None and entry.cwd and _REFERENCE.search(entry.cwd):
        return PreparedServer(
            server.name,
            transport,
            target,
            env_keys,
            header_keys,
            None,
            "Its working directory uses a variable Studio can't fill in.",
        )
    return PreparedServer(
        server.name, transport, target, env_keys, header_keys, entry, unresolved = unresolved
    )


def server_identity(url: str) -> Optional[tuple]:
    """What makes two servers the same one: the argv of a local program, or the address of a remote
    one with case-insensitive scheme and host and no trailing slash. Env and headers don't count, so
    a server already added with different credentials is still a duplicate."""
    if is_stdio(url):
        try:
            parts = parse_stdio_command(url)
        except ValueError:
            return None
        if not parts:
            return None
        program = os.path.normcase(parts[0]) if sys.platform == "win32" else parts[0]
        return ("stdio", program, *parts[1:])
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    return ("http", parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query)


def _launcher_name(command: str) -> str:
    name = os.path.basename(command).lower()
    for suffix in (".cmd", ".exe", ".bat", ".ps1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def find_program(
    command: str, env: Optional[Mapping[str, str]] = None, cwd: Optional[str] = None
) -> Optional[str]:
    """Where argv[0] resolves the way the spawn will look for it: on the server's own PATH when its
    env sets one, else Studio's, and for node/npm/npx also in the Node runtime Studio manages. A
    relative path counts from the working directory. None when nothing is there."""
    path = None
    for key, value in (env or {}).items():
        if (key.upper() if sys.platform == "win32" else key) == "PATH":
            path = value
    if path is None:
        path = os.environ.get("PATH", "")
    candidate = command
    if cwd and os.path.dirname(command) and not os.path.isabs(command):
        candidate = os.path.join(cwd, command)
    try:
        found = shutil.which(candidate, path = path)
        if (
            found is None
            and not os.path.dirname(command)
            and _launcher_name(command) in _NODE_LAUNCHERS
        ):
            from utils.node_runtime import managed_node_bin_dir

            bin_dir = managed_node_bin_dir()
            if bin_dir is not None:
                found = shutil.which(command, path = str(bin_dir))
    except (OSError, ValueError):
        return None
    return found
