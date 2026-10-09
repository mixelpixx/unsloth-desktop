# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from storage.studio_db import connect_studio_db
from utils.paths import studio_db_path, ensure_dir

_schema_lock = threading.Lock()
_schema_ready: set[Path] = set()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_servers (
            id TEXT NOT NULL PRIMARY KEY,
            display_name TEXT NOT NULL,
            url TEXT NOT NULL,
            headers_json TEXT,
            is_enabled INTEGER NOT NULL DEFAULT 1,
            use_oauth INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Backfill use_oauth for pre-existing DBs.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(mcp_servers)").fetchall()}
    if "use_oauth" not in cols:
        conn.execute("ALTER TABLE mcp_servers ADD COLUMN use_oauth INTEGER NOT NULL DEFAULT 0")
    # cwd: a local program's working directory (NULL = the backend's own).
    for column in (
        "builtin_id",
        "builtin_config_json",
        "image_input_mappings_json",
        "oauth_client_id",
        "oauth_client_secret",
        "cwd",
    ):
        if column not in cols:
            conn.execute(f"ALTER TABLE mcp_servers ADD COLUMN {column} TEXT")
    # A local program's lifecycle. Rows saved before it existed keep "per_chat", the isolation they were configured
    # under: switching them to one process shared by every chat would hand one conversation's server-side state (a
    # signed-in browser, an open SSH session, a loaded toolset) to the next without the owner choosing that. The
    # routes pick "shared" for servers created or imported from here on. idle_timeout_seconds NULL = the mode's
    # default, 0 = never.
    if "process_mode" not in cols:
        conn.execute(
            "ALTER TABLE mcp_servers ADD COLUMN process_mode TEXT NOT NULL DEFAULT 'per_chat'"
        )
    if "idle_timeout_seconds" not in cols:
        conn.execute("ALTER TABLE mcp_servers ADD COLUMN idle_timeout_seconds INTEGER")
    # Per-tool settings (core.inference.mcp_client: server_disabled_tools, server_ask_tools): JSON arrays of raw tool
    # names, NULL = none. The tools turned OFF are stored, so an existing row, and a tool a server adds later, stay on.
    for column in ("disabled_tools_json", "ask_tools_json"):
        if column not in cols:
            conn.execute(f"ALTER TABLE mcp_servers ADD COLUMN {column} TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS mcp_servers_builtin_id ON mcp_servers(builtin_id)"
    )


def reset_schema_state_for_tests() -> None:
    with _schema_lock:
        _schema_ready.clear()


def get_connection() -> sqlite3.Connection:
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    conn = connect_studio_db(db_path)
    conn.row_factory = sqlite3.Row
    if db_path not in _schema_ready:
        with _schema_lock:
            schema_path = db_path.resolve()
            if schema_path not in _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready.add(schema_path)
                except Exception:
                    conn.close()
                    raise
    return conn


def create_server(
    id: str,
    display_name: str,
    url: str,
    headers_json: Optional[str] = None,
    is_enabled: bool = True,
    use_oauth: bool = False,
    builtin_id: Optional[str] = None,
    builtin_config_json: Optional[str] = None,
    image_input_mappings_json: Optional[str] = None,
    oauth_client_id: Optional[str] = None,
    oauth_client_secret: Optional[str] = None,
    cwd: Optional[str] = None,
    process_mode: str = "per_chat",
    idle_timeout_seconds: Optional[int] = None,
) -> None:
    from core.inference.mcp_client import validate_mcp_address

    validate_mcp_address(url)
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO mcp_servers
                (id, display_name, url, headers_json,
                 is_enabled, use_oauth, created_at, updated_at, builtin_id, builtin_config_json,
                 image_input_mappings_json, oauth_client_id, oauth_client_secret, cwd,
                 process_mode, idle_timeout_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                display_name,
                url,
                headers_json,
                int(is_enabled),
                int(use_oauth),
                now,
                now,
                builtin_id,
                builtin_config_json,
                image_input_mappings_json,
                oauth_client_id,
                oauth_client_secret,
                cwd,
                process_mode,
                idle_timeout_seconds,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_server(id: str, changes: dict) -> bool:
    """Apply column updates and bump ``updated_at``. Returns True on a hit."""
    if not changes:
        return False
    if "url" in changes:
        from core.inference.mcp_client import validate_mcp_address
        validate_mcp_address(changes["url"])
    bool_cols = {"is_enabled", "use_oauth"}
    sets, params = [], []
    for col, value in changes.items():
        sets.append(f"{col} = ?")
        params.append(int(value) if col in bool_cols else value)
    sets.append("updated_at = ?")
    params.extend([datetime.now(timezone.utc).isoformat(), id])

    conn = get_connection()
    try:
        cursor = conn.execute(
            f"UPDATE mcp_servers SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_server(id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM mcp_servers WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_server(id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM mcp_servers WHERE id = ?", (id,)).fetchone()
        return _effective_row(dict(row)) if row else None
    finally:
        conn.close()


def list_servers() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM mcp_servers ORDER BY created_at").fetchall()
        return [_effective_row(dict(row)) for row in rows]
    finally:
        conn.close()


def get_server_for_tool(key: str) -> Optional[dict]:
    if key == "blender":
        return next((row for row in list_servers() if row.get("builtin_id") == key), None)
    return get_server(key)


def _effective_row(row: dict) -> dict:
    if row.get("builtin_id"):
        from integrations.blender.service import resolve_server
        return resolve_server(row)
    return row
