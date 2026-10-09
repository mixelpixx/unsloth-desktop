# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stateful stdio MCP server for the lifecycle tests: it counts its calls, can take a while to start, and can add
a tool at runtime and say so with notifications/tools/list_changed (as konnect does when it loads a toolset)."""

import json
import os
import time

import mcp.types
from fastmcp import Context, FastMCP


start_delay = float(os.environ.get("UNSLOTH_MCP_LIFECYCLE_START_DELAY") or 0)
if start_delay:
    time.sleep(start_delay)

launch_log = os.environ.get("UNSLOTH_MCP_LIFECYCLE_LOG")
if launch_log:
    with open(launch_log, "a", encoding = "utf-8") as log:
        log.write(f"{os.getpid()}\n")


server = FastMCP("lifecycle")
state = {"calls": 0}


@server.tool
def whoami() -> str:
    state["calls"] += 1
    return json.dumps({"pid": os.getpid(), "calls": state["calls"]})


@server.tool
async def load_toolset(name: str, ctx: Context) -> str:
    def toolset_tool() -> str:
        return json.dumps({"pid": os.getpid(), "tool": name})

    server.tool(name = name)(toolset_tool)
    await ctx.send_notification(mcp.types.ToolListChangedNotification())
    return f"loaded {name}"


@server.tool
def slow(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


if __name__ == "__main__":
    server.run(transport = "stdio", show_banner = False)
