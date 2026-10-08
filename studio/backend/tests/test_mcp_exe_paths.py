"""Local-program (.exe) MCP servers on Windows-style paths.

Regression: a launch with `-H 0.0.0.0` (the README's LAN example) withholds the loopback stdio default, and a
lone Windows path such as ``D:\\mcp\\server.exe`` was then answered with "must start with http:// or https://"
because urlparse reads the drive letter as a URL scheme. The user saw an .exe "rejected as not http" with no
hint that the network bind was the cause. Also covers Explorer's "Copy as path" quotes, which used to be
escaped into the program name, and the capabilities route the dialog reads before Save.
"""

import os

import pytest
from fastapi import HTTPException

from core.inference import mcp_client
from utils import host_policy


@pytest.fixture(autouse = True)
def _isolate_stdio_env():
    from state import tool_policy

    saved = os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP")
    saved_policy = tool_policy.get_tool_policy()
    host_policy._reset_loopback_default_state()
    os.environ.pop("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", None)
    yield
    host_policy._reset_loopback_default_state()
    tool_policy.set_tool_policy(saved_policy)
    if saved is None:
        os.environ.pop("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", None)
    else:
        os.environ["UNSLOTH_STUDIO_ALLOW_STDIO_MCP"] = saved


PROGRAM_VALUES = [
    r"D:\tools\my-mcp\server.exe",
    r"C:/tools/server.exe",
    r"\\fileserver\share\server.exe",
    r'"C:\Program Files\Foo\server.exe"',
    "server.exe",
    "./server",
    "/usr/local/bin/mcp-server",
    "~/bin/server",
]


def test_network_bind_records_why_stdio_is_off():
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    assert mcp_client.stdio_mcp_enabled() is False
    assert host_policy.stdio_mcp_withheld_reason() == "network"
    reason = mcp_client.stdio_mcp_disabled_reason()
    assert "-H 127.0.0.1" in reason
    assert "UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1" in reason
    assert ".exe" in reason


def test_colab_records_its_own_reason():
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1", is_colab = True)
    assert host_policy.stdio_mcp_withheld_reason() == "colab"
    assert "Colab" in mcp_client.stdio_mcp_disabled_reason()


def test_loopback_and_explicit_values_withhold_nothing(monkeypatch):
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert host_policy.stdio_mcp_withheld_reason() is None
    host_policy._reset_loopback_default_state()

    # An operator's explicit =1 on a network bind wins, and the bind is not blamed.
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    assert host_policy.stdio_mcp_withheld_reason() is None
    assert mcp_client.stdio_mcp_enabled() is True


def test_relaunch_on_loopback_clears_the_network_reason():
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert host_policy.stdio_mcp_withheld_reason() is None
    assert mcp_client.stdio_mcp_enabled() is True


@pytest.mark.parametrize("value", PROGRAM_VALUES)
def test_program_paths_get_the_stdio_reason_not_the_http_one(value):
    from routes.mcp_servers import _validate_url

    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    with pytest.raises(HTTPException) as exc:
        _validate_url(value)
    detail = exc.value.detail
    assert exc.value.status_code == 400
    assert not detail.startswith("MCP server address must start with"), detail
    assert "-H 127.0.0.1" in detail


@pytest.mark.parametrize("value", ["example.com", "example.com/mcp", "localhost:8000", "ftp://host/mcp"])
def test_url_like_values_still_get_the_http_hint(value):
    from routes.mcp_servers import _validate_url

    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    with pytest.raises(HTTPException) as exc:
        _validate_url(value)
    assert exc.value.detail.startswith("MCP server address must start with")


@pytest.mark.parametrize("value", PROGRAM_VALUES)
def test_program_paths_are_accepted_on_loopback(value):
    from routes.mcp_servers import _validate_url

    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert _validate_url(value)


@pytest.mark.parametrize("quote", ['"', "'"])
def test_encode_strips_copy_as_path_quotes(quote):
    from models.mcp_servers import McpStdioCommand
    from routes.mcp_servers import encode_stdio_command

    path = r"C:\Program Files\Foo\server.exe"
    encoded = encode_stdio_command(
        McpStdioCommand(command = f"  {quote}{path}{quote} ", arguments = ["--stdio"]),
        current_subject = "unsloth",
        via_api_key = False,
    )
    assert mcp_client.parse_stdio_command(encoded.url) == [path, "--stdio"]


def test_encode_keeps_an_unbalanced_quote():
    from routes.mcp_servers import _strip_wrapping_quotes

    assert _strip_wrapping_quotes('"C:\\x\\server.exe') == '"C:\\x\\server.exe'
    assert _strip_wrapping_quotes('"') == '"'


def test_capabilities_reports_the_gate():
    from routes.mcp_servers import get_mcp_capabilities

    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    on = get_mcp_capabilities(current_subject = "unsloth", via_api_key = False, no_credential = False)
    assert on.stdio_enabled is True and on.stdio_disabled_reason is None

    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    off = get_mcp_capabilities(current_subject = "unsloth", via_api_key = False, no_credential = False)
    assert off.stdio_enabled is False
    assert "-H 127.0.0.1" in off.stdio_disabled_reason


@pytest.mark.parametrize("via_api_key,no_credential", [(True, False), (False, True)])
def test_capabilities_never_offers_stdio_to_api_keys(via_api_key, no_credential):
    from routes.mcp_servers import get_mcp_capabilities

    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    caps = get_mcp_capabilities(
        current_subject = "unsloth", via_api_key = via_api_key, no_credential = no_credential
    )
    assert caps.stdio_enabled is False
    assert caps.stdio_disabled_reason
