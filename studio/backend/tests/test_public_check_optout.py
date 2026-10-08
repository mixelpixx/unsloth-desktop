# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK (#7307 Problem 8) and the desktop default.

A wildcard bind asks ifconfig.me for the public IP and check-host.net whether the
port is reachable. Both stay on by default on a server OS; setting the var skips
both, which is what lab and privacy-sensitive deployments asked for.

On Windows and macOS both are skipped by default: a desktop behind a home router
can only be told its router is unreachable, with an ssh tunnel to the router as the
suggested fix. UNSLOTH_STUDIO_FORCE_PUBLIC_CHECK turns them back on there, and the
disable switch still beats it everywhere.
"""

import socket
import urllib.request
from types import SimpleNamespace

import pytest

import run
from run import (
    DISABLE_PUBLIC_CHECK_ENV,
    FORCE_PUBLIC_CHECK_ENV,
    _resolve_external_ip,
    _verify_global_reachability,
    public_check_disabled,
    public_check_skipped,
)

IFCONFIG = "https://ifconfig.me"
CHECK_HOST = "check-host.net"


class _FakeSocket:
    """Stand-in for the step 3 UDP route lookup."""

    def connect(self, addr):
        pass

    def getsockname(self):
        return ("192.168.1.50", 0)

    def close(self):
        pass


@pytest.fixture
def calls(monkeypatch):
    """Record every outbound URL and fail it, so resolution reaches the LAN step. On a server OS,
    whatever this test runs on, so "by default" below means the server default."""
    seen = []

    def _urlopen(req, *args, **kwargs):
        seen.append(req if isinstance(req, str) else req.full_url)
        raise OSError("no network in this test")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: _FakeSocket())
    monkeypatch.setattr(run, "_on_desktop_os", lambda: False)
    monkeypatch.delenv(DISABLE_PUBLIC_CHECK_ENV, raising = False)
    monkeypatch.delenv(FORCE_PUBLIC_CHECK_ENV, raising = False)
    return seen


@pytest.fixture
def desktop(monkeypatch, calls):
    """The same recorder on Windows or macOS. The logger is quietened because an unconfigured one
    writes debug records to stdout, where a real launch filters them out at INFO."""
    monkeypatch.setattr(run, "_on_desktop_os", lambda: True)
    monkeypatch.setattr(run, "_stdout_color_ok", lambda: False)
    quiet = lambda *a, **k: None
    monkeypatch.setattr(run, "logger", SimpleNamespace(debug = quiet, info = quiet, warning = quiet))
    return calls


# ── public_check_disabled ───────────────────────────────────────────


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv(DISABLE_PUBLIC_CHECK_ENV, raising = False)
    assert public_check_disabled() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", " 1 "])
def test_disabling_values(monkeypatch, raw):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, raw)
    assert public_check_disabled() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "  ", "ture"])
def test_anything_else_leaves_it_on(monkeypatch, raw):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, raw)
    assert public_check_disabled() is False


# ── the two lookups ─────────────────────────────────────────────────


def test_public_ip_lookup_runs_by_default(calls):
    assert _resolve_external_ip() == "192.168.1.50"
    assert IFCONFIG in calls


def test_public_ip_lookup_skipped_when_disabled(monkeypatch, calls):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, "1")

    assert _resolve_external_ip() == "192.168.1.50", "the LAN address still resolves"
    assert IFCONFIG not in calls


def test_display_host_resolves_every_wildcard_alias(monkeypatch):
    monkeypatch.setattr(run, "_resolve_external_ip", lambda: "192.168.1.50")
    monkeypatch.setattr("lan_access.detect_lan_addresses", lambda _ip_version = 4: ["fd00::50"])
    for host in ("0.0.0.0", "0", "::ffff:0.0.0.0"):
        assert run._display_host_for_bind(host) == "192.168.1.50"
    for host in ("::", "::0", "0:0:0:0:0:0:0:0"):
        assert run._display_host_for_bind(host) == "fd00::50"

    monkeypatch.setattr("lan_access.detect_lan_addresses", lambda _ip_version = 4: [])
    assert run._display_host_for_bind("::") == "::"


def test_display_host_falls_back_to_ipv6_for_dual_stack_wildcard(monkeypatch):
    original_getaddrinfo = socket.getaddrinfo

    def dual_stack_wildcard(host, *args, **kwargs):
        if host == "dual-wildcard.test":
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 0)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0)),
            ]
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", dual_stack_wildcard)
    monkeypatch.setattr(run, "_resolve_external_ip", lambda: "0.0.0.0")
    monkeypatch.setattr("lan_access.detect_lan_addresses", lambda _ip_version = 4: ["fd00::50"])

    assert run._display_host_for_bind("dual-wildcard.test") == "fd00::50"


def test_reachability_probe_runs_by_default(calls):
    _verify_global_reachability("95.216.11.2", 8888)
    assert any(CHECK_HOST in url for url in calls)


def test_ipv6_reachability_probe_brackets_the_host(calls):
    import urllib.parse

    _verify_global_reachability("2001:4860:4860::8844", 8888)
    request_url = next(url for url in calls if CHECK_HOST in url)
    query = urllib.parse.parse_qs(urllib.parse.urlparse(request_url).query)
    assert query["host"] == ["[2001:4860:4860::8844]:8888"]


def test_reachability_probe_skipped_when_disabled(monkeypatch, calls, capsys):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, "1")

    _verify_global_reachability("95.216.11.2", 8888)
    capsys.readouterr()

    assert not any(CHECK_HOST in url for url in calls)
    assert run._public_reachable is None, "skipping must not claim a reachability result"


# ── the desktop default ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "platform, is_desktop", [("win32", True), ("darwin", True), ("linux", False)]
)
def test_desktop_platforms(monkeypatch, platform, is_desktop):
    monkeypatch.setattr(run.sys, "platform", platform)
    assert run._on_desktop_os() is is_desktop


def test_desktop_skips_both_lookups_by_default(desktop):
    assert public_check_disabled() is False, "the operator opt-out itself is untouched"
    assert public_check_skipped() is True

    assert _resolve_external_ip() == "192.168.1.50", "the LAN address still resolves"
    _verify_global_reachability("95.216.11.2", 8888)

    assert IFCONFIG not in desktop
    assert not any(CHECK_HOST in url for url in desktop)
    assert run._public_reachable is None, "skipping must not claim a reachability result"


def test_desktop_skip_is_one_line_naming_the_lan_url(desktop, capsys):
    _verify_global_reachability("95.216.11.2", 8888, lan_host = "192.168.1.50")
    out = capsys.readouterr().out

    assert out.count("\n") == 1, out
    assert "http://192.168.1.50:8888/" in out
    assert FORCE_PUBLIC_CHECK_ENV in out, "the line says how to run the check anyway"
    assert "NOT reachable" not in out and "ssh -L" not in out


def test_desktop_skip_brackets_an_ipv6_lan_host(desktop, capsys):
    _verify_global_reachability("2001:4860:4860::8844", 8888, lan_host = "fd00::50")
    assert "http://[fd00::50]:8888/" in capsys.readouterr().out


@pytest.mark.parametrize("lan_host", ["", "0.0.0.0", "::", "127.0.0.1"])
def test_desktop_skip_is_silent_without_a_usable_lan_address(desktop, capsys, lan_host):
    # Nothing is looked up to fill the gap: the line is only for an address the banner already had.
    _verify_global_reachability("95.216.11.2", 8888, lan_host = lan_host)
    assert capsys.readouterr().out == ""
    assert desktop == []


def test_desktop_private_address_note_is_unchanged(desktop, capsys):
    # The usual desktop case once ifconfig.me is skipped: the route lookup answers with a LAN address,
    # which is decided locally, before any skip, exactly as on a server.
    _verify_global_reachability("192.168.1.50", 8888, lan_host = "192.168.1.50")
    assert "private/LAN address" in capsys.readouterr().out
    assert run._public_reachable is False
    assert desktop == []


@pytest.mark.parametrize("raw", ["1", "true", "YES", " yes "])
def test_force_runs_both_lookups_on_a_desktop(desktop, monkeypatch, raw):
    monkeypatch.setenv(FORCE_PUBLIC_CHECK_ENV, raw)
    assert public_check_skipped() is False

    _resolve_external_ip()
    _verify_global_reachability("95.216.11.2", 8888)

    assert IFCONFIG in desktop
    assert any(CHECK_HOST in url for url in desktop)


@pytest.mark.parametrize("raw", ["0", "false", "off", ""])
def test_anything_else_does_not_force(desktop, monkeypatch, raw):
    monkeypatch.setenv(FORCE_PUBLIC_CHECK_ENV, raw)
    assert public_check_skipped() is True


def test_disable_beats_force(desktop, monkeypatch):
    monkeypatch.setenv(FORCE_PUBLIC_CHECK_ENV, "1")
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, "1")
    assert public_check_skipped() is True

    _resolve_external_ip()
    _verify_global_reachability("95.216.11.2", 8888)

    assert IFCONFIG not in desktop
    assert not any(CHECK_HOST in url for url in desktop)


def test_server_default_is_unchanged(calls, monkeypatch):
    assert public_check_skipped() is False
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, "1")
    assert public_check_skipped() is True
