# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A custom-location Windows install (UNSLOTH_STUDIO_HOME, install.ps1's env mode) must be
startable without a terminal, and its launcher must find a server started from one.

Env mode was built for throwaway CI roots, so New-StudioShortcuts wrote launch-studio.ps1 and
returned before any .lnk, and `unsloth studio update` (--shortcuts-only) hit the same return. A
user who saved the variable for their account is not a sandbox: they had no Start menu or
Desktop entry at all. And the env-mode launcher trusted only its own port file, so a server
started with `unsloth studio` was invisible to it and a click started a second one on the next
port against the same database.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from unsloth_pwsh_runner import PWSH, run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

requires_pwsh = pytest.mark.skipif(PWSH is None, reason = "PowerShell is unavailable")


def _text() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def _function(name: str) -> str:
    match = re.search(rf"    function {name} \{{.*?\n    \}}\n", _text(), flags = re.DOTALL)
    assert match is not None, f"install.ps1 no longer defines {name}"
    return match.group(0)


def _launcher_template() -> str:
    src = _text()
    start = src.index('$launcherContent = @"')
    return src[start : src.index('\n"@', start)]


def _new_studio_shortcuts() -> str:
    # Not _function(): the launcher here-string inside it has column-4 closing braces of its own.
    src = _text()
    start = src.index("    function New-StudioShortcuts {")
    return src[start : src.index("\n    if ($ShortcutsOnly) {", start)]


def _env_mode_gate() -> str:
    # The last env-mode branch before the .lnk work; the first one bakes the launcher's variables.
    body = _new_studio_shortcuts()
    end = body.index("$firstInstall = ")
    return body[body.rindex("if ($StudioRedirectMode -eq 'env') {", 0, end) : end]


# ---------------------------------------------------------------------------
# Shortcuts for a persisted custom root
# ---------------------------------------------------------------------------


def test_env_mode_creates_shortcuts_for_a_persisted_root_or_an_opt_in():
    gate = _env_mode_gate()
    assert "Test-StudioHomeIsPersisted -Path $StudioHome" in gate, (
        "env mode must still create shortcuts for a root the user persisted; without that a "
        "custom-location install can only be started from a terminal"
    )
    assert "$env:UNSLOTH_CREATE_SHORTCUTS" in gate
    for value in ("'1'", "'true'", "'yes'", "'on'"):
        assert value in gate, f"UNSLOTH_CREATE_SHORTCUTS no longer accepts {value}"
    # The skip is still there for a session-only root, and it says how to opt in.
    assert gate.count("return") == 1
    assert "UNSLOTH_CREATE_SHORTCUTS=1" in gate
    # The .lnk writer is reached from the same function once the gate passes.
    body = _new_studio_shortcuts()
    assert body.index("$wshell.CreateShortcut($linkPath)") > body.index(gate)


def test_shortcuts_only_still_goes_through_the_same_gate():
    # `unsloth studio update` reaches the installer only through --shortcuts-only.
    src = _text()
    block = src[src.index("    if ($ShortcutsOnly) {") :]
    assert "New-StudioShortcuts -ManagedPythonPath $ShortcutPython" in block[: block.index("\n    }\n")]


# Each case: persisted values by (name, scope), the path to test, and the expected answer. Values
# are relative to tmp_path; "<root>" is the install root, "<other>" another existing directory.
_PERSISTED_CASES = {
    "user var names the root": ({("UNSLOTH_STUDIO_HOME", "User"): "<root>"}, True),
    "trailing separator": ({("UNSLOTH_STUDIO_HOME", "User"): "<root>" + os.sep}, True),
    "machine var names the root": ({("UNSLOTH_STUDIO_HOME", "Machine"): "<root>"}, True),
    "legacy STUDIO_HOME names the root": ({("STUDIO_HOME", "User"): "<root>"}, True),
    "whitespace-only user value is unset": (
        {("UNSLOTH_STUDIO_HOME", "User"): "   ", ("UNSLOTH_STUDIO_HOME", "Machine"): "<root>"},
        True,
    ),
    "tilde expands against USERPROFILE": ({("UNSLOTH_STUDIO_HOME", "User"): "~/root"}, True),
    # A new session resolves UNSLOTH_STUDIO_HOME first, so a matching legacy name does not count.
    "UNSLOTH_STUDIO_HOME elsewhere beats a matching STUDIO_HOME": (
        {("UNSLOTH_STUDIO_HOME", "User"): "<other>", ("STUDIO_HOME", "User"): "<root>"},
        False,
    ),
    # A fresh process inherits the User value over the Machine one.
    "user value elsewhere beats a matching machine value": (
        {("UNSLOTH_STUDIO_HOME", "User"): "<other>", ("UNSLOTH_STUDIO_HOME", "Machine"): "<root>"},
        False,
    ),
    "nothing persisted (a session-only root)": ({}, False),
    "persisted path does not exist": ({("UNSLOTH_STUDIO_HOME", "User"): "<missing>"}, False),
}


@requires_pwsh
@pytest.mark.parametrize("case", sorted(_PERSISTED_CASES))
def test_persisted_root_detection(tmp_path, case):
    persisted, expected = _PERSISTED_CASES[case]
    root = tmp_path / "root"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()

    lookup = "[Environment]::GetEnvironmentVariable($name, $scope)"
    fn = _function("Test-StudioHomeIsPersisted")
    assert lookup in fn, "the persisted lookup changed shape; update this test's fake"
    # Off Windows there is no User or Machine scope at all, so the lookup is faked everywhere, from
    # process variables this test sets.
    fn = fn.replace(lookup, '[Environment]::GetEnvironmentVariable("FAKE_${name}_$scope")')

    env = {**os.environ, "USERPROFILE": str(tmp_path), "CASE_PATH": str(root)}
    for key in list(env):
        if key.startswith("FAKE_"):
            del env[key]
    for (name, scope), value in persisted.items():
        value = (
            value.replace("<root>", str(root))
            .replace("<other>", str(other))
            .replace("<missing>", str(tmp_path / "missing"))
        )
        env[f"FAKE_{name}_{scope}"] = value

    script = tmp_path / "persisted.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        + fn
        + "\n$StudioHome = (Resolve-Path -LiteralPath $env:CASE_PATH).Path\n"
        "'ANSWER:' + (Test-StudioHomeIsPersisted -Path $StudioHome)\n",
        encoding = "utf-8",
    )
    res = run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        env = env,
        verdict = "ANSWER:",
    )
    assert res.returncode == 0, res.stderr
    assert f"ANSWER:{expected}" in res.stdout, (case, res.stdout, res.stderr)


@requires_pwsh
@pytest.mark.skipif(os.name != "nt", reason = "case-insensitive paths are a Windows property")
def test_persisted_root_detection_ignores_case(tmp_path):
    root = tmp_path / "Root"
    root.mkdir()
    fn = _function("Test-StudioHomeIsPersisted").replace(
        "[Environment]::GetEnvironmentVariable($name, $scope)",
        '[Environment]::GetEnvironmentVariable("FAKE_${name}_$scope")',
    )
    script = tmp_path / "case.ps1"
    script.write_text(
        fn + "\n'ANSWER:' + (Test-StudioHomeIsPersisted -Path $env:CASE_PATH)\n", encoding = "utf-8"
    )
    env = {
        **os.environ,
        "FAKE_UNSLOTH_STUDIO_HOME_User": str(root).upper(),
        "CASE_PATH": str(root),
    }
    res = run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        env = env,
        verdict = "ANSWER:",
    )
    assert "ANSWER:True" in res.stdout, (res.stdout, res.stderr)


# ---------------------------------------------------------------------------
# The launcher finds a server started from a terminal
# ---------------------------------------------------------------------------


def _rendered_find_healthy_port() -> str:
    tpl = _launcher_template()
    start = tpl.index("function Find-HealthyStudioPort {")
    end = tpl.index("\n}\n", start) + len("\n}\n")
    return tpl[start:end].replace("`$", "$")


@requires_pwsh
def test_env_mode_launcher_scans_past_a_missing_or_stale_port_file(tmp_path):
    port_file = tmp_path / "studio.port"
    script = tmp_path / "find.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$portFile = '{str(port_file).replace(chr(39), chr(39) * 2)}'\n"
        "$script:healthy = @()\n"
        "$script:scanned = $false\n"
        "function Get-CandidatePorts { $script:scanned = $true; return @(8888, 8889) }\n"
        "function Test-StudioHealth { param([int]$Port) return ($Port -in $script:healthy) }\n"
        + _rendered_find_healthy_port()
        # Started from a terminal: healthy on 8888, no port file.
        + "$script:healthy = @(8888)\n"
        "'NOFILE:' + (Find-HealthyStudioPort)\n"
        # A stale cached port is dropped and the scan still finds the live one.
        "[System.IO.File]::WriteAllText($portFile, \"8890`n\")\n"
        "'STALE:' + (Find-HealthyStudioPort) + ':' + (Test-Path -LiteralPath $portFile)\n"
        # A healthy cached port answers without a scan.
        "[System.IO.File]::WriteAllText($portFile, \"8891`n\")\n"
        "$script:healthy = @(8888, 8891); $script:scanned = $false\n"
        "'CACHED:' + (Find-HealthyStudioPort) + ':' + $script:scanned\n"
        # Another install's server fails Test-StudioHealth (studio_root_id), so nothing is adopted.
        "Remove-Item -LiteralPath $portFile -Force\n"
        "$script:healthy = @()\n"
        "'FOREIGN:' + (Find-HealthyStudioPort)\n"
        "'DONE'\n",
        encoding = "utf-8",
    )
    res = run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        verdict = "DONE",
    )
    assert res.returncode == 0, res.stderr
    lines = res.stdout.splitlines()
    assert "NOFILE:8888" in lines, res.stdout
    assert "STALE:8888:False" in lines, res.stdout
    assert "CACHED:8891:False" in lines, res.stdout
    assert "FOREIGN:" in lines, res.stdout


def test_the_scan_cannot_adopt_another_installs_server():
    # What makes scanning safe in env mode: every candidate is checked against this install's id.
    tpl = _launcher_template()
    health = tpl[tpl.index("function Test-StudioHealth {") : tpl.index("function Get-CandidatePorts {")]
    assert "`$resp.studio_root_id -ne `$_ExpectedStudioRootId" in health
    assert "`$_ExpectedStudioRootId = '$_studioRootId'" in tpl


def test_launcher_waits_long_enough_for_a_cold_start():
    # A first start after a reboot or an update imports the whole backend from a cold disk cache;
    # 60 seconds reported failure on machines that came up a few seconds later.
    tpl = _launcher_template()
    match = re.search(r"^`\$timeoutSec = (\d+)$", tpl, flags = re.MULTILINE)
    assert match is not None
    assert int(match.group(1)) >= 120
    # The user-facing message reads the variable, so it cannot drift from the real budget.
    assert "within `$timeoutSec seconds" in tpl


def test_launcher_opens_studio_in_its_own_app_window():
    # A desktop app, not a browser tab: an Edge/Chrome app window (no tabs or address bar) on its own
    # profile, so its sign-in is its own and the browser process lives exactly as long as the window.
    tpl = _launcher_template()
    assert '"--app=`$url"' in tpl
    assert "('--user-data-dir=\"' + `$appProfileDir + '\"')" in tpl
    assert "`$appProfileDir = '`$_appProfileSq'" not in tpl  # baked at install time, not at launch
    assert "`$appProfileDir = '$_appProfileSq'" in tpl
    # No Edge or Chrome: the default browser, as before.
    assert re.search(r"if \(-not `\$browser\) \{\s*Start-Process `\$url", tpl)
    # Every path that used to open a tab now opens the window.
    assert 'Start-Process "http://localhost:' not in tpl
    assert tpl.count("Open-StudioWindow -Port") == 3


def test_launcher_runs_the_server_without_a_console_window():
    tpl = _launcher_template()
    launch = tpl[tpl.index("`$launchArgs = @(") : tpl.index("`$deadline = (Get-Date)", tpl.index("`$launchArgs = @("))]
    assert "-NoExit" not in launch
    assert "-WindowStyle Hidden" in launch
    # RemoteSigned beside the hidden window, never Bypass (test_launch_studio_launcher pins why).
    assert "'RemoteSigned'" in launch and "Bypass" not in launch


def test_closing_the_window_asks_before_stopping_only_what_this_launcher_started():
    tpl = _launcher_template()
    tail = tpl[tpl.index("`$window = Open-StudioWindow -Port `$healthyPort") :]
    assert "`$window.WaitForExit()" in tail
    assert "'YesNo' 'Question'" in tail
    assert "Stop-StudioBackend" in tail
    # Opening a second window on a running server hands off and exits: only the starter asks.
    existing = tpl[tpl.index("`$existingPort = Find-HealthyStudioPort") : tpl.index("`$launchMutex =")]
    assert "Open-StudioWindow -Port `$existingPort | Out-Null" in existing and "exit 0" in existing
    stop = tpl[tpl.index("function Stop-StudioBackend") : tpl.index("`$existingPort = Find-HealthyStudioPort")]
    assert "' studio stop\"" in stop
