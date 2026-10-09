# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Network-free Unsloth release version resolution for display-only UI."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from utils import _studio_release_build

_DEV_VERSION = "dev"
_GIT_TIMEOUT_SECONDS = 1.0
_STUDIO_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.][0-9A-Za-z.-]*)?$")
_GIT_DESCRIBE_SUFFIX_RE = re.compile(r"-\d+-g[0-9A-Fa-f]+(?:-dirty)?$")
_GIT_BRANCH_RE = re.compile(r"^[0-9A-Za-z._/-]+$")
_MAX_VERSION_LENGTH = 64


def is_valid_studio_release_version(value: object) -> bool:
    """Return True for Unsloth release tags such as ``v0.1.39-beta``."""
    if not isinstance(value, str):
        return False
    version = value.strip()
    if not version or len(version) > _MAX_VERSION_LENGTH:
        return False
    if version.endswith("-dirty") or _GIT_DESCRIBE_SUFFIX_RE.search(version):
        return False
    return _STUDIO_TAG_RE.fullmatch(version) is not None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_is_in_site_packages(path: Path) -> bool:
    return any(part in {"site-packages", "dist-packages"} for part in path.parts)


def _is_source_checkout(repo_root: Path) -> bool:
    return (repo_root / ".git").exists() and not _path_is_in_site_packages(Path(__file__).resolve())


def _exact_git_studio_tag(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "describe",
                "--tags",
                "--exact-match",
                "--match",
                "v[0-9]*",
                "HEAD",
            ],
            cwd = repo_root,
            check = False,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = _GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    tag = result.stdout.strip()
    return tag if is_valid_studio_release_version(tag) else None


def _git_branch(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd = repo_root,
            check = False,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = _GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    branch = result.stdout.strip()
    # "HEAD" means detached, e.g. a tag or commit checkout.
    if (
        not branch
        or branch == "HEAD"
        or len(branch) > _MAX_VERSION_LENGTH
        or _GIT_BRANCH_RE.fullmatch(branch) is None
    ):
        return None
    return branch


_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_GIT_STATUS_TIMEOUT_SECONDS = 3.0


def _git_stdout(repo_root: Path, args: list[str], timeout: float) -> str | None:
    try:
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        hidden = windows_hidden_subprocess_kwargs()
    except Exception:
        hidden = {}
    try:
        result = subprocess.run(
            ["git", *args],
            cwd = repo_root,
            check = False,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            **hidden,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def get_source_checkout_info(repo_root: Path | None = None) -> dict[str, object] | None:
    """Branch, short commit and whether tracked files are modified, for a source checkout.

    None when Unsloth is not running from a Git checkout. For diagnostics only: local git
    calls with short timeouts, never a fetch. ``dirty`` is None when git could not say in time.
    """
    resolved_repo_root = repo_root or _repo_root()
    if not _is_source_checkout(resolved_repo_root):
        return None
    commit = (_git_stdout(resolved_repo_root, ["rev-parse", "--short=10", "HEAD"], _GIT_TIMEOUT_SECONDS) or "").strip()
    status = _git_stdout(
        resolved_repo_root,
        ["status", "--porcelain", "--untracked-files=no"],
        _GIT_STATUS_TIMEOUT_SECONDS,
    )
    return {
        "branch": _git_branch(resolved_repo_root),
        "commit": commit if _GIT_COMMIT_RE.fullmatch(commit) else None,
        "dirty": None if status is None else bool(status.strip()),
    }


def get_studio_version(repo_root: Path | None = None) -> str:
    """Return the installed Unsloth release tag for display, or ``dev``.

    Intentionally separate from the PyPI ``unsloth`` package version used by
    update checks. Never performs network requests.
    """
    resolved_repo_root = repo_root or _repo_root()

    if _is_source_checkout(resolved_repo_root):
        git_tag = _exact_git_studio_tag(resolved_repo_root)
        if git_tag is not None:
            return git_tag
        branch = _git_branch(resolved_repo_root)
        return f"GitHub {branch}" if branch is not None else _DEV_VERSION

    stamped_version = _studio_release_build.STUDIO_RELEASE_VERSION
    if is_valid_studio_release_version(stamped_version):
        return stamped_version.strip()

    return _DEV_VERSION
