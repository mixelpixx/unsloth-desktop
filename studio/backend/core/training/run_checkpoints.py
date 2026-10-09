# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The saved checkpoints of one training run, and what can be done with each.

Callers hand over a run row and a checkpoint id ("checkpoint-<step>", or "final" for the run's own
save). Every path is derived here from the run's recorded output_dir; a client never names one.
Nothing here overwrites a checkpoint: a delete moves the files aside before removing them, and a
resume from an older checkpoint copies it into a new run folder, so the newer ones stay untouched.
"""

import json
import math
import os
import re
import shutil
import stat
import threading
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Optional

from loggers import get_logger
from utils.paths import outputs_root, resolve_output_dir
from utils.paths.storage_roots import within_account

logger = get_logger(__name__)

FINAL_CHECKPOINT_ID = "final"
_CHECKPOINT_ID = re.compile(r"checkpoint-(\d{1,12})")
# What makes a folder a loadable save: PEFT / full model configs, an MLX adapter, a Laya decision run.
_MODEL_MARKERS = (
    "adapter_config.json",
    "config.json",
    "adapters.safetensors",
    "rl_agent_config.json",
)
# Run-level training state that sits beside the checkpoints rather than in them.
_ROW_BOUND_MARKER = "unsloth_row_bound.json"
_TRAINER_STATE_MAX_BYTES = 64 * 1024 * 1024
_FORK_MARGIN_BYTES = 256 * 1024 * 1024
_FORK_NAME_ATTEMPTS = 100
_MAX_DIR_NAME_BYTES = 255

# Deletes and forks of one run's checkpoints are serialized, so a fork never copies a checkpoint
# that a concurrent delete is moving aside.
_mutation_lock = threading.Lock()

BestBasis = Literal["eval_loss", "train_loss"]
ResumeMode = Literal["in_place", "fork"]


class CheckpointActionError(Exception):
    """A refusal the route turns into an HTTP error with a stable code."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen = True)
class RunCheckpoint:
    id: str
    path: Path
    step: Optional[int]
    is_final: bool


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_real_dir(path: Path) -> bool:
    # lstat: a symlink to a directory is not one of the run's own folders.
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def _contained_child(run_dir: Path, child: Path) -> bool:
    """A real directory directly inside the run folder. Checked on the resolved path as well, so a
    junction or link named like a checkpoint cannot carry a delete or a copy somewhere else."""
    if not _is_real_dir(child) or not within_account(child):
        return False
    try:
        return Path(os.path.realpath(child)).parent == Path(os.path.realpath(run_dir))
    except OSError:
        return False


def _has_model_marker(path: Path) -> bool:
    return any((path / name).is_file() for name in _MODEL_MARKERS)


def _top_level_files(path: Path) -> list[Path]:
    files: list[Path] = []
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks = False):
                files.append(Path(entry.path))
    except OSError:
        return []
    return files


def run_output_dir(run: dict) -> Optional[Path]:
    """The run's folder under this account's outputs root, or None when it is gone or points elsewhere."""
    raw = run.get("output_dir")
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()
    if not Path(raw).is_absolute() and (
        PureWindowsPath(raw).is_absolute() or PurePosixPath(raw).is_absolute()
    ):
        return None
    try:
        root = outputs_root().expanduser().resolve(strict = False)
        path = resolve_output_dir(raw).resolve(strict = False)
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    if path == root or not _is_real_dir(path) or not within_account(path):
        return None
    return path


def _read_trainer_state(path: Path) -> Optional[dict]:
    state_file = path / "trainer_state.json"
    try:
        if not state_file.is_file() or state_file.stat().st_size > _TRAINER_STATE_MAX_BYTES:
            return None
        state = json.loads(state_file.read_text(encoding = "utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def _state_step(state: Optional[dict]) -> Optional[int]:
    step = state.get("global_step") if state else None
    return step if isinstance(step, int) and not isinstance(step, bool) and step >= 0 else None


def list_run_checkpoints(run: dict, run_dir: Path) -> list[RunCheckpoint]:
    """Newest first, the run's own final save ahead of the numbered checkpoints.

    A run a later resume continued shares its folder with that continuation, so it lists only the
    checkpoints it wrote itself, and not the final save, which belongs to the continuation."""
    resumed_later = bool(run.get("resumed_later"))
    final_step = run.get("final_step") if isinstance(run.get("final_step"), int) else None
    numbered: list[RunCheckpoint] = []
    try:
        children = list(run_dir.iterdir())
    except OSError:
        children = []
    for child in children:
        match = _CHECKPOINT_ID.fullmatch(child.name)
        if match is None or not _contained_child(run_dir, child) or not _has_model_marker(child):
            continue
        step = int(match.group(1))
        if resumed_later and final_step is not None and step > final_step:
            continue
        numbered.append(RunCheckpoint(child.name, child, step, False))
    numbered.sort(key = lambda entry: entry.step or 0, reverse = True)
    if resumed_later or not _has_model_marker(run_dir):
        return numbered
    state_step = _state_step(_read_trainer_state(run_dir))
    final = RunCheckpoint(
        FINAL_CHECKPOINT_ID,
        run_dir,
        state_step if state_step is not None else final_step,
        True,
    )
    return [final, *numbered]


def find_run_checkpoint(run: dict, checkpoint_id: str) -> tuple[Path, RunCheckpoint]:
    """Resolve an id against what is actually on disk; the id itself never becomes part of a path."""
    run_dir = run_output_dir(run)
    if run_dir is None:
        raise CheckpointActionError(404, "run_output_missing", "This run's output folder is gone.")
    for entry in list_run_checkpoints(run, run_dir):
        if entry.id == checkpoint_id:
            return run_dir, entry
    raise CheckpointActionError(404, "checkpoint_not_found", "Checkpoint not found.")


def _tree_bytes(path: Path) -> Optional[int]:
    total = 0
    try:
        # os.walk does not follow directory links, and lstat does not follow file links.
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    continue
    except OSError:
        return None
    return total


def checkpoint_size_bytes(entry: RunCheckpoint) -> Optional[int]:
    # The final save is the run folder's own files; its checkpoint subfolders are listed separately.
    if entry.is_final:
        total = 0
        for file in _top_level_files(entry.path):
            try:
                total += os.lstat(file).st_size
            except OSError:
                continue
        return total
    return _tree_bytes(entry.path)


def checkpoint_saved_at(entry: RunCheckpoint) -> Optional[str]:
    latest: Optional[float] = None
    for file in _top_level_files(entry.path):
        if entry.is_final and file.name == _ROW_BOUND_MARKER:
            continue
        try:
            mtime = os.lstat(file).st_mtime
        except OSError:
            continue
        latest = mtime if latest is None else max(latest, mtime)
    if latest is None:
        try:
            latest = os.lstat(entry.path).st_mtime
        except OSError:
            return None
    return datetime.fromtimestamp(latest, tz = timezone.utc).isoformat()


def _state_points(state: Optional[dict]) -> list[dict]:
    history = state.get("log_history") if state else None
    if not isinstance(history, list):
        return []
    points = []
    for record in history:
        if isinstance(record, dict) and isinstance(record.get("step"), int):
            points.append(
                {
                    "step": record["step"],
                    "loss": record.get("loss"),
                    "eval_loss": record.get("eval_loss"),
                    "epoch": record.get("epoch"),
                }
            )
    points.sort(key = lambda point: point["step"])
    return points


def _value_at_or_before(
    points: list[dict], key: str, step: int
) -> tuple[Optional[float], Optional[int]]:
    """The last finite ``key`` logged at or before ``step``: the trainer logs loss every few steps,
    so the save step itself often has none."""
    usable = [point for point in points if point["step"] > 0 and _finite(point.get(key))]
    index = bisect_right([point["step"] for point in usable], step)
    if index == 0:
        return None, None
    point = usable[index - 1]
    return float(point[key]), point["step"]


def _value_at(points: list[dict], key: str, step: int) -> Optional[float]:
    # Eval loss belongs to the weights it measured, so only an evaluation at the save step counts.
    for point in points:
        if point["step"] == step and _finite(point.get(key)):
            return float(point[key])
    return None


def checkpoint_metrics(entry: RunCheckpoint, metric_points: list[dict]) -> dict:
    """Training loss near the save, eval loss at it, and the epoch, from training_metrics first and
    the checkpoint's own trainer_state.json when the database has nothing for that step (a run from
    before metrics were stored, or the steps a resumed run inherited)."""
    result: dict[str, Any] = {
        "epoch": None,
        "train_loss": None,
        "train_loss_step": None,
        "eval_loss": None,
    }
    if entry.step is None:
        return result
    state = _read_trainer_state(entry.path)
    state_points = _state_points(state)
    loss, loss_step = _value_at_or_before(metric_points, "loss", entry.step)
    if loss is None:
        loss, loss_step = _value_at_or_before(state_points, "loss", entry.step)
    eval_loss = _value_at(metric_points, "eval_loss", entry.step)
    if eval_loss is None:
        eval_loss = _value_at(state_points, "eval_loss", entry.step)
    epoch = state.get("epoch") if state else None
    if not _finite(epoch):
        epoch, _ = _value_at_or_before(metric_points, "epoch", entry.step)
    result.update(
        epoch = float(epoch) if _finite(epoch) else None,
        train_loss = loss,
        train_loss_step = loss_step,
        eval_loss = eval_loss,
    )
    return result


def pick_best_checkpoint(rows: list[dict]) -> tuple[Optional[str], Optional[BestBasis]]:
    """Lowest eval loss when at least two checkpoints were evaluated at their save step, else lowest
    training loss. The basis is returned with the pick so the label can say which one it was. Ties
    go to the run's final save, then the earlier step."""
    for basis in ("eval_loss", "train_loss"):
        candidates = [row for row in rows if _finite(row.get(basis))]
        if len(candidates) >= 2:
            best = min(
                candidates,
                key = lambda row: (
                    row[basis],
                    0 if row.get("is_final") else 1,
                    row.get("step") if isinstance(row.get("step"), int) else 0,
                ),
            )
            return best["id"], basis
    return None, None


def _run_resume_blocker(run: dict) -> Optional[tuple[str, Optional[str]]]:
    """Why no checkpoint of this run can seed a resume, whichever one is picked."""
    from core.training.provenance import resource_provenance_resume_blocker
    from core.training.resume import _uses_s3_dataset, training_run_config

    if run.get("status") == "running":
        return "run_active", None
    if _uses_s3_dataset(run):
        return "s3_dataset", None
    try:
        reason = resource_provenance_resume_blocker(training_run_config(run))
    except Exception:
        logger.debug("Resume provenance check failed for run %s", run.get("id"), exc_info = True)
        return "provenance", None
    return ("provenance", reason) if reason else None


def _checkpoint_resume_blocker(run: dict, entry: RunCheckpoint) -> Optional[str]:
    from core.training.resume import is_resume_checkpoint_valid

    if entry.step is None or not is_resume_checkpoint_valid(entry.path, entry.step):
        return "no_trainer_state"
    total_steps = run.get("total_steps")
    if isinstance(total_steps, int) and total_steps > 0 and entry.step >= total_steps:
        return "finished"
    return None


def _same_path(first: str, second: Path) -> bool:
    try:
        return os.path.normcase(os.path.realpath(first)) == os.path.normcase(
            os.path.realpath(second)
        )
    except (OSError, ValueError):
        return False


def describe_run_checkpoints(run: dict, metric_points: list[dict]) -> dict:
    """The listing: one row per checkpoint with its metrics, size and what resuming from it would do."""
    from core.training.resume import can_resume_run, get_resume_checkpoint_path

    run_dir = run_output_dir(run)
    if run_dir is None:
        return {
            "run_name": None,
            "checkpoints": [],
            "best_checkpoint_id": None,
            "best_basis": None,
            "total_size_bytes": None,
        }
    entries = list_run_checkpoints(run, run_dir)
    in_place_path: Optional[str] = None
    if run.get("status") != "running" and can_resume_run(run):
        in_place_path = get_resume_checkpoint_path(str(run_dir))
    run_blocker = _run_resume_blocker(run) if entries else None
    steps = [entry.step for entry in entries]
    newest_step = None if None in steps or not steps else max(steps)

    rows = []
    for entry in entries:
        resume_mode: Optional[ResumeMode] = None
        blocked_code: Optional[str] = None
        blocked_reason: Optional[str] = None
        if (
            in_place_path
            and _same_path(in_place_path, entry.path)
            and newest_step is not None
            and entry.step == newest_step
        ):
            # Nothing in the folder is newer than this state, so continuing in place cannot save over
            # a later checkpoint or final adapter. Anything else goes through a fork.
            resume_mode = "in_place"
        elif run_blocker is not None:
            blocked_code, blocked_reason = run_blocker
        else:
            blocked_code = _checkpoint_resume_blocker(run, entry)
            if blocked_code is None:
                resume_mode = "fork"
        rows.append(
            {
                "id": entry.id,
                "step": entry.step,
                "is_final": entry.is_final,
                "is_adapter": (entry.path / "adapter_config.json").is_file()
                or (entry.path / "adapters.safetensors").is_file(),
                "path": str(entry.path),
                "saved_at": checkpoint_saved_at(entry),
                "size_bytes": checkpoint_size_bytes(entry),
                "resume_mode": resume_mode,
                "resume_blocked_code": blocked_code,
                "resume_blocked_reason": blocked_reason,
                **checkpoint_metrics(entry, metric_points),
            }
        )
    best_id, best_basis = pick_best_checkpoint(rows)
    return {
        "run_name": run_dir.name,
        "checkpoints": rows,
        "best_checkpoint_id": best_id,
        "best_basis": best_basis,
        "total_size_bytes": _tree_bytes(run_dir),
    }


def _active_training_dir() -> Optional[str]:
    from core.training import get_training_backend

    return get_training_backend().active_output_dir()


def _training_active() -> bool:
    from core.training import get_training_backend

    return bool(get_training_backend().is_training_active())


def _overlaps(path: Path, other: Optional[str]) -> bool:
    if not other:
        return False
    try:
        mine = Path(os.path.realpath(path))
        theirs = Path(os.path.realpath(other))
    except (OSError, ValueError):
        return False
    return mine == theirs or mine in theirs.parents or theirs in mine.parents


def _export_holds(path: Path) -> bool:
    try:
        from core.export import get_export_backend

        loaded = get_export_backend().current_checkpoint
    except Exception:
        return False
    return bool(loaded) and _overlaps(path, loaded)


def _stage_for_delete(run_dir: Path, entry: RunCheckpoint) -> tuple[Path, list[tuple[Path, Path]]]:
    """Move the checkpoint out of the way under a hidden name, reversibly: the bytes only go once
    the move has succeeded, and a failed purge can put everything back. Returns the staging folder
    and the (original, staged) pairs for a final save, whose files are moved one by one."""
    token = uuid.uuid4().hex
    if not entry.is_final:
        staged = run_dir / f".{entry.path.name}.deleting-{token}"
        entry.path.rename(staged)
        return staged, []
    staged = run_dir / f".final.deleting-{token}"
    staged.mkdir()
    moved: list[tuple[Path, Path]] = []
    try:
        for file in _top_level_files(run_dir):
            # The row-bound marker is training state for the checkpoints that stay.
            if file.name == _ROW_BOUND_MARKER:
                continue
            target = staged / file.name
            os.replace(file, target)
            moved.append((file, target))
    except OSError:
        _restore_final(staged, moved)
        raise
    return staged, moved


def _restore_final(staged: Path, moved: list[tuple[Path, Path]]) -> bool:
    restored = True
    for original, target in reversed(moved):
        try:
            os.replace(target, original)
        except OSError:
            restored = False
            logger.exception("Could not restore %s from %s", original, target)
    if restored:
        try:
            staged.rmdir()
        except OSError:
            pass
    return restored


def delete_run_checkpoint(run: dict, checkpoint_id: str, *, confirm_final: bool) -> dict:
    from core.training.lifecycle import training_lifecycle_guard

    if run.get("status") == "running":
        raise CheckpointActionError(
            409, "run_active", "Stop the run before deleting its checkpoints."
        )
    run_dir, entry = find_run_checkpoint(run, checkpoint_id)
    if entry.is_final and not confirm_final:
        raise CheckpointActionError(
            409,
            "final_confirm_required",
            "This is the run's final adapter. Confirm deleting it explicitly.",
        )
    size = checkpoint_size_bytes(entry)
    with _mutation_lock:
        with training_lifecycle_guard():
            if _overlaps(run_dir, _active_training_dir()):
                raise CheckpointActionError(
                    409, "checkpoint_in_use", "A training run is writing to this folder."
                )
            if _export_holds(entry.path):
                raise CheckpointActionError(
                    409, "checkpoint_in_use", "Export has this checkpoint loaded."
                )
            try:
                staged, moved = _stage_for_delete(run_dir, entry)
            except OSError:
                logger.warning(
                    "Could not move checkpoint %s of run %s aside", entry.id, run.get("id"),
                    exc_info = True,
                )
                raise CheckpointActionError(
                    409,
                    "checkpoint_in_use",
                    "The checkpoint could not be moved. Another program may have it open.",
                )
        try:
            shutil.rmtree(staged)
        except OSError:
            logger.exception("Failed to purge checkpoint %s of run %s", entry.id, run.get("id"))
            if entry.is_final:
                _restore_final(staged, moved)
            else:
                try:
                    staged.rename(entry.path)
                except OSError:
                    logger.error("Checkpoint %s remains on disk at %s", entry.id, staged)
            raise CheckpointActionError(
                500, "checkpoint_delete_failed", "The checkpoint could not be deleted."
            )
    logger.info("Deleted checkpoint %s of run %s", entry.id, run.get("id"))
    return {"status": "deleted", "checkpoint_id": entry.id, "freed_bytes": size}


def _fork_dir_name(source_name: str, step: int, attempt: int) -> str:
    suffix = f"_from-step-{step}" + (f"-{attempt}" if attempt > 1 else "")
    budget = _MAX_DIR_NAME_BYTES - len(suffix.encode("utf-8"))
    base = source_name.lstrip(".")
    while len(base.encode("utf-8")) > budget:
        base = base[:-1]
    return f"{base}{suffix}"


def _create_fork_dir(run_dir: Path, step: int) -> Path:
    """A new sibling folder; mkdir without exist_ok is the guarantee nothing already there is reused."""
    for attempt in range(1, _FORK_NAME_ATTEMPTS + 1):
        candidate = run_dir.parent / _fork_dir_name(run_dir.name, step, attempt)
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise CheckpointActionError(
        409, "fork_name_exhausted", "Could not find a free folder name for the new run."
    )


def _skip_links(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if os.path.islink(os.path.join(directory, name))}


def _copy_checkpoint(entry: RunCheckpoint, staging: Path) -> None:
    if not entry.is_final:
        shutil.copytree(entry.path, staging, ignore = _skip_links, copy_function = shutil.copy2)
        return
    # A final save that kept its trainer state: its own files, not the checkpoint folders beside them.
    staging.mkdir()
    for file in _top_level_files(entry.path):
        if file.name != _ROW_BOUND_MARKER:
            shutil.copy2(file, staging / file.name)


def _new_run_id() -> str:
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def fork_run_from_checkpoint(run: dict, checkpoint_id: str) -> dict:
    """Seed a new stopped run from one checkpoint, in a new folder, for the normal resume path to
    continue. The source folder is only read: a resume there from an older checkpoint would save
    over (or rotate away) the checkpoints that came after it."""
    from core.training.lifecycle import training_lifecycle_guard
    from core.training.resume import (
        is_resume_checkpoint_valid,
        normalize_resume_output_dir,
        training_run_config,
    )
    from storage.studio_db import create_forked_run, get_run_metric_points

    run_dir, entry = find_run_checkpoint(run, checkpoint_id)
    blocker = _run_resume_blocker(run)
    if blocker is not None:
        code, reason = blocker
        raise CheckpointActionError(
            409, code, reason or "This run's resources no longer allow resuming it."
        )
    checkpoint_blocker = _checkpoint_resume_blocker(run, entry)
    if checkpoint_blocker == "no_trainer_state":
        raise CheckpointActionError(
            409,
            checkpoint_blocker,
            "This checkpoint has no saved optimizer state, so training cannot continue from it.",
        )
    if checkpoint_blocker == "finished":
        raise CheckpointActionError(
            409, checkpoint_blocker, "This checkpoint is already at the run's last step."
        )
    assert entry.step is not None
    step = entry.step

    with _mutation_lock:
        with training_lifecycle_guard():
            if _training_active() or _overlaps(run_dir, _active_training_dir()):
                raise CheckpointActionError(
                    409, "training_active", "Stop the current training run first."
                )
        needed = (checkpoint_size_bytes(entry) or 0)
        needed += max(_FORK_MARGIN_BYTES, needed // 20)
        try:
            free = shutil.disk_usage(run_dir.parent).free
        except OSError:
            free = None
        if free is not None and free < needed:
            raise CheckpointActionError(
                507,
                "insufficient_disk_space",
                f"Copying this checkpoint needs about {needed // (1024 * 1024)} MB free.",
            )

        fork_dir = _create_fork_dir(run_dir, step)
        try:
            staging = fork_dir / f".checkpoint-{step}.copying-{uuid.uuid4().hex}"
            _copy_checkpoint(entry, staging)
            target = fork_dir / f"checkpoint-{step}"
            staging.rename(target)
            if not is_resume_checkpoint_valid(target, step):
                raise CheckpointActionError(
                    500, "fork_copy_invalid", "The copied checkpoint did not validate."
                )
            marker = run_dir / _ROW_BOUND_MARKER
            if marker.is_file() and not marker.is_symlink():
                shutil.copy2(marker, fork_dir / _ROW_BOUND_MARKER)

            # Stored exactly as the start route normalizes a resume request, which looks the row up by it.
            stored_dir = normalize_resume_output_dir(str(fork_dir))
            config = {
                **training_run_config(run),
                "resume_from_checkpoint": None,
                "forked_from": {
                    "run_id": run.get("id"),
                    "checkpoint": entry.id,
                    "step": step,
                },
            }
            if "output_dir" in config:
                config["output_dir"] = stored_dir
            metric_points = get_run_metric_points(str(run.get("id")))
            new_id = _new_run_id()
            create_forked_run(
                new_id,
                source_run_id = str(run.get("id")),
                model_name = str(run.get("model_name") or ""),
                dataset_name = str(run.get("dataset_name") or ""),
                config_json = json.dumps(config),
                started_at = datetime.now(timezone.utc).isoformat(),
                total_steps = run.get("total_steps")
                if isinstance(run.get("total_steps"), int)
                else None,
                final_step = step,
                final_loss = checkpoint_metrics(entry, metric_points)["train_loss"],
                output_dir = stored_dir,
                display_name = run.get("display_name"),
            )
        except BaseException:
            # Only ever the folder this call just created.
            shutil.rmtree(fork_dir, ignore_errors = True)
            raise
    logger.info(
        "Forked run %s from %s of run %s into %s", new_id, entry.id, run.get("id"), fork_dir
    )
    return {"run_id": new_id, "output_dir_name": fork_dir.name, "step": step}
