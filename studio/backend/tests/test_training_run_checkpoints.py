# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoints of a finished or stopped run: listing with sizes and metrics, the best pick, the
delete guard for the final adapter, and resuming from an older checkpoint without touching the
newer ones."""

import asyncio
import json
import pickle
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.training import provenance, run_checkpoints
from core.training.resume import can_resume_run
from routes import training_history
from storage import studio_db
from utils.paths import outputs_root

ROW_BOUND = "unsloth_row_bound.json"


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(run_checkpoints, "_active_training_dir", lambda: None)
    monkeypatch.setattr(run_checkpoints, "_training_active", lambda: False)
    monkeypatch.setattr(run_checkpoints, "_export_holds", lambda path: False)
    monkeypatch.setattr(provenance, "resource_provenance_resume_blocker", lambda config: None)
    monkeypatch.setattr(provenance, "resource_provenance_allows_resume", lambda config: True)
    root = outputs_root()
    root.mkdir(parents = True)
    return root


def _safetensors(path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    save_file({"base.lora_A.weight": np.zeros((4, 4), dtype = np.float32)}, str(path))


def _torch_zip(path: Path) -> None:
    # The shape resume.py validates: a zip holding a complete pickle at */data.pkl.
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"state": {}}, protocol = 2))


def _adapter(folder: Path) -> None:
    (folder / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/test-model", "r": 8}), encoding = "utf-8"
    )
    _safetensors(folder / "adapter_model.safetensors")


def _checkpoint(run_dir: Path, step: int, *, state: bool = True, log_history = None) -> Path:
    folder = run_dir / f"checkpoint-{step}"
    folder.mkdir(parents = True)
    _adapter(folder)
    if state:
        _torch_zip(folder / "optimizer.pt")
        _torch_zip(folder / "scheduler.pt")
        (folder / "trainer_state.json").write_text(
            json.dumps(
                {"global_step": step, "epoch": step / 20, "log_history": log_history or []}
            ),
            encoding = "utf-8",
        )
    return folder


def _folder_bytes(folder: Path) -> int:
    return sum(file.stat().st_size for file in folder.rglob("*") if file.is_file())


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(folder)): file.read_bytes()
        for file in sorted(folder.rglob("*"))
        if file.is_file()
    }


def _make_run(
    outputs: Path,
    *,
    run_id: str = "job_source",
    status: str = "stopped",
    steps = (10, 20, 30),
    total_steps: int = 40,
    final: bool = True,
    eval_losses: dict[int, float] | None = None,
) -> tuple[dict, Path]:
    run_dir = outputs / f"unsloth_test-model_{run_id}"
    run_dir.mkdir()
    for step in steps:
        _checkpoint(run_dir, step)
    if final:
        _adapter(run_dir)
        (run_dir / "tokenizer.json").write_text("{}", encoding = "utf-8")
    (run_dir / ROW_BOUND).write_text(json.dumps({"max_train_rows": 100}), encoding = "utf-8")
    output_dir = str(run_dir.resolve())
    started = datetime(2026, 1, 1, tzinfo = timezone.utc).isoformat()
    studio_db.create_run(
        run_id,
        "unsloth/test-model",
        "org/dataset",
        json.dumps({"hf_dataset": "org/dataset", "output_dir": output_dir}),
        started,
        total_steps,
        output_dir = output_dir,
    )
    last = max(steps) if steps else 0
    # Loss logged every 5 steps, falling until step 20 and then rising.
    losses = {step: 2.0 - step / 20 if step <= 20 else 1.0 + (step - 20) / 20 for step in range(5, last + 1, 5)}
    studio_db.insert_metrics_batch(
        run_id,
        [
            {
                "step": step,
                "loss": loss,
                "epoch": step / 20,
                "eval_loss": (eval_losses or {}).get(step),
            }
            for step, loss in losses.items()
        ],
    )
    studio_db.finish_run(
        run_id,
        status,
        started,
        last,
        losses.get(last),
        60.0,
        output_dir = output_dir,
    )
    return studio_db.get_run(run_id), run_dir


def _listing(run: dict) -> dict:
    return run_checkpoints.describe_run_checkpoints(run, studio_db.get_run_metric_points(run["id"]))


def _row(listing: dict, checkpoint_id: str) -> dict:
    return next(row for row in listing["checkpoints"] if row["id"] == checkpoint_id)


def test_listing_reports_steps_sizes_and_nearby_loss(outputs):
    run, run_dir = _make_run(outputs)

    listing = _listing(run)

    assert [row["id"] for row in listing["checkpoints"]] == [
        "final",
        "checkpoint-30",
        "checkpoint-20",
        "checkpoint-10",
    ]
    assert listing["run_name"] == run_dir.name
    middle = _row(listing, "checkpoint-20")
    assert middle["step"] == 20
    assert middle["epoch"] == pytest.approx(1.0)
    assert middle["train_loss"] == pytest.approx(1.0)
    assert middle["train_loss_step"] == 20
    assert middle["size_bytes"] == _folder_bytes(run_dir / "checkpoint-20")
    assert middle["is_adapter"] is True
    assert middle["saved_at"]
    # The final save is the run folder's own files, not its checkpoint folders too.
    final = _row(listing, "final")
    assert final["is_final"] is True
    assert final["step"] == 30
    assert final["size_bytes"] == sum(
        file.stat().st_size for file in run_dir.iterdir() if file.is_file()
    )
    assert listing["total_size_bytes"] == _folder_bytes(run_dir)


def test_loss_comes_from_the_last_log_at_or_before_the_save(outputs):
    run, run_dir = _make_run(outputs, steps = (10,))
    _checkpoint(run_dir, 12)

    row = _row(_listing(run), "checkpoint-12")

    assert row["train_loss_step"] == 10
    assert row["train_loss"] == pytest.approx(1.5)


def test_loss_falls_back_to_the_checkpoint_trainer_state(outputs):
    run, run_dir = _make_run(outputs, steps = ())
    _checkpoint(
        run_dir,
        50,
        log_history = [{"step": 45, "loss": 0.75}, {"step": 50, "eval_loss": 0.9}],
    )

    row = _row(_listing(run), "checkpoint-50")

    assert row["train_loss"] == pytest.approx(0.75)
    assert row["train_loss_step"] == 45
    assert row["eval_loss"] == pytest.approx(0.9)


def test_best_checkpoint_prefers_eval_loss_at_the_save_step(outputs):
    run, _ = _make_run(outputs, eval_losses = {10: 1.4, 20: 1.1, 30: 1.3})

    listing = _listing(run)

    assert listing["best_basis"] == "eval_loss"
    assert listing["best_checkpoint_id"] == "checkpoint-20"


def test_best_checkpoint_falls_back_to_lowest_training_loss(outputs):
    run, _ = _make_run(outputs)

    listing = _listing(run)

    # Training loss bottoms out at step 20 (1.0); 30 rose back to 1.5.
    assert listing["best_basis"] == "train_loss"
    assert listing["best_checkpoint_id"] == "checkpoint-20"


def test_pick_best_checkpoint_rules():
    rows = [
        {"id": "final", "is_final": True, "step": 30, "train_loss": 0.5, "eval_loss": None},
        {"id": "checkpoint-30", "is_final": False, "step": 30, "train_loss": 0.5, "eval_loss": 0.8},
        {"id": "checkpoint-20", "is_final": False, "step": 20, "train_loss": 0.7, "eval_loss": None},
    ]
    # One evaluated checkpoint is no comparison: fall back to training loss, and a tie goes to the final save.
    assert run_checkpoints.pick_best_checkpoint(rows) == ("final", "train_loss")
    assert run_checkpoints.pick_best_checkpoint([rows[2]]) == (None, None)
    rows[2]["eval_loss"] = float("nan")
    assert run_checkpoints.pick_best_checkpoint(rows) == ("final", "train_loss")
    rows[2]["eval_loss"] = 0.6
    assert run_checkpoints.pick_best_checkpoint(rows) == ("checkpoint-20", "eval_loss")


def test_resume_modes_never_offer_in_place_over_newer_saves(outputs):
    # A stopped run whose final adapter is no newer than its last checkpoint resumes in place from it.
    run, run_dir = _make_run(outputs, final = False)
    listing = _listing(run)
    assert _row(listing, "checkpoint-30")["resume_mode"] == "in_place"
    assert _row(listing, "checkpoint-20")["resume_mode"] == "fork"
    assert _row(listing, "checkpoint-10")["resume_mode"] == "fork"

    # A completed run's last step has nothing left to train; its older checkpoints fork.
    done, _ = _make_run(outputs, run_id = "job_done", status = "completed", total_steps = 30)
    listing = _listing(done)
    assert _row(listing, "checkpoint-30")["resume_blocked_code"] == "finished"
    assert _row(listing, "final")["resume_blocked_code"] == "no_trainer_state"
    assert _row(listing, "checkpoint-20")["resume_mode"] == "fork"


def test_checkpoint_without_trainer_state_cannot_seed_a_resume(outputs):
    run, run_dir = _make_run(outputs, steps = (10,))
    _checkpoint(run_dir, 20, state = False)

    row = _row(_listing(run), "checkpoint-20")

    assert row["resume_mode"] is None
    assert row["resume_blocked_code"] == "no_trainer_state"
    with pytest.raises(run_checkpoints.CheckpointActionError) as refused:
        run_checkpoints.fork_run_from_checkpoint(run, "checkpoint-20")
    assert refused.value.code == "no_trainer_state"


def test_delete_numbered_checkpoint_leaves_the_rest(outputs):
    run, run_dir = _make_run(outputs)
    before = _snapshot(run_dir / "checkpoint-30")

    result = run_checkpoints.delete_run_checkpoint(run, "checkpoint-20", confirm_final = False)

    assert result["checkpoint_id"] == "checkpoint-20"
    assert result["freed_bytes"] > 0
    assert not (run_dir / "checkpoint-20").exists()
    assert (run_dir / "checkpoint-10").is_dir()
    assert _snapshot(run_dir / "checkpoint-30") == before
    # Nothing staged is left behind under a hidden name.
    assert not [path for path in run_dir.iterdir() if path.name.startswith(".")]


def test_final_adapter_needs_the_stronger_confirm(outputs):
    run, run_dir = _make_run(outputs)

    with pytest.raises(run_checkpoints.CheckpointActionError) as refused:
        run_checkpoints.delete_run_checkpoint(run, "final", confirm_final = False)
    assert refused.value.status_code == 409
    assert refused.value.code == "final_confirm_required"
    assert (run_dir / "adapter_model.safetensors").is_file()

    run_checkpoints.delete_run_checkpoint(run, "final", confirm_final = True)

    assert not (run_dir / "adapter_model.safetensors").exists()
    assert not (run_dir / "adapter_config.json").exists()
    # Checkpoints, and the row bound they resume with, stay.
    assert (run_dir / ROW_BOUND).is_file()
    assert {path.name for path in run_dir.iterdir() if path.is_dir()} == {
        "checkpoint-10",
        "checkpoint-20",
        "checkpoint-30",
    }


def test_delete_refuses_a_running_run_and_an_active_folder(outputs, monkeypatch):
    run, run_dir = _make_run(outputs)

    with pytest.raises(run_checkpoints.CheckpointActionError) as running:
        run_checkpoints.delete_run_checkpoint(
            {**run, "status": "running"}, "checkpoint-10", confirm_final = False
        )
    assert running.value.code == "run_active"

    monkeypatch.setattr(run_checkpoints, "_active_training_dir", lambda: str(run_dir))
    with pytest.raises(run_checkpoints.CheckpointActionError) as active:
        run_checkpoints.delete_run_checkpoint(run, "checkpoint-10", confirm_final = False)
    assert active.value.code == "checkpoint_in_use"
    assert (run_dir / "checkpoint-10").is_dir()


def test_ids_are_matched_not_joined(outputs):
    run, run_dir = _make_run(outputs)
    outside = outputs / "checkpoint-99"
    outside.mkdir()
    _adapter(outside)

    for checkpoint_id in ("../checkpoint-99", "checkpoint-99", "checkpoint-020", ".", ""):
        with pytest.raises(run_checkpoints.CheckpointActionError) as missing:
            run_checkpoints.delete_run_checkpoint(run, checkpoint_id, confirm_final = True)
        assert missing.value.status_code == 404
    assert outside.is_dir()


def test_fork_resume_copies_into_a_new_folder_and_never_overwrites(outputs):
    run, run_dir = _make_run(outputs)
    source_before = _snapshot(run_dir)
    # A folder already holding the first name the fork would pick must not be reused.
    squatter = outputs / f"{run_dir.name}_from-step-10"
    squatter.mkdir()
    (squatter / "keep.txt").write_text("mine", encoding = "utf-8")

    result = run_checkpoints.fork_run_from_checkpoint(run, "checkpoint-10")

    assert _snapshot(run_dir) == source_before
    assert (squatter / "keep.txt").read_text(encoding = "utf-8") == "mine"
    assert sorted(path.name for path in squatter.iterdir()) == ["keep.txt"]
    fork_dir = outputs / result["output_dir_name"]
    assert fork_dir.name == f"{run_dir.name}_from-step-10-2"
    assert sorted(path.name for path in fork_dir.iterdir()) == sorted([ROW_BOUND, "checkpoint-10"])
    assert _snapshot(fork_dir / "checkpoint-10") == _snapshot(run_dir / "checkpoint-10")

    seed = studio_db.get_run(result["run_id"])
    assert seed["status"] == "stopped"
    assert seed["final_step"] == 10
    assert seed["total_steps"] == 40
    assert json.loads(seed["config_json"])["forked_from"] == {
        "run_id": run["id"],
        "checkpoint": "checkpoint-10",
        "step": 10,
    }
    # The ordinary resume path finds the fork by its folder and continues from the copy.
    assert studio_db.get_resumable_run_by_output_dir(seed["output_dir"])["id"] == seed["id"]
    assert can_resume_run(seed) is True
    assert studio_db.get_run_metric_points(seed["id"])[-1]["step"] == 10
    listing = _listing(seed)
    assert [row["id"] for row in listing["checkpoints"]] == ["checkpoint-10"]
    assert listing["checkpoints"][0]["resume_mode"] == "in_place"


def test_fork_refused_while_training_and_leaves_nothing(outputs, monkeypatch):
    run, run_dir = _make_run(outputs)
    monkeypatch.setattr(run_checkpoints, "_training_active", lambda: True)
    before = sorted(path.name for path in outputs.iterdir())

    with pytest.raises(run_checkpoints.CheckpointActionError) as refused:
        run_checkpoints.fork_run_from_checkpoint(run, "checkpoint-10")

    assert refused.value.code == "training_active"
    assert sorted(path.name for path in outputs.iterdir()) == before


def test_fork_refused_without_disk_space(outputs, monkeypatch):
    run, _ = _make_run(outputs)

    class Usage:
        free = 1024

    monkeypatch.setattr(run_checkpoints.shutil, "disk_usage", lambda path: Usage())
    before = sorted(path.name for path in outputs.iterdir())

    with pytest.raises(run_checkpoints.CheckpointActionError) as refused:
        run_checkpoints.fork_run_from_checkpoint(run, "checkpoint-10")

    assert refused.value.status_code == 507
    assert sorted(path.name for path in outputs.iterdir()) == before


def test_routes_map_refusals_and_reject_path_ids(outputs):
    run, run_dir = _make_run(outputs)
    app = FastAPI()
    app.include_router(training_history.router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    client = TestClient(app)

    listed = client.get(f"/api/train/runs/{run['id']}/checkpoints")
    assert listed.status_code == 200
    assert listed.json()["checkpoints"][0]["id"] == "final"
    assert client.get("/api/train/runs/job_missing/checkpoints").status_code == 404

    refused = client.delete(f"/api/train/runs/{run['id']}/checkpoints/final")
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "final_confirm_required"
    for bad_id in ("..%2Fescape", "checkpoint-1x", "Final"):
        response = client.delete(f"/api/train/runs/{run['id']}/checkpoints/{bad_id}")
        assert response.status_code in (404, 422)
    assert (run_dir / "adapter_model.safetensors").is_file()

    deleted = client.delete(f"/api/train/runs/{run['id']}/checkpoints/checkpoint-10")
    assert deleted.status_code == 200
    assert deleted.json()["checkpoint_id"] == "checkpoint-10"

    forked = client.post(f"/api/train/runs/{run['id']}/checkpoints/checkpoint-20/fork")
    assert forked.status_code == 200
    assert studio_db.get_run(forked.json()["run_id"])["final_step"] == 20


def test_route_reports_unknown_run(outputs):
    with pytest.raises(HTTPException) as missing:
        asyncio.run(
            training_history.fork_training_run_checkpoint(
                "job_missing", "checkpoint-10", current_subject = "owner"
            )
        )
    assert missing.value.status_code == 404
