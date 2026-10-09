# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The hardware check's GPU measurements, run as a short-lived child of the backend.

Never imported by the backend: creating a CUDA context pins a few hundred MiB of VRAM per card
for the life of the process, so the server starts this file with its own interpreter and lets
the context die with the child. Standard library and torch only, run with ``python -I``.

Protocol: one JSON object per stdout line.

* ``{"event": "devices", "devices": [...]}`` -- the cards torch sees, keyed by UUID.
* ``{"event": "load", "uuid": ...}`` -- a host-to-device copy loop is running on this card. The
  parent reads the PCIe link while it runs (an idle link downclocks, so its width at rest says
  nothing) and answers with any line on stdin; the loop stops on that line or after
  ``LOAD_HOLD_S``, whichever is first.
* ``{"event": "bandwidth", "uuid": ..., "h2d_gibs": ..., "d2h_gibs": ...}`` -- pinned copies.
* ``{"event": "pair", "a": uuid, "b": uuid, ...}`` -- peer access both ways and the measured
  device-to-device copy speed both ways.
* ``{"event": "device_error", "uuid": ..., "reason": ...}`` / ``{"event": "error", ...}``.
* ``{"event": "done"}`` last.

Arguments: ``--uuids GPU-a,GPU-b`` (the cards to measure, in the parent's order),
``--buffer-mib N`` and ``--budget-s S`` (time per timed copy loop).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

LOAD_HOLD_S = 3.0
WARMUP_COPIES = 2
MIN_TIMED_COPIES = 2
MAX_TIMED_COPIES = 64


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _lower_priority() -> None:
    # Windows gets BELOW_NORMAL from the parent's creation flags; elsewhere ask for a nicer slot.
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass


def _uuid_of(props) -> str:
    raw = str(getattr(props, "uuid", "") or "")
    if not raw:
        return ""
    return raw if raw.startswith(("GPU-", "MIG-")) else f"GPU-{raw}"


def _pci_of(props):
    bus = getattr(props, "pci_bus_id", None)
    if bus is None:
        return None
    domain = int(getattr(props, "pci_domain_id", 0) or 0)
    device = int(getattr(props, "pci_device_id", 0) or 0)
    return f"{domain:08X}:{int(bus):02X}:{device:02X}.0"


class _Go:
    """``go <uuid>`` lines on stdin release that card's load loop. A reader thread, because a
    pipe cannot be polled with a timeout on Windows; keyed by card, so a late answer for one
    card cannot cut the next card's hold short."""

    def __init__(self) -> None:
        self._released: set[str] = set()
        self._closed = False
        self._lock = threading.Lock()
        threading.Thread(target = self._read, name = "probe-stdin", daemon = True).start()

    def _read(self) -> None:
        try:
            for line in sys.stdin:
                parts = line.split()
                if len(parts) == 2 and parts[0] == "go":
                    with self._lock:
                        self._released.add(parts[1])
        except Exception:  # noqa: BLE001
            pass
        # A closed stdin means the parent is gone or done listening: never wait on it again.
        with self._lock:
            self._closed = True

    def released(self, uuid: str) -> bool:
        with self._lock:
            return self._closed or uuid in self._released


def _timed(copy, sync, budget_s: float, nbytes: int) -> float:
    """GiB/s over back-to-back synchronised copies, for about ``budget_s``."""
    started = time.perf_counter()
    copies = 0
    while copies < MIN_TIMED_COPIES or (
        copies < MAX_TIMED_COPIES and time.perf_counter() - started < budget_s
    ):
        copy()
        sync()
        copies += 1
    elapsed = max(time.perf_counter() - started, 1e-9)
    return copies * nbytes / elapsed / float(1 << 30)


def main(argv = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuids", default = "")
    parser.add_argument("--buffer-mib", type = int, default = 256)
    parser.add_argument("--budget-s", type = float, default = 0.5)
    args = parser.parse_args(argv)
    _lower_priority()

    wanted = [u.strip() for u in args.uuids.split(",") if u.strip()]
    nbytes = max(16, int(args.buffer_mib)) * (1 << 20)
    budget = max(0.05, float(args.budget_s))

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        emit({"event": "error", "reason": f"torch could not be imported: {type(exc).__name__}"})
        emit({"event": "done"})
        return 0
    try:
        available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        available = False
    if not available:
        emit({"event": "error", "reason": "CUDA is not available to torch"})
        emit({"event": "done"})
        return 0

    devices = []
    by_uuid: dict[str, int] = {}
    for ordinal in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(ordinal)
        except Exception:  # noqa: BLE001
            continue
        uuid = _uuid_of(props)
        devices.append(
            {"ordinal": ordinal, "uuid": uuid, "pci_bus_id": _pci_of(props), "name": props.name}
        )
        if uuid:
            by_uuid[uuid] = ordinal
    emit({"event": "devices", "devices": devices})

    targets = [u for u in wanted if u in by_uuid] if wanted else list(by_uuid)
    for uuid in wanted:
        if uuid not in by_uuid:
            emit({"event": "device_error", "uuid": uuid, "reason": "not visible to torch"})

    go = _Go()
    try:
        host = torch.empty(nbytes, dtype = torch.uint8, pin_memory = True)
    except Exception as exc:  # noqa: BLE001
        emit({"event": "error", "reason": f"pinned host memory unavailable: {type(exc).__name__}"})
        emit({"event": "done"})
        return 0

    buffers: dict[str, object] = {}
    for uuid in targets:
        ordinal = by_uuid[uuid]
        try:
            torch.cuda.set_device(ordinal)
            dev = torch.empty(nbytes, dtype = torch.uint8, device = f"cuda:{ordinal}")

            def sync(ordinal = ordinal) -> None:
                torch.cuda.synchronize(ordinal)

            def h2d(dev = dev) -> None:
                dev.copy_(host, non_blocking = True)

            def d2h(dev = dev) -> None:
                host.copy_(dev, non_blocking = True)

            for _ in range(WARMUP_COPIES):
                h2d()
                sync()
            # The link trains up under traffic; hold it there while the parent reads it.
            emit({"event": "load", "uuid": uuid})
            held = time.perf_counter()
            while time.perf_counter() - held < LOAD_HOLD_S and not go.released(uuid):
                h2d()
                sync()
            h2d_gibs = _timed(h2d, sync, budget, nbytes)
            d2h_gibs = _timed(d2h, sync, budget, nbytes)
            emit(
                {
                    "event": "bandwidth",
                    "uuid": uuid,
                    "h2d_gibs": round(h2d_gibs, 3),
                    "d2h_gibs": round(d2h_gibs, 3),
                    "buffer_mib": nbytes >> 20,
                }
            )
            buffers[uuid] = dev
        except Exception as exc:  # noqa: BLE001 -- one card must not cost the others
            emit({"event": "device_error", "uuid": uuid, "reason": f"{type(exc).__name__}: {exc}"[:200]})

    measured = [u for u in targets if u in buffers]
    for i, a in enumerate(measured):
        for b in measured[i + 1 :]:
            oa, ob = by_uuid[a], by_uuid[b]
            entry: dict = {"event": "pair", "a": a, "b": b}
            try:
                entry["peer_ab"] = bool(torch.cuda.can_device_access_peer(oa, ob))
                entry["peer_ba"] = bool(torch.cuda.can_device_access_peer(ob, oa))
            except Exception:  # noqa: BLE001
                entry["peer_ab"] = entry["peer_ba"] = None
            src, dst = buffers[a], buffers[b]

            def sync_both(oa = oa, ob = ob) -> None:
                torch.cuda.synchronize(oa)
                torch.cuda.synchronize(ob)

            try:
                entry["copy_ab_gibs"] = round(
                    _timed(lambda: dst.copy_(src, non_blocking = True), sync_both, budget, nbytes), 3
                )
                entry["copy_ba_gibs"] = round(
                    _timed(lambda: src.copy_(dst, non_blocking = True), sync_both, budget, nbytes), 3
                )
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
            emit(entry)

    buffers.clear()
    emit({"event": "done"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
