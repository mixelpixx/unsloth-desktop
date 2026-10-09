# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which card automatic placement prefers once the hardware check has measured the links.

Pure: no I/O and no Studio imports, so the loader (``core.inference.llama_cpp``), the load
guardrail (``core.inference.load_verdict``) and the training auto-selector all decide with the
same rule and cannot drift apart. The bandwidth map comes from
``utils.hardware.hardware_check.active_link_preference``, which returns None whenever the option
is off or the check has no current result, and every caller keeps its old behaviour on None.

The rule never trades safety for speed. A card is moved away from only when it is measurably
slower (below ``TIE_FRACTION`` of the alternative's host bandwidth: two x16 cards measuring 24.1
and 24.9 GiB/s are a tie and keep the old order), and only onto a card that holds the load with
the same margin the guardrail calls comfortable (``comfort_margin_mib``, the mirror of
``load_verdict.estimation_band_bytes``). A card that only fits tightly is never preferred.
"""

from __future__ import annotations

from typing import Collection, Mapping, Optional, Sequence

# Below this share of the alternative's host bandwidth a card counts as slower; above it the
# two are a tie and the existing order decides.
TIE_FRACTION = 0.85

# Mirrors load_verdict._BAND_MIN_BYTES / _BAND_FRACTION (estimation_band_bytes), in MiB.
BAND_MIN_MIB = 512.0
BAND_FRACTION = 0.05

# Auto context: the planner sizes the window to the card, so a fast card with a little less room
# than the roomiest one costs that much KV cache and nothing else. Up to this much is taken.
AUTO_CONTEXT_ROOM_TOLERANCE_MIB = 1024.0


def comfort_margin_mib(need_mib: float) -> float:
    """The headroom a preferred card must have beyond ``need_mib``."""
    return max(BAND_MIN_MIB, BAND_FRACTION * max(0.0, float(need_mib)))


def clearly_slower(slow_gibs: Optional[float], fast_gibs: Optional[float]) -> bool:
    """Whether ``slow_gibs`` is measurably below ``fast_gibs`` (both measured)."""
    if slow_gibs is None or fast_gibs is None or fast_gibs <= 0:
        return False
    return float(slow_gibs) < TIE_FRACTION * float(fast_gibs)


def faster_card(
    default: int,
    candidates: Sequence[tuple[int, float]],
    bandwidth: Optional[Mapping[int, float]],
    *,
    need_mib: float,
    shared: Collection[int] = (),
    floor_need_mib: Optional[float] = None,
    room_tolerance_mib: float = 0.0,
) -> Optional[int]:
    """A card to place a single-card load on instead of ``default``, or None to keep ``default``.

    ``candidates``: ``(index, usable MiB)`` per card in the caller's existing preference order
    (the order breaks ties). ``need_mib``: the footprint at the context the load asks for.

    An alternative qualifies when ``default`` is clearly slower than it and it holds
    ``need_mib`` comfortably. With ``floor_need_mib`` (Auto context, where the planner sizes the
    window to the card) it also qualifies holding the floor comfortably while having at most
    ``room_tolerance_mib`` less room than ``default``: the window then shrinks by at most that.

    ``shared``: cards another model already runs on. A default that is not shared is never
    traded for one that is: sharing a card splits its compute between the two models.
    """
    if not bandwidth:
        return None
    default_bw = bandwidth.get(default)
    if default_bw is None:
        # Never measured (skipped for lack of free memory): no evidence that it is slow.
        return None
    room = {int(idx): float(usable) for idx, usable in candidates}
    default_room = room.get(default)
    default_shared = default in shared
    qualified: list[tuple[int, float]] = []
    for idx, usable in candidates:
        idx = int(idx)
        if idx == default or (idx in shared and not default_shared):
            continue
        bw = bandwidth.get(idx)
        if not clearly_slower(default_bw, bw):
            continue
        holds_all = usable >= need_mib + comfort_margin_mib(need_mib)
        holds_window = (
            floor_need_mib is not None
            and default_room is not None
            and usable >= default_room - max(0.0, room_tolerance_mib)
            and usable >= floor_need_mib + comfort_margin_mib(floor_need_mib)
        )
        if holds_all or holds_window:
            qualified.append((idx, float(bw)))
    if not qualified:
        return None
    best = max(bw for _idx, bw in qualified)
    # Ties among the alternatives keep the caller's order.
    for idx, bw in qualified:
        if not clearly_slower(bw, best):
            return idx
    return None


def fast_first_order(ids: Sequence[int], bandwidth: Optional[Mapping[int, float]]) -> list[int]:
    """``ids`` with the fastest-linked card first when the current first card is clearly slower.

    The rest keep their order. Unchanged when nothing was measured or the first card is as fast
    as any other: device 0 is where llama.cpp runs the work it offloads from system RAM, so it is
    the one card whose link the whole load leans on.
    """
    order = [int(i) for i in ids]
    if len(order) < 2 or not bandwidth:
        return order
    measured = [(i, bandwidth[i]) for i in order if bandwidth.get(i) is not None]
    if not measured:
        return order
    best_idx, best_bw = max(measured, key = lambda item: item[1])
    if best_idx == order[0] or not clearly_slower(bandwidth.get(order[0]), best_bw):
        return order
    return [best_idx, *[i for i in order if i != best_idx]]
