"""Forgetting & reinforcement (cognitive-science flavored).

Retention follows a Ebbinghaus-style exponential decay whose half-life is
stretched by each successful retrieval (spacing effect):

    half_life_eff = half_life_days * min(1 + gain * ln(1 + access_count), max_mult)
    retention     = 0.5 ** (age_days / half_life_eff)

``age_days`` counts from when the fact became true (valid_at/created_at):
re-reading a memory reinforces it (longer half-life via access_count) but no
longer rejuvenates its age — otherwise hot memories would live forever.
Episodic memories fade faster than facts (episodic_mult < 1).

Memories below their floor are treated as forgotten: they stay in the
database (auditable, Zep-style) but stop surfacing in retrieval and the core
memory block, until a future query re-excites them.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from .types import Memory


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def age_days(mem: Memory, now: Optional[datetime] = None) -> float:
    # Age counts from when the fact became true — NOT last_accessed: a hit
    # reinforces (see effective_half_life) but must not reset the clock.
    now = now or datetime.now(timezone.utc)
    ref = _parse(mem.valid_at) or _parse(mem.created_at)
    if ref is None:
        return 0.0
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ref).total_seconds() / 86400.0)


def effective_half_life(mem: Memory, base_days: float, gain: float,
                        consolidation_mult: float = 1.0,
                        episodic_mult: float = 0.5,
                        max_mult: float = 4.0) -> float:
    """Half-life stretched by retrieval (spacing effect), capped at
    ``max_mult`` × base so hot memories can't become effectively immortal.
    Abstractions (source="consolidation") fade slower than episodes when a
    multiplier > 1 is configured — the distilled summary outlives its members.
    Episodic memories fade faster (episodic_mult < 1)."""
    stretch = min(1.0 + gain * math.log1p(mem.access_count), max_mult)
    hl = base_days * stretch
    if mem.memory_type == "episodic":
        hl *= episodic_mult
    if consolidation_mult != 1.0 and mem.source == "consolidation":
        hl *= consolidation_mult
    return hl


def retention(mem: Memory, base_days: float, gain: float, now: Optional[datetime] = None,
              consolidation_mult: float = 1.0,
              episodic_mult: float = 0.5,
              max_mult: float = 4.0) -> float:
    """Memory strength in [0, 1] — used for ranking and forgetting decisions."""
    a = age_days(mem, now)
    if a <= 0.0:
        return 1.0
    hl = effective_half_life(mem, base_days, gain, consolidation_mult,
                             episodic_mult, max_mult)
    return 0.5 ** (a / hl)


def is_forgotten(mem: Memory, base_days: float, gain: float, floor: float,
                 now: Optional[datetime] = None,
                 consolidation_mult: float = 1.0,
                 episodic_floor: Optional[float] = None,
                 episodic_mult: float = 0.5,
                 max_mult: float = 4.0) -> bool:
    if episodic_floor is not None and mem.memory_type == "episodic":
        floor = episodic_floor
    return retention(mem, base_days, gain, now, consolidation_mult,
                     episodic_mult, max_mult) < floor
