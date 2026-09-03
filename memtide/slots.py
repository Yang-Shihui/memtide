"""Open slot normalisation (B-batch).

``slot`` is an open hint, not a closed enum: the LLM may invent new slots
(e.g. spouse/pet). This module收敛同物异名 (city/住址 -> location) via a
small alias table plus syntactic normalisation. Unknown slots pass through;
illegal values become None. Memory systems must never mis-overwrite:
unmatched slots simply mean "no hint", the LLM semantic decision still runs.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# Suggested core slots (kept as examples for prompts, not a whitelist).
SUGGESTED_SLOTS = {"name", "location", "employer", "role", "age", "stack", "plan"}

# Multi-value slots that must never force UPDATE: preferences, routines.
# The extractor prompt says multi-value facts use null; the fake test server
# historically emits these, so normalise them to None for safety.
MULTI_VALUE_SLOTS = frozenset({"like", "routine_weekly", "routine_daily"})
SLOT_ALIASES: Dict[str, str] = {
    "city": "location",
    "town": "location",
    "address": "location",
    "住址": "location",
    "居住地": "location",
    "现居地": "location",
    "job": "role",
    "title": "role",
    "职位": "role",
    "职业": "role",
    "company": "employer",
    "公司": "employer",
    "单位": "employer",
    "tech-stack": "stack",
    "技术栈": "stack",
    "goal": "plan",
    "目标": "plan",
    "pet-name": "pet",
    "宠物名": "pet",
}

_SLOT_RE = re.compile(r"^[a-z0-9_\u4e00-\u9fa5-]{1,20}$")


def canonicalize_slot(raw: Optional[str], extra_aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Normalise a raw slot string to its canonical form (or None)."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if not s:
        return None
    aliases = dict(SLOT_ALIASES)
    if extra_aliases:
        for k, v in extra_aliases.items():
            if isinstance(k, str) and isinstance(v, str):
                aliases[k.strip().lower()] = v.strip().lower()
    s = aliases.get(s, s)
    if s in MULTI_VALUE_SLOTS:
        return None
    if not _SLOT_RE.match(s):
        return None
    return s


def same_slot(a: Optional[str], b: Optional[str],
              extra_aliases: Optional[Dict[str, str]] = None) -> bool:
    """Canonical equality used by gate/candidate paths."""
    ca, cb = canonicalize_slot(a, extra_aliases), canonicalize_slot(b, extra_aliases)
    return ca is not None and ca == cb
