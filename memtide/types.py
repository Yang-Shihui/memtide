"""Core data types for Memtide.

A memory is an *atomic fact* (Mem0-style), not a raw message. Every mutation
is audited (Zep-style bi-temporal: ``valid_at`` / ``invalid_at``), and every
retrieval *reinforces* the memory (spaced-repetition style), see decay.py.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:20]


class MemoryType:
    """Coarse category of an atomic fact."""

    FACT = "fact"                # general world/user facts
    PREFERENCE = "preference"    # likes / dislikes / habits
    EPISODIC = "episodic"        # event-ish memories tied to a moment
    PROCEDURAL = "procedural"    # how-to / workflow knowledge

    ALL = (FACT, PREFERENCE, EPISODIC, PROCEDURAL)


class Event:
    """Audit-log event kinds (memory_history.event)."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    ACCESS = "ACCESS"
    CONSOLIDATE = "CONSOLIDATE"   # merged into a background reflection summary


@dataclass
class Memory:
    """One atomic, retrievable fact."""

    text: str
    id: str = field(default_factory=new_id)
    memory_type: str = MemoryType.FACT
    user_id: str = "default"
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5            # 0..1 intrinsic weight (set by extractor)
    access_count: int = 0              # retrieval reinforcement counter
    last_accessed: Optional[str] = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    valid_at: str = field(default_factory=utcnow)   # fact became true at
    invalid_at: Optional[str] = None                # fact stopped being true (None = still valid)
    superseded_by: Optional[str] = None             # id of the memory that replaced/merged this one
    source: str = "conversation"                    # conversation | manual | consolidation
    attachments: List[Dict[str, Any]] = field(default_factory=list)  # media assets (multimodal.py)

    # ---- serialization -------------------------------------------------
    def to_row(self, embedding: Optional[bytes]) -> Dict[str, Any]:
        import json

        return {
            "attachments": json.dumps(self.attachments, ensure_ascii=False),
            "id": self.id,
            "text": self.text,
            "memory_type": self.memory_type,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "entities": json.dumps(self.entities, ensure_ascii=False),
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "superseded_by": self.superseded_by,
            "source": self.source,
            "embedding": embedding,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Memory":
        import json

        return cls(
            id=row["id"],
            text=row["text"],
            memory_type=row["memory_type"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            entities=json.loads(row["entities"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            valid_at=row["valid_at"],
            invalid_at=row["invalid_at"],
            superseded_by=row["superseded_by"] if "superseded_by" in row.keys() else None,
            source=row["source"],
            attachments=(json.loads(row["attachments"] or "[]")
                         if "attachments" in row.keys() else []),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "text": self.text,
            "memory_type": self.memory_type,
            "user_id": self.user_id,
            "entities": self.entities,
            "metadata": self.metadata,
            "importance": round(self.importance, 3),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "source": self.source,
        }
        if self.superseded_by:
            d["superseded_by"] = self.superseded_by
        if self.attachments:
            d["attachments"] = self.attachments
        if self.agent_id is not None:
            d["agent_id"] = self.agent_id
        if self.run_id is not None:
            d["run_id"] = self.run_id
        return d


@dataclass
class SearchResult:
    """A memory plus the scores that explain *why* it was retrieved."""

    memory: Memory
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)  # semantic/bm25/entity/recency

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.memory.id,
            "memory": self.memory.text,
            "memory_type": self.memory.memory_type,
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "user_id": self.memory.user_id,
            "created_at": self.memory.created_at,
        }
        if self.memory.attachments:
            d["attachments"] = self.memory.attachments
        return d


@dataclass
class ExtractedFact:
    """Intermediate artifact of the extraction stage.

    ``slot`` tags volatile attributes (name/location/role/...): a new fact
    for the same slot supersedes the old one during consolidation.
    """

    text: str
    memory_type: str = MemoryType.FACT
    importance: float = 0.5
    entities: List[str] = field(default_factory=list)
    slot: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)  # media provenance


@dataclass
class AddResult:
    """What happened during one ``engine.add`` call (Mem0-style operations)."""

    added: List[str] = field(default_factory=list)      # new memory ids
    updated: List[str] = field(default_factory=list)    # memory ids whose text was replaced
    deleted: List[str] = field(default_factory=list)    # invalidated memory ids
    noop: int = 0                                       # duplicates skipped
    facts: List[str] = field(default_factory=list)      # extracted facts (pre-gate)
    rejected: List[str] = field(default_factory=list)   # facts gated out (fully predicted)
    gate: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # fact -> GateDecision
    attachments: List[Dict[str, Any]] = field(default_factory=list)  # media stored this call

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "added": self.added,
            "updated": self.updated,
            "deleted": self.deleted,
            "noop": self.noop,
            "facts": self.facts,
            "rejected": self.rejected,
            "gate": self.gate,
        }
        if self.attachments:
            d["attachments"] = self.attachments
        return d
