"""Background consolidation (LangMem-style reflection).

A periodic pass over the memory bank that does higher-level updating no
single write can do:

1. CLUSTER   greedy density clustering: a memory plus every memory within
             cosine ``threshold`` of it forms a same-topic cluster (only
             clusters with >= ``min_cluster`` members count; already-
             consolidated summaries are excluded to keep one abstraction level
             per pass).
2. SUMMARIZE members are distilled into one summary memory (LLM when
             configured, deterministic template otherwise — same contract).
3. SUPERSEDE each member is invalidated (``invalid_at``) and linked to the
             summary via ``superseded_by``; a CONSOLIDATE event lands in the
             audit history, so nothing is lost and the chain is replayable.

The summary inherits the strongest member's importance (plus a small bonus),
which lifts it into the core-memory block — repeatedly observed patterns get
promoted, exactly the "distillation" LangMem describes. Because superseded
members are gone from FTS and the vector scan, this also *shrinks* retrieval
space while keeping information reachable through the summary.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .embeddings import cosine, pack, unpack
from .types import Memory, MemoryType

CONSOLIDATION_SUMMARY_PROMPT = """You are the background memory reflection module of an AI agent.
The facts below are same-topic memories about one user. Distill them into ONE
short summary memory (third person, keeping every distinct detail that
matters; drop repetition). Write it in the language the facts mostly use;
1-2 sentences, up to 4 for a large cluster.

Output STRICT JSON: {{"summary": "..."}}

Facts:
{facts}"""


def _is_zh(text: str) -> bool:
    return (len(re.findall(r"[\u4e00-\u9fff]", text))
            > len(re.findall(r"[a-zA-Z]", text)))


def find_clusters(memories: List[Memory], vectors: Dict[str, List[float]],
                  threshold: float, min_cluster: int) -> List[List[Memory]]:
    """Greedy density clustering over memory vectors."""
    remaining = list(memories)
    clusters: List[List[Memory]] = []
    while remaining:
        seed = remaining[0]
        group = [m for m in remaining
                 if cosine(vectors[seed.id], vectors[m.id]) >= threshold]
        if len(group) >= min_cluster:
            clusters.append(group)
            remaining = [m for m in remaining if m not in group]
        else:
            remaining = remaining[1:]
    return clusters


def consolidate(engine, user_id: str = "default", agent_id: Optional[str] = None,
                run_id: Optional[str] = None, min_cluster: Optional[int] = None,
                threshold: Optional[float] = None) -> Dict[str, Any]:
    """Run one reflection pass; returns an auditable report."""
    cfg = engine.cfg
    min_cluster = min_cluster if min_cluster is not None else cfg.consolidation_min_cluster
    threshold = threshold if threshold is not None else cfg.consolidation_similarity

    memories = engine.store.all_valid(user_id, agent_id, run_id)
    # one abstraction level per pass: never re-merge summaries
    memories = [m for m in memories if not m.metadata.get("consolidated")]
    if len(memories) < min_cluster:
        return {"clusters": 0, "summaries": [], "members_absorbed": 0}

    vectors = {}
    for m in memories:
        vec = unpack(engine.store.get_embedding(m.id), cfg.embedding_dim)
        if vec is not None:
            vectors[m.id] = vec
    memories = [m for m in memories if m.id in vectors]

    report: Dict[str, Any] = {"clusters": 0, "summaries": [], "members_absorbed": 0}
    for group in find_clusters(memories, vectors, threshold, min_cluster):
        group = sorted(group, key=lambda m: m.created_at)
        texts = [m.text for m in group]
        summary_text: Optional[str] = None
        data = engine.llm.complete_json(
            "You are the background memory reflection module of an AI agent.",
            CONSOLIDATION_SUMMARY_PROMPT.format(
                facts=json.dumps(texts, ensure_ascii=False)),
        )
        if data and data.get("summary"):
            summary_text = str(data["summary"]).strip()
        if not summary_text:
            # deterministic fallback promised by the module contract: never
            # let one bad LLM response kill the whole consolidation pass
            summary_text = " | ".join(texts)[:500]

        entities = sorted({e for m in group for e in m.entities})
        summary = Memory(
            text=summary_text,
            memory_type=MemoryType.FACT,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            entities=entities[:8],
            metadata={
                "consolidated": True,
                "members": [m.id for m in group],
                "gate": "consolidated",
            },
            importance=min(0.95, max(m.importance for m in group) + 0.05),
            source="consolidation",
        )
        engine.store.insert(summary, pack(engine.embedder.embed(summary_text)))
        engine.vector_store.upsert(
            summary.id, engine.embedder.embed(summary_text),
            {"memory_id": summary.id, "user_id": user_id,
             "agent_id": agent_id, "run_id": run_id})
        for m in group:
            engine.store.supersede(m.id, summary.id)
            engine.vector_store.delete(m.id)
        report["clusters"] += 1
        report["members_absorbed"] += len(group)
        report["summaries"].append({"id": summary.id, "text": summary_text,
                                    "members": [m.id for m in group]})
    return report
