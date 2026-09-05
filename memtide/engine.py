"""MemoryEngine — the public API.

Write path (Mem0-style, two phases):
  1. EXTRACTION   the LLM turns raw conversation into atomic facts.
  2. CONSOLIDATION  each fact is compared against similar existing memories;
                  the LLM decides ADD / UPDATE / DELETE / NOOP.

Read path (hybrid): see retrieval.py. Every read reinforces (decay.py).
Plus Letta-style ``render_context`` for the always-injected core block and
Zep-style ``get_history`` temporal audit.
"""

from __future__ import annotations

import functools
import json
import threading
from typing import Any, Dict, List, Optional

from .config import MemoryConfig
from .decay import retention, is_forgotten
from .embeddings import CachedEmbedder, base_embedder, cosine, make_embedder, unpack
from .llm import (
    CONSOLIDATION_PROMPT,
    FACT_EXTRACTION_PROMPT,
    BaseLLM,
    make_llm,
)
from .retrieval import Retriever, _extract_query_entities
from .storage import make_storage
from .types import AddResult, ExtractedFact, Memory, MemoryType, SearchResult, new_id, utcnow


def _locked(fn):
    """Serialize engine-state access. The engine may be driven from several
    threads (REST server lock, add_background pool, auto-reflect scheduler);
    every public entry point takes this lock so concurrent workers remain safe."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


def _normalize_ts(timestamp: Optional[str]) -> Optional[str]:
    """Validate/normalize a caller-supplied event timestamp (ISO-8601).

    Every stored timestamp is normalized to a UTC-offset string: naive
    values are read as UTC, aware ones are converted. This keeps the
    lexicographic created_at ordering used by compact/consolidation
    consistent even when an import mixes offsets."""
    if timestamp is None:
        return None
    from datetime import datetime, timezone

    t = timestamp.strip()
    if not t:
        return None
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"invalid timestamp {timestamp!r}: need ISO-8601, e.g. 2024-06-01T10:30:00") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class MemoryEngine:
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.cfg = config or MemoryConfig()
        self.store = make_storage(self.cfg)
        # exact-text cache in front of the embedder: repeated texts (dedup
        # checks, gate scans, re-embeds) stop hitting the API
        self.embedder = CachedEmbedder(make_embedder(self.cfg))
        self.llm: BaseLLM = make_llm(self.cfg)
        from .vectorstore import make_vector_store

        self.vector_store = make_vector_store(self.cfg, self.store, self.embedder)
        self.retriever = Retriever(self.store, self.cfg, self.embedder,
                                   vector_store=self.vector_store, llm=self.llm)
        from .gating import PredictiveGate

        self.gate = PredictiveGate(self.cfg)
        self._reflect_stop = None
        self._reflect_thread = None
        self._reflect_status = {"consecutive_failures": 0, "last_error": None}
        self._pool = None
        self._lock = threading.RLock()
        # Qdrant is an index, PostgreSQL is the source of truth. On startup,
        # repair a stale/empty collection before serving requests; this is
        # what prevents a dimension migration or lost vector volume from
        # silently reducing retrieval coverage.
        self._sync_vector_index_on_init()

    def _sync_vector_index_on_init(self) -> None:
        if not hasattr(self.vector_store, "count"):
            return
        expected = len(self.store.all_rows_with_embeddings())
        actual = self.vector_store.count()
        if actual != expected:
            self.rebuild_index()

    # ------------------------------------------------------------------ write
    @_locked
    def add(self, messages: Any, user_id: str = "default", agent_id: Optional[str] = None,
            run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
            infer: bool = True, timestamp: Optional[str] = None) -> AddResult:
        """Store memories from a conversation turn (string or message list).

        ``timestamp``: optional ISO-8601 time OF THE CONVERSATION ITSELF — use
        it when importing historical dialogues so created_at/valid_at and the
        audit trail reflect when the exchange happened, not when it was
        imported. Omit to stamp with the current time.
        """
        result = AddResult()
        ts = _normalize_ts(timestamp)
        from . import multimodal

        conversation, media_attachments = multimodal.process_messages(messages, self.cfg)

        if infer:
            facts = self._extract_facts(conversation)
        else:
            facts = [ExtractedFact(text=conversation.strip(), memory_type=MemoryType.EPISODIC, importance=0.4)]

        # media provenance: every fact from this turn links the attachments;
        # if a described medium yielded no facts at all, keep its description
        # itself as one episodic memory so the asset is never silently lost
        for fact in facts:
            fact.attachments = media_attachments
        if media_attachments and not facts:
            for att in media_attachments:
                if att.get("description"):
                    facts.append(ExtractedFact(
                        text=att["description"], memory_type=MemoryType.EPISODIC,
                        importance=0.45, attachments=[att]))

        if metadata is None:
            metadata = {}
        facts = facts[: self.cfg.max_facts_per_turn]
        result.facts = [f.text for f in facts]
        result.attachments = media_attachments

        # one similarity scan per add() call (not per fact) — all facts share it
        corpus = self.store.all_embeddings(user_id, agent_id, run_id)
        prepared: List[tuple] = []  # (fact, metadata, similar)
        for fact in facts:
            fact_meta = dict(metadata)
            if fact.slot:
                fact_meta.setdefault("slot", fact.slot)
            if fact.attachments:
                fact_meta.setdefault("modality", fact.attachments[0]["kind"])
            similar = self._similar_memories(fact, user_id, agent_id, run_id, corpus=corpus)

            # predictive-coding gate: encode only prediction error
            if self.cfg.gate_enabled:
                decision = self.gate.evaluate(fact, similar)
                result.gate[fact.text] = decision.to_dict()
                if not decision.store:
                    result.rejected.append(fact.text)
                    if decision.max_similarity >= self.cfg.dedup_threshold:
                        result.noop += 1
                    continue
                fact_meta["gate"] = decision.reason
                fact_meta["surprise_bits"] = round(decision.surprise_bits, 3)
                if decision.importance_delta:
                    fact = ExtractedFact(
                        text=fact.text, memory_type=fact.memory_type,
                        importance=min(0.98, fact.importance + decision.importance_delta),
                        entities=fact.entities, slot=fact.slot,
                        attachments=fact.attachments,
                    )
            prepared.append((fact, fact_meta, similar))

        # conflict resolution: ONE batched LLM call on the real path
        decisions = self._decide_operations(prepared)

        for fact, fact_meta, similar in prepared:
            self._apply_fact(fact, fact_meta, result, decisions.get(fact.text, []),
                             similar, user_id, agent_id, run_id, ts, corpus)
        return result

    def _similar_memories(self, fact: ExtractedFact, user_id: str,
                          agent_id: Optional[str], run_id: Optional[str],
                          corpus: Optional[List[tuple]] = None) -> List[tuple]:
        """(cosine, memory) pairs for one candidate fact, sorted desc.

        ``corpus``: optional precomputed [(id, embedding_bytes)] from
        ``store.all_embeddings()`` — one query shared across all facts of an
        add() call instead of N+1 roundtrips per fact. Only the top candidates
        are hydrated into Memory objects.
        """
        from .embeddings import cosine, unpack

        try:
            qvec = self.embedder.embed(fact.text)
        except Exception:
            return []
        if corpus is None:
            corpus = self.store.all_embeddings(user_id, agent_id, run_id)
        dim = self.cfg.embedding_dim
        scored: List[tuple] = []
        for mid, blob in corpus:
            vec = unpack(blob, dim)
            if vec is None:
                continue
            scored.append((cosine(qvec, vec), mid))
        scored.sort(key=lambda t: t[0], reverse=True)
        sims: List[tuple] = []
        for sim, mid in scored[:12]:  # top-K only; gate/consolidation need no more
            mem = self.store.get(mid)
            if mem is None or mem.invalid_at:
                continue
            sims.append((sim, mem))
        return sims

    def _extract_facts(self, conversation: str) -> List[ExtractedFact]:
        data = self.llm.complete_json(
            "You are a memory extraction engine for an AI assistant.",
            FACT_EXTRACTION_PROMPT.format(conversation=conversation),
        )
        facts: List[ExtractedFact] = []
        if not data:
            return facts
        for item in data.get("facts", []):
            try:
                mtype = item.get("type", MemoryType.FACT)
                if mtype not in MemoryType.ALL:
                    mtype = MemoryType.FACT
                slot = item.get("slot")
                from .slots import canonicalize_slot

                facts.append(ExtractedFact(
                    text=str(item["text"]).strip(),
                    memory_type=mtype,
                    importance=max(0.0, min(1.0, float(item.get("importance", 0.5)))),
                    entities=[str(e) for e in item.get("entities", [])][:8],
                    slot=canonicalize_slot(
                        slot,
                        getattr(self.cfg, "slot_aliases", None)),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return facts

    # ---------------------------------------------------- conflict resolution
    def _decide_operations(self, prepared: List[tuple]) -> Dict[str, List[dict]]:
        """Consolidation operations per fact against its similar memories.

        Real-LLM path batches all facts into ONE chat call (a single turn
        should not cost one roundtrip per fact); falls back to per-fact calls
        when the batch response is malformed. The local test server supplies deterministic protocol responses.
        """
        from .llm import CONSOLIDATION_BATCH_PROMPT

        out: Dict[str, List[dict]] = {}
        if not prepared:
            return out
        items = [{"fact": fact.text, "slot": fact.slot,
                  "candidates": self._candidate_payload(fact, similar)}
                 for fact, _, similar in prepared]
        if sum(1 for it in items if it["candidates"]) <= 1:
            for fact, _, similar in prepared:  # single undecided fact: plain call
                out[fact.text] = self._llm_decisions(
                    fact.text, self._candidate_payload(fact, similar), slot=fact.slot)
            return out
        try:
            data = self.llm.complete_json(
                "You are the memory consolidation module of an AI agent.",
                CONSOLIDATION_BATCH_PROMPT.format(
                    items=json.dumps(items, ensure_ascii=False)))
            results = (data or {}).get("results")
            if not isinstance(results, list):
                raise ValueError("batch response malformed")
            by_fact: Dict[str, List[dict]] = {}
            for r in results:
                if not (isinstance(r, dict) and isinstance(r.get("fact"), str)):
                    continue
                ops = [o for o in r.get("operations", [])
                       if isinstance(o, dict) and o.get("op") in ("NOOP", "UPDATE", "DELETE")
                       and o.get("id")]
                by_fact[r["fact"]] = ops
            for fact, _, similar in prepared:
                if fact.text in by_fact:
                    out[fact.text] = by_fact[fact.text]
                else:
                    out[fact.text] = self._llm_decisions(
                        fact.text, self._candidate_payload(fact, similar), slot=fact.slot)
        except Exception:
            # any batch failure -> per-fact decisions (previous behavior)
            for fact, _, similar in prepared:
                out[fact.text] = self._llm_decisions(
                    fact.text, self._candidate_payload(fact, similar), slot=fact.slot)
        return out

    def _candidate_payload(self, fact: ExtractedFact, similar: List[tuple]) -> List[dict]:
        """Candidates strong enough to consult the consolidator about.

        Slot is a hint, not a rule: same-meaning slots lower the bar so the
        LLM can decide UPDATE vs ADD (multi-value / time-qualified stay ADD).
        """
        from .slots import same_slot

        aliases = getattr(self.cfg, "slot_aliases", None)
        return [{"id": m.id, "text": m.text, "slot": m.metadata.get("slot")}
                for sim, m in similar[: self.cfg.similarity_candidates]
                if sim >= self.cfg.conflict_threshold
                or (sim >= self.cfg.gate_slot_floor and fact.slot
                    and same_slot(fact.slot, m.metadata.get("slot"), aliases))]

    def _llm_decisions(self, fact_text: str, candidates: List[dict],
                       slot: "str | None" = None) -> List[dict]:
        from .llm import CONSOLIDATION_PROMPT

        if not candidates:
            return []
        data = self.llm.complete_json(
            "You are the memory consolidation module of an AI agent.",
            CONSOLIDATION_PROMPT.format(fact=fact_text, slot=slot,
                                        candidates=json.dumps(candidates, ensure_ascii=False)))
        return (data or {}).get("operations", [])

    def _apply_fact(self, fact: ExtractedFact, metadata: Dict[str, Any],
                    result: AddResult, decisions: List[dict], similar: List[tuple],
                    user_id: str, agent_id: Optional[str], run_id: Optional[str],
                    timestamp: Optional[str], corpus: List[tuple]) -> None:
        """Execute one fact's decided operations, then ADD unless consumed."""
        handled_ids: set = set()
        for dec in decisions:
            op = dec.get("op")
            mid = dec.get("id")
            if op == "NOOP":
                result.noop += 1
                handled_ids.add(mid)
            elif op == "UPDATE":
                emb = self.retriever.embed_for_storage(fact.text)
                self.store.replace_text(mid, fact.text, fact.entities, emb,
                                        metadata=metadata)
                self._index_upsert_by_id(mid, fact.text, emb, user_id, agent_id, run_id)
                result.updated.append(mid)
                handled_ids.add(mid)
            elif op == "DELETE":
                self.store.soft_delete(mid)
                # same contract as delete(): soft-deleted rows are hidden by
                # the index-removal, not by the relational filter on the
                # search path — the stale vector would still occupy a
                # semantic_topk candidate slot until a manual rebuild
                self.vector_store.delete(mid)
                result.deleted.append(mid)
                handled_ids.add(mid)

        # default: ADD the new fact — unless a NOOP/UPDATE decision already
        # consumed it (a DELETE still lets the new fact in, replacing the old)
        consumed = any(d.get("op") in ("NOOP", "UPDATE") for d in decisions)
        if consumed:
            return

        mem = Memory(
            text=fact.text,
            memory_type=fact.memory_type,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            entities=list(dict.fromkeys(fact.entities + _extract_query_entities(fact.text)))[:8],
            metadata=metadata,
            importance=fact.importance,
            source="conversation",
            attachments=list(fact.attachments) if fact.attachments else [],
            **({"created_at": timestamp, "updated_at": timestamp, "valid_at": timestamp}
               if timestamp else {}),
        )
        emb = self.retriever.embed_for_storage(fact.text)
        self.store.insert(mem, emb)
        self._index_upsert(mem, emb)
        # facts added later in this same call must see this one (dup detection)
        corpus.append((mem.id, emb))
        result.added.append(mem.id)

    # ------------------------------------------------------------------- read
    @_locked
    def search(self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None,
               run_id: Optional[str] = None, limit: int = 10,
               include_forgotten: bool = False, memory_type: Optional[str] = None,
               slot: Optional[str] = None, reinforce: bool = True) -> List[SearchResult]:
        """Hybrid retrieval; every hit's access_count grows (reinforcement).

        ``memory_type``/``slot``: optional post-fusion filters (e.g. only
        "preference" memories, or only slot="location" facts).
        ``reinforce=False``: read-only recall without the access_count bump
        (used by render_context previews)."""
        return self.retriever.search(query, user_id=user_id, agent_id=agent_id,
                                     run_id=run_id, limit=limit,
                                     include_forgotten=include_forgotten,
                                     memory_type=memory_type, slot=slot,
                                     reinforce=reinforce)

    @_locked
    def get(self, memory_id: str) -> Optional[Memory]:
        return self.store.get(memory_id)

    @_locked
    def get_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                run_id: Optional[str] = None, limit: int = 100) -> List[Memory]:
        return self.store.all_valid(user_id, agent_id, run_id)[:limit]

    @_locked
    def update(self, memory_id: str, text: str) -> bool:
        """Manual correction of one memory. Refuses invalidated memories —
        replace_text clears invalid_at, so editing one would resurrect it."""
        mem = self.store.get(memory_id)
        if mem is None or mem.invalid_at:
            return False
        entities = list(dict.fromkeys(_extract_query_entities(text) + mem.entities))[:8]
        emb = self.retriever.embed_for_storage(text)
        self.store.replace_text(memory_id, text, entities, emb)
        # keep the vector index in sync, or search keeps ranking the old text
        self._index_upsert_by_id(memory_id, text, emb, mem.user_id, mem.agent_id, mem.run_id)
        return True

    @_locked
    def delete(self, memory_id: str, hard: bool = False) -> bool:
        # PG-first dual-write order (matches add/update): the relational store
        # is the source of truth; the vector index is rebuilt from it. A crash
        # between the two steps then leaves at worst a stale readable point
        # (filtered by store.get) — never a PG row invisible to search.
        if self.store.get(memory_id) is None:
            return False
        if hard:
            self.store.hard_delete(memory_id)
        else:
            self.store.soft_delete(memory_id)  # keep for temporal audit
        # Index second and fail loud: soft-deleted rows are filtered out of
        # search via index removal (store.get still returns them), so a failed
        # vector delete must surface — run rebuild_index() to reconcile.
        self.vector_store.delete(memory_id)
        return True

    # ------------------------------------------------------------ vector index
    def _index_upsert(self, mem: Memory, emb: bytes) -> None:
        from .embeddings import unpack

        self.vector_store.upsert(mem.id, unpack(emb, self.cfg.embedding_dim) or [],
                                 {"memory_id": mem.id, "user_id": mem.user_id,
                                  "agent_id": mem.agent_id, "run_id": mem.run_id})

    def _index_upsert_by_id(self, mem_id: str, text: str, emb: bytes,
                            user_id: Optional[str], agent_id: Optional[str],
                            run_id: Optional[str]) -> None:
        from .embeddings import unpack

        self.vector_store.upsert(mem_id, unpack(emb, self.cfg.embedding_dim) or [],
                                 {"memory_id": mem_id, "user_id": user_id,
                                  "agent_id": agent_id, "run_id": run_id})

    @_locked
    def rebuild_index(self) -> int:
        """Rebuild the vector index from the embeddings kept in relational
        storage (use after switching embedding models or losing the index)."""
        from .embeddings import unpack

        # Rebuild means exact mirror of active PostgreSQL rows: clear stale
        # points first, then upload the complete source-of-truth snapshot.
        self.vector_store.clear()
        points = []
        for row in self.store.all_rows_with_embeddings():
            vec = unpack(row.get("embedding"), self.cfg.embedding_dim)
            if vec is None:
                continue
            points.append((row["id"], vec,
                           {"memory_id": row["id"], "user_id": row["user_id"],
                            "agent_id": row.get("agent_id"), "run_id": row.get("run_id")}))
        self.vector_store.upsert_many(points)
        n = len(points)
        if hasattr(self.vector_store, "count"):
            actual = self.vector_store.count()
            if actual != n:
                raise RuntimeError(
                    f"Qdrant rebuild mismatch: expected {n} points, got {actual} "
                    "(fewer = lost writes, more = stale dirty points)")
        return n

    # ------------------------------------------------------- context rendering
    @_locked
    def render_context(self, user_id: str = "default", agent_id: Optional[str] = None,
                       query: Optional[str] = None) -> str:
        """Letta-style always-injected block: strongest stable facts + (optional)
        query-relevant memories. Feed this into your system prompt."""
        cfg = self.cfg
        lines: List[str] = []

        memories = self.store.all_valid(user_id, agent_id)
        scored = sorted(
            memories,
            key=lambda m: m.importance * (0.3 + 0.7 * retention(
                m, cfg.half_life_days, cfg.reinforcement_gain,
                consolidation_mult=cfg.consolidation_half_life_mult,
                episodic_mult=cfg.episodic_half_life_mult,
                max_mult=cfg.max_half_life_mult)),
            reverse=True,
        )
        for m in scored:
            if m.importance < 0.55:
                continue
            if is_forgotten(m, cfg.half_life_days, cfg.reinforcement_gain,
                             cfg.retention_floor,
                             consolidation_mult=cfg.consolidation_half_life_mult,
                             episodic_floor=cfg.episodic_floor,
                             episodic_mult=cfg.episodic_half_life_mult,
                             max_mult=cfg.max_half_life_mult):
                continue  # forgotten memories never enter the core block
            line = f"- {m.text}"
            if len("\n".join(lines + [line])) > cfg.core_max_chars:
                break
            lines.append(line)

        block = "## Memory\n" + ("\n".join(lines) if lines else "(no long-term memories yet)")
        if query:
            hits = self.search(query, user_id=user_id, agent_id=agent_id, limit=5,
                               reinforce=False)  # preview must not reinforce
            if hits:
                rel = "\n".join(f"- {h.memory.text}" for h in hits)
                block += "\n\n## Relevant to current query\n" + rel
        return block

    # ------------------------------------------------------------------ audit
    @_locked
    def get_history(self, memory_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Full audit trail (ADD/UPDATE/DELETE/ACCESS with prev/new values)."""
        return self.store.history(memory_id, limit)

    @_locked
    def consolidate_background(self, user_id: str = "default", agent_id: Optional[str] = None,
                               run_id: Optional[str] = None, min_cluster: Optional[int] = None,
                               threshold: Optional[float] = None) -> Dict[str, Any]:
        """LangMem-style reflection pass: cluster same-topic memories, distill a
        summary, supersede the members (invalid_at + superseded_by + audit log).
        Run it periodically (cron / idle hook / after every N turns)."""
        from . import consolidation

        return consolidation.consolidate(self, user_id=user_id, agent_id=agent_id,
                                         run_id=run_id, min_cluster=min_cluster,
                                         threshold=threshold)

    @_locked
    def stats(self) -> Dict[str, Any]:
        s = self.store.stats()
        base = base_embedder(self.embedder)
        s["backend"] = {
            "storage": self.cfg.storage_backend,
            "vector": self.cfg.vector_backend,
            # real model names — an ops console should show what is actually
            # serving, not just the driver family
            "llm": self.cfg.llm_model,
            "llm_backend": self.cfg.llm_backend,
            "embedding": getattr(base, "model", None) or "unknown",
            "dim": getattr(self.embedder, "dim", None) or self.cfg.embedding_dim,
        }
        try:
            import os

            s["media_files"] = (len(os.listdir(self.cfg.media_dir))
                                if os.path.isdir(self.cfg.media_dir) else 0)
        except OSError:
            s["media_files"] = 0
        # background reflection health: lets monitoring see a silently
        # failing loop (issues #4)
        s["auto_reflect"] = {
            "active": self.auto_reflect_active,
            **self._reflect_status,
        }
        return s

    @_locked
    def reset(self) -> None:
        """Wipe everything (dev/test helper)."""
        self.store.reset()
        self.vector_store.clear()

    def close(self) -> None:
        self.disable_auto_reflect()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self.store.close()

    # ------------------------------------------------------------- operations
    @_locked
    def compact(self, user_id: str = "default", agent_id: Optional[str] = None,
                run_id: Optional[str] = None,
                threshold: Optional[float] = None) -> Dict[str, Any]:
        """Pure-vector near-duplicate compaction (no LLM): within a scope,
        memories at cosine >= ``threshold`` (default dedup_threshold) form a
        duplicate cluster; the best member (highest importance, then newest)
        is kept and the rest are superseded into it (audit chain preserved).
        Complements consolidate_background(): that one abstracts same-TOPIC
        clusters, this one removes literal near-duplicates."""
        cfg = self.cfg
        threshold = threshold if threshold is not None else cfg.dedup_threshold
        dim = cfg.embedding_dim
        vecs: Dict[str, List[float]] = {}
        for mid, blob in self.store.all_embeddings(user_id, agent_id, run_id):
            vec = unpack(blob, dim)
            if vec is not None:
                vecs[mid] = vec
        memories = {m.id: m for m in self.store.all_valid(user_id, agent_id, run_id)}
        remaining = [mid for mid in vecs if mid in memories]
        report: Dict[str, Any] = {"clusters": 0, "absorbed": 0, "kept": []}
        while remaining:
            seed = remaining[0]
            group = [mid for mid in remaining
                     if cosine(vecs[seed], vecs[mid]) >= threshold]
            if len(group) >= 2:
                keep = max((memories[mid] for mid in group),
                           key=lambda m: (m.importance, m.created_at))
                for mid in group:
                    if mid != keep.id:
                        self.store.supersede(mid, keep.id)
                        self.vector_store.delete(mid)
                report["clusters"] += 1
                report["absorbed"] += len(group) - 1
                report["kept"].append(keep.id)
            for mid in group:
                remaining.remove(mid)
        return report

    @_locked
    def media_gc(self, delete: bool = False) -> Dict[str, Any]:
        """Drop orphan files in media_dir (no memory references them —
        invalidated memories still count, the audit must stay replayable).
        ``delete=False`` only reports what WOULD be removed."""
        import os as _os

        referenced = set()
        for mem in self.store.all_rows(include_invalid=True):
            for att in mem.attachments or []:
                if att.get("sha256"):
                    referenced.add(att["sha256"])
        media_dir = self.cfg.media_dir
        orphans: List[str] = []
        removed: List[str] = []
        kept = 0
        if _os.path.isdir(media_dir):
            for name in sorted(_os.listdir(media_dir)):
                if name.split(".")[0] in referenced:
                    kept += 1
                else:
                    orphans.append(name)
        if delete:
            for name in orphans:
                try:
                    _os.remove(_os.path.join(media_dir, name))
                    removed.append(name)
                except OSError:
                    kept += 1
        return {"orphan": len(orphans), "kept": kept,
                "removed": removed, "would_remove": orphans}

    # ---------------------------------------------------------- export/import
    @_locked
    def export_jsonl(self, path: Optional[str] = None, user_id: Optional[str] = None,
                     agent_id: Optional[str] = None, run_id: Optional[str] = None,
                     include_invalid: bool = True,
                     include_embeddings: bool = True) -> List[str]:
        """Dump memories as JSON lines (returns the lines; writes ``path`` if
        given). Includes full audit state; media files are NOT inside — copy
        media_dir alongside the dump."""
        import base64

        rows = self.store.all_rows(user_id, agent_id, run_id,
                                   include_invalid=include_invalid)
        blobs = (self.store.get_embeddings([m.id for m in rows])
                 if include_embeddings else {})
        lines = []
        for m in rows:
            d = m.to_dict()
            d["superseded_by"] = m.superseded_by
            d["access_count"] = m.access_count
            d["last_accessed"] = m.last_accessed
            if include_embeddings:
                blob = blobs.get(m.id)
                if blob:
                    d["embedding_b64"] = base64.b64encode(blob).decode()
            lines.append(json.dumps(d, ensure_ascii=False))
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        return lines

    @_locked
    def import_jsonl(self, lines, on_conflict: str = "skip") -> Dict[str, int]:
        """Import lines produced by export_jsonl (strings or parsed dicts).

        ``on_conflict``: skip (default) | overwrite. Embeddings ride along in
        the dump; missing ones are re-embedded. Refresh the vector index with
        rebuild_index() after a large import."""
        import base64

        stats = {"imported": 0, "skipped": 0, "reembedded": 0}
        for line in lines:
            d = json.loads(line) if isinstance(line, str) else dict(line)
            mid = d.get("id") or new_id()
            if self.store.get(mid) is not None:
                if on_conflict == "overwrite":
                    self.delete(mid, hard=True)
                else:
                    stats["skipped"] += 1
                    continue
            emb = None
            if d.get("embedding_b64"):
                emb = base64.b64decode(d["embedding_b64"])
            if emb is None:
                emb = self.retriever.embed_for_storage(d.get("text", ""))
                stats["reembedded"] += 1
            mem = Memory(
                id=mid,
                text=d.get("text", ""),
                memory_type=d.get("memory_type", MemoryType.FACT),
                user_id=d.get("user_id", "default"),
                agent_id=d.get("agent_id"),
                run_id=d.get("run_id"),
                entities=d.get("entities", []),
                metadata=d.get("metadata", {}),
                importance=float(d.get("importance", 0.5)),
                access_count=int(d.get("access_count", 0)),
                last_accessed=d.get("last_accessed"),
                created_at=d.get("created_at") or utcnow(),
                updated_at=d.get("updated_at") or utcnow(),
                valid_at=d.get("valid_at") or d.get("created_at") or utcnow(),
                invalid_at=d.get("invalid_at"),
                superseded_by=d.get("superseded_by"),
                source=d.get("source", "conversation"),
                attachments=d.get("attachments", []),
            )
            self.store.insert(mem, emb)
            if not mem.invalid_at:
                self._index_upsert(mem, emb)
            stats["imported"] += 1
        return stats

    # ------------------------------------------------------------ background
    def add_background(self, *args, **kwargs):
        """add() on a small internal thread pool; returns a Future (stdlib
        concurrent.futures). Use for latency-sensitive agents — shares the
        engine lock with REST writes, so it is mutually exclusive with them."""
        from concurrent.futures import ThreadPoolExecutor

        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=2,
                                            thread_name_prefix="memtide-add")
        return self._pool.submit(self.add, *args, **kwargs)

    def enable_auto_reflect(self, interval_seconds: int = 3600,
                            user_id: Optional[str] = None,
                            agent_id: Optional[str] = None,
                            run_id: Optional[str] = None) -> None:
        """Periodic background reflection (stdlib threading, daemon): every
        ``interval_seconds`` run consolidate_background() for one user (when
        given) or all distinct users in the store."""
        if self._reflect_stop is not None:
            return
        import threading

        stop = threading.Event()
        self._reflect_stop = stop
        self._reflect_status = {"consecutive_failures": 0, "last_error": None}

        def loop():
            while not stop.wait(interval_seconds):
                try:
                    uids = [user_id] if user_id else self.store.distinct_users()
                    for uid in uids:
                        self.consolidate_background(uid, agent_id=agent_id,
                                                    run_id=run_id)
                    self._reflect_status["consecutive_failures"] = 0
                    self._reflect_status["last_error"] = None
                except Exception as e:
                    # a failed pass must never kill the loop — but it must
                    # also never be invisible: log the first failure and
                    # then every 10th consecutive one
                    st = self._reflect_status
                    st["consecutive_failures"] += 1
                    st["last_error"] = f"{type(e).__name__}: {e}"
                    n = st["consecutive_failures"]
                    if n == 1 or n % 10 == 0:
                        print(f"[memtide] auto-reflect failed {n}x: {st['last_error']}")

        self._reflect_thread = threading.Thread(
            target=loop, name="memtide-auto-reflect", daemon=True)
        self._reflect_thread.start()

    def disable_auto_reflect(self) -> None:
        if self._reflect_stop is not None:
            self._reflect_stop.set()
        thread = self._reflect_thread
        self._reflect_stop = None
        self._reflect_status = {"consecutive_failures": 0, "last_error": None}
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)  # no orphaned pass overlapping a new loop
        self._reflect_thread = None

    @property
    def auto_reflect_active(self) -> bool:
        return self._reflect_stop is not None
