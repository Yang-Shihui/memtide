"""Hybrid retrieval: vector cosine + BM25 full-text (pg_search) + entity match + recency.

Candidates are fetched from three independent channels, ranked per channel,
fused with Reciprocal Rank Fusion (RRF), then re-scored with raw semantic
similarity, retention (forgetting curve) and importance. The components dict
on each SearchResult explains the final ordering.

Optional quality layers (all default off — see MemoryConfig):
- query_expansion: real-LLM deployments add an English translation and a
  paraphrase of the query; all variants' channel rankings fuse together.
- rerank_backend="http": the fused top-N goes to a cross-encoder (Jina/Cohere
  style POST /rerank) for reordering.
- mmr_lambda > 0: final top-k selection maximizes λ·score − (1−λ)·max-sim to
  the already-selected set (diversity).
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Dict, List, Optional, Tuple

from .config import MemoryConfig
from .decay import is_forgotten, retention
from .embeddings import cosine, pack, unpack
from .storage import Storage
from .types import SearchResult

QUERY_EXPANSION_PROMPT = """You optimize retrieval queries for a long-term memory store.
Given the query below, produce exactly two alternative search queries:
1. its English translation (or English paraphrase if it is already English),
2. a rephrasing that a memory entry about this topic would likely use.

Output STRICT JSON: {{"variants": ["...", "..."]}}

Query: {query}"""


def _extract_query_entities(query: str) -> List[str]:
    import re

    ents: List[str] = []
    m = re.search(r"[\u4e00-\u9fa5A-Za-z·]{1,15}(?:叫|名字是)([\u4e00-\u9fa5A-Za-z·]{1,15})", query)
    if m:
        ents.append(m.group(1))
    m = re.search(r"\b(?:name(?:'s| is)|named)\s+([A-Za-z][\w'-]{1,29})", query, re.I)
    if m:
        ents.append(m.group(1))
    # capitalized latin tokens (likely entities)
    ents.extend(t for t in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", query) if t not in {"The", "What", "Where", "Who", "My", "I"})
    # CJK 2-4 char runs as loose entity keys
    ents.extend(re.findall(r"[\u4e00-\u9fff]{2,4}", query))
    return list(dict.fromkeys(ents))


def _rrf(rankings: List[List[str]], k: int) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, mid in enumerate(ranking):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def expand_query(query: str, llm, timeout: float = 20.0) -> List[str]:
    """Query variants via the LLM ([original, translation, paraphrase]).

    Returns [query] on any failure — expansion must never break retrieval."""
    try:
        data = llm.complete_json(
            "You optimize retrieval queries for a long-term memory store.",
            QUERY_EXPANSION_PROMPT.format(query=query))
        variants = [str(v).strip() for v in (data or {}).get("variants", [])
                    if isinstance(v, str) and v.strip()]
        return [query] + variants[:2]
    except Exception:
        return [query]


def http_rerank(query: str, docs: List[str], base_url: str, model: str,
                api_key: Optional[str], timeout: float = 20.0) -> Optional[List[float]]:
    """Cross-encoder scores via a Jina/Cohere-style POST {base}/rerank.

    Returns scores aligned with ``docs``, or None on any failure (rerank is
    an optimization — it must degrade silently)."""
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/rerank",
        data=json.dumps({"model": model, "query": query, "documents": docs,
                         "top_n": len(docs)}).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        scores = [0.0] * len(docs)
        for r in data.get("results", []):
            idx = r.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs):
                scores[idx] = float(r.get("relevance_score", 0.0))
        return scores
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


class Retriever:
    def __init__(self, storage: Storage, config: MemoryConfig, embedder,
                 vector_store=None, llm=None):
        self.store = storage
        self.cfg = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm  # needed only for query expansion

    # -- channels -------------------------------------------------------------
    def _channels(self, query: str, qvec: List[float], user_id: Optional[str],
                  agent_id: Optional[str], run_id: Optional[str]
                  ) -> Tuple[List[Tuple[str, float]], List[str], List[str]]:
        """(semantic ranking, bm25 ids, entity ids) for one query variant."""
        sem = self.vector_store.search(qvec, user_id=user_id, agent_id=agent_id,
                                       run_id=run_id, topk=self.cfg.semantic_topk)
        bm25 = self.store.fts_search(query, self.cfg.bm25_topk, user_id=user_id,
                                     agent_id=agent_id, run_id=run_id)
        ents = _extract_query_entities(query)
        ent_ids = self.store.entity_lookup(ents, user_id=user_id, agent_id=agent_id,
                                           run_id=run_id) if ents else []
        # drop entity hits that don't actually contain the string (CJK 2-4 char keys are loose)
        ent_ids = [mid for mid in ent_ids
                   if (mem := self.store.get(mid)) and any(e.lower() in mem.text.lower() for e in ents)]
        return sem, bm25, ent_ids

    def search(self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None,
               run_id: Optional[str] = None, limit: Optional[int] = None,
               include_forgotten: bool = False, memory_type: Optional[str] = None,
               slot: Optional[str] = None) -> List[SearchResult]:
        cfg = self.cfg
        limit = limit or cfg.final_topk
        qvec = self.embedder.embed(query)

        # channel rankings for the query (+ expanded variants when enabled)
        rankings: List[List[str]] = []
        sem_map: Dict[str, float] = {}
        bm25_hits: set = set()
        ent_hits: set = set()
        variants = [query]
        if cfg.query_expansion and not getattr(self, "_expansion_disabled", False) and self.llm is not None:
            variants = expand_query(query, self.llm)
        for i, variant in enumerate(variants):
            vvec = qvec if i == 0 else self.embedder.embed(variant)
            sem, bm25, ent_ids = self._channels(variant, vvec, user_id, agent_id, run_id)
            rankings += [[mid for mid, _ in sem], bm25, ent_ids]
            bm25_hits.update(bm25)
            ent_hits.update(ent_ids)
            for mid, sim in sem:
                sem_map[mid] = max(sem_map.get(mid, 0.0), sim)

        fused = _rrf(rankings, cfg.rrf_k)
        if not fused:
            return []

        w = cfg.weights
        results: List[SearchResult] = []
        for mid, rrf_score in fused.items():
            mem = self.store.get(mid)
            if mem is None:
                continue
            ret = retention(mem, cfg.half_life_days, cfg.reinforcement_gain,
                            consolidation_mult=cfg.consolidation_half_life_mult)
            if not include_forgotten and is_forgotten(
                    mem, cfg.half_life_days, cfg.reinforcement_gain,
                    cfg.retention_floor, consolidation_mult=cfg.consolidation_half_life_mult):
                continue
            sem_sim = sem_map.get(mid, 0.0)
            score = (
                w["rrf"] * rrf_score
                + w["semantic"] * sem_sim
                + w["recency"] * ret
                + w["importance"] * mem.importance
            )
            results.append(SearchResult(
                memory=mem,
                score=score,
                components={
                    "rrf": rrf_score,
                    "semantic": sem_sim,
                    "bm25": 1.0 if mid in bm25_hits else 0.0,
                    "entity": 1.0 if mid in ent_hits else 0.0,
                    "retention": ret,
                },
            ))
        results.sort(key=lambda r: r.score, reverse=True)

        # optional cross-encoder re-ranking of the fused top-N
        if cfg.rerank_backend == "http" and cfg.rerank_base_url and len(results) > 1:
            head = results[: max(2, cfg.rerank_topk)]
            scores = http_rerank(query, [r.memory.text for r in head],
                                 cfg.rerank_base_url, cfg.rerank_model,
                                 cfg.rerank_api_key)
            if scores is not None:  # silent fallback keeps fused order on error
                for r, s in zip(head, scores):
                    r.components["rerank"] = round(s, 4)
                head.sort(key=lambda r: r.components["rerank"], reverse=True)
                results = head + results[len(head):]

        # post-fusion filters (types/slots stored on memories)
        if memory_type:
            results = [r for r in results if r.memory.memory_type == memory_type]
        if slot:
            results = [r for r in results if r.memory.metadata.get("slot") == slot]

        # MMR diversity selection
        if cfg.mmr_lambda and len(results) > 1:
            results = self._mmr(results, qvec, limit)

        top = results[:limit]
        # retrieval reinforces memory (spacing effect) — skip ACCESS noise in tests
        if top:
            self.store.mark_accessed([r.memory.id for r in top])
        return top

    def _mmr(self, results: List[SearchResult], qvec: List[float],
             limit: int) -> List[SearchResult]:
        """Greedy MMR: λ·score − (1−λ)·max cosine to already-selected items."""
        lam = self.cfg.mmr_lambda
        vecs: Dict[str, List[float]] = {}
        dim = self.cfg.embedding_dim

        def vec_of(r: SearchResult) -> List[float]:
            if r.memory.id not in vecs:
                vecs[r.memory.id] = unpack(
                    self.store.get_embedding(r.memory.id), dim) or []
            return vecs[r.memory.id]

        pool = list(results)
        selected = [pool.pop(0)]
        while pool and len(selected) < limit:
            best, best_val = None, None
            for r in pool:
                rv = vec_of(r)
                max_sim = max((cosine(rv, vec_of(s)) for s in selected), default=0.0)
                val = lam * r.score - (1 - lam) * max_sim
                if best_val is None or val > best_val:
                    best, best_val = r, val
            selected.append(best)
            pool.remove(best)
        return selected

    def embed_for_storage(self, text: str) -> bytes:
        return pack(self.embedder.embed(text))
