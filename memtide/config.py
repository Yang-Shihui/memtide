"""Configuration for a MemoryEngine instance."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryConfig:
    # --- storage (PostgreSQL only) ----------------------------------------
    storage_backend: str = "postgres"
    pg_dsn: str = ""                     # e.g. postgresql://memtide:pw@localhost:5432/memtide
    embedding_dim: int = 256             # learned on first embedding API call

    # --- vector store (Qdrant only) -----------------------------------------
    vector_backend: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "memtide"

    # --- predictive-coding gate (gating.py) ---------------------------------
    # The memory corpus is the prior; only prediction error is encoded.
    gate_enabled: bool = True
    gate_redundant_bits: float = 0.5    # surprise <= this -> REJECT (fully predicted)
    gate_novel_bits: float = 2.5        # surprise >= this -> NOVEL (importance boost)
    gate_importance_boost: float = 0.10
    gate_slot_floor: float = 0.40       # min cosine to treat same-slot memory as predictor
    # slot-scoped prior: for slotted facts (location/role/...), surprise is
    # computed against SAME-slot memories only — "user lives in X" is not
    # predicted by "user works at Y" just because the phrasing matches
    gate_slot_scoped: bool = True

    # --- write path -------------------------------------------------------
    max_facts_per_turn: int = 12         # cap extraction output
    dedup_threshold: float = 0.94        # cosine >= this => duplicate (NOOP/UPDATE)
    conflict_threshold: float = 0.55     # cosine >= this + conflict signals => UPDATE
    similarity_candidates: int = 5       # existing memories compared per new fact

    # --- retrieval (hybrid fusion) ----------------------------------------
    semantic_topk: int = 40              # Qdrant vector candidates
    bm25_topk: int = 40                  # PostgreSQL pg_search BM25 candidates
    rrf_k: int = 60                      # reciprocal-rank-fusion constant
    # per-channel RRF weights: the loose entity channel counts less than the
    # two main channels; expanded query variants count less than the original
    entity_channel_weight: float = 0.5
    expansion_variant_weight: float = 0.5
    weights: dict = field(default_factory=lambda: {
        "rrf": 1.0,
        "semantic": 0.15,   # raw cosine bonus on top of RRF
        "bm25": 0.05,       # exact keyword-hit bonus (hit = 1.0 else 0.0)
        "entity": 0.03,     # entity-hit bonus (hit = 1.0 else 0.0)
        "recency": 0.10,    # retention (Ebbinghaus) bonus
        "importance": 0.05,
    })
    final_topk: int = 10
    # post-fusion filters (search(memory_type=, slot=))
    # MMR diversity: 0 = off; e.g. 0.7 blends score and dissimilarity
    mmr_lambda: float = 0.0
    # optional cross-encoder re-ranking of the fused top-N ("none" | "http";
    # http = Jina/Cohere-style POST {rerank_base_url}/rerank)
    rerank_backend: str = "none"
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_api_key: Optional[str] = None
    rerank_topk: int = 20
    # LLM query expansion (real-LLM deployments only): retrieve for the query
    # plus an English translation + a paraphrase, fuse all rankings
    query_expansion: bool = False

    # --- forgetting / reinforcement ----------------------------------------
    half_life_days: float = 45.0         # base Ebbinghaus half-life
    reinforcement_gain: float = 0.4      # each access slows decay: hl *= 1+g*ln(1+n)
    retention_floor: float = 0.02        # below this a memory is "forgotten" (hidden)
    max_half_life_mult: float = 4.0      # cap on the reinforcement stretch
    # episodic memories fade faster and are forgotten sooner than facts
    episodic_half_life_mult: float = 0.5
    episodic_floor: float = 0.05

    # --- background consolidation (consolidation.py) --------------------------
    consolidation_min_cluster: int = 3   # smallest cluster worth distilling
    consolidation_similarity: float = 0.45   # real embeddings: same-topic ~0.55, cross-topic ~0.28
    # abstractions forget slower than episodes (effective half-life multiplier
    # for memories with source="consolidation")
    consolidation_half_life_mult: float = 3.0

    # --- core memory (Letta-style always-injected block) ---------------------
    core_max_chars: int = 1200

    # --- backends ------------------------------------------------------------
    # llm: any OpenAI-compatible endpoint (API key required)
    llm_backend: str = "openai"
    llm_base_url: str = "https://api.openai.com/v1"
    embedding_base_url: str = ""         # empty -> llm_base_url
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = None
    # embedding: "auto" tries dashscope -> openai (by available keys)
    embedding_backend: str = "auto"      # auto|openai|dashscope
    embedding_model: str = "text-embedding-3-small"
    dashscope_api_key: Optional[str] = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_embedding_model: str = "qwen3.7-text-embedding"

    # --- multimodal (multimodal.py) ----------------------------------------
    # Media parts are described to text (vision model / optional STT) and the
    # original bytes are stored content-addressed under media_dir.
    multimodal_enabled: bool = True
    media_dir: str = "memtide_media"          # content-addressed asset store
    max_media_bytes: int = 10 * 1024 * 1024
    # SECURITY: reading media from local paths is disabled by default — a REST
    # deployment would otherwise let any caller read arbitrary local files
    # ({"path": "/etc/passwd"} -> GET /media/<sha>). Enable only for trusted
    # embedders (CLI / in-process agents).
    media_allow_paths: bool = False
    # vision: empty base_url/model -> fall back to the main LLM endpoint
    # (works with any OpenAI-compatible vision model, e.g. qwen-vl-max)
    vision_base_url: str = ""
    vision_model: str = ""
    vision_api_key: Optional[str] = None
    # speech-to-text: empty stt_model disables transcription (audio kept as
    # reference-only attachments)
    stt_base_url: str = ""
    stt_model: str = ""

    # --- slot normalisation (slots.py) --------------------------------------
    # Open-hint slots: canonicalise aliases (city/住址 -> location) so the
    # same meaning with different names still matches. Extra user aliases
    # merge over the built-in table.
    slot_aliases: dict = field(default_factory=dict)

    # --- service / ops -------------------------------------------------------
    api_key: Optional[str] = None        # when set, REST requires it (X-API-Key)
    auto_reflect_seconds: int = 0        # >0: run consolidate_background periodically

    def resolve_api_key(self) -> Optional[str]:
        if self.llm_api_key:
            return self.llm_api_key
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")

    def resolve_dashscope_key(self) -> Optional[str]:
        if self.dashscope_api_key:
            return self.dashscope_api_key
        return os.environ.get("DASHSCOPE_API_KEY")


def config_from_env() -> "MemoryConfig":
    """Build a config from MEMTIDE_* / LLM_* / DASHSCOPE_* env variables
    (used by the REST server and Docker deployment)."""
    cfg = MemoryConfig(
        storage_backend=(os.environ.get("MEMTIDE_STORAGE") or "postgres"),
        pg_dsn=(os.environ.get("MEMTIDE_PG_DSN") or ""),
        vector_backend=(os.environ.get("MEMTIDE_VECTOR_BACKEND") or "qdrant"),
        qdrant_url=(os.environ.get("MEMTIDE_QDRANT_URL") or "http://localhost:6333"),
        qdrant_collection=(os.environ.get("MEMTIDE_QDRANT_COLLECTION") or "memtide"),
        llm_backend=(os.environ.get("LLM_BACKEND") or "openai"),
        llm_base_url=(os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"),
        llm_model=(os.environ.get("LLM_MODEL") or "gpt-4o-mini"),
        embedding_backend=(os.environ.get("EMBEDDING_BACKEND") or "auto"),
        embedding_model=(os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"),
        dashscope_embedding_model=(os.environ.get("DASHSCOPE_EMBEDDING_MODEL")
                                   or "qwen3.7-text-embedding"),
        dashscope_base_url=os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        multimodal_enabled=(os.environ.get("MEMTIDE_MULTIMODAL") or "1") not in ("0", "false", "no"),
        media_dir=(os.environ.get("MEMTIDE_MEDIA_DIR") or "memtide_media"),
        media_allow_paths=(os.environ.get("MEMTIDE_MEDIA_ALLOW_PATHS") or "0") in ("1", "true", "yes"),
        vision_base_url=(os.environ.get("MEMTIDE_VISION_BASE_URL") or ""),
        vision_model=(os.environ.get("MEMTIDE_VISION_MODEL") or ""),
        stt_base_url=(os.environ.get("MEMTIDE_STT_BASE_URL") or ""),
        stt_model=(os.environ.get("MEMTIDE_STT_MODEL") or ""),
    )
    if cfg.storage_backend != "postgres":
        raise ValueError("MEMTIDE_STORAGE 必须为 postgres；PostgreSQL 是唯一存储后端")
    if not cfg.pg_dsn:
        raise ValueError("请设置 MEMTIDE_PG_DSN（例如 postgresql://user:password@host:5432/db）")
    if cfg.vector_backend != "qdrant":
        raise ValueError("MEMTIDE_VECTOR_BACKEND 必须为 qdrant；Qdrant 是唯一向量后端")
    if os.environ.get("LLM_API_KEY"):
        cfg.llm_api_key = os.environ["LLM_API_KEY"]
    if os.environ.get("DASHSCOPE_API_KEY"):
        cfg.dashscope_api_key = os.environ["DASHSCOPE_API_KEY"]
    if os.environ.get("MEMTIDE_VISION_API_KEY"):
        cfg.vision_api_key = os.environ["MEMTIDE_VISION_API_KEY"]
    if os.environ.get("MEMTIDE_API_KEY"):
        cfg.api_key = os.environ["MEMTIDE_API_KEY"]
    if os.environ.get("MEMTIDE_AUTO_REFLECT"):
        try:
            cfg.auto_reflect_seconds = max(60, int(os.environ["MEMTIDE_AUTO_REFLECT"]))
        except ValueError:
            pass
    return cfg
