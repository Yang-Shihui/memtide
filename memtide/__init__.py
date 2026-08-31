"""Memtide — a stdlib-core agent memory engine.

Atomic, auditable, forgetting-aware memory for LLM agents:
- Mem0-style atomic-fact extraction & write-time consolidation
- Letta-style core memory block you inject into the system prompt
- Zep-style temporal audit (nothing is silently lost; updates/invalidations tracked)
- Hybrid retrieval: vector + BM25 full-text + entity + Ebbinghaus retention, RRF-fused

Production-ready agent memory with PostgreSQL, Qdrant, and any
OpenAI-compatible LLM/embedding endpoints.
"""

from .config import MemoryConfig
from .engine import MemoryEngine
from .types import AddResult, Event, ExtractedFact, Memory, MemoryType, SearchResult

__version__ = "0.1.0"

__all__ = [
    "MemoryEngine",
    "MemoryConfig",
    "Memory",
    "MemoryType",
    "Event",
    "AddResult",
    "ExtractedFact",
    "SearchResult",
    "__version__",
]
