"""Predictive-coding memory gate.

The existing memory corpus acts as the agent's generative prior: given a
candidate fact, we estimate how well the model predicts it and only let
*prediction error* through to encoding.

Theory mapping (see docs/predictive-coding-gate.md for full references):

- Predicted probability ``p_hat``: kernel-density estimate of the fact under
  the memory prior — ``p_hat = max_cosine_similarity ** 2`` (squared kernel).
- Surprise in bits: ``S = -log2(p_hat)`` — Shannon self-information, the
  first-order surrogate for Bayesian surprise (Itti & Baldi 2009, who define
  surprise as KL(posterior || prior); self-information is the KL for the
  degenerate one-fact observation).
- Three-way routing follows the schema / Multiple-Memory-Route account
  (van Kesteren et al. 2012; Quent, Henson & Greve 2021):

      S <= gate_redundant_bits   -> REJECT   (fully predicted: no error signal,
                                              no encoding — saves capacity)
      S >= gate_novel_bits       -> NOVEL    (off-schema: strong episodic
                                              encoding, importance boosted)
      otherwise                  -> INTEGRATE (schema-congruent: normal write,
                                              refines the prior)

  A special case overrides all three: a *volatile slot conflict* (the user
  moved cities, changed jobs) is by definition a large prediction error about
  an attribute memory exists to track, so it is always encoded — the gate
  routes it to consolidation as a forced store.

With an empty memory bank every fact is maximally surprising, which matches
intuition (a newborn agent encodes everything) — novelty boost then decays
naturally as the prior fills in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm import VOLATILE_SLOTS
from .types import ExtractedFact, Memory


@dataclass
class GateDecision:
    """Outcome of the gate for one candidate fact (kept on metadata for audit)."""

    store: bool
    reason: str                # "novel" | "integrate" | "redundant" | "volatile-update"
    surprise_bits: float
    max_similarity: float      # how strongly the best memory predicted the fact
    schema_fit: float          # mean of top-3 similarities (congruence with prior)
    importance_delta: float    # applied to fact.importance when encoding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "store": self.store,
            "reason": self.reason,
            "surprise_bits": round(self.surprise_bits, 3),
            "max_similarity": round(self.max_similarity, 3),
            "schema_fit": round(self.schema_fit, 3),
            "importance_delta": round(self.importance_delta, 3),
        }


class PredictiveGate:
    def __init__(self, config):
        self.cfg = config

    def evaluate(self, fact: ExtractedFact,
                 similar: List[Tuple[float, Memory]]) -> GateDecision:
        """``similar``: (cosine, memory) pairs, sorted by similarity desc."""
        cfg = self.cfg
        sims = [min(s, 1.0) for s, _ in similar]  # clamp float noise above 1.0
        max_sim = sims[0] if sims else 0.0
        schema_fit = sum(sims[:3]) / min(3, len(sims)) if sims else 0.0

        # --- prior prediction: kernel density over memory vectors -----------
        # For slotted facts (location/role/...) the prior is SAME-slot memories
        # only — "user lives in X" is not predicted by "user works at Y" just
        # because the sentence shape matches. A slotted fact with no same-slot
        # predecessor is unpredictable by definition -> fully surprising.
        if getattr(cfg, "gate_slot_scoped", True) and fact.slot:
            slot_sims = [min(s, 1.0) for s, mem in similar
                         if mem.metadata.get("slot") == fact.slot]
            if slot_sims:
                prior_sim = max(slot_sims)
            else:
                prior_sim = 0.0
        else:
            prior_sim = max_sim
        # max_sim**2 = squared-exponential-style kernel; 0.02 floor keeps the
        # log finite for a truly unseen fact (surprise saturates ~5.6 bits).
        p_hat = max(prior_sim, 0.02) ** 2
        surprise = -math.log2(p_hat) + 0.0  # +0.0 normalizes -0.0

        # --- override: volatile attribute conflict is always an error signal -
        if fact.slot in VOLATILE_SLOTS:
            for sim, mem in similar:
                if (mem.metadata.get("slot") == fact.slot
                        and sim >= cfg.gate_slot_floor
                        and mem.text != fact.text):
                    return GateDecision(True, "volatile-update", surprise,
                                        max_sim, schema_fit, 0.0)

        # --- three-way routing ----------------------------------------------
        if surprise <= cfg.gate_redundant_bits:
            return GateDecision(False, "redundant", surprise, max_sim,
                                schema_fit, 0.0)
        if surprise >= cfg.gate_novel_bits:
            return GateDecision(True, "novel", surprise, max_sim,
                                schema_fit, cfg.gate_importance_boost)
        return GateDecision(True, "integrate", surprise, max_sim,
                            schema_fit, 0.0)
