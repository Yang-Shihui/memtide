"""LLM backend: any OpenAI-compatible chat API.

The engine speaks the real OpenAI-compatible chat protocol only.
Point ``llm_base_url`` at any compatible endpoint (GLM, Qwen, OpenAI, vLLM,
one-api, ...); prompts and JSON parsing live here so behavior is identical
across backends.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, Optional

FACT_EXTRACTION_PROMPT = """You are a memory extraction engine for an AI assistant.
From the conversation below, extract atomic facts worth remembering long-term
about the user (and about the assistant's instructions/preferences).

Rules:
- One fact per item; short, self-contained, third-person ("The user's name is Li Lei").
- Write each fact in the same language as the conversation.
- Include: preferences, identity, relationships, goals, constraints, corrections.
- Do NOT include small talk, questions, or transient context.
- Do NOT output two items that state the same fact.
- "entities" lists concrete nouns worth indexing: person names, places,
  products, technologies, organizations (lowercase common nouns, keep names verbatim).
- "importance" rubric: identity/core preferences 0.8-0.9, goals/plans and
  corrections 0.7, useful context 0.4-0.6, trivial 0.2.
- "slot" marks single-value attributes that get REPLACED when they change.
  Use a short lowercase string (e.g. "name"|"role"|"employer"|"location"|
  "age"|"stack"|"plan", or invent one like "spouse"|"pet" when needed),
  or null for multi-value facts (likes, skills lists) and one-off events.
  Write time qualifiers into "text", never into "slot".
  (e.g. name -> name, lives in -> location, works at -> employer,
  is a ... engineer -> role, uses Rust -> stack, is working on -> plan)
- Output STRICT JSON: {{"facts": [{{"text": "...", "type": "fact|preference|episodic|procedural", "importance": 0.0-1.0, "entities": ["..."], "slot": null}}]}}

Conversation:
{conversation}"""

CONSOLIDATION_PROMPT = """You are the memory consolidation module of an AI agent.
Given a NEW fact and EXISTING memories, decide the operation for each candidate:

- "NOOP": candidate already states the same fact as the new fact, EVEN IF the
  wording differs (e.g. "likes Americano" vs "enjoys Americano coffee").
- "UPDATE": candidate should be REPLACED by a new value the new fact provides
  ("I moved to Shanghai" -> "lives in Hangzhou" becomes "lives in Shanghai").
- "DELETE": the new fact says the old one is no longer true and provides NO
  replacement value ("I quit coffee" -> "likes Americano" is deleted).
- "ADD" (returned implicitly): keep both, they are genuinely different facts.
Be conservative with ADD: near-duplicates with different phrasing are NOOP.

Candidates carry their "slot" tag when the memory tracks a single-value
attribute (location/role/...): a new fact for the SAME meaning slot is a
strong UPDATE signal — but not ironclad. Judge by meaning, not spelling
("city" vs "location" is the same slot). Multi-value facts (two homes,
Rust + Python) and time-qualified facts (last year vs this year) should be
kept as ADD even when slots match. Same-value restatement is NOOP.

Output STRICT JSON: {{"operations": [{{"id": "...", "op": "NOOP|UPDATE|DELETE", "reason": "..."}}]}}
Pick at most one operation per candidate; if nothing matches, return an empty list.

New fact: {fact}
New fact slot: {slot}
Candidates (JSON): {candidates}"""

CONSOLIDATION_BATCH_PROMPT = """You are the memory consolidation module of an AI agent.
Below are SEVERAL new facts, each with its own candidate memories. For EVERY
item decide the operation per candidate — same rules as ever:

- "NOOP": candidate already states the same fact, EVEN IF the wording differs.
- "UPDATE": candidate should be REPLACED by a new value the new fact provides.
- "DELETE": the new fact says the old one is no longer true and provides NO
  replacement value.
- no operation: keep both, genuinely different facts (near-duplicates with
  different phrasing are NOOP — be conservative about ADD).

Items with empty candidate lists need no decision.
Output STRICT JSON:
{{"results": [{{"fact": "<the item's fact text>", "operations": [{{"id": "...", "op": "NOOP|UPDATE|DELETE", "reason": "..."}}]}}]}}
Return one result per item, in order.

Items carry "slot" (the new fact's single-value attribute tag, or null).
Slots are hints: judge by meaning, not spelling; multi-value or
time-qualified facts stay ADD even when slots match.
Items (JSON): {items}"""


class BaseLLM:
    """Contract: return parsed JSON object for a chat prompt."""

    def complete_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class OpenAIChatLLM(BaseLLM):
    """Minimal stdlib client for POST {base_url}/chat/completions."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90.0):
        # 90s: reasoning models (deep thinking traces) occasionally exceed 60s
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        try:
            return self._post(payload)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404, 422):
                # many OpenAI-compatible servers (vLLM, one-api, ...) reject
                # response_format — retry without it
                payload.pop("response_format")
                return self._post(payload)
            raise

    def _post(self, payload: dict) -> Optional[Dict[str, Any]]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError:
            # let complete_json's response_format retry see real HTTP statuses
            # (HTTPError subclasses URLError — must not be swallowed here)
            raise
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach LLM endpoint {self.base_url}/chat/completions: {e}"
            ) from e
        message = data["choices"][0]["message"]
        content = message.get("content")
        if not content and message.get("reasoning_content"):
            # some reasoning models occasionally leave content empty
            content = message["reasoning_content"]
        return _loads_json(content or "")


def _loads_json(text: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON parsing (handles ```json fences and stray prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


# Suggested single-value slots (examples for prompts). The engine treats
# slot as an open hint — see slots.py. Kept for backwards compatibility.
from .slots import SUGGESTED_SLOTS as _SUGGESTED

VOLATILE_SLOTS = set(_SUGGESTED)


def make_llm(config) -> BaseLLM:
    """Real OpenAI-compatible chat client. Requires an API key — there is no
    alternate in-process backend; tests use a local protocol server at the HTTP boundary."""
    api_key = config.resolve_api_key()
    if not api_key:
        raise ValueError(
            "需要 LLM API key：设置 LLM_API_KEY 或 OPENAI_API_KEY（或在 MemoryConfig 里给 llm_api_key）")
    return OpenAIChatLLM(config.llm_base_url, api_key, config.llm_model)
