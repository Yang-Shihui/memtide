"""Local fake OpenAI-compatible server for the hermetic test suite.

The engine speaks the real OpenAI protocol only;
the hermetic tests spin up THIS server instead. It reproduces deterministic
rule-based behavior at the HTTP boundary:

- POST /chat/completions  routes by the system prompt: fact extraction uses
  the bilingual rule set, single/batch consolidation uses cosine + polarity
  rules over the fake embedder, reflection distills a template summary,
  query expansion returns no variants.
- POST /embeddings        returns deterministic hashed n-gram vectors.

No network leaves the process; no keys required (auth is ignored).
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

from memtide.embeddings import cosine


def _hash_bucket(token: str, dim: int, salt: int = 0) -> int:
    import hashlib

    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=str(salt).encode()).digest()
    return int.from_bytes(h, "big") % dim


class FakeEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _tokens(self, text: str) -> List[str]:
        words = text.lower().split()
        tokens: List[str] = []
        for w in words:
            tokens.append("w:" + w.strip(".,!?;:'\"()[]{}"))
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        tokens.extend("c:" + ch for ch in cjk)
        tokens.extend("b:" + a + b for a, b in zip(cjk, cjk[1:]))
        for w in words:
            t = w.strip(".,!?;:'\"()[]{}")
            if len(t) >= 3 and t.isascii():
                tokens.extend("g:" + t[i: i + 3] for i in range(len(t) - 2))
        return [t for t in tokens if len(t) > 2]

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = self._tokens(text)
        for tok in toks:
            vec[_hash_bucket(tok, self.dim)] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


EMB = FakeEmbedder(256)

_ZH_PATTERNS = [
    (r"(?:我|俺)?(?:叫|名字叫|名字是)(?P<x>[\u4e00-\u9fa5A-Za-z·]{1,15})", "用户的名字是{x}", "fact", 0.95, "name"),
    (r"(?:我|俺)?(?:特别|很|比较|超级)?(?<!不)(?<!太不)(?:喜欢|爱|热爱|偏爱|偏好)(?P<x>[^。！？!?,，]{1,24})", "用户喜欢{x}", "preference", 0.7, "like"),
    (r"(?:我|俺)?(?:不|不太|很不|最不)(?:喜欢|爱|想)(?P<x>[^。！？!?,，]{1,24})", "用户不喜欢{x}", "preference", 0.75, "like"),
    (r"(?:我|俺)?(?:讨厌|恨|受不了)(?P<x>[^。！？!?,，]{1,24})", "用户不喜欢{x}", "preference", 0.75, "like"),
    (r"(?:我|俺)?(?:是|是一名|是个)(?P<x>[^。！？!?,，\s]{1,20}?(?:工程师|学生|老师|医生|设计师|产品经理|程序员|研究员|律师|作家|记者|创业者|经理|教师|护士|司机|画家|编辑|运营|开发|分析师|架构师))", "用户是{x}", "fact", 0.85, "role"),
    (r"(?:我|俺)?(?:在|供职于|就职于)(?P<x>[^。！？!?,，\s]{1,20}?)(?:工作|上班|任职)", "用户在{x}工作", "fact", 0.85, "employer"),
    (r"(?:我|俺)?(?:换到|跳槽到|加入了|入职)(?P<x>[^。！？!?,，]{1,25}?)(?:工作)?\s*$", "用户在{x}工作", "fact", 0.85, "employer"),
    (r"(?:我|俺)?(?:搬到|搬家到|迁到|迁往|移居到?)(?P<x>[^。！？!?,，\s]{1,15})", "用户住在{x}", "fact", 0.9, "location"),
    (r"(?:我|俺)?(?:住在|家在|定居在|现居于|居住在)(?P<x>[^。！？!?,，\s]{1,15})", "用户住在{x}", "fact", 0.9, "location"),
    (r"(?:我|俺)(?:今年)?(?P<x>[0-9一二两三四五六七八九十百]{1,4})岁", "用户{x}岁", "fact", 0.8, "age"),
    (r"(?:我|俺)的(?P<k>[^。！？!?,，\s]{1,10}?)(?:是|为|叫)(?P<v>[^。！？!?,，\s]{1,20})", "用户的{k}是{v}", "fact", 0.7, None),
    (r"(?:我|俺)?(?:每周|每星期)(?P<x>[^。！？!?,，]{1,24})", "用户每周做{x}", "preference", 0.6, "routine_weekly"),
    (r"(?:我|俺)?每天(?P<x>[^。！？!?,，]{1,24})", "用户每天做{x}", "preference", 0.6, "routine_daily"),
    (r"(?:我|俺)?(?:去)?吃了(?P<x>[^。！？!?,，]{1,20})", "用户吃了{x}", "episodic", 0.5, None),
    (r"(?:我|俺)?(?:去了|来到)(?P<x>[^。！？!?,，]{1,20})", "用户去了{x}", "episodic", 0.5, None),
    (r"(?:请?记住|帮我记下|记一下)[:,：,，]?(?P<x>[^。！？!?,，]{2,40})", "{x}", "fact", 0.9, None),
    (r"(?:我|俺)?正在(?P<x>[^。！？!?,，]{1,30})", "用户正在{x}", "fact", 0.6, "plan"),
    (r"(?:我|俺)?(?:打算|准备|计划)(?P<x>[^。！？!?,，]{1,30})", "用户计划{x}", "fact", 0.6, "plan"),
    (r"(?:我|俺)?(?:正在)?(?:在用|用|在学|学|使用)(?P<x>Python|JavaScript|TypeScript|Java|Go|Rust|C\+\+|Ruby|PHP|Kotlin|Swift)\b", "用户使用{x}", "fact", 0.6, "stack"),
]

_EN_PATTERNS = [
    (r"\bmy name(?:'s| is)\s+(?P<x>[A-Za-z][\w'-]{1,29})", "The user's name is {x}", "fact", 0.95, "name"),
    (r"\bi(?:'m| am)\s+(?P<x>[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,3}?)\s*$", "The user is {x}", "fact", 0.7, "role"),
    (r"\bi\s+(?:really\s+)?(?:like|love|enjoy|prefer)\s+(?P<x>[^.,!?\n]{1,40})", "The user likes {x}", "preference", 0.7, "like"),
    (r"\bi\s+(?:hate|dislike|can't stand|don't like)\s+(?P<x>[^.,!?\n]{1,40})", "The user dislikes {x}", "preference", 0.75, "like"),
    (r"\bi\s+(?:work|am working)\s+(?:at|for)\s+(?P<x>[^.,!?\n]{1,30})", "The user works at {x}", "fact", 0.85, "employer"),
    (r"\bi\s+live\s+in\s+(?P<x>[^.,!?\n]{1,30})", "The user lives in {x}", "fact", 0.9, "location"),
    (r"\bi(?:'m| am)\s+(?P<x>\d{1,2})\s+years?\s+old\b", "The user is {x} years old", "fact", 0.8, "age"),
    (r"\bmy\s+(?P<k>[a-z][\w'-]{1,15})\s+(?:is|are)\s+(?P<v>[^.,!?\n]{1,40})", "The user's {k} is {v}", "fact", 0.7, None),
    (r"\b(?:remember|note)\s+that\s+(?P<x>[^.!?\n]{3,60})", "{x}", "fact", 0.9, None),
    (r"\bi(?:'m| am)\s+(?:working on|building|learning|studying)\s+(?P<x>[^.,!?\n]{1,40})", "The user is working on {x}", "fact", 0.6, "plan"),
    (r"\bi\s+use\s+(?P<x>Python|JavaScript|TypeScript|Java|Go|Rust|C\+\+|Ruby|PHP|Kotlin|Swift)\b", "The user uses {x}", "fact", 0.6, "stack"),
]

_ALL_PATTERNS = (
    [(re.compile(p), tpl, t, imp, slot) for p, tpl, t, imp, slot in _ZH_PATTERNS]
    + [(re.compile(p, re.IGNORECASE), tpl, t, imp, slot) for p, tpl, t, imp, slot in _EN_PATTERNS]
)

_NEG = re.compile(r"dislikes|hates|不(?:喜欢|想)|讨厌|恨")


def extract_facts_rules(conversation: str) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    seen = set()
    for raw in re.split(r"[。！？!?，,；;\n]+", conversation):
        line = re.sub(r"[（(][^）)]*[)）]", "", raw).strip()
        if not line:
            continue
        for pat, tpl, mtype, imp, slot in _ALL_PATTERNS:
            for m in pat.finditer(line):
                gd = m.groupdict()
                try:
                    text = tpl.format(**gd).strip()
                except (KeyError, IndexError):
                    continue
                text = re.sub(r"[了呢吗吧嘛啊]+$", "", text)
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                ents = [re.sub(r"[了呢吗吧嘛啊]+$", "", v).strip()
                        for v in gd.values() if v and len(v.strip()) > 1]
                facts.append({"text": text, "type": mtype, "importance": imp,
                              "entities": [e for e in ents if len(e) > 1], "slot": slot})
    return facts


def _shared_entityish(a: str, b: str) -> bool:
    strip = re.compile(r"不(?:喜欢|想)|不太|很不|最不|讨厌|恨|受不了|dislikes|hates|can't stand|don't like", re.I)
    ta, tb = strip.sub("", a), strip.sub("", b)
    if len(set(re.findall(r"[\u4e00-\u9fff]", ta)) & set(re.findall(r"[\u4e00-\u9fff]", tb))) >= 3:
        return True
    wa = {w for w in re.findall(r"[a-zA-Z]{3,}", ta.lower()) if w not in {"user", "the"}}
    wb = {w for w in re.findall(r"[a-zA-Z]{3,}", tb.lower()) if w not in {"user", "the"}}
    return bool(wa & wb)


def consolidate_rules(new_fact: str, candidates: List[Dict[str, Any]],
                      new_fact_slot: Any = None) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for c in candidates:
        sim = cosine(EMB.embed(new_fact), EMB.embed(c.get("text", "")))
        if sim >= 0.995:
            ops.append({"id": c["id"], "op": "NOOP", "reason": "exact duplicate"})
            continue
        if sim >= 0.88:
            ops.append({"id": c["id"], "op": "UPDATE", "reason": f"near-duplicate (cos={sim:.2f})"})
            continue
        if sim >= 0.55:
            new_neg = bool(_NEG.search(new_fact))
            old_neg = bool(_NEG.search(c.get("text", "")))
            if new_neg != old_neg and _shared_entityish(new_fact, c.get("text", "")):
                ops.append({"id": c["id"], "op": "UPDATE", "reason": f"polarity conflict (cos={sim:.2f})"})
                continue
        # Open-hint slot: same canonical slot is a strong UPDATE signal, but
        # not blind — require a similarity floor so two homes / Rust+Python
        # (multi-value, time-qualified) stay ADD. Mirrors gate_slot_floor.
        if (c.get("slot") and new_fact_slot
                and str(c.get("slot")).strip().lower() == str(new_fact_slot).strip().lower()
                and new_fact != c.get("text") and sim >= 0.35):
            ops.append({"id": c["id"], "op": "UPDATE",
                        "reason": f"slot '{c['slot']}' updated (cos={sim:.2f})"})
    return ops


def _summary(facts: List[str]) -> str:
    shown = [t if len(t) <= 40 else t[:38] + "…" for t in facts[:5]]
    zh = len(re.findall(r"[\u4e00-\u9fff]", " ".join(facts))) > len(re.findall(r"[a-zA-Z]", " ".join(facts)))
    return (f"综合记忆（{len(facts)}条同主题事实）：{'；'.join(shown)}" if zh
            else f"Summary of {len(facts)} related memories: {'; '.join(shown)}")


def _chat(body: Dict[str, Any]) -> str:
    system = body["messages"][0]["content"]
    user = body["messages"][-1]["content"]
    if "memory extraction engine" in system:
        facts = extract_facts_rules(user if isinstance(user, str) else "")
        return json.dumps({"facts": facts}, ensure_ascii=False)
    if "memory consolidation module" in system:
        if "SEVERAL new facts" in system:  # batch
            items = json.loads(user.split("Items (JSON):", 1)[1])
            results = []
            for item in items:
                ops = consolidate_rules(item["fact"], item["candidates"],
                                        new_fact_slot=item.get("slot"))
                results.append({"fact": item["fact"], "operations": ops})
            return json.dumps({"results": results}, ensure_ascii=False)
        fact = user.split("New fact:", 1)[1].split("\nNew fact slot:", 1)[0].strip()
        slot_m = re.search(r"New fact slot: (\w+)", user)
        cands = json.loads(user.split("Candidates (JSON):", 1)[1])
        return json.dumps({"operations": consolidate_rules(fact, cands,
                          new_fact_slot=slot_m.group(1) if slot_m else None)},
                          ensure_ascii=False)
    if "background memory reflection" in system:
        facts = json.loads(user.split("Facts:", 1)[1])
        return json.dumps({"summary": _summary(facts)}, ensure_ascii=False)
    return json.dumps({"variants": []}, ensure_ascii=False)  # query expansion etc.


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else {}
        if self.path.endswith("/embeddings"):
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            vecs = [EMB.embed(t) for t in texts]
            out = {"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]}
        else:
            content = _chat(body)
            out = {"choices": [{"message": {"content": content}}]}
        resp = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass


def start_fake_server():
    """Start once per test process; returns the base URL ('http://host:port')."""
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}"
