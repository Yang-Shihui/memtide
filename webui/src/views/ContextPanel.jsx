import React, { useState } from "react";
import { api } from "../api.js";

const EXAMPLE = `## Memory
- 用户的名字是林小满
- 用户住在上海
- 用户喜欢喝美式咖啡

## Relevant to current query
- 用户喜欢喝美式咖啡`;

export default function ContextPanel({ view }) {
  const { scope, notify, wrap } = view;
  const [userId, setUserId] = useState(scope.user_id || "default");
  const [query, setQuery] = useState("");
  const [block, setBlock] = useState(null);
  const [busy, setBusy] = useState(false);

  const doRender = wrap(async () => {
    setBusy(true);
    try {
      const res = await api.context(userId, query);
      setBlock(res.context);
    } finally {
      setBusy(false);
    }
  });

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(block);
      notify("已复制到剪贴板");
    } catch {
      notify("复制失败，请手动选择文本", true);
    }
  };

  return (
    <>
      <div className="panel">
        <div className="muted" style={{ marginBottom: 14 }}>
          把该用户的长期记忆压缩成一段文本，直接注入 agent 的 system prompt
          （Letta 式 core memory）。填入查询可在下方附加与问题最相关的记忆。
        </div>
        <div className="row">
          <input placeholder="user_id，如 demo" aria-label="user_id" style={{ width: 180 }} value={userId}
                 onChange={(e) => setUserId(e.target.value)} />
          <input placeholder="可选：用查询召回相关记忆，如 用户喜欢什么" aria-label="召回查询" style={{ flex: 1, minWidth: 240 }}
                 value={query} onChange={(e) => setQuery(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && !busy && doRender()} />
          <button className="btn" onClick={doRender} disabled={busy}>
            {busy ? "渲染中…" : "渲染核心记忆块"}
          </button>
        </div>
      </div>

      {block && (
        <div className="panel">
          <div className="row" style={{ marginBottom: 10 }}>
            <button className="btn ghost" onClick={copy}>复制全文</button>
            <span className="muted">{block.length} 字符 · 可直接拼入 system prompt</span>
          </div>
          <pre>{block}</pre>
        </div>
      )}

      {!block && (
        <div className="panel" style={{ textAlign: "center", padding: "44px 24px" }}>
          <div style={{ fontSize: 15, color: "var(--muted)", marginBottom: 10 }}>
            尚未渲染 —— 填好 user_id 后点击上方按钮
          </div>
          <pre style={{ display: "inline-block", textAlign: "left", maxWidth: 460 }}>{EXAMPLE}</pre>
          <div className="muted" style={{ marginTop: 10 }}>
            提示：试试 <code>demo</code> 用户（已填充演示数据）
          </div>
        </div>
      )}
    </>
  );
}
