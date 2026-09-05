import React, { useState } from "react";
import { api } from "../api.js";

const COMPONENTS = [
  ["semantic", "语义相似", "accent"],
  ["retention", "留存度", "green"],
  ["rrf", "RRF 融合", ""],
];

export default function SearchPlayground({ view }) {
  const { scope, notify } = view;
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(5);
  const [includeForgotten, setIncludeForgotten] = useState(false);
  const [memoryType, setMemoryType] = useState("");
  const [slot, setSlot] = useState("");
  const [hits, setHits] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const doSearch = async () => {
    if (!query.trim()) return;
    setBusy(true);
    setErr("");
    try {
      const res = await api.search({
        query,
        user_id: scope.user_id || undefined,
        agent_id: scope.agent_id || undefined,
        run_id: scope.run_id || undefined,
        limit,
        include_forgotten: includeForgotten,
        memory_type: memoryType || undefined,
        slot: slot || undefined,
      });
      setHits(res);
      if (res.length === 0) notify("无命中（可能已被遗忘，试试 include_forgotten）");
    } catch (e) {
      setErr(e.message);
    }
    setBusy(false);
  };

  return (
    <>
      <div className="panel">
        <div className="row">
          <input placeholder="输入查询，如：用户喜欢什么咖啡？" aria-label="检索查询" style={{ flex: 1 }}
                 value={query} onChange={(e) => setQuery(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && !busy && doSearch()} />
          <select value={limit} onChange={(e) => setLimit(+e.target.value)}>
            {[3, 5, 10, 20].map((n) => <option key={n} value={n}>top {n}</option>)}
          </select>
          <select value={memoryType} onChange={(e) => setMemoryType(e.target.value)}
                  title="按记忆类型过滤">
            <option value="">全部类型</option>
            <option value="fact">事实</option>
            <option value="preference">偏好</option>
            <option value="episodic">情景</option>
            <option value="procedural">程序</option>
          </select>
          <input placeholder="slot 过滤" aria-label="按 slot 过滤" style={{ width: 90 }} value={slot}
                 onChange={(e) => setSlot(e.target.value)} />
          <label className="check">
            <input type="checkbox" checked={includeForgotten}
                   onChange={(e) => setIncludeForgotten(e.target.checked)} />
            含已遗忘
          </label>
          <button className="btn" onClick={doSearch} disabled={busy}>
            {busy ? "检索中…" : "检索"}
          </button>
        </div>
        <div className="muted" style={{ marginTop: 8 }}>
          当前作用域：user={scope.user_id || "(全部)"} agent={scope.agent_id || "(全部)"} run={scope.run_id || "(全部)"}
          —— 在「记忆库」页顶部可修改
        </div>
      </div>

      {err && <div className="err">{err}</div>}
      {hits === null && !err && (
        <div className="panel" style={{ textAlign: "center", padding: "44px 20px" }}>
          <svg viewBox="0 0 48 48" width="44" height="44" style={{ marginBottom: 6 }} aria-hidden="true">
            <circle cx="20" cy="20" r="13" fill="none" stroke="#8FCBA8" strokeWidth="3.5" />
            <line x1="30" y1="30" x2="41" y2="41" stroke="#D9738F" strokeWidth="4" strokeLinecap="round" />
            <circle cx="16" cy="16" r="4" fill="#A9C7F2" opacity="0.7" />
          </svg>
          <div className="muted">输入查询，检索结果会展示 RRF 融合 / 语义相似 / 留存度的可解释得分条</div>
          <div className="muted" style={{ marginTop: 4 }}>试试「用户喜欢什么咖啡」，或切换类型/slot 过滤</div>
        </div>
      )}
      {hits && hits.length === 0 && <div className="empty">没有命中</div>}
      {hits &&
        hits.map((h) => (
          <div className="mem" key={h.id}>
            <div className="meta">
              <span className="badge">{(h.score * 100).toFixed(1)} 分</span>
              <span className="chip">user:{h.user_id}</span>
              <span className="chip">{new Date(h.created_at).toLocaleDateString()}</span>
            </div>
            <div className="text">{h.memory}</div>
            <div className="barline">
              <span className="lbl">语义相似</span>
              <div className="bar"><div style={{ width: `${(h.components.semantic || 0) * 100}%` }} /></div>
              <span className="val">{(h.components.semantic || 0).toFixed(2)}</span>
            </div>
            <div className="barline">
              <span className="lbl">留存度</span>
              <div className="bar green"><div style={{ width: `${(h.components.retention || 0) * 100}%` }} /></div>
              <span className="val">{(h.components.retention || 0).toFixed(2)}</span>
            </div>
            <div className="barline">
              <span className="lbl">RRF 融合</span>
              <div className="bar"><div style={{ width: `${Math.min(100, (h.components.rrf || 0) * 1500)}%` }} /></div>
              <span className="val">{(h.components.rrf || 0).toFixed(4)}</span>
            </div>
            <div className="meta" style={{ marginTop: 6 }}>
              {h.components.bm25 ? <span className="badge added">全文命中</span> : <span className="badge">无全文命中</span>}
              {h.components.entity ? <span className="badge integrate">实体命中</span> : <span className="badge">无实体命中</span>}
            </div>
          </div>
        ))}
    </>
  );
}
