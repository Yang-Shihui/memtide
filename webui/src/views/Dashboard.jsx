import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { relTime } from "../components/ui.jsx";

const EVENT_COLORS = {
  ADD: "var(--accent-deep)",
  UPDATE: "var(--apricot-deep)",
  ACCESS: "var(--muted)",
  CONSOLIDATE: "var(--lavender-deep)",
  DELETE: "var(--danger)",
};
const EVENT_ZH = { ADD: "新增", UPDATE: "更新", ACCESS: "检索", CONSOLIDATE: "反思", DELETE: "删除" };

const TYPE_COLORS = {
  fact: "var(--accent)",
  preference: "var(--pink-deep)",
  episodic: "var(--apricot)",
  procedural: "var(--blueberry)",
};
const TYPE_LABEL = { fact: "事实", preference: "偏好", episodic: "情景", procedural: "程序" };

const GATE_COLORS = {
  novel: "var(--accent)",
  integrate: "var(--blueberry)",
  "volatile-update": "var(--apricot)",
  consolidated: "var(--lavender-deep)",
  redundant: "var(--raspberry)",
};
const GATE_LABEL = {
  novel: "novel 高惊喜",
  integrate: "integrate 整合",
  "volatile-update": "volatile 属性更新",
  consolidated: "consolidated 反思概括",
  redundant: "redundant 拦下",
  unmarked: "未标记（原文直存/手动）",
};

function Donut({ counts }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (!total) return <div className="empty">暂无记忆</div>;
  const R = 52, C = 2 * Math.PI * R;
  let acc = 0;
  return (
    <div className="donutwrap">
      <svg viewBox="0 0 140 140" width="150" height="150" role="img" aria-label="记忆类型分布">
        <circle cx="70" cy="70" r={R} fill="none" stroke="var(--panel2)" strokeWidth="17" />
        {Object.entries(counts).map(([t, n]) => {
          const frac = n / total;
          const dash = `${frac * C} ${C}`;
          const el = (
            <circle key={t} cx="70" cy="70" r={R} fill="none"
                    stroke={TYPE_COLORS[t] || "var(--muted)"} strokeWidth="17"
                    strokeDasharray={dash} strokeDashoffset={-acc * C}
                    transform="rotate(-90 70 70)" strokeLinecap="butt">
              <title>{TYPE_LABEL[t] || t}：{n} 条</title>
            </circle>
          );
          acc += frac;
          return el;
        })}
        <text x="70" y="66" textAnchor="middle" className="donut-num">{total}</text>
        <text x="70" y="84" textAnchor="middle" className="donut-cap">条记忆</text>
      </svg>
      <div className="legend">
        {Object.entries(counts).map(([t, n]) => (
          <span className="lg" key={t}>
            <i style={{ background: TYPE_COLORS[t] || "var(--muted)" }} />
            {TYPE_LABEL[t] || t} <b>{n}</b>
          </span>
        ))}
      </div>
    </div>
  );
}

function GateBar({ counts, total }) {
  if (!total) return <div className="muted">暂无门控数据</div>;
  const order = ["novel", "integrate", "volatile-update", "consolidated", "redundant", "unmarked"];
  return (
    <>
      <div className="gatebar">
        {order.filter((g) => counts[g]).map((g) => (
          <div key={g} style={{ width: `${(counts[g] / total) * 100}%`, background: GATE_COLORS[g] }}
               title={`${GATE_LABEL[g]}：${counts[g]} 条`} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 10 }}>
        {order.filter((g) => counts[g]).map((g) => (
          <span className="lg" key={g}>
            <i style={{ background: GATE_COLORS[g] }} />
            {GATE_LABEL[g]} <b>{counts[g]}</b>
          </span>
        ))}
      </div>
    </>
  );
}

export default function Dashboard({ view }) {
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    let alive = true;
    let timer = null;
    const load = () => {
      Promise.all([api.stats(), api.history(null, 10)])
        .then(([s, h]) => { if (alive) { setStats(s); setRecent(h); } })
        .catch((e) => { if (alive) setErr(e.message); });
    };
    const tick = () => {
      if (document.hidden) return;  // don't poll a background tab
      load();
    };
    load();
    timer = setInterval(tick, 15000);  // stats + history only (server-side aggregates)
    return () => { alive = false; clearInterval(timer); };
  }, []);

  if (err) return <div className="err">{err}</div>;
  if (!stats) {
    return (
      <>
        <div className="grid cols3">
          {[0, 1, 2].map((i) => <div className="stat" key={i}><div className="skel" style={{ height: 40, width: 90 }} /><div className="skel" style={{ height: 13, width: 120, marginTop: 10 }} /></div>)}
        </div>
        <div className="grid cols2" style={{ marginTop: 22 }}>
          {[0, 1].map((i) => <div className="panel skel" key={i} style={{ height: 210 }} />)}
        </div>
      </>
    );
  }

  const events = stats.events || {};
  const maxEv = Math.max(1, ...Object.values(events));
  const backend = stats.backend || {};
  const typeCounts = stats.by_type || {};
  const gateCounts = stats.by_gate || {};
  const unmarked = (stats.active_memories || 0) - Object.values(gateCounts).reduce((a, b) => a + b, 0);
  if (unmarked > 0) gateCounts.unmarked = (gateCounts.unmarked || 0) + unmarked;

  return (
    <>
      <div className="grid cols3">
        <div className="stat">
          <div className="num">{stats.active_memories}</div>
          <div className="cap">有效记忆</div>
        </div>
        <div className="stat">
          <div className="num" style={{ color: "var(--apricot-deep)" }}>
            {stats.invalidated_memories}
          </div>
          <div className="cap">失效记忆（审计保留）</div>
        </div>
        <div className="stat">
          <div className="num" style={{ color: "var(--lavender-deep)" }}>
            {Object.values(events).reduce((a, b) => a + b, 0)}
          </div>
          <div className="cap">累计事件</div>
        </div>
      </div>

      <div className="grid cols2" style={{ marginTop: 22 }}>
        <div>
          <h2 style={{ marginTop: 0 }}>记忆构成</h2>
          <div className="panel">
            <Donut counts={typeCounts} />
          </div>
        </div>
        <div>
          <h2 style={{ marginTop: 0 }}>门控决策分布</h2>
          <div className="panel" style={{ paddingTop: 26 }}>
            <GateBar counts={gateCounts} total={stats.active_memories} />
            <div className="muted" style={{ marginTop: 14 }}>
              S = −log₂ p̂ —— 完全被旧记忆预测到的不编码；惊喜值越高，重要度加成越大。
            </div>
          </div>
        </div>
      </div>

      <div className="grid cols2" style={{ marginTop: 22 }}>
        <div>
          <h2 style={{ marginTop: 0 }}>事件分布</h2>
          <div className="panel">
            {Object.keys(events).length === 0 && <div className="empty">暂无事件</div>}
            {Object.entries(events)
              .sort((a, b) => b[1] - a[1])
              .map(([ev, n]) => (
                <div className="barline" key={ev}>
                  <span className="lbl" style={{ color: EVENT_COLORS[ev] || "var(--muted)" }}>
                    {EVENT_ZH[ev] || ev}
                  </span>
                  <div className="bar">
                    <div style={{ width: `${(n / maxEv) * 100}%`, background: EVENT_COLORS[ev] || "var(--accent)" }} />
                  </div>
                  <span className="val">{n}</span>
                </div>
              ))}
          </div>
          <h2>后端</h2>
          <div className="panel">
            <div className="kv"><span className="k">关系存储</span><span className="mono">{backend.storage}</span></div>
            <div className="kv"><span className="k">向量库</span><span className="mono">{backend.vector}</span></div>
            <div className="kv"><span className="k">LLM</span><span className="mono">{backend.llm}</span></div>
            <div className="kv"><span className="k">Embedding</span><span className="mono">{backend.embedding}</span></div>
            <div className="kv"><span className="k">向量维度</span><span className="mono">{backend.dim}</span></div>
          </div>
        </div>
        <div>
          <h2 style={{ marginTop: 0 }}>最近动态</h2>
          <div className="panel">
            {recent.length === 0 && <div className="empty">暂无事件</div>}
            {recent.map((e) => {
              const body = e.event === "ACCESS"
                ? "（记忆被检索命中，强化留存）"
                : (e.new_value || e.prev_value || "—");
              return (
                <div className="feed" key={e.seq}>
                  <span className={"badge " + (e.event === "DELETE" ? "rejected" : e.event === "UPDATE" || e.event === "CONSOLIDATE" ? "consolidated" : "added")}>
                    {EVENT_ZH[e.event] || e.event}
                  </span>
                  <span className="feed-body" title={body}>
                    {body.slice(0, 42)}{body.length > 42 && "…"}
                  </span>
                  <span className="feed-time" title={e.created_at}>{relTime(e.created_at)}</span>
                </div>
              );
            })}
            <div className="muted" style={{ marginTop: 10 }}>
              完整审计：<code>GET /history</code>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
