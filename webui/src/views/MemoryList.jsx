import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import Attachments from "./Attachments.jsx";
import { fmtTime, relTime, CopyButton } from "../components/ui.jsx";

const TYPE_LABEL = { fact: "事实", preference: "偏好", episodic: "情景", procedural: "程序" };
const GATE_LABEL = { novel: "novel", integrate: "integrate",
  "volatile-update": "volatile", consolidated: "consolidated" };
const GATE_TIP = { novel: "novel：高惊喜，加权编码", integrate: "integrate：整合编码",
  "volatile-update": "volatile：易变属性更新", consolidated: "consolidated：反思概括" };

export function MemoryCard({ mem, scope, onDetail, onEdit, onDelete }) {
  const invalid = !!mem.invalid_at;
  const gate = mem.metadata?.gate;
  const surprise = mem.metadata?.surprise_bits;
  return (
    <div className={"mem" + (invalid ? " invalid" : "")}>
      <div className="meta">
        <span className={"dot t-" + (mem.memory_type || "fact")}
              title={TYPE_LABEL[mem.memory_type] || mem.memory_type} />
        <span className="badge">{TYPE_LABEL[mem.memory_type] || mem.memory_type}</span>
        {gate && <span className={"badge " + gate} title={GATE_TIP[gate] || gate}>{GATE_LABEL[gate] || gate}</span>}
        {mem.metadata?.slot && <span className="chip">slot:{mem.metadata.slot}</span>}
        {mem.source !== "conversation" && <span className="chip">{mem.source}</span>}
        <span className="chip">user:{mem.user_id}</span>
        {mem.agent_id && <span className="chip">agent:{mem.agent_id}</span>}
      </div>
      <div className="text">{mem.text}</div>
      <Attachments items={mem.attachments} compact />
      <div className="meta">
        <span className="barline" style={{ width: 130, margin: 0 }}>
          <span className="lbl">重要度</span>
          <div className="bar green">
            <div style={{ width: `${Math.round(mem.importance * 100)}%` }} />
          </div>
        </span>
        <span>×{mem.access_count} 次检索</span>
        {surprise !== undefined && <span title="预测编码门控惊喜值">surprise {surprise} bits</span>}
        <span title={fmtTime(mem.created_at)}>{relTime(mem.created_at)}</span>
        {invalid && <span className="badge invalid" title={fmtTime(mem.invalid_at)}>已失效 {relTime(mem.invalid_at)}</span>}
        {mem.superseded_by && (
          <a onClick={() => onDetail(mem.superseded_by)} style={{ cursor: "pointer" }}>
            superseded_by →
          </a>
        )}
        <span className="actions">
          <button onClick={() => onDetail(mem.id)}>详情</button>
          {!invalid && <button onClick={() => onEdit(mem)}>编辑</button>}
          {!invalid && (
            <button className="del" onClick={() => onDelete(mem)}>
              删除
            </button>
          )}
        </span>
      </div>
    </div>
  );
}

export default function MemoryList({ view }) {
  const { scope, setScope, notify } = view;
  const [mems, setMems] = useState(null);
  const [err, setErr] = useState("");
  const [q, setQ] = useState("");
  const [includeInvalid, setIncludeInvalid] = useState(false);
  const [detailId, setDetailId] = useState(null);
  const [editing, setEditing] = useState(null);
  const [editText, setEditText] = useState("");
  const [deleting, setDeleting] = useState(null);

  const load = async () => {
    setErr("");
    try {
      const list = await api.list({
        ...scope,
        include_invalid: includeInvalid ? "true" : "",
        limit: 200,
      });
      setMems(list);
    } catch (e) {
      setErr(e.message);
    }
  };

  useEffect(() => { load(); }, [scope, includeInvalid]);

  // 文本过滤在渲染时做：输入即时生效，也不逐键重发请求
  const shown = (mems || []).filter(
    (m) => !q || m.text.toLowerCase().includes(q.toLowerCase()));

  const softDelete = async (mem) => {
    try {
      await api.remove(mem.id, false);
      notify("已软删除（审计保留）");
      setDeleting(null);
      load();
    } catch (e) {
      notify(e.message, true);
    }
  };

  const hardDelete = async (mem) => {
    const t = prompt("输入 DELETE 彻底删除（不可恢复）：");
    if (t !== "DELETE") return;
    try {
      await api.remove(mem.id, true);
      notify("已彻底删除");
      setDeleting(null);
      load();
    } catch (e) {
      notify(e.message, true);
    }
  };

  const doEdit = async () => {
    try {
      await api.update(editing.id, editText);
      notify("已更新");
      setEditing(null);
      load();
    } catch (e) {
      notify(e.message, true);
    }
  };

  return (
    <>
      <div className="panel">
        <div className="row">
          <input placeholder="user_id" aria-label="按 user_id 过滤" style={{ width: 120 }}
                 value={scope.user_id}
                 onChange={(e) => setScope({ ...scope, user_id: e.target.value })} />
          <input placeholder="agent_id（可选）" aria-label="按 agent_id 过滤" style={{ width: 140 }}
                 value={scope.agent_id}
                 onChange={(e) => setScope({ ...scope, agent_id: e.target.value })} />
          <input placeholder="文本过滤…" aria-label="按文本过滤" style={{ flex: 1 }} value={q}
                 onChange={(e) => setQ(e.target.value)} />
          <label className="check">
            <input type="checkbox" checked={includeInvalid}
                   onChange={(e) => setIncludeInvalid(e.target.checked)} />
            含失效/被取代
          </label>
          <button className="btn ghost" onClick={load}>刷新</button>
        </div>
      </div>

      {err && <div className="err">{err}</div>}
      {!mems && !err && (
        <>{[0, 1, 2].map((i) => (
          <div className="mem" key={i}>
            <div className="skel" style={{ height: 16, width: 180 }} />
            <div className="skel" style={{ height: 22, width: "55%", marginTop: 12 }} />
            <div className="skel" style={{ height: 13, width: 260, marginTop: 14 }} />
          </div>
        ))}</>
      )}
      {mems && shown.length === 0 && <div className="empty">没有匹配的记忆</div>}
      {mems &&
        shown.map((m) => (
          <MemoryCard
            key={m.id}
            mem={m}
            onDetail={setDetailId}
            onEdit={(mem) => { setEditing(mem); setEditText(mem.text); }}
            onDelete={setDeleting}
          />
        ))}

      {deleting && (
        <>
          <div className="mask" onClick={() => setDeleting(null)} />
          <div className="modal">
            <h2>删除记忆</h2>
            <div className="text" style={{ fontSize: 15 }}>{deleting.text}</div>
            <div className="row" style={{ marginTop: 14 }}>
              <button className="btn" onClick={() => softDelete(deleting)}>软删除（保留审计）</button>
              <button className="btn del" onClick={() => hardDelete(deleting)}>彻底删除…</button>
              <button className="btn ghost" onClick={() => setDeleting(null)}>取消</button>
            </div>
            <div className="muted" style={{ marginTop: 10 }}>
              软删除后可在「含失效/被取代」中查看；彻底删除需再输入 DELETE 确认，不可恢复。
            </div>
          </div>
        </>
      )}

      {editing && (
        <>
          <div className="mask" onClick={() => setEditing(null)} />
          <div className="modal">
            <h2>编辑记忆</h2>
            <textarea value={editText} onChange={(e) => setEditText(e.target.value)} />
            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn" onClick={doEdit}>保存</button>
              <button className="btn ghost" onClick={() => setEditing(null)}>取消</button>
              <span className="muted">保存会同步更新向量索引并记入审计日志</span>
            </div>
          </div>
        </>
      )}

      {detailId && <DetailDrawer id={detailId} onClose={() => setDetailId(null)} onJump={setDetailId} />}
    </>
  );
}

function DetailDrawer({ id, onClose, onJump }) {
  const [mem, setMem] = useState(null);
  const [hist, setHist] = useState([]);
  const [err, setErr] = useState("");
  const [histErr, setHistErr] = useState("");

  useEffect(() => {
    let alive = true;
    setMem(null);
    setHist([]);
    setErr("");
    api.get(id, true)
      .then((m) => { if (alive) setMem(m); })
      .catch((e) => { if (alive) setErr(e.message); });
    // a failing history fetch must not masquerade as "无事件"
    api.history(id)
      .then((h) => { if (alive) setHist(h); })
      .catch((e) => { if (alive) setHistErr(e.message); });
    return () => { alive = false; };
  }, [id]);

  return (
    <>
      <div className="mask" onClick={onClose} />
      <div className="drawer">
        <button className="close" onClick={onClose}>×</button>
        <h2>记忆详情</h2>
        {err && <div className="err">{err}</div>}
        {!mem && !err && <div className="empty">加载中…</div>}
        {mem && (
          <>
            <div className="panel">
              <div className="text" style={{ fontSize: 15 }}>{mem.text}</div>
              <div className="kv"><span className="k">id</span>
                <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <code>{mem.id}</code>
                  <CopyButton text={mem.id} />
                </span>
              </div>
              <div className="kv"><span className="k">类型</span><span>{TYPE_LABEL[mem.memory_type] || mem.memory_type}</span></div>
              <div className="kv"><span className="k">slot</span><span>{mem.metadata?.slot || "—"}</span></div>
              <div className="kv"><span className="k">门控</span><span title={GATE_TIP[mem.metadata?.gate] || ""}>{GATE_LABEL[mem.metadata?.gate] || "—"}</span></div>
              <div className="kv"><span className="k">surprise</span><span>{mem.metadata?.surprise_bits ?? "—"} bits</span></div>
              <div className="kv"><span className="k">重要度 / 检索次数</span><span>{mem.importance} / {mem.access_count}</span></div>
              <div className="kv"><span className="k">user / agent / run</span><span>{mem.user_id} / {mem.agent_id || "—"} / {mem.run_id || "—"}</span></div>
              <div className="kv"><span className="k">valid_at</span><span>{fmtTime(mem.valid_at)}</span></div>
              <div className="kv"><span className="k">invalid_at</span><span>{mem.invalid_at ? fmtTime(mem.invalid_at) : "仍有效"}</span></div>
              {mem.superseded_by && (
                <div className="kv">
                  <span className="k">superseded_by</span>
                  <a style={{ cursor: "pointer" }} onClick={() => onJump(mem.superseded_by)}>
                    {mem.superseded_by.slice(0, 10)}…
                  </a>
                </div>
              )}
              {mem.entities?.length > 0 && (
                <div className="kv"><span className="k">实体</span><span>{mem.entities.join("、")}</span></div>
              )}
              {mem.attachments?.length > 0 && (
                <div className="kv"><span className="k">附件</span><Attachments items={mem.attachments} /></div>
              )}
            </div>
            <h2>审计时间线</h2>
            <div className="tl">
              {hist.map((e) => (
                <div className={"ev " + e.event} key={e.seq}>
                  <span className="head">{e.event}</span>
                  <span className="when">{fmtTime(e.created_at)}</span>
                  {(e.prev_value || e.new_value) && (
                    <div className="body">
                      {e.prev_value && <div>旧: {e.prev_value.slice(0, 60)}{e.prev_value.length > 60 && "…"}</div>}
                      {e.new_value && <div>新: {e.new_value.slice(0, 60)}{e.new_value.length > 60 && "…"}</div>}
                    </div>
                  )}
                </div>
              ))}
              {hist.length === 0 && !histErr && <div className="empty">无事件</div>}
              {histErr && <div className="err">审计时间线加载失败：{histErr}</div>}
            </div>
          </>
        )}
      </div>
    </>
  );
}
