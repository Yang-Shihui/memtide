import React, { useState } from "react";
import { api } from "../api.js";
import Attachments from "./Attachments.jsx";

const GATE_LABEL = {
  novel: "编码（高惊喜）",
  integrate: "编码（整合）",
  "volatile-update": "编码（属性更新）",
  redundant: "拦下（已预测到）",
  consolidated: "反思概括",
};

export default function Operations({ view }) {
  const { scope, notify, wrap } = view;
  // ---- quick add ----
  const [text, setText] = useState("");
  const [infer, setInfer] = useState(true);
  const [image, setImage] = useState(null); // data URL of an attached image
  const [addResult, setAddResult] = useState(null);
  const [busy, setBusy] = useState(false);

  const pickImage = (e) => {
    const f = e.target.files?.[0];
    e.target.value = ""; // allow re-picking the same file
    if (!f) return;
    if (f.size > 10 * 1024 * 1024) {
      notify("图片超过 10MB 上限", true);
      return;
    }
    const r = new FileReader();
    r.onload = () => setImage(r.result);
    r.readAsDataURL(f);
  };

  const doAdd = async () => {
    if (!text.trim() && !image) return;
    setBusy(true);
    try {
      const body = {
        user_id: scope.user_id || "default",
        agent_id: scope.agent_id || undefined,
        run_id: scope.run_id || undefined,
        infer,
      };
      if (image) {
        const content = [];
        if (text.trim()) content.push({ type: "text", text });
        content.push({ type: "image_url", image_url: { url: image } });
        body.messages = [{ role: "user", content }];
      } else {
        body.text = text;
      }
      const res = await api.add(body);
      setAddResult(res);
      setText("");
      setImage(null);
      notify(`写入完成：新增 ${res.added.length}，更新 ${res.updated.length}，拦下 ${res.rejected.length}`);
    } catch (e) {
      notify(e.message, true);
    }
    setBusy(false);
  };

  // ---- consolidation ----
  const [conReport, setConReport] = useState(null);
  const [conBusy, setConBusy] = useState(false);
  const doConsolidate = async () => {
    setConBusy(true);
    try {
      const rep = await api.consolidate(scope.user_id || "default");
      setConReport(rep);
      notify(`反思完成：合并 ${rep.clusters} 簇 / 吸收 ${rep.members_absorbed} 条`);
    } catch (e) {
      notify(e.message, true);
    }
    setConBusy(false);
  };

  // ---- maintenance: compact + media gc ----
  const [compactReport, setCompactReport] = useState(null);
  // shared busy gate: compact/rebuild/gc are heavy — don't let double-clicks
  // fire duplicates
  const [maintBusy, setMaintBusy] = useState(false);
  const maint = (fn) => async () => {
    if (maintBusy) return;
    setMaintBusy(true);
    try {
      await fn();  // wrap() already reports failures
    } finally {
      setMaintBusy(false);
    }
  };
  const doCompact = maint(wrap(async () => {
    const r = await api.compact(scope.user_id || "default");
    setCompactReport(r);
    notify(r.absorbed ? `压实完成：${r.clusters} 簇 / 吸收 ${r.absorbed} 条近重复` : "没有发现近重复");
  }));
  const doMediaGc = maint(wrap(async () => {
    const r = await api.mediaGc(true);
    notify(r.orphan ? `已清理 ${r.removed.length} 个孤儿媒体文件` : "没有孤儿媒体文件");
  }));

  // ---- rebuild index ----
  const doRebuild = maint(wrap(async () => {
    const data = await api.rebuild();
    notify(`索引重建完成：${data.reindexed} 条`);
  }));

  // ---- reset ----
  const doReset = maint(async () => {
    const t = prompt("这将删除【所有用户】的全部记忆且不可恢复！输入 RESET 确认：");
    if (t !== "RESET") return;
    wrap(async () => {
      await api.reset();
      notify("记忆库已重置");
    })();
  });

  return (
    <>
      <h2>快速写入</h2>
      <div className="panel">
        <textarea aria-label={'要写入的对话文本'} placeholder={'输入对话文本，如：我叫李雷，住在杭州，喜欢喝美式咖啡\n（走完整管线：抽取 → 预测编码门控 → 冲突消解）'}
                  value={text} onChange={(e) => setText(e.target.value)} />
        {image && (
          <div className="row" style={{ marginTop: 8, alignItems: "center" }}>
            <img className="thumb" src={image} alt="待写入图片" style={{ width: 64, height: 64 }} />
            <button className="btn ghost" onClick={() => setImage(null)}>移除图片</button>
          </div>
        )}
        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn" onClick={doAdd} disabled={busy || (!text.trim() && !image)}>
            {busy ? "写入中…" : "写入记忆"}
          </button>
          <label className="btn ghost" style={{ cursor: "pointer" }}>
            {image ? "换一张图片" : "附加图片"}
            <input type="file" accept="image/*" style={{ display: "none" }} onChange={pickImage} />
          </label>
          <label className="check">
            <input type="checkbox" checked={infer} onChange={(e) => setInfer(e.target.checked)} />
            智能抽取（关闭则原文直存）
          </label>
          <span className="muted">
            作用域：user={scope.user_id || "default"} agent={scope.agent_id || "—"}
          </span>
        </div>
      </div>

      {addResult && (
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>写入结果</h2>
          {addResult.facts.length === 0 && <div className="empty">未抽取到值得记忆的事实</div>}
          {addResult.facts.map((f) => {
            const g = addResult.gate[f] || {};
            const rejected = addResult.rejected.includes(f);
            return (
              <div key={f} style={{ padding: "7px 0", borderBottom: "1px dashed var(--border)" }}>
                <span className={"badge " + (rejected ? "rejected" : g.reason || "added")}>
                  {rejected ? GATE_LABEL[g.reason] || "拦下" : GATE_LABEL[g.reason] || "已编码"}
                </span>
                <code style={{ fontSize: 12.5 }}>{f}</code>
                <span className="muted" style={{ marginLeft: 8 }}>
                  {g.surprise_bits !== undefined && `${g.surprise_bits} bits`}
                  {g.max_similarity !== undefined && ` · 与最相似旧记忆 cos=${g.max_similarity}`}
                </span>
              </div>
            );
          })}
          {addResult.attachments?.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <span className="chip">媒体附件（视觉模型已转文字）</span>
              <Attachments items={addResult.attachments} />
            </div>
          )}
        </div>
      )}

      <h2>后台反思</h2>
      <div className="panel">
        <div className="row">
          <button className="btn" onClick={doConsolidate} disabled={conBusy}>
            {conBusy ? "反思中…" : "运行后台反思（聚类 → 概括 → 取代）"}
          </button>
        </div>
        {conReport && (
          <div style={{ marginTop: 10 }}>
            {conReport.clusters === 0 ? (
              <div className="muted">没有找到足够大的同主题簇（需要 ≥3 条相似记忆）</div>
            ) : (
              conReport.summaries.map((s) => (
                <div key={s.id} style={{ padding: "6px 0" }}>
                  <span className="badge consolidated">概括</span>
                  {s.text}
                  <div className="muted">吸收 {s.members.length} 条原始记忆（已入审计链）</div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <h2>维护</h2>
      <div className="panel">
        <div className="row">
          <button className="btn ghost" onClick={doCompact} disabled={maintBusy}>近重复压实</button>
          <button className="btn ghost" onClick={doMediaGc} disabled={maintBusy}>清理孤儿媒体</button>
          <button className="btn ghost" onClick={doRebuild} disabled={maintBusy}>重建向量索引</button>
          <button className="btn danger" onClick={doReset} disabled={maintBusy}>重置记忆库…</button>
        </div>
        <div className="muted" style={{ marginTop: 8 }}>
          压实=同作用域内 cos≥去重阈值的近重复只保留最优一条（审计链保留）；孤儿媒体=无任何记忆引用的落盘文件。
          重建索引用于更换 embedding 模型或向量库数据丢失后恢复（数据以关系库为准）。
        </div>
        {compactReport && compactReport.absorbed > 0 && (
          <div style={{ marginTop: 8 }}>
            {compactReport.kept.map((id) => (
              <div key={id}><span className="badge consolidated">保留</span><code>{id}</code></div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
