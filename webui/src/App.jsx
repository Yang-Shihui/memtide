import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api.js";
import "./style.css";
import Dashboard from "./views/Dashboard.jsx";
import MemoryList from "./views/MemoryList.jsx";
import SearchPlayground from "./views/SearchPlayground.jsx";
import ContextPanel from "./views/ContextPanel.jsx";
import Operations from "./views/Operations.jsx";
import Logo from "./components/Logo.jsx";
import { getApiKey, setApiKey } from "./api.js";

const TABS = [
  ["dashboard", "总览"],
  ["memories", "记忆库"],
  ["search", "检索试玩"],
  ["context", "核心记忆"],
  ["ops", "操作"],
];

function tabFromHash() {
  const h = (location.hash || "").replace("#", "");
  return TABS.some(([id]) => id === h) ? h : "dashboard";
}

export default function App() {
  const [tab, setTab] = useState(tabFromHash);
  const [toast, setToast] = useState(null);
  const [scope, setScope] = useState({ user_id: "", agent_id: "", run_id: "" });
  const [needKey, setNeedKey] = useState(false);
  const [keyInput, setKeyInput] = useState(getApiKey());

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2600);
    return () => clearTimeout(t);
  }, [toast]);

  const switchTab = (id) => {
    setTab(id);
    history.replaceState(null, "", id === "dashboard" ? "#" : `#${id}`);
  };

  useEffect(() => {
    const on401 = () => setNeedKey(true);
    window.addEventListener("memtide-unauthorized", on401);
    return () => window.removeEventListener("memtide-unauthorized", on401);
  }, []);

  const submitKey = () => {
    setApiKey(keyInput.trim());
    setNeedKey(false);
    setToast({ msg: "API Key 已保存，重试操作即可", error: false });
  };

  const notify = (msg, error = false) => setToast({ msg, error });
  const wrap = (fn) => (...args) => fn(...args).catch((e) => notify(e.message, true));

  const view = {
    scope, setScope, notify, wrap,
  };

  return (
    <>
      <div className="topbar">
        <div className="logo">
          <Logo size={34} />
          <span>Memtide<span>·</span>记忆管理台</span>
        </div>
        <div className="subtitle">agent long-term memory console</div>
      </div>
      <div className="tabs">
        {TABS.map(([id, label]) => (
          <button key={id} className={tab === id ? "active" : ""} onClick={() => switchTab(id)}>
            {label}
          </button>
        ))}
      </div>
      {tab === "dashboard" && <Dashboard view={view} />}
      {tab === "memories" && <MemoryList view={view} />}
      {tab === "search" && <SearchPlayground view={view} />}
      {tab === "context" && <ContextPanel view={view} />}
      {tab === "ops" && <Operations view={view} />}
      {toast && <div className={"toast" + (toast.error ? " error" : "")}>{toast.msg}</div>}
      {needKey && (
        <>
          <div className="mask" />
          <div className="modal">
            <h2>需要 API Key</h2>
            <p className="muted">服务端启用了 MEMTIDE_API_KEY，输入后将继续保存在本浏览器。</p>
            <input type="password" style={{ width: "100%" }} value={keyInput} autoFocus
                   placeholder="API Key" onChange={(e) => setKeyInput(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && submitKey()} />
            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn" onClick={submitKey}>保存</button>
              <button className="btn ghost" onClick={() => setNeedKey(false)}>稍后</button>
            </div>
          </div>
        </>
      )}
    </>
  );
}
