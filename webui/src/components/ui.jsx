// Shared small components (copy button) + time formatting helpers.

import React, { useState } from "react";

export function fmtTime(ts) {
  if (!ts) return "—";
  return ts.replace("T", " ").slice(0, 19);
}

export function relTime(ts) {
  if (!ts) return "—";
  const t = new Date(ts).getTime();
  if (Number.isNaN(t)) return fmtTime(ts);
  const diff = (Date.now() - t) / 1000;
  if (diff < 0) return "刚刚";
  if (diff < 60) return "刚刚";
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
  if (diff < 7 * 86400) return `${Math.floor(diff / 86400)} 天前`;
  return ts.slice(0, 10);
}

export function CopyButton({ text, label = "复制", doneLabel = "已复制" }) {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // clipboard API needs https/localhost; fall back for http-LAN use
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setDone(true);
    setTimeout(() => setDone(false), 1500);
  };
  return (
    <button className={"copybtn" + (done ? " ok" : "")} onClick={copy}
            title={typeof text === "string" && text.length > 40 ? "复制到剪贴板" : undefined}>
      {done ? doneLabel : label}
    </button>
  );
}
