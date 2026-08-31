import React, { useEffect, useState } from "react";
import { api } from "../api";

const KIND_LABEL = { image: "图片", audio: "音频", video: "视频", file: "文件" };

// <img> tags can't carry X-API-Key, so fetch media through the authed API
// wrapper and hand the blob URL to the img element.
function AuthImage({ sha256, alt }) {
  const [src, setSrc] = useState(null);
  useEffect(() => {
    let alive = true;
    let url = null;
    api.mediaBlob(sha256)
      .then((u) => {
        if (!alive) {
          URL.revokeObjectURL(u);
          return;
        }
        url = u;
        setSrc(u);
      })
      .catch(() => {});
    return () => {
      alive = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [sha256]);
  if (!src) return <span className="thumb thumb-loading" aria-label={alt} />;
  return <img className="thumb" src={src} alt={alt} loading="lazy" />;
}

export default function Attachments({ items, compact = false }) {
  if (!items?.length) return null;
  return (
    <div className={"atts" + (compact ? " compact" : "")}>
      {items.map((a) => (
        <span className="att" key={a.id} title={a.description || a.mime || a.kind}>
          {a.kind === "image" && a.sha256 ? (
            <AuthImage sha256={a.sha256} alt={a.description || "image attachment"} />
          ) : (
            <span className="chip">{KIND_LABEL[a.kind] || a.kind}</span>
          )}
          {a.description && <span className="desc">{a.description}</span>}
        </span>
      ))}
    </div>
  );
}
