// Thin fetch wrappers over the memtide REST API.
// When the server runs with MEMTIDE_API_KEY, the console asks for the key once
// (401 -> event), stores it in localStorage and sends it as X-API-Key.
const KEY_STORAGE = "memtide-api-key";

export function getApiKey() {
  return localStorage.getItem(KEY_STORAGE) || "";
}

export function setApiKey(key) {
  if (key) localStorage.setItem(KEY_STORAGE, key);
  else localStorage.removeItem(KEY_STORAGE);
}

async function req(method, path, body) {
  const opt = { method, headers: {} };
  const key = getApiKey();
  if (key) opt.headers["X-API-Key"] = key;
  if (body !== undefined) {
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opt);
  if (resp.status === 401) {
    window.dispatchEvent(new CustomEvent("memtide-unauthorized"));
    throw new Error("需要 API Key（服务端返回 401 后会弹出输入框，保存后自动重试）");
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

export const api = {
  stats: () => req("GET", "/stats"),
  list: (scope = {}) => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(scope)) {
      if (v !== undefined && v !== null && v !== "") p.set(k, v);
    }
    return req("GET", `/memories?${p}`);
  },
  get: (id, includeInvalid = false) =>
    req("GET", `/memories/${id}${includeInvalid ? "?include_invalid=true" : ""}`),
  add: (body) => req("POST", "/memories", body),
  update: (id, text) => req("PUT", `/memories/${id}`, { text }),
  remove: (id, hard = false) => req("DELETE", `/memories/${id}${hard ? "?hard=true" : ""}`),
  search: (body) => req("POST", "/search", body),
  history: (memoryId, limit = 100) => {
    const p = new URLSearchParams();
    if (memoryId) p.set("memory_id", memoryId);
    p.set("limit", limit);
    return req("GET", `/history?${p}`);
  },
  context: (userId, query) => {
    const p = new URLSearchParams({ user_id: userId });
    if (query) p.set("query", query);
    return req("GET", `/context?${p}`);
  },
  consolidate: (userId) => req("POST", "/consolidate", { user_id: userId }),
  compact: (userId) => req("POST", "/compact", { user_id: userId }),
  mediaGc: (del = false) => req("POST", "/media/gc", { delete: del }),
  rebuild: () => req("POST", "/rebuild"),
  mediaBlob: async (sha256) => {
    // <img> tags can't carry X-API-Key; fetch media with auth ourselves
    const opt = { headers: {} };
    const key = getApiKey();
    if (key) opt.headers["X-API-Key"] = key;
    const resp = await fetch(`/media/${sha256}`, opt);
    if (resp.status === 401) {
      window.dispatchEvent(new CustomEvent("memtide-unauthorized"));
      throw new Error("需要 API Key（服务端返回 401 后会弹出输入框，保存后自动重试）");
    }
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return URL.createObjectURL(await resp.blob());
  },
  reset: () => req("POST", "/reset", { confirm: "RESET" }),
};
