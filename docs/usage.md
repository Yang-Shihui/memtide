# Memtide 使用说明（Usage Guide）

> 安装、快速上手、Python API / REST API / Web UI / CLI 全参考、部署与配置。
> 设计原理见 [design.md](design.md)，开发与测试见 [development.md](development.md)。

## 安装

```bash
# 方式一：pip 安装（含 UI 静态资源；psycopg 是唯一运行时依赖）
pip install .
source .env                 # MEMTIDE_PG_DSN + LLM/embedding keys
python -c "from memtide import MemoryEngine; MemoryEngine(); print('ok')"

# 方式二：Docker（PostgreSQL + Qdrant + REST + UI 生产栈）
docker compose up -d --build
```

Python ≥ 3.10。需要配置：**PostgreSQL**（`MEMTIDE_PG_DSN`）+ Qdrant + 任意 OpenAI 兼容
LLM/embedding 端点（`source .env` 即可，见 `.env.example`）。

## 30 秒上手（先 source .env：PG DSN + LLM/embedding 端点）

```python
from memtide import MemoryEngine, MemoryConfig

mem = MemoryEngine(config_from_env())               # 读 MEMTIDE_PG_DSN 等环境变量

# 写入：自动抽取原子事实 → 预测编码门控 → 冲突消解
res = mem.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id="alice")
print(res.facts)      # ['用户的名字是李雷', '用户住在杭州', '用户喜欢喝美式咖啡']
print(res.added)      # 新增记忆 id

# 变化：自动 UPDATE（不是并存！），旧值进审计日志
res = mem.add("我搬到上海了", user_id="alice")
print(res.updated)

# 检索：三路召回 RRF 融合，得分可解释
for h in mem.search("用户住哪", user_id="alice"):
    print(h.score, h.memory.text, h.components)

# 注入 system prompt 的核心记忆块（Letta 式）
print(mem.render_context(user_id="alice", query="用户的近况"))
```

完整多轮示例：[examples/quickstart.py](../examples/quickstart.py)、
[examples/chat_agent.py](../examples/chat_agent.py)。

## Python API 参考

### 写入 `add()`

```python
res = mem.add(
    messages,                      # str 或 [{'role','content'}]；content 支持 parts（见下）
    user_id="alice",               # 作用域：用户（必读参数，默认 "default"）
    agent_id=None, run_id=None,    # 作用域：agent / 会话（可选）
    metadata={"k": "v"},           # 附加到本批所有事实的元数据
    infer=True,                    # False = 原文直存（跳过抽取，保留原始对话）
    timestamp=None,                # ISO-8601：数据自身时间（导入历史对话用）；
                                   # 无时区按 UTC 读，一律归一为 UTC 存储
)
# res: AddResult
#   .facts     抽取出的候选事实
#   .added/.updated/.deleted  记忆 id 列表
#   .noop      被 NOOP 吸收的重复数
#   .rejected  被门控拦下的事实
#   .gate      {事实: {reason, surprise_bits, max_similarity}}
#   .attachments  本轮处理的媒体附件
```

**消息 content parts（多模态）**：

| part 类型 | 字段 | 说明 |
|---|---|---|
| `{"type":"text"}` | `text` | 普通文本 |
| `{"type":"image_url"}` | `image_url.url` | data URL / `https://` / 本地路径；可选 `caption` 直接给描述（跳过 vision 调用） |
| `{"type":"input_audio"}` | `input_audio.data/format` | base64 音频；配置 STT 后转文字 |
| `{"type":"file"}` | `url` / `path` | 任意文件仅存引用 |

```python
import base64
mem.add([{"role": "user", "content": [
    {"type": "text", "text": "看看这张图"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
]}], user_id="alice")
# 事实带 attachments 引用 → mem.search("图里有什么") 命中 → /media/{sha256} 取回原图
```

媒体规则：大小上限 10MB（`max_media_bytes`）；相同字节自动去重；vision 不可用
时素材照存、描述留空。**历史对话导入**一定带 `timestamp="2024-06-01T10:30:00"`，
让 created_at/valid_at/审计/遗忘曲线都用数据自身时间。

### 检索与读取

```python
mem.search(query, user_id=None, agent_id=None, run_id=None,
           limit=10, include_forgotten=False,
           memory_type=None, slot=None)     # → [SearchResult{score, components, memory}]
mem.get(memory_id)                              # → Memory | None
mem.get_all(user_id, agent_id=None, run_id=None, limit=100)
mem.render_context(user_id, agent_id=None, query=None)  # → 核心记忆块文本
mem.get_history(memory_id=None, limit=100)      # 审计事件（不带 id = 全库最近）
```

`memory_type`（fact/preference/episodic/procedural）与 `slot`（如 "location"；开放 hint，
过滤前经别名归一，`slot="city"` 也能命中 location 记忆）
为融合后过滤；`search` 命中会强化记忆（access_count++，越查越难忘）。

### 修改与删除

```python
mem.update(memory_id, "修正后的文本")   # 手动修正（拒绝已失效记忆，防复活）
mem.delete(memory_id)                   # 软删除：invalid_at + 审计保留
mem.delete(memory_id, hard=True)        # 彻底删除
```

### 反思与维护

```python
mem.consolidate_background(user_id="alice")  # 聚类→概括→取代（定期跑，如每 N 轮）
mem.compact(user_id="alice")                 # 近重复压实（纯向量，保留最优一条）
mem.media_gc(delete=False)                   # 孤儿媒体报告；delete=True 才真删
mem.enable_auto_reflect(3600)                # 每小时自动反思（全部用户）
mem.disable_auto_reflect()                   # 停止（close() 也会停止）
mem.rebuild_index()                          # 换 embedding 模型/向量库丢数据后重建
mem.stats()                                  # 计数 + 后端信息 + media_files
mem.reset()                                  # 清空（开发用）

# 后台写入（延迟敏感场景）：立即返回 Future，管线在线程池执行
fut = mem.add_background("我叫王五", user_id="bob")
result = fut.result(timeout=30)

# 导出 / 导入（JSONL，含审计状态；媒体文件需随 media_dir 拷贝）
mem.export_jsonl("dump.jsonl", user_id="alice")          # include_embeddings 默认 True
other.import_jsonl(open("dump.jsonl").read().splitlines(),
                   on_conflict="skip")                   # 或 "overwrite"
```

## REST API 参考

启动：`python -m memtide serve --port 8300`（读 `MEMTIDE_PG_DSN`、`MEMTIDE_QDRANT_URL` 等环境变量）
（Docker 栈已带，地址 `http://localhost:8300`）。

**交互式接口文档**：`http://localhost:8300/docs` —— 全部端点的参数表 + 示例 +
在线执行（自动携带浏览器保存的 API key）。

```bash
# 写入（文本 / 消息 / 多模态 parts / 时间戳）
curl -X POST localhost:8300/memories -H 'Content-Type: application/json' -d '{
  "text": "我叫李雷，住在杭州", "user_id": "alice"}'
curl -X POST localhost:8300/memories -d '{
  "messages": [{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"https://example.com/x.jpg"}}]}],
  "user_id": "alice"}'
curl -X POST localhost:8300/memories -d '{
  "text": "2024 年的历史对话", "user_id": "alice",
  "timestamp": "2024-03-01T09:00:00+08:00", "infer": false}'

# 检索
curl -X POST localhost:8300/search -d '{"query":"用户住哪","user_id":"alice","limit":3}'
#   → [{score, components:{rrf,semantic,bm25,entity,retention(+rerank)}, memory, attachments?}]

# 记忆 CRUD
curl "localhost:8300/memories?user_id=alice&limit=50"     # 列表（&include_invalid=true 含失效）
curl localhost:8300/memories/<id>                         # 单条（软删的 404，加 ?include_invalid=true 可见）
curl -X PUT localhost:8300/memories/<id> -d '{"text":"修正"}'
curl -X DELETE localhost:8300/memories/<id>               # 软删（?hard=true 彻底删）

# 记忆块 / 审计 / 媒体 / 反思 / 维护
curl "localhost:8300/context?user_id=alice&query=近况"
curl "localhost:8300/history?memory_id=<id>"
curl "localhost:8300/media/<sha256>"                      # 取回原始素材（不可变缓存）
curl -X POST localhost:8300/consolidate -d '{"user_id": "alice"}'  # 后台反思
curl -X POST localhost:8300/compact -d '{"user_id": "alice"}'      # 近重复压实
curl -X POST localhost:8300/media/gc -d '{"delete": true}'         # 孤儿媒体清理
curl -X POST localhost:8300/rebuild                       # 重建向量索引
curl "localhost:8300/export?user_id=alice&download=1"     # JSONL 导出（含审计+向量）
curl -X POST localhost:8300/import -d '{"lines": [{...}]}'
curl localhost:8300/stats
curl -X POST localhost:8300/reset -d '{"confirm":"RESET"}'   # 危险：清空全库
```

**鉴权**：服务端设 `MEMTIDE_API_KEY=xxx` 后，以上所有端点需带
`X-API-Key: xxx` 或 `Authorization: Bearer xxx`（静态页面本身公开，
浏览器控制台会弹出 key 输入框，保存后自动附带）。不设即开放（开发默认）。

错误约定：缺参/非法时间戳/超限媒体/非法 compact threshold → `400 {"error": ...}`；未认证 → 401（API 前缀；静态控制台保持公开）；
不存在 → 404；内部异常 → 500 JSON（仅异常类型 + 前 200 字符，不会断连）。

## Web UI（可视化管理台）

Docker 栈或 `MEMTIDE_STATIC_DIR` 指向构建产物时：`http://localhost:8300/` 是官网
落地页，管理台在 **`http://localhost:8300/console`**（`/ui/` 为兼容旧链接的别名）。五个页面：

| 页面 | 能做什么 |
|---|---|
| 总览 | 记忆数量/类型与门控分布/事件分布/后端状态（每 15 秒自动刷新，后台 tab 暂停） |
| 记忆库 | 列表、按作用域过滤、新建/编辑/软删/硬删；点详情看**审计时间线**（ADD→UPDATE→ACCESS 全链）与附件缩略图 |
| 检索试玩 | 输入 query 看融合得分成分条（semantic/bm25/entity/retention） |
| 核心记忆 | 预览实际注入 system prompt 的记忆块 |
| 操作 | 写入对话（可附图片）看门控决策可视化；跑后台反思；重建索引；重置 |

右上角作用域输入框（user/agent/run）对所有页面生效。

## CLI

```bash
python -m memtide add "我叫李雷，喜欢 Rust" --user alice     # 环境变量提供 PG/LLM 配置
python -m memtide search "用户喜欢什么语言" --user alice -k 3
python -m memtide list --user alice
python -m memtide context "用户的偏好"
python -m memtide history | stats | delete <id>
python -m memtide serve --port 8300
```

## Docker 部署

```bash
cp .env.example .env          # 填 LLM_* / DASHSCOPE_API_KEY
docker compose up -d --build
docker exec -it memtide-memtide-1 python /app/scripts/seed_demo.py   # 交互确认后播种；脚本化加 --yes
```

- 栈：`memtide`（REST+UI, :8300）+ `paradedb/paradedb:pg16`（PostgreSQL 16 + BM25）+ `qdrant`
- 卷：`memtide_pgdata` / `memtide_qdrantdata` / `memtide_mndata`（含 /data/media 媒体）
- **备份**：三个卷都要备（媒体文件不在数据库里）；恢复 = 恢复卷后 `rebuild_index()`
- 体检：`python scripts/live_check.py`（LLM 连通 / embedding 维度 / 全管线 / 冒烟）

## 配置参考

`MemoryConfig`（代码）与环境变量（部署）一一对应：

| 配置项 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| storage_backend | MEMTIDE_STORAGE | postgres | PostgreSQL 是唯一存储后端 |
| pg_dsn | MEMTIDE_PG_DSN | — | postgresql://user:pw@host:5432/db |
| vector_backend | MEMTIDE_VECTOR_BACKEND | qdrant | Qdrant（唯一向量后端） |
| qdrant_url | MEMTIDE_QDRANT_URL | http://localhost:6333 | |
| qdrant_collection | MEMTIDE_QDRANT_COLLECTION | memtide | 集合名 |
| llm_backend | LLM_BACKEND | openai | OpenAI 兼容端点（无离线模式） |
| llm_base_url/model/api_key | LLM_BASE_URL / LLM_MODEL / LLM_API_KEY | — | 任意 OpenAI 兼容端点 |
| embedding_backend | EMBEDDING_BACKEND | auto | auto/openai/dashscope |
| dashscope_api_key | DASHSCOPE_API_KEY | — | 兼容模式 key |
| multimodal_enabled | MEMTIDE_MULTIMODAL | 1 | 0 = 忽略媒体 parts |
| media_dir | MEMTIDE_MEDIA_DIR | memtide_media | 素材落盘目录 |
| media_allow_paths | MEMTIDE_MEDIA_ALLOW_PATHS | 0 | **安全开关**：允许 `{"path": 本地路径}` 读媒体。默认关闭——REST 部署下开启等于把本地文件暴露给调用方；仅可信内嵌场景（CLI/进程内 agent）开启 |
| vision_base_url/model | MEMTIDE_VISION_BASE_URL / MEMTIDE_VISION_MODEL | 继承主 LLM | 视觉端点（如 qwen-vl-max） |
| vision_api_key | MEMTIDE_VISION_API_KEY | 继承 LLM_API_KEY | |
| stt_model | MEMTIDE_STT_MODEL | 空 | 配置后音频才转写 |
| gate_enabled / 各阈值 | —（代码配置） | true / 0.5 / 2.5 bits | 门控，见 design.md §3.3 |
| gate_slot_scoped / gate_slot_floor | —（代码配置） | true / 0.40 | slot 范围先验开关与同槽冲突底线（经别名归一判定） |
| slot_aliases | —（代码配置） | {} | 追加自定义槽别名，合并覆盖内置表（slots.py） |
| entity_channel_weight / expansion_variant_weight | —（代码配置） | 0.5 / 0.5 | 实体通道与扩展变体的 RRF 融合权重 |
| max_half_life_mult / episodic_half_life_mult / episodic_floor | —（代码配置） | 4.0 / 0.5 / 0.05 | 遗忘曲线：拉伸封顶、分型半衰期、分型遗忘线 |
| mmr_lambda | —（代码配置） | 0.0 | >0 启用 MMR 多样性选择 |
| query_expansion | —（代码配置） | false | 检索前 LLM 生成翻译+改写变体（仅真实 LLM） |
| rerank_backend / rerank_base_url / rerank_model | —（代码配置） | none | cross-encoder 重排（Jina/Cohere 风格 /rerank） |
| consolidation_half_life_mult | —（代码配置） | 3.0 | 概括记忆半衰期倍数（抗衰） |
| api_key | MEMTIDE_API_KEY | 空 | 设置后 REST 全端点鉴权 |
| auto_reflect_seconds | MEMTIDE_AUTO_REFLECT | 0 | >0 时服务端周期自动反思（最小 60） |

关键可调项（代码配置）：`max_facts_per_turn=12`、`dedup_threshold=0.94`、
`conflict_threshold=0.55`、`half_life_days=45`、`retention_floor=0.02`、
`consolidation_similarity=0.45`。换 embedding 模型后运行
`python scripts/calibrate_gate.py` 重新标定门控阈值。

## 多模态记忆（细节）

- **为什么是"文字桥接"**：调研 Mem0（vision 转文字进文本管线）、MemOS（统一
  多模态记忆单元）、M3-Agent（原生视觉/听觉流编码）三条路线后选 Mem0 式——
  复用全部文本基础设施，文字查询跨模态命中，零新增依赖。详见 design.md §3.1/§7。
- **视觉端点**：默认复用主 LLM 配置；GLM-5.3-Flash 实测支持 OpenAI 图片格式。
  换专用视觉模型（如 DashScope qwen-vl-max）配 `MEMTIDE_VISION_*` 即可。
  注意 vision 模型普遍拒绝 <10px 的图片——测试请用真实尺寸图。
- **测试/演示无 vision 环境时**：在 part 上直接给 `caption` 字段即可跳过
  vision 调用（seed_demo 就是这样保证演示数据确定的）。

## 故障排查 FAQ

| 现象 | 原因与处理 |
|---|---|
| 检索结果里看不到旧记忆 | 遗忘曲线把它沉到 floor 之下了：`include_forgotten=True` 或检索命中它即"复忆" |
| 换了 embedding 模型后检索变差 | 向量维度/语义空间变了：POST /rebuild（或 `rebuild_index()`） |
| Qdrant 连不上 | 服务启动/请求会直接报错，不隐藏基础设施故障；检查 MEMTIDE_QDRANT_URL |
| Qdrant 集合维度不匹配 | 引擎自动重建集合后需要跑一次 rebuild_index() 回填 |
| 图片存了但 description 是 null | vision 端点不可用/拒绝（如图太小）；素材仍在，可换 MEMTIDE_VISION_* 或补 caption 重写 |
| 相似事实总被拦下（rejected 多） | 门控认为已被预测：合理行为；确认阈值可调 gate_redundant_bits |
| 启动报“请设置 MEMTIDE_PG_DSN” | PostgreSQL 是唯一后端：先起 docker compose 或指向已有 PG |
