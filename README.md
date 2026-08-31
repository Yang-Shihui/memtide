<div align="center">
  <img src="docs/assets/logo.svg" width="120" alt="Memtide logo"/>

# Memtide

[![CI](https://github.com/Yang-Shihui/memtide/actions/workflows/ci.yml/badge.svg)](https://github.com/Yang-Shihui/memtide/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**轻量级 Agent 记忆引擎** — 生产级 PostgreSQL + Qdrant、可插拔 LLM

波峰 M：字母的两个峰是两道潮，中央负空间的潮谷里藏着一块品牌绿——涨潮强化，落潮遗忘。

</div>

> **License note**: Memtide is MIT. The Docker image ships PostgreSQL with the
> ParadeDB `pg_search` extension (AGPL-3.0 community edition) — it runs as a
> separate database process, so your application code stays MIT.

**运行时固定使用 PostgreSQL + Qdrant + 真实 OpenAI 兼容端点**；核心
HTTP 客户端只用 Python 标准库，PostgreSQL 驱动 psycopg 是唯一运行时依赖。

Memtide 融合了 2026 年主流记忆框架的核心设计，外加一个原创机制：

| 借鉴自 | 设计 | 在 Memtide 中 |
|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | 记忆是**原子事实**而非原始消息；写时 LLM 决策 ADD/UPDATE/DELETE/NOOP | `engine.add()` 两阶段管线 |
| [Letta (MemGPT)](https://github.com/letta-ai/letta) | **常驻要点**：render_context() 按重要度×留存度实时选出 core memory 块注入 system prompt | `render_context()` |
| [Zep / Graphiti](https://github.com/getzep/graphiti) | **时间线审计**：失效不删除，`valid_at`/`invalid_at` 双时标 + 全量事件日志 | `get_history()` |
| LangMem / 认知科学 | **检索强化 + 遗忘曲线**：访问越多记得越牢，久不访问自然淡忘 | `decay.py`（Ebbinghaus） |
| [预测编码](docs/predictive-coding-gate.md)（Rao & Ballard; Itti & Baldi; van Kesteren） | **预测误差门控编码**：完全被先验预测到的信息不写入，越意外记得越牢 | `gating.py`（PredictiveGate） |

## 快速开始

```python
from memtide import MemoryEngine, MemoryConfig

# 生产模式：PostgreSQL + 任意 OpenAI 兼容 LLM/embedding 端点
mem = MemoryEngine(MemoryConfig(
    pg_dsn="postgresql://memtide:pw@localhost:5432/memtide",
    llm_base_url="http://your-llm/v1", llm_model="glm-5.3-flash",
    llm_api_key="sk-...", embedding_backend="dashscope", dashscope_api_key="sk-..."))

# 1. 写入：对话 → 原子事实 → 与旧记忆冲突消解
mem.add([{"role": "user", "content": "嗨，我叫李雷，住在杭州，喜欢喝美式咖啡"}], user_id="alice")
# → AddResult(facts=['用户的名字是李雷', '用户住在杭州', '用户喜欢喝美式咖啡'], added=[...])

# 2. 事实变化时自动 UPDATE（不是并存！），旧的值进入审计日志
mem.add("我搬到上海了", user_id="alice")
# → AddResult(updated=[...])  '用户住在杭州' 被替换为 '用户住在上海'

# 3. 混合检索：向量 + BM25 全文 + 实体三路召回，RRF 融合后按留存度+重要度重排，每路得分可解释
hits = mem.search("用户现在住在哪里？", user_id="alice")
# → [{'memory': '用户住在上海', 'score': 0.24,
#     'components': {'rrf': 0.02, 'semantic': 0.49, 'bm25': 0, 'entity': 0, 'retention': 1.0}}]

# 4. 注入 system prompt 的核心记忆块（Letta 风格）
system_prompt = "你是助理。\n" + mem.render_context(user_id="alice", query="用户的职业？")

# 5. 全量审计：ADD → UPDATE → ACCESS，每步带 prev/new 值
mem.get_history(memory_id=hits[0]["id"])
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/design.md](docs/design.md) | **设计方案**：架构总览、写入/检索/门控/反思/多模态各子系统详细设计与取舍 |
| [docs/usage.md](docs/usage.md) | **使用说明**：Python/REST API 全参考、Web UI、Docker 部署、配置表、FAQ |
| [docs/development.md](docs/development.md) | **开发文档**：环境搭建、测试体系、代码约定、扩展后端、UI 构建 |
| [docs/predictive-coding-gate.md](docs/predictive-coding-gate.md) | 预测编码门控的数学推导与实测标定 |

### 接入真实 LLM / embedding / 存储后端（可选）

```python
cfg = MemoryConfig(
    # 存储：PostgreSQL（唯一后端，pg_search BM25 全文检索）
    storage_backend="postgres", pg_dsn="postgresql://memtide:pw@localhost:5432/memtide",
    # 向量：Qdrant ANN（urllib 直连 REST，零 SDK）
    vector_backend="qdrant", qdrant_url="http://localhost:6333",
    # LLM：任何 OpenAI 兼容端点
    llm_backend="openai", llm_base_url="http://your-llm/v1",
    llm_model="GLM-5.3-Flash", llm_api_key="sk-...",
    # embedding：DashScope（qwen3.7-text-embedding），维度自动探测
    embedding_backend="dashscope", dashscope_api_key="sk-...",
)
mem = MemoryEngine(cfg)
```

真 LLM 下抽取/消解/反思由 LLM 完成，并输出 slot（易变属性）标签驱动冲突
更新；真 embedding 下检索为真语义向量（实测中文查询可跨语言命中英文记忆，
DashScope qwen3.7-text-embedding 同主题 cos≈0.56 / 跨主题 cos≈0.28）。

### 更换 LLM / embedding 端点

Memtide 与任何 OpenAI 兼容端点协作（DeepSeek、Qwen、vLLM、Ollama 等）：

```python
cfg = MemoryConfig(
    llm_backend="openai",
    llm_base_url="https://api.deepseek.com/v1",  # 或本地 vLLM/Ollama
    llm_model="deepseek-chat",
    llm_api_key="sk-...",          # 或环境变量 OPENAI_API_KEY / LLM_API_KEY
    embedding_backend="dashscope",  # 或 openai
)
mem = MemoryEngine(cfg)
```

抽取与冲突消解由 LLM 完成（提示词见 `llm.py`）；引擎只讲真实 OpenAI 协议，
测试用本地协议服务器（`tests/fake_openai.py`）保持确定性。

## Docker 部署

```bash
cp .env.example .env   # 填入 LLM / DASHSCOPE key
docker compose up -d --build
curl localhost:8300/stats   # 返回真实 LLM / embedding / PostgreSQL / Qdrant 名称
```

栈组成：`paradedb/paradedb:pg16`（PostgreSQL 16 + pg_search BM25 全文检索）、`qdrant`（向量 ANN 索引，
embedding 模型变更时自动重建集合）、`memtide`（REST 服务）。数据分别在
pgdata/qdrantdata volume 中持久化。

**真实端点验证**：`python3 scripts/live_check.py --docker` 顺序检查 LLM 连通、
embedding 维度、真实后端全链路（抽取/门控/检索/易变属性更新/LLM 反思/审计链）、
Docker REST 冒烟。真端点集成测试 `tests/test_live.py`（`MEMTIDE_LIVE=1` 门控，
默认跳过）。

## 可视化管理台（Web UI）

马卡龙浅色主题（奶油底 + 开心果绿/草莓粉/蓝莓/薰衣草语义徽章），设计规范由
[ui-ux-pro-max skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill)
约束：语义色 token、文本对比度 ≥4.5:1、8px 间距体系、可见焦点环、
prefers-reduced-motion 支持。演示数据可用 `scripts/seed_demo.py` 一键播种
（在容器内执行，幂等）。

REST 服务内置 React 管理台：启动后 **`http://localhost:8300/`** 是官网，**`/console`** 是管理台（五个页面），**`/docs`** 是可在线执行的接口文档：

| 页面 | 功能 |
|---|---|
| 总览 | 记忆数/失效数/事件分布图、后端信息 |
| 记忆库 | 过滤浏览（含失效/被取代记忆）、重要度条、门控徽章、surprise 值；编辑/删除（软/硬）/详情 |
| 详情抽屉 | 完整字段 + **审计时间线**（ADD→UPDATE→ACCESS→CONSOLIDATE 可视化）+ superseded_by 跳转 |
| 检索试玩 | query → 命中结果带**得分成分条**（语义/留存度/RRF/全文/实体命中），直观理解混合检索排序 |
| 核心记忆 | 渲染 `render_context()` 输出块预览 + 复制 |
| 操作 | 快速写入并可视化 AddResult（每条事实的编码/拦截决策 + surprise bits）、一键后台反思、重建索引、重置库 |

开发模式：`cd webui && npm install && npm run dev`（Vite 5173 端口，API 代理到
8300）。构建：`npm run build` 产物拷入 `memtide/static/`（Docker 镜像多阶段构建
自动完成）。`MEMTIDE_STATIC_DIR` 可自定义静态目录；目录不存在时服务退化为纯 API。

### REST API 服务

内置零依赖 HTTP 服务（标准库 `http.server` 实现），一行启动：

```bash
source .env && python -m memtide serve --port 8300
```

所有能力都以 JSON 接口暴露，任何语言的 agent 都能接入：

```bash
# 写入（带预测编码门控 + 冲突消解，返回完整决策明细）
curl -X POST localhost:8300/memories \
  -d '{"text": "我叫李雷，住在杭州", "user_id": "alice"}'

# 混合检索（返回可解释的 components 得分）
curl -X POST localhost:8300/search \
  -d '{"query": "用户住在哪里", "user_id": "alice", "limit": 3}'

# 其他接口
curl localhost:8300/memories?user_id=alice          # 列出
curl localhost:8300/memories/<id>                   # 单条
curl -X PUT localhost:8300/memories/<id> -d '{"text": "..."}'   # 修正
curl -X DELETE localhost:8300/memories/<id>         # 软删除（?hard=true 彻底删）
curl "localhost:8300/context?user_id=alice&query=..."           # core memory 块
curl -X POST localhost:8300/consolidate -d '{"user_id": "alice"}'  # 后台反思
curl localhost:8300/history                         # 审计日志
curl localhost:8300/stats
```

### 多模态记忆

OpenAI content-parts 格式的图片 / 音频 / 文件可以直接进管线。做法与 Mem0 一致
（业界主流）：**写入时把媒体归一化为文字描述**（可配置的 vision/STT 端点），
描述文本走正常的抽取 → 门控 → 冲突消解管线，原始字节以 sha256 内容寻址落盘为
附件，随记忆一起返回。文字查询即可跨模态召回图片记忆：

```python
eng.add([{"role": "user", "content": [
    {"type": "text", "text": "看看这张图"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},  # 或 https:// / 本地路径
]}], user_id="alice")
# → 抽出的原子事实带 attachments 引用（+ metadata.modality）
# → 检索 "图里有什么" 返回事实 + GET /media/{sha256} 取回原图
```

```bash
curl -X POST localhost:8300/memories -d '{"messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}]}]}'
curl localhost:8300/media/<sha256>     # 取回原始素材（不可变缓存）
```

vision 端点默认复用主 LLM 配置（任何 OpenAI 兼容视觉模型，如 qwen-vl-max），
可用 `MEMTIDE_VISION_BASE_URL` / `MEMTIDE_VISION_MODEL` / `MEMTIDE_VISION_API_KEY`
单独指定；音频转录需配置 `MEMTIDE_STT_MODEL`，未配置时音频仅存引用不转写；
vision 不可用时优雅降级——素材照存、描述留空，绝不丢数据。

### CLI

```bash
source .env && python -m memtide add "我叫李雷，喜欢 Rust"
python -m memtide search "用户喜欢什么语言"
python -m memtide context "用户的偏好"   # 渲染核心记忆块
python -m memtide history               # 审计日志
python -m memtide stats
```

## 架构

```
写入:  messages ──▶ Extractor ──▶ 原子事实(带 slot/重要度/实体)
                                        │
                          PredictiveGate 预测编码门控 ◀── 记忆先验
                          (完全被预测到的 → 拦下；越意外 → 重要度越高)
                                        │
                          对相似旧记忆做冲突消解 ◀── LLM/规则
                                        │
                          ADD / UPDATE / DELETE / NOOP
                                        │
存储:  PostgreSQL ── memories + BM25(中英可查) + entities + memory_history
                                        │
读取:  query ──▶ ┌ 向量余弦(top-40) ┐
                 ├ BM25 全文                 ├─▶ RRF 融合 ─▶ + 遗忘曲线留存度
                 └ 实体索引匹配     ┘              + 重要度
                                        │
                 检索命中 ──▶ access_count++ (间隔效应: 越用越难忘)
```

### 关键机制

- **预测编码门控**：记忆库即生成式先验，新事实按预测误差 S = -log2 p̂ 分流——
  S ≤ 0.5 bits 不存（完全被预测到）、中间正常整合、S ≥ 2.5 bits 加权编码
  （越意外记得越牢）；易变属性冲突强制写入。详见
  [docs/predictive-coding-gate.md](docs/predictive-coding-gate.md)。
- **易变属性槽位（slot）**：`name/location/role/employer/age/stack/plan` 属于"人会变"的事实——同一槽位出现新值时自动 **UPDATE**（旧值进历史），而非叠加矛盾记忆。喜好类（`like`）则允许多条并存，仅在极性冲突（喜欢↔不喜欢）时替换。
- **后台反思（LangMem 式）**：`mem.consolidate_background()` 定期把同主题高密度
  簇蒸馏成一条概括记忆（重要度取簇内最高 +0.05，使其更常驻 core memory 块，
  且半衰期 ×3 抗衰减），
  原事实标 `invalid_at` 并留 `superseded_by` 链接，审计日志记 CONSOLIDATE
  事件——记忆库越用越小、越用越精，且无信息丢失。
- **遗忘是软性的**：留存度 `retention = 0.5^(age / (half_life × (1 + 0.4·ln(1+access_count))))`，低于阈值的记忆不再浮现但保留在库中——被再次查询时即可"复忆"。
- **多租户隔离**：所有读写按 `user_id` / `agent_id` / `run_id` 作用域隔离。

## 项目结构

```
memtide/
├── memtide/
│   ├── engine.py      # MemoryEngine：add/search/render_context/get_history
│   ├── llm.py         # OpenAI 兼容客户端（提示词与 JSON 解析）
│   ├── embeddings.py  # OpenAI / DashScope embedder（auto 按 key 选择）
│   ├── retrieval.py   # 三路召回 + RRF 融合 + 可解释得分
│   ├── decay.py       # Ebbinghaus 遗忘曲线 + 检索强化
│   ├── gating.py      # 预测编码门控：surprise 三分流（novel/integrate/reject）
│   ├── consolidation.py  # 后台反思：聚类 → 概括 → supersede 链
│   ├── storage.py     # StorageBase 契约（PostgreSQL 唯一后端）
│   ├── pgstore.py     # PostgreSQL 后端（psycopg3 + pg_search BM25）
│   ├── vectorstore.py # Qdrant ANN（urllib REST）
│   ├── multimodal.py  # 多模态接入：媒体 parts → 文字描述 + sha256 附件
│   ├── types.py       # Memory / SearchResult / AddResult 数据类
│   ├── config.py      # MemoryConfig（全部阈值可调）
│   ├── server.py      # REST API（stdlib http.server，含 /media 素材服务）
│   └── cli.py         # 命令行入口
├── tests/test_memtide.py   # 73 个测试（hermetic 回归：全功能/性能路径/多模态/迁移/运维）+ 8 个 live 集成测试（真实端点含真实视觉，MEMTIDE_LIVE=1 门控）
└── examples/
```


## 测试

```bash
python3 -m unittest tests.test_memtide   # 73 tests, PG/Qdrant + 本地协议服务器，无外部网络
```

## License

MIT
