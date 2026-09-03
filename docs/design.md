# Memtide 设计方案（Architecture & Design）

> 本文是 Memtide 的完整设计文档：设计目标、架构总览、各子系统的详细设计与取舍。
> 使用说明见 [usage.md](usage.md)，开发与测试见 [development.md](development.md)，
> 门控的数学细节见 [predictive-coding-gate.md](predictive-coding-gate.md)。

## 1. 定位与设计目标

Memtide 是给 LLM agent 用的长期记忆引擎：agent 把对话交给它，它抽出值得记的原子事实，
消解冲突、门控编码、按遗忘曲线衰减；检索时三路召回融合、给出可解释的得分。

**设计目标（按优先级）**

1. **零 SDK 依赖核心**：LLM / Qdrant 客户端只用 Python 标准库——
   没有第三方 SDK 依赖树、没有供应链风险面；唯一运行时依赖是
   PostgreSQL 驱动 psycopg。生产部署 = PostgreSQL + 任意
   OpenAI 兼容 LLM/embedding 端点，docker compose 开箱即用。
2. **记忆质量优先**：不是"存聊天记录"，而是 Mem0 式原子事实 + 写时冲突消解 +
   认知科学启发的门控/遗忘，让记忆库越用越小、越用越准。
3. **可解释**：每条检索结果带 components 得分明细；每条记忆带完整审计链；
   每次写入返回 ADD/UPDATE/DELETE/NOOP 决策和门控的 surprise 值。
4. **可平滑升级**：更换 Qdrant 集合、LLM/embedding 端点或 reranker，全部由配置切换，业务代码无感知。

**非目标**

- 不做多进程并发写（单引擎实例 + 锁已覆盖 REST 多线程场景）
- 不提供离线/模拟模式（引擎只讲真实 OpenAI 协议；测试用本地假服务器）
- 不做细粒度权限 / 多租户配额（作用域 `user_id/agent_id/run_id` 只做隔离；
  鉴权仅有单一 `MEMTIDE_API_KEY` 门禁）
- 不做向量训练/微调（embedding 全部外接）

## 2. 总体架构

```
                        ┌──────────────────────────────────────────────┐
 写入 add()             │                MemoryEngine                  │
 messages ──────────────▶  multimodal 归一化 ─▶ extract 事实抽取       │
 (text / content-parts) │        │                  │                   │
                        │        │            PredictiveGate           │
                        │        │            surprise 三分流           │
                        │        │                  │                   │
                        │        │            冲突消解 (LLM/规则)       │
                        │        │            ADD/UPDATE/DELETE/NOOP   │
                        │        ▼                  ▼                   │
                        │  附件落盘(sha256)      Memory 持久化           │
                        └────────┬──────────────────┬───────────────────┘
                                 │                  │
              ┌──────────────────▼───┐   ┌──────────▼─────────────┐
              │ 媒体文件 (内容寻址)    │   │ StorageBase            │
              │ memtide_media/<sha>.  │   │ └ PostgresStorage      │
              └──────────────────────┘   │   memories + entities  │
                                         │   + memory_history     │
                                         │   + FTS(BM25)         │
                                         └──────────┬─────────────┘
                                                    │ embeddings 副本
                                         ┌──────────▼─────────────┐
                                         │ VectorStoreBase        │
                                         │ └ Qdrant ANN (REST)    │
                                         └────────────────────────┘

 读取 search()          query ─▶ ┌ 向量余弦 ┐
                                 ├ BM25 全文   ├─▶ RRF 融合 ─▶ +留存度+重要度
                                 └ 实体索引  ┘        │
                                                     ▼
                                  命中 → access_count++（间隔效应）
```

## 3. 写入管线（`engine.add()`）

单次 `add()` 五阶段同步完成（auto reflect 为可选后台线程，不参与单次写入）：

### 3.1 输入归一化（含多模态）

接受纯文本或 OpenAI 风格消息列表；content 可以是字符串或 content-parts 数组。
媒体 part（图片/音频/文件）由 `multimodal.py` 处理：

- 解析引用：data URL / `https://` / 本地路径 / base64 内联，大小上限 `max_media_bytes`
- 原始字节以 **sha256 内容寻址**落盘 `media_dir/<sha256>.<ext>`（同字节自动去重）
- 归一化为文字：图片走 vision 端点（OpenAI 兼容 chat/completions + image_url），
  音频走可选 STT（/audio/transcriptions），调用方也可直接在 part 上给 `caption`
- 生成 `[image: <描述>]` 占位符并入抽取文本，同时产出 `Attachment` 字典
  （`{id, kind, source, mime, sha256, description}`）

设计取舍（调研结论，详见 [usage.md § 多模态](usage.md#多模态记忆)）：
Mem0 / MemOS / M3-Agent 三条路线里选择 Mem0 式"**写入时媒体→文字桥接**"——
描述文本复用整条文本管线（抽取/门控/向量/全文检索），文字查询天然跨模态召回图片，
且零新增依赖。M3-Agent 式原生多模态编码器过重，MemOS 式统一记忆单元超出
单进程库的定位。失败模式设计：vision 不可用时素材照存、描述留空（优雅降级，
绝不因媒体丢失数据）。

### 3.2 事实抽取

- **真实 LLM 路径**：`FACT_EXTRACTION_PROMPT` 要求输出原子事实 JSON
  （text/type/importance/entities/slot），第三人称、单句自包含。
- **测试夹具**：`tests/fake_openai.py` 在本地假服务器里复刻了同构的双语规则
  抽取（模板/类型/重要度/slot），确定性输出——hermetic 测试的基石；引擎
  本体只讲真实协议。
- 两条路径同一契约（`BaseLLM.complete_json`），可整体替换。
- `slot` 为开放 hint（`slots.py`）：建议 7 槽（name/location/role/employer/age/stack/plan）仅作示例，
  LLM 可发明新槽（如 spouse/pet）；`SLOT_ALIASES` 别名表归一同物异名（city/住址→location），
  非法值归 `None`；多值事实（喜好、技能列表）用 `null`，永不强制 UPDATE。

### 3.3 预测编码门控（`gating.py`）

记忆库即生成式先验：新事实按预测误差 S = -log2 p̂ 分流。**slot 范围先验**
（`gate_slot_scoped=True` 默认开）：带 slot 的事实只在**同义 slot 记忆**
（经 `same_slot` + `slot_aliases` 判定，city == location）上取预测相似度——
"用户住在X"不会被句式相似的"用户工作在Y"抬高预测度；无同 slot 先验的新属性自然落入 NOVEL（新属性=意外）。

| S | 决策 | 语义 |
|---|---|---|
| ≤ 0.5 bits | REJECT | 完全被预测到，不存（防冗余堆积） |
| 中间 | INTEGRATE | 正常编码 |
| ≥ 2.5 bits | NOVEL | 越意外越重要，importance +0.10 |

同义 slot 冲突（规范化后相同 + 相似度 ≥ `gate_slot_floor` 0.40 + 文本不同）强制写入
（volatile-update）。阈值依赖 embedding 分布，换模型后跑
`scripts/calibrate_gate.py` 重新标定。数学推导见
[predictive-coding-gate.md](predictive-coding-gate.md)。

### 3.4 冲突消解

对每个候选事实，取相似度 ≥ 阈值的既有记忆（同义 slot 冲突降到 `gate_slot_floor` 门槛），交给：

- **LLM 决策**：`CONSOLIDATION_PROMPT` 输出每候选 NOOP/UPDATE/DELETE；
  提示词强调"措辞不同但同义 = NOOP"防止重复膨胀；slot 只是 hint——多值事实
  （两套房、Rust+Python）与时间限定事实（去年 vs 今年）即使 slot 相同也应 ADD。
  **批量优化**：一次
  `add()` 的全部事实合并进 `CONSOLIDATION_BATCH_PROMPT`，单次 LLM 调用
  完成整轮消解（解析失败自动回退逐条），写延迟与 token 成本大幅下降。
- **测试夹具**：`tests/fake_openai.py` 里的 `consolidate_rules()` 用余弦带 +
  极性检测（喜欢↔不喜欢）+ 同槽信号（三类）+ `sim≥0.35` 底线，与 LLM 契约一致。

决策执行：UPDATE = `replace_text`（旧值进审计）；DELETE = 软删除；
NOOP/UPDATE 消费掉新事实；否则 ADD 新记忆。相同表述重复写入被 NOOP 吸收，
因此记忆库不会线性膨胀。

### 3.5 持久化与审计

每条记忆是一个 `Memory`（见 `types.py`）：原子事实文本 + 类型/重要度/实体 +
**双时标**（`valid_at` 事实何时为真 / `invalid_at` 何时失效）+ `superseded_by`
取代链 + `attachments` 附件引用 + 5 个时间戳。所有变更写 `memory_history`
事件日志（ADD/UPDATE/DELETE/ACCESS/CONSOLIDATE，带 prev/new 值）——
Zep 式"失效不删除"，历史可回放。

**调用方时间戳**：`add(..., timestamp=ISO-8601)` 用于导入历史对话——
`created_at/valid_at` 与审计 ADD 事件都用数据自身时间，遗忘曲线按历史时间衰减
（久远记忆自动沉到 retention floor 之下，可 `include_forgotten` 复忆）。

## 4. 检索管线（`retrieval.py`）

三路独立召回（全部按 user/agent/run 作用域过滤）：

1. **向量**：query embedding → Qdrant ANN top-40
2. **全文**：ParadeDB pg_search BM25 索引（`@@@` 查询 + `score()` 排序），
   中文用 ngram tokenizer（min 2-gram）免外部分词器，中英可查
3. **实体**：实体表 LIKE 匹配 + 文本二次校验（防 CJK 宽松键误召回）

**RRF 融合**：加权 `score_rrf = Σ w/(k + rank)`，k=60——只用排名不用分数，
天然规避三路分数量纲不一。实体通道较宽松计 `×0.5`，查询扩展变体计 `×0.5`
（`entity_channel_weight` / `expansion_variant_weight`）。

**重排**：`final = 1.0·rrf + 0.15·cosine + 0.05·bm25 + 0.03·entity + 0.10·retention + 0.05·importance`。
每个命中的 components 字典完整暴露六项得分（+可选 rerank，UI 可视化成条形图）。

**可选质量层**（默认全关，配置启用，不改变默认行为）：

- **type/slot 过滤**：`search(memory_type=, slot=)` 融合后过滤——把已存的
  分类维度用于检索（如只查偏好、只查 location 事实）；slot 过滤先经别名归一，
  `slot="city"` 也能命中 `location` 记忆
- **查询扩展**（`query_expansion`，仅真实 LLM）：检索前一次 LLM 调用产出
  英译 + 改写两个变体，多路召回后融合（变体 `embed_batch` 一次调用，融合权重减半）——跨语言查询（中文 query 对英文
  事实）的主要补救
- **cross-encoder 重排**（`rerank_backend="http"`）：融合 top-N 送 Jina/Cohere
  风格 POST /rerank 精排，rerank 分主导、融合分断并列，失败静默回退融合序
- **MMR 多样性**（`mmr_lambda`）：得分先按池内最大归一再与差异度混合，最终 top-k 兼顾得分与相互差异

**检索强化**：命中即 `access_count++` 并记 ACCESS 事件——访问越多，
遗忘曲线半衰期越长（见 §5），模拟间隔效应。`render_context(query=)` 的预览
检索用 `reinforce=False`，只读不污染计数。

**写入路径性能**：相似度扫描走 `store.all_embeddings()` 单查询（一条 SQL
各一条 SQL），一次 `add()` 全部事实共享，候选只取 top-K 水合成 Memory
对象——旧实现的 N+1 逐行 `get_embedding` 已消除（100 条库、3 事实一轮
的扫描段实测 ~20x）。embedding 层有精确文本缓存（CachedEmbedder），
API 批量接口单次调用带多个文本。

## 5. 遗忘与巩固

- **留存度**（`decay.py`）：`retention = 0.5^(age / (half_life × min(1 + 0.4·ln(1+access_count), 4.0)))`，
  年龄从 `valid_at/created_at` 起算——访问拉长半衰期但不再回春，热门记忆也不会永生。
  **分型衰减**：episodic 半衰期 ×0.5（`episodic_half_life_mult`）、遗忘线 0.05（`episodic_floor`）。
  **概括记忆抗衰**：`source="consolidation"` 的记忆有效半衰期 ×3
  （`consolidation_half_life_mult`）——抽象层比细节忘得慢，防止概括先于其
  成员淡忘。低于遗忘线（fact 0.02 / episodic 0.05）视为"遗忘"：默认检索不再浮现，
  但数据仍在库中，可复忆。软性遗忘而非删除——删除只发生在显式 delete 或
  冲突消解。
- **后台反思**（`consolidation.py`，LangMem 式）：`consolidate_background()`
  对同用户记忆做贪心密度聚类（余弦 ≥ 0.45，簇 ≥ 3 条）→ 蒸馏概括
  （LLM 蒸馏）→ 概括记忆以簇内最高重要度 +0.05 入库，成员标
  `invalid_at + superseded_by` 并记 CONSOLIDATE 事件。**调度**：
  `enable_auto_reflect(interval, scope)` 内置守护线程定期执行（服务端
  `MEMTIDE_AUTO_REFLECT=秒数`），不设则纯手动。
- **近重复压实**：`compact(scope)` 纯向量去重（cos ≥ 去重阈值），保留最优
  一条、其余 supersede 入审计链——与反思互补（反思抽象同主题，压实删除
  字面重复），都不需要 LLM。

## 6. Core Memory 块（Letta 式）

`render_context()` 产出可直接注入 system prompt 的文本块：按
`importance × (0.3 + 0.7·retention)` 排序的重要事实（importance ≥ 0.55），
截断到 `core_max_chars`；可选附加与当前 query 最相关的 5 条检索结果。

## 7. 多模态子系统（`multimodal.py`）

- **Attachment 模型**：`{id(sha256 前 16), kind, source(文件名), mime, sha256, description}`。
  存储在 `memories.attachments` JSON 列（PostgreSQL）。
- **关联语义**：turn 级溯源——同一轮抽出的所有事实都携带该轮的附件引用
  （`metadata.modality` 标注模态）；若整个调用没有任何事实被抽出，描述本身
  兜底存为一条 episodic 记忆，保证素材可检索。
- **安全**：大小上限（读满 cap+1 即拒绝）、sha256 严格校验（`/media/` 端点
  拒绝非 64 hex）、内容寻址文件名不可穿越；**本地路径读取默认关闭**
  （`media_allow_paths=False`）——否则 REST 调用方可用 `{"path": "/etc/passwd"}`
  把任意本地文件变成可下载素材；媒体 URL 抓取限定 http(s) 且受大小上限约束。
- **跨模态检索**：文字 query → 描述文本的向量/全文命中 → 返回事实 + 附件引用
  → agent 用 `GET /media/{sha256}` 取回原图。（不引入 CLIP 式联合嵌入：
  文字桥接已覆盖主流场景，保持核心客户端精简。）

## 8. 存储层设计

**`StorageBase` 契约**（`storage.py`）：insert / replace_text / soft_delete /
hard_delete / supersede / mark_accessed / log_event / get / all_valid /
fts_search / entity_lookup / history / stats / reset。引擎层只依赖此契约。

**PostgresStorage（唯一后端）**：psycopg3（`pip install .`，psycopg 是唯一
运行时依赖）连接，
全文检索用 pg_search BM25 索引（AGPL-3.0 社区版扩展，随
`paradedb/paradedb:pg16` 镜像提供；中文 ngram tokenizer 免外部分词器），
embedding 存 float32 bytea
（Qdrant 索引重建副本），时间戳一律 ISO 字符串。旧列通过
`ADD COLUMN IF NOT EXISTS` 自动迁移。表结构：`memories`（主表，含
attachments 列）、`entities`（name↔memory_id，实体检索通道）、
`memory_history`（审计日志，seq 递增可回放）。

## 9. 向量层与 Embedding 层

- **QdrantVectorStore（唯一向量库）**：urllib 直连 REST（无 SDK），point ID =
  memory id 的 64-bit blake2b；构造时探测 embedder 维度，集合维度不符自动
  重建集合（提示跑 `rebuild_index()` 回填）；连接失败直接失败而非静默降级，
  生产环境不隐藏基础设施故障。
- **Embedder**：`OpenAIEmbedder` / `DashScopeEmbedder`（懒探测维度、精确文本
  缓存、批量接口）。`embedding_backend="auto"` 按可用 key 自动选。

## 10. REST 服务（`server.py`）

- 标准库 `ThreadingHTTPServer`；单引擎实例 + 统一引擎 RLock（REST/后台写入/auto-reflect 互斥）。
  PG 每线程一连接（`threading.local`），无连接池依赖。
- 路由全覆盖（见 usage.md），`ValueError → 400`（非法时间戳/超限媒体/非法 compact threshold），
  兜底 `Exception → 500 JSON`（仅异常类型 + 前 200 字符，不泄露 DSN/路径），
  参数容错（垃圾 limit 回落默认值，越界钳制 1–500）。
- **鉴权**：设 `MEMTIDE_API_KEY` 后所有数据端点要求 `X-API-Key` 或
  `Bearer` 头（静态控制台保持公开以便输入 key）；不设 = 开放（开发默认）。
- 静态托管：`/` 为官网落地页（landing.html，缺失时回退控制台），管理台在
  `/console`（SPA fallback 到自身 shell），`/docs` 为交互式 API 文档页，
  `/ui/*` 保留为旧链接别名；
  未知 API 前缀返回 JSON 404；路径穿越防护（解码前拼接 + resolve 前缀校验）。
- `/media/{sha256}`：内容寻址素材下载，不可变缓存头；`/media/gc` 孤儿清理。
- **导出/导入**：`GET /export`（JSONL，含审计状态与可选 embedding base64）、
  `POST /import`（skip/overwrite 冲突策略）；媒体文件不入包（内容寻址，
  随 media_dir 拷贝）。
- **后台写入**：Python 层 `add_background()` 返回 Future（内部 2 线程池）；
  REST 保持同步——异步 REST 需要任务存储+轮询端点，收益不成比例。

## 11. 测试策略

- **Hermetic（86 个）**：本地 OpenAI 协议服务器 + 隔离 PG schema，~15s，无外部网络。
  覆盖写管线/检索/门控/反思/审计/REST/UI 托管/时间戳/多模态/旧库迁移/
  性能路径（查询计数）/压实/媒体GC/导出导入/鉴权/调度器/槽归一/衰减分型/历次 bug 回归。
- **Live（8 个，MEMTIDE_LIVE=1 门控）**：真实 LLM + embedding + 视觉端点，
  验证真实语义相似度分布下的行为（阈值是按实测分布标定的）。
- **E2E**：`scripts/live_check.py` 四步体检 + `scripts/seed_demo.py` 演示数据 +
  `scripts/calibrate_gate.py` 门控阈值标定。

## 12. 已知权衡与限制

- Qdrant 是唯一向量后端；大规模部署通过 Qdrant 分片/索引参数扩展
- 测试假服务器复刻的是规则级行为：真实 LLM 的语义质量需 live 测试覆盖
- 单引擎锁：写多读高并发场景应多实例分片（按 user_id 路由）；
  `add_background()` 与其它写入已由引擎级 RLock 串行化
- 媒体文件不进数据库：备份需同时备份 media_dir（或 export 后随目录拷贝）
- 门控阈值（0.5/2.5 bits）按实测 embedding 分布标定，换 embedding 模型后
  跑 `scripts/calibrate_gate.py` 重新标定
- 检索的 type/slot 过滤发生在融合之后：候选不足时过滤后的结果可能少于
  limit（超大规模需要把过滤下推到通道层）
