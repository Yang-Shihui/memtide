# Memtide 开发文档（Development Guide）

> 环境搭建、测试、代码约定、如何扩展后端、UI 开发、打包与发布。
> 设计原理见 [design.md](design.md)，使用说明见 [usage.md](usage.md)。

## 环境搭建

```bash
git clone <repo> && cd memtide
docker compose up -d postgres              # 测试需要可达的 PG（compose 已发布 5432）
python -m unittest tests.test_memtide      # ~15s，PG/Qdrant + 本地协议服务器，无外部网络

# 真实端点联调（可选）：cp .env.example .env 填好 key 后
set -a && source .env && set +a
MEMTIDE_LIVE=1 MEMTIDE_LLM_BASE_URL=$LLM_BASE_URL MEMTIDE_LLM_KEY=$LLM_API_KEY \
MEMTIDE_LLM_MODEL=$LLM_MODEL MEMTIDE_DASHSCOPE_KEY=$DASHSCOPE_API_KEY \
  python -m unittest tests.test_live -v
```

## 项目结构

```
memtide/
├── memtide/                 # 核心包（LLM/Qdrant 客户端仅标准库；psycopg 是唯一运行时依赖）
│   ├── engine.py           #   MemoryEngine：add/search/context/update/history/...
│   ├── multimodal.py       #   媒体 parts → 文字描述 + sha256 附件
│   ├── llm.py              #   OpenAI 兼容客户端（提示词与 JSON 解析）
│   ├── embeddings.py       #   OpenAI / DashScope embedder + auto + 缓存
│   ├── retrieval.py        #   三路召回 + RRF + 重排（components 可解释）
│   ├── gating.py           #   预测编码门控：surprise 三分流
│   ├── consolidation.py    #   后台反思：聚类→概括→supersede
│   ├── decay.py            #   Ebbinghaus 留存度 + 检索强化
│   ├── storage.py          #   StorageBase 契约（PostgreSQL 唯一后端）
│   ├── pgstore.py          #   PostgresStorage（psycopg3 + pg_search BM25）
│   ├── vectorstore.py      #   Qdrant（urllib REST）+ 维度探测自愈
│   ├── server.py           #   标准库 HTTP REST（含 根路径静态控制台、/media/ 素材托管）
│   ├── types.py            #   Memory/SearchResult/AddResult/ExtractedFact
│   ├── config.py           #   MemoryConfig + config_from_env
│   ├── static/             #   UI 构建产物（由 webui 构建同步，勿手改）
│   └── cli.py              #   python -m memtide 入口
├── webui/                  # React 18 + Vite 管理台（马卡龙浅色主题）
│   ├── src/views/          #   Dashboard/MemoryList/SearchPlayground/ContextPanel/Operations
│   ├── src/components/     #   Logo.jsx 等
│   └── public/favicon.svg  #   favicon（= logo badge）
├── docs/                   # 设计方案/使用说明/开发文档/门控数学/Logo
├── tests/                  # test_memtide.py（hermetic）+ test_live.py（真实端点）
├── scripts/                # seed_demo.py（演示数据）/ live_check.py（真端点体检）
├── examples/               # quickstart.py / chat_agent.py
└── docker-compose.yml      # memtide + paradedb(pg16) + qdrant 生产栈
```

## 测试

| 套件 | 内容 | 运行 |
|---|---|---|
| hermetic（77 个） | 全功能回归 + 性能路径（查询计数）+ 历次 bug 回归；本地 OpenAI 协议服务器 + 独立 PG schema，无外部网络 | `python -m unittest tests.test_memtide` |
| live（8 个） | 真实 LLM/embedding/**视觉**端点；未配 key 自动跳过 | `MEMTIDE_LIVE=1 ... python -m unittest tests.test_live` |
| 体检 | 4 步真端点检查 | `python scripts/live_check.py` |
| 门控标定 | 改写对相似度分布 → 建议阈值 | `python scripts/calibrate_gate.py` |

约定：

- **新功能必须带 hermetic 测试**；断言要打在**序列化结果**上（`to_dict()`），
  防止对象属性正确但序列化遗漏（真踩过：SearchResult.attachments 死代码）。
- 涉及真实语义相似度的阈值判断，用 `infer=False` 确定性播种，或先在 live 测试
  里实测相似度分布再定阈值。
- 视觉类测试图片用 `tests.test_memtide.make_test_png()`（stdlib 生成 64×64），
  **不要用 1×1 像素**——vision 模型普遍拒绝 <10px 图。

## 代码约定

- **核心协议简洁**：HTTP 客户端使用标准库 urllib；PostgreSQL 驱动 `psycopg` 是唯一运行时依赖。
  新依赖必须放可选 extra 并惰性导入。
- 引擎只讲真实 OpenAI 协议；hermetic 测试通过 `tests/fake_openai.py`
  （本地假服务器）复刻规则级行为——改 prompt 路由时两处同步。
- 所有写操作进审计日志（`log_event`）；所有删除默认软删除。
- 时间戳一律 ISO 字符串；支持调用方时间戳的地方要同步审计事件时间。
- 提交前：`python -m unittest tests.test_memtide` 全绿 + `cd webui && npm run build`。

## 如何扩展

**新增存储后端**：实现 `storage.StorageBase` 全部方法（参考 pgstore.py 的写法：
get_raw 返回 dict-like、fts/entity 两通道带 user/agent/run 过滤、旧库迁移），
在 `make_storage()` 注册，补一组与 TestPersistenceAndCli 对等的测试。

**扩展向量库**：当前生产实现固定为 Qdrant。若未来引入新后端，必须实现
`vectorstore.VectorStoreBase`（upsert/search/delete/clear），在
`make_vector_store()` 显式注册，并保持基础设施不可用时 fail-fast，不能静默降级。

**新增 LLM/Embedder**：继承 `BaseLLM` / 实现 `embed()`，在 `make_llm` /
`make_embedder` 注册。

**新增消息 part 类型**：`multimodal.py` 的 `_kind_of()` 登记 kind →
`process_part()` 处理取数/落盘/描述 → 必要时在 `Attachment.kind` 白名单
（image/audio/video/file）中扩展；补 hermetic 测试。

## Web UI 开发

```bash
cd webui
npm install
npm run dev        # :5173，API 代理到 localhost:8300
npm run build      # 产物 dist/ → 同步到 memtide/static/
rm -rf ../memtide/static/assets ../memtide/static/index.html ../memtide/static/favicon.svg
cp -r dist/. ../memtide/static/
```

- 设计 token 在 `src/style.css` `:root`（马卡龙体系：奶油底/开心果主色/语义
  徽章色），新组件优先用现有 token 与 `.panel/.chip/.badge/.btn` 等既有类。
- 对比度红线：正文对浅底 ≥ 4.5:1（ui-ux-pro-max skill 的 CRITICAL 规则）。
- Logo 源文件：`docs/assets/logo.svg`（波峰 M，浅底用）、`logo-dark.svg`
  （深底白变体）与 `logo-badge.svg`（白底 app tile/favicon）；favicon、官网
  内联 SVG、UI 顶栏 `components/Logo.jsx` 三处需同步（改形状时一起改）。

## Docker 与打包

```bash
docker compose up -d --build      # 多阶段：node 构建 UI → python:3.13-slim 运行
python scripts/seed_demo.py       # 在容器内跑：docker exec memtide-memtide-1 ...
```

- `pyproject.toml` 用 package-data 打包 `memtide/static/**`，因此**构建 wheel 前先
  完成 UI build + 同步**。
- `.dockerignore` 排除 tests/examples/docs/webui 源码（镜像只装运行时）。
- 生产运行时固定为 PostgreSQL + Qdrant + 真实 LLM/embedding；测试使用本地协议夹具，
  但不会改变引擎的后端实现。

## Git 约定

- `.env`（真实密钥）**永远不入库**，已在 .gitignore；示例文件是 `.env.example`。
- 提交前检查：`git status` 无密钥文件、`git diff --staged` 无 `sk-` 前缀字符串。
- 提交信息用 Conventional Commits（feat/fix/docs/test/chore）。

## 文档索引

| 文档 | 内容 |
|---|---|
| [../README.md](../README.md) | 项目总览与快速开始 |
| [design.md](design.md) | 架构与各子系统详细设计 |
| [usage.md](usage.md) | 安装/API/部署/配置全参考 |
| [predictive-coding-gate.md](predictive-coding-gate.md) | 门控数学与实测标定 |
| [assets/logo.svg](assets/logo.svg) | Logo（浅底/深底/badge 三版） |
