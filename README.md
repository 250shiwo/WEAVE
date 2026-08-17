# Weave

为 AI Agent 设计的知识图谱长期记忆平台。以 **Kuzu 图数据库** 承载实体/关系事实、**LanceDB 向量数据库** 承载语义检索、**SQLite** 承载元数据与任务记录、**Redis** 承载会话缓存与任务队列，通过 **MCP（Streamable HTTP）** 与 **REST API** 双通道对外提供服务。

## 核心特性

- **混合检索（recall）**：查询向量化后以 LanceDB 双表（实体/文本块）为入口，经 Kuzu 图谱 1 跳扩展补充关联事实，结果标注来源（`graph` 已验证 / `vector` / `session` 未过滤原料）
- **双模式记忆（remember）**：无 `session_id` 直写永久知识图谱；有 `session_id` 先写会话缓存立即返回，由后台 improve 管线过滤沉淀
- **会话防污染**：会话原文仅存 Redis（带长度上限与 TTL），经 LLM 认知过滤器 + 重要性门禁后才允许进图，永不让闲聊污染长期记忆
- **知识演进**：事实变更通过 supersede（取代）语义版本化，检索自动排除已失效关系
- **异步摄取（cognify）**：txt/md 文档 base64 提交后进入队列，内嵌 worker 完成分块/抽取/入图，`task_status` 轮询进度
- **单进程拓扑**：REST `/v1` 与 MCP `/mcp` 同端口对外，worker 以协程内嵌于 API 进程（嵌入式数据库单写者约束，不拆独立 worker）
- **优先级队列**：单次 BRPOP 按 key 序消费 `[improve, cognify]` 双队列，improve 永远优先

## 技术栈

| 组件 | 选型 |
|---|---|
| 语言 / 框架 | Python ≥ 3.11 · FastAPI · Uvicorn |
| 图数据库 | Kuzu（嵌入式） |
| 向量数据库 | LanceDB（嵌入式） |
| 关系存储 | SQLite（SQLAlchemy 2.0） |
| 缓存 / 队列 | Redis ≥ 5.0 |
| LLM | DeepSeek（实体/关系抽取，OpenAI 兼容协议） |
| 向量化 | DashScope（OpenAI 兼容协议） |
| 依赖管理 | uv（uv.lock 冻结） |

## 快速开始（Docker，推荐）

前置：Docker Desktop。

```powershell
# 1. 配置密钥（见下文「配置」节），然后构建并启动 redis + weave
docker compose up -d --build

# 2. 验证
curl http://localhost:8000/v1/health   # -> {"status":"ok"}

# 3. 可选：连同前端 Web UI 一起启动（http://localhost:8080）
docker compose --profile full up -d --build
```

数据持久化在 named volume（`weave_weave_data` / `weave_redis_data`）中；`docker compose down` 停止，加 `-v` 连同数据一起清空。

> 注意：本地开发服务（127.0.0.1:8000/6379）与容器不要同时运行，端口会冲突；也不要让两个进程挂载同一份数据目录（Kuzu/LanceDB 文件锁约束）。

## 本地开发

前置：Python ≥ 3.11、[uv](https://docs.astral.sh/uv/)、Redis 运行于 `127.0.0.1:6379`。

```powershell
# 安装依赖（含 dev 组：pytest 等）
uv sync

# 只启动 Redis（用项目自带的 compose 服务）
docker compose up -d redis

# 启动服务（REST + MCP 同在 http://127.0.0.1:8000，内嵌 worker 随应用启动）
uv run weave

# 运行测试（fakeredis + Fake LLM/Embedding 客户端，无需真实外部依赖）
uv run pytest
```

前端是纯静态页面，无构建工具：直接用浏览器打开 `data/web-ui/index.html`，或经 compose 的 `web-ui` 服务访问 http://localhost:8080。

## 配置

配置唯一来源是环境变量 / 项目根目录 `.env`（pydantic-settings，优先级：环境变量 > .env > 默认值）。复制以下模板为 `.env` 并填入密钥：

```dotenv
# ---------- LLM（实体/关系抽取） ----------
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

# ---------- 向量化 ----------
DASHSCOPE_API_KEY=sk-your-dashscope-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding

# ---------- 服务鉴权（生产务必改成强随机值） ----------
WEAVE_API_KEY=dev-key
WEAVE_HOST=127.0.0.1
WEAVE_PORT=8000

# ---------- 存储路径 ----------
VECTOR_DB_PATH=./data/vector_db
GRAPH_DB_PATH=./data/graph_db
RELATIONAL_DB_PATH=./data/weave.db

# ---------- Redis ----------
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0

# ---------- 行为参数 ----------
SESSION_MAX_ITEMS=50      # 单会话最大记忆条目数
SESSION_TTL_DAYS=7        # 会话存活天数
CHUNK_SIZE=1500           # 文本分块字符数
CHUNK_OVERLAP=200         # 相邻分块重叠字符数
DB_CALL_TIMEOUT=30        # 单次数据库调用超时秒数
DB_CALL_MAX_RETRIES=2     # 数据库调用失败重试次数
QUEUE_POLL_TIMEOUT=5      # worker BRPOP 轮询超时秒数
```

Docker 部署时 compose 会自动注入 `.env`，并覆盖 `WEAVE_HOST=0.0.0.0`、`REDIS_HOST=redis` 两项容器内连接地址。

## REST API

Base URL：`http://localhost:8000/v1`。除 `GET /v1/health` 外全部需要请求头 `Authorization: Bearer <WEAVE_API_KEY>`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（认证豁免） |
| POST | `/memories` | 写入记忆：`{content, dataset?, session_id?}` |
| POST | `/recall` | 混合检索：`{query, dataset?, top_k?, session_id?}` → `{facts, chunks, session_items}` |
| POST | `/documents` | 异步摄取文档（202）：`{file_name, content_base64, dataset?}` → `{task_id, data_id}` |
| GET | `/tasks/{task_id}` | 查询异步任务状态（pending/running/completed/failed） |
| GET | `/datasets` | 列出全部数据集及规模统计 |
| DELETE | `/datasets/{name}` | 级联清空指定数据集（图/向量/元数据） |
| DELETE | `/memories?confirm=true` | 清空全部记忆（高危，必须显式确认） |

示例：

```powershell
# 写入一条永久记忆
curl -X POST http://localhost:8000/v1/memories `
  -H "Authorization: Bearer dev-key" -H "Content-Type: application/json" `
  -d '{"content": "Weave 项目使用 Kuzu 作为图数据库", "dataset": "project"}'

# 检索
curl -X POST http://localhost:8000/v1/recall `
  -H "Authorization: Bearer dev-key" -H "Content-Type: application/json" `
  -d '{"query": "Weave 用了什么图数据库", "dataset": "project"}'
```

错误统一为信封格式 `{"error": {"code", "message"}}`：400 业务校验失败、401 鉴权失败。

## MCP 接入

端点：`http://localhost:8000/mcp`（Streamable HTTP，无状态 + JSON 响应），同样需要 Bearer 头。

```json
{
  "mcpServers": {
    "weave": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer dev-key" }
    }
  }
}
```

- **Claude Code**：`claude mcp add --transport http weave http://localhost:8000/mcp --header "Authorization: Bearer dev-key"`
- **Claude Desktop**（仅 stdio）：用 `mcp-remote` 桥接 —— `npx -y mcp-remote http://localhost:8000/mcp --header "Authorization: Bearer dev-key"`

暴露 6 个工具：`remember` / `recall` / `forget` / `cognify_file` / `task_status` / `list_datasets`。建议用 `dataset` 隔离不同项目的记忆，用固定 `session_id` 标记单次任务会话；在 Agent 系统提示中引导"重要事实调 remember，需要历史上下文先 recall"。

## 项目结构

```
weave/
├── main.py            # 入口：装配依赖 + uvicorn 启动
├── worker.py          # 内嵌 worker：单循环 BRPOP 双队列（improve 优先）
├── config.py          # 全局配置（pydantic-settings，唯一配置来源）
├── api/               # 接入层
│   ├── app.py         #   FastAPI 组装：中间件/路由/MCP 挂载/lifespan
│   ├── rest.py        #   REST 路由（8 个端点薄壳）
│   ├── mcp_server.py  #   MCP 工具（6 个薄壳）
│   └── auth.py        #   Bearer 认证中间件（仅豁免 /v1/health 与 OPTIONS）
├── core/              # 核心服务层
│   ├── service.py     #   MemoryService 门面
│   ├── pipelines.py   #   remember/cognify/improve 管线
│   ├── retrieval.py   #   混合检索（向量入口 + 图扩展）
│   ├── extraction.py  #   LLM 实体/关系抽取与认知过滤
│   ├── chunking.py    #   文本分块
│   ├── session.py     #   会话缓存管理
│   ├── cleanup.py     #   过期会话/数据清理
│   └── models.py      #   领域模型
└── infra/             # 基础设施层
    ├── relational.py  #   SQLite（SQLAlchemy）
    ├── vector.py      #   LanceDB
    ├── graph.py       #   Kuzu
    ├── cache.py       #   Redis（缓存 + 队列，BRPOP 兜底）
    ├── embedding.py   #   DashScope 向量化客户端
    ├── llm.py         #   DeepSeek 客户端
    └── executor.py    #   DB 调用保护（超时/重试）
tests/                 # pytest 测试（fakeredis + Fake 客户端，可离线运行）
data/web-ui/           # 纯静态前端（无构建工具）
docs/superpowers/      # 设计文档与实施计划
```

## 设计文档

- 设计规格：[docs/superpowers/specs/2026-08-10-weave-design.md](docs/superpowers/specs/2026-08-10-weave-design.md)
- 实施计划：[docs/superpowers/plans/2026-08-10-weave-implementation.md](docs/superpowers/plans/2026-08-10-weave-implementation.md)
