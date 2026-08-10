# Weave 设计文档 — AI 智能体长期记忆知识图谱平台

- 日期：2026-08-10
- 状态：已与用户确认定稿
- 参考：借鉴 [cognee](https://github.com/topoteretes/cognee)，面向个人单用户场景裁剪，不完全复刻

## 1. 背景与目标

为 AI 智能体（首要使用者是 [MySmallAgent](https://github.com/250shiwo/MySmallAgent)，同时兼容任意支持 MCP/HTTP 的 agent）提供**跨会话持久化、可演进的长期记忆**服务。

- 通过 MCP（Streamable HTTP）提供工具调用，同时暴露 REST 开放 API
- 单机个人场景：全部使用嵌入式/本地基础设施，一条命令启动
- 知识以知识图谱 + 向量混合方式存储，检索时自动路由

## 2. 需求总结（已与用户确认）

| 维度 | 决策 |
|---|---|
| 记忆类型 | 永久记忆：语义记忆（用户画像/偏好/事实）+ 文档知识库；会话记忆：Redis 缓存，后台过滤沉淀进图 |
| 写入路径 | 短文本 `remember` 同步入图；文档 `cognify_file` 异步（Redis 队列 + 进程内 worker） |
| 检索 | 混合检索自动路由（向量入口 → 图扩展），对 agent 只暴露一个 `recall` |
| 可演进 | 保留 version 版本化（历史版本保留、`is_latest` 最新优先）；**去掉** feedback_weight / importance_weight，不做排序权重与反馈逻辑 |
| 组织隔离 | 单用户 + dataset 命名空间（无多用户/权限体系） |
| 认证 | 单一静态 API Key（Bearer token），MCP 与 REST 共用 |
| 会话防污染 | 会话记忆进图必须经过 LLM 过滤（improve），详见 §6 |

## 3. 架构（方案 A：单进程）

一个 Python 进程承载全部职责：FastAPI（REST）+ FastMCP（`/mcp`）+ 进程内 asyncio worker（消费 Redis 队列）。**关键约束**：SQLite / LanceDB / Kuzu 均为嵌入式单写者数据库（Kuzu 以独占锁打开库文件），单进程拓扑天然满足该约束。LLM 抽取与 Kuzu/LanceDB 原生调用等阻塞操作统一收敛到一个 **max_workers=1 的专用线程池**（串行化 DB 写入，匹配单写者约束），并用 `asyncio.wait_for` 包裹超时，防止原生调用卡死拖住事件循环（见 §8）。

```
weave/                        # Python 包, uv 管理, Python 3.11+
├── main.py                   # 入口: 组装 FastAPI app + 启动内嵌 worker, uvicorn 运行
├── config.py                 # pydantic-settings, 读取 .env
│
├── api/                      # 接口层 (薄, 只做协议转换 + 认证)
│   ├── app.py                # FastAPI 组装: 认证中间件 + REST 路由 + FastMCP(/mcp)
│   ├── rest.py               # REST 开放 API
│   ├── mcp_server.py         # FastMCP 工具 (Streamable HTTP)
│   └── auth.py               # Bearer token 校验
│
├── core/                     # 业务核心 (DB 无关, 可独立测试)
│   ├── models/               # DataPoint 基类 + Entity/TextChunk/Document 等 (pydantic)
│   ├── pipelines/            # remember 管线(同步) / cognify 管线(异步) / improve 管线(异步)
│   ├── tasks/                # 原子任务: 切块 / 实体关系抽取 / embedding / 版本更替
│   ├── retrieval/            # 混合检索: 向量入口 → 图扩展 → 汇总排序
│   └── session.py            # 会话记忆读写 (Redis)
│
├── infra/                    # 基础设施适配层 (每个一个窄接口, 进程级单例)
│   ├── relational.py         # SQLite (SQLAlchemy)
│   ├── vector.py             # LanceDB 封装
│   ├── graph.py              # Kuzu 封装
│   ├── cache.py              # Redis 封装 (会话记忆 + 任务队列)
│   ├── llm.py                # DeepSeek 客户端 (OpenAI 兼容, JSON 抽取)
│   └── embedding.py          # DashScope embedding 客户端
│
└── worker.py                 # 进程内 asyncio worker: BRPOP 消费任务队列
```

分层原则：`api` 是薄壳，MCP 工具与 REST 端点一一对应、调用同一个 `core` 服务函数；`core` 不感知具体数据库，只面向 `infra` 窄接口编程，未来换 Neo4j/Qdrant 只动 `infra`。

## 4. 数据模型

### 4.1 DataPoint 基类

基于 data.txt 草稿调整：去掉 `feedback_weight` / `importance_weight`，版本化用 `version` + `is_latest` 显式表达。

```
id: UUID
version: int = 1
type: str                     # 节点类型, 由子类填充
metadata: dict = {}
source_pipeline: str | None   # "remember" | "cognify" | "session_improve"
source_task: str | None       # 管线内哪个原子任务产生
source_node_set: list[str]    # dataset 归属标签, 用于过滤
is_latest: bool = True        # 被新版本取代时置 False; 检索只返回 True
created_at: datetime
updated_at: datetime
```

### 4.2 v1 节点类型

语义记忆与文档走同一套管线，统一为以下类型：

- `Document`：原始文档记录（文件名、内容哈希、大小）
- `TextChunk`：文本块，embedding 与检索的基本单位（remember 短文本 = 1 个 chunk；文档 = N 个 chunk）
- `Entity`：实体（name, entity_type, description）。**用户画像/偏好不单独建表**，就是 `entity_type=Person/Preference` 等的实体
- 关系不建节点类：存 Kuzu 图边 + SQLite `edges` 元数据表

### 4.3 各存储分工

**SQLite（关系库，SQLAlchemy 管理表结构）**：

| 表 | 内容 |
|---|---|
| `datasets` | id, name(唯一), description, created_at |
| `data` | 原始数据记录：id, name, raw_text 或文件元信息, content_hash(防重复摄入), status, created_at |
| `dataset_data` | dataset_id ↔ data_id 关联 |
| `edges` | 关系元数据：edge_id, source_id, target_id, relationship_type + DataPoint 字段(version, is_latest, source_pipeline 等) |
| `pipeline_runs` | 异步任务状态：task_id, pipeline_name, data_id, status(pending/running/completed/failed), error, created_at, updated_at |

**LanceDB（向量库）**：

| 表 | 内容 |
|---|---|
| `text_chunks` | chunk_id, vector, text(原文 payload), data_id, dataset, created_at |
| `entities` | entity_id, vector, name, entity_type, description, dataset, is_latest |

**Kuzu（图库）**：

- 节点表 `Entity(id, name, entity_type, description, version, is_latest, created_at)`
- 节点表 `TextChunk(id, data_id, dataset)`（轻量引用，原文在 LanceDB payload，按需回查）
- 边表 `RELATES_TO(Entity→Entity, edge_id, relationship_type, version, is_latest)`（单一边表 + 类型属性，避免动态建边表）
- 边表 `MENTIONS(TextChunk→Entity)`

**Redis**：

| 键 | 内容 |
|---|---|
| `weave:session:{session_id}` | 会话记忆 list：{content, ts, synced}，上限 50 条（可配），TTL 7 天（可配）兜底 |
| `weave:queue:cognify` | 文档摄取任务队列 |
| `weave:queue:improve` | 会话沉淀任务队列 |

### 4.4 版本更替规则（演进核心）

同一 `(主体实体, relationship_type)` 出现新客体时（如"用户喜欢浅烘焙"→"用户喜欢深烘焙"）：旧关系 `is_latest=False` 保留历史，新关系 `version+1`。SQLite `edges` 与 Kuzu `RELATES_TO` 同步标记。检索只看 `is_latest=True`，历史可追溯。

实体去重：按规范化 name（trim + 大小写归一）匹配合并；v1 不做实体消歧/本体生成。

## 5. 核心数据流

### 5.1 remember(content, dataset) — 同步（无 session_id）

agent 明确指定的记忆，视为已筛选：

1. 写 `data` 记录（content_hash 去重）+ 关联 dataset
2. 切块（短文本通常 1 块；切块规则：约 1500 字符 + 200 重叠，段落优先）
3. DeepSeek 抽取实体+关系三元组（JSON 结构化输出）
4. 实体按规范化 name 合并；关系执行 §4.4 版本更替
5. 顺序写三库：Kuzu（Entity/RELATES_TO/MENTIONS）→ LanceDB（chunk + entity 向量）→ SQLite（edges 元数据）
6. 返回：抽取到的实体数、关系数、版本更替数

### 5.2 cognify_file(file_name, content_base64, dataset) — 异步

1. 解码 base64 → 写 `data` 记录 + `pipeline_runs(pending)` → 任务入 `weave:queue:cognify` → **立即返回 task_id**
2. 进程内 worker BRPOP 取出 → 状态置 running → 加载文本（v1 支持 txt/md，预留 loader 接口）→ 切块 → 逐块走与 remember 相同的抽取/入图逻辑
3. 完成置 completed；失败置 failed + error，可重新提交
4. 调用方用 `task_status(task_id)` 轮询

### 5.3 会话记忆 — remember(content, session_id) + 后台 improve

无单独 save_session 工具，`remember` 带 session_id 即是会话记忆：

1. 写 Redis `weave:session:{id}`（毫秒级返回，**不触碰图**）
2. 自动入 `weave:queue:improve`
3. worker 消费时，同一会话多条积压自然攒批成一次 LLM 调用（成本控制）
4. improve 管线 = LLM 认知过滤 + 重要性门禁（见 §6），产出的候选事实走与 remember 相同的抽取/版本更替入图，`source_pipeline="session_improve"`
5. 处理成功的会话条目标记 `synced=true`（保留在缓存中供 recall，超上限裁剪最旧）

### 5.4 recall(query, dataset?, session_id?, top_k) — 同步，自动路由

1. query 向量化 → LanceDB 检索：`text_chunks` 与 `entities` 两张表各取 top_k 作为入口
2. Kuzu 图扩展：从入口实体出发遍历 1 跳邻居（v1 固定 1 跳，只看 `is_latest=True`），收集三元组事实与关联 chunk id（chunk 原文按需回查 LanceDB）
3. 汇总：向量分数为主排序，图扩展结果作为关联上下文附后；返回结构化结果 = 事实三元组列表 + 原文 chunk + 来源 dataset
4. 带 session_id 时先查 Redis 会话缓存，命中项标注 `source="session"`（未过滤原料），图结果标注 `source="graph"`（已验证事实），agent 自行区分置信度

## 6. 会话记忆防污染机制（对齐 cognee）

| 机制 | v1 落地 |
|---|---|
| ① 认知过滤器 | 会话记忆进图**必须**经过 improve 管线的 LLM 过滤：区分「持久事实/偏好」与「一次性事件/闲聊」，后者直接丢弃不进图。不允许任何会话原文直通图/向量库 |
| ② 重要性门禁 | 过滤器内含 keep/discard 判定（"值得跨会话长期记住吗"）。**不存储数值权重、不参与排序**（与去掉权重字段的决策一致），仅作进图前的一次性闸门 |
| ③ 上下文隔离 | 存储隔离：会话原文只存 Redis，永不进图/向量库；血缘隔离：会话来源事实标 `source_pipeline="session_improve"`，可溯源；检索隔离：recall 返回按 `source="session"/"graph"` 标注。配套内部方法 `forget_by_source(source_pipeline)`（core 层，v1 不暴露 MCP/REST 接口）：删除指定来源的全部关系事实（SQLite edges + Kuzu RELATES_TO，含历史版本）并置对应 data 记录状态；实体节点可能被多个来源共享，故保留不删 |

## 7. 接口设计

### 7.1 MCP 工具（Streamable HTTP，挂载 `/mcp`，v1 共 6 个）

| 工具 | 参数 | 说明 |
|---|---|---|
| `remember` | content, dataset="default", session_id=None | 双模式：无 session_id 同步入图；有则写缓存+后台 improve |
| `recall` | query, dataset=None, top_k=5, session_id=None | 混合检索自动路由 |
| `forget` | dataset=None, confirm=False | 删指定数据集（级联清三库+相关缓存）；省略 dataset 且 confirm=True 时清空全部记忆 |
| `cognify_file` | file_name, content_base64, dataset="default" | base64 文件同步落 data 记录、后台异步 cognify，返回 task_id |
| `task_status` | task_id | 查询异步任务状态/错误 |
| `list_datasets` | — | 列出数据集及规模 |

### 7.2 REST 开放 API（与 MCP 一一对应，同一套 core 函数）

| 端点 | 对应工具 |
|---|---|
| `POST /v1/memories` | remember |
| `POST /v1/recall` | recall |
| `POST /v1/documents`（JSON base64） | cognify_file |
| `GET /v1/tasks/{task_id}` | task_status |
| `GET /v1/datasets` | list_datasets |
| `DELETE /v1/datasets/{name}`；`DELETE /v1/memories?confirm=true` | forget |
| `GET /v1/health` | 健康检查（免认证） |

认证：除 `/v1/health` 外，所有端点（含 `/mcp`）校验 `Authorization: Bearer <WEAVE_API_KEY>`。

## 8. 错误处理

- **LLM/embedding 调用**：3 次指数退避重试。同步 remember 失败 → 图/向量不落半成品（先抽取成功，再统一写三库）；`data` 记录已落库的标记 `status=failed`，可凭 content_hash 幂等重试。异步任务失败 → `pipeline_runs` 置 failed + error，可重新提交
- **DB 原生调用防护**：所有 Kuzu/LanceDB 原生调用经专用单线程执行器提交，外层 `asyncio.wait_for` 包裹 `DB_CALL_TIMEOUT`（默认 30s）超时；锁竞争类瞬时错误重试 `DB_CALL_MAX_RETRIES`（默认 2）次，约束冲突等确定性错误不重试。已知限制：线程内真正卡死的原生调用无法强杀（线程泄漏至进程重启），但事件循环与 API 保持可用、调用方收到超时错误；若后续遇到 Kuzu 稳定性问题，再升级为 cognee 式子进程 harness
- **三库一致性**：无分布式事务，采用"SQLite 状态字段 + 顺序写"，v1 接受最终一致，失败记录可人工排查
- **文件锁冲突**：启动时检测 Kuzu/LanceDB 被其他进程占用 → 清晰报错退出
- **队列**：单 worker 单循环串行消费（匹配嵌入式 DB 单写者）。优先级实现模式写死：循环内单次 `BRPOP [weave:queue:improve, weave:queue:cognify]`（key 顺序即优先级，Redis 按 key 顺序扫描，improve 有货必先发）+ `QUEUE_POLL_TIMEOUT`（默认 5s）防止永久阻塞以便响应关闭信号；**禁止**拆成两个队列各自独立 BRPOP（两个并发消费者会使优先级失效）。队列积压打日志
- **错误返回**：REST 标准状态码 + `{error: {code, message}}`；MCP 工具返回结构化 `isError` 内容，agent 可读

## 9. 测试策略

- **单元**：切块器、抽取结果解析与版本更替逻辑（mock LLM）、检索排序（mock embedding）、improve 过滤器 keep/discard 解析
- **集成**：tmp 目录跑真实 SQLite/LanceDB/Kuzu + fakeredis，覆盖两条闭环：`remember→recall`、`remember(session)→improve→recall(graph)`
- **E2E**：真实启动服务，用 MCP Streamable HTTP client 跑全链路（remember / cognify_file / recall / forget）
- CI 全部 mock/录制，不依赖真实 API key

## 10. 配置（.env）

已有：DashScope（embedding）、DeepSeek（抽取 LLM）、LanceDB/Kuzu/SQLite 路径、Redis 连接。需补充：

```
WEAVE_API_KEY=<静态 token>
WEAVE_HOST=127.0.0.1
WEAVE_PORT=8000
SESSION_MAX_ITEMS=50        # 会话缓存上限
SESSION_TTL_DAYS=7          # 会话缓存 TTL
CHUNK_SIZE=1500             # 切块字符数
CHUNK_OVERLAP=200
DB_CALL_TIMEOUT=30          # Kuzu/LanceDB 原生调用超时(秒)
DB_CALL_MAX_RETRIES=2       # 锁竞争类瞬时错误重试次数
QUEUE_POLL_TIMEOUT=5        # worker BRPOP 阻塞超时(秒)
```

## 11. MySmallAgent 集成说明

mcp.json 配置示例：

```json
{
  "mcpServers": {
    "weave": { "url": "http://localhost:8000/mcp" }
  }
}
```

工具注册后名为 `mcp_weave_remember` 等。**风险点**：MySmallAgent 当前 mcp.json 示例未体现自定义 header 能力，若其 MCP 客户端不支持 Authorization header，需二选一：给 MySmallAgent 增加 header 配置项（推荐），或 Weave 额外支持 `?api_key=` query 参数（不推荐，仅兜底）。

## 12. 非目标（v1 明确不做）

- 权重/反馈/重要性衰减排序、memify 式后台图巩固
- 多用户/多租户/权限体系
- 实体消歧、本体（ontology）自动生成
- 图谱可视化 UI、开发者规则管理工具（tool.txt 中 cognee 的 UI/规则类工具）
- pdf/docx 等复杂格式解析（v1 仅 txt/md，预留 loader 接口）
- 分布式部署、独立 worker 进程（嵌入式 DB 单写者约束下无意义）
