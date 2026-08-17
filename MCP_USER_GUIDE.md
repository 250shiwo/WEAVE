# Weave MCP Server 用户指南

## 快速部署

### Docker Compose 启动（推荐）

```bash
# 1. 复制环境变量模板并修改密钥
cp .env.example .env
# 编辑 .env:
#   - WEAVE_API_KEY=<你的安全 token>
#   - DASHSCOPE_API_KEY=<DashScope API Key>
#   - DEEPSEEK_API_KEY=<DeepSeek API Key>

# 2. 一键启动
docker-compose --profile full up -d

# 3. 查看日志
docker-compose logs -f weave

# 4. 检查健康状态
curl http://localhost:8000/v1/health
# {"status":"ok"}

# 5. 停止服务
docker-compose down
```

### 生产环境部署

```bash
# 使用外部 Redis（AWS ElastiCache、Redis Cloud 等）
# 在 .env 中设置 REDIS_HOST=your-redis-endpoint.redislabs.com
docker-compose --profile production build -t weave:latest
docker-compose run weave  # 后台运行
```

## MCP 客户端配置

### MySmallAgent 配置 (mcp.json)

```json
{
  "mcpServers": {
    "weave": {
      "url": "http://host.docker.internal:8000/mcp"
    }
  }
}
```

**Windows 本地开发**: 使用 `http://host.docker.internal:8000/mcp`  
**Docker 内其他容器**: 使用 `http://weave:8000/mcp` (compose 网络)

### 通用 HTTP MCP 配置

任何支持 Streamable HTTP 传输的 MCP 客户端均可通过以下端点调用：

```
POST http://your-server:8000/mcp
Authorization: Bearer <WEAVE_API_KEY>
Content-Type: application/json
```

请求体格式:
```json
{
  "method": "tools/call",
  "params": {
    "name": "remember",
    "arguments": {
      "content": "用户的长期偏好是简洁回答",
      "dataset": "preferences"
    }
  },
  "id": 1
}
```

## 可用工具

### 记忆管理

| 工具 | 功能 | 示例 |
|------|------|------|
| `remember` | 存储永久记忆或会话缓存 | `{ "content": "我喜欢深烘焙咖啡", "dataset": "likes", "session_id": null }` |
| `recall` | 混合检索 (向量+图) | `{ "query": "用户喜欢什么", "top_k": 5, "session_id": "chat-1" }` |
| `forget` | 删除数据集 | `{ "dataset": "temp", "confirm": false }` |

### 文档摄取

| 工具 | 功能 | 示例 |
|------|------|------|
| `cognify_file` | base64 文档异步摄取 | `{ "file_name": "report.txt", "content_base64": "...", "dataset": "docs" }` |
| `task_status` | 查询异步任务进度 | `{ "task_id": "abc123..." }` |

### 元数据

| 工具 | 功能 |
|------|------|
| `list_datasets` | 列出所有数据集统计 |

## MCP 工具参数详解

### remember(content, dataset="default", session_id=None)

**永久记忆模式** (`session_id=null`):
- 立即 LLM 抽取实体关系 → 写入三库
- 失败重试机制自动兜底

**会话记忆模式** (`session_id="xxx"`):
- 只写 Redis 缓存 + 入队 improve
- 后台异步过滤沉淀进图
- 防污染机制保证闲聊不污染知识库

### recall(query, dataset=None, top_k=5, session_id=None)

- **自动路由**: query 向量化 → 双表取入口 → Kuzu 扩展 → 汇总排序
- **数据集过滤**: `dataset="default"` 限定范围
- **会话叠加**: `session_id="chat-1"` 时返回未过滤原文供上下文参考
- **性能**: 单次 recall 约 300ms (embedding + 向量检索 + 图遍历)

### cognify_file(file_name, content_base64, dataset="default")

**异步处理流程**:
1. 验证 base64 → 存 data 记录 (pending)
2. 切块 → 入队列 (improve/cognify queue)
3. worker 消费 → LLM 抽取 → 写入三库 (completed/failed)
4. 返回 task_id → 用 task_status 轮询

**支持格式**: `.txt`, `.md`, `.markdown`  
**限制**: DashScope embedding 单批上限 20 条 (已内部分批)

## REST API 对照

MCP 工具底层全部走 REST `/v1/*` 接口：

| MCP | REST | 说明 |
|-----|------|------|
| `remember` | POST `/v1/memories` | JSON body |
| `recall` | POST `/v1/recall` | JSON body |
| `forget(dataset)` | DELETE `/v1/datasets/{name}` | path 参数 |
| `forget(confirm)` | DELETE `/v1/memories?confirm=true` | query param |
| `cognify_file` | POST `/v1/documents` | base64 + filename |
| `task_status(task_id)` | GET `/v1/tasks/{id}` | path 参数 |
| `list_datasets` | GET `/v1/datasets` | 无参 |

健康检查无需认证：`GET /v1/health`

## 故障排查

### 容器无法启动

**症状**: `Exit code 1`, 日志显示 `IO exception: Could not set lock on file`  
**原因**: SQLite/Kuzu 文件被占用 (旧进程残留)  
**解决**:
```bash
docker-compose stop redis weave
docker volume rm weave_data
docker-compose up -d
```

### Embedding 400 错误

**症状**: `batch size is invalid, it should not be larger than 20`  
**原因**: 长文档提取出 >20 个实体，一次性提交超限  
**解决**: v0.1 已内置分批逻辑（20 条/批），升级镜像即可

### 检索永远返回同一结果

**症状**: 新上传文档后 recall 仍只返回旧内容  
**原因**: 
1. 文档摄取失败 → 查 `task_status`
2. 数据集隔离 → `dataset` 参数不匹配
3. 版本更替 → 同主体关系最新优先 (spec §4.4)

**调试步骤**:
```bash
# 1. 检查任务状态
curl http://localhost:8000/v1/tasks/<task_id> \
  -H "Authorization: Bearer dev-key"

# 2. 刷新 datasets
curl http://localhost:8000/v1/datasets \
  -H "Authorization: Bearer dev-key"

# 3. 强制清空测试 (危险操作!)
curl -X DELETE "http://localhost:8000/v1/memories?confirm=true" \
  -H "Authorization: Bearer dev-key"
```

### MCP 连接超时

**症状**: client 报 "Connection refused" 或 timeout  
**原因**:
1. 端口映射缺失 → `docker ps` 确认 8000:8000
2. 防火墙阻止 → 检查 `netsh advfirewall firewall add rule`
3. Redis 不可达 → `WEAVE_API_KEY` 正确且 Redis 健康

**诊断**:
```bash
# 1. 检查容器状态
docker-compose ps

# 2. 进入 weave 容器 shell
docker exec -it weave-memories bash

# 3. 在容器内 curl localhost:8000/v1/health
# 成功则暴露问题；失败则查 weave 容器日志
docker-compose logs weave
```

## 最佳实践

### 数据集组织策略

建议按主题/会话划分数据集：
```json
{ "dataset": "user_preferences" }   // 用户画像/偏好
{ "dataset": "project_x_docs" }     // 项目文档知识
{ "dataset": "session_2026-08-17" } // 当天对话流 (临时)
```

批量遗忘前先用 `list_datasets` 确认名称：
```bash
docker exec -it weave-memories \
  uv run python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/v1/datasets', headers={'Authorization':'Bearer dev-key'}).read())"
```

### Session ID 规范

建议使用结构化命名便于清理:
```
chat-{session_id}-{timestamp}
memory_{user}_{date}
```

配合会话 TTL: `.env` 中 `SESSION_TTL_DAYS=7` (默认), 过期自动清理

### 性能调优

大数据集优化参数:
```yaml
environment:
  - CHUNK_SIZE=2000       # 增大 chunk 减少抽取次数 (牺牲精度)
  - DB_CALL_TIMEOUT=60.0  # 长文档摄取增加超时
  - QUEUE_POLL_TIMEOUT=10 # worker 轮询间隔
```

平衡精度/速度：`CHUNK_SIZE=1500` (原值) 对大多数场景足够

## API 密钥安全

**生产环境必须替换默认 key!**

1. `WEAVE_API_KEY`: 生成强随机 Token (如 `openssl rand -hex 32`)
2. `DASHSCOPE_API_KEY`: 阿里云 DashScope 控制台申请
3. `DEEPSEEK_API_KEY`: DeepSeek OpenAPI 平台申请

不要在代码/commit 中硬编码：
```bash
# ✅ 正确：从 env_file 读取
.env.local:
  WEAVE_API_KEY=${SECRET_KEY:-generated_random_token}

# ❌ 错误：直接写在 compose.yml
docker-compose.yml:
  environment:
    - WEAVE_API_KEY="hardcoded_secret"  # commit 到 git！
```

## 备份与恢复

### 数据导出 (SQLite + 向量库 + 图库)

```bash
# 备份整个 data 目录
docker cp weave-memories:/data ./backup_weave_$(date +%Y%m%d).tar.gz
tar czf backup_$(date +%F).tar.gz ./data

# 或者仅导出 SQLite (便于迁移)
docker exec weave-memories \
  sqlite3 /data/weave.db ".dump" > weavedb.sql
```

### 数据恢复

```bash
# 替换数据卷
docker-compose down
rm -rf data
mkdir -p data
# 解压缩备份...
docker-compose up -d
```

## 常见问题 FAQ

**Q: 能否只用 Redis 缓存，不用它持久化？**  
A: 可以设置 `REDIS_HOST=""` 禁用队列，但记得 `weave_worker` 需要队列消费任务。建议保持 Redis 用于会话和任务编排。

**Q: 向量库/图库能换吗？**  
A: v0.1 仅支持 LanceDB/Kuzu/SQLite。若需 Neo4j/Qdrant/PgVector 需改 infra/层适配，预留了接口但未实现。

**Q: 如何监控资源消耗？**  
A: `docker stats weave-memories` 查看 CPU/内存/Disk I/O; 日志包含 embedding 耗时统计。典型场景：<1GB RAM, ~200MB Disk/天。

---

**更新日志**:
- 2026-08-14: 修复 DashScope 嵌入分批 (max 20); CORS 预检兼容
- 2026-08-11: 初始 MCP 支持上线

**联系支持**: GitHub Issues / internal Slack channel
