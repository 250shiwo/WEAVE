# Dockerfile: Weave - AI 记忆管理平台
# 镜像用途：通过 MCP Streamable HTTP 和 REST API 为 AI agent 提供长期记忆能力
# 
# 特性:
# - Python 3.11 官方精简基础镜像 (slim)
# - uv 加速依赖安装与执行
# - 非 root 用户运行，安全加固
# - 健康检查端口：8000 (REST/MCP), 6379 (Redis, 如需内置)
# - 数据持久化卷：/data (SQLite/LanceDB/Kuzu 存储), /tmp (缓存)

FROM python:3.11-slim AS base

LABEL name="weave"
LABEL description="Knowledge-graph long-term memory platform for AI agents"
LABEL version="v0.1.0"
LABEL maintainer="weave-team"

# --- 构建阶段 ---
FROM base AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 复制 pyproject.toml 到构建上下文根目录并安装
COPY pyproject.toml .

# uv sync: 安装生产依赖 + dev 工具 (用于 run 脚本)
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev --no-editable --platform linux-x86_64 --python-version 3.11

# --- 最终运行时镜像 ---
FROM base AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    WEAVE_HOST="0.0.0.0" \
    WEAVE_PORT="8000" \
    REDIS_HOST="redis" \
    REDIS_PORT="6379" \
    REDIS_DB="0" \
    VECTOR_DB_PATH="/data/vector_db" \
    GRAPH_DB_PATH="/data/graph_db" \
    RELATIONAL_DB_PATH="/data/weave.db" \
    CHUNK_SIZE="1500" \
    CHUNK_OVERLAP="200" \
    DB_CALL_TIMEOUT="30.0" \
    DB_CALL_MAX_RETRIES="2" \
    QUEUE_POLL_TIMEOUT="5" \
    SESSION_MAX_ITEMS="50" \
    SESSION_TTL_DAYS="7" \
    DEEPSEEK_MODEL="deepseek-v4-flash" \
    DASHSCOPE_EMBEDDING_MODEL="qwen3.7-text-embedding"

# 非 root 用户创建
RUN groupadd -r weave && useradd -r -g weave weave \
    && mkdir -p /data /tmp/caches \
    && chown -R weave:weave /data /tmp/caches

WORKDIR /app

# 从 builder 安装的生产环境
COPY --from=builder --chown=weave:weave /build/.venv /app/.venv
COPY --chown=weave:weave weave /app/weave
COPY --chown=weave:weave pyproject.toml /app/

# 健康检查命令（验证 HTTP 和 MCP 可用）
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${WEAVE_PORT}/v1/health || exit 1

# 暴露端口：REST API + MCP Streamable HTTP
EXPOSE ${WEAVE_PORT}

# 启动入口（默认 uv run weave）
USER weave
CMD ["uv", "run", "weave"]
