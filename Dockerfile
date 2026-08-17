# syntax=docker/dockerfile:1
# ============================================
# Weave - 多阶段构建镜像
#
# 阶段一 (builder): 使用 uv 按 uv.lock 冻结安装全部生产依赖到 /app/.venv，
#                   并将 weave 包自身安装进去（hatchling 构建）。
# 阶段二 (runtime): 仅拷贝 .venv 与源码，非 root 用户运行，内置健康检查。
#
# 说明：REST /v1 与 MCP /mcp 同端口（默认 8000）；worker 以协程内嵌于
#       API 进程（spec §8：嵌入式库单写者约束下不拆独立 worker 容器）。
# ============================================

# ---------- 构建阶段 ----------
# 若 ghcr.io 拉取困难，可改为 python:3.12-slim-bookworm 并先执行
# `RUN pip install --no-cache-dir uv`，后续命令不变。
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# UV_COMPILE_BYTECODE=1：安装时预编译 .pyc，加快容器首次启动速度；
# UV_LINK_MODE=copy：uv 默认用硬链接节省空间，跨阶段拷贝时硬链接会失效，
# 改为直接复制，避免运行时出现悬空链接警告。
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 第一层：仅借助 bind mount 读取依赖清单安装第三方依赖（不装项目自身）。
# 依赖清单不变时该层命中构建缓存，日常改源码不会触发依赖重装。
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# 第二层：拷贝真实源码，再安装 weave 包自身。
# --frozen 严格按 uv.lock 解析，保证镜像与开发环境依赖完全一致。
COPY pyproject.toml uv.lock ./
COPY weave ./weave
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- 运行阶段 ----------
FROM python:3.12-slim-bookworm AS runtime

# 固定 uid/gid 的非 root 用户：宿主机挂载卷时便于对齐文件属主权限
RUN groupadd --system --gid 1000 weave \
    && useradd --system --uid 1000 --gid weave --home-dir /app weave

WORKDIR /app

# 从构建阶段拷贝虚拟环境与源码；--chown 保证运行时用户可读可执行
COPY --from=builder --chown=weave:weave /app/.venv /app/.venv
COPY --from=builder --chown=weave:weave /app/weave /app/weave

# 数据目录：LanceDB 向量库 / Kuzu 图库 / SQLite 关系库的默认落盘位置
# （对应配置项 ./data/vector_db、./data/graph_db、./data/weave.db）。
# 预建并授权：named volume 首次挂载会继承镜像内该目录的属主与权限，
# 保证非 root 的 weave 用户对挂载卷可写。
RUN mkdir -p /app/data && chown weave:weave /app/data

# .venv 前置到 PATH，python / weave 命令直接可用；
# PYTHONUNBUFFERED 让日志实时输出到 docker logs；
# 容器内必须监听 0.0.0.0 才能接收宿主机端口映射转发的流量。
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WEAVE_HOST=0.0.0.0 \
    WEAVE_PORT=8000

USER weave

EXPOSE 8000

# slim 镜像不含 curl，改用 Python 标准库探活；
# /v1/health 是认证豁免端点（weave/api/auth.py），无需携带 token。
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('WEAVE_PORT','8000')+'/v1/health', timeout=4)"]

# 入口等价于 `uv run weave`：装配三库 + Redis + LLM 客户端后以 uvicorn 常驻
CMD ["python", "-m", "weave.main"]
