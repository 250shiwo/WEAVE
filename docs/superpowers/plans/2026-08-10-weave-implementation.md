# Weave 知识图谱记忆平台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Weave —— 借鉴 cognee 的单进程知识图谱记忆平台，通过 MCP (Streamable HTTP) 与 REST API 为 AI agent 提供跨会话持久化、版本化演进的长期记忆。

**Architecture:** 单 Python 进程：FastAPI(REST `/v1/*`) + FastMCP(`/mcp`) + 进程内 asyncio worker 消费 Redis 队列。SQLite/LanceDB/Kuzu 嵌入式三库经专用单线程执行器（max_workers=1）串行访问并带超时重试；LLM(DeepSeek)/embedding(DashScope) 用 AsyncOpenAI 原生异步。详见 `docs/superpowers/specs/2026-08-10-weave-design.md`（下称 spec）。

**Tech Stack:** Python 3.11+, uv, FastAPI, mcp(FastMCP), pydantic-settings, SQLAlchemy 2.0, lancedb, kuzu, redis(redis.asyncio), openai(AsyncOpenAI), pytest + pytest-asyncio + fakeredis。

**Spec 细化说明（不改变已确认决策）:** spec §3 将 LLM 与 DB 原生调用合并描述；本计划按 §8 的明确表述执行——单线程执行器只串行化 Kuzu/LanceDB/SQLite 原生调用，LLM/embedding 走 AsyncOpenAI 原生异步，避免长文档 cognify 阻塞 remember。

## Global Constraints

- Python `>=3.11`；依赖与命令一律用 `uv`（`uv sync` / `uv run pytest ...`）
- shell 是 Windows PowerShell：**禁止 `&&`**，多命令用 `;` 分隔
- `.env` 含真实 API key，**严禁提交、严禁打印**；`.gitignore` 已排除
- 所有 SQLite/LanceDB/Kuzu 原生调用**必须**经 `run_db()`（单线程执行器 + 超时 + 重试），禁止在事件循环线程直接调用
- Redis 队列常量：`QUEUE_IMPROVE = "weave:queue:improve"`，`QUEUE_COGNIFY = "weave:queue:cognify"`；worker 必须单循环单次 `BRPOP [QUEUE_IMPROVE, QUEUE_COGNIFY]`（key 序即优先级）
- 版本更替规则：同一 `(source_entity_id, relationship_type)` 新客体 → 旧边 `is_latest=False`、新边 `version+1`；检索只看 `is_latest=True`
- 实体 ID 确定性生成：`uuid5(NS, f"{dataset}:{norm_name}")`；边 ID：`uuid5(NS, f"{src_id}:{rel_type}:{dst_id}")`
- 会话原文只存 Redis，永不进图/向量库；会话衍生事实 `source_pipeline="session_improve"`
- pytest 配置 `asyncio_mode = "auto"`；测试不依赖真实 API key（LLM/embedding 注入 fake）
- 每个 Task 结束必须全部测试通过并 commit

## 文件结构（新建）

```
pyproject.toml
weave/
├── __init__.py
├── config.py            # Settings (pydantic-settings)
├── main.py              # 入口: uvicorn 启动 create_app()
├── api/
│   ├── __init__.py
│   ├── app.py           # create_app(): 中间件 + lifespan(worker+mcp) + 路由 + mount /mcp
│   ├── auth.py          # bearer_check 中间件函数
│   ├── rest.py          # APIRouter /v1/*
│   └── mcp_server.py    # create_mcp(): FastMCP + 6 工具
├── core/
│   ├── __init__.py
│   ├── models.py        # DataPoint/Entity/TextChunk/ExtractedGraph/RecallResult + id 工具
│   ├── chunking.py      # split_text()
│   ├── extraction.py    # extract_graph() / filter_session_facts() + prompt + 解析
│   ├── session.py       # 会话缓存助手 (append/get/unsynced/mark_synced)
│   ├── pipelines.py     # ingest_text() / improve_session() / run_cognify_task()
│   ├── retrieval.py     # hybrid_recall()
│   ├── cleanup.py       # forget_dataset() / forget_by_source()
│   └── service.py       # MemoryService 门面 (薄, 委托上面各模块)
├── infra/
│   ├── __init__.py
│   ├── executor.py      # run_db() 单线程执行器 + 超时 + 瞬时重试
│   ├── relational.py    # RelationalStore (SQLAlchemy/SQLite)
│   ├── vector.py        # VectorStore (LanceDB)
│   ├── graph.py         # GraphStore (Kuzu)
│   ├── cache.py         # Cache (redis.asyncio): 会话 + 队列
│   ├── llm.py           # with_retry() + LLMClient (DeepSeek)
│   └── embedding.py     # EmbeddingClient (DashScope)
└── worker.py            # worker_loop()
tests/
├── conftest.py          # settings/stores/fake_redis/fake_llm/fake_embedder fixtures
├── fakes.py             # FakeLLM / FakeEmbedder
└── test_*.py            # 每个 Task 指定
```

---

### Task 1: 项目脚手架与配置

**Files:**
- Create: `pyproject.toml`
- Create: `weave/__init__.py`（空文件）
- Create: `weave/config.py`
- Create: `tests/__init__.py`（空文件）
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`
- Modify: `.env`（追加 Weave 自身配置，**不提交**）

**Interfaces:**
- Consumes: 现有 `.env`（DASHSCOPE_*/DEEPSEEK_*/REDIS_*/*_DB_PATH）
- Produces: `Settings`（全部字段见代码）、`get_settings() -> Settings`；后续所有 Task 从 `weave.config import Settings`

- [ ] **Step 1: 写失败测试**

`tests/test_config.py`:

```python
from weave.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.setenv("WEAVE_API_KEY", "secret")
    s = Settings(_env_file=None)
    assert s.deepseek_api_key == "k1"
    assert s.weave_api_key == "secret"


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.chunk_size == 1500
    assert s.chunk_overlap == 200
    assert s.db_call_timeout == 30.0
    assert s.db_call_max_retries == 2
    assert s.queue_poll_timeout == 5
    assert s.session_max_items == 50
    assert s.session_ttl_days == 7
    assert s.weave_host == "127.0.0.1"
    assert s.weave_port == 8000
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave'`（若报 `uv` 无项目，先执行 Step 3 的 pyproject 再重跑确认 FAIL）

- [ ] **Step 3: 实现**

`pyproject.toml`:

```toml
[project]
name = "weave"
version = "0.1.0"
description = "Knowledge-graph long-term memory platform for AI agents"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "mcp>=1.9",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "sqlalchemy>=2.0",
    "lancedb>=0.17",
    "kuzu>=0.7",
    "redis>=5.0",
    "openai>=1.40",
    "httpx>=0.27",
]

[project.scripts]
weave = "weave.main:main"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "fakeredis>=2.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["weave"]
```

`weave/__init__.py` 与 `tests/__init__.py`：空文件。

`weave/config.py`:

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # DashScope（向量模型）
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_embedding_model: str = "qwen3.7-text-embedding"
    # DeepSeek（抽取模型）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    # 存储
    vector_db_path: str = "./data/vector_db"
    graph_db_path: str = "./data/graph_db"
    relational_db_path: str = "./data/weave.db"
    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    # Weave 服务
    weave_api_key: str = "dev-key"
    weave_host: str = "127.0.0.1"
    weave_port: int = 8000
    # 行为参数
    session_max_items: int = 50
    session_ttl_days: int = 7
    chunk_size: int = 1500
    chunk_overlap: int = 200
    db_call_timeout: float = 30.0
    db_call_max_retries: int = 2
    queue_poll_timeout: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`tests/conftest.py`:

```python
import pytest

from weave.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        vector_db_path=str(tmp_path / "vector_db"),
        graph_db_path=str(tmp_path / "graph_db"),
        relational_db_path=str(tmp_path / "weave.db"),
        weave_api_key="test-key",
        db_call_timeout=5.0,
        queue_poll_timeout=1,
    )
```

`.env` 追加（本地操作，不提交）:

```
WEAVE_API_KEY=<本地开发 token>
WEAVE_HOST=127.0.0.1
WEAVE_PORT=8000
SESSION_MAX_ITEMS=50
SESSION_TTL_DAYS=7
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
DB_CALL_TIMEOUT=30
DB_CALL_MAX_RETRIES=2
QUEUE_POLL_TIMEOUT=5
```

- [ ] **Step 4: 运行确认通过**

Run: `uv sync; uv run pytest tests/test_config.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml weave tests uv.lock; git commit -m "feat: 项目脚手架与配置管理"
```

---

### Task 2: DB 执行器（超时 + 瞬时重试）

**Files:**
- Create: `weave/infra/__init__.py`（空文件）
- Create: `weave/infra/executor.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: 无
- Produces: `async run_db(fn, *args, timeout: float = 30.0, max_retries: int = 2, **kwargs) -> Any`（Task 3/7/8 的全部 DB 访问入口）；`is_transient(exc) -> bool`

- [ ] **Step 1: 写失败测试**

`tests/test_executor.py`:

```python
import asyncio
import time

import pytest

from weave.infra.executor import run_db


async def test_run_db_success():
    assert await run_db(lambda x: x + 1, 41, timeout=1.0) == 42


async def test_run_db_timeout_keeps_loop_responsive():
    def stuck():
        time.sleep(5)

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await run_db(stuck, timeout=0.05, max_retries=0)
    # 事件循环未被拖死，后续调用可立即执行
    assert await run_db(lambda: "alive", timeout=1.0) == "alive"


async def test_run_db_no_retry_on_deterministic_error():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("constraint violation")

    with pytest.raises(ValueError):
        await run_db(bad, timeout=1.0, max_retries=2)
    assert calls["n"] == 1


async def test_run_db_retries_transient_lock_error():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Could not set lock on file")
        return "ok"

    assert await run_db(flaky, timeout=1.0, max_retries=2) == "ok"
    assert calls["n"] == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_executor.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.infra'`

- [ ] **Step 3: 实现**

`weave/infra/executor.py`:

```python
"""嵌入式 DB 调用的统一入口: 单线程执行器 + 超时 + 瞬时错误重试.

所有 SQLite/LanceDB/Kuzu 原生调用必须经 run_db() 提交:
- max_workers=1 串行化写入, 匹配嵌入式库单写者约束
- wait_for 超时防止原生调用卡死拖住事件循环 (线程内卡死无法强杀, 但 API 保持可用)
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weave-db")


def is_transient(exc: BaseException) -> bool:
    """锁竞争/超时类瞬时错误可重试; 约束冲突等确定性错误不重试."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    return "lock" in str(exc).lower()


async def run_db(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float = 30.0,
    max_retries: int = 2,
    **kwargs: Any,
) -> Any:
    loop = asyncio.get_running_loop()
    attempt = 0
    while True:
        try:
            future = loop.run_in_executor(_db_executor, lambda: fn(*args, **kwargs))
            return await asyncio.wait_for(future, timeout)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not is_transient(exc):
                raise
            await asyncio.sleep(0.2 * attempt)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_executor.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/infra tests/test_executor.py; git commit -m "feat: DB 单线程执行器 (超时+瞬时重试)"
```

---

### Task 3: SQLite 关系存储（RelationalStore）

**Files:**
- Create: `weave/infra/relational.py`
- Test: `tests/test_relational.py`

**Interfaces:**
- Consumes: `weave.infra.executor.run_db`（调用方负责包装，本类全同步）
- Produces（Task 9/12/13/14/15 依赖）:
  - `RelationalStore(path: str)`，`.close()`
  - `get_or_create_dataset(name: str, description: str = "") -> dict`（keys: id, name, description, created_at）
  - `list_datasets() -> list[dict]`
  - `dataset_stats() -> list[dict]`（keys: name, data_count, edge_count）
  - `create_data(data_id, name, raw_text, content_hash, source_pipeline) -> dict`
  - `get_data_by_hash(content_hash: str) -> dict | None`
  - `get_data(data_id: str) -> dict | None`
  - `set_data_status(data_id: str, status: str) -> None`
  - `link_dataset_data(dataset_id: str, data_id: str) -> None`
  - `upsert_edge(edge: dict) -> None`（edge keys: edge_id, source_id, target_id, relationship_type, dataset, version, is_latest, source_pipeline, source_task, created_at, updated_at）
  - `get_latest_edge(source_id: str, relationship_type: str) -> dict | None`
  - `set_edge_not_latest(edge_id: str, updated_at: str) -> None`
  - `delete_edges_by_source(source_pipeline: str) -> int`
  - `delete_dataset_rows(name: str) -> dict`（级联删 dataset/data/edges，返回 counts）
  - `create_pipeline_run(task_id: str, pipeline_name: str, data_id: str = "") -> None`
  - `update_pipeline_run(task_id: str, status: str, error: str = "") -> None`
  - `get_pipeline_run(task_id: str) -> dict | None`

- [ ] **Step 1: 写失败测试**

`tests/test_relational.py`:

```python
import pytest

from weave.infra.relational import RelationalStore


@pytest.fixture
def store(settings):
    s = RelationalStore(settings.relational_db_path)
    yield s
    s.close()


def test_dataset_and_data_crud(store):
    ds = store.get_or_create_dataset("default")
    assert ds["name"] == "default" and ds["id"]
    assert store.get_or_create_dataset("default")["id"] == ds["id"]  # 幂等

    store.create_data("d1", "note", "用户喜欢咖啡", "hash1", "remember")
    store.link_dataset_data(ds["id"], "d1")
    assert store.get_data_by_hash("hash1")["id"] == "d1"
    assert store.get_data("d1")["status"] == "created"
    store.set_data_status("d1", "completed")
    assert store.get_data("d1")["status"] == "completed"
    assert [d["name"] for d in store.list_datasets()] == ["default"]


def test_edge_metadata_supersede(store):
    base = dict(source_id="e1", target_id="e2", relationship_type="LIKES",
                dataset="default", version=1, is_latest=True,
                source_pipeline="remember", source_task="remember",
                created_at="2026-08-10T00:00:00+00:00",
                updated_at="2026-08-10T00:00:00+00:00")
    store.upsert_edge(dict(base, edge_id="edge1"))
    latest = store.get_latest_edge("e1", "LIKES")
    assert latest["edge_id"] == "edge1" and latest["is_latest"] is True

    store.set_edge_not_latest("edge1", "2026-08-10T01:00:00+00:00")
    store.upsert_edge(dict(base, edge_id="edge2", target_id="e3", version=2))
    latest = store.get_latest_edge("e1", "LIKES")
    assert latest["edge_id"] == "edge2" and latest["version"] == 2

    assert store.delete_edges_by_source("remember") == 2
    assert store.get_latest_edge("e1", "LIKES") is None


def test_pipeline_run_lifecycle(store):
    store.create_pipeline_run("t1", "cognify", "d1")
    assert store.get_pipeline_run("t1")["status"] == "pending"
    store.update_pipeline_run("t1", "running")
    store.update_pipeline_run("t1", "failed", "LLM error")
    run = store.get_pipeline_run("t1")
    assert run["status"] == "failed" and run["error"] == "LLM error"
    assert store.get_pipeline_run("missing") is None


def test_delete_dataset_rows_cascade(store):
    ds = store.get_or_create_dataset("temp")
    store.create_data("d1", "n", "x", "h1", "remember")
    store.link_dataset_data(ds["id"], "d1")
    store.upsert_edge(dict(edge_id="edge1", source_id="a", target_id="b",
                           relationship_type="R", dataset="temp", version=1,
                           is_latest=True, source_pipeline="remember",
                           source_task="remember", created_at="t", updated_at="t"))
    counts = store.delete_dataset_rows("temp")
    assert counts["data"] == 1 and counts["edges"] == 1
    assert store.get_data("d1") is None
    assert store.list_datasets() == []
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_relational.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.infra.relational'`

- [ ] **Step 3: 实现**

`weave/infra/relational.py`:

```python
"""SQLite 关系存储: datasets / data / dataset_data / edges / pipeline_runs.

全部为同步方法, 调用方必须经 run_db() 包装 (Global Constraints).
返回值为 dict 而非 ORM 对象, 避免跨线程 session 绑定问题.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, Integer, String, Text, create_engine, delete, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class DatasetRow(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    description: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(String)


class DataRow(Base):
    __tablename__ = "data"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    content_hash: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="created")
    source_pipeline: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(String)


class DatasetDataRow(Base):
    __tablename__ = "dataset_data"
    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    data_id: Mapped[str] = mapped_column(String, primary_key=True)


class EdgeRow(Base):
    __tablename__ = "edges"
    edge_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, index=True)
    target_id: Mapped[str] = mapped_column(String)
    relationship_type: Mapped[str] = mapped_column(String)
    dataset: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True)
    source_pipeline: Mapped[str] = mapped_column(String, default="")
    source_task: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(String)
    updated_at: Mapped[str] = mapped_column(String)


class PipelineRunRow(Base):
    __tablename__ = "pipeline_runs"
    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    pipeline_name: Mapped[str] = mapped_column(String)
    data_id: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String)
    updated_at: Mapped[str] = mapped_column(String)


def _to_dict(row) -> dict:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


class RelationalStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    # ---------- datasets ----------
    def get_or_create_dataset(self, name: str, description: str = "") -> dict:
        with Session(self.engine) as s:
            row = s.execute(select(DatasetRow).where(DatasetRow.name == name)).scalar_one_or_none()
            if row is None:
                row = DatasetRow(id=_new_id(), name=name, description=description, created_at=_utcnow())
                s.add(row)
                s.commit()
            return _to_dict(row)

    def list_datasets(self) -> list[dict]:
        with Session(self.engine) as s:
            return [_to_dict(r) for r in s.execute(select(DatasetRow)).scalars().all()]

    def dataset_stats(self) -> list[dict]:
        with Session(self.engine) as s:
            stats = []
            for ds in s.execute(select(DatasetRow)).scalars().all():
                data_count = s.execute(
                    select(func.count()).select_from(DatasetDataRow).where(DatasetDataRow.dataset_id == ds.id)
                ).scalar_one()
                edge_count = s.execute(
                    select(func.count()).select_from(EdgeRow).where(EdgeRow.dataset == ds.name)
                ).scalar_one()
                stats.append({"name": ds.name, "data_count": data_count, "edge_count": edge_count})
            return stats

    def delete_dataset_rows(self, name: str) -> dict:
        with Session(self.engine) as s:
            ds = s.execute(select(DatasetRow).where(DatasetRow.name == name)).scalar_one_or_none()
            if ds is None:
                return {"data": 0, "edges": 0}
            data_ids = [r.data_id for r in s.execute(
                select(DatasetDataRow).where(DatasetDataRow.dataset_id == ds.id)
            ).scalars().all()]
            s.execute(delete(DatasetDataRow).where(DatasetDataRow.dataset_id == ds.id))
            edge_n = s.execute(delete(EdgeRow).where(EdgeRow.dataset == name)).rowcount
            data_n = 0
            for did in data_ids:
                still_linked = s.execute(
                    select(func.count()).select_from(DatasetDataRow).where(DatasetDataRow.data_id == did)
                ).scalar_one()
                if still_linked == 0:
                    data_n += s.execute(delete(DataRow).where(DataRow.id == did)).rowcount
            s.execute(delete(DatasetRow).where(DatasetRow.id == ds.id))
            s.commit()
            return {"data": data_n, "edges": edge_n}

    # ---------- data ----------
    def create_data(self, data_id, name, raw_text, content_hash, source_pipeline) -> dict:
        with Session(self.engine) as s:
            row = DataRow(id=data_id, name=name, raw_text=raw_text, content_hash=content_hash,
                          status="created", source_pipeline=source_pipeline, created_at=_utcnow())
            s.add(row)
            s.commit()
            return _to_dict(row)

    def get_data_by_hash(self, content_hash: str) -> dict | None:
        with Session(self.engine) as s:
            row = s.execute(select(DataRow).where(DataRow.content_hash == content_hash)).scalar_one_or_none()
            return _to_dict(row) if row else None

    def get_data(self, data_id: str) -> dict | None:
        with Session(self.engine) as s:
            row = s.get(DataRow, data_id)
            return _to_dict(row) if row else None

    def set_data_status(self, data_id: str, status: str) -> None:
        with Session(self.engine) as s:
            row = s.get(DataRow, data_id)
            if row:
                row.status = status
                s.commit()

    def link_dataset_data(self, dataset_id: str, data_id: str) -> None:
        with Session(self.engine) as s:
            exists = s.get(DatasetDataRow, {"dataset_id": dataset_id, "data_id": data_id})
            if not exists:
                s.add(DatasetDataRow(dataset_id=dataset_id, data_id=data_id))
                s.commit()

    # ---------- edges ----------
    def upsert_edge(self, edge: dict) -> None:
        with Session(self.engine) as s:
            row = s.get(EdgeRow, edge["edge_id"])
            if row is None:
                s.add(EdgeRow(**edge))
            else:
                for k, v in edge.items():
                    setattr(row, k, v)
            s.commit()

    def get_latest_edge(self, source_id: str, relationship_type: str) -> dict | None:
        with Session(self.engine) as s:
            row = s.execute(
                select(EdgeRow).where(
                    EdgeRow.source_id == source_id,
                    EdgeRow.relationship_type == relationship_type,
                    EdgeRow.is_latest.is_(True),
                )
            ).scalar_one_or_none()
            return _to_dict(row) if row else None

    def set_edge_not_latest(self, edge_id: str, updated_at: str) -> None:
        with Session(self.engine) as s:
            row = s.get(EdgeRow, edge_id)
            if row:
                row.is_latest = False
                row.updated_at = updated_at
                s.commit()

    def delete_edges_by_source(self, source_pipeline: str) -> int:
        with Session(self.engine) as s:
            n = s.execute(delete(EdgeRow).where(EdgeRow.source_pipeline == source_pipeline)).rowcount
            s.commit()
            return n

    # ---------- pipeline runs ----------
    def create_pipeline_run(self, task_id: str, pipeline_name: str, data_id: str = "") -> None:
        with Session(self.engine) as s:
            s.add(PipelineRunRow(task_id=task_id, pipeline_name=pipeline_name, data_id=data_id,
                                 status="pending", created_at=_utcnow(), updated_at=_utcnow()))
            s.commit()

    def update_pipeline_run(self, task_id: str, status: str, error: str = "") -> None:
        with Session(self.engine) as s:
            row = s.get(PipelineRunRow, task_id)
            if row:
                row.status = status
                row.error = error
                row.updated_at = _utcnow()
                s.commit()

    def get_pipeline_run(self, task_id: str) -> dict | None:
        with Session(self.engine) as s:
            row = s.get(PipelineRunRow, task_id)
            return _to_dict(row) if row else None
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_relational.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/infra/relational.py tests/test_relational.py; git commit -m "feat: SQLite 关系存储 (datasets/data/edges/pipeline_runs)"
```

---

### Task 4: 核心数据模型与文本切块

**Files:**
- Create: `weave/core/__init__.py`（空文件）
- Create: `weave/core/models.py`
- Create: `weave/core/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: 无
- Produces（后续 Task 广泛使用）:
  - `utcnow() -> str`（ISO8601 UTC）、`new_id() -> str`、`norm_name(name: str) -> str`
  - `entity_id_for(dataset: str, name: str) -> str`（uuid5 确定性）、`edge_id_for(src_id: str, rel_type: str, dst_id: str) -> str`
  - `content_hash(text: str) -> str`（sha256 hex）
  - `DataPoint`（spec §4.1 全部字段）、`Entity(DataPoint)`（+name/norm_name/entity_type/description/dataset）、`TextChunk(DataPoint)`（+text/data_id/dataset/position）
  - `ExtractedEntity(name, entity_type="Thing", description="")`、`ExtractedRelationship(source, target, relationship_type, description="")`、`ExtractedGraph(entities=[], relationships=[])`
  - `RecallResult(facts: list[dict], chunks: list[dict], session_items: list[dict])`
  - `split_text(text: str, chunk_size: int, overlap: int) -> list[str]`

- [ ] **Step 1: 写失败测试**

`tests/test_chunking.py`:

```python
from weave.core.chunking import split_text
from weave.core.models import edge_id_for, entity_id_for, norm_name


def test_short_text_single_chunk():
    assert split_text("用户喜欢咖啡", 1500, 200) == ["用户喜欢咖啡"]
    assert split_text("   ", 1500, 200) == []


def test_paragraph_packing_respects_size():
    text = "\n\n".join(["段落一" * 10, "段落二" * 10, "段落三" * 10])
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 3
    assert all(len(c) <= 50 for c in chunks)


def test_long_paragraph_hard_split_with_overlap():
    text = "a" * 100
    chunks = split_text(text, chunk_size=40, overlap=10)
    assert len(chunks) == 3
    assert all(len(c) <= 40 for c in chunks)
    # 相邻块保留 10 字符重叠
    assert chunks[1][:10] == chunks[0][-10:]


def test_id_helpers_deterministic():
    assert norm_name("  User  Name ") == "user name"
    assert entity_id_for("default", "User") == entity_id_for("default", " user ")
    assert entity_id_for("default", "a") != entity_id_for("other", "a")
    assert edge_id_for("s", "LIKES", "d") == edge_id_for("s", "LIKES", "d")
    assert edge_id_for("s", "LIKES", "d") != edge_id_for("s", "HATES", "d")
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.core'`

- [ ] **Step 3: 实现**

`weave/core/models.py`:

```python
"""DataPoint 基类与 v1 节点类型 (spec §4.1/4.2) + 确定性 ID 工具.

去掉 feedback_weight/importance_weight; 版本化用 version + is_latest 显式表达.
实体/边使用确定性 uuid5: 同一 (dataset, 规范化名) 的实体天然幂等合并.
"""

import hashlib
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

_NS = uuid.UUID("8f1b3c2a-7d4e-4f5a-9b6c-1e2d3a4b5c6d")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def norm_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def entity_id_for(dataset: str, name: str) -> str:
    return uuid.uuid5(_NS, f"entity:{dataset}:{norm_name(name)}").hex


def edge_id_for(src_id: str, rel_type: str, dst_id: str) -> str:
    return uuid.uuid5(_NS, f"edge:{src_id}:{rel_type}:{dst_id}").hex


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DataPoint(BaseModel):
    id: str = Field(default_factory=new_id)
    version: int = 1
    type: str = "DataPoint"
    metadata: dict = Field(default_factory=dict)
    source_pipeline: str | None = None
    source_task: str | None = None
    source_node_set: list[str] = Field(default_factory=list)
    is_latest: bool = True
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)


class Entity(DataPoint):
    type: str = "Entity"
    name: str = ""
    norm_name: str = ""
    entity_type: str = "Thing"
    description: str = ""
    dataset: str = "default"


class TextChunk(DataPoint):
    type: str = "TextChunk"
    text: str = ""
    data_id: str = ""
    dataset: str = "default"
    position: int = 0


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str = "Thing"
    description: str = ""


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relationship_type: str
    description: str = ""


class ExtractedGraph(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class RecallResult(BaseModel):
    facts: list[dict] = Field(default_factory=list)
    chunks: list[dict] = Field(default_factory=list)
    session_items: list[dict] = Field(default_factory=list)
```

`weave/core/chunking.py`:

```python
"""文本切块: 段落优先组装, 超长段落硬切并保留重叠 (spec §5.1)."""


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        pieces = _hard_split(para, chunk_size, overlap) if len(para) > chunk_size else [para]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(1, chunk_size - overlap)
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + chunk_size])
        start += step
    return pieces
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/core tests/test_chunking.py; git commit -m "feat: DataPoint 数据模型与文本切块"
```

---

### Task 5: LLM 客户端与图抽取/会话过滤

**Files:**
- Create: `weave/infra/llm.py`
- Create: `weave/core/extraction.py`
- Create: `tests/fakes.py`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Consumes: `weave.core.models.ExtractedGraph`（Task 4）
- Produces:
  - `async with_retry(factory, retries: int = 3, base_delay: float = 0.5)`（Task 6 复用）
  - `LLMClient(api_key, base_url, model, client=None)`，`async complete_json(system: str, user: str) -> dict`
  - `EXTRACTION_SYSTEM` / `IMPROVE_SYSTEM`（prompt 常量）
  - `parse_extraction(payload: dict) -> ExtractedGraph`、`parse_improve(payload: dict) -> list[str]`
  - `async extract_graph(llm, text: str) -> ExtractedGraph`（llm 为任何有 `complete_json` 方法的对象，方便注入 fake）
  - `async filter_session_facts(llm, items: list[str]) -> list[str]`（返回 keep 的事实陈述句）
  - `tests/fakes.py`: `FakeLLM(responses)`（.calls 记录调用）、`FakeEmbedder(dim=8)`（确定性向量）

- [ ] **Step 1: 写失败测试**

`tests/fakes.py`:

```python
import hashlib
import math
import random


class FakeLLM:
    """按队列返回预设响应; .calls 记录每次调用."""

    def __init__(self, responses: list[dict] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    async def complete_json(self, system: str, user: str) -> dict:
        self.calls.append({"system": system, "user": user})
        if not self.responses:
            raise AssertionError("FakeLLM 无剩余响应")
        return self.responses.pop(0)


class FakeEmbedder:
    """确定性向量: 同文本同向量, 不同文本近似正交."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        rng = random.Random(hashlib.sha256(text.encode()).digest())
        v = [rng.uniform(-1, 1) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
```

`tests/test_extraction.py`:

```python
import pytest

from weave.core.extraction import (
    extract_graph,
    filter_session_facts,
    parse_extraction,
    parse_improve,
)
from weave.infra.llm import with_retry
from tests.fakes import FakeLLM


def test_parse_extraction_valid():
    payload = {
        "entities": [{"name": "用户", "entity_type": "Person", "description": "平台用户"}],
        "relationships": [{"source": "用户", "target": "深烘焙咖啡",
                           "relationship_type": "LIKES", "description": ""}],
    }
    g = parse_extraction(payload)
    assert g.entities[0].name == "用户"
    assert g.relationships[0].relationship_type == "LIKES"


def test_parse_extraction_tolerates_junk():
    g = parse_extraction({
        "entities": [{"name": "有效实体"}, {"no_name": True}, "garbage"],
        "relationships": [{"source": "a"}, {"source": "a", "target": "b", "relationship_type": "R"}],
    })
    assert [e.name for e in g.entities] == ["有效实体"]
    assert len(g.relationships) == 1
    assert parse_extraction({}).entities == []


def test_parse_improve_keep_only():
    payload = {"facts": [
        {"keep": True, "statement": "用户偏好简洁回答", "reason": "稳定偏好"},
        {"keep": False, "statement": "今天下雨了", "reason": "一次性事件"},
        {"keep": True, "statement": "", "reason": "空陈述丢弃"},
    ]}
    assert parse_improve(payload) == ["用户偏好简洁回答"]


async def test_extract_graph_calls_llm():
    llm = FakeLLM([{"entities": [{"name": "用户"}], "relationships": []}])
    g = await extract_graph(llm, "用户喜欢咖啡")
    assert g.entities[0].name == "用户"
    assert "用户喜欢咖啡" in llm.calls[0]["user"]


async def test_filter_session_facts_keep_discard():
    llm = FakeLLM([{"facts": [
        {"keep": True, "statement": "用户偏好简洁回答", "reason": "稳定偏好"},
        {"keep": False, "statement": "今天下雨了", "reason": "一次性事件"},
    ]}])
    kept = await filter_session_facts(llm, ["你说简洁点", "今天下雨了"])
    assert kept == ["用户偏好简洁回答"]


async def test_with_retry_succeeds_after_failures():
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert await with_retry(flaky, retries=3, base_delay=0.01) == "ok"
    assert attempts["n"] == 3


async def test_with_retry_gives_up():
    async def always_fail():
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        await with_retry(always_fail, retries=2, base_delay=0.01)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.infra.llm'`

- [ ] **Step 3: 实现**

`weave/infra/llm.py`:

```python
"""DeepSeek 抽取客户端 (OpenAI 兼容) + 通用异步重试 (3 次指数退避, spec §8)."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI


async def with_retry(
    factory: Callable[[], Awaitable[Any]], retries: int = 3, base_delay: float = 0.5
) -> Any:
    attempt = 0
    while True:
        try:
            return await factory()
        except Exception:
            attempt += 1
            if attempt >= retries:
                raise
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, client: AsyncOpenAI | None = None):
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def complete_json(self, system: str, user: str) -> dict:
        async def call() -> dict:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content or "{}")

        return await with_retry(call)
```

`weave/core/extraction.py`:

```python
"""实体关系抽取 (remember/cognify) 与会话过滤 (improve) — spec §5.1/§5.3/§6."""

from weave.core.models import ExtractedEntity, ExtractedGraph, ExtractedRelationship

EXTRACTION_SYSTEM = """你是知识图谱抽取器。从给定文本中抽取实体与关系，只输出 JSON：
{"entities": [{"name": "实体名", "entity_type": "类型如 Person/Preference/Project/Concept",
               "description": "一句话描述"}],
 "relationships": [{"source": "源实体名", "target": "目标实体名",
                    "relationship_type": "英文大写蛇形如 LIKES/WORKS_ON/LOCATED_IN",
                    "description": ""}]}
规则：实体名用规范名词；用户画像/偏好的实体类型用 Person/Preference；
没有可抽取内容时返回空数组。只输出 JSON，不要输出任何其他文字。"""

IMPROVE_SYSTEM = """你是长期记忆过滤器。给定 AI 助手的会话片段，判断哪些内容值得作为
跨会话的长期记忆（用户画像、偏好、稳定事实、重要决定），丢弃一次性事件、
闲聊、临时上下文。只输出 JSON：
{"facts": [{"keep": true, "statement": "精炼后的事实陈述句", "reason": "保留原因"},
            {"keep": false, "statement": "...", "reason": "丢弃原因"}]}
宁缺毋滥：不确定的一律 keep=false。只输出 JSON。"""


def parse_extraction(payload: dict) -> ExtractedGraph:
    if not isinstance(payload, dict):
        return ExtractedGraph()
    entities = [
        ExtractedEntity(
            name=str(e["name"]).strip(),
            entity_type=str(e.get("entity_type") or "Thing"),
            description=str(e.get("description") or ""),
        )
        for e in payload.get("entities") or []
        if isinstance(e, dict) and e.get("name")
    ]
    relationships = [
        ExtractedRelationship(
            source=str(r["source"]).strip(),
            target=str(r["target"]).strip(),
            relationship_type=str(r["relationship_type"]).strip().upper().replace(" ", "_"),
            description=str(r.get("description") or ""),
        )
        for r in payload.get("relationships") or []
        if isinstance(r, dict) and r.get("source") and r.get("target") and r.get("relationship_type")
    ]
    return ExtractedGraph(entities=entities, relationships=relationships)


def parse_improve(payload: dict) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return [
        str(f["statement"]).strip()
        for f in payload.get("facts") or []
        if isinstance(f, dict) and f.get("keep") is True and str(f.get("statement") or "").strip()
    ]


async def extract_graph(llm, text: str) -> ExtractedGraph:
    return parse_extraction(await llm.complete_json(EXTRACTION_SYSTEM, text))


async def filter_session_facts(llm, items: list[str]) -> list[str]:
    user = "\n".join(f"- {c}" for c in items)
    return parse_improve(await llm.complete_json(IMPROVE_SYSTEM, user))
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/infra/llm.py weave/core/extraction.py tests/fakes.py tests/test_extraction.py; git commit -m "feat: LLM 客户端、图抽取与会话过滤 (含重试)"
```

---

### Task 6: Embedding 客户端（DashScope）

**Files:**
- Create: `weave/infra/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `weave.infra.llm.with_retry`（Task 5）
- Produces: `EmbeddingClient(api_key, base_url, model, client=None)`，`async embed(texts: list[str]) -> list[list[float]]`（空列表返回空；Task 9/13 依赖）

- [ ] **Step 1: 写失败测试**

`tests/test_embedding.py`:

```python
from types import SimpleNamespace

from weave.infra.embedding import EmbeddingClient


class _StubEmbeddings:
    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times

    async def create(self, model, input):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("boom")
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])


def _client(fail_times: int = 0) -> EmbeddingClient:
    stub = SimpleNamespace(embeddings=_StubEmbeddings(fail_times))
    return EmbeddingClient("k", "http://x", "m", client=stub)


async def test_embed_returns_vectors():
    out = await _client().embed(["a", "b"])
    assert out == [[1.0, 0.0], [1.0, 0.0]]


async def test_embed_empty_input():
    assert await _client().embed([]) == []


async def test_embed_retries_transient_failure():
    assert await _client(fail_times=1).embed(["a"]) == [[1.0, 0.0]]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.infra.embedding'`

- [ ] **Step 3: 实现**

`weave/infra/embedding.py`:

```python
"""DashScope embedding 客户端 (OpenAI 兼容端点), 复用 with_retry."""

from openai import AsyncOpenAI

from weave.infra.llm import with_retry


class EmbeddingClient:
    def __init__(self, api_key: str, base_url: str, model: str, client: AsyncOpenAI | None = None):
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def call() -> list[list[float]]:
            resp = await self._client.embeddings.create(model=self._model, input=texts)
            return [d.embedding for d in resp.data]

        return await with_retry(call)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/infra/embedding.py tests/test_embedding.py; git commit -m "feat: DashScope embedding 客户端"
```

---

### Task 7: Kuzu 图存储（GraphStore）

**Files:**
- Create: `weave/infra/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `weave.core.models` 的 id 工具（测试中用）
- Produces（全同步，调用方经 `run_db` 包装；Task 9/13/14 依赖）:
  - `GraphStore(path: str)`，`.close()`
  - `upsert_entity(e: dict) -> None`（e keys: id/name/norm_name/entity_type/description/dataset/version/is_latest/created_at；按 id 存在则跳过）
  - `get_entity_by_id(entity_id: str) -> dict | None`
  - `get_entity_by_name(norm_name: str, dataset: str) -> dict | None`
  - `create_chunk(chunk_id: str, data_id: str, dataset: str, created_at: str) -> None`
  - `link_mentions(chunk_id: str, entity_ids: list[str]) -> None`
  - `mentioned_chunk_ids(entity_ids: list[str], dataset: str | None = None) -> list[str]`
  - `get_latest_relationship(source_id: str, rel_type: str) -> dict | None`（keys: edge_id/target_id/version）
  - `set_relationship_not_latest(edge_id: str) -> None`
  - `upsert_relationship(e: dict) -> None`（e keys: edge_id/source_id/target_id/relationship_type/version/is_latest/source_pipeline/dataset/created_at）
  - `neighbors(entity_ids: list[str], dataset: str | None = None) -> list[dict]`（1 跳 is_latest 边，keys: edge_id/source_id/source_name/relationship_type/target_id/target_name/version/dataset/source_pipeline）
  - `delete_dataset(dataset: str) -> None`、`delete_relationships_by_source(source_pipeline: str) -> int`、`count_entities(dataset: str | None = None) -> int`

注意：Kuzu 同步 API；查询参数用 `$param`；IN 列表内联（id 为 uuid hex，字符集安全）；`RELATES_TO` 边表带 `dataset`/`source_pipeline` 属性。

- [ ] **Step 1: 写失败测试**

`tests/test_graph.py`:

```python
import pytest

from weave.core.models import edge_id_for, entity_id_for, utcnow
from weave.infra.graph import GraphStore


@pytest.fixture
def graph(settings):
    g = GraphStore(settings.graph_db_path)
    yield g
    g.close()


def _entity(name: str, dataset: str = "default") -> dict:
    return dict(id=entity_id_for(dataset, name), name=name, norm_name=name.strip().lower(),
                entity_type="Thing", description="", dataset=dataset, version=1,
                is_latest=True, created_at=utcnow())


def _rel(src: str, rt: str, dst: str, version: int = 1, dataset: str = "default",
         sp: str = "remember") -> dict:
    return dict(edge_id=edge_id_for(src, rt, dst), source_id=src, target_id=dst,
                relationship_type=rt, version=version, is_latest=True,
                source_pipeline=sp, dataset=dataset, created_at=utcnow())


def test_entity_upsert_idempotent_and_get_by_name(graph):
    graph.upsert_entity(_entity("用户"))
    graph.upsert_entity(_entity("用户"))  # 幂等
    e = graph.get_entity_by_name("用户", "default")
    assert e["name"] == "用户" and e["entity_type"] == "Thing"
    assert graph.get_entity_by_name("不存在", "default") is None
    assert graph.count_entities("default") == 1


def test_relationship_supersession_flow(graph):
    u, light, dark = _entity("用户"), _entity("浅烘焙"), _entity("深烘焙")
    for e in (u, light, dark):
        graph.upsert_entity(e)
    graph.upsert_relationship(_rel(u["id"], "LIKES", light["id"]))
    latest = graph.get_latest_relationship(u["id"], "LIKES")
    assert latest["target_id"] == light["id"] and latest["version"] == 1

    # 版本更替: 旧边 is_latest=False, 新边 version=2
    graph.set_relationship_not_latest(latest["edge_id"])
    graph.upsert_relationship(_rel(u["id"], "LIKES", dark["id"], version=2))
    latest = graph.get_latest_relationship(u["id"], "LIKES")
    assert latest["target_id"] == dark["id"] and latest["version"] == 2


def test_neighbors_one_hop_with_dataset_filter(graph):
    u, coffee, tea = _entity("用户"), _entity("咖啡"), _entity("茶", "other")
    for e in (u, coffee, tea):
        graph.upsert_entity(e)
    graph.upsert_relationship(_rel(u["id"], "LIKES", coffee["id"]))
    graph.upsert_relationship(_rel(u["id"], "LIKES", tea["id"], dataset="other"))
    all_facts = graph.neighbors([u["id"]])
    assert len(all_facts) == 2
    default_facts = graph.neighbors([u["id"]], dataset="default")
    assert len(default_facts) == 1 and default_facts[0]["target_name"] == "咖啡"


def test_mentions_and_chunk_lookup(graph):
    e = _entity("咖啡")
    graph.upsert_entity(e)
    graph.create_chunk("c1", "d1", "default", utcnow())
    graph.link_mentions("c1", [e["id"]])
    assert graph.mentioned_chunk_ids([e["id"]]) == ["c1"]


def test_delete_dataset_detaches(graph):
    a, b = _entity("甲", "temp"), _entity("乙", "temp")
    graph.upsert_entity(a)
    graph.upsert_entity(b)
    graph.upsert_relationship(_rel(a["id"], "R", b["id"], dataset="temp"))
    graph.delete_dataset("temp")
    assert graph.count_entities("temp") == 0
    assert graph.neighbors([a["id"]]) == []


def test_delete_relationships_by_source(graph):
    a, b = _entity("甲"), _entity("乙")
    graph.upsert_entity(a)
    graph.upsert_entity(b)
    graph.upsert_relationship(_rel(a["id"], "R1", b["id"], sp="session_improve"))
    graph.upsert_relationship(_rel(b["id"], "R2", a["id"], sp="remember"))
    assert graph.delete_relationships_by_source("session_improve") == 1
    facts = graph.neighbors([a["id"]])
    assert len(facts) == 1 and facts[0]["relationship_type"] == "R2"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.infra.graph'`

- [ ] **Step 3: 实现**

`weave/infra/graph.py`:

```python
"""Kuzu 图存储 (spec §4.3): Entity/TextChunk 节点 + RELATES_TO/MENTIONS 边.

全同步方法, 调用方必须经 run_db() 包装. Kuzu 不支持关系主键,
边身份用 edge_id 属性 (确定性 uuid5, 见 core.models).
"""

from pathlib import Path

import kuzu

_SCHEMA = [
    "CREATE NODE TABLE IF NOT EXISTS Entity("
    "id STRING, name STRING, norm_name STRING, entity_type STRING, description STRING, "
    "dataset STRING, version INT64, is_latest BOOLEAN, created_at STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS TextChunk("
    "id STRING, data_id STRING, dataset STRING, created_at STRING, PRIMARY KEY(id))",
    "CREATE REL TABLE IF NOT EXISTS RELATES_TO("
    "FROM Entity TO Entity, edge_id STRING, relationship_type STRING, version INT64, "
    "is_latest BOOLEAN, source_pipeline STRING, dataset STRING, created_at STRING)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM TextChunk TO Entity)",
]


class GraphStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)
        for stmt in _SCHEMA:
            self._conn.execute(stmt)

    def close(self) -> None:
        conn, db = self._conn, self._db
        self._conn = self._db = None
        for obj in (conn, db):
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    # ---------- 内部 ----------
    def _rows(self, query: str, params: dict | None = None) -> list[dict]:
        result = self._conn.execute(query, params or {})
        cols = result.get_column_names()
        rows = []
        while result.has_next():
            rows.append(dict(zip(cols, result.get_next())))
        return rows

    @staticmethod
    def _id_list(ids: list[str]) -> str:
        return ",".join(f"'{i}'" for i in ids)  # uuid hex, 字符集安全

    # ---------- 实体 ----------
    def upsert_entity(self, e: dict) -> None:
        if self.get_entity_by_id(e["id"]):
            return
        self._conn.execute(
            "CREATE (n:Entity {id: $id, name: $name, norm_name: $norm_name, "
            "entity_type: $entity_type, description: $description, dataset: $dataset, "
            "version: $version, is_latest: $is_latest, created_at: $created_at})",
            dict(e),
        )

    def get_entity_by_id(self, entity_id: str) -> dict | None:
        rows = self._rows("MATCH (n:Entity {id: $id}) RETURN n", {"id": entity_id})
        return rows[0]["n"] if rows else None

    def get_entity_by_name(self, norm_name: str, dataset: str) -> dict | None:
        rows = self._rows(
            "MATCH (n:Entity) WHERE n.norm_name = $nn AND n.dataset = $ds RETURN n",
            {"nn": norm_name, "ds": dataset},
        )
        return rows[0]["n"] if rows else None

    def count_entities(self, dataset: str | None = None) -> int:
        if dataset:
            rows = self._rows(
                "MATCH (n:Entity) WHERE n.dataset = $ds RETURN count(n) AS n", {"ds": dataset})
        else:
            rows = self._rows("MATCH (n:Entity) RETURN count(n) AS n")
        return rows[0]["n"]

    # ---------- chunk 与 MENTIONS ----------
    def create_chunk(self, chunk_id: str, data_id: str, dataset: str, created_at: str) -> None:
        self._conn.execute(
            "CREATE (c:TextChunk {id: $id, data_id: $data_id, dataset: $dataset, created_at: $ca})",
            {"id": chunk_id, "data_id": data_id, "dataset": dataset, "ca": created_at},
        )

    def link_mentions(self, chunk_id: str, entity_ids: list[str]) -> None:
        for eid in entity_ids:
            self._conn.execute(
                "MATCH (c:TextChunk {id: $cid}), (e:Entity {id: $eid}) CREATE (c)-[:MENTIONS]->(e)",
                {"cid": chunk_id, "eid": eid},
            )

    def mentioned_chunk_ids(self, entity_ids: list[str], dataset: str | None = None) -> list[str]:
        if not entity_ids:
            return []
        q = (f"MATCH (c:TextChunk)-[:MENTIONS]->(e:Entity) "
             f"WHERE e.id IN [{self._id_list(entity_ids)}] ")
        params = {}
        if dataset:
            q += "AND c.dataset = $ds "
            params["ds"] = dataset
        q += "RETURN DISTINCT c.id AS chunk_id"
        return [r["chunk_id"] for r in self._rows(q, params)]

    # ---------- 关系 ----------
    def get_latest_relationship(self, source_id: str, rel_type: str) -> dict | None:
        rows = self._rows(
            "MATCH (a:Entity {id: $sid})-[r:RELATES_TO]->(b:Entity) "
            "WHERE r.relationship_type = $rt AND r.is_latest = true "
            "RETURN r.edge_id AS edge_id, b.id AS target_id, r.version AS version",
            {"sid": source_id, "rt": rel_type},
        )
        return rows[0] if rows else None

    def set_relationship_not_latest(self, edge_id: str) -> None:
        self._conn.execute(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.edge_id = $eid SET r.is_latest = false",
            {"eid": edge_id},
        )

    def upsert_relationship(self, e: dict) -> None:
        exists = self._rows(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.edge_id = $eid RETURN count(r) AS n",
            {"eid": e["edge_id"]},
        )[0]["n"]
        if exists:
            self._conn.execute(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.edge_id = $eid "
                "SET r.version = $version, r.is_latest = $is_latest, r.source_pipeline = $sp",
                {"eid": e["edge_id"], "version": e["version"],
                 "is_latest": e["is_latest"], "sp": e["source_pipeline"]},
            )
            return
        self._conn.execute(
            "MATCH (a:Entity {id: $src}), (b:Entity {id: $dst}) "
            "CREATE (a)-[:RELATES_TO {edge_id: $eid, relationship_type: $rt, version: $version, "
            "is_latest: $is_latest, source_pipeline: $sp, dataset: $ds, created_at: $ca}]->(b)",
            {"src": e["source_id"], "dst": e["target_id"], "eid": e["edge_id"],
             "rt": e["relationship_type"], "version": e["version"], "is_latest": e["is_latest"],
             "sp": e["source_pipeline"], "ds": e["dataset"], "ca": e["created_at"]},
        )

    def neighbors(self, entity_ids: list[str], dataset: str | None = None) -> list[dict]:
        if not entity_ids:
            return []
        ids = self._id_list(entity_ids)
        q = ("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
             f"WHERE r.is_latest = true AND (a.id IN [{ids}] OR b.id IN [{ids}]) ")
        params = {}
        if dataset:
            q += "AND r.dataset = $ds "
            params["ds"] = dataset
        q += ("RETURN r.edge_id AS edge_id, a.id AS source_id, a.name AS source_name, "
              "r.relationship_type AS relationship_type, b.id AS target_id, "
              "b.name AS target_name, r.version AS version, r.dataset AS dataset, "
              "r.source_pipeline AS source_pipeline")
        return self._rows(q, params)

    # ---------- 清理 ----------
    def delete_dataset(self, dataset: str) -> None:
        self._conn.execute("MATCH (c:TextChunk {dataset: $ds}) DETACH DELETE c", {"ds": dataset})
        self._conn.execute("MATCH (e:Entity {dataset: $ds}) DETACH DELETE e", {"ds": dataset})

    def delete_relationships_by_source(self, source_pipeline: str) -> int:
        n = self._rows(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.source_pipeline = $sp RETURN count(r) AS n",
            {"sp": source_pipeline},
        )[0]["n"]
        if n:
            self._conn.execute(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.source_pipeline = $sp DELETE r",
                {"sp": source_pipeline},
            )
        return n
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_graph.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/infra/graph.py tests/test_graph.py; git commit -m "feat: Kuzu 图存储 (实体/关系/版本更替/1跳检索)"
```

---

### Task 8: LanceDB 向量存储（VectorStore）

**Files:**
- Create: `weave/infra/vector.py`
- Test: `tests/test_vector.py`

**Interfaces:**
- Consumes: 无
- Produces（全同步，调用方经 `run_db` 包装；Task 9/13/14 依赖）:
  - `VectorStore(path: str)`，`.close()`
  - `add_chunks(rows: list[dict]) -> None`（row keys: chunk_id/vector/text/data_id/dataset/created_at）
  - `add_entities(rows: list[dict]) -> None`（row keys: entity_id/vector/name/entity_type/description/dataset/is_latest）
  - `search_chunks(vector: list[float], top_k: int, dataset: str | None = None) -> list[dict]`（含 `_distance`）
  - `search_entities(vector: list[float], top_k: int, dataset: str | None = None) -> list[dict]`
  - `get_chunks(chunk_ids: list[str]) -> list[dict]`（按 id 回查原文）
  - `delete_dataset(dataset: str) -> None`
  - 表不存在时 search/get 返回 `[]`（懒建表：首次 add 时以数据推断 schema）

- [ ] **Step 1: 写失败测试**

`tests/test_vector.py`:

```python
import pytest

from weave.infra.vector import VectorStore


@pytest.fixture
def vector(settings):
    v = VectorStore(settings.vector_db_path)
    yield v
    v.close()


def _chunk(cid: str, vec, text: str, dataset: str = "default") -> dict:
    return dict(chunk_id=cid, vector=vec, text=text, data_id="d1",
                dataset=dataset, created_at="2026-08-10T00:00:00+00:00")


def test_search_empty_db_returns_empty(vector):
    assert vector.search_chunks([1.0, 0.0], 5) == []
    assert vector.get_chunks(["x"]) == []


def test_add_and_search_chunks_ranking(vector):
    vector.add_chunks([
        _chunk("c1", [1.0, 0.0], "喜欢咖啡"),
        _chunk("c2", [0.0, 1.0], "喜欢茶"),
    ])
    hits = vector.search_chunks([1.0, 0.0], 2)
    assert hits[0]["chunk_id"] == "c1"  # 同向量距离最小
    assert {h["chunk_id"] for h in hits} == {"c1", "c2"}
    assert hits[0]["text"] == "喜欢咖啡"


def test_search_with_dataset_filter(vector):
    vector.add_chunks([
        _chunk("c1", [1.0, 0.0], "默认库内容"),
        _chunk("c2", [1.0, 0.0], "其他库内容", dataset="other"),
    ])
    hits = vector.search_chunks([1.0, 0.0], 5, dataset="other")
    assert [h["chunk_id"] for h in hits] == ["c2"]


def test_entities_table_and_get_chunks(vector):
    vector.add_entities([dict(entity_id="e1", vector=[1.0, 0.0], name="用户",
                              entity_type="Person", description="", dataset="default",
                              is_latest=True)])
    hits = vector.search_entities([1.0, 0.0], 1)
    assert hits[0]["entity_id"] == "e1" and hits[0]["name"] == "用户"

    vector.add_chunks([_chunk("c9", [0.5, 0.5], "回查文本")])
    rows = vector.get_chunks(["c9"])
    assert rows[0]["text"] == "回查文本"


def test_delete_dataset(vector):
    vector.add_chunks([_chunk("c1", [1.0, 0.0], "x"), _chunk("c2", [1.0, 0.0], "y", "other")])
    vector.delete_dataset("default")
    assert [h["chunk_id"] for h in vector.search_chunks([1.0, 0.0], 5)] == ["c2"]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_vector.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.infra.vector'`

- [ ] **Step 3: 实现**

`weave/infra/vector.py`:

```python
"""LanceDB 向量存储 (spec §4.3): text_chunks / entities 两张表, 懒建表.

全同步方法, 调用方必须经 run_db() 包装.
"""

from pathlib import Path

import lancedb

_TABLES = ("text_chunks", "entities")


def _esc(value: str) -> str:
    return value.replace("'", "''")


class VectorStore:
    def __init__(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(path)

    def close(self) -> None:
        self._db = None

    def _table(self, name: str):
        try:
            return self._db.open_table(name)
        except Exception:
            return None

    def _add(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        table = self._table(name)
        if table is None:
            self._db.create_table(name, rows)
        else:
            table.add(rows)

    def add_chunks(self, rows: list[dict]) -> None:
        self._add("text_chunks", rows)

    def add_entities(self, rows: list[dict]) -> None:
        self._add("entities", rows)

    def _search(self, name: str, vector: list[float], top_k: int,
                dataset: str | None = None) -> list[dict]:
        table = self._table(name)
        if table is None:
            return []
        query = table.search(vector)
        if dataset:
            query = query.where(f"dataset = '{_esc(dataset)}'")
        return query.limit(top_k).to_list()

    def search_chunks(self, vector: list[float], top_k: int,
                      dataset: str | None = None) -> list[dict]:
        return self._search("text_chunks", vector, top_k, dataset)

    def search_entities(self, vector: list[float], top_k: int,
                        dataset: str | None = None) -> list[dict]:
        return self._search("entities", vector, top_k, dataset)

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        table = self._table("text_chunks")
        if table is None or not chunk_ids:
            return []
        id_list = ",".join(f"'{_esc(i)}'" for i in chunk_ids)
        return table.search().where(f"chunk_id IN ({id_list})").to_list()

    def delete_dataset(self, dataset: str) -> None:
        for name in _TABLES:
            table = self._table(name)
            if table is not None:
                table.delete(f"dataset = '{_esc(dataset)}'")
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_vector.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/infra/vector.py tests/test_vector.py; git commit -m "feat: LanceDB 向量存储 (双表/检索/回查)"
```

---

### Task 9: MemoryService 门面与 remember 永久管线（ingest_text）

**Files:**
- Create: `weave/core/service.py`
- Create: `weave/core/pipelines.py`
- Modify: `weave/infra/relational.py`（新增 `is_data_linked` 方法）
- Modify: `tests/conftest.py`（新增 stores / make_service fixtures）
- Test: `tests/test_pipelines.py`

**Interfaces:**
- Consumes: Task 2-8 全部
- Produces:
  - `MemoryService(settings, relational, vector, graph, cache, llm, embedder)`；`async svc.db(fn, *args, **kwargs)`（= run_db + settings 超时/重试，所有 DB 调用的统一包装）
  - `async remember(content: str, dataset: str = "default", session_id: str | None = None) -> dict`（本 Task 只实现 session_id=None 分支）
  - `async list_datasets() -> list[dict]`（name/data_count/edge_count/entity_count）
  - `pipelines.ingest_text(svc, text, dataset, source_pipeline, source_task, data_name, data_id=None) -> dict`（返回 keys: data_id/dataset/deduplicated/entities/relationships/superseded/chunks）
  - `relational.is_data_linked(dataset_id: str, data_id: str) -> bool`

**关键实现约束（spec §8）:** 三阶段——① 所有 chunk 的 LLM 抽取先全部完成（失败则不写任何图/向量）；② 内存中归并实体并计算版本更替（只读 DB）；③ 统一写三库。`data_id` 传入时（cognify 路径）跳过去重与建记录，直接使用已有 data 记录。

- [ ] **Step 1: 写失败测试**

`tests/conftest.py` 追加（保留已有 settings fixture）:

```python
from weave.core.service import MemoryService
from weave.infra.graph import GraphStore
from weave.infra.relational import RelationalStore
from weave.infra.vector import VectorStore
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture
def stores(settings):
    rel = RelationalStore(settings.relational_db_path)
    vec = VectorStore(settings.vector_db_path)
    graph = GraphStore(settings.graph_db_path)
    yield rel, vec, graph
    graph.close()
    vec.close()
    rel.close()


@pytest.fixture
def make_service(settings, stores):
    def factory(llm=None, embedder=None, cache=None) -> MemoryService:
        rel, vec, graph = stores
        return MemoryService(settings, rel, vec, graph, cache,
                             llm or FakeLLM(), embedder or FakeEmbedder())
    return factory
```

`tests/test_pipelines.py`:

```python
import pytest

from weave.core.models import content_hash, entity_id_for
from tests.fakes import FakeLLM

LIGHT = {"entities": [{"name": "用户", "entity_type": "Person"}, {"name": "浅烘焙咖啡"}],
         "relationships": [{"source": "用户", "target": "浅烘焙咖啡",
                            "relationship_type": "LIKES"}]}
DARK = {"entities": [{"name": "用户", "entity_type": "Person"}, {"name": "深烘焙咖啡"}],
        "relationships": [{"source": "用户", "target": "深烘焙咖啡",
                           "relationship_type": "LIKES"}]}


async def test_remember_permanent_ingests_graph_and_vector(make_service, stores):
    svc = make_service(llm=FakeLLM([LIGHT]))
    result = await svc.remember("用户喜欢浅烘焙咖啡")
    assert result["mode"] == "permanent"
    assert result["entities"] == 2 and result["relationships"] == 1 and result["superseded"] == 0

    rel, vec, graph = stores
    assert graph.get_entity_by_name("用户", "default") is not None
    facts = graph.neighbors([entity_id_for("default", "用户")])
    assert facts[0]["target_name"] == "浅烘焙咖啡"
    qv = (await svc.embedder.embed(["用户喜欢浅烘焙咖啡"]))[0]
    assert vec.search_chunks(qv, 1)[0]["text"] == "用户喜欢浅烘焙咖啡"
    assert rel.get_data(result["data_id"])["status"] == "completed"


async def test_remember_supersedes_changed_fact(make_service, stores):
    svc = make_service(llm=FakeLLM([LIGHT, DARK]))
    await svc.remember("用户喜欢浅烘焙咖啡")
    result = await svc.remember("用户其实更喜欢深烘焙咖啡")
    assert result["superseded"] == 1

    rel, _, graph = stores
    uid = entity_id_for("default", "用户")
    latest = graph.get_latest_relationship(uid, "LIKES")
    assert latest["version"] == 2
    assert latest["target_id"] == entity_id_for("default", "深烘焙咖啡")
    sql_latest = rel.get_latest_edge(uid, "LIKES")
    assert sql_latest["version"] == 2 and sql_latest["is_latest"] is True


async def test_remember_deduplicates_same_text(make_service):
    llm = FakeLLM([LIGHT])
    svc = make_service(llm=llm)
    await svc.remember("用户喜欢浅烘焙咖啡")
    result = await svc.remember("用户喜欢浅烘焙咖啡")
    assert result["deduplicated"] is True
    assert len(llm.calls) == 1  # 未重复抽取


async def test_remember_llm_failure_writes_no_partial_graph(make_service, stores):
    class FailingLLM:
        async def complete_json(self, system, user):
            raise ConnectionError("LLM down")

    svc = make_service(llm=FailingLLM())
    with pytest.raises(ConnectionError):
        await svc.remember("一些文本")
    rel, _, graph = stores
    assert graph.count_entities() == 0
    assert rel.get_data_by_hash(content_hash("一些文本"))["status"] == "failed"


async def test_list_datasets(make_service):
    svc = make_service(llm=FakeLLM([LIGHT]))
    await svc.remember("用户喜欢浅烘焙咖啡")
    stats = await svc.list_datasets()
    assert stats[0]["name"] == "default"
    assert stats[0]["data_count"] == 1 and stats[0]["edge_count"] == 1
    assert stats[0]["entity_count"] == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_pipelines.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.core.service'`

- [ ] **Step 3: 实现**

`weave/infra/relational.py` 的 `RelationalStore` 中追加方法（放在 `link_dataset_data` 之后）:

```python
    def is_data_linked(self, dataset_id: str, data_id: str) -> bool:
        with Session(self.engine) as s:
            return s.get(DatasetDataRow, {"dataset_id": dataset_id, "data_id": data_id}) is not None
```

`weave/core/service.py`:

```python
"""MemoryService: core 门面, 持有全部存储/客户端, 委托 pipelines/retrieval/cleanup."""

from typing import Any

from weave.config import Settings
from weave.infra.executor import run_db


class MemoryService:
    def __init__(self, settings: Settings, relational, vector, graph, cache, llm, embedder):
        self.settings = settings
        self.relational = relational
        self.vector = vector
        self.graph = graph
        self.cache = cache
        self.llm = llm
        self.embedder = embedder

    async def db(self, fn, *args, **kwargs) -> Any:
        """所有嵌入式 DB 调用的统一入口: 单线程执行器 + 超时 + 瞬时重试."""
        return await run_db(fn, *args, timeout=self.settings.db_call_timeout,
                            max_retries=self.settings.db_call_max_retries, **kwargs)

    async def remember(self, content: str, dataset: str = "default",
                       session_id: str | None = None) -> dict:
        from weave.core.pipelines import ingest_text

        if session_id is not None:
            raise NotImplementedError("session 分支在 Task 11 实现")
        result = await ingest_text(self, content, dataset, "remember", "remember",
                                   data_name=f"remember:{content[:40]}")
        return {"mode": "permanent", **result}

    async def list_datasets(self) -> list[dict]:
        stats = await self.db(self.relational.dataset_stats)
        for row in stats:
            row["entity_count"] = await self.db(self.graph.count_entities, row["name"])
        return stats
```

`weave/core/pipelines.py`:

```python
"""写入管线: ingest_text (remember/cognify/improve 共用) — spec §5.1 三阶段写."""

from weave.core.chunking import split_text
from weave.core.extraction import extract_graph
from weave.core.models import content_hash, edge_id_for, entity_id_for, new_id, norm_name, utcnow


async def ingest_text(svc, text: str, dataset: str, source_pipeline: str,
                      source_task: str, data_name: str, data_id: str | None = None) -> dict:
    ds = await svc.db(svc.relational.get_or_create_dataset, dataset)

    if data_id is None:
        # remember/improve 路径: 同 dataset 内容哈希去重
        ch = content_hash(text)
        existing = await svc.db(svc.relational.get_data_by_hash, ch)
        if existing and await svc.db(svc.relational.is_data_linked, ds["id"], existing["id"]):
            return {"data_id": existing["id"], "dataset": dataset, "deduplicated": True,
                    "entities": 0, "relationships": 0, "superseded": 0, "chunks": 0}
        data_id = new_id()
        await svc.db(svc.relational.create_data, data_id, data_name, text, ch, source_pipeline)
        await svc.db(svc.relational.link_dataset_data, ds["id"], data_id)

    try:
        chunk_texts = split_text(text, svc.settings.chunk_size, svc.settings.chunk_overlap)
        now = utcnow()

        # 阶段1: 全部 LLM 抽取 (失败则不写任何图/向量)
        per_chunk = []
        for position, chunk_text in enumerate(chunk_texts):
            extracted = await extract_graph(svc.llm, chunk_text)
            per_chunk.append((position, chunk_text, extracted))

        # 阶段2: 内存归并实体 + 计算版本更替 (只读 DB)
        all_entities: dict[str, dict] = {}
        new_entity_ids: list[str] = []
        chunk_rows: list[dict] = []
        edge_plan: list[dict] = []
        supersede_plan: list[dict] = []

        async def ensure_entity(name: str, entity_type: str, description: str) -> str:
            eid = entity_id_for(dataset, name)
            if eid not in all_entities:
                all_entities[eid] = dict(
                    id=eid, name=name, norm_name=norm_name(name), entity_type=entity_type,
                    description=description, dataset=dataset, version=1,
                    is_latest=True, created_at=now)
                if not await svc.db(svc.graph.get_entity_by_id, eid):
                    new_entity_ids.append(eid)
            return eid

        for position, chunk_text, extracted in per_chunk:
            local: dict[str, str] = {}
            chunk_entity_ids: set[str] = set()
            for ee in extracted.entities:
                eid = await ensure_entity(ee.name, ee.entity_type, ee.description)
                local[norm_name(ee.name)] = eid
                chunk_entity_ids.add(eid)
            for rel in extracted.relationships:
                src_id = local.get(norm_name(rel.source)) or await ensure_entity(rel.source, "Thing", "")
                dst_id = local.get(norm_name(rel.target)) or await ensure_entity(rel.target, "Thing", "")
                if src_id == dst_id:
                    continue
                chunk_entity_ids.update({src_id, dst_id})
                prev = await svc.db(svc.graph.get_latest_relationship, src_id, rel.relationship_type)
                version = 1
                if prev is not None:
                    if prev["target_id"] == dst_id:
                        continue  # 重复事实, 跳过
                    supersede_plan.append(prev)
                    version = prev["version"] + 1
                edge_plan.append(dict(
                    edge_id=edge_id_for(src_id, rel.relationship_type, dst_id),
                    source_id=src_id, target_id=dst_id,
                    relationship_type=rel.relationship_type, version=version,
                    is_latest=True, source_pipeline=source_pipeline,
                    dataset=dataset, created_at=now))
            chunk_rows.append(dict(chunk_id=new_id(), text=chunk_text, position=position,
                                   entity_ids=sorted(chunk_entity_ids)))

        # 阶段3: 统一写三库 (Kuzu -> LanceDB -> SQLite)
        for prev in supersede_plan:
            await svc.db(svc.graph.set_relationship_not_latest, prev["edge_id"])
            await svc.db(svc.relational.set_edge_not_latest, prev["edge_id"], now)
        for entity in all_entities.values():
            await svc.db(svc.graph.upsert_entity, entity)
        for edge in edge_plan:
            await svc.db(svc.graph.upsert_relationship, edge)
            await svc.db(svc.relational.upsert_edge,
                         {**edge, "source_task": source_task, "updated_at": now})
        for row in chunk_rows:
            await svc.db(svc.graph.create_chunk, row["chunk_id"], data_id, dataset, now)
            if row["entity_ids"]:
                await svc.db(svc.graph.link_mentions, row["chunk_id"], row["entity_ids"])

        chunk_vecs = await svc.embedder.embed([r["text"] for r in chunk_rows])
        await svc.db(svc.vector.add_chunks, [
            dict(chunk_id=r["chunk_id"], vector=vec, text=r["text"], data_id=data_id,
                 dataset=dataset, created_at=now)
            for r, vec in zip(chunk_rows, chunk_vecs)
        ])
        new_entities = [all_entities[eid] for eid in new_entity_ids]
        if new_entities:
            entity_vecs = await svc.embedder.embed(
                [f"{e['name']} {e['description']}".strip() for e in new_entities])
            await svc.db(svc.vector.add_entities, [
                dict(entity_id=e["id"], vector=vec, name=e["name"],
                     entity_type=e["entity_type"], description=e["description"],
                     dataset=dataset, is_latest=True)
                for e, vec in zip(new_entities, entity_vecs)
            ])

        await svc.db(svc.relational.set_data_status, data_id, "completed")
    except Exception:
        await svc.db(svc.relational.set_data_status, data_id, "failed")
        raise

    return {"data_id": data_id, "dataset": dataset, "deduplicated": False,
            "entities": len(new_entity_ids), "relationships": len(edge_plan),
            "superseded": len(supersede_plan), "chunks": len(chunk_texts)}
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_pipelines.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/core/service.py weave/core/pipelines.py weave/infra/relational.py tests/conftest.py tests/test_pipelines.py; git commit -m "feat: remember 永久管线 (三阶段写/版本更替/去重)"
```

---

### Task 10: Redis 缓存（队列 + 会话）与会话助手

**Files:**
- Create: `weave/infra/cache.py`
- Create: `weave/core/session.py`
- Modify: `tests/conftest.py`（新增 fake_cache fixture）
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: 无（fakeredis 仅测试用）
- Produces:
  - 常量 `QUEUE_IMPROVE = "weave:queue:improve"`、`QUEUE_COGNIFY = "weave:queue:cognify"`
  - `Cache(host, port, db, client=None)`，`async close()`
  - 会话：`async session_append(session_id, content, max_items, ttl_seconds)`、`async session_get(session_id) -> list[dict]`（item keys: id/content/ts/synced）、`async session_unsynced(session_id) -> list[dict]`、`async session_mark_synced(session_id, ttl_seconds)`、`async session_clear(session_id)`
  - 队列：`async enqueue(queue, payload: dict)`、`async dequeue_priority(queues: list[str], timeout: int) -> tuple[str, dict] | None`（BRPOP，key 序即优先级）、`async queue_len(queue) -> int`
  - `weave.core.session`: `append_session(cache, settings, session_id, content)`、`get_session(cache, session_id)`、`get_unsynced(cache, session_id)`、`mark_synced(cache, settings, session_id)`

- [ ] **Step 1: 写失败测试**

`tests/conftest.py` 追加:

```python
import fakeredis.aioredis

from weave.infra.cache import Cache


@pytest.fixture
async def fake_cache(settings):
    c = Cache(settings.redis_host, settings.redis_port, settings.redis_db,
              client=fakeredis.aioredis.FakeRedis(decode_responses=True))
    yield c
    await c.close()
```

`tests/test_cache.py`:

```python
from weave.core.session import append_session, get_session, get_unsynced, mark_synced
from weave.infra.cache import QUEUE_COGNIFY, QUEUE_IMPROVE


async def test_session_append_trim_and_ttl(settings, fake_cache):
    for i in range(5):
        await append_session(fake_cache, settings, "s1", f"消息{i}")
    items = await get_session(fake_cache, "s1")
    assert [i["content"] for i in items] == [f"消息{i}" for i in range(5)]
    assert all(i["synced"] is False for i in items)

    settings.session_max_items = 3
    await append_session(fake_cache, settings, "s1", "消息5")
    items = await get_session(fake_cache, "s1")
    assert [i["content"] for i in items] == ["消息3", "消息4", "消息5"]  # 裁剪最旧
    ttl = await fake_cache._r.ttl("weave:session:s1")
    assert ttl > 0


async def test_session_unsynced_and_mark_synced(settings, fake_cache):
    await append_session(fake_cache, settings, "s2", "甲")
    await append_session(fake_cache, settings, "s2", "乙")
    assert len(await get_unsynced(fake_cache, "s2")) == 2
    await mark_synced(fake_cache, settings, "s2")
    assert await get_unsynced(fake_cache, "s2") == []
    assert len(await get_session(fake_cache, "s2")) == 2  # 原文保留供 recall


async def test_queue_priority_improve_first(fake_cache):
    await fake_cache.enqueue(QUEUE_COGNIFY, {"kind": "doc"})
    await fake_cache.enqueue(QUEUE_IMPROVE, {"kind": "session"})
    first = await fake_cache.dequeue_priority([QUEUE_IMPROVE, QUEUE_COGNIFY], timeout=1)
    assert first[0] == QUEUE_IMPROVE and first[1] == {"kind": "session"}
    second = await fake_cache.dequeue_priority([QUEUE_IMPROVE, QUEUE_COGNIFY], timeout=1)
    assert second[0] == QUEUE_COGNIFY
    assert await fake_cache.dequeue_priority([QUEUE_IMPROVE, QUEUE_COGNIFY], timeout=1) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.infra.cache'`

- [ ] **Step 3: 实现**

`weave/infra/cache.py`:

```python
"""Redis 封装: 会话记忆 list + 任务队列 (spec §4.3).

dequeue_priority 用单次 BRPOP 多 key: Redis 按 key 顺序扫描, 实现 improve 优先.
"""

import json
import time
import uuid

import redis.asyncio as aioredis

QUEUE_IMPROVE = "weave:queue:improve"
QUEUE_COGNIFY = "weave:queue:cognify"
_SESSION_PREFIX = "weave:session:"


class Cache:
    def __init__(self, host: str, port: int, db: int, client=None):
        self._r = client or aioredis.Redis(host=host, port=port, db=db, decode_responses=True)

    async def close(self) -> None:
        await self._r.aclose()

    # ---------- 会话记忆 ----------
    async def session_append(self, session_id: str, content: str,
                             max_items: int, ttl_seconds: int) -> None:
        key = _SESSION_PREFIX + session_id
        item = json.dumps({"id": uuid.uuid4().hex, "content": content,
                           "ts": time.time(), "synced": False})
        await self._r.rpush(key, item)
        await self._r.ltrim(key, -max_items, -1)
        await self._r.expire(key, ttl_seconds)

    async def session_get(self, session_id: str) -> list[dict]:
        rows = await self._r.lrange(_SESSION_PREFIX + session_id, 0, -1)
        return [json.loads(r) for r in rows]

    async def session_unsynced(self, session_id: str) -> list[dict]:
        return [i for i in await self.session_get(session_id) if not i["synced"]]

    async def session_mark_synced(self, session_id: str, ttl_seconds: int) -> None:
        key = _SESSION_PREFIX + session_id
        items = await self.session_get(session_id)
        if not items:
            return
        for i in items:
            i["synced"] = True
        await self._r.delete(key)
        await self._r.rpush(key, *[json.dumps(i) for i in items])
        await self._r.expire(key, ttl_seconds)

    async def session_clear(self, session_id: str) -> None:
        await self._r.delete(_SESSION_PREFIX + session_id)

    # ---------- 任务队列 ----------
    async def enqueue(self, queue: str, payload: dict) -> None:
        await self._r.lpush(queue, json.dumps(payload))

    async def dequeue_priority(self, queues: list[str], timeout: int) -> tuple[str, dict] | None:
        result = await self._r.brpop(queues, timeout=timeout)
        if result is None:
            return None
        queue, raw = result
        return queue, json.loads(raw)

    async def queue_len(self, queue: str) -> int:
        return await self._r.llen(queue)
```

`weave/core/session.py`:

```python
"""会话记忆助手: 隔离于图之外的 Redis 读写 (spec §6 防污染机制③)."""


def _ttl_seconds(settings) -> int:
    return settings.session_ttl_days * 86400


async def append_session(cache, settings, session_id: str, content: str) -> None:
    await cache.session_append(session_id, content, settings.session_max_items, _ttl_seconds(settings))


async def get_session(cache, session_id: str) -> list[dict]:
    return await cache.session_get(session_id)


async def get_unsynced(cache, session_id: str) -> list[dict]:
    return await cache.session_unsynced(session_id)


async def mark_synced(cache, settings, session_id: str) -> None:
    await cache.session_mark_synced(session_id, _ttl_seconds(settings))
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_cache.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/infra/cache.py weave/core/session.py tests/conftest.py tests/test_cache.py; git commit -m "feat: Redis 缓存 (会话记忆+优先级队列)"
```

---

### Task 11: improve 管线与 remember 会话分支

**Files:**
- Modify: `weave/core/pipelines.py`（追加 `improve_session`）
- Modify: `weave/core/service.py`（remember 的 session 分支 + `run_improve`）
- Test: `tests/test_improve.py`

**Interfaces:**
- Consumes: Task 5 `filter_session_facts`、Task 9 `ingest_text`、Task 10 session/queue
- Produces:
  - `pipelines.improve_session(svc, session_id: str, dataset: str = "default", task_id: str = "") -> dict`（keys: kept/discarded/ingested）
  - `service.run_improve(session_id: str, dataset: str = "default", task_id: str = "") -> dict`
  - `remember(content, dataset, session_id)` 完整双模式：session 分支返回 `{"mode": "session", "session_id", "queued": True, "task_id}`

**防污染约束（spec §6）:** 会话原文只进 Redis；improve 只有 LLM 过滤出的陈述句进 `ingest_text`；入图事实 `source_pipeline="session_improve"`；全部丢弃时不写图。

- [ ] **Step 1: 写失败测试**

`tests/test_improve.py`:

```python
from weave.core.models import entity_id_for
from weave.infra.cache import QUEUE_IMPROVE
from tests.fakes import FakeLLM

IMPROVE_OUT = {"facts": [
    {"keep": True, "statement": "用户偏好简洁回答", "reason": "稳定偏好"},
    {"keep": False, "statement": "今天在调试代码", "reason": "一次性事件"},
]}
EXTRACT_OUT = {"entities": [{"name": "用户", "entity_type": "Person"},
                            {"name": "简洁回答", "entity_type": "Preference"}],
               "relationships": [{"source": "用户", "target": "简洁回答",
                                  "relationship_type": "PREFERS"}]}


async def test_remember_session_writes_cache_and_enqueues(make_service, stores, fake_cache):
    svc = make_service(cache=fake_cache)
    result = await svc.remember("你回答简洁一点", session_id="chat-1")
    assert result["mode"] == "session" and result["queued"] is True and result["task_id"]
    assert len(await fake_cache.session_get("chat-1")) == 1
    assert await fake_cache.queue_len(QUEUE_IMPROVE) == 1
    _, _, graph = stores
    assert graph.count_entities() == 0  # 会话原文不进图


async def test_improve_filters_then_ingests(make_service, stores, fake_cache):
    llm = FakeLLM([IMPROVE_OUT, EXTRACT_OUT])
    svc = make_service(llm=llm, cache=fake_cache)
    await svc.remember("你回答简洁一点", session_id="chat-1")
    await svc.remember("今天在调试代码", session_id="chat-1")

    result = await svc.run_improve("chat-1")
    assert result["kept"] == 1 and result["discarded"] == 1
    assert result["ingested"]["relationships"] == 1

    rel, _, graph = stores
    uid = entity_id_for("default", "用户")
    latest = rel.get_latest_edge(uid, "PREFERS")
    assert latest["source_pipeline"] == "session_improve"  # 血缘隔离标记
    assert graph.neighbors([uid])[0]["target_name"] == "简洁回答"
    assert await fake_cache.session_unsynced("chat-1") == []  # 已标记 synced


async def test_improve_discards_everything_writes_nothing(make_service, stores, fake_cache):
    llm = FakeLLM([{"facts": [{"keep": False, "statement": "闲聊", "reason": "丢弃"}]}])
    svc = make_service(llm=llm, cache=fake_cache)
    await svc.remember("随便聊聊", session_id="chat-2")

    result = await svc.run_improve("chat-2")
    assert result["kept"] == 0 and result["ingested"] is None
    _, _, graph = stores
    assert graph.count_entities() == 0  # 全部丢弃, 不写图
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_improve.py -v`
Expected: FAIL（remember session 分支 NotImplementedError / improve_session 不存在）

- [ ] **Step 3: 实现**

`weave/core/pipelines.py` 追加:

```python
async def improve_session(svc, session_id: str, dataset: str = "default",
                          task_id: str = "") -> dict:
    """会话记忆沉淀: LLM 认知过滤 + 重要性门禁, 通过后才走标准入图 (spec §5.3/§6)."""
    from weave.core.extraction import filter_session_facts
    from weave.core.session import get_unsynced, mark_synced

    if task_id:
        await svc.db(svc.relational.update_pipeline_run, task_id, "running")
    try:
        items = await get_unsynced(svc.cache, session_id)
        if not items:
            if task_id:
                await svc.db(svc.relational.update_pipeline_run, task_id, "completed")
            return {"kept": 0, "discarded": 0, "ingested": None}
        statements = await filter_session_facts(svc.llm, [i["content"] for i in items])
        result = None
        if statements:
            result = await ingest_text(svc, "\n".join(statements), dataset,
                                       "session_improve", "improve",
                                       data_name=f"session:{session_id}")
        await mark_synced(svc.cache, svc.settings, session_id)
        if task_id:
            await svc.db(svc.relational.update_pipeline_run, task_id, "completed")
        return {"kept": len(statements), "discarded": len(items) - len(statements),
                "ingested": result}
    except Exception as exc:
        if task_id:
            await svc.db(svc.relational.update_pipeline_run, task_id, "failed", str(exc))
        raise
```

`weave/core/service.py` 修改 remember 并追加 run_improve（替换 NotImplementedError 分支）:

```python
    async def remember(self, content: str, dataset: str = "default",
                       session_id: str | None = None) -> dict:
        from weave.core.pipelines import ingest_text
        from weave.core.session import append_session
        from weave.infra.cache import QUEUE_IMPROVE
        from weave.core.models import new_id

        if session_id is not None:
            await append_session(self.cache, self.settings, session_id, content)
            task_id = new_id()
            await self.db(self.relational.create_pipeline_run, task_id, "improve")
            await self.cache.enqueue(QUEUE_IMPROVE,
                                     {"task_id": task_id, "session_id": session_id,
                                      "dataset": dataset})
            return {"mode": "session", "session_id": session_id,
                    "queued": True, "task_id": task_id}
        result = await ingest_text(self, content, dataset, "remember", "remember",
                                   data_name=f"remember:{content[:40]}")
        return {"mode": "permanent", **result}

    async def run_improve(self, session_id: str, dataset: str = "default",
                          task_id: str = "") -> dict:
        from weave.core.pipelines import improve_session

        return await improve_session(self, session_id, dataset, task_id)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_improve.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/core/pipelines.py weave/core/service.py tests/test_improve.py; git commit -m "feat: improve 管线与 remember 会话分支 (防污染过滤)"
```

---

### Task 12: cognify 管线（base64 文档异步摄取）与 worker

**Files:**
- Modify: `weave/core/service.py`（追加 `cognify_submit` / `run_cognify_task` / `task_status`）
- Create: `weave/worker.py`
- Test: `tests/test_cognify.py`

**Interfaces:**
- Consumes: Task 9 `ingest_text`（传 data_id 分支）、Task 10/11 队列与 `run_improve`
- Produces:
  - `async cognify_submit(file_name: str, content_base64: str, dataset: str = "default") -> dict`（keys: task_id/data_id/status[/deduplicated]；非法 base64 或非 txt/md 扩展名抛 `ValueError`）
  - `async run_cognify_task(task_id: str, data_id: str, dataset: str) -> None`（失败不抛出，pipeline_run 置 failed）
  - `async task_status(task_id: str) -> dict`（不存在返回 `{"task_id", "status": "not_found"}`）
  - `worker_loop(service, cache, poll_timeout: int, stop: asyncio.Event) -> None`

**队列约束（spec §8）:** 单循环单次 `BRPOP [QUEUE_IMPROVE, QUEUE_COGNIFY]`；payload 键名与 `run_improve`/`run_cognify_task` 参数名一致，直接 `**payload` 分发。

- [ ] **Step 1: 写失败测试**

`tests/test_cognify.py`:

```python
import asyncio
import base64

import pytest

from weave.infra.cache import QUEUE_COGNIFY, QUEUE_IMPROVE
from weave.worker import worker_loop
from tests.fakes import FakeLLM

DOC = "Weave 是知识图谱记忆平台。它使用 Kuzu 作为图数据库。"
DOC_B64 = base64.b64encode(DOC.encode()).decode()
DOC_OUT = {"entities": [{"name": "Weave", "entity_type": "Project"},
                        {"name": "Kuzu", "entity_type": "Concept"}],
           "relationships": [{"source": "Weave", "target": "Kuzu",
                              "relationship_type": "USES"}]}


async def test_cognify_submit_and_run_task(make_service, stores, fake_cache):
    svc = make_service(llm=FakeLLM([DOC_OUT]), cache=fake_cache)
    submit = await svc.cognify_submit("notes.txt", DOC_B64)
    assert submit["status"] == "pending" and submit["task_id"]
    assert await fake_cache.queue_len(QUEUE_COGNIFY) == 1

    _, payload = await fake_cache.dequeue_priority([QUEUE_COGNIFY], timeout=1)
    await svc.run_cognify_task(**payload)

    status = await svc.task_status(submit["task_id"])
    assert status["status"] == "completed" and status["pipeline_name"] == "cognify"
    _, _, graph = stores
    assert graph.get_entity_by_name("weave", "default") is not None


async def test_cognify_dedup_returns_completed(make_service, fake_cache):
    svc = make_service(llm=FakeLLM([DOC_OUT]), cache=fake_cache)
    first = await svc.cognify_submit("notes.txt", DOC_B64)
    _, payload = await fake_cache.dequeue_priority([QUEUE_COGNIFY], timeout=1)
    await svc.run_cognify_task(**payload)
    second = await svc.cognify_submit("notes.txt", DOC_B64)
    assert second["status"] == "completed" and second["deduplicated"] is True


async def test_cognify_rejects_bad_input(make_service, fake_cache):
    svc = make_service(cache=fake_cache)
    with pytest.raises(ValueError, match="文件类型"):
        await svc.cognify_submit("doc.pdf", DOC_B64)
    with pytest.raises(ValueError, match="base64"):
        await svc.cognify_submit("doc.txt", "!!!not-base64!!!")


async def test_run_cognify_task_failure_marks_failed(make_service, fake_cache):
    class FailingLLM:
        async def complete_json(self, system, user):
            raise ConnectionError("LLM down")

    svc = make_service(llm=FailingLLM(), cache=fake_cache)
    submit = await svc.cognify_submit("notes.txt", DOC_B64)
    _, payload = await fake_cache.dequeue_priority([QUEUE_COGNIFY], timeout=1)
    await svc.run_cognify_task(**payload)  # 不抛出
    status = await svc.task_status(submit["task_id"])
    assert status["status"] == "failed" and "LLM down" in status["error"]
    assert (await svc.task_status("missing"))["status"] == "not_found"


async def test_worker_consumes_queues_with_priority(make_service, fake_cache):
    llm = FakeLLM([{"facts": []}, DOC_OUT])
    svc = make_service(llm=llm, cache=fake_cache)
    order: list[str] = []
    orig_improve, orig_cognify = svc.run_improve, svc.run_cognify_task

    async def spy_improve(**kw):
        order.append("improve")
        await orig_improve(**kw)

    async def spy_cognify(**kw):
        order.append("cognify")
        await orig_cognify(**kw)

    svc.run_improve = spy_improve
    svc.run_cognify_task = spy_cognify

    # cognify 先入队, improve 后入队 —— 但 improve 必须先被消费
    submit = await svc.cognify_submit("notes.txt", DOC_B64)
    await svc.remember("随便聊聊", session_id="s9")

    stop = asyncio.Event()
    worker = asyncio.create_task(worker_loop(svc, fake_cache, 1, stop))
    for _ in range(100):
        if len(order) == 2:
            break
        await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(worker, timeout=5)

    assert order == ["improve", "cognify"]
    assert (await svc.task_status(submit["task_id"]))["status"] == "completed"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_cognify.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.worker'`（且 service 无 cognify_submit）

- [ ] **Step 3: 实现**

`weave/core/service.py` 追加方法:

```python
    async def cognify_submit(self, file_name: str, content_base64: str,
                             dataset: str = "default") -> dict:
        import base64
        from pathlib import Path

        from weave.core.models import content_hash, new_id
        from weave.infra.cache import QUEUE_COGNIFY

        suffix = Path(file_name).suffix.lower()
        if suffix not in {".txt", ".md", ".markdown"}:
            raise ValueError(f"暂不支持的文件类型: {suffix} (v1 支持 txt/md)")
        try:
            raw = base64.b64decode(content_base64).decode("utf-8")
        except Exception as exc:
            raise ValueError(f"无法解码 base64 文件内容: {exc}") from exc

        ds = await self.db(self.relational.get_or_create_dataset, dataset)
        ch = content_hash(raw)
        existing = await self.db(self.relational.get_data_by_hash, ch)
        if existing and await self.db(self.relational.is_data_linked, ds["id"], existing["id"]):
            task_id = new_id()
            await self.db(self.relational.create_pipeline_run, task_id, "cognify", existing["id"])
            await self.db(self.relational.update_pipeline_run, task_id, "completed")
            return {"task_id": task_id, "data_id": existing["id"],
                    "status": "completed", "deduplicated": True}

        data_id = new_id()
        await self.db(self.relational.create_data, data_id, file_name, raw, ch, "cognify")
        await self.db(self.relational.link_dataset_data, ds["id"], data_id)
        task_id = new_id()
        await self.db(self.relational.create_pipeline_run, task_id, "cognify", data_id)
        await self.cache.enqueue(QUEUE_COGNIFY,
                                 {"task_id": task_id, "data_id": data_id, "dataset": dataset})
        return {"task_id": task_id, "data_id": data_id, "status": "pending"}

    async def run_cognify_task(self, task_id: str, data_id: str, dataset: str) -> None:
        from weave.core.pipelines import ingest_text

        await self.db(self.relational.update_pipeline_run, task_id, "running")
        try:
            record = await self.db(self.relational.get_data, data_id)
            if record is None:
                raise ValueError(f"data 记录不存在: {data_id}")
            await ingest_text(self, record["raw_text"], dataset, "cognify", "cognify",
                              record["name"], data_id=data_id)
            await self.db(self.relational.update_pipeline_run, task_id, "completed")
        except Exception as exc:
            await self.db(self.relational.update_pipeline_run, task_id, "failed", str(exc))

    async def task_status(self, task_id: str) -> dict:
        run = await self.db(self.relational.get_pipeline_run, task_id)
        if run is None:
            return {"task_id": task_id, "status": "not_found"}
        return run
```

`weave/worker.py`:

```python
"""进程内 worker: 单循环单 BRPOP, key 顺序 [improve, cognify] 即优先级 (spec §8).

禁止拆成两个队列各自独立 BRPOP (并发消费者会使优先级失效).
"""

import asyncio
import logging

from weave.infra.cache import QUEUE_COGNIFY, QUEUE_IMPROVE

logger = logging.getLogger(__name__)


async def worker_loop(service, cache, poll_timeout: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        item = await cache.dequeue_priority([QUEUE_IMPROVE, QUEUE_COGNIFY], poll_timeout)
        if item is None:
            continue
        queue, payload = item
        try:
            if queue == QUEUE_IMPROVE:
                await service.run_improve(**payload)
            else:
                await service.run_cognify_task(**payload)
        except Exception:
            logger.exception("worker 任务失败: %s %s", queue, payload)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_cognify.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/core/service.py weave/worker.py tests/test_cognify.py; git commit -m "feat: cognify 异步文档摄取与进程内 worker"
```

---

### Task 13: 混合检索（hybrid_recall）

**Files:**
- Create: `weave/core/retrieval.py`
- Modify: `weave/core/service.py`（追加 `recall`）
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: Task 7 `neighbors`/`mentioned_chunk_ids`、Task 8 `search_*`/`get_chunks`、Task 10 `session_get`
- Produces:
  - `retrieval.hybrid_recall(svc, query, dataset=None, top_k=5, session_id=None) -> dict`
  - `service.recall(query, dataset=None, top_k=5, session_id=None) -> dict`（RecallResult 结构：facts/chunks/session_items）

**路由逻辑（spec §5.4）:** query 向量化 → `text_chunks` 与 `entities` 各取 top_k 入口 → Kuzu 1 跳扩展（仅 is_latest）→ 图关联 chunk 回查 LanceDB 补充 → 会话缓存叠加（`source="session"`）。fact 字典 keys: source/relationship_type/target/version/dataset/source_pipeline/origin；chunk 字典 keys: chunk_id/text/dataset/score/source（"vector"|"graph"）。

- [ ] **Step 1: 写失败测试**

`tests/test_retrieval.py`:

```python
from tests.fakes import FakeLLM
from tests.test_pipelines import DARK, LIGHT


async def test_recall_returns_facts_and_chunks(make_service, fake_cache):
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")

    result = await svc.recall("用户喜欢什么")
    assert result["facts"][0]["source"] == "用户"
    assert result["facts"][0]["relationship_type"] == "LIKES"
    assert result["facts"][0]["target"] == "浅烘焙咖啡"
    assert result["facts"][0]["origin"] == "graph"
    assert result["chunks"][0]["text"] == "用户喜欢浅烘焙咖啡"
    assert result["session_items"] == []


async def test_recall_excludes_superseded(make_service, fake_cache):
    svc = make_service(llm=FakeLLM([LIGHT, DARK]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")
    await svc.remember("用户其实更喜欢深烘焙咖啡")

    result = await svc.recall("用户喜欢什么烘焙")
    targets = [f["target"] for f in result["facts"]]
    assert targets == ["深烘焙咖啡"]  # 被取代的旧版本不出现


async def test_recall_dataset_isolation(make_service, fake_cache):
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡", dataset="personal")

    assert (await svc.recall("咖啡", dataset="personal"))["facts"] != []
    assert (await svc.recall("咖啡", dataset="other"))["facts"] == []


async def test_recall_session_overlay(make_service, fake_cache):
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")
    await svc.remember("我们明天讨论架构", session_id="chat-7")

    result = await svc.recall("架构", session_id="chat-7")
    assert result["session_items"][0]["content"] == "我们明天讨论架构"
    assert result["session_items"][0]["source"] == "session"
    # 不带 session_id 则无会话内容
    assert (await svc.recall("架构"))["session_items"] == []
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL（service 无 recall / 无 retrieval 模块）

- [ ] **Step 3: 实现**

`weave/core/retrieval.py`:

```python
"""混合检索自动路由 (spec §5.4): 向量入口 -> 图 1 跳扩展 -> 汇总, 会话缓存叠加."""

from weave.core.models import RecallResult
from weave.core.session import get_session


async def hybrid_recall(svc, query: str, dataset: str | None = None,
                        top_k: int = 5, session_id: str | None = None) -> dict:
    query_vector = (await svc.embedder.embed([query]))[0]
    chunk_hits = await svc.db(svc.vector.search_chunks, query_vector, top_k, dataset)
    entity_hits = await svc.db(svc.vector.search_entities, query_vector, top_k, dataset)

    entry_ids = [h["entity_id"] for h in entity_hits]
    facts = await svc.db(svc.graph.neighbors, entry_ids, dataset)

    # 图扩展: 入口实体 MENTIONS 的 chunk 回查原文, 补充进结果
    mentioned_ids = await svc.db(svc.graph.mentioned_chunk_ids, entry_ids, dataset)
    mentioned = await svc.db(svc.vector.get_chunks, mentioned_ids)

    seen = {h["chunk_id"] for h in chunk_hits}
    chunks = [
        dict(chunk_id=h["chunk_id"], text=h["text"], dataset=h["dataset"],
             score=h.get("_distance"), source="vector")
        for h in chunk_hits
    ]
    for row in mentioned:
        if row["chunk_id"] not in seen:
            chunks.append(dict(chunk_id=row["chunk_id"], text=row["text"],
                               dataset=row["dataset"], score=None, source="graph"))

    fact_dicts = [
        dict(source=f["source_name"], relationship_type=f["relationship_type"],
             target=f["target_name"], version=f["version"], dataset=f["dataset"],
             source_pipeline=f["source_pipeline"], origin="graph")
        for f in facts
    ]

    session_items = []
    if session_id:
        session_items = [
            dict(content=i["content"], ts=i["ts"], synced=i["synced"], source="session")
            for i in await get_session(svc.cache, session_id)
        ]

    return RecallResult(facts=fact_dicts, chunks=chunks,
                        session_items=session_items).model_dump()
```

`weave/core/service.py` 追加:

```python
    async def recall(self, query: str, dataset: str | None = None,
                     top_k: int = 5, session_id: str | None = None) -> dict:
        from weave.core.retrieval import hybrid_recall

        return await hybrid_recall(self, query, dataset, top_k, session_id)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/core/retrieval.py weave/core/service.py tests/test_retrieval.py; git commit -m "feat: 混合检索自动路由 (向量入口+图扩展+会话叠加)"
```

---

### Task 14: 清理与遗忘（forget / forget_by_source）

**Files:**
- Create: `weave/core/cleanup.py`
- Modify: `weave/core/service.py`（追加 `forget` / `forget_by_source`）
- Test: `tests/test_cleanup.py`

**Interfaces:**
- Consumes: Task 3 `delete_dataset_rows`/`delete_edges_by_source`、Task 7 `delete_dataset`/`delete_relationships_by_source`、Task 8 `delete_dataset`
- Produces:
  - `cleanup.forget_dataset(svc, dataset: str) -> dict`（keys: scope/data/edges）
  - `cleanup.forget_by_source(svc, source_pipeline: str) -> dict`（keys: scope/edges_sql/edges_graph）
  - `service.forget(dataset: str | None = None, confirm: bool = False) -> dict`（dataset=None 且 confirm=False 抛 ValueError）
  - `service.forget_by_source(source_pipeline: str) -> dict`（core 内部方法，spec §6③；不暴露 MCP/REST）

**语义:** forget 级联清三库；forget_by_source 只删关系事实（实体可能被多来源共享，保留，spec §6③）。

- [ ] **Step 1: 写失败测试**

`tests/test_cleanup.py`:

```python
import pytest

from weave.core.models import entity_id_for
from tests.fakes import FakeLLM
from tests.test_improve import EXTRACT_OUT, IMPROVE_OUT
from tests.test_pipelines import LIGHT


async def test_forget_dataset_cascades(make_service, stores, fake_cache):
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")
    result = await svc.forget(dataset="default")
    assert result["scope"] == "default" and result["edges"] == 1

    rel, vec, graph = stores
    assert graph.count_entities() == 0
    assert rel.list_datasets() == []
    qv = (await svc.embedder.embed(["咖啡"]))[0]
    assert vec.search_chunks(qv, 5) == []


async def test_forget_all_requires_confirm(make_service, fake_cache):
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")
    with pytest.raises(ValueError, match="confirm"):
        await svc.forget()
    result = await svc.forget(confirm=True)
    assert result["scope"] == "all"
    assert await svc.list_datasets() == []


async def test_forget_by_source_removes_only_session_facts(make_service, stores, fake_cache):
    llm = FakeLLM([LIGHT, IMPROVE_OUT, EXTRACT_OUT])
    svc = make_service(llm=llm, cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")                    # source_pipeline=remember
    await svc.remember("你回答简洁一点", session_id="chat-1")
    await svc.run_improve("chat-1")                             # source_pipeline=session_improve

    result = await svc.forget_by_source("session_improve")
    assert result["edges_graph"] == 1

    uid = entity_id_for("default", "用户")
    rel, _, graph = stores
    facts = graph.neighbors([uid])
    assert [f["relationship_type"] for f in facts] == ["LIKES"]  # remember 事实保留
    assert graph.get_entity_by_name("简洁回答", "default") is not None  # 实体共享保留
    assert rel.get_latest_edge(uid, "PREFERS") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_cleanup.py -v`
Expected: FAIL（service 无 forget / 无 cleanup 模块）

- [ ] **Step 3: 实现**

`weave/core/cleanup.py`:

```python
"""遗忘管线 (spec §5/§6): forget 级联清三库; forget_by_source 只删关系事实."""


async def forget_dataset(svc, dataset: str) -> dict:
    counts = await svc.db(svc.relational.delete_dataset_rows, dataset)
    await svc.db(svc.graph.delete_dataset, dataset)
    await svc.db(svc.vector.delete_dataset, dataset)
    return {"scope": dataset, **counts}


async def forget_by_source(svc, source_pipeline: str) -> dict:
    """按来源清理关系事实 (如 session_improve); 实体可能被多来源共享, 保留."""
    edges_sql = await svc.db(svc.relational.delete_edges_by_source, source_pipeline)
    edges_graph = await svc.db(svc.graph.delete_relationships_by_source, source_pipeline)
    return {"scope": f"source:{source_pipeline}",
            "edges_sql": edges_sql, "edges_graph": edges_graph}
```

`weave/core/service.py` 追加:

```python
    async def forget(self, dataset: str | None = None, confirm: bool = False) -> dict:
        from weave.core.cleanup import forget_dataset

        if dataset is None:
            if not confirm:
                raise ValueError("清空全部记忆需要 confirm=True")
            names = [d["name"] for d in await self.db(self.relational.list_datasets)]
            return {"scope": "all",
                    "datasets": [await forget_dataset(self, n) for n in names]}
        return await forget_dataset(self, dataset)

    async def forget_by_source(self, source_pipeline: str) -> dict:
        from weave.core.cleanup import forget_by_source as _impl

        return await _impl(self, source_pipeline)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_cleanup.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/core/cleanup.py weave/core/service.py tests/test_cleanup.py; git commit -m "feat: 遗忘管线 (dataset 级联/按来源清理)"
```

---

### Task 15: REST API、Bearer 认证与应用组装

**Files:**
- Create: `weave/api/__init__.py`（空文件）
- Create: `weave/api/auth.py`
- Create: `weave/api/rest.py`
- Create: `weave/api/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: Task 9-14 的 `MemoryService` 全部方法
- Produces:
  - `make_bearer_middleware(api_key: str)`（FastAPI http 中间件工厂；仅豁免 `/v1/health`）
  - `router`（`/v1` 前缀 APIRouter）
  - `build_service(settings) -> MemoryService`（生产装配：真实 Redis/LLM/embedding 客户端）
  - `create_app(settings, service=None) -> FastAPI`（本 Task 版本：lifespan 只启动 worker；MCP 挂载在 Task 16 加入）

**REST 端点（spec §7.2）:** `GET /v1/health`（免认证）、`POST /v1/memories`、`POST /v1/recall`、`POST /v1/documents`(202)、`GET /v1/tasks/{id}`、`GET /v1/datasets`、`DELETE /v1/datasets/{name}`、`DELETE /v1/memories?confirm=true`。`ValueError` → 400 `{error: {code, message}}`。

- [ ] **Step 1: 写失败测试**

`tests/test_api.py`:

```python
import base64

import pytest
from fastapi.testclient import TestClient

from weave.api.app import create_app
from tests.fakes import FakeLLM
from tests.test_pipelines import LIGHT

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def client(settings, make_service, fake_cache):
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    with TestClient(create_app(settings, service=svc)) as c:
        yield c


def test_health_is_open(client):
    assert client.get("/v1/health").status_code == 200


def test_auth_required(client):
    r = client.post("/v1/memories", json={"content": "x"})
    assert r.status_code == 401
    r = client.post("/v1/memories", json={"content": "x"},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_remember_and_recall_via_rest(client):
    r = client.post("/v1/memories", json={"content": "用户喜欢浅烘焙咖啡"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["mode"] == "permanent"

    r = client.post("/v1/recall", json={"query": "用户喜欢什么"}, headers=AUTH)
    assert r.json()["facts"][0]["target"] == "浅烘焙咖啡"


def test_documents_validation_error(client):
    bad = base64.b64encode(b"x").decode()
    r = client.post("/v1/documents",
                    json={"file_name": "a.pdf", "content_base64": bad}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


def test_datasets_and_task_status(client):
    assert client.get("/v1/datasets", headers=AUTH).status_code == 200
    r = client.get("/v1/tasks/missing", headers=AUTH)
    assert r.json()["status"] == "not_found"


def test_forget_via_rest(client):
    client.post("/v1/memories", json={"content": "用户喜欢浅烘焙咖啡"}, headers=AUTH)
    r = client.request("DELETE", "/v1/memories", headers=AUTH)  # 无 confirm
    assert r.status_code == 400
    r = client.delete("/v1/memories?confirm=true", headers=AUTH)
    assert r.status_code == 200 and r.json()["scope"] == "all"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.api'`

- [ ] **Step 3: 实现**

`weave/api/auth.py`:

```python
"""Bearer token 认证中间件 (spec §7.2): 仅豁免 /v1/health."""

from fastapi import Request
from fastapi.responses import JSONResponse

_EXEMPT_PATHS = {"/v1/health"}


def make_bearer_middleware(api_key: str):
    async def bearer(request: Request, call_next):
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)
        if request.headers.get("authorization", "") != f"Bearer {api_key}":
            return JSONResponse(
                {"error": {"code": "unauthorized", "message": "无效或缺失的 Bearer token"}},
                status_code=401,
            )
        return await call_next(request)

    return bearer
```

`weave/api/rest.py`:

```python
"""REST 开放 API (spec §7.2): 薄壳, 直接委托 MemoryService."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/v1")


class RememberIn(BaseModel):
    content: str
    dataset: str = "default"
    session_id: str | None = None


class RecallIn(BaseModel):
    query: str
    dataset: str | None = None
    top_k: int = 5
    session_id: str | None = None


class DocumentIn(BaseModel):
    file_name: str
    content_base64: str
    dataset: str = "default"


def _svc(request: Request):
    return request.app.state.service


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/memories")
async def remember(body: RememberIn, request: Request):
    return await _svc(request).remember(body.content, body.dataset, body.session_id)


@router.post("/recall")
async def recall(body: RecallIn, request: Request):
    return await _svc(request).recall(body.query, body.dataset, body.top_k, body.session_id)


@router.post("/documents", status_code=202)
async def cognify_file(body: DocumentIn, request: Request):
    return await _svc(request).cognify_submit(body.file_name, body.content_base64, body.dataset)


@router.get("/tasks/{task_id}")
async def task_status(task_id: str, request: Request):
    return await _svc(request).task_status(task_id)


@router.get("/datasets")
async def list_datasets(request: Request):
    return await _svc(request).list_datasets()


@router.delete("/datasets/{name}")
async def forget_dataset(name: str, request: Request):
    return await _svc(request).forget(dataset=name)


@router.delete("/memories")
async def forget_all(request: Request, confirm: bool = False):
    return await _svc(request).forget(dataset=None, confirm=confirm)
```

`weave/api/app.py`:

```python
"""FastAPI 应用组装: 认证中间件 + REST 路由 + lifespan 启动内嵌 worker.

MCP (Streamable HTTP) 挂载在 Task 16 加入.
"""

import asyncio
import contextlib

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from weave.api.auth import make_bearer_middleware
from weave.api.rest import router
from weave.config import Settings
from weave.core.service import MemoryService
from weave.infra.cache import Cache
from weave.infra.embedding import EmbeddingClient
from weave.infra.graph import GraphStore
from weave.infra.llm import LLMClient
from weave.infra.relational import RelationalStore
from weave.infra.vector import VectorStore
from weave.worker import worker_loop


def build_service(settings: Settings) -> MemoryService:
    return MemoryService(
        settings,
        RelationalStore(settings.relational_db_path),
        VectorStore(settings.vector_db_path),
        GraphStore(settings.graph_db_path),
        Cache(settings.redis_host, settings.redis_port, settings.redis_db),
        LLMClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model),
        EmbeddingClient(settings.dashscope_api_key, settings.dashscope_base_url,
                        settings.dashscope_embedding_model),
    )


def create_app(settings: Settings, service: MemoryService | None = None) -> FastAPI:
    svc = service or build_service(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        worker = asyncio.create_task(
            worker_loop(svc, svc.cache, settings.queue_poll_timeout, stop))
        try:
            yield
        finally:
            stop.set()
            worker.cancel()
            with contextlib.suppress(Exception):
                await worker

    app = FastAPI(title="Weave", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.service = svc
    app.middleware("http")(make_bearer_middleware(settings.weave_api_key))

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse({"error": {"code": "bad_request", "message": str(exc)}},
                            status_code=400)

    app.include_router(router)
    return app
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_api.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```powershell
git add weave/api tests/test_api.py; git commit -m "feat: REST 开放 API 与 Bearer 认证"
```

---

### Task 16: MCP 工具（Streamable HTTP）+ E2E + 入口

**Files:**
- Create: `weave/api/mcp_server.py`
- Modify: `weave/api/app.py`（lifespan 加入 mcp session manager；挂载 mcp app）
- Create: `weave/main.py`
- Test: `tests/test_e2e_mcp.py`

**Interfaces:**
- Consumes: Task 15 `create_app`、全部 service 方法
- Produces:
  - `create_mcp(get_service: Callable[[], MemoryService]) -> FastMCP`（闭包延迟取 service，避免导入期绑定）
  - 6 个 MCP 工具：`remember` / `recall` / `forget` / `cognify_file` / `task_status` / `list_datasets`（参数与 service 方法一致，spec §7.1）
  - `main.main()`（uvicorn 启动；`uv run weave` 可用）

**挂载要点:** FastMCP 用默认 `streamable_http_path="/mcp"`；`app.mount("/", mcp.streamable_http_app())` 必须放在 `include_router` **之后**（Starlette 按序匹配，`/v1/*` 先中）；lifespan 必须运行 `mcp.session_manager.run()`，否则 MCP 请求报 session 错误。

- [ ] **Step 1: 写失败测试**

`tests/test_e2e_mcp.py`:

```python
import asyncio
import json

import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from weave.api.app import create_app
from tests.fakes import FakeLLM
from tests.test_pipelines import LIGHT


@pytest.fixture
async def live_server(settings, make_service, fake_cache):
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings, service=svc),
                       host="127.0.0.1", port=0, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


def _payload(result) -> dict:
    assert not result.isError
    return json.loads(result.content[0].text)


async def test_mcp_full_flow(live_server):
    headers = {"Authorization": "Bearer test-key"}
    async with streamablehttp_client(live_server + "/mcp", headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {
                "remember", "recall", "forget", "cognify_file",
                "task_status", "list_datasets"}

            r = await session.call_tool("remember", {"content": "用户喜欢浅烘焙咖啡"})
            assert _payload(r)["mode"] == "permanent"

            r = await session.call_tool("recall", {"query": "用户喜欢什么"})
            assert _payload(r)["facts"][0]["target"] == "浅烘焙咖啡"

            r = await session.call_tool("list_datasets", {})
            assert _payload(r)[0]["name"] == "default"

            r = await session.call_tool("forget", {"dataset": "default"})
            assert _payload(r)["scope"] == "default"


async def test_mcp_auth_rejected(live_server):
    async with streamablehttp_client(live_server + "/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            with pytest.raises(Exception):
                await session.initialize()
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_e2e_mcp.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'weave.api.mcp_server'`

- [ ] **Step 3: 实现**

`weave/api/mcp_server.py`:

```python
"""FastMCP 工具定义 (spec §7.1): 薄壳, 委托 MemoryService.

get_service 为可调用闭包, 避免导入期与 app.state 绑定顺序问题.
"""

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP


def create_mcp(get_service: Callable) -> FastMCP:
    mcp = FastMCP("weave", stateless_http=True, json_response=True)

    @mcp.tool()
    async def remember(content: str, dataset: str = "default",
                       session_id: str | None = None) -> dict:
        """存储记忆。无 session_id：同步抽取写入永久知识图谱；
        有 session_id：先写会话缓存（立即返回），后台异步过滤沉淀进图。"""
        return await get_service().remember(content, dataset, session_id)

    @mcp.tool()
    async def recall(query: str, dataset: str | None = None, top_k: int = 5,
                     session_id: str | None = None) -> dict:
        """检索记忆。混合检索自动路由：向量相似度入口 + 知识图谱 1 跳扩展；
        传 session_id 时叠加会话缓存（source=session 为未过滤原料）。"""
        return await get_service().recall(query, dataset, top_k, session_id)

    @mcp.tool()
    async def forget(dataset: str | None = None, confirm: bool = False) -> dict:
        """删除记忆。指定 dataset：删除该数据集（级联清图/向量/元数据）；
        省略 dataset 且 confirm=True：清空全部记忆。"""
        return await get_service().forget(dataset, confirm)

    @mcp.tool()
    async def cognify_file(file_name: str, content_base64: str,
                           dataset: str = "default") -> dict:
        """异步摄取文档到知识图谱（base64 编码内容，v1 支持 txt/md）。
        立即返回 task_id，用 task_status 轮询进度。"""
        return await get_service().cognify_submit(file_name, content_base64, dataset)

    @mcp.tool()
    async def task_status(task_id: str) -> dict:
        """查询异步任务（cognify/improve）状态：pending/running/completed/failed。"""
        return await get_service().task_status(task_id)

    @mcp.tool()
    async def list_datasets() -> list[dict]:
        """列出全部数据集及规模（data_count/edge_count/entity_count）。"""
        return await get_service().list_datasets()

    return mcp
```

`weave/api/app.py` 的 `create_app` 替换为（变更点：mcp 挂载 + session manager lifespan）:

```python
def create_app(settings: Settings, service: MemoryService | None = None) -> FastAPI:
    from weave.api.mcp_server import create_mcp

    svc = service or build_service(settings)
    mcp = create_mcp(lambda: svc)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(mcp.session_manager.run())
            worker = asyncio.create_task(
                worker_loop(svc, svc.cache, settings.queue_poll_timeout, stop))
            try:
                yield
            finally:
                stop.set()
                worker.cancel()
                with contextlib.suppress(Exception):
                    await worker

    app = FastAPI(title="Weave", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.service = svc
    app.middleware("http")(make_bearer_middleware(settings.weave_api_key))

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse({"error": {"code": "bad_request", "message": str(exc)}},
                            status_code=400)

    app.include_router(router)
    # 注意: 必须在 include_router 之后挂载; /v1/* 先匹配, /mcp 落到 mcp app
    app.mount("/", mcp.streamable_http_app())
    return app
```

`weave/main.py`:

```python
"""入口: uv run weave / uv run python -m weave.main."""

import uvicorn

from weave.api.app import create_app
from weave.config import get_settings


def main() -> None:
    settings = get_settings()
    try:
        app = create_app(settings)
    except Exception as exc:
        # spec §8: Kuzu/LanceDB 被其他进程占用时给出清晰报错
        raise SystemExit(
            f"Weave 启动失败（可能是 Kuzu/LanceDB/SQLite 数据文件被其他进程占用）: {exc}"
        ) from exc
    uvicorn.run(app, host=settings.weave_host, port=settings.weave_port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_e2e_mcp.py -v`
Expected: `2 passed`

- [ ] **Step 5: 全量回归 + Commit**

Run: `uv run pytest -v`
Expected: 全部 passed（约 50 个测试）

```powershell
git add weave/api/mcp_server.py weave/api/app.py weave/main.py tests/test_e2e_mcp.py; git commit -m "feat: MCP Streamable HTTP 工具与 E2E"
```

---

## 完成验收清单（执行方自检）

- [ ] `uv run pytest` 全绿
- [ ] `uv run weave` 启动后：`curl http://127.0.0.1:8000/v1/health` 返回 `{"status":"ok"}`（无需 token）
- [ ] MySmallAgent 侧 mcp.json 配置 `{"mcpServers": {"weave": {"url": "http://127.0.0.1:8000/mcp"}}}` 可发现 6 个工具（header 支持见 spec §11 风险点）
- [ ] 真实 DeepSeek/DashScope 冒烟：`remember` 一条偏好 → `recall` 能取回事实三元组
