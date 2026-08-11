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
    """生产装配：按 settings 创建真实的三库/缓存/LLM/embedding 客户端并组出门面。

    参数:
        settings: 全局配置实例，提供全部存储路径、Redis 连接信息与
            DeepSeek/DashScope 的密钥、接入地址与模型名。
    返回:
        MemoryService: 依赖全部就位的服务门面；各客户端构造均为纯对象创建，
            真实连接在首次调用时惰性建立（与 infra 各模块现有行为一致）。
    """
    return MemoryService(
        settings,
        RelationalStore(settings.relational_db_path),  # SQLite 关系库（建五张表）
        VectorStore(settings.vector_db_path),  # LanceDB 向量库（懒建表）
        GraphStore(settings.graph_db_path),  # Kuzu 图库（建四张表）
        Cache(settings.redis_host, settings.redis_port, settings.redis_db),  # 真实 Redis
        LLMClient(settings.deepseek_api_key, settings.deepseek_base_url,
                  settings.deepseek_model),  # DeepSeek 抽取客户端
        EmbeddingClient(settings.dashscope_api_key, settings.dashscope_base_url,
                        settings.dashscope_embedding_model),  # DashScope 嵌入客户端
    )


def create_app(settings: Settings, service: MemoryService | None = None) -> FastAPI:
    """组装 FastAPI 应用：注入服务门面、注册认证中间件/错误处理器/路由与 lifespan。

    做什么: service 缺省时用 build_service 按 settings 装配真实依赖（生产路径），
        显式传入时直接使用（测试注入 fakeredis + Fake 客户端）；lifespan 在应用
        启动时创建内嵌 worker 协程轮询任务队列，关闭时先置位 stop 再取消 worker
        并吞掉取消异常，保证干净停机。docs/openapi/redoc 全部关闭（内部 API）。
    参数:
        settings: 全局配置实例（鉴权 token、队列轮询超时等取自它）。
        service: 可选的预装 MemoryService；为 None 时走 build_service 生产装配。
    返回:
        FastAPI: 组装完成、可直接运行的应用实例。
    """
    # 测试注入优先；未注入时按 settings 装配真实客户端
    svc = service or build_service(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理：启动时拉起内嵌 worker，关闭时优雅停止。

        参数:
            app: 当前 FastAPI 应用实例（本函数不使用，协议要求占位）。
        返回: 无（async context manager，yield 之间为应用运行期）。
        """
        # 停止信号：关闭阶段置位，worker 在本轮 BRPOP 返回后退出循环
        stop = asyncio.Event()
        # 内嵌 worker 协程：与 API 同进程轮询 [improve, cognify] 双队列；
        # svc.cache 即队列客户端（测试为 fakeredis，生产为真实 Redis）
        worker = asyncio.create_task(
            worker_loop(svc, svc.cache, settings.queue_poll_timeout, stop))
        try:
            yield  # 应用运行期：请求由下方路由处理
        finally:
            # 优雅停机两步：先置位 stop 让 worker 自行退出，再 cancel 兜底
            # （覆盖 worker 正阻塞在长 BRPOP 中的情况，双保险幂等）
            stop.set()
            worker.cancel()
            # 吞掉取消异常与一切退出异常：停机路径不允许再抛错。
            # 注意 asyncio.CancelledError 自 Python 3.8 起继承 BaseException，
            # 不在 Exception 体系内，必须显式列出才能被 suppress 捕获，
            # 否则 await 已取消的 worker 会把 CancelledError 抛给 lifespan 调用方
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker

    # docs/openapi/redoc 全部关闭：内部 API 不暴露交互式文档与 schema
    app = FastAPI(title="Weave", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    # 服务门面挂到应用状态，路由层经 request.app.state.service 取用
    app.state.service = svc
    # 注册 Bearer 认证中间件（函数工厂模式，仅豁免 /v1/health）
    app.middleware("http")(make_bearer_middleware(settings.weave_api_key))

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        """业务校验失败统一出口：ValueError -> 400 错误信封。

        参数:
            request: 触发异常的请求对象（本函数不使用，协议要求占位）。
            exc: 服务层/路由抛出的 ValueError，消息原样透传给调用方。
        返回:
            JSONResponse: {"error": {"code": "bad_request", "message": ...}}，状态码 400。
        """
        return JSONResponse({"error": {"code": "bad_request", "message": str(exc)}},
                            status_code=400)

    # 挂载 /v1 前缀的全部 REST 端点（8 个，见 weave/api/rest.py）
    app.include_router(router)
    return app
