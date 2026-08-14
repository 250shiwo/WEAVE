"""FastAPI 应用组装: 认证中间件 + REST 路由 + MCP 挂载 + lifespan 启动内嵌 worker.

MCP (Streamable HTTP) 子应用挂在根挂载点上，/v1/* 由 REST 路由优先匹配，
/mcp 落到 MCP 子应用；lifespan 同时托管 mcp session manager 与内嵌 worker。
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
    """组装 FastAPI 应用：注入服务门面、注册认证/错误处理/路由/MCP 与 lifespan。

    做什么: service 缺省时用 build_service 按 settings 装配真实依赖（生产路径），
        显式传入时直接使用（测试注入 fakeredis + Fake 客户端）；同一服务门面经
        闭包共享给 REST 路由与 6 个 MCP 工具。lifespan 在应用启动时先运行
        mcp session manager（否则 MCP 请求报 session 错误），再创建内嵌 worker
        协程轮询任务队列；关闭时先置位 stop 再取消 worker 并吞掉取消异常，
        最后由 ExitStack 逆序关闭 session manager，保证干净停机。
        docs/openapi/redoc 全部关闭（内部 API）。
    参数:
        settings: 全局配置实例（鉴权 token、队列轮询超时等取自它）。
        service: 可选的预装 MemoryService；为 None 时走 build_service 生产装配。
    返回:
        FastAPI: 组装完成、可直接运行的应用实例。
    """
    # 延迟导入：mcp_server 只在本函数内使用，避免模块级循环依赖风险
    from weave.api.mcp_server import create_mcp

    # 测试注入优先；未注入时按 settings 装配真实客户端
    svc = service or build_service(settings)
    # 闭包延迟取 service：工具被调用时才求值，规避导入期绑定顺序问题
    mcp = create_mcp(lambda: svc)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理：托管 mcp session manager 与内嵌 worker 的起停。

        参数:
            app: 当前 FastAPI 应用实例（本函数不使用，协议要求占位）。
        返回: 无（async context manager，yield 之间为应用运行期）。
        """
        # 停止信号：关闭阶段置位，worker 在本轮 BRPOP 返回后退出循环
        stop = asyncio.Event()
        # ExitStack 统一托管：先进入 mcp session manager（其 run() 是 async
        # context manager，内部启动会话任务组）；退出时按 LIFO 逆序关闭，
        # 即先停 worker（finally 段），再关 session manager
        async with contextlib.AsyncExitStack() as stack:
            # 注意：mcp.session_manager 依赖下方 streamable_http_app() 已被
            # 调用（mcp 2.x 中 session manager 由该方法创建）；lifespan 运行于
            # create_app 返回之后，此顺序天然成立
            await stack.enter_async_context(mcp.session_manager.run())
            # 内嵌 worker 协程：与 API 同进程轮询 [improve, cognify] 双队列；
            # svc.cache 即队列客户端（测试为 fakeredis，生产为真实 Redis）
            worker = asyncio.create_task(
                worker_loop(svc, svc.cache, settings.queue_poll_timeout, stop))
            try:
                yield  # 应用运行期：请求由下方路由与 MCP 子应用处理
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
    # 注册 Bearer 认证中间件（函数工厂模式，仅豁免 /v1/health 与 OPTIONS 预检）；
    # 中间件包裹整个应用（含挂载的 MCP 子应用），/mcp 与 /v1 同等受保护
    app.middleware("http")(make_bearer_middleware(settings.weave_api_key))
    # CORS 支持：允许浏览器前端（如 data/web-ui）跨域调用 API。
    # add_middleware 在 app.middleware("http") 之后注册，位于中间件栈更外层，
    # 因此 OPTIONS 预检请求由 CORSMiddleware 优先响应（不进入认证逻辑）。
    # 本地单用户工具场景，allow_origins 用 "*"（Bearer token 鉴权、无 cookie 凭证，
    # 不受 "*" 与 credentials 冲突限制）；对外暴露时应收紧为具体来源列表。
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    # 注意: 必须在 include_router 之后挂载; Starlette 按序匹配，
    # /v1/* 先中, /mcp 落到 mcp 子应用（默认 streamable_http_path="/mcp"）。
    # mcp 2.x 适配：stateless_http/json_response 从 1.x 的构造参数移到
    # streamable_http_app() 调用点，语义不变（无状态 HTTP + JSON 响应）
    app.mount("/", mcp.streamable_http_app(json_response=True, stateless_http=True))
    return app
