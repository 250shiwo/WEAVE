"""Bearer token 认证中间件 (spec §7.2): 仅豁免 /v1/health."""

from fastapi import Request
from fastapi.responses import JSONResponse

# 认证豁免路径集合：健康检查供负载均衡/探活调用，必须无需鉴权即可访问；
# 除它之外的全部端点（含未注册路径）都要求合法 Bearer token
_EXEMPT_PATHS = {"/v1/health"}


def make_bearer_middleware(api_key: str):
    """构造一个 FastAPI http 中间件：校验 Authorization 头中的 Bearer token。

    做什么: 返回一个符合 `app.middleware("http")` 协议的 async 可调用对象；
        对每个入站请求，先按路径放行豁免清单（仅 /v1/health），其余请求
        要求 Authorization 头精确等于 "Bearer <api_key>"，不匹配则直接
        以 401 错误信封短路响应（不再进入路由），匹配则放行给下一跳。
    参数:
        api_key: 服务端预期的鉴权 token（取自 settings.weave_api_key）；
            比较采用完整字符串精确相等，缺失与错误 token 不区分（防探测）。
    返回:
        callable: async (request, call_next) -> Response 的中间件函数。
    """
    async def bearer(request: Request, call_next):
        """Bearer 校验中间件本体（工厂闭包，捕获 api_key）。

        参数:
            request: 入站 HTTP 请求对象。
            call_next: ASGI 下一跳（路由处理链）。
        返回:
            Response: 豁免/鉴权通过时为下游响应；鉴权失败时为 401 JSON 响应。
        """
        # 豁免路径直接放行：/v1/health 无需任何鉴权头
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)
        # 头名大小写不敏感（Starlette headers 为不可变小写映射）；
        # 缺头时 get 返回 ""，与期望值必不相等，自然落入 401 分支
        if request.headers.get("authorization", "") != f"Bearer {api_key}":
            # 统一错误信封格式 {error: {code, message}}，与 400 处理器一致
            return JSONResponse(
                {"error": {"code": "unauthorized", "message": "无效或缺失的 Bearer token"}},
                status_code=401,
            )
        # 鉴权通过：交棒给后续中间件/路由处理
        return await call_next(request)

    return bearer
