"""MCP (Streamable HTTP) 端到端测试：真实 uvicorn 服务 + 官方 MCP 客户端全链路验证。

覆盖 spec §7.1：6 个工具的发现、remember -> recall -> list_datasets -> forget
主流程，以及无凭据时 MCP 握手被 Bearer 中间件拒绝（/mcp 与 /v1 同等受保护）。

适配说明（mcp 2.0.0）：本环境安装的 mcp 为 2.x（FastMCP 已拆分为独立包，
mcp 包内高层服务端类为 mcp.server.mcpserver.MCPServer），客户端 API 与简报
所依据的 1.x 存在机械差异，测试语义不变：
- streamablehttp_client -> streamable_http_client，且产出 (read, write) 2 元组；
- headers 不再直接透传，改由 create_mcp_http_client(headers=...) 注入；
- CallToolResult.isError -> is_error（snake_case 字段名）；
- 返回 list 的工具结果不再整体进单个文本块，需读 structured_content["result"]。
"""

import asyncio
import json

import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from weave.api.app import create_app
from tests.fakes import FakeLLM
from tests.test_pipelines import LIGHT


@pytest.fixture
async def live_server(settings, make_service, fake_cache):
    """启动一个绑定随机端口（port=0）的真实 uvicorn 服务，注入全部测试替身。

    做什么: 用 make_service 组出带 FakeLLM（预设 LIGHT 抽取响应）与 fakeredis
        缓存的 MemoryService，交给 create_app 生成完整应用（REST + MCP +
        内嵌 worker），由 uvicorn 在后台协程中真实监听 127.0.0.1 随机端口；
        测试结束后置位 should_exit 并等待服务协程退出，保证端口与 lifespan
        （含 mcp session manager 与 worker）干净回收。
    参数:
        settings: conftest 测试配置夹具（weave_api_key="test-key"）。
        make_service: conftest 服务工厂夹具（三库指向临时目录）。
        fake_cache: conftest fakeredis 缓存夹具（worker 轮询队列所需）。
    返回:
        str: yield 服务 base URL（http://127.0.0.1:<实际端口>）。
    """
    # LIGHT 供 remember 的同步抽取消费：单 chunk 只调一次 LLM
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    # port=0 让操作系统分配空闲端口，避免并行测试/本机占用冲突
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings, service=svc),
                       host="127.0.0.1", port=0, log_level="error"))
    task = asyncio.create_task(server.serve())
    # 轮询等待 uvicorn 完成绑定与启动（startup 含 lifespan：session manager + worker）
    while not server.started:
        await asyncio.sleep(0.05)
    # 从实际监听的 socket 取回系统分配的端口号
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    # 优雅停机：should_exit 触发 uvicorn shutdown（跑完 lifespan 退出段），
    # wait_for 兜底防止 lifespan 挂死阻塞测试套件
    server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


def _payload(result) -> dict:
    """取出返回 dict 的工具调用结果负载。

    做什么: 先断言调用未报错，再把首个 TextContent 的文本按 JSON 解析
        （mcp 2.0 对 dict 返回值序列化为单个 JSON 文本块）。
    参数:
        result: ClientSession.call_tool 的返回值（CallToolResult）。
    返回:
        dict: 工具返回的 JSON 对象。
    """
    assert not result.is_error
    return json.loads(result.content[0].text)


def _payload_list(result) -> list[dict]:
    """取出返回 list 的工具调用结果负载。

    做什么: mcp 2.0 对 list 返回值改为逐项文本块 + 结构化包装，
        列表整体需从 structured_content["result"] 读取。
    参数:
        result: ClientSession.call_tool 的返回值（CallToolResult）。
    返回:
        list[dict]: 工具返回的列表（元素保持原始顺序）。
    """
    assert not result.is_error
    return result.structured_content["result"]


async def test_mcp_full_flow(live_server):
    """验证 /mcp 恰好暴露 6 个工具，并打通 写入 -> 检索 -> 数据集列表 -> 遗忘 全流程。

    参数:
        live_server: 上面的真实服务夹具，yield base URL。
    """
    # Bearer 凭据：中间件对 /mcp 挂载点同样生效，无凭据会被 401 拒绝
    headers = {"Authorization": "Bearer test-key"}
    # mcp 2.0：headers 须通过预配置的 http_client 传入；连接产出 (read, write) 2 元组
    async with streamable_http_client(
            live_server + "/mcp",
            http_client=create_mcp_http_client(headers=headers)) as (read, write):
        async with ClientSession(read, write) as session:
            # MCP 握手：协商协议版本并获取服务器能力
            await session.initialize()
            # 工具发现：必须恰为 spec §7.1 的 6 个工具，不多不少
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {
                "remember", "recall", "forget", "cognify_file",
                "task_status", "list_datasets"}

            # 写入：无 session_id 走永久管线，同步抽取入图（消费 LIGHT 响应）
            r = await session.call_tool("remember", {"content": "用户喜欢浅烘焙咖啡"})
            assert _payload(r)["mode"] == "permanent"

            # 检索：混合检索应命中刚写入的 LIKES 事实，target 即客体实体名
            r = await session.call_tool("recall", {"query": "用户喜欢什么"})
            assert _payload(r)["facts"][0]["target"] == "浅烘焙咖啡"

            # 数据集列表：default 数据集已存在且排在首位
            r = await session.call_tool("list_datasets", {})
            assert _payload_list(r)[0]["name"] == "default"

            # 遗忘：按数据集级联清理，返回值回显被清理的范围
            r = await session.call_tool("forget", {"dataset": "default"})
            assert _payload(r)["scope"] == "default"


async def test_mcp_auth_rejected(live_server):
    """验证无 Authorization 头时 MCP initialize 握手失败（中间件先于会话建立拒绝）。

    参数:
        live_server: 上面的真实服务夹具，yield base URL。
    """
    # 不配置任何凭据：POST initialize 会被 Bearer 中间件以 401 拦截，
    # 客户端据此把 initialize 置为失败（与 test_mcp_full_flow 的成功路径互为对照）
    async with streamable_http_client(live_server + "/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            with pytest.raises(Exception):
                await session.initialize()
