"""MCP 工具定义 (spec §7.1): 薄壳, 委托 MemoryService.

get_service 为可调用闭包, 避免导入期与 app.state 绑定顺序问题.

适配说明（mcp 2.0.0）: 任务简报依据 mcp 1.x 的 mcp.server.fastmcp.FastMCP
编写；本环境安装 mcp 2.x，FastMCP 已拆分为独立包，mcp 包内等价的高层服务端
类为 mcp.server.mcpserver.MCPServer（同名 tool() 装饰器 / session_manager /
streamable_http_app() 均在）。1.x 构造参数 stateless_http/json_response 在
2.x 移至 streamable_http_app() 调用点（见 weave/api/app.py 的挂载行），
对外行为完全等价：无状态 HTTP + JSON 响应 + 默认 /mcp 路径。
"""

from collections.abc import Callable

from mcp.server.mcpserver import MCPServer


def create_mcp(get_service: Callable) -> MCPServer:
    """组装 Weave 的 MCP 服务：注册 spec §7.1 规定的 6 个工具。

    做什么: 创建名为 "weave" 的 MCPServer，把 MemoryService 的 6 个对外方法
        逐一包装为 MCP 工具；每个工具都是薄壳——仅在被调用时经 get_service()
        取服务门面并原样转发入参，自身不含任何业务逻辑。
    参数:
        get_service: 零参可调用闭包，返回 MemoryService 实例；延迟取值避免
            导入期与 app.state 的绑定顺序问题（服务门面在 create_app 中装配，
            闭包在每次工具调用时才求值，天然拿到最终实例）。
    返回:
        MCPServer: 工具注册完毕的 MCP 服务实例（简报所称的 FastMCP 在
            mcp 2.x 的等价物）；由调用方经 streamable_http_app() 生成
            ASGI 子应用挂载进 FastAPI。
    """
    # mcp 2.x 构造仅收服务器名；无状态/JSON 响应选项移到 streamable_http_app()
    mcp = MCPServer("weave")

    @mcp.tool()
    async def remember(content: str, dataset: str = "default",
                       session_id: str | None = None) -> dict:
        """存储记忆（session_id 决定双模式）。

        做什么: 无 session_id：同步抽取写入永久知识图谱；有 session_id：
            先写会话缓存（立即返回），后台异步过滤沉淀进图。
        参数:
            content: 待记忆文本。
            dataset: 目标数据集名，默认 "default"。
            session_id: 可选会话标识；非空时走会话暂存分支。
        返回:
            dict: 永久模式含 mode="permanent"/entities/relationships/
                superseded 等统计；会话模式含 mode="session"。
        """
        return await get_service().remember(content, dataset, session_id)

    @mcp.tool()
    async def recall(query: str, dataset: str | None = None, top_k: int = 5,
                     session_id: str | None = None) -> dict:
        """检索记忆（混合检索自动路由）。

        做什么: 向量相似度入口 + 知识图谱 1 跳扩展；传 session_id 时叠加
            会话缓存（source="session" 为未过滤原料）。
        参数:
            query: 查询文本。
            dataset: 可选数据集过滤；为 None 时跨全部数据集检索。
            top_k: 向量入口条数上限，默认 5。
            session_id: 可选会话标识；非空时叠加该会话的缓存原文。
        返回:
            dict: {facts, chunks, session_items}；fact 标 origin="graph"，
                chunk 标 source="vector"/"graph"，会话项标 source="session"。
        """
        return await get_service().recall(query, dataset, top_k, session_id)

    @mcp.tool()
    async def forget(dataset: str | None = None, confirm: bool = False) -> dict:
        """删除记忆（按数据集级联 / 全量清空）。

        做什么: 指定 dataset：删除该数据集（级联清图/向量/元数据）；
            省略 dataset 且 confirm=True：清空全部记忆。
        参数:
            dataset: 可选目标数据集名；为 None 时表示全量清空。
            confirm: 全量清空的确认开关，默认 False（防误删）。
        返回:
            dict: 含 scope（被清理的数据集名或 "all"）及各库清理统计。
        """
        return await get_service().forget(dataset, confirm)

    @mcp.tool()
    async def cognify_file(file_name: str, content_base64: str,
                           dataset: str = "default") -> dict:
        """异步摄取文档到知识图谱（v1 支持 txt/md）。

        做什么: 校验 base64 解码、落数据记录、内容去重后投入 cognify 队列，
            立即返回 task_id；实际分块/抽取/入图由内嵌 worker 异步完成，
            调用方用 task_status 轮询进度。
        参数:
            file_name: 原始文件名（扩展名用于类型校验）。
            content_base64: 文件内容的 base64 编码。
            dataset: 目标数据集名，默认 "default"。
        返回:
            dict: 含 task_id（队列任务 ID）与 data_id（数据记录 ID）。
        """
        return await get_service().cognify_submit(file_name, content_base64, dataset)

    @mcp.tool()
    async def task_status(task_id: str) -> dict:
        """查询异步任务（cognify/improve）状态。

        做什么: 读取 pipeline_run 记录，回显任务的当前状态与进度。
        参数:
            task_id: cognify_file 提交时返回的队列任务 ID。
        返回:
            dict: 含 status（pending/running/completed/failed）与
                错误信息（失败时）等字段。
        """
        return await get_service().task_status(task_id)

    @mcp.tool()
    async def list_datasets() -> list[dict]:
        """列出全部数据集及规模。

        做什么: 汇总三库统计——名称/数据条目数/边数来自关系库，
            实体数来自图库。
        参数: 无。
        返回:
            list[dict]: 每个数据集一项，含 name/data_count/edge_count/
                entity_count。
        """
        return await get_service().list_datasets()

    return mcp
