"""MemoryService: core 门面, 持有全部存储/客户端, 委托 pipelines/retrieval/cleanup."""

from typing import Any

from weave.config import Settings
from weave.infra.executor import run_db


class MemoryService:
    """记忆平台核心门面：聚合配置、三库实例与 LLM/嵌入客户端，对外提供统一业务入口。

    做什么: 持有 Settings、关系库/向量库/图库、缓存、LLM 与嵌入器的引用，
        所有具体业务流程委托给 core.pipelines / retrieval / cleanup 等模块实现；
        同时以 db() 方法为全部嵌入式 DB 调用提供统一的线程/超时/重试包装。
    """

    def __init__(self, settings: Settings, relational, vector, graph, cache, llm, embedder):
        """组装 MemoryService 实例（纯依赖注入，不做任何 IO）。

        参数:
            settings: 全局配置实例（超时/重试/切块参数等均取自它）。
            relational: RelationalStore，SQLite 关系存储。
            vector: VectorStore，LanceDB 向量存储。
            graph: GraphStore，Kuzu 图存储。
            cache: Redis 缓存客户端；remember 永久分支不使用，可为 None。
            llm: LLM 客户端，提供 async complete_json(system, user) -> dict。
            embedder: 嵌入客户端，提供 async embed(texts) -> list[list[float]]。
        """
        self.settings = settings
        self.relational = relational
        self.vector = vector
        self.graph = graph
        self.cache = cache
        self.llm = llm
        self.embedder = embedder

    async def db(self, fn, *args, **kwargs) -> Any:
        """所有嵌入式 DB 调用的统一入口: 单线程执行器 + 超时 + 瞬时重试.

        参数:
            fn: 要执行的同步 DB 可调用对象（如 relational/graph/vector 的方法）。
            *args: 透传给 fn 的位置参数。
            **kwargs: 透传给 fn 的关键字参数。
        返回:
            Any: fn 的返回值，原样回传。
        """
        # 超时与重试次数取自全局配置，统一所有 DB 调用的保护策略
        return await run_db(fn, *args, timeout=self.settings.db_call_timeout,
                            max_retries=self.settings.db_call_max_retries, **kwargs)

    async def remember(self, content: str, dataset: str = "default",
                       session_id: str | None = None) -> dict:
        """写入一条记忆：session_id 决定走永久管线还是会话暂存分支（双模式）。

        做什么:
            - 永久分支（session_id 为 None）：直接走 ingest_text 三阶段入图，
              血缘标记 "remember"。
            - 会话分支（session_id 非 None，spec §6 防污染）：原文只追加进
              Redis 会话 list（永不直接进图/向量库），创建 improve 管线
              运行记录并把任务入 QUEUE_IMPROVE 队列，等待 improve 流程
              异步做认知过滤后再沉淀入图。
        参数:
            content: 待记忆的原始文本。
            dataset: 目标数据集名，默认 "default"。
            session_id: 会话 ID；为 None 时走永久分支，非 None 时走会话暂存分支。
        返回:
            dict: 永久分支为 ingest_text 的结果外加 mode="permanent"，键为
                mode/data_id/dataset/deduplicated/entities/relationships/superseded/chunks；
                会话分支为 {mode: "session", session_id, queued: True, task_id}。
        """
        # 延迟导入避免循环依赖：pipelines 模块只依赖 svc 协议，不反向 import service
        from weave.core.pipelines import ingest_text
        from weave.core.session import append_session
        from weave.infra.cache import QUEUE_IMPROVE
        from weave.core.models import new_id

        if session_id is not None:
            # 会话分支第一步：原文只写 Redis 会话 list（带上限裁剪与 TTL 刷新）
            await append_session(self.cache, self.settings, session_id, content)
            # 生成任务 ID 并建立 pipeline_run 记录（初始 pending），供 worker 流转状态
            task_id = new_id()
            await self.db(self.relational.create_pipeline_run, task_id, "improve")
            # 任务载荷入 improve 队列（优先级高于 cognify），等待异步沉淀
            await self.cache.enqueue(QUEUE_IMPROVE,
                                     {"task_id": task_id, "session_id": session_id,
                                      "dataset": dataset})
            return {"mode": "session", "session_id": session_id,
                    "queued": True, "task_id": task_id}
        # 永久管线：source_pipeline/source_task 均标记为 "remember"，
        # data_name 取文本前 40 字符作为人类可读标题
        result = await ingest_text(self, content, dataset, "remember", "remember",
                                   data_name=f"remember:{content[:40]}")
        return {"mode": "permanent", **result}

    async def run_improve(self, session_id: str, dataset: str = "default",
                          task_id: str = "") -> dict:
        """触发一次会话记忆沉淀：improve_session 管线的门面入口。

        做什么: 把调用原样委托给 pipelines.improve_session——取会话未同步
            消息、LLM 认知过滤、保留事实走标准入图、标记已同步。
        参数:
            session_id: 会话标识。
            dataset: 目标数据集名，默认 "default"。
            task_id: 队列任务 ID；worker 消费队列时传入以流转 pipeline_run
                状态（running -> completed/failed），直接调用时缺省为空串，
                跳过状态流转。
        返回:
            dict: improve_session 的结果 {kept, discarded, ingested}。
        """
        # 延迟导入避免循环依赖；具体流程全部委托给 pipelines.improve_session
        from weave.core.pipelines import improve_session

        return await improve_session(self, session_id, dataset, task_id)

    async def list_datasets(self) -> list[dict]:
        """汇总各数据集统计：名称/数据条目数/边数来自关系库，实体数来自图库。

        参数: 无。
        返回:
            list[dict]: [{name, data_count, edge_count, entity_count}, ...]；
                无任何数据集时为空列表。
        """
        # data_count/edge_count 由 SQLite 统计（关联表 + edges 表）
        stats = await self.db(self.relational.dataset_stats)
        for row in stats:
            # entity_count 只能查图库：实体节点只存在于 Kuzu，按数据集名过滤
            row["entity_count"] = await self.db(self.graph.count_entities, row["name"])
        return stats
