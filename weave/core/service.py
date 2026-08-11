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
        """写入一条记忆：session_id 为 None 时走永久管线（ingest_text）。

        参数:
            content: 待记忆的原始文本。
            dataset: 目标数据集名，默认 "default"。
            session_id: 会话 ID；非 None 时走会话暂存分支（Task 11 实现），
                本任务只实现 None 的永久分支。
        返回:
            dict: ingest_text 的结果外加 mode="permanent"，键为
                mode/data_id/dataset/deduplicated/entities/relationships/superseded/chunks。
        异常:
            NotImplementedError: session_id 非 None（会话分支尚未实现）。
        """
        # 延迟导入避免循环依赖：pipelines 模块只依赖 svc 协议，不反向 import service
        from weave.core.pipelines import ingest_text

        if session_id is not None:
            raise NotImplementedError("session 分支在 Task 11 实现")
        # 永久管线：source_pipeline/source_task 均标记为 "remember"，
        # data_name 取文本前 40 字符作为人类可读标题
        result = await ingest_text(self, content, dataset, "remember", "remember",
                                   data_name=f"remember:{content[:40]}")
        return {"mode": "permanent", **result}

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
