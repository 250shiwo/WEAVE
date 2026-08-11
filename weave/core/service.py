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

    async def recall(self, query: str, dataset: str | None = None,
                     top_k: int = 5, session_id: str | None = None) -> dict:
        """回忆查询：混合检索的门面入口（remember 的查询侧配对）。

        做什么: 把调用原样委托给 retrieval.hybrid_recall——query 向量化后
            在 text_chunks/entities 双表各取 top_k 入口，经 Kuzu 1 跳图扩展
            （仅 is_latest 边）出事实，图关联块回查 LanceDB 原文补充，
            最后按 session_id 叠加会话缓存原文。
        参数:
            query: 查询文本。
            dataset: 可选数据集过滤；为 None 时跨全部数据集检索。
            top_k: 双表各自的向量入口条数上限，默认 5。
            session_id: 可选会话标识；非空时叠加该会话的缓存原文。
        返回:
            dict: RecallResult 结构 {facts, chunks, session_items}；
                fact 标 origin="graph"，chunk 标 source="vector"/"graph"，
                会话项标 source="session"。
        """
        # 延迟导入避免循环依赖：retrieval 只依赖 svc 协议，不反向 import service
        from weave.core.retrieval import hybrid_recall

        return await hybrid_recall(self, query, dataset, top_k, session_id)

    async def run_improve(self, session_id: str, dataset: str = "default",
                          task_id: str = "") -> dict:
        """触发一次会话记忆沉淀：improve_session 管线的门面入口。

        做什么: 把调用原样委托给 pipelines.improve_session——取会话未同步
            消息、LLM 认知过滤、保留事实走标准入图、按 id 标记已同步。
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

    async def cognify_submit(self, file_name: str, content_base64: str,
                             dataset: str = "default") -> dict:
        """提交一份 base64 编码的文档：校验解码、落记录、去重判定、入 cognify 队列。

        做什么:
            1. 校验文件扩展名（v1 仅支持 txt/md/markdown）并解码 base64 为
               UTF-8 文本，任一失败抛 ValueError（不产生任何写入）；
            2. 按解码后文本的内容哈希在数据集内联表去重：命中且记录为
               completed 时，只补一条已完成的 pipeline_run 记录作为审计痕迹，
               直接返回 completed + deduplicated（不入队、不重复抽取）；
            3. 命中但记录为非完成态（上次执行失败等）时，复用其 data_id
               新建 pending 运行记录并重新入队（绝不假报 completed——
               图中尚无数据，返回 pending 等待重跑）；
            4. 未命中时创建 data 记录（source_pipeline="cognify"）并关联数据集，
               创建 pending 状态的 pipeline_run，把 {task_id, data_id, dataset}
               载荷入 QUEUE_COGNIFY，等待 worker 异步执行 run_cognify_task。
        参数:
            file_name: 原始文件名（仅用扩展名判断类型，并作为 data 记录名称）。
            content_base64: 文件内容的 base64 编码字符串（解码后须为 UTF-8）。
            dataset: 目标数据集名，默认 "default"。
        返回:
            dict: 新任务（含失败重提）为 {task_id, data_id, status: "pending"}；
                去重短路（命中已完成记录）时额外带 deduplicated: True 且
                status 为 "completed"。
        异常:
            ValueError: 扩展名不支持或 base64 解码失败时抛出。
        """
        import base64
        from pathlib import Path

        from weave.core.models import content_hash, new_id
        from weave.infra.cache import QUEUE_COGNIFY

        # 扩展名白名单校验：v1 只支持纯文本/Markdown，其余直接拒绝
        suffix = Path(file_name).suffix.lower()
        if suffix not in {".txt", ".md", ".markdown"}:
            raise ValueError(f"暂不支持的文件类型: {suffix} (v1 支持 txt/md)")
        try:
            # base64 解码为字节串后再按 UTF-8 解码为文本
            raw = base64.b64decode(content_base64).decode("utf-8")
        except Exception as exc:
            # 统一包装为 ValueError，调用方只需捕获一种异常
            raise ValueError(f"无法解码 base64 文件内容: {exc}") from exc

        # 数据集按名获取/创建（幂等），后续关联与去重判定都基于它的内部 id
        ds = await self.db(self.relational.get_or_create_dataset, dataset)
        # 内容哈希去重：同文本同数据集只入库一次（联表查询限定数据集内，
        # 跨数据集的同内容互不干扰，也不会因同哈希多行而抛 MultipleResultsFound）
        ch = content_hash(raw)
        existing = await self.db(self.relational.get_data_by_hash, ch, ds["id"])
        if existing and existing["status"] == "completed":
            # 去重短路仅对已完成记录生效：补一条直接置 completed 的运行记录作审计，不入队
            task_id = new_id()
            await self.db(self.relational.create_pipeline_run, task_id, "cognify", existing["id"])
            await self.db(self.relational.update_pipeline_run, task_id, "completed")
            return {"task_id": task_id, "data_id": existing["id"],
                    "status": "completed", "deduplicated": True}

        if existing:
            # 非完成态（上次执行失败等）：复用已有 data_id 重新入队重跑，
            # 返回 pending 而非 completed（图中尚无数据，不假报完成、不静默丢数据）
            data_id = existing["id"]
        else:
            # 未命中：先落 data 记录（状态 created，由 run_cognify_task 流转终态）
            data_id = new_id()
            await self.db(self.relational.create_data, data_id, file_name, raw, ch, "cognify")
            await self.db(self.relational.link_dataset_data, ds["id"], data_id)
        # 建立 pending 状态的运行记录，供 task_status 查询进度
        task_id = new_id()
        await self.db(self.relational.create_pipeline_run, task_id, "cognify", data_id)
        # 载荷键名与 run_cognify_task 参数名一致，worker 直接 **payload 分发
        await self.cache.enqueue(QUEUE_COGNIFY,
                                 {"task_id": task_id, "data_id": data_id, "dataset": dataset})
        return {"task_id": task_id, "data_id": data_id, "status": "pending"}

    async def run_cognify_task(self, task_id: str, data_id: str, dataset: str) -> None:
        """执行一条 cognify 任务：取已落库文本走标准入图管线，流转运行状态。

        做什么: 先把 pipeline_run 置 running；按 data_id 取出 cognify_submit
            阶段已落库的 data 记录，复用 ingest_text 三阶段写（传 data_id=
            跳过去重建记录，血缘标记 source_pipeline/source_task 均为
            "cognify"）；成功置 completed，失败置 failed 并记录错误信息。
        参数:
            task_id: 队列任务 ID（pipeline_run 主键，由 cognify_submit 创建）。
            data_id: 待入图的数据记录 ID（记录已在提交阶段落库）。
            dataset: 目标数据集名。
        返回: 无。
        异常: 不抛出——任何失败都只把运行记录置 failed（含错误文本）后返回，
            worker 主循环的 except 仅作兜底。
        """
        from weave.core.pipelines import ingest_text

        # 状态流转第一步：pending -> running
        await self.db(self.relational.update_pipeline_run, task_id, "running")
        try:
            # 取提交阶段已落库的 data 记录；记录缺失视为失败（走 except 置 failed）
            record = await self.db(self.relational.get_data, data_id)
            if record is None:
                raise ValueError(f"data 记录不存在: {data_id}")
            # 传 data_id= 走 cognify 分支：跳过去重建记录，直接入图已有文本
            await ingest_text(self, record["raw_text"], dataset, "cognify", "cognify",
                              record["name"], data_id=data_id)
            # 全部写入完成：running -> completed
            await self.db(self.relational.update_pipeline_run, task_id, "completed")
        except Exception as exc:
            # 失败不抛出：running -> failed 并携带错误信息，供 task_status 查询
            await self.db(self.relational.update_pipeline_run, task_id, "failed", str(exc))

    async def task_status(self, task_id: str) -> dict:
        """查询队列任务的运行状态（pipeline_run 记录的只读视图）。

        参数:
            task_id: 队列任务 ID。
        返回:
            dict: 存在时返回完整运行记录行（task_id/pipeline_name/data_id/
                status/error/created_at/updated_at）；不存在时返回
                {"task_id": task_id, "status": "not_found"}，不抛异常。
        """
        run = await self.db(self.relational.get_pipeline_run, task_id)
        if run is None:
            # 未知任务：以 not_found 状态明示，而非返回 None 或抛异常
            return {"task_id": task_id, "status": "not_found"}
        return run

    async def forget(self, dataset: str | None = None, confirm: bool = False) -> dict:
        """遗忘入口：按数据集级联清三库；dataset=None 时清空全部记忆（需 confirm=True）。

        做什么: 删除侧门面方法，具体级联逻辑委托 cleanup.forget_dataset——
            - 指定 dataset：只清空该数据集（SQLite 关联行/边/孤立 data +
              Kuzu 节点边 + LanceDB 向量行）；
            - dataset 为 None：全清语义，属于高危操作，必须显式传
              confirm=True 才执行，否则抛 ValueError；确认后枚举关系库中
              全部数据集名，逐个走 forget_dataset 级联清空。
        参数:
            dataset: 待清空的数据集名；为 None 时表示清空全部记忆。
            confirm: 全清确认开关，仅 dataset 为 None 时生效；缺省 False
                时拒绝执行（防误操作）。
        返回:
            dict: 指定数据集时为 {"scope": 数据集名, "data", "edges"}；
                全清时为 {"scope": "all", "datasets": [各数据集的清理结果]}。
        异常:
            ValueError: dataset 为 None 且 confirm 为 False 时抛出，
                错误消息含 "confirm"。
        """
        # 延迟导入避免循环依赖：cleanup 只依赖 svc 协议，不反向 import service
        from weave.core.cleanup import forget_dataset

        if dataset is None:
            # 全清是高危操作：未显式确认时直接拒绝，不产生任何删除
            if not confirm:
                raise ValueError("清空全部记忆需要 confirm=True")
            # 枚举关系库中全部数据集名，作为全清的删除清单
            names = [d["name"] for d in await self.db(self.relational.list_datasets)]
            # 逐个数据集级联清三库，汇总每个数据集的清理结果
            return {"scope": "all",
                    "datasets": [await forget_dataset(self, n) for n in names]}
        # 单数据集清理：直接委托 cleanup.forget_dataset
        return await forget_dataset(self, dataset)

    async def forget_by_source(self, source_pipeline: str) -> dict:
        """按来源管线清理关系事实：core 内部方法 (spec §6③)，不暴露 MCP/REST。

        做什么: 把调用原样委托给 cleanup.forget_by_source——只删除该来源
            产生的 SQLite edges 元数据（含全部历史版本）与 Kuzu RELATES_TO
            边；实体节点可能被多来源共享，一律保留。
        参数:
            source_pipeline: 来源管线名（如 "session_improve"，用于清理
                会话沉淀产生的关系事实）。
        返回:
            dict: {"scope": "source:<来源>", "edges_sql", "edges_graph"}。
        """
        # 延迟导入避免循环依赖；重命名 _impl 以与本方法区分
        from weave.core.cleanup import forget_by_source as _impl

        return await _impl(self, source_pipeline)
