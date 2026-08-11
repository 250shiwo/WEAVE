"""写入管线: ingest_text (remember/cognify/improve 共用) — spec §5.1 三阶段写.

improve_session 为会话记忆沉淀管线 (spec §5.3/§6): 会话原文只进 Redis,
LLM 过滤后的陈述句才复用 ingest_text 入图, 血缘隔离标记 session_improve。
"""

from weave.core.chunking import split_text
from weave.core.extraction import extract_graph
from weave.core.models import content_hash, edge_id_for, entity_id_for, new_id, norm_name, utcnow


async def ingest_text(svc, text: str, dataset: str, source_pipeline: str,
                      source_task: str, data_name: str, data_id: str | None = None) -> dict:
    """把一段文本写入知识图谱：切块 -> LLM 抽取 -> 版本更替 -> 三库统一写入。

    做什么: 严格按三阶段执行——
        阶段1: 全部 chunk 的 LLM 抽取先全部完成（任一失败则不写任何图/向量，
            只把数据记录标记为 failed，保证不留半个子图）；
        阶段2: 在内存中归并实体（确定性 ID 幂等合并）、逐关系计算版本更替
            （此阶段只读 DB，不做任何写入）；
        阶段3: 统一写三库（Kuzu 图 -> LanceDB 向量 -> SQLite 元数据），
            最后把数据记录状态流转为 completed。
    参数:
        svc: MemoryService 门面（提供 db()/settings/llm/embedder/三库引用）。
        text: 待写入的原始文本。
        dataset: 目标数据集名（不存在则自动创建）。
        source_pipeline: 来源管线名（remember/cognify/improve），写入边与数据记录的血缘。
        source_task: 来源任务名，写入边元数据的血缘字段。
        data_name: 数据记录的人类可读名称（仅 data_id 为 None 的新建路径使用）。
        data_id: 可选；传入时（cognify 路径）跳过内容去重与建记录，
            直接使用已有 data 记录；为 None 时（remember/improve 路径）
            先按内容哈希去重，未命中则新建数据记录并关联数据集。
    返回:
        dict: {data_id, dataset, deduplicated, entities, relationships,
            superseded, chunks}；去重命中时 deduplicated=True 且各计数为 0。
    异常:
        Exception: 任一阶段失败（含 LLM 错误）时，先把数据记录标记 failed 再原样抛出。
    """
    # 数据集按名获取/创建（幂等），后续关联与血缘都基于它的内部 id
    ds = await svc.db(svc.relational.get_or_create_dataset, dataset)

    if data_id is None:
        # remember/improve 路径: 同 dataset 内容哈希去重
        ch = content_hash(text)
        existing = await svc.db(svc.relational.get_data_by_hash, ch)
        # 哈希命中且确实挂在当前数据集下才算重复：直接返回，不重复抽取/写入
        if existing and await svc.db(svc.relational.is_data_linked, ds["id"], existing["id"]):
            return {"data_id": existing["id"], "dataset": dataset, "deduplicated": True,
                    "entities": 0, "relationships": 0, "superseded": 0, "chunks": 0}
        # 未命中：生成新数据 ID，建记录（状态 created）并关联到数据集
        data_id = new_id()
        await svc.db(svc.relational.create_data, data_id, data_name, text, ch, source_pipeline)
        await svc.db(svc.relational.link_dataset_data, ds["id"], data_id)

    try:
        chunk_texts = split_text(text, svc.settings.chunk_size, svc.settings.chunk_overlap)
        now = utcnow()  # 本次写入统一使用同一时间戳，保证跨库时间一致

        # 阶段1: 全部 LLM 抽取 (失败则不写任何图/向量)
        per_chunk = []
        for position, chunk_text in enumerate(chunk_texts):
            # 直接 await LLM 调用（非 DB 调用，不经 svc.db 包装）；
            # 此处抛异常会进 except 分支：只标 failed，三库无任何写入
            extracted = await extract_graph(svc.llm, chunk_text)
            per_chunk.append((position, chunk_text, extracted))

        # 阶段2: 内存归并实体 + 计算版本更替 (只读 DB)
        all_entities: dict[str, dict] = {}  # eid -> 实体载荷（精确 9 键，供 kuzu upsert）
        new_entity_ids: list[str] = []  # 图库中尚不存在、本次新建实体 ID（保序）
        chunk_rows: list[dict] = []  # 待写入的文本块（chunk_id/text/position/entity_ids）
        edge_plan: list[dict] = []  # 待写入的新边计划
        supersede_plan: list[dict] = []  # 待置非最新的旧边（图库查询结果）

        async def ensure_entity(name: str, entity_type: str, description: str) -> str:
            """按名确保实体已进入内存归并表，返回其确定性 ID（闭包，读写外层归并状态）。

            参数:
                name: 实体原始名（内部按 dataset+规范化名算确定性 ID）。
                entity_type: 实体类别（首次见到该实体时采用）。
                description: 实体描述（首次见到该实体时采用）。
            返回:
                str: 该实体的确定性 ID；同一 (dataset, 规范化名) 多次调用只归并一次。
            """
            eid = entity_id_for(dataset, name)
            if eid not in all_entities:
                # 首次见到：构造精确 9 键实体载荷（kuzu 0.11 对多余参数键报错，勿增删键）
                all_entities[eid] = dict(
                    id=eid, name=name, norm_name=norm_name(name), entity_type=entity_type,
                    description=description, dataset=dataset, version=1,
                    is_latest=True, created_at=now)
                # 图库查不到的才算新实体（只读查询），需补写实体向量
                if not await svc.db(svc.graph.get_entity_by_id, eid):
                    new_entity_ids.append(eid)
            return eid

        for position, chunk_text, extracted in per_chunk:
            local: dict[str, str] = {}  # 本 chunk 内 规范化名 -> eid，供关系解析端点
            chunk_entity_ids: set[str] = set()  # 本 chunk 提及的全部实体（MENTIONS 用）
            for ee in extracted.entities:
                eid = await ensure_entity(ee.name, ee.entity_type, ee.description)
                local[norm_name(ee.name)] = eid
                chunk_entity_ids.add(eid)
            for rel in extracted.relationships:
                # 关系端点优先用本 chunk 已声明的实体（保留其类型/描述）；
                # 未声明的端点按 Thing 兜底补建，保证边两端必有实体
                src_id = local.get(norm_name(rel.source)) or await ensure_entity(rel.source, "Thing", "")
                dst_id = local.get(norm_name(rel.target)) or await ensure_entity(rel.target, "Thing", "")
                if src_id == dst_id:
                    continue  # 自环边无语义，跳过
                chunk_entity_ids.update({src_id, dst_id})
                # 查当前最新同类型边（只读）：决定是重复、新增还是更替
                prev = await svc.db(svc.graph.get_latest_relationship, src_id, rel.relationship_type)
                version = 1
                if prev is not None:
                    if prev["target_id"] == dst_id:
                        continue  # 重复事实, 跳过
                    # 目标变化 = 事实变更：旧边进入更替计划，新版本号 +1
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
        # 3.1 版本更替：先把旧边在图库与关系库双双置为非最新（历史保留可查）
        for prev in supersede_plan:
            await svc.db(svc.graph.set_relationship_not_latest, prev["edge_id"])
            await svc.db(svc.relational.set_edge_not_latest, prev["edge_id"], now)
        # 3.2 实体：确定性 ID upsert，已存在的实体幂等跳过
        for entity in all_entities.values():
            await svc.db(svc.graph.upsert_entity, entity)
        # 3.3 关系边：图库写边本体，关系库写版本/血缘元数据（含 source_task/updated_at）
        for edge in edge_plan:
            await svc.db(svc.graph.upsert_relationship, edge)
            await svc.db(svc.relational.upsert_edge,
                         {**edge, "source_task": source_task, "updated_at": now})
        # 3.4 文本块：图库建 TextChunk 节点并挂 MENTIONS 提及边
        for row in chunk_rows:
            await svc.db(svc.graph.create_chunk, row["chunk_id"], data_id, dataset, now)
            if row["entity_ids"]:
                await svc.db(svc.graph.link_mentions, row["chunk_id"], row["entity_ids"])

        # 3.5 文本块向量：嵌入后直接 LanceDB（嵌入非 DB 调用，不经 svc.db）
        chunk_vecs = await svc.embedder.embed([r["text"] for r in chunk_rows])
        await svc.db(svc.vector.add_chunks, [
            dict(chunk_id=r["chunk_id"], vector=vec, text=r["text"], data_id=data_id,
                 dataset=dataset, created_at=now)
            for r, vec in zip(chunk_rows, chunk_vecs)
        ])
        # 3.6 新实体向量：仅图库中不存在的实体需要补写（旧实体向量仍在有效期内）
        new_entities = [all_entities[eid] for eid in new_entity_ids]
        if new_entities:
            # 嵌入文本取 "名称 描述" 拼接，语义更完整；strip 防空描述留下尾空格
            entity_vecs = await svc.embedder.embed(
                [f"{e['name']} {e['description']}".strip() for e in new_entities])
            await svc.db(svc.vector.add_entities, [
                dict(entity_id=e["id"], vector=vec, name=e["name"],
                     entity_type=e["entity_type"], description=e["description"],
                     dataset=dataset, is_latest=True)
                for e, vec in zip(new_entities, entity_vecs)
            ])

        # 全部写入完成：数据记录状态流转为 completed
        await svc.db(svc.relational.set_data_status, data_id, "completed")
    except Exception:
        # 任一阶段失败：只把数据记录标记 failed 后原样上抛；
        # 阶段1 失败时三库尚无任何写入，天然无残留
        await svc.db(svc.relational.set_data_status, data_id, "failed")
        raise

    return {"data_id": data_id, "dataset": dataset, "deduplicated": False,
            "entities": len(new_entity_ids), "relationships": len(edge_plan),
            "superseded": len(supersede_plan), "chunks": len(chunk_texts)}


async def improve_session(svc, session_id: str, dataset: str = "default",
                          task_id: str = "") -> dict:
    """会话记忆沉淀: LLM 认知过滤 + 重要性门禁, 通过后才走标准入图 (spec §5.3/§6).

    做什么: 取出会话中尚未同步的原文消息，先经 LLM 认知过滤（只保留用户画像/
        偏好/稳定事实/重要决定，一次性事件与闲聊一律丢弃）；存在保留事实时
        才拼成文本走标准 ingest_text 三阶段入图，血缘标记 source_pipeline=
        "session_improve"；无论是否入图，最后都会把会话标记为已同步。
        防污染关键：会话原文永不直接入图，只有过滤后的陈述句可以进入。
    参数:
        svc: MemoryService 门面（提供 db()/settings/cache/llm/三库引用）。
        session_id: 会话标识，对应 Redis 中的会话 list。
        dataset: 目标数据集名，默认 "default"。
        task_id: 队列任务 ID；非空时同步流转 pipeline_run 状态
            （running -> completed/failed），为空（直接调用）时跳过状态流转。
    返回:
        dict: {kept: 保留事实数, discarded: 丢弃消息数,
            ingested: ingest_text 结果或 None（无保留/无未同步消息时不写图）}。
    异常:
        Exception: 任一环节失败时先把 pipeline_run 标记 failed（含错误信息），
            再原样上抛给调用方。
    """
    # 延迟导入：保持 pipelines 只依赖 svc 协议，避免模块级依赖扩散
    from weave.core.extraction import filter_session_facts
    from weave.core.session import get_unsynced, mark_synced

    # 有 task_id（队列消费路径）时先把运行记录置为 running
    if task_id:
        await svc.db(svc.relational.update_pipeline_run, task_id, "running")
    try:
        # 只取未同步的增量消息；已同步的历史原文不重复处理
        items = await get_unsynced(svc.cache, session_id)
        if not items:
            # 无待处理内容：直接置 completed 并返回零计数，不触碰 LLM 与图库
            if task_id:
                await svc.db(svc.relational.update_pipeline_run, task_id, "completed")
            return {"kept": 0, "discarded": 0, "ingested": None}
        # LLM 认知过滤（防污染门禁）：输入为消息原文列表，输出 keep 的陈述句
        statements = await filter_session_facts(svc.llm, [i["content"] for i in items])
        result = None
        if statements:
            # 重要性门禁通过：仅过滤后的陈述句拼成文本走标准三阶段入图；
            # 血缘隔离标记 source_pipeline="session_improve"、source_task="improve"
            result = await ingest_text(svc, "\n".join(statements), dataset,
                                       "session_improve", "improve",
                                       data_name=f"session:{session_id}")
        # 全部丢弃时跳过入图；无论是否入图都标记已同步（原文保留供 recall）
        await mark_synced(svc.cache, svc.settings, session_id)
        if task_id:
            await svc.db(svc.relational.update_pipeline_run, task_id, "completed")
        return {"kept": len(statements), "discarded": len(items) - len(statements),
                "ingested": result}
    except Exception as exc:
        # 失败路径：运行记录置 failed 并携带错误信息，异常原样上抛
        if task_id:
            await svc.db(svc.relational.update_pipeline_run, task_id, "failed", str(exc))
        raise
