"""混合检索自动路由 (spec §5.4): 向量入口 -> 图 1 跳扩展 -> 汇总, 会话缓存叠加.

查询侧主流程：query 向量化后在 LanceDB 的 text_chunks 与 entities 两张表
各取 top_k 作为入口；实体入口经 Kuzu 做 1 跳图扩展（neighbors 仅返回
is_latest 的最新关系边），并反查入口实体 MENTIONS 的文本块回查 LanceDB
原文补充进 chunks；最后按 session_id 叠加 Redis 会话缓存原文。
来源标注约定：图事实标 origin="graph"，文本块标 source="vector"/"graph"，
会话项标 source="session"。
"""

from weave.core.models import RecallResult
from weave.core.session import get_session


async def hybrid_recall(svc, query: str, dataset: str | None = None,
                        top_k: int = 5, session_id: str | None = None) -> dict:
    """混合检索主入口：向量检索定入口 -> 图 1 跳扩展出事实 -> 块原文补充 -> 会话叠加。

    做什么: 严格按路由顺序执行——
        1. query 向量化（非 DB 调用，直接 await embedder）；
        2. 双表入口：text_chunks 取向量命中块、entities 取入口实体，各 top_k；
        3. 图扩展：入口实体经 neighbors 查 1 跳事实（仅 is_latest 边，
           被版本更替取代的旧事实天然不出现）；
        4. 块补充：入口实体 MENTIONS 的块 ID 回查 LanceDB 原文，
           与向量命中块按 chunk_id 去重后合并（图来源块 score=None）；
        5. 会话叠加：session_id 非空时读取 Redis 会话全部原文逐条标注返回。
    参数:
        svc: MemoryService 门面（提供 db()/embedder/cache 与三库引用）。
        query: 查询文本。
        dataset: 可选数据集过滤；为 None 时跨全部数据集检索。
        top_k: 双表各自的向量入口条数上限，默认 5。
        session_id: 可选会话标识；非空时叠加该会话的缓存原文。
    返回:
        dict: RecallResult.model_dump()，键为 facts/chunks/session_items——
            fact 字典键: source/relationship_type/target/version/dataset/
                source_pipeline/origin("graph")；
            chunk 字典键: chunk_id/text/dataset/score/source("vector"|"graph")；
            session 项字典键: content/ts/synced/source("session")。
    """
    # 1. query 向量化：嵌入非 DB 调用，不经 svc.db 包装；取单条查询向量
    query_vector = (await svc.embedder.embed([query]))[0]
    # 2. 双表入口检索（均经 svc.db 统一包装：线程/超时/重试）：
    #    text_chunks 出向量命中块，entities 出图扩展的入口实体
    chunk_hits = await svc.db(svc.vector.search_chunks, query_vector, top_k, dataset)
    entity_hits = await svc.db(svc.vector.search_entities, query_vector, top_k, dataset)

    # 3. 图扩展：入口实体 ID 列表 -> neighbors 查 1 跳事实（仅 is_latest 边）
    entry_ids = [h["entity_id"] for h in entity_hits]
    facts = await svc.db(svc.graph.neighbors, entry_ids, dataset)

    # 图扩展: 入口实体 MENTIONS 的 chunk 回查原文, 补充进结果
    # 反查提及边拿块 ID，再回查 LanceDB 取块原文（mentioned_chunk_ids 空输入短路）
    mentioned_ids = await svc.db(svc.graph.mentioned_chunk_ids, entry_ids, dataset)
    mentioned = await svc.db(svc.vector.get_chunks, mentioned_ids)

    # 4. 块合并：向量命中块在前（带 _distance 作为 score），图来源块在后按 chunk_id 去重
    seen = {h["chunk_id"] for h in chunk_hits}  # 已收录块 ID 集合，防止图来源重复补充
    chunks = [
        dict(chunk_id=h["chunk_id"], text=h["text"], dataset=h["dataset"],
             score=h.get("_distance"), source="vector")  # 向量命中：score 取距离值
        for h in chunk_hits
    ]
    for row in mentioned:
        if row["chunk_id"] not in seen:
            # 图来源补充块：无相似度距离，score 置 None，source 标 "graph"
            chunks.append(dict(chunk_id=row["chunk_id"], text=row["text"],
                               dataset=row["dataset"], score=None, source="graph"))

    # 图查询行 -> 对外 fact 字典：端点改名 source/target，统一标注 origin="graph"
    fact_dicts = [
        dict(source=f["source_name"], relationship_type=f["relationship_type"],
             target=f["target_name"], version=f["version"], dataset=f["dataset"],
             source_pipeline=f["source_pipeline"], origin="graph")
        for f in facts
    ]

    # 5. 会话叠加：仅在传入 session_id 时读取会话缓存原文，逐条标注 source="session"
    session_items = []
    if session_id:
        session_items = [
            dict(content=i["content"], ts=i["ts"], synced=i["synced"], source="session")
            for i in await get_session(svc.cache, session_id)
        ]

    # 经 RecallResult 模型约束返回结构后转 dict（缺省字段自动补空列表）
    return RecallResult(facts=fact_dicts, chunks=chunks,
                        session_items=session_items).model_dump()
