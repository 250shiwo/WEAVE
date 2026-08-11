"""遗忘管线 (spec §5/§6): forget 级联清三库; forget_by_source 只删关系事实.

删除侧的两个核心流程：
1. forget_dataset——按数据集名级联清空 SQLite（关联行/边元数据/孤立 data 行/
   数据集行）、Kuzu（TextChunk 与 Entity 节点，DETACH 连带删边）、LanceDB
   （text_chunks/entities 两表向量行）；
2. forget_by_source——按来源管线只删除关系事实（SQLite edges 含全部历史版本 +
   Kuzu RELATES_TO 边），实体节点可能被多来源共享，一律保留（spec §6③，
   保证会话衍生事实可单独清理而不误伤共享实体）。
"""


async def forget_dataset(svc, dataset: str) -> dict:
    """按数据集名级联清空三库：SQLite 行 -> Kuzu 节点/边 -> LanceDB 向量行。

    做什么: 依次调用三个存储的 dataset 级删除接口——
        1. 关系库 delete_dataset_rows：删 dataset_data 关联、edges 边元数据、
           孤立 data 行（仍被其他数据集引用的共享 data 保留）与数据集行本身，
           返回 {"data": ..., "edges": ...} 删除计数；
        2. 图库 delete_dataset：DETACH DELETE 该数据集下全部 TextChunk 与
           Entity 节点，连带移除 MENTIONS/RELATES_TO 边，不留悬空关系；
        3. 向量库 delete_dataset：删 text_chunks/entities 两张表中该数据集的
           全部向量行。
        三个删除均经 svc.db 统一包装（单线程执行器 + 超时 + 瞬时重试），
        任一失败即向上抛出，不做静默吞错。
    参数:
        svc: MemoryService 门面（提供 db() 与 relational/graph/vector 三库引用）。
        dataset: 待清空的数据集名。
    返回:
        dict: {"scope": 数据集名, "data": 删除的数据行数, "edges": 删除的边行数}。
    """
    # 1. 关系库级联删除，拿回 data/edges 删除计数作为返回值主体
    counts = await svc.db(svc.relational.delete_dataset_rows, dataset)
    # 2. 图库删除该数据集全部节点（DETACH 连带删边），无计数返回
    await svc.db(svc.graph.delete_dataset, dataset)
    # 3. 向量库删除该数据集在两张表中的全部行，无计数返回
    await svc.db(svc.vector.delete_dataset, dataset)
    # scope 标记清理范围（数据集名），合并关系库返回的 data/edges 计数
    return {"scope": dataset, **counts}


async def forget_by_source(svc, source_pipeline: str) -> dict:
    """按来源清理关系事实 (如 session_improve); 实体可能被多来源共享, 保留.

    做什么: 只删除指定来源管线产生的关系事实，分两侧执行——
        1. SQLite edges 表元数据：delete_edges_by_source 按 source_pipeline
           批量删除（含全部历史版本，不只 is_latest），返回删除行数；
        2. Kuzu RELATES_TO 边：delete_relationships_by_source 按
           source_pipeline 删除全部匹配边，返回删除条数。
        实体节点与文本块一律不动：同一实体（如“用户”）可能被 remember 与
        session_improve 等多个来源共享，按来源清理时误删实体会破坏其他
        来源的事实（spec §6③）。
    参数:
        svc: MemoryService 门面（提供 db() 与 relational/graph 引用）。
        source_pipeline: 来源管线名（如 "remember" / "session_improve" /
            "cognify"）。
    返回:
        dict: {"scope": "source:<来源管线名>",
               "edges_sql": SQLite 删除的边行数,
               "edges_graph": 图库删除的边条数}。
    """
    # 1. 关系库：删除该来源的全部边元数据（含历史版本），拿删除行数
    edges_sql = await svc.db(svc.relational.delete_edges_by_source, source_pipeline)
    # 2. 图库：删除该来源的全部 RELATES_TO 边，拿删除条数
    edges_graph = await svc.db(svc.graph.delete_relationships_by_source, source_pipeline)
    # scope 以 "source:" 前缀标记按来源清理，与 dataset 级清理的 scope 区分
    return {"scope": f"source:{source_pipeline}",
            "edges_sql": edges_sql, "edges_graph": edges_graph}
