"""Kuzu 图存储 (spec §4.3): Entity/TextChunk 节点 + RELATES_TO/MENTIONS 边.

全同步方法, 调用方必须经 run_db() 包装. Kuzu 不支持关系主键,
边身份用 edge_id 属性 (确定性 uuid5, 见 core.models).
"""

from pathlib import Path

import kuzu

# 建表语句集：两张节点表 + 两张边表；IF NOT EXISTS 保证重复构造 GraphStore 幂等
_SCHEMA = [
    # Entity 节点表：承载实体全字段，id 为主键（entity_id_for 确定性 uuid5 hex）
    "CREATE NODE TABLE IF NOT EXISTS Entity("
    "id STRING, name STRING, norm_name STRING, entity_type STRING, description STRING, "
    "dataset STRING, version INT64, is_latest BOOLEAN, created_at STRING, PRIMARY KEY(id))",
    # TextChunk 节点表：文本切块，id 为主键，data_id 指向来源原始数据
    "CREATE NODE TABLE IF NOT EXISTS TextChunk("
    "id STRING, data_id STRING, dataset STRING, created_at STRING, PRIMARY KEY(id))",
    # RELATES_TO 边表：实体间关系；Kuzu 关系表不支持主键，边身份用 edge_id 属性表达
    "CREATE REL TABLE IF NOT EXISTS RELATES_TO("
    "FROM Entity TO Entity, edge_id STRING, relationship_type STRING, version INT64, "
    "is_latest BOOLEAN, source_pipeline STRING, dataset STRING, created_at STRING)",
    # MENTIONS 边表：文本块提及实体，无语义属性
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM TextChunk TO Entity)",
]


class GraphStore:
    """Kuzu 图存储门面：实体/文本块节点与关系边的全部读写入口。

    所有方法均为同步阻塞调用，禁止在事件循环中直接调用；调用方（Task 9 起）
    必须经 run_db() 包装到工作线程执行。
    """

    def __init__(self, path: str):
        """打开（必要时创建）Kuzu 数据库并确保四张表存在。

        参数:
            path: Kuzu 数据库目录路径；父目录不存在时自动创建。
        返回: 无。
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
        self._db = kuzu.Database(path)  # 打开/创建数据库
        self._conn = kuzu.Connection(self._db)  # 单连接串行执行，全同步
        for stmt in _SCHEMA:
            self._conn.execute(stmt)  # 逐条建表（IF NOT EXISTS，重复执行无副作用）

    def close(self) -> None:
        """关闭连接与数据库句柄，释放底层资源（可重复调用，幂等）。

        参数: 无。
        返回: 无。
        """
        conn, db = self._conn, self._db  # 先取出引用再置空，保证二次调用安全
        self._conn = self._db = None
        for obj in (conn, db):
            close = getattr(obj, "close", None)  # 兼容无 close 方法的未来版本
            if callable(close):
                try:
                    close()  # 尽力关闭，失败不抛异常（清理路径不掩盖业务错误）
                except Exception:
                    pass

    # ---------- 内部 ----------
    def _rows(self, query: str, params: dict | None = None) -> list[dict]:
        """执行查询并把结果集完整物化为 dict 列表。

        参数:
            query: Cypher 查询语句，字符串值一律用 $param 占位。
            params: 查询参数；键集合必须与语句中的 $param 完全一致
                （kuzu 0.11 对多余参数键会报错）。
        返回:
            list[dict]: 每行为 {列名: 值}；无结果时返回空列表。
        """
        result = self._conn.execute(query, params or {})
        cols = result.get_column_names()  # 列名顺序与 get_next() 值顺序一致
        rows = []
        while result.has_next():
            rows.append(dict(zip(cols, result.get_next())))  # 逐行打包为 dict
        return rows

    @staticmethod
    def _id_list(ids: list[str]) -> str:
        """把 ID 列表拼接为 Cypher IN 列表字面量片段。

        参数:
            ids: 实体/块 ID 列表，均为 uuid hex（字符集安全，无注入风险）。
        返回:
            str: 形如 "'a1b2','c3d4'" 的逗号分隔片段，直接内联进查询。
        """
        return ",".join(f"'{i}'" for i in ids)  # uuid hex, 字符集安全

    # ---------- 实体 ----------
    def upsert_entity(self, e: dict) -> None:
        """按 id 幂等写入实体节点；已存在则跳过（实体由确定性 ID 天然合并）。

        参数:
            e: 实体载荷，键为 id/name/norm_name/entity_type/description/dataset/
                version/is_latest/created_at；键集合与建参占位符一一对应。
        返回: 无。
        """
        if self.get_entity_by_id(e["id"]):
            return  # 同 id 已存在：幂等跳过，不产生重复节点
        self._conn.execute(
            "CREATE (n:Entity {id: $id, name: $name, norm_name: $norm_name, "
            "entity_type: $entity_type, description: $description, dataset: $dataset, "
            "version: $version, is_latest: $is_latest, created_at: $created_at})",
            dict(e),  # 九个键与查询的九个 $param 精确对应
        )

    def get_entity_by_id(self, entity_id: str) -> dict | None:
        """按主键 id 查询实体节点。

        参数:
            entity_id: 实体 ID（uuid hex）。
        返回:
            dict | None: 命中时返回节点属性 dict（kuzu 会附带 _id/_label 内部键），
            未命中返回 None。
        """
        rows = self._rows("MATCH (n:Entity {id: $id}) RETURN n", {"id": entity_id})
        return rows[0]["n"] if rows else None

    def get_entity_by_name(self, norm_name: str, dataset: str) -> dict | None:
        """按 (规范化名, 数据集) 查询实体节点。

        参数:
            norm_name: 规范化实体名（调用方需先做 norm_name 处理）。
            dataset: 数据集命名空间；同名实体跨数据集隔离。
        返回:
            dict | None: 命中返回节点属性 dict，未命中返回 None。
        """
        rows = self._rows(
            "MATCH (n:Entity) WHERE n.norm_name = $nn AND n.dataset = $ds RETURN n",
            {"nn": norm_name, "ds": dataset},
        )
        return rows[0]["n"] if rows else None

    def count_entities(self, dataset: str | None = None) -> int:
        """统计实体节点数量，可按数据集过滤。

        参数:
            dataset: 为 None 时统计全库实体，否则只统计该数据集。
        返回:
            int: 实体节点条数。
        """
        if dataset:
            rows = self._rows(
                "MATCH (n:Entity) WHERE n.dataset = $ds RETURN count(n) AS n", {"ds": dataset})
        else:
            rows = self._rows("MATCH (n:Entity) RETURN count(n) AS n")
        return rows[0]["n"]  # count 聚合恒返回一行

    # ---------- chunk 与 MENTIONS ----------
    def create_chunk(self, chunk_id: str, data_id: str, dataset: str, created_at: str) -> None:
        """创建 TextChunk 文本块节点。

        参数:
            chunk_id: 块 ID（主键，调用方指定）。
            data_id: 来源原始数据 ID（对应 relational.data.id）。
            dataset: 所属数据集命名空间。
            created_at: 创建时间（ISO8601 UTC 字符串）。
        返回: 无。
        """
        self._conn.execute(
            "CREATE (c:TextChunk {id: $id, data_id: $data_id, dataset: $dataset, created_at: $ca})",
            {"id": chunk_id, "data_id": data_id, "dataset": dataset, "ca": created_at},
        )

    def link_mentions(self, chunk_id: str, entity_ids: list[str]) -> None:
        """为文本块与一批实体建立 MENTIONS 提及边（逐条 CREATE）。

        参数:
            chunk_id: 已存在的 TextChunk 节点 ID。
            entity_ids: 被提及的实体 ID 列表；空列表时不产生任何写操作。
        返回: 无。
        """
        for eid in entity_ids:
            self._conn.execute(
                "MATCH (c:TextChunk {id: $cid}), (e:Entity {id: $eid}) CREATE (c)-[:MENTIONS]->(e)",
                {"cid": chunk_id, "eid": eid},
            )

    def mentioned_chunk_ids(self, entity_ids: list[str], dataset: str | None = None) -> list[str]:
        """反查提及了任一给定实体的文本块 ID 列表（去重）。

        参数:
            entity_ids: 实体 ID 列表；为空时直接返回空列表，不发起查询。
            dataset: 为 None 时不限数据集，否则只统计该数据集下的块。
        返回:
            list[str]: 命中的块 ID 列表（DISTINCT 去重，顺序不保证）。
        """
        if not entity_ids:
            return []  # 空输入短路，避免拼出非法的 IN [] 查询
        q = (f"MATCH (c:TextChunk)-[:MENTIONS]->(e:Entity) "
             f"WHERE e.id IN [{self._id_list(entity_ids)}] ")  # ID 为 uuid hex，内联安全
        params = {}
        if dataset:
            q += "AND c.dataset = $ds "  # 追加数据集过滤条件
            params["ds"] = dataset
        q += "RETURN DISTINCT c.id AS chunk_id"
        return [r["chunk_id"] for r in self._rows(q, params)]

    # ---------- 关系 ----------
    def get_latest_relationship(self, source_id: str, rel_type: str) -> dict | None:
        """查询某源实体指定类型的当前最新关系边。

        参数:
            source_id: 源实体 ID。
            rel_type: 关系类型，如 "LIKES"。
        返回:
            dict | None: 命中返回 {edge_id, target_id, version}，未命中返回 None。
        """
        rows = self._rows(
            "MATCH (a:Entity {id: $sid})-[r:RELATES_TO]->(b:Entity) "
            "WHERE r.relationship_type = $rt AND r.is_latest = true "  # 只取最新版本边
            "RETURN r.edge_id AS edge_id, b.id AS target_id, r.version AS version",
            {"sid": source_id, "rt": rel_type},
        )
        return rows[0] if rows else None

    def set_relationship_not_latest(self, edge_id: str) -> None:
        """把指定边标记为非最新版本（版本更替的第一步，历史边保留可查）。

        参数:
            edge_id: 边身份 ID（edge_id_for 生成的确定性 uuid5 hex）。
        返回: 无。
        """
        self._conn.execute(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.edge_id = $eid SET r.is_latest = false",
            {"eid": edge_id},
        )

    def upsert_relationship(self, e: dict) -> None:
        """按 edge_id 幂等写入关系边：已存在则更新版本属性，不存在则新建。

        参数:
            e: 关系载荷，键为 edge_id/source_id/target_id/relationship_type/
                version/is_latest/source_pipeline/dataset/created_at。
        返回: 无。
        """
        # 先按 edge_id 查重：Kuzu 关系表无主键，幂等只能靠属性查询实现
        exists = self._rows(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.edge_id = $eid RETURN count(r) AS n",
            {"eid": e["edge_id"]},
        )[0]["n"]
        if exists:
            # 已存在：只更新版本相关属性，两端实体与关系类型不变
            self._conn.execute(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.edge_id = $eid "
                "SET r.version = $version, r.is_latest = $is_latest, r.source_pipeline = $sp",
                {"eid": e["edge_id"], "version": e["version"],
                 "is_latest": e["is_latest"], "sp": e["source_pipeline"]},
            )
            return
        # 不存在：匹配两端实体后创建新边（端点缺失时 MATCH 为空、不产生边）
        self._conn.execute(
            "MATCH (a:Entity {id: $src}), (b:Entity {id: $dst}) "
            "CREATE (a)-[:RELATES_TO {edge_id: $eid, relationship_type: $rt, version: $version, "
            "is_latest: $is_latest, source_pipeline: $sp, dataset: $ds, created_at: $ca}]->(b)",
            {"src": e["source_id"], "dst": e["target_id"], "eid": e["edge_id"],
             "rt": e["relationship_type"], "version": e["version"], "is_latest": e["is_latest"],
             "sp": e["source_pipeline"], "ds": e["dataset"], "ca": e["created_at"]},
        )

    def neighbors(self, entity_ids: list[str], dataset: str | None = None) -> list[dict]:
        """查询一批实体的 1 跳邻居事实（仅 is_latest 的出/入边，双向匹配）。

        参数:
            entity_ids: 中心实体 ID 列表；为空时直接返回空列表。
            dataset: 为 None 时不限数据集，否则只返回该数据集下的边。
        返回:
            list[dict]: 每条事实含 edge_id/source_id/source_name/relationship_type/
            target_id/target_name/version/dataset/source_pipeline 九个键。
        """
        if not entity_ids:
            return []  # 空输入短路，避免拼出非法的 IN [] 查询
        ids = self._id_list(entity_ids)  # uuid hex 内联，字符集安全
        q = ("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
             f"WHERE r.is_latest = true AND (a.id IN [{ids}] OR b.id IN [{ids}]) ")  # 双向 1 跳
        params = {}
        if dataset:
            q += "AND r.dataset = $ds "  # 追加边级数据集过滤
            params["ds"] = dataset
        q += ("RETURN r.edge_id AS edge_id, a.id AS source_id, a.name AS source_name, "
              "r.relationship_type AS relationship_type, b.id AS target_id, "
              "b.name AS target_name, r.version AS version, r.dataset AS dataset, "
              "r.source_pipeline AS source_pipeline")
        return self._rows(q, params)

    # ---------- 清理 ----------
    def delete_dataset(self, dataset: str) -> None:
        """删除指定数据集下的全部文本块与实体节点（DETACH 级联删边）。

        参数:
            dataset: 待清空的数据集命名空间。
        返回: 无；先删 TextChunk 再删 Entity，DETACH DELETE 会连带移除
            MENTIONS/RELATES_TO 边，不残留悬空关系。
        """
        self._conn.execute("MATCH (c:TextChunk {dataset: $ds}) DETACH DELETE c", {"ds": dataset})
        self._conn.execute("MATCH (e:Entity {dataset: $ds}) DETACH DELETE e", {"ds": dataset})

    def delete_relationships_by_source(self, source_pipeline: str) -> int:
        """删除指定来源管线产生的全部关系边，并返回删除条数。

        参数:
            source_pipeline: 来源管线名（如 "session_improve"）。
        返回:
            int: 实际删除的边条数；无匹配时为 0 且不发起 DELETE。
        """
        # 先统计待删条数作为返回值（DELETE 语句本身不返回条数）
        n = self._rows(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.source_pipeline = $sp RETURN count(r) AS n",
            {"sp": source_pipeline},
        )[0]["n"]
        if n:
            self._conn.execute(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.source_pipeline = $sp DELETE r",
                {"sp": source_pipeline},
            )
        return n
