"""LanceDB 向量存储 (spec §4.3): text_chunks / entities 两张表, 懒建表.

全同步方法, 调用方必须经 run_db() 包装. 两张表均不在启动时创建,
首次 add 时以首批数据推断 schema 建表; 表不存在时 search/get 返回 [].
"""

from pathlib import Path

import lancedb

# 本模块管理的全部表名：delete_dataset 需要逐表清理
_TABLES = ("text_chunks", "entities")


def _esc(value: str) -> str:
    """转义 SQL 字符串字面量中的单引号，防止过滤表达式注入或语法错误。

    参数:
        value: 待拼入 where/delete 表达式的原始字符串值。
    返回:
        str: 将每个单引号替换为两个单引号后的安全值（SQL 标准转义）。
    """
    return value.replace("'", "''")


class VectorStore:
    """LanceDB 向量存储门面：text_chunks / entities 两张表的写入与检索入口。

    所有方法均为同步阻塞调用，禁止在事件循环中直接调用；调用方（Task 9 起）
    必须经 run_db() 包装到工作线程执行。表采用懒建表策略：构造时不建表，
    首次 add 时由首批数据推断 schema 创建，故维度由首次写入的向量决定。
    """

    def __init__(self, path: str):
        """连接（必要时创建）LanceDB 数据库目录；不创建任何表。

        参数:
            path: LanceDB 数据库目录路径；不存在时自动创建（含父目录）。
        返回: 无。
        """
        Path(path).mkdir(parents=True, exist_ok=True)  # 确保数据库目录存在
        self._db = lancedb.connect(path)  # 打开/创建数据库连接（仅建目录，不建表）

    def close(self) -> None:
        """释放数据库连接句柄（可重复调用，幂等）。

        参数: 无。
        返回: 无。
        """
        self._db = None  # lancedb 连接无显式 close，置空交由 GC 释放文件句柄

    # ---------- 内部 ----------
    def _table(self, name: str):
        """按名打开已存在的表；表不存在时返回 None 而不是抛异常。

        参数:
            name: 表名（"text_chunks" 或 "entities"）。
        返回:
            打开的 LanceDB 表对象；表尚未懒创建时返回 None。
        """
        try:
            return self._db.open_table(name)
        except Exception:
            return None  # 表不存在（含从未 add 过）时按空表语义处理

    def _add(self, name: str, rows: list[dict]) -> None:
        """向指定表追加行；表不存在时以首批数据推断 schema 建表（懒建表）。

        参数:
            name: 表名（"text_chunks" 或 "entities"）。
            rows: 行字典列表；键集合即表 schema，vector 列维度由首行决定。
        返回: 无。
        """
        if not rows:
            return  # 空批次直接跳过，避免用空数据建出无 schema 的表
        table = self._table(name)
        if table is None:
            self._db.create_table(name, rows)  # 首次写入：以 rows 推断 schema 建表
        else:
            table.add(rows)  # 后续写入：向既有表追加（schema 须一致）

    # ---------- 写入 ----------
    def add_chunks(self, rows: list[dict]) -> None:
        """写入文本块向量行到 text_chunks 表。

        参数:
            rows: 行字典列表，每行键为 chunk_id/vector/text/data_id/dataset/
                created_at；vector 为等长 float 列表。
        返回: 无。
        """
        self._add("text_chunks", rows)

    def add_entities(self, rows: list[dict]) -> None:
        """写入实体向量行到 entities 表。

        参数:
            rows: 行字典列表，每行键为 entity_id/vector/name/entity_type/
                description/dataset/is_latest；vector 为等长 float 列表。
        返回: 无。
        """
        self._add("entities", rows)

    # ---------- 检索 ----------
    def _search(self, name: str, vector: list[float], top_k: int,
                dataset: str | None = None) -> list[dict]:
        """在指定表中按向量相似度检索，可选按 dataset 过滤。

        参数:
            name: 表名（"text_chunks" 或 "entities"）。
            vector: 查询向量，维度须与表内 vector 列一致。
            top_k: 返回的最大结果数，按距离升序取前 top_k 条。
            dataset: 可选数据集过滤；为 None 时跨全部数据集检索。
        返回:
            list[dict]: 命中行字典列表（含全部载荷列与 _distance），
            按距离升序；表不存在时返回空列表。
        """
        table = self._table(name)
        if table is None:
            return []  # 懒建表语义：从未写入过即空结果
        query = table.search(vector)
        if dataset:
            query = query.where(f"dataset = '{_esc(dataset)}'")  # 单引号已转义
        return query.limit(top_k).to_list()

    def search_chunks(self, vector: list[float], top_k: int,
                      dataset: str | None = None) -> list[dict]:
        """在 text_chunks 表中检索最相似的文本块。

        参数:
            vector: 查询向量，维度须与写入时一致。
            top_k: 返回的最大结果数，按距离升序。
            dataset: 可选数据集过滤；为 None 时跨全部数据集检索。
        返回:
            list[dict]: 命中行字典列表（含 chunk_id/text/_distance 等列）；
            表不存在时返回空列表。
        """
        return self._search("text_chunks", vector, top_k, dataset)

    def search_entities(self, vector: list[float], top_k: int,
                        dataset: str | None = None) -> list[dict]:
        """在 entities 表中检索最相似的实体。

        参数:
            vector: 查询向量，维度须与写入时一致。
            top_k: 返回的最大结果数，按距离升序。
            dataset: 可选数据集过滤；为 None 时跨全部数据集检索。
        返回:
            list[dict]: 命中行字典列表（含 entity_id/name/_distance 等列）；
            表不存在时返回空列表。
        """
        return self._search("entities", vector, top_k, dataset)

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """按 chunk_id 批量回查文本块原文（无向量的过滤扫描）。

        参数:
            chunk_ids: 待回查的 chunk_id 列表；空列表直接返回空结果。
        返回:
            list[dict]: 命中行字典列表（含 text 等全部列），顺序不保证；
            表不存在时返回空列表。
        """
        table = self._table("text_chunks")
        if table is None or not chunk_ids:
            return []  # 空表或空 id 列表均无结果
        # 拼 IN 过滤表达式，id 逐个转义单引号
        id_list = ",".join(f"'{_esc(i)}'" for i in chunk_ids)
        return table.search().where(f"chunk_id IN ({id_list})").to_list()

    # ---------- 删除 ----------
    def delete_dataset(self, dataset: str) -> None:
        """删除指定数据集在两张表中的全部行（dataset 级清理）。

        参数:
            dataset: 待删除的数据集名；不存在的表自动跳过。
        返回: 无。
        """
        for name in _TABLES:
            table = self._table(name)
            if table is not None:
                table.delete(f"dataset = '{_esc(dataset)}'")  # 单引号已转义
