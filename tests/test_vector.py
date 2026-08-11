"""LanceDB 向量存储测试：验证懒建表、双表写入、相似度排序、dataset 过滤、按 id 回查与按 dataset 删除。"""

import pytest

from weave.infra.vector import VectorStore


@pytest.fixture
def vector(settings):
    """构建一个指向临时 LanceDB 目录的 VectorStore 实例，测试结束后关闭释放资源。

    参数:
        settings: conftest 提供的测试配置夹具，其 vector_db_path 指向
            tmp_path 下的隔离目录，保证测试之间互不影响。
    返回:
        VectorStore: 尚未建表（懒建表，首次 add 时才创建）、可直接读写的
        向量存储实例。
    """
    v = VectorStore(settings.vector_db_path)  # 在临时目录连接/创建向量库
    yield v
    v.close()  # 释放数据库句柄，避免临时文件被占用


def _chunk(cid: str, vec, text: str, dataset: str = "default") -> dict:
    """构造一个符合 add_chunks 接口约定的文本块字典。

    参数:
        cid: 文本块 ID（chunk_id 主键语义）。
        vec: 嵌入向量，测试用二维向量即可验证距离排序。
        text: 文本块原文。
        dataset: 所属数据集命名空间，默认 "default"。
    返回:
        dict: 含 chunk_id/vector/text/data_id/dataset/created_at 六个键的
        文本块载荷；data_id 固定为 "d1"，created_at 固定为同一时刻。
    """
    return dict(chunk_id=cid, vector=vec, text=text, data_id="d1",
                dataset=dataset, created_at="2026-08-10T00:00:00+00:00")


def test_search_empty_db_returns_empty(vector):
    """空库（表尚未创建）时 search/get 必须返回空列表而不是抛异常。"""
    assert vector.search_chunks([1.0, 0.0], 5) == []  # 表不存在，检索返回 []
    assert vector.get_chunks(["x"]) == []  # 表不存在，回查返回 []


def test_add_and_search_chunks_ranking(vector):
    """写入两条文本块后，相似度检索必须按距离升序返回且命中全部内容。"""
    vector.add_chunks([
        _chunk("c1", [1.0, 0.0], "喜欢咖啡"),
        _chunk("c2", [0.0, 1.0], "喜欢茶"),
    ])
    hits = vector.search_chunks([1.0, 0.0], 2)
    assert hits[0]["chunk_id"] == "c1"  # 同向量距离最小
    assert {h["chunk_id"] for h in hits} == {"c1", "c2"}  # top_k=2 全覆盖
    assert hits[0]["text"] == "喜欢咖啡"  # 载荷字段原样返回


def test_search_with_dataset_filter(vector):
    """带 dataset 过滤的检索只返回该数据集内的结果，其他数据集不可见。"""
    vector.add_chunks([
        _chunk("c1", [1.0, 0.0], "默认库内容"),
        _chunk("c2", [1.0, 0.0], "其他库内容", dataset="other"),
    ])
    hits = vector.search_chunks([1.0, 0.0], 5, dataset="other")
    assert [h["chunk_id"] for h in hits] == ["c2"]  # 只剩 other 数据集的一条


def test_entities_table_and_get_chunks(vector):
    """entities 表独立写入与检索；get_chunks 按 id 批量回查原文。"""
    vector.add_entities([dict(entity_id="e1", vector=[1.0, 0.0], name="用户",
                              entity_type="Person", description="", dataset="default",
                              is_latest=True)])
    hits = vector.search_entities([1.0, 0.0], 1)
    assert hits[0]["entity_id"] == "e1" and hits[0]["name"] == "用户"  # 字段完整返回

    vector.add_chunks([_chunk("c9", [0.5, 0.5], "回查文本")])
    rows = vector.get_chunks(["c9"])  # 无向量的过滤扫描回查
    assert rows[0]["text"] == "回查文本"


def test_delete_dataset(vector):
    """delete_dataset 删除指定数据集的全部行，其他数据集不受影响。"""
    vector.add_chunks([_chunk("c1", [1.0, 0.0], "x"), _chunk("c2", [1.0, 0.0], "y", "other")])
    vector.delete_dataset("default")  # 删掉 default，应只剩 other
    assert [h["chunk_id"] for h in vector.search_chunks([1.0, 0.0], 5)] == ["c2"]
