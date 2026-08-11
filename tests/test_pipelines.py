"""remember 永久写入管线测试：验证三阶段写、版本更替、内容去重、失败不留残图与数据集统计。

测试通过 conftest 的 make_service/stores 夹具组装 MemoryService 与真实三库
（均指向临时目录），LLM 与嵌入使用 Fake 替身，全程无任何外部服务依赖。
"""

import pytest

from weave.core.models import content_hash, entity_id_for
from tests.fakes import FakeLLM

# 预设抽取结果：用户喜欢浅烘焙咖啡（2 实体 + 1 关系）
LIGHT = {"entities": [{"name": "用户", "entity_type": "Person"}, {"name": "浅烘焙咖啡"}],
         "relationships": [{"source": "用户", "target": "浅烘焙咖啡",
                            "relationship_type": "LIKES"}]}
# 预设抽取结果：同一用户的偏好变更为深烘焙咖啡（用于触发版本更替）
DARK = {"entities": [{"name": "用户", "entity_type": "Person"}, {"name": "深烘焙咖啡"}],
        "relationships": [{"source": "用户", "target": "深烘焙咖啡",
                           "relationship_type": "LIKES"}]}


async def test_remember_permanent_ingests_graph_and_vector(make_service, stores):
    """验证 remember 永久模式把实体/关系写入图库、文本块写入向量库、数据记录标记完成。

    参数:
        make_service: conftest 服务工厂夹具。
        stores: 三库夹具，用于直接断言底层存储状态。
    """
    svc = make_service(llm=FakeLLM([LIGHT]))  # 只预设一次抽取响应：单 chunk 只调一次 LLM
    result = await svc.remember("用户喜欢浅烘焙咖啡")
    # 返回值断言：永久模式 + 2 实体 + 1 关系 + 无版本更替
    assert result["mode"] == "permanent"
    assert result["entities"] == 2 and result["relationships"] == 1 and result["superseded"] == 0

    rel, vec, graph = stores
    # 图库断言：实体按名可查，LIKES 事实的目标实体正确
    assert graph.get_entity_by_name("用户", "default") is not None
    facts = graph.neighbors([entity_id_for("default", "用户")])
    assert facts[0]["target_name"] == "浅烘焙咖啡"
    # 向量库断言：用同文本的确定性向量检索，命中文本块即原文
    qv = (await svc.embedder.embed(["用户喜欢浅烘焙咖啡"]))[0]
    assert vec.search_chunks(qv, 1)[0]["text"] == "用户喜欢浅烘焙咖啡"
    # 关系库断言：数据记录状态流转到 completed
    assert rel.get_data(result["data_id"])["status"] == "completed"


async def test_remember_supersedes_changed_fact(make_service, stores):
    """验证同一实体同一关系类型的事实变更时产生新版本：旧边置非最新，新边 version=2。

    参数:
        make_service: conftest 服务工厂夹具。
        stores: 三库夹具，用于断言图库与关系库中的版本状态。
    """
    svc = make_service(llm=FakeLLM([LIGHT, DARK]))  # 两次 remember 各消费一个响应
    await svc.remember("用户喜欢浅烘焙咖啡")
    result = await svc.remember("用户其实更喜欢深烘焙咖啡")
    assert result["superseded"] == 1  # 恰好更替一条旧事实

    rel, _, graph = stores
    uid = entity_id_for("default", "用户")
    # 图库断言：最新 LIKES 边指向深烘焙咖啡且版本号为 2
    latest = graph.get_latest_relationship(uid, "LIKES")
    assert latest["version"] == 2
    assert latest["target_id"] == entity_id_for("default", "深烘焙咖啡")
    # 关系库断言：最新边元数据与图库一致（version=2 且 is_latest=True）
    sql_latest = rel.get_latest_edge(uid, "LIKES")
    assert sql_latest["version"] == 2 and sql_latest["is_latest"] is True


async def test_remember_deduplicates_same_text(make_service):
    """验证同数据集相同文本重复写入时按内容哈希去重，不再调用 LLM 抽取。

    参数:
        make_service: conftest 服务工厂夹具。
    """
    llm = FakeLLM([LIGHT])  # 队列只有一个响应：若第二次也抽取会因队列空而失败
    svc = make_service(llm=llm)
    await svc.remember("用户喜欢浅烘焙咖啡")
    result = await svc.remember("用户喜欢浅烘焙咖啡")  # 完全相同的文本
    assert result["deduplicated"] is True
    assert len(llm.calls) == 1  # 未重复抽取


async def test_remember_llm_failure_writes_no_partial_graph(make_service, stores):
    """验证 LLM 抽取失败时不写任何图/向量（三阶段写的第一阶段原子性），数据记录标记 failed。

    参数:
        make_service: conftest 服务工厂夹具。
        stores: 三库夹具，用于断言图库无残留、数据状态为 failed。
    """
    class FailingLLM:
        """永远抛连接错误的 LLM 替身，模拟 LLM 服务不可用。"""

        async def complete_json(self, system, user):
            """模拟 LLM 调用失败：无条件抛 ConnectionError。"""
            raise ConnectionError("LLM down")

    svc = make_service(llm=FailingLLM())
    with pytest.raises(ConnectionError):
        await svc.remember("一些文本")
    rel, _, graph = stores
    # 阶段 1 即失败：图库必须零实体（无任何部分写入）
    assert graph.count_entities() == 0
    # 数据记录已建但状态必须流转为 failed，便于后续排查/重试
    assert rel.get_data_by_hash(content_hash("一些文本"))["status"] == "failed"


async def test_list_datasets(make_service):
    """验证 list_datasets 汇总三库统计：数据条目数、边数来自关系库，实体数来自图库。

    参数:
        make_service: conftest 服务工厂夹具。
    """
    svc = make_service(llm=FakeLLM([LIGHT]))
    await svc.remember("用户喜欢浅烘焙咖啡")
    stats = await svc.list_datasets()
    assert stats[0]["name"] == "default"
    assert stats[0]["data_count"] == 1 and stats[0]["edge_count"] == 1
    assert stats[0]["entity_count"] == 2
