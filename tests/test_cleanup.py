"""清理与遗忘测试 (Task 14) — spec §5/§6: forget 级联清三库、forget_by_source 只删关系事实。

验证三件事：
1. forget(dataset=...) 级联清空 SQLite/LanceDB/Kuzu 三库中该数据集的全部内容；
2. forget() 全清必须显式 confirm=True，否则抛 ValueError（防误操作）；
3. forget_by_source 只删除指定来源的关系事实（含历史版本），实体节点因可能
   被多来源共享而保留（spec §6③ 会话衍生事实的可清理性）。
"""

import pytest

from weave.core.models import entity_id_for
from tests.fakes import FakeLLM
from tests.test_improve import EXTRACT_OUT, IMPROVE_OUT
from tests.test_pipelines import LIGHT


async def test_forget_dataset_cascades(make_service, stores, fake_cache):
    """forget(dataset="default") 级联清空三库：图库零实体、关系库零数据集、向量库零命中。

    参数:
        make_service: conftest 服务工厂夹具（注入 fake_cache）。
        stores: 三库夹具，用于直接断言各库底层状态已清空。
        fake_cache: fakeredis 缓存夹具。
    """
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")  # 写入 2 实体 + 1 关系 + 1 文本块
    result = await svc.forget(dataset="default")
    # 返回 scope 为数据集名，edges 为关系库删除的边行数（1 条 LIKES）
    assert result["scope"] == "default" and result["edges"] == 1

    rel, vec, graph = stores
    assert graph.count_entities() == 0  # 图库：实体节点全部删除
    assert rel.list_datasets() == []  # 关系库：数据集行（含边元数据）全部删除
    # 向量库：删除后用任意向量检索 text_chunks 表应无任何命中
    qv = (await svc.embedder.embed(["咖啡"]))[0]
    assert vec.search_chunks(qv, 5) == []


async def test_forget_all_requires_confirm(make_service, fake_cache):
    """forget() 全清保护：缺 confirm 抛 ValueError；confirm=True 后清空全部数据集。

    参数:
        make_service: conftest 服务工厂夹具（注入 fake_cache）。
        fake_cache: fakeredis 缓存夹具。
    """
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")
    # 不传 dataset 且未 confirm：必须拒绝执行，错误消息含 "confirm"
    with pytest.raises(ValueError, match="confirm"):
        await svc.forget()
    result = await svc.forget(confirm=True)
    assert result["scope"] == "all"  # 全清分支的 scope 固定为 "all"
    assert await svc.list_datasets() == []  # 全部数据集已级联清空


async def test_forget_by_source_removes_only_session_facts(make_service, stores, fake_cache):
    """forget_by_source 只删 session_improve 关系事实：remember 事实与共享实体保留。

    参数:
        make_service: conftest 服务工厂夹具（注入 fake_cache）。
        stores: 三库夹具，用于断言图库/关系库中事实与实体的保留状态。
        fake_cache: fakeredis 缓存夹具（会话分支与 improve 必需）。
    """
    # 三次 LLM 调用：remember 抽取 -> improve 过滤 -> 保留陈述抽取
    llm = FakeLLM([LIGHT, IMPROVE_OUT, EXTRACT_OUT])
    svc = make_service(llm=llm, cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")                    # source_pipeline=remember
    await svc.remember("你回答简洁一点", session_id="chat-1")
    await svc.run_improve("chat-1")                             # source_pipeline=session_improve

    result = await svc.forget_by_source("session_improve")
    assert result["edges_graph"] == 1  # 图库只删 1 条 PREFERS 边（血缘 session_improve）

    uid = entity_id_for("default", "用户")
    rel, _, graph = stores
    facts = graph.neighbors([uid])
    assert [f["relationship_type"] for f in facts] == ["LIKES"]  # remember 事实保留
    assert graph.get_entity_by_name("简洁回答", "default") is not None  # 实体共享保留
    assert rel.get_latest_edge(uid, "PREFERS") is None  # SQLite 边元数据（含历史版本）已删
