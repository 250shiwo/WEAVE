"""Kuzu 图存储测试：验证实体/关系/文本块节点的读写、版本更替、1 跳检索与清理行为。"""

import pytest

from weave.core.models import edge_id_for, entity_id_for, utcnow
from weave.infra.graph import GraphStore


@pytest.fixture
def graph(settings):
    """构建一个指向临时 Kuzu 数据库目录的 GraphStore 实例，测试结束后关闭释放资源。

    参数:
        settings: conftest 提供的测试配置夹具，其 graph_db_path 指向
            tmp_path 下的隔离目录，保证测试之间互不影响。
    返回:
        GraphStore: 已自动建表（Entity/TextChunk 节点表与 RELATES_TO/MENTIONS 边表）、
        可直接读写的图存储实例。
    """
    g = GraphStore(settings.graph_db_path)  # 在临时目录建库建表
    yield g
    g.close()  # 释放连接与数据库句柄，避免临时文件被占用


def _entity(name: str, dataset: str = "default") -> dict:
    """构造一个符合 upsert_entity 接口约定的实体字典。

    参数:
        name: 实体原始名；id 用 entity_id_for(dataset, name) 确定性生成，
            norm_name 取去空白转小写形式。
        dataset: 所属数据集命名空间，默认 "default"。
    返回:
        dict: 含 id/name/norm_name/entity_type/description/dataset/version/
        is_latest/created_at 九个键的实体载荷，version 固定为 1。
    """
    return dict(id=entity_id_for(dataset, name), name=name, norm_name=name.strip().lower(),
                entity_type="Thing", description="", dataset=dataset, version=1,
                is_latest=True, created_at=utcnow())


def _rel(src: str, rt: str, dst: str, version: int = 1, dataset: str = "default",
         sp: str = "remember") -> dict:
    """构造一个符合 upsert_relationship 接口约定的关系字典。

    参数:
        src: 源实体 ID。
        rt: 关系类型，如 "LIKES"。
        dst: 目标实体 ID。
        version: 关系版本号，默认 1。
        dataset: 边所属数据集，默认 "default"。
        sp: 来源管线名，默认 "remember"。
    返回:
        dict: 含 edge_id/source_id/target_id/relationship_type/version/is_latest/
        source_pipeline/dataset/created_at 九个键的关系载荷；edge_id 由
        edge_id_for(src, rt, dst) 确定性生成。
    """
    return dict(edge_id=edge_id_for(src, rt, dst), source_id=src, target_id=dst,
                relationship_type=rt, version=version, is_latest=True,
                source_pipeline=sp, dataset=dataset, created_at=utcnow())


def test_entity_upsert_idempotent_and_get_by_name(graph):
    """验证实体写入幂等、按规范化名查询与计数。

    做什么: 同一实体重复 upsert 不得产生重复节点；get_entity_by_name 能按
        (norm_name, dataset) 取回实体且未知名返回 None；count_entities 计数正确。
    参数: 无（使用 graph 夹具）。
    返回: 无；断言查询字段值与计数符合预期。
    """
    graph.upsert_entity(_entity("用户"))
    graph.upsert_entity(_entity("用户"))  # 幂等：重复写入应被跳过
    e = graph.get_entity_by_name("用户", "default")
    # 取回的实体字段必须与写入一致
    assert e["name"] == "用户" and e["entity_type"] == "Thing"
    # 查询不存在的实体名应返回 None
    assert graph.get_entity_by_name("不存在", "default") is None
    # 幂等写入后该数据集只有 1 个实体节点
    assert graph.count_entities("default") == 1


def test_relationship_supersession_flow(graph):
    """验证关系的版本更替流程：旧边置 is_latest=False 后新边成为最新版本。

    做什么: 先建立 version=1 的 LIKES 边并确认其为最新；再将其置为非最新、
        写入指向另一目标的 version=2 新边；get_latest_relationship 应返回新边。
    参数: 无（使用 graph 夹具）。
    返回: 无；断言更替前后最新边的 target_id 与 version 变化符合预期。
    """
    u, light, dark = _entity("用户"), _entity("浅烘焙"), _entity("深烘焙")
    for e in (u, light, dark):
        graph.upsert_entity(e)
    # 写入 version=1 的初始边：用户 -LIKES-> 浅烘焙
    graph.upsert_relationship(_rel(u["id"], "LIKES", light["id"]))
    latest = graph.get_latest_relationship(u["id"], "LIKES")
    # 初始最新边应指向浅烘焙且版本为 1
    assert latest["target_id"] == light["id"] and latest["version"] == 1

    # 版本更替: 旧边 is_latest=False, 新边 version=2
    graph.set_relationship_not_latest(latest["edge_id"])
    graph.upsert_relationship(_rel(u["id"], "LIKES", dark["id"], version=2))
    latest = graph.get_latest_relationship(u["id"], "LIKES")
    # 更替后最新边应指向深烘焙且版本为 2
    assert latest["target_id"] == dark["id"] and latest["version"] == 2


def test_neighbors_one_hop_with_dataset_filter(graph):
    """验证 1 跳邻居检索：只返回 is_latest 边，且支持按 dataset 过滤。

    做什么: 用户在 default 数据集 LIKES 咖啡、在 other 数据集 LIKES 茶；
        不带过滤应返回两条事实，带 dataset="default" 只返回咖啡那条。
    参数: 无（使用 graph 夹具）。
    返回: 无；断言两种查询返回的事实条数与目标实体名符合预期。
    """
    u, coffee, tea = _entity("用户"), _entity("咖啡"), _entity("茶", "other")
    for e in (u, coffee, tea):
        graph.upsert_entity(e)
    # 同一对用户实体在不同 dataset 下各写一条 LIKES 边
    graph.upsert_relationship(_rel(u["id"], "LIKES", coffee["id"]))
    graph.upsert_relationship(_rel(u["id"], "LIKES", tea["id"], dataset="other"))
    # 不过滤数据集时应拿到全部 2 条最新边
    all_facts = graph.neighbors([u["id"]])
    assert len(all_facts) == 2
    # 过滤 default 数据集时只剩咖啡那条边
    default_facts = graph.neighbors([u["id"]], dataset="default")
    assert len(default_facts) == 1 and default_facts[0]["target_name"] == "咖啡"


def test_mentions_and_chunk_lookup(graph):
    """验证文本块节点创建、MENTIONS 边建立与按实体反查块 ID。

    做什么: 创建 TextChunk 节点 c1 并建立其到实体咖啡的 MENTIONS 边；
        mentioned_chunk_ids 按实体 ID 应反查出 ["c1"]。
    参数: 无（使用 graph 夹具）。
    返回: 无；断言反查结果恰好为写入的块 ID 列表。
    """
    e = _entity("咖啡")
    graph.upsert_entity(e)
    graph.create_chunk("c1", "d1", "default", utcnow())
    graph.link_mentions("c1", [e["id"]])
    # 按实体反查提及它的文本块，应得到 c1
    assert graph.mentioned_chunk_ids([e["id"]]) == ["c1"]


def test_delete_dataset_detaches(graph):
    """验证按数据集清理：实体与其关系一并删除（DETACH DELETE）。

    做什么: 在 temp 数据集写入两个实体和一条关系后 delete_dataset("temp")；
        该数据集实体计数应归零，邻居查询应返回空列表。
    参数: 无（使用 graph 夹具）。
    返回: 无；断言清理后的计数与邻居查询结果为空。
    """
    a, b = _entity("甲", "temp"), _entity("乙", "temp")
    graph.upsert_entity(a)
    graph.upsert_entity(b)
    graph.upsert_relationship(_rel(a["id"], "R", b["id"], dataset="temp"))
    # 删除整个 temp 数据集：节点与关联边都应被移除
    graph.delete_dataset("temp")
    assert graph.count_entities("temp") == 0
    # 节点已删，任何邻居查询都应返回空
    assert graph.neighbors([a["id"]]) == []


def test_delete_relationships_by_source(graph):
    """验证按来源管线清理关系：只删指定 source_pipeline 的边并返回删除条数。

    做什么: 写入两条不同来源的边（session_improve 与 remember）；
        按 session_improve 删除应返回 1，剩余邻居查询只剩 remember 的 R2 边。
    参数: 无（使用 graph 夹具）。
    返回: 无；断言删除条数与剩余关系的类型符合预期。
    """
    a, b = _entity("甲"), _entity("乙")
    graph.upsert_entity(a)
    graph.upsert_entity(b)
    # 两条边：R1 来自 session_improve，R2 来自 remember
    graph.upsert_relationship(_rel(a["id"], "R1", b["id"], sp="session_improve"))
    graph.upsert_relationship(_rel(b["id"], "R2", a["id"], sp="remember"))
    # 只应删掉 R1 这一条，返回删除条数 1
    assert graph.delete_relationships_by_source("session_improve") == 1
    # 剩余的邻居事实只剩 R2（remember 来源的边不受影响）
    facts = graph.neighbors([a["id"]])
    assert len(facts) == 1 and facts[0]["relationship_type"] == "R2"
