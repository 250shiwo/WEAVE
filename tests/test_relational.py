"""SQLite 关系存储测试：验证 datasets/data/edges/pipeline_runs 五张表的 CRUD、幂等与级联删除行为。"""

import pytest

from weave.infra.relational import RelationalStore


@pytest.fixture
def store(settings):
    """构建一个指向临时 SQLite 文件的 RelationalStore 实例，测试结束后关闭释放资源。

    参数:
        settings: conftest 提供的测试配置夹具，其 relational_db_path 指向
            tmp_path 下的隔离数据库文件，保证测试之间互不影响。
    返回:
        RelationalStore: 已自动建表、可直接读写的存储实例。
    """
    s = RelationalStore(settings.relational_db_path)  # 在临时目录建库建表
    yield s
    s.close()  # 释放连接池，避免临时文件被占用


def test_dataset_and_data_crud(store):
    """验证数据集与数据条目的基本读写：按名幂等建数据集、按哈希去重查数据、状态流转。

    做什么: 覆盖 get_or_create_dataset 的幂等性（同名返回同一 id）、
        create_data/link_dataset_data/get_data_by_hash/get_data/set_data_status/
        list_datasets 的正常读写路径。
    参数: 无（使用 store 夹具）。
    返回: 无；断言各读取接口返回的字段值符合写入预期。
    """
    # 首次调用创建数据集，返回的 id 必须非空
    ds = store.get_or_create_dataset("default")
    assert ds["name"] == "default" and ds["id"]
    # 幂等：同名再次调用不得新建行，应返回同一个 id
    assert store.get_or_create_dataset("default")["id"] == ds["id"]

    # 写入一条数据并关联到数据集；初始状态应为 created
    store.create_data("d1", "note", "用户喜欢咖啡", "hash1", "remember")
    store.link_dataset_data(ds["id"], "d1")
    # 按内容哈希可反查同一条数据（去重依赖此接口）
    assert store.get_data_by_hash("hash1")["id"] == "d1"
    assert store.get_data("d1")["status"] == "created"
    # 状态流转：created -> completed
    store.set_data_status("d1", "completed")
    assert store.get_data("d1")["status"] == "completed"
    # 数据集列表应且仅应包含刚创建的 default
    assert [d["name"] for d in store.list_datasets()] == ["default"]


def test_edge_metadata_supersede(store):
    """验证边元数据的版本替代（supersede）与按来源管线批量删除。

    做什么: 同一 source_id + relationship_type 下，旧边置为 not latest 后写入 v2 新边，
        get_latest_edge 应只返回最新版本；delete_edges_by_source 按来源管线清边并返回删除数。
    参数: 无（使用 store 夹具）。
    返回: 无；断言最新边指针随 supersede 切换、删除计数准确。
    """
    # 构造一条 v1 边的公共字段
    base = dict(source_id="e1", target_id="e2", relationship_type="LIKES",
                dataset="default", version=1, is_latest=True,
                source_pipeline="remember", source_task="remember",
                created_at="2026-08-10T00:00:00+00:00",
                updated_at="2026-08-10T00:00:00+00:00")
    store.upsert_edge(dict(base, edge_id="edge1"))
    # 初始最新边为 edge1
    latest = store.get_latest_edge("e1", "LIKES")
    assert latest["edge_id"] == "edge1" and latest["is_latest"] is True

    # supersede：旧边标记 not latest，再写入指向 e3 的 v2 新边
    store.set_edge_not_latest("edge1", "2026-08-10T01:00:00+00:00")
    store.upsert_edge(dict(base, edge_id="edge2", target_id="e3", version=2))
    # 最新边指针应切换到 edge2
    latest = store.get_latest_edge("e1", "LIKES")
    assert latest["edge_id"] == "edge2" and latest["version"] == 2

    # 按来源管线删除：两条边均来自 remember，应全部删除
    assert store.delete_edges_by_source("remember") == 2
    assert store.get_latest_edge("e1", "LIKES") is None


def test_pipeline_run_lifecycle(store):
    """验证管线运行记录的生命周期：pending -> running -> failed（带错误信息）。

    做什么: 覆盖 create_pipeline_run 的初始状态、update_pipeline_run 的状态/错误更新、
        get_pipeline_run 对存在与不存在 task_id 的两种返回。
    参数: 无（使用 store 夹具）。
    返回: 无；断言状态流转与错误信息持久化正确。
    """
    # 创建运行记录，初始状态固定为 pending
    store.create_pipeline_run("t1", "cognify", "d1")
    assert store.get_pipeline_run("t1")["status"] == "pending"
    # 状态推进：pending -> running -> failed，error 记录失败原因
    store.update_pipeline_run("t1", "running")
    store.update_pipeline_run("t1", "failed", "LLM error")
    run = store.get_pipeline_run("t1")
    assert run["status"] == "failed" and run["error"] == "LLM error"
    # 查询不存在的 task_id 应返回 None 而非抛异常
    assert store.get_pipeline_run("missing") is None


def test_delete_dataset_rows_cascade(store):
    """验证按数据集名级联删除：关联数据行、边行与数据集行一并清除，返回各类删除计数。

    做什么: 先造出一个含 1 条数据、1 条边的 temp 数据集，调用 delete_dataset_rows 后
        确认计数准确、数据行不可再查、数据集列表为空。
    参数: 无（使用 store 夹具）。
    返回: 无；断言 counts 中 data/edges 计数及删除后的不可见性。
    """
    # 准备：1 个数据集 + 1 条数据（已关联）+ 1 条属于该数据集的边
    ds = store.get_or_create_dataset("temp")
    store.create_data("d1", "n", "x", "h1", "remember")
    store.link_dataset_data(ds["id"], "d1")
    store.upsert_edge(dict(edge_id="edge1", source_id="a", target_id="b",
                           relationship_type="R", dataset="temp", version=1,
                           is_latest=True, source_pipeline="remember",
                           source_task="remember", created_at="t", updated_at="t"))
    # 级联删除：返回各类行的删除计数
    counts = store.delete_dataset_rows("temp")
    assert counts["data"] == 1 and counts["edges"] == 1
    # 删除后：数据行不可查，数据集列表为空
    assert store.get_data("d1") is None
    assert store.list_datasets() == []
