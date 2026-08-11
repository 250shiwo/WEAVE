"""最终评审修复回归测试 (C1/I1/I2/I3)：合并前必须修复项的红绿证据。

C1: 跨 dataset 同内容写入后，去重判定限定在数据集内（联表查询），
    第三次 remember 不得抛 MultipleResultsFound；
I1: 去重短路仅对 completed 记录生效——失败的 remember 可重试入图，
    失败的 cognify 重新提交返回 pending 并可重跑至 completed；
I2: improve 按条目 id 打标（Redis SET），处理窗口内新 append 的消息
    永不被误标 synced，下一轮 improve 仍会处理它；
I3: 单次 ingest 内同 (source, relationship_type) 的冲突事实在批内
    互相更替，保证同 (主体, 类型) 至多一条 is_latest=True 的最新边。
"""

import base64

from weave.core.models import entity_id_for
from weave.core.session import append_session
from weave.infra.cache import QUEUE_COGNIFY
from tests.fakes import FakeLLM

# 预设抽取结果：用户喜欢浅烘焙咖啡（2 实体 + 1 关系）
LIGHT = {"entities": [{"name": "用户", "entity_type": "Person"}, {"name": "浅烘焙咖啡"}],
         "relationships": [{"source": "用户", "target": "浅烘焙咖啡",
                            "relationship_type": "LIKES"}]}
# 预设抽取结果：空子图（跨数据集写同内容时消费，避免与 default 数据集的实体纠缠）
EMPTY = {"entities": [], "relationships": []}
# 预设抽取结果：单 chunk 内同 source 同 type 不同 target 的两条冲突关系（I3 批内更替）
CONFLICT = {"entities": [{"name": "用户", "entity_type": "Person"}],
            "relationships": [{"source": "用户", "target": "浅烘焙咖啡",
                               "relationship_type": "LIKES"},
                              {"source": "用户", "target": "深烘焙咖啡",
                               "relationship_type": "LIKES"}]}
# cognify 测试用文档与其 base64 编码
DOC = "Weave 是知识图谱记忆平台。它使用 Kuzu 作为图数据库。"
DOC_B64 = base64.b64encode(DOC.encode()).decode()
DOC_OUT = {"entities": [{"name": "Weave", "entity_type": "Project"},
                        {"name": "Kuzu", "entity_type": "Concept"}],
           "relationships": [{"source": "Weave", "target": "Kuzu",
                              "relationship_type": "USES"}]}


class FailingLLM:
    """永远抛连接错误的 LLM 替身，模拟 LLM 服务不可用（I1 失败前置）。"""

    async def complete_json(self, system, user):
        """模拟 LLM 调用失败：无条件抛 ConnectionError。

        参数:
            system: 系统提示词（忽略）。
            user: 用户消息（忽略）。
        异常:
            ConnectionError: 恒定抛出。
        """
        raise ConnectionError("LLM down")


class MidAppendLLM:
    """竞态复现替身：首次过滤调用期间向同一会话追加一条新消息（I2 用）。

    improve 的处理窗口是“读出未同步集合 -> LLM 过滤 -> 打标 synced”；
    在 LLM 调用内追加消息，正好落在窗口中间（已读出、尚未打标），
    与生产环境 improve 运行期间 remember 会话分支写入新消息等价。
    """

    def __init__(self, cache, settings, session_id: str):
        """初始化替身：持有缓存/配置/目标会话，准备两次全丢弃的过滤响应。

        参数:
            cache: fakeredis 缓存夹具（直接在其上 append 新消息）。
            settings: 测试配置（append_session 需要 TTL/max_items 换算）。
            session_id: 竞态写入的目标会话。
        """
        self._cache = cache
        self._settings = settings
        self._session_id = session_id
        self.calls = 0  # 记录过滤调用次数，供断言“第二轮确实处理了 m2”

    async def complete_json(self, system, user) -> dict:
        """模拟过滤调用：首次调用先窗口内追加 m2 再返回全丢弃；二次调用直接全丢弃。

        参数:
            system: 系统提示词（忽略）。
            user: 用户消息（忽略）。
        返回:
            dict: 全丢弃的过滤结果（keep=False，不触发入图）。
        """
        self.calls += 1
        if self.calls == 1:
            # 竞态窗口：improve 已读出未同步集合、尚未打标时，新消息进入会话
            await append_session(self._cache, self._settings, self._session_id, "临时的想法")
            return {"facts": [{"keep": False, "statement": "闲聊", "reason": "丢弃"}]}
        return {"facts": [{"keep": False, "statement": "临时想法", "reason": "丢弃"}]}


async def test_c1_cross_dataset_same_content_no_crash(make_service, stores):
    """C1 回归：同内容写 dataset A -> 写 dataset B -> 再写 A，全程无异常且命中去重。

    参数:
        make_service: conftest 服务工厂夹具。
        stores: 三库夹具，用于断言 A/B 两数据集各自只持有一条数据。
    断言:
        第三次 remember 不抛 MultipleResultsFound，返回 deduplicated=True
        （去重限定在数据集内，联表查询唯一命中 A 的记录）；同内容在
        data 表按 A/B 各存一行，跨库去重互不干扰。
    """
    svc = make_service(llm=FakeLLM([LIGHT, EMPTY]))  # A 正常抽取；B 消费空子图响应
    await svc.remember("同一段记忆文本", dataset="A")
    await svc.remember("同一段记忆文本", dataset="B")  # 同内容跨库：B 必须全量入图
    third = await svc.remember("同一段记忆文本", dataset="A")  # 修复前此处必抛 500
    assert third["deduplicated"] is True and third["relationships"] == 0

    rel, _, _ = stores
    # 同内容按数据集维度各存一行：A/B 互不串库、互不重复建行
    stats = {s["name"]: s for s in rel.dataset_stats()}
    assert stats["A"]["data_count"] == 1 and stats["B"]["data_count"] == 1


async def test_i1_remember_retry_after_llm_failure(make_service, stores):
    """I1 回归（remember 侧）：LLM 失败置 failed 后，换正常 LLM 重试同内容实际入图。

    参数:
        make_service: conftest 服务工厂夹具（两次组装共享同一套三库）。
        stores: 三库夹具，用于断言数据状态与图库实体。
    断言:
        重试返回 deduplicated=False 且实体/关系计数非零；数据记录最终
        completed；图库可查到抽取实体（修复前重试被去重短路，永不入图）。
    """
    svc = make_service(llm=FailingLLM())
    try:
        await svc.remember("用户喜欢浅烘焙咖啡")
    except ConnectionError:
        pass  # 预期的首次失败：数据记录已置 failed

    # 换正常 LLM 重试同内容：复用 failed 记录重走管线（共享 stores，图谱状态连续）
    svc2 = make_service(llm=FakeLLM([LIGHT]))
    result = await svc2.remember("用户喜欢浅烘焙咖啡")
    assert result["deduplicated"] is False
    assert result["entities"] == 2 and result["relationships"] == 1

    rel, _, graph = stores
    assert rel.get_data(result["data_id"])["status"] == "completed"
    assert graph.get_entity_by_name("用户", "default") is not None


async def test_i1_cognify_resubmit_after_failure_returns_pending(make_service, stores, fake_cache):
    """I1 回归（cognify 侧）：失败后重提同文件返回 pending，重跑后 completed 且图中有实体。

    参数:
        make_service: conftest 服务工厂夹具。
        stores: 三库夹具，用于断言图库实体。
        fake_cache: fakeredis 缓存夹具，用于手动出队消费任务。
    断言:
        重提返回 status="pending"（不是 completed 假审计！）且复用原 data_id；
        手动跑 run_cognify_task 后运行记录 completed，图库中可查到文档实体。
    """
    svc = make_service(llm=FailingLLM(), cache=fake_cache)
    first = await svc.cognify_submit("notes.txt", DOC_B64)
    _, payload = await fake_cache.dequeue_priority([QUEUE_COGNIFY], timeout=1)
    await svc.run_cognify_task(**payload)  # 失败不抛出：data/pipeline_run 均置 failed
    assert (await svc.task_status(first["task_id"]))["status"] == "failed"

    # 换正常 LLM 重新提交同文件：不得被去重短路，须重新入队等待消费
    svc2 = make_service(llm=FakeLLM([DOC_OUT]), cache=fake_cache)
    second = await svc2.cognify_submit("notes.txt", DOC_B64)
    assert second["status"] == "pending" and second["data_id"] == first["data_id"]
    assert await fake_cache.queue_len(QUEUE_COGNIFY) == 1

    _, payload = await fake_cache.dequeue_priority([QUEUE_COGNIFY], timeout=1)
    await svc2.run_cognify_task(**payload)
    assert (await svc2.task_status(second["task_id"]))["status"] == "completed"
    _, _, graph = stores
    assert graph.get_entity_by_name("weave", "default") is not None  # 图中有实体，无静默丢数据


async def test_i2_message_appended_during_improve_not_marked(make_service, stores,
                                                             fake_cache, settings):
    """I2 回归：improve 处理窗口内新 append 的消息不被误标 synced，下轮仍被处理。

    参数:
        make_service: conftest 服务工厂夹具。
        stores: 三库夹具（本测试不入图，仅保持接口形态一致）。
        fake_cache: fakeredis 缓存夹具，用于断言 unsynced 集合。
        settings: 测试配置夹具（MidAppendLLM 追加消息时换算 TTL/上限）。
    断言:
        第一轮 improve 只评估 m1（过滤调用内追加 m2，落在“已读未打标”窗口）；
        m2 仍在 unsynced；第二轮 improve 把 m2 评估后打标，unsynced 才清空。
    """
    llm = MidAppendLLM(fake_cache, settings, "s-race")
    svc = make_service(llm=llm, cache=fake_cache)
    await svc.remember("随便聊聊", session_id="s-race")

    # 第一轮 improve：读出 m1 后，LLM 过滤调用内窗口追加 m2；本轮只应给 m1 打标
    first = await svc.run_improve("s-race")
    assert first["discarded"] == 1 and llm.calls == 1

    # 修复前：m2 被全量重写误标 synced 而丢失；修复后：m2 仍在未同步集合
    unsynced = await fake_cache.session_unsynced("s-race")
    assert [i["content"] for i in unsynced] == ["临时的想法"]

    # 第二轮 improve 消费 m2（第二次过滤调用即“m2 被处理”的证据）
    second = await svc.run_improve("s-race")
    assert second["discarded"] == 1 and llm.calls == 2
    assert await fake_cache.session_unsynced("s-race") == []


async def test_i3_batch_internal_supersede_same_source_type(make_service, stores):
    """I3 回归：单次 ingest 内同 (source, type) 双目标关系在批内更替，只留一条最新边。

    参数:
        make_service: conftest 服务工厂夹具。
        stores: 三库夹具，用于断言图库与关系库两侧的版本状态。
    断言:
        get_latest_relationship 只返回第二条（version=2 指向深烘焙咖啡）；
        neighbors 只返回一条最新事实；关系库最新边 version=2 且历史行已落库
        （edges 表共 2 行），满足 §4.4 同 (主体, 类型) 至多一条最新边。
    """
    svc = make_service(llm=FakeLLM([CONFLICT]))  # 单 chunk 双关系：一次 LLM 调用
    result = await svc.remember("我以前喜欢浅烘焙，现在喜欢深烘焙")
    assert result["relationships"] == 2  # 两条关系都落库（一条作为历史版本）
    assert result["superseded"] == 0  # 批内更替不走 DB 旧边置换计划

    rel, _, graph = stores
    uid = entity_id_for("default", "用户")
    # 图库断言：最新边唯一且指向第二个目标，版本号为 2
    latest = graph.get_latest_relationship(uid, "LIKES")
    assert latest["version"] == 2
    assert latest["target_id"] == entity_id_for("default", "深烘焙咖啡")
    facts = graph.neighbors([uid])
    assert len(facts) == 1 and facts[0]["target_name"] == "深烘焙咖啡"
    # 关系库侧同步验证：最新边 version=2；历史版本行一并落库（共 2 行边元数据）
    sql_latest = rel.get_latest_edge(uid, "LIKES")
    assert sql_latest["version"] == 2 and sql_latest["is_latest"] is True
    stats = {s["name"]: s for s in rel.dataset_stats()}
    assert stats["default"]["edge_count"] == 2
