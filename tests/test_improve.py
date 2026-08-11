"""improve 管线与 remember 会话分支测试 (Task 11) — spec §6 防污染约束。

验证三件事：
1. remember 的 session 分支只写 Redis 会话缓存并入 improve 队列，原文不进图；
2. improve 先经 LLM 认知过滤，通过的事实才走标准入图（血缘 session_improve）；
3. 全部丢弃时不写任何图数据，只把会话标记为已同步。
"""

from weave.core.models import entity_id_for
from weave.infra.cache import QUEUE_IMPROVE
from tests.fakes import FakeLLM

# FakeLLM 第 1 个预设响应：improve 过滤结果（保留一条稳定偏好、丢弃一条一次性事件）
IMPROVE_OUT = {"facts": [
    {"keep": True, "statement": "用户偏好简洁回答", "reason": "稳定偏好"},
    {"keep": False, "statement": "今天在调试代码", "reason": "一次性事件"},
]}
# FakeLLM 第 2 个预设响应：对保留陈述句的实体关系抽取结果
EXTRACT_OUT = {"entities": [{"name": "用户", "entity_type": "Person"},
                            {"name": "简洁回答", "entity_type": "Preference"}],
               "relationships": [{"source": "用户", "target": "简洁回答",
                                  "relationship_type": "PREFERS"}]}


async def test_remember_session_writes_cache_and_enqueues(make_service, stores, fake_cache):
    """remember 会话分支：原文只写 Redis 并入 improve 队列，不触碰图库。

    参数:
        make_service: MemoryService 工厂夹具（注入 fake_cache）。
        stores: 三库夹具，用于断言图库无任何写入。
        fake_cache: fakeredis 缓存夹具，用于断言会话 list 与队列状态。
    """
    svc = make_service(cache=fake_cache)
    result = await svc.remember("你回答简洁一点", session_id="chat-1")
    # 返回会话模式标记、已入队标记与非空任务 ID
    assert result["mode"] == "session" and result["queued"] is True and result["task_id"]
    # 会话原文已进入 Redis 会话 list（1 条）
    assert len(await fake_cache.session_get("chat-1")) == 1
    # improve 队列积压 1 个任务
    assert await fake_cache.queue_len(QUEUE_IMPROVE) == 1
    _, _, graph = stores
    assert graph.count_entities() == 0  # 会话原文不进图


async def test_improve_filters_then_ingests(make_service, stores, fake_cache):
    """improve 管线：LLM 过滤后只有保留的事实入图，且血缘标记 session_improve。

    参数:
        make_service: MemoryService 工厂夹具。
        stores: 三库夹具，用于断言关系库血缘与图库邻居。
        fake_cache: fakeredis 缓存夹具，用于断言会话已标记同步。
    """
    # 两次 LLM 调用：先 improve 过滤，后对保留陈述做实体关系抽取
    llm = FakeLLM([IMPROVE_OUT, EXTRACT_OUT])
    svc = make_service(llm=llm, cache=fake_cache)
    await svc.remember("你回答简洁一点", session_id="chat-1")
    await svc.remember("今天在调试代码", session_id="chat-1")

    result = await svc.run_improve("chat-1")
    # 两条会话消息：保留 1 条、丢弃 1 条；入图产生 1 条关系
    assert result["kept"] == 1 and result["discarded"] == 1
    assert result["ingested"]["relationships"] == 1

    rel, _, graph = stores
    uid = entity_id_for("default", "用户")
    latest = rel.get_latest_edge(uid, "PREFERS")
    assert latest["source_pipeline"] == "session_improve"  # 血缘隔离标记
    # 图库中“用户”实体的邻居事实指向“简洁回答”
    assert graph.neighbors([uid])[0]["target_name"] == "简洁回答"
    assert await fake_cache.session_unsynced("chat-1") == []  # 已标记 synced


async def test_improve_discards_everything_writes_nothing(make_service, stores, fake_cache):
    """improve 全部丢弃：不写任何图数据，ingested 为 None。

    参数:
        make_service: MemoryService 工厂夹具。
        stores: 三库夹具，用于断言图库始终为空。
        fake_cache: fakeredis 缓存夹具。
    """
    # LLM 过滤响应：唯一一条闲聊被 keep=False 丢弃；无第二次抽取调用
    llm = FakeLLM([{"facts": [{"keep": False, "statement": "闲聊", "reason": "丢弃"}]}])
    svc = make_service(llm=llm, cache=fake_cache)
    await svc.remember("随便聊聊", session_id="chat-2")

    result = await svc.run_improve("chat-2")
    # 保留 0 条；全部丢弃时不走 ingest_text，ingested 为 None
    assert result["kept"] == 0 and result["ingested"] is None
    _, _, graph = stores
    assert graph.count_entities() == 0  # 全部丢弃, 不写图
