"""Redis 缓存封装测试：会话记忆 list（追加/裁剪/TTL/synced 集合打标）+ 优先级队列。

会话原文只存 Redis、永不进图（防污染机制③的存储基础），本模块验证：
1. 会话追加保序、超出上限裁剪最旧条目、键带 TTL；
2. unsynced 过滤与 mark_synced 按 id 打标后原文仍保留；
3. 单次 BRPOP 多 key 实现 improve 队列优先消费、空队列超时返回 None。
"""

from weave.core.session import append_session, get_session, get_unsynced, mark_synced
from weave.infra.cache import QUEUE_COGNIFY, QUEUE_IMPROVE


async def test_session_append_trim_and_ttl(settings, fake_cache):
    """验证会话追加的保序性、max_items 裁剪与 TTL 刷新。

    参数:
        settings: 测试配置夹具（session_max_items 默认 50，测试中途改为 3）。
        fake_cache: 注入 FakeRedis 的 Cache 夹具。
    断言:
        追加 5 条后内容保序且 synced 全为 False；把上限改为 3 再追加 1 条后
        只剩最新 3 条（最旧的被裁掉）；会话键的 TTL 大于 0（已设置过期时间）。
    """
    for i in range(5):
        await append_session(fake_cache, settings, "s1", f"消息{i}")
    items = await get_session(fake_cache, "s1")
    assert [i["content"] for i in items] == [f"消息{i}" for i in range(5)]
    assert all(i["synced"] is False for i in items)

    settings.session_max_items = 3
    await append_session(fake_cache, settings, "s1", "消息5")
    items = await get_session(fake_cache, "s1")
    assert [i["content"] for i in items] == ["消息3", "消息4", "消息5"]  # 裁剪最旧
    ttl = await fake_cache._r.ttl("weave:session:s1")
    assert ttl > 0


async def test_session_unsynced_and_mark_synced(settings, fake_cache):
    """验证 unsynced 过滤与 mark_synced 按 id 打标后原文保留（供 recall）。

    参数:
        settings: 测试配置夹具（提供 session_ttl_days 换算标记后的 TTL）。
        fake_cache: 注入 FakeRedis 的 Cache 夹具。
    断言:
        追加 2 条后 unsynced 为 2；按条目 id 打标后 unsynced 为空，
        但 get_session 仍能取到 2 条原文（标记不清除内容），且 synced
        标记由 synced 集合派生为 True。
    """
    await append_session(fake_cache, settings, "s2", "甲")
    await append_session(fake_cache, settings, "s2", "乙")
    unsynced = await get_unsynced(fake_cache, "s2")
    assert len(unsynced) == 2
    # 按本次处理过的条目 id 打标（improve 流程的真实调用形态）
    await mark_synced(fake_cache, settings, "s2", [i["id"] for i in unsynced])
    assert await get_unsynced(fake_cache, "s2") == []
    items = await get_session(fake_cache, "s2")
    assert len(items) == 2  # 原文保留供 recall
    assert all(i["synced"] is True for i in items)  # synced 标记由 SET 派生


async def test_queue_priority_improve_first(fake_cache):
    """验证单次 BRPOP 多 key 的优先级语义：improve 队列先于 cognify 被消费。

    参数:
        fake_cache: 注入 FakeRedis 的 Cache 夹具。
    断言:
        先往 cognify、再往 improve 各入队 1 条，第一次消费仍取到 improve
        （key 序即优先级，与入队先后无关）；第二次取到 cognify；
        两队列都空后 BRPOP 超时返回 None。
    """
    await fake_cache.enqueue(QUEUE_COGNIFY, {"kind": "doc"})
    await fake_cache.enqueue(QUEUE_IMPROVE, {"kind": "session"})
    first = await fake_cache.dequeue_priority([QUEUE_IMPROVE, QUEUE_COGNIFY], timeout=1)
    assert first[0] == QUEUE_IMPROVE and first[1] == {"kind": "session"}
    second = await fake_cache.dequeue_priority([QUEUE_IMPROVE, QUEUE_COGNIFY], timeout=1)
    assert second[0] == QUEUE_COGNIFY
    assert await fake_cache.dequeue_priority([QUEUE_IMPROVE, QUEUE_COGNIFY], timeout=1) is None
