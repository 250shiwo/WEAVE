"""会话记忆助手: 隔离于图之外的 Redis 读写 (spec §6 防污染机制③).

对 Cache 会话方法的薄封装：统一从 settings 换算 TTL、读取 max_items，
让上层（API/MCP/improve 流程）不必关心 Redis 键结构与配置细节。
会话原文经此模块只落 Redis，永不进入图库与向量库。
"""


def _ttl_seconds(settings) -> int:
    """把配置中的会话存活天数换算成秒数（Redis EXPIRE 用）。

    参数:
        settings: 全局 Settings 实例，使用其 session_ttl_days 字段。
    返回:
        int: 会话键的过期秒数（session_ttl_days * 86400）。
    """
    return settings.session_ttl_days * 86400  # 天 -> 秒


async def append_session(cache, settings, session_id: str, content: str) -> None:
    """向指定会话追加一条消息（自动带上限裁剪与 TTL 刷新）。

    参数:
        cache: Cache 缓存封装实例（测试时注入 FakeRedis 客户端）。
        settings: 全局 Settings 实例，提供 session_max_items 与 session_ttl_days。
        session_id: 会话标识。
        content: 消息原文（只存 Redis，不进图）。
    返回: 无。
    """
    # 上限与 TTL 均取自 settings，调用方无需关心配置换算
    await cache.session_append(session_id, content, settings.session_max_items, _ttl_seconds(settings))


async def get_session(cache, session_id: str) -> list[dict]:
    """读取会话全部消息（含已同步），按插入顺序返回。

    参数:
        cache: Cache 缓存封装实例。
        session_id: 会话标识。
    返回:
        list[dict]: 消息项字典列表（id/content/ts/synced），会话不存在时为 []。
    """
    return await cache.session_get(session_id)


async def get_unsynced(cache, session_id: str) -> list[dict]:
    """读取会话中未同步的消息，供 improve 流程取增量待处理内容。

    参数:
        cache: Cache 缓存封装实例。
        session_id: 会话标识。
    返回:
        list[dict]: synced=False 的消息项字典列表，全部已同步时为 []。
    """
    return await cache.session_unsynced(session_id)


async def mark_synced(cache, settings, session_id: str) -> None:
    """把会话全部消息标记为已同步，并刷新 TTL（原文保留供 recall）。

    参数:
        cache: Cache 缓存封装实例。
        settings: 全局 Settings 实例，提供 session_ttl_days 换算标记后的 TTL。
        session_id: 会话标识。
    返回: 无。
    """
    await cache.session_mark_synced(session_id, _ttl_seconds(settings))
