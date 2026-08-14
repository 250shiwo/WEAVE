"""Redis 封装: 会话记忆 list + synced 集合 + 任务队列 (spec §4.3).

dequeue_priority 用单次 BRPOP 多 key: Redis 按 key 顺序扫描, 实现 improve 优先.

会话记忆防污染的存储基础：会话原文只存 Redis（list 结构，带 TTL），
永不写入图库；已同步状态存独立的 synced 集合（Redis SET，按条目 id
打标），避免"读全量->翻标记->重写"在处理窗口内误标新到消息（竞态）。
任务队列用 list 实现，LPUSH 入队、BRPOP 阻塞出队，调用方把高优先级
队列名放在 list 前面即可。
"""

import json
import logging
import time
import uuid

import redis
import redis.asyncio as aioredis

QUEUE_IMPROVE = "weave:queue:improve"  # 会话整理（improve）任务队列键，优先级高于 cognify
QUEUE_COGNIFY = "weave:queue:cognify"  # 文档认知（cognify）任务队列键
_SESSION_PREFIX = "weave:session:"  # 会话记忆 list 的键前缀，完整键为 前缀+session_id

logger = logging.getLogger(__name__)


def _synced_key(session_id: str) -> str:
    """生成会话 synced 集合的完整 Redis 键（SET，存已同步条目的 id）。

    参数:
        session_id: 会话标识。
    返回:
        str: 形如 weave:session:{session_id}:synced 的键名。
    """
    return f"{_SESSION_PREFIX}{session_id}:synced"


class Cache:
    """Redis 异步客户端封装：会话记忆 list 与任务队列的统一入口。

    构造时可注入外部 client（测试用 fakeredis 替换真实连接）；未注入时
    按 host/port/db 自建 aioredis.Redis。所有方法均为 async，可直接在
    事件循环中 await（redis.asyncio 本身是非阻塞 IO，无需 run_db 包装）。
    """

    def __init__(self, host: str, port: int, db: int, client=None):
        """初始化缓存封装；client 为 None 时自建真实 Redis 连接。

        参数:
            host: Redis 监听地址（client 缺省时生效）。
            port: Redis 端口（client 缺省时生效）。
            db: Redis 逻辑库编号（client 缺省时生效）。
            client: 可选的外部 Redis 异步客户端（如 fakeredis 的 FakeRedis）；
                传入后忽略 host/port/db，便于测试替换。
        返回: 无。
        """
        # decode_responses=True：读写均为 str，避免 bytes/str 来回解码
        self._r = client or aioredis.Redis(host=host, port=port, db=db, decode_responses=True)

    async def close(self) -> None:
        """关闭底层 Redis 连接，释放连接池资源（幂等可重复调用）。

        参数: 无。
        返回: 无。
        """
        await self._r.aclose()

    # ---------- 会话记忆 ----------
    async def session_append(self, session_id: str, content: str,
                             max_items: int, ttl_seconds: int) -> None:
        """向会话记忆 list 尾部追加一条消息，并裁剪长度、刷新两键 TTL。

        做什么: RPUSH 一条 JSON 序列化的消息项（id/content/ts/synced 占位），
            随后 LTRIM 只保留最新 max_items 条（裁剪最旧），最后 EXPIRE
            刷新会话列表键的存活时间；同时 EXPIRE synced 集合键，
            保证两键 TTL 对齐（同生共死，集合不泄漏）。
        参数:
            session_id: 会话标识，拼在键前缀后定位 Redis list。
            content: 消息原文（会话原文只存 Redis，永不进图）。
            max_items: 单个会话最多保留的条目数，超出裁掉最旧的。
            ttl_seconds: 会话键的过期秒数（由 session_ttl_days 换算而来）。
        返回: 无。
        """
        key = _SESSION_PREFIX + session_id  # 会话 list 的完整 Redis 键
        # 消息项四字段：uuid 十六进制 id、原文、Unix 秒时间戳、未同步占位
        # （synced 真值由 synced 集合派生，此处仅保持消息项结构稳定）
        item = json.dumps({"id": uuid.uuid4().hex, "content": content,
                           "ts": time.time(), "synced": False})
        await self._r.rpush(key, item)  # 追加到 list 尾部（时间序即插入序）
        await self._r.ltrim(key, -max_items, -1)  # 只保留尾部最新 max_items 条，裁掉最旧
        await self._r.expire(key, ttl_seconds)  # 刷新列表键 TTL：活跃会话永不过期
        # synced 集合键同步续期：两键 TTL 对齐，避免集合先于列表过期导致重复处理
        await self._r.expire(_synced_key(session_id), ttl_seconds)

    async def session_get(self, session_id: str) -> list[dict]:
        """读取会话全部消息（含已同步与未同步），按插入顺序返回。

        做什么: LRANGE 取全量条目后，SMEMBERS synced 集合逐个回填 synced
            标记（条目 id 在集合中即已同步）；synced 不以消息项内字段为准
            （append 时恒为 False 占位），而是按 id 从集合派生——这是
            recall 与 unsynced 的统一事实来源。
        参数:
            session_id: 会话标识。
        返回:
            list[dict]: 消息项字典列表，每项键为 id/content/ts/synced；
            会话不存在或已过期时返回空列表。
        """
        rows = await self._r.lrange(_SESSION_PREFIX + session_id, 0, -1)  # 取全量（0 到末尾）
        synced_ids = await self._r.smembers(_synced_key(session_id))  # 已同步条目 id 集合
        items = [json.loads(r) for r in rows]  # 逐条反序列化 JSON 为 dict
        for i in items:
            i["synced"] = i["id"] in synced_ids  # 覆盖占位：以 synced 集合为准
        return items

    async def session_unsynced(self, session_id: str) -> list[dict]:
        """读取会话中尚未同步（id 不在 synced 集合中）的消息，供 improve 流程取增量。

        参数:
            session_id: 会话标识。
        返回:
            list[dict]: 未同步消息项字典列表（字段同 session_get）；
            全部已同步或会话不存在时返回空列表。
        """
        # session_get 已按 synced 集合回填标记，此处只留未同步的
        return [i for i in await self.session_get(session_id) if not i["synced"]]

    async def session_mark_synced(self, session_id: str, ttl_seconds: int,
                                  ids: list[str]) -> None:
        """按条目 id 把会话消息标记为已同步（SADD synced 集合），并刷新集合 TTL。

        做什么: 只把本次实际处理过的条目 id SADD 进 synced 集合（不再
            "读全量->翻标记->重写 list"）：improve 处理窗口内新 append 的
            消息 id 不在集合中，永不会被误标（竞态修复）；原文保留供
            recall 检索，只加标记不清除内容。ids 为空时直接返回，
            避免误建空键。
        参数:
            session_id: 会话标识。
            ttl_seconds: synced 集合键的过期秒数（与会话列表键 TTL 对齐）。
            ids: 本次实际处理过的条目 id 列表（含被过滤丢弃的——它们已被评估过）。
        返回: 无。
        """
        if not ids:
            return  # 无可标记条目：避免 SADD 空集造出无意义键
        key = _synced_key(session_id)  # 会话 synced 集合的完整 Redis 键
        await self._r.sadd(key, *ids)  # 按 id 打标：集合外的新消息不受影响
        await self._r.expire(key, ttl_seconds)  # 刷新集合键 TTL：与列表键同生共死

    async def session_clear(self, session_id: str) -> None:
        """删除整个会话记忆（list 与 synced 集合两键一并删除，键不存在时为无害空操作）。

        参数:
            session_id: 会话标识。
        返回: 无。
        """
        await self._r.delete(_SESSION_PREFIX + session_id, _synced_key(session_id))

    # ---------- 任务队列 ----------
    async def enqueue(self, queue: str, payload: dict) -> None:
        """把任务载荷 JSON 序列化后 LPUSH 入队（与 BRPOP 配合构成 FIFO）。

        参数:
            queue: 队列键（QUEUE_IMPROVE 或 QUEUE_COGNIFY）。
            payload: 任务载荷字典，须可 JSON 序列化。
        返回: 无。
        """
        await self._r.lpush(queue, json.dumps(payload))  # 左进右出：LPUSH + BRPOP = 先进先出

    async def dequeue_priority(self, queues: list[str], timeout: int) -> tuple[str, dict] | None:
        """单次 BRPOP 阻塞消费多个队列，key 顺序即优先级。

        做什么: 一次 BRPOP 传入全部队列键，Redis 按列表顺序扫描、命中第一个
            非空队列即弹出（禁止拆成多个独立 BRPOP，否则优先级语义失效）；
            全部为空则阻塞至 timeout 后返回 None。BRPOP 的客户端读超时竞争
            与连接中断也在此兜底为返回 None——异常若上抛到 worker_loop 的
            try 之外会使 worker 任务静默死亡、队列无人消费，因此韧性逻辑
            内聚在本层，调用方只需按"超时无任务"继续轮询。
        参数:
            queues: 队列键列表，靠前的优先级更高（如 [improve, cognify]）。
            timeout: 阻塞等待秒数，超时返回 None。
        返回:
            tuple[str, dict] | None: (命中的队列键, 反序列化后的载荷字典)；
            超时无任务、客户端读超时竞争或连接中断（Redis 重启中）时
            均返回 None。
        """
        try:
            result = await self._r.brpop(queues, timeout=timeout)  # 单次多 key BRPOP：key 序即优先级
        except redis.exceptions.TimeoutError:
            # 客户端读超时竞争：redis 8.x 客户端 socket 读超时与 BRPOP 服务端
            # timeout 接近，服务端即将返回 nil 时客户端可能竞争先触发抛错；
            # 语义等同"本轮无任务"，兜底返回 None 让 worker 继续轮询
            return None
        except redis.exceptions.ConnectionError:
            # Redis 可能重启中：记 warning 后按无任务处理，worker 继续轮询等其恢复
            logger.warning("BRPOP 连接中断（Redis 重启中?），本轮按无任务处理，继续轮询")
            return None
        if result is None:
            return None  # 超时：所有队列均无任务
        queue, raw = result  # brpop 返回 (键名, 弹出的 JSON 字符串)
        return queue, json.loads(raw)

    async def queue_len(self, queue: str) -> int:
        """查询队列当前积压的任务数（LLEN，键不存在时为 0）。

        参数:
            queue: 队列键（QUEUE_IMPROVE 或 QUEUE_COGNIFY）。
        返回:
            int: 队列中待消费的任务条数。
        """
        return await self._r.llen(queue)
