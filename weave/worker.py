"""进程内 worker: 单循环单 BRPOP, key 顺序 [improve, cognify] 即优先级 (spec §8).

禁止拆成两个队列各自独立 BRPOP (并发消费者会使优先级失效).
"""

import asyncio
import logging

from weave.infra.cache import QUEUE_COGNIFY, QUEUE_IMPROVE

logger = logging.getLogger(__name__)


async def worker_loop(service, cache, poll_timeout: int, stop: asyncio.Event) -> None:
    """队列消费主循环：单次 BRPOP 双队列，按 key 顺序保证 improve 优先于 cognify。

    做什么: 循环调用 cache.dequeue_priority 一次传入 [QUEUE_IMPROVE, QUEUE_COGNIFY]
        两个键——Redis BRPOP 按 key 顺序扫描、命中第一个非空队列即弹出，因此
        improve 任务天然先于 cognify 被消费（与入队先后无关）；取到任务后按
        队列名分发：payload 键名与 run_improve / run_cognify_task 的参数名
        一致，直接 **payload 解包调用。超时无任务则继续轮询，stop 置位后退出。
    参数:
        service: MemoryService 门面，提供 run_improve / run_cognify_task。
        cache: 缓存客户端，提供 dequeue_priority（单次多 key BRPOP；其内部
            已兜底客户端读超时竞争与连接中断为返回 None，worker 永不因此退出）。
        poll_timeout: 每轮 BRPOP 的阻塞秒数；超时返回 None 继续下一轮，
            同时它也是 stop 信号的最大响应延迟。
        stop: 停止信号；置位后当前轮 BRPOP 返回即退出循环（幂等可重复置位）。
    返回: 无（协程随 stop 置位而结束）。
    异常: 不抛出——单个任务失败只记录日志，主循环继续消费后续任务；
        run_cognify_task 内部已自行兜底（置 failed 不上抛），此 except 是
        针对 run_improve 上抛与意外错误的最后防线。
    """
    logger.info("worker_loop 启动, poll_timeout=%s", poll_timeout)
    while not stop.is_set():
        # 单次 BRPOP 传入两个队列键：key 序即优先级，improve 永远先于 cognify；
        # 读超时竞争/连接中断已由 cache.dequeue_priority 内聚兜底为返回 None
        item = await cache.dequeue_priority([QUEUE_IMPROVE, QUEUE_COGNIFY], poll_timeout)
        if item is None:
            continue  # 本轮超时无任务：回到循环头检查 stop 后继续阻塞轮询
        queue, payload = item  # 命中的队列键与反序列化后的任务载荷
        try:
            if queue == QUEUE_IMPROVE:
                # improve payload 为 {task_id, session_id, dataset}，键名即参数名
                await service.run_improve(**payload)
            else:
                # cognify payload 为 {task_id, data_id, dataset}，键名即参数名
                await service.run_cognify_task(**payload)
        except Exception:
            # 兜底防线：单任务失败不中断主循环，只记录日志继续消费
            logger.exception("worker 任务失败: %s %s", queue, payload)
