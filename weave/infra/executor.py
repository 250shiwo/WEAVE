"""嵌入式 DB 调用的统一入口: 单线程执行器 + 超时 + 瞬时错误重试.

所有 SQLite/LanceDB/Kuzu 原生调用必须经 run_db() 提交:
- max_workers=1 串行化写入, 匹配嵌入式库单写者约束
- wait_for 超时防止原生调用卡死拖住事件循环 (线程内卡死无法强杀, 但 API 保持可用)

Q:为什么不调用 DB 自带的方法,而通过一个统一的入口?
A: 通过统一入口可以方便地添加超时保护和瞬时错误重试，而不需要在每个调用点都添加。

解决了阻塞事件循环，拖慢所有请求,多协程并发写入报错崩溃,遇到死锁/慢查询卡死整个服务,每次自己写 try-except 处理锁冲突的问题

"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# 模块级单例线程池：整个进程共享，max_workers=1 保证所有 DB 调用严格串行，
# 避免并发写入触发嵌入式库（Kuzu/LanceDB/SQLite）的单写者冲突；
# thread_name_prefix 便于在线程转储/日志中快速定位 DB 线程。
_db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weave-db")


def is_transient(exc: BaseException) -> bool:
    """判断异常是否为可重试的瞬时错误。

    做什么: 供 run_db 决定失败后要重试还是直接向上抛出。
        锁竞争/超时类瞬时错误可重试; 约束冲突等确定性错误不重试。
    参数:
        exc: 本次 DB 调用抛出的异常实例。
    返回:
        bool: True 表示瞬时错误（重试可能成功），False 表示确定性错误（重试无意义）。
    """
    # 等待超时属于瞬时故障：可能只是一时拥塞，重试有机会成功
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    # Kuzu/LanceDB/SQLite 的文件锁竞争错误信息中均含 "lock"（不区分大小写），
    # 例如 "Could not set lock on file"；锁会被持有者释放，故属于瞬时错误
    return "lock" in str(exc).lower()


async def run_db(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float = 30.0,
    max_retries: int = 2,
    **kwargs: Any,
) -> Any:
    """将同步 DB 调用提交到单线程执行器执行，带超时保护与瞬时错误重试。

    做什么: 把阻塞的原生 DB 调用转移到独立的 DB 线程中执行，避免拖住事件循环；
        等待超过 timeout 秒则抛 TimeoutError；遇到瞬时错误按指数退避重试，
        确定性错误或重试次数耗尽则将异常原样向上抛出。
    参数:
        fn: 要执行的同步可调用对象（如 Kuzu/LanceDB/SQLite 的原生方法）。
        *args: 透传给 fn 的位置参数。
        timeout: 单次尝试的最长等待秒数，默认 30.0。
        max_retries: 瞬时错误允许的最大重试次数，默认 2（即最多执行 3 次）。
        **kwargs: 透传给 fn 的关键字参数。
    返回:
        Any: fn 的返回值，原样回传。
    异常:
        TimeoutError: 单次等待超过 timeout 且重试耗尽（或不可重试）时抛出。
        Exception: fn 抛出的确定性错误（如约束冲突）不重试，原样抛出。
    """
    # 获取当前运行中的事件循环，用于把任务调度到线程池
    loop = asyncio.get_running_loop()
    # 已失败次数计数：首次执行不算重试，失败后 +1
    attempt = 0
    while True:
        try:
            # 把 fn(*args, **kwargs) 提交到单线程执行器；
            # 用 lambda 包裹以延迟参数绑定，返回 asyncio Future
            future = loop.run_in_executor(_db_executor, lambda: fn(*args, **kwargs))
            # 带超时等待结果：超时则取消等待并抛 TimeoutError
            # （线程内已开始的原生调用无法强杀，但事件循环与 API 保持可用）
            return await asyncio.wait_for(future, timeout)
        except Exception as exc:
            # 本次尝试失败，计入重试计数
            attempt += 1
            # 重试次数耗尽，或属于确定性错误（不可重试）：原样向上抛出
            if attempt > max_retries or not is_transient(exc):
                raise
            # 瞬时错误：线性退避（0.2s、0.4s……）后重试，给锁持有者释放的时间
            await asyncio.sleep(0.2 * attempt)
