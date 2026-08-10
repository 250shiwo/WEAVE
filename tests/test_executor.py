"""DB 单线程执行器测试：验证 run_db 的正常返回、超时保护、确定性错误不重试、瞬时锁错误重试。"""

import asyncio
import time

import pytest

from weave.infra.executor import run_db


async def test_run_db_success():
    """验证 run_db 能在线程池中执行同步函数并原样返回其结果。

    做什么: 提交一个 lambda 到执行器，确认返回值经过 asyncio 边界后不丢失、不变形。
    参数: 无（pytest-asyncio auto 模式自动驱动协程）。
    返回: 无；断言 41 + 1 == 42 即说明参数透传与结果回传均正确。
    """
    assert await run_db(lambda x: x + 1, 41, timeout=1.0) == 42


async def test_run_db_timeout_keeps_loop_responsive():
    """验证原生调用卡死时 run_db 按 timeout 抛出超时错误，且执行器随后仍可服务新调用。

    做什么: 先提交一个 sleep 0.5 秒的阻塞任务并设极短超时（且禁止重试），
        确认超时异常被抛出；随后立即再提交一个正常任务，
        确认单线程执行器从卡死任务中恢复后依旧可用（事件循环未被拖死）。
    参数: 无。
    返回: 无；断言后续调用返回 "alive"。
    """

    def stuck():
        # 模拟 Kuzu/LanceDB 原生调用在线程内短暂卡死：无法被外部强杀，0.5s 后自行释放
        time.sleep(0.5)

    # timeout=0.05 让等待快速超时；max_retries=0 禁止重试以便直接观察异常
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await run_db(stuck, timeout=0.05, max_retries=0)
    # 事件循环未被拖死，后续调用可立即执行（stuck 线程 0.5s 后释放，1.0s 预算足够）
    assert await run_db(lambda: "alive", timeout=1.0) == "alive"


async def test_run_db_no_retry_on_deterministic_error():
    """验证确定性错误（如约束冲突）不重试：首次抛出即向上传播。

    做什么: 提交一个总是抛 ValueError 的函数，max_retries=2，
        确认 ValueError 原样抛出且函数只被调用 1 次（重试确定性错误毫无意义）。
    参数: 无。
    返回: 无；断言调用计数为 1。
    """
    calls = {"n": 0}

    def bad():
        # 记录调用次数，用于断言"未发生重试"
        calls["n"] += 1
        raise ValueError("constraint violation")

    with pytest.raises(ValueError):
        await run_db(bad, timeout=1.0, max_retries=2)
    # 确定性错误不应触发任何重试
    assert calls["n"] == 1


async def test_run_db_retries_transient_lock_error():
    """验证瞬时锁错误（如嵌入式库文件锁竞争）会自动重试并最终成功。

    做什么: 提交一个首次抛 "Could not set lock on file"、第二次正常返回的函数，
        确认 run_db 识别锁错误为瞬时错误、重试一次后返回 "ok"，总调用次数为 2。
    参数: 无。
    返回: 无；断言返回值与调用计数。
    """
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            # 模拟 Kuzu/LanceDB 并发写时的文件锁竞争：瞬时故障，重试即可恢复
            raise RuntimeError("Could not set lock on file")
        return "ok"

    assert await run_db(flaky, timeout=1.0, max_retries=2) == "ok"
    # 第一次失败 + 一次重试成功 = 共 2 次调用
    assert calls["n"] == 2
