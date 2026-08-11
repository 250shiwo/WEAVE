"""cognify 管线与进程内 worker 测试 (Task 12) — spec §8 队列优先级约束。

验证五件事：
1. cognify_submit 解码 base64、落 data/pipeline_run 记录并入 cognify 队列，
   run_cognify_task 走 ingest_text 入图后状态流转为 completed；
2. 同内容重复提交时按内容哈希去重，直接返回 completed + deduplicated；
3. 非法输入（非 txt/md 扩展名、非法 base64）抛 ValueError；
4. run_cognify_task 失败不抛出，pipeline_run 置 failed 并携带错误信息，
   未知 task_id 的 task_status 返回 not_found；
5. worker_loop 单循环单次 BRPOP [improve, cognify]，key 序即优先级——
   cognify 先入队、improve 后入队，improve 仍必须先被消费。
"""

import asyncio
import base64

import pytest

from weave.infra.cache import QUEUE_COGNIFY, QUEUE_IMPROVE
from weave.worker import worker_loop
from tests.fakes import FakeLLM

# 测试用文档原文与其 base64 编码（cognify_submit 的入参形式）
DOC = "Weave 是知识图谱记忆平台。它使用 Kuzu 作为图数据库。"
DOC_B64 = base64.b64encode(DOC.encode()).decode()
# FakeLLM 预设的文档抽取结果：两个实体 + 一条 USES 关系
DOC_OUT = {"entities": [{"name": "Weave", "entity_type": "Project"},
                        {"name": "Kuzu", "entity_type": "Concept"}],
           "relationships": [{"source": "Weave", "target": "Kuzu",
                              "relationship_type": "USES"}]}


async def test_cognify_submit_and_run_task(make_service, stores, fake_cache):
    """cognify 全链路：提交入队 -> 消费执行 -> 状态 completed 且实体入图。

    参数:
        make_service: MemoryService 工厂夹具（注入 FakeLLM 与 fake_cache）。
        stores: 三库夹具，用于断言图库中实体已写入。
        fake_cache: fakeredis 缓存夹具，用于断言队列积压与手动出队。
    """
    svc = make_service(llm=FakeLLM([DOC_OUT]), cache=fake_cache)
    submit = await svc.cognify_submit("notes.txt", DOC_B64)
    # 提交返回 pending 状态与非空任务 ID；cognify 队列积压 1 个任务
    assert submit["status"] == "pending" and submit["task_id"]
    assert await fake_cache.queue_len(QUEUE_COGNIFY) == 1

    # 手动出队并按 payload 键名直接分发（与 worker 行为一致）
    _, payload = await fake_cache.dequeue_priority([QUEUE_COGNIFY], timeout=1)
    await svc.run_cognify_task(**payload)

    # 运行记录终态为 completed，管线名为 cognify
    status = await svc.task_status(submit["task_id"])
    assert status["status"] == "completed" and status["pipeline_name"] == "cognify"
    # 图库中可查到抽取出的实体（按规范化名小写查询）
    _, _, graph = stores
    assert graph.get_entity_by_name("weave", "default") is not None


async def test_cognify_dedup_returns_completed(make_service, fake_cache):
    """同内容重复提交：内容哈希去重命中，直接返回 completed + deduplicated。

    参数:
        make_service: MemoryService 工厂夹具。
        fake_cache: fakeredis 缓存夹具，用于手动出队执行第一次任务。
    """
    svc = make_service(llm=FakeLLM([DOC_OUT]), cache=fake_cache)
    first = await svc.cognify_submit("notes.txt", DOC_B64)
    # 先消费掉第一次提交，让数据记录状态变为 completed
    _, payload = await fake_cache.dequeue_priority([QUEUE_COGNIFY], timeout=1)
    await svc.run_cognify_task(**payload)
    # 第二次提交同内容：去重命中，不再入队，直接返回已完成
    second = await svc.cognify_submit("notes.txt", DOC_B64)
    assert second["status"] == "completed" and second["deduplicated"] is True


async def test_cognify_rejects_bad_input(make_service, fake_cache):
    """非法输入校验：不支持的扩展名与非法 base64 均抛 ValueError。

    参数:
        make_service: MemoryService 工厂夹具（本测试不触发 LLM，用默认替身）。
        fake_cache: fakeredis 缓存夹具。
    """
    svc = make_service(cache=fake_cache)
    # .pdf 不在 v1 支持列表（txt/md/markdown），错误信息含“文件类型”
    with pytest.raises(ValueError, match="文件类型"):
        await svc.cognify_submit("doc.pdf", DOC_B64)
    # 非法 base64 字符串无法解码，错误信息含“base64”
    with pytest.raises(ValueError, match="base64"):
        await svc.cognify_submit("doc.txt", "!!!not-base64!!!")


async def test_run_cognify_task_failure_marks_failed(make_service, fake_cache):
    """失败路径：run_cognify_task 不抛出，pipeline_run 置 failed 并记录错误。

    参数:
        make_service: MemoryService 工厂夹具（注入必定失败的 LLM 替身）。
        fake_cache: fakeredis 缓存夹具。
    """
    class FailingLLM:
        """必定失败的 LLM 替身：模拟 LLM 服务不可用。"""

        async def complete_json(self, system, user):
            """每次调用都抛 ConnectionError。

            参数:
                system: 系统提示词（忽略）。
                user: 用户消息（忽略）。
            异常:
                ConnectionError: 恒定抛出，模拟 LLM 服务宕机。
            """
            raise ConnectionError("LLM down")

    svc = make_service(llm=FailingLLM(), cache=fake_cache)
    submit = await svc.cognify_submit("notes.txt", DOC_B64)
    _, payload = await fake_cache.dequeue_priority([QUEUE_COGNIFY], timeout=1)
    await svc.run_cognify_task(**payload)  # 不抛出
    # 运行记录终态为 failed，错误信息包含底层异常文本
    status = await svc.task_status(submit["task_id"])
    assert status["status"] == "failed" and "LLM down" in status["error"]
    # 未知 task_id：返回 not_found 而非抛异常
    assert (await svc.task_status("missing"))["status"] == "not_found"


async def test_worker_consumes_queues_with_priority(make_service, fake_cache):
    """worker 优先级：cognify 先入队、improve 后入队，improve 必须先被消费。

    参数:
        make_service: MemoryService 工厂夹具。
        fake_cache: fakeredis 缓存夹具，worker 直接在其上轮询。
    """
    # 两次 LLM 调用：先 improve 过滤（{"facts": []} 全部丢弃），后 cognify 抽取
    llm = FakeLLM([{"facts": []}, DOC_OUT])
    svc = make_service(llm=llm, cache=fake_cache)
    order: list[str] = []  # 记录两个分发目标的实际执行顺序
    orig_improve, orig_cognify = svc.run_improve, svc.run_cognify_task

    async def spy_improve(**kw):
        """improve 探针：记录执行顺序后委托原实现。

        参数:
            **kw: 队列 payload 原样透传（task_id/session_id/dataset）。
        """
        order.append("improve")
        await orig_improve(**kw)

    async def spy_cognify(**kw):
        """cognify 探针：记录执行顺序后委托原实现。

        参数:
            **kw: 队列 payload 原样透传（task_id/data_id/dataset）。
        """
        order.append("cognify")
        await orig_cognify(**kw)

    svc.run_improve = spy_improve
    svc.run_cognify_task = spy_cognify

    # cognify 先入队, improve 后入队 —— 但 improve 必须先被消费
    submit = await svc.cognify_submit("notes.txt", DOC_B64)
    await svc.remember("随便聊聊", session_id="s9")

    # 启动 worker 协程轮询，直到两个任务都被消费（或最多等 5 秒）
    stop = asyncio.Event()
    worker = asyncio.create_task(worker_loop(svc, fake_cache, 1, stop))
    for _ in range(100):
        if len(order) == 2:
            break
        await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(worker, timeout=5)

    # 单 BRPOP key 序保证 improve 先于 cognify，与入队顺序无关
    assert order == ["improve", "cognify"]
    # cognify 任务最终也正常完成
    assert (await svc.task_status(submit["task_id"]))["status"] == "completed"
