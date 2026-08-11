"""混合检索（hybrid_recall）测试：验证 recall 的向量入口、图扩展、版本过滤、数据集隔离与会话叠加。

测试通过 conftest 的 make_service/fake_cache 夹具组装 MemoryService（三库指向临时目录、
缓存为 fakeredis），LLM 与嵌入使用 Fake 替身，全程无任何外部服务依赖。
测试数据常量 LIGHT/DARK 复用 tests/test_pipelines.py 中的预设抽取结果。
"""

from tests.fakes import FakeLLM
from tests.test_pipelines import DARK, LIGHT


async def test_recall_returns_facts_and_chunks(make_service, fake_cache):
    """验证 recall 返回图事实（标 origin="graph"）与向量命中块，无会话时 session_items 为空。

    参数:
        make_service: conftest 服务工厂夹具。
        fake_cache: conftest 假 Redis 缓存夹具（本用例不触发会话读取，仅保证依赖完整）。
    """
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")

    result = await svc.recall("用户喜欢什么")
    # 图事实断言：1 跳邻居边映射为 fact 字典，来源/关系类型/目标与写入一致
    assert result["facts"][0]["source"] == "用户"
    assert result["facts"][0]["relationship_type"] == "LIKES"
    assert result["facts"][0]["target"] == "浅烘焙咖啡"
    assert result["facts"][0]["origin"] == "graph"
    # 向量命中块断言：检索入口 text_chunks 命中原文块
    assert result["chunks"][0]["text"] == "用户喜欢浅烘焙咖啡"
    # 未传 session_id：不叠加任何会话内容
    assert result["session_items"] == []


async def test_recall_excludes_superseded(make_service, fake_cache):
    """验证被版本更替取代的旧事实不出现在 recall 结果中（图扩展只看 is_latest 边）。

    参数:
        make_service: conftest 服务工厂夹具。
        fake_cache: conftest 假 Redis 缓存夹具。
    """
    svc = make_service(llm=FakeLLM([LIGHT, DARK]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")
    await svc.remember("用户其实更喜欢深烘焙咖啡")

    result = await svc.recall("用户喜欢什么烘焙")
    targets = [f["target"] for f in result["facts"]]
    assert targets == ["深烘焙咖啡"]  # 被取代的旧版本不出现


async def test_recall_dataset_isolation(make_service, fake_cache):
    """验证 recall 按数据集隔离：入口向量检索与图扩展都带 dataset 过滤，跨数据集查不到事实。

    参数:
        make_service: conftest 服务工厂夹具。
        fake_cache: conftest 假 Redis 缓存夹具。
    """
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡", dataset="personal")

    # 同数据集可检索到事实；换无关数据集则实体入口为空、图事实随之为空
    assert (await svc.recall("咖啡", dataset="personal"))["facts"] != []
    assert (await svc.recall("咖啡", dataset="other"))["facts"] == []


async def test_recall_session_overlay(make_service, fake_cache):
    """验证会话缓存叠加：带 session_id 时原文以 source="session" 返回，不带则不叠加。

    参数:
        make_service: conftest 服务工厂夹具。
        fake_cache: conftest 假 Redis 缓存夹具（会话原文所在）。
    """
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    await svc.remember("用户喜欢浅烘焙咖啡")
    # 会话分支：原文只进 Redis 会话 list（不进图/向量库），等待 improve 异步沉淀
    await svc.remember("我们明天讨论架构", session_id="chat-7")

    result = await svc.recall("架构", session_id="chat-7")
    # 会话叠加断言：原文按插入序返回，标注来源为 session
    assert result["session_items"][0]["content"] == "我们明天讨论架构"
    assert result["session_items"][0]["source"] == "session"
    # 不带 session_id 则无会话内容
    assert (await svc.recall("架构"))["session_items"] == []
