"""图抽取/会话过滤解析与通用异步重试的行为测试（Task 5 验收）。

覆盖七组行为：
- parse_extraction 把合法负载解析为 ExtractedGraph；
- parse_extraction 容忍垃圾数据（缺字段的实体/关系、非 dict 项被丢弃，空负载返回空图）；
- parse_improve 只保留 keep=true 且陈述非空的事实；
- extract_graph 调用 LLM 并解析其结果，原文进入 user 消息；
- filter_session_facts 按 keep/discard 门禁过滤会话片段；
- with_retry 在前两次失败后第三次成功（指数退避重试）；
- with_retry 重试次数耗尽后原样抛出最后一次异常。
"""

import pytest

from weave.core.extraction import (
    extract_graph,
    filter_session_facts,
    parse_extraction,
    parse_improve,
)
from weave.infra.llm import with_retry
from tests.fakes import FakeLLM


def test_parse_extraction_valid():
    """合法负载解析为 ExtractedGraph：实体名与关系类型原样保留。"""
    payload = {
        "entities": [{"name": "用户", "entity_type": "Person", "description": "平台用户"}],
        "relationships": [{"source": "用户", "target": "深烘焙咖啡",
                           "relationship_type": "LIKES", "description": ""}],
    }
    g = parse_extraction(payload)
    assert g.entities[0].name == "用户"
    assert g.relationships[0].relationship_type == "LIKES"


def test_parse_extraction_tolerates_junk():
    """垃圾数据容忍：缺 name 的实体、缺字段的关系、非 dict 项被丢弃；空负载返回空图。"""
    g = parse_extraction({
        "entities": [{"name": "有效实体"}, {"no_name": True}, "garbage"],
        "relationships": [{"source": "a"}, {"source": "a", "target": "b", "relationship_type": "R"}],
    })
    assert [e.name for e in g.entities] == ["有效实体"]
    assert len(g.relationships) == 1
    assert parse_extraction({}).entities == []


def test_parse_improve_keep_only():
    """只保留 keep=true 且陈述非空的事实；keep=false 与空陈述都被丢弃。"""
    payload = {"facts": [
        {"keep": True, "statement": "用户偏好简洁回答", "reason": "稳定偏好"},
        {"keep": False, "statement": "今天下雨了", "reason": "一次性事件"},
        {"keep": True, "statement": "", "reason": "空陈述丢弃"},
    ]}
    assert parse_improve(payload) == ["用户偏好简洁回答"]


async def test_extract_graph_calls_llm():
    """extract_graph 恰好调一次 LLM：原文进入 user 消息，返回被解析为 ExtractedGraph。"""
    llm = FakeLLM([{"entities": [{"name": "用户"}], "relationships": []}])
    g = await extract_graph(llm, "用户喜欢咖啡")
    assert g.entities[0].name == "用户"
    assert "用户喜欢咖啡" in llm.calls[0]["user"]


async def test_filter_session_facts_keep_discard():
    """filter_session_facts 门禁：保留稳定偏好陈述，丢弃一次性事件。"""
    llm = FakeLLM([{"facts": [
        {"keep": True, "statement": "用户偏好简洁回答", "reason": "稳定偏好"},
        {"keep": False, "statement": "今天下雨了", "reason": "一次性事件"},
    ]}])
    kept = await filter_session_facts(llm, ["你说简洁点", "今天下雨了"])
    assert kept == ["用户偏好简洁回答"]


async def test_with_retry_succeeds_after_failures():
    """with_retry 重试：前两次抛错、第三次成功返回，总共恰好调用 3 次。"""
    attempts = {"n": 0}

    async def flaky():
        """抖动函数：前两次抛 ConnectionError，第三次返回 "ok"。"""
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert await with_retry(flaky, retries=3, base_delay=0.01) == "ok"
    assert attempts["n"] == 3


async def test_with_retry_gives_up():
    """with_retry 放弃：一直失败时重试次数耗尽后原样抛出最后一次异常。"""
    async def always_fail():
        """永远抛 ConnectionError 的失败函数。"""
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        await with_retry(always_fail, retries=2, base_delay=0.01)
