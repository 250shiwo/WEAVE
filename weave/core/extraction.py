"""实体关系抽取 (remember/cognify) 与会话过滤 (improve) — spec §5.1/§5.3/§6."""

from weave.core.models import ExtractedEntity, ExtractedGraph, ExtractedRelationship

# 抽取系统提示词：约束 LLM 只输出固定结构的 JSON（实体 + 关系两个数组），
# 实体名用规范名词、关系类型用英文大写蛇形，保证下游解析与建边的稳定性。
EXTRACTION_SYSTEM = """你是知识图谱抽取器。从给定文本中抽取实体与关系，只输出 JSON：
{"entities": [{"name": "实体名", "entity_type": "类型如 Person/Preference/Project/Concept",
               "description": "一句话描述"}],
 "relationships": [{"source": "源实体名", "target": "目标实体名",
                    "relationship_type": "英文大写蛇形如 LIKES/WORKS_ON/LOCATED_IN",
                    "description": ""}]}
规则：实体名用规范名词；用户画像/偏好的实体类型用 Person/Preference；
没有可抽取内容时返回空数组。只输出 JSON，不要输出任何其他文字。"""

# 过滤系统提示词：会话防污染核心——只保留值得跨会话长期记忆的内容
# （用户画像、偏好、稳定事实、重要决定），一次性事件/闲聊一律丢弃；宁缺毋滥。
IMPROVE_SYSTEM = """你是长期记忆过滤器。给定 AI 助手的会话片段，判断哪些内容值得作为
跨会话的长期记忆（用户画像、偏好、稳定事实、重要决定），丢弃一次性事件、
闲聊、临时上下文。只输出 JSON：
{"facts": [{"keep": true, "statement": "精炼后的事实陈述句", "reason": "保留原因"},
            {"keep": false, "statement": "...", "reason": "丢弃原因"}]}
宁缺毋滥：不确定的一律 keep=false。只输出 JSON。"""


def parse_extraction(payload: dict) -> ExtractedGraph:
    """把 LLM 抽取响应解析为 ExtractedGraph，容忍垃圾数据。

    做什么: 从 LLM 返回的 JSON dict 中提取实体与关系列表；非 dict 负载、
        缺 name 的实体、缺 source/target/relationship_type 任一字段的关系、
        非 dict 项一律丢弃（宁缺毋滥，不让脏数据进图）。
    参数:
        payload: LLM 输出的 JSON 解析结果（可能不完整或含垃圾项）。
    返回:
        ExtractedGraph: 解析出的实体与关系；负载不可用时返回空图。
    """
    if not isinstance(payload, dict):
        return ExtractedGraph()  # 负载连 dict 都不是：无法解析，返回空图
    entities = [
        ExtractedEntity(
            name=str(e["name"]).strip(),  # 实体名去首尾空白
            entity_type=str(e.get("entity_type") or "Thing"),  # 缺类型兜底为 Thing
            description=str(e.get("description") or ""),  # 缺描述兜底为空串
        )
        for e in payload.get("entities") or []  # entities 缺失/为 None 时按空列表处理
        if isinstance(e, dict) and e.get("name")  # 只接受有 name 的 dict 项
    ]
    relationships = [
        ExtractedRelationship(
            source=str(r["source"]).strip(),
            target=str(r["target"]).strip(),
            # 关系类型统一为英文大写蛇形（如 "likes coffee" -> "LIKES_COFFEE"）
            relationship_type=str(r["relationship_type"]).strip().upper().replace(" ", "_"),
            description=str(r.get("description") or ""),
        )
        for r in payload.get("relationships") or []
        # 三要素缺一不可：缺任一字段的关系无法成边，直接丢弃
        if isinstance(r, dict) and r.get("source") and r.get("target") and r.get("relationship_type")
    ]
    return ExtractedGraph(entities=entities, relationships=relationships)


def parse_improve(payload: dict) -> list[str]:
    """把 LLM 过滤响应解析为要保留的事实陈述句列表。

    做什么: 只保留 keep 严格为 true 且陈述非空（strip 后有内容）的条目。
        这是会话防污染门禁：一次性事件/闲聊/空陈述一律丢弃，不进长期记忆。
    参数:
        payload: LLM 输出的 JSON 解析结果，形如 {"facts": [...]}。
    返回:
        list[str]: 保留的事实陈述句（已 strip），保持原顺序；负载不可用时空列表。
    """
    if not isinstance(payload, dict):
        return []  # 负载不可用：一条都不保留（宁缺毋滥）
    return [
        str(f["statement"]).strip()
        for f in payload.get("facts") or []
        # keep 必须严格为 True（真值 1/"true" 等不算），且陈述 strip 后非空
        if isinstance(f, dict) and f.get("keep") is True and str(f.get("statement") or "").strip()
    ]


async def extract_graph(llm, text: str) -> ExtractedGraph:
    """用 LLM 从文本中抽取实体关系子图。

    参数:
        llm: 任何提供 async complete_json(system, user) -> dict 方法的对象
            （生产环境是 LLMClient，测试中是 FakeLLM）。
        text: 待抽取的原始文本。
    返回:
        ExtractedGraph: 抽取并解析后的实体关系子图。
    """
    # 原文直接作为 user 消息，输出格式由系统提示词约束
    return parse_extraction(await llm.complete_json(EXTRACTION_SYSTEM, text))


async def filter_session_facts(llm, items: list[str]) -> list[str]:
    """用 LLM 过滤会话片段，只保留值得长期记忆的事实陈述。

    参数:
        llm: 任何提供 async complete_json(system, user) -> dict 方法的对象。
        items: 会话片段列表（AI 助手的回复/陈述）。
    返回:
        list[str]: 通过 keep/discard 门禁的事实陈述句。
    """
    # 会话片段拼成 markdown 无序列表，一条一行，便于 LLM 逐条判断
    user = "\n".join(f"- {c}" for c in items)
    return parse_improve(await llm.complete_json(IMPROVE_SYSTEM, user))
