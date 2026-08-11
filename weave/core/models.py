"""DataPoint 基类与 v1 节点类型 (spec §4.1/4.2) + 确定性 ID 工具。

去掉 feedback_weight/importance_weight；版本化用 version + is_latest 显式表达。
实体/边使用确定性 uuid5：同一 (dataset, 规范化名) 的实体天然幂等合并。
"""

import hashlib
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

# uuid5 命名空间：项目内固定，保证同一输入永远生成同一 ID（重启/跨进程稳定）
_NS = uuid.UUID("8f1b3c2a-7d4e-4f5a-9b6c-1e2d3a4b5c6d")


def utcnow() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。

    返回:
        str: 如 "2026-08-11T12:00:00.123456+00:00"，统一用字符串存时间便于跨库序列化。
    """
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    """生成随机 ID（uuid4 的 32 位 hex，无连字符）。

    返回:
        str: 全局唯一 ID，用于 DataPoint 及子类实例的主键。
    """
    return uuid.uuid4().hex


def norm_name(name: str) -> str:
    """规范化实体名：去首尾空白、内部连续空白压缩为单个空格、转小写。

    参数:
        name: 原始实体名，如 "  User  Name "。
    返回:
        str: 规范化结果，如 "user name"；用于实体幂等合并的比较基准。
    """
    return " ".join(name.strip().lower().split())


def entity_id_for(dataset: str, name: str) -> str:
    """生成实体的确定性 ID（uuid5）。

    同一 (dataset, 规范化名) 永远得到同一 ID，重复抽取同一实体时天然幂等合并。

    参数:
        dataset: 数据集命名空间，隔离不同来源的同名实体。
        name: 实体名（内部会先 norm_name 规范化再参与计算）。
    返回:
        str: 32 位 hex 实体 ID。
    """
    return uuid.uuid5(_NS, f"entity:{dataset}:{norm_name(name)}").hex


def edge_id_for(src_id: str, rel_type: str, dst_id: str) -> str:
    """生成边的确定性 ID（uuid5）。

    参数:
        src_id: 源实体 ID。
        rel_type: 关系类型，如 "LIKES"。
        dst_id: 目标实体 ID。
    返回:
        str: 32 位 hex 边 ID；同一 (源, 类型, 目标) 三元组永远得到同一 ID。
    """
    return uuid.uuid5(_NS, f"edge:{src_id}:{rel_type}:{dst_id}").hex


def content_hash(text: str) -> str:
    """计算文本内容的 sha256 hex 摘要，用于去重与变更检测。

    参数:
        text: 待哈希的文本（按 utf-8 编码）。
    返回:
        str: 64 位 hex 摘要。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DataPoint(BaseModel):
    """所有可存储数据点的基类（spec §4.1 全字段）。

    版本化模型：每次变更产生 version+1 的新行并置旧行 is_latest=False，
    查询默认只取 is_latest=True 的最新版本。
    """

    id: str = Field(default_factory=new_id)  # 主键，默认随机 uuid4 hex
    version: int = 1  # 版本号，从 1 开始递增
    type: str = "DataPoint"  # 节点类型判别字段，子类覆盖
    metadata: dict = Field(default_factory=dict)  # 自由扩展的附属信息
    source_pipeline: str | None = None  # 产生本数据点的管线名（如 ingest/extract）
    source_task: str | None = None  # 产生本数据点的任务 ID
    source_node_set: list[str] = Field(default_factory=list)  # 溯源：由哪些上游节点导出
    is_latest: bool = True  # 是否最新版本（软删除/历史版本置 False）
    created_at: str = Field(default_factory=utcnow)  # 创建时间（ISO8601 UTC）
    updated_at: str = Field(default_factory=utcnow)  # 更新时间（ISO8601 UTC）


class Entity(DataPoint):
    """知识图谱实体节点（spec §4.2）。"""

    type: str = "Entity"
    name: str = ""  # 实体原始名
    norm_name: str = ""  # 规范化名（norm_name 结果），用于幂等合并
    entity_type: str = "Thing"  # 实体类别（Person/Place/...，默认 Thing）
    description: str = ""  # 实体描述
    dataset: str = "default"  # 所属数据集命名空间


class TextChunk(DataPoint):
    """文本切块节点（spec §4.2），由原始文档切块而来。"""

    type: str = "TextChunk"
    text: str = ""  # 切块文本内容
    data_id: str = ""  # 来源文档/数据的 ID
    dataset: str = "default"  # 所属数据集命名空间
    position: int = 0  # 在原文中的块序号（从 0 开始）


class ExtractedEntity(BaseModel):
    """LLM 抽取结果中的实体（中间结构，落库前转为 Entity）。"""

    name: str  # 实体名（必填）
    entity_type: str = "Thing"  # 实体类别，默认 Thing
    description: str = ""  # 实体描述


class ExtractedRelationship(BaseModel):
    """LLM 抽取结果中的关系（中间结构，落库前解析为实体间边）。"""

    source: str  # 源实体名
    target: str  # 目标实体名
    relationship_type: str  # 关系类型（必填）
    description: str = ""  # 关系描述


class ExtractedGraph(BaseModel):
    """一次抽取得到的完整子图：实体列表 + 关系列表。"""

    entities: list[ExtractedEntity] = Field(default_factory=list)  # 抽取到的实体
    relationships: list[ExtractedRelationship] = Field(default_factory=list)  # 抽取到的关系


class RecallResult(BaseModel):
    """回忆查询的统一返回结构：三类来源的结果合并返回。"""

    facts: list[dict] = Field(default_factory=list)  # 结构化事实（图查询结果）
    chunks: list[dict] = Field(default_factory=list)  # 语义检索命中的文本块
    session_items: list[dict] = Field(default_factory=list)  # 会话记忆中的条目
