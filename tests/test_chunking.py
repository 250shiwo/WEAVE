"""文本切块器与确定性 ID 工具的行为测试（Task 4 验收）。

覆盖四组行为：
- 短文本整块返回、纯空白文本返回空列表；
- 段落优先组装：能合并则合并，但每块长度不得超过 chunk_size；
- 超长段落硬切：按 chunk_size/overlap 滑动开窗，相邻块保留重叠字符；
- norm_name / entity_id_for / edge_id_for 的确定性与区分度。
"""

from weave.core.chunking import split_text
from weave.core.models import edge_id_for, entity_id_for, norm_name


def test_short_text_single_chunk():
    """短文本不切割原样返回；strip 后为空的文本不产生任何块。"""
    assert split_text("用户喜欢咖啡", 1500, 200) == ["用户喜欢咖啡"]
    assert split_text("   ", 1500, 200) == []


def test_paragraph_packing_respects_size():
    """段落优先组装：每段 30 字符，两两合并需 30+2+30=62 > 50，故三段各成一块。"""
    text = "\n\n".join(["段落一" * 10, "段落二" * 10, "段落三" * 10])
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 3
    assert all(len(c) <= 50 for c in chunks)


def test_long_paragraph_hard_split_with_overlap():
    """超长段落硬切：100 字符按窗口 40、步长 30 切为 [0:40] [30:70] [60:100] 三块。"""
    text = "a" * 100
    chunks = split_text(text, chunk_size=40, overlap=10)
    assert len(chunks) == 3
    assert all(len(c) <= 40 for c in chunks)
    # 相邻块保留 10 字符重叠
    assert chunks[1][:10] == chunks[0][-10:]


def test_id_helpers_deterministic():
    """norm_name 去首尾空白、压缩内部空白并转小写；实体/边 ID 确定性且随入参变化。"""
    assert norm_name("  User  Name ") == "user name"
    # 同一 dataset 下名字仅大小写/空白不同 -> 同一实体 ID（幂等合并）
    assert entity_id_for("default", "User") == entity_id_for("default", " user ")
    # 不同 dataset 同名 -> 不同实体 ID
    assert entity_id_for("default", "a") != entity_id_for("other", "a")
    # 边 ID 由 (源, 关系类型, 目标) 唯一确定
    assert edge_id_for("s", "LIKES", "d") == edge_id_for("s", "LIKES", "d")
    assert edge_id_for("s", "LIKES", "d") != edge_id_for("s", "HATES", "d")
