"""文本切块: 段落优先组装, 超长段落硬切并保留重叠 (spec §5.1).

切块策略：
1. 文本 strip 后为空 -> 返回空列表；
2. 整体不超过 chunk_size -> 整块返回，不切割；
3. 否则按空行（\\n\\n）拆成段落，顺序贪心组装：能塞进当前块就合并，
   塞不下就封存当前块、另起新块，保证每块长度 <= chunk_size；
4. 单个段落超过 chunk_size 时先硬切（滑动窗口，相邻片保留 overlap
   字符重叠），再参与组装。
"""


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """把长文本切成不超过 chunk_size 字符的块列表。

    参数:
        text: 待切块文本。
        chunk_size: 每块最大字符数。
        overlap: 硬切时相邻块的重叠字符数（段落组装不产生重叠）。
    返回:
        list[str]: 切块结果；空文本返回 []，块顺序与原文一致。
    """
    text = text.strip()
    if not text:
        return []  # 纯空白文本没有可索引内容
    if len(text) <= chunk_size:
        return [text]  # 短文本整块返回，避免无谓切割
    chunks: list[str] = []
    current = ""  # 正在组装中的当前块
    # 按空行拆段，丢弃空白段落；段落顺序即原文顺序
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        # 超长段落先硬切成 <= chunk_size 的片，普通段落整体作为一片
        pieces = _hard_split(para, chunk_size, overlap) if len(para) > chunk_size else [para]
        for piece in pieces:
            # 尝试把本片并入当前块（块间以空行连接）
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= chunk_size:
                current = candidate  # 塞得下：合并进当前块
            else:
                if current:
                    chunks.append(current)  # 塞不下：封存当前块
                current = piece  # 本片作为新块起点
    if current:
        chunks.append(current)  # 封存最后一个未封存的块
    return chunks


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """把单个超长段落按滑动窗口硬切，相邻片保留 overlap 字符重叠。

    参数:
        text: 长度超过 chunk_size 的单个段落。
        chunk_size: 每片最大字符数。
        overlap: 相邻片重叠字符数，须小于 chunk_size 才有意义。
    返回:
        list[str]: 切片列表，每片 <= chunk_size，首尾相接覆盖原文。
    """
    step = max(1, chunk_size - overlap)  # 窗口步长；max(1,...) 防止 overlap >= chunk_size 时死循环
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break  # 本片已覆盖到文本末尾：剩余尾巴完全被重叠区包含，不再产生冗余小片
        start += step
    return pieces
