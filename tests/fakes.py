"""测试替身 (fakes)：FakeLLM 与 FakeEmbedder。

FakeLLM 按预设响应队列依次返回，并记录每次调用的 system/user 入参，
用于在无真实 LLM API 的情况下测试图抽取与会话过滤流程；
FakeEmbedder 生成确定性单位向量（同文本必同向量），供向量检索相关测试使用。
"""

import hashlib
import math
import random


class FakeLLM:
    """按队列返回预设响应; .calls 记录每次调用."""

    def __init__(self, responses: list[dict] | None = None):
        """初始化 FakeLLM。

        参数:
            responses: 预设响应队列，每次 complete_json 调用弹出队首一个；
                为 None 时视为空队列（任何调用都会触发 AssertionError）。
        """
        self.responses = list(responses or [])  # 复制为新列表，避免调用方后续修改影响队列
        self.calls: list[dict] = []  # 调用记录：[{"system": ..., "user": ...}, ...]

    async def complete_json(self, system: str, user: str) -> dict:
        """模拟 LLM 的 JSON 补全调用：记录入参并弹出下一个预设响应。

        参数:
            system: 系统提示词（原样记录到 .calls，供断言）。
            user: 用户消息（原样记录到 .calls，供断言）。
        返回:
            dict: 预设响应队列的队首元素（弹出后队列长度减一）。
        异常:
            AssertionError: 响应队列已空仍被调用，说明被测代码多调了一次 LLM。
        """
        self.calls.append({"system": system, "user": user})  # 先记录，便于断言调用次数与入参
        if not self.responses:
            raise AssertionError("FakeLLM 无剩余响应")
        return self.responses.pop(0)  # 弹出队首：响应严格按添加顺序消费


class FakeEmbedder:
    """确定性向量: 同文本同向量, 不同文本近似正交."""

    def __init__(self, dim: int = 8):
        """初始化 FakeEmbedder。

        参数:
            dim: 生成向量的维度，默认 8（测试用低维即可）。
        """
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """为每个文本生成确定性单位向量。

        参数:
            texts: 待嵌入的文本列表。
        返回:
            list[list[float]]: 与 texts 等长、顺序一致的向量列表；
                同一文本永远得到同一向量（不依赖任何外部服务）。
        """
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        """生成单个文本的确定性单位向量（内部实现）。

        参数:
            text: 待嵌入文本。
        返回:
            list[float]: 长度为 self.dim 的单位向量（L2 范数为 1）。
        """
        # 用文本的 sha256 摘要作为随机种子：同文本必同向量，不同文本近似正交
        rng = random.Random(hashlib.sha256(text.encode()).digest())
        v = [rng.uniform(-1, 1) for _ in range(self.dim)]  # 各分量取 [-1, 1) 均匀随机值
        norm = math.sqrt(sum(x * x for x in v)) or 1.0  # L2 范数；or 1.0 防御零向量除零
        return [x / norm for x in v]  # 归一化为单位向量，便于余弦相似度计算
