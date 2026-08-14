"""DashScope embedding 客户端的行为测试（Task 6 验收）。

覆盖三组行为：
- embed 对多条文本返回与输入一一对应的向量列表；
- embed 对空列表直接返回空列表（不发起任何远程调用）；
- embed 遇到瞬时失败（ConnectionError）时经 with_retry 重试后成功。
"""

from types import SimpleNamespace

from weave.infra.embedding import EmbeddingClient


class _StubEmbeddings:
    """OpenAI embeddings 接口的桩实现：可按需先失败若干次再返回固定向量。"""

    def __init__(self, fail_times: int = 0):
        """初始化桩对象。

        参数:
            fail_times: 前若干次 create 调用抛 ConnectionError 的次数，
                默认 0 表示第一次就成功，用于模拟瞬时故障后的恢复。
        """
        self.fail_times = fail_times

    async def create(self, model, input):
        """模拟 embeddings.create：先按剩余失败次数抛错，再为每条输入返回固定向量。

        参数:
            model: 模型名（桩中忽略，仅为匹配真实接口签名）。
            input: 待嵌入的文本列表，每条文本对应返回一个 [1.0, 0.0] 向量。
        返回:
            SimpleNamespace: 模拟的响应对象，data 为与 input 等长的向量列表。
        异常:
            ConnectionError: fail_times 未耗尽时抛出，模拟瞬时网络故障。
        """
        if self.fail_times > 0:
            self.fail_times -= 1  # 消耗一次失败配额，模拟瞬时故障
            raise ConnectionError("boom")
        # 为每条输入文本生成一个固定的二维向量，模拟真实响应的 data 结构
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])


def _client(fail_times: int = 0) -> EmbeddingClient:
    """构造注入了桩 embeddings 的 EmbeddingClient，不触碰真实网络。

    参数:
        fail_times: 桩对象在成功前要先失败的次数，默认 0。
    返回:
        EmbeddingClient: 底层 AsyncOpenAI 被 SimpleNamespace 桩替换的客户端。
    """
    stub = SimpleNamespace(embeddings=_StubEmbeddings(fail_times))
    return EmbeddingClient("k", "http://x", "m", client=stub)


async def test_embed_returns_vectors():
    """embed 对两条输入返回两个向量，顺序与输入一一对应。"""
    out = await _client().embed(["a", "b"])
    assert out == [[1.0, 0.0], [1.0, 0.0]]


async def test_embed_empty_input():
    """embed 对空列表直接返回空列表（短路，不发起远程调用）。"""
    assert await _client().embed([]) == []


async def test_embed_retries_transient_failure():
    """embed 遇到一次瞬时 ConnectionError 后由 with_retry 重试并成功返回向量。"""
    assert await _client(fail_times=1).embed(["a"]) == [[1.0, 0.0]]


class _BatchProbeEmbeddings:
    """记录每批 input 大小的桩：验证超限输入被自动分批（DashScope 单批上限 20）。"""

    def __init__(self):
        """初始化探针桩，batch_sizes 记录每次 create 收到的条数。"""
        self.batch_sizes: list[int] = []

    async def create(self, model, input):
        """记录本批条数，并为每条输入返回其在全局输入中的序号向量。

        参数:
            model: 模型名（桩中忽略）。
            input: 本批待嵌入文本列表。
        返回:
            SimpleNamespace: data 为与 input 等长的向量列表，向量值取文本
                自身的序号（如 "t7" -> [7.0]），用于验证跨批拼接后顺序不乱。
        """
        self.batch_sizes.append(len(input))
        # 文本形如 "t0".."tN"，取出序号作为向量，便于校验总顺序与输入一致
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(t[1:])]) for t in input]
        )


async def test_embed_batches_when_exceeding_limit():
    """45 条输入按 batch_size=20 自动分为 3 批（20/20/5），结果顺序与输入一致。"""
    probe = _BatchProbeEmbeddings()
    stub = SimpleNamespace(embeddings=probe)
    client = EmbeddingClient("k", "http://x", "m", client=stub, batch_size=20)

    texts = [f"t{i}" for i in range(45)]
    out = await client.embed(texts)

    # 断言分三批调用且每批不超上限
    assert probe.batch_sizes == [20, 20, 5]
    # 断言返回 45 个向量且顺序与输入一一对应（跨批拼接不乱序）
    assert out == [[float(i)] for i in range(45)]
