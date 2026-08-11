"""DashScope embedding 客户端 (OpenAI 兼容端点), 复用 with_retry."""

from openai import AsyncOpenAI

from weave.infra.llm import with_retry


class EmbeddingClient:
    """DashScope 向量嵌入客户端（OpenAI 兼容协议），只暴露批量文本嵌入。

    通过 OpenAI 兼容端点调用 DashScope 的 text-embedding 模型；空输入直接
    短路返回空列表，网络层失败经 with_retry 做 3 次指数退避重试。
    供 Task 9（向量写入）与 Task 13（召回）复用。
    """

    def __init__(self, api_key: str, base_url: str, model: str, client: AsyncOpenAI | None = None):
        """初始化 embedding 客户端。

        参数:
            api_key: DashScope API key（仅在 client 未传入时使用）。
            base_url: OpenAI 兼容接口地址，如
                "https://dashscope.aliyuncs.com/compatible-mode/v1"。
            model: 嵌入模型名，如 "text-embedding-v3"。
            client: 可选的已构造 AsyncOpenAI 实例（测试时注入桩对象）；
                为 None 时用 api_key/base_url 新建。
        """
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本，返回与输入一一对应的向量列表（带重试）。

        参数:
            texts: 待嵌入的文本列表；空列表直接返回空列表，不发起远程调用。
        返回:
            list[list[float]]: 与 texts 等长、顺序一致的向量列表。
        """
        if not texts:
            return []  # 空输入短路：避免无意义的远程调用

        async def call() -> list[list[float]]:
            """单次调用：请求 embeddings 接口并按输入顺序取出向量。"""
            resp = await self._client.embeddings.create(model=self._model, input=texts)
            # resp.data 与 input 顺序一致，逐条取出 embedding 组成向量列表
            return [d.embedding for d in resp.data]

        return await with_retry(call)  # 网络抖动/限流失败自动 3 次指数退避重试
