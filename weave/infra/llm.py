"""DeepSeek 抽取客户端 (OpenAI 兼容) + 通用异步重试 (3 次指数退避, spec §8)."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI


async def with_retry(
    factory: Callable[[], Awaitable[Any]], retries: int = 3, base_delay: float = 0.5
) -> Any:
    """对异步操作按指数退避重试，直到成功或重试次数耗尽。

    做什么: 反复调用 factory()，成功则立即返回其结果；失败则按
        base_delay * 2^(attempt-1) 秒退避后重试（第 1 次失败等 base_delay 秒，
        第 2 次等 2*base_delay 秒……），尝试次数达到 retries 后放弃，
        把最后一次异常原样抛出。供 LLM 调用及 Task 6 的其他外部调用复用。
    参数:
        factory: 无参异步可调用对象，每次尝试都会重新调用它。
        retries: 最大尝试次数（含首次），默认 3。
        base_delay: 指数退避的基础秒数，默认 0.5。
    返回:
        Any: factory() 成功时的返回值，原样回传。
    异常:
        Exception: 尝试次数耗尽时，将最后一次失败的异常原样抛出。
    """
    attempt = 0  # 已失败次数：首次执行不算失败，失败后 +1
    while True:
        try:
            return await factory()  # 成功：直接返回结果，退出循环
        except Exception:
            attempt += 1  # 本次尝试失败，计入失败计数
            if attempt >= retries:
                raise  # 尝试次数耗尽：放弃，把最后一次异常原样向上抛出
            # 指数退避：第 1 次失败等 base_delay 秒，之后每次翻倍，给远端恢复时间
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))


class LLMClient:
    """DeepSeek 抽取客户端（OpenAI 兼容协议），只暴露 JSON 补全。

    所有请求使用 response_format=json_object 强制 JSON 输出、temperature=0
    保证抽取结果确定性；网络层失败经 with_retry 做 3 次指数退避重试。
    """

    def __init__(self, api_key: str, base_url: str, model: str, client: AsyncOpenAI | None = None):
        """初始化 LLM 客户端。

        参数:
            api_key: DeepSeek API key（仅在 client 未传入时使用）。
            base_url: OpenAI 兼容接口地址，如 "https://api.deepseek.com"。
            model: 模型名，如 "deepseek-chat"。
            client: 可选的已构造 AsyncOpenAI 实例（测试时注入桩对象）；
                为 None 时用 api_key/base_url 新建。
        """
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def complete_json(self, system: str, user: str) -> dict:
        """调用 chat completions 并把响应内容解析为 dict（带重试）。

        参数:
            system: 系统提示词（抽取/过滤规则与输出格式约束）。
            user: 用户消息（待处理文本）。
        返回:
            dict: 模型输出的 JSON 解析结果；空内容按 "{}" 兜底解析为空 dict。
        """
        async def call() -> dict:
            """单次调用：构造 JSON-mode 请求并解析响应内容为 dict。"""
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},  # 系统提示词：抽取/过滤规则
                    {"role": "user", "content": user},  # 用户消息：待处理文本
                ],
                response_format={"type": "json_object"},  # 强制 JSON 输出，便于解析
                temperature=0,  # 关闭随机性：抽取/过滤要求确定性输出
            )
            # 内容为空时按 "{}" 兜底，避免 json.loads(None) 抛 TypeError
            return json.loads(resp.choices[0].message.content or "{}")

        return await with_retry(call)  # 网络抖动/限流失败自动 3 次指数退避重试
