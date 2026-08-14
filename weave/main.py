"""入口: uv run weave / uv run python -m weave.main.

装配生产依赖（真实三库/Redis/LLM 客户端）后以 uvicorn 启动 FastAPI 应用；
启动失败（典型为 Kuzu/LanceDB/SQLite 数据文件被其他进程占用）时给出
清晰的中文报错并以非零码退出。
"""

import logging

import uvicorn

from weave.api.app import create_app
from weave.config import get_settings


def main() -> None:
    """Weave 进程入口：读配置 -> 装配应用 -> uvicorn 常驻运行。

    参数: 无（配置经 get_settings() 从环境变量/.env 读取）。
    返回: 无；uvicorn.run 阻塞直至收到停止信号。
    异常:
        SystemExit: create_app 装配失败（数据文件被占用等）时抛出，
            消息中含原始异常摘要，进程以非零码退出。
    """
    settings = get_settings()
    # 生产可观测性：INFO 级别让 weave.worker 的启动/任务失败日志可见
    logging.basicConfig(level=logging.INFO)
    # httpx 每次 HTTP 请求都打 INFO 日志（LLM/嵌入调用频繁），噪音大；压到 WARNING 只留异常
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        app = create_app(settings)
    except Exception as exc:
        # spec §8: Kuzu/LanceDB 被其他进程占用时给出清晰报错；
        # from exc 保留原始异常链，便于向上追溯真实根因
        raise SystemExit(
            f"Weave 启动失败（可能是 Kuzu/LanceDB/SQLite 数据文件被其他进程占用）: {exc}"
        ) from exc
    # 阻塞式运行：REST /v1 与 MCP /mcp 同端口对外，直到进程收到停止信号
    uvicorn.run(app, host=settings.weave_host, port=settings.weave_port)


if __name__ == "__main__":
    main()
