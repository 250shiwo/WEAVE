"""pytest 共享夹具：为全部测试提供隔离、可重复的 Settings 实例。"""

import pytest

from weave.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """构建一个面向测试的 Settings 实例。

    所有存储路径都指向 pytest 提供的临时目录，并使用测试专用
    API key 与更短的超时时间，保证测试不读写真实数据、快速失败。

    参数:
        tmp_path: pytest 内置夹具，为每次测试提供独立的临时目录。

    返回:
        Settings: 存储路径指向 tmp_path、weave_api_key 为 "test-key"、
        db_call_timeout 缩短为 5 秒、queue_poll_timeout 缩短为 1 秒
        的配置实例。
    """
    return Settings(
        _env_file=None,  # 不读取本地 .env，避免测试受开发环境影响
        vector_db_path=str(tmp_path / "vector_db"),  # 向量库指向临时目录
        graph_db_path=str(tmp_path / "graph_db"),  # 图库指向临时目录
        relational_db_path=str(tmp_path / "weave.db"),  # SQLite 指向临时目录
        weave_api_key="test-key",  # 测试专用鉴权 token
        db_call_timeout=5.0,  # 缩短 DB 调用超时，让失败更快暴露
        queue_poll_timeout=1,  # 缩短队列轮询超时，避免测试长时间阻塞
    )
