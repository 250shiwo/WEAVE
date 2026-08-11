"""pytest 共享夹具：为全部测试提供隔离、可重复的 Settings 实例。"""

import fakeredis.aioredis
import pytest

from weave.config import Settings
from weave.core.service import MemoryService
from weave.infra.cache import Cache
from weave.infra.graph import GraphStore
from weave.infra.relational import RelationalStore
from weave.infra.vector import VectorStore
from tests.fakes import FakeEmbedder, FakeLLM


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


@pytest.fixture
def stores(settings):
    """构建一套指向临时目录的三库实例（SQLite / LanceDB / Kuzu）。

    做什么: 按 settings 中的测试路径分别创建关系库、向量库、图库实例，
        供测试直接断言各库底层状态；测试结束后按 图 -> 向量 -> 关系
        的顺序全部关闭，释放文件句柄避免临时目录被占用。
    参数:
        settings: 上面的 settings 夹具，三个存储路径均指向 tmp_path。
    返回:
        tuple: (RelationalStore, VectorStore, GraphStore) 三元组。
    """
    rel = RelationalStore(settings.relational_db_path)  # SQLite 关系库（建五张表）
    vec = VectorStore(settings.vector_db_path)  # LanceDB 向量库（懒建表）
    graph = GraphStore(settings.graph_db_path)  # Kuzu 图库（建四张表）
    yield rel, vec, graph
    # 逆序关闭：先关图库与向量库，最后关关系库，保证句柄全部释放
    graph.close()
    vec.close()
    rel.close()


@pytest.fixture
def make_service(settings, stores):
    """构建 MemoryService 工厂：按测试需要注入不同的 llm/embedder/cache 替身。

    做什么: 返回一个工厂函数，每次调用都用共享的三库实例（stores 夹具）
        组装一个 MemoryService；未显式传入 llm/embedder 时使用默认的
        FakeLLM/FakeEmbedder（无外部依赖、确定性行为），cache 默认 None
        （remember 永久分支不触碰缓存）。
    参数:
        settings: 测试配置夹具。
        stores: 三库夹具 (rel, vec, graph)。
    返回:
        callable: 工厂函数 factory(llm=None, embedder=None, cache=None) -> MemoryService。
    """
    def factory(llm=None, embedder=None, cache=None) -> MemoryService:
        """组装一个 MemoryService 实例（工厂的闭包实现）。

        参数:
            llm: LLM 替身，默认 None 时新建空响应队列的 FakeLLM。
            embedder: 嵌入替身，默认 None 时新建 FakeEmbedder。
            cache: 缓存客户端，默认 None（永久写入路径不使用缓存）。
        返回:
            MemoryService: 依赖全部就位的服务门面实例。
        """
        rel, vec, graph = stores  # 解包共享三库，与测试断言侧使用同一批实例
        return MemoryService(settings, rel, vec, graph, cache,
                             llm or FakeLLM(), embedder or FakeEmbedder())
    return factory


@pytest.fixture
async def fake_cache(settings):
    """构建一个注入 fakeredis 假客户端的 Cache 实例（无真实 Redis 依赖）。

    做什么: 用 fakeredis.aioredis.FakeRedis 作为底层客户端构造 Cache，
        在进程内模拟完整 Redis 协议行为（list/expire/BRPOP 等），
        让会话记忆与任务队列测试无需启动真实 Redis 服务。
    参数:
        settings: 上面的 settings 夹具，提供 redis_host/port/db 配置值
            （仅作构造参数透传，实际连接由 FakeRedis 接管）。
    返回:
        Cache: 底层客户端为 FakeRedis（decode_responses=True，返回 str）
            的缓存封装实例；测试结束后关闭连接释放资源。
    """
    c = Cache(settings.redis_host, settings.redis_port, settings.redis_db,
              client=fakeredis.aioredis.FakeRedis(decode_responses=True))
    yield c
    await c.close()
