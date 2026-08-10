"""Weave 全局配置模块。

本模块是整个平台唯一的配置来源：所有子系统（API 服务、MCP 服务、
三类存储、任务队列等）都应通过 `get_settings()` 获取同一份缓存的
`Settings` 实例，而不是各自读取环境变量。

配置优先级（pydantic-settings 默认规则）：
    显式传参 > 进程环境变量 > .env 文件 > 字段默认值。
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Weave 全局配置项，字段即环境变量（不区分大小写）的映射。

    例如字段 `deepseek_api_key` 对应环境变量 `DEEPSEEK_API_KEY`。
    .env 中出现但此处未定义的键会被静默忽略（extra="ignore"），
    因此可以与其他工具共享同一个 .env 文件。
    """

    # env_file 指定默认读取项目根目录的 .env；env_file_encoding 保证
    # 中文注释/值在 Windows 下不出现乱码；extra="ignore" 忽略未知键。
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- DashScope（向量模型） ----------
    dashscope_api_key: str = ""  # DashScope API 密钥，留空表示未配置
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # OpenAI 兼容接入地址
    dashscope_embedding_model: str = "qwen3.7-text-embedding"  # 文本嵌入模型名
    # ---------- DeepSeek（抽取模型） ----------
    deepseek_api_key: str = ""  # DeepSeek API 密钥，留空表示未配置
    deepseek_base_url: str = "https://api.deepseek.com"  # DeepSeek 接入地址
    deepseek_model: str = "deepseek-v4-flash"  # 实体/关系抽取所用模型名
    # ---------- 存储 ----------
    vector_db_path: str = "./data/vector_db"  # LanceDB 向量库目录
    graph_db_path: str = "./data/graph_db"  # Kuzu 图数据库目录
    relational_db_path: str = "./data/weave.db"  # SQLite 关系库文件路径
    # ---------- Redis ----------
    redis_host: str = "127.0.0.1"  # Redis 监听地址
    redis_port: int = 6379  # Redis 端口
    redis_db: int = 0  # Redis 逻辑库编号
    # ---------- Weave 服务 ----------
    weave_api_key: str = "dev-key"  # 调用方鉴权 token，本地开发默认值
    weave_host: str = "127.0.0.1"  # 服务监听地址
    weave_port: int = 8000  # 服务监听端口
    # ---------- 行为参数 ----------
    session_max_items: int = 50  # 单个会话最多保留的记忆条目数
    session_ttl_days: int = 7  # 会话存活天数，过期后清理
    chunk_size: int = 1500  # 文本分块大小（字符数）
    chunk_overlap: int = 200  # 相邻分块的重叠字符数
    db_call_timeout: float = 30.0  # 单次数据库调用超时秒数
    db_call_max_retries: int = 2  # 数据库调用失败后的最大重试次数
    queue_poll_timeout: int = 5  # 队列阻塞取任务的轮询超时秒数


@lru_cache
def get_settings() -> Settings:
    """获取全局唯一的 Settings 单例。

    借助 lru_cache 缓存，进程内多次调用返回同一个实例，
    避免重复解析 .env 与环境变量；测试需要不同配置时应直接
    构造 `Settings(...)` 而不是调用本函数。

    返回:
        Settings: 按默认来源（环境变量 + .env）解析出的配置实例。
    """
    return Settings()
