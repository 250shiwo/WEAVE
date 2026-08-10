"""配置模块测试：验证 Settings 能正确读取环境变量并使用约定的默认值。"""

from weave.config import Settings


def test_settings_reads_env(monkeypatch):
    """验证 Settings 能从进程环境变量中读取配置值。

    参数:
        monkeypatch: pytest 内置夹具，用于在测试期间临时设置环境变量，
            测试结束后自动还原，避免污染其它用例。

    断言:
        DEEPSEEK_API_KEY、WEAVE_API_KEY 两个环境变量分别被映射到
        Settings.deepseek_api_key 与 Settings.weave_api_key 字段。
    """
    # 模拟运行环境中注入的两个密钥环境变量
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.setenv("WEAVE_API_KEY", "secret")
    # _env_file=None 表示不读取 .env 文件，只验证环境变量来源
    s = Settings(_env_file=None)
    assert s.deepseek_api_key == "k1"
    assert s.weave_api_key == "secret"


def test_settings_defaults():
    """验证 Settings 在无任何外部输入时使用设计约定的默认值。

    构造时不传任何参数且禁用 .env 读取，全部字段应回落到
    类定义中的默认值，保证开箱即用。

    断言:
        分块参数、DB 调用防护参数、队列轮询参数、会话参数、
        服务监听地址等默认值均与设计文档一致。
    """
    s = Settings(_env_file=None)
    # 分块默认值
    assert s.chunk_size == 1500
    assert s.chunk_overlap == 200
    # DB 调用超时与重试默认值
    assert s.db_call_timeout == 30.0
    assert s.db_call_max_retries == 2
    # 队列轮询默认值
    assert s.queue_poll_timeout == 5
    # 会话默认值
    assert s.session_max_items == 50
    assert s.session_ttl_days == 7
    # 服务监听默认值
    assert s.weave_host == "127.0.0.1"
    assert s.weave_port == 8000
