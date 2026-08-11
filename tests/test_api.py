"""REST 开放 API 与 Bearer 认证测试 (Task 15) — spec §7.2。

验证六件事：
1. GET /v1/health 免认证，无鉴权头也返回 200；
2. 其余端点缺失或错误的 Bearer token 一律 401；
3. POST /v1/memories 写入 + POST /v1/recall 召回的 REST 全链路；
4. POST /v1/documents 非法输入经 ValueError 异常处理器返回 400 bad_request 错误信封；
5. GET /v1/datasets 可用，未知 task_id 的 GET /v1/tasks/{id} 返回 not_found；
6. DELETE /v1/memories 无 confirm 时 400（高危全清需显式确认），confirm=true 时返回 scope=all。

测试经 TestClient 走完整 ASGI 栈（认证中间件 + 路由 + lifespan 内嵌 worker），
service 由 conftest 夹具注入（真实三库指向临时目录 + fakeredis + FakeLLM），
全程无任何外部服务依赖。
"""

import base64

import pytest
from fastapi.testclient import TestClient

from weave.api.app import create_app
from tests.fakes import FakeLLM
from tests.test_pipelines import LIGHT

# 测试专用鉴权头：与 conftest settings 夹具的 weave_api_key="test-key" 对应
AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def client(settings, make_service, fake_cache):
    """构建注入测试 service 的 TestClient（走完整应用，含 lifespan 启动的内嵌 worker）。

    做什么: 用 make_service 工厂组装 MemoryService（FakeLLM 预设 LIGHT 抽取
        响应、fakeredis 作为缓存），再交给 create_app 组包 FastAPI 应用；
        以 with 语法进入 TestClient 上下文，触发 lifespan 启动 worker 协程，
        测试结束退出上下文时 lifespan 置位 stop 并取消 worker，保证资源清理。
    参数:
        settings: conftest 测试配置夹具（weave_api_key="test-key"）。
        make_service: conftest 服务工厂夹具（共享三库指向临时目录）。
        fake_cache: fakeredis 缓存夹具，供会话/队列及 lifespan worker 轮询使用。
    返回:
        TestClient: lifespan 已启动（worker 运行中）的同步测试客户端。
    """
    svc = make_service(llm=FakeLLM([LIGHT]), cache=fake_cache)
    # with 进入时执行 lifespan：创建并启动内嵌 worker；退出时 stop 置位并取消 worker
    with TestClient(create_app(settings, service=svc)) as c:
        yield c


def test_health_is_open(client):
    """验证 /v1/health 是认证豁免端点：不带任何鉴权头也返回 200。

    参数:
        client: 上面的 TestClient 夹具。
    """
    # 健康检查供探活使用，必须免认证；无 Authorization 头也应 200
    assert client.get("/v1/health").status_code == 200


def test_auth_required(client):
    """验证除 /v1/health 外的端点都受 Bearer 认证保护：缺失或错误 token 均 401。

    参数:
        client: 上面的 TestClient 夹具。
    """
    # 完全不带 Authorization 头：认证中间件直接拒绝
    r = client.post("/v1/memories", json={"content": "x"})
    assert r.status_code == 401
    # 带了 Authorization 头但 token 错误：同样拒绝（不区分缺失/错误，防探测）
    r = client.post("/v1/memories", json={"content": "x"},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_remember_and_recall_via_rest(client):
    """验证 REST 写入/召回全链路：POST /v1/memories 永久入图，POST /v1/recall 命中事实。

    参数:
        client: 上面的 TestClient 夹具（FakeLLM 已预设 LIGHT 抽取响应）。
    """
    # 永久写入（无 session_id）：FakeLLM 消费 LIGHT 响应，抽取 2 实体 + 1 关系
    r = client.post("/v1/memories", json={"content": "用户喜欢浅烘焙咖啡"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["mode"] == "permanent"

    # 混合检索：查询文本经确定性 FakeEmbedder 向量化，图扩展出 LIKES 事实
    r = client.post("/v1/recall", json={"query": "用户喜欢什么"}, headers=AUTH)
    assert r.json()["facts"][0]["target"] == "浅烘焙咖啡"


def test_documents_validation_error(client):
    """验证 ValueError 异常处理器：非法文档输入返回 400 + bad_request 错误信封。

    参数:
        client: 上面的 TestClient 夹具。
    """
    # .pdf 不在 v1 支持列表（txt/md/markdown），cognify_submit 抛 ValueError，
    # 由应用级异常处理器统一转为 400 {"error": {"code": "bad_request", ...}}
    bad = base64.b64encode(b"x").decode()
    r = client.post("/v1/documents",
                    json={"file_name": "a.pdf", "content_base64": bad}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


def test_datasets_and_task_status(client):
    """验证 GET /v1/datasets 可用，且未知 task_id 的 GET /v1/tasks/{id} 返回 not_found。

    参数:
        client: 上面的 TestClient 夹具。
    """
    # 数据集列表：空库也应正常 200（空列表），不抛异常
    assert client.get("/v1/datasets", headers=AUTH).status_code == 200
    # 未知任务：task_status 以 not_found 状态明示，而非 404 或异常
    r = client.get("/v1/tasks/missing", headers=AUTH)
    assert r.json()["status"] == "not_found"


def test_forget_via_rest(client):
    """验证 DELETE /v1/memories 全清保护：无 confirm 时 400，confirm=true 时清空全部。

    参数:
        client: 上面的 TestClient 夹具（FakeLLM 已预设 LIGHT 抽取响应）。
    """
    # 先写入一条记忆，让全清有实际对象（FakeLLM 消费 LIGHT 响应）
    client.post("/v1/memories", json={"content": "用户喜欢浅烘焙咖啡"}, headers=AUTH)
    # 无 confirm 查询参数：forget(dataset=None, confirm=False) 抛 ValueError -> 400
    r = client.request("DELETE", "/v1/memories", headers=AUTH)  # 无 confirm
    assert r.status_code == 400
    # 显式 confirm=true：执行全清，返回 scope="all"
    r = client.delete("/v1/memories?confirm=true", headers=AUTH)
    assert r.status_code == 200 and r.json()["scope"] == "all"
