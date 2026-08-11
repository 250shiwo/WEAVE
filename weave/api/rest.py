"""REST 开放 API (spec §7.2): 薄壳, 直接委托 MemoryService.

路由层不做任何业务逻辑：请求体经 pydantic 模型校验后原样透传给
app.state.service（MemoryService 门面），返回值直接序列化为 JSON 响应；
业务校验失败（ValueError）由 app 层注册的异常处理器统一转 400 错误信封。
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

# 全部端点统一挂在 /v1 前缀下（create_app 中 include_router 时生效）
router = APIRouter(prefix="/v1")


class RememberIn(BaseModel):
    """POST /v1/memories 请求体：一条待写入的记忆文本。

    字段:
        content: 待记忆的原始文本（必填）。
        dataset: 目标数据集名，默认 "default"。
        session_id: 会话 ID；缺省 None 走永久管线，非 None 走会话暂存分支。
    """

    content: str
    dataset: str = "default"
    session_id: str | None = None


class RecallIn(BaseModel):
    """POST /v1/recall 请求体：一次混合检索查询。

    字段:
        query: 查询文本（必填）。
        dataset: 可选数据集过滤；缺省 None 表示跨全部数据集。
        top_k: 双表各自的向量入口条数上限，默认 5。
        session_id: 可选会话标识；非空时叠加该会话的缓存原文。
    """

    query: str
    dataset: str | None = None
    top_k: int = 5
    session_id: str | None = None


class DocumentIn(BaseModel):
    """POST /v1/documents 请求体：一份 base64 编码的待摄取文档。

    字段:
        file_name: 原始文件名（v1 仅支持 txt/md/markdown 扩展名）。
        content_base64: 文件内容的 base64 编码（解码后须为 UTF-8 文本）。
        dataset: 目标数据集名，默认 "default"。
    """

    file_name: str
    content_base64: str
    dataset: str = "default"


def _svc(request: Request):
    """从应用状态取出 MemoryService 门面（create_app 启动时注入）。

    参数:
        request: 当前请求对象，经 request.app 拿到 FastAPI 应用实例。
    返回:
        MemoryService: app.state.service 上挂载的服务门面。
    """
    return request.app.state.service


@router.get("/health")
async def health():
    """健康检查端点（认证豁免，供探活使用）。

    参数: 无。
    返回:
        dict: 固定 {"status": "ok"}。
    """
    return {"status": "ok"}


@router.post("/memories")
async def remember(body: RememberIn, request: Request):
    """写入一条记忆：无 session_id 走永久管线，有 session_id 走会话暂存队列。

    参数:
        body: RememberIn 请求体（content/dataset/session_id）。
        request: 请求对象，用于取服务门面。
    返回:
        dict: MemoryService.remember 的结果（永久含 entities 等统计，
            会话含 task_id 与 queued=True）。
    """
    # 参数按位置透传，与服务层签名 remember(content, dataset, session_id) 对齐
    return await _svc(request).remember(body.content, body.dataset, body.session_id)


@router.post("/recall")
async def recall(body: RecallIn, request: Request):
    """回忆查询：向量入口 + 图扩展 + 会话叠加的混合检索。

    参数:
        body: RecallIn 请求体（query/dataset/top_k/session_id）。
        request: 请求对象，用于取服务门面。
    返回:
        dict: {facts, chunks, session_items} 检索结果。
    """
    # 参数按位置透传，与服务层签名 recall(query, dataset, top_k, session_id) 对齐
    return await _svc(request).recall(body.query, body.dataset, body.top_k, body.session_id)


@router.post("/documents", status_code=202)
async def cognify_file(body: DocumentIn, request: Request):
    """提交一份文档进入异步 cognify 管线：校验落记录后入队，202 表示已受理。

    参数:
        body: DocumentIn 请求体（file_name/content_base64/dataset）。
        request: 请求对象，用于取服务门面。
    返回:
        dict: {task_id, data_id, status}；去重命中时 status="completed"。
    异常:
        ValueError: 扩展名不支持或 base64 解码失败（由 app 异常处理器转 400）。
    """
    # 参数按位置透传，与服务层签名 cognify_submit(file_name, content_base64, dataset) 对齐
    return await _svc(request).cognify_submit(body.file_name, body.content_base64, body.dataset)


@router.get("/tasks/{task_id}")
async def task_status(task_id: str, request: Request):
    """查询队列任务运行状态（pipeline_run 的只读视图）。

    参数:
        task_id: 路径参数，队列任务 ID。
        request: 请求对象，用于取服务门面。
    返回:
        dict: 完整运行记录；未知任务返回 {"task_id": ..., "status": "not_found"}。
    """
    return await _svc(request).task_status(task_id)


@router.get("/datasets")
async def list_datasets(request: Request):
    """汇总全部数据集统计（数据条目数/边数/实体数）。

    参数:
        request: 请求对象，用于取服务门面。
    返回:
        list[dict]: [{name, data_count, edge_count, entity_count}, ...]。
    """
    return await _svc(request).list_datasets()


@router.delete("/datasets/{name}")
async def forget_dataset(name: str, request: Request):
    """清空指定数据集：SQLite 关联/边/孤立 data + Kuzu 节点边 + LanceDB 向量行。

    参数:
        name: 路径参数，待清空的数据集名。
        request: 请求对象，用于取服务门面。
    返回:
        dict: {"scope": 数据集名, "data", "edges"} 清理统计。
    """
    # 关键字传参明确语义：只清这一个数据集，confirm 不参与（单数据集非高危）
    return await _svc(request).forget(dataset=name)


@router.delete("/memories")
async def forget_all(request: Request, confirm: bool = False):
    """清空全部记忆（高危操作）：必须显式传 ?confirm=true 才执行。

    参数:
        request: 请求对象，用于取服务门面。
        confirm: 查询参数，全清确认开关；缺省 False 时服务层抛 ValueError -> 400。
    返回:
        dict: {"scope": "all", "datasets": [...]} 清理统计。
    异常:
        ValueError: confirm 为 False 时抛出（由 app 异常处理器转 400）。
    """
    # dataset=None 即全清语义；confirm 原样透传给服务层做高危确认
    return await _svc(request).forget(dataset=None, confirm=confirm)
