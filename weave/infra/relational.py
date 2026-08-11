"""SQLite 关系存储: datasets / data / dataset_data / edges / pipeline_runs.

全部为同步方法, 调用方必须经 run_db() 包装 (Global Constraints).
返回值为 dict 而非 ORM 对象, 避免跨线程 session 绑定问题.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, Integer, String, Text, create_engine, delete, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


def _utcnow() -> str:
    """生成当前 UTC 时间的 ISO 8601 字符串。

    做什么: 为各表的 created_at / updated_at 字段提供统一格式的时间戳，
        保证全库时间格式一致且带时区信息。
    参数: 无。
    返回:
        str: 带 +00:00 时区的 ISO 格式时间字符串，如 "2026-08-10T00:00:00+00:00"。
    """
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """生成一个新的随机主键 ID。

    做什么: 为需要内部生成主键的行（如 datasets）提供全局唯一 ID。
    参数: 无。
    返回:
        str: 32 位十六进制 UUID 字符串（无连字符）。
    """
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类：集中登记五张表的元数据，供 create_all 统一建表。"""


class DatasetRow(Base):
    """datasets 表：一个数据集（如 default）承载一批记忆数据与图谱边。

    字段:
        id: 主键，内部生成的 UUID hex。
        name: 数据集名称，全局唯一，业务上按 name 定位数据集。
        description: 数据集描述，默认空串。
        created_at: 创建时间（UTC ISO 字符串）。
    """

    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    description: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(String)


class DataRow(Base):
    """data 表：一条原始记忆数据（remember 管线写入，cognify 管线消费）。

    字段:
        id: 主键，由调用方（remember 管线）指定。
        name: 数据名称/标题。
        raw_text: 原始文本内容（长度不限，用 Text）。
        content_hash: 内容哈希，加索引用于按哈希去重查询。
        status: 处理状态（created -> processing -> completed/failed），默认 created。
        source_pipeline: 来源管线名（如 remember）。
        created_at: 创建时间（UTC ISO 字符串）。
    """

    __tablename__ = "data"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    content_hash: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="created")
    source_pipeline: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(String)


class DatasetDataRow(Base):
    """dataset_data 关联表：数据集与数据条目的多对多关系，联合主键防重复关联。"""

    __tablename__ = "dataset_data"
    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    data_id: Mapped[str] = mapped_column(String, primary_key=True)


class EdgeRow(Base):
    """edges 表：图谱边的元数据（边本体存于 Kuzu 图库，此处记录版本与血缘）。

    字段:
        edge_id: 主键，与图库中的边 ID 一致。
        source_id: 起点实体 ID，加索引用于按实体查其最新边。
        target_id: 终点实体 ID。
        relationship_type: 关系类型（如 LIKES）。
        dataset: 所属数据集名（按业务键 name 关联，随数据集级联删除）。
        version: 版本号，从 1 开始，supersede 时递增。
        is_latest: 是否为当前最新版本（旧版本行保留用于追溯）。
        source_pipeline / source_task: 写入该边的管线与任务名（血缘追踪）。
        created_at / updated_at: 创建/最后更新时间（UTC ISO 字符串）。
    """

    __tablename__ = "edges"
    edge_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, index=True)
    target_id: Mapped[str] = mapped_column(String)
    relationship_type: Mapped[str] = mapped_column(String)
    dataset: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True)
    source_pipeline: Mapped[str] = mapped_column(String, default="")
    source_task: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(String)
    updated_at: Mapped[str] = mapped_column(String)


class PipelineRunRow(Base):
    """pipeline_runs 表：一次管线执行的生命周期记录（pending -> running -> completed/failed）。

    字段:
        task_id: 主键，队列任务 ID。
        pipeline_name: 管线名（remember/cognify/improve）。
        data_id: 关联的数据条目 ID，可为空串（不关联具体数据时）。
        status: 执行状态，默认 pending。
        error: 失败时的错误信息，无错误为空串（可能较长，用 Text）。
        created_at / updated_at: 创建/最后更新时间（UTC ISO 字符串）。
    """

    __tablename__ = "pipeline_runs"
    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    pipeline_name: Mapped[str] = mapped_column(String)
    data_id: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String)
    updated_at: Mapped[str] = mapped_column(String)


def _to_dict(row) -> dict:
    """把 ORM 行对象转换为普通 dict。

    做什么: 按表的列定义取出全部列值组成 dict；调用方拿到的结果与 session 解绑，
        可安全跨线程传递（配合 run_db 在线程池中执行的调用方式，避免懒加载触雷）。
    参数:
        row: 任意 ORM 行对象（DatasetRow/DataRow/EdgeRow/PipelineRunRow 等）。
    返回:
        dict: {列名: 列值} 的普通字典。
    """
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


class RelationalStore:
    """SQLite 关系存储门面：封装五张表的全部读写操作。

    所有方法均为同步阻塞实现，自身不做任何线程/异步处理；
    并发安全由调用方通过 run_db() 的单线程执行器保证。
    """

    def __init__(self, path: str):
        """打开（不存在则创建）SQLite 数据库并建好全部五张表。

        参数:
            path: SQLite 数据库文件路径；父目录不存在时自动创建。
        """
        # 先确保数据库文件的父目录存在，否则 SQLite 打开时报 unable to open database file
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：连接实际在 run_db 的 DB 线程中使用，与创建线程不同，
        # 必须关闭 SQLite 默认的同线程校验；串行安全由 run_db 单线程执行器保证
        self.engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False}
        )
        # 按 ORM 元数据建表：已存在的表自动跳过，重复构造实例幂等
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        """关闭引擎并释放连接池占用的文件句柄。

        参数: 无。
        返回: 无。
        """
        self.engine.dispose()

    # ---------- datasets ----------
    def get_or_create_dataset(self, name: str, description: str = "") -> dict:
        """按名称获取数据集，不存在则创建（同名重复调用幂等）。

        参数:
            name: 数据集名称（全库唯一）。
            description: 数据集描述，仅在首次创建时写入。
        返回:
            dict: 数据集行 {id, name, description, created_at}。
        """
        with Session(self.engine) as s:
            # 先按唯一名查询：命中则直接复用，保证重复调用返回同一 id
            row = s.execute(select(DatasetRow).where(DatasetRow.name == name)).scalar_one_or_none()
            if row is None:
                # 未命中：创建新行并立即提交，使其他调用方随后可查
                row = DatasetRow(id=_new_id(), name=name, description=description, created_at=_utcnow())
                s.add(row)
                s.commit()
            return _to_dict(row)

    def list_datasets(self) -> list[dict]:
        """列出全部数据集。

        参数: 无。
        返回:
            list[dict]: 数据集行 dict 列表；无任何数据集时为空列表。
        """
        with Session(self.engine) as s:
            return [_to_dict(r) for r in s.execute(select(DatasetRow)).scalars().all()]

    def dataset_stats(self) -> list[dict]:
        """统计每个数据集下的数据条目数与边数。

        参数: 无。
        返回:
            list[dict]: [{name, data_count, edge_count}, ...]；
                data_count 经 dataset_data 关联表计数，edge_count 按 edges.dataset 名计数。
        """
        with Session(self.engine) as s:
            stats = []
            # 逐数据集分别计数（数据量小，N+1 查询可接受，逻辑更直白）
            for ds in s.execute(select(DatasetRow)).scalars().all():
                # 数据条目数：关联表中该数据集 id 的行数
                data_count = s.execute(
                    select(func.count()).select_from(DatasetDataRow).where(DatasetDataRow.dataset_id == ds.id)
                ).scalar_one()
                # 边数：edges 表按数据集名（业务键）计数
                edge_count = s.execute(
                    select(func.count()).select_from(EdgeRow).where(EdgeRow.dataset == ds.name)
                ).scalar_one()
                stats.append({"name": ds.name, "data_count": data_count, "edge_count": edge_count})
            return stats

    def delete_dataset_rows(self, name: str) -> dict:
        """按数据集名级联删除其关联行：dataset_data 关联、edges 边、孤立 data 行及数据集本身。

        做什么: 先收集该数据集关联的 data_id 列表并删除关联行；再按数据集名删除全部边；
            对每个 data_id 检查是否仍被其他数据集引用，无任何引用才删除 data 行
            （多对多共享的数据不被误删）；最后删除数据集行并整体提交。
        参数:
            name: 要删除的数据集名称。
        返回:
            dict: {"data": 删除的数据行数, "edges": 删除的边行数}；
                数据集不存在时返回 {"data": 0, "edges": 0}。
        """
        with Session(self.engine) as s:
            ds = s.execute(select(DatasetRow).where(DatasetRow.name == name)).scalar_one_or_none()
            if ds is None:
                # 数据集不存在：无可删内容，直接返回零计数
                return {"data": 0, "edges": 0}
            # 先收集该数据集关联的全部 data_id（删除关联行后便无从反查）
            data_ids = [r.data_id for r in s.execute(
                select(DatasetDataRow).where(DatasetDataRow.dataset_id == ds.id)
            ).scalars().all()]
            # 删除本数据集的关联行
            s.execute(delete(DatasetDataRow).where(DatasetDataRow.dataset_id == ds.id))
            # 按数据集名删除其全部边（含历史版本），rowcount 即删除数
            edge_n = s.execute(delete(EdgeRow).where(EdgeRow.dataset == name)).rowcount
            data_n = 0
            for did in data_ids:
                # 关联行已删，此处统计该 data 在剩余关联表中的引用数
                still_linked = s.execute(
                    select(func.count()).select_from(DatasetDataRow).where(DatasetDataRow.data_id == did)
                ).scalar_one()
                # 仅当不再被任何数据集引用时才删除 data 行，避免误删共享数据
                if still_linked == 0:
                    data_n += s.execute(delete(DataRow).where(DataRow.id == did)).rowcount
            # 最后删除数据集行本身，并一次性提交整个级联事务
            s.execute(delete(DatasetRow).where(DatasetRow.id == ds.id))
            s.commit()
            return {"data": data_n, "edges": edge_n}

    # ---------- data ----------
    def create_data(self, data_id, name, raw_text, content_hash, source_pipeline) -> dict:
        """写入一条数据记录，初始状态固定为 created。

        参数:
            data_id: 数据主键，由调用方指定。
            name: 数据名称/标题。
            raw_text: 原始文本内容。
            content_hash: 内容哈希，供 get_data_by_hash 去重查询。
            source_pipeline: 来源管线名（如 remember）。
        返回:
            dict: 新写入的数据行。
        """
        with Session(self.engine) as s:
            row = DataRow(id=data_id, name=name, raw_text=raw_text, content_hash=content_hash,
                          status="created", source_pipeline=source_pipeline, created_at=_utcnow())
            s.add(row)
            s.commit()
            return _to_dict(row)

    def get_data_by_hash(self, content_hash: str) -> dict | None:
        """按内容哈希查询数据（remember 去重依赖此接口）。

        参数:
            content_hash: 内容哈希字符串。
        返回:
            dict | None: 命中的数据行；不存在时返回 None。
        """
        with Session(self.engine) as s:
            row = s.execute(select(DataRow).where(DataRow.content_hash == content_hash)).scalar_one_or_none()
            return _to_dict(row) if row else None

    def get_data(self, data_id: str) -> dict | None:
        """按主键查询数据。

        参数:
            data_id: 数据主键。
        返回:
            dict | None: 命中的数据行；不存在时返回 None。
        """
        with Session(self.engine) as s:
            row = s.get(DataRow, data_id)
            return _to_dict(row) if row else None

    def set_data_status(self, data_id: str, status: str) -> None:
        """更新数据处理状态；数据不存在时静默忽略。

        参数:
            data_id: 数据主键。
            status: 新状态（如 processing/completed/failed）。
        返回: 无。
        """
        with Session(self.engine) as s:
            row = s.get(DataRow, data_id)
            if row:
                # 仅命中时更新并提交；未命中视为无操作（幂等，便于管线放心调用）
                row.status = status
                s.commit()

    def link_dataset_data(self, dataset_id: str, data_id: str) -> None:
        """建立数据集与数据的关联；重复关联幂等跳过。

        参数:
            dataset_id: 数据集主键。
            data_id: 数据主键。
        返回: 无。
        """
        with Session(self.engine) as s:
            # 按联合主键查询：已存在则不再插入，保证重复 link 幂等
            exists = s.get(DatasetDataRow, {"dataset_id": dataset_id, "data_id": data_id})
            if not exists:
                s.add(DatasetDataRow(dataset_id=dataset_id, data_id=data_id))
                s.commit()

    def is_data_linked(self, dataset_id: str, data_id: str) -> bool:
        """判断指定数据是否已关联到指定数据集（remember 去重判定依赖此接口）。

        做什么: 内容哈希命中只能说明文本曾经写入过，还必须确认该数据确实挂在
            当前数据集下，才算真正的重复写入；跨数据集的同名同文数据不视为重复。
        参数:
            dataset_id: 数据集主键。
            data_id: 数据主键。
        返回:
            bool: True 表示关联行已存在（同数据集重复），False 表示未关联。
        """
        with Session(self.engine) as s:
            # 联合主键精确查询：命中即已关联
            return s.get(DatasetDataRow, {"dataset_id": dataset_id, "data_id": data_id}) is not None

    # ---------- edges ----------
    def upsert_edge(self, edge: dict) -> None:
        """插入或更新一条边元数据（按 edge_id 判存在）。

        参数:
            edge: 边字段 dict，键须覆盖 EdgeRow 全部列：edge_id, source_id, target_id,
                relationship_type, dataset, version, is_latest, source_pipeline,
                source_task, created_at, updated_at。
        返回: 无。
        """
        with Session(self.engine) as s:
            row = s.get(EdgeRow, edge["edge_id"])
            if row is None:
                # 不存在：整包字段直接构造新行
                s.add(EdgeRow(**edge))
            else:
                # 已存在：逐字段覆盖更新
                for k, v in edge.items():
                    setattr(row, k, v)
            s.commit()

    def get_latest_edge(self, source_id: str, relationship_type: str) -> dict | None:
        """查询某实体某类关系的当前最新版本边（is_latest=True）。

        参数:
            source_id: 起点实体 ID。
            relationship_type: 关系类型。
        返回:
            dict | None: 最新边行；不存在时返回 None。
        """
        with Session(self.engine) as s:
            row = s.execute(
                select(EdgeRow).where(
                    EdgeRow.source_id == source_id,
                    EdgeRow.relationship_type == relationship_type,
                    EdgeRow.is_latest.is_(True),
                )
            ).scalar_one_or_none()
            return _to_dict(row) if row else None

    def set_edge_not_latest(self, edge_id: str, updated_at: str) -> None:
        """把指定边标记为非最新版本（supersede 旧边时调用）；边不存在时静默忽略。

        参数:
            edge_id: 边主键。
            updated_at: 本次变更时间（由调用方提供，与写入的新边时间戳保持一致）。
        返回: 无。
        """
        with Session(self.engine) as s:
            row = s.get(EdgeRow, edge_id)
            if row:
                # 翻转最新标记并记录变更时间；历史行保留用于追溯
                row.is_latest = False
                row.updated_at = updated_at
                s.commit()

    def delete_edges_by_source(self, source_pipeline: str) -> int:
        """按来源管线批量删除边元数据（含全部历史版本）。

        参数:
            source_pipeline: 来源管线名（如 remember）。
        返回:
            int: 实际删除的行数。
        """
        with Session(self.engine) as s:
            # rowcount 在 commit 前先取，避免提交后结果失效
            n = s.execute(delete(EdgeRow).where(EdgeRow.source_pipeline == source_pipeline)).rowcount
            s.commit()
            return n

    # ---------- pipeline runs ----------
    def create_pipeline_run(self, task_id: str, pipeline_name: str, data_id: str = "") -> None:
        """创建一条管线运行记录，初始状态固定为 pending。

        参数:
            task_id: 队列任务 ID（主键）。
            pipeline_name: 管线名（remember/cognify/improve）。
            data_id: 关联的数据条目 ID，可选，默认空串。
        返回: 无。
        """
        with Session(self.engine) as s:
            s.add(PipelineRunRow(task_id=task_id, pipeline_name=pipeline_name, data_id=data_id,
                                 status="pending", created_at=_utcnow(), updated_at=_utcnow()))
            s.commit()

    def update_pipeline_run(self, task_id: str, status: str, error: str = "") -> None:
        """更新管线运行状态与错误信息；task_id 不存在时静默忽略。

        参数:
            task_id: 队列任务 ID。
            status: 新状态（running/completed/failed）。
            error: 错误信息，成功或无错误时为空串。
        返回: 无。
        """
        with Session(self.engine) as s:
            row = s.get(PipelineRunRow, task_id)
            if row:
                # 更新状态、错误信息并刷新 updated_at
                row.status = status
                row.error = error
                row.updated_at = _utcnow()
                s.commit()

    def get_pipeline_run(self, task_id: str) -> dict | None:
        """按 task_id 查询管线运行记录。

        参数:
            task_id: 队列任务 ID。
        返回:
            dict | None: 命中的运行记录行；不存在时返回 None。
        """
        with Session(self.engine) as s:
            row = s.get(PipelineRunRow, task_id)
            return _to_dict(row) if row else None
