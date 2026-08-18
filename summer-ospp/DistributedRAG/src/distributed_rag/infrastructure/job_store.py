from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, UniqueConstraint, create_engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ..config import DatabaseConfig
from ..domain.models import JobStatus, StageName, StageStatus


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "rag_jobs"
    __table_args__ = (UniqueConstraint("document_version", "pipeline_version", name="uq_job_version_pipeline"),)

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    document_version: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    pipeline_version: Mapped[str] = mapped_column(String(80), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=JobStatus.PENDING.value)
    error_kind: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class StageRecord(Base):
    __tablename__ = "rag_job_stages"
    __table_args__ = (UniqueConstraint("job_id", "stage_name", name="uq_job_stage"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stage_name: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=StageStatus.PENDING.value)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_kind: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class DocumentVersionRecord(Base):
    __tablename__ = "rag_document_versions"

    document_version: Mapped[str] = mapped_column(String(80), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    parser_version: Mapped[str] = mapped_column(String(80), nullable=False)
    chunking_version: Mapped[str] = mapped_column(String(80), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(120), nullable=False)
    published: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class JobStore:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        engine_args: Dict[str, Any] = {"pool_pre_ping": True}
        if not config.url.startswith("sqlite"):
            engine_args["pool_size"] = config.pool_size
        self.engine = create_engine(config.url, **engine_args)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.Session.begin() as session:
            yield session

    def create_or_get_job(self, document_id: str, document_version: str, pipeline_version: str, source_uri: str, trace_id: Optional[str] = None) -> JobRecord:
        try:
            with self.session() as session:
                existing = session.scalar(
                    select(JobRecord).where(
                        JobRecord.document_version == document_version,
                        JobRecord.pipeline_version == pipeline_version,
                    )
                )
                if existing:
                    return existing
                record = JobRecord(
                    job_id=f"job_{uuid.uuid4().hex}",
                    trace_id=trace_id or f"trace_{uuid.uuid4().hex}",
                    document_id=document_id,
                    document_version=document_version,
                    pipeline_version=pipeline_version,
                    source_uri=source_uri,
                    status=JobStatus.PENDING.value,
                )
                session.add(record)
                session.flush()
                return record
        except IntegrityError:
            with self.Session() as session:
                existing = session.scalar(select(JobRecord).where(
                    JobRecord.document_version == document_version,
                    JobRecord.pipeline_version == pipeline_version,
                ))
                if not existing:
                    raise
                return existing

    def register_document(self, *, document_id: str, document_version: str, checksum: str, source_uri: str, parser_version: str, chunking_version: str, embedding_version: str) -> None:
        try:
            with self.session() as session:
                if session.get(DocumentVersionRecord, document_version):
                    return
                session.add(DocumentVersionRecord(
                    document_id=document_id,
                    document_version=document_version,
                    checksum=checksum,
                    source_uri=source_uri,
                    parser_version=parser_version,
                    chunking_version=chunking_version,
                    embedding_version=embedding_version,
                ))
        except IntegrityError:
            return

    def start_job(self, job_id: str) -> None:
        self._set_job(job_id, status=JobStatus.RUNNING.value, error_kind=None, error_message=None)

    def finish_job(self, job_id: str, chunk_count: int) -> None:
        with self.session() as session:
            job = session.get(JobRecord, job_id)
            if not job:
                raise KeyError(job_id)
            version = session.get(DocumentVersionRecord, job.document_version)
            if not version:
                raise KeyError(job.document_version)
            session.execute(
                update(DocumentVersionRecord)
                .where(
                    DocumentVersionRecord.document_id == job.document_id,
                    DocumentVersionRecord.document_version != job.document_version,
                )
                .values(published=False)
            )
            version.published = True
            version.chunk_count = chunk_count
            version.published_at = _now()
            job.status = JobStatus.SUCCEEDED.value
            job.updated_at = _now()

    def fail_job(self, job_id: str, error_kind: str, error_message: str) -> None:
        self._set_job(job_id, status=JobStatus.FAILED.value, error_kind=error_kind, error_message=error_message[:4000])

    def request_cancel(self, job_id: str) -> None:
        self._set_job(job_id, status=JobStatus.CANCEL_REQUESTED.value)

    def mark_cancelled(self, job_id: str) -> None:
        self._set_job(job_id, status=JobStatus.CANCELLED.value)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self.Session() as session:
            record = session.get(JobRecord, job_id)
            return bool(record and record.status == JobStatus.CANCEL_REQUESTED.value)

    def heartbeat(self, job_id: str) -> None:
        self._set_job(job_id)

    def is_stale(self, job_id: str) -> bool:
        with self.Session() as session:
            record = session.get(JobRecord, job_id)
            if not record:
                raise KeyError(job_id)
            updated_at = record.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age = (_now() - updated_at).total_seconds()
            return age > self.config.stale_job_seconds

    def stage_started(self, job_id: str, stage: StageName) -> None:
        with self.session() as session:
            record = self._get_or_create_stage(session, job_id, stage)
            record.status = StageStatus.RUNNING.value
            record.started_at = _now()
            record.finished_at = None
            record.error_kind = None
            record.error_message = None

    def stage_retry(self, job_id: str, stage: StageName) -> None:
        with self.session() as session:
            record = self._get_or_create_stage(session, job_id, stage)
            record.retry_count += 1

    def stage_finished(self, job_id: str, stage: StageName) -> None:
        with self.session() as session:
            record = self._get_or_create_stage(session, job_id, stage)
            record.status = StageStatus.SUCCEEDED.value
            record.finished_at = _now()

    def stage_failed(self, job_id: str, stage: StageName, error_kind: str, error_message: str) -> None:
        with self.session() as session:
            record = self._get_or_create_stage(session, job_id, stage)
            record.status = StageStatus.FAILED.value
            record.finished_at = _now()
            record.error_kind = error_kind
            record.error_message = error_message[:4000]

    def close_open_stages(self, job_id: str, except_stage: StageName, status: StageStatus) -> None:
        with self.session() as session:
            records = session.scalars(select(StageRecord).where(
                StageRecord.job_id == job_id,
                StageRecord.status == StageStatus.RUNNING.value,
                StageRecord.stage_name != except_stage.value,
            )).all()
            for record in records:
                record.status = status.value
                record.finished_at = _now()

    def retry_failed_stages(self, job_id: str) -> None:
        with self.session() as session:
            records = session.scalars(select(StageRecord).where(
                StageRecord.job_id == job_id,
                StageRecord.status == StageStatus.FAILED.value,
            )).all()
            for record in records:
                record.retry_count += 1
                record.status = StageStatus.PENDING.value
                record.error_kind = None
                record.error_message = None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.Session() as session:
            job = session.get(JobRecord, job_id)
            if not job:
                return None
            stages = session.scalars(select(StageRecord).where(StageRecord.job_id == job_id).order_by(StageRecord.id)).all()
            version = session.get(DocumentVersionRecord, job.document_version)
            return {
                "job_id": job.job_id,
                "trace_id": job.trace_id,
                "document_id": job.document_id,
                "document_version": job.document_version,
                "status": job.status,
                "error_kind": job.error_kind,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "published": bool(version and version.published),
                "chunk_count": version.chunk_count if version else 0,
                "stages": [
                    {
                        "stage": item.stage_name,
                        "status": item.status,
                        "retry_count": item.retry_count,
                        "started_at": item.started_at.isoformat() if item.started_at else None,
                        "finished_at": item.finished_at.isoformat() if item.finished_at else None,
                        "error_kind": item.error_kind,
                        "error_message": item.error_message,
                    }
                    for item in stages
                ],
            }

    def published_versions(self, document_ids: Optional[List[str]] = None) -> List[str]:
        with self.Session() as session:
            query = select(DocumentVersionRecord.document_version).where(DocumentVersionRecord.published.is_(True))
            if document_ids:
                query = query.where(DocumentVersionRecord.document_id.in_(document_ids))
            return list(session.scalars(query).all())

    def health(self) -> bool:
        with self.Session() as session:
            session.execute(select(1))
        return True

    def _set_job(self, job_id: str, **updates: Any) -> None:
        with self.session() as session:
            record = session.get(JobRecord, job_id)
            if not record:
                raise KeyError(job_id)
            for key, value in updates.items():
                setattr(record, key, value)
            record.updated_at = _now()

    @staticmethod
    def _get_or_create_stage(session: Session, job_id: str, stage: StageName) -> StageRecord:
        record = session.scalar(select(StageRecord).where(StageRecord.job_id == job_id, StageRecord.stage_name == stage.value))
        if record:
            return record
        record = StageRecord(job_id=job_id, stage_name=stage.value, status=StageStatus.PENDING.value)
        session.add(record)
        session.flush()
        return record
