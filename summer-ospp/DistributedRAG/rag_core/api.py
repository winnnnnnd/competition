from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from .observability import refresh_runtime_metrics
from .service import DistributedRAGService


@lru_cache(maxsize=1)
def get_service() -> DistributedRAGService:
    return DistributedRAGService()


def create_app() -> FastAPI:
    app = FastAPI(title="DistributedRAG API", version="2.0.0")

    @app.get("/health")
    def health():
        result = get_service().health()
        if not result["healthy"]:
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.get("/metrics")
    def metrics():
        refresh_runtime_metrics()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/documents")
    async def upload_document(file: UploadFile = File(...)):
        content = await file.read()
        return await run_in_threadpool(
            get_service().submit_ingestion,
            file.filename or "document.bin",
            content,
            {"content_type": file.content_type or ""},
        )

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        value = get_service().job(job_id)
        if not value:
            raise HTTPException(status_code=404, detail="Job not found")
        return value

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return get_service().cancel(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/v1/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        try:
            return get_service().retry(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/v1/query")
    def query(question: str = Form(...), document_ids: Optional[List[str]] = Form(None), use_hyde: bool = Form(True)):
        return get_service().ask(question, document_ids=document_ids, use_hyde=use_hyde)

    return app
