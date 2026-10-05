"""Read-only retrieval API. Corpus administration stays in the offline CLI."""

import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings
from .http_limits import RequestBodyLimit
from .models import validate_textbook_id
from .search import SearchEngine
from .service import create_engine

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=100, strict=True)
    textbook_id: str | None = Field(default=None, max_length=100)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must contain nonempty text")
        return value.strip()

    @field_validator("textbook_id")
    @classmethod
    def valid_id(cls, value: str | None) -> str | None:
        return validate_textbook_id(value) if value is not None else None


class Passage(BaseModel):
    chunk_id: str
    textbook_id: str
    textbook_title: str
    chapter: str
    page: int
    content: str
    score: float
    citation: str


class SearchResponse(BaseModel):
    results: list[Passage]
    request_id: str


def create_app(settings: Settings | None = None, engine: SearchEngine | None = None) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not config.allow_unauthenticated and len(config.api_key) < 32:
            raise RuntimeError("MEDICAL_RAG_API_KEY must be at least 32 characters")
        app.state.engine = engine or await run_in_threadpool(create_engine, config)
        await run_in_threadpool(app.state.engine.build_index)
        yield

    app = FastAPI(
        title="Medical RAG Engine",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def authenticate(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if config.allow_unauthenticated:
            return
        if x_api_key is None or not secrets.compare_digest(
            x_api_key.encode(), config.api_key.encode()
        ):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        # Exception messages and tracebacks can include extracted text or queries.
        logger.error("Retrieval failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "Retrieval failed", "request_id": request_id},
            headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
        )

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def ready(request: Request) -> dict:
        if not request.app.state.engine.build_index():
            raise HTTPException(status_code=503, detail="Corpus is empty")
        return {"status": "ready"}

    @app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(authenticate)])
    def search(payload: SearchRequest, request: Request) -> dict:
        results = request.app.state.engine.search(
            payload.query,
            payload.top_k,
            payload.textbook_id,
        )
        return {
            "results": [result.to_dict() for result in results],
            "request_id": request.state.request_id,
        }

    app.add_middleware(RequestBodyLimit)
    return app
