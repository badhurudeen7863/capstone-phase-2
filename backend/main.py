"""
ExpenseIQ - FastAPI application entry point.

Run from the repository root:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from .config import API_V1_PREFIX, PROJECT_NAME
from .database import SessionLocal, init_db
from .logging_setup import get_logger
from .models.db_models import Category
from .routers import analytics, auth, budgets, categories, expenses, forecasting
from .services.ml_service import ml_service

logger = get_logger(__name__)

SEED_CATEGORIES = [
    ("Rent", "Fixed"),
    ("Utilities", "Fixed"),
    ("Food & Groceries", "Variable"),
    ("Health", "Variable"),
    ("Transport", "Variable"),
    ("Entertainment", "Variable"),
    ("Other", "Variable"),
]


def seed_categories() -> None:
    db = SessionLocal()
    try:
        if db.query(Category).count() == 0:
            db.add_all([Category(name=n, type=t) for n, t in SEED_CATEGORIES])
            db.commit()
            logger.info("Seeded %d default categories.", len(SEED_CATEGORIES))
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_categories()
    ml_service.load()
    logger.info("ExpenseIQ API started.")
    yield
    logger.info("ExpenseIQ API shutting down.")


app = FastAPI(
    title=PROJECT_NAME,
    version="2.0.0",
    description="AI-based monthly expenses prediction & forecasting (XGBoost) with "
    "JWT auth, MySQL/SQLite persistence, CSV bulk upload, anomaly detection "
    "and a predictions audit log.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    return JSONResponse(status_code=400, content={"detail": "Database constraint violated"})


for router in (auth.router, categories.router, expenses.router, analytics.router, forecasting.router, budgets.router):
    app.include_router(router, prefix=API_V1_PREFIX)


@app.get("/healthz", tags=["system"])
def healthz() -> dict:
    return {
        "status": "ok",
        "model_loaded": ml_service.is_loaded,
        "model_version": ml_service.model_version if ml_service.is_loaded else None,
        "model_metrics": ml_service.deployed_metrics if ml_service.is_loaded else None,
    }


@app.get("/", tags=["system"])
def root() -> dict:
    return {"name": PROJECT_NAME, "docs": "/docs", "health": "/healthz"}
