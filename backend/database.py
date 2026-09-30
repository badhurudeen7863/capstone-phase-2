"""
SQLAlchemy engine / session management (report: database.py with pooling).

Works with SQLite (zero-setup default) and MySQL 8 (project target) through
the DATABASE_URL environment variable.
"""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import DATABASE_URL
from .logging_setup import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_connect_args = {}
_engine_kwargs: dict = {"pool_pre_ping": True}

if DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False
    logger.info("Using SQLite database (zero-setup mode): %s", DATABASE_URL)
else:  # MySQL 8 as per project report - connection pooling
    _engine_kwargs.update(pool_size=10, max_overflow=20, pool_recycle=1800)
    logger.info("Using MySQL database: %s", DATABASE_URL.split("@")[-1])

engine = create_engine(DATABASE_URL, connect_args=_connect_args, **_engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (idempotent)."""
    from .models import db_models  # noqa: F401  (register mappers)

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured (%d tables).", len(Base.metadata.tables))
