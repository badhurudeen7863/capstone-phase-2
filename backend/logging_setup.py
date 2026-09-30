"""
Logging configuration (report section 3.5.1 - "centralised logging").

* console handler  -> uvicorn / streamlit terminal
* rotating file    -> logs/server.log
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import LOG_DIR, LOG_FILE, LOG_LEVEL

_CONFIGURED = False


def setup_logging() -> None:
    """Idempotently configure root + app loggers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.addHandler(console)
    root.addHandler(file_handler)

    # keep third-party loggers quiet-ish
    for noisy in ("uvicorn.access", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
