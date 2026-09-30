"""
Central configuration for the ExpenseIQ backend.

Everything is environment-driven so the same code base runs:
  * out-of-the-box in VS Code with zero setup (SQLite fallback), and
  * against a real MySQL 8 server (as specified in the project report)
    by setting DATABASE_URL in .env, e.g.
    DATABASE_URL=mysql+pymysql://root:password@localhost:3306/expenseiq
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repository root (the folder that contains backend/, frontend/, data/, ...)
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# ---------------------------------------------------------------- database ---
# Default: SQLite file inside data/ so the project runs with no DB server.
# The report's production target is MySQL 8 -> set DATABASE_URL in .env.
DATABASE_URL = _env(
    "DATABASE_URL",
    f"sqlite:///{(BASE_DIR / 'data' / 'expenseiq.db').as_posix()}",
)

# -------------------------------------------------------------------- auth ---
SECRET_KEY = _env("SECRET_KEY", "expenseiq-dev-secret-change-me-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(_env("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))

# ---------------------------------------------------------------------- ml ---
MODEL_PATH = Path(_env("MODEL_PATH", str(BASE_DIR / "ml_models" / "xgboost_expense_forecaster_v2.pkl")))
METRICS_PATH = Path(_env("METRICS_PATH", str(BASE_DIR / "ml_models" / "model_metrics.json")))

# ------------------------------------------------------- forecasting rules ---
MIN_HISTORY_DAYS = int(_env("MIN_HISTORY_DAYS", "30"))     # >= 30 days of history required to predict
FORECAST_WINDOW_DAYS = int(_env("FORECAST_WINDOW_DAYS", "90"))  # last 90 days used for features
ANOMALY_SIGMA = float(_env("ANOMALY_SIGMA", "3.0"))        # value > roll30_mean + 3 * roll30_std
CONFIDENCE_Z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}      # CI = pred +/- z * RMSE

# ------------------------------------------------------------------ logging --
LOG_DIR = Path(_env("LOG_DIR", str(BASE_DIR / "logs")))
LOG_FILE = LOG_DIR / "server.log"
LOG_LEVEL = _env("LOG_LEVEL", "INFO")

# --------------------------------------------------------------------- api ---
API_V1_PREFIX = "/api/v1"
PROJECT_NAME = "ExpenseIQ - AI-Based Monthly Expenses Prediction and Forecasting"
