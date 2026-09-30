"""
Shared pytest fixtures.

Isolates every test run: temporary SQLite DB, temporary model artefact
(a tiny XGBoost trained on synthetic series so the API->ML handoff is real).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = ROOT / ".pytest_tmp"
_TMP.mkdir(exist_ok=True)

import os  # noqa: E402

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["MODEL_PATH"] = str(_TMP / "model.pkl")
os.environ["METRICS_PATH"] = str(_TMP / "metrics.json")
os.environ["LOG_DIR"] = str(_TMP / "logs")
os.environ["SECRET_KEY"] = "test-secret"

import joblib  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402
from backend.services.feature_engineering import (  # noqa: E402
    FEATURE_NAMES,
    build_daily_series,
    make_training_set,
)


def _synthetic_daily(user_scale: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    base = 800 * user_scale
    season = 1 + 0.35 * np.sin(2 * np.pi * (idx.month - 3) / 12)
    festival = np.where(idx.month.isin([10, 11]), 1.3, 1.0)
    weekly = np.where(idx.dayofweek.isin([4, 5, 6]), 1.5, 0.7)
    amounts = base * season * festival * weekly * (1 + rng.normal(0, 0.15, len(idx)))
    s = pd.Series(np.maximum(amounts, 0.0), index=idx, name="amount")
    # a couple of spikes so winsorisation/anomaly paths execute
    s.iloc[400] += 9000 * user_scale
    s.iloc[900] += 7000 * user_scale
    return s


def _train_tiny_model() -> None:
    from xgboost import XGBRegressor

    daily = {1: _synthetic_daily(1.0, 1), 2: _synthetic_daily(1.3, 2)}
    X, y, meta = make_training_set(daily)
    model = XGBRegressor(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=0, verbosity=0)
    model.fit(X, y)
    pred = model.predict(X)
    base = X["rolling_mean_30"].clip(lower=1.0) * X["days_in_target_month"]
    tot = pred * base
    yt = meta["y_total"].to_numpy()
    rmse = float(np.sqrt(np.mean((tot - yt) ** 2)))
    r2 = float(1 - np.sum((yt - tot) ** 2) / np.sum((yt - yt.mean()) ** 2))
    artifact = {
        "model": model,
        "target_space": "ratio",
        "model_name": "XGBoost",
        "model_version": "test-v1",
        "feature_names": FEATURE_NAMES,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {"deployed": {"mae": round(rmse * 0.8, 2), "mse": round(rmse**2, 2),
                                 "rmse": round(rmse, 2), "r2": round(r2, 4)}},
    }
    joblib.dump(artifact, os.environ["MODEL_PATH"])
    import json

    Path(os.environ["METRICS_PATH"]).write_text(json.dumps(artifact["metrics"], indent=2))


_train_tiny_model()

if (_TMP / "test.db").exists():
    (_TMP / "test.db").unlink()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:   # lifespan: init_db + seed categories + load model
        yield c


@pytest.fixture(scope="session")
def auth(client):
    res = client.post(
        "/api/v1/auth/register",
        json={"name": "Test User", "email": "tester@example.com", "password": "secret123"},
    )
    assert res.status_code == 201, res.text
    body = res.json()
    return {"token": body["access_token"], "user_id": body["user"]["user_id"]}


@pytest.fixture(scope="session")
def headers(auth):
    return {"Authorization": f"Bearer {auth['token']}"}


@pytest.fixture(scope="session")
def seeded_user(client, headers, auth):
    """~14 months of transactions so forecasting has >= 30 days of history."""
    rng = np.random.default_rng(7)
    cats = {c["name"]: c["category_id"] for c in client.get("/api/v1/categories", headers=headers).json()}
    idx = pd.date_range("2025-07-01", "2026-08-25", freq="D")
    payload = []
    for d in idx:
        if d.day == 1:
            payload.append((8500.0, d, cats["Rent"], "Monthly rent"))
        if d.day == 8:
            payload.append((1800.0 + 300 * np.sin(d.month), d, cats["Utilities"], "Electricity bill"))
        if d.dayofweek == 5:
            payload.append((1400 * (1 + 0.3 * np.sin(d.month)) + rng.normal(0, 80), d,
                            cats["Food & Groceries"], "Supermarket run"))
        if d.dayofweek == 2 and rng.random() < 0.5:
            payload.append((120 + rng.normal(0, 30), d, cats["Transport"], "Fuel / commute"))
    for amount, d, cat, note in payload:
        r = client.post(
            "/api/v1/expenses",
            headers=headers,
            json={"amount": round(max(amount, 20), 2), "transaction_date": d.date().isoformat(),
                  "category_id": cat, "notes": note},
        )
        assert r.status_code == 201, r.text
    return {"user_id": auth["user_id"], "n": len(payload)}
