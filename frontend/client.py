"""Thin HTTP client used by the Streamlit dashboard to talk to the FastAPI backend."""
from __future__ import annotations

import os
from typing import Any

import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000") + "/api/v1"


class APIError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


class APIClient:
    def __init__(self, token: str | None = None):
        self.token = token

    # ------------------------------------------------------------------ core
    def _req(self, method: str, path: str, **kwargs) -> Any:
        headers = kwargs.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = requests.request(method, API_BASE + path, headers=headers, timeout=60, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(resp.status_code, detail)
        return resp.json() if resp.status_code != 204 else None

    # ------------------------------------------------------------------ auth
    def login(self, email: str, password: str) -> dict:
        return self._req("POST", "/auth/login", json={"email": email, "password": password})

    def register(self, name: str, email: str, password: str) -> dict:
        return self._req("POST", "/auth/register", json={"name": name, "email": email, "password": password})

    def me(self) -> dict:
        return self._req("GET", "/auth/me")

    # ------------------------------------------------------------ master data
    def categories(self) -> list[dict]:
        return self._req("GET", "/categories")

    # ------------------------------------------------------------ transactions
    def list_expenses(self, **params) -> list[dict]:
        return self._req("GET", "/expenses", params=params)

    def create_expense(self, payload: dict) -> dict:
        return self._req("POST", "/expenses", json=payload)

    def update_expense(self, txn_id: int, payload: dict) -> dict:
        return self._req("PUT", f"/expenses/{txn_id}", json=payload)

    def delete_expense(self, txn_id: int) -> None:
        return self._req("DELETE", f"/expenses/{txn_id}")

    def upload_csv(self, file_bytes: bytes, filename: str) -> dict:
        return self._req(
            "POST",
            "/expenses/upload-csv",
            files={"file": (filename, file_bytes, "text/csv")},
        )

    # --------------------------------------------------------------- analytics
    def summary(self) -> dict:
        return self._req("GET", "/analytics/summary")

    def monthly(self) -> list[dict]:
        return self._req("GET", "/analytics/monthly")

    def category_summary(self) -> list[dict]:
        return self._req("GET", "/analytics/categories")

    def category_month(self) -> list[dict]:
        return self._req("GET", "/analytics/category-month")

    def anomalies(self) -> list[dict]:
        return self._req("GET", "/analytics/anomalies")

    def spend_by_day(self) -> list[dict]:
        return self._req("GET", "/analytics/spend-by-day")

    # ------------------------------------------------------------- forecasting
    def forecast(self, user_id: int, horizon: int = 1, confidence: float = 0.95) -> dict:
        return self._req(
            "GET",
            f"/forecasting/predict/{user_id}",
            params={"horizon": horizon, "confidence": confidence},
        )

    def forecast_history(self) -> list[dict]:
        return self._req("GET", "/forecasting/history")
