"""Report end-to-end flow: register -> 3 months bulk data -> forecast -> audit log."""
from __future__ import annotations

import io
import random
from datetime import date, timedelta


def _three_month_statement() -> bytes:
    rng = random.Random(11)
    lines = ["Txn Date,Narration,Debit"]
    start = date(2026, 5, 1)
    for i in range(120):
        d = start + timedelta(days=rng.randint(0, 89))
        kind = rng.choice(
            ["UBER TRIP", "SWIGGY PAYMENT", "TNEB ELECTRICITY", "HOUSE RENT PAYMENT",
             "APOLLO PHARMACY", "PVR CINEMAS", "BIGBASKET SUPERMARKET"]
        )
        amt = round(rng.uniform(90, 2600), 2)
        lines.append(f"{d.strftime('%d/%m/%Y')},{kind},{amt}")
    return "\n".join(lines).encode()


def test_end_to_end_flow(client):
    # 1. register
    reg = client.post(
        "/api/v1/auth/register",
        json={"name": "E2E User", "email": "e2e@example.com", "password": "secret123"},
    )
    assert reg.status_code == 201
    token = reg.json()["access_token"]
    uid = reg.json()["user"]["user_id"]
    h = {"Authorization": f"Bearer {token}"}

    # 2. login again (JWT roundtrip)
    lg = client.post("/api/v1/auth/login", json={"email": "e2e@example.com", "password": "secret123"})
    assert lg.status_code == 200

    # 3. bulk-load 3 months of statement data
    up = client.post(
        "/api/v1/expenses/upload-csv",
        headers=h,
        files={"file": ("statement.csv", _three_month_statement(), "text/csv")},
    )
    assert up.status_code == 200 and up.json()["inserted"] == 120

    # 4. forecast
    fc = client.get(f"/api/v1/forecasting/predict/{uid}", headers=h)
    assert fc.status_code == 200, fc.text
    body = fc.json()
    assert body["predicted_amount"] > 0
    assert body["lower_bound"] < body["upper_bound"]

    # 5. prediction logged for drift audit
    hist = client.get("/api/v1/forecasting/history", headers=h).json()
    assert any(row["forecast_month"] == body["target_month"] for row in hist)

    # 6. analytics reflect the uploaded spend
    summary = client.get("/api/v1/analytics/summary", headers=h).json()
    assert summary["transaction_count"] == 120
    assert summary["total_spend"] > 0
