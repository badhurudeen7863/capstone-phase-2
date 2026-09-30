"""Report test list: bad payloads must be rejected with 422 / 401 / 403 / 400."""
from __future__ import annotations


def test_negative_amount_rejected(client, headers):
    r = client.post(
        "/api/v1/expenses",
        headers=headers,
        json={"amount": -500, "transaction_date": "2026-08-01", "category_id": 1},
    )
    assert r.status_code == 422


def test_zero_amount_rejected(client, headers):
    r = client.post(
        "/api/v1/expenses",
        headers=headers,
        json={"amount": 0, "transaction_date": "2026-08-01", "category_id": 1},
    )
    assert r.status_code == 422


def test_invalid_date_rejected(client, headers):
    r = client.post(
        "/api/v1/expenses",
        headers=headers,
        json={"amount": 500, "transaction_date": "2026-15-45", "category_id": 1},
    )
    assert r.status_code == 422


def test_sql_injection_rejected(client, headers):
    payloads = ["x' OR '1'='1", "1; DROP TABLE users--", "a' UNION SELECT password_hash FROM users--"]
    for note in payloads:
        r = client.post(
            "/api/v1/expenses",
            headers=headers,
            json={"amount": 500, "transaction_date": "2026-08-01", "category_id": 1, "notes": note},
        )
        assert r.status_code == 422, note


def test_unauthenticated_access_blocked(client):
    assert client.get("/api/v1/expenses").status_code == 401
    assert client.get("/api/v1/analytics/summary").status_code == 401


def test_bad_token_rejected(client):
    r = client.get("/api/v1/expenses", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_cross_user_forecast_forbidden(client, headers, auth):
    other_user = auth["user_id"] + 999
    r = client.get(f"/api/v1/forecasting/predict/{other_user}", headers=headers)
    assert r.status_code == 403


def test_forecast_without_history_is_400(client, headers):
    res = client.post(
        "/api/v1/auth/register",
        json={"name": "Empty User", "email": "empty@example.com", "password": "secret123"},
    )
    token = res.json()["access_token"]
    uid = res.json()["user"]["user_id"]
    r = client.get(f"/api/v1/forecasting/predict/{uid}", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400
