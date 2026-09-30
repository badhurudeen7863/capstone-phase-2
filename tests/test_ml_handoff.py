"""API -> ML handoff: forecast contract, CI ordering, metrics, audit log."""
from __future__ import annotations

from backend.services.feature_engineering import FEATURE_NAMES


def test_forecast_contract(client, headers, seeded_user):
    uid = seeded_user["user_id"]
    r = client.get(f"/api/v1/forecasting/predict/{uid}", headers=headers,
                   params={"horizon": 1, "confidence": 0.95})
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ["user_id", "target_month", "predicted_amount", "lower_bound",
                "upper_bound", "model_metrics", "anomaly_flag", "history_days",
                "model_version", "features_used"]:
        assert key in body
    assert body["lower_bound"] <= body["predicted_amount"] <= body["upper_bound"]
    m = body["model_metrics"]
    assert {"r2", "mae", "mse", "rmse"} <= set(m)
    assert body["features_used"] == FEATURE_NAMES
    assert body["history_days"] >= 30


def test_confidence_widens_interval(client, headers, seeded_user):
    uid = seeded_user["user_id"]
    wide = client.get(f"/api/v1/forecasting/predict/{uid}", headers=headers,
                      params={"confidence": 0.99}).json()
    narrow = client.get(f"/api/v1/forecasting/predict/{uid}", headers=headers,
                        params={"confidence": 0.90}).json()
    span_w = wide["upper_bound"] - wide["lower_bound"]
    span_n = narrow["upper_bound"] - narrow["lower_bound"]
    assert span_w > span_n


def test_multi_horizon(client, headers, seeded_user):
    uid = seeded_user["user_id"]
    r = client.get(f"/api/v1/forecasting/predict/{uid}", headers=headers, params={"horizon": 3})
    assert r.status_code == 200
    assert r.json()["horizon_months"] == 3


def test_prediction_audit_log(client, headers, seeded_user):
    uid = seeded_user["user_id"]
    client.get(f"/api/v1/forecasting/predict/{uid}", headers=headers)
    hist = client.get("/api/v1/forecasting/history", headers=headers).json()
    assert hist, "forecast must be written to predictions_log"
    row = hist[0]
    assert {"prediction_id", "forecast_month", "predicted_amount", "lower_bound",
            "upper_bound", "actual_val", "drift"} <= set(row)


def test_analytics_endpoints(client, headers, seeded_user):
    assert client.get("/api/v1/analytics/summary", headers=headers).status_code == 200
    monthly = client.get("/api/v1/analytics/monthly", headers=headers).json()
    assert monthly and {"month", "total", "count"} <= set(monthly[0])
    cats = client.get("/api/v1/analytics/categories", headers=headers).json()
    assert cats and abs(sum(c["share"] for c in cats) - 100.0) < 0.5
    anomalies = client.get("/api/v1/analytics/anomalies", headers=headers).json()
    assert isinstance(anomalies, list)
