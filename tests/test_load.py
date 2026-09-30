"""Lightweight concurrency/load test (no external tooling required)."""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor


def test_concurrent_analytics_load(client, headers, seeded_user):
    n = 60
    latencies: list[float] = []
    errors: list[int] = []

    def one(_) -> int:
        t0 = time.perf_counter()
        r = client.get("/api/v1/analytics/summary", headers=headers)
        latencies.append((time.perf_counter() - t0) * 1000)
        return r.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = list(pool.map(one, range(n)))

    assert all(code == 200 for code in errors), errors
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    print(f"\nload test: n={n} mean={statistics.mean(latencies):.1f}ms p95={p95:.1f}ms")
    assert p95 < 3000, f"p95 latency too high: {p95:.0f} ms"


def test_concurrent_forecast_load(client, headers, seeded_user):
    uid = seeded_user["user_id"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(
            lambda _: client.get(f"/api/v1/forecasting/predict/{uid}", headers=headers).status_code,
            range(12),
        ))
    assert all(c == 200 for c in codes), codes
