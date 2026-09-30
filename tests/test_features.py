"""Report unit tests: rolling mean, sin/cos cycle wrap, feature pipeline sanity."""
from __future__ import annotations

import math
from datetime import date

import pandas as pd

from backend.services.feature_engineering import (
    apply_sine_cosine_encoding,
    build_daily_series,
    encode_target_month,
    generate_time_features,
    rolling_mean,
)


def test_rolling_mean_unit():
    out = rolling_mean([10, 20, 30, 40], 2)
    assert math.isnan(out[0])
    assert out[1:] == [15.0, 25.0, 35.0]


def test_sine_cosine_cycle_wrap():
    dec = encode_target_month(date(2026, 12, 1))
    jan_next_cycle = {k: v for k, v in encode_target_month(date(2026, 12, 1)).items()}
    # month 13 must wrap to month 1
    df = pd.DataFrame({"m": [1, 13, 12, 24]})
    enc = apply_sine_cosine_encoding(df, month_values=df["m"])
    assert enc["sin_month"].iloc[0] == enc["sin_month"].iloc[1]
    assert enc["cos_month"].iloc[0] == enc["cos_month"].iloc[1]
    assert enc["sin_month"].iloc[2] == enc["sin_month"].iloc[3]
    assert dec == jan_next_cycle


def test_sine_cosine_bounds_and_period():
    df = pd.DataFrame({"m": range(1, 13)})
    enc = apply_sine_cosine_encoding(df, month_values=df["m"])
    assert ((enc["sin_month"] ** 2 + enc["cos_month"] ** 2) - 1).abs().max() < 1e-9


def test_generate_time_features_columns_and_lags():
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    s = pd.Series(range(60), index=idx, dtype=float)
    feats = generate_time_features(s)
    for col in ["lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30",
                "rolling_std_30", "expanding_std", "ema_7"]:
        assert col in feats.columns
    assert feats["lag_1"].iloc[1] == 0.0
    assert feats["lag_7"].iloc[7] == 0.0
    assert feats["rolling_mean_7"].iloc[8] == 4.0  # mean of shifted values 1..7


def test_build_daily_series_fills_gaps_with_zero():
    rows = [(date(2026, 1, 1), 100.0), (date(2026, 1, 4), 50.0)]
    daily = build_daily_series(rows)
    assert len(daily) == 4
    assert daily.iloc[1] == 0.0 and daily.iloc[2] == 0.0
    assert daily.iloc[0] == 100.0 and daily.iloc[3] == 50.0


def test_build_daily_series_aggregates_same_day():
    rows = [(date(2026, 1, 1), 100.0), (date(2026, 1, 1), 25.0)]
    daily = build_daily_series(rows)
    assert len(daily) == 1 and daily.iloc[0] == 125.0
