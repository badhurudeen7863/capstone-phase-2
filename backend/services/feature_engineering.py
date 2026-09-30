"""
Automated feature engineering (report sections 3.4 / 3.5.1).

Daily spend series  ->  lag, rolling, expanding, EMA features
Target month        ->  sin/cos cyclic encoding (month & day-of-month)

Forecast target definition (resolves the report's daily-vs-monthly ambiguity):
    y = total spend of the month AFTER the feature cut-off date ("as_of").
    Features are computed on the daily series resampled to calendar days
    (missing days filled with 0), using the last FORECAST_WINDOW_DAYS (90)
    days ending at as_of - exactly the window described in the report.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import FORECAST_WINDOW_DAYS

FEATURE_NAMES = [
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_30",
    "expanding_std",
    "ema_7",
    "sin_month",
    "cos_month",
    "month_of_year",
    "days_in_target_month",
]


# --------------------------------------------------------------- primitives --
def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Plain rolling mean used by the unit test: [10,20,30,40], w=2 -> [NaN,15,25,35]."""
    return pd.Series(list(values), dtype=float).rolling(window).mean().tolist()


def build_daily_series(rows: Iterable[tuple[date | datetime | str, float]]) -> pd.Series:
    """Sum amounts per calendar day, reindex to a contiguous daily range, fill gaps with 0."""
    if not isinstance(rows, list):
        rows = list(rows)
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]), name="amount")
    df = pd.DataFrame(rows, columns=["d", "amount"])
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    daily = df.groupby("d")["amount"].sum()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_idx, fill_value=0.0).astype(float).rename("amount")


def generate_time_features(daily: pd.Series) -> pd.DataFrame:
    """Lag t-1/t-7/t-30, rolling mean 7/30, rolling std 30, expanding std, EMA."""
    s = daily.astype(float)
    feats = pd.DataFrame(index=s.index)
    feats["lag_1"] = s.shift(1)
    feats["lag_7"] = s.shift(7)
    feats["lag_30"] = s.shift(30)
    feats["rolling_mean_7"] = s.shift(1).rolling(7).mean()
    feats["rolling_mean_30"] = s.shift(1).rolling(30).mean()
    feats["rolling_std_30"] = s.shift(1).rolling(30).std()
    feats["expanding_std"] = s.shift(1).expanding(min_periods=7).std()
    feats["ema_7"] = s.shift(1).ewm(span=7, adjust=False).mean()
    return feats


def apply_sine_cosine_encoding(
    df: pd.DataFrame,
    *,
    month_values: pd.Series | None = None,
    day_values: pd.Series | None = None,
) -> pd.DataFrame:
    """Cyclic sin/cos encoding: sin(2*pi*m/12), cos(2*pi*m/12) (and day-of-month).

    Encoding uses modulo 12 (and modulo days-in-month), so month 13 wraps to
    month 1 - this is what the cycle-wrap unit test asserts.
    """
    out = df.copy()
    if month_values is not None:
        m = month_values.astype(int) % 12
        out["sin_month"] = np.sin(2 * np.pi * m / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * m / 12.0)
    if day_values is not None:
        d = day_values.astype(int) % 31
        out["sin_day"] = np.sin(2 * np.pi * d / 31.0)
        out["cos_day"] = np.cos(2 * np.pi * d / 31.0)
    return out


def encode_target_month(target_month: date | pd.Timestamp) -> dict[str, float]:
    """sin/cos + numeric encoding of the *target* month (the month being forecast)."""
    ts = pd.Timestamp(target_month)
    m = ts.month % 12
    return {
        "sin_month": float(math.sin(2 * math.pi * m / 12.0)),
        "cos_month": float(math.cos(2 * math.pi * m / 12.0)),
        "month_of_year": float(ts.month),
        "days_in_target_month": float(ts.days_in_month),
    }


def winsorize_anomalies(daily: pd.Series, variable: pd.Series | None = None, sigma: float = 3.0) -> pd.Series:
    """Remove 3-sigma spike days from the series used for features / rescaling.

    The spike mask is computed on the *variable-spend* daily series (rent and
    utility payment days inflate sigma so much that real spikes hide below a
    3-sigma threshold on the all-spend series).  Masked days keep only their
    fixed-cost component, so a single hospital bill cannot drag the next
    month's forecast up.  Raw anomalies are still reported by services.anomaly.
    """
    if daily is None or daily.empty or len(daily) < 31:
        return daily
    if variable is not None and not variable.empty and len(variable) >= 31:
        mean = variable.shift(1).rolling(30).mean()
        std = variable.shift(1).rolling(30).std()
        mask = (variable > mean + sigma * std).fillna(False)
        mask = mask.reindex(daily.index, fill_value=False)
        out = daily.copy()
        fixed_part = (daily - variable.reindex(daily.index, fill_value=0.0)).clip(lower=0.0)
        out[mask] = fixed_part[mask]
        return out
    mean = daily.shift(1).rolling(30).mean()
    std = daily.shift(1).rolling(30).std()
    med = daily.shift(1).rolling(30).median()
    mask = (daily > mean + sigma * std).fillna(False)
    out = daily.copy()
    out[mask] = med[mask]
    return out


# ------------------------------------------------------- forecast features ---
def build_forecast_features(
    daily: pd.Series,
    as_of: pd.Timestamp,
    target_month: pd.Timestamp,
    window_days: int = FORECAST_WINDOW_DAYS,
) -> dict[str, float] | None:
    """Feature vector for 'predict the total of target_month' using data up to as_of.

    Returns None when the window lacks enough history for the 30-day statistics.
    """
    as_of = pd.Timestamp(as_of).normalize()
    if daily.empty or as_of < daily.index.min():
        return None
    window = daily.loc[max(daily.index.min(), as_of - pd.Timedelta(days=window_days - 1)): as_of]
    if len(window) < 31:
        return None
    feats = generate_time_features(window).iloc[[-1]]
    row = feats.iloc[0]
    if row[["lag_30", "rolling_mean_30", "rolling_std_30"]].isna().any():
        return None
    vec = {name: float(row[name]) for name in FEATURE_NAMES if name in row.index}
    vec.update(encode_target_month(target_month))
    return {k: (0.0 if (isinstance(v, float) and math.isnan(v)) else v) for k, v in vec.items()}


def make_training_set(
    user_daily: dict[int, pd.Series],
    user_daily_variable: dict[int, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build (X, y, meta) across users.

    For every month-end `e` of month m (where month m+1 exists in the series):
        features = build_forecast_features(daily, as_of=e, target=m+1)
        y        = total spend of month m+1
    """
    xs, ys, metas = [], [], []
    for user_id, raw in user_daily.items():
        if raw.empty:
            continue
        var = (user_daily_variable or {}).get(user_id)
        daily = winsorize_anomalies(raw, var)
        monthly = daily.resample("MS").sum()
        month_starts = monthly.index
        for i in range(len(month_starts) - 1):
            m_start = month_starts[i]
            next_start = month_starts[i + 1]
            as_of = next_start - pd.Timedelta(days=1)          # month-end of m
            target = next_start                                # month m+1
            feat = build_forecast_features(daily, as_of=as_of, target_month=target)
            if feat is None:
                continue
            y_total = float(monthly.iloc[i + 1])
            # stationary target: ratio of next-month total to recent daily rate.
            # Trees cannot extrapolate the inflation trend in level space; the
            # ratio cancels it and is rescaled at inference time (see ml_service).
            base = max(feat["rolling_mean_30"], 1.0) * feat["days_in_target_month"]
            xs.append(feat)
            ys.append(y_total / base)
            metas.append({"user_id": user_id, "as_of": as_of, "target_month": target,
                          "base": base, "y_total": y_total})
    X = pd.DataFrame(xs, columns=FEATURE_NAMES)
    y = pd.Series(ys, name="next_month_total")
    meta = pd.DataFrame(metas)
    return X, y, meta


def features_for_prediction(
    daily: pd.Series,
    as_of: pd.Timestamp,
    target_month: pd.Timestamp,
    horizon: int = 1,
) -> dict[str, float] | None:
    """Features for horizon h>1: recursively extend the daily series with the
    previously predicted monthly total spread uniformly over that month's days."""
    return build_forecast_features(daily, as_of=as_of, target_month=target_month)


def extend_daily_with_forecast(
    daily: pd.Series, predicted_total: float, month_start: pd.Timestamp
) -> pd.Series:
    """Uniform daily imputation of a forecast month (used for multi-month horizons)."""
    month_end = month_start + pd.offsets.MonthEnd(0)
    days = pd.date_range(month_start, month_end, freq="D")
    per_day = float(predicted_total) / len(days)
    ext = daily.copy()
    new = pd.Series(per_day, index=days, name="amount")
    ext = ext[~ext.index.isin(days)]
    return pd.concat([ext, new]).sort_index()
