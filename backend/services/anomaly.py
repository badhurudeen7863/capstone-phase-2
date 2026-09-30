"""
Anomaly detection (report: value > rolling_30_mean + 3 * rolling_30_std).

Applied at two levels:
  1. daily transaction level  (single-day spend spikes)
  2. category-month level     (a category's monthly total is unusual)
"""
from __future__ import annotations

import pandas as pd

from ..config import ANOMALY_SIGMA
from ..models.schemas import AnomalyOut


def detect_daily_anomalies(daily: pd.Series, sigma: float = ANOMALY_SIGMA) -> list[AnomalyOut]:
    """Flag days whose spend exceeds roll30 mean + sigma * roll30 std."""
    if daily is None or daily.empty or len(daily) < 31:
        return []
    mean = daily.shift(1).rolling(30).mean()
    std = daily.shift(1).rolling(30).std()
    threshold = mean + sigma * std
    mask = (daily > threshold).fillna(False)
    out: list[AnomalyOut] = []
    for ts in daily.index[mask]:
        out.append(
            AnomalyOut(
                level="transaction",
                date=ts.date(),
                amount=float(daily.loc[ts]),
                threshold=float(threshold.loc[ts]),
                description=f"Daily spend exceeded rolling 30-day mean + {sigma:g} sigma",
            )
        )
    return out


def detect_transaction_anomalies(
    txns: list[tuple], sigma: float = ANOMALY_SIGMA, min_history: int = 10
) -> list[AnomalyOut]:
    """Transaction-level rule (report: 'applied to transactions').

    Each transaction is compared with the rolling window of the previous 30
    transactions *of the same category*: flag when
        amount > rolling_mean + sigma * rolling_std   AND   amount >= 2 * mean
    The relative guard stops ordinary large grocery runs from tripping alerts.
    txns: list of (date, category, amount, description|None)
    """
    if not txns:
        return []
    df = pd.DataFrame(txns, columns=["d", "category", "amount", "description"])
    df["d"] = pd.to_datetime(df["d"])
    df = df.sort_values("d")
    out: list[AnomalyOut] = []
    for cat, sub in df.groupby("category"):
        amounts = sub["amount"].to_numpy(float)
        for i in range(min_history, len(amounts)):
            window = amounts[max(0, i - 30):i]
            mean, std = float(window.mean()), float(window.std(ddof=1)) if len(window) > 1 else (float(window.mean()), 0.0)
            thr = mean + sigma * std
            if amounts[i] > thr and amounts[i] >= 2.0 * mean:
                row = sub.iloc[i]
                out.append(
                    AnomalyOut(
                        level="transaction",
                        date=row["d"].date(),
                        category=cat,
                        amount=float(amounts[i]),
                        threshold=round(thr, 2),
                        description=row["description"],
                    )
                )
    return out


def detect_category_month_anomalies(
    txns: list[tuple], sigma: float = ANOMALY_SIGMA
) -> list[AnomalyOut]:
    """txns: list of (date, category_name, amount). Flags unusual category-month totals."""
    if not txns:
        return []
    df = pd.DataFrame(txns, columns=["d", "category", "amount"])
    df["d"] = pd.to_datetime(df["d"])
    out: list[AnomalyOut] = []
    for _period_unused, cat, series in _per_category(df):
        if len(series) < 6:
            continue
        mean = series.shift(1).rolling(3, min_periods=3).mean()
        std = series.shift(1).rolling(3, min_periods=3).std()
        thr = mean + sigma * std
        flag = ((series > thr) & (series >= 2.0 * series.shift(1).rolling(3, min_periods=3).mean())).fillna(False)
        for p in series.index[flag]:
            out.append(
                AnomalyOut(
                    level="category_month",
                    date=p.to_timestamp().date(),
                    category=cat,
                    amount=float(series.loc[p]),
                    threshold=float(thr.loc[p]),
                    description=f"{cat} monthly total exceeded its rolling mean + {sigma:g} sigma",
                )
            )
    return out


def _per_category(df: pd.DataFrame):
    """Yield (None, category, monthly Series) per category over time (gaps zero-filled)."""
    df = df.copy()
    df["period"] = df["d"].dt.to_period("M")
    grouped = df.groupby(["category", "period"])["amount"].sum()
    for cat, sub in grouped.groupby(level=0):
        series = sub.droplevel(0).sort_index()
        full = pd.period_range(series.index.min(), series.index.max(), freq="M")
        series = series.reindex(full, fill_value=0.0)
        yield None, cat, series
