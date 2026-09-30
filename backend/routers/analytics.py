"""Analytics endpoints powering the Streamlit dashboard & analytics tabs."""
from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models.db_models import Category, Transaction, User
from ..models.schemas import AnomalyOut
from ..services.anomaly import detect_category_month_anomalies, detect_transaction_anomalies
from ..services.feature_engineering import build_daily_series

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _user_txns(db: Session, user: User):
    return (
        db.query(Transaction)
        .filter(Transaction.user_id == user.user_id)
        .order_by(Transaction.transaction_date)
        .all()
    )


@router.get("/summary")
def summary(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    txns = _user_txns(db, user)
    if not txns:
        return {"total_spend": 0.0, "transaction_count": 0, "this_month": 0.0,
                "last_month": 0.0, "delta_pct": None, "daily_average": 0.0,
                "avg_monthly": 0.0, "top_category": None, "first_date": None, "last_date": None}
    df = pd.DataFrame(
        [(t.transaction_date, float(t.amount), t.category.name) for t in txns],
        columns=["d", "amount", "category"],
    )
    df["month"] = pd.to_datetime(df["d"]).dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum()
    cur = pd.Timestamp.today().to_period("M")
    this_month = float(monthly.get(cur, 0.0))
    last_month = float(monthly.get(cur - 1, 0.0))
    delta = ((this_month - last_month) / last_month * 100.0) if last_month else None
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    n_days = (df["d"].max() - df["d"].min()).days + 1
    return {
        "total_spend": round(float(df["amount"].sum()), 2),
        "transaction_count": int(len(df)),
        "this_month": round(this_month, 2),
        "last_month": round(last_month, 2),
        "delta_pct": round(delta, 2) if delta is not None else None,
        "daily_average": round(float(df["amount"].sum()) / n_days, 2),
        "avg_monthly": round(float(monthly.mean()), 2),
        "top_category": {"name": cat_totals.index[0], "total": round(float(cat_totals.iloc[0]), 2)}
        if len(cat_totals) else None,
        "first_date": df["d"].min().isoformat(),
        "last_date": df["d"].max().isoformat(),
    }


@router.get("/monthly")
def monthly_series(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    txns = _user_txns(db, user)
    if not txns:
        return []
    df = pd.DataFrame([(t.transaction_date, float(t.amount)) for t in txns], columns=["d", "amount"])
    monthly = pd.to_datetime(df["d"]).dt.to_period("M").value_counts().sort_index()
    totals = df.assign(month=pd.to_datetime(df["d"]).dt.to_period("M")).groupby("month")["amount"].sum()
    return [
        {"month": str(p), "total": round(float(totals.get(p, 0.0)), 2), "count": int(monthly.get(p, 0))}
        for p in totals.index
    ]


@router.get("/categories")
def category_summary(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    txns = _user_txns(db, user)
    if not txns:
        return []
    df = pd.DataFrame(
        [(t.transaction_date, float(t.amount), t.category.name, t.category.type) for t in txns],
        columns=["d", "amount", "category", "type"],
    )
    g = df.groupby(["category", "type"])["amount"].agg(["sum", "count"]).reset_index()
    g["share"] = g["sum"] / g["sum"].sum() * 100.0
    cur = pd.Timestamp.today().to_period("M")
    df["month"] = pd.to_datetime(df["d"]).dt.to_period("M")
    this_month = df[df["month"] == cur].groupby("category")["amount"].sum()
    g = g.sort_values("sum", ascending=False)
    return [
        {
            "category": rec["category"],
            "type": rec["type"],
            "total": round(float(rec["sum"]), 2),
            "count": int(rec["count"]),
            "share": round(float(rec["share"]), 2),
            "this_month": round(float(this_month.get(rec["category"], 0.0)), 2),
        }
        for rec in g.to_dict(orient="records")
    ]


@router.get("/category-month")
def category_month_matrix(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    """Predicted/actual distribution: monthly totals per category (last 12 months)."""
    txns = _user_txns(db, user)
    if not txns:
        return []
    df = pd.DataFrame(
        [(t.transaction_date, float(t.amount), t.category.name) for t in txns],
        columns=["d", "amount", "category"],
    )
    df["month"] = pd.to_datetime(df["d"]).dt.to_period("M").dt.to_timestamp()
    piv = df.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0.0)
    piv = piv.tail(12)
    return [
        {"month": idx.date().isoformat(), **{c: round(float(v), 2) for c, v in row.items()}}
        for idx, row in piv.iterrows()
    ]


@router.get("/anomalies", response_model=list[AnomalyOut])
def anomalies(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[AnomalyOut]:
    txns = _user_txns(db, user)
    if not txns:
        return []
    out = detect_transaction_anomalies(
        [(t.transaction_date, t.category.name, float(t.amount), t.notes) for t in txns]
    )
    out += detect_category_month_anomalies(
        [(t.transaction_date, t.category.name, float(t.amount)) for t in txns]
    )
    return sorted(out, key=lambda a: a.date, reverse=True)[:100]


@router.get("/spend-by-day")
def spend_by_day(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    txns = _user_txns(db, user)
    if not txns:
        return []
    daily = build_daily_series([(t.transaction_date, float(t.amount)) for t in txns])
    return [{"date": d.date().isoformat(), "amount": round(float(v), 2)} for d, v in daily.items()]
