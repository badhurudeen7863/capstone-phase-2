"""
Forecasting endpoints (report: GET /api/v1/forecasting/predict/{user_id}).

Rules implemented from the report:
  * >= 30 days of history required, else HTTP 400
  * features from the last 90 days of the daily-resampled series (gaps = 0)
  * CI = prediction +/- 1.96 * RMSE (generalised per confidence level, widens with horizon)
  * anomaly_flag when the input window contains 3-sigma anomalies
  * every forecast is written to predictions_log (audit + drift monitoring);
    actual_val is back-filled once the target month has ended
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..config import MIN_HISTORY_DAYS
from ..database import get_db
from ..deps import get_current_user
from ..logging_setup import get_logger
from ..models.db_models import PredictionLog, Transaction, User
from ..models.schemas import ForecastResponse, PredictionLogOut
from ..services.anomaly import detect_transaction_anomalies
from ..services.feature_engineering import (
    build_daily_series,
    build_forecast_features,
    extend_daily_with_forecast,
    winsorize_anomalies,
)
from ..services.ml_service import ModelNotLoadedError, ml_service

logger = get_logger(__name__)
router = APIRouter(prefix="/forecasting", tags=["forecasting"])


def _daily_for(db: Session, user_id: int) -> pd.Series:
    txns = (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(Transaction.transaction_date)
        .all()
    )
    return build_daily_series([(t.transaction_date, float(t.amount)) for t in txns])


@router.get("/predict/{user_id}", response_model=ForecastResponse)
def predict(
    user_id: int,
    horizon: int = Query(1, ge=1, le=6, description="months ahead"),
    confidence: float = Query(0.95, ge=0.50, le=0.999),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ForecastResponse:
    if current_user.user_id != user_id:
        raise HTTPException(status_code=403, detail="You can only forecast your own expenses")
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not trained yet - run scripts/train_model.py")

    daily = _daily_for(db, user_id)
    txns = (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(Transaction.transaction_date)
        .all()
    )
    if daily.empty:
        raise HTTPException(status_code=400, detail="No transaction history found")
    span_days = (daily.index.max() - daily.index.min()).days + 1
    if span_days < MIN_HISTORY_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_HISTORY_DAYS} days of history are required "
            f"(found {span_days})",
        )

    variable = [(t.transaction_date, float(t.amount)) for t in txns if t.category.type == "Variable"]
    var_series = build_daily_series(variable)
    anomalies = detect_transaction_anomalies(
        [(t.transaction_date, t.category.name, float(t.amount), t.notes) for t in txns]
    )
    daily = winsorize_anomalies(daily, var_series)   # features/base on cleaned series
    as_of = daily.index.max()
    window_start = as_of - pd.Timedelta(days=89)
    anomalies_in_window = sum(1 for a in anomalies if pd.Timestamp(a.date) >= window_start)

    # ---- recursive multi-horizon forecast ---------------------------------
    work_daily = daily
    predicted_total = 0.0
    lower = upper = 0.0
    cur_month_start = pd.Timestamp(as_of).normalize().replace(day=1)
    for h in range(1, horizon + 1):
        target = cur_month_start + pd.DateOffset(months=h)
        prev_month_end = target - pd.DateOffset(days=1)
        feat = build_forecast_features(work_daily, as_of=prev_month_end, target_month=target)
        if feat is None:
            raise HTTPException(status_code=400, detail="Not enough history for the requested horizon")
        predicted_total = ml_service.predict(feat)
        lower, upper = ml_service.confidence_interval(predicted_total, confidence, horizon=h)
        work_daily = extend_daily_with_forecast(work_daily, predicted_total, target)

    target_month = cur_month_start + pd.DateOffset(months=horizon)

    # ---- audit log ---------------------------------------------------------
    existing = (
        db.query(PredictionLog)
        .filter(
            PredictionLog.user_id == user_id,
            PredictionLog.forecast_month == target_month.date(),
        )
        .order_by(PredictionLog.prediction_id.desc())
        .first()
    )
    if existing is None:
        log_row = PredictionLog(
            user_id=user_id,
            forecast_month=target_month.date(),
            predicted_amount=round(predicted_total, 2),
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            model_version=ml_service.model_version,
        )
        db.add(log_row)
        db.commit()
    logger.info(
        "forecast user=%s target=%s pred=%.2f CI=[%.2f, %.2f] anomalies=%s",
        user_id, target_month.date(), predicted_total, lower, upper, anomalies_in_window,
    )

    return ForecastResponse(
        user_id=user_id,
        target_month=target_month.date(),
        horizon_months=horizon,
        predicted_amount=round(max(predicted_total, 0.0), 2),
        lower_bound=max(lower, 0.0),
        upper_bound=upper,
        confidence_level=confidence,
        model_metrics=ml_service.deployed_metrics,
        anomaly_flag=anomalies_in_window > 0,
        anomalies_in_window=anomalies_in_window,
        model_version=ml_service.model_version,
        history_days=int(span_days),
        features_used=ml_service.feature_names,
    )


@router.get("/history", response_model=list[PredictionLogOut])
def history(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[PredictionLogOut]:
    rows = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id == user.user_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(200)
        .all()
    )
    today = date.today()
    out: list[PredictionLogOut] = []
    for r in rows:
        actual = r.actual_val
        if actual is None and r.forecast_month.replace(day=1) < today.replace(day=1):
            # month has ended -> back-fill actual for drift audit
            month_end = r.forecast_month.replace(day=28) + timedelta(days=4)
            month_end = month_end.replace(day=1) - timedelta(days=1)
            total = (
                db.query(Transaction)
                .filter(
                    Transaction.user_id == r.user_id,
                    Transaction.transaction_date >= r.forecast_month,
                    Transaction.transaction_date <= month_end,
                )
                .with_entities(Transaction.amount)
                .all()
            )
            if total:
                actual = round(sum(float(a[0]) for a in total), 2)
                r.actual_val = actual
        out.append(
            PredictionLogOut(
                prediction_id=r.prediction_id,
                user_id=r.user_id,
                forecast_month=r.forecast_month,
                predicted_amount=r.predicted_amount,
                lower_bound=r.lower_bound,
                upper_bound=r.upper_bound,
                confidence_level=r.confidence_level,
                model_version=r.model_version,
                actual_val=actual,
                drift=round(actual - r.predicted_amount, 2) if actual is not None else None,
                created_at=r.created_at,
            )
        )
    db.commit()
    return out
