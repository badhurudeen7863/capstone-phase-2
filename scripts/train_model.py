"""
Model training & comparison (report section 4.x):
LinearRegression vs RandomForest vs XGBoost, time-series cross-validation,
hold-out metrics MAE / MSE / RMSE / R2, artefact + metrics JSON export.

Usage:  python scripts/train_model.py [--quick]
Outputs:
  ml_models/xgboost_expense_forecaster_v2.pkl   (joblib artefact)
  ml_models/model_metrics.json                  (honest comparison + deployed metrics)
  ml_models/feature_importance.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from backend.logging_setup import get_logger
from backend.services.feature_engineering import FEATURE_NAMES, make_training_set

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


VARIABLE_CATEGORIES = {"Food & Groceries", "Health", "Transport", "Entertainment", "Other"}


def load_daily_series(csv_path: Path) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Returns (all-spend daily series, variable-spend daily series) per user."""
    from backend.services.feature_engineering import build_daily_series

    df = pd.read_csv(csv_path, parse_dates=["date"])
    all_s: dict[int, pd.Series] = {}
    var_s: dict[int, pd.Series] = {}
    for i, (email, grp) in enumerate(df.groupby("user_email"), start=1):
        all_s[i] = build_daily_series(list(zip(grp["date"], grp["amount"])))
        vgrp = grp[grp["category"].isin(VARIABLE_CATEGORIES)]
        var_s[i] = build_daily_series(list(zip(vgrp["date"], vgrp["amount"])))
    return all_s, var_s


def build_models(quick: bool) -> dict:
    if quick:
        return {
            "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
            "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.9,
                                    colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=5,
                                    random_state=42, n_jobs=-1, verbosity=0),
        }
    return {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=4,
                                              random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9,
                                colsample_bytree=0.9, reg_lambda=1.5, min_child_weight=4,
                                random_state=42, n_jobs=-1, verbosity=0),
    }


def evaluate(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mse": round(mse, 2),
        "rmse": round(float(np.sqrt(mse)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    csv_path = BASE_DIR / "data" / "transactions.csv"
    if not csv_path.exists():
        raise SystemExit("data/transactions.csv missing - run scripts/generate_synthetic_data.py first")

    t0 = time.time()
    daily, daily_var = load_daily_series(csv_path)
    X, y, meta = make_training_set(daily, daily_var)
    logger.info("Training set: %d samples, %d features", len(X), X.shape[1])
    if len(X) < 40:
        raise SystemExit("Not enough training samples - generate more history")

    # chronological ordering & time-based split (no leakage)
    order = np.argsort(meta["as_of"].values)
    X, y, meta = X.iloc[order], y.iloc[order], meta.iloc[order]
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    models = build_models(args.quick)
    tscv = TimeSeriesSplit(n_splits=4)
    comparison: dict[str, dict] = {}
    trained: dict[str, object] = {}

    y_tr_total = meta["y_total"].iloc[:split].to_numpy()
    y_te_total = meta["y_total"].iloc[split:].to_numpy()

    def to_total(pred_ratio, part: pd.DataFrame) -> np.ndarray:
        base = (part["rolling_mean_30"].clip(lower=1.0) * part["days_in_target_month"]).to_numpy()
        return np.asarray(pred_ratio) * base

    for name, model in models.items():
        cv_scores = []
        for tr_idx, va_idx in tscv.split(X_tr):
            m = build_models(args.quick)[name]
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            pred = to_total(m.predict(X_tr.iloc[va_idx]), X_tr.iloc[va_idx])
            cv_scores.append(float(np.sqrt(mean_squared_error(y_tr_total[va_idx], pred))))
        model.fit(X_tr, y_tr)
        pred = to_total(model.predict(X_te), X_te)
        metrics = evaluate(y_te_total, pred)
        metrics["cv_rmse"] = round(float(np.mean(cv_scores)), 2)
        comparison[name] = metrics
        trained[name] = model
        logger.info("%-16s holdout MAE=%.2f RMSE=%.2f R2=%.4f CV-RMSE=%.2f",
                    name, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["cv_rmse"])

    # deploy the best hold-out RMSE model (report deploys XGBoost)
    best_name = min(comparison, key=lambda k: comparison[k]["rmse"])
    best = trained[best_name]

    out_dir = BASE_DIR / "ml_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    version = f"v2-{time.strftime('%Y%m%d')}"
    artifact = {
        "model": best,
        "target_space": "ratio",
        "model_name": best_name,
        "model_version": version,
        "feature_names": FEATURE_NAMES,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": int(len(X)),
        "metrics": {"deployed": comparison[best_name], "comparison": comparison},
    }
    model_path = out_dir / "xgboost_expense_forecaster_v2.pkl"
    joblib.dump(artifact, model_path)

    metrics_doc = {
        "model_name": best_name,
        "model_version": version,
        "trained_at": artifact["trained_at"],
        "target_space": "ratio (next-month total / rolling_mean_30 * days); rescaled at inference",
        "n_training_samples": int(split),
        "n_holdout_samples": int(len(X_te)),
        "feature_names": FEATURE_NAMES,
        "comparison": comparison,
        "deployed": comparison[best_name],
        "ci_rule": "pred +/- z * RMSE (z=1.96 for 95%), widens with sqrt(horizon)",
        "anomaly_rule": "value > rolling_30_mean + 3 * rolling_30_std",
        "training_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "model_metrics.json").write_text(json.dumps(metrics_doc, indent=2))

    # feature importance plot (tree models only)
    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=FEATURE_NAMES).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        imp.plot.barh(ax=ax, color="#1a7f5a")
        ax.set_title(f"{best_name} feature importance")
        ax.set_xlabel("importance")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_importance.png", dpi=130)
        plt.close(fig)

    print("\n=== Model comparison (hold-out) ===")
    print(pd.DataFrame(comparison).T.to_string())
    print(f"\nDeployed: {best_name} ({version}) -> {model_path}")
    print(f"Metrics JSON -> {out_dir / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
