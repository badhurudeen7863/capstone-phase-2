"""
ML inference service: loads the trained artefact at startup (FastAPI lifespan)
and serves forecasts with confidence intervals.

Artefact (joblib): {"model", "feature_names", "model_name", "model_version",
                    "trained_at", "metrics"}
Metrics JSON: per-model comparison + deployed model metrics (MAE, MSE, RMSE, R2).
CI rule from the report:  prediction +/- 1.96 * RMSE  (z generalised per confidence level).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..config import CONFIDENCE_Z, METRICS_PATH, MODEL_PATH
from ..logging_setup import get_logger
from .feature_engineering import FEATURE_NAMES

logger = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


class MLService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.artifact: dict[str, Any] | None = None
        self.metrics: dict[str, Any] | None = None

    # ------------------------------------------------------------- loading ---
    def load(self, model_path: Path | None = None, metrics_path: Path | None = None) -> bool:
        model_path = Path(model_path or MODEL_PATH)
        metrics_path = Path(metrics_path or METRICS_PATH)
        with self._lock:
            if not model_path.exists():
                logger.warning("Model artefact not found at %s - run scripts/train_model.py", model_path)
                self.artifact = None
                return False
            self.artifact = joblib.load(model_path)
            if metrics_path.exists():
                self.metrics = json.loads(metrics_path.read_text())
            else:
                self.metrics = self.artifact.get("metrics", {})
            logger.info(
                "Loaded model %s (version %s, R2=%.4f, RMSE=%.2f)",
                self.model_name,
                self.model_version,
                self.metrics.get("deployed", {}).get("r2", float("nan")),
                self.metrics.get("deployed", {}).get("rmse", float("nan")),
            )
            return True

    @property
    def is_loaded(self) -> bool:
        return self.artifact is not None

    @property
    def model_name(self) -> str:
        return str((self.artifact or {}).get("model_name", "unknown"))

    @property
    def model_version(self) -> str:
        return str((self.artifact or {}).get("model_version", "unknown"))

    @property
    def feature_names(self) -> list[str]:
        return list((self.artifact or {}).get("feature_names", FEATURE_NAMES))

    @property
    def deployed_metrics(self) -> dict[str, float]:
        dep = (self.metrics or {}).get("deployed", {})
        return {
            "r2": float(dep.get("r2", 0.0)),
            "mae": float(dep.get("mae", 0.0)),
            "mse": float(dep.get("mse", 0.0)),
            "rmse": float(dep.get("rmse", 0.0)),
        }

    # ---------------------------------------------------------- inference ---
    def predict(self, features: dict[str, float]) -> float:
        """Returns the forecast MONTHLY TOTAL in currency units.

        The artefact models the stationary ratio (next-month total / recent
        daily rate); we rescale here: total = ratio * rolling_mean_30 * days."""
        if not self.is_loaded:
            raise ModelNotLoadedError(
                "No trained model loaded. Run: python scripts/train_model.py"
            )
        names = self.feature_names
        row = pd.DataFrame([[float(features.get(n, 0.0)) for n in names]], columns=names)
        pred = self.artifact["model"].predict(row)
        ratio = float(pred[0])  # report bug fix: prediction is an array -> take [0]
        if self.artifact.get("target_space") == "ratio":
            base = max(float(features.get("rolling_mean_30", 0.0)), 1.0) * float(
                features.get("days_in_target_month", 30.44)
            )
            return ratio * base
        return ratio

    def confidence_interval(self, predicted: float, confidence: float = 0.95, horizon: int = 1) -> tuple[float, float]:
        z = CONFIDENCE_Z.get(round(confidence, 2), 1.96)
        rmse = self.deployed_metrics["rmse"]
        half = z * rmse * (horizon ** 0.5)  # CI widens with horizon
        return round(predicted - half, 2), round(predicted + half, 2)


ml_service = MLService()
