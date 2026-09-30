"""Pydantic v2 schemas - request/response validation (report: Pydantic validation, 422 on bad input)."""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Rule-based guard: reject obvious SQL-injection payloads in free-text fields.
_SQLI_RE = re.compile(
    r"(--|;|\bunion\b\s+\bselect\b|\bdrop\b\s+\btable\b|\bdelete\b\s+\bfrom\b|"
    r"\bor\b\s+\d+\s*=\s*\d+|\band\b\s+\d+\s*=\s*\d+|'\s*(or|and)\s*'|xp_cmdshell)",
    re.IGNORECASE,
)


def _reject_sqli(value: str | None, field: str) -> str | None:
    if value and _SQLI_RE.search(value):
        raise ValueError(f"{field} contains forbidden SQL-like syntax")
    return value


# --------------------------------------------------------------------- auth --
class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=190)
    password: str = Field(min_length=6, max_length=128)

    @field_validator("name")
    @classmethod
    def _name_safe(cls, v: str) -> str:
        return _reject_sqli(v, "name") or v


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_id: int
    name: str
    email: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# --------------------------------------------------------------- categories --
class CategoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    category_id: int
    name: str
    type: str


# ------------------------------------------------------------- transactions --
class TransactionCreate(BaseModel):
    amount: float = Field(gt=0, le=10_000_000, description="Amount must be > 0")
    transaction_date: date
    category_id: int = Field(ge=1)
    notes: Optional[str] = Field(default=None, max_length=500)

    @field_validator("transaction_date")
    @classmethod
    def _date_sane(cls, v: date) -> date:
        if v.year < 2000 or v.year > 2100:
            raise ValueError("transaction_date out of supported range (2000-2100)")
        return v

    @field_validator("notes")
    @classmethod
    def _notes_safe(cls, v: Optional[str]) -> Optional[str]:
        return _reject_sqli(v, "notes")


class TransactionUpdate(BaseModel):
    amount: Optional[float] = Field(default=None, gt=0, le=10_000_000)
    transaction_date: Optional[date] = None
    category_id: Optional[int] = Field(default=None, ge=1)
    notes: Optional[str] = Field(default=None, max_length=500)

    @field_validator("notes")
    @classmethod
    def _notes_safe(cls, v: Optional[str]) -> Optional[str]:
        return _reject_sqli(v, "notes")


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    category_id: int
    category_name: str | None = None
    category_type: str | None = None
    amount: float
    transaction_date: date
    notes: str | None = None

    @classmethod
    def from_orm_row(cls, txn) -> "TransactionOut":
        return cls(
            id=txn.id,
            user_id=txn.user_id,
            category_id=txn.category_id,
            category_name=txn.category.name if txn.category else None,
            category_type=txn.category.type if txn.category else None,
            amount=float(txn.amount),
            transaction_date=txn.transaction_date,
            notes=txn.notes,
        )


class CSVUploadReport(BaseModel):
    inserted: int
    skipped: int
    errors: list[str] = Field(default_factory=list)
    category_mapping: dict[str, str] = Field(default_factory=dict)
    date_formats_seen: list[str] = Field(default_factory=list)


# -------------------------------------------------------------- forecasting --
class ForecastResponse(BaseModel):
    user_id: int
    target_month: date                      # first day of the forecast month
    horizon_months: int = 1
    predicted_amount: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    model_metrics: dict                     # R2, MAE, MSE, RMSE of deployed model
    anomaly_flag: bool
    anomalies_in_window: int = 0
    model_version: str | None = None
    history_days: int
    features_used: list[str] = Field(default_factory=list)


class PredictionLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    prediction_id: int
    user_id: int
    forecast_month: date
    predicted_amount: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    model_version: str | None
    actual_val: float | None
    drift: float | None = None              # actual - predicted, when actual is known
    created_at: datetime


# ------------------------------------------------------------------ budgets --
class BudgetUpsert(BaseModel):
    category_id: int = Field(ge=1)
    month: date
    amount: float = Field(ge=0)


class BudgetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    budget_id: int
    user_id: int
    category_id: int
    category_name: str | None = None
    month: date
    amount: float


# ---------------------------------------------------------------- analytics --
class AnomalyOut(BaseModel):
    level: str            # 'transaction' | 'category_month'
    date: date
    category: str | None = None
    amount: float
    threshold: float
    description: str | None = None
