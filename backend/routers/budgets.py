"""Monthly budgets per category (upsert + list)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models.db_models import Budget, Category, User
from ..models.schemas import BudgetOut, BudgetUpsert

router = APIRouter(prefix="/budgets", tags=["budgets"])


@router.get("", response_model=list[BudgetOut])
def list_budgets(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[BudgetOut]:
    rows = db.query(Budget).filter(Budget.user_id == user.user_id).order_by(Budget.month.desc()).all()
    return [
        BudgetOut(
            budget_id=b.budget_id,
            user_id=b.user_id,
            category_id=b.category_id,
            category_name=b.category.name if b.category else None,
            month=b.month,
            amount=float(b.amount),
        )
        for b in rows
    ]


@router.put("", response_model=BudgetOut)
def upsert_budget(
    payload: BudgetUpsert, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> BudgetOut:
    cat = db.get(Category, payload.category_id)
    if cat is None:
        raise HTTPException(status_code=404, detail="Category not found")
    month = payload.month.replace(day=1)
    row = (
        db.query(Budget)
        .filter(
            Budget.user_id == user.user_id,
            Budget.category_id == payload.category_id,
            Budget.month == month,
        )
        .first()
    )
    if row is None:
        row = Budget(user_id=user.user_id, category_id=payload.category_id, month=month, amount=payload.amount)
        db.add(row)
    else:
        row.amount = payload.amount
    db.commit()
    db.refresh(row)
    return BudgetOut(
        budget_id=row.budget_id,
        user_id=row.user_id,
        category_id=row.category_id,
        category_name=row.category.name if row.category else None,
        month=row.month,
        amount=float(row.amount),
    )
