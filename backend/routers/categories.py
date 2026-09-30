"""Category listing (seeded: Rent, Utilities, Food & Groceries, Health, Transport, Entertainment, Other)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models.db_models import Category, User
from ..models.schemas import CategoryOut

router = APIRouter(prefix="/categories", tags=["categories"])


@router.get("", response_model=list[CategoryOut])
def list_categories(
    db: Session = Depends(get_db), _user: User = Depends(get_current_user)
) -> list[CategoryOut]:
    cats = db.query(Category).order_by(Category.category_id).all()
    return [CategoryOut.model_validate(c) for c in cats]
