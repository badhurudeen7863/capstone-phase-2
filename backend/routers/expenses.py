"""
Expense CRUD + CSV bulk upload.
Users can only ever read/modify their own transactions (row-level ownership checks).
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..logging_setup import get_logger
from ..models.db_models import Category, Transaction, User
from ..models.schemas import (
    CSVUploadReport,
    TransactionCreate,
    TransactionOut,
    TransactionUpdate,
)
from ..services.csv_service import parse_upload

logger = get_logger(__name__)
router = APIRouter(prefix="/expenses", tags=["expenses"])


def _out(txn: Transaction) -> TransactionOut:
    return TransactionOut.from_orm_row(txn)


@router.post("", response_model=TransactionOut, status_code=status.HTTP_201_CREATED)
def create_expense(
    payload: TransactionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> TransactionOut:
    cat = db.get(Category, payload.category_id)
    if cat is None:
        raise HTTPException(status_code=404, detail="Category not found")
    txn = Transaction(
        user_id=user.user_id,
        category_id=payload.category_id,
        amount=payload.amount,
        transaction_date=payload.transaction_date,
        notes=payload.notes,
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)
    logger.info("txn created id=%s user=%s amount=%.2f", txn.id, user.user_id, txn.amount)
    return _out(txn)


@router.get("", response_model=list[TransactionOut])
def list_expenses(
    category_id: int | None = None,
    month: str | None = Query(None, description="YYYY-MM"),
    search: str | None = None,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[TransactionOut]:
    q = db.query(Transaction).filter(Transaction.user_id == user.user_id)
    if category_id:
        q = q.filter(Transaction.category_id == category_id)
    if month:
        try:
            y, m = (int(p) for p in month.split("-"))
            start, end = date(y, m, 1), date(y + (m == 12), (m % 12) + 1, 1)
        except ValueError:
            raise HTTPException(status_code=422, detail="month must be YYYY-MM")
        q = q.filter(Transaction.transaction_date >= start, Transaction.transaction_date < end)
    if search:
        like = f"%{search}%"
        cat_ids = [c.category_id for c in db.query(Category).filter(Category.name.ilike(like)).all()]
        q = q.filter(Transaction.notes.ilike(like) | Transaction.category_id.in_(cat_ids or [-1]))
    q = q.order_by(Transaction.transaction_date.desc(), Transaction.id.desc())
    txns = q.offset(offset).limit(limit).all()
    return [_out(t) for t in txns]


@router.put("/{txn_id}", response_model=TransactionOut)
def update_expense(
    txn_id: int,
    payload: TransactionUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> TransactionOut:
    txn = db.get(Transaction, txn_id)
    if txn is None or txn.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Transaction not found")
    data = payload.model_dump(exclude_unset=True)
    if "category_id" in data and db.get(Category, data["category_id"]) is None:
        raise HTTPException(status_code=404, detail="Category not found")
    for field, value in data.items():
        setattr(txn, field, value)
    db.commit()
    db.refresh(txn)
    logger.info("txn updated id=%s user=%s fields=%s", txn.id, user.user_id, list(data))
    return _out(txn)


@router.delete("/{txn_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_expense(
    txn_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    txn = db.get(Transaction, txn_id)
    if txn is None or txn.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Transaction not found")
    db.delete(txn)
    db.commit()
    logger.info("txn deleted id=%s user=%s", txn_id, user.user_id)


@router.post("/upload-csv", response_model=CSVUploadReport)
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> CSVUploadReport:
    """Bulk import: regex date normalisation + rule-based category mapping,
    inserted in a single DB transaction."""
    raw = await file.read()
    df, errors, mapping, formats = parse_upload(raw)

    name_to_cat = {c.name: c for c in db.query(Category).all()}
    inserted = 0
    if not df.empty:
        objs = []
        for rec in df.to_dict(orient="records"):
            cat = name_to_cat.get(rec["category_name"]) or name_to_cat.get("Other")
            if cat is None:  # categories not seeded yet
                raise HTTPException(status_code=500, detail="Categories not seeded")
            objs.append(
                Transaction(
                    user_id=user.user_id,
                    category_id=cat.category_id,
                    amount=rec["amount"],
                    transaction_date=rec["transaction_date"],
                    notes=rec["notes"],
                )
            )
        db.add_all(objs)  # single transaction batch insert
        db.commit()
        inserted = len(objs)
    logger.info("CSV upload user=%s inserted=%s skipped=%s", user.user_id, inserted, len(errors))
    return CSVUploadReport(
        inserted=inserted,
        skipped=len(errors),
        errors=errors[:50],
        category_mapping=dict(list(mapping.items())[:50]),
        date_formats_seen=formats,
    )
