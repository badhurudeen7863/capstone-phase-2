"""
Seed the database: categories, demo users, and the multi-year transaction CSV.

Usage:  python scripts/seed_db.py [--reset]

Demo logins (password for all):  demo1234
  demo@expenseiq.app | aisha@example.com | rahul@example.com
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import Base, SessionLocal, engine, init_db
from backend.logging_setup import get_logger
from backend.models.db_models import Category, Transaction, User
from backend.services.security import hash_password

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
PASSWORD = "demo1234"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop all tables first")
    args = ap.parse_args()

    if args.reset:
        from backend.models import db_models  # noqa: F401
        Base.metadata.drop_all(bind=engine)
        logger.info("Dropped all tables.")

    init_db()
    db = SessionLocal()
    try:
        # ---- categories
        if db.query(Category).count() == 0:
            with open(BASE_DIR / "data" / "categories.csv", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    db.add(Category(name=row["name"], type=row["type"]))
            db.commit()
            logger.info("Categories seeded.")
        cats = {c.name: c.category_id for c in db.query(Category).all()}

        # ---- users (derived from the dataset so both stay in sync)
        users: dict[str, int] = {}
        with open(BASE_DIR / "data" / "transactions.csv", newline="", encoding="utf-8") as fh:
            seen = {}
            for r in csv.DictReader(fh):
                seen.setdefault(r["user_email"], r["name"])
        for email, name in seen.items():
            u = db.query(User).filter(User.email == email).first()
            if u is None:
                u = User(name=name, email=email, password_hash=hash_password(PASSWORD))
                db.add(u)
                db.commit()
                db.refresh(u)
                logger.info("Created user %s", email)
            users[email] = u.user_id

        # ---- transactions (batch insert, single commit per chunk)
        existing = db.query(Transaction).count()
        if existing:
            logger.info("Transactions already present (%d). Use --reset to reload.", existing)
            return
        rows = []
        with open(BASE_DIR / "data" / "transactions.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(
                    Transaction(
                        user_id=users[r["user_email"]],
                        category_id=cats[r["category"]],
                        amount=float(r["amount"]),
                        transaction_date=datetime.strptime(r["date"], "%Y-%m-%d").date(),
                        notes=r["description"],
                    )
                )
        for i in range(0, len(rows), 2000):
            db.add_all(rows[i:i + 2000])
            db.commit()
        logger.info("Inserted %d transactions for %d users.", len(rows), len(users))
        print(f"Seeded {len(rows)} transactions. Demo login: demo@expenseiq.app / {PASSWORD}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
