"""
Synthetic multi-year household expense generator (the report's dataset is synthetic).

Produces:
  data/transactions.csv     - multi-year, multi-user labelled transactions
  data/sample_upload.csv    - messy 'bank statement' for demoing CSV bulk upload
                              (mixed date formats + raw merchant descriptions)

Data-generating process (documented in README):
  * smooth seasonal curve + sharp Diwali (Oct/Nov) festival block
  * linear inflation trend, piecewise transport regime switch (2023)
  * non-linear electricity response  900 + 3200 * heat^1.7
  * controlled stochastic noise + a fixed schedule of rare anomaly events
This structure is additive-smooth with sharp blocks, so the
LinearRegression / RandomForest / XGBoost comparison is meaningful.

Usage:  python scripts/generate_synthetic_data.py [--seed 42] [--start 2016-01-01] [--end 2026-08-31]
"""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "data"

USERS = [
    {"email": "demo@expenseiq.app", "name": "Demo User", "scale": 1.00},
    {"email": "aisha@example.com", "name": "Aisha Rahman", "scale": 1.25},
    {"email": "rahul@example.com", "name": "Rahul Kumar", "scale": 0.80},
    {"email": "meera@example.com", "name": "Meera Iyer", "scale": 1.10},
]

# Rare, deliberately injected anomalies (also power the dashboard alerts demo).
ANOMALY_SCHEDULE = [
    ("demo@expenseiq.app", date(2026, 7, 14), 9800.0, "Hospital admission"),
    ("demo@expenseiq.app", date(2026, 2, 9), 7600.0, "Emergency dental surgery"),
    ("aisha@example.com", date(2024, 5, 22), 8900.0, "Hospital admission"),
    ("rahul@example.com", date(2023, 12, 3), 7200.0, "Accident repair"),
]

TRANSPORT_DAYS = [2, 4, 8, 11, 15, 18, 22, 25, 29]
GROCERY_DAYS = [3, 10, 17, 24]
VEG_DAYS = [6, 20]
ENT_DAYS = [7, 16, 27]


def _month_season(m: int) -> float:
    """Smooth seasonal curve (summer cooling + winter)."""
    return 0.18 * np.sin(2 * np.pi * (m - 3) / 12.0) + 0.10 * np.cos(2 * np.pi * (m - 10) / 12.0)


def _txn(u: dict, d: date, category: str, amount: float, notes: str) -> dict:
    return {
        "user_email": u["email"],
        "name": u["name"],
        "date": d.isoformat(),
        "category": category,
        "description": notes,
        "amount": max(20.0, round(amount, 2)),
    }


def generate(start: date, end: date, rng: np.random.Generator, pyrng: random.Random) -> pd.DataFrame:
    rows: list[dict] = []
    anomaly_by_user: dict[str, list[tuple[date, float, str]]] = {}
    for email, d, amt, note in ANOMALY_SCHEDULE:
        anomaly_by_user.setdefault(email, []).append((d, amt, note))

    first = start.replace(day=1)
    last = end.replace(day=1)
    m = first
    while m <= last:
        months_since = (m.year - first.year) * 12 + (m.month - first.month)
        inflation = 1.0 + 0.0045 * months_since                 # ~5.4 % / year
        season = float(_month_season(m.month))
        festival = 1.35 if m.month in (10, 11) else 1.0         # sharp Diwali block
        heat = max(0.0, season + 0.15)
        commute_base = 95.0 if m < date(2023, 1, 1) else 165.0  # piecewise regime switch
        days_in_month = (pd.Timestamp(m) + pd.offsets.MonthEnd(0)).day

        for u in USERS:
            s = u["scale"]

            def add(day: int, cat: str, amount: float, notes: str) -> None:
                if 1 <= day <= days_in_month and m.replace(day=day) <= end:
                    rows.append(_txn(u, m.replace(day=day), cat, amount, notes))

            # ---- fixed contracts -------------------------------------------
            add(1, "Rent", 8500 * s * inflation + rng.normal(0, 50), "Monthly rent")
            if m.month % 3 == 1:
                add(5, "Rent", 2200 * s + rng.normal(0, 120), "Society maintenance")
            add(8, "Utilities", (900 + 3200 * heat ** 1.7) * s * inflation + rng.normal(0, 50), "Electricity bill")
            add(10, "Utilities", 799 * s * inflation + rng.normal(0, 8), "Broadband internet")

            # ---- variable spend ---------------------------------------------
            for day in GROCERY_DAYS:
                base = 1500 * (1 + 0.75 * season) * festival
                add(day, "Food & Groceries", base * s * inflation * (1 + rng.normal(0, 0.06)), "Supermarket run")
            for day in VEG_DAYS:
                add(day, "Food & Groceries", 450 * s * inflation * (1 + rng.normal(0, 0.08)), "Vegetables and fruit")
            if pyrng.random() < 0.20:
                add(pyrng.randint(11, 26), "Food & Groceries", 800 * s * inflation * (1 + rng.normal(0, 0.10)), "Kirana store top-up")
            for day in TRANSPORT_DAYS:
                add(day, "Transport", commute_base * s * inflation * (1 + rng.normal(0, 0.12)), "Fuel / commute")
            if pyrng.random() < 0.85:
                add(12, "Health", 700 * s * (1 + rng.normal(0, 0.08)), "Doctor visit")
            if pyrng.random() < 0.15:
                add(21, "Health", 1500 * s * (1 + rng.normal(0, 0.10)), "Pharmacy / lab tests")
            for day in ENT_DAYS:
                add(day, "Entertainment", 380 * festival * s * inflation * (1 + rng.normal(0, 0.12)), "Movies / dining out")

            # ---- injected anomalies ------------------------------------------
            for d, amt, note in anomaly_by_user.get(u["email"], []):
                if m <= d.replace(day=1) <= last and d.month == m.month and d.year == m.year:
                    rows.append(_txn(u, d, "Health", amt * s, note))
        m = (m + pd.offsets.MonthBegin(1)).to_pydatetime().date()

    df = pd.DataFrame(rows).sort_values(["user_email", "date"]).reset_index(drop=True)
    return df


def make_sample_upload(rng: random.Random) -> pd.DataFrame:
    """Messy statement: mixed date formats + raw merchant names (for CSV upload demo)."""
    merchants = [
        ("SWIGGY PAYMENT", "Food & Groceries"), ("BIGBASKET SUPERMARKET", "Food & Groceries"),
        ("UBER TRIP", "Transport"), ("INDIAN OIL PETROL", "Transport"),
        ("TNEB ELECTRICITY", "Utilities"), ("ACT FIBERNET", "Utilities"),
        ("APOLLO PHARMACY", "Health"), ("CULT GYM MEMBERSHIP", "Health"),
        ("PVR CINEMAS", "Entertainment"), ("HOUSE RENT PAYMENT", "Rent"),
        ("DMART GROCERS", "Food & Groceries"), ("RAPIDO BIKE", "Transport"),
    ]
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%b %d, %Y"]
    rows = []
    d = date(2026, 6, 1)
    for _ in range(40):
        desc, _cat = rng.choice(merchants)
        day = d + timedelta(days=rng.randint(0, 85))
        rows.append({
            "Txn Date": day.strftime(rng.choice(fmts)),
            "Narration": desc,
            "Debit": round(rng.uniform(120, 4200), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-08-31")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pyrng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    df = generate(date.fromisoformat(args.start), date.fromisoformat(args.end), rng, pyrng)
    df.to_csv(OUT / "transactions.csv", index=False)
    print(f"wrote {OUT / 'transactions.csv'}: {len(df)} rows, "
          f"{df['user_email'].nunique()} users, {df['date'].min()} .. {df['date'].max()}")

    up = make_sample_upload(pyrng)
    up.to_csv(OUT / "sample_upload.csv", index=False)
    print(f"wrote {OUT / 'sample_upload.csv'}: {len(up)} messy rows")

    cats = pd.DataFrame(
        [("Rent", "Fixed"), ("Utilities", "Fixed"), ("Food & Groceries", "Variable"),
         ("Health", "Variable"), ("Transport", "Variable"), ("Entertainment", "Variable"),
         ("Other", "Variable")],
        columns=["name", "type"],
    )
    cats.to_csv(OUT / "categories.csv", index=False)
    print(f"wrote {OUT / 'categories.csv'}")


if __name__ == "__main__":
    main()
