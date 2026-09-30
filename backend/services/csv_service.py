"""
CSV bulk upload (report: pandas parsing, regex date normalisation,
rule-based category mapping, single-transaction batch insert).
"""
from __future__ import annotations

import io
import re
from datetime import datetime

import pandas as pd

DATE_CANDIDATE_COLS = ["date", "transaction_date", "txn_date", "value_date", "dt"]
AMOUNT_CANDIDATE_COLS = ["amount", "value", "debit", "spend", "price", "total"]
DESC_CANDIDATE_COLS = ["description", "notes", "narration", "details", "merchant", "particulars"]
CAT_CANDIDATE_COLS = ["category", "category_name", "type_of_expense"]

# Regex date normalisation: day-first (Indian format) preference.
_DATE_PATTERNS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), ("%Y-%m-%d",)),
    (re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"), ("%Y/%m/%d",)),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"), ("%d/%m/%Y",)),
    (re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"), ("%d-%m-%Y",)),
    (re.compile(r"^\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}$"), ("%d %b %Y",)),
    (re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$"), ("%b %d, %Y", "%b %d %Y")),
    (re.compile(r"^\d{2}-[A-Za-z]{3}-\d{4}$"), ("%d-%b-%Y",)),
]

# Rule-based category mapping keywords -> seeded category name.
CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Rent", re.compile(r"\brent\b|lease|society maintenance|landlord", re.I)),
    ("Utilities", re.compile(r"electricity|power bill|water bill|internet|broadband|fibernet|wifi|gas bill|phone bill|recharge|tneb|bsnl|tatasky|discom", re.I)),
    ("Transport", re.compile(r"uber|ola|rapido|bike|fuel|petrol|diesel|bus|train|metro|taxi|cab|parking|fastag|flight", re.I)),
    ("Health", re.compile(r"doctor|pharma|medicine|hospital|clinic|gym|fitness|dentist|lab test|health", re.I)),
    ("Entertainment", re.compile(r"movie|cinema|netflix|amazon prime|spotify|concert|game|restaurant|cafe|swiggy instamart|zomato pro", re.I)),
    ("Food & Groceries", re.compile(r"grocer|supermarket|kirana|swiggy|zomato|food|bakery|vegetable|fruit|milk|mart", re.I)),
]
FALLBACK_CATEGORY = "Other"


def normalise_date(raw: str) -> tuple[datetime | None, str | None]:
    """Try regex-driven format detection; returns (parsed, format_seen)."""
    s = str(raw).strip()
    for pattern, formats in _DATE_PATTERNS:
        if pattern.match(s):
            for candidate in formats:
                try:
                    return datetime.strptime(s, candidate), candidate
                except ValueError:
                    continue
    # last resort: pandas guess
    try:
        return pd.to_datetime(s, dayfirst=True), "parsed_fallback"
    except Exception:
        return None, None


def map_category(text: str | None) -> str:
    if not text:
        return FALLBACK_CATEGORY
    for name, pattern in CATEGORY_RULES:
        if pattern.search(text):
            return name
    return FALLBACK_CATEGORY


def parse_upload(raw_bytes: bytes) -> tuple[pd.DataFrame, list[str], dict[str, str], list[str]]:
    """Returns (normalised dataframe, error list, category mapping used, date formats seen)."""
    errors: list[str] = []
    mapping: dict[str, str] = {}
    formats: set[str] = set()

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # malformed csv
        return pd.DataFrame(), [f"unreadable CSV: {exc}"], {}, []

    cols = {c.strip().lower(): c for c in df.columns}

    def pick(candidates: list[str], substrings: list[str]) -> str | None:
        for c in candidates:
            if c in cols:
                return cols[c]
        for key, original in cols.items():
            if any(sub in key for sub in substrings):
                return original
        return None

    date_col = pick(DATE_CANDIDATE_COLS, ["date"])
    amount_col = pick(AMOUNT_CANDIDATE_COLS, ["amount", "debit", "value", "spend", "price"])
    if date_col is None or amount_col is None:
        return pd.DataFrame(), ["CSV must contain a date column and an amount column"], {}, []
    desc_col = pick(DESC_CANDIDATE_COLS, ["desc", "narration", "note", "merchant", "particular", "detail"])
    cat_col = pick(CAT_CANDIDATE_COLS, ["categ"])

    records = []
    for i, row in df.iterrows():
        parsed, fmt = normalise_date(row[date_col])
        if parsed is None:
            errors.append(f"row {i + 2}: unparseable date '{row[date_col]}'")
            continue
        formats.add(fmt or "unknown")
        try:
            amount = float(str(row[amount_col]).replace(",", "").replace("₹", "").strip())
        except ValueError:
            errors.append(f"row {i + 2}: unparseable amount '{row[amount_col]}'")
            continue
        if amount <= 0:
            errors.append(f"row {i + 2}: non-positive amount {amount}")
            continue
        desc = str(row[desc_col]) if desc_col else ""
        if cat_col and pd.notna(row[cat_col]):
            cat = str(row[cat_col]).strip()
            mapped = cat  # honour an explicit category column when present
        else:
            mapped = map_category(desc)
        mapping[desc or f"row {i + 2}"] = mapped
        records.append(
            {
                "transaction_date": parsed.date(),
                "amount": round(amount, 2),
                "notes": desc or None,
                "category_name": mapped,
            }
        )
    return pd.DataFrame(records), errors, mapping, sorted(formats)
