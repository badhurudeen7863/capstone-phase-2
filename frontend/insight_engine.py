"""Pure, additive insight calculations for the optional Insights Studio page.

This module has no database, API, or Streamlit dependencies. It operates only on
records returned by the existing ExpenseIQ API, so the original backend and
storage schema remain untouched.
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping


_CADENCES = (
    (7, "Weekly", 2),
    (14, "Fortnightly", 3),
    (30, "Monthly", 5),
    (90, "Quarterly", 12),
)
_MONTHS_PER_YEAR = {"Weekly": 52, "Fortnightly": 26, "Monthly": 12, "Quarterly": 4}


def _as_date(value: Any) -> date | None:
    """Parse an API date (or Python date) without raising on malformed rows."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _as_amount(value: Any) -> float:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return 0.0
    return amount if amount > 0 else 0.0


def _month_key(value: Any) -> tuple[int, int] | None:
    """Return (year, month) for YYYY-MM or an ISO date string."""
    if value is None:
        return None
    text = str(value)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        text += "-01"
    parsed = _as_date(text)
    return (parsed.year, parsed.month) if parsed else None


def _normalize_label(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    # Receipt/reference numbers often make the same merchant look different.
    text = re.sub(r"\b\d{3,}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def detect_recurring_expenses(
    transactions: Iterable[Mapping[str, Any]], *, min_occurrences: int = 3
) -> list[dict[str, Any]]:
    """Find likely recurring payments from repeated labels and regular dates.

    The detector looks for at least ``min_occurrences`` distinct payment dates
    with weekly, fortnightly, monthly, or quarterly spacing. Amounts may vary,
    which is useful for bills such as electricity. Matches are heuristics rather
    than proof of a subscription.
    """
    if min_occurrences < 3:
        raise ValueError("min_occurrences must be at least 3")

    groups: dict[str, list[tuple[date, float, str, str]]] = defaultdict(list)
    for transaction in transactions:
        amount = _as_amount(transaction.get("amount"))
        txn_date = _as_date(transaction.get("transaction_date"))
        if not amount or txn_date is None:
            continue

        notes = str(transaction.get("notes") or "").strip()
        category = str(transaction.get("category_name") or "Other").strip() or "Other"
        category_type = str(transaction.get("category_type") or "").strip().casefold()
        # A fixed category is a reasonable fallback when an imported transaction
        # has no merchant description. Broad variable categories are not.
        label = notes or (category if category_type == "fixed" else "")
        key = _normalize_label(label)
        if not key:
            continue
        groups[key].append((txn_date, amount, label, category))

    matches: list[dict[str, Any]] = []
    for events in groups.values():
        by_day: dict[date, list[float]] = defaultdict(list)
        for txn_date, amount, _, _ in events:
            by_day[txn_date].append(amount)
        dated_amounts = sorted((day, median(amounts)) for day, amounts in by_day.items())
        if len(dated_amounts) < min_occurrences:
            continue

        gaps = [(dated_amounts[i][0] - dated_amounts[i - 1][0]).days for i in range(1, len(dated_amounts))]
        typical_days, cadence, tolerance = min(
            _CADENCES, key=lambda item: abs(median(gaps) - item[0])
        )
        if any(gap < 1 or abs(gap - typical_days) > tolerance for gap in gaps):
            continue

        amounts = [amount for _, amount in dated_amounts]
        average_amount = mean(amounts)
        amount_cv = (pstdev(amounts) / average_amount) if average_amount else 1.0
        interval_score = max(
            0.0,
            1.0 - mean(abs(gap - typical_days) for gap in gaps) / tolerance,
        )
        amount_score = max(0.0, 1.0 - min(amount_cv, 1.0))
        occurrence_score = min(1.0, len(dated_amounts) / 6.0)
        confidence = round(100 * (0.60 * interval_score + 0.25 * amount_score + 0.15 * occurrence_score))
        if confidence < 60:
            continue

        # Use the most common original merchant label and category for display.
        labels = Counter(event[2] for event in events)
        categories = Counter(event[3] for event in events)
        expected_amount = float(median(amounts))
        monthly_equivalent = expected_amount * _MONTHS_PER_YEAR[cadence] / 12.0
        annual_estimate = expected_amount * _MONTHS_PER_YEAR[cadence]
        matches.append(
            {
                "description": labels.most_common(1)[0][0],
                "category": categories.most_common(1)[0][0],
                "frequency": cadence,
                "expected_amount": round(expected_amount, 2),
                "monthly_equivalent": round(monthly_equivalent, 2),
                "annual_estimate": round(annual_estimate, 2),
                "occurrences": len(dated_amounts),
                "confidence": confidence,
                "first_seen": dated_amounts[0][0].isoformat(),
                "last_seen": dated_amounts[-1][0].isoformat(),
            }
        )

    return sorted(matches, key=lambda item: (item["confidence"], item["annual_estimate"]), reverse=True)


def _select_month_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    months: int,
    today: date | None,
    completed_only: bool = False,
) -> list[Mapping[str, Any]]:
    if months < 1:
        raise ValueError("months must be at least 1")
    as_of = today or date.today()
    dated_rows = [
        (key, row)
        for row in rows
        if (key := _month_key(row.get("month"))) is not None
    ]
    dated_rows.sort(key=lambda item: item[0])
    completed = [row for key, row in dated_rows if key < (as_of.year, as_of.month)]
    available = completed if completed_only else (completed or [row for _, row in dated_rows])
    return available[-months:]


def average_monthly_categories(
    category_month_rows: Iterable[Mapping[str, Any]],
    *,
    months: int = 3,
    today: date | None = None,
    completed_only: bool = False,
) -> dict[str, float]:
    """Average per-category spend across the latest completed available months.

    The current month is excluded when historical months exist. If the account
    has no completed month yet, available rows are used instead.
    """
    selected = _select_month_rows(
        category_month_rows, months=months, today=today, completed_only=completed_only
    )
    if not selected:
        return {}
    categories = sorted({str(key) for row in selected for key in row if key != "month"})
    return {
        category: round(
            mean(_as_amount(row.get(category, 0.0)) for row in selected), 2
        )
        for category in categories
    }


def average_monthly_total(
    monthly_rows: Iterable[Mapping[str, Any]],
    *,
    months: int = 3,
    today: date | None = None,
    completed_only: bool = False,
) -> float:
    """Average total of the latest completed available months."""
    selected = _select_month_rows(
        monthly_rows, months=months, today=today, completed_only=completed_only
    )
    if not selected:
        return 0.0
    return round(mean(_as_amount(row.get("total", 0.0)) for row in selected), 2)


def forecast_horizon_for_current_month(
    last_transaction_date: Any, *, today: date | None = None, max_horizon: int = 6
) -> int | None:
    """Choose an existing API horizon that targets this or the next month.

    Forecasts are anchored to the latest stored transaction month. Return None
    instead of presenting a stale prediction when that target is over six months
    away (or outside the endpoint's supported horizon).
    """
    last_date = _as_date(last_transaction_date)
    if last_date is None or max_horizon < 1:
        return None
    as_of = today or date.today()
    month_gap = (as_of.year - last_date.year) * 12 + as_of.month - last_date.month
    horizon = max(1, month_gap)
    return horizon if horizon <= max_horizon else None


def analyze_spending_patterns(
    transactions: Iterable[Mapping[str, Any]],
    category_month_rows: Iterable[Mapping[str, Any]] = (),
    *,
    today: date | None = None,
    category_types: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build explainable weekly, weekend, and completed-month spend insights.

    The calculations use only transaction/category-month records returned by
    ExpenseIQ's existing API. Weekly patterns are anchored to the latest
    recorded transaction, so older accounts with no current-month activity can
    still receive useful insights.
    """
    as_of = today or date.today()
    category_types = category_types or {}
    valid: list[tuple[date, str, float]] = []
    for transaction in transactions:
        txn_date = _as_date(transaction.get("transaction_date"))
        amount = _as_amount(transaction.get("amount"))
        category = str(transaction.get("category_name") or "Other").strip() or "Other"
        if txn_date is not None and txn_date <= as_of and amount > 0:
            valid.append((txn_date, category, amount))
    if not valid:
        return []

    patterns: list[dict[str, Any]] = []
    latest_date = max(row[0] for row in valid)

    # Compare the latest recorded seven-day period with the preceding week.
    recent_start = latest_date - timedelta(days=6)
    previous_start = latest_date - timedelta(days=13)
    recent_by_category: dict[str, float] = defaultdict(float)
    previous_by_category: dict[str, float] = defaultdict(float)
    for txn_date, category, amount in valid:
        if recent_start <= txn_date <= latest_date:
            recent_by_category[category] += amount
        elif previous_start <= txn_date < recent_start:
            previous_by_category[category] += amount
    for category in sorted(set(recent_by_category) | set(previous_by_category)):
        recent_amount = recent_by_category.get(category, 0.0)
        previous_amount = previous_by_category.get(category, 0.0)
        increase = recent_amount - previous_amount
        threshold = max(200.0, previous_amount * 0.25)
        if recent_amount > previous_amount and increase >= threshold:
            change_pct = (increase / previous_amount * 100.0) if previous_amount else None
            change_text = f" ({change_pct:.0f}% increase)" if change_pct is not None else " (new weekly spend)"
            patterns.append(
                {
                    "kind": "weekly_increase",
                    "category": category,
                    "title": f"{category} spending increased this week",
                    "detail": (
                        f"The latest tracked 7 days total {_inr_pattern(recent_amount)} versus "
                        f"{_inr_pattern(previous_amount)} in the preceding 7 days{change_text}."
                    ),
                    "score": increase,
                }
            )

    # Compare average spending per weekend calendar day against weekdays over
    # the latest 90 recorded days (including days with no transaction as zero).
    window_start = max(min(row[0] for row in valid), latest_date - timedelta(days=89))
    window_days = (latest_date - window_start).days + 1
    if window_days >= 21:
        weekend_days = weekday_days = 0
        daily_by_category: dict[str, dict[date, float]] = defaultdict(lambda: defaultdict(float))
        for txn_date, category, amount in valid:
            if window_start <= txn_date <= latest_date:
                daily_by_category[category][txn_date] += amount
        for offset in range(window_days):
            current_day = window_start + timedelta(days=offset)
            if current_day.weekday() >= 5:
                weekend_days += 1
            else:
                weekday_days += 1
        for category, daily in daily_by_category.items():
            active_weekend_days = sum(day.weekday() >= 5 and amount > 0 for day, amount in daily.items())
            active_weekdays = sum(day.weekday() < 5 and amount > 0 for day, amount in daily.items())
            if active_weekend_days < 2 or active_weekdays < 3 or not weekend_days or not weekday_days:
                continue
            weekend_total = sum(amount for day, amount in daily.items() if day.weekday() >= 5)
            weekday_total = sum(amount for day, amount in daily.items() if day.weekday() < 5)
            weekend_average = weekend_total / weekend_days
            weekday_average = weekday_total / weekday_days
            increase_per_day = weekend_average - weekday_average
            if increase_per_day <= 0 or increase_per_day < 25:
                continue
            change_pct = (increase_per_day / weekday_average * 100.0) if weekday_average else None
            change_text = f" ({change_pct:.0f}% higher)" if change_pct is not None else " (no weekday spend recorded)"
            patterns.append(
                {
                    "kind": "weekend_pattern",
                    "category": category,
                    "title": f"{category} spending is higher on weekends",
                    "detail": (
                        f"Average daily spend was {_inr_pattern(weekend_average)} on weekends and "
                        f"{_inr_pattern(weekday_average)} on weekdays in the latest recorded period{change_text}."
                    ),
                    "score": increase_per_day,
                }
            )

    # Month-to-month comparisons use completed months only. Rows with no
    # category key are treated as zero because the API returns a category matrix.
    month_rows: list[tuple[tuple[int, int], Mapping[str, Any]]] = []
    for row in category_month_rows:
        key = _month_key(row.get("month"))
        if key is not None and key < (as_of.year, as_of.month):
            month_rows.append((key, row))
    month_rows.sort(key=lambda item: item[0])
    if month_rows:
        latest_key, latest_row = month_rows[-1]
        variable_categories = {
            str(key): _as_amount(value)
            for key, value in latest_row.items()
            if key != "month"
            and str(category_types.get(str(key), "Variable")).casefold() != "fixed"
            and _as_amount(value) > 0
        }
        variable_total = sum(variable_categories.values())
        if variable_categories and variable_total > 0:
            top_category = max(variable_categories, key=variable_categories.get)
            top_amount = variable_categories[top_category]
            share = top_amount / variable_total * 100.0
            month_label = date(latest_key[0], latest_key[1], 1).strftime("%b %Y")
            patterns.append(
                {
                    "kind": "top_category",
                    "category": top_category,
                    "title": f"{top_category} is your largest variable-spend category",
                    "detail": (
                        f"In {month_label}, it accounted for {_inr_pattern(top_amount)} "
                        f"({share:.0f}% of variable-category spending)."
                    ),
                    "score": top_amount,
                }
            )
    if len(month_rows) >= 2:
        previous_key, previous_row = month_rows[-2]
        latest_key, latest_row = month_rows[-1]
        category_names = {
            str(key)
            for row in (previous_row, latest_row)
            for key in row
            if key != "month"
        }
        previous_label = date(previous_key[0], previous_key[1], 1).strftime("%b %Y")
        latest_label = date(latest_key[0], latest_key[1], 1).strftime("%b %Y")
        for category in sorted(category_names):
            previous_amount = _as_amount(previous_row.get(category, 0.0))
            latest_amount = _as_amount(latest_row.get(category, 0.0))
            increase = latest_amount - previous_amount
            threshold = max(200.0, previous_amount * 0.25)
            if latest_amount <= previous_amount or increase < threshold:
                continue
            change_text = (
                f" ({increase / previous_amount * 100.0:.0f}% increase)"
                if previous_amount > 0
                else " (new spending in the latest tracked month)"
            )
            patterns.append(
                {
                    "kind": "monthly_increase",
                    "category": category,
                    "title": f"{category} spending increased",
                    "detail": (
                        f"{latest_label}: {_inr_pattern(latest_amount)}; "
                        f"{previous_label}: {_inr_pattern(previous_amount)}{change_text}."
                    ),
                    "score": increase,
                }
            )

    patterns.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return [{key: value for key, value in item.items() if key != "score"} for item in patterns[:5]]


def _inr_pattern(value: float) -> str:
    return "₹" + format(float(value), ",.0f")


def build_category_spending_alerts(
    current_category_spend: Mapping[str, Any],
    monthly_baseline: Mapping[str, Any],
    category_types: Mapping[str, Any] | None = None,
    *,
    today: date | None = None,
    increase_threshold: float = 0.30,
    minimum_overage: float = 200.0,
) -> list[dict[str, Any]]:
    """Flag variable-category month-to-date spend above its normal pace.

    The baseline is scaled by the fraction of the month elapsed. Fixed costs are
    excluded because recurring rent/bills often land early in the month.
    """
    as_of = today or date.today()
    days_in_month = calendar.monthrange(as_of.year, as_of.month)[1]
    expected_fraction = as_of.day / days_in_month
    category_types = category_types or {}
    alerts: list[dict[str, Any]] = []
    for category, raw_current in current_category_spend.items():
        current_amount = _as_amount(raw_current)
        baseline = _as_amount(monthly_baseline.get(category, 0.0))
        if current_amount <= 0 or baseline <= 0:
            continue
        if str(category_types.get(category, "Variable")).casefold() == "fixed":
            continue
        expected_to_date = baseline * expected_fraction
        overage = current_amount - expected_to_date
        if overage < minimum_overage or current_amount < expected_to_date * (1.0 + increase_threshold):
            continue
        alerts.append(
            {
                "category": category,
                "current_spend": round(current_amount, 2),
                "normal_monthly_average": round(baseline, 2),
                "expected_to_date": round(expected_to_date, 2),
                "overage": round(overage, 2),
                "increase_pct": round(overage / expected_to_date * 100.0, 1) if expected_to_date else None,
            }
        )
    return sorted(alerts, key=lambda item: item["overage"], reverse=True)


def build_saving_recommendations(
    monthly_income: Any,
    current_expenses: Any,
    predicted_expenses: Any | None,
    monthly_category_baseline: Mapping[str, Any],
    current_category_spend: Mapping[str, Any],
    category_types: Mapping[str, Any] | None = None,
    spending_patterns: Iterable[Mapping[str, Any]] = (),
    *,
    historical_expenses: Any | None = None,
) -> dict[str, Any]:
    """Return a transparent savings estimate plus one category-level action.

    When a model forecast is unavailable, recent historical spend is used as a
    labelled fallback. The planning expense is the larger of the forecast and
    current month-to-date actual, making the savings estimate conservative.
    """
    income = _as_amount(monthly_income)
    current = _as_amount(current_expenses)
    try:
        prediction = float(predicted_expenses) if predicted_expenses is not None else 0.0
    except (TypeError, ValueError):
        prediction = 0.0
    source = "ML forecast" if prediction > 0 else ""
    if prediction <= 0:
        try:
            prediction = float(historical_expenses) if historical_expenses is not None else 0.0
        except (TypeError, ValueError):
            prediction = 0.0
        if prediction <= 0:
            prediction = sum(_as_amount(value) for value in monthly_category_baseline.values())
        if prediction > 0:
            source = "recent monthly average"
        else:
            prediction = 0.0

    # A partial month by itself is not a full monthly forecast. Use current
    # spend conservatively only when a forecast or completed-month baseline exists.
    has_expense_estimate = prediction > 0
    planning_expense = max(prediction, current) if has_expense_estimate else 0.0
    category_types = category_types or {}
    pattern_categories = {
        str(pattern.get("category"))
        for pattern in spending_patterns
        if pattern.get("category")
    }
    eligible: list[tuple[str, float, bool]] = []
    categories = set(monthly_category_baseline) | set(current_category_spend)
    for category in sorted(categories):
        if str(category_types.get(category, "Variable")).casefold() == "fixed":
            continue
        reference = max(
            _as_amount(monthly_category_baseline.get(category, 0.0)),
            _as_amount(current_category_spend.get(category, 0.0)),
        )
        if reference > 0:
            eligible.append((str(category), reference, str(category) in pattern_categories))
    eligible.sort(key=lambda item: (item[2], item[1]), reverse=True)
    category_cut = None
    category_recommendation = None
    if eligible and income > 0:
        category, reference, _ = eligible[0]
        raw_cut = min(reference * 0.10, income * 0.05)
        category_cut = round(raw_cut / 50.0) * 50.0 if raw_cut >= 50 else round(raw_cut, 2)
        if category_cut > 0:
            category_recommendation = {
                "category": category,
                "suggested_cut": round(category_cut, 2),
                "reference_spend": round(reference, 2),
            }

    suggested_saving = max(income - planning_expense, 0.0) if income > 0 and has_expense_estimate else None
    shortfall = max(planning_expense - income, 0.0) if income > 0 and has_expense_estimate else None
    if income <= 0:
        status = "income_required"
    elif not has_expense_estimate:
        status = "history_required"
    else:
        status = "ready"
    return {
        "status": status,
        "monthly_income": round(income, 2),
        "current_expenses": round(current, 2),
        "predicted_expenses": round(prediction, 2) if prediction > 0 else None,
        "expense_source": source or None,
        "planning_expense": round(planning_expense, 2) if has_expense_estimate else None,
        "suggested_saving": round(suggested_saving, 2) if suggested_saving is not None else None,
        "shortfall": round(shortfall, 2) if shortfall is not None else None,
        "category_recommendation": category_recommendation,
    }


def simulate_category_cuts(
    monthly_baseline: Mapping[str, Any], cut_percentages: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply non-persistent category reduction percentages to a monthly baseline."""
    breakdown: dict[str, dict[str, float]] = {}
    for category, raw_amount in sorted(monthly_baseline.items()):
        amount = _as_amount(raw_amount)
        try:
            cut = float(cut_percentages.get(category, 0.0))
        except (TypeError, ValueError):
            cut = 0.0
        cut = min(100.0, max(0.0, cut))
        simulated = amount * (1.0 - cut / 100.0)
        breakdown[category] = {
            "baseline": round(amount, 2),
            "cut_percent": round(cut, 1),
            "simulated": round(simulated, 2),
            "savings": round(amount - simulated, 2),
        }

    baseline_total = sum(row["baseline"] for row in breakdown.values())
    scenario_total = sum(row["simulated"] for row in breakdown.values())
    monthly_savings = baseline_total - scenario_total
    return {
        "categories": breakdown,
        "baseline_monthly": round(baseline_total, 2),
        "scenario_monthly": round(scenario_total, 2),
        "monthly_savings": round(monthly_savings, 2),
        "annual_savings": round(monthly_savings * 12.0, 2),
    }


def project_month_end_spend(
    transactions: Iterable[Mapping[str, Any]],
    *,
    today: date | None = None,
    monthly_limit: float | None = None,
) -> dict[str, Any]:
    """Project month-end outflow from current month-to-date spending pace."""
    as_of = today or date.today()
    days_in_month = calendar.monthrange(as_of.year, as_of.month)[1]
    elapsed_days = as_of.day
    remaining_days = max(days_in_month - elapsed_days, 0)

    spent = 0.0
    count = 0
    for transaction in transactions:
        txn_date = _as_date(transaction.get("transaction_date"))
        amount = _as_amount(transaction.get("amount"))
        if (
            txn_date is not None
            and txn_date.year == as_of.year
            and txn_date.month == as_of.month
            and txn_date <= as_of
            and amount > 0
        ):
            spent += amount
            count += 1

    has_data = count > 0
    daily_rate = spent / elapsed_days if elapsed_days else 0.0
    projected = daily_rate * days_in_month if has_data else 0.0
    limit = None if monthly_limit is None else max(0.0, float(monthly_limit))
    daily_allowance = None
    if limit is not None and remaining_days > 0:
        daily_allowance = max((limit - spent) / remaining_days, 0.0)

    if limit is None:
        status = "limit_not_set"
    elif not has_data:
        status = "no_spend_yet"
    else:
        status = "at_risk" if projected > limit else "on_track"

    overage = max(projected - limit, 0.0) if limit is not None else None
    return {
        "spent_to_date": round(spent, 2),
        "projected_month_end": round(projected, 2),
        "current_daily_rate": round(daily_rate, 2),
        "days_in_month": days_in_month,
        "elapsed_days": elapsed_days,
        "remaining_days": remaining_days,
        "transactions_count": count,
        "has_data": has_data,
        "monthly_limit": round(limit, 2) if limit is not None else None,
        "remaining_daily_allowance": round(daily_allowance, 2) if daily_allowance is not None else None,
        "projected_overage": round(overage, 2) if overage is not None else None,
        "status": status,
    }
