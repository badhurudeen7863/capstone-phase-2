"""Unit tests for the optional, additive Insights Studio calculations."""
from datetime import date, timedelta

from frontend.insight_engine import (
    analyze_spending_patterns,
    average_monthly_categories,
    average_monthly_total,
    build_category_spending_alerts,
    build_saving_recommendations,
    detect_recurring_expenses,
    forecast_horizon_for_current_month,
    project_month_end_spend,
    simulate_category_cuts,
)


def test_detects_regular_monthly_payment_and_estimates_annual_cost():
    transactions = [
        {"transaction_date": f"2025-{month:02d}-01", "amount": amount,
         "notes": "StreamPlus subscription", "category_name": "Entertainment"}
        for month, amount in [(1, 499), (2, 499), (3, 549), (4, 499)]
    ]

    matches = detect_recurring_expenses(transactions)

    assert len(matches) == 1
    assert matches[0]["frequency"] == "Monthly"
    assert matches[0]["expected_amount"] == 499
    assert matches[0]["annual_estimate"] == 5988
    assert matches[0]["confidence"] >= 70


def test_irregular_transactions_are_not_called_recurring():
    transactions = [
        {"transaction_date": "2025-01-01", "amount": 100, "notes": "One-off shop"},
        {"transaction_date": "2025-01-05", "amount": 100, "notes": "One-off shop"},
        {"transaction_date": "2025-01-29", "amount": 100, "notes": "One-off shop"},
    ]

    assert detect_recurring_expenses(transactions) == []


def test_category_baseline_excludes_partial_current_month():
    rows = [
        {"month": "2025-01-01", "Food": 100, "Rent": 1000},
        {"month": "2025-02-01", "Food": 200, "Rent": 1000},
        {"month": "2025-03-01", "Food": 300, "Rent": 1000},
        {"month": "2025-04-01", "Food": 900, "Rent": 1000},
    ]

    baseline = average_monthly_categories(rows, months=3, today=date(2025, 4, 16))

    assert baseline == {"Food": 200.0, "Rent": 1000.0}


def test_monthly_total_baseline_uses_latest_completed_months():
    rows = [
        {"month": "2025-01", "total": 1000},
        {"month": "2025-02", "total": 2000},
        {"month": "2025-03", "total": 3000},
        {"month": "2025-04", "total": 9000},
    ]

    assert average_monthly_total(rows, months=2, today=date(2025, 4, 10)) == 2500


def test_what_if_simulation_is_non_mutating_and_annualizes_savings():
    baseline = {"Food": 200, "Rent": 1000}
    result = simulate_category_cuts(baseline, {"Food": 25, "Rent": 50})

    assert result["baseline_monthly"] == 1200
    assert result["scenario_monthly"] == 650
    assert result["monthly_savings"] == 550
    assert result["annual_savings"] == 6600
    assert baseline == {"Food": 200, "Rent": 1000}


def test_month_end_pace_projects_spend_and_daily_room():
    transactions = [
        {"transaction_date": "2025-04-01", "amount": 50},
        {"transaction_date": "2025-04-10", "amount": 150},
        {"transaction_date": "2025-04-11", "amount": 500},  # future relative to as_of
        {"transaction_date": "2025-03-31", "amount": 999},
    ]

    result = project_month_end_spend(transactions, today=date(2025, 4, 10), monthly_limit=500)

    assert result["spent_to_date"] == 200
    assert result["projected_month_end"] == 600
    assert result["remaining_daily_allowance"] == 15
    assert result["projected_overage"] == 100
    assert result["status"] == "at_risk"


def test_detects_higher_weekend_spending_from_daily_transactions():
    today = date(2025, 4, 30)
    start = today - timedelta(days=89)
    transactions = []
    for offset in range(90):
        day = start + timedelta(days=offset)
        amount = 300 if day.weekday() >= 5 else 100
        transactions.append({
            "transaction_date": day.isoformat(),
            "amount": amount,
            "category_name": "Food & Groceries",
        })

    patterns = analyze_spending_patterns(transactions, today=today)

    weekend = next(pattern for pattern in patterns if pattern["kind"] == "weekend_pattern")
    assert weekend["category"] == "Food & Groceries"
    assert "higher on weekends" in weekend["title"]


def test_reports_largest_variable_category_from_real_month_data():
    transactions = [{
        "transaction_date": "2025-01-15",
        "amount": 100,
        "category_name": "Food",
    }]
    category_months = [{
        "month": "2025-01-01",
        "Rent": 10000,
        "Food": 4000,
        "Transport": 1500,
    }]

    patterns = analyze_spending_patterns(
        transactions,
        category_months,
        today=date(2025, 2, 1),
        category_types={"Rent": "Fixed", "Food": "Variable", "Transport": "Variable"},
    )

    largest = next(pattern for pattern in patterns if pattern["kind"] == "top_category")
    assert largest["category"] == "Food"
    assert "₹4,000" in largest["detail"]


def test_detects_category_increase_between_completed_months():
    transactions = [{
        "transaction_date": "2025-03-15",
        "amount": 100,
        "category_name": "Food",
    }]
    category_months = [
        {"month": "2025-02-01", "Food": 2000},
        {"month": "2025-03-01", "Food": 3500},
        {"month": "2025-04-01", "Food": 8000},  # partial current month is excluded
    ]

    patterns = analyze_spending_patterns(transactions, category_months, today=date(2025, 4, 16))

    monthly = next(pattern for pattern in patterns if pattern["kind"] == "monthly_increase")
    assert monthly["category"] == "Food"
    assert "75% increase" in monthly["detail"]
    assert "Apr 2025" not in monthly["detail"]


def test_category_alert_compares_current_pace_with_historical_average():
    alerts = build_category_spending_alerts(
        {"Food": 3500, "Rent": 10000},
        {"Food": 2000, "Rent": 10000},
        {"Food": "Variable", "Rent": "Fixed"},
        today=date(2025, 4, 30),
    )

    assert len(alerts) == 1
    assert alerts[0]["category"] == "Food"
    assert alerts[0]["expected_to_date"] == 2000
    assert alerts[0]["overage"] == 1500
    assert alerts[0]["increase_pct"] == 75


def test_saving_plan_uses_income_forecast_current_spend_and_patterns():
    plan = build_saving_recommendations(
        monthly_income=15000,
        current_expenses=8000,
        predicted_expenses=12000,
        monthly_category_baseline={"Food": 4000, "Shopping": 5000, "Rent": 7000},
        current_category_spend={"Food": 1800, "Shopping": 2000, "Rent": 7000},
        category_types={"Food": "Variable", "Shopping": "Variable", "Rent": "Fixed"},
        spending_patterns=[{"category": "Shopping", "kind": "monthly_increase"}],
    )

    assert plan["status"] == "ready"
    assert plan["planning_expense"] == 12000
    assert plan["suggested_saving"] == 3000
    assert plan["category_recommendation"] == {
        "category": "Shopping", "suggested_cut": 500.0, "reference_spend": 5000.0
    }


def test_savings_and_alerts_handle_empty_history_without_inventing_numbers():
    assert analyze_spending_patterns([], [], today=date(2025, 4, 16)) == []
    assert build_category_spending_alerts({}, {}, today=date(2025, 4, 16)) == []
    plan = build_saving_recommendations(15000, 0, None, {}, {}, historical_expenses=None)
    assert plan["status"] == "history_required"
    assert plan["suggested_saving"] is None


def test_forecast_horizon_targets_current_month_or_skips_stale_history():
    assert forecast_horizon_for_current_month("2025-02-20", today=date(2025, 4, 1)) == 2
    assert forecast_horizon_for_current_month("2024-01-01", today=date(2025, 4, 1)) is None
