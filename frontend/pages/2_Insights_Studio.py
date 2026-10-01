"""Optional, read-only ExpenseIQ Insights Studio.

This is a Streamlit multipage add-on. It uses existing authenticated API
endpoints and does not write transactions, budgets, or database schema changes.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.client import APIClient, APIError
from frontend.insight_engine import (
    average_monthly_categories,
    average_monthly_total,
    detect_recurring_expenses,
    project_month_end_spend,
    simulate_category_cuts,
)

st.set_page_config(page_title="ExpenseIQ Insights Studio", page_icon="✨", layout="wide")
st.markdown(
    """
<style>
html, body, [class*="st-"] { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: #f4f6f5; }
div[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1250px; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #e6eae8;
  border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.insight-card { background:#fff; border:1px solid #e6eae8; border-radius:12px;
  padding:16px 20px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
.insight-muted { color:#667085; font-size:.86rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_transactions(token: str, month: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": 5000}
    if month:
        params["month"] = month
    return APIClient(token).list_expenses(**params)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_categories(token: str) -> list[dict[str, Any]]:
    return APIClient(token).categories()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_category_month(token: str) -> list[dict[str, Any]]:
    return APIClient(token).category_month()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_monthly(token: str) -> list[dict[str, Any]]:
    return APIClient(token).monthly()


def _inr(value: Any) -> str:
    try:
        return "₹" + format(float(value), ",.2f")
    except (TypeError, ValueError):
        return "—"


def _login_page() -> None:
    st.title("ExpenseIQ · Insights Studio")
    st.caption("Sign in with your existing ExpenseIQ account to open the three new insights.")
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        mode = st.radio("Account", ["Login", "Register"], horizontal=True)
        with st.form("insights_auth_form"):
            name = st.text_input("Full name") if mode == "Register" else ""
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in" if mode == "Login" else "Create account", type="primary")
        if submitted:
            client = APIClient()
            try:
                response = (
                    client.login(email.strip(), password)
                    if mode == "Login"
                    else client.register(name.strip(), email.strip(), password)
                )
                st.session_state.token = response["access_token"]
                st.session_state.user = response["user"]
                st.rerun()
            except APIError as exc:
                st.error(str(exc.detail))
            except Exception as exc:  # connection failures are not APIError responses
                st.error(f"Could not reach the ExpenseIQ API: {exc}")
        st.caption("The existing FastAPI service must be running. No data is changed by this add-on.")


def _render_recurring_radar(token: str) -> None:
    st.title("Recurring Payment Radar")
    st.caption("Find likely bills and subscriptions from repeated merchant descriptions and payment dates.")
    with st.spinner("Scanning your latest transactions…"):
        transactions = _cached_transactions(token)
    matches = detect_recurring_expenses(transactions)

    monthly_outflow = sum(item["monthly_equivalent"] for item in matches)
    most_confident = max((item["confidence"] for item in matches), default=0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Likely recurring patterns", f"{len(matches)}")
    c2.metric("Estimated recurring outflow / month", _inr(monthly_outflow))
    c3.metric("Highest pattern confidence", f"{most_confident}%" if matches else "—")

    if not transactions:
        st.info("No transactions were found for this account yet.")
    elif not matches:
        st.info("No strong recurring patterns found. The detector needs at least three payments with a regular weekly, fortnightly, monthly, or quarterly cadence.")
    else:
        table = pd.DataFrame(
            [
                {
                    "Description": item["description"],
                    "Category": item["category"],
                    "Cadence": item["frequency"],
                    "Typical payment": _inr(item["expected_amount"]),
                    "Est. monthly": _inr(item["monthly_equivalent"]),
                    "Seen": item["occurrences"],
                    "Confidence": f"{item['confidence']}%",
                    "Last seen": item["last_seen"],
                }
                for item in matches
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        top = matches[:10][::-1]
        fig = go.Figure(
            go.Bar(
                x=[item["monthly_equivalent"] for item in top],
                y=[item["description"] for item in top],
                orientation="h",
                marker_color="#1a7f5a",
                text=[_inr(item["monthly_equivalent"]) for item in top],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Estimated monthly cost of likely recurring payments",
            height=max(280, 42 * len(top)),
            margin=dict(l=0, r=10, t=45, b=0),
            xaxis=dict(tickprefix="₹", tickformat=",.0f"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("How this estimate works"):
        st.write(
            "Descriptions are normalized to reduce reference-number differences. "
            "A match needs at least three distinct payment dates with a steady weekly, "
            "fortnightly, monthly, or quarterly interval. Amount consistency and the "
            "number of observations contribute to confidence. This is a pattern finder, "
            "not confirmation that a merchant will charge you again."
        )
        st.caption("Scans up to the latest 5,000 transactions returned by the existing API.")


def _render_what_if_lab(token: str) -> None:
    st.title("Budget What-if Lab")
    st.caption("Try hypothetical category reductions and see their monthly and annual impact — without saving a budget or changing expenses.")
    category_rows = _cached_category_month(token)
    categories = _cached_categories(token)
    baseline = average_monthly_categories(category_rows, months=3, today=date.today())
    if not baseline:
        st.info("Add a few months of transactions to build a category baseline.")
        return

    category_types = {str(item.get("name")): str(item.get("type", "Variable")) for item in categories}
    include_fixed = st.checkbox("Include fixed-cost categories in the scenario", value=False)
    adjustable = [
        name for name in baseline
        if include_fixed or category_types.get(name, "Variable").casefold() != "fixed"
    ]
    st.caption("Baseline averages up to the three latest completed months with recorded transactions; the current partial month is excluded when history is available.")

    if not adjustable:
        st.info("There are no adjustable categories in the selected history. Turn on fixed-cost categories to include them.")
        return

    st.markdown("#### Choose reductions")
    st.caption("Set a 0–50% hypothetical reduction for each category. This is a planning simulation only.")
    columns = st.columns(2)
    cuts: dict[str, int] = {}
    for index, name in enumerate(adjustable):
        with columns[index % 2]:
            amount = baseline[name]
            cuts[name] = st.slider(
                f"{name} · monthly baseline {_inr(amount)}",
                min_value=0,
                max_value=50,
                value=0,
                step=5,
                key=f"insights_whatif_{name}",
            )

    result = simulate_category_cuts(baseline, cuts)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline monthly spend", _inr(result["baseline_monthly"]))
    c2.metric("Scenario monthly spend", _inr(result["scenario_monthly"]))
    c3.metric("Potential savings / month", _inr(result["monthly_savings"]))
    st.caption(f"Annualized at the same hypothetical pace: {_inr(result['annual_savings'])}.")

    plotted = sorted(
        result["categories"].items(),
        key=lambda item: item[1]["baseline"],
        reverse=True,
    )
    fig = go.Figure()
    fig.add_bar(name="Baseline", x=[item[1]["baseline"] for item in plotted], y=[item[0] for item in plotted], orientation="h", marker_color="#a8cbb9")
    fig.add_bar(name="Scenario", x=[item[1]["simulated"] for item in plotted], y=[item[0] for item in plotted], orientation="h", marker_color="#1a7f5a")
    fig.update_layout(
        title="Monthly baseline vs. scenario",
        barmode="group",
        height=max(300, 42 * len(plotted)),
        margin=dict(l=0, r=10, t=45, b=0),
        xaxis=dict(tickprefix="₹", tickformat=",.0f"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("The estimated annual amount assumes the same monthly reduction for 12 months; it is not a forecast or a guaranteed saving.")


def _pace_chart(transactions: list[dict[str, Any]], today: date, result: dict[str, Any]) -> go.Figure:
    days_in_month = result["days_in_month"]
    daily_spend: dict[int, float] = {}
    for item in transactions:
        try:
            txn_date = date.fromisoformat(str(item.get("transaction_date", ""))[:10])
            amount = max(0.0, float(item.get("amount", 0.0)))
        except (TypeError, ValueError):
            continue
        if txn_date.year == today.year and txn_date.month == today.month and txn_date <= today:
            daily_spend[txn_date.day] = daily_spend.get(txn_date.day, 0.0) + amount

    actual_days = list(range(1, today.day + 1))
    actual_cumulative: list[float] = []
    running = 0.0
    for day in actual_days:
        running += daily_spend.get(day, 0.0)
        actual_cumulative.append(running)

    future_days = list(range(today.day, days_in_month + 1))
    forecast_values = [
        result["spent_to_date"] + result["current_daily_rate"] * (day - today.day)
        for day in future_days
    ]
    figure = go.Figure()
    figure.add_scatter(x=actual_days, y=actual_cumulative, mode="lines+markers", name="Actual to date", line=dict(color="#2f6bff", width=2.5))
    figure.add_scatter(x=future_days, y=forecast_values, mode="lines", name="Straight-line pace", line=dict(color="#1a7f5a", width=2.5, dash="dash"))
    if result["monthly_limit"] is not None:
        figure.add_hline(y=result["monthly_limit"], line_dash="dot", line_color="#e8590c", annotation_text="Monthly cap")
    figure.update_layout(
        title="Current-month spending pace",
        height=330,
        margin=dict(l=0, r=10, t=45, b=0),
        xaxis_title="Day of month",
        yaxis=dict(tickprefix="₹", tickformat=",.0f"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def _render_pace_monitor(token: str) -> None:
    st.title("Month-end Pace Monitor")
    st.caption("Estimate where this month may finish and get an early warning against a spending cap you choose.")
    today = date.today()
    monthly_rows = _cached_monthly(token)
    historical_average = average_monthly_total(monthly_rows, months=3, today=today)
    suggested_limit = round(historical_average / 500.0) * 500.0 if historical_average else 0.0
    if "insights_monthly_limit" not in st.session_state:
        st.session_state.insights_monthly_limit = float(suggested_limit)

    limit_value = st.number_input(
        "Your monthly spending cap (₹)",
        min_value=0.0,
        step=500.0,
        key="insights_monthly_limit",
        help="A positive value enables the warning. The value is kept only in this Streamlit session; it is not written to your ExpenseIQ account.",
    )
    month = today.strftime("%Y-%m")
    with st.spinner("Calculating this month's pace…"):
        transactions = _cached_transactions(token, month=month)
    result = project_month_end_spend(
        transactions,
        today=today,
        monthly_limit=float(limit_value) if limit_value > 0 else None,
    )

    if result["status"] == "at_risk":
        st.warning(f"At the current pace, spending may finish about {_inr(result['projected_overage'])} above your cap.")
    elif result["status"] == "on_track":
        st.success("Your current straight-line pace is within the monthly cap.")
    elif result["status"] == "no_spend_yet":
        st.info("No transactions have been recorded this month yet. The monitor will update when you add some.")
    else:
        st.info("Enter a positive monthly cap above to enable the early-warning comparison.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Spent so far", _inr(result["spent_to_date"]), f"{result['transactions_count']} transactions · day {result['elapsed_days']}")
    c2.metric("Straight-line month-end estimate", _inr(result["projected_month_end"]))
    allowance = result["remaining_daily_allowance"]
    c3.metric(
        "Daily room for rest of month",
        _inr(allowance) if allowance is not None else "Set a cap",
        f"{result['remaining_days']} days remaining",
    )
    st.plotly_chart(_pace_chart(transactions, today, result), width="stretch")
    st.caption("Estimate = month-to-date spend ÷ calendar days elapsed × days in month. It is a simple pace indicator, not the XGBoost forecast. Your cap is not saved to the database.")


def main() -> None:
    token = st.session_state.get("token")
    if not token:
        _login_page()
        return

    client = APIClient(token)
    try:
        user = client.me()
    except APIError as exc:
        if exc.status in (401, 403):
            st.session_state.pop("token", None)
            st.session_state.pop("user", None)
        st.error(f"Your ExpenseIQ session could not be validated: {exc.detail}")
        _login_page()
        return
    except Exception as exc:
        st.error(f"Could not reach the ExpenseIQ API. Start the backend and try again. ({exc})")
        return

    st.title("ExpenseIQ · Insights Studio")
    st.caption(f"Welcome, {user.get('name', 'there')} — three read-only tools built on your existing expense history.")
    with st.sidebar:
        st.markdown("## ✨ Insights Studio")
        feature = st.radio(
            "Choose an insight",
            ["Recurring Payment Radar", "Budget What-if Lab", "Month-end Pace Monitor"],
            label_visibility="collapsed",
        )
        st.caption("Add-on only · existing expenses and budgets are unchanged")
        if st.button("Sign out", width="stretch"):
            st.session_state.pop("token", None)
            st.session_state.pop("user", None)
            st.rerun()

    try:
        if feature == "Recurring Payment Radar":
            _render_recurring_radar(token)
        elif feature == "Budget What-if Lab":
            _render_what_if_lab(token)
        else:
            _render_pace_monitor(token)
    except APIError as exc:
        st.error(f"ExpenseIQ API error: {exc.detail}")
    except Exception as exc:
        st.error(f"The insight could not be loaded: {exc}")


main()
