"""
ExpenseIQ - Streamlit dashboard (report section 3.5.1 / sample output pages).

Run from the repository root:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.client import APIError, APIClient
from frontend.insight_engine import (
    analyze_spending_patterns,
    average_monthly_categories,
    average_monthly_total,
    build_category_spending_alerts,
    build_saving_recommendations,
    forecast_horizon_for_current_month,
)

GREEN = "#1a7f5a"
GREEN_DARK = "#14603f"
CATEGORY_COLORS = {
    "Rent": "#0e9fc0",
    "Utilities": "#2f6bff",
    "Food & Groceries": "#1a7f5a",
    "Health": "#f59f00",
    "Transport": "#e8590c",
    "Entertainment": "#9c36b5",
    "Other": "#868e96",
}

st.set_page_config(page_title="ExpenseIQ", page_icon="₹", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="st-"] { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: #f4f6f5; }
div[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1250px; }
div[data-testid="stTabs"] button { font-size: 0.86rem; padding: 0.35rem 0.95rem; border-radius: 8px; }
div[data-testid="stTabs"] button[aria-selected="true"] { background: #e3f2ea; color: #14603f; font-weight: 600; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #e6eae8; border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
div[data-testid="stMetric"] label { color: #667085 !important; font-weight: 500; font-size: .78rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700; }
.card { background:#fff; border:1px solid #e6eae8; border-radius:12px; padding:16px 20px;
        box-shadow:0 1px 2px rgba(16,24,40,.04); }
.card h4 { margin:0 0 10px 0; font-size:.72rem; letter-spacing:.08em; color:#98a2b3;
           text-transform:uppercase; font-weight:600; }
.brandbar { display:flex; align-items:center; gap:10px; padding:6px 4px 14px 4px; }
.brandlogo { width:30px; height:30px; border-radius:8px; background:#14603f; color:#fff;
             display:flex; align-items:center; justify-content:center; font-weight:700; }
.brandname { font-weight:700; font-size:1.05rem; color:#101828; }
.pill { margin-left:auto; font-size:.72rem; color:#1a7f5a; background:#e3f2ea;
        padding:4px 10px; border-radius:999px; font-weight:600; }
.bigpred { font-size:2.1rem; font-weight:800; color:#101828; }
.predcard { background:#eef7f2; border:1px solid #cfe8db; border-radius:12px; padding:16px 20px; }
.predlabel { font-size:.7rem; letter-spacing:.08em; color:#1a7f5a; font-weight:700; }
.featinfo td { padding:3px 0; font-size:.78rem; color:#475467; }
.featinfo td:last-child { text-align:right; font-weight:600; color:#101828; }
.footnote { font-size:.7rem; color:#98a2b3; margin-top:8px; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------- helpers
def inr(x: float | None, decimals: int = 2) -> str:
    if x is None:
        return "—"
    return "₹" + format(float(x), f",.{decimals}f")


def month_label(iso: str) -> str:
    s = str(iso)
    if len(s) == 7:  # "YYYY-MM" period string from /analytics/monthly
        s += "-01"
    return datetime.fromisoformat(s).strftime("%b %y")


def get_client() -> APIClient:
    return APIClient(st.session_state.get("token"))


def brandbar() -> None:
    st.markdown(
        f"""<div class="brandbar">
          <div class="brandlogo">₹</div><div class="brandname">ExpenseIQ</div>
          <div class="pill">● {st.session_state.get('model_status', 'model active')}</div>
        </div>""",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=20, show_spinner=False)
def _cached(fn_name: str, token: str, **params):
    client = APIClient(token)
    return getattr(client, fn_name)(**params)


# ----------------------------------------------------------------------- auth
def login_page() -> None:
    brandbar()
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Sign in to ExpenseIQ")
        st.caption("AI-based monthly expenses prediction & forecasting")
        mode = st.radio("Mode", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        email = st.text_input("Email", value="demo@expenseiq.app" if mode == "Login" else "")
        password = st.text_input("Password", type="password", value="demo1234" if mode == "Login" else "")
        if mode == "Register":
            name = st.text_input("Full name")
        if st.button("Sign in" if mode == "Login" else "Create account", type="primary", width='stretch'):
            client = APIClient()
            try:
                if mode == "Login":
                    res = client.login(email.strip(), password)
                else:
                    res = client.register(name.strip(), email.strip(), password)
                st.session_state.token = res["access_token"]
                st.session_state.user = res["user"]
                st.rerun()
            except APIError as exc:
                st.error(f"{exc.detail}")
        st.markdown(
            "<div class='footnote'>Demo account: demo@expenseiq.app / demo1234 · "
            "JWT bearer tokens · bcrypt password hashing</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ dashboard
def tab_dashboard(client: APIClient) -> None:
    s = client.summary()
    monthly = client.monthly()
    cats = client.category_summary()
    recent = client.list_expenses(limit=8)
    anomalies = client.anomalies()

    c1, c2, c3 = st.columns(3)
    delta = s.get("delta_pct")
    c1.metric("This Month", inr(s["this_month"]),
              None if delta is None else f"{delta:+.0f}% from last month")
    c2.metric("Transactions", f"{s['transaction_count']:,}", "all time")
    c3.metric("Daily Average", inr(s["daily_average"]), "current month rate")

    if anomalies:
        latest = anomalies[0]
        st.markdown(
            f"<div class='card' style='border-color:#f5c2c7;background:#fdf0f0'>"
            f"<h4 style='color:#b42318'>Anomaly alert</h4>"
            f"<span style='font-size:.85rem;color:#475467'>{latest['date']} · "
            f"{latest.get('category') or 'daily spend'} · {inr(latest['amount'])} "
            f"(threshold {inr(latest['threshold'])}) — {latest['description'] or ''}</span></div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([2, 1])
    with left:
        st.markdown("<div class='card'><h4>Monthly spending trend</h4>", unsafe_allow_html=True)
        df = pd.DataFrame(monthly)
        fig = go.Figure()
        if not df.empty:
            fig.add_scatter(
                x=[month_label(m) for m in df["month"]],
                y=df["total"],
                mode="lines",
                line=dict(color=GREEN, width=2.2, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(26,127,90,0.08)",
                name="spend",
            )
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=5, b=0), showlegend=False,
                          yaxis=dict(tickprefix="₹", tickformat=",." + "0f"),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><h4>This month by category</h4>", unsafe_allow_html=True)
        cur = [c for c in cats if c["this_month"] > 0] or cats[:5]
        total = sum(c["this_month"] or c["total"] for c in cur) or 1
        for c in sorted(cur, key=lambda x: -(x["this_month"] or x["total"])):
            val = c["this_month"] or c["total"]
            pct = val / total * 100
            color = CATEGORY_COLORS.get(c["category"], "#868e96")
            st.markdown(
                f"<div style='margin:6px 0'><div style='display:flex;justify-content:space-between;"
                f"font-size:.8rem;color:#475467'><span>{c['category']}</span><span>{pct:.0f}%</span></div>"
                f"<div style='height:5px;border-radius:3px;background:#eef1f0'>"
                f"<div style='height:5px;border-radius:3px;width:{max(pct, 1):.1f}%;background:{color}'></div></div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>Recent transactions</h4>", unsafe_allow_html=True)
    if recent:
        df = pd.DataFrame(
            [(t["transaction_date"], t["notes"] or t["category_name"], t["category_name"], inr(t["amount"]))
             for t in recent],
            columns=["Date", "Description", "Category", "Amount"],
        )
        st.dataframe(df, hide_index=True, width='stretch', height=260)
    else:
        st.info("No transactions yet - add one or upload a CSV.")
    st.markdown("</div>", unsafe_allow_html=True)
    render_dashboard_insights(client, s, monthly, cats, anomalies, st.session_state.user["user_id"])


def render_dashboard_insights(
    client: APIClient,
    summary: dict,
    monthly: list[dict],
    category_summary: list[dict],
    anomalies: list[dict],
    user_id: int,
) -> None:
    """Render data-driven patterns, saving guidance, and early-warning alerts."""
    today = date.today()
    token = client.token or ""
    transaction_error = None
    month_matrix_error = None
    try:
        transactions = _cached("list_expenses", token, limit=5000)
    except Exception as exc:
        transactions = []
        transaction_error = exc
    try:
        category_month_rows = _cached("category_month", token)
    except Exception as exc:
        category_month_rows = []
        month_matrix_error = exc

    category_types = {
        str(item.get("category")): str(item.get("type", "Variable"))
        for item in category_summary
    }
    patterns = analyze_spending_patterns(
        transactions,
        category_month_rows,
        today=today,
        category_types=category_types,
    )
    completed_category_baseline = average_monthly_categories(
        category_month_rows, months=3, today=today, completed_only=True
    )
    historical_monthly_average = average_monthly_total(
        monthly, months=3, today=today, completed_only=True
    )
    current_category_spend = {
        str(item.get("category")): item.get("this_month", 0.0)
        for item in category_summary
    }
    category_alerts = build_category_spending_alerts(
        current_category_spend,
        completed_category_baseline,
        category_types,
        today=today,
    )

    st.markdown("---")
    st.markdown("### 1. Spending Patterns")
    st.caption(
        "Insights use your transactions and existing monthly category history. "
        "Weekly/monthly increases need at least a 25% and ₹200 rise; weekend patterns "
        "compare average daily spend over up to 90 recorded days."
    )
    if transaction_error:
        st.warning(f"Transaction patterns could not be loaded right now: {transaction_error}")
    elif patterns:
        for insight in patterns:
            st.info(f"{insight['title']} — {insight['detail']}")
    else:
        st.caption("No strong repeated pattern was detected yet. More dated transactions make weekly and weekend comparisons more reliable.")
    if month_matrix_error:
        st.caption("Month-to-month category comparisons are temporarily unavailable; other pattern checks can still run.")

    st.markdown("### 2. Saving Recommendations")
    st.caption(
        "Enter your monthly income. It stays in this Streamlit session, not the database. "
        "Suggested savings = income − the larger of the expense estimate or current spend so far."
    )
    income_col, forecast_col = st.columns([1, 1])
    income_key = f"dashboard_monthly_income_{user_id}"
    with income_col:
        monthly_income = st.number_input(
            "Monthly income (₹)",
            min_value=0.0,
            step=500.0,
            format="%.2f",
            key=income_key,
            help="Used only to calculate this session's savings estimate.",
        )
    forecast_key = f"dashboard_saving_forecast_{user_id}"
    forecast_error_key = f"dashboard_saving_forecast_error_{user_id}"
    with forecast_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        refresh_forecast = st.button("Get fresh XGBoost forecast", key=f"dashboard_forecast_button_{user_id}")
    if refresh_forecast:
        horizon = forecast_horizon_for_current_month(summary.get("last_date"), today=today)
        if horizon is None:
            st.session_state.pop(forecast_key, None)
            st.session_state[forecast_error_key] = (
                "The latest transaction history is too old for a current-month forecast. "
                "The saving estimate will use completed-month spending history instead."
            )
        else:
            try:
                forecast_result = client.forecast(user_id, horizon=horizon, confidence=0.95)
                target_month = date.fromisoformat(str(forecast_result["target_month"])[:10])
                current_month = today.replace(day=1)
                if target_month.replace(day=1) < current_month:
                    st.session_state.pop(forecast_key, None)
                    st.session_state[forecast_error_key] = (
                        "The model returned a past target month, so it was not used as a future saving estimate."
                    )
                else:
                    st.session_state[forecast_key] = {
                        "target_month": target_month.isoformat(),
                        "predicted_amount": float(forecast_result["predicted_amount"]),
                        "model_version": forecast_result.get("model_version"),
                        "last_transaction_date": summary.get("last_date"),
                        "transaction_count": summary.get("transaction_count"),
                        "total_spend": summary.get("total_spend"),
                    }
                    st.session_state.pop(forecast_error_key, None)
            except APIError as exc:
                st.session_state.pop(forecast_key, None)
                st.session_state[forecast_error_key] = str(exc.detail)
            except Exception as exc:
                st.session_state.pop(forecast_key, None)
                st.session_state[forecast_error_key] = f"Forecast service unavailable: {exc}"
    if st.session_state.get(forecast_error_key):
        st.caption(f"XGBoost forecast not used: {st.session_state[forecast_error_key]}")

    forecast_result = st.session_state.get(forecast_key)
    if forecast_result and any(
        forecast_result.get(key) != summary.get(summary_key)
        for key, summary_key in (
            ("last_transaction_date", "last_date"),
            ("transaction_count", "transaction_count"),
            ("total_spend", "total_spend"),
        )
    ):
        forecast_result = None
        st.session_state.pop(forecast_key, None)
    if forecast_result:
        try:
            forecast_period = date.fromisoformat(forecast_result["target_month"])
            if forecast_period.replace(day=1) < today.replace(day=1):
                forecast_result = None
                st.session_state.pop(forecast_key, None)
        except (KeyError, TypeError, ValueError):
            forecast_result = None
            st.session_state.pop(forecast_key, None)
    predicted_expenses = forecast_result.get("predicted_amount") if forecast_result else None
    historical_expenses = historical_monthly_average or None
    saving_plan = build_saving_recommendations(
        monthly_income,
        summary.get("this_month", 0.0),
        predicted_expenses,
        completed_category_baseline,
        current_category_spend,
        category_types,
        patterns,
        historical_expenses=historical_expenses,
    )

    if saving_plan["status"] == "income_required":
        st.info("Enter monthly income above to calculate a personalized savings estimate.")
    elif saving_plan["status"] == "history_required":
        st.info("There is not enough completed-month history for a safe monthly estimate yet. Add more expenses or try the XGBoost forecast after at least 30 days of history.")
    else:
        plan_cols = st.columns(4)
        plan_cols[0].metric("Monthly income", inr(saving_plan["monthly_income"]))
        plan_cols[1].metric("Spent this month", inr(saving_plan["current_expenses"]))
        plan_cols[2].metric("Planning expense", inr(saving_plan["planning_expense"]))
        if saving_plan["shortfall"] > 0:
            plan_cols[3].metric("Income gap", inr(saving_plan["shortfall"]))
            st.warning("The current expense estimate is above income. Review the variable-category suggestion below.")
        else:
            plan_cols[3].metric("Suggested saving", inr(saving_plan["suggested_saving"]))
            st.success(f"A practical starting target is to save about {inr(saving_plan['suggested_saving'])} this month.")
        target_text = ""
        if forecast_result:
            target_text = f" for {month_label(forecast_result['target_month'])}"
        st.caption(
            f"Expense basis: {saving_plan['expense_source']}{target_text}. "
            "The planning expense uses the greater of that estimate and current month-to-date actual spending."
        )

    category_recommendation = saving_plan.get("category_recommendation")
    if category_recommendation:
        st.info(
            f"Try reducing {category_recommendation['category']} by about "
            f"{inr(category_recommendation['suggested_cut'])} per month. "
            f"This is based on recent spending of {inr(category_recommendation['reference_spend'])} "
            "(about 10% of that category, capped at 5% of income); fixed costs are excluded."
        )

    st.markdown("### 3. Expense Alerts")
    st.caption("Category pace alerts require spending to be at least 30% and ₹200 above the historical pace; fixed costs are excluded. The existing 3σ anomaly detector is also shown below.")
    alert_count = 0
    for alert in category_alerts:
        increase_text = (
            f" ({alert['increase_pct']:.0f}% above its normal pace)"
            if alert["increase_pct"] is not None
            else ""
        )
        st.warning(
            f"Warning: {alert['category']} spending is higher than its normal pattern. "
            f"Spent {inr(alert['current_spend'])} so far; the usual pace by day {today.day} "
            f"is about {inr(alert['expected_to_date'])} (recent monthly average "
            f"{inr(alert['normal_monthly_average'])}){increase_text}."
        )
        alert_count += 1
    if anomalies:
        st.markdown("**Existing 3σ anomaly detector**")
        for anomaly in anomalies[:5]:
            level = "category-month" if anomaly.get("level") == "category_month" else "transaction/day"
            st.warning(
                f"{anomaly.get('date', 'Date unavailable')} · {anomaly.get('category') or level}: "
                f"{inr(anomaly.get('amount'))} (threshold {inr(anomaly.get('threshold'))}). "
                f"{anomaly.get('description') or 'Unusually high spending detected.'}"
            )
            alert_count += 1
    if alert_count == 0:
        if not completed_category_baseline:
            st.info("No alert is active. Category comparisons will appear after you have at least one completed month of history.")
        else:
            st.success("No unusually high category spending or 3σ anomaly was detected by the current checks.")


# ------------------------------------------------------------------ add expense
def tab_add_expense(client: APIClient) -> None:
    st.markdown("### Add Transaction")
    cats = client.categories()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        with st.form("add_txn", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            amount = col_a.number_input("Amount (₹)", min_value=0.0, step=100.0, format="%.2f")
            txn_date = col_b.date_input("Date", value=date.today())
            options = {f"{c['name']} · {c['type']}": c["category_id"] for c in cats}
            cat_label = st.selectbox("Category", list(options))
            notes = st.text_input("Description (optional)", placeholder="e.g. Weekly grocery run")
            b1, b2 = st.columns([1, 1])
            submitted = b1.form_submit_button("Submit Transaction", type="primary")
            cleared = b2.form_submit_button("Clear")
            if submitted:
                if amount <= 0:
                    st.error("Amount must be greater than 0 (schema-validated).")
                else:
                    try:
                        client.create_expense({
                            "amount": round(amount, 2),
                            "transaction_date": txn_date.isoformat(),
                            "category_id": options[cat_label],
                            "notes": notes or None,
                        })
                        st.success("Transaction saved.")
                        st.cache_data.clear()
                    except APIError as exc:
                        st.error(f"Validation failed: {exc.detail}")
            if cleared:
                st.cache_data.clear()
        st.markdown(
            "<div class='footnote'>Schema-validated · MySQL/SQLite persisted · "
            "Anomaly detection active</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("<div class='card'><h4>Bulk CSV upload</h4>", unsafe_allow_html=True)
        up = st.file_uploader("Bank statement / expense export (.csv)", type=["csv"], label_visibility="collapsed")
        if up is not None:
            if st.button("Parse & import", type="primary"):
                try:
                    rep = client.upload_csv(up.getvalue(), up.name)
                    st.success(f"Inserted {rep['inserted']} rows, skipped {rep['skipped']}.")
                    st.caption("Date formats detected: " + ", ".join(rep["date_formats_seen"]))
                    if rep["category_mapping"]:
                        st.caption("Rule-based category mapping applied, e.g. " +
                                   "; ".join(f"{k} → {v}" for k, v in list(rep["category_mapping"].items())[:4]))
                    st.cache_data.clear()
                except APIError as exc:
                    st.error(str(exc.detail))
        st.markdown(
            "<div class='footnote'>Regex date normalisation (dd/mm/yyyy, dd-Mon-yyyy, …) "
            "+ keyword category mapping; single batch insert.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------- forecast
def tab_forecast(client: APIClient, user_id: int) -> None:
    st.markdown("### Expense Forecast")
    left, right = st.columns([1, 2.6])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        model_label = st.selectbox("Model", ["XGBoost"], disabled=False)
        confidence = st.selectbox("Confidence Level", ["90%", "95%", "99%"], index=1)
        horizon = st.selectbox("Horizon", ["Next Month", "Next 3 Months", "Next 6 Months"])
        run = st.button("Run Forecast", type="primary", width='stretch')
        metrics = st.session_state.get("model_metrics") or {}
        st.markdown(
            f"""<table class='featinfo' style='width:100%'>
            <tr><td colspan='2' style='padding-top:10px'><b style='font-size:.7rem;letter-spacing:.08em;color:#98a2b3'>FEATURE INFO</b></td></tr>
            <tr><td>Lag Features</td><td>t-1, t-7, t-30</td></tr>
            <tr><td>Rolling Windows</td><td>7d / 30d</td></tr>
            <tr><td>Cyclic Encoding</td><td>sin/cos(month)</td></tr>
            <tr><td>Anomaly Threshold</td><td>3σ</td></tr>
            <tr><td>R² Score</td><td>{metrics.get('r2', '—')}</td></tr>
            </table>""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        if run or st.session_state.get("forecast_result"):
            if run:
                h = {"Next Month": 1, "Next 3 Months": 3, "Next 6 Months": 6}[horizon]
                conf = float(confidence.strip("%")) / 100.0
                try:
                    res = client.forecast(user_id, horizon=h, confidence=conf)
                    st.session_state.forecast_result = res
                except APIError as exc:
                    st.session_state.forecast_result = None
                    st.error(str(exc.detail))
            res = st.session_state.get("forecast_result")
            if res:
                mlabel = datetime.fromisoformat(res["target_month"]).strftime("%B %Y").upper()
                st.markdown(
                    f"""<div class='predcard'>
                    <div class='predlabel'>PREDICTED — {mlabel}</div>
                    <div class='bigpred'>{inr(res['predicted_amount'])}</div>
                    <div style='font-size:.8rem;color:#475467'>{res['confidence_level']:.0%} confidence interval:
                    <b>{inr(res['lower_bound'])}</b> – <b>{inr(res['upper_bound'])}</b>
                    {' · ⚠ anomalies detected in input window' if res['anomaly_flag'] else ''}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                m = res["model_metrics"]
                c1, c2, c3 = st.columns(3)
                c1.metric("R² score", f"{m['r2']:.2f}", "variance explained")
                c2.metric("MAE", inr(m["mae"], 0), "average absolute error")
                c3.metric("RMSE", inr(m["rmse"], 0), "CI width basis")

                st.markdown("<div class='card'><h4>Historical + forecast trend</h4>", unsafe_allow_html=True)
                monthly = pd.DataFrame(client.monthly())
                fig = go.Figure()
                if not monthly.empty:
                    hist = monthly.tail(12)
                    xs = [month_label(x) for x in hist["month"]]
                    fig.add_scatter(x=xs, y=hist["total"], mode="lines+markers",
                                    line=dict(color="#2f6bff", width=2), name="actual")
                    # CI band for the forecast point(s)
                    fx = xs + [datetime.fromisoformat(res["target_month"]).strftime("%b %y")]
                    fig.add_scatter(x=fx, y=list(hist["total"]) + [res["upper_bound"]],
                                    mode="lines", line=dict(width=0), showlegend=False, name="upper")
                    fig.add_scatter(x=fx, y=list(hist["total"]) + [res["lower_bound"]],
                                    mode="lines", line=dict(width=0),
                                    fill="tonexty", fillcolor="rgba(26,127,90,0.18)",
                                    showlegend=False, name="CI")
                    fig.add_scatter(x=[datetime.fromisoformat(res["target_month"]).strftime("%b %y")],
                                    y=[res["predicted_amount"]], mode="markers",
                                    marker=dict(color=GREEN, size=10), name="forecast")
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=5, b=0),
                                  yaxis=dict(tickprefix="₹", tickformat=",." + "0f"),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

                hist_rows = client.forecast_history()
                if hist_rows:
                    with st.expander("Prediction audit log (drift monitoring)"):
                        df = pd.DataFrame(
                            [(h["forecast_month"], inr(h["predicted_amount"]),
                              inr(h["lower_bound"], 0) + " – " + inr(h["upper_bound"], 0),
                              inr(h["actual_val"]) if h["actual_val"] is not None else "pending",
                              inr(h["drift"]) if h["drift"] is not None else "—")
                             for h in hist_rows],
                            columns=["Target month", "Predicted", "CI", "Actual", "Drift"],
                        )
                        st.dataframe(df, hide_index=True, width='stretch')
        else:
            st.markdown(
                "<div class='card' style='padding:48px;text-align:center;color:#98a2b3'>"
                "📈<br/>Configure settings and run forecast</div>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------- transactions
def tab_transactions(client: APIClient) -> None:
    st.markdown("### Transactions")
    cats = client.categories()
    c1, c2, c3 = st.columns([3, 1.4, 1.4])
    search = c1.text_input("Search by description or category…", label_visibility="collapsed")
    cat_sel = c2.selectbox("Category filter", ["All Categories"] + [c["name"] for c in cats],
                           label_visibility="collapsed")
    monthly = client.monthly()
    month_sel = c3.selectbox("Month filter", ["All Months"] + [m["month"] for m in monthly][::-1],
                             label_visibility="collapsed")

    params: dict = {"limit": 1000}
    if search:
        params["search"] = search
    if cat_sel != "All Categories":
        params["category_id"] = next(c["category_id"] for c in cats if c["name"] == cat_sel)
    if month_sel != "All Months":
        params["month"] = month_sel
    txns = client.list_expenses(**params)

    total = sum(t["amount"] for t in txns)
    st.caption(f"{len(txns)} records · **{inr(total)}**")
    if txns:
        df = pd.DataFrame(
            [(t["transaction_date"], t["notes"] or t["category_name"], t["category_name"], t["amount"], t["id"])
             for t in txns],
            columns=["Date", "Description", "Category", "Amount", "id"],
        )
        def color_badge(val: str) -> str:
            color = CATEGORY_COLORS.get(val, "#868e96")
            return f"background:{color}1a;color:{color};padding:2px 10px;border-radius:999px;font-size:.75rem"
        df["Category"] = df["Category"].map(lambda v: f"<span style='{color_badge(v)}'>{v}</span>")
        df["Amount"] = df["Amount"].map(lambda v: inr(v))
        st.write(df.drop(columns=["id"]).to_html(index=False, escape=False), unsafe_allow_html=True)
        with st.expander("Delete a transaction"):
            pick = st.selectbox("Transaction", [f"{t['transaction_date']} · {t['notes'] or t['category_name']} · {inr(t['amount'])} (id {t['id']})" for t in txns], label_visibility="collapsed")
            tid = int(pick.split("(id ")[-1].strip(")"))
            if st.button("Delete selected", type="primary"):
                client.delete_expense(tid)
                st.cache_data.clear()
                st.success(f"Deleted transaction #{tid}")
                st.rerun()
    else:
        st.info("No transactions match the filters.")


# ------------------------------------------------------------------- analytics
def tab_analytics(client: APIClient) -> None:
    st.markdown("### Analytics")
    s = client.summary()
    cats = client.category_summary()
    monthly = client.monthly()
    category_fields = {"category", "type", "total", "share"}
    if not isinstance(cats, list) or any(
        not isinstance(category, dict) or not category_fields.issubset(category)
        for category in cats
    ):
        st.warning("Category analytics returned incomplete data. Refresh the page and try again.")
        cats = []

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spend", inr(s["total_spend"]), "all time")
    c2.metric("Avg Monthly", inr(s["avg_monthly"]), "based on history")
    top = s.get("top_category") or {"name": "—", "total": 0}
    c3.metric("Top Category", top["name"], inr(top["total"]))

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='card'><h4>Spend by category</h4>", unsafe_allow_html=True)
        if cats:
            df = pd.DataFrame(cats)
            fig = go.Figure(go.Bar(
                x=df["total"], y=df["category"], orientation="h",
                marker_color=[CATEGORY_COLORS.get(c, "#868e96") for c in df["category"]],
            ))
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=5, b=0), showlegend=False,
                              xaxis=dict(tickprefix="₹", tickformat=",." + "0f"),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Add expenses to see spending by category.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><h4>Monthly comparison</h4>", unsafe_allow_html=True)
        if monthly:
            df = pd.DataFrame(monthly).tail(6)
            colors = ["#a8cbb9"] * (len(df) - 1) + [GREEN_DARK]
            fig = go.Figure(go.Bar(x=[month_label(m) for m in df["month"]], y=df["total"], marker_color=colors))
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=5, b=0), showlegend=False,
                              yaxis=dict(tickprefix="₹", tickformat=",." + "0f"),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Add expenses to see monthly spending.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>Category summary</h4>", unsafe_allow_html=True)
    if cats:
        rows = []
        for c in cats:
            color = CATEGORY_COLORS.get(c["category"], "#868e96")
            rows.append(
                f"<tr><td style='padding:8px 4px'><span style='display:inline-block;width:8px;height:8px;"
                f"border-radius:50%;background:{color};margin-right:8px'></span>{c['category']}</td>"
                f"<td><span style='background:{color}1a;color:{color};padding:2px 10px;border-radius:999px;"
                f"font-size:.72rem'>{c['type']}</span></td>"
                f"<td style='font-weight:600'>{inr(c['total'])}</td>"
                f"<td style='width:35%'><div style='height:5px;border-radius:3px;background:#eef1f0'>"
                f"<div style='height:5px;border-radius:3px;width:{max(c['share'], 0.5):.1f}%;background:{color}'></div></div></td>"
                f"<td style='color:#667085'>{c['share']:.0f}%</td></tr>"
            )
        st.markdown(
            "<table style='width:100%;font-size:.85rem;color:#475467'>"
            "<tr style='color:#98a2b3;font-size:.7rem;letter-spacing:.06em'><th style='text-align:left'>CATEGORY</th>"
            "<th style='text-align:left'>TYPE</th><th style='text-align:left'>TOTAL SPENT</th>"
            "<th style='text-align:left'>SHARE</th><th></th></tr>" + "".join(rows) + "</table>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No category spending to summarize yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    anomalies = client.anomalies()
    if anomalies:
        st.markdown("<div class='card'><h4>Anomaly alerts (3σ rule)</h4>", unsafe_allow_html=True)
        df = pd.DataFrame(
            [(a["date"], a["level"], a.get("category") or "—", inr(a["amount"]), inr(a["threshold"]),
              a["description"] or "") for a in anomalies[:20]],
            columns=["Date", "Level", "Category", "Amount", "Threshold", "Rule"],
        )
        st.dataframe(df, hide_index=True, width='stretch', height=300)
        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------ main
def main() -> None:
    if "token" not in st.session_state:
        login_page()
        return

    client = get_client()
    try:
        user = client.me()
    except APIError:
        st.session_state.clear()
        login_page()
        return
    st.session_state.user = user

    try:
        import requests as _rq
        health = _rq.get(os.getenv("API_BASE_URL", "http://localhost:8000") + "/healthz", timeout=5).json()
        st.session_state.model_status = "model active" if health.get("model_loaded") else "model missing"
        st.session_state.model_metrics = health.get("model_metrics") or {}
    except Exception:
        st.session_state.model_status = "api offline"
        st.session_state.model_metrics = {}

    brandbar()
    tabs = st.tabs(["Dashboard", "Add Expense", "Forecast", "Transactions", "Analytics"])

    with tabs[0]:
        tab_dashboard(client)
    with tabs[1]:
        tab_add_expense(client)
    with tabs[2]:
        tab_forecast(client, user["user_id"])
    with tabs[3]:
        tab_transactions(client)
    with tabs[4]:
        tab_analytics(client)

    st.markdown(
        "<div class='footnote' style='text-align:center;padding:18px'>ExpenseIQ · FastAPI + XGBoost + "
        "SQLAlchemy · JWT auth · predictions audit log · logs/server.log</div>",
        unsafe_allow_html=True,
    )


main()
