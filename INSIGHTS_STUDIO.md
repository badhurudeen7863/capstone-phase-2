# ExpenseIQ Insights (additive features)

The existing **Dashboard** now includes three data-driven sections. They reuse the
current authenticated API, analytics, anomaly detector, and XGBoost forecast; no
backend endpoints, database tables, or duplicate ML models were added.

## Dashboard additions

1. **Spending Patterns** — compares the latest tracked 7 days with the prior 7,
   checks average weekend versus weekday spending over recent history, and compares
   the latest two completed category-months. It shows only patterns supported by
   the user's transactions; a partial current month is not used for month-to-month
   comparisons.
2. **Saving Recommendations** — enter monthly income in the Dashboard. The plan
   compares it with an XGBoost expense forecast (when requested and available),
   current month-to-date expenses, and recent completed-month averages. It estimates
   savings headroom and suggests a possible variable-category reduction. Income is
   kept only in the current Streamlit session, not in the database.
3. **Expense Alerts** — compares variable-category spending so far this month with
   its historical monthly average scaled to the elapsed days, and also displays
   alerts from the existing 3σ anomaly endpoint. Fixed costs such as rent are not
   treated as pace anomalies.

Empty/new accounts are handled with explanatory messages instead of fabricated
values. If the ML forecast is unavailable or history is too old, the saving plan
uses completed-month averages when available. Clicking **Get fresh XGBoost
forecast** calls the existing forecast endpoint and records its normal audit-log
entry.

## Run the project

From the repository root, install and initialize the project once:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/seed_db.py --reset
python scripts/train_model.py
```

Start the API and dashboard in separate terminals:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
streamlit run frontend/app.py
```

Sign in, open **Dashboard**, enter monthly income under **Saving Recommendations**,
and optionally click **Get fresh XGBoost forecast**. For this particular dashboard
feature, you can also keep using the usual forecast tab, transaction list, and
analytics; none were removed.

## Earlier optional Insights Studio page

The separate `Insights Studio` page is also retained. It includes the Recurring
Payment Radar, Budget What-if Lab, and Month-end Pace Monitor. It can be selected
from Streamlit's page navigation or launched directly:

```bash
streamlit run frontend/pages/2_Insights_Studio.py
```

## Checks

Pure calculations and edge cases are covered in `tests/test_insight_engine.py`:

```bash
python -m pytest tests/test_insight_engine.py -q
```
