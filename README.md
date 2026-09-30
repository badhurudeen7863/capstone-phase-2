# ExpenseIQ — AI-Based Monthly Expenses Prediction and Forecasting

A complete, runnable implementation of the BSc project report
*"AI-Based Monthly Expenses Prediction and Forecasting"* (Bharathiar University,
NIFT-TEA College of Knitwear Fashion, Tirupur — March 2026).

Full stack: **FastAPI + SQLAlchemy + JWT + XGBoost + Streamlit**, shipped with a
multi-year synthetic dataset, a trained model artefact, MySQL DDL, tests and
VS Code run configurations.

---

## 1. Features (all report requirements implemented)

| Report requirement | Where it lives |
|---|---|
| XGBoost monthly expense forecasting (vs LinearRegression / RandomForest) | `scripts/train_model.py`, `backend/services/ml_service.py` |
| Automated feature engineering: lags t-1/t-7/t-30, rolling mean 7/30, rolling std 30, expanding std, EMA | `backend/services/feature_engineering.py` |
| Sin/cos cyclic encoding of month (and day-of-month helper) | `feature_engineering.apply_sine_cosine_encoding` |
| Confidence intervals (pred ± 1.96·RMSE, generalised per confidence level, √h widening) | `ml_service.confidence_interval` |
| Anomaly detection (value > rolling-mean + 3σ, transaction level + category-month level) | `backend/services/anomaly.py` |
| FastAPI backend, async, model loaded in lifespan, Pydantic validation, 422 on bad input | `backend/main.py`, `backend/models/schemas.py` |
| Normalised MySQL 8 schema (3NF) via SQLAlchemy ORM (+ zero-setup SQLite fallback) | `backend/models/db_models.py`, `mysql/schema.sql`, `backend/database.py` |
| JWT auth (bcrypt hashing, time-limited tokens, row-level ownership) | `backend/services/security.py`, `backend/deps.py`, `backend/routers/auth.py` |
| Expense CRUD + filters | `backend/routers/expenses.py` |
| CSV bulk upload (pandas, regex date normalisation, rule-based category mapping, single batch insert) | `backend/services/csv_service.py` |
| Predictions audit log with drift monitoring (`actual_val` back-filled when the month ends) | `backend/models/db_models.py::PredictionLog`, `GET /forecasting/history` |
| Streamlit dashboard: KPI cards, trend chart with shaded CI, category bars, anomaly alerts, model metrics, audit log | `frontend/app.py` |
| Logging to `logs/server.log` (rotating) + request logging middleware | `backend/logging_setup.py`, `backend/main.py` |
| Tests (validation, unit, roundtrip, ML handoff, end-to-end, load) | `tests/` |
| Budgets table + endpoints | `backend/routers/budgets.py` |

### Modelling decisions (documented deviations / report-bug fixes)

1. **Target definition.** The report mixes daily features with a monthly target.
   Here: features are computed on the daily-resampled series (gaps = 0) at each
   month-end, and the target is the **total of the following month**.
2. **Stationary ratio target.** Tree ensembles cannot extrapolate the inflation
   trend in level space, so the model predicts
   `ratio = next_month_total / (rolling_mean_30 × days_in_month)` and the API
   rescales at inference time. This is applied identically in training and
   serving (no train/serve skew).
3. **Anomaly-safe features.** 3σ spike days are winsorised (spike mask computed on
   the *variable-spend* series, because rent days inflate σ) before features and
   the rescaling base are computed; raw anomalies are still reported in the UI.
4. **Report code bugs fixed:** metrics are computed from real training runs (not
   hard-coded 45.50 / 0.87), and `float(prediction_array)` uses `[0]`.
5. **Honest model selection.** The best hold-out model is deployed; with the
   bundled dataset that is XGBoost (see `ml_models/model_metrics.json`).

### Current bundled metrics (hold-out, currency units)

| Model | MAE | RMSE | R² | CV-RMSE |
|---|---|---|---|---|
| LinearRegression | 2071.70 | 2573.13 | 0.7949 | 2152.96 |
| RandomForest | 1912.71 | 2488.35 | 0.8082 | 1898.78 |
| **XGBoost (deployed)** | **1852.45** | **2479.47** | **0.8096** | **1816.48** |

*(re-run `python scripts/train_model.py` to reproduce; numbers vary slightly with data seed)*

---

## 2. Project layout

```
expense_forecaster/
├── backend/
│   ├── main.py                  # FastAPI app, lifespan (DB init + model load), CORS, logging middleware
│   ├── config.py                # env-driven settings (.env)
│   ├── database.py              # engine/session (SQLite default, MySQL 8 with pooling)
│   ├── logging_setup.py         # rotating logs/server.log
│   ├── deps.py                  # JWT bearer dependency
│   ├── models/
│   │   ├── db_models.py         # users, categories, transactions, predictions_log, budgets
│   │   └── schemas.py           # Pydantic v2 validation (amount>0, ISO dates, SQL-injection guard)
│   ├── routers/
│   │   ├── auth.py              # register / login / me
│   │   ├── expenses.py          # CRUD + CSV bulk upload
│   │   ├── categories.py        # category list
│   │   ├── analytics.py         # summary, monthly, categories, anomalies, spend-by-day
│   │   ├── forecasting.py       # GET /forecasting/predict/{user_id}, /forecasting/history
│   │   └── budgets.py           # budget upsert/list
│   └── services/
│       ├── feature_engineering.py  # lags/rolling/EMA/sin-cos, training-set builder, winsorisation
│       ├── anomaly.py              # 3σ rules (transaction + category-month)
│       ├── ml_service.py           # artefact loading, inference, CI
│       ├── csv_service.py          # regex date normalisation + keyword category mapping
│       └── security.py             # bcrypt + PyJWT
├── frontend/
│   ├── app.py                   # Streamlit dashboard (Dashboard / Add Expense / Forecast / Transactions / Analytics)
│   └── client.py                # typed HTTP client for the API
├── scripts/
│   ├── generate_synthetic_data.py  # multi-year, multi-user dataset + messy statement sample
│   ├── seed_db.py                  # categories + demo users + transactions
│   └── train_model.py              # LR vs RF vs XGBoost, CV, artefact + metrics export
├── data/                        # transactions.csv (≈11.5k rows, 2016-2026, 4 users), sample_upload.csv, categories.csv
├── ml_models/                   # xgboost_expense_forecaster_v2.pkl, model_metrics.json, feature_importance.png
├── mysql/schema.sql             # MySQL 8 DDL (3NF) + category seed
├── notebooks/eda_and_training.ipynb
├── tests/                       # 26 pytest tests
├── .vscode/                     # launch.json (5 configs + full-stack compound), tasks.json, settings, extensions
├── requirements.txt
├── .env.example
└── pytest.ini
```

---

## 3. Quick start (any machine, 5 commands)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_synthetic_data.py   # already generated in data/, re-run anytime
python scripts/seed_db.py --reset           # creates data/expenseiq.db + demo users
python scripts/train_model.py               # writes ml_models/*.pkl + metrics JSON

# terminal 1
uvicorn backend.main:app --reload --port 8000      # API + Swagger at /docs
# terminal 2
streamlit run frontend/app.py                      # dashboard at http://localhost:8501
```

**Demo logins** (password `demo1234`): `demo@expenseiq.app`, `aisha@example.com`,
`rahul@example.com`, `meera@example.com`. Register your own account from the UI.

> No MySQL installed? No problem — the default `DATABASE_URL` is a SQLite file, so
> the project runs out of the box. Switch to MySQL 8 (the report's target) any time:
> run `mysql -u root -p < mysql/schema.sql`, then set `DATABASE_URL` in `.env`.

---

## 4. Running in VS Code

1. **Open the folder** `expense_forecaster` in VS Code. Install the recommended
   extensions when prompted (Python, Jupyter).
2. Pick your interpreter (Ctrl/Cmd+Shift+P → *Python: Select Interpreter* → the venv
   you created above).
3. **One-click setup:** *Terminal → Run Task…* → **ExpenseIQ: One-click setup**
   (installs deps → generates data → seeds DB → trains the model).
   Individual tasks exist for each step, plus **ExpenseIQ: Run tests**.
4. **Run / debug:** press **F5** and choose
   - *ExpenseIQ: Backend (FastAPI)* — API on :8000 with debugger + `--reload`
   - *ExpenseIQ: Dashboard (Streamlit)* — UI on :8501
   - *ExpenseIQ: Full stack (API + Dashboard)* — both at once (compound)
   - *ExpenseIQ: Train model*, *ExpenseIQ: Seed database*, *ExpenseIQ: Tests (pytest)*
5. Breakpoints work everywhere (backend, services, scripts). The pytest explorer
   shows the 26 tests from `tests/`.
6. Logs stream to the terminals **and** to `logs/server.log`.

---

## 5. API reference (prefix `/api/v1`, interactive docs at `/docs`)

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /auth/register` | – | create account → JWT |
| `POST /auth/login` | – | login → JWT |
| `GET /auth/me` | ✔ | current user |
| `GET /categories` | ✔ | seeded categories (Fixed/Variable) |
| `POST /expenses` | ✔ | add expense (amount > 0, ISO date, injection-guarded notes) |
| `GET /expenses` | ✔ | list; filters `category_id`, `month=YYYY-MM`, `search`, paging |
| `PUT /expenses/{id}` · `DELETE /expenses/{id}` | ✔ | update / delete (owner only) |
| `POST /expenses/upload-csv` | ✔ | bulk import; returns insert/skip report + mapping |
| `GET /analytics/summary` | ✔ | KPIs: total, this/last month, Δ%, daily avg, top category |
| `GET /analytics/monthly` · `/categories` · `/category-month` · `/spend-by-day` | ✔ | chart series |
| `GET /analytics/anomalies` | ✔ | 3σ alerts (transaction + category-month) |
| `GET /forecasting/predict/{user_id}?horizon=1..6&confidence=0.90/0.95/0.99` | ✔ (self only) | forecast + CI + metrics + anomaly flag; needs ≥ 30 days history (else 400) |
| `GET /forecasting/history` | ✔ | audit log with actuals & drift once months close |
| `GET/PUT /budgets` | ✔ | monthly category budgets |
| `GET /healthz` | – | liveness + model status/metrics |

Validation behaviour: negative/zero amount → **422**, `2026-15-45` → **422**,
SQL-injection strings in free text → **422**, missing/invalid JWT → **401**,
forecasting another user → **403**, < 30 days history → **400**.

---

## 6. Dataset & DGP

`scripts/generate_synthetic_data.py` produces 2016-01 → 2026-08 for 4 users
(≈11.5k transactions): smooth seasonal curve, sharp Diwali (Oct/Nov) block,
~5.4 %/yr inflation, non-linear electricity response `900 + 3200·heat^1.7`,
a 2023 transport regime switch, controlled noise and four rare injected
anomaly events (two inside 2026 so the dashboard alert demo always has data).
`data/sample_upload.csv` is a deliberately messy bank statement (five date
formats, raw merchant names) for the CSV-upload feature.

## 7. Tests

```bash
pytest            # or: python -m pytest -v
```
26 tests: payload validation & injection guards, rolling-mean unit test
(`[10,20,30,40] → [NaN,15,25,35]`), sin/cos cycle wrap, daily-series gap filling,
DB→API CRUD roundtrip, CSV upload mapping, ownership enforcement, API→ML
forecast contract + CI ordering + audit log, register→bulk-load→forecast
end-to-end, and a threaded load test (p95 latency assertion).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Model not trained yet` (503) | run `python scripts/train_model.py` |
| Port busy | change `--port` / `--server.port`; UI reads `API_BASE_URL` env |
| MySQL connection error | check `DATABASE_URL` in `.env`, or delete it to fall back to SQLite |
| Empty dashboard | seed data: `python scripts/seed_db.py --reset`, then log in as demo user |
| Re-generate everything | task **ExpenseIQ: One-click setup**, or the four commands in §3 |
