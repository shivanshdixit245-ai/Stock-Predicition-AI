# Adaptive Market Intelligence & Risk Signal Platform

## Project Overview

A production-grade, end-to-end data science system that predicts stock price movements using ensemble machine learning, technical indicators, market regime detection, sentiment analysis, and rigorous statistical validation. Built for a data scientist CV targeting FAANG/MAANG-level roles.

---

## Instructions for AI Agent (Antigravity)

### Step 1 — Read ALL documents before writing any code
Read every `.md` file in this folder in this order:
1. `README.md` (this file)
2. `PRD.md`
3. `Architecture.md`
4. `Features.md`
5. `ML_Pipeline.md`
6. `Backtesting.md`
7. `WalkForwardValidation.md`
8. `ModelDrift.md`
9. `ExperimentTracking.md`
10. `Database.md`
11. `API.md`
12. `TechStack.md`
13. `UIUX.md`
14. `Scaling.md`
15. `Security.md`
16. `Deployment.md`
17. `AI_Instructions.md`

### Step 2 — Build one module at a time
Do NOT generate the entire codebase at once. Build module by module, confirm each works, then move to the next.

### Step 3 — Module build order
```
1. src/data/loader.py           — data ingestion
2. src/data/preprocessor.py     — cleaning, normalisation
3. src/features/technical.py    — indicator engineering
4. src/features/sentiment.py    — NLP pipeline
5. src/features/selector.py     — SHAP-based feature selection
6. src/models/regime.py         — HMM regime detection
7. src/models/trainer.py        — walk-forward training loop
8. src/models/ensemble.py       — model stacking + calibration
9. src/models/uncertainty.py    — conformal prediction intervals
10. src/backtest/engine.py      — vectorised backtest
11. src/backtest/stats.py       — metrics + bootstrap significance
12. src/drift/monitor.py        — drift detection + retrain trigger
13. src/api/routes.py           — FastAPI layer
14. app.py                      — Streamlit dashboard
```

### Step 4 — Code standards
- Python 3.11+
- Type hints on every function
- Docstrings on every class and function
- Logging via `loguru`, not `print`
- Config via `pydantic-settings` / `.env` file
- No hardcoded values — all constants in `config.py`
- Unit tests in `tests/` using `pytest`

### Step 5 — Always explain code for learning
After writing each module, add a comment block at the top explaining:
- What the module does
- Why this approach was chosen
- What a data science interviewer would ask about it

---

## Project Output Structure

```
stock-ai/
├── README.md
├── config.py
├── requirements.txt
├── .env.example
├── app.py                     ← Streamlit dashboard entry point
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/
│   │   ├── technical.py
│   │   ├── sentiment.py
│   │   └── selector.py
│   ├── models/
│   │   ├── regime.py
│   │   ├── trainer.py
│   │   ├── ensemble.py
│   │   └── uncertainty.py
│   ├── backtest/
│   │   ├── engine.py
│   │   └── stats.py
│   └── drift/
│       └── monitor.py
│
├── api/
│   ├── main.py
│   └── routes.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_backtest.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── mlruns/                    ← MLflow experiment store
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
└── docs/
    └── *.md
```

---

## Resume Bullet (final version)
> "Built an end-to-end adaptive trading signal platform using ensemble ML (XGBoost + LightGBM) with walk-forward cross-validation, conformal prediction intervals, and Hidden Markov Model regime detection; validated statistical significance of alpha via bootstrap permutation testing (p<0.05); built model drift monitor with automated retraining; deployed as Streamlit dashboard with MLflow experiment tracking across 60+ experiments."
