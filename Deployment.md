# Deployment — Specification

## Local Development (Primary)

```bash
# 1. Clone repo
git clone https://github.com/yourusername/stock-ai.git
cd stock-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env: add NEWS_API_KEY

# 5. Run Streamlit dashboard
python -m streamlit run app.py

# 6. (Optional) Run MLflow UI in separate terminal
mlflow ui --port 5000

# 7. (Optional) Run FastAPI in separate terminal
uvicorn api.main:app --reload --port 8000
```

---

## Streamlit Cloud (Free Public Demo)

Deploy the dashboard publicly for portfolio purposes:

1. Push code to GitHub (do NOT push `.env` or data files)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Add `NEWS_API_KEY` as a Secret in Streamlit Cloud settings
5. Set main file: `app.py`

**Important:** Pre-compute and commit model artefacts to GitHub (models are small .pkl files). Otherwise the app will need to retrain on every cold start.

---

## GitHub Repository Best Practices

### Files to commit
```
✅ All .py source files
✅ requirements.txt
✅ README.md
✅ docs/*.md
✅ .env.example (with placeholder values)
✅ data/models/{TICKER}/*.pkl  (pre-trained models)
✅ data/models/{TICKER}/feature_list.json
✅ data/models/{TICKER}/training_metadata.json
```

### Files to NOT commit
```
❌ .env (API keys)
❌ data/raw/ (cached API responses — too large)
❌ data/processed/ (regenerated on run)
❌ mlruns/ (MLflow logs — optional, can commit if small)
❌ __pycache__/
❌ *.pyc
```

---

## Docker (Optional, shows DevOps awareness)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t stock-ai .
docker run -p 8501:8501 --env-file .env stock-ai
```

---

## README for GitHub

The GitHub README is extremely important for your CV — hiring managers will read it. It should include:

1. Project title + one-line description
2. Architecture diagram (screenshot or ASCII)
3. Key technical differentiators (walk-forward CV, conformal prediction, etc.)
4. GIF or screenshot of the dashboard
5. Quickstart commands
6. Results table (walk-forward F1, Sharpe, bootstrap p-value)
7. The resume bullet
8. Experiment table from MLflow (screenshot)
