# Security — Specification

## Threat Model

This is a portfolio project, not a financial product. Security focus is: protect API keys, prevent accidental data exposure, demonstrate professional practices.

---

## API Key Management

Never hardcode API keys. Use `.env` file:

```bash
# .env (never commit this to GitHub)
NEWS_API_KEY=your_key_here
```

```python
# config.py — loaded via pydantic-settings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    news_api_key: str = ""
    
    class Config:
        env_file = ".env"
```

`.gitignore` must include:
```
.env
data/raw/
data/models/
mlruns/
*.pkl
*.parquet
```

---

## Input Validation

All ticker inputs validated before hitting external APIs:

```python
import re

def validate_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if not re.match(r'^[A-Z]{1,5}$', ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")
    return ticker
```

FastAPI automatically validates request bodies via Pydantic schemas.

---

## Rate Limiting

NewsAPI free tier: 100 requests/day. Implement a request counter:

```python
from collections import defaultdict
from datetime import date

_request_counts = defaultdict(int)

def check_rate_limit(api_name: str, limit: int) -> bool:
    today = str(date.today())
    key = f"{api_name}_{today}"
    if _request_counts[key] >= limit:
        raise RateLimitError(f"{api_name} daily limit reached")
    _request_counts[key] += 1
```

---

## Error Handling

Never expose raw exceptions to the dashboard. All errors caught and logged:

```python
from loguru import logger

try:
    df = load_stock_data(ticker, start, end)
except Exception as e:
    logger.error(f"Failed to load data for {ticker}: {e}")
    st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    st.stop()
```

---

## No Financial Advice Disclaimer

Add to Streamlit sidebar:
```python
st.sidebar.warning(
    "This is a research tool for educational purposes only. "
    "Not financial advice. Past performance does not guarantee future results."
)
```
