import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import streamlit as st

CACHE_PATH = Path("data/raw/ticker_universe.parquet")
CACHE_TTL_DAYS = 7

@st.cache_data(ttl=604800)
def get_all_tickers() -> list[str]:
    """
    Fetch all US stock tickers from GitHub, cache locally for 7 days.
    Auto-refreshes weekly to handle new listings and delistings.
    
    DS Interview Note: Weekly TTL balances freshness vs API cost.
    """
    if CACHE_PATH.exists():
        age = datetime.now() - datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
        if age < timedelta(days=CACHE_TTL_DAYS):
            logger.info(f"Loading ticker universe from cache (age: {age.days} days)")
            return pd.read_parquet(CACHE_PATH)["ticker"].tolist()

    logger.info("Cache expired or missing — fetching fresh ticker list")
    sources = [
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt",
    ]

    tickers = set()
    for url in sources:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                batch = [t.strip() for t in response.text.splitlines() if t.strip()]
                tickers.update(batch)
                logger.success(f"Fetched {len(batch)} tickers from {url}")
        except Exception as e:
            logger.warning(f"Failed to fetch from {url}: {e}")

    if not tickers:
        logger.warning("All sources failed — using fallback popular tickers")
        tickers = {"AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA",
                   "JPM","BAC","V","MA","WMT","COST","JNJ","PFE",
                   "XOM","CVX","AMD","INTC","QCOM","SPY","QQQ"}

    df = pd.DataFrame(sorted(tickers), columns=["ticker"])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    logger.success(f"Saved {len(df)} tickers to cache")
    return df["ticker"].tolist()


def search_tickers(query: str, all_tickers: list[str]) -> list[str]:
    """Return top 20 tickers matching query prefix."""
    q = query.upper().strip()
    if len(q) < 1:
        return []
    return [t for t in all_tickers if t.startswith(q)][:20]
