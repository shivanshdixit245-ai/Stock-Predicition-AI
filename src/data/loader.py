"""
=== DATA SCIENTIST LEARNING NOTES ===

What this module does:
  Fetches and caches raw stock price (OHLCV) and financial news data.

Why we use Parquet instead of CSV:
  Parquet is a columnar storage format that is significantly faster for financial time-series data.
  It preserves data types (integers, floats, datetimes) and supports efficient compression (Snappy),
  reducing both disk I/O and storage footprint.

What a FAANG interviewer might ask:
  Q: "How do you handle API failures or rate limits in a production pipeline?"
  A: I implemented custom exceptions like `DataUnavailableError` and used a caching layer
     to minimize redundant API calls. In a real production system, I would also add 
     exponential backoff retries and circuit breakers.
  
  Q: "Why cache raw data separately from processed features?"
  A: It allows us to iterate on feature engineering logic without re-fetching data from 
     external APIs (which are often slow or rate-limited). Raw data is the "source of truth".

Data leakage risk in this module:
  Minimal, as this module only fetches raw historical data. However, news timestamps 
  must be carefully aligned to the next trading day's open in the preprocessor to avoid 
  lookahead bias (using news that broke after the close to predict the same day's close).
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import yfinance as yf
from loguru import logger
from newsapi import NewsApiClient

from config import settings


class DataUnavailableError(Exception):
    """Raised when data cannot be fetched from the API."""
    pass


def load_with_cache(
    fetch_fn: Callable[..., pd.DataFrame],
    cache_path: Path,
    allow_empty: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from cache if it exists, otherwise fetch and cache it.
    """
    if cache_path.exists():
        logger.info(f"Loading data from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    
    logger.info(f"Cache miss. Fetching data...")
    df = fetch_fn(**kwargs)
    
    if df.empty and not allow_empty:
        raise DataUnavailableError(f"No data returned for {kwargs.get('ticker', 'unknown')}")
    
    # Ensure cache directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving fetched data to cache: {cache_path}")
    df.to_parquet(cache_path, index=True, compression="snappy")
    
    return df


def load_stock_data(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance and cache it.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL).
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).
        use_cache: Whether to use the local cache.
        
    Returns:
        pd.DataFrame: Stock price data.
        
    Raises:
        DataUnavailableError: If data cannot be fetched.
    """
    cache_filename = f"{ticker}_{start}_{end}.parquet"
    cache_path = settings.data_prices_dir / cache_filename
    
    def fetch_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            logger.info(f"Fetching yfinance data for {ticker} from {start} to {end}")
            df = yf.download(ticker, start=start, end=end, progress=False)
            
            logger.info(f"YF downloaded data shape: {df.shape}")
            if not df.empty:
                logger.info(f"YF columns: {df.columns.tolist()}")
            
            # Handle MultiIndex for columns (yfinance 0.2.x default)
            if isinstance(df.columns, pd.MultiIndex):
                logger.info("MultiIndex columns detected. Flattening levels...")
                df.columns = df.columns.get_level_values(0)
                
            # Standardize column names to lowercase
            df.columns = [str(col).lower() for col in df.columns]
            return df
        except Exception as e:
            logger.error(f"Error fetching data from yfinance for {ticker}: {e}")
            raise DataUnavailableError(f"yfinance fetch failed: {e}")

    if use_cache:
        return load_with_cache(fetch_ticker_data, cache_path, ticker=ticker, start=start, end=end)
    else:
        return fetch_ticker_data(ticker, start, end)


def load_news(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch financial news headlines for a ticker and cache them.
    
    Args:
        ticker: Stock ticker symbol.
        start: Start date string.
        end: End date string.
        use_cache: Whether to use the local cache.
        
    Returns:
        pd.DataFrame: News headlines data.
        
    Note:
        Currently implements NewsAPI. RSS fallback can be added if needed.
    """
    cache_filename = f"{ticker}_{start}_{end}.parquet"
    cache_path = settings.data_news_dir / cache_filename
    
    def fetch_news_data(ticker: str, start: str, end: str) -> pd.DataFrame:
        api_key = settings.news_api_key
        if not api_key:
            logger.warning("NEWS_API_KEY not found in settings. Skipping news fetch.")
            return pd.DataFrame()
            
        try:
            newsapi = NewsApiClient(api_key=api_key)
            logger.info(f"Fetching NewsAPI headlines for {ticker}")
            
            # Simple query for ticker
            all_articles = newsapi.get_everything(
                q=ticker,
                from_param=start,
                to=end,
                language='en',
                sort_by='relevancy'
            )
            
            articles = all_articles.get('articles', [])
            if not articles:
                return pd.DataFrame()
                
            df = pd.DataFrame(articles)
            # Keep only relevant columns
            cols = ['publishedAt', 'title', 'description', 'source', 'url']
            df = df[cols]
            df['date'] = pd.to_datetime(df['publishedAt']).dt.date
            return df
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data for {ticker}: {e}")
            return pd.DataFrame() # Return empty instead of crashing for news

    if use_cache:
        return load_with_cache(fetch_news_data, cache_path, allow_empty=True, ticker=ticker, start=start, end=end)
    else:
        return fetch_news_data(ticker, start, end)


def load_cached_ticker(ticker: str) -> dict | None:
    """
    Find the most recent parquet file for this ticker in the prices dir.
    Returns the data + metadata if found.
    """
    prices_dir = settings.data_prices_dir
    files = list(prices_dir.glob(f"{ticker}_*.parquet"))
    if not files:
        return None
    
    # Get latest by mtime
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    df = pd.read_parquet(latest_file)
    
    # Calc age
    age_minutes = int((datetime.now().timestamp() - latest_file.stat().st_mtime) / 60)
    
    # Metadata fallback
    meta_path = settings.data_raw_dir / "metadata" / f"{ticker}_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {
            "company_name": ticker,
            "current_price": float(df["close"].iloc[-1]) if not df.empty else 0.0,
            "price_change_pct": 0.0,
            "fetched_at": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        }
    
    return {
        "df": df,
        "meta": meta,
        "age_minutes": age_minutes
    }


def verify_and_fetch_realtime(ticker: str) -> dict:
    """
    Fetch live data, perform quality checks, and save to cache.
    """
    if not ticker:
        raise ValueError("Ticker symbol cannot be empty")
        
    logger.info(f"Verifying and fetching real-time data for {ticker}...")
    
    # 1. Fetch (last 5 years)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    
    try:
        df = load_stock_data(ticker, start_date, end_date, use_cache=False)
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        # Re-raise with a cleaner message for the UI
        raise DataUnavailableError(f"yfinance fetch failed: {str(e)}")

    if df.empty:
        raise DataUnavailableError(f"No price data found for ticker {ticker}. Verify the symbol is correct.")

    # 2. Quality Checks
    confidence = 100.0
    warnings = []
    quality = "HIGH"
    
    if len(df) < 250:
        confidence -= 50
        warnings.append("Insufficient historical data (< 1 year)")
        quality = "LOW"
    
    # Check for gaps (rough approx of trading days)
    # yfinance handles most gaps but we check length
    if len(df) < 100:
        confidence -= 30
        warnings.append("Critically low historical data for stable modeling")
        quality = "LOW"

    # 3. Enhanced Metadata
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        meta = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "current_price": float(df["close"].iloc[-1]),
            "price_change_pct": float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100) if len(df) > 1 else 0.0,
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "rsi_14": 50.0, # Placeholder
            "fetched_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.warning(f"Could not fetch full metadata for {ticker}: {e}")
        meta = {
            "ticker": ticker,
            "company_name": ticker,
            "current_price": float(df["close"].iloc[-1]),
            "price_change_pct": 0.0,
            "sector": "Unknown",
            "market_cap": 0,
            "rsi_14": 50.0,
            "fetched_at": datetime.now().isoformat()
        }
    
    # Save meta
    meta_dir = settings.data_raw_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / f"{ticker}_meta.json", "w") as f:
        json.dump(meta, f)
        
    return {
        "df": df,
        "meta": meta,
        "confidence_score": confidence,
        "data_quality": quality,
        "warnings": warnings
    }


if __name__ == "__main__":
    # Demo block
    TICKER = "AAPL"
    START = "2023-01-01"
    END = "2023-12-31"
    
    logger.info(f"--- Running DataLoader Demo for {TICKER} ---")
    
    try:
        # Load stock data
        prices_df = load_stock_data(TICKER, START, END)
        logger.success(f"Loaded {len(prices_df)} price rows for {TICKER}")
        print(prices_df.head())
        
        # Load news data (will likely be empty without API key, but should handle gracefully)
        news_df = load_news(TICKER, START, END)
        if not news_df.empty:
            logger.success(f"Loaded {len(news_df)} news headlines")
            print(news_df.head())
        else:
            logger.info("No news headlines found (or API key missing).")
            
    except DataUnavailableError as e:
        logger.error(f"Demo failed: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in demo: {e}")
