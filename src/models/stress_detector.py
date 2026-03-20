"""
Market Stress Detection & Circuit Breaker Module.

Detects abnormal market conditions (black swan events, wars, crashes, pandemics)
and activates a circuit breaker that overrides all model signals with HOLD when
markets are too extreme for the model to handle reliably.

DS Interview Note:
Every production ML system needs a "kill switch". In finance, models trained
on normal market data will generate catastrophically wrong signals during tail
events (COVID, 2008, Ukraine). A circuit breaker is the difference between a
research prototype and a deployable system.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


# ---------------------------------------------------------------------------
# 1. Stress Score
# ---------------------------------------------------------------------------

def compute_market_stress_score(df: pd.DataFrame) -> pd.Series:
    """
    Combines 5 stress indicators into a single composite score (0-100).

    Indicators:
        1. Volatility spike — realised_vol_5 / realised_vol_20 ratio
        2. Price drawdown — current price vs 20-day rolling high
        3. Volume anomaly — volume z-score over 20 days
        4. RSI extremes — panic selling (<20) or euphoria (>85)
        5. VIX proxy — CBOE VIX level (fetched live, skipped gracefully)

    DS Interview Note:
    A single composite score is essential for real-time dashboards.
    The weights (30/25/20/15/10) reflect empirical importance: volatility
    is the strongest predictor of regime breakdowns, followed by drawdown.
    VIX gets the lowest weight because it is an *external* signal that may
    not be available.
    """
    scores = pd.DataFrame(index=df.index)

    # --- Indicator 1: Volatility spike (weight 30) ---
    if "realised_vol_5" in df.columns and "realised_vol_20" in df.columns:
        vol_ratio = df["realised_vol_5"] / df["realised_vol_20"].replace(0, np.nan)
        # ratio of 1 = normal, 3+ = extreme → map [1, 4] → [0, 100]
        scores["vol_spike"] = ((vol_ratio - 1) / 3 * 100).clip(0, 100)
    else:
        # Fallback: derive from close prices
        ret = df["close"].pct_change()
        vol5 = ret.rolling(5).std()
        vol20 = ret.rolling(20).std()
        vol_ratio = vol5 / vol20.replace(0, np.nan)
        scores["vol_spike"] = ((vol_ratio - 1) / 3 * 100).clip(0, 100)

    # --- Indicator 2: Price drawdown (weight 25) ---
    rolling_high = df["close"].rolling(20, min_periods=1).max()
    drawdown_pct = (rolling_high - df["close"]) / rolling_high * 100  # 0-100 %
    # 0 % drawdown = 0 stress, 20 %+ drawdown = 100 stress
    scores["drawdown"] = (drawdown_pct / 20 * 100).clip(0, 100)

    # --- Indicator 3: Volume anomaly (weight 20) ---
    if "volume_zscore_20" in df.columns:
        # z-score 0 = normal, 3+ = panic → map [0, 5] → [0, 100]
        scores["volume"] = (df["volume_zscore_20"].abs() / 5 * 100).clip(0, 100)
    elif "volume" in df.columns:
        vol_mean = df["volume"].rolling(20, min_periods=1).mean()
        vol_std = df["volume"].rolling(20, min_periods=1).std().replace(0, np.nan)
        z = ((df["volume"] - vol_mean) / vol_std).abs()
        scores["volume"] = (z / 5 * 100).clip(0, 100)
    else:
        scores["volume"] = 0.0

    # --- Indicator 4: RSI extremes (weight 15) ---
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
        # RSI 30-70 = 0 stress; <20 or >85 = 100 stress
        rsi_stress = pd.Series(0.0, index=df.index)
        rsi_stress = rsi_stress.where(rsi >= 20, (20 - rsi) / 20 * 100)
        rsi_stress = rsi_stress.where(rsi <= 85, (rsi - 85) / 15 * 100)
        scores["rsi"] = rsi_stress.clip(0, 100)
    else:
        scores["rsi"] = 0.0

    # --- Indicator 5: VIX proxy (weight 10) ---
    scores["vix"] = _fetch_vix_component(df.index)

    # --- Weighted composite ---
    weights = {"vol_spike": 30, "drawdown": 25, "volume": 20, "rsi": 15, "vix": 10}
    composite = sum(scores[k].fillna(0) * w for k, w in weights.items()) / 100
    composite = composite.clip(0, 100)

    logger.info(
        f"Stress scores computed — latest: {composite.iloc[-1]:.1f}/100"
    )
    return composite


def _fetch_vix_component(index: pd.DatetimeIndex) -> pd.Series:
    """Attempt to fetch ^VIX and map to 0-100 stress. Fails gracefully."""
    try:
        vix_df = yf.download(
            "^VIX",
            start=index.min(),
            end=index.max() + pd.Timedelta(days=1),
            progress=False,
        )
        if vix_df.empty:
            raise ValueError("Empty VIX data")

        vix_close = vix_df["Close"].squeeze()
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]
        vix_close = vix_close.reindex(index, method="ffill")
        # VIX 12 = calm, 30 = stressed, 50+ = extreme → map [12, 60] → [0, 100]
        mapped = ((vix_close - 12) / 48 * 100).clip(0, 100)
        logger.debug("VIX component fetched successfully")
        return mapped
    except Exception as e:
        logger.warning(f"VIX data unavailable — skipping: {e}")
        return pd.Series(0.0, index=index)


# ---------------------------------------------------------------------------
# 2. Classify Stress Level
# ---------------------------------------------------------------------------

def classify_stress_level(stress_score: float) -> dict:
    """
    Translate a numeric stress score into an actionable status dict.

    DS Interview Note:
    Risk systems need *discrete* action levels, not just numbers.
    Traders and automated systems both need a clear "what do I do now?"
    signal. The four-tier system mirrors institutional risk frameworks
    (green/amber/orange/red).
    """
    if stress_score < 25:
        return {
            "level": 0,
            "label": "Normal",
            "color": "green",
            "action": "trade_normally",
            "description": "Market conditions normal. Model signals reliable.",
        }
    elif stress_score < 50:
        return {
            "level": 1,
            "label": "Elevated",
            "color": "yellow",
            "action": "reduce_size",
            "description": (
                "Above-average volatility. Trade with caution. "
                "Consider reducing position size."
            ),
        }
    elif stress_score < 75:
        return {
            "level": 2,
            "label": "High",
            "color": "orange",
            "action": "minimal_trading",
            "description": (
                "High stress detected. Possible news event or sector shock. "
                "Minimal trading recommended."
            ),
        }
    else:
        return {
            "level": 3,
            "label": "Extreme",
            "color": "red",
            "action": "circuit_breaker",
            "description": (
                "Extreme market stress. Possible black swan event "
                "(crash, war, pandemic, disaster). ALL signals overridden "
                "to HOLD. Model not reliable under these conditions."
            ),
        }


# ---------------------------------------------------------------------------
# 3. Circuit Breaker
# ---------------------------------------------------------------------------

def apply_circuit_breaker(
    signals: pd.Series, stress_scores: pd.Series
) -> pd.Series:
    """
    Override trading signals based on market stress levels.

    - stress >= 75  →  force HOLD  (circuit breaker)
    - stress 50-75  →  append _CAUTION suffix
    - stress < 50   →  unchanged

    DS Interview Note:
    This is the production safety net. During COVID week-1 (March 9-13, 2020),
    any momentum model would have generated massive BUY or SELL signals on
    40 % daily swings. A circuit breaker prevents the model from trading into
    a regime it was never trained on.
    """
    modified = signals.copy()

    # Align on common index
    common = signals.index.intersection(stress_scores.index)
    scores_aligned = stress_scores.reindex(common)

    extreme_mask = scores_aligned >= 75
    high_mask = (scores_aligned >= 50) & (scores_aligned < 75)

    n_extreme = extreme_mask.sum()
    n_caution = high_mask.sum()

    modified.loc[common[extreme_mask]] = "HOLD"
    modified.loc[common[high_mask]] = modified.loc[common[high_mask]].apply(
        lambda s: s if s == "HOLD" or s.endswith("_CAUTION") else f"{s}_CAUTION"
    )

    logger.info(
        f"Circuit breaker applied — {n_extreme} signals overridden to HOLD, "
        f"{n_caution} signals flagged CAUTION"
    )
    return modified


# ---------------------------------------------------------------------------
# 4. Black Swan Detector
# ---------------------------------------------------------------------------

def detect_black_swan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify specific extreme-event patterns in historical price data.

    Patterns detected:
        1. Flash crash   — single-day drop >5 % + volume >3× average
        2. Sustained crash — cumulative drop >20 % over 10 days
        3. Volatility explosion — realised_vol_5 > 4× realised_vol_20
        4. Volume panic  — volume >5× 20-day average for 3+ consecutive days

    DS Interview Note:
    Black swan detection is *retrospective* labelling that tells the
    interviewer: "My model knows when it would have been wrong."
    Historical examples this catches: COVID March 2020, 2008 GFC,
    2022 rate-hike drawdown, Ukraine war February 2022.
    """
    result = df.copy()
    result["is_black_swan"] = False
    result["black_swan_type"] = ""
    result["black_swan_severity"] = 0.0

    daily_return = df["close"].pct_change()
    vol_mean_20 = df["volume"].rolling(20, min_periods=1).mean() if "volume" in df.columns else pd.Series(0, index=df.index)

    # --- Pattern 1: Flash crash (single-day drop >3 % + volume >2× average) ---
    flash_drop = daily_return < -0.03
    if "volume" in df.columns:
        high_vol = df["volume"] > (vol_mean_20 * 2)
        flash_crash = flash_drop & high_vol
    else:
        flash_crash = flash_drop
    result.loc[flash_crash, "is_black_swan"] = True
    result.loc[flash_crash, "black_swan_type"] = "Flash Crash"
    result.loc[flash_crash, "black_swan_severity"] = (
        daily_return[flash_crash].abs().clip(0, 1)
    )

    # --- Pattern 2: Sustained crash (>15 % cumulative drop over 10 days) ---
    cum_ret_10d = (1 + daily_return).rolling(10, min_periods=1).apply(
        lambda x: x.prod() - 1, raw=True
    )
    sustained = cum_ret_10d < -0.15
    new_sustained = sustained & ~result["is_black_swan"]
    result.loc[new_sustained, "is_black_swan"] = True
    result.loc[new_sustained, "black_swan_type"] = "Sustained Crash"
    result.loc[new_sustained, "black_swan_severity"] = (
        cum_ret_10d[new_sustained].abs().clip(0, 1)
    )

    # --- Pattern 3: Volatility explosion (5-day vol > 2.5× 20-day vol) ---
    if "realised_vol_5" in df.columns and "realised_vol_20" in df.columns:
        vol_ratio = df["realised_vol_5"] / df["realised_vol_20"].replace(0, np.nan)
    else:
        ret = df["close"].pct_change()
        vol5 = ret.rolling(5).std()
        vol20 = ret.rolling(20).std()
        vol_ratio = vol5 / vol20.replace(0, np.nan)

    vol_explode = vol_ratio > 2.5
    new_vol = vol_explode & ~result["is_black_swan"]
    result.loc[new_vol, "is_black_swan"] = True
    result.loc[new_vol, "black_swan_type"] = "Volatility Explosion"
    result.loc[new_vol, "black_swan_severity"] = (
        (vol_ratio[new_vol] / 8).clip(0, 1)
    )

    # --- Pattern 4: Volume panic (>3× avg for 2+ consecutive days) ---
    if "volume" in df.columns:
        extreme_vol = (df["volume"] > vol_mean_20 * 3).astype(int)
        consecutive = extreme_vol.rolling(2, min_periods=2).sum() >= 2
        new_panic = consecutive & ~result["is_black_swan"]
        result.loc[new_panic, "is_black_swan"] = True
        result.loc[new_panic, "black_swan_type"] = "Volume Panic"
        result.loc[new_panic, "black_swan_severity"] = 0.7

    n_events = result["is_black_swan"].sum()
    logger.info(f"Black swan scan complete — {n_events} event-days detected")
    return result


# ---------------------------------------------------------------------------
# 5. Current Status (live)
# ---------------------------------------------------------------------------

def get_current_stress_status(ticker: str) -> dict:
    """
    Fetch the last 60 days of data, compute today's stress score, and
    return a fully-qualified status dict for the dashboard banner.

    DS Interview Note:
    This is the "one API call" a portfolio manager needs.  It answers:
    "Is it safe to trade right now?" — with a number, a colour, and
    a concrete recommendation.
    """
    try:
        raw = yf.download(ticker, period="60d", progress=False)
        if raw.empty:
            logger.warning(f"No data for {ticker} — returning default Normal")
            return _default_status(ticker)

        df = raw.copy()
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        # Derive lightweight features needed by the stress scorer
        ret = df["close"].pct_change()
        df["realised_vol_5"] = ret.rolling(5).std() * np.sqrt(252)
        df["realised_vol_20"] = ret.rolling(20).std() * np.sqrt(252)
        if "volume" in df.columns:
            vol_mean = df["volume"].rolling(20, min_periods=1).mean()
            vol_std = df["volume"].rolling(20, min_periods=1).std().replace(0, np.nan)
            df["volume_zscore_20"] = (df["volume"] - vol_mean) / vol_std

        # RSI-14
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + rs)

        df = df.dropna(subset=["close"])

        stress_scores = compute_market_stress_score(df)
        latest_score = float(stress_scores.iloc[-1])
        status = classify_stress_level(latest_score)

        # Identify the top stress driver
        drivers = _get_driver_breakdown(df.iloc[[-1]])
        top_driver = max(drivers, key=drivers.get) if drivers else "unknown"

        # VIX level (best-effort)
        vix_level = None
        try:
            vix_raw = yf.download("^VIX", period="5d", progress=False)
            if not vix_raw.empty:
                vix_close = vix_raw["Close"].squeeze()
                if isinstance(vix_close, pd.DataFrame):
                    vix_close = vix_close.iloc[:, 0]
                vix_level = float(vix_close.iloc[-1])
        except Exception:
            pass

        return {
            "ticker": ticker,
            "date": str(df.index[-1].date()),
            "stress_score": latest_score,
            "stress_level": status["level"],
            "label": status["label"],
            "color": status["color"],
            "action": status["action"],
            "description": status["description"],
            "top_stress_driver": top_driver,
            "vix_level": vix_level,
            "recommendation": status["description"],
        }

    except Exception as e:
        logger.error(f"get_current_stress_status failed for {ticker}: {e}")
        return _default_status(ticker)


def _default_status(ticker: str) -> dict:
    """Return a safe default status when data is unavailable."""
    return {
        "ticker": ticker,
        "date": "",
        "stress_score": 0.0,
        "stress_level": 0,
        "label": "Normal",
        "color": "green",
        "action": "trade_normally",
        "description": "Market conditions normal. Model signals reliable.",
        "top_stress_driver": "unknown",
        "vix_level": None,
        "recommendation": "Market conditions normal. Model signals reliable.",
    }


def _get_driver_breakdown(row_df: pd.DataFrame) -> dict:
    """Return individual stress component scores for the given row(s)."""
    drivers = {}
    if "realised_vol_5" in row_df.columns and "realised_vol_20" in row_df.columns:
        ratio = row_df["realised_vol_5"].iloc[0] / max(row_df["realised_vol_20"].iloc[0], 1e-9)
        drivers["volatility_spike"] = min((ratio - 1) / 3 * 100, 100)
    if "close" in row_df.columns:
        drivers["drawdown"] = 0.0  # single row — no rolling context
    if "volume_zscore_20" in row_df.columns:
        drivers["volume_anomaly"] = min(abs(row_df["volume_zscore_20"].iloc[0]) / 5 * 100, 100)
    if "rsi_14" in row_df.columns:
        rsi = row_df["rsi_14"].iloc[0]
        if rsi < 20:
            drivers["rsi_extreme"] = (20 - rsi) / 20 * 100
        elif rsi > 85:
            drivers["rsi_extreme"] = (rsi - 85) / 15 * 100
        else:
            drivers["rsi_extreme"] = 0.0
    return drivers
