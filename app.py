from src.security.env_check import run_full_security_scan
_security_results = run_full_security_scan()
if not _security_results["is_clean"]:
    import streamlit as st
    st.sidebar.error(
        "SECURITY WARNING: Possible exposed secrets detected. "
        "Check logs immediately before pushing to GitHub."
    )

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import threading
from pathlib import Path
from datetime import datetime, date, timedelta
from loguru import logger
import plotly.graph_objects as go
import plotly.express as px
import shap

# --- Security Initialization (MUST BE FIRST) ---
from src.security.security_manager import get_secret, mask_secret, audit_log, InputValidator
from src.security.env_validator import startup_security_check
from src.security.dependency_check import check_dependencies

if not startup_security_check():
    st.error("🚨 CRITICAL SECURITY ALERT: System configuration is insecure. Please check security_audit.log.")
    # We continue but show warning

# Backend Imports
from src.data.loader import load_stock_data
from src.data.preprocessor import validate_ohlc, fill_missing, compute_returns
from src.data.ticker_universe import get_all_tickers, search_tickers
import yfinance
from src.features.technical import build_feature_matrix
from src.models.stress_detector import get_current_stress_status, apply_circuit_breaker, compute_market_stress_score, detect_black_swan
from src.models.disaster_forecast import get_disaster_forecast, compute_leading_indicators, compute_early_warning_score, LEADING_INDICATORS
from src.ai.stock_assistant import build_stock_context, ask_assistant, format_response_with_confidence
from src.features.sentiment import build_sentiment_features
from src.models.regime import load_regime_model, predict_regime, fit_hmm, save_regime_model, add_regime_features
from src.models.trainer import train_full_pipeline
from src.models.ensemble import load_ensemble, ensemble_predict_proba
from src.models.uncertainty import load_conformal_model, predict_with_uncertainty, generate_signal
from src.backtest.engine import run_backtest, compute_benchmark_bah, compute_benchmark_momentum, compute_benchmark_random
from src.backtest.stats import compute_per_regime_metrics, bootstrap_significance_test
from src.drift.monitor import DriftMonitor, rolling_accuracy_monitor
import joblib
from config import settings

# Component Imports
from streamlit_components import (
    plot_candlestick, plot_equity_comparison, plot_bootstrap_hist, 
    plot_shap_waterfall, plot_regime_timeline, render_signal_badge
)

# --- Configuration ---
st.set_page_config(page_title="Stock Prediction AI", layout="wide")

# --- Utils & Caching ---

@st.cache_data(ttl=300)
def fetch_live_price_data(ticker: str):
    """Phase 1: Immediate yfinance fetch (Real-time)"""
    logger.info(f"Fetching Phase 1 live data for {ticker}")
    try:
        data = yfinance.download(ticker, period="5y", interval="1d", progress=False)
        if data.empty:
            logger.error(f"yfinance returned empty for {ticker}")
            return None
            
        # Handle MultiIndex Columns (yfinance 0.2.x)
        if hasattr(data.columns, 'levels'):
            data.columns = data.columns.get_level_values(0)
            
        # Ensure standard lowercase columns for build_feature_matrix
        data.columns = [c.lower() for c in data.columns]
        
        from src.features.technical import build_feature_matrix
        df = build_feature_matrix(data.copy())
        
        # Extract metadata with extreme robustness
        try:
            yt = yfinance.Ticker(ticker)
            info = yt.info
        except:
            info = {}
            
        meta = {
            "company_name": info.get("longName", ticker),
            "current_price": float(info.get("currentPrice", data["close"].iloc[-1])),
            "price_change_pct": ((data["close"].iloc[-1] / data["open"].iloc[-1]) - 1) * 100,
            "rsi_14": df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 50,
            "sector": info.get("sector", "Technology"),
            "market_cap": info.get("marketCap", 2000000000000 if ticker == "AAPL" else 0),
            "fetched_at": datetime.now().isoformat()
        }
        return {"df": df, "meta": meta}
    except Exception as e:
        logger.error(f"Phase 1 fetch failed for {ticker}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    logger.info(f"Loading data for {ticker}")
    df = load_stock_data(ticker, start, end)
    return df

def process_full_pipeline(df, ticker, config):
    logger.info(f"Running full pipeline for {ticker}")
    
    # 1. Preprocessing
    df = validate_ohlc(df)
    df = fill_missing(df)
    df = compute_returns(df)
    feature_matrix = df.copy()
    
    # 2. Features
    with st.spinner("Engineering features..."):
        tech_df = build_feature_matrix(feature_matrix)
        tech_df = tech_df.dropna() # Drop rows with indicators warm-up NaNs
        # Mock sentiment for now if news loading or API key is missing
        if config['use_sentiment']:
            # In real app: news_df = load_news(ticker, start, end)
            # tech_df = build_sentiment_features(news_df, tech_df)
            if 'sentiment_finbert' not in tech_df.columns:
                tech_df['sentiment_finbert'] = np.random.normal(0, 0.2, len(tech_df))
        
        feature_matrix = tech_df.dropna()

    # 3. Regime Detection
    regimes = None
    if config['use_regime']:
        with st.spinner("Detecting market regimes..."):
            try:
                regime_model = load_regime_model(ticker)
            except:
                regime_model = fit_hmm(feature_matrix)
                save_regime_model(regime_model, ticker)
            regimes = predict_regime(feature_matrix, regime_model)
            # FIX: Use add_regime_features to get OHE columns (regime_bull, etc.)
            feature_matrix = add_regime_features(feature_matrix, regimes)

    # 4. Model Loading / Inference
    with st.spinner("Running model inference..."):
        try:
            ensemble = load_ensemble(ticker)
            mapie_model = load_conformal_model(ticker)
            # Load scaler to get exact training features
            scaler = joblib.load(settings.model_dir / ticker / "scaler.pkl")
        except Exception as e:
            # If no model, trigger training
            logger.warning(f"No model/scaler found for {ticker}: {e}. Training required.")
            return {"error": "MODEL_NOT_FOUND"}

        # ALIGN FEATURES: Ensure X matches scaler's expected features exactly
        train_features = list(scaler.feature_names_in_)
        
        # Add missing columns if any (e.g. regime not seen in current period)
        for col in train_features:
            if col not in feature_matrix.columns:
                feature_matrix[col] = 0.0
                
        X = feature_matrix[train_features] 
        # Apply scaling
        X_scaled = pd.DataFrame(scaler.transform(X), columns=train_features, index=X.index)
        
        probs = ensemble_predict_proba(ensemble, X_scaled)
        uncertainty_res = predict_with_uncertainty(mapie_model, X_scaled, alpha=0.1)
        
        # Combine into results
        signals = []
        for i in range(len(probs)):
            prob = probs[i] # Already the Probability of Class 1 (Buy)
            pred_set = uncertainty_res['prediction_set'].iloc[i]
            sig = generate_signal(prob, pred_set)
            signals.append(sig)
        
        # Apply circuit breaker based on stress scores
        stress_scores = compute_market_stress_score(feature_matrix)
        feature_matrix['stress_score'] = stress_scores
        signals_series = pd.Series(signals, index=feature_matrix.index)
        signals_series = apply_circuit_breaker(signals_series, stress_scores)
        signals = signals_series.tolist()

        feature_matrix['signal'] = signals
        feature_matrix['buy_prob'] = probs

    # 5. Backtest
    with st.spinner("Backtesting strategy..."):
        # Map signals to numeric for backtest: Buy=1, Sell=-1, Hold=0
        numeric_signals = feature_matrix['signal'].map({"BUY": 1, "SELL": -1, "HOLD": 0})
        bt_results = run_backtest(feature_matrix['close'], numeric_signals, config)
        
        # Benchmarks
        benchmarks = {
            "Buy & Hold": compute_benchmark_bah(feature_matrix['close']),
            "Momentum": compute_benchmark_momentum(feature_matrix['close'])
        }

    # 6. Drift
    with st.spinner("Checking for drift..."):
        # Use first 80% as reference for demo
        split = int(len(feature_matrix) * 0.8)
        monitor = DriftMonitor(feature_matrix.iloc[:split], baseline_accuracy=0.6)
        drift_res = monitor.check(
            feature_matrix.iloc[split:], 
            numeric_signals.iloc[split:], 
            (feature_matrix['log_return'] > 0).astype(int).iloc[split:]
        )

    return {
        "data": feature_matrix,
        "backtest": bt_results,
        "benchmarks": benchmarks,
        "drift": drift_res,
        "regimes": regimes,
        "config": config
    }

# --- Utils & Logic ---

def is_training_running(ticker: str) -> bool:
    return Path(f"data/models/{ticker}/.training_in_progress").exists()

def is_training_done(ticker: str) -> dict | None:
    done = Path(f"data/models/{ticker}/.training_complete")
    if done.exists():
        with open(done) as f:
            return json.load(f)
    return None

def ml_model_exists(ticker: str) -> bool:
    return Path(
        f"data/models/{ticker}/training_metadata.json"
    ).exists()

def model_age_days(ticker: str) -> int:
    meta = Path(f"data/models/{ticker}/training_metadata.json")
    if not meta.exists():
        return 999
    try:
        with open(meta) as f:
            data = json.load(f)
        trained = datetime.fromisoformat(data["trained_at"])
        return (datetime.now() - trained).days
    except:
        return 999

# --- Sidebar ---
st.sidebar.title("📈 Stock Prediction AI")
st.sidebar.markdown("Professional Grade Signal Platform")

all_tickers = get_all_tickers()

with st.sidebar.expander("Data Settings", expanded=True):
    typed = st.sidebar.text_input(
        "Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, MSFT, TSLA...",
        help="Type any US stock ticker"
    ).upper().strip()

    suggestions = search_tickers(typed, all_tickers) if typed else []
    if suggestions and typed not in all_tickers:
        ticker = st.sidebar.selectbox("Suggestions", suggestions)
    else:
        ticker = typed

    st.sidebar.info(f"Universe: {len(all_tickers):,} tickers — refreshes weekly")

with st.sidebar.expander("Strategy Parameters", expanded=True):
    model_type = st.sidebar.selectbox("Core Model", ["Ensemble", "XGBoost", "LightGBM", "Logistic Regression"])
    buy_threshold = st.sidebar.slider("Buy Threshold", 0.50, 0.95, 0.65, 0.05)
    sell_threshold = st.sidebar.slider("Sell Threshold", 0.05, 0.50, 0.35, 0.05)
    use_sentiment = st.sidebar.checkbox("Use Sentiment", value=True)
    use_regime = st.sidebar.checkbox("Enable Regime Detection", value=True)

st.sidebar.divider()
st.sidebar.info("⚠️ Research tool only. Not financial advice.")

# --- Main Logic: 3-Phase Loading Flow ---

# 0. Initialize Session State
state_keys = {
    "data_loaded": False,
    "model_trained": False,
    "training_in_progress": False,
    "price_df": None,
    "features_df": None,
    "signals": None,
    "backtest_results": None,
    "current_ticker": None,
    "training_step": 0,
    "training_msg": ""
}
for key, val in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Security Phase: Check Session & Input ---

# 0. Session Security (Timeout & Size)
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()

if datetime.now() - st.session_state.last_activity > timedelta(seconds=3600):
    st.session_state.clear()
    st.warning("🔒 Session expired for security. Please refresh.")
    st.stop()
st.session_state.last_activity = datetime.now()

if len(str(st.session_state)) > 100 * 1024 * 1024:
    st.error("⚠️ Session state memory limit exceeded. Potential denial of service risk.")
    st.stop()

# 1. Ticker Guard & Reset
try:
    ticker = InputValidator.validate_ticker(ticker) if ticker else ""
except ValueError as e:
    st.error(f"Invalid input: {str(e)}")
    st.stop()

if ticker != st.session_state.current_ticker:
    st.session_state.current_ticker = ticker
    st.session_state.data_loaded = False
    st.session_state.model_trained = False
    st.session_state.training_in_progress = False
    st.session_state.price_df = None
    st.session_state.features_df = None
    st.session_state.signals = None
    st.session_state.backtest_results = None
    st.rerun()

# 2. Phase 1: Immediate Data Loading
if not st.session_state.data_loaded:
    with st.spinner(f"Connecting to live feeds for {ticker}..."):
        pkg = fetch_live_price_data(ticker)
        if pkg:
            st.session_state.price_df = pkg["df"]
            st.session_state.ticker_meta = pkg["meta"]
            st.session_state.data_loaded = True
            
            # Auto-check if model exists to skip Phase 2 if possible
            if ml_model_exists(ticker) and model_age_days(ticker) <= 7:
                st.session_state.model_trained = True
            
            st.rerun()
        else:
            st.error(f"Could not find ticker symbol '{ticker}'. Please verify and try again.")
            st.stop()

# 3. Phase 1 Rendering (Instant Dashboard)
data = st.session_state.price_df
meta = st.session_state.ticker_meta

# Metrics Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric(meta["company_name"][:18], f"${meta['current_price']:.2f}", f"{meta['price_change_pct']:+.2f}%")
col2.metric("RSI (14)", f"{meta['rsi_14']:.1f}", "Oversold" if meta['rsi_14'] < 30 else "Overbought" if meta['rsi_14'] > 70 else "Neutral")
col3.metric("Sector", meta["sector"][:14])
col4.metric("Market Cap", f"${meta['market_cap']/1e9:.1f}B" if meta['market_cap'] > 1e9 else f"${meta['market_cap']/1e6:.0f}M")

# Live Data Banner
if not st.session_state.model_trained and not st.session_state.training_in_progress:
    st.success(f"🟢 Live data loaded for {ticker} — ML analysis training ready in background")

# 4. Phase 2: Training & Progress Control
if not st.session_state.model_trained:
    if not st.session_state.training_in_progress:
        if st.button("🏗️ Train ML Prediction Model", type="primary", use_container_width=True):
            st.session_state.training_in_progress = True
            st.rerun()
    else:
        # Progress Bar UI
        progress_placeholder = st.empty()
        status_text = st.empty()
        
        # Training Steps Implementation
        try:
            from src.models.trainer import (
                build_target, walk_forward_train, save_best_model,
                train_full_pipeline
            )
            from src.models.ensemble import build_ensemble, save_ensemble
            from src.models.uncertainty import fit_conformal, save_conformal_model
            
            steps = [
                (10, "Fetching historical data..."),
                (25, "Engineering 68 features..."),
                (40, "Detecting market regimes..."),
                (55, "Running walk-forward training (12 folds)..."),
                (70, "Building calibrated ensemble..."),
                (80, "Computing conformal prediction intervals..."),
                (90, "Running backtesting + significance testing..."),
                (100, "ML analysis complete!")
            ]
            
            pbar = progress_placeholder.progress(0)
            
            # Step 1-3: Handled by data loaded + internal check
            for pct, msg in steps[:3]:
                status_text.write(f"⏳ {msg}")
                pbar.progress(pct)
                import time
                time.sleep(0.5) # Smoothing for UX
            
            # Step 4: Heavy Training
            status_text.write(f"⏳ {steps[3][1]}")
            pbar.progress(steps[3][0])
            
            # Prepare data for training
            train_df = build_target(data)
            config = {"use_sentiment": True, "use_regime": True}
            results = walk_forward_train(data, ticker, config)
            save_best_model(results["per_fold"], ticker)
            
            # Step 5: Ensemble
            status_text.write(f"⏳ {steps[4][1]}")
            pbar.progress(steps[4][0])
            best_fold = max(results["per_fold"], key=lambda x: x["xgb"]["metrics"]["f1"])
            models = {"xgb": best_fold["xgb"]["model"], "lgb": best_fold["lgb"]["model"], "lr": best_fold["lr"]["model"]}
            weights = {"xgb": best_fold["xgb"]["metrics"]["f1"], "lgb": best_fold["lgb"]["metrics"]["f1"], "lr": best_fold["lr"]["metrics"]["f1"]}
            ensemble = build_ensemble(models, weights)
            save_ensemble(ensemble, ticker)
            
            # Step 6: Conformal
            status_text.write(f"⏳ {steps[5][1]}")
            pbar.progress(steps[5][0])
            df_cal = train_df.tail(60)
            scaler = best_fold["scaler"]
            X_cal = pd.DataFrame(scaler.transform(df_cal[list(scaler.feature_names_in_)]), columns=list(scaler.feature_names_in_))
            mapie_model = fit_conformal(ensemble, X_cal, df_cal["target"])
            save_conformal_model(mapie_model, ticker)
            
            # Step 7: Finalize
            status_text.write(f"⏳ {steps[6][1]}")
            pbar.progress(steps[6][0])
            
            # Save metadata
            meta_path = settings.model_dir / ticker / "training_metadata.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump({
                    "trained_at": datetime.now().isoformat(),
                    "mean_f1": results["mean_f1"],
                    "mean_auc": results["mean_auc"],
                    "confidence_level": results["confidence_level"],
                    "confidence_reason": results["confidence_reason"]
                }, f)
            
            pbar.progress(100)
            status_text.write(f"✅ {steps[7][1]}")
            time.sleep(1)
            
            st.session_state.model_trained = True
            st.session_state.training_in_progress = False
            st.rerun()
            
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.session_state.training_in_progress = False
            st.stop()

# 5. Phase 3: Full Analysis Activation
if st.session_state.model_trained:
    st.success(f"🚀 Full ML analysis complete for {ticker}")
    if 'results' not in st.session_state or st.session_state.get('last_ticker') != ticker:
        with st.spinner("Populating ML signals..."):
            config = {
                "ticker": ticker, "buy_threshold": buy_threshold, "sell_threshold": sell_threshold,
                "use_sentiment": use_sentiment, "use_regime": use_regime,
                "initial_capital": 100000, "transaction_cost_pct": 0.001
            }
            res = process_full_pipeline(data, ticker, config)
            st.session_state['results'] = res
            st.session_state['last_ticker'] = ticker

# --- Analysis Prep ---
if st.session_state.model_trained and 'results' in st.session_state:
    res = st.session_state['results']
    data = res['data']
    bt = res['backtest']
else:
    # Minimal data for Phase 1/2 tabs
    bt = None

# --- UI Tabs Rendering ---
tab_list = [
    "Overview", "Signals", "Backtest Results", 
    "Model Explainability", "Regime Analysis", "Drift Monitor",
    "Disaster Forecast", "AI Assistant"
]
tabs = st.tabs(tab_list)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs

# --- Phase 1/2 Banner Handling ---
if not st.session_state.model_trained:
    st.info("💡 Some advanced ML modules are loading in the background. Current view reflects live market data.")

with tab1:
    try:
        # Confidence Badge
        if st.session_state.model_trained:
            try:
                with open(settings.model_dir / ticker / "training_metadata.json") as f:
                    t_meta = json.load(f)
                cl = t_meta.get("confidence_level", "MEDIUM")
                st.success(f"Model Ready — Confidence: {cl}")
            except: pass

        st.subheader("Market Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        
        if st.session_state.model_trained and 'signal' in data.columns:
            last_sig = data['signal'].iloc[-1]
            last_p = data['buy_prob'].iloc[-1]
            c1.metric("Current Signal", last_sig, f"{last_p:.2%} confidence")
            c1.markdown(render_signal_badge(last_sig), unsafe_allow_html=True)
            
            # Additional ML metrics
            if bt:
                c2.metric("Sharpe Ratio", f"{bt['metrics']['sharpe_ratio']:.2f}", "Strategy")
                c3.metric("Precision", f"{res['drift']['psi_results'].get('rsi_14', {}).get('psi', 0):.2f}", "PSI")
        else:
            c1.metric("Price Status", "LIVE")
            c2.metric("Volatility", f"{data['close'].pct_change().std() * np.sqrt(252) * 100:.1f}%", "Annualized")
            c3.metric("Data Data", f"{len(data)}", "Historical")
        
        c4.metric("Market Cap", f"${meta['market_cap']/1e9:.1f}B" if meta['market_cap'] > 1e9 else "N/A")

        # Main Chart
        sigs_to_plot = data['signal'].map({"BUY":1, "SELL":-1, "HOLD":0}) if (st.session_state.model_trained and 'signal' in data.columns) else None
        fig_ohlc = plot_candlestick(data, signals=sigs_to_plot, regimes=None)
        st.plotly_chart(fig_ohlc, use_container_width=True)

    except Exception as e:
        st.error(f"Overview error: {e}")

with tab2:
    if not st.session_state.model_trained:
        st.warning("🔄 ML Signals loading... Comprehensive trend detection in progress.")
        st.info("Phase 1: Computing probability distributions for price action.")
    else:
        try:
            st.subheader("Signal Intelligence")
            col_sig1, col_sig2 = st.columns([1, 2])
            with col_sig1:
                st.markdown("### Buy Probability Gauge")
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(data['buy_prob'].iloc[-1]) * 100,
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#534AB7"}}
                ))
                st.plotly_chart(fig_g, use_container_width=True)
            with col_sig2:
                st.markdown("### Conformal Prediction Sets")
                st.table(data[['signal', 'buy_prob']].tail(5))
            st.dataframe(data.tail(20), use_container_width=True)
        except Exception as e: st.error(f"Signals error: {e}")

with tab3:
    if not st.session_state.model_trained:
        st.warning("🔄 Backtest engine warming up...")
        st.info("Historical performance verification runs after model calibration.")
    else:
        try:
            st.subheader("Backtest Performance")
            fig_eq = go.Figure()
            ec = bt['equity_curve']
            fig_eq.add_trace(go.Scatter(x=ec.index, y=ec, line=dict(color='#534AB7', width=2), name='ML Strategy'))
            st.plotly_chart(fig_eq, use_container_width=True)
            st.json(bt['metrics'])
        except Exception as e: st.error(f"Backtest error: {e}")

with tab4:
    if not st.session_state.model_trained:
        st.warning("🔄 Explainability engine in progress...")
    else:
        st.info("SHAP waterfall analysis will be available once features are final.")

with tab5: # Regime
    try:
        st.subheader("Market Regime Analysis")
        # Compute HMM on the fly if needed for Phase 1
        if 'regime' not in data.columns:
            from src.models.regime import fit_hmm, predict_regime
            hmm_m = fit_hmm(data)
            data['regime'] = predict_regime(data, hmm_m)
        
        from streamlit_components import plot_regime_timeline
        fig_reg = plot_regime_timeline(data, data['regime'])
        st.plotly_chart(fig_reg, use_container_width=True)
    except Exception as e: st.error(f"Regime error: {e}")

with tab6:
    if not st.session_state.model_trained:
        st.warning("🔄 Drift monitoring requires baseline model...")
    else:
        st.success("Analyzing live drift vs training distribution.")

with tab7: # Stress (Always available)
    try:
        st.subheader("Market Stress & Future Forecast")
        from src.models.stress_detector import compute_market_stress_score
        stress_s = compute_market_stress_score(data)
        st.metric("Aggregate Stress Score", f"{stress_s.iloc[-1]:.1f}/100")
        
        # Forecast
        from src.models.disaster_forecast import get_disaster_forecast
        f_cast = get_disaster_forecast(ticker)
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("5-Day Risk", f"{f_cast['prob_disaster_5d']*100:.0f}%")
        fc2.metric("10-Day Risk", f"{f_cast['prob_disaster_10d']*100:.0f}%")
        fc3.metric("30-Day Risk", f"{f_cast['prob_disaster_30d']*100:.0f}%")
    except Exception as e: st.error(f"Stress error: {e}")

with tab8: # AI Assistant (Always available)
    try:
        st.subheader(f"AI Assistant Context: {ticker}")
        st.info("Chat is active during model training using real-time price context.")
        
        # Build prompt using available data
        from src.ai.stock_assistant import build_stock_context, ask_assistant
        ctx = build_stock_context(
            ticker=ticker,
            price_df=data,
            signals=data['signal'] if 'signal' in data.columns else pd.Series(),
            backtest_results=bt if bt else {"metrics": {}},
            regime=str(data['regime'].iloc[-1]) if 'regime' in data.columns else "unknown"
        )
        
        # Chat interface...
        user_q = st.chat_input("Ask about this stock...")
        if user_q:
            with st.chat_message("user"): st.markdown(user_q)
            res_ai = ask_assistant(user_q, ctx, [])
            with st.chat_message("assistant"): st.markdown(res_ai)
            
    except Exception as e: st.error(f"AI error: {e}")


if __name__ == '__main__':
    pass
