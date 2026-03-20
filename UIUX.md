# UI/UX — Streamlit Dashboard Specification

## Design Principles
- Clean, professional — looks like a real fintech product
- Data-dense but readable
- Every chart is interactive (Plotly, not matplotlib)
- Mobile-aware (Streamlit's default is responsive enough)

---

## Layout Overview

```
┌─────────────────────────────────────────────────────┐
│  SIDEBAR                  │  MAIN CONTENT           │
│  ─────────────────────    │  ─────────────────────  │
│  Ticker: [AAPL ▾]        │  [Tab navigation]        │
│  Start:  [2022-01-01]    │                          │
│  End:    [2024-01-01]    │                          │
│  ─────────────────────    │                          │
│  Model:  [Ensemble ▾]    │                          │
│  Buy threshold: [0.65]   │                          │
│  Sell threshold: [0.35]  │                          │
│  ─────────────────────    │                          │
│  [Run Analysis]          │                          │
└─────────────────────────────────────────────────────┘
```

---

## Sidebar Components

```python
st.sidebar.title("Configuration")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("today"))

st.sidebar.divider()
st.sidebar.subheader("Model Settings")
model_type = st.sidebar.selectbox("Model", ["Ensemble", "XGBoost", "LightGBM", "Logistic Regression"])
buy_threshold = st.sidebar.slider("Buy threshold", 0.5, 0.95, 0.65, 0.05)
sell_threshold = st.sidebar.slider("Sell threshold", 0.05, 0.5, 0.35, 0.05)
use_sentiment = st.sidebar.checkbox("Include sentiment", value=True)
use_regime = st.sidebar.checkbox("Regime detection", value=True)

st.sidebar.divider()
run_button = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)
```

---

## Tab 1: Overview

### Metric cards row (4 columns)
```python
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Signal", "BUY", delta="73% confidence")
col2.metric("Sharpe Ratio", "1.42", delta="+0.41 vs benchmark")
col3.metric("Model F1", "0.61", delta="±0.04 (walk-forward)")
col4.metric("Drift Status", "Stable", delta=None)
```

### Main price chart
Full-width Plotly candlestick chart with:
- OHLCV candlesticks (green/red)
- Volume bars at bottom
- SMA20 and SMA50 overlays (toggle)
- Bollinger Bands overlay (toggle)
- RSI subplot below price (toggle)
- MACD subplot below RSI (toggle)
- BUY signals: green upward triangle on candle
- SELL signals: red downward triangle on candle
- HOLD zones: light grey background shading
- Regime colouring: faint background tint (green=bull, red=bear, grey=sideways)

---

## Tab 2: Signals

### Prediction confidence gauge
```python
# Plotly gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=current_prob * 100,
    title={"text": "Buy Probability"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#534AB7"},
        "steps": [
            {"range": [0, 35], "color": "#FCEBEB"},
            {"range": [35, 65], "color": "#F1EFE8"},
            {"range": [65, 100], "color": "#E1F5EE"},
        ],
        "threshold": {
            "line": {"color": "black", "width": 2},
            "thickness": 0.75,
            "value": 65,
        },
    },
))
```

### Conformal prediction set
```
Today's prediction: BUY  (set: {BUY})  ← high confidence, single class
Tomorrow estimate:  HOLD (set: {BUY, SELL}) ← uncertain, both classes
```
Show as a styled table with colour coding.

### Recent signals table
Last 20 trading days: Date | Price | Signal | Probability | Actual | Correct?
Highlight rows where conformal set contained both classes (uncertain predictions).

---

## Tab 3: Backtest Results

### Performance summary cards
```
Total Return    │  Sharpe Ratio  │  Max Drawdown  │  Bootstrap p-value
────────────────┼────────────────┼────────────────┼────────────────────
+71.2%          │  1.42          │  -18.2%        │  0.023 ✓
vs BnH: +34.1%  │  vs BnH: 0.81  │  vs BnH: -22.4%│  Significant at 95%
```

### Equity curve
Multi-line Plotly chart:
- ML Strategy (purple, solid)
- Buy & Hold (grey, dashed)
- MA Momentum (teal, dotted)
- Drawdown shading (red fill, 15% opacity)

### Bootstrap significance plot
Histogram of null Sharpe ratios (grey bars) + vertical line at observed Sharpe (purple) + shaded 95th percentile region. p-value annotation.

### Per-regime performance table
| Regime | Trades | Win Rate | Sharpe | Notes |
|--------|--------|----------|--------|-------|
| Bull | 48 | 62% | 1.8 | Strong signal |
| Bear | 22 | 44% | 0.3 | Weaker |
| Sideways | 67 | 53% | 0.9 | Moderate |

---

## Tab 4: Model Explainability

### SHAP waterfall plot (most recent prediction)
```python
import shap
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("shap_waterfall.png", bbox_inches="tight")
st.image("shap_waterfall.png")
```

### Feature importance bar chart
Horizontal bar chart: top 15 features by mean absolute SHAP value.

### Calibration curve
Plotly scatter: predicted probability (x) vs. empirical accuracy (y). Diagonal reference line. Shows model is well-calibrated.

### Walk-forward F1 per fold
Bar chart: fold number (x) vs. F1 score (y). Horizontal line at mean F1. Shaded band at ±1 std.

---

## Tab 5: Regime Analysis

### Regime timeline
Price chart with colour-coded background by regime. Legend: Bull / Bear / Sideways.

### Regime distribution pie chart
What % of the test period was each regime.

### HMM transition matrix heatmap
Show probability of transitioning from regime A to regime B. Useful for understanding market dynamics.

---

## Tab 6: Drift Monitor

### Feature PSI table
| Feature | Reference Mean | Current Mean | PSI | Status |
|---------|---------------|--------------|-----|--------|
| rsi_14 | 52.3 | 49.1 | 0.08 | 🟢 Stable |
| macd | 0.42 | -0.21 | 0.24 | 🔴 Drift |

### Rolling accuracy chart
Line chart: rolling 20-day accuracy over time. Reference line at baseline. Alert zones shaded in red.

### Retrain recommendation
```python
if drift_result["should_retrain"]:
    st.warning(f"Retraining recommended: {drift_result['retrain_reason']}")
    if st.button("Retrain Model Now"):
        with st.spinner("Retraining..."):
            run_id = train_and_log(ticker, config)
        st.success(f"Retrained successfully. MLflow run: {run_id}")
```

---

## Streamlit Performance Tips

```python
# Cache expensive computations
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    return data_loader.load_stock_data(ticker, start, end)

@st.cache_resource
def load_model(ticker):
    return load_trained_model(ticker)

# Use spinners for long operations
with st.spinner("Running analysis..."):
    results = run_full_pipeline(ticker, config)
```

All tabs must load in < 3 seconds. Pre-compute features and model predictions on "Run Analysis" click, cache results for the session.
