import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.data.loader import load_stock_data
from src.data.preprocessor import validate_ohlc, fill_missing, compute_returns
from src.features.technical import build_feature_matrix
from src.models.regime import load_regime_model, predict_regime, add_regime_features
from src.models.ensemble import load_ensemble, ensemble_predict_proba
from src.models.uncertainty import load_conformal_model, predict_with_uncertainty, generate_signal
from src.backtest.engine import run_backtest
from config import settings

def debug_pipeline():
    ticker = "MSFT"
    start = "2023-01-01"
    end = "2023-12-31"
    
    print(f"Loading data for {ticker}...")
    df = load_stock_data(ticker, start, end)
    df = validate_ohlc(df)
    df = fill_missing(df)
    df = compute_returns(df)
    
    print("Building features...")
    feature_matrix = build_feature_matrix(df)
    feature_matrix = feature_matrix.dropna()
    
    print("Adding regime...")
    regime_model = load_regime_model(ticker)
    regimes = predict_regime(df, regime_model)
    feature_matrix = add_regime_features(feature_matrix, regimes)
    
    print("Loading models...")
    ensemble = load_ensemble(ticker)
    mapie_model = load_conformal_model(ticker)
    scaler = joblib.load(settings.model_dir / ticker / "scaler.pkl")
    
    train_features = list(scaler.feature_names_in_)
    for col in train_features:
        if col not in feature_matrix.columns:
            feature_matrix[col] = 0.0
            
    X = feature_matrix[train_features]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=train_features, index=X.index)
    
    print("Predicting...")
    probs = ensemble_predict_proba(ensemble, X_scaled)
    uncertainty_res = predict_with_uncertainty(mapie_model, X_scaled, alpha=0.1)
    
    signals = []
    for i in range(len(probs)):
        prob = probs[i]
        pred_set = uncertainty_res['prediction_set'].iloc[i]
        sig = generate_signal(prob, pred_set)
        signals.append(sig)
        
    feature_matrix['signal'] = signals
    feature_matrix['buy_prob'] = probs
    
    print(f"Probabilities stats: min={np.min(probs):.4f}, max={np.max(probs):.4f}, mean={np.mean(probs):.4f}")
    
    print(f"Signals unique values: {pd.Series(signals).value_counts().to_dict()}")
    
    print("Backtesting...")
    numeric_signals = feature_matrix['signal'].map({"BUY": 1, "SELL": -1, "HOLD": 0})
    config = {"initial_capital": 100000, "cost_pct": 0.001}
    bt_results = run_backtest(feature_matrix['close'], numeric_signals, config)
    
    print(f"Backtest metrics: {bt_results['metrics']}")
    print(f"Equity curve head/tail: {bt_results['equity_curve'].head(2).tolist()}, {bt_results['equity_curve'].tail(2).tolist()}")

if __name__ == "__main__":
    debug_pipeline()
