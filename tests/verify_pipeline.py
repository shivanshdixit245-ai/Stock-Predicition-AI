import pandas as pd
from src.data.loader import load_stock_data
from src.data.preprocessor import validate_ohlc, fill_missing, compute_returns
from src.features.technical import build_feature_matrix
from src.models.trainer import train_full_pipeline
from loguru import logger
import os

def verify():
    ticker = "AAPL"
    logger.info(f"Verifying full pipeline for {ticker}...")
    
    # 1. Load small data for speed
    prices = load_stock_data(ticker, "2022-01-01", "2024-03-18")
    df = validate_ohlc(prices)
    df = fill_missing(df)
    df = compute_returns(df)
    df = build_feature_matrix(df).dropna()
    
    # 2. Run pipeline
    config = {"use_sentiment": False, "use_regime": True}
    try:
        train_full_pipeline(df, ticker, config)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Check artifacts
    model_dir = f"data/models/{ticker}"
    expected_files = [
        "xgb_model.pkl", "lgb_model.pkl", "lr_model.pkl", "scaler.pkl",
        "calibrated_ensemble.pkl", "mapie_conformal.pkl"
    ]
    
    all_exist = True
    for f in expected_files:
        path = os.path.join(model_dir, f)
        if os.path.exists(path):
            logger.success(f"Verified: {f} exists.")
        else:
            logger.error(f"Missing: {f}")
            all_exist = False
            
    if all_exist:
        logger.success("All artifacts created successfully!")
    else:
        logger.error("Verification failed!")

if __name__ == "__main__":
    verify()
