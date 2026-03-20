from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime

class TrainRequest(BaseModel):
    ticker: str = Field(..., pattern="^[A-Z]{1,5}$")
    start_date: str = "2019-01-01"
    end_date: str = "2023-12-31"
    use_sentiment: bool = True
    use_regime: bool = True
    n_cv_folds: int = 12
    model_types: List[str] = ["xgboost", "lightgbm", "logistic"]

class TrainResponse(BaseModel):
    run_id: str
    status: str
    mean_f1: float
    std_f1: float
    sharpe_ratio: float
    bootstrap_pvalue: float
    mlflow_run_url: str

class FeatureImportance(BaseModel):
    feature: str
    shap_value: float
    feature_value: float

class PredictResponse(BaseModel):
    ticker: str
    date: str
    signal: str
    buy_probability: float
    prediction_set: List[int]
    confidence: str
    top_3_features: List[FeatureImportance]
    regime: str
    model_version: str

class BacktestMetrics(BaseModel):
    total_return: float
    cagr: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int

class BacktestResponse(BaseModel):
    ticker: str
    period: str
    metrics: BacktestMetrics
    benchmarks: Dict[str, Dict[str, float]]
    significance: Dict[str, Any]

class DataResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    n_rows: int
    columns: List[str]
    preview: List[Dict[str, Any]]

class DriftResponse(BaseModel):
    ticker: str
    last_checked: datetime
    overall_status: str
    should_retrain: bool
    reason: str
    features: Dict[str, Dict[str, Any]]
    rolling_accuracy: Dict[str, Any]
