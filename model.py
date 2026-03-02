import numpy as np
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../backend/.env")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def fetch_real_ohlcv(symbol: str):
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return None
    
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        ts = data.get("Time Series (Daily)")
        if not ts:
            return None
        
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.astype(float)
        return df
    except Exception:
        return None

def make_mock_ohlcv(n=120):
    rng = np.random.default_rng(42)
    prices = np.cumsum(rng.normal(0, 1, n)) + 100
    prices = np.maximum(prices, 1)
    high = prices + rng.random(n) * 2
    low = prices - rng.random(n) * 2
    openp = prices + rng.normal(0, 0.5, n)
    volume = rng.integers(1000, 5000, n)
    df = pd.DataFrame({
        "open": openp,
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume
    })
    return df

def train_or_dummy(features: pd.DataFrame, target: pd.Series):
    if XGB_AVAILABLE:
        try:
            model = XGBClassifier(
                n_estimators=60,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=1,
                verbosity=0
            )
            model.fit(features, target)
            return model
        except Exception:
            pass
    return None

def heuristic_proba(row: pd.Series):
    rsi = row.get("RSI", 50.0)
    macd = row.get("MACD", 0.0)
    ma20 = row.get("MA20", row.get("close", 0))
    ma50 = row.get("MA50", row.get("close", 0))
    score = 0.5
    if rsi > 55: score += 0.1
    if rsi < 45: score -= 0.1
    if macd > 0: score += 0.1
    if macd < 0: score -= 0.1
    if ma20 > ma50: score += 0.05
    if ma20 < ma50: score -= 0.05
    return max(0.05, min(0.95, score))
