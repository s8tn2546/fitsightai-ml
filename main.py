from fastapi import FastAPI
from pydantic import BaseModel
from indicators import compute_indicators
from model import make_mock_ohlcv, train_or_dummy, heuristic_proba, fetch_real_ohlcv
import pandas as pd

app = FastAPI(title="ML Service")

class PredictIn(BaseModel):
    symbol: str

@app.get("/health")
def health():
    return {"ok": True, "service": "ml", "time": pd.Timestamp.utcnow().isoformat()}

@app.post("/predict-stock")
def predict_stock(inp: PredictIn):
    df = fetch_real_ohlcv(inp.symbol)
    is_real = True
    if df is None or len(df) < 50:
        df = make_mock_ohlcv(200)
        is_real = False
    
    df = compute_indicators(df)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()
    features = df[["close", "MA20", "MA50", "RSI", "MACD", "MACD_SIGNAL", "ATR", "VOL_CHANGE", "BB_UPPER", "BB_LOWER", "BB_MIDDLE", "STOCH_K", "STOCH_D"]]
    target = df["target"]
    model = train_or_dummy(features.iloc[:-1], target.iloc[:-1])
    x_last = features.iloc[[-1]]
    if model is not None:
        proba_up = float(model.predict_proba(x_last)[0][1])
    else:
        proba_up = float(heuristic_proba(x_last.iloc[0]))
    pred = "UP" if proba_up >= 0.5 else "DOWN"
    indicators = {
        "MA20": float(df["MA20"].iloc[-1]),
        "MA50": float(df["MA50"].iloc[-1]),
        "RSI": float(df["RSI"].iloc[-1]),
        "MACD": float(df["MACD"].iloc[-1]),
        "ATR": float(df["ATR"].iloc[-1]),
        "VOL_CHANGE": float(df["VOL_CHANGE"].iloc[-1]),
        "BB_UPPER": float(df["BB_UPPER"].iloc[-1]),
        "BB_LOWER": float(df["BB_LOWER"].iloc[-1]),
        "STOCH_K": float(df["STOCH_K"].iloc[-1]),
    }
    return {
        "prediction": pred,
        "confidence": round(proba_up if pred == "UP" else 1 - proba_up, 4),
        "indicators": indicators,
        "is_real_data": is_real,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
