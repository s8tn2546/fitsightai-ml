import pandas as pd

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Simple Moving Averages
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA50"] = df["close"].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # ATR (Average True Range) - Simplified
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()
    
    # Bollinger Bands
    df["BB_MIDDLE"] = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["BB_UPPER"] = df["BB_MIDDLE"] + (std * 2)
    df["BB_LOWER"] = df["BB_MIDDLE"] - (std * 2)
    
    # Stochastic Oscillator
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["STOCH_K"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["STOCH_D"] = df["STOCH_K"].rolling(window=3).mean()

    df["VOL_CHANGE"] = df["volume"].pct_change().fillna(0)
    df = df.bfill().ffill()
    return df
