# -*- coding: utf-8 -*-
"""
🤖 ML Signal Service API
Dual-endpoint API for live predictions and historical signals
- Live Signal: For dashboard predictions
- Historical Signals: For backtesting engine
- Market Data: Serves real data via YFinance (acting as Supabase proxy)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from ml.predictor import MLEngine
from contracts.schema import StockData, MLSignal

import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import uvicorn

app = FastAPI(
    title="ML Signal Service",
    description="AI-Powered Stock Prediction API with Live & Historical Endpoints",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================
# ROOT ENDPOINT
# =================================================
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the ML Signal Service API",
        "status": "online",
        "endpoints": {
            "live_signal": "/api/v1/ml/signal/live [POST]",
            "historical_signals": "/api/v1/ml/signal/historical [POST]",
            "health": "/health [GET]"
        }
    }

# =================================================
# LOAD MODELS
# =================================================
MODEL_PATH = "C:/infosys1/AI-powered-stock-and-ETF-trading-platform/ml/models/rf_model.pkl"

# Update model loading to use the absolute path
try:
    rf_model = joblib.load(MODEL_PATH)
    xgb_model = joblib.load(MODEL_PATH.replace("rf_model.pkl", "xgb_model.pkl"))
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Models not loaded: {e}")
    rf_model = None
    xgb_model = None

FEATURES = ["Daily_Return", "Volatility", "SMA_ratio", "EMA_ratio", "MACD"]

# =================================================
# FEATURE ENGINEERING
# =================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix yfinance multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    df["Daily_Return"] = close.pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(14).std()

    df["SMA20"] = close.rolling(20).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    df["SMA_ratio"] = close / df["SMA20"]
    df["EMA_ratio"] = close / df["EMA20"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    df.dropna(inplace=True)
    return df

# =================================================
# HELPER: YFINANCE DATA PROVIDER
# =================================================
def fetch_real_market_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    """
    Fetch real data using yfinance to populate API responses
    This replaces Supabase if credentials are missing
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return []
        
        # MEANINGFUL FIX: Flatten MultiIndex columns (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Format columns to match typical API/DB response
        data = []
        for _, row in df.iterrows():
            record = {
                "ticker": ticker.upper(),
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            }
            # Calculate RSI if enough data
            data.append(record)
            
        return data  # Returns oldest to newest by default
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return []

def calculate_rsi(prices: pd.Series, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =================================================
# REQUEST SCHEMA
# =================================================
class TickerRequest(BaseModel):
    ticker: str

# =================================================
# 1️⃣ LIVE SIGNAL API (Dashboard)
# =================================================
@app.post("/api/v1/ml/signal/live")
def get_live_signal(request: TickerRequest):
    """
    Used by Dashboard → Predict Signal button
    Returns only today's signal using the unified MLEngine
    """
    ticker = request.ticker.upper()
    try:
        # 1. Fetch data and calculate indicators (simplified)
        # We need to construct a StockData object. 
        # Since scripts/run_api.py already has fetch_stock_data, 
        # we can just use that or reimplement it here briefly.
        
        # For consistency, let's use the engine's predictor which now handles RSS too.
        # But first we need StockData. I'll use a helper or just yfinance.
        
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Calculate standard indicators for StockData
        # (Simplified implementation of indicator calculation)
        close = df['Close']
        df['rsi'] = 100 - (100 / (1 + (close.diff().where(close.diff() > 0, 0).rolling(14).mean() / -close.diff().where(close.diff() < 0, 0).rolling(14).mean())))
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['ema_12'] = close.ewm(span=12).mean()
        df['ema_26'] = close.ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        latest = df.iloc[-1]
        subset = df.tail(100)
        
        stock_data = StockData(
            symbol=ticker,
            current_price=float(latest['Close']),
            price_change=float(latest['Close'] - df.iloc[-2]['Close']) if len(df) > 1 else 0.0,
            price_change_pct=float((latest['Close'] - df.iloc[-2]['Close'])/df.iloc[-2]['Close']*100) if len(df) > 1 else 0.0,
            last_updated=datetime.now(),
            market_status="Open",
            dates=subset.index.strftime('%Y-%m-%d').tolist(),
            opens=subset['Open'].tolist(),
            highs=subset['High'].tolist(),
            lows=subset['Low'].tolist(),
            closes=subset['Close'].tolist(),
            volumes=subset['Volume'].tolist(),
            rsi=subset['rsi'].tolist(),
            sma_20=subset['sma_20'].tolist(),
            sma_50=subset['sma_50'].tolist(),
            ema_12=subset['ema_12'].tolist(),
            ema_26=subset['ema_26'].tolist(),
            macd=subset['macd'].tolist(),
            macd_signal=subset['macd_signal'].tolist(),
            macd_hist=subset['macd_hist'].tolist()
        )
        
        # 2. Call the Engine
        engine = MLEngine()
        prediction = engine.predict(stock_data, skip_api=True)
        
        # 3. Format response to match legacy expectations if needed, 
        # but MLSignal should be mostly compatible.
        result = prediction.dict()
        # Add historical-compatible fields if necessary
        result["signal"] = result["action"]
        result["expected_return"] = 0.0 # Heuristics don't provide easy return %
        
        return result
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================================================
# 2️⃣ HISTORICAL SIGNALS API (Backtesting)
# =================================================
@app.post("/api/v1/ml/signal/historical")
def get_historical_signals(request: TickerRequest):
    """
    Used by Backtesting Engine
    Returns 5 years OHLCV + ML signals
    """
    if not rf_model or not xgb_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    ticker = request.ticker.upper()

    try:
        df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical data")

        df = create_features(df)
        if df.empty:
            raise HTTPException(status_code=404, detail="Not enough data for features")

        X = df[FEATURES]
        rf_preds = rf_model.predict(X)
        xgb_preds = xgb_model.predict(X)
        avg_preds = (rf_preds + xgb_preds) / 2

        df["Signal"] = np.where(avg_preds > 0, 1, -1)

        # Convert to JSON-safe structure
        records = []
        for date, row in df.iterrows():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "signal": int(row["Signal"])
            })

        return {
            "ticker": ticker,
            "rows": records,
            "total_rows": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================================================
# HEALTH CHECK
# =================================================
@app.get("/health")
def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": rf_model is not None and xgb_model is not None,
        "version": "2.1.0"
    }

# =================================================
# DATA PIPELINE CONTROL
# =================================================
@app.post("/run-pipeline")
def run_pipeline():
    """Trigger data pipeline execution"""
    # In a real app, this would start the Airflow/Prefect job
    # Here we can perhaps trigger a background update of cached data
    return {
        "status": "pipeline_started",
        "timestamp": datetime.now().isoformat(),
        "message": "Data pipeline execution triggered (Simulation)"
    }

# =================================================
# STOCK DATA ENDPOINTS (Implementing Real Logic)
# =================================================

@app.get("/supabase/recent/{ticker}")
def get_recent_data(ticker: str, days: int = Query(30, description="Number of days")):
    """Get recent stock data for a ticker with REAL data"""
    try:
        # Fetch slightly more to ensure coverage
        data = fetch_real_market_data(ticker, period=f"{days+10}d")
        
        # Filter strictly for days requested (approx)
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        filtered_data = [d for d in data if d["date"] >= cutoff_date]
        
        # Reverse to be descending (latest first) as expected by frontend
        filtered_data.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "ticker": ticker.upper(),
            "days": days,
            "count": len(filtered_data),
            "data": filtered_data
        }
    except Exception as e:
        return {"data": [], "error": str(e)}

@app.get("/supabase/ticker/{ticker}")
def get_ticker_data(
    ticker: str,
    start_date: str = Query("2024-01-01", description="Start date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Max records")
):
    """Get ticker data with date range and limit"""
    try:
        # Convert start_date to period approx or just fetch max and filter
        data = fetch_real_market_data(ticker, period="2y") # Safe default
        
        filtered_data = [d for d in data if d["date"] >= start_date]
        filtered_data.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "ticker": ticker.upper(),
            "start_date": start_date,
            "limit": limit,
            "count": min(len(filtered_data), limit),
            "data": filtered_data[:limit]
        }
    except Exception as e:
        return {"data": [], "error": str(e)}

# --- Market Overview ---
@app.get("/supabase/latest")
def get_latest_market(limit: int = Query(10, description="Number of latest records")):
    """Get latest market data for top accessible tickers"""
    # Simulate a "market scan" by fetching a basket of popular stocks
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "SPY"]
    
    market_data = []
    try:
        # Fetch in bulk if possible, or loop (loop is safer for yfinance reliability here)
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        for t in TICKERS[:limit]:
            # Just get 1 day
            d = fetch_real_market_data(t, period="5d") # 5d to handle weekends
            if d:
                latest = d[-1] # Newest is last in fetch_real_market_data default sort
                latest["ticker"] = t # Ensure ticker is present
                market_data.append(latest)
                
        return {
            "limit": limit,
            "data": market_data,
            "count": len(market_data)
        }
    except Exception as e:
        return {"data": [], "error": str(e)}

@app.get("/supabase/top-performers")
def get_top_performers(top_n: int = Query(10, description="Top N performers")):
    """Get top performing stocks from our basket"""
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"]
    
    performers = []
    try:
        for t in TICKERS:
            d = fetch_real_market_data(t, period="1mo")
            if d and len(d) > 1:
                start_price = d[0]["close"]
                end_price = d[-1]["close"]
                if start_price > 0:
                    change = ((end_price - start_price) / start_price) * 100
                    performers.append({
                        "ticker": t,
                        "change_pct": round(change, 2),
                        "start_price": start_price,
                        "end_price": end_price
                    })
        
        performers.sort(key=lambda x: x["change_pct"], reverse=True)
        
        return {
            "top_n": top_n,
            "performers": performers[:top_n]
        }
    except Exception as e:
        return {"performers": [], "error": str(e)}

# --- Analysis & Filtering ---
@app.get("/supabase/stats/{ticker}")
def get_ticker_stats(
    ticker: str,
    start_date: str = Query("2024-01-01", description="Start date")
):
    """Get statistical analysis for a ticker"""
    try:
        data = fetch_real_market_data(ticker, period="1y")
        filtered_data = [d for d in data if d["date"] >= start_date]
        
        if not filtered_data:
            return {"stats": {}}
            
        closes = [d["close"] for d in filtered_data]
        volumes = [d["volume"] for d in filtered_data]
        
        stats = {
            "ticker": ticker.upper(),
            "period_start": start_date,
            "price_high": max(closes),
            "price_low": min(closes),
            "price_avg": sum(closes) / len(closes),
            "volume_avg": sum(volumes) / len(volumes),
            "data_points": len(closes)
        }
        
        return {"stats": stats}
    except Exception as e:
        return {"stats": {}, "error": str(e)}

@app.get("/supabase/rsi-search")
def search_by_rsi(
    min_rsi: float = Query(0, description="Minimum RSI"),
    max_rsi: float = Query(30, description="Maximum RSI")
):
    """Search stocks by RSI range"""
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    results = []
    
    try:
        for t in TICKERS:
            # Need about 14+ days for RSI
            d = fetch_real_market_data(t, period="2mo") 
            if len(d) > 15:
                # Convert to df for easy calc
                df = pd.DataFrame(d)
                df.set_index("date", inplace=True)
                df["rsi"] = calculate_rsi(df["close"])
                
                current_rsi = df["rsi"].iloc[-1]
                
                if min_rsi <= current_rsi <= max_rsi:
                    latest = d[-1]
                    latest["rsi"] = current_rsi
                    results.append(latest)
                    
        return {
            "min_rsi": min_rsi,
            "max_rsi": max_rsi,
            "results": results
        }
    except Exception as e:
         return {"results": [], "error": str(e)}

# Update the absolute path to the model file
MODEL_PATH = "C:/infosys1/AI-powered-stock-and-ETF-trading-platform/ml/models/rf_model.pkl"
# Update the code to use MODEL_PATH when loading the model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
