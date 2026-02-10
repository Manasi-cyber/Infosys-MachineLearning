# -*- coding: utf-8 -*-
import sys
from pathlib import Path
# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import random
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
try:
    import xgboost as xgb
except ImportError:
    pass
from contracts.schema import StockData, MLSignal
import requests 
from typing import List
import feedparser

class MLEngine:
    # Model Metadata
    MODEL_TYPE = "Ensemble ML (Random Forest + XGBoost)"
    MODEL_VERSION = "v3.0.0"
    LAST_TRAINED = "2025-01-08"
    PREDICTION_FREQUENCY = "Real-time (on-demand)"
    
    def __init__(self):
        """Load trained models on initialization"""
        self.models_loaded = False
        self.rf_model = None
        self.xgb_model = None
        self._load_models()
    
    def _load_models(self):
        """Load trained Random Forest and XGBoost models"""
        try:
            # Models are in ml/models/ relative to project root
            # predictor.py is in ml/ folder
            models_path = Path(__file__).parent / "models"
            rf_path = models_path / "rf_model.pkl"
            xgb_path = models_path / "xgb_model.pkl"
            
            if rf_path.exists() and xgb_path.exists():
                self.rf_model = joblib.load(rf_path)
                self.xgb_model = joblib.load(xgb_path)
                self.models_loaded = True
                print("✅ [MLEngine] Loaded trained ML models (RF + XGBoost)")
            else:
                print(f"⚠️ [MLEngine] Models not found at {models_path}, using heuristic fallback")
                # List contents to help debug
                if models_path.exists():
                    print(f"Contents of {models_path}: {[f.name for f in models_path.iterdir()]}")
                else:
                    print(f"Directory {models_path} does not exist")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}, using heuristic fallback")
    
    def _create_features_from_stock_data(self, data: StockData) -> pd.DataFrame:
        """Create features from StockData for model prediction"""
        try:
            # Calculate features from stock data
            closes = np.array(data.closes)
            
            # Daily return (latest)
            daily_return = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
            
            # Volatility (14-day rolling std of returns)
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns[-14:]) if len(returns) >= 14 else np.std(returns)
            
            # SMA ratio (current price / SMA20)
            sma_20 = data.sma_20[-1] if data.sma_20 and len(data.sma_20) > 0 else closes[-1]
            sma_ratio = closes[-1] / sma_20 if sma_20 > 0 else 1.0
            
            # EMA ratio (current price / EMA20)
            # Calculate EMA20 if not available
            if len(closes) >= 20:
                ema_20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
            else:
                ema_20 = closes[-1]
            ema_ratio = closes[-1] / ema_20 if ema_20 > 0 else 1.0
            
            # MACD
            macd = data.macd[-1] if data.macd and len(data.macd) > 0 else 0
            
            # Create feature dataframe
            features = pd.DataFrame({
                'Daily_Return': [daily_return],
                'Volatility': [volatility],
                'SMA_ratio': [sma_ratio],
                'EMA_ratio': [ema_ratio],
                'MACD': [macd]
            })
            
            return features
        except Exception as e:
            print(f"Error creating features: {e}")
            return None
    def _fetch_rss_news(self, ticker: str) -> List[str]:
        """Fetch real-time news headlines from Google News RSS."""
        try:
            rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            headlines = [entry.title for entry in feed.entries[:5]]
            return headlines
        except Exception as e:
            print(f"RSS Fetch Error: {e}")
            return []

    def _generate_ai_analysis(self, symbol: str, action: str, confidence: float, factors: list, news: list = []) -> tuple:
        """Sends technical data to Ollama to get a quirky Coach-style summary and market mood."""
        # Simple fallback
        fallback_reasoning = f"Analysis: {symbol} is a {action} ({confidence:.1f}%) based on {', '.join(factors[:2])}."
        fallback_mood = "Market conditions appear neutral."
        
        try:
            news_text = "\n".join([f"- {n}" for n in news[:5]]) if news else "No specific news available."
            
            prompt = (
                f"You are a quirky, witty, and wise AI Trading Coach. Your personality is a mix of high-energy wall street trader and a Zen master.\n\n"
                f"Market Data for {symbol}:\n"
                f"- Signal: {action} ({confidence:.1f}% confidence)\n"
                f"- Technical Factors: {', '.join(factors)}\n"
                f"- Recent News Headlines:\n{news_text}\n\n"
                f"Task:\n"
                f"1. Give a 'Market Mood' (one short, punchy sentence) that captures the vibe of the news and technicals.\n"
                f"2. Provide a 'Coach Explanation' (1-2 paragraphs). Start with something catchy like '{action}ing {symbol} today...' or 'Listen up, {symbol} is showing some heat...'. "
                f"Use a quirky, slightly irreverent but professional tone. Mention the technicals and news context in a way that feels 'inside baseball'. Make it sound like you're giving a secret tip while being realistic about risks.\n\n"
                f"Format your response exactly as:\n"
                f"MOOD: [Market Mood]\n"
                f"EXPLANATION: [Coach Explanation]"
            )

            res = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=500
            )
            
            text = res.json().get("response", "").strip()
            
            if "MOOD:" in text and "EXPLANATION:" in text:
                parts = text.split("EXPLANATION:")
                mood_part = parts[0].replace("MOOD:", "").strip()
                exp_part = parts[1].strip()
                return exp_part, mood_part
            else:
                return text, fallback_mood

        except Exception as e:
            print(f"GenAI Error: {e}")
            return fallback_reasoning, fallback_mood
    def _predict_with_models(self, data: StockData) -> tuple:
        """Use trained models to predict next day return"""
        try:
            features = self._create_features_from_stock_data(data)
            if features is None:
                return None, None
            
            # Get predictions from both models
            rf_pred = float(self.rf_model.predict(features)[0])
            xgb_pred = float(self.xgb_model.predict(features)[0])
            
            # Average the predictions
            avg_pred = (rf_pred + xgb_pred) / 2
            
            # Calculate predicted price
            current_price = data.current_price
            predicted_price = current_price * (1 + avg_pred)
            expected_return_pct = avg_pred * 100
            
            # Determine confidence based on prediction magnitude (cap at 95%)
            confidence = min(70 + min(abs(expected_return_pct) * 3, 25), 95.0)
            
            return predicted_price, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None
    
    def predict(self, data: StockData, skip_api: bool = False, news: list = []) -> MLSignal:
        """
        Generates a trading signal using (in order):
        1. Live API (Port 8000) - To match Alerts service exactly
        2. Local Trained Models (if loaded)
        3. Technical Heuristics (fallback)
        """
        import requests
        
        # 0️⃣ Fetch News if empty
        if not news:
            print(f"📰 Fetching RSS news for {data.symbol}...")
            news = self._fetch_rss_news(data.symbol)
        
        # 1️⃣ Try Live API (Sync with Alerts)
        try:
            if skip_api:
                raise Exception("Local prediction requested")
            url = "http://localhost:8000/api/v1/ml/signal/live"
            # Extract ticker. data.symbol is cleaner.
            ticker = data.symbol
            
            # fast timeout to not block UI if API is down
            response = requests.post(url, json={"ticker": ticker}, timeout=500.0)
            
            if response.status_code == 200:
                api_data = response.json()
                # API returns: {signal, confidence, expected_return ...}
                
                action = api_data.get("signal", "HOLD")
                confidence = float(api_data.get("confidence", 0.0))
                
                # Determine Confidence Level
                if confidence >= 85: conf_level = "Very High"
                elif confidence >= 70: conf_level = "High"
                elif confidence >= 55: conf_level = "Medium"
                else: conf_level = "Low"
                
                signal_value = 1 if action == "BUY" else (-1 if action == "SELL" else 0)
                
                # Generate explanations based on API data
                exp_return = api_data.get("expected_return", 0.0)
                
                # Handling Old API vs New API (Decimal vs Percent)
                # If absolute value is < 0.2, it MIGHT be decimal (5% -> 0.05).
                # But ML returns can be small. Since we updated API to send %, we trust it roughly.
                # However, if it's extremely small (like 0.001), it might display as 0.00%
                # Let's ensure non-zero display logic isn't the problem, but source is.
                
                # Use reasoning from API if available
                api_reasoning = api_data.get("reasoning", "")
                market_mood = api_data.get("market_mood", "Neutral")
                top_news = api_data.get("top_news", [])
                
                # Use factors from API if available
                api_factors = api_data.get("key_factors", [])
                                
                # Use feature importance from API if available
                api_feat_imp = api_data.get("feature_importance", {})

                # Unified Logic: Use API factors if valid, else calculate local fallback
                factors_to_use = api_factors if api_factors and len(api_factors) > 0 else []

                # If no API factors, calculate local technical reasons
                if not factors_to_use:
                    local_reasons = []
                    # RSI
                    rsi = data.rsi[-1] if data.rsi and len(data.rsi) > 0 else 50
                    if rsi < 30: local_reasons.append(f"RSI is Oversold ({rsi:.1f})")
                    elif rsi > 70: local_reasons.append(f"RSI is Overbought ({rsi:.1f})")
                    else: local_reasons.append(f"RSI is Neutral ({rsi:.1f})")
                    
                    # SMA Trend
                    sma_50 = data.sma_50[-1] if data.sma_50 else data.current_price
                    if data.current_price > sma_50: local_reasons.append("Price in Uptrend (Above SMA 50)")
                    else: local_reasons.append("Price in Downtrend (Below SMA 50)")
                    
                    # MACD
                    macd = data.macd[-1] if data.macd else 0
                    macd_sig = data.macd_signal[-1] if data.macd_signal else 0
                    if macd > macd_sig: local_reasons.append("MACD Bullish Crossover")
                    else: local_reasons.append("MACD Bearish Momentum")
                    
                    # Add ML context
                    local_reasons.append(f"ML Model Expects {exp_return:.2f}% Return")
                    
                    factors_to_use = local_reasons

                final_reasoning = api_reasoning
                api_factors = factors_to_use

                # Calculate dynamic feature importance for UI (Explainability) if missing
                if not api_feat_imp:
                    # Fallback Logic
                    local_importance = {}
                    local_importance["AI Model Confidence"] = 40.0
                    total_imp = 40.0
                    api_feat_imp = {"AI Model Confidence": 100.0}

                print("🤖 Attempting to generate AI Reasoning...")
                # If API didn't provide rich reasoning (or if we want to augment it locally)
                # But typically API should provide it.
                # However, your prompt update is in this file (local MLEngine), so we MUST call _generate_ai_analysis HERE 
                # if we want the new prompt format, unless API uses THIS file too.
                # API uses THIS file. So API calls _generate_ai_analysis.
                # If API returns detailed reasoning, we can just use it.
                # But let's call it locally if API reasoning is empty or simple.
                
                if not api_reasoning or len(api_reasoning) < 20: 
                     ai_reasoning, mood_from_ai = self._generate_ai_analysis(
                        symbol=data.symbol,
                        action=action,
                        confidence=confidence,
                        factors=factors_to_use,
                        news=top_news
                    )
                     market_mood = mood_from_ai
                else:
                     ai_reasoning = api_reasoning

                return MLSignal(
                    action=action,
                    signal_value=signal_value,
                    timestamp=datetime.now(),
                    prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    confidence=confidence,
                    confidence_level=conf_level,
                    reasoning=ai_reasoning,
                    key_factors=api_factors,
                    feature_importance=api_feat_imp,
                    prediction_frequency=MLEngine.PREDICTION_FREQUENCY,
                    model_type=MLEngine.MODEL_TYPE + " (API)",
                    model_version=MLEngine.MODEL_VERSION,
                    last_trained=MLEngine.LAST_TRAINED,
                    market_mood=market_mood,
                    top_news=top_news
                )
        except Exception as e:
            # Silently fail API and use local logic
            print(f"⚠️ [MLEngine] API Signal Error: {e}")
            pass

        # 2️⃣ Try using trained models locally
        if self.models_loaded:
            predicted_price, ml_confidence = self._predict_with_models(data)
            if predicted_price is not None:
                # Determine signal from ML prediction
                current_price = data.current_price
                expected_return = ((predicted_price / current_price) - 1) * 100
                
                # Combine ML prediction with Technical Context for stronger signals
                # Many ML models predict very conservative returns (e.g., 0.05%), 
                # so we combine this with technicals to force a decision.
                
                technical_score = 0
                rsi = data.rsi[-1] if data.rsi and len(data.rsi) > 0 else 50
                if rsi < 40: technical_score += 1  # Oversold bias
                if rsi > 60: technical_score -= 1  # Overbought bias
                
                sma_50 = data.sma_50[-1] if data.sma_50 else data.current_price
                if data.current_price > sma_50: technical_score += 0.5 # Uptrend
                
                # Decision Logic
                # If ML predicts even slightly positive (>0.02%) OR (non-negative AND technicals bullish)
                if expected_return > 0.02 or (expected_return > -0.05 and technical_score > 0):
                    action = "BUY"
                    # Boost confidence if technicals agree
                    confidence = min(ml_confidence * (1.1 if technical_score > 0 else 1.0), 96.0)
                    reasons = [
                        f"ML predicts positive outlook ({expected_return:.2f}%)",
                        f"Target: ₹{predicted_price:.2f}",
                        "Technical factors support bullish trend" if technical_score > 0 else "Bullish signal despite neutral technicals"
                    ]
                elif expected_return < -0.02 or (expected_return < 0.05 and technical_score < 0):
                    action = "SELL"
                    confidence = min(ml_confidence * (1.1 if technical_score < 0 else 1.0), 96.0)
                    reasons = [
                        f"ML predicts negative/weak outlook ({expected_return:.2f}%)",
                        f"Target: ₹{predicted_price:.2f}",
                        "Technical factors suggest weakness"
                    ]
                else:
                    action = "HOLD"
                    confidence = ml_confidence
                    reasons = [
                        f"Flat prediction ({expected_return:.2f}%)",
                        "No strong directional signal found",
                        "Wait for clearer trend"
                    ]
                
                # Add technical indicator context
                rsi = data.rsi[-1] if data.rsi and len(data.rsi) > 0 else 50
                macd = data.macd[-1] if data.macd and len(data.macd) > 0 else 0
                macd_signal = data.macd_signal[-1] if data.macd_signal and len(data.macd_signal) > 0 else 0
                
                if rsi < 30:
                    reasons.append(f"RSI is oversold ({rsi:.1f})")
                elif rsi > 70:
                    reasons.append(f"RSI is overbought ({rsi:.1f})")
                
                if macd > macd_signal:
                    reasons.append("MACD is bullish")
                else:
                    reasons.append("MACD is bearish")
                
                confidence_level = "Very High" if confidence >= 85 else "High" if confidence >= 70 else "Medium" if confidence >= 55 else "Low"
                
                signal_value = 1 if action == "BUY" else (-1 if action == "SELL" else 0)
                

                ai_reasoning, market_mood = self._generate_ai_analysis(
                    symbol=data.symbol,
                    action=action,
                    confidence=confidence,
                    factors=reasons,
                    news=news
                )

                return MLSignal(
                    action=action,
                    signal_value=signal_value,
                    timestamp=datetime.now(),
                    prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    confidence=confidence,
                    confidence_level=confidence_level,
                    reasoning=ai_reasoning,
                    key_factors=reasons[:5],
                    feature_importance={
                        "ML Prediction": 85.0,
                        "RSI": 75.0,
                        "MACD": 70.0,
                        "Price Momentum": 65.0
                    },
                    prediction_frequency="Real-time",
                    model_type=self.MODEL_TYPE,
                    model_version=self.MODEL_VERSION,
                    last_trained=self.LAST_TRAINED,
                    market_mood=market_mood,
                    top_news=news
                )
        
        # Fallback to technical indicator heuristics
        rsi = data.rsi[-1] if data.rsi and len(data.rsi) > 0 else 50
        macd = data.macd[-1] if data.macd and len(data.macd) > 0 else 0
        macd_signal = data.macd_signal[-1] if data.macd_signal and len(data.macd_signal) > 0 else 0
        macd_hist = data.macd_hist[-1] if data.macd_hist and len(data.macd_hist) > 0 else 0
        price = data.current_price
        sma_20 = data.sma_20[-1] if data.sma_20 and len(data.sma_20) > 0 else price
        sma_50 = data.sma_50[-1] if data.sma_50 and len(data.sma_50) > 0 else price
        
        score = 0
        reasons = []
        
        # RSI Logic
        if rsi < 30:
            score += 2
            reasons.append(f"RSI is oversold ({rsi:.1f}), suggesting a potential rebound.")
        elif rsi > 70:
            score -= 2
            reasons.append(f"RSI is overbought ({rsi:.1f}), suggesting a potential pullback.")
        else:
            reasons.append(f"RSI is neutral ({rsi:.1f}).")
            
        # MACD Logic
        if macd > macd_signal:
            score += 1
            reasons.append("MACD line is above the signal line (Bullish).")
        else:
            score -= 1
            reasons.append("MACD line is below the signal line (Bearish).")
            
        # Trend Logic
        if price > sma_50:
            score += 1
            reasons.append("Price is above the 50-day SMA (Uptrend).")
        else:
            score -= 1
            reasons.append("Price is below the 50-day SMA (Downtrend).")
        
        # MACD Histogram momentum
        if abs(macd_hist) > 0:
            if macd_hist > 0:
                score += 0.5
                reasons.append("MACD histogram shows increasing bullish momentum.")
            else:
                score -= 0.5
                reasons.append("MACD histogram shows increasing bearish momentum.")
        
        # Golden/Death Cross detection
        if len(data.sma_20) > 1 and len(data.sma_50) > 1:
            prev_20 = data.sma_20[-2]
            prev_50 = data.sma_50[-2]
            if prev_20 <= prev_50 and sma_20 > sma_50:
                score += 2
                reasons.append("🌟 Golden Cross detected (SMA 20 crossed above SMA 50).")
            elif prev_20 >= prev_50 and sma_20 < sma_50:
                score -= 2
                reasons.append("💀 Death Cross detected (SMA 20 crossed below SMA 50).")
            
        # Determine Action
        if score >= 2:
            action = "BUY"
            confidence = 75 + (score * 5)
        elif score <= -2:
            action = "SELL"
            confidence = 75 + (abs(score) * 5)
        else:
            action = "HOLD"
            confidence = 50.0
            
        # Cap confidence
        confidence = min(98.5, max(50.0, confidence))
        
        # Confidence Level (categorical)
        if confidence >= 85:
            confidence_level = "Very High"
        elif confidence >= 70:
            confidence_level = "High"
        elif confidence >= 55:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # 2️⃣ Numerical Signal Encoding
        signal_value = 1 if action == "BUY" else (-1 if action == "SELL" else 0)
        
        # 4️⃣ Generate quirky GenAI Explanation
        ai_reasoning, market_mood = self._generate_ai_analysis(
            symbol=data.symbol,
            action=action,
            confidence=confidence,
            factors=reasons,
            news=news
        )
        
        # 6️⃣ Calculate Feature Importance (weighted by contribution to score)
        feature_importance = {}
        
        # RSI contribution
        if rsi < 30:
            feature_importance["RSI (Oversold)"] = 2.0 / max(abs(score), 1) * 100
        elif rsi > 70:
            feature_importance["RSI (Overbought)"] = 2.0 / max(abs(score), 1) * 100
        else:
            feature_importance["RSI (Neutral)"] = 0.5 / max(abs(score), 1) * 100
        
        # MACD contribution
        if macd > macd_signal:
            feature_importance["MACD (Bullish)"] = 1.0 / max(abs(score), 1) * 100
        else:
            feature_importance["MACD (Bearish)"] = 1.0 / max(abs(score), 1) * 100
        
        # Trend contribution
        if price > sma_50:
            feature_importance["Trend (Uptrend)"] = 1.0 / max(abs(score), 1) * 100
        else:
            feature_importance["Trend (Downtrend)"] = 1.0 / max(abs(score), 1) * 100
        
        # Volume analysis
        if data.volumes and len(data.volumes) > 1:
            vol_change = ((data.volumes[-1] - data.volumes[-2]) / data.volumes[-2]) * 100
            feature_importance["Volume"] = min(abs(vol_change) / 10, 15.0)
        else:
            feature_importance["Volume"] = 5.0
        
        # Cross detection contribution
        if len(data.sma_20) > 1 and len(data.sma_50) > 1:
            prev_20 = data.sma_20[-2]
            prev_50 = data.sma_50[-2]
            if prev_20 <= prev_50 and sma_20 > sma_50:
                feature_importance["Golden Cross"] = 2.0 / max(abs(score), 1) * 100
            elif prev_20 >= prev_50 and sma_20 < sma_50:
                feature_importance["Death Cross"] = 2.0 / max(abs(score), 1) * 100
        
        # Normalize feature importance to sum to 100%
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: (v / total_importance) * 100 for k, v in feature_importance.items()}
        else:
             feature_importance = {
                "Market Volatility": 50.0,
                "Trend Analysis": 50.0
            }
        
        # 3️⃣ Timestamp alignment
        current_timestamp = datetime.now()
        prediction_date = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        return MLSignal(
            # 1️⃣ Core Output
            action=action,
            signal_value=signal_value,
            timestamp=current_timestamp,
            prediction_date=prediction_date,
            reasoning=ai_reasoning,
            confidence=confidence,
            confidence_level=confidence_level,
            key_factors=[r.split('(')[0].strip() for r in reasons[:5]],
            feature_importance=feature_importance,
            prediction_frequency=MLEngine.PREDICTION_FREQUENCY,
            model_type=MLEngine.MODEL_TYPE,
            model_version=MLEngine.MODEL_VERSION,
            last_trained=MLEngine.LAST_TRAINED,
            market_mood=market_mood,
            top_news=news
        )