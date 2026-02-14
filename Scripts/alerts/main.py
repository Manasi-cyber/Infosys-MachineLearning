import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import random # Used for simulation only
import requests # Used for real ML API calls

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from tzlocal import get_localzone

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 1. CONFIGURATION (STRICTLY FROM ENV)
# ==========================================
# Load from .env or system environment variables
EMAIL_SENDER = os.getenv("SMTP_USER", "saikarthikreddykuppireddy@gmail.com")
EMAIL_PASSWORD = os.getenv("SMTP_PASSWORD", "fmfg dncy yuku fvqt")

# ==========================================
# 2. INITIALIZE APP & SCHEDULER
# ==========================================
app = FastAPI(title="Alerts Middleware API", version="2.0")

# Initialize the scheduler to run jobs in the background
# ⚠️ Forced Timezone: Asia/Kolkata (IST)
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

if not scheduler.running:
    scheduler.start()
    print("✅ [Scheduler] BackgroundScheduler started (Timezone: Asia/Kolkata)")

# ==========================================
# 3. ML API INTEGRATION (The Input Source)
# ==========================================
def fetch_ml_signal(ticker: str):
    """
    DEBUG EXPLANATION:
    - This function acts as a bridge to the 'Signals API' running on port 8000.
    - It sends a POST request with the 'ticker' symbol.
    - If successful (HTTP 200), it returns the raw JSON signal data (Buy/Sell/Hold, Confidence, Price).
    - If it fails, it logs the error and returns None.
    """
    print(f"DEBUG: [fetch_ml_signal] Fetching live signal for {ticker} from Port 8000...")
    # FIXED: Pointing to Port 8000 where ML API runs
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/api/v1/ml/signal/live",
            json={"ticker": ticker},
            timeout=500
        )
        print(f"DEBUG: [fetch_ml_signal] Response Code: {resp.status_code}")
        if resp.status_code != 200:
             print(f"⚠️ ML API Error: {resp.text}")
             return None
        data = resp.json()
        print(f"DEBUG: [fetch_ml_signal] Received data: {data}")
        return data
    except Exception as e:
        print(f"⚠️ ML API Connection Failed: {e}")
        return None

def fetch_backtest_result(ticker: str):
    """
    DEBUG EXPLANATION:
    - Connects to the 'Backtesting API' on port 8002.
    - This is crucial because it calculates the 'Confidence Score' based on historical performance.
    - If this service is down, we might fallback to live confidence or zero.
    """
    print(f"DEBUG: [fetch_backtest_result] Fetching backtest metrics for {ticker} from Port 8002...")
    # FIXED: Pointing to Port 8002 where Backtesting API runs
    try:
        resp = requests.post(
            "http://127.0.0.1:8002/api/v1/backtest/run",
            json={"ticker": ticker},
            timeout=5
        )
        print(f"DEBUG: [fetch_backtest_result] Response Code: {resp.status_code}")
        if resp.status_code != 200:
            print(f"⚠️ Backtest API Error: {resp.text}")
            return None
        data = resp.json()
        print(f"DEBUG: [fetch_backtest_result] Received Confidence Score: {data.get('confidence_score', 'N/A')}")
        return data
    except Exception as e:
        print(f"⚠️ Backtest API Connection Failed: {e}")
        return None

def get_usd_inr_rate():
    """
    Fetches the latest USD to INR exchange rate from the ML API.
    Used to display US stock prices in Indian Rupees for the user.
    """
    try:
        # Ticker for USD/INR on YFinance is 'USDINR=X'
        resp = requests.get("http://127.0.0.1:8000/supabase/ticker/USDINR=X?limit=1", timeout=5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                rate = float(data[0]["close"])
                print(f"DEBUG: [Exchange Rate] 1 USD = {rate:.2f} INR")
                return rate
    except Exception as e:
        print(f"⚠️ [Exchange Rate] Failed to fetch live rate, using manual fallback (84.0): {e}")
    return 84.0

# ==========================================
# 4. EMAIL ALERT LOGIC (The Output)
# ==========================================
def send_email_alert(user_email: str, ticker: str, signal_data: dict):
    """
    Sends a formatted HTML email to the user using Gmail SMTP.
    Matches Dashboard AI Signals aesthetics.
    """
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract Data
        signal_text = signal_data.get('signal_text', 'UNKNOWN') # e.g. "BUY 🚀"
        price = float(signal_data.get('current_price', 0.0))
        confidence = float(signal_data.get('confidence', 0.0))
        conf_level = signal_data.get('confidence_level', 'Medium')
        frequency = signal_data.get('prediction_frequency', 'Real-time')
        
        # Color coding for signal
        if "BUY" in signal_text:
            signal_color = "#00cc66" # Green
        elif "SELL" in signal_text:
            signal_color = "#ff4444" # Red
        else:
            signal_color = "#ffbb00" # Yellow

        subject = f"🚨 {signal_text} Alert: {ticker} ({confidence}%)"
        
        # HTML Body
        # Detect currency and convert if it's a US stock
        is_us_stock = not (ticker.endswith(".NS") or ticker.endswith(".BO"))
        currency_symbol = "₹"
        
        if is_us_stock:
            rate = get_usd_inr_rate()
            original_price = price
            price = price * rate
            print(f"DEBUG: [Conversion] {ticker} Price: ${original_price:.2f} → ₹{price:,.2f} (Rate: {rate})")
        
        body = f"""
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
            </style>
        </head>
        <body style="font-family: 'Inter', Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 0;">
            <div style="max-width: 600px; margin: 20px auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                
                <!-- HEADER -->
                <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 30px 20px; text-align: center;">
                    <div style="color: #60a5fa; font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px;">AI Powered Signals</div>
                    <div style="font-size: 24px; font-weight: 800; color: #ffffff; margin: 0;">Trading Alert 🔔</div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 6px;">{current_time}</div>
                </div>
                
                <!-- SIGNAL CARD -->
                <div style="padding: 32px 24px;">
                    <div style="text-align: center; margin-bottom: 32px;">
                        <div style="font-size: 18px; color: #64748b; font-weight: 600; margin-bottom: 12px;">{ticker}</div>
                        <div style="font-size: 42px; font-weight: 800; color: {signal_color}; letter-spacing: -1px; line-height: 1;">
                            {signal_text}
                        </div>
                    </div>
                    
                    <!-- METRICS GRID -->
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="border-bottom: 1px solid #eef2f6;">
                                <td style="padding: 12px 0; color: #64748b; font-size: 14px;">Market Price (INR)</td>
                                <td style="padding: 12px 0; font-weight: 700; color: #0f172a; text-align: right; font-size: 16px;">{currency_symbol}{price:,.2f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eef2f6;">
                                <td style="padding: 12px 0; color: #64748b; font-size: 14px;">AI Confidence</td>
                                <td style="padding: 12px 0; font-weight: 700; color: #0f172a; text-align: right; font-size: 16px;">{confidence}%</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eef2f6;">
                                <td style="padding: 12px 0; color: #64748b; font-size: 14px;">Strength Level</td>
                                <td style="padding: 12px 0; font-weight: 700; color: #0f172a; text-align: right; font-size: 14px;">
                                    <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 20px; font-size: 12px;">{conf_level}</span>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding: 12px 0; color: #64748b; font-size: 14px;">Trigger</td>
                                <td style="padding: 12px 0; font-weight: 600; color: #475569; text-align: right; font-size: 14px;">{frequency}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <!-- FOOTER -->
                    <div style="margin-top: 32px; text-align: center;">
                        <a href="http://localhost:8501" style="text-decoration: none; background: #f1f5f9; color: #475569; padding: 12px 24px; border-radius: 8px; font-weight: 600; font-size: 14px; display: inline-block;">Open Dashboard</a>
                        <p style="margin-top: 24px; font-size: 11px; color: #94a3b8; line-height: 1.5;">
                            Automated alert generated by AI Signals Platform.<br>
                            Not financial advice. Trade at your own risk.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        # Setup the email
        msg = MIMEText(body, 'html')
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = user_email

        # Send via Gmail SSL
        password_clean = EMAIL_PASSWORD.replace(" ", "")
        print(f"DEBUG: Attempting to connect to smtp.gmail.com:465 for {EMAIL_SENDER}...")
        
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15) as server:
                server.login(EMAIL_SENDER, password_clean)
                server.sendmail(EMAIL_SENDER, user_email, msg.as_string())
            print(f"✅ [Success] Email sent to {user_email} for {ticker}")
        except smtplib.SMTPAuthenticationError as e:
            print(f"❌ [Auth Error] SMTP login failed. Check App Password for {EMAIL_SENDER}. Error: {e}")
            raise
        except smtplib.SMTPException as e:
            print(f"❌ [SMTP Error] smtplib error during handshake/send: {e}")
            raise
        except Exception as e:
            print(f"❌ [Network/TLS Error] {e}")
            raise

    except Exception as e:
        print(f"❌ [Alert Execution Error] Unexpected error in send_email_alert: {e}")
        raise

# ==========================================
# 5. THE CORE JOB (Runs at scheduled times)
# ==========================================
def check_and_alert_job(user_email: str, ticker: str, force: bool = False, alert_type: str = "Scheduled"):
    """
    Scheduled job to check signals and alert user.
    force: If True, sends email regardless of confidence threshold (for Ignore/Test).
    alert_type: Display string for email (e.g. "Scheduled @ 10:00" or "Instant Report")
    """
    print(f"\n⏰ [Scheduler] Running check for {ticker} (User: {user_email}) | Type: {alert_type}")

    try:
        # 1️⃣ Live ML signal
        live = fetch_ml_signal(ticker)

        # 2️⃣ Backtesting (already includes confidence)
        bt = fetch_backtest_result(ticker)
        
        if not live:
             print(f"⚠️ [SKIP] No live data for {ticker}. Is the ML Signal API (Port 8000) running?")
             if force:
                 print("⚠️ FORCE MODE: Using dummy data for testing purposes.")
                 live = {
                     "signal": "TEST_SIGNAL",
                     "current_price": 0.0,
                     "confidence": 0.0
                 }
             else:
                 return False, "ML API connection failed or returned no data."

        if bt:
            confidence = bt.get("confidence_score", 0)
        else:
            confidence = float(live.get("confidence", 0))
            print(f"ℹ️ [Fallback] Backtest data unavailable. Using Live Confidence: {confidence}%")

        # Determine Signal Emoji
        raw_signal = live.get("signal", "HOLD")
        if raw_signal == "BUY":
            signal_text = "BUY 🚀"
        elif raw_signal == "SELL":
            signal_text = "SELL 🔻"
        else:
            signal_text = "HOLD ⏸️"

        # Determine Confidence Level
        if confidence >= 80:
            conf_level = "Very High"
        elif confidence >= 60:
            conf_level = "High"
        elif confidence >= 40:
            conf_level = "Medium"
        else:
            conf_level = "Low"

        # Prepare Data Payload
        email_data = {
            "signal_text": signal_text,
            "current_price": live.get("current_price", 0.0),
            "confidence": f"{confidence:.2f}",
            "confidence_level": conf_level,
            "prediction_frequency": alert_type
        }

        # 3️⃣ Send alert (Threshold removed by user request)
        print(f"🚀 SIGNAL ({confidence:.2f}%) → Sending Email")
        send_email_alert(user_email, ticker, email_data)

    except Exception as e:
        print(f"❌ Alert job failed for {ticker}: {e}")
        return False, str(e)
    
    return True, "Success"

# ==========================================
# 6. API ENDPOINTS (For Dashboard Team)
# ==========================================

# Input Model: This is what the Dashboard sends you
class AlertRequest(BaseModel):
    user_email: str
    ticker_name: str
    alert_time: str # "HH:MM" format (24h)

@app.post("/create-alert")
async def create_alert(request: AlertRequest):
    """
    DEBUG EXPLANATION:
    - Receives a request from the Streamlit Frontend to monitor a stock.
    - Creates a Cron job in the background scheduler at the specified time.
    - Logs the request to the terminal.
    """
    email = request.user_email
    ticker = request.ticker_name
    time_str = request.alert_time
    
    base_id = f"{email}_{ticker}"

    print(f"📥 [API Received] Request to alert {email} on {ticker} at {time_str}")
    
    # DEBUG: Print current time to compare
    now_str = datetime.now(get_localzone()).strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f"    ↳ Server Time: {now_str}")

    try:
        # Parse time
        try:
            hour, minute = map(int, time_str.split(':'))
        except ValueError:
             raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM")
             
        # Create a unique job ID based on time
        # This allows multiple alerts for the same ticker at different times if needed,
        # but the UI currently sets one time for a batch.
        job_id = f"{base_id}_{hour:02d}{minute:02d}"

        # Schedule Job
        
        # Pass the time string into the job arguments so the email says "Scheduled @ 10:00"
        job_type_str = f"Scheduled Daily @ {time_str}"
        
        scheduler.add_job(check_and_alert_job, 'cron', hour=hour, minute=minute,
                          id=job_id, args=[email, ticker, False, job_type_str], replace_existing=True)

        print(f"✅ [Scheduler] Job Created: {job_id} | Schedule: {hour:02d}:{minute:02d} IST")

        return {
            "status": "success",
            "message": f"Alert set for {ticker} at {time_str}",
            "user": email
        }

    except Exception as e:
        print(f"ERROR: Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/stop-alert/{user_email}/{ticker_name}")
async def stop_alert(user_email: str, ticker_name: str):
    """
    DEBUG EXPLANATION:
    - Helper endpoint to unschedule jobs when the user clicks 'Stop' or 'Trash'.
    - Removes ALL scheduled jobs for this user/ticker pair.
    """
    base_id = f"{user_email}_{ticker_name}"
    print(f"DEBUG: Stopping all alerts for {base_id}...")
    
    deleted_count = 0
    try:
        # Iterate over all jobs and remove those that match the base_id prefix
        for job in scheduler.get_jobs():
            if job.id.startswith(base_id):
                scheduler.remove_job(job.id)
                deleted_count += 1
        
        if deleted_count > 0:
             return {"status": "success", "message": f"Stopped {deleted_count} alerts for {ticker_name}"}
        else:
             return {"status": "warning", "message": "No active alerts found to stop"}
             
    except Exception as e:
        print(f"DEBUG: Stop alert warning: {e}")
        return {"status": "warning", "message": f"Error stopping alerts: {str(e)}"}

# --- Added for Frontend Compatibility ---
@app.get("/active-alerts")
def get_active_alerts():
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "next_run": str(job.next_run_time),
            "args": job.args
        })
    return {"count": len(jobs), "jobs": jobs}

class InstantReportRequest(BaseModel):
    user_email: str
    ticker_name: str

@app.post("/instant-report")
def instant_report(request: InstantReportRequest):
    # Passes force=True to ensure email is sent for testing purposes
    success, message = check_and_alert_job(request.user_email, request.ticker_name, force=True, alert_type="Instant Report (Manual)")
    if success:
        return {"status": "success", "message": "Report sent"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to send report: {message}")

@app.delete("/clear-all-alerts")
def clear_all_alerts():
    scheduler.remove_all_jobs()
    return {"status": "success", "message": "All active alerts cleared."}
# ----------------------------------------

@app.get("/")
def health_check():
    return {"status": "active", "system": "Alerts Middleware API"}

@app.get("/health")
def health_alias():
     return {"status": "active", "system": "Alerts Middleware API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
