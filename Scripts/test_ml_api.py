import requests
try:
    r = requests.post('http://127.0.0.1:8000/api/v1/ml/signal/live', json={'ticker': 'AAPL'}, timeout=120)
    data = r.json()
    print(f"Price: {data.get('current_price')}")
    print(f"Signal: {data.get('signal')}")
    print(f"Confidence: {data.get('confidence')}")
except Exception as e:
    print(f"Error: {e}")
