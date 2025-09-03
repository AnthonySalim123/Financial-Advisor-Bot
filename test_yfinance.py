# Create a test file: test_yfinance.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Testing yfinance connection...")

# Test 1: Simple fetch
try:
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    print("✓ Connection successful")
    print(f"  Company: {info.get('longName', 'N/A')}")
except Exception as e:
    print(f"✗ Info fetch failed: {e}")

# Test 2: Historical data with different period
try:
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1mo")  # Try shorter period
    if not data.empty:
        print(f"✓ Historical data fetched: {len(data)} days")
        print(f"  Latest price: ${data['Close'].iloc[-1]:.2f}")
    else:
        print("✗ Historical data is empty")
except Exception as e:
    print(f"✗ Historical fetch failed: {e}")

# Test 3: Direct download
try:
    data = yf.download("AAPL", start="2024-01-01", end="2024-12-31", progress=False)
    if not data.empty:
        print(f"✓ Direct download successful: {len(data)} days")
except Exception as e:
    print(f"✗ Direct download failed: {e}")