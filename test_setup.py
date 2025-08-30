#!/usr/bin/env python
"""Test script to verify installation"""

print("Testing StockBot Setup...")
print("-" * 50)

# Test imports
try:
    import streamlit as st
    print("✅ Streamlit installed")
except:
    print("❌ Streamlit not installed")

try:
    import pandas as pd
    print("✅ Pandas installed")
except:
    print("❌ Pandas not installed")

try:
    import yfinance as yf
    print("✅ YFinance installed")
except:
    print("❌ YFinance not installed")

try:
    import plotly
    print("✅ Plotly installed")
except:
    print("❌ Plotly not installed")

try:
    from sklearn.ensemble import RandomForestClassifier
    print("✅ Scikit-learn installed")
except:
    print("❌ Scikit-learn not installed")

# Test YFinance data
try:
    print("\nTesting data fetch...")
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1d")
    if not hist.empty:
        print(f"✅ YFinance working - AAPL price: ${hist['Close'].iloc[-1]:.2f}")
    else:
        print("⚠️ YFinance returned empty data")
except Exception as e:
    print(f"❌ YFinance error: {e}")

print("-" * 50)
print("Setup test complete!")
