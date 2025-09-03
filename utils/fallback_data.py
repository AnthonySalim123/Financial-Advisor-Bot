"""
Fallback data generator for when yfinance is unavailable
This uses synthetic but realistic data for development/testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_stock_data(symbol, period='2y'):
    """
    Generate synthetic stock data that mimics real patterns
    """
    # Determine number of days
    period_days = {
        '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
        '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
    }
    days = period_days.get(period, 365)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base prices for different stocks
    base_prices = {
        'AAPL': 150, 'MSFT': 350, 'GOOGL': 140, 
        'NVDA': 450, 'AMZN': 170, 'META': 350,
        'JPM': 150, 'BAC': 35, 'GS': 350,
        'JNJ': 160, 'PFE': 30, 'MRNA': 100
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    price_series = base_price * np.exp(np.cumsum(returns))
    
    # Add some trend
    trend = np.linspace(0, 0.2, len(dates))  # 20% upward trend over period
    price_series = price_series * (1 + trend)
    
    # Generate OHLCV data
    data = []
    for i, date in enumerate(dates):
        close = price_series[i]
        open_price = close * (1 + np.random.uniform(-0.01, 0.01))
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.02))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.02))
        volume = int(50000000 * (1 + np.random.uniform(-0.5, 0.5)))
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    print(f"Generated synthetic data for {symbol}: {len(df)} days")
    return df

def get_synthetic_info(symbol):
    """
    Get synthetic company info
    """
    info_map = {
        'AAPL': {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'marketCap': 3000000000000,
            'trailingPE': 28.5,
            'forwardPE': 26.2,
            'dividendYield': 0.0044,
            'beta': 1.25
        },
        'MSFT': {
            'symbol': 'MSFT',
            'longName': 'Microsoft Corporation',
            'sector': 'Technology',
            'marketCap': 2800000000000,
            'trailingPE': 35.2,
            'forwardPE': 30.1,
            'dividendYield': 0.0072,
            'beta': 0.93
        },
        'GOOGL': {
            'symbol': 'GOOGL',
            'longName': 'Alphabet Inc.',
            'sector': 'Technology',
            'marketCap': 1700000000000,
            'trailingPE': 26.8,
            'forwardPE': 23.5,
            'dividendYield': 0,
            'beta': 1.07
        }
    }
    
    return info_map.get(symbol, {
        'symbol': symbol,
        'longName': f'{symbol} Corporation',
        'sector': 'Technology',
        'marketCap': 500000000000,
        'trailingPE': 25.0,
        'forwardPE': 22.0,
        'dividendYield': 0.01,
        'beta': 1.0
    })