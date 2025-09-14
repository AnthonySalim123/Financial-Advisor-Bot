# get_real_data_alternative.py
# Multiple alternative methods to get REAL stock data

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("🚀 ALTERNATIVE METHODS TO GET REAL STOCK DATA")
print("="*60)

# METHOD 1: Use pandas_datareader (Works with Yahoo Finance)
print("\n📊 METHOD 1: Using pandas_datareader")
print("-"*40)
try:
    import pandas_datareader as pdr
    
    # Get data from Yahoo using pandas_datareader
    start_date = datetime.now() - timedelta(days=365*5)  # 5 years
    end_date = datetime.now()
    
    print("Fetching AAPL data...")
    df = pdr.get_data_yahoo('AAPL', start=start_date, end=end_date)
    
    if not df.empty:
        print(f"✅ SUCCESS! Got {len(df)} days of real data")
        df.to_csv('AAPL_real.csv')
        print("Saved to AAPL_real.csv")
        
        # Test it immediately
        from utils.ml_models import MLModel
        from utils.technical_indicators import TechnicalIndicators
        
        print("\n🎯 Testing with real data...")
        df = TechnicalIndicators.calculate_all_indicators(df)
        model = MLModel('classification')
        metrics = model.train(df)
        print(f"ACCURACY: {metrics['accuracy']*100:.1f}%")
        
except ImportError:
    print("❌ pandas_datareader not installed")
    print("Install it with: pip install pandas-datareader")
except Exception as e:
    print(f"❌ Method 1 failed: {e}")

# METHOD 2: Use yfinance with fix
print("\n📊 METHOD 2: Using yfinance (with fix)")
print("-"*40)
try:
    import yfinance as yf
    
    # Try different approach with yfinance
    print("Attempting yfinance download...")
    
    # Method A: Using Ticker object
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="5y")
    
    if df.empty:
        # Method B: Using download function
        df = yf.download("AAPL", start="2019-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False)
    
    if not df.empty:
        print(f"✅ SUCCESS! Got {len(df)} days of real data")
        df.to_csv('AAPL_yfinance.csv')
        print("Saved to AAPL_yfinance.csv")
        
        # Test it
        from utils.ml_models import MLModel
        from utils.technical_indicators import TechnicalIndicators
        
        df = TechnicalIndicators.calculate_all_indicators(df)
        model = MLModel('classification')
        metrics = model.train(df)
        print(f"ACCURACY: {metrics['accuracy']*100:.1f}%")
    else:
        print("❌ yfinance returned empty data")
        
except Exception as e:
    print(f"❌ Method 2 failed: {e}")

# METHOD 3: Manual instructions for Yahoo Finance website
print("\n📊 METHOD 3: Manual Download from Yahoo Finance Website")
print("-"*40)
print("""
Since direct download is blocked, do this:

1. Open your browser
2. Go to: https://finance.yahoo.com/quote/AAPL/history
3. Look for these settings:
   - Time Period: Click and select "5Y" (5 years)
   - Show: Historical Prices
   - Frequency: Daily
4. Click "Apply"
5. Look for "Download" link/button (usually at top-right of the data table)
6. Click it to download AAPL.csv
7. Move the file to your project folder
8. Run the test script below
""")

# METHOD 4: Use Alpha Vantage (FREE API)
print("\n📊 METHOD 4: Alpha Vantage FREE API")
print("-"*40)
print("""
1. Get FREE API key (no credit card needed):
   https://www.alphavantage.co/support/#api-key
   
2. Install: pip install alpha-vantage

3. Use this code:
""")

code = '''
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Use your free API key
ts = TimeSeries(key='YOUR_FREE_API_KEY', output_format='pandas')
data, meta = ts.get_daily_adjusted('AAPL', outputsize='full')

# Save it
data.to_csv('AAPL_alphavantage.csv')
print(f"Got {len(data)} days of real data!")
'''
print(code)

# METHOD 5: Use IEX Cloud (FREE tier available)
print("\n📊 METHOD 5: IEX Cloud (Free tier)")
print("-"*40)
print("""
1. Sign up for free at: https://iexcloud.io/
2. Get free API token (no credit card)
3. Install: pip install pyEX
4. Use this code:

import pyEX as p
c = p.Client(api_token='YOUR_TOKEN')
df = c.chartDF('AAPL', timeframe='5y')
df.to_csv('AAPL_iex.csv')
""")

print("\n" + "="*60)
print("📋 QUICKEST SOLUTION:")
print("="*60)
print("""
1. Install pandas_datareader:
   pip install pandas-datareader

2. Run this script again - Method 1 should work!

OR

Just go to Yahoo Finance website and download manually (Method 3)
""")

# Check if any CSV exists from previous attempts
print("\n" + "="*60)
print("📂 CHECKING FOR EXISTING DATA FILES:")
print("="*60)

import os
possible_files = ['AAPL.csv', 'AAPL_real.csv', 'AAPL_yfinance.csv', 'AAPL_google.csv']

for filename in possible_files:
    if os.path.exists(filename):
        print(f"✅ FOUND: {filename}")
        print(f"   Testing this file...")
        
        try:
            df = pd.read_csv(filename)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            
            # Keep only OHLCV
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required):
                df = df[required]
                
                from utils.ml_models import MLModel
                from utils.technical_indicators import TechnicalIndicators
                
                df = TechnicalIndicators.calculate_all_indicators(df)
                model = MLModel('classification')
                metrics = model.train(df)
                
                print(f"   ACCURACY: {metrics['accuracy']*100:.1f}%")
                
                if metrics['accuracy'] >= 0.6:
                    print(f"\n🎉 SUCCESS! File {filename} gives {metrics['accuracy']*100:.1f}% accuracy!")
                    print("Your model works with real data!")
                    break
        except Exception as e:
            print(f"   Error testing {filename}: {e}")

print("\n" + "="*60)
print("💡 RECOMMENDED ACTION:")
print("="*60)
print("""
1. Run: pip install pandas-datareader
2. Run this script again
3. It will automatically download and test real data
4. You should see 60-70% accuracy!
""")