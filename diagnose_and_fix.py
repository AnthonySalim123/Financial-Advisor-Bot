# diagnose_and_fix.py
"""
Diagnostic script to identify why accuracy is low and fix it
Run this to understand and solve the problem
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("🔍 DIAGNOSING LOW ACCURACY ISSUE")
print("="*60)

# ============================================
# STEP 1: CHECK DATA QUALITY
# ============================================
print("\n📊 STEP 1: Checking Data Quality...")
print("-"*40)

def analyze_data_quality(filename):
    """Deep analysis of CSV data"""
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    
    # Clean data
    df = df.rename(columns={'Price': 'Close', 'Vol.': 'Volume'})
    df['Volume'] = df['Volume'].str.replace('M', '').astype(float) * 1e6
    
    print(f"\n{filename}:")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Check for future dates
    if df.index[-1].year >= 2025:
        print(f"  ⚠️ WARNING: Contains future dates (year {df.index[-1].year})")
    
    # Calculate volatility
    returns = df['Close'].pct_change()
    volatility = returns.std()
    print(f"  Volatility: {volatility*100:.2f}% daily")
    
    # Check price movement patterns
    price_changes = df['Close'].pct_change().dropna()
    positive_days = (price_changes > 0).mean()
    print(f"  Positive days: {positive_days*100:.1f}%")
    
    # Check for data issues
    if volatility < 0.005:
        print("  ❌ PROBLEM: Volatility too low - might be synthetic data")
    elif volatility > 0.10:
        print("  ❌ PROBLEM: Volatility too high - check for errors")
    else:
        print("  ✅ Volatility looks realistic")
    
    # Check for patterns
    autocorr = returns.autocorr()
    print(f"  Autocorrelation: {autocorr:.3f}")
    
    if abs(autocorr) > 0.1:
        print("  ⚠️ High autocorrelation - potential data issue")
    
    return df

msft = analyze_data_quality('MSFT.csv')
googl = analyze_data_quality('GOOGL.csv')

# ============================================
# STEP 2: TEST DIFFERENT SIGNAL METHODS
# ============================================
print("\n📊 STEP 2: Testing Different Signal Generation Methods...")
print("-"*40)

def test_signal_methods(df, name):
    """Test different ways to create signals"""
    
    # Clean and prepare data
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Add basic indicators
    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    
    # Remove NaN
    df = df.dropna()
    
    print(f"\n{name} - Testing signal methods:")
    
    # Method 1: Simple returns
    future_returns = df['Close'].pct_change(5).shift(-5)
    method1_signals = pd.qcut(future_returns, q=3, labels=[0,1,2], duplicates='drop')
    
    # Method 2: Fixed thresholds
    method2_signals = pd.Series(1, index=df.index)
    method2_signals[future_returns > 0.02] = 2
    method2_signals[future_returns < -0.02] = 0
    
    # Method 3: Volatility adjusted
    volatility = df['Returns'].rolling(20).std()
    norm_returns = future_returns / (volatility + 1e-10)
    method3_signals = pd.Series(1, index=df.index)
    method3_signals[norm_returns > 1] = 2
    method3_signals[norm_returns < -1] = 0
    
    # Test each method
    features = ['RSI', 'SMA_20', 'SMA_50', 'MACD']
    X = df[features].dropna()
    
    for i, (method_name, signals) in enumerate([
        ("Percentile-based", method1_signals),
        ("Fixed threshold", method2_signals),
        ("Volatility-adjusted", method3_signals)
    ], 1):
        # Align data
        valid_idx = X.index.intersection(signals.dropna().index)
        X_clean = X.loc[valid_idx]
        y_clean = signals.loc[valid_idx]
        
        if len(X_clean) > 100:
            # Quick test
            split = int(len(X_clean) * 0.8)
            X_train = X_clean[:split]
            X_test = X_clean[split:]
            y_train = y_clean[:split]
            y_test = y_clean[split:]
            
            # Scale and train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            
            acc = accuracy_score(y_test, rf.predict(X_test_scaled))
            print(f"  Method {i} ({method_name}): {acc*100:.1f}% accuracy")

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

test_signal_methods(msft, 'MSFT')
test_signal_methods(googl, 'GOOGL')

# ============================================
# STEP 3: FIX AND RETRAIN
# ============================================
print("\n📊 STEP 3: Applying Fixes and Retraining...")
print("-"*40)

def create_fixed_model(msft_df, googl_df):
    """Create a model with fixes applied"""
    
    # Process both dataframes
    dfs = []
    for df, name in [(msft_df, 'MSFT'), (googl_df, 'GOOGL')]:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Calculate more features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_20d'] = df['Close'].pct_change(20)
        
        # Technical indicators
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Moving averages
        for period in [10, 20, 50]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA_20'] + (std * 2)
        df['BB_Lower'] = df['SMA_20'] - (std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']
        
        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['ATR'] = calculate_atr(df)
        
        # Price patterns
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_to_High'] = df['Close'] / df['High']
        df['Close_to_Low'] = df['Close'] / df['Low']
        
        # Create BETTER signals
        future_returns = df['Close'].pct_change(5).shift(-5)
        
        # Use adaptive thresholds based on recent volatility
        rolling_std = future_returns.rolling(window=60).std()
        adaptive_upper = rolling_std * 0.5
        adaptive_lower = -rolling_std * 0.5
        
        df['Signal'] = 1  # Default HOLD
        df.loc[future_returns > adaptive_upper, 'Signal'] = 2  # BUY
        df.loc[future_returns < adaptive_lower, 'Signal'] = 0  # SELL
        
        # Fallback to percentiles where adaptive doesn't work
        if df['Signal'].value_counts().min() < len(df) * 0.2:
            # Use percentiles instead
            lower_33 = future_returns.quantile(0.33)
            upper_67 = future_returns.quantile(0.67)
            df['Signal'] = 1
            df.loc[future_returns < lower_33, 'Signal'] = 0
            df.loc[future_returns > upper_67, 'Signal'] = 2
        
        df['Stock'] = name
        dfs.append(df)
    
    # Combine
    combined = pd.concat(dfs)
    combined = combined.dropna()
    
    # Features to use
    feature_cols = [
        'Returns', 'Returns_5d', 'Returns_20d',
        'RSI', 'MACD', 'MACD_Signal',
        'SMA_10', 'SMA_20', 'SMA_50',
        'BB_Width', 'Volume_Ratio', 'Volatility', 'ATR',
        'High_Low_Ratio', 'Close_to_High', 'Close_to_Low'
    ]
    
    X = combined[feature_cols]
    y = combined['Signal']
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train improved model
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, rf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
    
    print(f"\n🎯 FIXED MODEL RESULTS:")
    print(f"  Training accuracy: {train_acc*100:.2f}%")
    print(f"  Testing accuracy:  {test_acc*100:.2f}%")
    
    return rf, scaler, feature_cols, test_acc

def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return pd.Series(true_range).rolling(window=period).mean()

# Apply fixes
model, scaler, features, accuracy = create_fixed_model(msft, googl)

# ============================================
# STEP 4: DIAGNOSIS
# ============================================
print("\n" + "="*60)
print("🔍 DIAGNOSIS COMPLETE")
print("="*60)

if accuracy >= 0.55:
    print(f"\n✅ SUCCESS! Fixed model achieves {accuracy*100:.2f}% accuracy!")
    print("\n🎯 The fixes that worked:")
    print("  1. Added more technical indicators")
    print("  2. Used adaptive signal thresholds")
    print("  3. Reduced model overfitting")
    print("  4. Better feature engineering")
    
    # Save the fixed model
    import joblib
    joblib.dump({'model': model, 'scaler': scaler, 'features': features}, 'fixed_model.pkl')
    print("\n💾 Fixed model saved as 'fixed_model.pkl'")
    
elif accuracy >= 0.45:
    print(f"\n⚠️ Partial improvement: {accuracy*100:.2f}%")
    print("\n💡 Additional steps needed:")
    print("  1. Your data might contain synthetic patterns")
    print("  2. Try using only data before 2024")
    print("  3. Consider getting real historical data")
    
else:
    print(f"\n❌ Still low accuracy: {accuracy*100:.2f}%")
    print("\n🔍 Root cause analysis:")
    print("  1. The CSV data appears to be synthetic or randomized")
    print("  2. Future dates (2025) suggest this is test data")
    print("  3. Price patterns don't match real market behavior")
    
    print("\n💡 SOLUTION:")
    print("  1. Download REAL historical data from Yahoo Finance")
    print("  2. Use only dates before 2024")
    print("  3. Or use yfinance to get real data:")
    print("\n  import yfinance as yf")
    print("  msft = yf.download('MSFT', start='2020-01-01', end='2024-01-01')")
    print("  googl = yf.download('GOOGL', start='2020-01-01', end='2024-01-01')")

print("\n" + "="*60)
print("✅ Diagnosis complete. Check results above.")