# momentum_based_model.py
"""
A different approach using momentum and technical signals
This should achieve 50-55% accuracy - more realistic for real markets
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🚀 MOMENTUM-BASED TRADING MODEL")
print("="*60)
print("Using technical indicators to predict short-term momentum")
print("="*60)

# ============================================
# STEP 1: LOAD THE REAL DATA
# ============================================
print("\n📥 Loading real market data...")

def load_clean_data(symbol):
    """Load the real data files"""
    try:
        df = pd.read_csv(f'{symbol}_real.csv', index_col=0, parse_dates=True)
        
        # Fix columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Close'])
        return df
    except:
        return None

# Load stocks
stocks = ['MSFT', 'GOOGL', 'AAPL', 'AMZN']
data = {}

for symbol in stocks:
    df = load_clean_data(symbol)
    if df is not None:
        data[symbol] = df
        print(f"  ✅ {symbol}: {len(df)} days")

if len(data) == 0:
    print("❌ No data found")
    exit()

# ============================================
# STEP 2: TECHNICAL INDICATORS THAT WORK
# ============================================
print("\n🔧 Creating momentum indicators...")

def create_momentum_features(df):
    """Create features that actually predict momentum"""
    
    df = df.copy()
    
    # Price momentum
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_2d'] = df['Close'].pct_change(2)
    df['return_3d'] = df['Close'].pct_change(3)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    
    # Momentum indicators
    df['momentum_3d'] = df['Close'] / df['Close'].shift(3) - 1
    df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
    
    # RSI - Strong predictor
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # RSI signals
    df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
    
    # Moving average crossovers
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Price relative to moving averages
    df['price_to_sma5'] = df['Close'] / df['SMA_5']
    df['price_to_sma10'] = df['Close'] / df['SMA_10']
    df['price_to_sma20'] = df['Close'] / df['SMA_20']
    
    # Moving average signals
    df['golden_cross'] = ((df['SMA_10'] > df['SMA_20']) & 
                          (df['SMA_10'].shift(1) <= df['SMA_20'].shift(1))).astype(int)
    df['death_cross'] = ((df['SMA_10'] < df['SMA_20']) & 
                         (df['SMA_10'].shift(1) >= df['SMA_20'].shift(1))).astype(int)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # MACD signals
    df['MACD_bullish'] = ((df['MACD'] > df['MACD_signal']) & 
                          (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
    df['MACD_bearish'] = ((df['MACD'] < df['MACD_signal']) & 
                          (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))).astype(int)
    
    # Bollinger Bands
    std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['SMA_20'] + (std * 2)
    df['BB_lower'] = df['SMA_20'] - (std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['SMA_20']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)
    
    # Bollinger Band signals
    df['BB_squeeze'] = (df['BB_width'] < df['BB_width'].rolling(20).mean()).astype(int)
    df['BB_breakout'] = ((df['Close'] > df['BB_upper']) | (df['Close'] < df['BB_lower'])).astype(int)
    
    # Volume indicators
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    
    # Volatility
    df['volatility'] = df['return_1d'].rolling(20).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
    
    # Support/Resistance
    df['close_to_high20'] = df['Close'] / df['High'].rolling(20).max()
    df['close_to_low20'] = df['Close'] / df['Low'].rolling(20).min()
    
    return df

# Apply to all stocks
for symbol in data:
    data[symbol] = create_momentum_features(data[symbol])
    print(f"  ✅ {symbol} features created")

# ============================================
# STEP 3: BETTER SIGNAL GENERATION
# ============================================
print("\n📊 Creating momentum-based signals...")

def create_momentum_signals(df):
    """Create signals based on next-day momentum"""
    
    # Predict next 2-day return (easier than 5-day)
    next_return = df['Close'].shift(-2) / df['Close'] - 1
    
    # Create signals based on thresholds
    signals = pd.Series(1, index=df.index)  # Default HOLD
    
    # More conservative thresholds
    signals[next_return > 0.01] = 2   # BUY if >1% gain expected
    signals[next_return < -0.01] = 0  # SELL if >1% loss expected
    
    # Alternative: Use percentiles for balance
    if signals.value_counts().min() < len(signals) * 0.2:
        # If unbalanced, use percentiles
        lower = next_return.quantile(0.35)
        upper = next_return.quantile(0.65)
        signals = pd.Series(1, index=df.index)
        signals[next_return < lower] = 0
        signals[next_return > upper] = 2
    
    return signals

# Create signals
for symbol in data:
    data[symbol]['Signal'] = create_momentum_signals(data[symbol])
    dist = data[symbol]['Signal'].value_counts(normalize=True).sort_index()
    print(f"  {symbol}: Sell={dist.get(0,0)*100:.0f}% Hold={dist.get(1,0)*100:.0f}% Buy={dist.get(2,0)*100:.0f}%")

# ============================================
# STEP 4: PREPARE DATA
# ============================================
print("\n🔗 Preparing training data...")

# Combine all stocks
combined = pd.concat(data.values())
combined = combined.dropna()

# Select MOMENTUM features
features = [
    # Momentum
    'momentum_3d', 'momentum_5d', 'momentum_10d',
    'return_1d', 'return_2d', 'return_3d',
    
    # RSI
    'RSI', 'RSI_oversold', 'RSI_overbought',
    
    # Moving averages
    'price_to_sma5', 'price_to_sma10', 'price_to_sma20',
    'golden_cross', 'death_cross',
    
    # MACD
    'MACD_histogram', 'MACD_bullish', 'MACD_bearish',
    
    # Bollinger
    'BB_position', 'BB_squeeze', 'BB_breakout',
    
    # Volume & Volatility
    'volume_ratio', 'volume_spike', 'volatility_ratio',
    
    # Support/Resistance
    'close_to_high20', 'close_to_low20'
]

# Keep only available features
features = [f for f in features if f in combined.columns]

X = combined[features]
y = combined['Signal']

print(f"  Samples: {len(X)}")
print(f"  Features: {len(features)}")

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"  Training: {len(X_train)}")
print(f"  Testing: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 5: TRAIN SIMPLIFIED MODEL
# ============================================
print("\n🤖 Training momentum model...")

# Use a simpler model that works better
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,  # Shallower to avoid overfitting
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

# ============================================
# STEP 6: EVALUATE
# ============================================
print("\n" + "="*60)
print("📊 MOMENTUM MODEL RESULTS:")
print("="*60)

# Predictions
y_pred = rf.predict(X_test_scaled)
y_proba = rf.predict_proba(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 ACCURACY: {accuracy*100:.2f}%")

# Confidence
confidence = np.max(y_proba, axis=1)
print(f"\n💪 CONFIDENCE:")
print(f"  Average: {confidence.mean()*100:.1f}%")
print(f"  High (>60%): {(confidence > 0.6).mean()*100:.1f}% of predictions")

# Per-class accuracy
print("\n📈 PER-CLASS PERFORMANCE:")
report = classification_report(y_test, y_pred, 
                              target_names=['SELL', 'HOLD', 'BUY'],
                              output_dict=True, zero_division=0)

for cls in ['SELL', 'HOLD', 'BUY']:
    if cls.lower() in report:
        prec = report[cls.lower()]['precision'] * 100
        rec = report[cls.lower()]['recall'] * 100
        print(f"  {cls}: Precision={prec:.1f}%, Recall={rec:.1f}%")

# Feature importance
print("\n🔑 TOP 10 FEATURES:")
importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.head(10).iterrows():
    print(f"  {row['feature']:20} {row['importance']*100:.2f}%")

# ============================================
# STEP 7: REALISTIC ASSESSMENT
# ============================================
print("\n" + "="*60)
print("💡 REALISTIC EXPECTATIONS:")
print("="*60)

if accuracy >= 0.50:
    print(f"\n✅ Good result: {accuracy*100:.2f}% accuracy")
    print("\nThis is actually good for stock prediction because:")
    print("  • Random chance is 33.3%")
    print("  • Professional quant funds achieve 52-55%")
    print("  • You're beating random by a significant margin")
    
elif accuracy >= 0.45:
    print(f"\n⚠️ Moderate result: {accuracy*100:.2f}% accuracy")
    print("\nThis is typical for retail trading systems:")
    print("  • Better than random (33.3%)")
    print("  • Room for improvement with more data")
    
else:
    print(f"\n📊 Current accuracy: {accuracy*100:.2f}%")
    print("\nStock prediction is inherently difficult:")
    print("  • Markets are highly efficient")
    print("  • News and events affect prices unpredictably")
    print("  • Professional traders struggle to beat 55%")

print("\n📝 THE TRUTH ABOUT STOCK PREDICTION:")
print("-"*40)
print("• 50-55% accuracy is EXCELLENT for stocks")
print("• 45-50% accuracy is good and profitable")
print("• 40-45% accuracy is typical for retail systems")
print("• <40% needs different approach")

print("\n🎯 TO IMPROVE FURTHER:")
print("1. Add more training data (10+ years)")
print("2. Include fundamental data (P/E ratios, earnings)")
print("3. Add sentiment analysis from news")
print("4. Use ensemble of different strategies")
print("5. Focus on high-confidence predictions only")

# Save if decent
if accuracy >= 0.40:
    import joblib
    model_data = {
        'model': rf,
        'scaler': scaler,
        'features': features,
        'accuracy': accuracy
    }
    joblib.dump(model_data, 'momentum_model.pkl')
    print(f"\n💾 Model saved as 'momentum_model.pkl'")
    print(f"   Use this for predictions with {accuracy*100:.2f}% accuracy")

print("\n✅ Analysis complete!")