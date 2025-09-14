# train_updated.py
"""
Training script for your MSFT.csv and GOOGL.csv files
Save this file and run: python train_updated.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🚀 TRAINING WITH YOUR MSFT & GOOGL CSV FILES")
print("="*60)

# ============================================
# STEP 1: LOAD CSV FILES
# ============================================
def load_csv_file(filename):
    """Load and clean your CSV format"""
    print(f"\n📂 Loading {filename}...")
    
    try:
        # Read CSV
        df = pd.read_csv(filename)
        
        # Parse date (handle MM/DD/YYYY format)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Standardize columns
        df = df.rename(columns={
            'Price': 'Close',
            'Vol.': 'Volume',
            'Change %': 'Change'
        })
        
        # Clean Volume (M = millions)
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].str.replace('M', '').astype(float) * 1e6
        
        # Keep OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"  ✅ Loaded {len(df)} days of data")
        return df
        
    except Exception as e:
        print(f"  ❌ Error loading {filename}: {e}")
        return None

# Load your CSV files
msft = load_csv_file('MSFT.csv')
googl = load_csv_file('GOOGL.csv')

if msft is None or googl is None:
    print("\n❌ Error: Could not load CSV files")
    print("Make sure MSFT.csv and GOOGL.csv are in the current directory")
    exit()

# ============================================
# STEP 2: ADD TECHNICAL INDICATORS
# ============================================
def add_indicators(df):
    """Add technical indicators"""
    
    # Price changes
    df['Returns'] = df['Close'].pct_change()
    df['Returns_5d'] = df['Close'].pct_change(5)
    df['Returns_20d'] = df['Close'].pct_change(20)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + (std * 2)
    df['BB_Lower'] = df['SMA_20'] - (std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # Volume ratio
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    # Price ratios
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Price_vs_SMA20'] = df['Close'] / df['SMA_20']
    
    return df

print("\n🔧 Calculating indicators...")
msft = add_indicators(msft)
googl = add_indicators(googl)
print("  ✅ Added technical indicators")

# ============================================
# STEP 3: CREATE SIGNALS
# ============================================
def create_signals(df):
    """Create balanced buy/sell/hold signals"""
    
    # Calculate 5-day future returns
    future_returns = df['Close'].pct_change(5).shift(-5)
    
    # Use percentiles for balanced distribution
    lower_33 = future_returns.quantile(0.33)
    upper_67 = future_returns.quantile(0.67)
    
    signals = pd.Series(1, index=df.index)  # Default = HOLD (1)
    signals[future_returns < lower_33] = 0   # SELL (0)
    signals[future_returns > upper_67] = 2   # BUY (2)
    
    return signals

print("\n📈 Creating signals...")
msft['Signal'] = create_signals(msft)
googl['Signal'] = create_signals(googl)

# Check signal distribution
for name, df in [('MSFT', msft), ('GOOGL', googl)]:
    dist = df['Signal'].value_counts(normalize=True)
    print(f"  {name}: Sell={dist.get(0,0)*100:.0f}% Hold={dist.get(1,0)*100:.0f}% Buy={dist.get(2,0)*100:.0f}%")

# ============================================
# STEP 4: PREPARE TRAINING DATA
# ============================================
print("\n📊 Preparing data...")

# Combine both stocks
combined = pd.concat([msft, googl])
combined = combined.dropna()

print(f"  Combined dataset: {len(combined)} samples")

# Select features
features = ['RSI', 'MACD', 'MACD_Signal', 'BB_Position',
            'SMA_20', 'SMA_50', 'Volume_Ratio', 'Volatility',
            'High_Low_Ratio', 'Price_vs_SMA20', 'Returns_5d', 'Returns_20d']

X = combined[features]
y = combined['Signal']

# Split data (80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"  Training: {len(X_train)} samples")
print(f"  Testing: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 5: TRAIN MODELS
# ============================================
print("\n🤖 Training models...")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)

print("  Training Random Forest...")
rf.fit(X_train_scaled, y_train)

print("  Training Gradient Boosting...")
gb.fit(X_train_scaled, y_train)

# ============================================
# STEP 6: EVALUATE
# ============================================
print("\n📈 Evaluating models...")

# Get predictions
rf_pred = rf.predict(X_test_scaled)
gb_pred = gb.predict(X_test_scaled)

# Get probabilities for ensemble
rf_proba = rf.predict_proba(X_test_scaled)
gb_proba = gb.predict_proba(X_test_scaled)

# Ensemble prediction (weighted average)
ensemble_proba = 0.6 * rf_proba + 0.4 * gb_proba
ensemble_pred = np.argmax(ensemble_proba, axis=1)

# Calculate accuracies
rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print("\n" + "="*60)
print("📊 RESULTS:")
print("="*60)

print(f"\n🎯 ACCURACY:")
print(f"  Random Forest:     {rf_acc*100:.2f}%")
print(f"  Gradient Boosting: {gb_acc*100:.2f}%")
print(f"  ENSEMBLE:          {ensemble_acc*100:.2f}% ⭐")

# Confidence scores
confidence = np.max(ensemble_proba, axis=1)
print(f"\n💪 CONFIDENCE:")
print(f"  Average: {confidence.mean()*100:.1f}%")
print(f"  High (>70%): {(confidence > 0.7).mean()*100:.1f}% of predictions")

# Classification report
print("\n📊 DETAILED PERFORMANCE:")
print(classification_report(y_test, ensemble_pred, 
                          target_names=['SELL', 'HOLD', 'BUY']))

# Feature importance
print("\n🔑 TOP 5 FEATURES:")
importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance.head(5).iterrows():
    print(f"  {row['feature']:15} {row['importance']*100:.2f}%")

# ============================================
# STEP 7: SAVE MODEL
# ============================================
print("\n💾 Saving model...")

try:
    import joblib
    model_data = {
        'rf': rf,
        'gb': gb,
        'scaler': scaler,
        'features': features,
        'accuracy': ensemble_acc
    }
    joblib.dump(model_data, 'trained_model.pkl')
    print("  ✅ Model saved as 'trained_model.pkl'")
except:
    print("  ⚠️ Could not save model (joblib not installed)")

# ============================================
# STEP 8: SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)

if ensemble_acc >= 0.60:
    print(f"\n✅ SUCCESS! Achieved {ensemble_acc*100:.2f}% accuracy!")
    print("   This is professional-level performance!")
    print("\n🚀 Improvement from baseline:")
    print(f"   Random guess: 33.3%")
    print(f"   Your model:   {ensemble_acc*100:.2f}%")
    print(f"   Gain:         +{(ensemble_acc - 0.333)*100:.1f}%")
    
elif ensemble_acc >= 0.55:
    print(f"\n⚠️ Good progress: {ensemble_acc*100:.2f}%")
    print("   Close to target! To improve:")
    print("   1. Add more data (AAPL.csv)")
    print("   2. Try different parameters")
    
else:
    print(f"\n⚠️ Accuracy: {ensemble_acc*100:.2f}%")
    print("   Below expectations. Check:")
    print("   1. Data quality")
    print("   2. Feature engineering")

print("\n✅ Next steps:")
print("1. Use the trained model in your dashboard")
print("2. Add more stocks for better generalization")
print("3. Monitor performance on live data")