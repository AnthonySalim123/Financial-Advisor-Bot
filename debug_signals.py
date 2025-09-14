# debug_signals.py
# This will show us exactly what's wrong with your signals

from utils.ml_models import MLModel
from utils.data_processor import get_data_processor
from utils.technical_indicators import TechnicalIndicators
import pandas as pd
import numpy as np

print("🔍 Debugging Signal Generation...")
print("="*60)

# Test 1: Check signal distribution
processor = get_data_processor()
df = processor.fetch_stock_data('AAPL', period='2y')

if not df.empty:
    print(f"✅ Loaded {len(df)} days of data")
    
    # Add indicators
    df = TechnicalIndicators.calculate_all_indicators(df)
    
    # Create model
    model = MLModel('classification')
    
    # Test the create_target_variable method
    print("\n📊 Testing Signal Creation:")
    signals = model.create_target_variable(df)
    
    # Check distribution
    distribution = signals.value_counts(normalize=True).sort_index()
    print("\nSignal Distribution:")
    print(f"  SELL (0): {distribution.get(0, 0)*100:.1f}%")
    print(f"  HOLD (1): {distribution.get(1, 0)*100:.1f}%")
    print(f"  BUY (2): {distribution.get(2, 0)*100:.1f}%")
    
    # Check if it's imbalanced
    if distribution.get(1, 0) > 0.6:
        print("\n❌ PROBLEM FOUND: Too many HOLD signals!")
        print("   The create_target_variable method is NOT fixed properly")
        
        # Show what thresholds are being used
        future_returns = df['Close'].pct_change(periods=5).shift(-5)
        volatility = df['Close'].pct_change().rolling(20).std()
        
        print(f"\n📈 Debugging Info:")
        print(f"   Average volatility: {volatility.mean()*100:.2f}%")
        print(f"   Average future returns: {future_returns.mean()*100:.2f}%")
        print(f"   Future returns std: {future_returns.std()*100:.2f}%")
        
        # Test with correct thresholds
        print("\n🔧 Testing with CORRECT thresholds:")
        rolling_std = future_returns.rolling(window=60, min_periods=30).std()
        rolling_mean = future_returns.rolling(window=60, min_periods=30).mean()
        
        upper_threshold = rolling_mean + (0.5 * rolling_std)
        lower_threshold = rolling_mean - (0.5 * rolling_std)
        
        upper_threshold = upper_threshold.fillna(0.01)
        lower_threshold = lower_threshold.fillna(-0.01)
        
        correct_signals = pd.Series(1, index=df.index)
        correct_signals[future_returns > upper_threshold] = 2
        correct_signals[future_returns < lower_threshold] = 0
        
        correct_dist = correct_signals.value_counts(normalize=True).sort_index()
        print("\nCORRECT Signal Distribution Should Be:")
        print(f"  SELL (0): {correct_dist.get(0, 0)*100:.1f}%")
        print(f"  HOLD (1): {correct_dist.get(1, 0)*100:.1f}%")
        print(f"  BUY (2): {correct_dist.get(2, 0)*100:.1f}%")
        
    else:
        print("\n✅ Signal distribution looks balanced!")
    
    # Test 2: Check features being created
    print("\n📊 Testing Feature Engineering:")
    features_df = model.engineer_features(df)
    
    print(f"Total features created: {len(features_df.columns)}")
    
    # Check for key features
    expected_features = [
        'Distance_from_High_20',
        'Distance_from_Low_20', 
        'RSI_14',
        'RSI_7',
        'BB_Position_20',
        'BB_Position_30',
        'Momentum_10',
        'Price_to_SMA_50',
        'ATR_14',
        'MFI_14'
    ]
    
    missing_features = []
    for feat in expected_features:
        if feat not in features_df.columns:
            missing_features.append(feat)
    
    if missing_features:
        print(f"\n❌ Missing important features: {missing_features}")
        print("   The engineer_features method is NOT fully updated")
    else:
        print("\n✅ All key features are present!")
    
    # List actual features
    print(f"\nFirst 20 features found:")
    for i, col in enumerate(features_df.columns[:20]):
        print(f"  {i+1}. {col}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("-"*60)

print("""
If you see:
1. Too many HOLD signals (>60%) → create_target_variable NOT fixed
2. Missing features → engineer_features NOT fully updated  
3. <30 total features → Using OLD feature engineering

SOLUTION:
Double-check that you replaced the ENTIRE methods in ml_models.py,
not just parts of them.
""")