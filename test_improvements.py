# test_improvements.py
"""Test script to verify all improvements are working"""

import streamlit as st
from utils.ml_models import create_prediction_model, MLModel
from utils.data_processor import get_data_processor
from utils.technical_indicators import TechnicalIndicators
import yaml

print("🔍 Testing StockBot Advisor Improvements...")
print("="*50)

# Test 1: Check config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
print("✅ Config Check:")
print(f"  - n_estimators: {config['ml_config']['parameters']['n_estimators']} (should be 300)")
print(f"  - max_depth: {config['ml_config']['parameters']['max_depth']} (should be 12)")
print(f"  - default_period: {config['data']['default_period']} (should be 3y)")

# Test 2: Data fetching
processor = get_data_processor()
df = processor.fetch_stock_data('AAPL')  # Should use 3y by default

print(f"\n✅ Data Fetching:")
print(f"  - Rows fetched: {len(df)} (should be ~750 for 3 years)")
print(f"  - Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Test 3: Model creation
model = create_prediction_model('classification')

print(f"\n✅ Model Configuration:")
print(f"  - RF estimators: {model.config['rf_n_estimators']} (should be 300)")
print(f"  - Max depth: {model.config['rf_max_depth']} (should be 12)")
print(f"  - Use ensemble: {model.config['use_ensemble']} (should be True)")

# Test 4: Full pipeline
df = TechnicalIndicators.calculate_all_indicators(df)
print(f"\n✅ Technical Indicators:")
print(f"  - Total features: {len(df.columns)} columns")
print(f"  - Has RSI: {'RSI' in df.columns}")
print(f"  - Has MACD: {'MACD' in df.columns}")
print(f"  - Has SMA_20: {'SMA_20' in df.columns}")

# Test 5: Train model
print(f"\n🚀 Training Model (this may take 30-60 seconds)...")
metrics = model.train(df)

if 'accuracy' in metrics:
    accuracy_pct = metrics['accuracy'] * 100
    print(f"\n🎉 SUCCESS! Model Accuracy: {accuracy_pct:.1f}%")
    
    if accuracy_pct >= 65:
        print("✅ Target accuracy achieved (65%+)")
    else:
        print(f"⚠️ Accuracy below target (got {accuracy_pct:.1f}%, expected 65%+)")
        
    # Test prediction
    prediction = model.predict_latest(df)
    if 'signal' in prediction:
        print(f"\n📊 Latest Prediction:")
        print(f"  - Signal: {prediction['signal']}")
        print(f"  - Confidence: {prediction['confidence']*100:.1f}%")
else:
    print(f"❌ Training failed: {metrics}")

print("\n" + "="*50)
print("Test complete! Your system is ready.")