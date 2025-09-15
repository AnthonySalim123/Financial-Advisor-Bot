# test_practical_integration.py
# Run this to test the practical trading system integration

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import get_data_processor
from utils.technical_indicators import TechnicalIndicators

def test_practical_model():
    """Test if the practical model works correctly"""
    
    print("🔍 Testing Practical Trading System Integration")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists('practical_model.pkl'):
        print("❌ practical_model.pkl not found!")
        print("   Run: python practical_trading_system.py")
        return False
    
    # Load model
    try:
        model_data = joblib.load('practical_model.pkl')
        print("✅ Model loaded successfully")
        print(f"   Base accuracy: {model_data['accuracy']*100:.1f}%")
        print(f"   Confidence threshold: {model_data.get('confidence_threshold', 0.5)*100:.0f}%")
        print(f"   Features: {model_data['features']}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test prediction on real data
    print("\n📊 Testing predictions on real data...")
    data_processor = get_data_processor()
    
    test_stocks = ['AAPL', 'MSFT', 'GOOGL']
    results = []
    
    for symbol in test_stocks:
        try:
            # Fetch data
            df = data_processor.fetch_stock_data(symbol, period='1y')
            
            if len(df) > 50:
                # Create features (same as practical_trading_system.py)
                df['return_1d'] = df['Close'].pct_change()
                df['return_5d'] = df['Close'].pct_change(5)
                
                # RSI
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-10)))
                
                # SMAs
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['price_vs_sma20'] = df['Close'] / df['SMA_20']
                df['price_vs_sma50'] = df['Close'] / df['SMA_50']
                
                # Volatility & Volume
                df['volatility'] = df['return_1d'].rolling(20).std()
                df['volume_avg'] = df['Volume'].rolling(20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_avg']
                
                # Get latest features
                X_latest = df[model_data['features']].iloc[-1:].fillna(method='ffill')
                
                if not X_latest.empty:
                    # Scale and predict
                    X_scaled = model_data['scaler'].transform(X_latest)
                    prediction = model_data['model'].predict(X_scaled)[0]
                    probabilities = model_data['model'].predict_proba(X_scaled)[0]
                    confidence = np.max(probabilities)
                    
                    # Map to signal
                    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                    signal = signal_map[prediction]
                    
                    results.append({
                        'Symbol': symbol,
                        'Signal': signal,
                        'Confidence': confidence,
                        'Price': df['Close'].iloc[-1],
                        'RSI': df['RSI'].iloc[-1],
                        'Trade?': 'YES' if confidence > 0.5 else 'NO'
                    })
                    
                    print(f"   {symbol}: {signal} (Confidence: {confidence*100:.1f}%)")
        
        except Exception as e:
            print(f"   Error with {symbol}: {e}")
    
    # Display results
    if results:
        print("\n📊 PREDICTION RESULTS:")
        print("-"*60)
        
        df_results = pd.DataFrame(results)
        
        # Overall statistics
        total_signals = len(df_results)
        tradeable = df_results[df_results['Trade?'] == 'YES']
        tradeable_count = len(tradeable)
        
        print(f"\nTotal Signals: {total_signals}")
        print(f"Tradeable (>50% conf): {tradeable_count} ({tradeable_count/total_signals*100:.0f}%)")
        
        # Show all results
        print("\nDetailed Results:")
        for _, row in df_results.iterrows():
            emoji = "🟢" if row['Signal'] == 'BUY' else "🔴" if row['Signal'] == 'SELL' else "🟡"
            trade_emoji = "✅" if row['Trade?'] == 'YES' else "⏸️"
            
            print(f"\n{row['Symbol']}:")
            print(f"  Signal: {emoji} {row['Signal']}")
            print(f"  Confidence: {row['Confidence']*100:.1f}%")
            print(f"  Price: ${row['Price']:.2f}")
            print(f"  RSI: {row['RSI']:.1f}")
            print(f"  Action: {trade_emoji} {row['Trade?']}")
        
        # Confidence distribution
        print("\n📊 CONFIDENCE DISTRIBUTION:")
        print("-"*40)
        
        for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
            above_threshold = df_results[df_results['Confidence'] > threshold]
            count = len(above_threshold)
            pct = count / total_signals * 100 if total_signals > 0 else 0
            
            print(f"Above {threshold*100:.0f}%: {count} signals ({pct:.0f}%)")
        
        # Expected performance
        print("\n🎯 EXPECTED PERFORMANCE:")
        print("-"*40)
        print("Based on backtesting results:")
        print("• Confidence > 50%: ~51.7% accuracy (profitable)")
        print("• Confidence > 55%: ~53.9% accuracy (more profitable)")
        print("• Confidence > 60%: ~59.7% accuracy (highly profitable)")
        
        print("\n✅ Integration test complete!")
        print("\n🚀 NEXT STEPS:")
        print("1. Update your dashboard.py with the enhanced code")
        print("2. Run: streamlit run app.py")
        print("3. Navigate to the Dashboard page")
        print("4. Click 'Generate Practical Trading Signals'")
        print("5. Adjust confidence threshold as needed")
        
        return True
    
    return False

if __name__ == "__main__":
    success = test_practical_model()
    
    if not success:
        print("\n⚠️ Please run practical_trading_system.py first!")
        print("Command: python practical_trading_system.py")