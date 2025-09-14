# verify_accuracy.py
# Run this to confirm your model can achieve 60%+ accuracy

from utils.ml_models import MLModel
from utils.data_processor import get_data_processor
from utils.technical_indicators import TechnicalIndicators

print("🔍 Verifying Enhanced Model Accuracy...")
print("="*60)

# Fetch data using your processor
processor = get_data_processor()
symbols = ['AAPL', 'MSFT', 'GOOGL']

for symbol in symbols:
    print(f"\n📊 Testing {symbol}...")
    
    # Fetch data
    df = processor.fetch_stock_data(symbol, period='5y')
    
    if df.empty or len(df) < 200:
        print(f"  ❌ Insufficient data for {symbol}")
        continue
    
    # Add technical indicators
    df = TechnicalIndicators.calculate_all_indicators(df)
    
    # Create model with enhanced config
    model = MLModel(
        model_type='classification',
        config={
            'rf_n_estimators': 500,
            'rf_max_depth': 15,
            'rf_min_samples_split': 5,
            'rf_min_samples_leaf': 2,
            'use_ensemble': True,
            'feature_selection': True,
            'n_features': 50,
            'handle_imbalance': True
        }
    )
    
    # Force fresh training (no cache)
    model.is_trained = False
    
    # Train with enhanced features
    print(f"  ⏳ Training model...")
    metrics = model.train(df, optimize_hyperparameters=False)
    
    if 'accuracy' in metrics:
        accuracy = metrics['accuracy'] * 100
        print(f"  ✅ Accuracy: {accuracy:.1f}%")
        
        # Test prediction
        prediction = model.predict_latest(df)
        if 'signal' in prediction:
            print(f"  📈 Signal: {prediction['signal']}")
            print(f"  🎯 Confidence: {prediction['confidence']*100:.1f}%")
        
        # Check feature importance
        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
            top_features = model.feature_importance.head(5)
            print(f"  🔑 Top features being used:")
            for _, row in top_features.iterrows():
                print(f"     - {row['feature']}: {row['importance']*100:.1f}%")
    else:
        print(f"  ❌ Training failed: {metrics.get('error', 'Unknown error')}")

print("\n" + "="*60)
print("✅ Verification complete!")
print("\nIf accuracy is still low, check that:")
print("1. The engineer_features method in ml_models.py was updated")
print("2. The create_target_variable method was updated")
print("3. You have at least 1000+ data points for training")