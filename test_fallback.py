"""Test the fallback data system"""
import sys
sys.path.append('.')

from utils.data_processor import get_data_processor
from utils.technical_indicators import TechnicalIndicators

print("Testing fallback data system...")

# Get data processor
processor = get_data_processor()

# Test fetching data
symbols = ['AAPL', 'MSFT', 'GOOGL']

for symbol in symbols:
    print(f"\nTesting {symbol}:")
    
    # Fetch data (will use fallback automatically)
    df = processor.fetch_stock_data(symbol, period='1y')
    
    if not df.empty:
        print(f"  ✓ Data fetched: {len(df)} days")
        print(f"  ✓ Latest price: ${df['Close'].iloc[-1]:.2f}")
        
        # Calculate indicators
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
        print(f"  ✓ Indicators calculated")
        
        # Check for required columns
        required = ['RSI', 'MACD', 'SMA_20']
        missing = [col for col in required if col not in df_with_indicators.columns]
        if not missing:
            print(f"  ✓ All required indicators present")
        else:
            print(f"  ✗ Missing indicators: {missing}")
    else:
        print(f"  ✗ Failed to fetch data")

print("\n" + "="*50)
print("Fallback system test complete!")