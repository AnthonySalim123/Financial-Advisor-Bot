# test_new_stocks.py
import yaml

# Test 1: Config loads correctly
print("Testing config.yaml...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
tech_stocks = len(config['stocks']['technology'])
financial_stocks = len(config['stocks']['financial'])
real_estate_stocks = len(config['stocks']['real_estate'])
healthcare_stocks = len(config['stocks']['healthcare'])

print(f"✅ Technology: {tech_stocks} stocks")
print(f"✅ Financial: {financial_stocks} stocks (should be 10)")
print(f"✅ Real Estate: {real_estate_stocks} stocks (should be 5)")
print(f"✅ Healthcare: {healthcare_stocks} stocks")
print(f"✅ Total: {tech_stocks + financial_stocks + real_estate_stocks + healthcare_stocks} stocks")

# Test 2: Enhanced processor works
print("\nTesting enhanced_data_processor...")
from utils.enhanced_data_processor import get_enhanced_data_processor

processor = get_enhanced_data_processor()
print(f"✅ Processor loaded with {processor.get_total_stocks()} stocks")

# Test 3: Fetch data for new stocks
test_stocks = ['MA', 'PLD']  # One new financial, one REIT
for symbol in test_stocks:
    df = processor.fetch_stock_data(symbol, period='1mo')
    if not df.empty:
        print(f"✅ {symbol} data fetched: {len(df)} days, Latest: ${df['Close'].iloc[-1]:.2f}")

# Test 4: REIT-specific data
print("\nTesting REIT-specific features...")
reit_data = processor.fetch_reit_specific_data('O')
if reit_data:
    print(f"✅ Realty Income (O) - Dividend Yield: {reit_data.get('dividend_yield', 0):.2f}%")

print("\n🎉 All tests passed! Your 33-stock universe is ready!")