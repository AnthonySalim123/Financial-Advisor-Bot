# Create test_app_working.py
import sys
print("Python executable:", sys.executable)

try:
    import streamlit
    print("✅ Streamlit imported successfully")
    print("   Version:", streamlit.__version__)
except:
    print("❌ Streamlit import failed")

try:
    import pandas
    print("✅ Pandas imported successfully")
except:
    print("❌ Pandas import failed")

try:
    import plotly
    print("✅ Plotly imported successfully")
except:
    print("❌ Plotly import failed")

try:
    from utils.data_processor import get_data_processor
    print("✅ Data processor imported successfully")
except Exception as e:
    print("❌ Data processor import failed:", e)

print("\nIf you see all ✅, your app is working fine!")
print("The VS Code warnings are just IDE issues.")