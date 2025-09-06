"""
Centralized session state management
"""

def initialize_portfolio():
    """Initialize portfolio with consistent structure"""
    return {
        'holdings': {
            'AAPL': {'shares': 50, 'avg_price': 150.00},
            'MSFT': {'shares': 30, 'avg_price': 350.00},
            'GOOGL': {'shares': 20, 'avg_price': 135.00},
            'NVDA': {'shares': 25, 'avg_price': 450.00}
        },
        'cash': 25000,
        'total_value': 125430,
        'daily_return': 1.9,
        'total_return': 25.4
    }

def init_session_state():
    """Initialize all session state variables"""
    import streamlit as st
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = initialize_portfolio()
    
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Anthony Salim',
            'risk_tolerance': 'Moderate',
            'investment_horizon': '1-3 years',
            'member_since': '2025',
            'initial_capital': 100000,
            'currency': 'USD'
        }
    
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []
    
    if 'last_update' not in st.session_state:
        from datetime import datetime
        st.session_state.last_update = datetime.now()