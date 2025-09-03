"""
StockBot Advisor - Main Application
Author: Anthony Winata Salim
Student Number: 230726051
Course: CM3070 Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration with minimalistic theme
st.set_page_config(
    page_title="StockBot Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # StockBot Advisor
        AI-Powered Financial Advisory Platform
        
        **Author:** Anthony Winata Salim  
        **Student Number:** 230726051  
        **Course:** CM3070 Project
        """
    }
)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from yaml file"""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Custom CSS for minimalistic design
def load_custom_css():
    """Apply custom CSS for minimalistic black/white/gray theme"""
    st.markdown("""
    <style>
        /* Import Inter font for modern look */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main container */
        .main {
            padding: 2rem;
            background-color: #FFFFFF;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #000000;
            font-weight: 300;
            letter-spacing: -0.02em;
        }
        
        h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid #E9ECEF;
            padding-bottom: 1rem;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #F8F9FA;
            border-right: 1px solid #E9ECEF;
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding: 2rem 1rem;
        }
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border: 1px solid #E9ECEF;
            padding: 1rem;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin: 0.5rem 0;
        }
        
        div[data-testid="metric-container"] label {
            color: #6C757D;
            font-size: 0.875rem;
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        div[data-testid="metric-container"] div[data-testid="metric-value"] {
            color: #000000;
            font-size: 1.5rem;
            font-weight: 500;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #000000;
            border-radius: 4px;
            padding: 0.5rem 1.5rem;
            font-weight: 400;
            transition: all 0.2s ease;
            text-transform: none;
            letter-spacing: 0;
        }
        
        .stButton > button:hover {
            background-color: #000000;
            color: #FFFFFF;
            border-color: #000000;
        }
        
        /* Primary button style */
        .stButton > button[kind="primary"] {
            background-color: #000000;
            color: #FFFFFF;
            border: 1px solid #000000;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #212529;
            border-color: #212529;
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stMultiSelect > div > div > div {
            border: none;
            border-bottom: 1px solid #E9ECEF;
            border-radius: 0;
            padding: 0.5rem 0;
            background-color: transparent;
            color: #000000;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-bottom: 2px solid #000000;
            box-shadow: none;
        }
        
        /* Tables */
        .dataframe {
            border: none;
            font-size: 0.875rem;
        }
        
        .dataframe thead tr th {
            background-color: #F8F9FA;
            border-bottom: 2px solid #E9ECEF;
            color: #000000;
            font-weight: 500;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid #E9ECEF;
        }
        
        .dataframe tbody tr:hover {
            background-color: #F8F9FA;
        }
        
        .dataframe tbody tr td {
            color: #212529;
            font-family: 'SF Mono', Monaco, monospace;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            border-bottom: 1px solid #E9ECEF;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            padding: 0 1rem;
            background-color: transparent;
            border: none;
            color: #6C757D;
            font-weight: 400;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            border-bottom: 2px solid #000000;
            color: #000000;
            font-weight: 500;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #F8F9FA;
            border: 1px solid #E9ECEF;
            border-radius: 4px;
            color: #000000;
            font-weight: 400;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #E9ECEF;
        }
        
        /* Info boxes */
        .stAlert {
            background-color: #F8F9FA;
            border: 1px solid #E9ECEF;
            border-radius: 4px;
            color: #212529;
        }
        
        /* Success/Error colors */
        .success-text {
            color: #28A745;
            font-weight: 500;
        }
        
        .error-text {
            color: #DC3545;
            font-weight: 500;
        }
        
        /* Plotly charts */
        .js-plotly-plot {
            border: 1px solid #E9ECEF;
            border-radius: 4px;
            padding: 1rem;
            background-color: #FFFFFF;
        }
        
        /* Remove streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F8F9FA;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #6C757D;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #000000;
        }
        
        /* Loading animation */
        .stSpinner > div {
            border-color: #000000;
        }
        
        /* Divider */
        hr {
            border: none;
            border-top: 1px solid #E9ECEF;
            margin: 2rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Guest User',
            'risk_tolerance': 'Moderate',
            'investment_horizon': '1-3 years',
            'initial_capital': 100000,
            'currency': 'USD'
        }
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'holdings': {},
            'cash': 100000,
            'total_value': 100000,
            'daily_return': 0,
            'total_return': 0
        }
    
    if 'watchlist' not in st.session_state:
        # Initialize with 5 popular stocks
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    if 'theme' not in st.session_state:
        st.session_state.theme = 'minimal'

# Cache functions for data fetching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period='1d'):
    """Fetch stock data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        info = ticker.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

@st.cache_data(ttl=300)
def fetch_market_overview():
    """Fetch market indices data"""
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        'SPY': 'SPY ETF'
    }
    
    market_data = {}
    for symbol, name in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d')
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - prev) / prev) * 100 if prev != 0 else 0
                market_data[name] = {
                    'value': current,
                    'change': change
                }
        except:
            market_data[name] = {'value': 0, 'change': 0}
    
    return market_data

# Sidebar components
def render_sidebar():
    """Render the sidebar with user profile and navigation"""
    with st.sidebar:
        # Logo/Title
        st.markdown("# 📊 StockBot Advisor")
        st.markdown("---")
        
        # User Profile Section
        st.markdown("### User Profile")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**{st.session_state.user_profile['name']}**")
        with col2:
            if st.button("Edit", key="edit_profile"):
                st.session_state.show_profile_edit = True
        
        st.caption(f"Risk: {st.session_state.user_profile['risk_tolerance']}")
        st.caption(f"Horizon: {st.session_state.user_profile['investment_horizon']}")
        
        st.markdown("---")
        
        # Portfolio Summary
        st.markdown("### Portfolio Summary")
        st.metric("Total Value", f"${st.session_state.portfolio['total_value']:,.2f}")
        col1, col2 = st.columns(2)
        with col1:
            change_color = "🟢" if st.session_state.portfolio['daily_return'] >= 0 else "🔴"
            st.caption(f"{change_color} Daily: {st.session_state.portfolio['daily_return']:.2f}%")
        with col2:
            change_color = "🟢" if st.session_state.portfolio['total_return'] >= 0 else "🔴"
            st.caption(f"{change_color} Total: {st.session_state.portfolio['total_return']:.2f}%")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("Report generation coming soon!")
        
        if st.button("🎯 Run Analysis", use_container_width=True):
            st.info("Running analysis...")
        
        st.markdown("---")
        
        # Market Status
        st.markdown("### Market Status")
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        
        if now.weekday() < 5 and market_open <= now <= market_close:
            st.success("🟢 Market Open")
        else:
            st.error("🔴 Market Closed")
        
        st.caption(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # Footer
        st.caption("CM3070 Project")
        st.caption("© 2025 Anthony Winata Salim")

# Main dashboard content
def render_dashboard():
    """Render the main dashboard"""
    # Header
    st.markdown("# Stock Market Dashboard")
    st.markdown("Real-time market data and AI-powered insights")
    
    # Market Overview
    st.markdown("---")
    st.markdown("## Market Overview")
    
    market_data = fetch_market_overview()
    # Display market indices
    if market_data and len(market_data) > 0:
        cols = st.columns(len(market_data))
        for i, (name, data) in enumerate(market_data.items()):
            with cols[i]:
                change_symbol = "↑" if data['change'] >= 0 else "↓"
                change_color = "green" if data['change'] >= 0 else "red"
                st.metric(
                    label=name,
                    value=f"${data['value']:,.2f}",
                    delta=f"{change_symbol} {abs(data['change']):.2f}%"
                )
    else:
        # Fallback display when no market data
        st.info("Market data loading... Using synthetic data")
        # Display sample market data
        sample_data = {
            'S&P 500': {'value': 4782.15, 'change': 0.85},
            'NASDAQ': {'value': 15123.45, 'change': 1.23},
            'DOW': {'value': 38456.78, 'change': 0.45},
            'VIX': {'value': 18.65, 'change': -2.15}
        }
        cols = st.columns(len(sample_data))
        for i, (name, data) in enumerate(sample_data.items()):
            with cols[i]:
                st.metric(
                    label=name,
                    value=f"${data['value']:,.2f}",
                    delta=f"{data['change']:+.2f}%"
                )
    
    # Tabs for different views
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Watchlist", "🎯 AI Signals", "📊 Top Movers", "📰 Market News"])
    
    with tab1:
        render_watchlist()
    
    with tab2:
        render_ai_signals()
    
    with tab3:
        render_top_movers()
    
    with tab4:
        render_market_news()

def render_watchlist():
    """Render watchlist component"""
    st.markdown("### Your Watchlist")
    
    # Add stock to watchlist
    col1, col2 = st.columns([3, 1])
    with col1:
        new_symbol = st.text_input("Add Symbol", placeholder="Enter stock symbol...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", key="add_watchlist"):
            if new_symbol and new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.success(f"Added {new_symbol.upper()} to watchlist")
                st.rerun()
    
    # Display watchlist
    if st.session_state.watchlist:
        watchlist_data = []
        
        for symbol in st.session_state.watchlist:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = ((current - prev) / prev) * 100 if prev != 0 else 0
                    volume = hist['Volume'].iloc[-1]
                    
                    watchlist_data.append({
                        'Symbol': symbol,
                        'Price': f"${current:.2f}",
                        'Change': f"{change:+.2f}%",
                        'Volume': f"{volume:,.0f}",
                        'Action': '🗑️'
                    })
            except:
                continue
        
        if watchlist_data:
            df = pd.DataFrame(watchlist_data)
            
            # Display with custom styling
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Price": st.column_config.TextColumn("Price", width="small"),
                    "Change": st.column_config.TextColumn("Change", width="small"),
                    "Volume": st.column_config.TextColumn("Volume", width="medium"),
                    "Action": st.column_config.TextColumn("", width="small")
                }
            )
    else:
        st.info("Your watchlist is empty. Add stocks to track them here.")

def render_ai_signals():
    """Render AI trading signals"""
    st.markdown("### AI Trading Signals")
    
    # Sample signals (in production, these would come from ML model)
    signals = [
        {'Stock': 'AAPL', 'Signal': 'BUY', 'Confidence': 78, 'Reason': 'RSI oversold, MACD bullish crossover'},
        {'Stock': 'MSFT', 'Signal': 'HOLD', 'Confidence': 65, 'Reason': 'Mixed indicators, await confirmation'},
        {'Stock': 'GOOGL', 'Signal': 'SELL', 'Confidence': 72, 'Reason': 'Overbought conditions, divergence detected'},
    ]
    
    for signal in signals:
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            if signal['Signal'] == 'BUY':
                st.markdown(f"**{signal['Stock']}** - <span class='success-text'>BUY</span>", unsafe_allow_html=True)
            elif signal['Signal'] == 'SELL':
                st.markdown(f"**{signal['Stock']}** - <span class='error-text'>SELL</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**{signal['Stock']}** - HOLD")
        
        with col2:
            st.progress(signal['Confidence'] / 100)
            st.caption(f"{signal['Confidence']}% confidence")
        
        with col3:
            st.caption(signal['Reason'])
        
        st.markdown("---")

def render_top_movers():
    """Render top movers section"""
    st.markdown("### Top Movers")
    
    config = st.session_state.config
    all_stocks = []
    
    # Collect all stocks from config
    for sector in ['technology', 'financial', 'healthcare']:
        if sector in config['stocks']:
            all_stocks.extend([s['symbol'] for s in config['stocks'][sector]])
    
    # Get random movers for demonstration
    movers_data = []
    for _ in range(5):
        if all_stocks:
            symbol = np.random.choice(all_stocks)
            change = np.random.uniform(-5, 5)
            movers_data.append({
                'Symbol': symbol,
                'Change': f"{change:+.2f}%",
                'Volume': f"{np.random.randint(1000000, 10000000):,}"
            })
    
    if movers_data:
        df = pd.DataFrame(movers_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_market_news():
    """Render market news section"""
    st.markdown("### Latest Market News")
    
    # Sample news (in production, this would come from news API)
    news_items = [
        {
            'title': 'Fed Signals Potential Rate Cuts in Q2 2025',
            'time': '2 hours ago',
            'summary': 'Federal Reserve officials hint at possible rate adjustments amid cooling inflation...'
        },
        {
            'title': 'Tech Stocks Rally on AI Optimism',
            'time': '4 hours ago',
            'summary': 'Major technology companies see gains as AI adoption accelerates across industries...'
        },
        {
            'title': 'Energy Sector Under Pressure',
            'time': '6 hours ago',
            'summary': 'Oil prices decline as supply concerns ease and demand forecasts are revised...'
        }
    ]
    
    for news in news_items:
        st.markdown(f"**{news['title']}**")
        st.caption(news['time'])
        st.write(news['summary'])
        st.markdown("---")

# Main application
def main():
    # Initialize
    init_session_state()
    load_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_dashboard()

if __name__ == "__main__":
    main()