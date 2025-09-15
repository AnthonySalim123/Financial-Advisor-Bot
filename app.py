"""
StockBot Advisor - Main Application
Financial Advisory Bot with AI-Powered Stock Analysis
Author: Anthony Winata Salim
Student Number: 230726051
Course: CM3070 Project
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import json
import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules with error handling
try:
    from utils.data_processor import get_data_processor
except ImportError:
    from utils.data_processor import StockDataProcessor
    def get_data_processor():
        return StockDataProcessor()

try:
    from utils.technical_indicators import TechnicalIndicators
except ImportError:
    print("Warning: TechnicalIndicators module not found")
    TechnicalIndicators = None

try:
    from utils.ml_models import create_prediction_model
except ImportError:
    print("Warning: ML models module not found")
    create_prediction_model = None

# Fix the LLM explainer import
try:
    from utils.llm_explainer import LLMExplainer  # Changed from get_llm_explainer
    llm_explainer = LLMExplainer()  # Create instance if needed
except ImportError:
    print("Warning: LLM explainer module not found")
    llm_explainer = None

# Optional database import
try:
    from utils.database import get_database
except ImportError:
    print("Warning: Database module not found")
    get_database = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="StockBot Advisor - AI-Powered Investment Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables with proper defaults"""
    
    # Initialize portfolio with ALL required keys
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'holdings': {},  # {symbol: {'shares': float, 'avg_cost': float}}
            'cash': 100000.0,
            'total_value': 100000.0,
            'daily_return': 0.0,
            'daily_return_pct': 0.0,
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'previous_total_value': 100000.0,
            'transactions': [],
            'last_updated': datetime.now().isoformat()
        }
    else:
        # Ensure all required keys exist (for backward compatibility)
        portfolio_defaults = {
            'holdings': {},
            'cash': 100000.0,
            'total_value': 100000.0,
            'daily_return': 0.0,
            'daily_return_pct': 0.0,
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'previous_total_value': 100000.0,
            'transactions': [],
            'last_updated': datetime.now().isoformat()
        }
        
        for key, default_value in portfolio_defaults.items():
            if key not in st.session_state.portfolio:
                st.session_state.portfolio[key] = default_value
    
    # Initialize user profile
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Guest User',
            'email': 'guest@stockbot.com',
            'risk_tolerance': 'Moderate',
            'investment_horizon': '1-3 years',
            'initial_capital': 100000.0,
            'currency': 'USD',
            'experience_level': 'Intermediate',
            'created_at': datetime.now().isoformat()
        }
    
    # Initialize watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Initialize market data cache
    if 'market_data_cache' not in st.session_state:
        st.session_state.market_data_cache = {}
    
    # Initialize predictions cache
    if 'predictions_cache' not in st.session_state:
        st.session_state.predictions_cache = {}
    
    # Initialize UI state variables
    if 'show_profile_editor' not in st.session_state:
        st.session_state.show_profile_editor = False
    
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False
    
    # Initialize selected page
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = 'Dashboard'
    
    # Initialize backtest results
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    
    # Initialize analysis preferences
    if 'analysis_preferences' not in st.session_state:
        st.session_state.analysis_preferences = {
            'selected_stock': 'AAPL',
            'analysis_period': '1y',
            'show_indicators': True,
            'show_ml_predictions': True,
            'chart_type': 'candlestick'
        }

def apply_custom_css():
    """Apply custom CSS for minimalistic design"""
    st.markdown("""
    <style>
    /* Main theme - Minimalistic Black & White */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8F9FA;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #000000;
        font-family: 'Inter', 'SF Pro Display', sans-serif;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #000000;
        color: #FFFFFF;
        border: none;
        padding: 8px 20px;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #333333;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border: 1px solid #E5E5E5;
        border-radius: 6px;
        padding: 8px 12px;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        border: 1px solid #E5E5E5;
        border-radius: 6px;
    }
    
    /* Data frames */
    .dataframe {
        border: none !important;
        font-family: 'SF Mono', monospace;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #D4EDDA;
        color: #155724;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #28A745;
    }
    
    .stError {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #DC3545;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0px 20px;
        background-color: transparent;
        border-radius: 6px;
        color: #666666;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        border-radius: 6px;
        border: 1px solid #E5E5E5;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F8F9FA;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)

def update_portfolio_values():
    """Update portfolio values including daily returns"""
    try:
        data_processor = get_data_processor()
        
        # Ensure portfolio exists
        if 'portfolio' not in st.session_state:
            initialize_session_state()
        
        portfolio = st.session_state.portfolio
        
        # Calculate current portfolio value
        total_value = portfolio.get('cash', 100000.0)
        
        # Add value of holdings
        holdings = portfolio.get('holdings', {})
        for symbol, holding in holdings.items():
            try:
                # Get current price
                df = data_processor.fetch_stock_data(symbol, period='2d')
                
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    previous_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                    
                    # Calculate position value
                    shares = holding.get('shares', 0)
                    current_value = shares * current_price
                    previous_value = shares * previous_price
                    
                    # Add to total
                    total_value += current_value
                    
                    # Track daily change for this position
                    holding['current_price'] = current_price
                    holding['current_value'] = current_value
                    holding['daily_change'] = current_value - previous_value
                    
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                # Use last known value if available
                if 'current_value' in holding:
                    total_value += holding['current_value']
        
        # Calculate daily return
        previous_total = portfolio.get('previous_total_value', portfolio.get('total_value', 100000.0))
        daily_return = total_value - previous_total
        daily_return_pct = (daily_return / previous_total * 100) if previous_total > 0 else 0
        
        # Calculate total return from initial capital
        initial_capital = st.session_state.user_profile.get('initial_capital', 100000.0)
        total_return = total_value - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0
        
        # Update portfolio
        portfolio.update({
            'total_value': total_value,
            'previous_total_value': previous_total,
            'daily_return': daily_return,
            'daily_return_pct': daily_return_pct,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating portfolio values: {e}")

def render_sidebar():
    """Render the sidebar with user profile and portfolio info"""
    with st.sidebar:
        st.markdown("## 📈 StockBot Advisor")
        st.markdown("---")
        
        # User Profile Section
        st.markdown("### 👤 User Profile")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{st.session_state.user_profile.get('name', 'Guest User')}**")
        with col2:
            if st.button("Edit", key="edit_profile"):
                st.session_state.show_profile_editor = not st.session_state.get('show_profile_editor', False)
        
        st.write(f"Risk: {st.session_state.user_profile.get('risk_tolerance', 'Moderate')}")
        st.write(f"Horizon: {st.session_state.user_profile.get('investment_horizon', '1-3 years')}")
        
        # Show profile editor if requested
        if st.session_state.get('show_profile_editor', False):
            with st.expander("Edit Profile", expanded=True):
                new_name = st.text_input("Name", value=st.session_state.user_profile.get('name', 'Guest User'))
                new_risk = st.select_slider(
                    "Risk Tolerance",
                    options=["Conservative", "Moderate", "Aggressive"],
                    value=st.session_state.user_profile.get('risk_tolerance', 'Moderate')
                )
                new_horizon = st.selectbox(
                    "Investment Horizon",
                    ["< 1 year", "1-3 years", "3-5 years", "> 5 years"],
                    index=1
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save", key="save_profile"):
                        st.session_state.user_profile.update({
                            'name': new_name,
                            'risk_tolerance': new_risk,
                            'investment_horizon': new_horizon
                        })
                        st.session_state.show_profile_editor = False
                        st.rerun()
                with col2:
                    if st.button("Cancel", key="cancel_profile"):
                        st.session_state.show_profile_editor = False
                        st.rerun()
        
        st.markdown("---")
        
        # Portfolio Summary Section
        st.markdown("### 💼 Portfolio Summary")
        
        # Update portfolio values before displaying
        update_portfolio_values()
        
        # Get portfolio values safely
        portfolio = st.session_state.portfolio
        total_value = portfolio.get('total_value', 100000.0)
        daily_return = portfolio.get('daily_return', 0.0)
        daily_return_pct = portfolio.get('daily_return_pct', 0.0)
        total_return = portfolio.get('total_return', 0.0)
        total_return_pct = portfolio.get('total_return_pct', 0.0)
        
        st.markdown(f"**Total Value**")
        st.markdown(f"# ${total_value:,.2f}")
        
        # Daily change display with safe color selection
        change_color = "🟢" if daily_return >= 0 else "🔴"
        st.markdown(f"{change_color} **Daily:** ${daily_return:+,.2f} ({daily_return_pct:+.2f}%)")
        
        # Total return display
        return_color = "🟢" if total_return >= 0 else "🔴"
        st.markdown(f"{return_color} **Total:** ${total_return:+,.2f} ({total_return_pct:+.2f}%)")
        
        st.markdown("---")
        
        # Watchlist Section
        st.markdown("### 👁️ Watchlist")
        
        watchlist = st.session_state.get('watchlist', ['AAPL', 'MSFT', 'GOOGL'])
        for symbol in watchlist[:5]:  # Show top 5
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button(symbol, key=f"watch_{symbol}", use_container_width=True):
                    st.session_state.analysis_preferences['selected_stock'] = symbol
                    st.session_state.selected_page = 'Analysis'
                    st.rerun()
            with col2:
                # Show mini price indicator
                try:
                    data_processor = get_data_processor()
                    df = data_processor.fetch_stock_data(symbol, period='2d')
                    if not df.empty and len(df) > 1:
                        change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                        color = "🟢" if change >= 0 else "🔴"
                        st.write(f"{color} {change:+.1f}%")
                except:
                    st.write("--")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", key="refresh_data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("📊 Report", key="view_report", use_container_width=True):
                st.session_state.show_report = True
        
        st.markdown("---")
        
        # Market Status
        st.markdown("### 🌍 Market Status")
        
        # Determine market status based on current time
        try:
            import pytz
            et_tz = pytz.timezone('US/Eastern')
            current_time = datetime.now(et_tz)
            market_open = current_time.replace(hour=9, minute=30, second=0)
            market_close = current_time.replace(hour=16, minute=0, second=0)
            
            if current_time.weekday() < 5 and market_open <= current_time <= market_close:
                st.success("🟢 Market Open")
            else:
                st.error("🔴 Market Closed")
        except ImportError:
            # Fallback if pytz is not installed
            current_time = datetime.now()
            if current_time.weekday() < 5 and 9 <= current_time.hour < 16:
                st.success("🟢 Market Open")
            else:
                st.error("🔴 Market Closed")
        
        st.caption(f"Last updated: {datetime.now().strftime('%I:%M %p')}")

def render_dashboard():
    """Render the main dashboard"""
    st.title("📊 Market Dashboard")
    st.caption("Real-time market overview and AI-powered insights")
    
    # Market Overview
    st.markdown("## Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Fetch market indices
    data_processor = get_data_processor()
    
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^VIX': 'VIX'
    }
    
    for i, (symbol, name) in enumerate(indices.items()):
        col = [col1, col2, col3, col4][i]
        with col:
            try:
                df = data_processor.fetch_stock_data(symbol, period='2d')
                if not df.empty and len(df) > 1:
                    current = df['Close'].iloc[-1]
                    previous = df['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    
                    st.metric(
                        label=name,
                        value=f"{current:,.2f}",
                        delta=f"{change:+.2f}%"
                    )
                else:
                    st.metric(label=name, value="--", delta="--")
            except Exception as e:
                st.metric(label=name, value="--", delta="--")
    
    st.markdown("---")
    
    # AI Recommendations Section
    st.markdown("## 🤖 AI Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Top Picks", "📈 Trending", "⚠️ Alerts"])
    
    with tab1:
        render_top_picks()
    
    with tab2:
        render_trending_stocks()
    
    with tab3:
        render_market_alerts()
    
    st.markdown("---")
    
    # Portfolio Performance
    st.markdown("## 📊 Portfolio Performance")
    render_portfolio_chart()

def render_top_picks():
    """Render AI top stock picks"""
    col1, col2, col3 = st.columns(3)
    
    # Sample top picks (replace with actual ML predictions)
    picks = [
        {'symbol': 'AAPL', 'signal': 'BUY', 'confidence': 78, 'reason': 'Strong momentum'},
        {'symbol': 'MSFT', 'signal': 'HOLD', 'confidence': 65, 'reason': 'Consolidating'},
        {'symbol': 'GOOGL', 'signal': 'BUY', 'confidence': 72, 'reason': 'Oversold bounce'}
    ]
    
    for i, pick in enumerate(picks):
        col = [col1, col2, col3][i]
        with col:
            signal_color = "🟢" if pick['signal'] == 'BUY' else "🟡" if pick['signal'] == 'HOLD' else "🔴"
            
            st.markdown(f"### {pick['symbol']}")
            st.markdown(f"{signal_color} **{pick['signal']}**")
            st.progress(pick['confidence'] / 100)
            st.caption(f"Confidence: {pick['confidence']}%")
            st.caption(f"Reason: {pick['reason']}")
            
            if st.button(f"Analyze", key=f"analyze_{pick['symbol']}"):
                st.session_state.analysis_preferences['selected_stock'] = pick['symbol']
                st.session_state.selected_page = 'Analysis'
                st.rerun()

def render_trending_stocks():
    """Render trending stocks"""
    data_processor = get_data_processor()
    
    # Get trending stocks (example with watchlist)
    trending = st.session_state.get('watchlist', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'])
    
    # Create a DataFrame for display
    trending_data = []
    
    for symbol in trending:
        try:
            df = data_processor.fetch_stock_data(symbol, period='5d')
            if not df.empty:
                current_price = df['Close'].iloc[-1]
                change_1d = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
                volume = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].mean()
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1
                
                trending_data.append({
                    'Symbol': symbol,
                    'Price': f"${current_price:.2f}",
                    '1D Change': f"{change_1d:+.2f}%",
                    'Volume vs Avg': f"{volume_ratio:.1f}x",
                    'Trend': '📈' if change_1d > 0 else '📉'
                })
        except:
            continue
    
    if trending_data:
        df_trending = pd.DataFrame(trending_data)
        st.dataframe(df_trending, use_container_width=True, hide_index=True)
    else:
        st.info("No trending data available")

def render_market_alerts():
    """Render market alerts"""
    alerts = [
        {'type': 'warning', 'message': 'VIX above 20 - Market volatility elevated'},
        {'type': 'info', 'message': 'AAPL approaching 52-week high'},
        {'type': 'success', 'message': 'Portfolio up 5% this month'}
    ]
    
    for alert in alerts:
        if alert['type'] == 'warning':
            st.warning(f"⚠️ {alert['message']}")
        elif alert['type'] == 'info':
            st.info(f"ℹ️ {alert['message']}")
        elif alert['type'] == 'success':
            st.success(f"✅ {alert['message']}")

def render_portfolio_chart():
    """Render portfolio performance chart"""
    # Generate sample data (replace with actual portfolio history)
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    values = 100000 + np.cumsum(np.random.randn(30) * 1000)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='black', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,0,0,0.05)'
    ))
    
    fig.update_layout(
        title="30-Day Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400,
        template='plotly_white',
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_analysis():
    """Render the analysis page"""
    st.title("🔍 Stock Analysis")
    
    # Stock selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_stock = st.selectbox(
            "Select Stock",
            options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            index=0,
            key="stock_selector"
        )
    
    with col2:
        period = st.selectbox(
            "Time Period",
            options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
            index=5,
            key="period_selector"
        )
    
    with col3:
        if st.button("🔄 Refresh", key="refresh_analysis"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch and display data
    data_processor = get_data_processor()
    df = data_processor.fetch_stock_data(selected_stock, period=period)
    
    if not df.empty:
        # Calculate indicators if TechnicalIndicators is available
        if TechnicalIndicators:
            df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Display charts and analysis
        st.markdown(f"## {selected_stock} Analysis")
        
        # Price chart
        render_price_chart(df, selected_stock)
        
        # Metrics
        render_stock_metrics(df, selected_stock)
        
        # Technical indicators
        if TechnicalIndicators:
            render_technical_indicators(df)
        
        # AI Prediction
        render_ai_prediction(df, selected_stock)
    else:
        st.error("Unable to fetch data for the selected stock")

def render_price_chart(df, symbol):
    """Render interactive price chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} Price", "RSI", "Volume")
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='gray'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)

def render_stock_metrics(df, symbol):
    """Render stock metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
    
    with col2:
        st.metric("Day Range", f"${df['Low'].iloc[-1]:.2f} - ${df['High'].iloc[-1]:.2f}")
    
    with col3:
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    
    with col4:
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric("RSI", f"{rsi:.1f}", rsi_signal)

def render_technical_indicators(df):
    """Render technical indicators section"""
    st.markdown("### 📊 Technical Indicators")
    
    indicators = {}
    
    if 'RSI' in df.columns:
        indicators['RSI'] = df['RSI'].iloc[-1]
    
    if 'MACD' in df.columns:
        indicators['MACD'] = df['MACD'].iloc[-1]
    
    if 'BB_Position' in df.columns:
        indicators['BB Position'] = df['BB_Position'].iloc[-1] * 100
    
    if indicators:
        indicator_df = pd.DataFrame([indicators])
        st.dataframe(indicator_df, use_container_width=True)

def render_ai_prediction(df, symbol):
    """Render AI prediction section"""
    st.markdown("### 🤖 AI Prediction")
    
    # Placeholder for ML prediction (implement actual model prediction)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal = "BUY"  # Replace with actual prediction
        signal_color = "🟢" if signal == "BUY" else "🟡" if signal == "HOLD" else "🔴"
        st.markdown(f"### {signal_color} {signal}")
    
    with col2:
        confidence = 75  # Replace with actual confidence
        st.markdown("### Confidence")
        st.progress(confidence / 100)
        st.caption(f"{confidence}%")
    
    with col3:
        st.markdown("### Explanation")
        st.caption("Strong momentum with RSI recovering from oversold conditions")

def render_portfolio():
    """Render portfolio page"""
    st.title("💼 Portfolio Management")
    st.info("Portfolio management features coming soon!")

def render_backtesting():
    """Render backtesting page"""
    st.title("📈 Strategy Backtesting")
    st.info("Backtesting features coming soon!")

def render_education():
    """Render education page"""
    st.title("🎓 Education Center")
    st.info("Educational content coming soon!")

def render_settings():
    """Render settings page"""
    st.title("⚙️ Settings")
    st.info("Settings configuration coming soon!")

def main():
    """Main application function"""
    
    # CRITICAL: Initialize session state first
    initialize_session_state()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Navigation
    pages = {
        "Dashboard": render_dashboard,
        "Analysis": render_analysis,
        "Portfolio": render_portfolio,
        "Backtesting": render_backtesting,
        "Education": render_education,
        "Settings": render_settings
    }
    
    # Page selector in main area
    st.markdown("---")
    selected_page = st.radio(
        "Navigation",
        list(pages.keys()),
        horizontal=True,
        key="page_selector",
        label_visibility="collapsed"
    )
    st.markdown("---")
    
    # Render selected page
    pages[selected_page]()
    
    # Footer
    st.markdown("---")
    st.caption("StockBot Advisor v1.0.0 | CM3070 Project | © 2025 Anthony Winata Salim")

if __name__ == "__main__":
    main()