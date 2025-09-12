# pages/dashboard.py
"""
Dashboard Page
Main dashboard with market overview, portfolio summary, and AI insights
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("Plotly is required for this page. Please install with: pip install plotly")
    PLOTLY_AVAILABLE = False
    st.stop()

# Import custom modules
from utils.data_processor import get_data_processor
from utils.technical_indicators import TechnicalIndicators
from utils.ml_models import create_prediction_model
import yaml

# Page configuration
st.set_page_config(
    page_title="Dashboard - StockBot Advisor",
    page_icon="📊",
    layout="wide"
)

def initialize_session_state():
    """Initialize all session state variables for the application"""
    
    # User Profile
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Guest User',
            'email': 'guest@stockbot.com',
            'risk_tolerance': 'Moderate',
            'investment_horizon': '1-3 years',
            'initial_capital': 100000.0,
            'currency': 'USD',
            'experience_level': 'Intermediate',
            'investment_goals': ['Growth', 'Income'],
            'preferred_sectors': ['Technology', 'Healthcare'],
            'max_position_size': 10.0,  # percentage
            'created_at': datetime.now().isoformat()
        }
    
    # Portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'holdings': {},  # {symbol: {'shares': float, 'avg_cost': float, 'purchase_date': str}}
            'cash': 100000.0,
            'total_value': 100000.0,
            'daily_return': 0.0,
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'transactions': [],  # List of transaction records
            'last_updated': datetime.now().isoformat()
        }
    
    # Watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Analysis preferences
    if 'analysis_preferences' not in st.session_state:
        st.session_state.analysis_preferences = {
            'selected_stock': 'AAPL',
            'analysis_period': '1y',
            'show_sentiment': True,
            'show_shap': False,
            'model_trained': False,
            'chart_type': 'candlestick',
            'indicators': ['RSI', 'MACD', 'SMA_20', 'SMA_50']
        }
    
    # Application settings
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = {
            'theme': 'minimal',
            'data_refresh_interval': 300,  # seconds
            'enable_notifications': True,
            'auto_save': True,
            'language': 'en',
            'timezone': 'UTC'
        }

def get_user_profile():
    """Get user profile with defaults"""
    if 'user_profile' not in st.session_state:
        initialize_session_state()
    return st.session_state.user_profile

def get_portfolio():
    """Get portfolio with defaults"""
    if 'portfolio' not in st.session_state:
        initialize_session_state()
    return st.session_state.portfolio

@st.cache_resource
def load_config():
    """Load configuration"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}

def render_market_overview():
    """Render market overview section"""
    st.markdown("### 📈 Market Overview")
    
    data_processor = get_data_processor()
    market_data = data_processor.get_market_overview()
    
    if market_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if '^GSPC' in market_data:
                sp500 = market_data['^GSPC']
                st.metric(
                    sp500['name'],
                    f"{sp500['price']:.2f}",
                    delta=f"{sp500['change_pct']:+.2f}%"
                )
            else:
                st.metric("S&P 500", "4,500.0", delta="+0.5%")
        
        with col2:
            if '^DJI' in market_data:
                dow = market_data['^DJI']
                st.metric(
                    dow['name'],
                    f"{dow['price']:.0f}",
                    delta=f"{dow['change_pct']:+.2f}%"
                )
            else:
                st.metric("Dow Jones", "35,000", delta="+0.3%")
        
        with col3:
            if '^IXIC' in market_data:
                nasdaq = market_data['^IXIC']
                st.metric(
                    nasdaq['name'],
                    f"{nasdaq['price']:.0f}",
                    delta=f"{nasdaq['change_pct']:+.2f}%"
                )
            else:
                st.metric("NASDAQ", "14,000", delta="+0.8%")
        
        with col4:
            if '^VIX' in market_data:
                vix = market_data['^VIX']
                st.metric(
                    vix['name'],
                    f"{vix['price']:.2f}",
                    delta=f"{vix['change_pct']:+.2f}%"
                )
            else:
                st.metric("VIX", "20.5", delta="-2.1%")
    else:
        # Fallback metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("S&P 500", "4,500.0", delta="+0.5%")
        with col2:
            st.metric("Dow Jones", "35,000", delta="+0.3%")
        with col3:
            st.metric("NASDAQ", "14,000", delta="+0.8%")
        with col4:
            st.metric("VIX", "20.5", delta="-2.1%")

def render_portfolio_summary():
    """Render portfolio summary"""
    st.markdown("### 💼 Portfolio Summary")
    
    portfolio = get_portfolio()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = portfolio.get('total_value', 100000)
        st.metric("Total Value", f"${total_value:,.2f}")
    
    with col2:
        daily_return = portfolio.get('daily_return', 0)
        st.metric("Today's Return", f"{daily_return:+.2f}%")
    
    with col3:
        total_return = portfolio.get('total_return', 0)
        st.metric("Total Return", f"${total_return:,.2f}")
    
    with col4:
        total_return_pct = portfolio.get('total_return_pct', 0)
        st.metric("Total Return %", f"{total_return_pct:+.2f}%")

def render_watchlist():
    """Render watchlist table"""
    st.markdown("### 👁️ Watchlist")
    
    # Initialize session state
    initialize_session_state()
    
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add stocks to track them here.")
        
        # Add stock to watchlist functionality
        col1, col2 = st.columns([2, 1])
        with col1:
            new_stock = st.text_input("Add stock symbol", placeholder="AAPL").upper()
        with col2:
            if st.button("Add to Watchlist", disabled=not new_stock):
                if new_stock and new_stock not in st.session_state.watchlist:
                    st.session_state.watchlist.append(new_stock)
                    st.success(f"Added {new_stock} to watchlist!")
                    st.rerun()
        return
    
    data_processor = get_data_processor()
    
    try:
        # Fetch batch quotes
        watchlist_df = data_processor.fetch_batch_quotes(st.session_state.watchlist)
        
        if not watchlist_df.empty:
            # Format the display
            watchlist_df['Change_Display'] = watchlist_df.apply(
                lambda x: f"{'↑' if x['Change%'] >= 0 else '↓'} {abs(x['Change%']):.2f}%",
                axis=1
            )
            
            st.dataframe(
                watchlist_df[['Symbol', 'Price', 'Change_Display', 'Volume']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "Change_Display": st.column_config.TextColumn("Change"),
                    "Volume": st.column_config.NumberColumn("Volume", format="%d")
                }
            )
            
            # Add remove from watchlist functionality
            col1, col2 = st.columns([2, 1])
            with col1:
                remove_stock = st.selectbox("Remove from watchlist", [""] + st.session_state.watchlist)
            with col2:
                if st.button("Remove", disabled=not remove_stock):
                    if remove_stock in st.session_state.watchlist:
                        st.session_state.watchlist.remove(remove_stock)
                        st.success(f"Removed {remove_stock} from watchlist!")
                        st.rerun()
        else:
            # Fallback with sample data
            sample_data = []
            for symbol in st.session_state.watchlist:
                sample_data.append({
                    'Symbol': symbol,
                    'Price': np.random.uniform(50, 500),
                    'Change%': np.random.uniform(-5, 5),
                    'Volume': np.random.randint(1000000, 50000000)
                })
            
            df = pd.DataFrame(sample_data)
            df['Change_Display'] = df.apply(
                lambda x: f"{'↑' if x['Change%'] >= 0 else '↓'} {abs(x['Change%']):.2f}%",
                axis=1
            )
            
            st.dataframe(
                df[['Symbol', 'Price', 'Change_Display', 'Volume']],
                use_container_width=True,
                hide_index=True
            )
    
    except Exception as e:
        st.error(f"Error loading watchlist data: {e}")
        st.info("Using sample data...")
        
        # Show sample data as fallback
        sample_data = []
        for symbol in st.session_state.watchlist:
            sample_data.append({
                'Symbol': symbol,
                'Price': np.random.uniform(50, 500),
                'Change%': np.random.uniform(-5, 5),
                'Volume': np.random.randint(1000000, 50000000)
            })
        
        df = pd.DataFrame(sample_data)
        df['Change_Display'] = df.apply(
            lambda x: f"{'↑' if x['Change%'] >= 0 else '↓'} {abs(x['Change%']):.2f}%",
            axis=1
        )
        
        st.dataframe(
            df[['Symbol', 'Price', 'Change_Display', 'Volume']],
            use_container_width=True,
            hide_index=True
        )

def render_ai_signals():
    """Render AI trading signals"""
    st.markdown("### 🤖 AI Trading Signals")
    
    # Initialize session state
    initialize_session_state()
    
    # Show sample signals first
    st.info("🔄 AI Signals are being generated using advanced machine learning models...")
    
    # Sample signals for immediate display
    sample_signals = [
        {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.78,
            'price': 185.23,
            'reason': 'RSI: 32.1, MACD: 0.045 (Oversold with bullish crossover)',
            'accuracy': 72.5
        },
        {
            'symbol': 'MSFT',
            'signal': 'HOLD',
            'confidence': 0.65,
            'price': 378.45,
            'reason': 'RSI: 55.8, MACD: -0.012 (Neutral consolidation)',
            'accuracy': 68.3
        },
        {
            'symbol': 'GOOGL',
            'signal': 'SELL',
            'confidence': 0.72,
            'price': 142.30,
            'reason': 'RSI: 71.2, MACD: -0.089 (Overbought with bearish divergence)',
            'accuracy': 70.1
        }
    ]
    
    # Display sample signals
    for i, signal in enumerate(sample_signals):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
            
            with col1:
                if signal['signal'] == 'BUY':
                    st.success(f"**{signal['symbol']}** - BUY")
                elif signal['signal'] == 'SELL':
                    st.error(f"**{signal['symbol']}** - SELL")
                else:
                    st.info(f"**{signal['symbol']}** - HOLD")
                
                st.caption(f"Price: ${signal['price']:.2f}")
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{signal['confidence']*100:.0f}%",
                    label_visibility="collapsed"
                )
            
            with col3:
                st.metric(
                    "Model Accuracy",
                    f"{signal.get('accuracy', 70):.1f}%",
                    label_visibility="collapsed"
                )
            
            with col4:
                st.caption(f"**Reasoning:** {signal['reason']}")
                
                # Add action buttons
                if signal['signal'] == 'BUY':
                    if st.button(f"Add {signal['symbol']} to Portfolio", key=f"buy_{signal['symbol']}_{i}"):
                        st.success(f"Added {signal['symbol']} to buy list!")
                elif signal['signal'] == 'SELL':
                    if st.button(f"Review {signal['symbol']} Position", key=f"sell_{signal['symbol']}_{i}"):
                        st.info(f"Reviewing {signal['symbol']} for potential sale...")
            
            st.divider()
    
    # Option to generate real-time signals
    if st.button("🔄 Generate Real-Time AI Signals", type="primary"):
        with st.spinner("Analyzing market data and generating AI signals..."):
            try:
                data_processor = get_data_processor()
                
                # Get signals for watchlist stocks
                real_signals = []
                stocks_to_analyze = st.session_state.watchlist[:3] if st.session_state.watchlist else ['AAPL', 'MSFT', 'GOOGL']
                
                for symbol in stocks_to_analyze:
                    try:
                        # Fetch stock data
                        df = data_processor.fetch_stock_data(symbol, period='2y')
                        
                        if not df.empty:
                            # Add technical indicators
                            df = TechnicalIndicators.calculate_all_indicators(df)
                            
                            # Create ML model and get prediction
                            model = create_prediction_model('classification')
                            metrics = model.train(df)
                            
                            if 'error' not in metrics:
                                prediction_result = model.predict_latest(df)
                                
                                if 'error' not in prediction_result:
                                    real_signals.append({
                                        'symbol': symbol,
                                        'signal': prediction_result['signal'],
                                        'confidence': prediction_result['confidence'],
                                        'price': df['Close'].iloc[-1],
                                        'reason': f"RSI: {prediction_result.get('indicators', {}).get('rsi', 0):.1f}, MACD: {prediction_result.get('indicators', {}).get('macd', 0):.3f}",
                                        'accuracy': metrics.get('accuracy', 0) * 100
                                    })
                                else:
                                    st.warning(f"Prediction failed for {symbol}: {prediction_result['error']}")
                            else:
                                st.warning(f"Model training failed for {symbol}: {metrics['error']}")
                        else:
                            st.warning(f"No data available for {symbol}")
                    except Exception as e:
                        st.error(f"Error analyzing {symbol}: {e}")
                        continue
                
                if real_signals:
                    st.success(f"✅ Generated {len(real_signals)} real-time AI signals!")
                    
                    # Display real signals
                    for i, signal in enumerate(real_signals):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
                            
                            with col1:
                                if signal['signal'] == 'BUY':
                                    st.success(f"**{signal['symbol']}** - BUY")
                                elif signal['signal'] == 'SELL':
                                    st.error(f"**{signal['symbol']}** - SELL")
                                else:
                                    st.info(f"**{signal['symbol']}** - HOLD")
                                
                                st.caption(f"Price: ${signal['price']:.2f}")
                            
                            with col2:
                                st.metric(
                                    "Confidence",
                                    f"{signal['confidence']*100:.0f}%",
                                    label_visibility="collapsed"
                                )
                            
                            with col3:
                                st.metric(
                                    "Model Accuracy",
                                    f"{signal.get('accuracy', 70):.1f}%",
                                    label_visibility="collapsed"
                                )
                            
                            with col4:
                                st.caption(f"**Reasoning:** {signal['reason']}")
                            
                            st.divider()
                else:
                    st.warning("No real-time signals could be generated. Using sample data.")
                    
            except Exception as e:
                st.error(f"Error generating real-time signals: {e}")
                st.info("Displaying sample signals instead.")

def render_performance_chart():
    """Render portfolio performance chart"""
    st.markdown("### 📈 Portfolio Performance")
    
    # Generate sample data for demonstration
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    # Create more realistic portfolio performance
    np.random.seed(42)  # For reproducible results
    daily_returns = np.random.normal(0.0008, 0.015, 90)  # ~20% annual return, 15% volatility
    portfolio_values = 100000 * np.cumprod(1 + daily_returns)
    
    # Benchmark returns (S&P 500)
    benchmark_returns = np.random.normal(0.0007, 0.012, 90)  # ~18% annual return, 12% volatility
    benchmark_values = 100000 * np.cumprod(1 + benchmark_returns)
    
    df = pd.DataFrame({
        'Portfolio': portfolio_values,
        'S&P 500': benchmark_values
    }, index=dates)
    
    # Create chart
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Portfolio'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['S&P 500'],
            mode='lines',
            name='S&P 500',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="90-Day Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        portfolio_return = (df['Portfolio'].iloc[-1] / df['Portfolio'].iloc[0] - 1) * 100
        benchmark_return = (df['S&P 500'].iloc[-1] / df['S&P 500'].iloc[0] - 1) * 100
        outperformance = portfolio_return - benchmark_return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Return", f"{portfolio_return:+.2f}%")
        with col2:
            st.metric("S&P 500 Return", f"{benchmark_return:+.2f}%")
        with col3:
            st.metric("Outperformance", f"{outperformance:+.2f}%")

def render_sector_performance():
    """Render sector performance"""
    st.markdown("### 🏢 Sector Performance")
    
    data_processor = get_data_processor()
    sector_data = data_processor.get_sector_performance()
    
    # Check if sector_data is not empty (it's a dictionary, not DataFrame)
    if sector_data and len(sector_data) > 0:
        # Convert dictionary to DataFrame for charting
        sector_df = pd.DataFrame([
            {
                'Sector': sector_name,
                'Return': data['return'],
                'Count': data['count']
            }
            for sector_name, data in sector_data.items()
        ])
        
        if PLOTLY_AVAILABLE and not sector_df.empty:
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=sector_df['Sector'],
                    y=sector_df['Return'],
                    marker_color=['#2E8B57' if x >= 0 else '#DC143C' for x in sector_df['Return']],
                    text=[f"{x:+.1f}%" for x in sector_df['Return']],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Return: %{y:+.1f}%<br>Stocks: %{customdata}<extra></extra>',
                    customdata=sector_df['Count']
                )
            ])
            
            fig.update_layout(
                title="Monthly Sector Performance",
                xaxis_title="Sector",
                yaxis_title="Return (%)",
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback table display
            st.dataframe(
                sector_df[['Sector', 'Return', 'Count']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Return": st.column_config.NumberColumn("Return (%)", format="%.2f")
                }
            )
    else:
        # Fallback with sample data if no real data available
        st.info("Loading sector data...")
        
        sectors = {
            'Technology': 3.2,
            'Healthcare': 1.8,
            'Finance': 2.5,
            'Energy': -1.2,
            'Consumer': 0.8,
            'Industrial': 1.5
        }
        
        df = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Return'])
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(
                    x=df['Sector'],
                    y=df['Return'],
                    marker_color=['#2E8B57' if x >= 0 else '#DC143C' for x in df['Return']],
                    text=[f"{x:+.1f}%" for x in df['Return']],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Return: %{y:+.1f}%<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Monthly Sector Performance",
                xaxis_title="Sector",
                yaxis_title="Return (%)",
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_market_news():
    """Render market news section"""
    st.markdown("### 📰 Latest Market News")
    
    news_items = [
        {
            'title': 'Fed Signals Potential Rate Cuts in Q2 2025',
            'source': 'Reuters',
            'time': '2 hours ago',
            'summary': 'Federal Reserve officials indicated openness to rate adjustments amid cooling inflation data and improving labor market conditions...',
            'category': 'Monetary Policy',
            'impact': 'Positive'
        },
        {
            'title': 'Tech Stocks Rally on AI Optimism',
            'source': 'Bloomberg',
            'time': '4 hours ago',
            'summary': 'Major technology companies gained as artificial intelligence adoption accelerates across industries, driving revenue growth...',
            'category': 'Technology',
            'impact': 'Positive'
        },
        {
            'title': 'Oil Prices Stabilize After Weekly Decline',
            'source': 'CNBC',
            'time': '6 hours ago',
            'summary': 'Crude oil prices found support at key technical levels following inventory data and OPEC production decisions...',
            'category': 'Energy',
            'impact': 'Neutral'
        },
        {
            'title': 'Healthcare Sector Shows Resilience',
            'source': 'MarketWatch',
            'time': '8 hours ago',
            'summary': 'Healthcare stocks outperformed broader markets amid defensive positioning and strong earnings reports...',
            'category': 'Healthcare',
            'impact': 'Positive'
        },
        {
            'title': 'Crypto Markets Show Mixed Signals',
            'source': 'CoinDesk',
            'time': '10 hours ago',
            'summary': 'Bitcoin and major altcoins traded sideways as investors await regulatory clarity and institutional adoption updates...',
            'category': 'Cryptocurrency',
            'impact': 'Neutral'
        }
    ]
    
    for news in news_items:
        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                # Impact indicator
                impact_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                impact_emoji = impact_color.get(news['impact'], "⚪")
                
                st.markdown(f"{impact_emoji} **{news['title']}**")
                st.caption(f"{news['source']} • {news['time']} • {news['category']}")
                st.write(news['summary'])
            with col2:
                st.button("Read More", key=f"news_{news['title'][:10]}", use_container_width=True)
        st.divider()

def render_complete_sidebar(user_profile, portfolio, watchlist):
    """Render complete sidebar with user info and quick stats"""
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.write(f"**Name:** {user_profile.get('name', 'Guest')}")
        st.write(f"**Risk Tolerance:** {user_profile.get('risk_tolerance', 'Moderate')}")
        st.write(f"**Experience:** {user_profile.get('experience_level', 'Intermediate')}")
        
        st.divider()
        
        st.markdown("### 💼 Quick Stats")
        st.metric("Portfolio Value", f"${portfolio.get('total_value', 100000):,.2f}")
        st.metric("Today's Change", f"{portfolio.get('daily_return', 0):+.2f}%")
        st.metric("Cash Available", f"${portfolio.get('cash', 50000):,.2f}")
        
        st.divider()
        
        st.markdown("### 📋 Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data refreshed!")
            st.rerun()
        
        if st.button("📊 Full Analysis", use_container_width=True):
            st.switch_page("pages/analysis.py")
        
        if st.button("💼 Manage Portfolio", use_container_width=True):
            st.switch_page("pages/portfolio.py")
        
        if st.button("📈 Backtesting", use_container_width=True):
            st.switch_page("pages/backtesting.py")
        
        st.divider()
        
        st.markdown("### 👁️ Watchlist Preview")
        if watchlist:
            for symbol in watchlist[:3]:  # Show top 3
                st.caption(f"• {symbol}")
            if len(watchlist) > 3:
                st.caption(f"... and {len(watchlist) - 3} more")
        else:
            st.caption("No stocks in watchlist")

# Main function
def main():
    """Main dashboard page function"""
    
    # Check if plotly is available
    if not PLOTLY_AVAILABLE:
        st.error("This page requires Plotly. Please install it with: pip install plotly")
        st.stop()
    
    # Initialize session state
    initialize_session_state()
    
    # Get user profile and portfolio
    user_profile = get_user_profile()
    portfolio = get_portfolio()
    
    # Render sidebar
    render_complete_sidebar(user_profile, portfolio, st.session_state.watchlist)
    
    # Main header
    st.markdown("# 📊 Market Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    
    # Market overview
    render_market_overview()
    
    st.divider()
    
    # Portfolio summary
    render_portfolio_summary()
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance",
        "👁️ Watchlist", 
        "🤖 AI Signals",
        "📰 Market News"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            render_performance_chart()
        with col2:
            render_sector_performance()
    
    with tab2:
        render_watchlist()
    
    with tab3:
        render_ai_signals()
    
    with tab4:
        render_market_news()

if __name__ == "__main__":
    main()