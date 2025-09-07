"""
Dashboard Page
Main dashboard with market overview and portfolio summary
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from utils.data_processor import get_data_processor
from utils.technical_indicators import get_indicator_calculator
from components.charts import ChartComponents, render_chart
from components.metrics import MetricComponents
from components.sidebar import render_complete_sidebar
from components.alerts import AlertComponents

# Page configuration
st.set_page_config(
    page_title="Dashboard - StockBot Advisor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for minimalistic design
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    h1 {
        font-weight: 300;
        border-bottom: 1px solid #E9ECEF;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'total_value': 125430,
            'daily_return': 1.9,
            'total_return': 25.4,
            'cash': 25000
        }
    
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Anthony Salim',
            'risk_tolerance': 'Moderate',
            'investment_horizon': '1-3 years',
            'member_since': '2025'
        }

def render_market_overview():
    """Render market overview section with real data"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    st.markdown("## Market Overview")
    
    # Define market indices tickers
    market_tickers = {
        "S&P 500": "^GSPC",
        "DOW JONES": "^DJI",
        "NASDAQ": "^IXIC",
        "VIX": "^VIX",
        "GOLD": "GC=F",
        "10Y BOND": "^TNX",
        "USD INDEX": "DX-Y.NYB",
        "CRUDE OIL": "CL=F"
    }
    
    indices = []
    
    # Fetch real data for each ticker
    for name, ticker in market_tickers.items():
        try:
            # Fetch data for last 2 days
            data = yf.download(ticker, 
                              start=(datetime.now() - timedelta(days=5)), 
                              end=datetime.now(), 
                              progress=False)
            
            if not data.empty and len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2]
                change_pct = ((current_price - previous_price) / previous_price) * 100
                
                indices.append({
                    "name": name,
                    "value": current_price,
                    "change": change_pct
                })
            else:
                # Fallback to dummy data if fetch fails
                indices.append({
                    "name": name,
                    "value": 0,
                    "change": 0
                })
        except:
            # Use dummy data if API fails
            indices.append({
                "name": name,
                "value": 0,
                "change": 0
            })
    
    # If no data was fetched, use fallback data
    if not indices or all(idx["value"] == 0 for idx in indices):
        indices = [
            {"name": "S&P 500", "value": 4542.45, "change": -0.35},
            {"name": "DOW JONES", "value": 35322.21, "change": 0.52},
            {"name": "NASDAQ", "value": 14777.84, "change": -0.78},
            {"name": "VIX", "value": 17.80, "change": 2.15},
            {"name": "GOLD", "value": 1835.70, "change": 0.95},
            {"name": "10Y BOND", "value": 4.42, "change": -0.42},
            {"name": "USD INDEX", "value": 93.58, "change": -0.18},
            {"name": "CRUDE OIL", "value": 75.20, "change": 1.85}
        ]
    
    # Create columns for display
    cols = st.columns(len(indices))
    
    # Display each index
    for i, index in enumerate(indices):
        with cols[i]:
            color = "#28A745" if index["change"] >= 0 else "#DC3545"
            arrow = "↑" if index["change"] >= 0 else "↓"
            
            st.markdown(f"""
            <div style="
                background: white;
                border: 1px solid #E9ECEF;
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="
                    color: #6C757D;
                    font-size: 0.75rem;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                ">
                    {index["name"]}
                </div>
                <div style="
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #000000;
                    font-family: monospace;
                ">
                    {index["value"]:,.2f}
                </div>
                <div style="
                    color: {color};
                    font-size: 0.875rem;
                    font-weight: 500;
                ">
                    {arrow} {abs(index["change"]):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

def render_portfolio_summary():
    """Render portfolio summary"""
    st.markdown("## Portfolio Performance")
    
    portfolio = st.session_state.portfolio
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # FIXED: Added 'f' before the triple quotes to make it an f-string
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">💼</span>
                <span style="color: #6C757D; font-size: 0.875rem; text-transform: uppercase;">TOTAL VALUE</span>
            </div>
            <div style="font-size: 2rem; font-weight: bold; color: #000;">
                ${portfolio['total_value']:,.1f}
            </div>
            <div style="color: {'#28A745' if portfolio.get('daily_return', 0) >= 0 else '#DC3545'}; font-size: 0.875rem; margin-top: 0.5rem;">
                {'↑' if portfolio.get('daily_return', 0) >= 0 else '↓'} {abs(portfolio.get('daily_return', 0)):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        daily_pnl = portfolio['total_value'] * (portfolio.get('daily_return', 0) / 100)
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">📊</span>
                <span style="color: #6C757D; font-size: 0.875rem; text-transform: uppercase;">DAILY P&L</span>
            </div>
            <div style="font-size: 2rem; font-weight: bold; color: #000;">
                ${daily_pnl:,.1f}
            </div>
            <div style="color: {'#28A745' if daily_pnl >= 0 else '#DC3545'}; font-size: 0.875rem; margin-top: 0.5rem;">
                {'↑' if daily_pnl >= 0 else '↓'} {abs(portfolio.get('daily_return', 0)):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">📈</span>
                <span style="color: #6C757D; font-size: 0.875rem; text-transform: uppercase;">TOTAL RETURN</span>
            </div>
            <div style="font-size: 2rem; font-weight: bold; color: #000;">
                {portfolio.get('total_return', 0):.1f}%
            </div>
            <div style="color: {'#28A745' if portfolio.get('total_return', 0) >= 0 else '#DC3545'}; font-size: 0.875rem; margin-top: 0.5rem;">
                {'↑' if portfolio.get('total_return', 0) >= 0 else '↓'} +{abs(portfolio.get('total_return', 0)):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">🎯</span>
                <span style="color: #6C757D; font-size: 0.875rem; text-transform: uppercase;">WIN RATE</span>
            </div>
            <div style="font-size: 2rem; font-weight: bold; color: #000;">
                68%
            </div>
            <div style="color: #6C757D; font-size: 0.75rem; margin-top: 0.5rem;">
                Last 30 days
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">💵</span>
                <span style="color: #6C757D; font-size: 0.875rem; text-transform: uppercase;">CASH AVAILABLE</span>
            </div>
            <div style="font-size: 2rem; font-weight: bold; color: #000;">
                ${portfolio.get('cash', 0):,.1f}
            </div>
            <div style="color: #6C757D; font-size: 0.75rem; margin-top: 0.5rem;">
                Ready to invest
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_watchlist():
    """Render watchlist section"""
    st.markdown("### 👁️ Watchlist")
    
    # Add new stock to watchlist
    col1, col2 = st.columns([4, 1])
    with col1:
        new_symbol = st.text_input(
            "Add to watchlist",
            placeholder="Enter stock symbol (e.g., AAPL)",
            label_visibility="collapsed"
        )
    with col2:
        if st.button("Add", use_container_width=True):
            if new_symbol and new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.success(f"Added {new_symbol.upper()} to watchlist")
                st.rerun()
    
    # Display watchlist
    if st.session_state.watchlist:
        data_processor = get_data_processor()
        watchlist_data = data_processor.fetch_batch_quotes(st.session_state.watchlist)
        
        if not watchlist_data.empty:
            # Format data for display
            watchlist_data['Change_Display'] = watchlist_data.apply(
                lambda x: f"{'↑' if x['Change%'] >= 0 else '↓'} {abs(x['Change%']):.2f}%",
                axis=1
            )
            
            # Display table
            st.dataframe(
                watchlist_data[['Symbol', 'Price', 'Change_Display', 'Volume']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "Change_Display": st.column_config.TextColumn("Change"),
                    "Volume": st.column_config.NumberColumn("Volume", format="%d")
                }
            )
        else:
            # Fallback with sample data
            sample_data = []
            for symbol in st.session_state.watchlist:
                sample_data.append({
                    'Symbol': symbol,
                    'Price': np.random.uniform(50, 500),
                    'Change': np.random.uniform(-5, 5),
                    'Volume': np.random.randint(1000000, 50000000)
                })
            
            df = pd.DataFrame(sample_data)
            df['Change_Display'] = df.apply(
                lambda x: f"{'↑' if x['Change'] >= 0 else '↓'} {abs(x['Change']):.2f}%",
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
    
    # Sample signals - in production, these would come from ML model
    signals = [
        {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.78,
            'price': 185.23,
            'target': 195.00,
            'reason': 'RSI oversold, MACD bullish crossover'
        },
        {
            'symbol': 'MSFT',
            'signal': 'HOLD',
            'confidence': 0.65,
            'price': 378.45,
            'target': 380.00,
            'reason': 'Consolidating, awaiting breakout'
        },
        {
            'symbol': 'GOOGL',
            'signal': 'SELL',
            'confidence': 0.72,
            'price': 142.30,
            'target': 135.00,
            'reason': 'Overbought, resistance at $145'
        }
    ]
    
    for signal in signals:
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            if signal['signal'] == 'BUY':
                st.success(f"**{signal['symbol']}** - BUY")
            elif signal['signal'] == 'SELL':
                st.error(f"**{signal['symbol']}** - SELL")
            else:
                st.info(f"**{signal['symbol']}** - HOLD")
        
        with col2:
            st.metric(
                "Confidence",
                f"{signal['confidence']*100:.0f}%",
                label_visibility="collapsed"
            )
        
        with col3:
            st.caption(signal['reason'])

def render_performance_chart():
    """Render portfolio performance chart"""
    st.markdown("### Portfolio Performance")
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    portfolio_values = 100000 * (1 + np.random.randn(90).cumsum() * 0.01)
    benchmark_values = 100000 * (1 + np.random.randn(90).cumsum() * 0.008)
    
    df = pd.DataFrame({
        'Portfolio': portfolio_values,
        'S&P 500': benchmark_values
    }, index=dates)
    
    # Create chart
    fig = ChartComponents.create_line_chart(
        df,
        columns=['Portfolio', 'S&P 500'],
        title="90-Day Performance",
        height=400
    )
    
    render_chart(fig)

def render_sector_performance():
    """Render sector performance"""
    st.markdown("### Sector Performance")
    
    data_processor = get_data_processor()
    sector_data = data_processor.get_sector_performance()
    
    if not sector_data.empty:
        # Create bar chart
        fig = ChartComponents.create_bar_chart(
            sector_data,
            x_col='Sector',
            y_col='Weekly_Change%',
            title="Weekly Sector Performance",
            height=300,
            color_positive=True
        )
        render_chart(fig)
    else:
        # Fallback with sample data
        sectors = {
            'Technology': 3.2,
            'Healthcare': 1.8,
            'Finance': 2.5,
            'Energy': -1.2,
            'Consumer': 0.8,
            'Industrial': 1.5
        }
        
        df = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Performance'])
        fig = ChartComponents.create_bar_chart(
            df,
            x_col='Sector',
            y_col='Performance',
            title="Weekly Sector Performance (%)",
            height=300,
            color_positive=True
        )
        render_chart(fig)

def render_market_news():
    """Render market news section"""
    st.markdown("### 📰 Latest Market News")
    
    news_items = [
        {
            'title': 'Fed Signals Potential Rate Cuts in Q2 2025',
            'source': 'Reuters',
            'time': '2 hours ago',
            'summary': 'Federal Reserve officials indicated openness to rate adjustments...'
        },
        {
            'title': 'Tech Stocks Rally on AI Optimism',
            'source': 'Bloomberg',
            'time': '4 hours ago',
            'summary': 'Major technology companies gained as AI adoption accelerates...'
        },
        {
            'title': 'Oil Prices Stabilize After Weekly Decline',
            'source': 'CNBC',
            'time': '6 hours ago',
            'summary': 'Crude oil prices found support at key technical levels...'
        }
    ]
    
    for news in news_items:
        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{news['title']}**")
                st.caption(f"{news['source']} • {news['time']}")
                st.write(news['summary'])
            with col2:
                st.button("Read More", key=f"news_{news['title'][:10]}")
        st.divider()

# Main function
def main():
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_complete_sidebar(
        st.session_state.user_profile,
        st.session_state.portfolio,
        st.session_state.watchlist[:5]
    )
    
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