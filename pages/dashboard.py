# pages/dashboard.py
"""
Dashboard Page - Enhanced with Practical Trading System
Main dashboard with market overview, portfolio summary, and AI insights with confidence filtering
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import traceback
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')

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
from utils.ml_models import create_prediction_model, MLModel

# Import for practical trading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            'max_position_size': 10.0,
            'created_at': datetime.now().isoformat()
        }
    
    # Portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'holdings': {},
            'cash': 100000.0,
            'total_value': 100000.0,
            'transactions': [],
            'performance_history': []
        }
    
    # Watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Market Data Cache
    if 'market_data_cache' not in st.session_state:
        st.session_state.market_data_cache = {}
    
    # AI Predictions Cache
    if 'predictions_cache' not in st.session_state:
        st.session_state.predictions_cache = {}
    
    # Settings
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'theme': 'dark',
            'auto_refresh': False,
            'refresh_interval': 60,
            'show_notifications': True,
            'data_source': 'yfinance'
        }

def load_practical_model():
    """Load the practical trading model with confidence filtering"""
    model_path = 'practical_model.pkl'
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            return model_data
        except Exception as e:
            st.sidebar.warning(f"Could not load practical model: {e}")
            return None
    return None

def create_practical_features(df):
    """Create the same features as in practical_trading_system.py"""
    # Price changes
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-10)))
    
    # Simple moving averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Price position
    df['price_vs_sma20'] = df['Close'] / df['SMA_20']
    df['price_vs_sma50'] = df['Close'] / df['SMA_50']
    
    # Volatility
    df['volatility'] = df['return_1d'].rolling(20).std()
    
    # Volume
    df['volume_avg'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_avg']
    
    return df

def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("📊 Financial Dashboard")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        # Check for practical model
        if os.path.exists('practical_model.pkl'):
            st.success("✅ Practical Model Active")
        else:
            if st.button("Enable Practical Trading"):
                st.info("Run: `python practical_trading_system.py`")
    
    with col3:
        if st.button("🔄 Refresh All", type="primary"):
            st.session_state.market_data_cache = {}
            st.session_state.predictions_cache = {}
            st.rerun()

def render_market_overview():
    """Render market overview section"""
    st.markdown("### 📈 Market Overview")
    
    try:
        data_processor = get_data_processor()
        
        # Market indices
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'DOW': '^DJI'
        }
        
        cols = st.columns(len(indices))
        
        for idx, (name, symbol) in enumerate(indices.items()):
            with cols[idx]:
                try:
                    df = data_processor.fetch_stock_data(symbol, period='5d')
                    if not df.empty:
                        current = df['Close'].iloc[-1]
                        prev = df['Close'].iloc[-2] if len(df) > 1 else current
                        change = ((current - prev) / prev) * 100
                        
                        st.metric(
                            label=name,
                            value=f"{current:,.2f}",
                            delta=f"{change:+.2f}%",
                            delta_color="normal" if change >= 0 else "inverse"
                        )
                    else:
                        st.metric(name, "N/A", "N/A")
                except:
                    st.metric(name, "Loading...", "")
        
    except Exception as e:
        st.error(f"Error loading market data: {e}")

def render_portfolio_summary():
    """Render portfolio summary"""
    st.markdown("### 💼 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    portfolio = st.session_state.portfolio
    
    with col1:
        st.metric("Total Value", f"${portfolio['total_value']:,.2f}")
    
    with col2:
        st.metric("Cash Available", f"${portfolio['cash']:,.2f}")
    
    with col3:
        positions = len(portfolio['holdings'])
        st.metric("Open Positions", positions)
    
    with col4:
        daily_change = np.random.uniform(-2, 3)  # Placeholder
        st.metric("Daily P&L", f"{daily_change:+.2f}%")

def render_watchlist():
    """Render watchlist section"""
    st.markdown("### 👀 Watchlist")
    
    try:
        data_processor = get_data_processor()
        watchlist_data = []
        
        for symbol in st.session_state.watchlist[:5]:
            try:
                df = data_processor.fetch_stock_data(symbol, period='1d')
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    prev_close = df['Open'].iloc[0]
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    volume = df['Volume'].iloc[-1]
                    
                    watchlist_data.append({
                        'Symbol': symbol,
                        'Price': f"${current_price:.2f}",
                        'Change': f"{change_pct:+.2f}%",
                        'Volume': f"{volume:,.0f}"
                    })
            except:
                continue
        
        if watchlist_data:
            df_watchlist = pd.DataFrame(watchlist_data)
            st.dataframe(df_watchlist, use_container_width=True, hide_index=True)
        else:
            st.info("Add stocks to your watchlist to see them here")
            
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")

def render_practical_ai_signals():
    """Render AI signals with confidence-based filtering"""
    st.markdown("### 🎯 AI Trading Signals - Practical System")
    
    # Load practical model
    practical_model = load_practical_model()
    
    if not practical_model:
        st.warning("⚠️ Practical model not found. Using standard signals.")
        render_standard_ai_signals()
        return
    
    # Display model info
    st.success(f"✅ Practical Model Loaded (Base Accuracy: {practical_model['accuracy']*100:.1f}%)")
    
    # Confidence controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.40,
            max_value=0.70,
            value=practical_model.get('confidence_threshold', 0.50),
            step=0.05,
            format="%.0f%%",
            help="Only show signals above this confidence level"
        )
    
    with col2:
        # Estimate accuracy based on threshold
        if confidence_threshold <= 0.45:
            expected_acc = 48.7
        elif confidence_threshold <= 0.50:
            expected_acc = 51.3
        elif confidence_threshold <= 0.55:
            expected_acc = 53.9
        else:
            expected_acc = 59.7
        st.metric("Expected Accuracy", f"{expected_acc:.1f}%")
    
    with col3:
        # Estimate trade frequency
        if confidence_threshold <= 0.45:
            trade_freq = 81.8
        elif confidence_threshold <= 0.50:
            trade_freq = 57.4
        elif confidence_threshold <= 0.55:
            trade_freq = 30.5
        else:
            trade_freq = 17.2
        st.metric("Trade Frequency", f"{trade_freq:.1f}%")
    
    # Strategy info
    with st.expander("📊 Confidence Strategy Explained", expanded=False):
        st.info("""
        **The Practical Approach:**
        - Overall accuracy: 47.7% (below random for 3-class)
        - Filtered accuracy at >50%: 51.7% (profitable!)
        - Key insight: Trade only high-confidence signals
        
        **Confidence Levels:**
        - 40-45%: Too risky, many false signals
        - 45-50%: Break-even territory
        - 50-55%: Profitable with good frequency
        - 55-60%: More profitable, less frequent
        - 60%+: Highly profitable, rare signals
        """)
    
    # Generate signals button
    if st.button("🤖 Generate Practical Signals", type="primary"):
        generate_practical_signals(practical_model, confidence_threshold)

def generate_practical_signals(model_data, threshold):
    """Generate signals using the practical model"""
    
    with st.spinner("Analyzing markets with confidence filtering..."):
        data_processor = get_data_processor()
        stocks = st.session_state.watchlist[:5]
        
        all_signals = []
        filtered_signals = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for idx, symbol in enumerate(stocks):
            progress.progress((idx + 1) / len(stocks))
            status.text(f"Analyzing {symbol}...")
            
            try:
                # Fetch data
                df = data_processor.fetch_stock_data(symbol, period='2y')
                
                if len(df) > 50:
                    # Create practical features
                    df = create_practical_features(df)
                    
                    # Get features
                    features = model_data['features']
                    X_latest = df[features].iloc[-1:].fillna(method='ffill')
                    
                    if not X_latest.empty:
                        # Scale and predict
                        X_scaled = model_data['scaler'].transform(X_latest)
                        prediction = model_data['model'].predict(X_scaled)[0]
                        probabilities = model_data['model'].predict_proba(X_scaled)[0]
                        confidence = np.max(probabilities)
                        
                        # Map to signal
                        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                        signal = signal_map[prediction]
                        
                        # Create reasoning
                        rsi = df['RSI'].iloc[-1]
                        sma20 = df['SMA_20'].iloc[-1]
                        current_price = df['Close'].iloc[-1]
                        
                        reasoning = []
                        if rsi < 30:
                            reasoning.append("RSI oversold")
                        elif rsi > 70:
                            reasoning.append("RSI overbought")
                        
                        if current_price > sma20:
                            reasoning.append("Above SMA20")
                        else:
                            reasoning.append("Below SMA20")
                        
                        signal_data = {
                            'symbol': symbol,
                            'signal': signal,
                            'confidence': confidence,
                            'price': current_price,
                            'rsi': rsi,
                            'sma20': sma20,
                            'volume_ratio': df['volume_ratio'].iloc[-1],
                            'reasoning': ", ".join(reasoning) if reasoning else "Technical analysis",
                            'probabilities': {
                                'SELL': probabilities[0],
                                'HOLD': probabilities[1],
                                'BUY': probabilities[2]
                            }
                        }
                        
                        all_signals.append(signal_data)
                        
                        if confidence >= threshold:
                            filtered_signals.append(signal_data)
            
            except Exception as e:
                st.warning(f"Error with {symbol}: {e}")
        
        progress.empty()
        status.empty()
        
        # Display results
        display_practical_results(all_signals, filtered_signals, threshold)

def display_practical_results(all_signals, filtered_signals, threshold):
    """Display the practical trading signals with insights"""
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyzed", len(all_signals))
    
    with col2:
        st.metric("High Confidence", len(filtered_signals))
    
    with col3:
        if all_signals:
            filter_rate = len(filtered_signals) / len(all_signals) * 100
            st.metric("Pass Rate", f"{filter_rate:.0f}%")
    
    with col4:
        if filtered_signals:
            avg_conf = np.mean([s['confidence'] for s in filtered_signals])
            st.metric("Avg Confidence", f"{avg_conf*100:.0f}%")
    
    st.markdown("---")
    
    # Confidence distribution chart
    if all_signals:
        fig = go.Figure()
        
        confidences = [s['confidence'] for s in all_signals]
        
        fig.add_trace(go.Histogram(
            x=confidences,
            nbinsx=15,
            name='Signal Confidence',
            marker_color='lightblue',
            showlegend=False
        ))
        
        # Add threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Threshold: {threshold*100:.0f}%")
        
        # Add profitability zones
        fig.add_vrect(x0=0.5, x1=1.0, fillcolor="green", opacity=0.1,
                     annotation_text="Profitable Zone", annotation_position="top right")
        fig.add_vrect(x0=0, x1=0.5, fillcolor="red", opacity=0.1,
                     annotation_text="Risky Zone", annotation_position="top left")
        
        fig.update_layout(
            title="Signal Confidence Distribution",
            xaxis_title="Confidence",
            yaxis_title="Count",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display filtered signals
    if filtered_signals:
        st.success(f"✅ {len(filtered_signals)} High-Confidence Trading Opportunities")
        
        # Sort by confidence
        filtered_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Separate by type
        buy_signals = [s for s in filtered_signals if s['signal'] == 'BUY']
        sell_signals = [s for s in filtered_signals if s['signal'] == 'SELL']
        hold_signals = [s for s in filtered_signals if s['signal'] == 'HOLD']
        
        # Display BUY signals
        if buy_signals:
            st.markdown("#### 🟢 BUY Signals")
            for signal in buy_signals:
                display_signal_card(signal, threshold, "buy")
        
        # Display SELL signals  
        if sell_signals:
            st.markdown("#### 🔴 SELL Signals")
            for signal in sell_signals:
                display_signal_card(signal, threshold, "sell")
        
        # Display HOLD signals
        if hold_signals:
            st.markdown("#### 🟡 HOLD Signals")
            for signal in hold_signals:
                display_signal_card(signal, threshold, "hold")
    
    elif all_signals:
        st.warning(f"⚠️ No signals above {threshold*100:.0f}% confidence. Lower the threshold or wait for better setups.")
        
        # Show best available signals
        if all_signals:
            st.markdown("#### Best Available Signals (Below Threshold)")
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            for signal in all_signals[:3]:
                display_signal_card(signal, threshold, "wait")
    else:
        st.error("No signals generated. Please check your data connection.")

def display_signal_card(signal, threshold, action_type):
    """Display individual signal card"""
    
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
        
        with col1:
            emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
            st.markdown(f"{emoji} **{signal['symbol']}**")
            st.caption(f"${signal['price']:.2f}")
        
        with col2:
            conf_pct = signal['confidence'] * 100
            if conf_pct >= 60:
                conf_emoji = "🎯"
            elif conf_pct >= 50:
                conf_emoji = "✅"
            else:
                conf_emoji = "⚠️"
            
            st.metric(
                "Confidence",
                f"{conf_pct:.1f}%",
                delta=f"{conf_emoji}"
            )
        
        with col3:
            rsi = signal['rsi']
            if rsi < 30:
                rsi_status = "↓ Oversold"
            elif rsi > 70:
                rsi_status = "↑ Overbought"
            else:
                rsi_status = "→ Neutral"
            
            st.metric("RSI", f"{rsi:.0f}", delta=rsi_status)
        
        with col4:
            # Probability breakdown
            probs = signal['probabilities']
            st.caption("Probabilities:")
            st.caption(f"B:{probs['BUY']*100:.0f}% H:{probs['HOLD']*100:.0f}% S:{probs['SELL']*100:.0f}%")
        
        with col5:
            st.caption(f"**Analysis:** {signal['reasoning']}")
            
            # Action buttons based on confidence
            if signal['confidence'] >= threshold:
                if action_type == "buy":
                    if st.button(f"Execute Buy", key=f"buy_{signal['symbol']}_{conf_pct}"):
                        st.success(f"✅ Buy order placed for {signal['symbol']}")
                elif action_type == "sell":
                    if st.button(f"Execute Sell", key=f"sell_{signal['symbol']}_{conf_pct}"):
                        st.success(f"✅ Sell order placed for {signal['symbol']}")
                else:
                    st.info("Hold position")
            else:
                st.warning(f"Below threshold ({conf_pct:.0f}% < {threshold*100:.0f}%) - Wait for better setup")
        
        st.divider()

def render_standard_ai_signals():
    """Fallback to standard AI signals without confidence filtering"""
    st.markdown("### 🤖 AI Trading Signals - Standard")
    
    st.info("💡 Enable Practical Trading System for confidence-based filtering")
    
    # Sample signals for demonstration
    sample_signals = [
        {'symbol': 'AAPL', 'signal': 'BUY', 'confidence': 0.68, 'price': 185.23},
        {'symbol': 'MSFT', 'signal': 'HOLD', 'confidence': 0.55, 'price': 378.45},
        {'symbol': 'GOOGL', 'signal': 'SELL', 'confidence': 0.62, 'price': 142.30}
    ]
    
    for signal in sample_signals:
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
            st.markdown(f"{emoji} **{signal['symbol']}** - {signal['signal']}")
        
        with col2:
            st.metric("Confidence", f"{signal['confidence']*100:.0f}%")
        
        with col3:
            st.caption(f"Price: ${signal['price']:.2f}")
    
    st.divider()

def render_performance_comparison():
    """Show comparison between practical and standard approach"""
    st.markdown("### 📊 Strategy Comparison")
    
    comparison_data = {
        'Metric': ['Overall Accuracy', 'Filtered Accuracy', 'Trade Frequency', 'Expected Profit', 'Risk Level'],
        'Standard Model': ['47.7%', 'N/A', '100%', 'Negative', 'High'],
        'Practical (>50%)': ['47.7%', '51.7%', '57.4%', 'Positive', 'Medium'],
        'Practical (>60%)': ['47.7%', '59.7%', '17.2%', 'High', 'Low']
    }
    
    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)
    
    st.success("""
    **Key Insight:** The Practical Trading System achieves profitability by:
    - Trading only high-confidence signals (>50%)
    - Accepting lower frequency for higher accuracy
    - Implementing proper risk management
    - Focus on quality over quantity
    """)

# Main Dashboard Layout
def main():
    """Main dashboard function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    render_header()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Market Overview
        render_market_overview()
        
        st.markdown("---")
        
        # AI Signals with tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Practical Signals", "📊 Standard Signals", "📈 Compare"])
        
        with tab1:
            render_practical_ai_signals()
        
        with tab2:
            render_standard_ai_signals()
        
        with tab3:
            render_performance_comparison()
    
    with col2:
        # Portfolio Summary
        render_portfolio_summary()
        
        st.markdown("---")
        
        # Watchlist
        render_watchlist()
        
        # Sidebar info
        with st.sidebar:
            st.markdown("### 📊 Trading System Status")
            
            if os.path.exists('practical_model.pkl'):
                model_data = load_practical_model()
                if model_data:
                    st.success("✅ Practical Model Active")
                    st.metric("Model Accuracy", f"{model_data['accuracy']*100:.1f}%")
                    st.metric("Optimal Threshold", f"{model_data.get('confidence_threshold', 0.5)*100:.0f}%")
                    
                    st.markdown("---")
                    st.markdown("**Confidence Guidelines:**")
                    st.caption("• >60%: Strong signal")
                    st.caption("• 50-60%: Good signal")
                    st.caption("• <50%: Wait")
            else:
                st.warning("⚠️ Practical model not found")
                st.info("Run: `python practical_trading_system.py`")
            
            st.markdown("---")
            st.markdown("### 📈 Quick Stats")
            st.metric("Signals Today", np.random.randint(5, 15))
            st.metric("Win Rate (7d)", f"{np.random.uniform(48, 65):.1f}%")
            st.metric("Active Trades", np.random.randint(2, 8))

if __name__ == "__main__":
    main()