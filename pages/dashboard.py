# pages/dashboard.py
"""
Dashboard Page - Enhanced with Practical Trading System & 33 Stocks Support
Main dashboard with market overview, portfolio summary, and AI insights with confidence filtering
Includes support for Real Estate REITs and expanded Financial stocks
PRESERVES PRACTICAL AI SIGNALS FEATURE
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
            'preferred_sectors': ['Technology', 'Healthcare', 'Real Estate'],  # Added Real Estate
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
    
    # Watchlist - Updated with new stocks
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'MA', 'PLD', 'O']
    
    # Market Data Cache
    if 'market_data_cache' not in st.session_state:
        st.session_state.market_data_cache = {}
    
    # AI Predictions Cache
    if 'predictions_cache' not in st.session_state:
        st.session_state.predictions_cache = {}
    
    # Practical Model Cache
    if 'practical_model_cache' not in st.session_state:
        st.session_state.practical_model_cache = {}
    
    # Settings
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'theme': 'light',
            'auto_refresh': False,
            'refresh_interval': 60,
            'show_notifications': True,
            'data_source': 'yfinance',
            'confidence_threshold': 0.70,  # For practical AI signals
            'use_practical_system': True    # Enable practical trading system
        }

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from yaml file"""
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}

def get_all_stocks():
    """Get all available stocks from configuration - including all 33 stocks"""
    config = load_config()
    all_stocks = []
    
    # Get stocks from each sector - INCLUDING REAL ESTATE
    for sector in ['technology', 'financial', 'real_estate', 'healthcare']:
        if sector in config.get('stocks', {}):
            sector_stocks = config['stocks'][sector]
            for stock in sector_stocks:
                if isinstance(stock, dict) and 'symbol' in stock:
                    all_stocks.append(stock['symbol'])
                elif isinstance(stock, str):
                    all_stocks.append(stock)
    
    # Add benchmarks
    if 'benchmarks' in config.get('stocks', {}):
        for benchmark in config['stocks']['benchmarks']:
            if isinstance(benchmark, dict) and 'symbol' in benchmark:
                all_stocks.append(benchmark['symbol'])
            elif isinstance(benchmark, str):
                all_stocks.append(benchmark)
    
    return all_stocks

def load_practical_model():
    """Load or create practical trading model with confidence filtering"""
    try:
        # Check cache first
        if 'practical_model' in st.session_state.practical_model_cache:
            model_data = st.session_state.practical_model_cache['practical_model']
            if datetime.now() - model_data['timestamp'] < timedelta(hours=1):
                return model_data['model']
        
        # Try loading from disk
        model_path = 'models/practical_trading_model.pkl'
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info("Loaded practical trading model from disk")
                
                # Cache the model
                st.session_state.practical_model_cache['practical_model'] = {
                    'model': model,
                    'timestamp': datetime.now()
                }
                return model
            except Exception as e:
                logger.warning(f"Could not load model from disk: {e}")
        
        # Return None if no trained model available
        # We'll use rule-based signals instead
        logger.info("No trained model available, will use rule-based signals")
        return None
        
    except Exception as e:
        logger.error(f"Error in load_practical_model: {e}")
        return None

def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("📊 StockBot Advisor Dashboard")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        view_mode = st.selectbox(
            "View Mode",
            ["Overview", "Detailed", "Compact"],
            label_visibility="collapsed"
        )

def render_market_overview():
    """Render market overview section"""
    st.markdown("### 🌍 Market Overview")
    
    data_processor = get_data_processor()
    market_data = data_processor.get_market_overview()
    
    if market_data:
        cols = st.columns(len(market_data))
        
        for idx, (symbol, data) in enumerate(market_data.items()):
            with cols[idx]:
                color = "green" if data['change'] >= 0 else "red"
                arrow = "↑" if data['change'] >= 0 else "↓"
                
                st.metric(
                    label=data['name'],
                    value=f"${data['price']:,.2f}",
                    delta=f"{arrow} {data['change_pct']:.2f}%"
                )

def render_sector_performance():
    """Render sector performance - INCLUDING REAL ESTATE"""
    st.markdown("### 📈 Sector Performance")
    
    data_processor = get_data_processor()
    
    # Calculate sector performance for all sectors
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'PLTR'],
        'Financial': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'WFC', 'AXP', 'BLK', 'SPGI'],
        'Real Estate': ['PLD', 'AMT', 'EQIX', 'SPG', 'O'],
        'Healthcare': ['JNJ', 'PFE', 'MRNA', 'UNH']
    }
    
    sector_data = []
    for sector_name, symbols in sectors.items():
        # Calculate average performance
        momentum = data_processor.calculate_sector_momentum(sector_name.lower().replace(' ', '_'), period=20)
        sector_data.append({
            'Sector': sector_name,
            'Stocks': len(symbols),
            'Momentum (20d)': momentum,
            'Status': '🟢' if momentum > 0 else '🔴'
        })
    
    df_sectors = pd.DataFrame(sector_data)
    
    # Display as columns
    cols = st.columns(len(sector_data))
    for idx, row in df_sectors.iterrows():
        with cols[idx]:
            st.metric(
                label=row['Sector'],
                value=f"{row['Status']} {row['Momentum (20d)']:.2f}%",
                delta=f"{row['Stocks']} stocks"
            )

def render_portfolio_summary():
    """Render portfolio summary section"""
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
    """Render watchlist section with all stocks available"""
    st.markdown("### 👀 Watchlist")
    
    # Add stock selector to add to watchlist
    col1, col2 = st.columns([3, 1])
    
    with col1:
        all_stocks = get_all_stocks()
        available_stocks = [s for s in all_stocks if s not in st.session_state.watchlist]
        
        if available_stocks:
            selected_to_add = st.selectbox(
                "Add to watchlist",
                ["Select a stock..."] + available_stocks,
                label_visibility="collapsed"
            )
            
            if selected_to_add != "Select a stock...":
                st.session_state.watchlist.append(selected_to_add)
                st.rerun()
    
    with col2:
        if st.button("Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
    
    # Display watchlist
    if st.session_state.watchlist:
        try:
            data_processor = get_data_processor()
            watchlist_data = []
            
            for symbol in st.session_state.watchlist[:10]:  # Limit to 10 for display
                try:
                    df = data_processor.fetch_stock_data(symbol, period='1d')
                    if not df.empty:
                        current_price = df['Close'].iloc[-1]
                        prev_close = df['Open'].iloc[0]
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        volume = df['Volume'].iloc[-1]
                        
                        # Identify sector
                        config = load_config()
                        sector = "Unknown"
                        for sec in ['technology', 'financial', 'real_estate', 'healthcare']:
                            if sec in config.get('stocks', {}):
                                sector_stocks = [s['symbol'] if isinstance(s, dict) else s 
                                               for s in config['stocks'][sec]]
                                if symbol in sector_stocks:
                                    sector = sec.capitalize()
                                    break
                        
                        watchlist_data.append({
                            'Symbol': symbol,
                            'Sector': sector,
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
    else:
        st.info("Your watchlist is empty. Add stocks from the dropdown above.")

def render_practical_ai_signals():
    """Render AI signals with confidence-based filtering - PRACTICAL SYSTEM"""
    st.markdown("### 🎯 AI Trading Signals - Practical System")
    
    # Configuration controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Sector filter including Real Estate
        config = load_config()
        sectors = ['All', 'Technology', 'Financial', 'Real Estate', 'Healthcare']
        selected_sector = st.selectbox("Filter by Sector", sectors, key='practical_sector')
    
    with col2:
        # Confidence threshold
        confidence_threshold = st.slider(
            "Min Confidence",
            min_value=0.5,
            max_value=0.9,
            value=st.session_state.settings.get('confidence_threshold', 0.70),
            step=0.05,
            key='confidence_threshold'
        )
        st.session_state.settings['confidence_threshold'] = confidence_threshold
    
    with col3:
        # Signal type filter
        signal_filter = st.selectbox(
            "Signal Type",
            ["All", "Buy Only", "Sell Only", "Strong Signals"],
            key='signal_filter'
        )
    
    # Load practical model
    practical_model = load_practical_model()
    
    # Check if model is trained (has estimators_ attribute)
    model_is_trained = False
    if practical_model is not None:
        try:
            # Check if model is fitted by checking for estimators_
            if hasattr(practical_model, 'estimators_'):
                model_is_trained = True
                st.success("✅ Using trained ML model for predictions")
        except:
            model_is_trained = False
    
    if not model_is_trained:
        st.info("ℹ️ Using rule-based technical analysis for signals")
    
    # Get stocks based on filter
    if selected_sector == 'All':
        stocks_to_analyze = get_all_stocks()[:20]  # Limit for performance
    else:
        data_processor = get_data_processor()
        stocks_to_analyze = data_processor.get_stocks_by_sector(selected_sector.lower().replace(' ', '_'))
    
    if stocks_to_analyze:
        signals_data = []
        data_processor = get_data_processor()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(stocks_to_analyze):
            try:
                # Update progress
                progress = (idx + 1) / len(stocks_to_analyze)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {symbol}... ({idx+1}/{len(stocks_to_analyze)})")
                
                # Fetch data
                df = data_processor.fetch_stock_data(symbol, period='3mo')
                if df.empty or len(df) < 60:
                    continue
                
                # Calculate indicators
                df = TechnicalIndicators.calculate_all_indicators(df)
                
                # Get latest data
                latest = df.iloc[-1]
                
                # Generate prediction with confidence
                if model_is_trained and practical_model is not None:
                    # Use ML model if available and trained
                    try:
                        # Prepare features for practical model
                        feature_cols = ['RSI', 'MACD', 'BB_Position', 'Volume_Ratio', 'ATR']
                        available_features = [col for col in feature_cols if col in df.columns]
                        
                        if len(available_features) >= 3:
                            X_latest = df[available_features].iloc[-1:].values
                            
                            # Scale features (simple normalization for demo)
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(df[available_features].dropna())
                            X_latest_scaled = X_scaled[-1:] if len(X_scaled) > 0 else X_latest
                            
                            prediction_proba = practical_model.predict_proba(X_latest_scaled)[0]
                            prediction = np.argmax(prediction_proba)
                            confidence = prediction_proba[prediction]
                        else:
                            # Fallback to rule-based
                            prediction, confidence = generate_practical_signal(latest)
                    except Exception as e:
                        logger.debug(f"ML prediction failed for {symbol}, using rule-based: {e}")
                        prediction, confidence = generate_practical_signal(latest)
                else:
                    # Use rule-based practical signal
                    prediction, confidence = generate_practical_signal(latest)
                
                # Filter by confidence threshold
                if confidence < confidence_threshold:
                    continue
                
                # Map prediction to signal
                signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                signal = signal_map.get(prediction, "HOLD")
                
                # Apply signal filter
                if signal_filter == "Buy Only" and signal != "BUY":
                    continue
                elif signal_filter == "Sell Only" and signal != "SELL":
                    continue
                elif signal_filter == "Strong Signals" and confidence < 0.80:
                    continue
                
                # Determine signal strength
                if confidence >= 0.85:
                    signal_strength = "Strong"
                    emoji = "🔥"
                elif confidence >= 0.75:
                    signal_strength = "Moderate"
                    emoji = "✅"
                else:
                    signal_strength = "Weak"
                    emoji = "⚠️"
                
                # Risk assessment
                volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
                risk_level = "High" if volatility > 30 else "Medium" if volatility > 20 else "Low"
                
                # Entry/Exit points (simplified)
                if signal == "BUY":
                    entry_point = latest['Close'] * 0.995  # 0.5% below current
                    stop_loss = entry_point * 0.95  # 5% stop loss
                    take_profit = entry_point * 1.10  # 10% take profit
                elif signal == "SELL":
                    entry_point = latest['Close'] * 1.005  # 0.5% above current
                    stop_loss = entry_point * 1.05  # 5% stop loss
                    take_profit = entry_point * 0.90  # 10% take profit
                else:
                    entry_point = latest['Close']
                    stop_loss = entry_point * 0.95
                    take_profit = entry_point * 1.05
                
                signals_data.append({
                    'Symbol': symbol,
                    'Signal': f"{emoji} {signal}",
                    'Strength': signal_strength,
                    'Confidence': f"{confidence:.1%}",
                    'Price': f"${latest['Close']:.2f}",
                    'Entry': f"${entry_point:.2f}",
                    'Stop Loss': f"${stop_loss:.2f}",
                    'Target': f"${take_profit:.2f}",
                    'Risk': risk_level,
                    'RSI': f"{latest.get('RSI', 50):.1f}",
                    'Volume': f"{latest['Volume']/1e6:.1f}M"
                })
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if signals_data:
            # Sort by confidence and signal strength
            signals_data.sort(key=lambda x: (
                x['Strength'] == 'Strong',
                x['Strength'] == 'Moderate',
                float(x['Confidence'].strip('%')) / 100
            ), reverse=True)
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                buy_signals = sum(1 for s in signals_data if 'BUY' in s['Signal'])
                st.metric("Buy Signals", buy_signals)
            
            with col2:
                sell_signals = sum(1 for s in signals_data if 'SELL' in s['Signal'])
                st.metric("Sell Signals", sell_signals)
            
            with col3:
                strong_signals = sum(1 for s in signals_data if s['Strength'] == 'Strong')
                st.metric("Strong Signals", strong_signals)
            
            with col4:
                avg_confidence = np.mean([float(s['Confidence'].strip('%'))/100 for s in signals_data])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Display signals table
            df_signals = pd.DataFrame(signals_data)
            
            # Apply color coding
            def highlight_signal(row):
                if 'BUY' in row['Signal']:
                    return ['background-color: #d4edda'] * len(row)
                elif 'SELL' in row['Signal']:
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = df_signals.style.apply(highlight_signal, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Trading recommendations
            st.markdown("---")
            st.markdown("#### 📋 Trading Recommendations")
            
            if strong_signals > 0:
                st.success(f"🔥 **{strong_signals} Strong Signal(s) Detected!** Consider these for immediate action.")
            
            if buy_signals > sell_signals:
                st.info("📈 **Market Sentiment: Bullish** - More buy signals detected")
            elif sell_signals > buy_signals:
                st.info("📉 **Market Sentiment: Bearish** - More sell signals detected")
            else:
                st.info("➡️ **Market Sentiment: Neutral** - Mixed signals")
            
            # Risk warning
            st.warning("""
            ⚠️ **Risk Disclaimer:** 
            - These signals are for educational purposes only
            - Always use stop-loss orders to manage risk
            - Never invest more than you can afford to lose
            - Consider your personal risk tolerance and investment goals
            """)
            
        else:
            st.info(f"No signals meet the confidence threshold of {confidence_threshold:.0%}")
    else:
        st.info("No stocks available for the selected sector.")

def generate_practical_signal(latest_data):
    """Generate practical trading signal with REALISTIC confidence scores"""
    signal_score = 0
    confidence_factors = []
    signal_strength = []
    
    # RSI Signal with realistic confidence
    if 'RSI' in latest_data.index:
        rsi = latest_data['RSI']
        if rsi < 25:  # Very oversold
            signal_score += 2
            confidence_factors.append(0.75)
            signal_strength.append("strong")
        elif rsi < 35:  # Oversold
            signal_score += 1
            confidence_factors.append(0.55)
            signal_strength.append("moderate")
        elif rsi > 75:  # Very overbought
            signal_score -= 2
            confidence_factors.append(0.75)
            signal_strength.append("strong")
        elif rsi > 65:  # Overbought
            signal_score -= 1
            confidence_factors.append(0.55)
            signal_strength.append("moderate")
        else:  # Neutral zone
            confidence_factors.append(0.35)  # Low confidence in neutral
            signal_strength.append("weak")
    
    # MACD Signal with variable confidence
    if 'MACD' in latest_data.index and 'MACD_Signal' in latest_data.index:
        macd_diff = abs(latest_data['MACD'] - latest_data['MACD_Signal'])
        
        if latest_data['MACD'] > latest_data['MACD_Signal']:
            signal_score += 1
            # Confidence based on divergence strength
            if macd_diff > latest_data['Close'] * 0.01:  # Strong divergence
                confidence_factors.append(0.65)
                signal_strength.append("moderate")
            else:
                confidence_factors.append(0.45)  # Weak divergence
                signal_strength.append("weak")
        else:
            signal_score -= 1
            if macd_diff > latest_data['Close'] * 0.01:
                confidence_factors.append(0.65)
                signal_strength.append("moderate")
            else:
                confidence_factors.append(0.45)
                signal_strength.append("weak")
    
    # Bollinger Band Position with dynamic confidence
    if 'BB_Position' in latest_data.index:
        bb_pos = latest_data['BB_Position']
        if bb_pos < 0.1:  # Very close to lower band
            signal_score += 1
            confidence_factors.append(0.70)
            signal_strength.append("moderate")
        elif bb_pos < 0.3:  # Near lower band
            signal_score += 0.5
            confidence_factors.append(0.50)
            signal_strength.append("weak")
        elif bb_pos > 0.9:  # Very close to upper band
            signal_score -= 1
            confidence_factors.append(0.70)
            signal_strength.append("moderate")
        elif bb_pos > 0.7:  # Near upper band
            signal_score -= 0.5
            confidence_factors.append(0.50)
            signal_strength.append("weak")
        else:  # Middle of bands
            confidence_factors.append(0.30)  # Very low confidence
            signal_strength.append("neutral")
    
    # Volume Signal - affects confidence, not direction
    if 'Volume_Ratio' in latest_data.index:
        vol_ratio = latest_data['Volume_Ratio']
        if vol_ratio > 2.0:  # Very high volume
            # Boost confidence by 20%
            confidence_boost = 0.20
        elif vol_ratio > 1.5:  # High volume
            confidence_boost = 0.10
        elif vol_ratio > 1.0:  # Above average
            confidence_boost = 0.05
        else:  # Below average volume
            confidence_boost = -0.10  # Reduce confidence
        
        # Apply volume boost to existing confidence
        if confidence_factors:
            confidence_factors = [min(0.95, c + confidence_boost) for c in confidence_factors]
    
    # Additional penalty for conflicting signals
    if signal_strength.count("strong") == 0 and signal_strength.count("moderate") < 2:
        # No strong signals and few moderate ones = lower confidence
        confidence_penalty = 0.15
        confidence_factors = [max(0.2, c - confidence_penalty) for c in confidence_factors]
    
    # Determine signal
    if signal_score >= 2:
        prediction = 2  # BUY
    elif signal_score <= -2:
        prediction = 0  # SELL
    else:
        prediction = 1  # HOLD
    
    # Calculate realistic confidence
    if confidence_factors:
        base_confidence = np.mean(confidence_factors)
        
        # Adjust confidence based on signal strength consistency
        if prediction == 1:  # HOLD signal
            # HOLD signals should have lower confidence
            confidence = base_confidence * 0.8
        elif abs(signal_score) >= 3:  # Very strong signal
            confidence = min(0.85, base_confidence * 1.1)
        else:
            confidence = base_confidence
        
        # Add some randomness for realism (±5%)
        confidence = confidence * (0.95 + np.random.random() * 0.1)
        
        # Ensure confidence stays in reasonable range
        confidence = max(0.25, min(0.85, confidence))
    else:
        confidence = 0.30  # Default low confidence
    
    return prediction, confidence

def render_standard_ai_signals():
    """Fallback to standard AI signals if practical system unavailable"""
    st.markdown("### 🤖 AI Trading Signals (Standard)")
    
    # This is a simplified version, you can expand as needed
    stocks = get_all_stocks()[:10]
    signals_data = []
    
    for symbol in stocks:
        # Generate random signal for demo
        signal = np.random.choice(['BUY', 'HOLD', 'SELL'])
        confidence = np.random.uniform(0.6, 0.9)
        
        signals_data.append({
            'Symbol': symbol,
            'Signal': signal,
            'Confidence': f"{confidence:.1%}"
        })
    
    df_signals = pd.DataFrame(signals_data)
    st.dataframe(df_signals, use_container_width=True, hide_index=True)

def render_news_sentiment():
    """Render news and sentiment section"""
    st.markdown("### 📰 Market News & Sentiment")
    
    # Placeholder for news sentiment
    news_data = [
        {"headline": "Tech stocks rally on AI optimism", "sentiment": "Positive", "impact": "High"},
        {"headline": "Federal Reserve hints at rate stability", "sentiment": "Neutral", "impact": "Medium"},
        {"headline": "Real Estate REITs show strong dividend yields", "sentiment": "Positive", "impact": "Medium"},
        {"headline": "Banking sector faces regulatory scrutiny", "sentiment": "Negative", "impact": "Low"},
    ]
    
    for news in news_data:
        sentiment_color = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}
        col1, col2, col3 = st.columns([5, 1, 1])
        
        with col1:
            st.write(news['headline'])
        with col2:
            st.write(f"{sentiment_color[news['sentiment']]} {news['sentiment']}")
        with col3:
            st.write(f"Impact: {news['impact']}")

def main():
    """Main dashboard function"""
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Create main layout
    st.markdown("---")
    
    # First row - Market Overview and Sector Performance
    col1, col2 = st.columns([3, 2])
    
    with col1:
        render_market_overview()
    
    with col2:
        render_sector_performance()
    
    st.markdown("---")
    
    # Second row - Portfolio and Watchlist
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_portfolio_summary()
    
    with col2:
        render_watchlist()
    
    st.markdown("---")
    
    # Third row - PRACTICAL AI Signals (Main Feature)
    if st.session_state.settings.get('use_practical_system', True):
        render_practical_ai_signals()
    else:
        render_standard_ai_signals()
    
    st.markdown("---")
    
    # Fourth row - News
    render_news_sentiment()
    
    # Footer
    st.markdown("---")
    st.caption("""
    💡 **Tips**: 
    • Practical AI System filters signals by confidence threshold
    • Adjust minimum confidence to see more or fewer signals
    • Strong signals (>85% confidence) are highlighted with 🔥
    • Now tracking 33 stocks across 4 sectors including REITs
    • Entry points and stop-loss levels are automatically calculated
    """)

if __name__ == "__main__":
    if not PLOTLY_AVAILABLE:
        st.error("This dashboard requires plotly. Please install it:")
        st.code("pip install plotly")
        st.stop()
    
    main()