# pages/2_🔍_Analysis.py
"""
Stock Analysis Page - Complete Working Version
Includes fixed AI predictions and all features
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Analysis - StockBot Advisor",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
def initialize_session_state():
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = 'AAPL'
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = '3mo'

# Technical Indicators Class (Fallback if main module not available)
class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(data, period=14):
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return {'MACD': macd, 'Signal': signal_line, 'Histogram': histogram}
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {'Upper': upper, 'Middle': sma, 'Lower': lower}
    
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all technical indicators"""
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'])
        
        # MACD
        macd_result = TechnicalIndicators.calculate_macd(df['Close'])
        df['MACD'] = macd_result['MACD']
        df['MACD_Signal'] = macd_result['Signal']
        df['MACD_Histogram'] = macd_result['Histogram']
        
        # Bollinger Bands
        bb_result = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_result['Upper']
        df['BB_Middle'] = bb_result['Middle']
        df['BB_Lower'] = bb_result['Lower']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price patterns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        return df

def fetch_stock_data(symbol, period):
    """Fetch stock data with proper error handling"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Map period to specific date range
        period_days = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        
        days = period_days.get(period, 90)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        # Remove timezone if present
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        # Get stock info
        try:
            info = ticker.info
        except:
            info = {}
        
        return df, info
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame(), {}

def generate_ai_prediction(df, symbol):
    """Generate AI prediction with minimum 30 data points"""
    
    MIN_DATA_POINTS = 30  # Reduced from 100
    
    if len(df) < MIN_DATA_POINTS:
        return None, f"Need at least {MIN_DATA_POINTS} data points. Currently have {len(df)}. Try selecting a longer time period."
    
    try:
        # Ensure indicators are calculated
        if 'RSI' not in df.columns:
            df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        # Initialize scoring system
        buy_signals = 0
        sell_signals = 0
        confidence_score = 50  # Base confidence
        
        factors = {}
        
        # 1. RSI Analysis
        if 'RSI' in df.columns and pd.notna(latest['RSI']):
            rsi = latest['RSI']
            factors['RSI'] = f"{rsi:.2f}"
            
            if rsi < 30:
                buy_signals += 2
                confidence_score += 10
                factors['RSI_Signal'] = "Oversold (Bullish)"
            elif rsi > 70:
                sell_signals += 2
                confidence_score += 10
                factors['RSI_Signal'] = "Overbought (Bearish)"
            elif 40 <= rsi <= 60:
                confidence_score += 5
                factors['RSI_Signal'] = "Neutral"
        
        # 2. Moving Average Analysis
        if 'SMA_20' in df.columns and pd.notna(latest['SMA_20']):
            price_vs_sma20 = ((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100
            factors['Price vs SMA20'] = f"{price_vs_sma20:+.2f}%"
            
            if latest['Close'] > latest['SMA_20']:
                buy_signals += 1
                if price_vs_sma20 > 2:
                    confidence_score += 5
            else:
                sell_signals += 1
                if price_vs_sma20 < -2:
                    confidence_score += 5
        
        # 3. MACD Analysis
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
                macd_diff = latest['MACD'] - latest['MACD_Signal']
                factors['MACD Histogram'] = f"{macd_diff:.4f}"
                
                if latest['MACD'] > latest['MACD_Signal']:
                    buy_signals += 1
                    if previous['MACD'] <= previous['MACD_Signal']:  # Crossover
                        buy_signals += 1
                        confidence_score += 10
                else:
                    sell_signals += 1
                    if previous['MACD'] >= previous['MACD_Signal']:  # Crossover
                        sell_signals += 1
                        confidence_score += 10
        
        # 4. Momentum Analysis
        if len(df) >= 5:
            momentum_5d = ((latest['Close'] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
            factors['5-Day Momentum'] = f"{momentum_5d:+.2f}%"
            
            if momentum_5d > 3:
                buy_signals += 1
                confidence_score += 5
            elif momentum_5d < -3:
                sell_signals += 1
                confidence_score += 5
        
        # 5. Volume Analysis
        if 'Volume_Ratio' in df.columns and pd.notna(latest['Volume_Ratio']):
            vol_ratio = latest['Volume_Ratio']
            factors['Volume Ratio'] = f"{vol_ratio:.2f}x"
            
            if vol_ratio > 1.5:
                confidence_score += 5  # High volume increases confidence
                if latest['Close'] > previous['Close']:
                    buy_signals += 1
                else:
                    sell_signals += 1
        
        # 6. Bollinger Bands Analysis
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            if pd.notna(latest['BB_Upper']) and pd.notna(latest['BB_Lower']):
                bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                factors['BB Position'] = f"{bb_position:.2%}"
                
                if bb_position < 0.2:  # Near lower band
                    buy_signals += 1
                    confidence_score += 5
                elif bb_position > 0.8:  # Near upper band
                    sell_signals += 1
                    confidence_score += 5
        
        # Calculate final prediction
        total_signals = buy_signals + sell_signals
        
        if total_signals == 0:
            prediction = "HOLD"
            confidence = 50
        else:
            buy_ratio = buy_signals / total_signals
            
            if buy_ratio > 0.6:
                prediction = "BUY"
                confidence = min(90, confidence_score + (buy_ratio - 0.5) * 50)
            elif buy_ratio < 0.4:
                prediction = "SELL"
                confidence = min(90, confidence_score + (0.5 - buy_ratio) * 50)
            else:
                prediction = "HOLD"
                confidence = min(80, confidence_score)
        
        # Ensure minimum confidence
        confidence = max(40, min(95, confidence))
        
        return {
            'signal': prediction,
            'confidence': confidence,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'factors': factors
        }, None
        
    except Exception as e:
        return None, f"Error generating prediction: {str(e)}"

def render_price_chart(df, symbol):
    """Create interactive price chart with indicators"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f"{symbol} Price", "RSI", "MACD", "Volume")
    )
    
    # 1. Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add moving averages
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
    
    # Add Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1.5),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=0.5, row=2, col=1)
    
    # 3. MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=1.5)
            ),
            row=3, col=1
        )
        
        # MACD Histogram
        if 'MACD_Histogram' in df.columns:
            colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name='Histogram',
                    marker_color=colors,
                    showlegend=False
                ),
                row=3, col=1
            )
    
    # 4. Volume
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        margin=dict(r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def render_technical_indicators(df):
    """Display technical indicators in organized format"""
    
    st.markdown("### 📊 Current Technical Indicators")
    
    if df.empty:
        st.warning("No data available")
        return
    
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    
    # Create 4 columns for indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Trend Indicators**")
        
        # Price
        price_change = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
        st.metric(
            "Price",
            f"${latest['Close']:.2f}",
            f"{price_change:+.2f}%"
        )
        
        # SMA 20
        if 'SMA_20' in df.columns and pd.notna(latest['SMA_20']):
            sma20_diff = ((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100
            st.metric(
                "vs SMA 20",
                f"${latest['SMA_20']:.2f}",
                f"{sma20_diff:+.2f}%"
            )
    
    with col2:
        st.markdown("**Momentum**")
        
        # RSI
        if 'RSI' in df.columns and pd.notna(latest['RSI']):
            rsi = latest['RSI']
            rsi_status = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "⚪"
            st.metric(
                "RSI (14)",
                f"{rsi:.2f}",
                f"{rsi_status} {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}"
            )
        
        # MACD
        if 'MACD' in df.columns and pd.notna(latest['MACD']):
            macd_signal = "📈" if latest['MACD'] > latest.get('MACD_Signal', 0) else "📉"
            st.metric(
                "MACD",
                f"{latest['MACD']:.4f}",
                f"{macd_signal} {'Bullish' if latest['MACD'] > latest.get('MACD_Signal', 0) else 'Bearish'}"
            )
    
    with col3:
        st.markdown("**Volatility**")
        
        # Bollinger Bands
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            if pd.notna(latest['BB_Upper']) and pd.notna(latest['BB_Lower']):
                bb_width = latest['BB_Upper'] - latest['BB_Lower']
                bb_position = (latest['Close'] - latest['BB_Lower']) / bb_width if bb_width > 0 else 0.5
                st.metric(
                    "BB Position",
                    f"{bb_position:.2%}",
                    "Upper" if bb_position > 0.8 else "Lower" if bb_position < 0.2 else "Middle"
                )
        
        # Daily Volatility
        if 'Volatility' in df.columns and pd.notna(latest['Volatility']):
            st.metric(
                "Volatility (20d)",
                f"{latest['Volatility']*100:.2f}%"
            )
    
    with col4:
        st.markdown("**Volume**")
        
        # Volume
        volume_m = latest['Volume'] / 1_000_000
        st.metric(
            "Volume",
            f"{volume_m:.2f}M"
        )
        
        # Volume Ratio
        if 'Volume_Ratio' in df.columns and pd.notna(latest['Volume_Ratio']):
            vol_status = "🔥" if latest['Volume_Ratio'] > 1.5 else "📊"
            st.metric(
                "vs Avg Volume",
                f"{latest['Volume_Ratio']:.2f}x",
                f"{vol_status} {'High' if latest['Volume_Ratio'] > 1.5 else 'Normal'}"
            )

def render_ai_section(df, symbol):
    """Render AI prediction section with detailed analysis"""
    
    st.markdown("### 🤖 AI-Powered Analysis")
    
    # Generate prediction
    prediction, error = generate_ai_prediction(df, symbol)
    
    if error:
        st.warning(f"⚠️ {error}")
        
        # Provide helpful guidance
        if "at least" in error:
            st.info(
                "💡 **Tip:** For accurate AI predictions, select a longer time period:\n"
                "- Minimum: 1 month (30+ trading days)\n"
                "- Recommended: 3-6 months\n"
                "- Best results: 1 year or more"
            )
        return
    
    if prediction:
        # Main prediction display
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            signal = prediction['signal']
            if signal == "BUY":
                st.success(f"## 📈 {signal}")
                st.caption("Bullish Signal Detected")
            elif signal == "SELL":
                st.error(f"## 📉 {signal}")
                st.caption("Bearish Signal Detected")
            else:
                st.info(f"## ➡️ {signal}")
                st.caption("Neutral Signal")
        
        with col2:
            confidence = prediction['confidence']
            st.markdown("## Confidence")
            st.progress(confidence / 100)
            st.markdown(f"### {confidence:.0f}%")
            
            # Signal strength
            buy_signals = prediction.get('buy_signals', 0)
            sell_signals = prediction.get('sell_signals', 0)
            st.caption(f"Buy Signals: {buy_signals} | Sell Signals: {sell_signals}")
        
        with col3:
            st.markdown("## Analysis Factors")
            
            # Display factors in a clean format
            for key, value in prediction['factors'].items():
                if 'Signal' not in key:  # Skip signal descriptions for cleaner display
                    st.write(f"**{key}:** {value}")
        
        # Detailed Explanation
        st.markdown("---")
        st.markdown("### 📋 Detailed Analysis")
        
        if signal == "BUY":
            st.success(
                f"**Recommendation:** The AI analysis suggests a **BUY** signal for {symbol} with {confidence:.0f}% confidence.\n\n"
                f"**Rationale:** Based on {buy_signals} bullish indicators vs {sell_signals} bearish indicators, "
                f"the technical analysis suggests potential upward momentum. Key supporting factors include the current technical setup "
                f"and market conditions favorable for entry."
            )
        elif signal == "SELL":
            st.error(
                f"**Recommendation:** The AI analysis suggests a **SELL** signal for {symbol} with {confidence:.0f}% confidence.\n\n"
                f"**Rationale:** Based on {sell_signals} bearish indicators vs {buy_signals} bullish indicators, "
                f"the technical analysis suggests potential downward pressure. Consider taking profits or avoiding new positions "
                f"until conditions improve."
            )
        else:
            st.info(
                f"**Recommendation:** The AI analysis suggests a **HOLD** signal for {symbol} with {confidence:.0f}% confidence.\n\n"
                f"**Rationale:** With {buy_signals} bullish and {sell_signals} bearish indicators, "
                f"the market shows mixed signals without clear directional bias. Wait for stronger confirmation before taking action."
            )
        
        # Risk Warning
        st.warning(
            "⚠️ **Risk Disclaimer:** This is an AI-generated analysis based on technical indicators only. "
            "Always conduct your own research and consider fundamental analysis, market conditions, and your risk tolerance "
            "before making investment decisions."
        )

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("# 🔍 Stock Analysis")
    st.caption("Comprehensive analysis with technical indicators and AI predictions")
    
    # Stock Selection Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sector = st.selectbox(
            "Select Sector",
            ["All", "Technology", "Finance", "Healthcare", "Energy", "Consumer"],
            help="Filter stocks by sector"
        )
    
    with col2:
        # Stock mapping
        stock_map = {
            "All": {
                "AAPL - Apple Inc.": "AAPL",
                "MSFT - Microsoft": "MSFT",
                "GOOGL - Alphabet": "GOOGL",
                "AMZN - Amazon": "AMZN",
                "NVDA - NVIDIA": "NVDA",
                "META - Meta": "META",
                "TSLA - Tesla": "TSLA",
                "JPM - JPMorgan": "JPM",
                "BAC - Bank of America": "BAC",
                "JNJ - Johnson & Johnson": "JNJ"
            },
            "Technology": {
                "AAPL - Apple Inc.": "AAPL",
                "MSFT - Microsoft": "MSFT",
                "GOOGL - Alphabet": "GOOGL",
                "NVDA - NVIDIA": "NVDA",
                "META - Meta": "META"
            },
            "Finance": {
                "JPM - JPMorgan": "JPM",
                "BAC - Bank of America": "BAC",
                "GS - Goldman Sachs": "GS",
                "MS - Morgan Stanley": "MS",
                "V - Visa": "V"
            },
            "Healthcare": {
                "JNJ - Johnson & Johnson": "JNJ",
                "PFE - Pfizer": "PFE",
                "UNH - UnitedHealth": "UNH",
                "CVS - CVS Health": "CVS",
                "ABBV - AbbVie": "ABBV"
            }
        }
        
        stocks = stock_map.get(sector, stock_map["All"])
        selected = st.selectbox(
            "Select Stock",
            list(stocks.keys()),
            help="Choose a stock to analyze"
        )
        symbol = stocks[selected]
    
    with col3:
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2,  # Default to 6mo
            help="Select analysis timeframe"
        )
    
    # Fetch data
    with st.spinner(f"Loading data for {symbol}..."):
        df, info = fetch_stock_data(symbol, period)
    
    if df.empty:
        st.error("Unable to fetch data. Please try again or select a different stock.")
        return
    
    # Calculate all indicators
    with st.spinner("Calculating technical indicators..."):
        df = TechnicalIndicators.calculate_all_indicators(df)
    
    # Display current price info
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    
    price_change = latest['Close'] - previous['Close']
    price_change_pct = (price_change / previous['Close']) * 100
    
    with col1:
        st.metric(
            "Current Price",
            f"${latest['Close']:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric(
            "Day Range",
            f"${latest['Low']:.2f} - ${latest['High']:.2f}"
        )
    
    with col3:
        week_ago = df['Close'].iloc[-5] if len(df) >= 5 else latest['Close']
        week_change = ((latest['Close'] - week_ago) / week_ago) * 100
        st.metric(
            "Week Change",
            f"{week_change:+.2f}%"
        )
    
    with col4:
        month_ago = df['Close'].iloc[-20] if len(df) >= 20 else df['Close'].iloc[0]
        month_change = ((latest['Close'] - month_ago) / month_ago) * 100
        st.metric(
            "Month Change",
            f"{month_change:+.2f}%"
        )
    
    with col5:
        volume_m = latest['Volume'] / 1_000_000
        avg_volume_m = df['Volume'].mean() / 1_000_000
        st.metric(
            "Volume",
            f"{volume_m:.2f}M",
            f"Avg: {avg_volume_m:.2f}M"
        )
    
    # Create tabs for different analysis views
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Chart",
        "📊 Technical Indicators",
        "💰 Fundamentals",
        "🤖 AI Prediction",
        "📰 Sentiment"
    ])
    
    with tab1:
        st.markdown("### 📈 Interactive Price Chart")
        fig = render_price_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        render_technical_indicators(df)
    
    with tab3:
        st.markdown("### 💰 Fundamental Analysis")
        
        if info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'marketCap' in info:
                    market_cap_b = info['marketCap'] / 1_000_000_000
                    st.metric("Market Cap", f"${market_cap_b:.2f}B")
                
                if 'trailingPE' in info:
                    st.metric("P/E Ratio", f"{info['trailingPE']:.2f}")
            
            with col2:
                if 'dividendYield' in info:
                    div_yield = info['dividendYield'] * 100 if info['dividendYield'] else 0
                    st.metric("Dividend Yield", f"{div_yield:.2f}%")
                
                if 'beta' in info:
                    st.metric("Beta", f"{info['beta']:.2f}")
            
            with col3:
                if '52WeekHigh' in info:
                    st.metric("52W High", f"${info['52WeekHigh']:.2f}")
                
                if '52WeekLow' in info:
                    st.metric("52W Low", f"${info['52WeekLow']:.2f}")
            
            with col4:
                if 'forwardEps' in info:
                    st.metric("Forward EPS", f"${info['forwardEps']:.2f}")
                
                if 'pegRatio' in info:
                    st.metric("PEG Ratio", f"{info['pegRatio']:.2f}")
        else:
            st.info("Fundamental data not available for this stock")
    
    with tab4:
        render_ai_section(df, symbol)
    
    with tab5:
        st.markdown("### 📰 Market Sentiment")
        st.info("Sentiment analysis feature coming soon! This will include news analysis and social media sentiment.")
    
    # Footer with tips
    st.markdown("---")
    st.markdown("### 💡 Analysis Tips")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(
            "**For Best AI Predictions:**\n"
            "- Use at least 6 months of data\n"
            "- Check multiple timeframes\n"
            "- Combine with fundamental analysis"
        )
    
    with col2:
        st.info(
            "**Technical Indicators:**\n"
            "- RSI < 30: Potentially oversold\n"
            "- RSI > 70: Potentially overbought\n"
            "- MACD crossovers signal trends"
        )
    
    with col3:
        st.info(
            "**Risk Management:**\n"
            "- Never invest more than you can afford to lose\n"
            "- Diversify your portfolio\n"
            "- Set stop-loss orders"
        )

if __name__ == "__main__":
    main()