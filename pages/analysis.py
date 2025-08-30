"""
Analysis Page
Detailed stock analysis with ML predictions and technical indicators
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
from utils.technical_indicators import TechnicalIndicators
from utils.ml_models import create_prediction_model
from utils.llm_explainer import get_explainer
from components.charts import ChartComponents, render_chart
from components.metrics import MetricComponents
from components.sidebar import render_complete_sidebar
import yaml

# Page configuration
st.set_page_config(
    page_title="Analysis - StockBot Advisor",
    page_icon="🔍",
    layout="wide"
)

# Load configuration
@st.cache_resource
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = 'AAPL'
    
    if 'analysis_period' not in st.session_state:
        st.session_state.analysis_period = '3mo'
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

@st.cache_data(ttl=300)
def fetch_and_analyze_stock(symbol: str, period: str):
    """Fetch stock data and calculate indicators"""
    data_processor = get_data_processor()
    
    # Fetch stock data
    df = data_processor.fetch_stock_data(symbol, period)
    
    if df.empty:
        return None, None
    
    # Calculate technical indicators
    df = TechnicalIndicators.calculate_all_indicators(df)
    
    # Get stock info
    info = data_processor.fetch_stock_info(symbol)
    
    return df, info

def render_stock_selector():
    """Render stock selection controls"""
    config = load_config()
    
    # Get all stocks from config
    all_stocks = []
    for sector in ['technology', 'financial', 'healthcare']:
        if sector in config['stocks']:
            all_stocks.extend([s['symbol'] for s in config['stocks'][sector]])
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        selected = st.selectbox(
            "Select Stock for Analysis",
            all_stocks,
            index=all_stocks.index(st.session_state.selected_stock) if st.session_state.selected_stock in all_stocks else 0
        )
        st.session_state.selected_stock = selected
    
    with col2:
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=1
        )
        st.session_state.analysis_period = period
    
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col4:
        if st.button("📥 Export Report", use_container_width=True):
            st.success("Report exported!")
    
    return selected, period

def render_price_chart(df: pd.DataFrame, symbol: str):
    """Render main price chart with indicators"""
    st.markdown("### Price Chart")
    
    # Chart type selection
    chart_type = st.radio(
        "Chart Type",
        ["Candlestick", "Line", "Area"],
        horizontal=True
    )
    
    # Indicator selection
    indicators = st.multiselect(
        "Add Indicators",
        ["SMA_20", "SMA_50", "EMA_12", "EMA_26", "BB_Upper", "BB_Lower"],
        default=["SMA_20", "SMA_50"]
    )
    
    if chart_type == "Candlestick":
        fig = ChartComponents.create_candlestick_chart(
            df,
            title=f"{symbol} - {st.session_state.analysis_period}",
            indicators=indicators
        )
    else:
        fig = ChartComponents.create_line_chart(
            df,
            columns=['Close'] + indicators,
            title=f"{symbol} - {st.session_state.analysis_period}"
        )
    
    render_chart(fig)

def render_technical_indicators(df: pd.DataFrame):
    """Render technical indicator charts"""
    st.markdown("### Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart
        if 'RSI' in df.columns:
            fig_rsi = ChartComponents.create_indicator_chart(
                df,
                'RSI',
                title="RSI (14)",
                thresholds={'Overbought': 70, 'Oversold': 30}
            )
            render_chart(fig_rsi)
    
    with col2:
        # MACD Chart
        if 'MACD' in df.columns:
            macd_df = df[['MACD', 'MACD_Signal']].dropna()
            fig_macd = ChartComponents.create_line_chart(
                macd_df,
                columns=['MACD', 'MACD_Signal'],
                title="MACD",
                height=250
            )
            render_chart(fig_macd)
    
    # Volume analysis
    if 'Volume' in df.columns:
        st.markdown("### Volume Analysis")
        volume_df = df[['Volume', 'Volume_SMA']].dropna()
        fig_volume = ChartComponents.create_bar_chart(
            volume_df.tail(30).reset_index(),
            x_col='Date',
            y_col='Volume',
            title="30-Day Volume",
            height=250
        )
        render_chart(fig_volume)

def render_ml_prediction(df: pd.DataFrame, symbol: str):
    """Render ML predictions and signals"""
    st.markdown("### 🤖 AI Prediction")
    
    with st.spinner("Running ML analysis..."):
        # Create and train model
        model = create_prediction_model('classification')
        
        # Train model
        metrics = model.train(df)
        
        if 'error' not in metrics:
            # Get prediction
            prediction_result = model.predict_next(df)
            
            if 'error' not in prediction_result:
                # Display signal
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    MetricComponents.render_signal_strength(
                        prediction_result['signal'],
                        prediction_result['confidence'],
                        prediction_result
                    )
                
                with col2:
                    # Get explanation
                    explainer = get_explainer()
                    explanation = explainer.generate_explanation(
                        prediction_result['signal'],
                        prediction_result['confidence'],
                        prediction_result
                    )
                    st.markdown(explanation)
                
                # Model metrics
                st.markdown("#### Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                with col2:
                    st.metric("Precision", f"{metrics['precision']*100:.1f}%")
                with col3:
                    st.metric("Recall", f"{metrics['recall']*100:.1f}%")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
                
                # Feature importance
                if prediction_result.get('feature_importance'):
                    st.markdown("#### Feature Importance")
                    importance_df = pd.DataFrame(prediction_result['feature_importance'])
                    fig = ChartComponents.create_bar_chart(
                        importance_df.head(5),
                        x_col='importance',
                        y_col='feature',
                        title="Top 5 Features",
                        height=200,
                        orientation='h'
                    )
                    render_chart(fig)
            else:
                st.error(f"Prediction error: {prediction_result['error']}")
        else:
            st.error(f"Training error: {metrics['error']}")

def render_fundamental_analysis(info: Dict):
    """Render fundamental analysis"""
    st.markdown("### Fundamental Analysis")
    
    # Valuation metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### Valuation")
        metrics = {
            "P/E Ratio": info.get('pe_ratio', 'N/A'),
            "P/B Ratio": info.get('price_to_book', 'N/A'),
            "PEG Ratio": info.get('peg_ratio', 'N/A')
        }
        for key, value in metrics.items():
            if value != 'N/A' and value != 0:
                st.metric(key, f"{value:.2f}")
            else:
                st.metric(key, "N/A")
    
    with col2:
        st.markdown("#### Profitability")
        metrics = {
            "Profit Margin": f"{info.get('profit_margin', 0)*100:.1f}%",
            "ROE": f"{info.get('roe', 0)*100:.1f}%",
            "ROA": f"{info.get('roa', 0)*100:.1f}%"
        }
        for key, value in metrics.items():
            st.metric(key, value)
    
    with col3:
        st.markdown("#### Growth")
        metrics = {
            "Revenue Growth": f"{info.get('revenue_growth', 0)*100:.1f}%",
            "Earnings Growth": f"{info.get('earnings_growth', 0)*100:.1f}%",
            "Target Price": f"${info.get('target_price', 0):.2f}"
        }
        for key, value in metrics.items():
            st.metric(key, value)
    
    with col4:
        st.markdown("#### Financial Health")
        metrics = {
            "Current Ratio": f"{info.get('current_ratio', 0):.2f}",
            "Debt/Equity": f"{info.get('debt_to_equity', 0):.2f}",
            "Beta": f"{info.get('beta', 1):.2f}"
        }
        for key, value in metrics.items():
            st.metric(key, value)

def render_risk_analysis(df: pd.DataFrame):
    """Render risk analysis"""
    st.markdown("### Risk Analysis")
    
    # Calculate risk metrics
    returns = df['Close'].pct_change().dropna()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        volatility = returns.std() * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{volatility:.1f}%")
    
    with col2:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col3:
        max_dd = ((df['Close'] / df['Close'].cummax()) - 1).min() * 100
        st.metric("Max Drawdown", f"{max_dd:.1f}%")
    
    with col4:
        var_95 = returns.quantile(0.05) * 100
        st.metric("VaR (95%)", f"{var_95:.2f}%")
    
    # Risk distribution
    st.markdown("#### Returns Distribution")
    returns_df = pd.DataFrame({'Returns': returns * 100})
    fig = ChartComponents.create_bar_chart(
        returns_df.value_counts().reset_index().head(20),
        x_col='Returns',
        y_col='count',
        title="Daily Returns Distribution (%)",
        height=250
    )
    render_chart(fig)

def render_trading_signals(df: pd.DataFrame):
    """Render trading signals summary"""
    st.markdown("### Trading Signals Summary")
    
    # Generate signals
    df = TechnicalIndicators.generate_signals(df)
    
    latest = df.iloc[-1]
    
    # Signal summary
    col1, col2, col3, col4, col5 = st.columns(5)
    
    signals = {
        'RSI': latest.get('RSI_Signal', 0),
        'MACD': latest.get('MACD_Signal_Line', 0),
        'MA': latest.get('MA_Signal', 0),
        'BB': latest.get('BB_Signal', 0),
        'Stoch': latest.get('Stoch_Signal', 0)
    }
    
    for i, (name, signal) in enumerate(signals.items()):
        with [col1, col2, col3, col4, col5][i]:
            if signal > 0:
                st.success(f"{name}: BUY")
            elif signal < 0:
                st.error(f"{name}: SELL")
            else:
                st.info(f"{name}: NEUTRAL")
    
    # Overall signal
    overall = latest.get('Combined_Signal', 0)
    strength = latest.get('Signal_Strength', 0)
    
    st.markdown("---")
    
    if overall > 0:
        st.success(f"### Overall Signal: BUY (Strength: {strength:.0f}%)")
    elif overall < 0:
        st.error(f"### Overall Signal: SELL (Strength: {strength:.0f}%)")
    else:
        st.info(f"### Overall Signal: HOLD (Strength: {strength:.0f}%)")

# Main function
def main():
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    if 'user_profile' in st.session_state:
        render_complete_sidebar(
            st.session_state.user_profile,
            st.session_state.get('portfolio', {}),
            st.session_state.get('watchlist', [])[:5]
        )
    
    # Main header
    st.markdown("# 🔍 Stock Analysis")
    st.caption("Comprehensive technical and fundamental analysis with AI predictions")
    
    # Stock selector
    symbol, period = render_stock_selector()
    
    # Fetch and analyze data
    with st.spinner(f"Analyzing {symbol}..."):
        df, info = fetch_and_analyze_stock(symbol, period)
    
    if df is not None and not df.empty:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Charts",
            "🤖 AI Prediction",
            "📈 Indicators",
            "💼 Fundamentals",
            "⚠️ Risk Analysis"
        ])
        
        with tab1:
            render_price_chart(df, symbol)
            render_trading_signals(df)
        
        with tab2:
            render_ml_prediction(df, symbol)
        
        with tab3:
            render_technical_indicators(df)
        
        with tab4:
            if info:
                render_fundamental_analysis(info)
            else:
                st.info("Fundamental data not available")
        
        with tab5:
            render_risk_analysis(df)
    else:
        st.error(f"Unable to fetch data for {symbol}. Please try again.")

if __name__ == "__main__":
    main()