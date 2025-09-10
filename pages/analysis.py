# pages/analysis.py
"""
Analysis Page
Detailed stock analysis with ML predictions, technical indicators, and sentiment analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Plotly imports with error handling
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
    page_title="Analysis - StockBot Advisor",
    page_icon="🔍",
    layout="wide"
)

# Load configuration
@st.cache_resource
def load_config():
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}

def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = 'AAPL'
    
    if 'analysis_period' not in st.session_state:
        st.session_state.analysis_period = '1y'
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    if 'show_sentiment' not in st.session_state:
        st.session_state.show_sentiment = True
    
    if 'show_shap' not in st.session_state:
        st.session_state.show_shap = False

@st.cache_data(ttl=300)
def fetch_and_analyze_stock(symbol: str, period: str, include_sentiment: bool = True):
    """Fetch stock data and calculate indicators with optional sentiment"""
    data_processor = get_data_processor()
    
    try:
        # Fetch stock data with or without sentiment
        if include_sentiment:
            df = data_processor.fetch_stock_data_with_sentiment(symbol, period)
        else:
            df = data_processor.fetch_stock_data(symbol, period)
        
        if df.empty:
            return None, None, None
        
        # Calculate technical indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Get stock info
        info = data_processor.fetch_stock_info(symbol)
        
        # Get sentiment analysis if available
        sentiment_data = None
        if include_sentiment:
            sentiment_data = data_processor.fetch_sentiment_analysis(symbol)
        
        return df, info, sentiment_data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None, None

def render_stock_selector():
    """Render stock selection controls"""
    config = load_config()
    
    # Get all stocks from config
    all_stocks = []
    
    # Add stocks from each sector
    for sector in ['technology', 'financial', 'healthcare']:
        if sector in config.get('stocks', {}):
            sector_stocks = [s['symbol'] for s in config['stocks'][sector]]
            all_stocks.extend(sector_stocks)
    
    # Add benchmarks
    if 'benchmarks' in config.get('stocks', {}):
        benchmark_stocks = [b['symbol'] for b in config['stocks']['benchmarks']]
        all_stocks.extend(benchmark_stocks)
    
    # Fallback stocks if config is empty
    if not all_stocks:
        all_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'BAC', 'JNJ', 'PFE']
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        selected_symbol = st.selectbox(
            "Select Stock",
            options=all_stocks,
            index=all_stocks.index(st.session_state.selected_stock) if st.session_state.selected_stock in all_stocks else 0,
            key="stock_selector"
        )
    
    with col2:
        period = st.selectbox(
            "Time Period",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=['1mo', '3mo', '6mo', '1y', '2y', '5y'].index(st.session_state.analysis_period),
            key="period_selector"
        )
    
    with col3:
        include_sentiment = st.checkbox(
            "Include Sentiment",
            value=st.session_state.show_sentiment,
            help="Include sentiment analysis in the analysis"
        )
    
    with col4:
        show_advanced = st.checkbox(
            "Advanced Analysis",
            value=st.session_state.show_shap,
            help="Show SHAP explainability and advanced features"
        )
    
    # Update session state
    st.session_state.selected_stock = selected_symbol
    st.session_state.analysis_period = period
    st.session_state.show_sentiment = include_sentiment
    st.session_state.show_shap = show_advanced
    
    return selected_symbol, period, include_sentiment, show_advanced

def render_stock_overview(info: Dict, latest_data: pd.Series):
    """Render stock overview metrics"""
    if not info or 'error' in info:
        st.warning("Stock information not available")
        return
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = latest_data.get('Close', info.get('current_price', 0))
        prev_close = latest_data.get('Close', 0) - latest_data.get('Close_Open_Spread', 0)
        change = current_price - prev_close if prev_close != 0 else 0
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{change_pct:+.2f}%"
        )
    
    with col2:
        market_cap = info.get('market_cap', 0)
        if market_cap > 1e12:
            market_cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap > 1e9:
            market_cap_str = f"${market_cap/1e9:.2f}B"
        elif market_cap > 1e6:
            market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = f"${market_cap:.0f}"
            
        st.metric(
            label="Market Cap",
            value=market_cap_str
        )
    
    with col3:
        pe_ratio = info.get('pe_ratio', 0)
        st.metric(
            label="P/E Ratio",
            value=f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        )
    
    with col4:
        volume = latest_data.get('Volume', 0)
        avg_volume = info.get('average_volume', 0)
        volume_ratio = (volume / avg_volume) if avg_volume > 0 else 1
        
        if volume > 1e9:
            volume_str = f"{volume/1e9:.2f}B"
        elif volume > 1e6:
            volume_str = f"{volume/1e6:.2f}M"
        elif volume > 1e3:
            volume_str = f"{volume/1e3:.2f}K"
        else:
            volume_str = f"{volume:.0f}"
            
        st.metric(
            label="Volume",
            value=volume_str,
            delta=f"{volume_ratio:.1f}x avg" if avg_volume > 0 else None
        )
    
    with col5:
        beta = info.get('beta', 1)
        st.metric(
            label="Beta",
            value=f"{beta:.2f}" if beta else "N/A"
        )
    
    # Additional info
    st.markdown(f"""
    **{info.get('name', 'Unknown Company')}** | 
    Sector: {info.get('sector', 'N/A')} | 
    Industry: {info.get('industry', 'N/A')}
    """)

def render_price_chart(df: pd.DataFrame, symbol: str):
    """Render interactive price chart with technical indicators"""
    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts")
        return
    
    st.markdown("### 📈 Price Chart & Technical Indicators")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Moving averages
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
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
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
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
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
        
        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        if 'MACD_Histogram' in df.columns:
            colors = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Technical Analysis",
        xaxis_title="Date",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_sentiment_analysis(symbol: str, sentiment_data: Dict):
    """Render sentiment analysis section"""
    st.markdown("### 📰 Sentiment Analysis")
    
    try:
        if not sentiment_data or 'error' in sentiment_data:
            st.warning(f"Sentiment analysis unavailable: {sentiment_data.get('error', 'Unknown error')}")
            return
        
        # Overall sentiment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_score = sentiment_data.get('overall_sentiment_score', 0)
            overall_label = sentiment_data.get('overall_sentiment_label', 'neutral')
            confidence = sentiment_data.get('overall_confidence', 0)
            
            st.metric(
                label="Overall Sentiment",
                value=overall_label.title(),
                delta=f"{overall_score:+.3f} (conf: {confidence:.1%})"
            )
        
        with col2:
            news_sentiment = sentiment_data.get('news_sentiment', {})
            news_score = news_sentiment.get('sentiment_score', 0)
            articles_count = news_sentiment.get('articles_analyzed', 0)
            
            st.metric(
                label="News Sentiment",
                value=f"{news_score:+.3f}",
                delta=f"{articles_count} articles"
            )
        
        with col3:
            market_sentiment = sentiment_data.get('market_sentiment', {})
            market_score = market_sentiment.get('market_sentiment_score', 0)
            
            st.metric(
                label="Market Sentiment",
                value=f"{market_score:+.3f}",
                delta="Technical indicators"
            )
        
        # Sentiment breakdown
        if news_sentiment.get('articles') and PLOTLY_AVAILABLE:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Sentiment Distribution")
                
                # Sentiment distribution pie chart
                distribution = news_sentiment.get('distribution', {})
                if distribution:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Positive', 'Negative', 'Neutral'],
                        values=[
                            distribution.get('positive', 0),
                            distribution.get('negative', 0),
                            distribution.get('neutral', 0)
                        ],
                        hole=0.4,
                        marker_colors=['#2E8B57', '#DC143C', '#708090']
                    )])
                    
                    fig_pie.update_layout(
                        height=300,
                        showlegend=True,
                        margin=dict(t=20, b=20, l=20, r=20)
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("#### Recent News Articles")
                
                # News articles
                articles = news_sentiment.get('articles', [])[:5]  # Show top 5
                
                for i, article in enumerate(articles):
                    sentiment_emoji = "🟢" if article['sentiment'] == 'positive' else "🔴" if article['sentiment'] == 'negative' else "⚪"
                    
                    with st.expander(f"{sentiment_emoji} {article['title'][:60]}..."):
                        st.write(f"**Sentiment:** {article['sentiment'].title()}")
                        st.write(f"**Confidence:** {article['score']:.2f}")
                        if article.get('summary'):
                            st.write(f"**Summary:** {article['summary']}")
        
        # Market sentiment details
        if market_sentiment:
            st.markdown("#### Market Sentiment Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                price_pos = market_sentiment.get('price_position', 0.5)
                st.metric(
                    "Price Position",
                    f"{price_pos:.1%}",
                    help="Position relative to recent high/low range"
                )
            
            with col2:
                volume_trend = market_sentiment.get('volume_trend', 0)
                st.metric(
                    "Volume Trend",
                    f"{volume_trend:+.1%}",
                    help="Recent volume vs average"
                )
            
            with col3:
                momentum_5d = market_sentiment.get('momentum_5d', 0)
                st.metric(
                    "5-Day Momentum",
                    f"{momentum_5d:+.1%}",
                    help="5-day price momentum"
                )
            
            with col4:
                momentum_20d = market_sentiment.get('momentum_20d', 0)
                st.metric(
                    "20-Day Momentum",
                    f"{momentum_20d:+.1%}",
                    help="20-day price momentum"
                )
    
    except Exception as e:
        st.error(f"Error loading sentiment analysis: {e}")

def render_ml_prediction(df: pd.DataFrame, symbol: str, include_sentiment: bool = True):
    """Render ML predictions and signals"""
    st.markdown("### 🤖 AI Prediction Engine")
    
    with st.spinner("Running ML analysis..."):
        try:
            # Create and train model
            model = create_prediction_model('classification')
            
            # Train model
            metrics = model.train(df)
            
            if 'error' in metrics:
                st.error(f"Model training failed: {metrics['error']}")
                return None, None  # Return None tuple explicitly
            
            # Get prediction
            prediction_result = model.predict_latest(df)
            
            if 'error' in prediction_result:
                st.error(f"Prediction failed: {prediction_result['error']}")
                return None, None  # Return None tuple explicitly
            
            # Display main prediction
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                signal = prediction_result['signal']
                confidence = prediction_result['confidence']
                
                # Signal color
                signal_color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                    <h2>{signal_color} {signal}</h2>
                    <p style="font-size: 18px;">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Probability breakdown
                probabilities = prediction_result.get('probabilities', {})
                if probabilities:
                    st.markdown("**Signal Probabilities:**")
                    for signal_type, prob in probabilities.items():
                        st.write(f"{signal_type}: {prob:.1%}")
            
            with col3:
                # Key indicators
                indicators = prediction_result.get('indicators', {})
                st.markdown("**Key Technical Indicators:**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"RSI: {indicators.get('rsi', 0):.1f}")
                    st.write(f"MACD: {indicators.get('macd', 0):.3f}")
                with col_b:
                    st.write(f"SMA 20: ${indicators.get('sma_20', 0):.2f}")
                    st.write(f"Volatility: {indicators.get('volatility', 0):.3f}")
                
                # Sentiment info if available
                if prediction_result.get('has_sentiment') and prediction_result.get('sentiment'):
                    sentiment = prediction_result['sentiment']
                    st.markdown("**Sentiment Indicators:**")
                    st.write(f"Overall: {sentiment.get('overall_score', 0):+.3f}")
                    st.write(f"News: {sentiment.get('news_sentiment', 0):+.3f}")
            
            # Model performance metrics
            st.markdown("#### Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1_score', 0)*100:.1f}%")
            
            # Feature importance
            feature_importance = prediction_result.get('feature_importance', [])
            if feature_importance and PLOTLY_AVAILABLE:
                st.markdown("#### Top Contributing Features")
                
                # Create feature importance chart
                top_features = feature_importance[:10]
                
                fig_features = go.Figure(go.Bar(
                    x=[f['importance'] for f in top_features],
                    y=[f['feature'] for f in top_features],
                    orientation='h',
                    marker_color='steelblue'
                ))
                
                fig_features.update_layout(
                    title="Feature Importance (Top 10)",
                    xaxis_title="Importance",
                    yaxis_title="Features",
                    height=400
                )
                
                st.plotly_chart(fig_features, use_container_width=True)
            
            # Training info
            with st.expander("Training Details"):
                st.write(f"**Training samples:** {metrics.get('train_samples', 0)}")
                st.write(f"**Test samples:** {metrics.get('test_samples', 0)}")
                st.write(f"**Features used:** {metrics.get('n_features', 0)}")
                st.write(f"**Sentiment features:** {'Yes' if metrics.get('has_sentiment_features') else 'No'}")
                
                # Model scores
                model_scores = metrics.get('model_scores', {})
                if model_scores:
                    st.write("**Individual Model Performance:**")
                    for model_name, scores in model_scores.items():
                        st.write(f"- {model_name}: Train={scores['train']:.3f}, Test={scores['test']:.3f}")
            
            return model, prediction_result
            
        except Exception as e:
            st.error(f"ML analysis failed: {e}")
            return None, None  # Return None tuple explicitly

def render_shap_analysis(model, df: pd.DataFrame, symbol: str):
    """Render SHAP explainability analysis"""
    st.markdown("### 🔍 Advanced Explainability (SHAP)")
    
    try:
        # Check if model has explainability
        if not hasattr(model, 'explainer') or model.explainer is None:
            if st.button("Initialize SHAP Explainer"):
                with st.spinner("Initializing SHAP explainer..."):
                    try:
                        # Use sentiment-enhanced features if available
                        has_sentiment = any(col.startswith('sentiment_') for col in df.columns)
                        
                        if has_sentiment:
                            features_df = model.engineer_features_with_sentiment(df)
                        else:
                            features_df = model.engineer_features(df)
                        
                        # Use last 100 rows as background for efficiency
                        background_data = features_df.tail(100)[model.selected_features]
                        model.add_explainability(background_data)
                        st.success("SHAP explainer initialized!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initialize explainer: {e}")
            return
        
        # Get latest data for explanation
        has_sentiment = any(col.startswith('sentiment_') for col in df.columns)
        
        if has_sentiment:
            features_df = model.engineer_features_with_sentiment(df)
        else:
            features_df = model.engineer_features(df)
        
        latest_data = features_df.tail(10)[model.selected_features]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Select Sample for Explanation:**")
            sample_idx = st.selectbox(
                "Sample Index", 
                range(len(latest_data)), 
                index=len(latest_data)-1,
                format_func=lambda x: f"Sample {x+1} (Latest)" if x == len(latest_data)-1 else f"Sample {x+1}"
            )
        
        with col2:
            st.write("**Analysis Type:**")
            analysis_type = st.selectbox(
                "Choose Analysis",
                ["Single Prediction", "Feature Contributions"]
            )
        
        try:
            if analysis_type == "Single Prediction":
                # Single prediction explanation
                single_data = latest_data.iloc[[sample_idx]]
                explanations = model.explain_predictions(single_data)
                
                if 'error' not in explanations:
                    # Show prediction info
                    prediction = explanations['predictions'][0]
                    # Updated signal mapping for [0, 1, 2] classes
                    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", signal_map.get(prediction, prediction))
                    
                    if explanations['probabilities'] is not None:
                        probs = explanations['probabilities'][0]
                        with col2:
                            st.metric("Confidence", f"{max(probs):.1%}")
                        with col3:
                            prob_dict = {signal_map[i]: prob for i, prob in enumerate(probs)}
                            st.write("**Probabilities:**")
                            for signal, prob in prob_dict.items():
                                st.write(f"{signal}: {prob:.1%}")
                    
                    # Feature importance from SHAP
                    shap_importance = explanations.get('feature_importance')
                    if shap_importance is not None and not shap_importance.empty and PLOTLY_AVAILABLE:
                        st.markdown("#### SHAP Feature Importance")
                        
                        # Top 15 features
                        top_shap_features = shap_importance.head(15)
                        
                        fig_shap = go.Figure(go.Bar(
                            y=top_shap_features['feature'],
                            x=top_shap_features['importance'],
                            orientation='h',
                            marker_color='steelblue'
                        ))
                        
                        fig_shap.update_layout(
                            title="SHAP Feature Importance (Top 15)",
                            xaxis_title="Mean |SHAP Value|",
                            yaxis_title="Features",
                            height=600
                        )
                        
                        st.plotly_chart(fig_shap, use_container_width=True)
                
            elif analysis_type == "Feature Contributions":
                # SHAP plots
                plots = model.create_shap_plots(latest_data, sample_idx)
                
                if 'error' not in plots:
                    # Waterfall plot
                    if 'waterfall' in plots and PLOTLY_AVAILABLE:
                        st.markdown("#### SHAP Waterfall Plot")
                        st.plotly_chart(plots['waterfall'], use_container_width=True)
                    
                    # Additional insights
                    st.markdown("#### Key Insights")
                    st.info("""
                    **SHAP Waterfall Plot Explanation:**
                    - Green bars show features that push the prediction toward the positive class
                    - Red bars show features that push the prediction toward the negative class
                    - The base value is the average prediction across all samples
                    - Each feature's contribution moves from the base value toward the final prediction
                    """)
                else:
                    st.error(f"SHAP analysis failed: {plots['error']}")
        
        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
    
    except Exception as e:
        st.error(f"Error in SHAP analysis: {e}")

def render_technical_summary(df: pd.DataFrame):
    """Render technical analysis summary"""
    st.markdown("### 📊 Technical Analysis Summary")
    
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rsi = latest.get('RSI', 50)
        rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
        
        st.metric(
            label="RSI",
            value=f"{rsi:.1f}",
            help=f"Relative Strength Index: {rsi_signal}"
        )
        st.write(f"{rsi_color} {rsi_signal}")
    
    with col2:
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_trend = "Bullish" if macd > macd_signal else "Bearish"
        macd_color = "🟢" if macd > macd_signal else "🔴"
        
        st.metric(
            label="MACD",
            value=f"{macd:.3f}",
            delta=f"Signal: {macd_signal:.3f}"
        )
        st.write(f"{macd_color} {macd_trend}")
    
    with col3:
        sma_20 = latest.get('SMA_20', 0)
        current_price = latest.get('Close', 0)
        sma_trend = "Above" if current_price > sma_20 else "Below"
        sma_color = "🟢" if current_price > sma_20 else "🔴"
        
        st.metric(
            label="SMA 20",
            value=f"${sma_20:.2f}",
            help="20-day Simple Moving Average"
        )
        st.write(f"{sma_color} Price {sma_trend}")
    
    with col4:
        volatility = latest.get('Volatility_20', 0)
        vol_level = "High" if volatility > 0.03 else "Normal" if volatility > 0.01 else "Low"
        vol_color = "🔴" if volatility > 0.03 else "🟡" if volatility > 0.01 else "🟢"
        
        st.metric(
            label="Volatility",
            value=f"{volatility:.3f}",
            help="20-day price volatility"
        )
        st.write(f"{vol_color} {vol_level}")

def main():
    """Main analysis page function"""
    st.title("🔍 Stock Analysis")
    st.markdown("Advanced technical analysis with AI predictions and sentiment insights")
    
    # Check if plotly is available
    if not PLOTLY_AVAILABLE:
        st.error("This page requires Plotly. Please install it with: pip install plotly")
        st.stop()
    
    # Initialize session state
    initialize_session_state()
    
    # Stock selector
    selected_symbol, period, include_sentiment, show_advanced = render_stock_selector()
    
    # Fetch and analyze data
    with st.spinner(f"Analyzing {selected_symbol}..."):
        df, info, sentiment_data = fetch_and_analyze_stock(selected_symbol, period, include_sentiment)
    
    if df is None or df.empty:
        st.error("No data available for the selected stock and period.")
        return
    
    # Stock overview
    latest_data = df.iloc[-1]
    render_stock_overview(info, latest_data)
    
    st.divider()
    
    # Main analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "🤖 AI Prediction", "📰 Sentiment", "🔍 Advanced"])
    
    with tab1:
        render_price_chart(df, selected_symbol)
        st.divider()
        render_technical_summary(df)
    
    with tab2:
        # Handle the None return case properly
        ml_result = render_ml_prediction(df, selected_symbol, include_sentiment)
        
        # Check if we got valid results
        if ml_result is not None and ml_result[0] is not None:
            model, prediction_result = ml_result
            
            if show_advanced and model is not None:
                st.divider()
                render_shap_analysis(model, df, selected_symbol)
        else:
            st.warning("ML prediction is not available. Please check the data and try again.")
    
    with tab3:
        if include_sentiment and sentiment_data:
            render_sentiment_analysis(selected_symbol, sentiment_data)
        else:
            st.info("Enable sentiment analysis in the controls above to see sentiment insights.")
    
    with tab4:
        if show_advanced:
            st.markdown("### 🔬 Advanced Analytics")
            
            # Additional technical indicators
            st.markdown("#### Extended Technical Indicators")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'BB_Position' in df.columns:
                    bb_pos = latest_data.get('BB_Position', 0.5)
                    st.metric("Bollinger Position", f"{bb_pos:.3f}")
                
                if 'Stoch_K' in df.columns:
                    stoch_k = latest_data.get('Stoch_K', 50)
                    st.metric("Stochastic %K", f"{stoch_k:.1f}")
            
            with col2:
                if 'ATR' in df.columns:
                    atr = latest_data.get('ATR', 0)
                    st.metric("ATR", f"{atr:.2f}")
                
                if 'OBV' in df.columns:
                    obv = latest_data.get('OBV', 0)
                    obv_str = f"{obv/1e6:.1f}M" if obv > 1e6 else f"{obv:.0f}"
                    st.metric("OBV", obv_str)
            
            with col3:
                if 'VWAP' in df.columns:
                    vwap = latest_data.get('VWAP', 0)
                    st.metric("VWAP", f"${vwap:.2f}")
                
                # Market regime
                market_regime = latest_data.get('Market_Regime', 0)
                regime_label = "Bullish" if market_regime > 0 else "Bearish" if market_regime < 0 else "Neutral"
                st.metric("Market Regime", regime_label)
            
            # Data quality info
            st.markdown("#### Data Quality")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Data Points:** {len(df)}")
                st.write(f"**Date Range:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            
            with col2:
                missing_data = df.isnull().sum().sum()
                st.write(f"**Missing Values:** {missing_data}")
                st.write(f"**Features Available:** {len(df.columns)}")
            
            with col3:
                sentiment_features = len([col for col in df.columns if col.startswith('sentiment_')])
                st.write(f"**Sentiment Features:** {sentiment_features}")
                st.write(f"**Technical Indicators:** {len([col for col in df.columns if col in ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower']])}")
        
        else:
            st.info("Enable 'Advanced Analysis' in the controls above to see detailed analytics.")

if __name__ == "__main__":
    main()