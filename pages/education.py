"""
Education Page
Educational resources and learning materials
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from utils.llm_explainer import get_explainer
from components.charts import ChartComponents, render_chart
from components.sidebar import render_complete_sidebar

# Page configuration
st.set_page_config(
    page_title="Education - StockBot Advisor",
    page_icon="🎓",
    layout="wide"
)

def render_indicator_explanation(indicator_name: str):
    """Render detailed explanation of technical indicator"""
    explainer = get_explainer()
    explanation = explainer.explain_indicator(indicator_name)
    st.markdown(explanation)

def render_interactive_rsi():
    """Render interactive RSI demonstration"""
    st.markdown("### Interactive RSI Calculator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        period = st.slider("RSI Period", 5, 30, 14)
        overbought = st.slider("Overbought Level", 60, 90, 70)
        oversold = st.slider("Oversold Level", 10, 40, 30)
        
        st.markdown("""
        **Understanding RSI:**
        - RSI measures momentum
        - Values range from 0 to 100
        - Higher values = overbought
        - Lower values = oversold
        """)
    
    with col2:
        # Generate sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = 100 * (1 + np.random.randn(100).cumsum() * 0.01)
        
        # Calculate RSI
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Create chart
        df = pd.DataFrame({'RSI': rsi}, index=dates)
        
        fig = ChartComponents.create_indicator_chart(
            df,
            'RSI',
            title=f"RSI ({period})",
            thresholds={'Overbought': overbought, 'Oversold': oversold}
        )
        render_chart(fig)

def render_interactive_macd():
    """Render interactive MACD demonstration"""
    st.markdown("### Interactive MACD Calculator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fast = st.slider("Fast EMA", 5, 20, 12)
        slow = st.slider("Slow EMA", 20, 35, 26)
        signal = st.slider("Signal EMA", 5, 15, 9)
        
        st.markdown("""
        **Understanding MACD:**
        - MACD = Fast EMA - Slow EMA
        - Signal = EMA of MACD
        - Histogram = MACD - Signal
        - Crossovers indicate trends
        """)
    
    with col2:
        # Generate sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.01))
        
        # Calculate MACD
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Create chart
        df = pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line
        }, index=dates)
        
        fig = ChartComponents.create_line_chart(
            df,
            columns=['MACD', 'Signal'],
            title="MACD Indicator",
            height=300
        )
        render_chart(fig)

def render_concepts_library():
    """Render trading concepts library"""
    st.markdown("## 📚 Trading Concepts Library")
    
    concepts = {
        "Technical Analysis": {
            "description": "Study of historical price patterns to predict future movements",
            "topics": ["Chart Patterns", "Support & Resistance", "Trend Analysis", "Volume Analysis"]
        },
        "Fundamental Analysis": {
            "description": "Evaluation of a company's intrinsic value through financial statements",
            "topics": ["P/E Ratio", "Earnings Reports", "Balance Sheet", "Cash Flow"]
        },
        "Risk Management": {
            "description": "Strategies to protect capital and minimize losses",
            "topics": ["Position Sizing", "Stop Losses", "Diversification", "Risk-Reward Ratio"]
        },
        "Portfolio Theory": {
            "description": "Mathematical framework for assembling portfolios",
            "topics": ["Modern Portfolio Theory", "Efficient Frontier", "Asset Allocation", "Rebalancing"]
        }
    }
    
    for concept, details in concepts.items():
        with st.expander(concept):
            st.markdown(f"**{details['description']}**")
            st.markdown("**Key Topics:**")
            for topic in details['topics']:
                st.markdown(f"- {topic}")

def render_trading_strategies():
    """Render trading strategies guide"""
    st.markdown("## 🎯 Trading Strategies")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trend Following",
        "Mean Reversion",
        "Momentum",
        "Value Investing"
    ])
    
    with tab1:
        st.markdown("""
        ### Trend Following Strategy
        
        **Principle:** "The trend is your friend" - ride existing market trends
        
        **Key Indicators:**
        - Moving Averages (SMA, EMA)
        - MACD
        - ADX (Average Directional Index)
        
        **Entry Signals:**
        - Price breaks above resistance
        - Moving average crossovers
        - MACD bullish crossover
        
        **Exit Signals:**
        - Price breaks below support
        - Moving average cross under
        - MACD bearish crossover
        
        **Risk Management:**
        - Use trailing stops
        - Position size based on ATR
        - Maximum 2% risk per trade
        """)
    
    with tab2:
        st.markdown("""
        ### Mean Reversion Strategy
        
        **Principle:** Prices tend to revert to their average over time
        
        **Key Indicators:**
        - RSI (Relative Strength Index)
        - Bollinger Bands
        - Stochastic Oscillator
        
        **Entry Signals:**
        - RSI < 30 (oversold)
        - Price touches lower Bollinger Band
        - Stochastic < 20
        
        **Exit Signals:**
        - RSI > 70 (overbought)
        - Price touches upper Bollinger Band
        - Stochastic > 80
        
        **Risk Management:**
        - Tight stop losses
        - Quick profit taking
        - Avoid trending markets
        """)
    
    with tab3:
        st.markdown("""
        ### Momentum Strategy
        
        **Principle:** Strong stocks tend to continue performing well
        
        **Key Indicators:**
        - Rate of Change (ROC)
        - Relative Strength
        - Volume indicators
        
        **Entry Signals:**
        - Breakout on high volume
        - Strong relative strength
        - Positive earnings surprise
        
        **Exit Signals:**
        - Momentum divergence
        - Volume decline
        - Break of support level
        
        **Risk Management:**
        - Scale into positions
        - Regular profit taking
        - Sector diversification
        """)
    
    with tab4:
        st.markdown("""
        ### Value Investing Strategy
        
        **Principle:** Buy undervalued stocks and hold for long term
        
        **Key Metrics:**
        - P/E Ratio < Industry Average
        - P/B Ratio < 1.5
        - Strong Free Cash Flow
        - Low Debt/Equity
        
        **Entry Signals:**
        - Stock trading below intrinsic value
        - Strong fundamentals
        - Temporary market pessimism
        
        **Exit Signals:**
        - Reaches fair value
        - Fundamental deterioration
        - Better opportunities available
        
        **Risk Management:**
        - Thorough due diligence
        - Long-term perspective
        - Margin of safety
        """)

def render_quiz():
    """Render interactive quiz"""
    st.markdown("## 🎯 Test Your Knowledge")
    
    questions = [
        {
            "question": "What does RSI measure?",
            "options": ["Volume", "Momentum", "Volatility", "Trend"],
            "correct": 1,
            "explanation": "RSI (Relative Strength Index) measures momentum - the speed and magnitude of price changes."
        },
        {
            "question": "When is a stock considered overbought using RSI?",
            "options": ["RSI < 30", "RSI > 70", "RSI = 50", "RSI < 20"],
            "correct": 1,
            "explanation": "A stock is typically considered overbought when RSI is above 70."
        },
        {
            "question": "What does MACD stand for?",
            "options": [
                "Moving Average Convergence Divergence",
                "Market Analysis and Chart Dynamics",
                "Maximum Average Current Direction",
                "Momentum And Chart Development"
            ],
            "correct": 0,
            "explanation": "MACD stands for Moving Average Convergence Divergence."
        }
    ]
    
    for i, q in enumerate(questions):
        st.markdown(f"**Question {i+1}: {q['question']}**")
        
        answer = st.radio(
            "Select your answer:",
            q['options'],
            key=f"q_{i}",
            label_visibility="collapsed"
        )
        
        if st.button(f"Check Answer", key=f"check_{i}"):
            if q['options'].index(answer) == q['correct']:
                st.success(f"✅ Correct! {q['explanation']}")
            else:
                st.error(f"❌ Incorrect. {q['explanation']}")
        
        st.divider()

def render_resources():
    """Render additional resources"""
    st.markdown("## 📖 Additional Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 Recommended Books
        - "A Random Walk Down Wall Street" - Burton Malkiel
        - "The Intelligent Investor" - Benjamin Graham
        - "Technical Analysis of the Financial Markets" - John Murphy
        - "Market Wizards" - Jack Schwager
        
        ### 🎥 Video Tutorials
        - [Introduction to Technical Analysis](#)
        - [Understanding Financial Statements](#)
        - [Risk Management Basics](#)
        - [Building Your First Portfolio](#)
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 Useful Links
        - [SEC Investor Education](https://www.investor.gov/)
        - [Investopedia](https://www.investopedia.com/)
        - [Yahoo Finance](https://finance.yahoo.com/)
        - [TradingView](https://www.tradingview.com/)
        
        ### 📊 Practice Tools
        - Paper Trading Simulator
        - Technical Analysis Playground
        - Portfolio Optimizer
        - Risk Calculator
        """)

def render_glossary():
    """Render financial glossary"""
    st.markdown("## 📖 Financial Glossary")
    
    terms = {
        "Alpha": "Excess return relative to a benchmark",
        "Beta": "Measure of volatility relative to the market",
        "Bull Market": "Period of rising stock prices",
        "Bear Market": "Period of declining stock prices (20%+ drop)",
        "Dividend": "Payment made by companies to shareholders",
        "EPS": "Earnings Per Share - company profit divided by shares",
        "IPO": "Initial Public Offering - first sale of stock to public",
        "Liquidity": "How easily an asset can be converted to cash",
        "Market Cap": "Total value of a company's shares",
        "P/E Ratio": "Price to Earnings - valuation metric",
        "Short Selling": "Betting against a stock by borrowing and selling",
        "Volatility": "Degree of price variation",
        "Yield": "Income return on an investment"
    }
    
    # Create searchable glossary
    search_term = st.text_input("Search glossary:", placeholder="Enter a term...")
    
    filtered_terms = {k: v for k, v in terms.items() 
                     if search_term.lower() in k.lower()} if search_term else terms
    
    for term, definition in sorted(filtered_terms.items()):
        st.markdown(f"**{term}:** {definition}")

# Main function
def main():
    # Render sidebar
    if 'user_profile' in st.session_state:
        render_complete_sidebar(
            st.session_state.user_profile,
            st.session_state.get('portfolio', {}),
            st.session_state.get('watchlist', [])[:5]
        )
    
    # Main header
    st.markdown("# 🎓 Education Center")
    st.caption("Learn about investing, technical analysis, and trading strategies")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Technical Indicators",
        "📚 Concepts",
        "🎯 Strategies",
        "✅ Quiz",
        "📖 Resources"
    ])
    
    with tab1:
        st.markdown("## Technical Indicators")
        
        # Indicator selector
        indicator = st.selectbox(
            "Select an indicator to learn about:",
            ["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Volume"]
        )
        
        # Display explanation
        render_indicator_explanation(indicator.lower().replace(' ', ''))
        
        st.divider()
        
        # Interactive demonstrations
        if indicator == "RSI":
            render_interactive_rsi()
        elif indicator == "MACD":
            render_interactive_macd()
        else:
            st.info("Interactive demonstration coming soon!")
    
    with tab2:
        render_concepts_library()
        st.divider()
        render_glossary()
    
    with tab3:
        render_trading_strategies()
    
    with tab4:
        render_quiz()
        
        # Progress tracking
        st.markdown("### 📈 Your Learning Progress")
        progress = st.progress(0.3)
        st.caption("30% Complete - Keep learning to unlock more content!")
    
    with tab5:
        render_resources()

if __name__ == "__main__":
    main()