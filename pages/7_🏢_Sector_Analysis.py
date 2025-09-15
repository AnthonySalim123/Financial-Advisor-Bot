# pages/7_🏢_Sector_Analysis.py
"""
Sector Analysis Page - Focus on Real Estate and Financial Sectors
Comprehensive sector performance analysis and comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from utils.enhanced_data_processor import get_enhanced_data_processor
from utils.technical_indicators import TechnicalIndicators

# Try importing plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("⚠️ Plotly is required for this page. Please install with: pip install plotly")
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Sector Analysis - StockBot Advisor",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .sector-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E9ECEF;
        transition: all 0.3s ease;
    }
    .sector-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .sector-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #212529;
    }
    .reit-metric {
        background: #FFF3CD;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border-left: 4px solid #FFC107;
    }
    .financial-metric {
        background: #D1ECF1;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border-left: 4px solid #17A2B8;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6C757D;
        margin: 0;
    }
    .positive { color: #28A745; }
    .negative { color: #DC3545; }
    .neutral { color: #6C757D; }
    
    /* Better table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    .dataframe thead th {
        background-color: #F8F9FA !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_sector' not in st.session_state:
        st.session_state.selected_sector = 'real_estate'
    
    if 'comparison_sectors' not in st.session_state:
        st.session_state.comparison_sectors = ['real_estate', 'financial']
    
    if 'analysis_period' not in st.session_state:
        st.session_state.analysis_period = '3mo'

@st.cache_data(ttl=300)
def fetch_sector_data(sector: str, period: str = '3mo'):
    """Fetch comprehensive sector data"""
    processor = get_enhanced_data_processor()
    
    sector_data = {
        'stocks': [],
        'performance': [],
        'metrics': {},
        'correlations': None
    }
    
    stocks = processor.get_stocks_by_sector(sector)
    
    for symbol in stocks:
        try:
            # Fetch stock data
            df = processor.fetch_stock_data(symbol, period)
            
            if not df.empty:
                # Calculate performance
                returns = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                
                stock_info = {
                    'symbol': symbol,
                    'current_price': df['Close'].iloc[-1],
                    'period_return': returns,
                    'volatility': df['Close'].pct_change().std() * np.sqrt(252) * 100,
                    'volume': df['Volume'].mean(),
                    'data': df
                }
                
                # Add sector-specific data
                if sector == 'real_estate':
                    reit_data = processor.fetch_reit_specific_data(symbol)
                    stock_info.update(reit_data)
                elif sector == 'financial':
                    financial_data = processor.fetch_financial_sector_data(symbol)
                    stock_info.update(financial_data)
                
                sector_data['stocks'].append(stock_info)
                
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {e}")
    
    # Calculate sector metrics
    if sector_data['stocks']:
        returns = [s['period_return'] for s in sector_data['stocks']]
        sector_data['metrics'] = {
            'avg_return': np.mean(returns),
            'best_performer': max(sector_data['stocks'], key=lambda x: x['period_return']),
            'worst_performer': min(sector_data['stocks'], key=lambda x: x['period_return']),
            'avg_volatility': np.mean([s['volatility'] for s in sector_data['stocks']]),
            'total_volume': sum([s['volume'] for s in sector_data['stocks']])
        }
    
    return sector_data

def render_sector_overview():
    """Render sector overview section"""
    st.markdown("## 📊 Sector Performance Overview")
    
    processor = get_enhanced_data_processor()
    
    # Create columns for each sector
    sectors = ['technology', 'financial', 'real_estate', 'healthcare']
    cols = st.columns(len(sectors))
    
    sector_colors = {
        'technology': '#007BFF',
        'financial': '#28A745',
        'real_estate': '#FD7E14',
        'healthcare': '#6F42C1'
    }
    
    for idx, sector in enumerate(sectors):
        with cols[idx]:
            # Calculate sector momentum
            momentum = processor.calculate_sector_momentum(sector, period=20)
            
            # Determine status
            if momentum > 5:
                status = "🔥 Hot"
                color_class = "positive"
            elif momentum > 0:
                status = "📈 Rising"
                color_class = "positive"
            elif momentum < -5:
                status = "❄️ Cold"
                color_class = "negative"
            elif momentum < 0:
                status = "📉 Falling"
                color_class = "negative"
            else:
                status = "➡️ Flat"
                color_class = "neutral"
            
            # Render card
            st.markdown(f"""
            <div class="sector-card">
                <div class="sector-title">{sector.upper()}</div>
                <div class="metric-value {color_class}">{momentum:+.2f}%</div>
                <div class="metric-label">20-Day Momentum</div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

def render_real_estate_analysis():
    """Detailed Real Estate/REIT Analysis"""
    st.markdown("## 🏢 Real Estate Sector Analysis (REITs)")
    
    processor = get_enhanced_data_processor()
    
    # Fetch REIT data
    reit_data = fetch_sector_data('real_estate', st.session_state.analysis_period)
    
    if not reit_data['stocks']:
        st.warning("No REIT data available")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance", 
        "💰 Dividend Analysis", 
        "📈 Technical Analysis",
        "🎯 Investment Signals"
    ])
    
    with tab1:
        # Performance comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if PLOTLY_AVAILABLE:
                # REIT Performance Chart
                fig = go.Figure()
                
                for stock in reit_data['stocks']:
                    df = stock['data']
                    normalized = (df['Close'] / df['Close'].iloc[0] - 1) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=normalized,
                        mode='lines',
                        name=stock['symbol'],
                        hovertemplate='%{x}<br>Return: %{y:.2f}%<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="REIT Performance Comparison",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    hovermode='x unified',
                    height=400,
                    showlegend=True,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install plotly for interactive charts: pip install plotly")
        
        with col2:
            # REIT Metrics Summary
            st.markdown("### 📊 Key Metrics")
            
            for stock in reit_data['stocks']:
                with st.expander(f"{stock['symbol']} - ${stock['current_price']:.2f}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Return", f"{stock['period_return']:.2f}%")
                        st.metric("Volatility", f"{stock['volatility']:.2f}%")
                    with col_b:
                        div_yield = stock.get('dividend_yield', 0)
                        # Fix excessive dividend yield display
                        if div_yield > 100:
                            div_yield = div_yield / 100
                        st.metric("Div Yield", f"{div_yield:.2f}%")
                        st.metric("P/B Ratio", f"{stock.get('price_to_book', 0):.2f}")
    
    with tab2:
        # Dividend Analysis
        st.markdown("### 💰 REIT Dividend Analysis")
        
        # Create dividend comparison
        dividend_data = []
        for stock in reit_data['stocks']:
            div_yield = stock.get('dividend_yield', 0)
            # Fix excessive dividend yield
            if div_yield > 100:
                div_yield = div_yield / 100
            
            dividend_data.append({
                'Symbol': stock['symbol'],
                'Dividend Yield': div_yield,
                'Payout Ratio': stock.get('payout_ratio', 0) * 100,
                'Sector': stock.get('reit_sector', 'N/A'),
                'Rate Sensitivity': stock.get('rate_sensitivity', 'N/A')
            })
        
        dividend_df = pd.DataFrame(dividend_data)
        
        if PLOTLY_AVAILABLE:
            # Dividend yield bar chart
            fig = px.bar(
                dividend_df,
                x='Symbol',
                y='Dividend Yield',
                color='Dividend Yield',
                color_continuous_scale='Viridis',
                title='REIT Dividend Yields',
                labels={'Dividend Yield': 'Yield (%)'}
            )
            fig.update_layout(showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        # Display dividend table
        st.dataframe(
            dividend_df.style.format({
                'Dividend Yield': '{:.2f}%',
                'Payout Ratio': '{:.1f}%'
            }).background_gradient(subset=['Dividend Yield'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )
        
        # Dividend insights
        avg_yield = dividend_df['Dividend Yield'].mean()
        st.markdown(f"""
        <div class="reit-metric">
            <strong>📊 REIT Dividend Insights:</strong><br>
            • Average Yield: {avg_yield:.2f}%<br>
            • Best Yield: {dividend_df.loc[dividend_df['Dividend Yield'].idxmax(), 'Symbol']} 
              ({dividend_df['Dividend Yield'].max():.2f}%)<br>
            • REITs must distribute 90% of taxable income<br>
            • Current 10-Year Treasury: ~4.5% (Compare with REIT yields)<br>
            • Tax Note: REIT dividends often taxed as ordinary income
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        # Technical Analysis
        st.markdown("### 📈 Technical Analysis")
        
        selected_reit = st.selectbox(
            "Select REIT for Analysis",
            [s['symbol'] for s in reit_data['stocks']],
            key='reit_technical'
        )
        
        # Get selected REIT data
        selected_data = next(s for s in reit_data['stocks'] if s['symbol'] == selected_reit)
        df = selected_data['data']
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        if PLOTLY_AVAILABLE:
            # Create technical chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(f"{selected_reit} Price", "RSI", "Volume")
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            if 'SMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['SMA_20'], 
                              name='SMA 20', line=dict(color='orange')),
                    row=1, col=1
                )
            if 'SMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['SMA_50'], 
                              name='SMA 50', line=dict(color='blue')),
                    row=1, col=1
                )
            
            # RSI
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Volume
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'),
                row=3, col=1
            )
            
            fig.update_layout(height=700, showlegend=False, template='plotly_white')
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators summary
        col1, col2, col3, col4 = st.columns(4)
        
        latest = df.iloc[-1]
        with col1:
            rsi_value = latest.get('RSI', 50)
            st.metric("RSI", f"{rsi_value:.2f}")
        with col2:
            if 'MACD' in df.columns:
                macd_signal = "Bullish" if latest['MACD'] > latest.get('MACD_Signal', 0) else "Bearish"
                st.metric("MACD", macd_signal)
        with col3:
            if 'BB_Position' in df.columns:
                bb_pos = latest['BB_Position']
                bb_signal = "Oversold" if bb_pos < 0.2 else "Overbought" if bb_pos > 0.8 else "Neutral"
                st.metric("BB Position", bb_signal)
        with col4:
            volume_avg = df['Volume'].mean()
            volume_ratio = latest['Volume'] / volume_avg
            st.metric("Volume vs Avg", f"{volume_ratio:.2f}x")
    
    with tab4:
        # Investment Signals
        st.markdown("### 🎯 REIT Investment Signals")
        
        signals = []
        for stock in reit_data['stocks']:
            # Simple signal generation based on multiple factors
            signal_score = 0
            reasons = []
            
            # Dividend yield signal
            div_yield = stock.get('dividend_yield', 0)
            if div_yield > 100:
                div_yield = div_yield / 100
            
            if div_yield > 5:
                signal_score += 2
                reasons.append(f"High yield ({div_yield:.1f}%)")
            elif div_yield > 4:
                signal_score += 1
                reasons.append(f"Good yield ({div_yield:.1f}%)")
            
            # Performance signal
            if stock['period_return'] > 5:
                signal_score += 1
                reasons.append("Strong momentum")
            elif stock['period_return'] < -5:
                signal_score -= 1
                reasons.append("Weak momentum")
            
            # Volatility signal
            if stock['volatility'] < 20:
                signal_score += 1
                reasons.append("Low volatility")
            elif stock['volatility'] > 30:
                signal_score -= 1
                reasons.append("High volatility")
            
            # Determine signal
            if signal_score >= 3:
                signal = "Strong Buy"
                color = "🟢"
            elif signal_score >= 2:
                signal = "Buy"
                color = "🟢"
            elif signal_score >= 1:
                signal = "Hold"
                color = "🟡"
            else:
                signal = "Caution"
                color = "🔴"
            
            signals.append({
                'Symbol': stock['symbol'],
                'Signal': f"{color} {signal}",
                'Score': signal_score,
                'Reasons': ', '.join(reasons),
                'Price': f"${stock['current_price']:.2f}",
                'Return': f"{stock['period_return']:.2f}%"
            })
        
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values('Score', ascending=False)
        
        st.dataframe(
            signals_df,
            use_container_width=True,
            hide_index=True
        )
        
        st.info("""
        💡 **REIT Investment Tips:**
        - Compare REIT yields to 10-Year Treasury rates
        - Consider interest rate environment (rising rates can hurt REITs)
        - Look for REITs with sustainable payout ratios (<90% of FFO)
        - Diversify across REIT sectors (industrial, data centers, residential)
        - Factor in tax implications of REIT dividends
        """)

def render_financial_sector_analysis():
    """Detailed Financial Sector Analysis"""
    st.markdown("## 💰 Financial Sector Analysis")
    
    processor = get_enhanced_data_processor()
    
    # Fetch financial sector data
    financial_data = fetch_sector_data('financial', st.session_state.analysis_period)
    
    if not financial_data['stocks']:
        st.warning("No financial sector data available")
        return
    
    # Separate banks from other financials
    banks = [s for s in financial_data['stocks'] 
            if s['symbol'] in ['JPM', 'BAC', 'WFC', 'GS', 'MS']]
    payment_processors = [s for s in financial_data['stocks'] 
                         if s['symbol'] in ['V', 'MA', 'AXP']]
    others = [s for s in financial_data['stocks'] 
             if s['symbol'] in ['BLK', 'SPGI']]
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏦 Banks", "💳 Payment Processors", "📊 Other Financials"])
    
    with tab1:
        st.markdown("### 🏦 Banking Stocks Analysis")
        
        if banks:
            # Banking metrics
            bank_metrics = []
            for stock in banks:
                bank_metrics.append({
                    'Symbol': stock['symbol'],
                    'Price': f"${stock['current_price']:.2f}",
                    'Return': f"{stock['period_return']:.2f}%",
                    'P/B Ratio': stock.get('price_to_book', 0),
                    'ROE': stock.get('roe', 0),
                    'Tier 1 Ratio': stock.get('tier1_capital_ratio', 0),
                    'Efficiency': stock.get('efficiency_ratio', 0)
                })
            
            bank_df = pd.DataFrame(bank_metrics)
            
            # Display metrics
            st.dataframe(
                bank_df.style.format({
                    'P/B Ratio': '{:.2f}',
                    'ROE': '{:.1f}%',
                    'Tier 1 Ratio': '{:.1f}%',
                    'Efficiency': '{:.1f}%'
                }).background_gradient(subset=['ROE'], cmap='Greens'),
                use_container_width=True,
                hide_index=True
            )
            
            # Banking health indicator
            avg_roe = np.mean([s.get('roe', 0) for s in banks])
            avg_tier1 = np.mean([s.get('tier1_capital_ratio', 0) for s in banks])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="financial-metric">
                    <strong>Banking Sector Health</strong><br>
                    • Average ROE: {avg_roe:.1f}%<br>
                    • Average Tier 1 Ratio: {avg_tier1:.1f}%<br>
                    • Health Status: {'💚 Strong' if avg_roe > 12 else '💛 Moderate' if avg_roe > 8 else '🔴 Weak'}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="financial-metric">
                    <strong>Interest Rate Impact</strong><br>
                    • Rising rates: Generally positive for banks<br>
                    • Net Interest Margin expansion expected<br>
                    • Credit risk: Monitor loan defaults
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 💳 Payment Processors Analysis")
        
        if payment_processors:
            # Payment processor metrics
            payment_metrics = []
            for stock in payment_processors:
                payment_metrics.append({
                    'Symbol': stock['symbol'],
                    'Price': f"${stock['current_price']:.2f}",
                    'Return': f"{stock['period_return']:.2f}%",
                    'P/E Ratio': stock.get('pe_ratio', 0),
                    'Profit Margin': stock.get('profit_margin', 0),
                    'Operating Margin': stock.get('operating_margin', 0),
                    'Revenue Growth': stock.get('revenue_growth', 0)
                })
            
            payment_df = pd.DataFrame(payment_metrics)
            
            st.dataframe(
                payment_df.style.format({
                    'P/E Ratio': '{:.1f}',
                    'Profit Margin': '{:.1f}%',
                    'Operating Margin': '{:.1f}%',
                    'Revenue Growth': '{:.1f}%'
                }).background_gradient(subset=['Profit Margin'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )
            
            # Payment sector insights
            avg_margin = np.mean([s.get('profit_margin', 0) for s in payment_processors])
            
            st.markdown(f"""
            <div class="financial-metric">
                <strong>💳 Payment Sector Insights</strong><br>
                • Average Profit Margin: {avg_margin:.1f}%<br>
                • Trend: Digital payments growing 15-20% annually<br>
                • Moat: Strong network effects and brand recognition<br>
                • Risk: Regulatory changes, competition from fintech
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📊 Asset Management & Financial Services")
        
        if others:
            # Other financial services
            other_metrics = []
            for stock in others:
                other_metrics.append({
                    'Symbol': stock['symbol'],
                    'Price': f"${stock['current_price']:.2f}",
                    'Return': f"{stock['period_return']:.2f}%",
                    'P/E Ratio': stock.get('pe_ratio', 0),
                    'Market Cap': f"${stock.get('market_cap', 0)/1e9:.1f}B",
                    'Beta': stock.get('beta', 1)
                })
            
            other_df = pd.DataFrame(other_metrics)
            
            st.dataframe(
                other_df.style.format({
                    'P/E Ratio': '{:.1f}',
                    'Beta': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Special mention for BlackRock
            if 'BLK' in [s['symbol'] for s in others]:
                blk_data = next(s for s in others if s['symbol'] == 'BLK')
                aum = blk_data.get('aum_estimate', 0)
                
                st.markdown(f"""
                <div class="financial-metric">
                    <strong>BlackRock (BLK) Highlights</strong><br>
                    • Estimated AUM: ${aum:.1f} Trillion<br>
                    • World's largest asset manager<br>
                    • iShares ETF platform leader<br>
                    • Aladdin risk management system
                </div>
                """, unsafe_allow_html=True)

def render_sector_comparison():
    """Render sector comparison analysis"""
    st.markdown("## 🔄 Sector Comparison")
    
    processor = get_enhanced_data_processor()
    
    # Get sector summary
    summary = processor.get_sector_summary()
    
    if summary:
        # Create comparison dataframe
        comparison_data = []
        for sector, data in summary.items():
            comparison_data.append({
                'Sector': sector.capitalize(),
                'Stocks': data['stock_count'],
                'Momentum (20d)': f"{data['momentum']:.2f}%",
                'Avg P/E': data['avg_pe'],
                'Avg Div Yield': f"{data['avg_dividend_yield']:.2f}%",
                'Market Cap': f"${data['total_market_cap']/1e12:.2f}T"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        st.dataframe(
            comparison_df.style.format({
                'Avg P/E': '{:.1f}'
            }).background_gradient(subset=['Avg P/E'], cmap='coolwarm'),
            use_container_width=True,
            hide_index=True
        )
        
        # Correlation matrix
        st.markdown("### 🔗 Sector Correlations")
        
        correlations = processor.get_sector_correlations()
        
        if not correlations.empty and PLOTLY_AVAILABLE:
            fig = go.Figure(data=go.Heatmap(
                z=correlations.values,
                x=correlations.columns,
                y=correlations.index,
                colorscale='RdBu',
                zmid=0,
                text=correlations.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Sector Correlation Matrix (6-Month)",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            📊 **Correlation Insights:**
            - Values close to 1: Sectors move together
            - Values close to -1: Sectors move opposite
            - Values near 0: Little correlation (good for diversification)
            """)

# Main execution
def main():
    initialize_session_state()
    
    # Header
    st.title("🏢 Advanced Sector Analysis")
    st.markdown("Deep dive into your expanded 33-stock universe across 4 sectors")
    
    # Show summary metrics
    processor = get_enhanced_data_processor()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", "25")
    with col2:
        st.metric("Total ETFs", "4")
    with col3:
        st.metric("Sectors", "4")
    with col4:
        st.metric("New Additions", "10")
    
    # Period selector
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.session_state.analysis_period = st.select_slider(
            "Analysis Period",
            options=['1mo', '3mo', '6mo', '1y', '2y'],
            value=st.session_state.analysis_period
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🏢 Real Estate",
        "💰 Financial",
        "🔄 Comparison"
    ])
    
    with tab1:
        render_sector_overview()
    
    with tab2:
        render_real_estate_analysis()
    
    with tab3:
        render_financial_sector_analysis()
    
    with tab4:
        render_sector_comparison()
    
    # Footer
    st.markdown("---")
    st.caption(f"""
    📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
    💡 Data from yfinance | 
    🎯 33 Stocks + 4 ETFs tracked
    """)

if __name__ == "__main__":
    if not PLOTLY_AVAILABLE:
        st.error("This page requires plotly for visualizations. Please install it:")
        st.code("pip install plotly")
        st.stop()
    
    main()