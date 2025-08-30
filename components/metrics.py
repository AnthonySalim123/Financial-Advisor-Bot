"""
Metrics Components Module
Custom metric cards and KPI displays with minimalistic design
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

class MetricComponents:
    """Reusable metric components with consistent styling"""
    
    @staticmethod
    def render_metric_card(title: str, 
                          value: Union[str, float, int],
                          delta: Optional[Union[str, float]] = None,
                          delta_color: str = "normal",
                          subtitle: Optional[str] = None,
                          icon: Optional[str] = None):
        """
        Render a custom metric card with minimalistic design
        
        Args:
            title: Metric title
            value: Main value to display
            delta: Change value
            delta_color: Color scheme for delta ('normal', 'inverse', 'off')
            subtitle: Additional subtitle text
            icon: Optional emoji icon
        """
        # Format value
        if isinstance(value, (int, float)):
            if value >= 1000000:
                value_str = f"${value/1000000:.2f}M"
            elif value >= 1000:
                value_str = f"${value/1000:.1f}K"
            else:
                value_str = f"${value:.2f}"
        else:
            value_str = str(value)
        
        # Create HTML for custom metric card
        card_html = f"""
        <div style="
            background: white;
            border: 1px solid #E9ECEF;
            border-radius: 4px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        ">
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.5rem;
            ">
                <h4 style="
                    color: #6C757D;
                    font-size: 0.875rem;
                    font-weight: 400;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">
                    {icon + ' ' if icon else ''}{title}
                </h4>
            </div>
            <div style="
                color: #000000;
                font-size: 1.75rem;
                font-weight: 500;
                font-family: 'SF Mono', Monaco, monospace;
                margin: 0.5rem 0;
            ">
                {value_str}
            </div>
        """
        
        if delta is not None:
            # Format delta
            if isinstance(delta, (int, float)):
                delta_str = f"{delta:+.2f}%"
                color = "#28A745" if delta >= 0 else "#DC3545"
                arrow = "↑" if delta >= 0 else "↓"
            else:
                delta_str = str(delta)
                color = "#6C757D"
                arrow = ""
            
            if delta_color == "inverse":
                color = "#DC3545" if delta >= 0 else "#28A745"
            elif delta_color == "off":
                color = "#6C757D"
            
            card_html += f"""
            <div style="
                color: {color};
                font-size: 0.875rem;
                font-weight: 500;
            ">
                {arrow} {delta_str}
            </div>
            """
        
        if subtitle:
            card_html += f"""
            <div style="
                color: #6C757D;
                font-size: 0.75rem;
                margin-top: 0.25rem;
            ">
                {subtitle}
            </div>
            """
        
        card_html += "</div>"
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric_row(metrics: List[Dict]):
        """
        Render a row of metrics
        
        Args:
            metrics: List of metric dictionaries with keys: title, value, delta, subtitle
        """
        cols = st.columns(len(metrics))
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                MetricComponents.render_metric_card(
                    title=metric.get('title', ''),
                    value=metric.get('value', 0),
                    delta=metric.get('delta'),
                    delta_color=metric.get('delta_color', 'normal'),
                    subtitle=metric.get('subtitle'),
                    icon=metric.get('icon')
                )
    
    @staticmethod
    def render_portfolio_summary(portfolio_data: Dict):
        """
        Render portfolio summary metrics
        
        Args:
            portfolio_data: Dictionary with portfolio information
        """
        st.markdown("### Portfolio Overview")
        
        # Calculate metrics
        total_value = portfolio_data.get('total_value', 100000)
        daily_return = portfolio_data.get('daily_return', 0)
        total_return = portfolio_data.get('total_return', 0)
        cash = portfolio_data.get('cash', 0)
        
        # First row of metrics
        metrics_row1 = [
            {
                'title': 'Total Value',
                'value': total_value,
                'delta': daily_return,
                'icon': '💼'
            },
            {
                'title': 'Daily P&L',
                'value': total_value * (daily_return / 100),
                'delta': daily_return,
                'subtitle': 'Since yesterday',
                'icon': '📊'
            },
            {
                'title': 'Total Return',
                'value': f"{total_return:.2f}%",
                'delta': total_return,
                'subtitle': 'Since inception',
                'icon': '📈'
            },
            {
                'title': 'Cash Available',
                'value': cash,
                'subtitle': f"{(cash/total_value)*100:.1f}% of portfolio",
                'icon': '💵'
            }
        ]
        
        MetricComponents.render_metric_row(metrics_row1)
    
    @staticmethod
    def render_risk_metrics(risk_data: Dict):
        """
        Render risk metrics display
        
        Args:
            risk_data: Dictionary with risk metrics
        """
        st.markdown("### Risk Analysis")
        
        metrics = [
            {
                'title': 'Sharpe Ratio',
                'value': f"{risk_data.get('sharpe_ratio', 0):.2f}",
                'subtitle': 'Risk-adjusted return',
                'icon': '⚖️'
            },
            {
                'title': 'Max Drawdown',
                'value': f"{risk_data.get('max_drawdown', 0):.1f}%",
                'delta_color': 'inverse',
                'icon': '📉'
            },
            {
                'title': 'Volatility',
                'value': f"{risk_data.get('volatility', 0):.1f}%",
                'subtitle': 'Annual',
                'icon': '〰️'
            },
            {
                'title': 'Beta',
                'value': f"{risk_data.get('beta', 1):.2f}",
                'subtitle': 'vs S&P 500',
                'icon': '🎯'
            }
        ]
        
        MetricComponents.render_metric_row(metrics)
    
    @staticmethod
    def render_performance_metrics(performance_data: Dict):
        """
        Render performance metrics
        
        Args:
            performance_data: Dictionary with performance data
        """
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Returns")
            periods = ['1D', '1W', '1M', '3M', '6M', '1Y']
            for period in periods:
                value = performance_data.get(f'return_{period}', 0)
                color = "🟢" if value >= 0 else "🔴"
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    padding: 0.25rem 0;
                    border-bottom: 1px solid #F8F9FA;
                ">
                    <span style="color: #6C757D; font-size: 0.875rem;">{period}</span>
                    <span style="color: #000; font-weight: 500; font-family: monospace;">
                        {color} {value:+.2f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Statistics")
            stats = {
                'Win Rate': f"{performance_data.get('win_rate', 0):.1f}%",
                'Avg Win': f"{performance_data.get('avg_win', 0):.2f}%",
                'Avg Loss': f"{performance_data.get('avg_loss', 0):.2f}%",
                'Best Day': f"{performance_data.get('best_day', 0):.2f}%",
                'Worst Day': f"{performance_data.get('worst_day', 0):.2f}%",
                'Trades': performance_data.get('num_trades', 0)
            }
            
            for stat, value in stats.items():
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    padding: 0.25rem 0;
                    border-bottom: 1px solid #F8F9FA;
                ">
                    <span style="color: #6C757D; font-size: 0.875rem;">{stat}</span>
                    <span style="color: #000; font-weight: 500; font-family: monospace;">
                        {value}
                    </span>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def render_signal_strength(signal: str, confidence: float, indicators: Dict):
        """
        Render signal strength indicator
        
        Args:
            signal: BUY, SELL, or HOLD
            confidence: Confidence level (0-1)
            indicators: Dictionary of technical indicators
        """
        # Determine colors and icons
        if signal == 'BUY':
            color = '#28A745'
            icon = '📈'
            action = 'BUY'
        elif signal == 'SELL':
            color = '#DC3545'
            icon = '📉'
            action = 'SELL'
        else:
            color = '#6C757D'
            icon = '⏸️'
            action = 'HOLD'
        
        # Create signal card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}10 0%, {color}05 100%);
            border: 2px solid {color};
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">
                {icon}
            </div>
            <h2 style="
                color: {color};
                margin: 0.5rem 0;
                font-weight: 600;
            ">
                {action} SIGNAL
            </h2>
            <div style="
                color: #000;
                font-size: 1.5rem;
                font-weight: 500;
                margin: 0.5rem 0;
            ">
                {confidence * 100:.1f}% Confidence
            </div>
            <div style="margin-top: 1rem;">
                <progress value="{confidence}" max="1" style="
                    width: 100%;
                    height: 20px;
                    -webkit-appearance: none;
                    appearance: none;
                ">
                </progress>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show supporting indicators
        st.markdown("#### Supporting Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi = indicators.get('RSI', 50)
            rsi_status = 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div style="color: #6C757D; font-size: 0.75rem;">RSI</div>
                <div style="color: #000; font-size: 1.25rem; font-weight: 500;">{rsi:.1f}</div>
                <div style="color: #6C757D; font-size: 0.75rem;">{rsi_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            macd = indicators.get('MACD', 0)
            macd_status = 'Bullish' if macd > 0 else 'Bearish'
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div style="color: #6C757D; font-size: 0.75rem;">MACD</div>
                <div style="color: #000; font-size: 1.25rem; font-weight: 500;">{macd:.3f}</div>
                <div style="color: #6C757D; font-size: 0.75rem;">{macd_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            volume_ratio = indicators.get('Volume_Ratio', 1)
            volume_status = 'High' if volume_ratio > 1.5 else 'Low' if volume_ratio < 0.5 else 'Normal'
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div style="color: #6C757D; font-size: 0.75rem;">Volume</div>
                <div style="color: #000; font-size: 1.25rem; font-weight: 500;">{volume_ratio:.1f}x</div>
                <div style="color: #6C757D; font-size: 0.75rem;">{volume_status}</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_market_status(is_open: bool, next_open: Optional[datetime] = None):
        """
        Render market status indicator
        
        Args:
            is_open: Whether market is currently open
            next_open: Next market open time
        """
        if is_open:
            status_html = """
            <div style="
                background: #28A74510;
                border: 1px solid #28A745;
                border-radius: 4px;
                padding: 0.75rem;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <span style="color: #28A745; font-weight: 500;">
                    🟢 Market Open
                </span>
            </div>
            """
        else:
            next_open_str = next_open.strftime("%B %d, %I:%M %p") if next_open else "Monday 9:30 AM"
            status_html = f"""
            <div style="
                background: #DC354510;
                border: 1px solid #DC3545;
                border-radius: 4px;
                padding: 0.75rem;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <div style="color: #DC3545; font-weight: 500;">
                    🔴 Market Closed
                </div>
                <div style="color: #6C757D; font-size: 0.75rem; margin-top: 0.25rem;">
                    Opens: {next_open_str}
                </div>
            </div>
            """
        
        st.markdown(status_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_watchlist_item(symbol: str, price: float, change: float, volume: int):
        """
        Render a watchlist item with metrics
        
        Args:
            symbol: Stock symbol
            price: Current price
            change: Price change percentage
            volume: Trading volume
        """
        color = "#28A745" if change >= 0 else "#DC3545"
        arrow = "↑" if change >= 0 else "↓"
        
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #E9ECEF;
            border-radius: 4px;
            padding: 0.75rem;
            margin: 0.5rem 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s ease;
            cursor: pointer;
        "
        onmouseover="this.style.backgroundColor='#F8F9FA'"
        onmouseout="this.style.backgroundColor='white'"
        >
            <div>
                <div style="font-weight: 600; color: #000;">{symbol}</div>
                <div style="font-size: 0.75rem; color: #6C757D;">Vol: {volume:,.0f}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 500; color: #000; font-family: monospace;">
                    ${price:.2f}
                </div>
                <div style="color: {color}; font-size: 0.875rem;">
                    {arrow} {abs(change):.2f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Utility functions
def display_loading_metrics():
    """Display loading placeholders for metrics"""
    with st.container():
        cols = st.columns(4)
        for col in cols:
            with col:
                st.markdown("""
                <div style="
                    background: #F8F9FA;
                    border-radius: 4px;
                    height: 100px;
                    animation: pulse 1.5s ease-in-out infinite;
                ">
                </div>
                """, unsafe_allow_html=True)

def format_large_number(value: float) -> str:
    """Format large numbers for display"""
    if abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.2f}"