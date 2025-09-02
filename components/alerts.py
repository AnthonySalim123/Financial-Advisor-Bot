"""
Alert Components Module
Real-time alerts and notifications for stock movements and signals
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class AlertComponents:
    """Manage and display trading alerts and notifications"""
    
    @staticmethod
    def create_alert(alert_type: str, symbol: str, message: str, 
                    severity: str = "info", data: Dict = None) -> Dict:
        """
        Create a new alert
        
        Args:
            alert_type: Type of alert (price, signal, news, risk)
            symbol: Stock symbol
            message: Alert message
            severity: Alert severity (info, warning, success, error)
            data: Additional alert data
        
        Returns:
            Alert dictionary
        """
        alert = {
            'id': datetime.now().timestamp(),
            'timestamp': datetime.now(),
            'type': alert_type,
            'symbol': symbol,
            'message': message,
            'severity': severity,
            'data': data or {},
            'read': False,
            'dismissed': False
        }
        
        return alert
    
    @staticmethod
    def render_alert_card(alert: Dict):
        """
        Render a single alert card
        
        Args:
            alert: Alert dictionary
        """
        # Determine icon and color based on severity
        icons = {
            'info': '📢',
            'warning': '⚠️',
            'success': '✅',
            'error': '🚨'
        }
        
        colors = {
            'info': '#17A2B8',
            'warning': '#FFC107',
            'success': '#28A745',
            'error': '#DC3545'
        }
        
        icon = icons.get(alert['severity'], '📢')
        color = colors.get(alert['severity'], '#17A2B8')
        
        # Time ago calculation
        time_diff = datetime.now() - alert['timestamp']
        if time_diff.seconds < 60:
            time_ago = "Just now"
        elif time_diff.seconds < 3600:
            time_ago = f"{time_diff.seconds // 60} minutes ago"
        elif time_diff.seconds < 86400:
            time_ago = f"{time_diff.seconds // 3600} hours ago"
        else:
            time_ago = f"{time_diff.days} days ago"
        
        # Render alert
        alert_html = f"""
        <div style="
            background: white;
            border-left: 4px solid {color};
            border-radius: 4px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            {'opacity: 0.7;' if alert.get('read') else ''}
        ">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex-grow: 1;">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                        <strong style="color: #000;">{alert['symbol']}</strong>
                        <span style="color: #6C757D; margin-left: 0.5rem; font-size: 0.875rem;">
                            {time_ago}
                        </span>
                    </div>
                    <div style="color: #212529; margin-bottom: 0.5rem;">
                        {alert['message']}
                    </div>
                    {f"<div style='color: #6C757D; font-size: 0.875rem;'>{alert['data'].get('details', '')}</div>" 
                     if alert.get('data', {}).get('details') else ''}
                </div>
                <button style="
                    background: none;
                    border: none;
                    color: #6C757D;
                    cursor: pointer;
                    font-size: 1.2rem;
                    padding: 0;
                ">×</button>
            </div>
        </div>
        """
        
        st.markdown(alert_html, unsafe_allow_html=True)
    
    @staticmethod
    def check_price_alerts(current_prices: Dict, alert_settings: Dict) -> List[Dict]:
        """
        Check for price-based alerts
        
        Args:
            current_prices: Current stock prices
            alert_settings: User alert settings
        
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        for symbol, price in current_prices.items():
            settings = alert_settings.get(symbol, {})
            
            # Check price thresholds
            if settings.get('price_above') and price > settings['price_above']:
                alerts.append(AlertComponents.create_alert(
                    'price',
                    symbol,
                    f"Price crossed above ${settings['price_above']:.2f}",
                    'success',
                    {'current_price': price, 'threshold': settings['price_above']}
                ))
            
            if settings.get('price_below') and price < settings['price_below']:
                alerts.append(AlertComponents.create_alert(
                    'price',
                    symbol,
                    f"Price dropped below ${settings['price_below']:.2f}",
                    'warning',
                    {'current_price': price, 'threshold': settings['price_below']}
                ))
            
            # Check percentage changes
            if settings.get('prev_close'):
                change_pct = ((price - settings['prev_close']) / settings['prev_close']) * 100
                
                if abs(change_pct) > settings.get('change_threshold', 5):
                    severity = 'success' if change_pct > 0 else 'error'
                    alerts.append(AlertComponents.create_alert(
                        'price',
                        symbol,
                        f"Significant move: {change_pct:+.2f}%",
                        severity,
                        {'change_pct': change_pct, 'current_price': price}
                    ))
        
        return alerts
    
    @staticmethod
    def check_signal_alerts(signals: Dict, confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Check for ML signal alerts
        
        Args:
            signals: ML model signals
            confidence_threshold: Minimum confidence for alerts
        
        Returns:
            List of signal alerts
        """
        alerts = []
        
        for symbol, signal_data in signals.items():
            if signal_data['confidence'] >= confidence_threshold:
                if signal_data['signal'] == 'BUY':
                    alerts.append(AlertComponents.create_alert(
                        'signal',
                        symbol,
                        f"Strong BUY signal detected ({signal_data['confidence']*100:.0f}% confidence)",
                        'success',
                        signal_data
                    ))
                elif signal_data['signal'] == 'SELL':
                    alerts.append(AlertComponents.create_alert(
                        'signal',
                        symbol,
                        f"Strong SELL signal detected ({signal_data['confidence']*100:.0f}% confidence)",
                        'error',
                        signal_data
                    ))
        
        return alerts
    
    @staticmethod
    def check_risk_alerts(portfolio: Dict, risk_metrics: Dict) -> List[Dict]:
        """
        Check for portfolio risk alerts
        
        Args:
            portfolio: Portfolio data
            risk_metrics: Risk metrics
        
        Returns:
            List of risk alerts
        """
        alerts = []
        
        # Check max drawdown
        if risk_metrics.get('max_drawdown', 0) < -15:
            alerts.append(AlertComponents.create_alert(
                'risk',
                'PORTFOLIO',
                f"Maximum drawdown exceeded: {risk_metrics['max_drawdown']:.1f}%",
                'error',
                risk_metrics
            ))
        
        # Check volatility
        if risk_metrics.get('volatility', 0) > 30:
            alerts.append(AlertComponents.create_alert(
                'risk',
                'PORTFOLIO',
                f"High volatility detected: {risk_metrics['volatility']:.1f}%",
                'warning',
                risk_metrics
            ))
        
        # Check concentration risk
        for symbol, holding in portfolio.get('holdings', {}).items():
            position_pct = (holding['value'] / portfolio['total_value']) * 100
            if position_pct > 20:
                alerts.append(AlertComponents.create_alert(
                    'risk',
                    symbol,
                    f"Position exceeds 20% of portfolio ({position_pct:.1f}%)",
                    'warning',
                    {'position_pct': position_pct}
                ))
        
        return alerts
    
    @staticmethod
    def render_alert_center():
        """Render the alert center interface"""
        st.markdown("### 🔔 Alert Center")
        
        # Alert filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            alert_type = st.selectbox(
                "Type",
                ["All", "Price", "Signal", "Risk", "News"],
                label_visibility="collapsed"
            )
        
        with col2:
            severity = st.selectbox(
                "Severity",
                ["All", "Info", "Warning", "Success", "Error"],
                label_visibility="collapsed"
            )
        
        with col3:
            time_range = st.selectbox(
                "Time",
                ["Today", "This Week", "This Month", "All Time"],
                label_visibility="collapsed"
            )
        
        with col4:
            if st.button("Clear All", use_container_width=True):
                st.session_state.alerts = []
                st.rerun()
        
        # Get alerts
        alerts = st.session_state.get('alerts', [])
        
        # Filter alerts
        if alert_type != "All":
            alerts = [a for a in alerts if a['type'] == alert_type.lower()]
        
        if severity != "All":
            alerts = [a for a in alerts if a['severity'] == severity.lower()]
        
        # Display alerts
        if alerts:
            for alert in sorted(alerts, key=lambda x: x['timestamp'], reverse=True):
                AlertComponents.render_alert_card(alert)
        else:
            st.info("No alerts to display")
    
    @staticmethod
    def render_alert_settings():
        """Render alert configuration settings"""
        st.markdown("### ⚙️ Alert Settings")
        
        # Get watchlist
        watchlist = st.session_state.get('watchlist', [])
        
        if not watchlist:
            st.info("Add stocks to your watchlist to configure alerts")
            return
        
        # Alert settings for each stock
        for symbol in watchlist:
            with st.expander(f"{symbol} Alert Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.number_input(
                        f"Alert if price above",
                        key=f"{symbol}_above",
                        min_value=0.0,
                        step=1.0,
                        format="%.2f"
                    )
                    
                    st.number_input(
                        f"Alert if price below",
                        key=f"{symbol}_below",
                        min_value=0.0,
                        step=1.0,
                        format="%.2f"
                    )
                
                with col2:
                    st.slider(
                        f"Price change threshold (%)",
                        key=f"{symbol}_change",
                        min_value=1,
                        max_value=20,
                        value=5
                    )
                    
                    st.slider(
                        f"Signal confidence threshold (%)",
                        key=f"{symbol}_confidence",
                        min_value=50,
                        max_value=100,
                        value=70
                    )
                
                st.multiselect(
                    "Alert types",
                    ["Price alerts", "Signal alerts", "Volume alerts", "News alerts"],
                    default=["Price alerts", "Signal alerts"],
                    key=f"{symbol}_types"
                )
    
    @staticmethod
    def create_sample_alerts() -> List[Dict]:
        """Create sample alerts for demonstration"""
        sample_alerts = [
            AlertComponents.create_alert(
                'signal',
                'AAPL',
                'Strong BUY signal detected with 85% confidence',
                'success',
                {'rsi': 28, 'macd': 'bullish', 'confidence': 0.85}
            ),
            AlertComponents.create_alert(
                'price',
                'NVDA',
                'Price increased by 5.2% today',
                'success',
                {'change_pct': 5.2, 'current_price': 475.50}
            ),
            AlertComponents.create_alert(
                'risk',
                'PORTFOLIO',
                'Portfolio volatility above normal range',
                'warning',
                {'volatility': 28.5, 'normal_range': '15-25%'}
            ),
            AlertComponents.create_alert(
                'signal',
                'GOOGL',
                'SELL signal triggered - RSI overbought',
                'error',
                {'rsi': 78, 'price': 142.30}
            ),
            AlertComponents.create_alert(
                'price',
                'MSFT',
                'Price approaching 52-week high',
                'info',
                {'current': 378.50, '52w_high': 384.00}
            )
        ]
        
        # Adjust timestamps
        for i, alert in enumerate(sample_alerts):
            alert['timestamp'] = datetime.now() - timedelta(minutes=i*30)
        
        return sample_alerts

# Utility functions for alert management
def process_alerts():
    """Process and update alerts"""
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    # Add new alerts (in production, this would check real conditions)
    # For demo, we'll use sample alerts
    if len(st.session_state.alerts) == 0:
        st.session_state.alerts = AlertComponents.create_sample_alerts()
    
    # Count unread alerts
    unread_count = sum(1 for a in st.session_state.alerts if not a.get('read', False))
    
    return unread_count

def mark_alerts_read():
    """Mark all alerts as read"""
    for alert in st.session_state.get('alerts', []):
        alert['read'] = True

def get_alert_summary() -> Dict:
    """Get summary of current alerts"""
    alerts = st.session_state.get('alerts', [])
    
    summary = {
        'total': len(alerts),
        'unread': sum(1 for a in alerts if not a.get('read', False)),
        'by_type': {},
        'by_severity': {}
    }
    
    for alert in alerts:
        # Count by type
        alert_type = alert.get('type', 'unknown')
        summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
        
        # Count by severity
        severity = alert.get('severity', 'info')
        summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
    
    return summary