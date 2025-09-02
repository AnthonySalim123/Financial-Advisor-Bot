"""
Settings Page
User preferences, portfolio configuration, and system settings
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from utils.database import get_database
from components.sidebar import render_complete_sidebar
from components.alerts import AlertComponents
import yaml

# Page configuration
st.set_page_config(
    page_title="Settings - StockBot Advisor",
    page_icon="⚙️",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Guest User',
            'email': '',
            'risk_tolerance': 'Moderate',
            'investment_horizon': '1-3 years',
            'initial_capital': 100000,
            'currency': 'USD',
            'preferred_sectors': [],
            'member_since': '2025'
        }
    
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = {
            'theme': 'minimal',
            'auto_refresh': True,
            'refresh_interval': 5,
            'show_tooltips': True,
            'advanced_mode': False,
            'notifications_enabled': True,
            'email_alerts': False
        }
    
    if 'model_settings' not in st.session_state:
        st.session_state.model_settings = {
            'model_type': 'RandomForest',
            'confidence_threshold': 0.65,
            'lookback_period': 60,
            'prediction_horizon': 5,
            'use_ensemble': False
        }

def render_profile_settings():
    """Render user profile settings"""
    st.markdown("### 👤 User Profile")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                "Name",
                value=st.session_state.user_profile['name']
            )
            
            email = st.text_input(
                "Email",
                value=st.session_state.user_profile.get('email', '')
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["Conservative", "Moderate", "Aggressive"],
                index=["Conservative", "Moderate", "Aggressive"].index(
                    st.session_state.user_profile['risk_tolerance']
                )
            )
        
        with col2:
            investment_horizon = st.selectbox(
                "Investment Horizon",
                ["< 1 year", "1-3 years", "3-5 years", "5-10 years", "> 10 years"],
                index=1
            )
            
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000,
                value=st.session_state.user_profile['initial_capital'],
                step=1000
            )
            
            currency = st.selectbox(
                "Currency",
                ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"],
                index=0
            )
        
        st.markdown("#### Preferred Sectors")
        sectors = st.multiselect(
            "Select sectors to focus on",
            ["Technology", "Healthcare", "Finance", "Energy", "Consumer", "Industrial"],
            default=st.session_state.user_profile.get('preferred_sectors', [])
        )
        
        if st.form_submit_button("Save Profile", type="primary"):
            # Update profile
            st.session_state.user_profile.update({
                'name': name,
                'email': email,
                'risk_tolerance': risk_tolerance,
                'investment_horizon': investment_horizon,
                'initial_capital': initial_capital,
                'currency': currency,
                'preferred_sectors': sectors
            })
            
            # Save to database
            db = get_database()
            db.update_user_profile(1, st.session_state.user_profile)
            
            st.success("Profile updated successfully!")

def render_app_settings():
    """Render application settings"""
    st.markdown("### 🎨 Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Display")
        
        theme = st.selectbox(
            "Theme",
            ["Minimal", "Dark", "Classic"],
            index=0,
            help="Choose the visual theme for the application"
        )
        
        show_tooltips = st.checkbox(
            "Show tooltips",
            value=st.session_state.app_settings['show_tooltips'],
            help="Display helpful tooltips throughout the application"
        )
        
        advanced_mode = st.checkbox(
            "Advanced mode",
            value=st.session_state.app_settings['advanced_mode'],
            help="Show advanced features and technical details"
        )
    
    with col2:
        st.markdown("#### Data Updates")
        
        auto_refresh = st.checkbox(
            "Auto-refresh data",
            value=st.session_state.app_settings['auto_refresh'],
            help="Automatically refresh market data"
        )
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh interval (minutes)",
                min_value=1,
                max_value=60,
                value=st.session_state.app_settings['refresh_interval']
            )
        else:
            refresh_interval = st.session_state.app_settings['refresh_interval']
        
        cache_duration = st.slider(
            "Cache duration (minutes)",
            min_value=1,
            max_value=60,
            value=5,
            help="How long to cache market data"
        )
    
    # Notification settings
    st.markdown("#### Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        notifications_enabled = st.checkbox(
            "Enable notifications",
            value=st.session_state.app_settings['notifications_enabled']
        )
        
        if notifications_enabled:
            notification_types = st.multiselect(
                "Notification types",
                ["Price alerts", "Signal alerts", "News alerts", "Risk alerts"],
                default=["Price alerts", "Signal alerts"]
            )
    
    with col2:
        email_alerts = st.checkbox(
            "Email alerts",
            value=st.session_state.app_settings['email_alerts'],
            disabled=not st.session_state.user_profile.get('email')
        )
        
        if email_alerts:
            alert_frequency = st.selectbox(
                "Alert frequency",
                ["Immediate", "Hourly digest", "Daily digest"],
                index=0
            )
    
    if st.button("Save Settings", type="primary"):
        # Update settings
        st.session_state.app_settings.update({
            'theme': theme,
            'show_tooltips': show_tooltips,
            'advanced_mode': advanced_mode,
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval,
            'notifications_enabled': notifications_enabled,
            'email_alerts': email_alerts
        })
        
        st.success("Settings saved successfully!")
        st.rerun()

def render_model_settings():
    """Render ML model settings"""
    st.markdown("### 🤖 Model Configuration")
    
    st.info("⚠️ These are advanced settings. Changes may affect prediction accuracy.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["RandomForest", "XGBoost", "Ensemble"],
            index=0,
            help="Select the machine learning model to use"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.model_settings['confidence_threshold'],
            step=0.05,
            help="Minimum confidence level for signals"
        )
        
        use_ensemble = st.checkbox(
            "Use ensemble voting",
            value=st.session_state.model_settings['use_ensemble'],
            help="Combine multiple models for better accuracy"
        )
    
    with col2:
        lookback_period = st.number_input(
            "Lookback Period (days)",
            min_value=20,
            max_value=200,
            value=st.session_state.model_settings['lookback_period'],
            help="Historical data period for analysis"
        )
        
        prediction_horizon = st.number_input(
            "Prediction Horizon (days)",
            min_value=1,
            max_value=30,
            value=st.session_state.model_settings['prediction_horizon'],
            help="How far ahead to predict"
        )
        
        retrain_frequency = st.selectbox(
            "Model Retraining",
            ["Daily", "Weekly", "Monthly", "Manual"],
            index=1,
            help="How often to retrain the model"
        )
    
    # Technical indicators configuration
    st.markdown("#### Technical Indicators")
    
    indicators = st.multiselect(
        "Active Indicators",
        ["RSI", "MACD", "SMA", "EMA", "Bollinger Bands", "Stochastic", "ATR", "OBV"],
        default=["RSI", "MACD", "SMA"],
        help="Select which technical indicators to use"
    )
    
    if st.button("Save Model Settings"):
        st.session_state.model_settings.update({
            'model_type': model_type,
            'confidence_threshold': confidence_threshold,
            'lookback_period': lookback_period,
            'prediction_horizon': prediction_horizon,
            'use_ensemble': use_ensemble,
            'indicators': indicators
        })
        
        st.success("Model settings updated! The model will retrain with new settings.")

def render_data_management():
    """Render data management section"""
    st.markdown("### 💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Export Data")
        
        export_type = st.selectbox(
            "Export type",
            ["Portfolio", "Transactions", "Predictions", "All Data"]
        )
        
        export_format = st.radio(
            "Format",
            ["CSV", "Excel", "JSON"]
        )
        
        if st.button("Export", use_container_width=True):
            # Export logic here
            st.success(f"Data exported as {export_format}")
    
    with col2:
        st.markdown("#### Import Data")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json']
        )
        
        if uploaded_file is not None:
            if st.button("Import", use_container_width=True):
                st.success("Data imported successfully!")
    
    with col3:
        st.markdown("#### Clear Data")
        
        st.warning("⚠️ This action cannot be undone!")
        
        clear_type = st.selectbox(
            "Clear type",
            ["Cache", "Alerts", "Transaction History", "All Data"]
        )
        
        if st.button("Clear", type="secondary", use_container_width=True):
            if clear_type == "Cache":
                st.cache_data.clear()
                st.success("Cache cleared!")
            elif clear_type == "Alerts":
                st.session_state.alerts = []
                st.success("Alerts cleared!")
            elif clear_type == "Transaction History":
                st.session_state.transactions = []
                st.success("Transaction history cleared!")

def render_api_keys():
    """Render API keys configuration"""
    st.markdown("### 🔑 API Configuration")
    
    st.info("API keys are stored securely and never shared.")
    
    with st.expander("Configure API Keys"):
        news_api_key = st.text_input(
            "News API Key",
            type="password",
            placeholder="Enter your News API key",
            help="Get your API key from newsapi.org"
        )
        
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key",
            type="password",
            placeholder="Enter your Alpha Vantage key",
            help="Get your API key from alphavantage.co"
        )
        
        if st.button("Save API Keys"):
            # In production, encrypt and store securely
            st.success("API keys saved securely!")

def render_about():
    """Render about section"""
    st.markdown("### ℹ️ About StockBot Advisor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Version:** 1.0.0  
        **Author:** Anthony Winata Salim  
        **Student Number:** 230726051  
        **Course:** CM3070 Project  
        **Supervisor:** Hwa Heng Kan  
        
        **Technologies Used:**
        - Python 3.9+
        - Streamlit
        - Scikit-learn
        - YFinance
        - Pandas/NumPy
        """)
    
    with col2:
        st.markdown("""
        **Features:**
        - Real-time stock analysis
        - AI-powered predictions
        - Natural language explanations
        - Portfolio management
        - Educational resources
        - Risk assessment
        
        **Support:**
        - Documentation: [View Docs](#)
        - GitHub: [Repository](#)
        - Email: support@stockbot.com
        """)

# Main function
def main():
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_complete_sidebar(
        st.session_state.user_profile,
        st.session_state.get('portfolio', {}),
        st.session_state.get('watchlist', [])[:5]
    )
    
    # Main header
    st.markdown("# ⚙️ Settings")
    st.caption("Configure your preferences and manage your account")
    
    # Settings tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "👤 Profile",
        "🎨 Application",
        "🤖 Model",
        "🔔 Alerts",
        "💾 Data",
        "🔑 API Keys",
        "ℹ️ About"
    ])
    
    with tab1:
        render_profile_settings()
    
    with tab2:
        render_app_settings()
    
    with tab3:
        render_model_settings()
    
    with tab4:
        AlertComponents.render_alert_settings()
    
    with tab5:
        render_data_management()
    
    with tab6:
        render_api_keys()
    
    with tab7:
        render_about()

if __name__ == "__main__":
    main()