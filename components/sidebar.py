"""
Sidebar Components Module
Reusable sidebar navigation and profile components
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class SidebarComponents:
    """Reusable sidebar components with consistent styling"""
    
    @staticmethod
    def render_user_profile(user_data: Dict):
        """
        Render user profile section in sidebar
        
        Args:
            user_data: Dictionary with user information
        """
        st.sidebar.markdown("### 👤 User Profile")
        
        # User info card
        st.sidebar.markdown(f"""
        <div style="
            background: #F8F9FA;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        ">
            <div style="
                font-weight: 600;
                color: #000;
                font-size: 1.1rem;
                margin-bottom: 0.5rem;
            ">
                {user_data.get('name', 'Guest User')}
            </div>
            <div style="color: #6C757D; font-size: 0.875rem;">
                <div>Risk: {user_data.get('risk_tolerance', 'Moderate')}</div>
                <div>Horizon: {user_data.get('investment_horizon', '1-3 years')}</div>
                <div>Member since: {user_data.get('member_since', '2025')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Edit profile button
        if st.sidebar.button("⚙️ Edit Profile", use_container_width=True):
            st.session_state.show_profile_modal = True
    
    @staticmethod
    def render_portfolio_summary(portfolio_data: Dict):
        """
        Render portfolio summary in sidebar
        
        Args:
            portfolio_data: Dictionary with portfolio information
        """
        st.sidebar.markdown("### 💼 Portfolio")
        
        total_value = portfolio_data.get('total_value', 100000)
        daily_return = portfolio_data.get('daily_return', 0)
        total_return = portfolio_data.get('total_return', 0)
        
        # Portfolio value
        st.sidebar.metric(
            label="Total Value",
            value=f"${total_value:,.2f}",
            delta=f"{daily_return:+.2f}%"
        )
        
        # Quick stats
        col1, col2 = st.sidebar.columns(2)
        with col1:
            color = "🟢" if daily_return >= 0 else "🔴"
            st.caption(f"{color} Day: {daily_return:+.2f}%")
        with col2:
            color = "🟢" if total_return >= 0 else "🔴"
            st.caption(f"{color} Total: {total_return:+.2f}%")
        
        st.sidebar.markdown("---")
    
    @staticmethod
    def render_navigation(current_page: str = "Dashboard"):
        """
        Render navigation menu
        
        Args:
            current_page: Currently active page
        """
        st.sidebar.markdown("### 🧭 Navigation")
        
        pages = {
            "Dashboard": "📊",
            "Analysis": "🔍",
            "Portfolio": "💼",
            "Backtesting": "📈",
            "Education": "🎓",
            "Settings": "⚙️"
        }
        
        for page, icon in pages.items():
            if st.sidebar.button(
                f"{icon} {page}",
                use_container_width=True,
                type="primary" if page == current_page else "secondary"
            ):
                st.session_state.current_page = page
                st.rerun()
    
    @staticmethod
    def render_quick_actions():
        """Render quick action buttons"""
        st.sidebar.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state.last_update = datetime.now()
                st.rerun()
        
        with col2:
            if st.button("📊 Report", use_container_width=True):
                st.session_state.show_report_modal = True
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🎯 Analyze", use_container_width=True):
                st.session_state.run_analysis = True
        
        with col2:
            if st.button("📧 Alerts", use_container_width=True):
                st.session_state.show_alerts_modal = True
        
        st.sidebar.markdown("---")
    
    @staticmethod
    def render_market_status():
        """Render market status indicator"""
        st.sidebar.markdown("### 🏛️ Market Status")
        
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if weekday and market hours
        is_weekday = now.weekday() < 5
        is_market_hours = market_open <= now <= market_close
        is_open = is_weekday and is_market_hours
        
        if is_open:
            st.sidebar.success("🟢 Market Open")
            
            # Time until close
            time_to_close = market_close - now
            hours = int(time_to_close.total_seconds() // 3600)
            minutes = int((time_to_close.total_seconds() % 3600) // 60)
            st.sidebar.caption(f"Closes in {hours}h {minutes}m")
        else:
            st.sidebar.error("🔴 Market Closed")
            
            # Next open time
            if now > market_close or not is_weekday:
                # Next trading day
                days_ahead = 1 if now.weekday() < 4 else (7 - now.weekday())
                next_open = now + timedelta(days=days_ahead)
                next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
                st.sidebar.caption(f"Opens: {next_open.strftime('%a %I:%M %p')}")
            else:
                st.sidebar.caption(f"Opens: {market_open.strftime('%I:%M %p')}")
        
        # Last update time
        last_update = st.session_state.get('last_update', now)
        st.sidebar.caption(f"Updated: {last_update.strftime('%H:%M:%S')}")
        
        st.sidebar.markdown("---")
    
    @staticmethod
    def render_watchlist_mini(watchlist: List[str], limit: int = 5):
        """
        Render mini watchlist in sidebar
        
        Args:
            watchlist: List of stock symbols
            limit: Maximum number of items to show
        """
        st.sidebar.markdown("### 👁️ Watchlist")
        
        if watchlist:
            for symbol in watchlist[:limit]:
                # In production, fetch real data
                # For demo, use random values
                import random
                price = random.uniform(50, 500)
                change = random.uniform(-5, 5)
                
                color = "#28A745" if change >= 0 else "#DC3545"
                arrow = "↑" if change >= 0 else "↓"
                
                st.sidebar.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    padding: 0.5rem;
                    margin: 0.25rem 0;
                    background: white;
                    border: 1px solid #E9ECEF;
                    border-radius: 4px;
                ">
                    <span style="font-weight: 500;">{symbol}</span>
                    <span style="color: {color};">
                        {arrow} {abs(change):.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            if len(watchlist) > limit:
                st.sidebar.caption(f"+ {len(watchlist) - limit} more")
        else:
            st.sidebar.info("Add stocks to your watchlist")
        
        st.sidebar.markdown("---")
    
    @staticmethod
    def render_notifications(notifications: List[Dict]):
        """
        Render notifications section
        
        Args:
            notifications: List of notification dictionaries
        """
        st.sidebar.markdown("### 🔔 Notifications")
        
        if notifications:
            unread_count = sum(1 for n in notifications if not n.get('read', False))
            
            if unread_count > 0:
                st.sidebar.markdown(f"""
                <div style="
                    background: #DC3545;
                    color: white;
                    border-radius: 12px;
                    padding: 0.25rem 0.5rem;
                    font-size: 0.75rem;
                    display: inline-block;
                    margin-bottom: 0.5rem;
                ">
                    {unread_count} new
                </div>
                """, unsafe_allow_html=True)
            
            for notif in notifications[:3]:
                icon = notif.get('icon', '📢')
                title = notif.get('title', 'Notification')
                time = notif.get('time', 'Just now')
                is_read = notif.get('read', False)
                
                bg_color = "white" if is_read else "#F8F9FA"
                
                st.sidebar.markdown(f"""
                <div style="
                    background: {bg_color};
                    border: 1px solid #E9ECEF;
                    border-radius: 4px;
                    padding: 0.5rem;
                    margin: 0.25rem 0;
                    font-size: 0.875rem;
                ">
                    <div>{icon} {title}</div>
                    <div style="color: #6C757D; font-size: 0.75rem;">{time}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.sidebar.button("View All", use_container_width=True):
                st.session_state.show_notifications_modal = True
        else:
            st.sidebar.info("No new notifications")
        
        st.sidebar.markdown("---")
    
    @staticmethod
    def render_theme_selector():
        """Render theme selector"""
        st.sidebar.markdown("### 🎨 Theme")
        
        theme = st.sidebar.radio(
            "Choose theme",
            ["Minimal", "Dark", "Classic"],
            index=0,
            horizontal=True
        )
        
        if theme != st.session_state.get('theme', 'Minimal'):
            st.session_state.theme = theme
            st.rerun()
    
    @staticmethod
    def render_footer():
        """Render sidebar footer"""
        st.sidebar.markdown("---")
        
        st.sidebar.markdown("""
        <div style="
            text-align: center;
            padding: 1rem 0;
            color: #6C757D;
            font-size: 0.75rem;
        ">
            <div>StockBot Advisor v1.0</div>
            <div>CM3070 Project</div>
            <div>© 2025 Anthony Winata Salim</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_help_section():
        """Render help section"""
        with st.sidebar.expander("❓ Help & Support"):
            st.markdown("""
            **Quick Tips:**
            - Press 'R' to refresh data
            - Press 'A' to run analysis
            - Press '?' for keyboard shortcuts
            
            **Support:**
            - Email: support@stockbot.com
            - Docs: [View Documentation](#)
            - Tutorial: [Watch Video](#)
            
            **Keyboard Shortcuts:**
            - `Ctrl/Cmd + K`: Quick search
            - `Ctrl/Cmd + /`: Toggle sidebar
            - `Esc`: Close modals
            """)

# Utility functions for sidebar operations
def init_sidebar():
    """Initialize sidebar with all components"""
    # Custom CSS for sidebar
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 280px !important;
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        
        section[data-testid="stSidebar"] button {
            transition: all 0.2s ease;
        }
        
        section[data-testid="stSidebar"] button:hover {
            transform: translateX(2px);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Dashboard'
    
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Minimal'
    
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []

def render_complete_sidebar(user_data: Dict, portfolio_data: Dict, watchlist: List[str]):
    """
    Render complete sidebar with all components
    
    Args:
        user_data: User profile data
        portfolio_data: Portfolio summary data
        watchlist: List of watchlist symbols
    """
    init_sidebar()
    
    # Logo/Title
    st.sidebar.markdown("""
    <div style="
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #000;
        margin-bottom: 1rem;
    ">
        <h1 style="
            font-size: 1.5rem;
            font-weight: 300;
            margin: 0;
            letter-spacing: -0.02em;
        ">
            📊 StockBot Advisor
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # User profile
    SidebarComponents.render_user_profile(user_data)
    
    # Portfolio summary
    SidebarComponents.render_portfolio_summary(portfolio_data)
    
    # Quick actions
    SidebarComponents.render_quick_actions()
    
    # Market status
    SidebarComponents.render_market_status()
    
    # Mini watchlist
    SidebarComponents.render_watchlist_mini(watchlist)
    
    # Notifications (if any)
    if st.session_state.get('notifications'):
        SidebarComponents.render_notifications(st.session_state.notifications)
    
    # Help section
    SidebarComponents.render_help_section()
    
    # Footer
    SidebarComponents.render_footer()