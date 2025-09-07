"""
Portfolio Page
Portfolio management, tracking, and optimization
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
from utils.database import get_database
from components.charts import ChartComponents, render_chart
from components.metrics import MetricComponents
from components.sidebar import render_complete_sidebar
from utils.session_state import init_session_state

# Page configuration
st.set_page_config(
    page_title="Portfolio - StockBot Advisor",
    page_icon="💼",
    layout="wide"
)



def calculate_portfolio_value():
    """Calculate current portfolio value"""
    
    data_processor = get_data_processor()
    total_value = st.session_state.portfolio['cash']
    
    holdings = st.session_state.portfolio['holdings']
    holdings_data = []
    
    for symbol, holding in holdings.items():
        try:
            # Fetch current price
            df = data_processor.fetch_stock_data(symbol, '1d')
            if not df.empty:
                current_price = df['Close'].iloc[-1]
            else:
                current_price = holding['avg_price']
            
            # Calculate values
            shares = holding['shares']
            avg_price = holding['avg_price']
            current_value = shares * current_price
            cost_basis = shares * avg_price
            gain_loss = current_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            holdings_data.append({
                'Symbol': symbol,
                'Shares': shares,
                'Avg Cost': avg_price,
                'Current Price': current_price,
                'Current Value': current_value,
                'Cost Basis': cost_basis,
                'Gain/Loss': gain_loss,
                'Gain/Loss %': gain_loss_pct
            })
            
            total_value += current_value
        except:
            # Use fallback values
            current_value = holding['shares'] * holding['avg_price']
            total_value += current_value
            
            holdings_data.append({
                'Symbol': symbol,
                'Shares': holding['shares'],
                'Avg Cost': holding['avg_price'],
                'Current Price': holding['avg_price'],
                'Current Value': current_value,
                'Cost Basis': current_value,
                'Gain/Loss': 0,
                'Gain/Loss %': 0
            })
    
    st.session_state.portfolio['total_value'] = total_value
    return pd.DataFrame(holdings_data)

def render_portfolio_summary():
    """Render portfolio overview metrics"""
    st.markdown("## Portfolio Overview")
    
    portfolio = st.session_state.portfolio
    
    # Calculate metrics
    total_value = portfolio.get('total_value', 125430)
    invested = total_value - portfolio.get('cash', 25000)
    cash = portfolio.get('cash', 25000)
    total_return = portfolio.get('total_return', 25.4)
    positions = len(portfolio.get('holdings', {}))
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Calculate percentage change
        daily_return = portfolio.get('daily_return', 1.9)
        color = "#28A745" if daily_return >= 0 else "#DC3545"
        arrow = "↑" if daily_return >= 0 else "↓"
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">💼</span>
                <span style="color: #6C757D; font-size: 0.75rem; text-transform: uppercase;">TOTAL VALUE</span>
            </div>
            <div style="font-size: 1.75rem; font-weight: bold; color: #000;">
                ${total_value:,.0f}
            </div>
            <div style="color: {color}; font-size: 0.875rem; margin-top: 0.5rem;">
                {arrow} {abs(daily_return):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        invested_pct = (invested / (invested + cash)) * 100 if (invested + cash) > 0 else 0
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">📊</span>
                <span style="color: #6C757D; font-size: 0.75rem; text-transform: uppercase;">INVESTED</span>
            </div>
            <div style="font-size: 1.75rem; font-weight: bold; color: #000;">
                ${invested:,.0f}
            </div>
            <div style="color: #6C757D; font-size: 0.875rem; margin-top: 0.5rem;">
                {invested_pct:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cash_pct = (cash / total_value) * 100 if total_value > 0 else 0
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">💵</span>
                <span style="color: #6C757D; font-size: 0.75rem; text-transform: uppercase;">CASH</span>
            </div>
            <div style="font-size: 1.75rem; font-weight: bold; color: #000;">
                ${cash:,.0f}
            </div>
            <div style="color: #6C757D; font-size: 0.875rem; margin-top: 0.5rem;">
                {cash_pct:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        return_color = "#28A745" if total_return >= 0 else "#DC3545"
        return_arrow = "↑" if total_return >= 0 else "↓"
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">📈</span>
                <span style="color: #6C757D; font-size: 0.75rem; text-transform: uppercase;">TOTAL RETURN</span>
            </div>
            <div style="font-size: 1.75rem; font-weight: bold; color: #000;">
                {total_return:.1f}%
            </div>
            <div style="color: {return_color}; font-size: 0.875rem; margin-top: 0.5rem;">
                {return_arrow} {abs(total_return):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">💼</span>
                <span style="color: #6C757D; font-size: 0.75rem; text-transform: uppercase;">POSITIONS</span>
            </div>
            <div style="font-size: 1.75rem; font-weight: bold; color: #000;">
                {positions}
            </div>
            <div style="color: #6C757D; font-size: 0.875rem; margin-top: 0.5rem;">
                Active holdings
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_holdings_table():
    """Render holdings table"""
    st.markdown("### Current Holdings")
    
    holdings_df = calculate_portfolio_value()
    
    if not holdings_df.empty:
        # Format for display
        display_df = holdings_df.copy()
        display_df['Avg Cost'] = display_df['Avg Cost'].apply(lambda x: f"${x:.2f}")
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
        display_df['Current Value'] = display_df['Current Value'].apply(lambda x: f"${x:,.2f}")
        display_df['Cost Basis'] = display_df['Cost Basis'].apply(lambda x: f"${x:,.2f}")
        display_df['Gain/Loss'] = display_df.apply(
            lambda row: f"${row['Gain/Loss']:,.2f}",
            axis=1
        )
        display_df['Gain/Loss %'] = display_df['Gain/Loss %'].apply(
            lambda x: f"{'🟢' if x >= 0 else '🔴'} {x:+.1f}%"
        )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                "Gain/Loss %": st.column_config.TextColumn("Return")
            }
        )
    else:
        st.info("No holdings in portfolio")

def render_allocation_chart():
    """Render portfolio allocation charts"""
    st.markdown("### Portfolio Allocation")
    
    holdings_df = calculate_portfolio_value()
    
    if not holdings_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Allocation pie chart
            allocation_data = holdings_df.set_index('Symbol')['Current Value'].to_dict()
            allocation_data['Cash'] = st.session_state.portfolio['cash']
            
            fig = ChartComponents.create_pie_chart(
                allocation_data,
                title="Asset Allocation",
                height=400
            )
            render_chart(fig)
        
        with col2:
            # Sector allocation (simplified)
            tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'PLTR']
            finance_symbols = ['JPM', 'BAC', 'GS', 'MS', 'V']
            healthcare_symbols = ['JNJ', 'PFE', 'MRNA', 'UNH']
            
            sector_allocation = {
                'Technology': 0,
                'Finance': 0,
                'Healthcare': 0,
                'Other': 0,
                'Cash': st.session_state.portfolio['cash']
            }
            
            for _, row in holdings_df.iterrows():
                if row['Symbol'] in tech_symbols:
                    sector_allocation['Technology'] += row['Current Value']
                elif row['Symbol'] in finance_symbols:
                    sector_allocation['Finance'] += row['Current Value']
                elif row['Symbol'] in healthcare_symbols:
                    sector_allocation['Healthcare'] += row['Current Value']
                else:
                    sector_allocation['Other'] += row['Current Value']
            
            # Remove empty sectors
            sector_allocation = {k: v for k, v in sector_allocation.items() if v > 0}
            
            fig = ChartComponents.create_pie_chart(
                sector_allocation,
                title="Sector Allocation",
                height=400
            )
            render_chart(fig)

def render_performance_chart():
    """Render portfolio performance chart"""
    st.markdown("### Portfolio Performance")
    
    # Generate sample historical data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    initial_value = 100000
    
    # Simulate portfolio performance
    returns = np.random.randn(90) * 0.01
    returns[0] = 0
    portfolio_values = initial_value * (1 + returns).cumprod()
    
    # Simulate benchmark
    benchmark_returns = np.random.randn(90) * 0.008
    benchmark_returns[0] = 0
    benchmark_values = initial_value * (1 + benchmark_returns).cumprod()
    
    df = pd.DataFrame({
        'Portfolio': portfolio_values,
        'S&P 500': benchmark_values
    }, index=dates)
    
    # Create chart
    fig = ChartComponents.create_line_chart(
        df,
        columns=['Portfolio', 'S&P 500'],
        title="90-Day Performance Comparison",
        height=400
    )
    render_chart(fig)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    portfolio_return = ((portfolio_values[-1] - initial_value) / initial_value) * 100
    benchmark_return = ((benchmark_values[-1] - initial_value) / initial_value) * 100
    alpha = portfolio_return - benchmark_return
    
    with col1:
        st.metric("Portfolio Return", f"{portfolio_return:.2f}%")
    with col2:
        st.metric("Benchmark Return", f"{benchmark_return:.2f}%")
    with col3:
        st.metric("Alpha", f"{alpha:+.2f}%")
    with col4:
        st.metric("Sharpe Ratio", f"{np.random.uniform(0.5, 1.5):.2f}")

def render_add_transaction():
    """Render add transaction form"""
    st.markdown("### Add Transaction")
    
    with st.form("add_transaction"):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            action = st.selectbox("Action", ["BUY", "SELL"])
        
        with col2:
            symbol = st.text_input("Symbol", placeholder="AAPL")
        
        with col3:
            shares = st.number_input("Shares", min_value=1, value=10)
        
        with col4:
            price = st.number_input("Price", min_value=0.01, value=100.00, step=0.01)
        
        with col5:
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Execute", use_container_width=True)
        
        if submit and symbol:
            # Update holdings
            holdings = st.session_state.portfolio['holdings']
            total_cost = shares * price
            
            if action == "BUY":
                if st.session_state.portfolio['cash'] >= total_cost:
                    if symbol in holdings:
                        # Update existing position
                        old_shares = holdings[symbol]['shares']
                        old_avg = holdings[symbol]['avg_price']
                        new_shares = old_shares + shares
                        new_avg = ((old_shares * old_avg) + (shares * price)) / new_shares
                        holdings[symbol] = {'shares': new_shares, 'avg_price': new_avg}
                    else:
                        # New position
                        holdings[symbol] = {'shares': shares, 'avg_price': price}
                    
                    st.session_state.portfolio['cash'] -= total_cost
                    
                    # Add to transactions
                    st.session_state.transactions.append({
                        'Date': datetime.now(),
                        'Action': action,
                        'Symbol': symbol,
                        'Shares': shares,
                        'Price': price,
                        'Total': total_cost
                    })
                    
                    st.success(f"Bought {shares} shares of {symbol} at ${price:.2f}")
                    st.rerun()
                else:
                    st.error("Insufficient cash!")
            
            elif action == "SELL":
                if symbol in holdings and holdings[symbol]['shares'] >= shares:
                    holdings[symbol]['shares'] -= shares
                    if holdings[symbol]['shares'] == 0:
                        del holdings[symbol]
                    
                    st.session_state.portfolio['cash'] += total_cost
                    
                    # Add to transactions
                    st.session_state.transactions.append({
                        'Date': datetime.now(),
                        'Action': action,
                        'Symbol': symbol,
                        'Shares': shares,
                        'Price': price,
                        'Total': total_cost
                    })
                    
                    st.success(f"Sold {shares} shares of {symbol} at ${price:.2f}")
                    st.rerun()
                else:
                    st.error("Insufficient shares!")

def render_transactions_history():
    """Render transaction history"""
    st.markdown("### Transaction History")
    
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')
        df['Price'] = df['Price'].apply(lambda x: f"${x:.2f}")
        df['Total'] = df['Total'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Date"),
                "Action": st.column_config.TextColumn("Action"),
                "Symbol": st.column_config.TextColumn("Symbol"),
                "Shares": st.column_config.NumberColumn("Shares"),
                "Price": st.column_config.TextColumn("Price"),
                "Total": st.column_config.TextColumn("Total")
            }
        )
    else:
        st.info("No transactions yet")

def render_rebalancing_suggestions():
    """Render portfolio rebalancing suggestions"""
    st.markdown("### Rebalancing Suggestions")
    
    suggestions = [
        {
            'Action': 'Reduce',
            'Symbol': 'NVDA',
            'Reason': 'Position exceeds 15% of portfolio',
            'Suggested': 'Sell 10 shares'
        },
        {
            'Action': 'Increase',
            'Symbol': 'JPM',
            'Reason': 'Underweight in financials sector',
            'Suggested': 'Buy 20 shares'
        },
        {
            'Action': 'Add',
            'Symbol': 'JNJ',
            'Reason': 'No healthcare exposure',
            'Suggested': 'Buy 15 shares'
        }
    ]
    
    for suggestion in suggestions:
        col1, col2, col3, col4 = st.columns([1, 1, 3, 2])
        
        with col1:
            if suggestion['Action'] == 'Reduce':
                st.error(suggestion['Action'])
            elif suggestion['Action'] == 'Increase':
                st.warning(suggestion['Action'])
            else:
                st.success(suggestion['Action'])
        
        with col2:
            st.write(suggestion['Symbol'])
        
        with col3:
            st.write(suggestion['Reason'])
        
        with col4:
            st.write(suggestion['Suggested'])

# Main function
def main():
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    if 'user_profile' in st.session_state:
        render_complete_sidebar(
            st.session_state.user_profile,
            st.session_state.portfolio,
            st.session_state.get('watchlist', [])[:5]
        )
    
    # Main header
    st.markdown("# 💼 Portfolio Management")
    st.caption("Track and manage your investment portfolio")
    
    # Portfolio summary
    render_portfolio_summary()
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Holdings",
        "📈 Performance",
        "➕ Transactions",
        "🎯 Allocation",
        "⚖️ Rebalancing"
    ])
    
    with tab1:
        render_holdings_table()
        st.divider()
        render_add_transaction()
    
    with tab2:
        render_performance_chart()
        
        # Risk metrics
        st.markdown("### Risk Metrics")
        performance_data = {
            'return_1D': 1.5,
            'return_1W': 3.2,
            'return_1M': 5.8,
            'return_3M': 12.3,
            'return_6M': 18.5,
            'return_1Y': 25.4,
            'win_rate': 68,
            'avg_win': 2.3,
            'avg_loss': -1.8,
            'best_day': 4.5,
            'worst_day': -3.2,
            'num_trades': 42
        }
        MetricComponents.render_performance_metrics(performance_data)
    
    with tab3:
        render_transactions_history()
    
    with tab4:
        render_allocation_chart()
    
    with tab5:
        render_rebalancing_suggestions()

if __name__ == "__main__":
    main()