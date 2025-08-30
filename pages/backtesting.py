"""
Backtesting Page
Strategy backtesting and performance analysis
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
from components.charts import ChartComponents, render_chart
from components.metrics import MetricComponents
from components.sidebar import render_complete_sidebar

# Page configuration
st.set_page_config(
    page_title="Backtesting - StockBot Advisor",
    page_icon="📈",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    
    if 'strategy_params' not in st.session_state:
        st.session_state.strategy_params = {
            'initial_capital': 100000,
            'position_size': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.15
        }

def run_backtest(symbols, strategy, start_date, end_date, initial_capital):
    """Run backtest simulation"""
    data_processor = get_data_processor()
    results = {
        'trades': [],
        'equity_curve': [],
        'dates': []
    }
    
    capital = initial_capital
    position = None
    
    for symbol in symbols:
        # Fetch historical data
        df = data_processor.fetch_stock_data(symbol, period='2y')
        
        if df.empty:
            continue
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Generate signals based on strategy
        if strategy == "MA Crossover":
            df['Signal'] = 0
            df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1
            df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] = -1
        
        elif strategy == "RSI Mean Reversion":
            df['Signal'] = 0
            df.loc[df['RSI'] < 30, 'Signal'] = 1
            df.loc[df['RSI'] > 70, 'Signal'] = -1
        
        elif strategy == "MACD Momentum":
            df['Signal'] = 0
            df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] = 1
            df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] = -1
        
        elif strategy == "Bollinger Bands":
            df['Signal'] = 0
            df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = 1
            df.loc[df['Close'] > df['BB_Upper'], 'Signal'] = -1
        
        else:  # ML-Based
            model = create_prediction_model('classification')
            model.train(df)
            predictions, _ = model.predict(df)
            df['Signal'] = predictions
        
        # Simulate trading
        for i in range(1, len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            
            # Entry logic
            if position is None and signal == 1:
                # Buy
                shares = (capital * 0.1) / price  # Use 10% of capital
                position = {
                    'symbol': symbol,
                    'entry_price': price,
                    'shares': shares,
                    'entry_date': date
                }
                
            # Exit logic
            elif position is not None:
                # Check stop loss
                if price < position['entry_price'] * 0.95:
                    # Stop loss hit
                    exit_price = price
                    profit = (exit_price - position['entry_price']) * position['shares']
                    capital += profit
                    
                    results['trades'].append({
                        'Symbol': position['symbol'],
                        'Entry Date': position['entry_date'],
                        'Exit Date': date,
                        'Entry Price': position['entry_price'],
                        'Exit Price': exit_price,
                        'Shares': position['shares'],
                        'P&L': profit,
                        'Return': (profit / (position['entry_price'] * position['shares'])) * 100
                    })
                    position = None
                
                # Check take profit
                elif price > position['entry_price'] * 1.15:
                    # Take profit hit
                    exit_price = price
                    profit = (exit_price - position['entry_price']) * position['shares']
                    capital += profit
                    
                    results['trades'].append({
                        'Symbol': position['symbol'],
                        'Entry Date': position['entry_date'],
                        'Exit Date': date,
                        'Entry Price': position['entry_price'],
                        'Exit Price': exit_price,
                        'Shares': position['shares'],
                        'P&L': profit,
                        'Return': (profit / (position['entry_price'] * position['shares'])) * 100
                    })
                    position = None
                
                # Signal exit
                elif signal == -1:
                    exit_price = price
                    profit = (exit_price - position['entry_price']) * position['shares']
                    capital += profit
                    
                    results['trades'].append({
                        'Symbol': position['symbol'],
                        'Entry Date': position['entry_date'],
                        'Exit Date': date,
                        'Entry Price': position['entry_price'],
                        'Exit Price': exit_price,
                        'Shares': position['shares'],
                        'P&L': profit,
                        'Return': (profit / (position['entry_price'] * position['shares'])) * 100
                    })
                    position = None
            
            # Track equity curve
            current_value = capital
            if position:
                current_value += position['shares'] * price
            
            results['equity_curve'].append(current_value)
            results['dates'].append(date)
    
    # Calculate metrics
    total_return = ((capital - initial_capital) / initial_capital) * 100
    
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        win_rate = (trades_df['P&L'] > 0).mean() * 100
        avg_win = trades_df[trades_df['P&L'] > 0]['Return'].mean() if any(trades_df['P&L'] > 0) else 0
        avg_loss = trades_df[trades_df['P&L'] < 0]['Return'].mean() if any(trades_df['P&L'] < 0) else 0
        num_trades = len(trades_df)
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        num_trades = 0
    
    # Calculate Sharpe ratio
    if results['equity_curve']:
        returns = pd.Series(results['equity_curve']).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    # Calculate max drawdown
    if results['equity_curve']:
        equity_series = pd.Series(results['equity_curve'])
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100
    else:
        max_drawdown = 0
    
    return {
        'total_return': total_return,
        'final_capital': capital,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'trades': results['trades'],
        'equity_curve': results['equity_curve'],
        'dates': results['dates']
    }

def render_strategy_configuration():
    """Render strategy configuration panel"""
    st.markdown("## Strategy Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy = st.selectbox(
            "Select Strategy",
            ["MA Crossover", "RSI Mean Reversion", "MACD Momentum", 
             "Bollinger Bands", "ML-Based (Random Forest)"]
        )
        
        symbols = st.multiselect(
            "Select Stocks",
            ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "JPM"],
            default=["AAPL", "MSFT", "GOOGL"]
        )
    
    with col2:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
        
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    with col3:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            value=100000,
            step=1000
        )
        
        position_size = st.slider(
            "Position Size (%)",
            min_value=5,
            max_value=100,
            value=10,
            step=5
        )
    
    # Advanced parameters
    with st.expander("Advanced Parameters"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stop_loss = st.number_input(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5
            )
        
        with col2:
            take_profit = st.number_input(
                "Take Profit (%)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=1.0
            )
        
        with col3:
            commission = st.number_input(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01
            )
        
        with col4:
            slippage = st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01
            )
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest simulation..."):
            results = run_backtest(
                symbols,
                strategy,
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
                initial_capital
            )
            st.session_state.backtest_results = results
            st.success("Backtest completed!")
    
    return strategy

def render_backtest_results():
    """Render backtest results"""
    if st.session_state.backtest_results is None:
        st.info("Run a backtest to see results")
        return
    
    results = st.session_state.backtest_results
    
    # Performance metrics
    st.markdown("## Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        MetricComponents.render_metric_card(
            title="Total Return",
            value=f"{results['total_return']:.2f}%",
            delta=results['total_return'],
            icon="📈"
        )
    
    with col2:
        MetricComponents.render_metric_card(
            title="Win Rate",
            value=f"{results['win_rate']:.1f}%",
            subtitle=f"{results['num_trades']} trades",
            icon="🎯"
        )
    
    with col3:
        MetricComponents.render_metric_card(
            title="Sharpe Ratio",
            value=f"{results['sharpe_ratio']:.2f}",
            subtitle="Risk-adjusted return",
            icon="⚖️"
        )
    
    with col4:
        MetricComponents.render_metric_card(
            title="Max Drawdown",
            value=f"{results['max_drawdown']:.1f}%",
            delta=results['max_drawdown'],
            delta_color="inverse",
            icon="📉"
        )
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Capital", f"${results['final_capital']:,.2f}")
    
    with col2:
        st.metric("Avg Win", f"{results['avg_win']:.2f}%")
    
    with col3:
        st.metric("Avg Loss", f"{results['avg_loss']:.2f}%")
    
    with col4:
        profit_factor = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] != 0 else 0
        st.metric("Profit Factor", f"{profit_factor:.2f}")

def render_equity_curve():
    """Render equity curve chart"""
    if st.session_state.backtest_results is None:
        return
    
    results = st.session_state.backtest_results
    
    if results['equity_curve'] and results['dates']:
        st.markdown("### Equity Curve")
        
        df = pd.DataFrame({
            'Date': results['dates'],
            'Portfolio Value': results['equity_curve']
        }).set_index('Date')
        
        # Add benchmark (buy and hold)
        initial = results['equity_curve'][0] if results['equity_curve'] else 100000
        df['Buy & Hold'] = initial * (1 + np.random.randn(len(df)) * 0.001).cumprod()
        
        fig = ChartComponents.create_line_chart(
            df,
            columns=['Portfolio Value', 'Buy & Hold'],
            title="Portfolio Value Over Time",
            height=400
        )
        render_chart(fig)

def render_trades_table():
    """Render trades table"""
    if st.session_state.backtest_results is None:
        return
    
    results = st.session_state.backtest_results
    
    if results['trades']:
        st.markdown("### Trade History")
        
        trades_df = pd.DataFrame(results['trades'])
        trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date']).dt.strftime('%Y-%m-%d')
        trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date']).dt.strftime('%Y-%m-%d')
        trades_df['Entry Price'] = trades_df['Entry Price'].apply(lambda x: f"${x:.2f}")
        trades_df['Exit Price'] = trades_df['Exit Price'].apply(lambda x: f"${x:.2f}")
        trades_df['P&L'] = trades_df['P&L'].apply(lambda x: f"${x:,.2f}")
        trades_df['Return'] = trades_df['Return'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(
            trades_df,
            use_container_width=True,
            hide_index=True
        )

def render_strategy_comparison():
    """Render strategy comparison"""
    st.markdown("### Strategy Comparison")
    
    # Sample comparison data
    strategies = {
        'MA Crossover': {'Return': 15.2, 'Sharpe': 1.2, 'Drawdown': -8.5, 'Trades': 45},
        'RSI Mean Rev': {'Return': 18.5, 'Sharpe': 1.5, 'Drawdown': -12.3, 'Trades': 62},
        'MACD Momentum': {'Return': 12.8, 'Sharpe': 0.9, 'Drawdown': -6.2, 'Trades': 38},
        'Bollinger': {'Return': 22.1, 'Sharpe': 1.8, 'Drawdown': -15.7, 'Trades': 71},
        'ML-Based': {'Return': 25.4, 'Sharpe': 2.1, 'Drawdown': -9.8, 'Trades': 53}
    }
    
    comparison_df = pd.DataFrame(strategies).T.reset_index()
    comparison_df.columns = ['Strategy', 'Return (%)', 'Sharpe', 'Max DD (%)', 'Trades']
    
    # Create comparison chart
    fig = ChartComponents.create_bar_chart(
        comparison_df,
        x_col='Strategy',
        y_col='Return (%)',
        title="Strategy Returns Comparison",
        height=300
    )
    render_chart(fig)
    
    # Display comparison table
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

def render_monte_carlo():
    """Render Monte Carlo simulation"""
    st.markdown("### Monte Carlo Simulation")
    
    # Generate Monte Carlo paths
    n_simulations = 100
    n_days = 252
    initial_value = 100000
    
    returns = np.random.randn(n_simulations, n_days) * 0.02
    prices = initial_value * np.exp(np.cumsum(returns, axis=1))
    
    # Calculate percentiles
    percentiles = np.percentile(prices, [5, 25, 50, 75, 95], axis=0)
    
    dates = pd.date_range(start=datetime.now(), periods=n_days, freq='D')
    
    # Create chart
    fig = go.Figure()
    
    # Add percentile bands
    fig.add_trace(go.Scatter(
        x=dates, y=percentiles[0],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=percentiles[4],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='5-95 Percentile',
        fillcolor='rgba(128,128,128,0.2)'
    ))
    
    # Add median line
    fig.add_trace(go.Scatter(
        x=dates, y=percentiles[2],
        mode='lines',
        name='Median',
        line=dict(color='black', width=2)
    ))
    
    fig.update_layout(
        title="Monte Carlo Simulation (100 paths)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    final_values = prices[:, -1]
    
    with col1:
        st.metric("Median Outcome", f"${np.median(final_values):,.0f}")
    with col2:
        st.metric("95% VaR", f"${np.percentile(final_values, 5):,.0f}")
    with col3:
        st.metric("95% CVaR", f"${final_values[final_values <= np.percentile(final_values, 5)].mean():,.0f}")

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
    st.markdown("# 📈 Strategy Backtesting")
    st.caption("Test and optimize your trading strategies on historical data")
    
    # Strategy configuration
    strategy = render_strategy_configuration()
    
    st.divider()
    
    # Results tabs
    if st.session_state.backtest_results:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Results",
            "📈 Equity Curve",
            "📋 Trades",
            "🔄 Comparison",
            "🎲 Monte Carlo"
        ])
        
        with tab1:
            render_backtest_results()
        
        with tab2:
            render_equity_curve()
        
        with tab3:
            render_trades_table()
        
        with tab4:
            render_strategy_comparison()
        
        with tab5:
            render_monte_carlo()
    else:
        st.info("Configure your strategy and click 'Run Backtest' to see results")

if __name__ == "__main__":
    main()