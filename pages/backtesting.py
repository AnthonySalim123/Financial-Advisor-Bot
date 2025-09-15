# pages/4_📈_Backtesting.py
"""
Production Version - Strategy Backtesting Module
Final version with all features and professional UI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing custom modules
try:
    from utils.technical_indicators import TechnicalIndicators
except ImportError:
    # Fallback if TechnicalIndicators not available
    class TechnicalIndicators:
        @staticmethod
        def calculate_rsi(data, period=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        @staticmethod
        def calculate_macd(data, fast=12, slow=26, signal=9):
            ema_fast = data.ewm(span=fast, adjust=False).mean()
            ema_slow = data.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return {'MACD': macd, 'Signal': signal_line}

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
        st.session_state.strategy_params = {}

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        # Remove timezone if present
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculate all technical indicators"""
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'])
    
    # MACD
    macd_result = TechnicalIndicators.calculate_macd(df['Close'])
    df['MACD'] = macd_result['MACD']
    df['MACD_Signal'] = macd_result['Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def generate_signals(df, strategy_type):
    """Generate trading signals based on strategy"""
    signals = pd.Series(0, index=df.index, dtype=int)
    
    if strategy_type == "MA Crossover":
        # Simple MA crossover strategy
        fast_ma = df['SMA_10'].values
        slow_ma = df['SMA_30'].values
        
        for i in range(1, len(df)):
            if pd.notna(fast_ma[i]) and pd.notna(slow_ma[i]):
                # Bullish crossover
                if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                    signals.iloc[i] = 1
                # Bearish crossover
                elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                    signals.iloc[i] = -1
    
    elif strategy_type == "RSI Mean Reversion":
        # RSI-based strategy
        rsi = df['RSI'].values
        
        for i in range(1, len(df)):
            if pd.notna(rsi[i]):
                # Buy on oversold
                if rsi[i] < 30 and rsi[i-1] >= 30:
                    signals.iloc[i] = 1
                # Sell on overbought
                elif rsi[i] > 70 and rsi[i-1] <= 70:
                    signals.iloc[i] = -1
    
    elif strategy_type == "MACD Momentum":
        # MACD strategy
        macd = df['MACD'].values
        signal_line = df['MACD_Signal'].values
        
        for i in range(1, len(df)):
            if pd.notna(macd[i]) and pd.notna(signal_line[i]):
                # Bullish crossover
                if macd[i] > signal_line[i] and macd[i-1] <= signal_line[i-1]:
                    signals.iloc[i] = 1
                # Bearish crossover
                elif macd[i] < signal_line[i] and macd[i-1] >= signal_line[i-1]:
                    signals.iloc[i] = -1
    
    elif strategy_type == "Bollinger Bands":
        # Bollinger Bands strategy
        close = df['Close'].values
        upper = df['BB_Upper'].values
        lower = df['BB_Lower'].values
        
        for i in range(1, len(df)):
            if pd.notna(upper[i]) and pd.notna(lower[i]):
                # Buy when price touches lower band
                if close[i] <= lower[i] and close[i-1] > lower[i-1]:
                    signals.iloc[i] = 1
                # Sell when price touches upper band
                elif close[i] >= upper[i] and close[i-1] < upper[i-1]:
                    signals.iloc[i] = -1
    
    return signals

def run_backtest(symbols, strategy_type, start_date, end_date, initial_capital, params):
    """Run the backtest simulation"""
    
    results = {
        'trades': [],
        'equity_curve': [initial_capital],
        'dates': [],
        'metrics': {},
        'daily_returns': []
    }
    
    # Parameters
    POSITION_SIZE_PCT = params.get('position_size', 0.1)
    STOP_LOSS = params.get('stop_loss', 0.05)
    TAKE_PROFIT = params.get('take_profit', 0.15)
    COMMISSION = params.get('commission', 0.001)
    
    cash = initial_capital
    position = 0
    entry_price = 0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    all_data = pd.DataFrame()
    
    for symbol in symbols:
        # Fetch data
        df = fetch_stock_data(symbol, start_date, end_date)
        
        if df.empty:
            st.warning(f"No data available for {symbol}")
            continue
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Generate signals
        df['Signal'] = generate_signals(df, strategy_type)
        
        # Simulate trading
        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            
            # Entry logic
            if position == 0 and signal == 1:
                shares = (cash * POSITION_SIZE_PCT) / price
                cost = shares * price * (1 + COMMISSION)
                
                if cost <= cash:
                    cash -= cost
                    position = shares
                    entry_price = price
                    total_trades += 1
                    
                    results['trades'].append({
                        'Date': date,
                        'Symbol': symbol,
                        'Type': 'BUY',
                        'Price': price,
                        'Shares': shares,
                        'Value': cost
                    })
            
            # Exit logic
            elif position > 0:
                exit_trade = False
                exit_reason = ""
                
                # Check exit conditions
                return_pct = (price - entry_price) / entry_price
                
                if signal == -1:
                    exit_trade = True
                    exit_reason = "Signal"
                elif return_pct >= TAKE_PROFIT:
                    exit_trade = True
                    exit_reason = "Take Profit"
                elif return_pct <= -STOP_LOSS:
                    exit_trade = True
                    exit_reason = "Stop Loss"
                elif i == len(df) - 1:
                    exit_trade = True
                    exit_reason = "End Period"
                
                if exit_trade:
                    proceeds = position * price * (1 - COMMISSION)
                    pnl = proceeds - (position * entry_price)
                    pnl_pct = (pnl / (position * entry_price)) * 100
                    
                    cash += proceeds
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    results['trades'].append({
                        'Date': date,
                        'Symbol': symbol,
                        'Type': 'SELL',
                        'Price': price,
                        'Shares': position,
                        'Value': proceeds,
                        'PnL': pnl,
                        'Return%': pnl_pct,
                        'Reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Track equity
            equity = cash + (position * price if position > 0 else 0)
            results['equity_curve'].append(equity)
            results['dates'].append(date)
    
    # Calculate metrics
    final_value = cash + (position * df['Close'].iloc[-1] if position > 0 else 0)
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    # Calculate daily returns for Sharpe ratio
    equity_array = np.array(results['equity_curve'])
    if len(equity_array) > 1:
        daily_returns = np.diff(equity_array) / equity_array[:-1]
        results['daily_returns'] = daily_returns
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) * 100
        
        # Volatility
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
    else:
        sharpe = 0
        max_drawdown = 0
        volatility = 0
    
    results['metrics'] = {
        'initial_capital': initial_capital,
        'final_capital': final_value,
        'total_return': total_return,
        'num_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'volatility': volatility
    }
    
    return results

def render_strategy_configuration():
    """Render strategy configuration panel"""
    st.markdown("## 📊 Strategy Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy = st.selectbox(
            "Select Strategy",
            ["MA Crossover", "RSI Mean Reversion", "MACD Momentum", "Bollinger Bands"],
            help="Choose the trading strategy to backtest"
        )
        
        symbols = st.multiselect(
            "Select Stocks",
            ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA"],
            default=["AAPL"],
            help="Select one or more stocks to test"
        )
    
    with col2:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now(),
            help="Backtest start date"
        )
        
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now(),
            help="Backtest end date"
        )
    
    with col3:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            value=100000,
            step=1000,
            help="Starting capital for the backtest"
        )
    
    # Advanced parameters
    with st.expander("⚙️ Advanced Parameters"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            position_size = st.slider(
                "Position Size (%)",
                min_value=5,
                max_value=100,
                value=10,
                step=5,
                help="Percentage of capital per trade"
            )
        
        with col2:
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Stop loss percentage"
            )
        
        with col3:
            take_profit = st.slider(
                "Take Profit (%)",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                help="Take profit percentage"
            )
        
        with col4:
            commission = st.slider(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Trading commission percentage"
            )
    
    # Run backtest button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            if not symbols:
                st.error("Please select at least one stock")
                return None
            
            with st.spinner("Running backtest... This may take a moment."):
                params = {
                    'position_size': position_size / 100,
                    'stop_loss': stop_loss / 100,
                    'take_profit': take_profit / 100,
                    'commission': commission / 100
                }
                
                results = run_backtest(
                    symbols, strategy, start_date, end_date, 
                    initial_capital, params
                )
                
                st.session_state.backtest_results = results
                st.success("✅ Backtest completed successfully!")
    
    return strategy

def render_results():
    """Render backtest results"""
    if st.session_state.backtest_results is None:
        st.info("👆 Configure your strategy and click 'Run Backtest' to see results")
        return
    
    results = st.session_state.backtest_results
    metrics = results['metrics']
    
    # Performance Summary
    st.markdown("## 📈 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = metrics['total_return']
        st.metric(
            "📊 Total Return",
            f"{metrics['total_return']:.2f}%",
            delta=f"{delta:.2f}%" if delta != 0 else None
        )
    
    with col2:
        st.metric(
            "⚖️ Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}"
        )
    
    with col3:
        st.metric(
            "🎯 Win Rate",
            f"{metrics['win_rate']:.1f}%",
            delta=f"{metrics['num_trades']} trades"
        )
    
    with col4:
        st.metric(
            "📉 Max Drawdown",
            f"{metrics['max_drawdown']:.1f}%",
            delta=f"-{metrics['max_drawdown']:.1f}%"
        )
    
    # Detailed Metrics
    st.markdown("## 📊 Detailed Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Initial Capital", f"${metrics['initial_capital']:,.2f}")
        st.metric("Final Capital", f"${metrics['final_capital']:,.2f}")
    
    with col2:
        st.metric("Total Trades", metrics['num_trades'])
        st.metric("Volatility", f"{metrics['volatility']:.1f}%")
    
    with col3:
        st.metric("Winning Trades", metrics['winning_trades'])
        st.metric("Losing Trades", metrics['losing_trades'])
    
    with col4:
        avg_win = 0
        avg_loss = 0
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            if 'PnL' in trades_df.columns:
                wins = trades_df[trades_df['PnL'] > 0]['PnL']
                losses = trades_df[trades_df['PnL'] < 0]['PnL']
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = losses.mean() if len(losses) > 0 else 0
        
        st.metric("Avg Win", f"${avg_win:.2f}")
        st.metric("Avg Loss", f"${avg_loss:.2f}")

def render_charts():
    """Render charts"""
    if st.session_state.backtest_results is None:
        return
    
    results = st.session_state.backtest_results
    
    tab1, tab2, tab3 = st.tabs(["📈 Equity Curve", "📊 Trades", "📉 Drawdown"])
    
    with tab1:
        if len(results['equity_curve']) > 1:
            fig = go.Figure()
            
            # Equity curve
            fig.add_trace(go.Scatter(
                x=list(range(len(results['equity_curve']))),
                y=results['equity_curve'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 255, 0.1)'
            ))
            
            # Initial capital line
            fig.add_hline(
                y=results['metrics']['initial_capital'],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital"
            )
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Trading Days",
                yaxis_title="Portfolio Value ($)",
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            
            # Format display
            display_df = trades_df.copy()
            
            # Format numeric columns
            for col in ['Price', 'Value']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
            
            if 'Shares' in display_df.columns:
                display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:.2f}")
            
            if 'PnL' in display_df.columns:
                display_df['PnL'] = display_df['PnL'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "-"
                )
            
            if 'Return%' in display_df.columns:
                display_df['Return%'] = display_df['Return%'].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "-"
                )
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed")
    
    with tab3:
        if len(results['equity_curve']) > 1:
            # Calculate drawdown
            equity_array = np.array(results['equity_curve'])
            running_max = np.maximum.accumulate(equity_array)
            drawdown = ((equity_array - running_max) / running_max) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(drawdown))),
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ))
            
            fig.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Trading Days",
                yaxis_title="Drawdown (%)",
                height=400,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Main function
def main():
    initialize_session_state()
    
    # Header
    st.markdown("# 📈 Strategy Backtesting")
    st.caption("Test and optimize your trading strategies with comprehensive historical analysis")
    
    # Strategy configuration
    render_strategy_configuration()
    
    # Divider
    st.divider()
    
    # Results section
    render_results()
    
    # Charts section
    if st.session_state.backtest_results:
        st.divider()
        render_charts()
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Past performance does not guarantee future results. This is for educational purposes only.")

if __name__ == "__main__":
    main()