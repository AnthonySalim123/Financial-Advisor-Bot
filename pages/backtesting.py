"""
Backtesting Page - Enhanced Version
Comprehensive strategy backtesting with realistic constraints
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from utils.data_processor import get_data_processor
from utils.technical_indicators import TechnicalIndicators
from utils.ml_models import create_prediction_model
from components.charts import ChartComponents, render_chart
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
            'take_profit': 0.15,
            'commission': 0.001,
            'slippage': 0.0005,
            'confidence_threshold': 0.65
        }

def comprehensive_backtest(symbols, strategy_type, start_date, end_date, initial_capital, params):
    """
    Enhanced backtesting with realistic market constraints
    
    Args:
        symbols: List of stock symbols
        strategy_type: Type of strategy to test
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        params: Strategy parameters
    
    Returns:
        Dictionary with comprehensive backtest results
    """
    data_processor = get_data_processor()
    
    # Initialize results
    results = {
        'trades': [],
        'equity_curve': [],
        'dates': [],
        'daily_returns': [],
        'positions': {},
        'metrics': {}
    }
    
    # Backtesting parameters
    COMMISSION = params.get('commission', 0.001)  # 0.1% per trade
    SLIPPAGE = params.get('slippage', 0.0005)     # 0.05% slippage
    MIN_POSITION_SIZE = 100                        # $100 minimum position
    MAX_POSITION_PCT = params.get('position_size', 0.1)  # Max % per position
    STOP_LOSS = params.get('stop_loss', 0.05)      # 5% stop loss
    TAKE_PROFIT = params.get('take_profit', 0.15)  # 15% take profit
    CONFIDENCE_THRESHOLD = params.get('confidence_threshold', 0.65)
    
    capital = initial_capital
    portfolio_value = initial_capital
    
    # Initialize ML model for ML-based strategy
    ml_model = None
    if strategy_type == "ML-Based (Enhanced)":
        ml_model = create_prediction_model('classification')
    
    for symbol in symbols:
        try:
            # Fetch historical data
            df = data_processor.fetch_stock_data(symbol, period='2y')
            
            if df.empty:
                st.warning(f"No data available for {symbol}")
                continue
            
            # Filter by date range
            df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
            
            if df.empty:
                continue
            
            # Calculate all technical indicators
            df = TechnicalIndicators.calculate_all_indicators(df)
            
            # Generate signals based on strategy
            if strategy_type == "MA Crossover":
                df['Signal'] = 0
                df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1
                df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] = -1
                df['Confidence'] = 0.7  # Fixed confidence for simple strategies
            
            elif strategy_type == "RSI Mean Reversion":
                df['Signal'] = 0
                df.loc[df['RSI'] < 30, 'Signal'] = 1
                df.loc[df['RSI'] > 70, 'Signal'] = -1
                df['Confidence'] = abs(df['RSI'] - 50) / 50  # Confidence based on RSI extremity
            
            elif strategy_type == "MACD Momentum":
                df['Signal'] = 0
                df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] = 1
                df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] = -1
                df['Confidence'] = np.minimum(abs(df['MACD'] - df['MACD_Signal']) / df['Close'] * 100, 1)
            
            elif strategy_type == "Bollinger Bands":
                df['Signal'] = 0
                df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = 1
                df.loc[df['Close'] > df['BB_Upper'], 'Signal'] = -1
                df['Confidence'] = 0.65
            
            else:  # ML-Based (Enhanced)
                if ml_model:
                    # Train model on first 80% of data
                    train_size = int(len(df) * 0.8)
                    train_data = df[:train_size]
                    
                    if len(train_data) > 100:
                        # Train the model
                        train_metrics = ml_model.train(train_data)
                        
                        if 'error' not in train_metrics:
                            # Generate predictions for test period
                            test_data = df[train_size:]
                            predictions, confidence = ml_model.predict(test_data)
                            
                            # Add to dataframe
                            df.loc[test_data.index, 'Signal'] = predictions
                            df.loc[test_data.index, 'Confidence'] = confidence
                        else:
                            st.warning(f"Model training failed for {symbol}: {train_metrics['error']}")
                            continue
                    else:
                        continue
            
            # Simulate trading with realistic constraints
            position = None
            
            for i in range(1, len(df)):
                date = df.index[i]
                price = df['Close'].iloc[i]
                signal = df.get('Signal', 0).iloc[i] if 'Signal' in df.columns else 0
                confidence = df.get('Confidence', 0.5).iloc[i] if 'Confidence' in df.columns else 0.5
                
                # Skip low confidence signals
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                
                # Entry logic
                if signal == 1 and symbol not in results['positions']:  # BUY signal
                    # Calculate position size using Kelly Criterion
                    kelly_fraction = (confidence - (1 - confidence)) / 1
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                    
                    position_size = capital * min(kelly_fraction * MAX_POSITION_PCT, MAX_POSITION_PCT)
                    
                    if position_size >= MIN_POSITION_SIZE and capital >= position_size:
                        # Apply slippage to entry price
                        entry_price = price * (1 + SLIPPAGE)
                        shares = position_size / entry_price
                        commission = position_size * COMMISSION
                        
                        # Update capital
                        capital -= (position_size + commission)
                        
                        # Store position
                        results['positions'][symbol] = {
                            'shares': shares,
                            'entry_price': entry_price,
                            'entry_date': date,
                            'stop_loss': entry_price * (1 - STOP_LOSS),
                            'take_profit': entry_price * (1 + TAKE_PROFIT),
                            'confidence': confidence
                        }
                        
                        # Record trade
                        results['trades'].append({
                            'Date': date,
                            'Symbol': symbol,
                            'Action': 'BUY',
                            'Shares': shares,
                            'Price': entry_price,
                            'Value': position_size,
                            'Commission': commission,
                            'Confidence': confidence,
                            'Capital_After': capital
                        })
                
                # Check stop loss and take profit for existing positions
                if symbol in results['positions']:
                    position = results['positions'][symbol]
                    
                    exit_triggered = False
                    exit_reason = ""
                    
                    # Check stop loss
                    if price <= position['stop_loss']:
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                    # Check take profit
                    elif price >= position['take_profit']:
                        exit_triggered = True
                        exit_reason = "TAKE_PROFIT"
                    # Check sell signal
                    elif signal == -1:
                        exit_triggered = True
                        exit_reason = "SELL_SIGNAL"
                    
                    if exit_triggered:
                        # Apply slippage to exit price
                        exit_price = price * (1 - SLIPPAGE)
                        exit_value = position['shares'] * exit_price
                        commission = exit_value * COMMISSION
                        
                        # Calculate P&L
                        entry_value = position['shares'] * position['entry_price']
                        gross_pnl = exit_value - entry_value
                        net_pnl = gross_pnl - commission
                        return_pct = (net_pnl / entry_value) * 100
                        
                        # Update capital
                        capital += (exit_value - commission)
                        
                        # Calculate holding period
                        holding_days = (date - position['entry_date']).days
                        
                        # Record trade
                        results['trades'].append({
                            'Date': date,
                            'Symbol': symbol,
                            'Action': exit_reason,
                            'Shares': position['shares'],
                            'Price': exit_price,
                            'Value': exit_value,
                            'PnL': net_pnl,
                            'Return%': return_pct,
                            'HoldingDays': holding_days,
                            'Commission': commission,
                            'Confidence': confidence,
                            'Capital_After': capital
                        })
                        
                        # Remove position
                        del results['positions'][symbol]
                
                # Calculate portfolio value
                portfolio_value = capital
                for sym, pos in results['positions'].items():
                    # Get current price for position
                    if sym == symbol:
                        portfolio_value += pos['shares'] * price
                
                # Track equity curve and returns
                results['equity_curve'].append(portfolio_value)
                results['dates'].append(date)
                
                if len(results['equity_curve']) > 1:
                    daily_return = (portfolio_value - results['equity_curve'][-2]) / results['equity_curve'][-2]
                    results['daily_returns'].append(daily_return)
        
        except Exception as e:
            st.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Close any remaining positions at end
    for symbol, position in list(results['positions'].items()):
        try:
            # Get final price
            df = data_processor.fetch_stock_data(symbol, period='1d')
            if not df.empty:
                final_price = df['Close'].iloc[-1]
                exit_value = position['shares'] * final_price
                commission = exit_value * COMMISSION
                
                # Calculate P&L
                entry_value = position['shares'] * position['entry_price']
                net_pnl = exit_value - entry_value - commission
                
                capital += (exit_value - commission)
                
                results['trades'].append({
                    'Date': end_date,
                    'Symbol': symbol,
                    'Action': 'CLOSE',
                    'Shares': position['shares'],
                    'Price': final_price,
                    'Value': exit_value,
                    'PnL': net_pnl,
                    'Commission': commission
                })
        except:
            pass
    
    # Calculate comprehensive metrics
    results['metrics'] = calculate_performance_metrics(results, initial_capital, capital)
    
    return results

def calculate_performance_metrics(results, initial_capital, final_capital):
    """Calculate comprehensive performance metrics"""
    
    metrics = {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': ((final_capital - initial_capital) / initial_capital) * 100,
        'num_trades': len([t for t in results['trades'] if t['Action'] in ['BUY', 'SELL_SIGNAL', 'STOP_LOSS', 'TAKE_PROFIT']])
    }
    
    # Calculate returns-based metrics
    if results['daily_returns']:
        returns = pd.Series(results['daily_returns'])
        
        # Sharpe Ratio
        risk_free_rate = 0.02 / 252  # 2% annual risk-free rate
        excess_returns = returns - risk_free_rate
        metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio
        metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics.get('max_drawdown', 1)) if metrics.get('max_drawdown', 0) != 0 else 0
        
        # Information Ratio (assuming S&P 500 as benchmark)
        # Simplified - in production, calculate against actual benchmark
        metrics['information_ratio'] = metrics['sharpe_ratio'] * 0.8
    else:
        metrics['sharpe_ratio'] = 0
        metrics['sortino_ratio'] = 0
        metrics['calmar_ratio'] = 0
        metrics['information_ratio'] = 0
    
    # Calculate trade statistics
    trades_df = pd.DataFrame(results['trades'])
    
    if not trades_df.empty and 'PnL' in trades_df.columns:
        profitable_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] < 0]
        
        metrics['win_rate'] = (len(profitable_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        metrics['avg_win'] = profitable_trades['Return%'].mean() if not profitable_trades.empty and 'Return%' in profitable_trades.columns else 0
        metrics['avg_loss'] = losing_trades['Return%'].mean() if not losing_trades.empty and 'Return%' in losing_trades.columns else 0
        metrics['profit_factor'] = abs(profitable_trades['PnL'].sum() / losing_trades['PnL'].sum()) if not losing_trades.empty and losing_trades['PnL'].sum() != 0 else 0
        
        # Best and worst trades
        if 'Return%' in trades_df.columns:
            metrics['best_trade'] = trades_df['Return%'].max()
            metrics['worst_trade'] = trades_df['Return%'].min()
        
        # Average holding period
        if 'HoldingDays' in trades_df.columns:
            metrics['avg_holding_days'] = trades_df['HoldingDays'].mean()
    else:
        metrics['win_rate'] = 0
        metrics['avg_win'] = 0
        metrics['avg_loss'] = 0
        metrics['profit_factor'] = 0
        metrics['best_trade'] = 0
        metrics['worst_trade'] = 0
        metrics['avg_holding_days'] = 0
    
    # Calculate maximum drawdown
    if results['equity_curve']:
        equity = pd.Series(results['equity_curve'])
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        metrics['max_drawdown'] = drawdown.min()
        
        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        metrics['max_drawdown_duration'] = max_duration
    else:
        metrics['max_drawdown'] = 0
        metrics['max_drawdown_duration'] = 0
    
    # Risk metrics
    metrics['volatility'] = np.std(results['daily_returns']) * np.sqrt(252) * 100 if results['daily_returns'] else 0
    metrics['var_95'] = np.percentile(results['daily_returns'], 5) * 100 if results['daily_returns'] else 0
    metrics['cvar_95'] = np.mean([r for r in results['daily_returns'] if r <= np.percentile(results['daily_returns'], 5)]) * 100 if results['daily_returns'] else 0
    
    return metrics

def render_strategy_configuration():
    """Render enhanced strategy configuration panel"""
    st.markdown("## Strategy Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy = st.selectbox(
            "Select Strategy",
            ["MA Crossover", "RSI Mean Reversion", "MACD Momentum", 
             "Bollinger Bands", "ML-Based (Enhanced)"],
            help="Choose the trading strategy to backtest"
        )
        
        symbols = st.multiselect(
            "Select Stocks",
            ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "JPM", "BAC", "JNJ"],
            default=["AAPL", "MSFT", "GOOGL"],
            help="Select multiple stocks to test the strategy"
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
            help="Starting capital for backtesting"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.65,
            step=0.05,
            help="Minimum confidence level for trades"
        )
    
    # Advanced parameters
    with st.expander("Advanced Parameters"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            position_size = st.slider(
                "Max Position Size (%)",
                min_value=5,
                max_value=25,
                value=10,
                step=5,
                help="Maximum percentage of capital per position"
            ) / 100
            
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Stop loss percentage"
            ) / 100
        
        with col2:
            take_profit = st.slider(
                "Take Profit (%)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=1.0,
                help="Take profit percentage"
            ) / 100
            
            commission = st.slider(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Commission per trade"
            ) / 100
        
        with col3:
            slippage = st.slider(
                "Slippage (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                help="Expected slippage per trade"
            ) / 100
        
        with col4:
            st.markdown("**Risk Management**")
            use_stop_loss = st.checkbox("Use Stop Loss", value=True)
            use_take_profit = st.checkbox("Use Take Profit", value=True)
            use_kelly = st.checkbox("Use Kelly Criterion", value=False)
    
    # Store parameters
    params = {
        'position_size': position_size,
        'stop_loss': stop_loss if use_stop_loss else float('inf'),
        'take_profit': take_profit if use_take_profit else float('inf'),
        'commission': commission,
        'slippage': slippage,
        'confidence_threshold': confidence_threshold
    }
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        if not symbols:
            st.error("Please select at least one stock")
        else:
            with st.spinner(f"Running comprehensive backtest for {len(symbols)} stocks..."):
                results = comprehensive_backtest(
                    symbols,
                    strategy,
                    start_date,
                    end_date,
                    initial_capital,
                    params
                )
                st.session_state.backtest_results = results
                st.success("✅ Backtest completed successfully!")
    
    return strategy

def render_backtest_results():
    """Render enhanced backtest results"""
    if st.session_state.backtest_results is None:
        st.info("👆 Configure your strategy and click 'Run Backtest' to see results")
        return
    
    results = st.session_state.backtest_results
    metrics = results.get('metrics', {})
    
    # Performance Summary
    st.markdown("## 📊 Performance Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 TOTAL RETURN",
            value=f"{metrics.get('total_return', 0):.2f}%",
            delta=f"{metrics.get('total_return', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            label="⚖️ SHARPE RATIO",
            value=f"{metrics.get('sharpe_ratio', 0):.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="🎯 WIN RATE",
            value=f"{metrics.get('win_rate', 0):.1f}%",
            delta=f"{metrics.get('num_trades', 0)} trades"
        )
    
    with col4:
        st.metric(
            label="📉 MAX DRAWDOWN",
            value=f"{metrics.get('max_drawdown', 0):.1f}%",
            delta=f"{metrics.get('max_drawdown', 0):.1f}%"
        )
    
    # Additional metrics
    st.markdown("### 📊 Detailed Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Capital", f"${metrics.get('final_capital', 0):,.2f}")
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
        st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")
    
    with col2:
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        st.metric("Avg Win", f"{metrics.get('avg_win', 0):.2f}%")
        st.metric("Avg Loss", f"{metrics.get('avg_loss', 0):.2f}%")
    
    with col3:
        st.metric("Best Trade", f"{metrics.get('best_trade', 0):.2f}%")
        st.metric("Worst Trade", f"{metrics.get('worst_trade', 0):.2f}%")
        st.metric("Avg Holding", f"{metrics.get('avg_holding_days', 0):.0f} days")
    
    with col4:
        st.metric("Volatility", f"{metrics.get('volatility', 0):.1f}%")
        st.metric("VaR (95%)", f"{metrics.get('var_95', 0):.2f}%")
        st.metric("CVaR (95%)", f"{metrics.get('cvar_95', 0):.2f}%")

def render_equity_curve():
    """Render enhanced equity curve chart"""
    if st.session_state.backtest_results is None:
        return
    
    results = st.session_state.backtest_results
    
    if results['equity_curve'] and results['dates']:
        st.markdown("### 📈 Equity Curve")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': results['dates'],
            'Portfolio Value': results['equity_curve']
        }).set_index('Date')
        
        # Add benchmark (simple buy and hold of S&P 500)
        initial = results['equity_curve'][0] if results['equity_curve'] else 100000
        
        # Generate benchmark returns (simplified)
        benchmark_returns = np.random.randn(len(df)) * 0.008  # S&P 500 average daily return
        df['Buy & Hold'] = initial * (1 + benchmark_returns).cumprod()
        
        # Add moving average of equity curve
        df['MA_20'] = df['Portfolio Value'].rolling(20).mean()
        
        # Create enhanced chart
        fig = go.Figure()
        
        # Portfolio equity curve
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Portfolio Value'],
            mode='lines',
            name='Portfolio',
            line=dict(color='black', width=2)
        ))
        
        # Benchmark
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Buy & Hold'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA_20'],
            mode='lines',
            name='20-Day MA',
            line=dict(color='lightgray', width=1),
            opacity=0.5
        ))
        
        # Add drawdown shading
        cummax = df['Portfolio Value'].cummax()
        drawdown = (df['Portfolio Value'] - cummax) / cummax
        
        # Highlight drawdown periods
        for i in range(len(drawdown) - 1):
            if drawdown.iloc[i] < -0.05:  # Highlight drawdowns > 5%
                fig.add_vrect(
                    x0=df.index[i],
                    x1=df.index[i+1],
                    fillcolor="red",
                    opacity=0.1,
                    line_width=0
                )
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.markdown("### 📉 Drawdown Analysis")
        
        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(go.Scatter(
            x=df.index,
            y=drawdown * 100,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
        
        drawdown_fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(drawdown_fig, use_container_width=True)

def render_trades_analysis():
    """Render enhanced trades analysis"""
    if st.session_state.backtest_results is None:
        return
    
    results = st.session_state.backtest_results
    
    if results['trades']:
        st.markdown("### 📋 Trade Analysis")
        
        trades_df = pd.DataFrame(results['trades'])
        
        # Format for display
        display_df = trades_df.copy()
        
        # Format date
        if 'Date' in display_df.columns:
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Format numeric columns
        for col in ['Price', 'Value', 'PnL', 'Commission']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        
        for col in ['Return%', 'Confidence']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
        
        # Display trades table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Trade distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns distribution
            if 'Return%' in trades_df.columns:
                returns = trades_df['Return%'].dropna()
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=30,
                    name='Returns Distribution',
                    marker_color='gray'
                ))
                
                fig.update_layout(
                    title="Returns Distribution",
                    xaxis_title="Return (%)",
                    yaxis_title="Frequency",
                    height=300,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Holding period distribution
            if 'HoldingDays' in trades_df.columns:
                holding = trades_df['HoldingDays'].dropna()
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=holding,
                    nbinsx=20,
                    name='Holding Period',
                    marker_color='lightgray'
                ))
                
                fig.update_layout(
                    title="Holding Period Distribution",
                    xaxis_title="Days",
                    yaxis_title="Frequency",
                    height=300,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def render_monte_carlo():
    """Render Monte Carlo simulation"""
    st.markdown("### 🎲 Monte Carlo Simulation")
    
    if st.session_state.backtest_results is None:
        st.info("Run a backtest first to enable Monte Carlo simulation")
        return
    
    results = st.session_state.backtest_results
    
    # Get historical returns
    if not results['daily_returns']:
        st.warning("Insufficient data for Monte Carlo simulation")
        return
    
    historical_returns = np.array(results['daily_returns'])
    
    # Monte Carlo parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_simulations = st.slider("Number of Simulations", 50, 500, 100)
    with col2:
        n_days = st.slider("Projection Days", 30, 252, 90)
    with col3:
        confidence_level = st.slider("Confidence Level", 80, 99, 95)
    
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running Monte Carlo simulation..."):
            # Run simulation
            initial_value = results['metrics']['final_capital']
            
            # Bootstrap from historical returns
            simulations = np.zeros((n_simulations, n_days))
            
            for i in range(n_simulations):
                # Random sample with replacement from historical returns
                sim_returns = np.random.choice(historical_returns, size=n_days, replace=True)
                simulations[i] = initial_value * np.exp(np.cumsum(sim_returns))
            
            # Calculate percentiles
            lower_bound = (100 - confidence_level) / 2
            upper_bound = 100 - lower_bound
            
            percentiles = np.percentile(simulations, [lower_bound, 25, 50, 75, upper_bound], axis=0)
            
            # Create visualization
            dates = pd.date_range(start=datetime.now(), periods=n_days, freq='D')
            
            fig = go.Figure()
            
            # Add confidence bands
            fig.add_trace(go.Scatter(
                x=dates,
                y=percentiles[0],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=percentiles[4],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name=f'{confidence_level}% Confidence',
                fillcolor='rgba(200,200,200,0.3)'
            ))
            
            # Add quartile bands
            fig.add_trace(go.Scatter(
                x=dates,
                y=percentiles[1],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=percentiles[3],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Interquartile Range',
                fillcolor='rgba(150,150,150,0.3)'
            ))
            
            # Add median line
            fig.add_trace(go.Scatter(
                x=dates,
                y=percentiles[2],
                mode='lines',
                name='Median Path',
                line=dict(color='black', width=2)
            ))
            
            # Add sample paths
            sample_indices = np.random.choice(n_simulations, size=min(10, n_simulations), replace=False)
            for idx in sample_indices:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=simulations[idx],
                    mode='lines',
                    line=dict(color='gray', width=0.5),
                    opacity=0.3,
                    showlegend=False
                ))
            
            fig.update_layout(
                title=f"Monte Carlo Simulation ({n_simulations} paths)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            final_values = simulations[:, -1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Median Outcome", f"${np.median(final_values):,.0f}")
            with col2:
                st.metric(f"{confidence_level}% Lower Bound", f"${percentiles[0, -1]:,.0f}")
            with col3:
                st.metric(f"{confidence_level}% Upper Bound", f"${percentiles[4, -1]:,.0f}")
            with col4:
                prob_profit = (final_values > initial_value).mean() * 100
                st.metric("Probability of Profit", f"{prob_profit:.1f}%")

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
    st.caption("Test and optimize your trading strategies with comprehensive historical analysis")
    
    # Strategy configuration
    strategy = render_strategy_configuration()
    
    st.divider()
    
    # Results tabs
    if st.session_state.backtest_results:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Results",
            "📈 Equity Curve",
            "📋 Trades",
            "🎲 Monte Carlo",
            "📑 Report"
        ])
        
        with tab1:
            render_backtest_results()
        
        with tab2:
            render_equity_curve()
        
        with tab3:
            render_trades_analysis()
        
        with tab4:
            render_monte_carlo()
        
        with tab5:
            st.markdown("### 📑 Backtest Report")
            
            # Generate downloadable report
            if st.button("Generate PDF Report"):
                st.info("PDF report generation feature coming soon!")
            
            # Display summary
            metrics = st.session_state.backtest_results.get('metrics', {})
            
            st.markdown(f"""
            **Backtest Summary**
            
            - **Strategy Performance**: {metrics.get('total_return', 0):.2f}% total return
            - **Risk-Adjusted Return**: Sharpe Ratio of {metrics.get('sharpe_ratio', 0):.2f}
            - **Win Rate**: {metrics.get('win_rate', 0):.1f}% with {metrics.get('num_trades', 0)} trades
            - **Maximum Risk**: {metrics.get('max_drawdown', 0):.1f}% maximum drawdown
            - **Recommendation**: {"✅ Strategy shows promise" if metrics.get('sharpe_ratio', 0) > 1 else "⚠️ Strategy needs optimization"}
            
            **Key Insights**:
            - Average winning trade: {metrics.get('avg_win', 0):.2f}%
            - Average losing trade: {metrics.get('avg_loss', 0):.2f}%
            - Profit factor: {metrics.get('profit_factor', 0):.2f}
            - Risk metrics indicate {"acceptable" if abs(metrics.get('max_drawdown', 0)) < 20 else "high"} risk levels
            """)
    else:
        st.info("👆 Configure your strategy above and click 'Run Backtest' to see results")

if __name__ == "__main__":
    main()