"""
Data Processing Module
Handles all data fetching, caching, and preprocessing for the StockBot Advisor
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Tuple, Optional
import yaml
import json
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add fallback import
try:
    from utils.fallback_data import generate_synthetic_stock_data, get_synthetic_info
    FALLBACK_AVAILABLE = True
    logger.info("Fallback data module available")
except:
    FALLBACK_AVAILABLE = False
    logger.warning("Fallback data module not available")

class StockDataProcessor:
    """Main class for processing stock market data"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.cache = {}
        self.last_update = {}
        
    def _load_config(self, config_path):
        """Load configuration from yaml file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def get_stock_list(self) -> List[str]:
        """Get complete list of stocks from configuration"""
        stocks = []
        
        # Get stocks from each sector
        for sector in ['technology', 'financial', 'healthcare']:
            if sector in self.config.get('stocks', {}):
                stocks.extend([s['symbol'] for s in self.config['stocks'][sector]])
        
        # Add benchmarks
        if 'benchmarks' in self.config.get('stocks', {}):
            stocks.extend([b['symbol'] for b in self.config['stocks']['benchmarks']])
        
        return stocks
    
    def fetch_stock_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance with automatic fallback to synthetic data
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self.cache:
                last_update = self.last_update.get(cache_key, datetime.min)
                if datetime.now() - last_update < timedelta(minutes=5):
                    logger.info(f"Using cached data for {symbol}")
                    return self.cache[cache_key]
            
            # Try fetching from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Check if data is valid
            if not data.empty:
                # Process successful data
                data = data.round(2)
                data.index = pd.to_datetime(data.index)
                
                # Add additional columns
                data['Returns'] = data['Close'].pct_change()
                data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
                data['High_Low_Spread'] = data['High'] - data['Low']
                data['Close_Open_Spread'] = data['Close'] - data['Open']
                
                # Cache the data
                self.cache[cache_key] = data
                self.last_update[cache_key] = datetime.now()
                
                logger.info(f"Successfully fetched {symbol} from yfinance")
                return data
            else:
                raise ValueError("Empty data from yfinance")
                
        except Exception as e:
            logger.warning(f"YFinance failed for {symbol}: {e}")
            
            # USE FALLBACK SYNTHETIC DATA
            if FALLBACK_AVAILABLE:
                logger.info(f"Using synthetic fallback data for {symbol}")
                from utils.fallback_data import generate_synthetic_stock_data
                
                data = generate_synthetic_stock_data(symbol, period)
                
                # Add calculated columns
                data['Returns'] = data['Close'].pct_change()
                data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
                data['High_Low_Spread'] = data['High'] - data['Low']
                data['Close_Open_Spread'] = data['Close'] - data['Open']
                
                # Cache the data
                self.cache[cache_key] = data
                self.last_update[cache_key] = datetime.now()
                
                return data
            else:
                logger.error(f"No data available for {symbol} and fallback not available")
                return pd.DataFrame()
    
    def fetch_stock_info(self, symbol: str) -> Dict:
        """
        Fetch stock information and fundamentals with fallback
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid info
            if info and 'symbol' in info:
                # Extract key metrics
                metrics = {
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'roa': info.get('returnOnAssets', 0),
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'earnings_growth': info.get('earningsGrowth', 0),
                    'current_ratio': info.get('currentRatio', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'free_cash_flow': info.get('freeCashflow', 0),
                    'beta': info.get('beta', 1),
                    '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                    '52_week_low': info.get('fiftyTwoWeekLow', 0),
                    'average_volume': info.get('averageVolume', 0),
                    'current_price': info.get('currentPrice', 0),
                    'target_price': info.get('targetMeanPrice', 0),
                    'recommendation': info.get('recommendationKey', 'none'),
                }
                
                logger.info(f"Successfully fetched info for {symbol} from yfinance")
                return metrics
                
        except Exception as e:
            logger.warning(f"YFinance info failed for {symbol}: {e}")
        
        # USE FALLBACK
        if FALLBACK_AVAILABLE:
            from utils.fallback_data import get_synthetic_info
            logger.info(f"Using synthetic info for {symbol}")
            info = get_synthetic_info(symbol)
            
            # Return synthetic info with all fields
            return {
                'symbol': symbol,
                'name': info.get('longName', f'{symbol} Corporation'),
                'sector': info.get('sector', 'Technology'),
                'industry': info.get('industry', 'Software'),
                'market_cap': info.get('marketCap', 500000000000),
                'pe_ratio': info.get('trailingPE', 25.0),
                'forward_pe': info.get('forwardPE', 22.0),
                'peg_ratio': info.get('pegRatio', 1.5),
                'price_to_book': info.get('priceToBook', 5.0),
                'dividend_yield': info.get('dividendYield', 0.01),
                'profit_margin': info.get('profitMargins', 0.20),
                'operating_margin': info.get('operatingMargins', 0.25),
                'roe': info.get('returnOnEquity', 0.30),
                'roa': info.get('returnOnAssets', 0.15),
                'revenue_growth': info.get('revenueGrowth', 0.15),
                'earnings_growth': info.get('earningsGrowth', 0.20),
                'current_ratio': info.get('currentRatio', 2.0),
                'debt_to_equity': info.get('debtToEquity', 0.5),
                'free_cash_flow': info.get('freeCashflow', 10000000000),
                'beta': info.get('beta', 1.0),
                '52_week_high': 200,
                '52_week_low': 100,
                'average_volume': 50000000,
                'current_price': 150,
                'target_price': 175,
                'recommendation': 'buy',
            }
        
        return {'symbol': symbol, 'error': 'No data available'}
    
    def fetch_batch_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch real-time quotes for multiple symbols with fallback
        
        Args:
            symbols: List of stock ticker symbols
        
        Returns:
            DataFrame with current quotes
        """
        quotes_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    volume = hist['Volume'].iloc[-1]
                    
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                    
                    quotes_data.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'Change': change,
                        'Change%': change_pct,
                        'Volume': volume,
                        'Prev_Close': prev_close,
                        'Day_High': hist['High'].iloc[-1],
                        'Day_Low': hist['Low'].iloc[-1]
                    })
                    logger.info(f"Fetched quote for {symbol} from yfinance")
                else:
                    raise ValueError("Empty or insufficient data")
                    
            except Exception as e:
                logger.warning(f"YFinance quote failed for {symbol}: {e}")
                
                # USE FALLBACK
                if FALLBACK_AVAILABLE:
                    logger.info(f"Using synthetic quote for {symbol}")
                    
                    # Generate synthetic quote
                    base_prices = {
                        'AAPL': 150, 'MSFT': 350, 'GOOGL': 140,
                        'NVDA': 450, 'AMZN': 170, 'META': 350,
                        'JPM': 150, 'BAC': 35, 'GS': 350,
                        'JNJ': 160, 'PFE': 30, 'MRNA': 100
                    }
                    
                    base_price = base_prices.get(symbol, 100)
                    current_price = base_price * (1 + np.random.uniform(-0.02, 0.02))
                    prev_close = base_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    quotes_data.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'Change': change,
                        'Change%': change_pct,
                        'Volume': int(50000000 * (1 + np.random.uniform(-0.5, 0.5))),
                        'Prev_Close': prev_close,
                        'Day_High': current_price * 1.01,
                        'Day_Low': current_price * 0.99
                    })
        
        return pd.DataFrame(quotes_data)
    
    def calculate_returns(self, data: pd.DataFrame, periods: List[int] = [1, 5, 20, 60, 252]) -> pd.DataFrame:
        """
        Calculate returns for different periods
        
        Args:
            data: DataFrame with price data
            periods: List of periods for return calculation
        
        Returns:
            DataFrame with returns
        """
        returns_df = pd.DataFrame(index=data.index)
        
        for period in periods:
            returns_df[f'Return_{period}D'] = data['Close'].pct_change(period)
        
        # Add cumulative returns
        returns_df['Cumulative_Return'] = (1 + data['Close'].pct_change()).cumprod() - 1
        
        return returns_df
    
    def calculate_risk_metrics(self, data: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate risk metrics for the stock
        
        Args:
            data: DataFrame with price data
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Dictionary with risk metrics
        """
        daily_returns = data['Close'].pct_change().dropna()
        
        # Calculate metrics
        metrics = {
            'volatility_daily': daily_returns.std(),
            'volatility_annual': daily_returns.std() * np.sqrt(252),
            'sharpe_ratio': (daily_returns.mean() - risk_free_rate/252) / daily_returns.std() * np.sqrt(252),
            'sortino_ratio': self._calculate_sortino_ratio(daily_returns, risk_free_rate),
            'max_drawdown': self._calculate_max_drawdown(data['Close']),
            'var_95': daily_returns.quantile(0.05),
            'cvar_95': daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean(),
            'skewness': daily_returns.skew(),
            'kurtosis': daily_returns.kurtosis()
        }
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0
        
        excess_return = returns.mean() - risk_free_rate/252
        return excess_return / downside_std * np.sqrt(252)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def get_market_overview(self) -> pd.DataFrame:
        """
        Get overview of major market indices with fallback
        
        Returns:
            DataFrame with market indices data
        """
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX',
            'GLD': 'Gold',
            'TLT': 'Bonds',
            'DX-Y.NYB': 'US Dollar',
            'CL=F': 'Crude Oil'
        }
        
        market_data = []
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                
                if not hist.empty and len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change_pct = ((current - prev) / prev) * 100 if prev != 0 else 0
                    
                    market_data.append({
                        'Index': name,
                        'Symbol': symbol,
                        'Value': current,
                        'Change%': change_pct
                    })
                    logger.info(f"Fetched {name} from yfinance")
                else:
                    raise ValueError("Empty data")
                    
            except Exception as e:
                logger.warning(f"YFinance failed for {name}: {e}")
                
                # USE FALLBACK - Generate synthetic market data
                if FALLBACK_AVAILABLE:
                    logger.info(f"Using synthetic data for {name}")
                    
                    # Synthetic values for indices
                    base_values = {
                        'S&P 500': 4500,
                        'Dow Jones': 35000,
                        'NASDAQ': 15000,
                        'VIX': 18,
                        'Gold': 1800,
                        'Bonds': 140,
                        'US Dollar': 95,
                        'Crude Oil': 75
                    }
                    
                    base_value = base_values.get(name, 100)
                    current = base_value * (1 + np.random.uniform(-0.02, 0.02))
                    change_pct = np.random.uniform(-2, 2)
                    
                    market_data.append({
                        'Index': name,
                        'Symbol': symbol,
                        'Value': current,
                        'Change%': change_pct
                    })
        
        return pd.DataFrame(market_data)
    
    def get_sector_performance(self) -> pd.DataFrame:
        """
        Get sector performance data with fallback
        
        Returns:
            DataFrame with sector performance
        """
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLC': 'Communication Services'
        }
        
        sector_data = []
        
        for symbol, sector in sector_etfs.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                
                if not hist.empty and len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    week_ago = hist['Close'].iloc[0]
                    change_pct = ((current - week_ago) / week_ago) * 100 if week_ago != 0 else 0
                    
                    sector_data.append({
                        'Sector': sector,
                        'ETF': symbol,
                        'Weekly_Change%': change_pct
                    })
                    logger.info(f"Fetched {sector} from yfinance")
                else:
                    raise ValueError("Empty data")
                    
            except Exception as e:
                logger.warning(f"YFinance failed for {sector}: {e}")
                
                # USE FALLBACK
                if FALLBACK_AVAILABLE:
                    logger.info(f"Using synthetic data for {sector}")
                    
                    # Generate synthetic sector performance
                    change_pct = np.random.uniform(-5, 5)
                    
                    sector_data.append({
                        'Sector': sector,
                        'ETF': symbol,
                        'Weekly_Change%': change_pct
                    })
        
        return pd.DataFrame(sector_data)
    
    def export_data(self, data: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Export data to file
        
        Args:
            data: DataFrame to export
            filename: Output filename
            format: Export format (csv, excel, json)
        """
        try:
            if format == 'csv':
                data.to_csv(filename)
            elif format == 'excel':
                data.to_excel(filename, engine='openpyxl')
            elif format == 'json':
                data.to_json(filename, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False

# Singleton instance
@st.cache_resource
def get_data_processor():
    """Get or create data processor instance"""
    return StockDataProcessor()