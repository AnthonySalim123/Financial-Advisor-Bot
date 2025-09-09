# utils/data_processor.py
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

# Add sentiment analyzer import
try:
    from utils.sentiment_analyzer import get_sentiment_analyzer
    SENTIMENT_AVAILABLE = True
    logger.info("Sentiment analyzer available")
except:
    SENTIMENT_AVAILABLE = False
    logger.warning("Sentiment analyzer not available")

class StockDataProcessor:
    """Main class for processing stock market data"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.cache = {}
        self.last_update = {}
        self.sentiment_cache = {}
        
        # Initialize sentiment analyzer
        if SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = get_sentiment_analyzer()
                logger.info("✅ Sentiment analyzer initialized")
            except Exception as e:
                logger.warning(f"Sentiment analyzer initialization failed: {e}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
        
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
    
    def fetch_stock_data_with_sentiment(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """
        Fetch stock data with sentiment features integrated
        
        Args:
            symbol: Stock ticker symbol
            period: Time period
            
        Returns:
            DataFrame with OHLCV data and sentiment features
        """
        # Get base stock data
        df = self.fetch_stock_data(symbol, period)
        
        if df.empty:
            return df
        
        # Add sentiment features if available
        if self.sentiment_analyzer:
            try:
                sentiment_features = self.sentiment_analyzer.get_sentiment_features(symbol)
                
                # Add sentiment features as constant columns (latest sentiment applied to all rows)
                for feature, value in sentiment_features.items():
                    df[f'sentiment_{feature}'] = value
                
                logger.info(f"Added sentiment features for {symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to add sentiment features for {symbol}: {e}")
                # Add neutral sentiment features as fallback
                neutral_features = {
                    'sentiment_score': 0.0,
                    'sentiment_confidence': 0.0,
                    'news_sentiment': 0.0,
                    'news_confidence': 0.0,
                    'news_articles_count': 0,
                    'market_sentiment': 0.0,
                    'price_position': 0.5,
                    'volume_trend': 0.0,
                    'momentum_5d': 0.0,
                    'momentum_20d': 0.0
                }
                
                for feature, value in neutral_features.items():
                    df[f'sentiment_{feature}'] = value
        
        return df
    
    def fetch_sentiment_analysis(self, symbol: str, use_cache: bool = True) -> Dict:
        """
        Fetch comprehensive sentiment analysis for a symbol
        
        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.sentiment_analyzer:
            return {'error': 'Sentiment analyzer not available'}
        
        try:
            # Check cache
            if use_cache and symbol in self.sentiment_cache:
                cache_entry = self.sentiment_cache[symbol]
                cache_time = datetime.fromisoformat(cache_entry['timestamp'])
                
                # Use cache if less than 30 minutes old
                if datetime.now() - cache_time < timedelta(minutes=30):
                    logger.info(f"Using cached sentiment for {symbol}")
                    return cache_entry['data']
            
            # Get fresh sentiment analysis
            sentiment_data = self.sentiment_analyzer.get_comprehensive_sentiment(symbol)
            
            # Cache the result
            self.sentiment_cache[symbol] = {
                'data': sentiment_data,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Fetched fresh sentiment analysis for {symbol}")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return {'error': str(e)}
    
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
                        'JNJ': 160, 'PFE': 45, 'UNH': 500
                    }
                    
                    base_price = base_prices.get(symbol, 100)
                    change_pct = np.random.uniform(-3, 3)
                    current_price = base_price * (1 + change_pct/100)
                    
                    quotes_data.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'Change': current_price - base_price,
                        'Change%': change_pct,
                        'Volume': np.random.randint(1000000, 10000000),
                        'Prev_Close': base_price,
                        'Day_High': current_price * 1.02,
                        'Day_Low': current_price * 0.98
                    })
        
        return pd.DataFrame(quotes_data)
    
    def get_market_overview(self) -> Dict:
        """
        Get market overview with major indices
        
        Returns:
            Dictionary with market data
        """
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX'
        }
        
        market_data = {}
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                
                if not hist.empty and len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = current - previous
                    change_pct = (change / previous) * 100 if previous != 0 else 0
                    
                    market_data[symbol] = {
                        'name': name,
                        'price': current,
                        'change': change,
                        'change_pct': change_pct
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                # Fallback with synthetic data
                market_data[symbol] = {
                    'name': name,
                    'price': 4500.0 if symbol == '^GSPC' else 35000.0 if symbol == '^DJI' else 14000.0 if symbol == '^IXIC' else 20.0,
                    'change': np.random.uniform(-50, 50),
                    'change_pct': np.random.uniform(-2, 2)
                }
        
        return market_data
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.last_update.clear()
        self.sentiment_cache.clear()
        logger.info("Cache cleared")


# Singleton instance
_data_processor_instance = None

def get_data_processor() -> StockDataProcessor:
    """Get singleton data processor instance"""
    global _data_processor_instance
    if _data_processor_instance is None:
        _data_processor_instance = StockDataProcessor()
    return _data_processor_instance

# Print status on import
logger.info("Data processor module loaded successfully")