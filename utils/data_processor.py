# utils/data_processor.py
"""
Data Processing Module - Updated with 33 Stocks Support
Handles all data fetching, caching, and preprocessing for the StockBot Advisor
Includes support for Real Estate REITs and expanded Financial stocks
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
    # Add this method to your StockDataProcessor class in utils/data_processor.py

    def fetch_stock_data(self, symbol: str, period: str = '1y', 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock data from yfinance with timezone handling
        FIXED VERSION - Removes timezone info for consistent handling
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}_{start_date}_{end_date}"
            if cache_key in self.cache:
                last_update = self.last_update.get(cache_key, datetime.min)
                if datetime.now() - last_update < timedelta(minutes=5):
                    return self.cache[cache_key]
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                df = ticker.history(period=period, auto_adjust=True)
            
            # FIX: Remove timezone information to avoid comparison issues
            # This is the simplest solution that works consistently
            if not df.empty and hasattr(df.index, 'tz'):
                if df.index.tz is not None:
                    # Convert to UTC then remove timezone
                    df.index = df.index.tz_convert('UTC').tz_localize(None)
            
            # Cache the result
            self.cache[cache_key] = df
            self.last_update[cache_key] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            
            # Try fallback data if available
            if FALLBACK_AVAILABLE:
                logger.info(f"Using fallback data for {symbol}")
                return generate_synthetic_stock_data(symbol, period, start_date, end_date)
            
            return pd.DataFrame()
        
    """Main class for processing stock market data - Enhanced with 33 stocks"""
    
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
        """Get complete list of stocks from configuration - INCLUDING REAL ESTATE"""
        stocks = []
        
        # Get stocks from each sector - UPDATED to include real_estate
        for sector in ['technology', 'financial', 'real_estate', 'healthcare']:
            if sector in self.config.get('stocks', {}):
                sector_stocks = self.config['stocks'][sector]
                # Handle both dict and string formats in config
                for stock_item in sector_stocks:
                    if isinstance(stock_item, dict) and 'symbol' in stock_item:
                        stocks.append(stock_item['symbol'])
                    elif isinstance(stock_item, str):
                        stocks.append(stock_item)
        
        # Add benchmarks
        if 'benchmarks' in self.config.get('stocks', {}):
            benchmark_items = self.config['stocks']['benchmarks']
            for benchmark in benchmark_items:
                if isinstance(benchmark, dict) and 'symbol' in benchmark:
                    stocks.append(benchmark['symbol'])
                elif isinstance(benchmark, str):
                    stocks.append(benchmark)
        
        return stocks
    
    def get_stocks_by_sector(self, sector: str) -> List[str]:
        """Get stocks for a specific sector"""
        stocks = []
        
        if sector in self.config.get('stocks', {}):
            sector_stocks = self.config['stocks'][sector]
            for stock_item in sector_stocks:
                if isinstance(stock_item, dict) and 'symbol' in stock_item:
                    stocks.append(stock_item['symbol'])
                elif isinstance(stock_item, str):
                    stocks.append(stock_item)
        
        return stocks
    
    def calculate_sector_momentum(self, sector: str, period: int = 20) -> float:
        """
        Calculate sector momentum score
        
        Args:
            sector: Sector name
            period: Lookback period in days
        
        Returns:
            Momentum score (-100 to 100)
        """
        try:
            stocks = self.get_stocks_by_sector(sector)
            if not stocks:
                return 0
            
            momentum_scores = []
            
            for symbol in stocks:
                df = self.fetch_stock_data(symbol, period='2mo')
                if not df.empty and len(df) >= period:
                    # Calculate momentum as percentage change
                    momentum = ((df['Close'].iloc[-1] / df['Close'].iloc[-period]) - 1) * 100
                    momentum_scores.append(momentum)
            
            if momentum_scores:
                return round(np.mean(momentum_scores), 2)
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating sector momentum: {e}")
            return 0
    
    def fetch_reit_specific_data(self, symbol: str) -> Dict:
        """
        Fetch REIT-specific metrics
        
        Args:
            symbol: REIT ticker symbol
            
        Returns:
            Dictionary with REIT metrics
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get REIT-specific metrics
            dividend_yield = info.get('dividendYield', 0)
            # Fix for excessive dividend yield (if it's already in percentage)
            if dividend_yield > 1:
                dividend_yield = dividend_yield
            else:
                dividend_yield = dividend_yield * 100
            
            reit_metrics = {
                'symbol': symbol,
                'dividend_yield': dividend_yield,
                'payout_ratio': info.get('payoutRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'market_cap': info.get('marketCap', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'beta': info.get('beta', 1),
                'trailing_pe': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'reit_sector': self._classify_reit_sector(symbol),
                'rate_sensitivity': self._calculate_rate_sensitivity(symbol, info.get('beta', 1))
            }
            
            return reit_metrics
            
        except Exception as e:
            logger.error(f"Error fetching REIT data for {symbol}: {e}")
            return {}
    
    def _classify_reit_sector(self, symbol: str) -> str:
        """Classify REIT by sector"""
        reit_sectors = {
            'PLD': 'Industrial/Logistics',
            'AMT': 'Infrastructure/Towers',
            'EQIX': 'Data Centers',
            'SPG': 'Retail/Malls',
            'O': 'Retail/Triple-Net-Lease'
        }
        return reit_sectors.get(symbol, 'Diversified')
    
    def _calculate_rate_sensitivity(self, symbol: str, beta: float) -> str:
        """Calculate interest rate sensitivity for REITs"""
        if beta > 1.2:
            return "High"
        elif beta > 0.8:
            return "Moderate"
        else:
            return "Low"
    
    def fetch_financial_sector_data(self, symbol: str) -> Dict:
        """
        Fetch financial sector specific metrics
        
        Args:
            symbol: Financial stock ticker
            
        Returns:
            Dictionary with financial metrics
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Determine if it's a bank
            is_bank = symbol in ['JPM', 'BAC', 'GS', 'MS', 'WFC']
            
            financial_metrics = {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
                'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 1),
                'book_value': info.get('bookValue', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
            }
            
            # Bank-specific metrics
            if is_bank:
                financial_metrics.update({
                    'is_bank': True,
                    'tier1_capital_ratio': self._estimate_tier1_ratio(symbol),
                    'efficiency_ratio': self._calculate_efficiency_ratio(info)
                })
            
            # Payment processors specific (V, MA, AXP)
            if symbol in ['V', 'MA', 'AXP']:
                financial_metrics.update({
                    'payment_processor': True,
                    'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
                    'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
                })
            
            return financial_metrics
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {e}")
            return {}
    
    def _estimate_tier1_ratio(self, symbol: str) -> float:
        """Estimate Tier 1 Capital Ratio for banks"""
        tier1_estimates = {
            'JPM': 14.3,
            'BAC': 13.8,
            'WFC': 12.1,
            'GS': 14.7,
            'MS': 15.2
        }
        return tier1_estimates.get(symbol, 13.0)
    
    def _calculate_efficiency_ratio(self, info: Dict) -> float:
        """Calculate efficiency ratio for banks"""
        operating_margin = info.get('operatingMargins', 0)
        if operating_margin > 0:
            return round((1 - operating_margin) * 100, 2)
        return 0
    
    def get_sector_summary(self) -> Dict:
        """Get summary statistics for all sectors"""
        summary = {}
        
        for sector in ['technology', 'financial', 'real_estate', 'healthcare']:
            stocks = self.get_stocks_by_sector(sector)
            
            if stocks:
                sector_data = {
                    'stock_count': len(stocks),
                    'stocks': stocks,
                    'momentum': self.calculate_sector_momentum(sector)
                }
                summary[sector] = sector_data
        
        return summary
    
    # ============= KEEP ALL YOUR EXISTING METHODS BELOW =============
    
    def fetch_stock_data(self, symbol: str, period: str = '3y', interval: str = '1d') -> pd.DataFrame:
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
            
            if info:
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
                    '52_week_high': info.get('fiftyTwoWeekHigh', 200),
                    '52_week_low': info.get('fiftyTwoWeekLow', 100),
                    'average_volume': info.get('averageVolume', 50000000),
                    'current_price': info.get('currentPrice', 150),
                    'target_price': info.get('targetMeanPrice', 175),
                    'recommendation': info.get('recommendationKey', 'buy'),
                }
        except:
            pass
        
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
                        'Change %': change_pct,
                        'Volume': volume
                    })
                else:
                    # Fallback
                    quotes_data.append({
                        'Symbol': symbol,
                        'Price': 100.0,
                        'Change': 0.0,
                        'Change %': 0.0,
                        'Volume': 1000000
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to fetch quote for {symbol}: {e}")
                # Fallback
                quotes_data.append({
                    'Symbol': symbol,
                    'Price': 100.0,
                    'Change': 0.0,
                    'Change %': 0.0,
                    'Volume': 1000000
                })
        
        return pd.DataFrame(quotes_data)
    
    def get_market_overview(self) -> Dict:
        """
        Get market overview data
        
        Returns:
            Dictionary with market indices data
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
    
    def get_sector_performance(self) -> Dict:
        """
        Get sector performance data - UPDATED with real_estate
        
        Returns:
            Dictionary with sector performance data
        """
        try:
            # Use sectors from config
            sector_performance = {}
            
            for sector in ['technology', 'financial', 'real_estate', 'healthcare']:
                stocks = self.get_stocks_by_sector(sector)
                
                if stocks:
                    sector_returns = []
                    for symbol in stocks[:3]:  # Sample first 3 stocks for efficiency
                        try:
                            ticker = yf.Ticker(symbol)
                            hist = ticker.history(period='1mo')
                            
                            if not hist.empty and len(hist) >= 2:
                                current_price = hist['Close'].iloc[-1]
                                start_price = hist['Close'].iloc[0]
                                monthly_return = (current_price / start_price - 1) * 100
                                sector_returns.append(monthly_return)
                        except:
                            continue
                    
                    if sector_returns:
                        avg_return = np.mean(sector_returns)
                        sector_performance[sector.capitalize()] = {
                            'return': avg_return,
                            'count': len(stocks)
                        }
            
            return sector_performance
            
        except Exception as e:
            logger.error(f"Error calculating sector performance: {e}")
            return {}

# Singleton instance
_data_processor_instance = None

def get_data_processor() -> StockDataProcessor:
    """Get singleton data processor instance"""
    global _data_processor_instance
    if _data_processor_instance is None:
        _data_processor_instance = StockDataProcessor()
    return _data_processor_instance

# Alias for compatibility with enhanced_data_processor imports
def get_enhanced_data_processor() -> StockDataProcessor:
    """Get enhanced data processor (alias for compatibility)"""
    return get_data_processor()

# Print status on import
logger.info("Data processor module loaded successfully with 33 stocks support")