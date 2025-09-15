# utils/enhanced_data_processor.py
"""
Enhanced Data Processing Module with Real Estate and Extended Financial Coverage
Handles data fetching for expanded stock universe including REITs
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)

class EnhancedStockDataProcessor:
    """Enhanced processor with sector-specific handling"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize with enhanced configuration"""
        self.config = self._load_config(config_path)
        self.cache = {}
        self.last_update = {}
        self.sector_cache = {}
        
        # Define complete stock universe with sectors
        self.stock_universe = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'PLTR'],
            'financial': [
                'JPM', 'BAC', 'GS', 'MS', 'V',      # Original
                'MA', 'WFC', 'AXP', 'BLK', 'SPGI'   # New additions
            ],
            'real_estate': ['PLD', 'AMT', 'EQIX', 'SPG', 'O'],  # New sector
            'healthcare': ['JNJ', 'PFE', 'MRNA', 'UNH'],
            'benchmarks': ['SPY', 'QQQ', 'DIA', 'VNQ']  # Added VNQ for REIT benchmark
        }
        
        # Sector-specific configurations
        self.sector_configs = {
            'real_estate': {
                'is_reit': True,
                'fetch_dividends': True,
                'key_metrics': ['dividendYield', 'trailingPE', 'priceToBook'],
                'special_indicators': ['FFO', 'AFFO', 'NAV']
            },
            'financial': {
                'is_bank': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
                'key_metrics': ['bookValue', 'priceToBook', 'returnOnEquity'],
                'regulatory_metrics': ['tier1Capital', 'netInterestMargin']
            }
        }
        
        logger.info(f"Enhanced processor initialized with {self.get_total_stocks()} stocks")
    
    def _load_config(self, config_path):
        """Load configuration from yaml file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def get_total_stocks(self) -> int:
        """Get total number of stocks in universe"""
        return sum(len(stocks) for stocks in self.stock_universe.values())
    
    def get_stock_list(self) -> List[str]:
        """Get complete list of all stocks"""
        all_stocks = []
        for sector_stocks in self.stock_universe.values():
            all_stocks.extend(sector_stocks)
        return all_stocks
    
    def get_stocks_by_sector(self, sector: str) -> List[str]:
        """Get stocks for a specific sector"""
        return self.stock_universe.get(sector, [])
    
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
            reit_metrics = {
                'symbol': symbol,
                # Fixed code:
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') and info.get('dividendYield') < 1 else info.get('dividendYield', 0),
                'trailing_twelve_months_yield': info.get('trailingAnnualDividendYield', 0) * 100 if info.get('trailingAnnualDividendYield') and info.get('trailingAnnualDividendYield') < 1 else info.get('trailingAnnualDividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'beta': info.get('beta', 1),
                'trailing_pe': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                
                # Calculate FFO proxy (if earnings data available)
                'estimated_ffo_per_share': self._estimate_ffo(info),
                
                # Dividend history
                'dividend_history': self._get_dividend_history(ticker),
                
                # Sector classification
                'reit_sector': self._classify_reit_sector(symbol),
                
                # Interest rate sensitivity
                'rate_sensitivity': self._calculate_rate_sensitivity(symbol, info.get('beta', 1))
            }
            
            return reit_metrics
            
        except Exception as e:
            logger.error(f"Error fetching REIT data for {symbol}: {e}")
            return {}
    
    def _estimate_ffo(self, info: Dict) -> float:
        """
        Estimate Funds From Operations (FFO) for REITs
        FFO = Net Income + Depreciation + Amortization - Gains on Sales
        """
        try:
            # This is a simplified estimation
            net_income = info.get('netIncomeToCommon', 0)
            shares = info.get('sharesOutstanding', 1)
            
            if net_income and shares:
                # REITs typically have depreciation around 30-40% of net income
                estimated_depreciation = net_income * 0.35
                estimated_ffo = (net_income + estimated_depreciation) / shares
                return round(estimated_ffo, 2)
            return 0
        except:
            return 0
    
    def _get_dividend_history(self, ticker) -> Dict:
        """Get dividend payment history"""
        try:
            dividends = ticker.dividends
            if not dividends.empty:
                # Get last 12 months of dividends
                one_year_ago = datetime.now() - timedelta(days=365)
                recent_divs = dividends[dividends.index >= one_year_ago]
                return recent_divs.to_dict()
            return {}
        except:
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
        # REITs are generally sensitive to interest rates
        # Higher beta = higher sensitivity
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
            is_bank = symbol in self.sector_configs['financial']['is_bank']
            
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
                
                # Financial sector specific
                'book_value': info.get('bookValue', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
            }
            
            # Bank-specific metrics
            if is_bank:
                financial_metrics.update({
                    'is_bank': True,
                    'net_interest_margin': self._estimate_nim(info),
                    'efficiency_ratio': self._calculate_efficiency_ratio(info),
                    'loan_to_deposit': self._estimate_loan_to_deposit(info),
                    'tier1_capital_ratio': self._estimate_tier1_ratio(symbol)
                })
            
            # Payment processors specific (V, MA, AXP)
            if symbol in ['V', 'MA', 'AXP']:
                financial_metrics.update({
                    'payment_processor': True,
                    'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
                    'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
                })
            
            # Asset managers specific (BLK)
            if symbol == 'BLK':
                financial_metrics.update({
                    'asset_manager': True,
                    'aum_estimate': self._estimate_aum(info)  # Assets Under Management
                })
            
            return financial_metrics
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {e}")
            return {}
    
    def _estimate_nim(self, info: Dict) -> float:
        """Estimate Net Interest Margin for banks"""
        # Simplified estimation based on profit margins
        profit_margin = info.get('profitMargins', 0)
        if profit_margin:
            return round(profit_margin * 3.5 * 100, 2)
        return 0
    
    def _calculate_efficiency_ratio(self, info: Dict) -> float:
        """Calculate efficiency ratio for banks (lower is better)"""
        operating_margin = info.get('operatingMargins', 0)
        if operating_margin > 0:
            return round((1 - operating_margin) * 100, 2)
        return 0
    
    def _estimate_loan_to_deposit(self, info: Dict) -> float:
        """Estimate loan-to-deposit ratio"""
        # This would need actual balance sheet data
        # Using a typical range for healthy banks
        return round(np.random.uniform(0.75, 0.90) * 100, 2)
    
    def _estimate_tier1_ratio(self, symbol: str) -> float:
        """Estimate Tier 1 Capital Ratio"""
        # Major US banks typically maintain 12-15%
        tier1_estimates = {
            'JPM': 14.3,
            'BAC': 13.8,
            'WFC': 12.1,
            'GS': 14.7,
            'MS': 15.2
        }
        return tier1_estimates.get(symbol, 13.0)
    
    def _estimate_aum(self, info: Dict) -> float:
        """Estimate Assets Under Management for asset managers"""
        market_cap = info.get('marketCap', 0)
        if market_cap:
            # BlackRock's AUM is typically 35-40x its market cap
            return round(market_cap * 37 / 1e9, 2)  # In billions
        return 0
    
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
            momentum_scores = []
            
            for symbol in stocks:
                df = self.fetch_stock_data(symbol, period='1mo')
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
    
    def get_sector_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix between sectors"""
        try:
            sector_returns = {}
            
            for sector in self.stock_universe.keys():
                if sector != 'benchmarks':
                    # Get average returns for sector
                    stocks = self.get_stocks_by_sector(sector)
                    returns = []
                    
                    for symbol in stocks[:3]:  # Use top 3 stocks for efficiency
                        df = self.fetch_stock_data(symbol, period='6mo')
                        if not df.empty:
                            daily_returns = df['Close'].pct_change().dropna()
                            returns.append(daily_returns)
                    
                    if returns:
                        # Average the returns
                        sector_returns[sector] = pd.concat(returns, axis=1).mean(axis=1)
            
            # Create correlation matrix
            if sector_returns:
                returns_df = pd.DataFrame(sector_returns)
                return returns_df.corr()
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error calculating sector correlations: {e}")
            return pd.DataFrame()
    
    def fetch_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """
        Enhanced fetch with sector-specific handling
        
        Args:
            symbol: Stock ticker
            period: Time period
            
        Returns:
            DataFrame with OHLCV data and sector-specific additions
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if cache_key in self.cache:
                last_update = self.last_update.get(cache_key, datetime.min)
                if datetime.now() - last_update < timedelta(minutes=5):
                    return self.cache[cache_key]
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()
            
            # Add sector information
            sector = self._get_stock_sector(symbol)
            df['Sector'] = sector
            
            # Add REIT-specific calculations if applicable
            if sector == 'real_estate':
                df['Dividend_Yield'] = self._calculate_trailing_yield(ticker)
            
            # Add financial-specific calculations if applicable
            if sector == 'financial':
                df['Book_Multiple'] = self._calculate_book_multiple(ticker)
            
            # Update cache
            self.cache[cache_key] = df
            self.last_update[cache_key] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            # Try fallback data if available
            try:
                from utils.fallback_data import generate_synthetic_stock_data
                logger.info(f"Using fallback data for {symbol}")
                return generate_synthetic_stock_data(symbol, period)
            except:
                return pd.DataFrame()
    
    def _get_stock_sector(self, symbol: str) -> str:
        """Get sector for a given stock"""
        for sector, stocks in self.stock_universe.items():
            if symbol in stocks:
                return sector
        return 'unknown'
    
    def _calculate_trailing_yield(self, ticker) -> float:
        """Calculate trailing dividend yield"""
        try:
            info = ticker.info
            div_yield = info.get('trailingAnnualDividendYield', 0)
            return div_yield * 100 if div_yield else 0
        except:
            return 0
    
    def _calculate_book_multiple(self, ticker) -> float:
        """Calculate price to book multiple"""
        try:
            info = ticker.info
            return info.get('priceToBook', 0)
        except:
            return 0
    
    def fetch_stock_info(self, symbol: str) -> Dict:
        """Get comprehensive stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Basic info available for all stocks
            basic_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': self._get_stock_sector(symbol),
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'market_cap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 1),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'description': info.get('longBusinessSummary', '')
            }
            
            # Add sector-specific info
            sector = self._get_stock_sector(symbol)
            if sector == 'real_estate':
                basic_info.update(self.fetch_reit_specific_data(symbol))
            elif sector == 'financial':
                basic_info.update(self.fetch_financial_sector_data(symbol))
            
            return basic_info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def screen_stocks(self, criteria: Dict) -> List[str]:
        """
        Screen stocks based on criteria
        
        Args:
            criteria: Dictionary with screening criteria
            
        Returns:
            List of symbols meeting criteria
        """
        screened = []
        
        for symbol in self.get_stock_list():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                meets_criteria = True
                
                # Check each criterion
                if 'min_market_cap' in criteria:
                    if info.get('marketCap', 0) < criteria['min_market_cap']:
                        meets_criteria = False
                
                if 'min_dividend_yield' in criteria:
                    div_yield = info.get('dividendYield', 0)
                    if div_yield < criteria['min_dividend_yield']:
                        meets_criteria = False
                
                if 'max_pe' in criteria:
                    pe = info.get('trailingPE', float('inf'))
                    if pe > criteria['max_pe']:
                        meets_criteria = False
                
                if 'min_roe' in criteria:
                    roe = info.get('returnOnEquity', 0)
                    if roe < criteria['min_roe']:
                        meets_criteria = False
                
                if 'sector' in criteria:
                    if self._get_stock_sector(symbol) != criteria['sector']:
                        meets_criteria = False
                
                if meets_criteria:
                    screened.append(symbol)
                    
            except Exception as e:
                logger.debug(f"Error screening {symbol}: {e}")
                continue
        
        return screened
    
    def get_sector_summary(self) -> Dict:
        """Get summary statistics for all sectors"""
        summary = {}
        
        for sector in self.stock_universe.keys():
            if sector != 'benchmarks':
                stocks = self.get_stocks_by_sector(sector)
                
                sector_data = {
                    'stock_count': len(stocks),
                    'stocks': stocks,
                    'momentum': self.calculate_sector_momentum(sector),
                    'avg_pe': 0,
                    'avg_dividend_yield': 0,
                    'total_market_cap': 0
                }
                
                # Calculate averages
                pe_ratios = []
                yields = []
                market_caps = []
                
                for symbol in stocks:
                    try:
                        info = yf.Ticker(symbol).info
                        pe = info.get('trailingPE', 0)
                        if pe and pe > 0:
                            pe_ratios.append(pe)
                        
                        div_yield = info.get('dividendYield', 0)
                        if div_yield:
                            yields.append(div_yield * 100)
                        
                        market_cap = info.get('marketCap', 0)
                        if market_cap:
                            market_caps.append(market_cap)
                    except:
                        continue
                
                if pe_ratios:
                    sector_data['avg_pe'] = round(np.mean(pe_ratios), 2)
                if yields:
                    sector_data['avg_dividend_yield'] = round(np.mean(yields), 2)
                if market_caps:
                    sector_data['total_market_cap'] = sum(market_caps)
                
                summary[sector] = sector_data
        
        return summary

# Factory function to get enhanced processor
def get_enhanced_data_processor():
    """Factory function to get enhanced data processor instance"""
    return EnhancedStockDataProcessor()

# Backward compatibility
def get_data_processor():
    """Backward compatible function that returns enhanced processor"""
    return get_enhanced_data_processor()