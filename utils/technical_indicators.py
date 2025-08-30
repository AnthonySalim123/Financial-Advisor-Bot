"""
Technical Indicators Module
Calculates various technical indicators for stock analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: Price series
            window: Period for SMA
        
        Returns:
            SMA series
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price series
            window: Period for EMA
        
        Returns:
            EMA series
        """
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            data: Price series
            period: RSI period
        
        Returns:
            RSI series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        
        Returns:
            Dictionary with MACD, Signal, and Histogram
        """
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'MACD': macd,
            'Signal': signal,
            'Histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, 
                                 window: int = 20, 
                                 num_std: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price series
            window: Period for moving average
            num_std: Number of standard deviations
        
        Returns:
            Dictionary with Upper, Middle, and Lower bands
        """
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        # Calculate band position (0-1 scale)
        band_position = (data - lower) / (upper - lower)
        band_position = band_position.clip(0, 1)  # Clip to 0-1 range
        
        return {
            'Upper': upper,
            'Middle': middle,
            'Lower': lower,
            'Position': band_position,
            'Width': upper - lower
        }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, 
                           low: pd.Series, 
                           close: pd.Series,
                           k_period: int = 14,
                           d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: Period for %K
            d_period: Period for %D (moving average of %K)
        
        Returns:
            Dictionary with %K and %D
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'K': k_percent,
            'D': d_percent
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, 
                     low: pd.Series, 
                     close: pd.Series,
                     period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (volatility indicator)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
        
        Returns:
            ATR series
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume
        
        Args:
            close: Close price series
            volume: Volume series
        
        Returns:
            OBV series
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_vwap(high: pd.Series, 
                      low: pd.Series, 
                      close: pd.Series,
                      volume: pd.Series) -> pd.Series:
        """
        Calculate Volume-Weighted Average Price
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
        
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def calculate_pivot_points(high: pd.Series, 
                             low: pd.Series, 
                             close: pd.Series) -> Dict[str, float]:
        """
        Calculate Pivot Points (support and resistance levels)
        
        Args:
            high: High price
            low: Low price
            close: Close price
        
        Returns:
            Dictionary with pivot levels
        """
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = r1 + (high - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = s1 - (high - low)
        
        return {
            'Pivot': pivot,
            'R1': r1,
            'R2': r2,
            'R3': r3,
            'S1': s1,
            'S2': s2,
            'S3': s3
        }
    
    @staticmethod
    def calculate_fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            high: Recent high price
            low: Recent low price
        
        Returns:
            Dictionary with Fibonacci levels
        """
        diff = high - low
        
        levels = {
            '0%': high,
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50%': high - diff * 0.5,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786,
            '100%': low
        }
        
        return levels
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary for indicator parameters
        
        Returns:
            DataFrame with all indicators added
        """
        if config is None:
            config = {
                'sma_short': 20,
                'sma_long': 50,
                'ema_short': 12,
                'ema_long': 26,
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2,
                'stoch_k': 14,
                'stoch_d': 3,
                'atr_period': 14
            }
        
        # Create a copy to avoid modifying original
        result = df.copy()
        
        try:
            # Moving Averages
            result['SMA_20'] = TechnicalIndicators.calculate_sma(result['Close'], config['sma_short'])
            result['SMA_50'] = TechnicalIndicators.calculate_sma(result['Close'], config['sma_long'])
            result['EMA_12'] = TechnicalIndicators.calculate_ema(result['Close'], config['ema_short'])
            result['EMA_26'] = TechnicalIndicators.calculate_ema(result['Close'], config['ema_long'])
            
            # RSI
            result['RSI'] = TechnicalIndicators.calculate_rsi(result['Close'], config['rsi_period'])
            
            # MACD
            macd_dict = TechnicalIndicators.calculate_macd(
                result['Close'], 
                config['macd_fast'], 
                config['macd_slow'], 
                config['macd_signal']
            )
            result['MACD'] = macd_dict['MACD']
            result['MACD_Signal'] = macd_dict['Signal']
            result['MACD_Histogram'] = macd_dict['Histogram']
            
            # Bollinger Bands
            bb_dict = TechnicalIndicators.calculate_bollinger_bands(
                result['Close'], 
                config['bb_period'], 
                config['bb_std']
            )
            result['BB_Upper'] = bb_dict['Upper']
            result['BB_Middle'] = bb_dict['Middle']
            result['BB_Lower'] = bb_dict['Lower']
            result['BB_Position'] = bb_dict['Position']
            result['BB_Width'] = bb_dict['Width']
            
            # Stochastic
            stoch_dict = TechnicalIndicators.calculate_stochastic(
                result['High'], 
                result['Low'], 
                result['Close'],
                config['stoch_k'],
                config['stoch_d']
            )
            result['Stoch_K'] = stoch_dict['K']
            result['Stoch_D'] = stoch_dict['D']
            
            # ATR
            result['ATR'] = TechnicalIndicators.calculate_atr(
                result['High'], 
                result['Low'], 
                result['Close'],
                config['atr_period']
            )
            
            # OBV
            result['OBV'] = TechnicalIndicators.calculate_obv(result['Close'], result['Volume'])
            
            # VWAP (for intraday)
            result['VWAP'] = TechnicalIndicators.calculate_vwap(
                result['High'], 
                result['Low'], 
                result['Close'], 
                result['Volume']
            )
            
            # Volume indicators
            result['Volume_SMA'] = result['Volume'].rolling(window=20).mean()
            result['Volume_Ratio'] = result['Volume'] / result['Volume_SMA']
            
            # Price patterns
            result['High_Low_Spread'] = result['High'] - result['Low']
            result['Close_Open_Spread'] = result['Close'] - result['Open']
            
            # Trend indicators
            result['Price_vs_SMA20'] = (result['Close'] - result['SMA_20']) / result['SMA_20'] * 100
            result['Price_vs_SMA50'] = (result['Close'] - result['SMA_50']) / result['SMA_50'] * 100
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return result
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            DataFrame with signals added
        """
        result = df.copy()
        
        # Initialize signals
        result['Signal'] = 0
        result['Signal_Strength'] = 0
        
        # RSI signals
        result.loc[result['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold = Buy
        result.loc[result['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought = Sell
        result.loc[(result['RSI'] >= 30) & (result['RSI'] <= 70), 'RSI_Signal'] = 0
        
        # MACD signals
        result['MACD_Signal_Line'] = 0
        result.loc[result['MACD'] > result['MACD_Signal'], 'MACD_Signal_Line'] = 1
        result.loc[result['MACD'] < result['MACD_Signal'], 'MACD_Signal_Line'] = -1
        
        # Moving Average signals
        result['MA_Signal'] = 0
        result.loc[result['SMA_20'] > result['SMA_50'], 'MA_Signal'] = 1
        result.loc[result['SMA_20'] < result['SMA_50'], 'MA_Signal'] = -1
        
        # Bollinger Band signals
        result['BB_Signal'] = 0
        result.loc[result['Close'] < result['BB_Lower'], 'BB_Signal'] = 1  # Buy
        result.loc[result['Close'] > result['BB_Upper'], 'BB_Signal'] = -1  # Sell
        
        # Stochastic signals
        result['Stoch_Signal'] = 0
        result.loc[result['Stoch_K'] < 20, 'Stoch_Signal'] = 1  # Oversold
        result.loc[result['Stoch_K'] > 80, 'Stoch_Signal'] = -1  # Overbought
        
        # Combine signals
        signal_columns = ['RSI_Signal', 'MACD_Signal_Line', 'MA_Signal', 'BB_Signal', 'Stoch_Signal']
        result['Combined_Signal'] = result[signal_columns].sum(axis=1)
        
        # Generate final signal based on combined score
        result.loc[result['Combined_Signal'] >= 2, 'Signal'] = 1  # Buy
        result.loc[result['Combined_Signal'] <= -2, 'Signal'] = -1  # Sell
        
        # Signal strength (confidence)
        result['Signal_Strength'] = np.abs(result['Combined_Signal']) / len(signal_columns) * 100
        
        return result
    
    @staticmethod
    def identify_patterns(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Identify common chart patterns
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with identified patterns
        """
        patterns = {}
        
        # Get recent data for pattern detection
        recent = df.tail(20)
        
        # Golden Cross (50-day MA crosses above 200-day MA)
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            if len(df) >= 2:
                patterns['golden_cross'] = (
                    df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] and
                    df['SMA_50'].iloc[-2] <= df['SMA_200'].iloc[-2]
                )
        
        # Death Cross (50-day MA crosses below 200-day MA)
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            if len(df) >= 2:
                patterns['death_cross'] = (
                    df['SMA_50'].iloc[-1] < df['SMA_200'].iloc[-1] and
                    df['SMA_50'].iloc[-2] >= df['SMA_200'].iloc[-2]
                )
        
        # Bullish/Bearish divergence
        if 'RSI' in df.columns:
            price_trend = recent['Close'].iloc[-1] > recent['Close'].iloc[0]
            rsi_trend = recent['RSI'].iloc[-1] > recent['RSI'].iloc[0]
            patterns['bullish_divergence'] = not price_trend and rsi_trend
            patterns['bearish_divergence'] = price_trend and not rsi_trend
        
        # Support/Resistance break
        if len(recent) >= 10:
            resistance = recent['High'].max()
            support = recent['Low'].min()
            current = recent['Close'].iloc[-1]
            patterns['resistance_break'] = current > resistance * 0.98
            patterns['support_break'] = current < support * 1.02
        
        return patterns

# Create a singleton instance
_indicator_calculator = TechnicalIndicators()

def get_indicator_calculator():
    """Get the technical indicator calculator instance"""
    return _indicator_calculator