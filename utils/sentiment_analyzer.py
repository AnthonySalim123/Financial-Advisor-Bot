# utils/sentiment_analyzer.py
"""
Market Sentiment Analysis Module
Analyzes news sentiment using FinBERT and social media indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import logging
from datetime import datetime, timedelta
import requests
import re
from collections import Counter
import json

import streamlit as st
from datetime import datetime

def initialize_session_state():
    """Initialize all session state variables for the application"""
    
    # User Profile
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Guest User',
            'email': 'guest@stockbot.com',
            'risk_tolerance': 'Moderate',
            'investment_horizon': '1-3 years',
            'initial_capital': 100000.0,
            'currency': 'USD',
            'experience_level': 'Intermediate',
            'investment_goals': ['Growth', 'Income'],
            'preferred_sectors': ['Technology', 'Healthcare'],
            'max_position_size': 10.0,  # percentage
            'created_at': datetime.now().isoformat()
        }
    
    # Portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'holdings': {},  # {symbol: {'shares': float, 'avg_cost': float, 'purchase_date': str}}
            'cash': 100000.0,
            'total_value': 100000.0,
            'daily_return': 0.0,
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'transactions': [],  # List of transaction records
            'last_updated': datetime.now().isoformat()
        }
    
    # Watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Analysis preferences
    if 'analysis_preferences' not in st.session_state:
        st.session_state.analysis_preferences = {
            'selected_stock': 'AAPL',
            'analysis_period': '1y',
            'show_sentiment': True,
            'show_shap': False,
            'model_trained': False,
            'chart_type': 'candlestick',
            'indicators': ['RSI', 'MACD', 'SMA_20', 'SMA_50']
        }
    
    # Market data cache
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
    
    # Application settings
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = {
            'theme': 'minimal',
            'data_refresh_interval': 300,  # seconds
            'enable_notifications': True,
            'auto_save': True,
            'language': 'en',
            'timezone': 'UTC'
        }
    
    # Backtesting state
    if 'backtesting' not in st.session_state:
        st.session_state.backtesting = {
            'strategy': None,
            'results': None,
            'parameters': {},
            'last_run': None
        }
    
    # Education progress
    if 'education_progress' not in st.session_state:
        st.session_state.education_progress = {
            'completed_modules': [],
            'current_module': None,
            'quiz_scores': {},
            'certificates': []
        }

def get_user_profile():
    """Get user profile with defaults"""
    if 'user_profile' not in st.session_state:
        initialize_session_state()
    return st.session_state.user_profile

def get_portfolio():
    """Get portfolio with defaults"""
    if 'portfolio' not in st.session_state:
        initialize_session_state()
    return st.session_state.portfolio

def update_user_profile(updates: dict):
    """Update user profile"""
    if 'user_profile' not in st.session_state:
        initialize_session_state()
    st.session_state.user_profile.update(updates)

def update_portfolio(updates: dict):
    """Update portfolio"""
    if 'portfolio' not in st.session_state:
        initialize_session_state()
    st.session_state.portfolio.update(updates)
    st.session_state.portfolio['last_updated'] = datetime.now().isoformat()

def add_to_watchlist(symbol: str):
    """Add symbol to watchlist"""
    if 'watchlist' not in st.session_state:
        initialize_session_state()
    if symbol not in st.session_state.watchlist:
        st.session_state.watchlist.append(symbol)

def remove_from_watchlist(symbol: str):
    """Remove symbol from watchlist"""
    if 'watchlist' not in st.session_state:
        initialize_session_state()
    if symbol in st.session_state.watchlist:
        st.session_state.watchlist.remove(symbol)

def add_transaction(transaction: dict):
    """Add transaction to portfolio"""
    portfolio = get_portfolio()
    portfolio['transactions'].append({
        **transaction,
        'timestamp': datetime.now().isoformat()
    })
    update_portfolio(portfolio)

def clear_session_state():
    """Clear all session state (for logout/reset)"""
    keys_to_clear = [
        'user_profile', 'portfolio', 'watchlist', 'analysis_preferences',
        'market_data', 'backtesting', 'education_progress'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Try importing transformers with fallback
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

# Try importing alternative sentiment libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketSentimentAnalyzer:
    """
    Comprehensive sentiment analysis for financial markets
    """
    
    def __init__(self):
        """Initialize sentiment analyzer with multiple models"""
        self.sentiment_model = None
        self.tokenizer = None
        self.fallback_mode = False
        
        # Initialize FinBERT model
        self._initialize_models()
        
        # Sentiment weights for aggregation
        self.sentiment_weights = {
            'news': 0.6,
            'social': 0.2,
            'market': 0.2
        }
    
    def _initialize_models(self):
        """Initialize sentiment analysis models with fallbacks"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Try to load FinBERT
                model_name = "ProsusAI/finbert"
                logger.info("Loading FinBERT model...")
                
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=-1  # Use CPU
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info("✅ FinBERT model loaded successfully")
                
            else:
                raise ImportError("Transformers not available")
                
        except Exception as e:
            logger.warning(f"FinBERT loading failed: {e}")
            logger.info("🔄 Falling back to TextBlob sentiment analysis")
            
            if TEXTBLOB_AVAILABLE:
                self.fallback_mode = True
                logger.info("✅ TextBlob fallback enabled")
            else:
                logger.error("❌ No sentiment analysis libraries available")
    
    def analyze_news_sentiment(self, symbol: str, max_articles: int = 10) -> Dict:
        """
        Analyze news sentiment for a stock symbol
        
        Args:
            symbol: Stock ticker symbol
            max_articles: Maximum number of articles to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Fetch news from yfinance
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.warning(f"No news found for {symbol}")
                return self._create_empty_sentiment_result("No news available")
            
            # Limit articles
            news = news[:max_articles]
            
            sentiments = []
            processed_articles = []
            
            for article in news:
                try:
                    # Get article details
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    
                    # Combine title and summary for analysis
                    text = f"{title}. {summary}".strip()
                    
                    if len(text) < 10:  # Skip very short texts
                        continue
                    
                    # Analyze sentiment
                    sentiment_result = self._analyze_text_sentiment(text)
                    
                    if sentiment_result:
                        sentiments.append(sentiment_result)
                        processed_articles.append({
                            'title': title,
                            'summary': summary[:200] + '...' if len(summary) > 200 else summary,
                            'sentiment': sentiment_result['label'],
                            'score': sentiment_result['score'],
                            'confidence': sentiment_result.get('confidence', 0),
                            'publish_time': article.get('providerPublishTime', 0)
                        })
                        
                except Exception as e:
                    logger.warning(f"Error processing article: {e}")
                    continue
            
            if not sentiments:
                return self._create_empty_sentiment_result("No processable articles")
            
            # Calculate aggregate sentiment
            sentiment_score = self._calculate_aggregate_sentiment(sentiments)
            
            # Get sentiment distribution
            distribution = self._get_sentiment_distribution(sentiments)
            
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'sentiment_label': self._score_to_label(sentiment_score),
                'confidence': np.mean([s.get('confidence', s['score']) for s in sentiments]),
                'articles_analyzed': len(processed_articles),
                'articles': processed_articles,
                'distribution': distribution,
                'analysis_time': datetime.now().isoformat(),
                'model_used': 'FinBERT' if not self.fallback_mode else 'TextBlob'
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed for {symbol}: {e}")
            return self._create_empty_sentiment_result(f"Analysis failed: {str(e)}")
    
    def _analyze_text_sentiment(self, text: str) -> Optional[Dict]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment results
        """
        try:
            if not self.fallback_mode and self.sentiment_model:
                # Use FinBERT
                # Truncate text to model's max length
                if self.tokenizer:
                    tokens = self.tokenizer.encode(text, truncation=True, max_length=512)
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                
                result = self.sentiment_model(text)[0]
                
                # Convert FinBERT labels to standardized format
                label_mapping = {
                    'positive': 'positive',
                    'negative': 'negative',
                    'neutral': 'neutral',
                    'POSITIVE': 'positive',
                    'NEGATIVE': 'negative',
                    'NEUTRAL': 'neutral'
                }
                
                return {
                    'label': label_mapping.get(result['label'].lower(), 'neutral'),
                    'score': result['score'],
                    'confidence': result['score']
                }
                
            elif self.fallback_mode and TEXTBLOB_AVAILABLE:
                # Use TextBlob fallback
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                # Convert polarity to label
                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return {
                    'label': label,
                    'score': abs(polarity),
                    'confidence': min(abs(polarity) * 2, 1.0)  # Scale confidence
                }
            
            else:
                # No sentiment analysis available
                return {
                    'label': 'neutral',
                    'score': 0.5,
                    'confidence': 0.0
                }
                
        except Exception as e:
            logger.warning(f"Text sentiment analysis failed: {e}")
            return None
    
    def _calculate_aggregate_sentiment(self, sentiments: List[Dict]) -> float:
        """
        Calculate aggregate sentiment score from individual sentiments
        
        Args:
            sentiments: List of sentiment dictionaries
            
        Returns:
            Aggregate sentiment score (-1 to 1)
        """
        if not sentiments:
            return 0.0
        
        # Weight sentiments by confidence
        weighted_scores = []
        total_weight = 0
        
        for sentiment in sentiments:
            score = sentiment['score']
            confidence = sentiment.get('confidence', score)
            
            # Convert label to numeric score
            if sentiment['label'] == 'positive':
                numeric_score = score
            elif sentiment['label'] == 'negative':
                numeric_score = -score
            else:  # neutral
                numeric_score = 0
            
            weight = confidence
            weighted_scores.append(numeric_score * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        aggregate_score = sum(weighted_scores) / total_weight
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, aggregate_score))
    
    def _get_sentiment_distribution(self, sentiments: List[Dict]) -> Dict:
        """Get distribution of sentiment labels"""
        if not sentiments:
            return {'positive': 0, 'negative': 0, 'neutral': 0}
        
        labels = [s['label'] for s in sentiments]
        counter = Counter(labels)
        total = len(sentiments)
        
        return {
            'positive': counter.get('positive', 0) / total,
            'negative': counter.get('negative', 0) / total,
            'neutral': counter.get('neutral', 0) / total
        }
    
    def _score_to_label(self, score: float) -> str:
        """Convert numeric sentiment score to label"""
        if score > 0.2:
            return 'positive'
        elif score < -0.2:
            return 'negative'
        else:
            return 'neutral'
    
    def _create_empty_sentiment_result(self, reason: str) -> Dict:
        """Create empty sentiment result with default values"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'articles_analyzed': 0,
            'articles': [],
            'distribution': {'positive': 0, 'negative': 0, 'neutral': 1},
            'analysis_time': datetime.now().isoformat(),
            'model_used': 'none',
            'error': reason
        }
    
    def analyze_market_sentiment_indicators(self, symbol: str) -> Dict:
        """
        Analyze market-based sentiment indicators
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with market sentiment metrics
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent price data
            hist = ticker.history(period="1mo")
            
            if hist.empty:
                return {'error': 'No market data available'}
            
            # Calculate market sentiment indicators
            current_price = hist['Close'].iloc[-1]
            month_high = hist['High'].max()
            month_low = hist['Low'].min()
            
            # Price position in range
            price_position = (current_price - month_low) / (month_high - month_low) if month_high != month_low else 0.5
            
            # Volume trend
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].tail(5).mean()
            volume_trend = (recent_volume / avg_volume - 1) if avg_volume > 0 else 0
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Momentum
            momentum_5d = (current_price / hist['Close'].iloc[-6] - 1) if len(hist) > 5 else 0
            momentum_20d = (current_price / hist['Close'].iloc[-21] - 1) if len(hist) > 20 else 0
            
            return {
                'price_position': price_position,
                'volume_trend': volume_trend,
                'volatility': volatility,
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d,
                'market_sentiment_score': self._calculate_market_sentiment_score(
                    price_position, volume_trend, momentum_5d, momentum_20d
                )
            }
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_market_sentiment_score(self, price_pos: float, vol_trend: float, 
                                        mom_5d: float, mom_20d: float) -> float:
        """Calculate overall market sentiment score"""
        # Normalize inputs
        price_sentiment = (price_pos - 0.5) * 2  # -1 to 1
        volume_sentiment = np.tanh(vol_trend)    # -1 to 1
        momentum_sentiment = (np.tanh(mom_5d * 10) + np.tanh(mom_20d * 5)) / 2  # -1 to 1
        
        # Weighted average
        market_sentiment = (
            price_sentiment * 0.3 +
            volume_sentiment * 0.2 +
            momentum_sentiment * 0.5
        )
        
        return max(-1.0, min(1.0, market_sentiment))
    
    def get_comprehensive_sentiment(self, symbol: str) -> Dict:
        """
        Get comprehensive sentiment analysis combining multiple sources
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with comprehensive sentiment analysis
        """
        try:
            # Get news sentiment
            news_sentiment = self.analyze_news_sentiment(symbol)
            
            # Get market sentiment
            market_sentiment = self.analyze_market_sentiment_indicators(symbol)
            
            # Calculate overall sentiment
            overall_score = 0.0
            confidence = 0.0
            components = {}
            
            # News sentiment component
            if 'sentiment_score' in news_sentiment:
                news_score = news_sentiment['sentiment_score']
                news_conf = news_sentiment.get('confidence', 0)
                components['news'] = {
                    'score': news_score,
                    'confidence': news_conf,
                    'weight': self.sentiment_weights['news']
                }
                overall_score += news_score * self.sentiment_weights['news'] * news_conf
                confidence += self.sentiment_weights['news'] * news_conf
            
            # Market sentiment component
            if 'market_sentiment_score' in market_sentiment:
                market_score = market_sentiment['market_sentiment_score']
                market_conf = 0.8  # Market data is generally reliable
                components['market'] = {
                    'score': market_score,
                    'confidence': market_conf,
                    'weight': self.sentiment_weights['market']
                }
                overall_score += market_score * self.sentiment_weights['market'] * market_conf
                confidence += self.sentiment_weights['market'] * market_conf
            
            # Normalize by total confidence
            if confidence > 0:
                overall_score = overall_score / confidence
            
            return {
                'symbol': symbol,
                'overall_sentiment_score': overall_score,
                'overall_sentiment_label': self._score_to_label(overall_score),
                'overall_confidence': min(confidence, 1.0),
                'components': components,
                'news_sentiment': news_sentiment,
                'market_sentiment': market_sentiment,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive sentiment analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'overall_sentiment_score': 0.0,
                'overall_sentiment_label': 'neutral',
                'overall_confidence': 0.0,
                'error': str(e)
            }
    
    def get_sentiment_features(self, symbol: str) -> Dict:
        """
        Get sentiment features for ML model integration
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment features for ML
        """
        try:
            sentiment_data = self.get_comprehensive_sentiment(symbol)
            
            # Extract features for ML model
            features = {
                'sentiment_score': sentiment_data.get('overall_sentiment_score', 0.0),
                'sentiment_confidence': sentiment_data.get('overall_confidence', 0.0),
                'news_sentiment': sentiment_data.get('news_sentiment', {}).get('sentiment_score', 0.0),
                'news_confidence': sentiment_data.get('news_sentiment', {}).get('confidence', 0.0),
                'news_articles_count': sentiment_data.get('news_sentiment', {}).get('articles_analyzed', 0),
                'market_sentiment': sentiment_data.get('market_sentiment', {}).get('market_sentiment_score', 0.0),
                'price_position': sentiment_data.get('market_sentiment', {}).get('price_position', 0.5),
                'volume_trend': sentiment_data.get('market_sentiment', {}).get('volume_trend', 0.0),
                'momentum_5d': sentiment_data.get('market_sentiment', {}).get('momentum_5d', 0.0),
                'momentum_20d': sentiment_data.get('market_sentiment', {}).get('momentum_20d', 0.0)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Sentiment feature extraction failed for {symbol}: {e}")
            # Return neutral features as fallback
            return {
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


def create_sentiment_analyzer() -> MarketSentimentAnalyzer:
    """Factory function to create sentiment analyzer"""
    return MarketSentimentAnalyzer()


def get_sentiment_analyzer() -> MarketSentimentAnalyzer:
    """Get singleton sentiment analyzer instance"""
    if not hasattr(get_sentiment_analyzer, '_instance'):
        get_sentiment_analyzer._instance = create_sentiment_analyzer()
    return get_sentiment_analyzer._instance


# Print status on import
logger.info("Sentiment Analysis module loaded")
if TRANSFORMERS_AVAILABLE:
    logger.info("✅ FinBERT sentiment analysis available")
elif TEXTBLOB_AVAILABLE:
    logger.info("⚠️ Using TextBlob fallback for sentiment analysis")
else:
    logger.warning("❌ No sentiment analysis libraries available")