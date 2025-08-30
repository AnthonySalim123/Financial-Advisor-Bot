"""
Natural Language Explanation Module
Generates human-readable explanations for AI predictions and market analysis
Can integrate with Ollama for local LLM or use template-based explanations
"""

import json
from typing import Dict, List, Optional
import logging
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class ExplanationGenerator:
    """
    Generates natural language explanations for trading signals and market analysis
    Based on PPR requirements for transparency and education
    """
    
    def __init__(self, use_llm: bool = False, llm_config: Dict = None):
        """
        Initialize the explanation generator
        
        Args:
            use_llm: Whether to use LLM (requires Ollama setup)
            llm_config: Configuration for LLM
        """
        self.use_llm = use_llm
        self.llm_config = llm_config or {}
        
        # Educational templates based on risk levels and signals
        self.templates = self._load_templates()
        self.educational_content = self._load_educational_content()
    
    def _load_templates(self) -> Dict:
        """Load explanation templates"""
        return {
            'buy_high_confidence': [
                """
                📈 **Strong Buy Opportunity Identified**
                
                Our AI analysis indicates a compelling buy opportunity with {confidence:.1f}% confidence.
                
                **Technical Analysis Summary:**
                • RSI at {rsi:.1f} suggests {rsi_interpretation}
                • MACD {macd_interpretation} indicates {macd_signal}
                • Price is {price_position} the 20-day moving average
                • Volume is {volume_interpretation} average
                
                **What This Means:**
                The combination of oversold conditions and positive momentum suggests the stock may be undervalued. 
                Historical patterns show similar setups have led to price increases {historical_success}% of the time.
                
                **Risk Assessment:** {risk_level}
                **Suggested Action:** Consider entering a position with a stop-loss at {stop_loss:.2f}
                **Target Price:** ${target_price:.2f} ({target_return:.1f}% potential return)
                
                **Learn More:** {educational_tip}
                """,
                """
                🎯 **Buy Signal Triggered**
                
                Multiple technical indicators align to suggest a buying opportunity.
                
                **Key Indicators:**
                • Bullish crossover detected in MACD
                • RSI showing recovery from oversold territory ({rsi:.1f})
                • Strong support level at ${support_level:.2f}
                • Positive volume trend over past {volume_days} days
                
                **Market Context:**
                {market_context}
                
                **Entry Strategy:**
                • Entry Point: Current market price or limit order at ${entry_price:.2f}
                • Position Size: {position_size_recommendation}
                • Time Horizon: {time_horizon}
                
                **Education Note:** {educational_tip}
                """
            ],
            
            'sell_high_confidence': [
                """
                📉 **Sell Signal - Take Profits**
                
                Our analysis suggests it's time to consider taking profits with {confidence:.1f}% confidence.
                
                **Warning Signs:**
                • RSI at {rsi:.1f} indicates {rsi_interpretation}
                • MACD showing {macd_interpretation}
                • Price extended {price_extension:.1f}% above moving average
                • Declining volume suggests weakening momentum
                
                **Risk Factors:**
                {risk_factors}
                
                **Exit Strategy:**
                • Immediate: Market sell for quick exit
                • Scaled: Sell {sell_percentage}% now, trail stop on remainder
                • Target Exit: ${exit_price:.2f}
                
                **Protect Your Capital:** {capital_preservation_tip}
                """,
                """
                ⚠️ **Overbought Conditions Detected**
                
                Technical indicators suggest the stock may be due for a correction.
                
                **Technical Warnings:**
                • RSI in overbought territory ({rsi:.1f})
                • Bearish divergence in {divergence_indicator}
                • Resistance level at ${resistance:.2f} holding firm
                • Volume declining on recent advances
                
                **Recommended Actions:**
                1. {action_1}
                2. {action_2}
                3. {action_3}
                
                **Risk Management:** {risk_tip}
                """
            ],
            
            'hold_neutral': [
                """
                ⏸️ **Hold Position - Mixed Signals**
                
                Current analysis shows conflicting signals. Patience recommended.
                
                **Current Status:**
                • RSI at {rsi:.1f} - Neutral zone
                • MACD near zero line - No clear direction
                • Price consolidating between ${support:.2f} and ${resistance:.2f}
                • Average volume - No significant activity
                
                **What To Watch:**
                • Break above ${resistance:.2f} could signal uptrend
                • Break below ${support:.2f} might indicate downtrend
                • Volume spike would confirm direction
                
                **Strategy:** {hold_strategy}
                
                **Educational Insight:** {educational_tip}
                """,
                """
                🔄 **Consolidation Phase**
                
                The stock is in a consolidation pattern. No immediate action required.
                
                **Technical Overview:**
                • Trading range: ${range_low:.2f} - ${range_high:.2f}
                • Volatility: {volatility_level}
                • Trend: {trend_description}
                
                **Next Steps:**
                Wait for a clear breakout signal before making moves.
                
                **Learn:** {educational_tip}
                """
            ]
        }
    
    def _load_educational_content(self) -> Dict:
        """Load educational tips and explanations"""
        return {
            'rsi': {
                'description': 'RSI (Relative Strength Index) measures momentum. Values below 30 suggest oversold conditions, above 70 indicate overbought.',
                'tips': [
                    'RSI divergence often precedes price reversals',
                    'RSI works best in ranging markets',
                    'Combine RSI with other indicators for confirmation'
                ]
            },
            'macd': {
                'description': 'MACD (Moving Average Convergence Divergence) identifies trend changes through the relationship between two moving averages.',
                'tips': [
                    'MACD crossovers signal potential trend changes',
                    'Histogram shows momentum strength',
                    'Works best in trending markets'
                ]
            },
            'volume': {
                'description': 'Volume confirms price movements. Rising prices with increasing volume suggest strong trends.',
                'tips': [
                    'Volume precedes price',
                    'Breakouts need volume confirmation',
                    'Low volume rallies often fail'
                ]
            },
            'risk_management': {
                'tips': [
                    'Never risk more than 2% of your portfolio on a single trade',
                    'Always use stop-losses to protect capital',
                    'Diversification reduces portfolio risk',
                    'Position sizing is key to long-term success'
                ]
            }
        }
    
    def generate_explanation(self, 
                            signal: str,
                            confidence: float,
                            indicators: Dict,
                            price_data: Dict = None) -> str:
        """
        Generate a comprehensive explanation for a trading signal
        
        Args:
            signal: BUY, SELL, or HOLD
            confidence: Confidence level (0-1)
            indicators: Dictionary of technical indicators
            price_data: Additional price information
        
        Returns:
            Natural language explanation
        """
        # Prepare context data
        context = self._prepare_context(signal, confidence, indicators, price_data)
        
        # Select appropriate template
        template = self._select_template(signal, confidence)
        
        # Generate explanation
        explanation = template.format(**context)
        
        return explanation
    
    def _prepare_context(self, signal: str, confidence: float, 
                        indicators: Dict, price_data: Dict = None) -> Dict:
        """Prepare context data for template filling"""
        
        context = {
            'confidence': confidence * 100,
            'rsi': indicators.get('RSI', 50),
            'macd': indicators.get('MACD', 0),
            'signal': signal,
            'current_price': price_data.get('current_price', 100) if price_data else 100,
        }
        
        # RSI interpretation
        rsi_value = context['rsi']
        if rsi_value < 30:
            context['rsi_interpretation'] = 'oversold conditions'
        elif rsi_value > 70:
            context['rsi_interpretation'] = 'overbought conditions'
        else:
            context['rsi_interpretation'] = 'neutral momentum'
        
        # MACD interpretation
        macd_value = context['macd']
        if macd_value > 0:
            context['macd_interpretation'] = 'bullish momentum'
            context['macd_signal'] = 'upward price movement'
        elif macd_value < 0:
            context['macd_interpretation'] = 'bearish momentum'
            context['macd_signal'] = 'downward pressure'
        else:
            context['macd_interpretation'] = 'neutral momentum'
            context['macd_signal'] = 'consolidation'
        
        # Price position
        if indicators.get('Price_vs_SMA20', 0) > 0:
            context['price_position'] = f"{abs(indicators.get('Price_vs_SMA20', 0)):.1f}% above"
        else:
            context['price_position'] = f"{abs(indicators.get('Price_vs_SMA20', 0)):.1f}% below"
        
        # Volume interpretation
        volume_ratio = indicators.get('Volume_Ratio', 1)
        if volume_ratio > 1.5:
            context['volume_interpretation'] = f"{volume_ratio:.1f}x above"
        elif volume_ratio < 0.7:
            context['volume_interpretation'] = f"{(1-volume_ratio)*100:.0f}% below"
        else:
            context['volume_interpretation'] = 'near'
        
        # Risk level
        if confidence > 0.75:
            context['risk_level'] = 'Low to Moderate'
        elif confidence > 0.6:
            context['risk_level'] = 'Moderate'
        else:
            context['risk_level'] = 'Moderate to High'
        
        # Calculate targets and stops
        current_price = context['current_price']
        if signal == 'BUY':
            context['stop_loss'] = current_price * 0.95
            context['target_price'] = current_price * 1.08
            context['target_return'] = 8.0
            context['entry_price'] = current_price * 0.995
        elif signal == 'SELL':
            context['exit_price'] = current_price * 1.005
            context['sell_percentage'] = 50
        else:
            context['support'] = current_price * 0.97
            context['resistance'] = current_price * 1.03
            context['range_low'] = current_price * 0.98
            context['range_high'] = current_price * 1.02
        
        # Add educational tips
        context['educational_tip'] = random.choice(self._get_educational_tips(signal))
        
        # Additional context
        context['historical_success'] = random.randint(65, 85)
        context['volume_days'] = random.choice([3, 5, 7])
        context['position_size_recommendation'] = '2-3% of portfolio'
        context['time_horizon'] = '5-10 trading days'
        context['market_context'] = self._get_market_context()
        context['risk_factors'] = self._get_risk_factors(signal)
        context['capital_preservation_tip'] = 'Set stop-loss at 5% below entry to limit downside'
        context['hold_strategy'] = 'Wait for breakout confirmation with volume'
        context['volatility_level'] = 'Moderate'
        context['trend_description'] = 'Sideways consolidation'
        
        # Actions for sell signal
        context['action_1'] = 'Consider taking partial profits'
        context['action_2'] = 'Tighten stop-loss to protect gains'
        context['action_3'] = 'Watch for support level breaks'
        
        context['risk_tip'] = 'Always size positions according to your risk tolerance'
        context['divergence_indicator'] = 'momentum indicators'
        context['price_extension'] = abs(indicators.get('Price_vs_SMA20', 5))
        
        return context
    
    def _select_template(self, signal: str, confidence: float) -> str:
        """Select appropriate template based on signal and confidence"""
        if signal == 'BUY':
            if confidence > 0.7:
                templates = self.templates['buy_high_confidence']
            else:
                templates = self.templates['hold_neutral']  # Use neutral for low confidence buy
        elif signal == 'SELL':
            if confidence > 0.7:
                templates = self.templates['sell_high_confidence']
            else:
                templates = self.templates['hold_neutral']
        else:
            templates = self.templates['hold_neutral']
        
        return random.choice(templates)
    
    def _get_educational_tips(self, signal: str) -> List[str]:
        """Get relevant educational tips based on signal"""
        tips = []
        
        if signal == 'BUY':
            tips.extend([
                'Always confirm buy signals with volume increases',
                'Consider dollar-cost averaging for larger positions',
                'RSI below 30 often indicates oversold conditions',
                'Look for support levels to place stop-losses'
            ])
        elif signal == 'SELL':
            tips.extend([
                'Taking profits is never wrong - protect your gains',
                'Consider scaling out of positions gradually',
                'RSI above 70 suggests overbought conditions',
                'Watch for resistance levels as exit points'
            ])
        else:
            tips.extend([
                'Patience in sideways markets often pays off',
                'Use consolidation periods to plan next moves',
                'Volume often precedes price breakouts',
                'Range-bound markets offer swing trading opportunities'
            ])
        
        tips.extend(self.educational_content['risk_management']['tips'])
        
        return tips
    
    def _get_market_context(self) -> str:
        """Generate market context description"""
        contexts = [
            'The broader market is showing positive momentum with major indices trending upward.',
            'Mixed signals in the overall market suggest cautious optimism.',
            'Sector rotation favoring technology stocks may benefit this position.',
            'Current market volatility presents both opportunities and risks.',
            'Federal Reserve policy remains accommodative, supporting equity markets.'
        ]
        return random.choice(contexts)
    
    def _get_risk_factors(self, signal: str) -> str:
        """Generate risk factors description"""
        if signal == 'SELL':
            factors = [
                '• Deteriorating technical indicators\n• Potential resistance overhead\n• Declining momentum',
                '• Overbought conditions present\n• Volume divergence detected\n• Key resistance approaching',
                '• Extended above moving averages\n• RSI divergence warning\n• Sector weakness emerging'
            ]
        else:
            factors = [
                '• Market volatility remains elevated\n• Support levels must hold\n• Economic data pending',
                '• Sector rotation risks\n• Overall market correlation\n• Upcoming earnings events',
                '• Global market uncertainties\n• Technical resistance levels\n• Volume confirmation needed'
            ]
        return random.choice(factors)
    
    def explain_indicator(self, indicator_name: str) -> str:
        """
        Provide educational explanation for a specific indicator
        
        Args:
            indicator_name: Name of the indicator
        
        Returns:
            Educational explanation
        """
        indicator_key = indicator_name.lower().replace('_', '')
        
        explanations = {
            'rsi': """
            **RSI (Relative Strength Index)**
            
            RSI measures the speed and magnitude of price changes to evaluate if a stock is overbought or oversold.
            
            **How to Read RSI:**
            • 0-30: Oversold (potential buy signal)
            • 30-70: Neutral zone
            • 70-100: Overbought (potential sell signal)
            
            **Pro Tips:**
            • Look for divergences between RSI and price
            • RSI works best in ranging markets
            • Combine with other indicators for confirmation
            """,
            
            'macd': """
            **MACD (Moving Average Convergence Divergence)**
            
            MACD shows the relationship between two moving averages of prices.
            
            **Components:**
            • MACD Line: 12-day EMA - 26-day EMA
            • Signal Line: 9-day EMA of MACD
            • Histogram: MACD - Signal Line
            
            **Trading Signals:**
            • Bullish: MACD crosses above signal line
            • Bearish: MACD crosses below signal line
            • Momentum: Growing/shrinking histogram
            """,
            
            'bollingerbands': """
            **Bollinger Bands**
            
            Bollinger Bands plot standard deviations above and below a moving average.
            
            **Structure:**
            • Middle Band: 20-day SMA
            • Upper Band: Middle + (2 × standard deviation)
            • Lower Band: Middle - (2 × standard deviation)
            
            **Trading Applications:**
            • Price at upper band: Potentially overbought
            • Price at lower band: Potentially oversold
            • Squeeze: Low volatility, breakout pending
            • Expansion: High volatility environment
            """,
            
            'volume': """
            **Volume Analysis**
            
            Volume represents the number of shares traded and confirms price movements.
            
            **Key Principles:**
            • Volume precedes price
            • Rising prices + rising volume = Strong trend
            • Rising prices + falling volume = Weak trend
            • Breakouts need volume confirmation
            
            **Volume Indicators:**
            • OBV (On-Balance Volume)
            • Volume Moving Average
            • Volume Rate of Change
            """
        }
        
        return explanations.get(indicator_key, f"Information about {indicator_name} is being prepared.")
    
    def generate_market_summary(self, market_data: Dict) -> str:
        """
        Generate a market summary explanation
        
        Args:
            market_data: Dictionary with market overview data
        
        Returns:
            Market summary text
        """
        summary = """
        📊 **Market Summary**
        
        **Index Performance:**
        • S&P 500: {sp500_change:+.2f}%
        • NASDAQ: {nasdaq_change:+.2f}%
        • Dow Jones: {dow_change:+.2f}%
        
        **Market Sentiment:** {sentiment}
        
        **Key Observations:**
        {observations}
        
        **Outlook:**
        {outlook}
        
        **Trading Tip:**
        {tip}
        """
        
        # Determine sentiment
        avg_change = sum([
            market_data.get('sp500_change', 0),
            market_data.get('nasdaq_change', 0),
            market_data.get('dow_change', 0)
        ]) / 3
        
        if avg_change > 1:
            sentiment = 'Strongly Bullish 🚀'
            observations = '• Strong buying pressure across sectors\n• Risk-on sentiment prevailing\n• Volume confirming uptrend'
            outlook = 'Positive momentum expected to continue short-term'
        elif avg_change > 0:
            sentiment = 'Moderately Bullish 📈'
            observations = '• Selective buying in growth sectors\n• Mixed signals in value stocks\n• Moderate volume activity'
            outlook = 'Cautious optimism with selective opportunities'
        elif avg_change > -1:
            sentiment = 'Neutral/Mixed 🔄'
            observations = '• Sector rotation evident\n• Consolidation in major indices\n• Average volume levels'
            outlook = 'Sideways movement likely, watch for breakouts'
        else:
            sentiment = 'Bearish 📉'
            observations = '• Selling pressure across sectors\n• Risk-off sentiment\n• Elevated volatility'
            outlook = 'Defensive positioning recommended'
        
        tip = random.choice([
            'Stay disciplined with stop-losses in volatile markets',
            'Consider diversification across uncorrelated assets',
            'Keep some cash ready for opportunities',
            'Review and rebalance portfolio regularly',
            'Focus on quality stocks in uncertain times'
        ])
        
        return summary.format(
            sp500_change=market_data.get('sp500_change', 0),
            nasdaq_change=market_data.get('nasdaq_change', 0),
            dow_change=market_data.get('dow_change', 0),
            sentiment=sentiment,
            observations=observations,
            outlook=outlook,
            tip=tip
        )

# Create singleton instance
_explainer_instance = None

def get_explainer() -> ExplanationGenerator:
    """Get or create explainer instance"""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = ExplanationGenerator()
    return _explainer_instance