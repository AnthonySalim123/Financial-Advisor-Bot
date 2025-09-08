"""
Ollama Integration Module - Enhanced Version
Provides natural language explanations using local LLM for transparency
Implements the XAI requirements from the PPR
Author: Anthony Winata Salim
"""

import requests
import json
import subprocess
import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import platform
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaExplainer:
    """
    Enhanced Ollama integration for natural language explanations
    Addresses PPR requirements for transparency and education
    """
    
    def __init__(self, model_preference: List[str] = None, config: Dict = None):
        """
        Initialize the Ollama explainer with enhanced configuration
        
        Args:
            model_preference: List of preferred models in order
            config: Additional configuration options
        """
        self.base_url = "http://localhost:11434"
        self.model_preference = model_preference or [
            "llama3",      # Best overall performance
            "mistral",     # Good for financial analysis
            "llama2",      # Fallback option
            "phi"          # Lightweight alternative
        ]
        
        self.config = config or {
            'temperature': 0.7,
            'max_tokens': 500,
            'top_p': 0.9,
            'timeout': 30,
            'retry_attempts': 3,
            'cache_ttl': 3600  # Cache for 1 hour
        }
        
        self.current_model = None
        self.is_available = False
        self.response_cache = {}
        self.cache_timestamps = {}
        
        # Educational templates for fallback
        self.educational_templates = self._load_educational_templates()
        
        # Initialize Ollama
        self._initialize_ollama()
    
    def _initialize_ollama(self) -> bool:
        """Enhanced Ollama initialization with better error handling"""
        try:
            # Check if Ollama is installed
            if not self._is_ollama_installed():
                logger.warning("Ollama is not installed. Using template-based explanations.")
                return False
            
            # Start Ollama service if not running
            if not self._is_ollama_running():
                logger.info("Starting Ollama service...")
                self._start_ollama_service()
            
            # Select and pull the best available model
            self.current_model = self._select_best_model()
            
            if self.current_model:
                logger.info(f"Using model: {self.current_model}")
                self.is_available = True
                return True
            else:
                logger.warning("No suitable models found. Using fallback explanations.")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            self.is_available = False
            return False
    
    def _is_ollama_installed(self) -> bool:
        """Check if Ollama is installed on the system"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def _start_ollama_service(self) -> bool:
        """Start the Ollama service"""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            elif system == "Linux":
                subprocess.Popen(["systemctl", "start", "ollama"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            elif system == "Windows":
                subprocess.Popen(["ollama", "serve"],
                               creationflags=subprocess.CREATE_NO_WINDOW)
            
            # Wait for service to start
            for _ in range(10):
                time.sleep(1)
                if self._is_ollama_running():
                    logger.info("Ollama service started successfully")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False
    
    def _select_best_model(self) -> Optional[str]:
        """Select the best available model from preferences"""
        try:
            # Get list of available models
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code != 200:
                return None
            
            available_models = {
                model['name'].split(':')[0] 
                for model in response.json().get('models', [])
            }
            
            # Select first available model from preferences
            for preferred_model in self.model_preference:
                if preferred_model in available_models:
                    return preferred_model
            
            # If no preferred model is available, try to pull one
            for preferred_model in self.model_preference[:2]:  # Try first two
                if self._pull_model(preferred_model):
                    return preferred_model
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            return None
    
    def _pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"Pulling model: {model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300  # 5 minutes timeout for download
            )
            
            if response.status_code == 200:
                # Process the stream to show progress
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'status' in data:
                            logger.info(f"Pull status: {data['status']}")
                
                logger.info(f"Successfully pulled model: {model_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def generate_explanation(self, signal_data: Dict, 
                           use_cache: bool = True) -> str:
        """
        Generate comprehensive explanation using LLM or fallback templates
        
        Args:
            signal_data: Dictionary containing signal information
            use_cache: Whether to use cached responses
        
        Returns:
            Natural language explanation
        """
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(signal_data)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.info("Using cached explanation")
                return cached_response
        
        # Try LLM generation
        if self.is_available and self.current_model:
            explanation = self._generate_llm_explanation(signal_data)
            if explanation:
                # Cache the response
                if use_cache:
                    self._cache_response(cache_key, explanation)
                return explanation
        
        # Fallback to template-based explanation
        logger.info("Using template-based explanation")
        return self._generate_template_explanation(signal_data)
    
    def _generate_llm_explanation(self, signal_data: Dict) -> Optional[str]:
        """Generate explanation using Ollama LLM"""
        
        # Construct enhanced prompt
        prompt = self._construct_prompt(signal_data)
        
        # Try to get response with retry logic
        for attempt in range(self.config['retry_attempts']):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.current_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config['temperature'],
                            "top_p": self.config['top_p'],
                            "num_predict": self.config['max_tokens']
                        }
                    },
                    timeout=self.config['timeout']
                )
                
                if response.status_code == 200:
                    result = response.json()
                    explanation = result.get('response', '').strip()
                    
                    if explanation and len(explanation) > 50:
                        return self._enhance_explanation(explanation, signal_data)
                    
                logger.warning(f"Attempt {attempt + 1} failed: Invalid response")
                
            except requests.Timeout:
                logger.warning(f"Attempt {attempt + 1} timed out")
            except Exception as e:
                logger.error(f"Error generating explanation: {e}")
            
            if attempt < self.config['retry_attempts'] - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _construct_prompt(self, signal_data: Dict) -> str:
        """Construct a comprehensive prompt for the LLM"""
        
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0.5) * 100
        indicators = signal_data.get('indicators', {})
        price_data = signal_data.get('price_data', {})
        
        prompt = f"""You are an expert financial advisor providing clear, educational explanations to retail investors.

CURRENT RECOMMENDATION:
Signal: {signal}
Confidence Level: {confidence:.1f}%
Stock: {signal_data.get('symbol', 'N/A')}
Current Price: ${price_data.get('current_price', 0):.2f}

TECHNICAL INDICATORS:
- RSI (14): {indicators.get('RSI', 50):.1f} - {"Overbought" if indicators.get('RSI', 50) > 70 else "Oversold" if indicators.get('RSI', 50) < 30 else "Neutral"}
- MACD: {indicators.get('MACD', 0):.3f}
- MACD Signal: {indicators.get('MACD_Signal', 0):.3f}
- Moving Average (20): ${indicators.get('SMA_20', 0):.2f}
- Moving Average (50): ${indicators.get('SMA_50', 0):.2f}
- Volume Ratio: {indicators.get('Volume_Ratio', 1):.2f}x average
- Bollinger Band Position: {indicators.get('BB_Position', 0.5):.1%}
- ATR (Volatility): {indicators.get('ATR', 0):.2f}

FEATURE IMPORTANCE (ML Model):
{json.dumps(signal_data.get('feature_importance', {}), indent=2)}

MARKET CONTEXT:
- Volatility: {signal_data.get('volatility', 'Normal')}
- Trend: {signal_data.get('trend', 'Neutral')}
- Sector Performance: {signal_data.get('sector_performance', 'Average')}

USER PROFILE:
- Risk Tolerance: {signal_data.get('risk_tolerance', 'Moderate')}
- Investment Horizon: {signal_data.get('investment_horizon', '1-3 years')}

Please provide a comprehensive explanation that includes:

1. PRIMARY REASON for the {signal} signal (1-2 sentences)
2. SUPPORTING EVIDENCE from the technical indicators (2-3 key points)
3. RISK FACTORS to consider (2 specific risks)
4. SUGGESTED ACTION with specific entry/exit points
5. EDUCATIONAL INSIGHT about one of the indicators used

Keep the explanation under 200 words, use simple language, and be specific with numbers.
Format the response with clear sections and bullet points for readability."""

        return prompt
    
    def _generate_template_explanation(self, signal_data: Dict) -> str:
        """Generate template-based explanation as fallback"""
        
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0.5) * 100
        indicators = signal_data.get('indicators', {})
        
        # Select appropriate template based on signal and confidence
        if signal == 'BUY':
            if confidence > 70:
                template = self.educational_templates['buy_high_confidence']
            else:
                template = self.educational_templates['buy_moderate_confidence']
        elif signal == 'SELL':
            if confidence > 70:
                template = self.educational_templates['sell_high_confidence']
            else:
                template = self.educational_templates['sell_moderate_confidence']
        else:
            template = self.educational_templates['hold']
        
        # Fill in the template with actual data
        explanation = template.format(
            confidence=confidence,
            rsi=indicators.get('RSI', 50),
            macd=indicators.get('MACD', 0),
            macd_signal=indicators.get('MACD_Signal', 0),
            sma_20=indicators.get('SMA_20', 0),
            sma_50=indicators.get('SMA_50', 0),
            volume_ratio=indicators.get('Volume_Ratio', 1),
            symbol=signal_data.get('symbol', 'the stock'),
            current_price=signal_data.get('price_data', {}).get('current_price', 100)
        )
        
        return explanation
    
    def _load_educational_templates(self) -> Dict[str, str]:
        """Load educational templates for different scenarios"""
        return {
            'buy_high_confidence': """
📈 **Strong Buy Signal Detected**

With {confidence:.1f}% confidence, our analysis indicates a compelling buying opportunity.

**Key Indicators:**
- RSI at {rsi:.1f} suggests the stock is recovering from oversold conditions
- MACD ({macd:.3f}) has crossed above its signal line ({macd_signal:.3f}), indicating bullish momentum
- Price is trading above the 20-day moving average (${sma_20:.2f})
- Volume is {volume_ratio:.1f}x the average, confirming strong interest

**Risk Considerations:**
- Set a stop-loss 5% below entry price to limit downside
- Market volatility could trigger short-term fluctuations

**Suggested Action:**
Consider entering a position at current levels with a target of 10-15% gain.

**Educational Note:**
The RSI (Relative Strength Index) measures momentum - values below 30 indicate oversold conditions, which often precede price rebounds.
""",
            
            'buy_moderate_confidence': """
📊 **Moderate Buy Signal**

Analysis shows a potential buying opportunity with {confidence:.1f}% confidence.

**Supporting Factors:**
- Technical indicators are showing early bullish signs
- RSI at {rsi:.1f} is approaching favorable levels
- Price action suggests potential upward movement

**Caution Points:**
- Signal strength is moderate - consider smaller position size
- Wait for confirmation from additional indicators

**Recommendation:**
Consider a gradual entry strategy, building position over time.
""",
            
            'sell_high_confidence': """
📉 **Strong Sell Signal Detected**

With {confidence:.1f}% confidence, indicators suggest taking profits or reducing exposure.

**Warning Signs:**
- RSI at {rsi:.1f} indicates overbought conditions
- MACD showing bearish divergence
- Volume patterns suggest distribution phase

**Risk Management:**
- Consider taking partial profits to lock in gains
- Tighten stop-losses to protect remaining position

**Action Steps:**
Exit or reduce position size, especially if holding significant gains.
""",
            
            'sell_moderate_confidence': """
⚠️ **Moderate Sell Signal**

Analysis indicates potential downside risk with {confidence:.1f}% confidence.

**Concerns:**
- Technical indicators showing weakness
- Momentum appears to be fading
- Consider defensive positioning

**Suggestions:**
- Review position size and risk exposure
- Consider partial profit-taking
""",
            
            'hold': """
⏸️ **Hold Position**

Current analysis suggests maintaining existing position with {confidence:.1f}% confidence.

**Market Assessment:**
- RSI at {rsi:.1f} is in neutral territory
- No clear directional signal from MACD
- Price consolidating near moving averages

**Strategy:**
- Monitor for clearer signals before making changes
- Maintain current stop-loss levels
- Be patient for better entry/exit opportunities

**Learn More:**
Consolidation periods often precede significant moves. Use this time to research and prepare for the next opportunity.
"""
        }
    
    def _enhance_explanation(self, explanation: str, signal_data: Dict) -> str:
        """Enhance the LLM explanation with additional formatting and data"""
        
        # Add confidence badge
        confidence = signal_data.get('confidence', 0.5) * 100
        confidence_badge = "🟢" if confidence > 70 else "🟡" if confidence > 50 else "🔴"
        
        # Add header
        enhanced = f"{confidence_badge} **AI Analysis - {confidence:.1f}% Confidence**\n\n"
        enhanced += explanation
        
        # Add footer with metadata
        enhanced += f"\n\n---\n*Analysis generated at {datetime.now().strftime('%H:%M:%S')} "
        enhanced += f"using {self.current_model or 'template engine'}*"
        
        return enhanced
    
    def _generate_cache_key(self, signal_data: Dict) -> str:
        """Generate a cache key for the signal data"""
        # Create a stable hash of the relevant data
        key_data = {
            'signal': signal_data.get('signal'),
            'confidence': round(signal_data.get('confidence', 0), 2),
            'symbol': signal_data.get('symbol'),
            'indicators': {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in signal_data.get('indicators', {}).items()
            }
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if still valid"""
        if cache_key in self.response_cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.config['cache_ttl']:
                return self.response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache a response with timestamp"""
        self.response_cache[cache_key] = response
        self.cache_timestamps[cache_key] = time.time()
        
        # Clean old cache entries
        self._clean_cache()
    
    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.config['cache_ttl']
        ]
        
        for key in expired_keys:
            del self.response_cache[key]
            del self.cache_timestamps[key]
    
    def generate_batch_explanations(self, signal_list: List[Dict]) -> List[str]:
        """Generate explanations for multiple signals efficiently"""
        explanations = []
        
        for signal_data in signal_list:
            explanation = self.generate_explanation(signal_data)
            explanations.append(explanation)
            
            # Small delay to avoid overwhelming the API
            if self.is_available:
                time.sleep(0.5)
        
        return explanations
    
    def get_educational_content(self, topic: str) -> str:
        """Get educational content about specific topics"""
        educational_content = {
            'rsi': """
**Understanding RSI (Relative Strength Index)**

RSI measures momentum on a scale of 0-100:
- Above 70: Potentially overbought (price may fall)
- Below 30: Potentially oversold (price may rise)
- 50: Neutral momentum

Tips: RSI works best in ranging markets. In strong trends, it can stay overbought/oversold for extended periods.
""",
            'macd': """
**Understanding MACD (Moving Average Convergence Divergence)**

MACD shows the relationship between two moving averages:
- MACD Line: 12-day EMA minus 26-day EMA
- Signal Line: 9-day EMA of MACD
- Histogram: Difference between MACD and Signal

When MACD crosses above Signal = Bullish
When MACD crosses below Signal = Bearish
""",
            'risk_management': """
**Essential Risk Management Rules**

1. Never risk more than 2% on a single trade
2. Always use stop-losses
3. Diversify across sectors
4. Size positions based on confidence
5. Keep emotions in check

Remember: Protecting capital is more important than making profits.
"""
        }
        
        return educational_content.get(topic, "Educational content coming soon...")
    
    def get_status(self) -> Dict:
        """Get current status of the Ollama integration"""
        return {
            'available': self.is_available,
            'model': self.current_model,
            'base_url': self.base_url,
            'cache_size': len(self.response_cache),
            'models_available': self._get_available_models()
        }
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                return [
                    model['name'].split(':')[0] 
                    for model in response.json().get('models', [])
                ]
        except:
            pass
        return []
    
    def test_connection(self) -> bool:
        """Test if Ollama is properly configured and working"""
        try:
            test_data = {
                'signal': 'BUY',
                'confidence': 0.75,
                'symbol': 'TEST',
                'indicators': {
                    'RSI': 35,
                    'MACD': 0.5,
                    'SMA_20': 100
                }
            }
            
            explanation = self.generate_explanation(test_data, use_cache=False)
            return len(explanation) > 50
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

# Create singleton instance
_ollama_instance = None

def get_ollama_explainer() -> OllamaExplainer:
    """Get or create Ollama explainer instance"""
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaExplainer()
    return _ollama_instance