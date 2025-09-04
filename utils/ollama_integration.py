# CREATE this new file for actual LLM integration
import requests
import json
from typing import Dict
import subprocess
import time

class OllamaExplainer:
    """Actual Ollama integration for natural language explanations"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "llama2"  # or "mistral" for better performance
        self._ensure_ollama_running()
    
    def _ensure_ollama_running(self):
        """Start Ollama if not running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                subprocess.Popen(["ollama", "serve"])
                time.sleep(5)
        except:
            # Start Ollama
            subprocess.Popen(["ollama", "serve"])
            time.sleep(5)
            # Pull model if not exists
            subprocess.run(["ollama", "pull", self.model])
    
    def generate_explanation(self, signal_data: Dict) -> str:
        """Generate explanation using actual LLM"""
        
        prompt = f"""
        You are a financial advisor explaining a trading recommendation to a retail investor.
        
        Signal: {signal_data['signal']}
        Confidence: {signal_data['confidence']*100:.1f}%
        
        Technical Indicators:
        - RSI: {signal_data['indicators']['RSI']:.1f}
        - MACD: {signal_data['indicators']['MACD']:.3f}
        - Volume Ratio: {signal_data['indicators']['Volume_Ratio']:.2f}
        
        Feature Importance:
        {json.dumps(signal_data.get('feature_importance', {}), indent=2)}
        
        Market Context:
        - Volatility: {signal_data.get('volatility', 'Normal')}
        - Trend: {signal_data.get('trend', 'Neutral')}
        
        Please provide:
        1. A clear explanation of WHY this signal was generated
        2. What the key indicators are telling us
        3. Risk factors to consider
        4. Suggested action steps
        5. Educational insight about the indicators
        
        Keep the explanation under 200 words and use simple language.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "max_tokens": 300
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                # Fallback to enhanced template
                return self._generate_template_explanation(signal_data)
                
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._generate_template_explanation(signal_data)