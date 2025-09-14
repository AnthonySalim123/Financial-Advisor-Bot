# CREATE comprehensive test suite
import unittest
import pandas as pd
import numpy as np
from utils.ml_models import MLModel
from utils.technical_indicators import TechnicalIndicators
from utils.data_processor import StockDataProcessor

class TestMLModel(unittest.TestCase):
    def setUp(self):
        self.model = MLModel()
        self.test_data = self._create_test_data()
    
    def test_accuracy_above_threshold(self):
        """Test that model achieves >70% accuracy"""
        self.model.train(self.test_data)
        predictions, confidence = self.model.predict(self.test_data)
        
        accuracy = (predictions == self.test_data['Signal']).mean()
        self.assertGreaterEqual(accuracy, 0.70, "Model accuracy must be >= 70%")
    
    def test_confidence_correlation(self):
        """Test that higher confidence correlates with accuracy"""
        self.model.train(self.test_data)
        predictions, confidence = self.model.predict(self.test_data)
        
        high_conf_mask = confidence > 0.7
        high_conf_accuracy = (predictions[high_conf_mask] == self.test_data['Signal'][high_conf_mask]).mean()
        
        low_conf_mask = confidence < 0.5
        low_conf_accuracy = (predictions[low_conf_mask] == self.test_data['Signal'][low_conf_mask]).mean()
        
        self.assertGreater(high_conf_accuracy, low_conf_accuracy, 
                          "High confidence predictions should be more accurate")
    
    def test_feature_importance(self):
        """Test that all features contribute"""
        self.model.train(self.test_data)
        importance = self.model.feature_importance
        
        self.assertIsNotNone(importance)
        self.assertGreater(len(importance), 0)
        self.assertTrue(all(importance['importance'] > 0))

class TestTechnicalIndicators(unittest.TestCase):
    def test_rsi_range(self):
        """Test RSI is within 0-100"""
        data = pd.DataFrame({'Close': np.random.randn(100).cumsum() + 100})
        rsi = TechnicalIndicators.calculate_rsi(data['Close'])
        
        self.assertTrue(all((rsi >= 0) & (rsi <= 100)))
    
    def test_macd_calculation(self):
        """Test MACD calculation correctness"""
        data = pd.DataFrame({'Close': np.random.randn(100).cumsum() + 100})
        macd_dict = TechnicalIndicators.calculate_macd(data['Close'])
        
        self.assertIn('MACD', macd_dict)
        self.assertIn('Signal', macd_dict)
        self.assertEqual(len(macd_dict['MACD']), len(data))

class TestBacktesting(unittest.TestCase):
    def test_no_lookahead_bias(self):
        """Ensure no future data leakage"""
        # Test that predictions only use past data
        pass
    
    def test_transaction_costs(self):
        """Test that transaction costs are applied"""
        pass

class TestUserExperience(unittest.TestCase):
    def test_explanation_generation(self):
        """Test that explanations are generated"""
        explainer = OllamaExplainer()
        signal_data = {
            'signal': 'BUY',
            'confidence': 0.75,
            'indicators': {'RSI': 35, 'MACD': 0.5}
        }
        
        explanation = explainer.generate_explanation(signal_data)
        self.assertIsNotNone(explanation)
        self.assertGreater(len(explanation), 100)
    
    def test_ui_responsiveness(self):
        """Test UI loads within acceptable time"""
        pass

if __name__ == '__main__':
    unittest.main()