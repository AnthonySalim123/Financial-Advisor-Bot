"""
Machine Learning Models Module
Implements Random Forest and other ML models for stock prediction
Based on the prototype implementation from the PPR
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, Tuple, Optional, List
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StockPredictionModel:
    """
    Machine Learning model for stock prediction
    Implements the hybrid approach from the PPR
    """
    
    def __init__(self, model_type='classification', config=None):
        """
        Initialize the prediction model
        
        Args:
            model_type: 'classification' for buy/sell/hold signals, 'regression' for price prediction
            config: Model configuration dictionary
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_trained = False
        
        # Default configuration based on PPR
        self.config = config or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'test_size': 0.2,
            'lookback_period': 60,
            'prediction_horizon': 5,
            'buy_threshold': 0.02,
            'sell_threshold': -0.02
        }
        
        # Feature columns from PPR
        self.feature_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
            'Stoch_K', 'Volume_Ratio', 'ATR', 'SMA_20', 'SMA_50',
            'Price_vs_SMA20', 'Price_vs_SMA50'
        ]
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                min_samples_split=self.config['min_samples_split'],
                min_samples_leaf=self.config['min_samples_leaf'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                min_samples_split=self.config['min_samples_split'],
                min_samples_leaf=self.config['min_samples_leaf'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            DataFrame with features ready for ML
        """
        features = df.copy()
        
        # Add additional features
        features['Returns'] = features['Close'].pct_change()
        features['Log_Returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['High_Low_Ratio'] = features['High'] / features['Low']
        features['Close_Open_Ratio'] = features['Close'] / features['Open']
        
        # Add lagged features
        for i in [1, 2, 3, 5]:
            features[f'Returns_Lag_{i}'] = features['Returns'].shift(i)
            features[f'Volume_Lag_{i}'] = features['Volume'].shift(i)
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            features[f'Returns_Mean_{window}'] = features['Returns'].rolling(window).mean()
            features[f'Returns_Std_{window}'] = features['Returns'].rolling(window).std()
            features[f'Volume_Mean_{window}'] = features['Volume'].rolling(window).mean()
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target variable for prediction
        Based on PPR implementation
        
        Args:
            df: DataFrame with price data
        
        Returns:
            Series with target variable
        """
        if self.model_type == 'classification':
            # Calculate future returns
            future_returns = df['Close'].pct_change(self.config['prediction_horizon']).shift(-self.config['prediction_horizon'])
            
            # Create signals: Buy=1, Hold=0, Sell=-1
            target = pd.Series(index=df.index, data=0)
            target[future_returns > self.config['buy_threshold']] = 1
            target[future_returns < self.config['sell_threshold']] = -1
            
            return target
        else:
            # For regression, predict future price
            return df['Close'].shift(-self.config['prediction_horizon'])
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the model on historical data
        
        Args:
            df: DataFrame with features and price data
        
        Returns:
            Dictionary with training metrics
        """
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Get available feature columns
            available_features = [col for col in self.feature_columns if col in features_df.columns]
            
            # Add any additional features that exist
            extra_features = [col for col in features_df.columns 
                            if ('Returns' in col or 'Volume' in col or 'Ratio' in col) 
                            and col not in available_features]
            available_features.extend(extra_features[:10])  # Limit extra features
            
            X = features_df[available_features].dropna()
            y = self.create_target_variable(features_df).loc[X.index].dropna()
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            if len(X) < 100:
                raise ValueError("Insufficient data for training (need at least 100 samples)")
            
            # Time series split (maintain chronological order)
            split_index = int(len(X) * (1 - self.config['test_size']))
            X_train = X[:split_index]
            X_test = X[split_index:]
            y_train = y[:split_index]
            y_test = y[split_index:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Get feature importance
            self.feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            if self.model_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            else:
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            
            # Store the feature columns used
            self.used_features = available_features
            
            logger.info(f"Model trained successfully with {len(available_features)} features")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {'error': str(e)}
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with features
        
        Returns:
            Tuple of (predictions, probabilities/confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Use only the features used during training
            X = features_df[self.used_features].dropna()
            
            if len(X) == 0:
                raise ValueError("No valid samples for prediction")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Get probabilities/confidence
            if self.model_type == 'classification':
                probabilities = self.model.predict_proba(X_scaled)
                confidence = np.max(probabilities, axis=1)
            else:
                # For regression, use prediction intervals as confidence
                predictions_all = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                confidence = 1 - (np.std(predictions_all, axis=0) / np.mean(predictions_all, axis=0))
                confidence = np.clip(confidence, 0, 1)
            
            return predictions, confidence
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([]), np.array([])
    
    def predict_next(self, df: pd.DataFrame) -> Dict:
        """
        Predict the next signal/price based on latest data
        Similar to PPR implementation
        
        Args:
            df: DataFrame with latest data
        
        Returns:
            Dictionary with prediction details
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Get the latest prediction
            predictions, confidence = self.predict(df)
            
            if len(predictions) == 0:
                return {'error': 'No valid prediction'}
            
            latest_prediction = predictions[-1]
            latest_confidence = confidence[-1]
            
            # Get latest indicators for explanation
            latest_indicators = df.iloc[-1]
            
            # Create result based on model type
            if self.model_type == 'classification':
                signal_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}
                result = {
                    'signal': signal_map.get(int(latest_prediction), 'HOLD'),
                    'confidence': float(latest_confidence),
                    'prediction_raw': int(latest_prediction),
                    'rsi': float(latest_indicators.get('RSI', 0)),
                    'macd': float(latest_indicators.get('MACD', 0)),
                    'bb_position': float(latest_indicators.get('BB_Position', 0.5)),
                    'volume_ratio': float(latest_indicators.get('Volume_Ratio', 1)),
                    'feature_importance': self.feature_importance.head(5).to_dict('records') if self.feature_importance is not None else []
                }
            else:
                current_price = df['Close'].iloc[-1]
                price_change = latest_prediction - current_price
                price_change_pct = (price_change / current_price) * 100
                
                result = {
                    'predicted_price': float(latest_prediction),
                    'current_price': float(current_price),
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct),
                    'confidence': float(latest_confidence),
                    'horizon_days': self.config['prediction_horizon']
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_next: {e}")
            return {'error': str(e)}
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """
        Backtest the strategy on historical data
        
        Args:
            df: DataFrame with historical data
            initial_capital: Starting capital for backtesting
        
        Returns:
            Dictionary with backtest results
        """
        try:
            # Prepare data
            features_df = self.prepare_features(df)
            
            # Use time series split for walk-forward analysis
            tscv = TimeSeriesSplit(n_splits=5)
            
            all_returns = []
            all_trades = []
            
            for train_index, test_index in tscv.split(features_df):
                # Get train and test data
                train_data = features_df.iloc[train_index]
                test_data = features_df.iloc[test_index]
                
                # Train on this fold
                self.train(train_data)
                
                # Predict on test data
                predictions, confidence = self.predict(test_data)
                
                if len(predictions) == 0:
                    continue
                
                # Simulate trading
                capital = initial_capital
                position = 0
                trades = []
                
                for i in range(len(predictions)):
                    signal = predictions[i]
                    price = test_data['Close'].iloc[i]
                    
                    if self.model_type == 'classification':
                        if signal == 1 and position == 0:  # Buy
                            position = capital / price
                            capital = 0
                            trades.append({'type': 'BUY', 'price': price})
                        elif signal == -1 and position > 0:  # Sell
                            capital = position * price
                            position = 0
                            trades.append({'type': 'SELL', 'price': price})
                
                # Close any open position
                if position > 0:
                    capital = position * test_data['Close'].iloc[-1]
                
                # Calculate returns
                returns = (capital - initial_capital) / initial_capital
                all_returns.append(returns)
                all_trades.extend(trades)
            
            # Calculate metrics
            if all_returns:
                total_return = np.mean(all_returns)
                sharpe_ratio = np.mean(all_returns) / np.std(all_returns) if np.std(all_returns) > 0 else 0
                win_rate = len([r for r in all_returns if r > 0]) / len(all_returns)
                max_drawdown = np.min(all_returns)
                
                backtest_results = {
                    'total_return': float(total_return),
                    'sharpe_ratio': float(sharpe_ratio),
                    'win_rate': float(win_rate),
                    'max_drawdown': float(max_drawdown),
                    'num_trades': len(all_trades),
                    'avg_return': float(np.mean(all_returns)),
                    'std_return': float(np.std(all_returns))
                }
            else:
                backtest_results = {'error': 'No valid backtest results'}
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'config': self.config,
            'used_features': self.used_features,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.config = model_data['config']
        self.used_features = model_data['used_features']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_explanation(self, prediction_result: Dict) -> str:
        """
        Generate human-readable explanation for prediction
        Based on PPR requirements
        
        Args:
            prediction_result: Dictionary from predict_next()
        
        Returns:
            String explanation
        """
        if 'error' in prediction_result:
            return f"Unable to generate prediction: {prediction_result['error']}"
        
        if self.model_type == 'classification':
            signal = prediction_result['signal']
            confidence = prediction_result['confidence'] * 100
            rsi = prediction_result['rsi']
            macd = prediction_result['macd']
            
            if signal == 'BUY':
                explanation = f"""
                📈 **Strong Buy Signal Detected**
                
                The AI model recommends a **BUY** position with {confidence:.1f}% confidence.
                
                **Key Indicators:**
                - RSI at {rsi:.1f} indicates the stock is {'oversold' if rsi < 30 else 'not overbought'}
                - MACD at {macd:.3f} shows {'bullish momentum' if macd > 0 else 'potential reversal'}
                - Technical analysis suggests upward price movement
                
                **Risk Level:** {'Low' if confidence > 75 else 'Moderate' if confidence > 60 else 'High'}
                """
            elif signal == 'SELL':
                explanation = f"""
                📉 **Sell Signal Detected**
                
                The AI model recommends a **SELL** position with {confidence:.1f}% confidence.
                
                **Key Indicators:**
                - RSI at {rsi:.1f} indicates {'overbought conditions' if rsi > 70 else 'weakening momentum'}
                - MACD at {macd:.3f} shows {'bearish divergence' if macd < 0 else 'momentum loss'}
                - Technical analysis suggests downward price movement
                
                **Risk Level:** {'High' if confidence > 75 else 'Moderate'}
                """
            else:
                explanation = f"""
                ⏸️ **Hold Position Recommended**
                
                The AI model recommends **HOLDING** with {confidence:.1f}% confidence.
                
                **Key Indicators:**
                - RSI at {rsi:.1f} shows neutral momentum
                - MACD at {macd:.3f} indicates consolidation
                - Mixed signals suggest waiting for clearer direction
                
                **Risk Level:** Low
                """
        else:
            explanation = f"""
            **Price Prediction**
            
            Predicted Price: ${prediction_result['predicted_price']:.2f}
            Current Price: ${prediction_result['current_price']:.2f}
            Expected Change: {prediction_result['price_change_pct']:.2f}%
            Time Horizon: {prediction_result['horizon_days']} days
            Confidence: {prediction_result['confidence']*100:.1f}%
            """
        
        return explanation

# Factory function to create model instances
def create_prediction_model(model_type='classification', config=None):
    """
    Factory function to create prediction models
    
    Args:
        model_type: Type of model ('classification' or 'regression')
        config: Model configuration
    
    Returns:
        StockPredictionModel instance
    """
    return StockPredictionModel(model_type, config)