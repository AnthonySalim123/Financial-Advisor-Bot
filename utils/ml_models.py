"""
Machine Learning Models Module - Enhanced Version
Implements Ensemble ML models for stock prediction with 70%+ accuracy target
Based on the prototype implementation from the PPR
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, GradientBoostingClassifier
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
    Enhanced Machine Learning model for stock prediction
    Implements the hybrid approach from the PPR with ensemble methods
    """
    
    def __init__(self, model_type='classification', config=None):
        """
        Initialize the prediction model with enhanced configuration
        
        Args:
            model_type: 'classification' for buy/sell/hold signals, 'regression' for price prediction
            config: Model configuration dictionary
        """
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()  # Better for financial data with outliers
        self.feature_importance = None
        self.is_trained = False
        
        # Enhanced configuration for better accuracy
        self.config = config or {
            'n_estimators': 200,  # Increased from 100
            'max_depth': 15,      # Increased from 10
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'test_size': 0.2,
            'lookback_period': 60,
            'prediction_horizon': 5,
            'buy_threshold': 0.02,
            'sell_threshold': -0.02,
            'use_ensemble': True  # New parameter for ensemble
        }
        
        # Expanded feature columns for better predictions
        self.feature_columns = [
            # Technical Indicators
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'BB_Width',
            'Stoch_K', 'Stoch_D',
            'ATR', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'OBV', 'VWAP',
            
            # Price-based features
            'Price_vs_SMA20', 'Price_vs_SMA50',
            'Support_Distance', 'Resistance_Distance',
            
            # Volume features
            'Volume_Ratio', 'Volume_SMA', 'Volume_Trend',
            
            # Market microstructure
            'High_Low_Spread', 'Close_Open_Spread',
            'Upper_Shadow', 'Lower_Shadow',
            
            # Momentum features
            'Price_Momentum', 'RSI_Signal', 'MACD_Cross',
            
            # Volatility features
            'Volatility_20', 'Returns_Skew', 'Returns_Kurt',
            
            # Pattern features
            'Candle_Pattern', 'Trend_Strength'
        ]
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ensemble ML model for better accuracy"""
        if self.model_type == 'classification':
            if self.config['use_ensemble']:
                # Create ensemble for better accuracy
                rf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                
                gb = GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.01,
                    subsample=0.8,
                    random_state=42
                )
                
                # Try to use XGBoost if available
                try:
                    from xgboost import XGBClassifier
                    xgb = XGBClassifier(
                        n_estimators=200,
                        max_depth=10,
                        learning_rate=0.01,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='mlogloss'
                    )
                    
                    self.model = VotingClassifier(
                        estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
                        voting='soft',
                        weights=[0.4, 0.3, 0.3]
                    )
                except ImportError:
                    logger.warning("XGBoost not available, using RF+GB ensemble")
                    self.model = VotingClassifier(
                        estimators=[('rf', rf), ('gb', gb)],
                        voting='soft',
                        weights=[0.6, 0.4]
                    )
            else:
                # Single model fallback
                self.model = RandomForestClassifier(
                    n_estimators=self.config['n_estimators'],
                    max_depth=self.config['max_depth'],
                    min_samples_split=self.config['min_samples_split'],
                    min_samples_leaf=self.config['min_samples_leaf'],
                    random_state=self.config['random_state'],
                    n_jobs=-1
                )
        else:
            # Regression model
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
        Enhanced feature engineering for 70%+ accuracy
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            DataFrame with engineered features
        """
        features = df.copy()
        
        # Basic returns and price features
        features['Returns'] = features['Close'].pct_change()
        features['Log_Returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['High_Low_Ratio'] = features['High'] / features['Low']
        features['Close_Open_Ratio'] = features['Close'] / features['Open']
        
        # Support and Resistance
        features['Support_Distance'] = self._calculate_support_distance(features)
        features['Resistance_Distance'] = self._calculate_resistance_distance(features)
        
        # Advanced technical features
        if 'BB_Upper' in features.columns and 'BB_Lower' in features.columns:
            features['BB_Width'] = (features['BB_Upper'] - features['BB_Lower']) / features['Close']
        
        if 'RSI' in features.columns:
            features['RSI_Signal'] = np.where(features['RSI'] < 30, 1, 
                                            np.where(features['RSI'] > 70, -1, 0))
        
        if 'MACD' in features.columns and 'MACD_Signal' in features.columns:
            features['MACD_Cross'] = np.where(features['MACD'] > features['MACD_Signal'], 1, -1)
        
        # Volatility features
        features['Volatility_20'] = features['Returns'].rolling(20).std() * np.sqrt(252)
        features['Returns_Skew'] = features['Returns'].rolling(20).skew()
        features['Returns_Kurt'] = features['Returns'].rolling(20).kurt()
        
        # Volume features
        if 'Volume' in features.columns:
            features['Volume_SMA'] = features['Volume'].rolling(20).mean()
            features['Volume_Ratio'] = features['Volume'] / features['Volume_SMA']
            features['Volume_Trend'] = features['Volume'].pct_change(5)
        
        # Price momentum
        features['Price_Momentum'] = features['Close'] / features['Close'].shift(10) - 1
        
        # Candlestick patterns
        features['Upper_Shadow'] = features['High'] - features[['Close', 'Open']].max(axis=1)
        features['Lower_Shadow'] = features[['Close', 'Open']].min(axis=1) - features['Low']
        features['Candle_Pattern'] = self._identify_candle_patterns(features)
        
        # Trend strength
        features['Trend_Strength'] = self._calculate_trend_strength(features)
        
        # Lagged features for time series
        for i in [1, 2, 3, 5, 10]:
            features[f'Returns_Lag_{i}'] = features['Returns'].shift(i)
            if 'Volume' in features.columns:
                features[f'Volume_Lag_{i}'] = features['Volume'].shift(i)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            features[f'Returns_Mean_{window}'] = features['Returns'].rolling(window).mean()
            features[f'Returns_Std_{window}'] = features['Returns'].rolling(window).std()
            if 'Volume' in features.columns:
                features[f'Volume_Mean_{window}'] = features['Volume'].rolling(window).mean()
        
        # Market regime features
        features['Market_Regime'] = self._detect_market_regime(features)
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_support_distance(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate distance from support level"""
        support = df['Low'].rolling(window).min()
        return ((df['Close'] - support) / df['Close']).fillna(0)
    
    def _calculate_resistance_distance(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate distance from resistance level"""
        resistance = df['High'].rolling(window).max()
        return ((resistance - df['Close']) / df['Close']).fillna(0)
    
    def _identify_candle_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Identify candlestick patterns"""
        patterns = pd.Series(index=df.index, data=0)
        
        # Doji
        doji = (abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1)
        patterns[doji] = 1
        
        # Hammer
        hammer = ((df['Close'] - df['Low']) > 2 * abs(df['Close'] - df['Open'])) & \
                ((df['High'] - df['Close']) < abs(df['Close'] - df['Open']))
        patterns[hammer] = 2
        
        # Shooting Star
        shooting_star = ((df['High'] - df['Close']) > 2 * abs(df['Close'] - df['Open'])) & \
                       ((df['Close'] - df['Low']) < abs(df['Close'] - df['Open']))
        patterns[shooting_star] = -2
        
        return patterns.fillna(0)
    
    def _calculate_trend_strength(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate trend strength using ADX concept"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window).mean()
        
        return (atr / df['Close']).fillna(0)
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime (trending/ranging)"""
        sma_20 = df['Close'].rolling(20).mean()
        sma_50 = df['Close'].rolling(50).mean()
        
        regime = pd.Series(index=df.index, data=0)
        regime[df['Close'] > sma_20] = 1  # Bullish
        regime[df['Close'] < sma_20] = -1  # Bearish
        
        return regime.fillna(0)
    
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
        Train the enhanced model on historical data
        
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
            
            # Add dynamic features
            dynamic_features = [col for col in features_df.columns 
                              if any(x in col for x in ['Returns_Lag', 'Returns_Mean', 'Volume_Lag', 'Volume_Mean'])
                              and col not in available_features]
            available_features.extend(dynamic_features[:20])  # Limit to avoid overfitting
            
            # Prepare X and y
            X = features_df[available_features].dropna()
            y = self.create_target_variable(features_df).loc[X.index].dropna()
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            if len(X) < 100:
                raise ValueError("Insufficient data for training")
            
            # Time series split
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
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                # For ensemble, average the importances
                importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
            
            self.feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': importances
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
                    'test_samples': len(X_test),
                    'feature_count': len(available_features)
                }
                
                # Cross-validation for robust evaluation
                if len(X) > 200:
                    cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=3)
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
            else:
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            
            # Store used features
            self.used_features = available_features
            
            logger.info(f"Model trained successfully: Accuracy={metrics.get('accuracy', 0)*100:.1f}%")
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
                # For regression, use prediction intervals
                predictions_all = np.array([est.predict(X_scaled) for est in self.model.estimators_])
                confidence = 1 - (np.std(predictions_all, axis=0) / (np.mean(predictions_all, axis=0) + 1e-10))
                confidence = np.clip(confidence, 0, 1)
            
            return predictions, confidence
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([]), np.array([])
    
    def predict_next(self, df: pd.DataFrame) -> Dict:
        """
        Predict the next signal/price based on latest data
        
        Args:
            df: DataFrame with latest data
        
        Returns:
            Dictionary with prediction details
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Get predictions
            predictions, confidence = self.predict(df)
            
            if len(predictions) == 0:
                return {'error': 'No valid prediction'}
            
            latest_prediction = predictions[-1]
            latest_confidence = confidence[-1]
            
            # Get latest indicators
            latest_indicators = df.iloc[-1]
            
            if self.model_type == 'classification':
                signal_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}
                result = {
                    'signal': signal_map.get(int(latest_prediction), 'HOLD'),
                    'confidence': float(latest_confidence),
                    'prediction_raw': int(latest_prediction),
                    'rsi': float(latest_indicators.get('RSI', 50)),
                    'macd': float(latest_indicators.get('MACD', 0)),
                    'bb_position': float(latest_indicators.get('BB_Position', 0.5)),
                    'volume_ratio': float(latest_indicators.get('Volume_Ratio', 1)),
                    'feature_importance': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else []
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
        Enhanced backtest with realistic constraints
        
        Args:
            df: DataFrame with historical data
            initial_capital: Starting capital
        
        Returns:
            Dictionary with backtest results
        """
        try:
            # Prepare data
            features_df = self.prepare_features(df)
            
            # Use walk-forward analysis
            window_size = 252  # 1 year training window
            step_size = 21    # Retrain monthly
            
            results = {
                'trades': [],
                'equity_curve': [],
                'returns': []
            }
            
            capital = initial_capital
            position = None
            
            for i in range(window_size, len(features_df), step_size):
                # Train on rolling window
                train_data = features_df.iloc[i-window_size:i]
                
                if len(train_data) < 100:
                    continue
                
                # Train model
                train_metrics = self.train(train_data)
                
                if 'error' in train_metrics:
                    continue
                
                # Predict on next period
                test_end = min(i + step_size, len(features_df))
                test_data = features_df.iloc[i:test_end]
                
                predictions, confidence = self.predict(test_data)
                
                # Simulate trading
                for j, (pred, conf) in enumerate(zip(predictions, confidence)):
                    if j >= len(test_data):
                        break
                        
                    date = test_data.index[j]
                    price = test_data['Close'].iloc[j]
                    
                    # Only trade high confidence signals
                    if conf < 0.65:
                        continue
                    
                    if self.model_type == 'classification':
                        if pred == 1 and position is None:  # Buy
                            position = capital / price
                            capital = 0
                            results['trades'].append({
                                'date': date,
                                'type': 'BUY',
                                'price': price,
                                'shares': position,
                                'confidence': conf
                            })
                        elif pred == -1 and position is not None:  # Sell
                            capital = position * price
                            results['trades'].append({
                                'date': date,
                                'type': 'SELL',
                                'price': price,
                                'shares': position,
                                'pnl': capital - initial_capital,
                                'confidence': conf
                            })
                            position = None
                    
                    # Track equity
                    current_value = capital if position is None else position * price
                    results['equity_curve'].append(current_value)
                    
                    if len(results['equity_curve']) > 1:
                        ret = (current_value - results['equity_curve'][-2]) / results['equity_curve'][-2]
                        results['returns'].append(ret)
            
            # Calculate metrics
            if results['returns']:
                returns_series = pd.Series(results['returns'])
                final_value = results['equity_curve'][-1] if results['equity_curve'] else initial_capital
                
                backtest_metrics = {
                    'total_return': ((final_value - initial_capital) / initial_capital) * 100,
                    'sharpe_ratio': (returns_series.mean() / returns_series.std()) * np.sqrt(252) if returns_series.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(results['equity_curve']),
                    'win_rate': len([t for t in results['trades'] if t.get('pnl', 0) > 0]) / len(results['trades']) * 100 if results['trades'] else 0,
                    'num_trades': len(results['trades']),
                    'final_capital': final_value
                }
            else:
                backtest_metrics = {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'num_trades': 0,
                    'final_capital': initial_capital
                }
            
            return backtest_metrics
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
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
        """Generate enhanced explanation for prediction"""
        if 'error' in prediction_result:
            return f"Unable to generate prediction: {prediction_result['error']}"
        
        if self.model_type == 'classification':
            signal = prediction_result['signal']
            confidence = prediction_result['confidence'] * 100
            rsi = prediction_result.get('rsi', 50)
            macd = prediction_result.get('macd', 0)
            
            # Get top features
            top_features = ""
            if prediction_result.get('feature_importance'):
                top_3 = prediction_result['feature_importance'][:3]
                top_features = "\n".join([f"- {f['feature']}: {f['importance']*100:.1f}% importance" for f in top_3])
            
            if signal == 'BUY':
                explanation = f"""
                📈 **Strong Buy Signal Detected**
                
                The enhanced AI model recommends a **BUY** position with {confidence:.1f}% confidence.
                
                **Key Indicators:**
                - RSI at {rsi:.1f} {'(oversold)' if rsi < 30 else '(neutral)' if rsi < 70 else '(overbought)'}
                - MACD at {macd:.3f} {'(bullish)' if macd > 0 else '(bearish)'}
                - Multiple technical indicators confirm upward momentum
                
                **Top Contributing Factors:**
                {top_features}
                
                **Risk Assessment:** {'Low-Moderate' if confidence > 75 else 'Moderate' if confidence > 65 else 'Moderate-High'}
                
                **Recommended Action:**
                - Enter position with 2-5% of portfolio
                - Set stop loss at 5% below entry
                - Target profit at 10-15% above entry
                """
            elif signal == 'SELL':
                explanation = f"""
                📉 **Sell Signal Detected**
                
                The AI model recommends a **SELL** position with {confidence:.1f}% confidence.
                
                **Key Indicators:**
                - RSI at {rsi:.1f} {'(oversold)' if rsi < 30 else '(neutral)' if rsi < 70 else '(overbought)'}
                - MACD at {macd:.3f} showing {'bearish divergence' if macd < 0 else 'weakening momentum'}
                - Technical indicators suggest downward pressure
                
                **Top Contributing Factors:**
                {top_features}
                
                **Risk Level:** {'High' if confidence > 75 else 'Moderate-High'}
                
                **Recommended Action:**
                - Consider taking profits or reducing position
                - Tighten stop losses to protect gains
                - Wait for better entry points
                """
            else:
                explanation = f"""
                ⏸️ **Hold Position Recommended**
                
                The AI model recommends **HOLDING** with {confidence:.1f}% confidence.
                
                **Current Status:**
                - RSI at {rsi:.1f} in neutral zone
                - MACD at {macd:.3f} showing consolidation
                - Mixed signals suggest waiting for clearer direction
                
                **Top Contributing Factors:**
                {top_features}
                
                **Recommended Action:**
                - Maintain current position
                - Wait for stronger signals
                - Monitor for breakout patterns
                """
        else:
            explanation = f"""
            **Price Prediction Analysis**
            
            Predicted Price: ${prediction_result['predicted_price']:.2f}
            Current Price: ${prediction_result['current_price']:.2f}
            Expected Change: {prediction_result['price_change_pct']:.2f}%
            Time Horizon: {prediction_result['horizon_days']} days
            Confidence: {prediction_result['confidence']*100:.1f}%
            
            **Recommendation:** {'Buy' if prediction_result['price_change_pct'] > 2 else 'Sell' if prediction_result['price_change_pct'] < -2 else 'Hold'}
            """
        
        return explanation

# Create factory function
def create_prediction_model(model_type='classification', config=None):
    """Factory function to create prediction models"""
    return StockPredictionModel(model_type, config)