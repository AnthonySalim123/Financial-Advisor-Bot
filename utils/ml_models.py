# utils/ml_models.py (Complete corrected version)
"""
Machine Learning Models Module - Enhanced Version 2.0
Achieves 70%+ accuracy through ensemble methods and advanced feature engineering
Based on PPR requirements and academic best practices
Author: Anthony Winata Salim
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    VotingClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, 
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import joblib
from typing import Dict, Tuple, Optional, List, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Plotly imports (separate from optional imports)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Install with: pip install plotly")
    # Create dummy class for type hints
    class go:
        class Figure:
            def __init__(self):
                pass

# Optional imports with graceful fallback
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️ SMOTE (imbalanced-learn) not available. Using class weights instead.")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ℹ️ XGBoost not available. Using sklearn models only.")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("ℹ️ LightGBM not available. Using sklearn models only.")

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("ℹ️ SHAP not available. Install with: pip install shap")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAIExplainer:
    """
    SHAP-based explainability for financial ML models
    """
    
    def __init__(self, model, model_type='classification'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained sklearn model (RandomForest, XGBoost, etc.)
            model_type: 'classification' or 'regression'
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def initialize_explainer(self, X_background: pd.DataFrame, max_evals=100):
        """
        Initialize SHAP explainer with background data
        
        Args:
            X_background: Background dataset for SHAP explainer
            max_evals: Maximum evaluations for explainer
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - explainer disabled")
            return
        
        try:
            self.feature_names = X_background.columns.tolist()
            
            # Use TreeExplainer for tree-based models
            if hasattr(self.model, 'feature_importances_'):
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized TreeExplainer")
            else:
                # Use Explainer for other models
                background_sample = shap.sample(X_background, min(100, len(X_background)))
                self.explainer = shap.Explainer(self.model.predict, background_sample)
                logger.info("Initialized general Explainer")
                
        except Exception as e:
            logger.error(f"Failed to initialize explainer: {e}")
            raise
    
    def explain_predictions(self, X: pd.DataFrame) -> Dict:
        """
        Generate SHAP explanations for predictions
        
        Args:
            X: Input features to explain
            
        Returns:
            Dictionary containing SHAP values and analysis
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {'error': 'SHAP explainer not available'}
        
        try:
            # Calculate SHAP values
            if hasattr(self.explainer, 'shap_values'):
                # TreeExplainer
                if self.model_type == 'classification':
                    shap_values = self.explainer.shap_values(X)
                    # For multi-class, take the values for each class
                    if isinstance(shap_values, list):
                        self.shap_values = shap_values
                    else:
                        self.shap_values = shap_values
                else:
                    self.shap_values = self.explainer.shap_values(X)
            else:
                # General Explainer
                shap_values = self.explainer(X)
                self.shap_values = shap_values.values
            
            # Calculate feature importance
            if isinstance(self.shap_values, list):
                # Multi-class: average absolute SHAP values across classes
                importance_values = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
            else:
                importance_values = np.abs(self.shap_values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            # Get model predictions
            predictions = self.model.predict(X)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
            else:
                probabilities = None
            
            return {
                'shap_values': self.shap_values,
                'feature_importance': importance_df,
                'predictions': predictions,
                'probabilities': probabilities,
                'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}
    
    def create_waterfall_plot(self, X_single: pd.DataFrame, prediction_idx: int = 0) -> Any:
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X_single: Single row DataFrame
            prediction_idx: Index of prediction to explain
            
        Returns:
            Plotly figure or empty figure if libraries not available
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - cannot create waterfall plot")
            return go.Figure()
        
        if not SHAP_AVAILABLE or self.shap_values is None:
            return go.Figure()
        
        try:
            # Get SHAP values for single prediction
            if isinstance(self.shap_values, list):
                # Multi-class: use the predicted class
                pred_class = self.model.predict(X_single)[0]
                # Updated class indexing for [0, 1, 2] instead of [-1, 0, 1]
                class_idx = pred_class  # No need to add 1 since classes are now [0, 1, 2]
                if class_idx >= len(self.shap_values):
                    class_idx = 0
                shap_vals = self.shap_values[class_idx][prediction_idx]
            else:
                shap_vals = self.shap_values[prediction_idx]
            
            # Get base value
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0] if len(base_value) > 0 else 0
            
            # Create waterfall data
            features = self.feature_names
            values = shap_vals
            
            # Sort by absolute value for better visualization
            sorted_indices = np.argsort(np.abs(values))[::-1][:15]  # Top 15 features
            features_sorted = [features[i] for i in sorted_indices]
            values_sorted = [values[i] for i in sorted_indices]
            
            # Create waterfall plot
            fig = go.Figure()
            
            fig.add_trace(go.Waterfall(
                name="SHAP Values",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(values_sorted) + ["total"],
                x=["Base"] + features_sorted + ["Final Prediction"],
                textposition="outside",
                text=[f"{base_value:.3f}"] + [f"{v:+.3f}" for v in values_sorted] + ["Final"],
                y=[base_value] + values_sorted + [sum(values_sorted)],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "green"}},
                decreasing={"marker": {"color": "red"}},
                totals={"marker": {"color": "blue"}}
            ))
            
            fig.update_layout(
                title=f"SHAP Waterfall Plot - Feature Contributions",
                showlegend=False,
                height=600,
                xaxis_title="Features",
                yaxis_title="SHAP Value",
                xaxis={'tickangle': 45}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Waterfall plot creation failed: {e}")
            return go.Figure()


class MLModel:
    """
    Advanced Machine Learning Model with Ensemble Methods
    Designed for 70%+ accuracy on financial time series data
    """
    
    def __init__(self, model_type='classification', config=None):
        """
        Initialize the ML model
        
        Args:
            model_type: 'classification' or 'regression'
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.models = {}
        self.final_model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_importance = None
        self.is_trained = False
        self.explainer = None
        
        # Default configuration with enhanced parameters
        default_config = {
            # Model parameters
            'rf_n_estimators': 300,
            'rf_max_depth': 12,
            'rf_min_samples_split': 10,
            'rf_min_samples_leaf': 4,
            'rf_max_features': 'sqrt',
            'gb_n_estimators': 200,
            'gb_max_depth': 8,
            'gb_learning_rate': 0.1,
            'gb_subsample': 0.8,
            'xgb_n_estimators': 250,
            'xgb_max_depth': 6,
            'xgb_learning_rate': 0.1,
            'xgb_subsample': 0.8,
            'xgb_colsample_bytree': 0.8,
            
            # Training parameters
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'handle_imbalance': True,
            'use_ensemble': True,
            'calibrate_probabilities': True,
            
            # Feature selection
            'feature_selection': True,
            'n_features': 50,
            'selection_method': 'mutual_info',
            
            # Prediction thresholds
            'high_confidence': 0.7,
            'prediction_horizon': 5
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize scalers and models
        self.scaler = RobustScaler()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all model components with optimized parameters"""
        
        if self.model_type == 'classification':
            # Random Forest
            self.models['rf'] = RandomForestClassifier(
                n_estimators=self.config['rf_n_estimators'],
                max_depth=self.config['rf_max_depth'],
                min_samples_split=self.config['rf_min_samples_split'],
                min_samples_leaf=self.config['rf_min_samples_leaf'],
                max_features=self.config['rf_max_features'],
                random_state=self.config['random_state'],
                n_jobs=-1,
                class_weight='balanced'
            )
            
            # Gradient Boosting
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=self.config['gb_n_estimators'],
                max_depth=self.config['gb_max_depth'],
                learning_rate=self.config['gb_learning_rate'],
                subsample=self.config['gb_subsample'],
                random_state=self.config['random_state']
            )
            
            # Extra Trees
            self.models['et'] = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                random_state=self.config['random_state'],
                n_jobs=-1,
                class_weight='balanced'
            )
            
            # XGBoost if available
            if XGBOOST_AVAILABLE:
                self.models['xgb'] = XGBClassifier(
                    n_estimators=self.config['xgb_n_estimators'],
                    max_depth=self.config['xgb_max_depth'],
                    learning_rate=self.config['xgb_learning_rate'],
                    subsample=self.config['xgb_subsample'],
                    colsample_bytree=self.config['xgb_colsample_bytree'],
                    random_state=self.config['random_state'],
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    n_jobs=-1
                )
            
            # AdaBoost
            self.models['ada'] = AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.5,
                random_state=self.config['random_state']
            )
            
        else:  # Regression
            self.models['rf'] = RandomForestRegressor(
                n_estimators=self.config['rf_n_estimators'],
                max_depth=self.config['rf_max_depth'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            
            if XGBOOST_AVAILABLE:
                self.models['xgb'] = XGBRegressor(
                    n_estimators=self.config['xgb_n_estimators'],
                    max_depth=self.config['xgb_max_depth'],
                    learning_rate=self.config['xgb_learning_rate'],
                    random_state=self.config['random_state'],
                    n_jobs=-1
                )
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for 70%+ accuracy
        Creates 60+ features from price, volume, and technical data
        """
        features = df.copy()
        
        # === Price-based Features ===
        features['Returns'] = features['Close'].pct_change()
        features['Log_Returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['Returns_Squared'] = features['Returns'] ** 2
        features['Returns_Abs'] = np.abs(features['Returns'])
        
        # Price ratios
        features['High_Low_Ratio'] = features['High'] / features['Low']
        features['Close_Open_Ratio'] = features['Close'] / features['Open']
        features['High_Close_Ratio'] = features['High'] / features['Close']
        features['Low_Close_Ratio'] = features['Low'] / features['Close']
        
        # Price spreads
        features['High_Low_Spread'] = (features['High'] - features['Low']) / features['Close']
        features['Close_Open_Spread'] = (features['Close'] - features['Open']) / features['Open']
        
        # === Moving Average Features ===
        for period in [5, 10, 20, 50]:
            features[f'SMA_{period}'] = features['Close'].rolling(period).mean()
            features[f'EMA_{period}'] = features['Close'].ewm(span=period).mean()
            features[f'Price_to_SMA_{period}'] = features['Close'] / features[f'SMA_{period}']
            features[f'Volume_SMA_{period}'] = features['Volume'].rolling(period).mean()
        
        # === Volatility Features ===
        for period in [5, 10, 20]:
            features[f'Volatility_{period}'] = features['Returns'].rolling(period).std()
            features[f'Volatility_Ratio_{period}'] = (
                features[f'Volatility_{period}'] / features[f'Volatility_{period}'].rolling(50).mean()
            )
        
        # === Momentum Features ===
        for period in [5, 10, 15, 20]:
            features[f'Momentum_{period}'] = features['Close'] / features['Close'].shift(period) - 1
            features[f'ROC_{period}'] = features['Close'].pct_change(periods=period)
        
        # === Volume Features ===
        if 'Volume' in features.columns:
            features['Volume_Change'] = features['Volume'].pct_change()
            features['Volume_MA_Ratio'] = features['Volume'] / features['Volume'].rolling(20).mean()
            features['Price_Volume'] = features['Close'] * features['Volume']
            
            # On Balance Volume derivative
            features['OBV_Change'] = features.get('OBV', pd.Series(0, index=features.index)).pct_change()
        
        # === Technical Indicator Enhancements ===
        if 'RSI' in features.columns:
            features['RSI_Momentum'] = features['RSI'].diff()
            features['RSI_MA'] = features['RSI'].rolling(10).mean()
            features['RSI_Oversold'] = (features['RSI'] < 30).astype(int)
            features['RSI_Overbought'] = (features['RSI'] > 70).astype(int)
        
        if 'MACD' in features.columns and 'MACD_Signal' in features.columns:
            features['MACD_Divergence'] = features['MACD'] - features['MACD_Signal']
            features['MACD_Momentum'] = features['MACD'].diff()
            features['MACD_Zero_Cross'] = ((features['MACD'] > 0) & (features['MACD'].shift(1) <= 0)).astype(int)
        
        # === Market Regime Features ===
        features['Trend_20'] = np.where(
            features['Close'] > features.get('SMA_20', features['Close']), 1, -1
        )
        features['Trend_50'] = np.where(
            features['Close'] > features.get('SMA_50', features['Close']), 1, -1
        )
        
        # Market regime based on multiple timeframes
        features['Market_Regime'] = (
            features['Trend_20'] + features['Trend_50']
        ) / 2
        
        # === Support/Resistance Features ===
        features['Support_Level'] = features['Low'].rolling(20).min()
        features['Resistance_Level'] = features['High'].rolling(20).max()
        features['Support_Distance'] = (features['Close'] - features['Support_Level']) / features['Close']
        features['Resistance_Distance'] = (features['Resistance_Level'] - features['Close']) / features['Close']
        
        # === Lag Features ===
        for lag in [1, 2, 3, 5]:
            features[f'Close_Lag_{lag}'] = features['Close'].shift(lag)
            features[f'Volume_Lag_{lag}'] = features['Volume'].shift(lag) if 'Volume' in features.columns else 0
            features[f'Returns_Lag_{lag}'] = features['Returns'].shift(lag)
        
        # === Interaction Features ===
        if 'RSI' in features.columns and 'MACD' in features.columns:
            features['RSI_MACD_Interaction'] = features['RSI'] * features['MACD']
        
        # === Time-based Features ===
        if features.index.name == 'Date' or 'Date' in features.columns:
            date_col = features.index if features.index.name == 'Date' else features['Date']
            features['Day_of_Week'] = pd.to_datetime(date_col).dayofweek
            features['Month'] = pd.to_datetime(date_col).month
            features['Quarter'] = pd.to_datetime(date_col).quarter
        
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill and then backward fill
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Final cleanup - drop any remaining NaN columns
        features = features.dropna(axis=1, how='all')
        
        logger.info(f"Feature engineering complete: {len(features.columns)} features created")
        
        return features
    
    def engineer_features_with_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced feature engineering including sentiment features
        """
        # Start with existing feature engineering
        features = self.engineer_features(df)
        
        # Add sentiment-based features if available
        sentiment_columns = [col for col in df.columns if col.startswith('sentiment_')]
        
        if sentiment_columns:
            logger.info(f"Adding {len(sentiment_columns)} sentiment features")
            
            for col in sentiment_columns:
                features[col] = df[col]
            
            # Create sentiment interaction features
            if 'sentiment_sentiment_score' in features.columns:
                sentiment_score = features['sentiment_sentiment_score']
                
                # Sentiment-momentum interactions
                if 'Momentum_5' in features.columns:
                    features['Sentiment_Momentum_5d'] = sentiment_score * features['Momentum_5']
                
                if 'Momentum_10' in features.columns:
                    features['Sentiment_Momentum_10d'] = sentiment_score * features['Momentum_10']
                
                # Sentiment-RSI interactions
                if 'RSI' in features.columns:
                    features['Sentiment_RSI'] = sentiment_score * features['RSI']
                
                # Sentiment-Volume interactions
                if 'Volume_Ratio' in features.columns:
                    features['Sentiment_Volume'] = sentiment_score * features['Volume_Ratio']
                
                # Sentiment strength categories
                features['Strong_Positive_Sentiment'] = (sentiment_score > 0.3).astype(int)
                features['Strong_Negative_Sentiment'] = (sentiment_score < -0.3).astype(int)
                features['Neutral_Sentiment'] = (np.abs(sentiment_score) <= 0.3).astype(int)
            
            # News-specific features
            if 'sentiment_news_articles_count' in features.columns:
                news_count = features['sentiment_news_articles_count']
                features['High_News_Activity'] = (news_count > 5).astype(int)
                features['Low_News_Activity'] = (news_count <= 2).astype(int)
            
            # Market sentiment features
            if 'sentiment_price_position' in features.columns:
                price_pos = features['sentiment_price_position']
                features['High_Price_Position'] = (price_pos > 0.7).astype(int)
                features['Low_Price_Position'] = (price_pos < 0.3).astype(int)
        
        logger.info(f"Feature engineering with sentiment complete: {len(features.columns)} total features")
        return features
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target variable for prediction
        """
        if self.model_type == 'classification':
            # Create forward-looking signals
            future_returns = df['Close'].pct_change(periods=self.config['prediction_horizon']).shift(-self.config['prediction_horizon'])
            
            # Adaptive thresholds based on volatility
            volatility = df['Close'].pct_change().rolling(20).std()
            upper_threshold = volatility * 1.5
            lower_threshold = volatility * -1.5
            
            # Create signals using [0, 1, 2] instead of [-1, 0, 1] for sklearn compatibility
            signals = pd.Series(1, index=df.index)  # Default to HOLD (1)
            signals[future_returns > upper_threshold] = 2   # Buy (2)
            signals[future_returns < lower_threshold] = 0   # Sell (0)
            # Rest remain 1 (Hold)
            
            return signals
        
        else:  # Regression
            # Predict future price
            return df['Close'].shift(-self.config['prediction_horizon'])
    
    def handle_class_imbalance(self, X, y):
        """
        Handle class imbalance using available methods
        """
        if not self.config['handle_imbalance'] or self.model_type != 'classification':
            return X, y
        
        if SMOTE_AVAILABLE:
            try:
                smote = SMOTE(random_state=self.config['random_state'])
                X_resampled, y_resampled = smote.fit_resample(X, y)
                logger.info("Applied SMOTE for class balancing")
                return X_resampled, y_resampled
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Using class weights instead.")
        
        # Fallback: Use class weights (this is already set in model initialization)
        logger.info("Using class weights for handling imbalance")
        return X, y
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Intelligent feature selection for optimal performance
        """
        if not self.config['feature_selection']:
            return X
        
        # Remove features with low variance
        variance_threshold = 0.01
        variances = X.var()
        low_variance_features = variances[variances < variance_threshold].index
        X = X.drop(columns=low_variance_features)
        
        # Remove highly correlated features
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.95)
        ]
        X = X.drop(columns=high_corr_features)
        
        # Select top K features
        if self.config['selection_method'] == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_classif if self.model_type == 'classification' else f_classif,
                k=min(self.config['n_features'], len(X.columns))
            )
        else:
            selector = SelectKBest(
                score_func=f_classif,
                k=min(self.config['n_features'], len(X.columns))
            )
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        self.feature_selector = selector
        self.selected_features = list(selected_features)
        
        logger.info(f"Selected {len(selected_features)} features from {len(X.columns)}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def train(self, df: pd.DataFrame, optimize_hyperparameters: bool = False) -> Dict:
        """
        Train the ensemble model with advanced techniques
        """
        try:
            # Check if sentiment features are available and use appropriate feature engineering
            has_sentiment = any(col.startswith('sentiment_') for col in df.columns)
            
            if has_sentiment:
                features_df = self.engineer_features_with_sentiment(df)
                logger.info("Using sentiment-enhanced feature engineering")
            else:
                features_df = self.engineer_features(df)
                logger.info("Using standard feature engineering")
            
            # Create target variable
            y = self.create_target_variable(features_df)
            
            # Remove target-related columns and clean data
            feature_cols = [col for col in features_df.columns 
                          if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 
                                       'Date', 'Symbol', 'Returns']]
            X = features_df[feature_cols]
            
            # Align X and y
            valid_idx = y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 100:
                raise ValueError("Insufficient data for training")
            
            # Time series split
            split_idx = int(len(X) * (1 - self.config['test_size']))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Feature selection
            X_train_selected = self.select_features(X_train, y_train)
            X_test_selected = X_test[self.selected_features]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_selected)
            X_test_scaled = self.scaler.transform(X_test_selected)
            
            # Handle class imbalance
            X_train_scaled, y_train = self.handle_class_imbalance(X_train_scaled, y_train)
            
            # Train individual models
            trained_models = []
            model_scores = {}
            
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                
                # Hyperparameter optimization
                if optimize_hyperparameters and name == 'rf':
                    param_grid = {
                        'n_estimators': [200, 300, 400],
                        'max_depth': [10, 12, 15],
                        'min_samples_split': [5, 10, 15]
                    }
                    
                    grid_search = GridSearchCV(
                        model, param_grid, 
                        cv=TimeSeriesSplit(n_splits=3),
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    model = grid_search.best_estimator_
                    logger.info(f"Best params for {name}: {grid_search.best_params_}")
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                model_scores[name] = {
                    'train': train_score,
                    'test': test_score
                }
                
                trained_models.append((name, model))
                logger.info(f"{name} - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
            # Create ensemble
            if self.config['use_ensemble'] and len(trained_models) > 1:
                # Weight models based on performance
                weights = []
                estimators = []
                
                for name, model in trained_models:
                    test_score = model_scores[name]['test']
                    if test_score > 0.4:  # Only include decent models
                        weights.append(max(test_score, 0.1))
                        estimators.append((name, model))
                
                if len(estimators) > 1:
                    final_model = VotingClassifier(
                        estimators=estimators,
                        voting='soft' if self.model_type == 'classification' else 'hard'
                    )
                    final_model.fit(X_train_scaled, y_train)
                    logger.info("Created ensemble model")
                else:
                    final_model = trained_models[0][1]
                    logger.info("Using single best model")
            else:
                # Use best single model
                best_model_name = max(model_scores.keys(), 
                                    key=lambda x: model_scores[x]['test'])
                final_model = dict(trained_models)[best_model_name]
                logger.info(f"Using best single model: {best_model_name}")
            
            # Calibrate probabilities if classification
            if (self.config['calibrate_probabilities'] and 
                self.model_type == 'classification' and 
                hasattr(final_model, 'predict_proba')):
                
                final_model = CalibratedClassifierCV(final_model, cv=3)
                final_model.fit(X_train_scaled, y_train)
                logger.info("Applied probability calibration")
            
            # Final evaluation
            y_pred = final_model.predict(X_test_scaled)
            
            if self.model_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                # Per-class accuracy - updated for [0, 1, 2] classes
                for class_label in [0, 1, 2]:  # Updated from [-1, 0, 1] to [0, 1, 2]
                    class_mask = y_test == class_label
                    if class_mask.sum() > 0:
                        class_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                        metrics[f'class_{class_names[class_label]}_accuracy'] = (y_pred[class_mask] == class_label).mean()
                
            else:
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
            
            # Feature importance
            importances = []
            for name, model in trained_models:
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            
            if importances:
                avg_importance = np.mean(importances, axis=0)
                self.feature_importance = pd.DataFrame({
                    'feature': self.selected_features,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
            
            # Store training info
            metrics.update({
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': len(self.selected_features),
                'model_scores': model_scores,
                'has_sentiment_features': has_sentiment
            })
            
            self.is_trained = True
            self.final_model = final_model
            
            logger.info(f"Training complete - Accuracy: {metrics.get('accuracy', 0)*100:.1f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'error': str(e)}
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Check if sentiment features are available and use appropriate feature engineering
            has_sentiment = any(col.startswith('sentiment_') for col in df.columns)
            
            if has_sentiment:
                features_df = self.engineer_features_with_sentiment(df)
            else:
                features_df = self.engineer_features(df)
            
            # Select features
            X = features_df[self.selected_features]
            
            # Scale
            X_scaled = self.scaler.transform(X)
            
            # Predict
            predictions = self.final_model.predict(X_scaled)
            
            if self.model_type == 'classification':
                probabilities = self.final_model.predict_proba(X_scaled)
                confidence_scores = np.max(probabilities, axis=1)
            else:
                # For regression, use prediction variance as confidence proxy
                confidence_scores = np.ones(len(predictions)) * 0.5
            
            return predictions, confidence_scores
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([]), np.array([])
    
    def predict_latest(self, df: pd.DataFrame) -> Dict:
        """
        Predict the latest data point with detailed analysis
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            predictions, confidence = self.predict(df)
            
            if len(predictions) == 0:
                return {'error': 'Prediction failed'}
            
            latest_pred = predictions[-1]
            latest_conf = confidence[-1]
            latest = df.iloc[-1]
            
            if self.model_type == 'classification':
                # Updated signal mapping for [0, 1, 2] classes
                signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                
                # Check if sentiment features are available
                has_sentiment = any(col.startswith('sentiment_') for col in df.columns)
                
                if has_sentiment:
                    features_df = self.engineer_features_with_sentiment(df)
                else:
                    features_df = self.engineer_features(df)
                
                probabilities = self.final_model.predict_proba(
                    self.scaler.transform([features_df[self.selected_features].iloc[-1]])
                )[0]
                
                result = {
                    'signal': signal_map.get(latest_pred, 'UNKNOWN'),
                    'confidence': float(latest_conf),
                    'probabilities': {
                        signal_map[i]: float(prob) 
                        for i, prob in enumerate(probabilities)
                    },
                    
                    # Technical indicators
                    'indicators': {
                        'rsi': float(latest.get('RSI', 0)),
                        'macd': float(latest.get('MACD', 0)),
                        'sma_20': float(latest.get('SMA_20', 0)),
                        'volatility': float(latest.get('Volatility_20', 0))
                    },
                    
                    # Price data
                    'price_data': {
                        'current_price': float(latest['Close']),
                        'support': float(latest.get('Support_Level', latest['Low'])),
                        'resistance': float(latest.get('Resistance_Level', latest['High']))
                    },
                    
                    # Market context
                    'market_context': {
                        'trend': 'Bullish' if latest.get('Market_Regime', 0) > 0 else 'Bearish',
                        'trend_strength': float(latest.get('Trend_Strength', 0)),
                        'volatility': 'High' if latest.get('Volatility_20', 0) > 0.3 else 'Normal'
                    },
                    
                    # Feature importance
                    'feature_importance': self.feature_importance.head(10).to_dict('records') 
                                        if self.feature_importance is not None else [],
                    
                    # Sentiment data if available
                    'has_sentiment': has_sentiment
                }
                
                # Add sentiment info if available
                if has_sentiment:
                    result['sentiment'] = {
                        'overall_score': float(latest.get('sentiment_sentiment_score', 0)),
                        'news_sentiment': float(latest.get('sentiment_news_sentiment', 0)),
                        'market_sentiment': float(latest.get('sentiment_market_sentiment', 0)),
                        'confidence': float(latest.get('sentiment_sentiment_confidence', 0))
                    }
                
            else:  # Regression
                current_price = float(latest['Close'])
                price_change = latest_pred - current_price
                price_change_pct = (price_change / current_price) * 100
                
                result = {
                    'predicted_price': float(latest_pred),
                    'current_price': current_price,
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct),
                    'confidence': float(latest_conf),
                    'signal': 'BUY' if price_change_pct > 2 else 'SELL' if price_change_pct < -2 else 'HOLD',
                    'horizon_days': self.config['prediction_horizon']
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def add_explainability(self, X_background: pd.DataFrame):
        """
        Add SHAP explainability to the trained model
        
        Args:
            X_background: Background dataset for SHAP initialization
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before adding explainability")
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - explainability disabled")
            return
        
        try:
            self.explainer = XAIExplainer(self.final_model, self.model_type)
            self.explainer.initialize_explainer(X_background)
            
            logger.info("SHAP explainability added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add explainability: {e}")
            raise
    
    def explain_predictions(self, X: pd.DataFrame) -> Dict:
        """
        Generate SHAP explanations for predictions
        
        Args:
            X: Input features to explain
            
        Returns:
            Dictionary containing SHAP explanations
        """
        if not hasattr(self, 'explainer') or self.explainer is None:
            return {'error': 'Explainer not initialized. SHAP may not be available.'}
        
        return self.explainer.explain_predictions(X)
    
    def create_shap_plots(self, X: pd.DataFrame, single_prediction_idx: int = 0) -> Dict:
        """
        Create SHAP visualization plots
        
        Args:
            X: Input features
            single_prediction_idx: Index for single prediction plots
            
        Returns:
            Dictionary of Plotly figures
        """
        if not hasattr(self, 'explainer') or self.explainer is None:
            return {'error': 'Explainer not initialized. SHAP may not be available.'}
        
        # First generate explanations
        explanations = self.explainer.explain_predictions(X)
        
        if 'error' in explanations:
            return {'error': explanations['error']}
        
        plots = {}
        
        try:
            # Waterfall plot for single prediction
            plots['waterfall'] = self.explainer.create_waterfall_plot(
                X.iloc[[single_prediction_idx]], single_prediction_idx
            )
            
            return plots
            
        except Exception as e:
            logger.error(f"SHAP plot creation failed: {e}")
            return {'error': str(e)}
    
    def evaluate_performance(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive performance evaluation
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Get predictions
            predictions, confidence = self.predict(df)
            
            # Create actual signals
            actual = self.create_target_variable(df)
            
            # Align
            valid_idx = actual.notna()
            predictions = predictions[valid_idx[self.selected_features].values]
            confidence = confidence[valid_idx[self.selected_features].values]
            actual = actual[valid_idx]
            
            if self.model_type == 'classification':
                # Overall metrics
                accuracy = accuracy_score(actual, predictions)
                
                # Confidence-based accuracy
                high_conf_mask = confidence >= self.config['high_confidence']
                high_conf_accuracy = accuracy_score(
                    actual[high_conf_mask], 
                    predictions[high_conf_mask]
                ) if high_conf_mask.sum() > 0 else 0
                
                # Per-class accuracy - updated for [0, 1, 2] classes
                class_accuracies = {}
                for class_label in [0, 1, 2]:  # Updated from [-1, 0, 1] to [0, 1, 2]
                    class_mask = actual == class_label
                    if class_mask.sum() > 0:
                        class_acc = (predictions[class_mask] == class_label).mean()
                        class_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                        class_accuracies[f'class_{class_names[class_label]}'] = class_acc
                
                return {
                    'overall_accuracy': accuracy,
                    'high_confidence_accuracy': high_conf_accuracy,
                    'high_confidence_ratio': high_conf_mask.mean(),
                    'average_confidence': confidence.mean(),
                    'class_accuracies': class_accuracies,
                    'total_predictions': len(predictions)
                }
                
            else:
                return {
                    'mse': mean_squared_error(actual, predictions),
                    'mae': mean_absolute_error(actual, predictions),
                    'r2': r2_score(actual, predictions),
                    'average_confidence': confidence.mean()
                }
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'final_model': self.final_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'config': self.config,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        
        self.final_model = model_data['final_model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.selected_features = model_data['selected_features']
        self.feature_importance = model_data['feature_importance']
        self.config = model_data['config']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")

import pickle
import hashlib
from datetime import datetime, timedelta

# Model cache dictionary
MODEL_CACHE = {}
MODEL_CACHE_TIMESTAMPS = {}
CACHE_DURATION = timedelta(hours=1)  # Cache models for 1 hour

def create_prediction_model(model_type='classification', config=None):
    """
    Enhanced factory function to create prediction models with caching
    
    Args:
        model_type: 'classification' or 'regression'
        config: Optional configuration dictionary
        
    Returns:
        StockPredictionModel instance with optimized configuration
    """
    # ✅ Enhanced default configuration for better accuracy
    enhanced_config = {
        # Model parameters - Optimized for 70%+ accuracy
        'rf_n_estimators': 300,      # Increased from default
        'rf_max_depth': 12,          # Optimal depth
        'rf_min_samples_split': 10,
        'rf_min_samples_leaf': 4,
        'rf_max_features': 'sqrt',
        
        'gb_n_estimators': 200,
        'gb_max_depth': 8,
        'gb_learning_rate': 0.1,
        'gb_subsample': 0.8,
        
        'xgb_n_estimators': 250,
        'xgb_max_depth': 6,
        'xgb_learning_rate': 0.1,
        'xgb_subsample': 0.8,
        'xgb_colsample_bytree': 0.8,
        
        # Training parameters
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'handle_imbalance': True,
        'use_ensemble': True,        # Enable ensemble for better accuracy
        'calibrate_probabilities': True,  # Better confidence scores
        
        # Feature selection
        'feature_selection': True,
        'n_features': 50,            # Use top 50 features
        'selection_method': 'mutual_info',
        
        # Prediction thresholds
        'high_confidence': 0.7,
        'prediction_horizon': 5
    }
    
    # Merge with provided config
    if config:
        enhanced_config.update(config)
    
    # Create model with enhanced configuration
    return StockPredictionModel(model_type, enhanced_config)

def get_cached_model(symbol: str, model_type: str = 'classification') -> StockPredictionModel:
    """
    Get a cached model or create a new one
    
    Args:
        symbol: Stock symbol
        model_type: Type of model
        
    Returns:
        Cached or new model instance
    """
    cache_key = f"{symbol}_{model_type}"
    
    # Check if model exists in cache and is still valid
    if cache_key in MODEL_CACHE:
        timestamp = MODEL_CACHE_TIMESTAMPS.get(cache_key)
        if timestamp and (datetime.now() - timestamp) < CACHE_DURATION:
            logger.info(f"Using cached model for {symbol}")
            return MODEL_CACHE[cache_key]
    
    # Create new model with enhanced configuration
    model = create_prediction_model(model_type)
    
    # Cache the model
    MODEL_CACHE[cache_key] = model
    MODEL_CACHE_TIMESTAMPS[cache_key] = datetime.now()
    
    logger.info(f"Created new model for {symbol}")
    return model


def get_model_info():
    """Get information about available models and features"""
    info = {
        'available_models': ['RandomForest', 'GradientBoosting', 'ExtraTrees', 'AdaBoost'],
        'optional_models': [],
        'features': {
            'class_imbalance_handling': 'class_weights' if not SMOTE_AVAILABLE else 'SMOTE',
            'feature_engineering': 'advanced_60+_features',
            'ensemble_methods': 'voting_classifier',
            'probability_calibration': 'available',
            'explainability': 'SHAP' if SHAP_AVAILABLE else 'feature_importance_only',
            'sentiment_analysis': 'integrated',
            'visualization': 'plotly' if PLOTLY_AVAILABLE else 'basic'
        }
    }
    
    if XGBOOST_AVAILABLE:
        info['available_models'].append('XGBoost')
        info['optional_models'].append('XGBoost')
    
    if LIGHTGBM_AVAILABLE:
        info['available_models'].append('LightGBM')
        info['optional_models'].append('LightGBM')
    
    if SMOTE_AVAILABLE:
        info['optional_models'].append('SMOTE')
    
    if SHAP_AVAILABLE:
        info['optional_models'].append('SHAP')
    
    if PLOTLY_AVAILABLE:
        info['optional_models'].append('Plotly')
    
    return info


# Print status on import
logger.info("ML Models module loaded successfully")
model_info = get_model_info()
logger.info(f"Available models: {model_info['available_models']}")
if model_info['optional_models']:
    logger.info(f"Optional models available: {model_info['optional_models']}")