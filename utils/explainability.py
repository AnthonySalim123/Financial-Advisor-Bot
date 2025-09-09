# utils/explainability.py
import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

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
        if self.explainer is None:
            raise ValueError("Explainer must be initialized first")
        
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
    
    def create_waterfall_plot(self, X_single: pd.DataFrame, prediction_idx: int = 0) -> go.Figure:
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X_single: Single row DataFrame
            prediction_idx: Index of prediction to explain
            
        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise ValueError("Must run explain_predictions first")
        
        try:
            # Get SHAP values for single prediction
            if isinstance(self.shap_values, list):
                # Multi-class: use the predicted class
                pred_class = self.model.predict(X_single)[0]
                class_idx = pred_class + 1  # Assuming classes are -1, 0, 1
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
            
            # Create cumulative values for waterfall
            cumulative = [base_value]
            for val in values_sorted:
                cumulative.append(cumulative[-1] + val)
            
            # Create colors
            colors = ['green' if v > 0 else 'red' for v in values_sorted]
            
            fig = go.Figure()
            
            # Add waterfall bars
            fig.add_trace(go.Waterfall(
                name="SHAP Values",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(values_sorted) + ["total"],
                x=["Base"] + features_sorted + ["Final Prediction"],
                textposition="outside",
                text=[f"{base_value:.3f}"] + [f"{v:+.3f}" for v in values_sorted] + [f"{cumulative[-1]:.3f}"],
                y=[base_value] + values_sorted + [cumulative[-1] - base_value],
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
            # Return empty figure
            return go.Figure()
    
    def create_summary_plot(self) -> go.Figure:
        """
        Create SHAP summary plot showing feature importance
        
        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise ValueError("Must run explain_predictions first")
        
        try:
            # Handle multi-class case
            if isinstance(self.shap_values, list):
                # Average across classes
                shap_vals = np.mean(self.shap_values, axis=0)
            else:
                shap_vals = self.shap_values
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap = np.abs(shap_vals).mean(axis=0)
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=True)
            
            # Take top 20 features
            top_features = importance_df.tail(20)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=top_features['feature'],
                x=top_features['importance'],
                orientation='h',
                marker_color='steelblue',
                text=[f'{val:.3f}' for val in top_features['importance']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance Summary",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Summary plot creation failed: {e}")
            return go.Figure()
    
    def create_force_plot(self, X_single: pd.DataFrame, prediction_idx: int = 0) -> go.Figure:
        """
        Create force plot showing positive and negative feature contributions
        
        Args:
            X_single: Single row DataFrame
            prediction_idx: Index of prediction to explain
            
        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            raise ValueError("Must run explain_predictions first")
        
        try:
            # Get SHAP values for single prediction
            if isinstance(self.shap_values, list):
                pred_class = self.model.predict(X_single)[0]
                class_idx = max(0, min(pred_class + 1, len(self.shap_values) - 1))
                shap_vals = self.shap_values[class_idx][prediction_idx]
            else:
                shap_vals = self.shap_values[prediction_idx]
            
            # Separate positive and negative contributions
            positive_features = []
            positive_values = []
            negative_features = []
            negative_values = []
            
            for i, (feature, value) in enumerate(zip(self.feature_names, shap_vals)):
                if value > 0:
                    positive_features.append(feature)
                    positive_values.append(value)
                else:
                    negative_features.append(feature)
                    negative_values.append(abs(value))
            
            # Sort by magnitude
            if positive_values:
                pos_sorted = sorted(zip(positive_features, positive_values), 
                                  key=lambda x: x[1], reverse=True)[:10]
                positive_features, positive_values = zip(*pos_sorted) if pos_sorted else ([], [])
            
            if negative_values:
                neg_sorted = sorted(zip(negative_features, negative_values), 
                                  key=lambda x: x[1], reverse=True)[:10]
                negative_features, negative_values = zip(*neg_sorted) if neg_sorted else ([], [])
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Positive Contributions', 'Negative Contributions'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Add positive contributions
            if positive_values:
                fig.add_trace(
                    go.Bar(
                        y=list(positive_features),
                        x=list(positive_values),
                        orientation='h',
                        marker_color='green',
                        name='Positive',
                        text=[f'+{val:.3f}' for val in positive_values],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
            
            # Add negative contributions
            if negative_values:
                fig.add_trace(
                    go.Bar(
                        y=list(negative_features),
                        x=list(negative_values),
                        orientation='h',
                        marker_color='red',
                        name='Negative',
                        text=[f'-{val:.3f}' for val in negative_values],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="SHAP Force Plot - Feature Contributions",
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Force plot creation failed: {e}")
            return go.Figure()
    
    def get_feature_explanations(self, X_single: pd.DataFrame, top_n: int = 10) -> Dict:
        """
        Get textual explanations for top contributing features
        
        Args:
            X_single: Single row DataFrame
            top_n: Number of top features to explain
            
        Returns:
            Dictionary with feature explanations
        """
        if self.shap_values is None:
            raise ValueError("Must run explain_predictions first")
        
        try:
            # Get SHAP values for single prediction
            if isinstance(self.shap_values, list):
                pred_class = self.model.predict(X_single)[0]
                class_idx = max(0, min(pred_class + 1, len(self.shap_values) - 1))
                shap_vals = self.shap_values[class_idx][0]
            else:
                shap_vals = self.shap_values[0]
            
            # Get feature values
            feature_values = X_single.iloc[0].to_dict()
            
            # Create feature importance with values
            feature_impact = []
            for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_vals)):
                feature_impact.append({
                    'feature': feature,
                    'shap_value': shap_val,
                    'feature_value': feature_values.get(feature, 0),
                    'abs_impact': abs(shap_val),
                    'direction': 'positive' if shap_val > 0 else 'negative'
                })
            
            # Sort by absolute impact
            feature_impact.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            # Get top features
            top_features = feature_impact[:top_n]
            
            # Generate explanations
            explanations = {
                'prediction': self.model.predict(X_single)[0],
                'confidence': max(self.model.predict_proba(X_single)[0]) if hasattr(self.model, 'predict_proba') else None,
                'top_features': top_features,
                'summary': self._generate_explanation_summary(top_features)
            }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Feature explanation generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_explanation_summary(self, top_features: List[Dict]) -> str:
        """
        Generate human-readable explanation summary
        
        Args:
            top_features: List of top feature dictionaries
            
        Returns:
            Summary string
        """
        if not top_features:
            return "No significant features identified."
        
        summary_parts = []
        
        # Most important feature
        most_important = top_features[0]
        direction = "strongly supports" if most_important['direction'] == 'positive' else "strongly opposes"
        summary_parts.append(
            f"The {most_important['feature']} (value: {most_important['feature_value']:.3f}) "
            f"{direction} this prediction with an impact of {most_important['shap_value']:+.3f}."
        )
        
        # Additional important features
        if len(top_features) > 1:
            positive_features = [f for f in top_features[1:4] if f['direction'] == 'positive']
            negative_features = [f for f in top_features[1:4] if f['direction'] == 'negative']
            
            if positive_features:
                pos_names = [f['feature'] for f in positive_features]
                summary_parts.append(f"Supporting factors include: {', '.join(pos_names)}.")
            
            if negative_features:
                neg_names = [f['feature'] for f in negative_features]
                summary_parts.append(f"Opposing factors include: {', '.join(neg_names)}.")
        
        return " ".join(summary_parts)


def create_explainer(model, model_type='classification') -> XAIExplainer:
    """
    Factory function to create XAI explainer
    
    Args:
        model: Trained model
        model_type: 'classification' or 'regression'
        
    Returns:
        XAIExplainer instance
    """
    return XAIExplainer(model, model_type)