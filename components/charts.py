"""
Chart Components Module
Reusable chart components with minimalistic black/white/gray theme
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st

class ChartComponents:
    """Reusable chart components with consistent styling"""
    
    # Minimalistic color scheme
    COLORS = {
        'background': '#FFFFFF',
        'grid': '#E9ECEF',
        'text': '#212529',
        'primary': '#000000',
        'secondary': '#6C757D',
        'success': '#28A745',
        'danger': '#DC3545',
        'candle_up': '#000000',
        'candle_down': '#DC3545',
        'volume': 'rgba(108, 117, 125, 0.3)'
    }
    
    # Default layout configuration - FIXED: Removed 'weight' properties
    DEFAULT_LAYOUT = {
        'plot_bgcolor': COLORS['background'],
        'paper_bgcolor': COLORS['background'],
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 12,
            'color': COLORS['text']
        },
        'margin': dict(l=0, r=0, t=30, b=0),
        'hovermode': 'x unified',
        'hoverlabel': dict(
            bgcolor='white',
            font_size=12,
            font_family='monospace'
        ),
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'showgrid': False,
            'zeroline': False,
            'color': COLORS['text']
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'showgrid': True,
            'zeroline': False,
            'color': COLORS['text']
        }
    }
    
    @staticmethod
    def create_candlestick_chart(df: pd.DataFrame, 
                                 title: str = "",
                                 height: int = 500,
                                 show_volume: bool = True,
                                 indicators: List[str] = None) -> go.Figure:
        """
        Create a minimalistic candlestick chart
        
        Args:
            df: DataFrame with OHLCV data
            title: Chart title
            height: Chart height in pixels
            show_volume: Whether to show volume bars
            indicators: List of indicators to overlay
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color=ChartComponents.COLORS['candle_up'],
            decreasing_line_color=ChartComponents.COLORS['candle_down'],
            increasing_fillcolor=ChartComponents.COLORS['candle_up'],
            decreasing_fillcolor=ChartComponents.COLORS['candle_down'],
            line=dict(width=1),
            hoverlabel=dict(namelength=0)
        ))
        
        # Add volume bars if requested
        if show_volume and 'Volume' in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                yaxis='y2',
                marker_color=ChartComponents.COLORS['volume'],
                hoverlabel=dict(namelength=0)
            ))
        
        # Add technical indicators
        if indicators:
            colors = ['#000000', '#6C757D', '#ADB5BD']
            for i, indicator in enumerate(indicators):
                if indicator in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[indicator],
                        name=indicator,
                        line=dict(
                            color=colors[i % len(colors)],
                            width=1
                        ),
                        hoverlabel=dict(namelength=0)
                    ))
        
        # Update layout - FIXED: Removed 'weight' from font
        layout = ChartComponents.DEFAULT_LAYOUT.copy()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 16}  # Removed 'weight' property
            },
            'height': height,
            'xaxis': {
                **layout['xaxis'],
                'rangeslider': {'visible': False},
                'type': 'date'
            },
            'yaxis': {
                **layout['yaxis'],
                'title': 'Price ($)',
                'side': 'right'
            },
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1,
                'font': {'size': 10}
            }
        })
        
        if show_volume:
            layout['yaxis2'] = {
                'title': 'Volume',
                'overlaying': 'y',
                'side': 'left',
                'showgrid': False,
                'range': [0, df['Volume'].max() * 4]
            }
        
        fig.update_layout(layout)
        
        return fig
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame,
                         columns: List[str],
                         title: str = "",
                         height: int = 400,
                         show_legend: bool = True) -> go.Figure:
        """
        Create a minimalistic line chart
        
        Args:
            df: DataFrame with data
            columns: List of column names to plot
            title: Chart title
            height: Chart height in pixels
            show_legend: Whether to show legend
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = ['#000000', '#6C757D', '#ADB5BD', '#DEE2E6']
        
        for i, col in enumerate(columns):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode='lines',
                    line=dict(
                        color=colors[i % len(colors)],
                        width=1.5
                    ),
                    hoverlabel=dict(namelength=0)
                ))
        
        layout = ChartComponents.DEFAULT_LAYOUT.copy()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 14}  # Removed 'weight' property
            },
            'height': height,
            'showlegend': show_legend,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 10}
            }
        })
        
        fig.update_layout(layout)
        
        return fig
    
    @staticmethod
    def create_indicator_chart(df: pd.DataFrame,
                              indicator: str,
                              title: str = "",
                              height: int = 250,
                              thresholds: Dict = None) -> go.Figure:
        """
        Create an indicator chart (RSI, MACD, etc.)
        
        Args:
            df: DataFrame with indicator data
            indicator: Name of the indicator column
            title: Chart title
            height: Chart height in pixels
            thresholds: Dictionary with threshold lines
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add main indicator line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[indicator],
            name=indicator,
            line=dict(color=ChartComponents.COLORS['primary'], width=1.5),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 0, 0.05)'
        ))
        
        # Add threshold lines
        if thresholds:
            for name, value in thresholds.items():
                fig.add_hline(
                    y=value,
                    line_dash="dash",
                    line_color=ChartComponents.COLORS['secondary'],
                    line_width=1,
                    annotation_text=name,
                    annotation_position="right",
                    annotation_font_size=10
                )
        
        layout = ChartComponents.DEFAULT_LAYOUT.copy()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 12}  # Removed 'weight' property
            },
            'height': height,
            'showlegend': False
        })
        
        fig.update_layout(layout)
        
        return fig
    
    @staticmethod
    def create_heatmap(correlation_matrix: pd.DataFrame,
                      title: str = "Correlation Heatmap",
                      height: int = 400) -> go.Figure:
        """
        Create a correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Chart title
            height: Chart height in pixels
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale=[
                [0, '#DC3545'],
                [0.5, '#FFFFFF'],
                [1, '#28A745']
            ],
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(
                title="Correlation",
                titleside="right",
                tickmode="linear",
                tick0=-1,
                dtick=0.5,
                thickness=10,
                len=0.7
            )
        ))
        
        layout = ChartComponents.DEFAULT_LAYOUT.copy()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 14}  # Removed 'weight' property
            },
            'height': height,
            'xaxis': {
                'side': 'bottom',
                'tickangle': -45
            },
            'yaxis': {
                'side': 'left'
            }
        })
        
        fig.update_layout(layout)
        
        return fig
    
    @staticmethod
    def create_pie_chart(data: Dict,
                        title: str = "",
                        height: int = 400,
                        show_legend: bool = True) -> go.Figure:
        """
        Create a minimalistic pie chart
        
        Args:
            data: Dictionary with labels as keys and values as values
            title: Chart title
            height: Chart height in pixels
            show_legend: Whether to show legend
        
        Returns:
            Plotly figure object
        """
        colors = ['#000000', '#212529', '#495057', '#6C757D', '#ADB5BD', '#CED4DA', '#DEE2E6', '#E9ECEF', '#F8F9FA']
        
        fig = go.Figure(data=[go.Pie(
            labels=list(data.keys()),
            values=list(data.values()),
            hole=0.4,
            marker=dict(
                colors=colors[:len(data)],
                line=dict(color='white', width=2)
            ),
            textfont=dict(size=12, color='white'),
            textposition='inside',
            textinfo='percent+label'
        )])
        
        layout = ChartComponents.DEFAULT_LAYOUT.copy()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 14}  # Removed 'weight' property
            },
            'height': height,
            'showlegend': show_legend,
            'legend': {
                'orientation': 'v',
                'yanchor': 'middle',
                'y': 0.5,
                'xanchor': 'left',
                'x': 1.1,
                'font': {'size': 10}
            }
        })
        
        fig.update_layout(layout)
        
        return fig
    
    @staticmethod
    def create_bar_chart(data: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        title: str = "",
                        height: int = 400,
                        orientation: str = 'v',
                        color_positive: bool = True) -> go.Figure:
        """
        Create a minimalistic bar chart
        
        Args:
            data: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            height: Chart height in pixels
            orientation: 'v' for vertical, 'h' for horizontal
            color_positive: Color bars based on positive/negative values
        
        Returns:
            Plotly figure object
        """
        if color_positive:
            colors = ['#28A745' if val >= 0 else '#DC3545' for val in data[y_col]]
        else:
            colors = ChartComponents.COLORS['primary']
        
        fig = go.Figure(data=[go.Bar(
            x=data[x_col] if orientation == 'v' else data[y_col],
            y=data[y_col] if orientation == 'v' else data[x_col],
            orientation=orientation,
            marker_color=colors,
            text=data[y_col].round(2),
            textposition='outside' if orientation == 'v' else 'auto',
            textfont=dict(size=10)
        )])
        
        layout = ChartComponents.DEFAULT_LAYOUT.copy()
        layout.update({
            'title': {
                'text': title,
                'font': {'size': 14}  # Removed 'weight' property
            },
            'height': height,
            'showlegend': False,
            'xaxis': {
                **layout['xaxis'],
                'tickangle': -45 if orientation == 'v' else 0
            }
        })
        
        fig.update_layout(layout)
        
        return fig
    
    @staticmethod
    def create_gauge_chart(value: float,
                          title: str = "",
                          min_val: float = 0,
                          max_val: float = 100,
                          height: int = 300) -> go.Figure:
        """
        Create a gauge chart for metrics
        
        Args:
            value: Current value
            title: Chart title
            min_val: Minimum value
            max_val: Maximum value
            height: Chart height in pixels
        
        Returns:
            Plotly figure object
        """
        # Determine color based on value
        if value < 30:
            bar_color = ChartComponents.COLORS['danger']
        elif value < 70:
            bar_color = ChartComponents.COLORS['secondary']
        else:
            bar_color = ChartComponents.COLORS['success']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 14}},
            gauge={
                'axis': {
                    'range': [min_val, max_val],
                    'tickwidth': 1,
                    'tickcolor': ChartComponents.COLORS['secondary']
                },
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': ChartComponents.COLORS['grid'],
                'steps': [
                    {'range': [min_val, 30], 'color': '#F8F9FA'},
                    {'range': [30, 70], 'color': '#E9ECEF'},
                    {'range': [70, max_val], 'color': '#DEE2E6'}
                ],
                'threshold': {
                    'line': {'color': ChartComponents.COLORS['primary'], 'width': 2},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        layout = ChartComponents.DEFAULT_LAYOUT.copy()
        layout.update({
            'height': height,
            'margin': dict(l=20, r=20, t=40, b=20)
        })
        
        fig.update_layout(layout)
        
        return fig
    
    @staticmethod
    def create_sparkline(data: pd.Series,
                        height: int = 60,
                        show_points: bool = False) -> go.Figure:
        """
        Create a small sparkline chart
        
        Args:
            data: Series with data
            height: Chart height in pixels
            show_points: Whether to show data points
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Determine color based on trend
        color = ChartComponents.COLORS['success'] if data.iloc[-1] > data.iloc[0] else ChartComponents.COLORS['danger']
        
        fig.add_trace(go.Scatter(
            y=data.values,
            mode='lines+markers' if show_points else 'lines',
            line=dict(color=color, width=1),
            marker=dict(size=3) if show_points else None,
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
        ))
        
        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            hovermode=False
        )
        
        return fig

# Utility functions for common chart operations
def render_chart(fig: go.Figure, use_container_width: bool = True):
    """
    Render a Plotly chart in Streamlit with consistent settings
    
    Args:
        fig: Plotly figure object
        use_container_width: Whether to use full container width
    """
    st.plotly_chart(
        fig,
        use_container_width=use_container_width,
        config={
            'displayModeBar': False,
            'staticPlot': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'chart',
                'height': None,
                'width': None,
                'scale': 2
            }
        }
    )

def create_mini_chart(data: pd.Series, title: str = "", metric_value: str = ""):
    """
    Create a mini chart with metric for dashboards
    
    Args:
        data: Series with data
        title: Chart title
        metric_value: Main metric value to display
    """
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.metric(title, metric_value)
    
    with col2:
        fig = ChartComponents.create_sparkline(data)
        render_chart(fig)