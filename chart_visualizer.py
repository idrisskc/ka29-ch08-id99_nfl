# =======================================================
# chart_visualizer.py - Advanced Chart Visualizations for NFL KPIs
# =======================================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =======================================================
# 🎨 COLOR SCHEMES
# =======================================================
COLOR_BG = "#0B0C10"
COLOR_PANEL = "#1B263B"
COLOR_ACCENT = "#C1121F"
COLOR_GOLD = "#FFD700"
TEXT_COLOR = "#E6EEF8"

NFL_COLORS = ['#C1121F', '#FFD700', '#1E90FF', '#32CD32', '#FF6347', '#9370DB', '#FF8C00', '#4169E1']


# =======================================================
# 📊 SPECIALIZED CHART FUNCTIONS
# =======================================================

def create_heatmap_chart(df, kpi_name):
    """Create Heatmap visualization"""
    try:
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            if len(df.columns) >= 3:
                z_col = df.columns[2]
                # Pivot for proper heatmap
                pivot_df = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_df.values,
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    colorscale='RdYlGn',
                    hoverongaps=False
                ))
            else:
                fig = px.density_heatmap(
                    df, 
                    x=x_col, 
                    y=y_col,
                    color_continuous_scale='RdYlGn',
                    template='plotly_dark'
                )
            
            fig.update_layout(
                title=f"{kpi_name} - Heatmap",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Heatmap error: {e}")
        return None


def create_radar_chart(df, kpi_name):
    """Create Radar/Spider Chart"""
    try:
        if len(df) > 0 and len(df.columns) >= 2:
            # Take top categories
            top_data = df.head(6)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=top_data[top_data.columns[1]].values,
                theta=top_data[top_data.columns[0]].values,
                fill='toself',
                line_color=COLOR_ACCENT,
                fillcolor='rgba(193, 18, 31, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, top_data[top_data.columns[1]].max() * 1.2]),
                    bgcolor=COLOR_PANEL
                ),
                showlegend=False,
                title=f"{kpi_name} - Radar Chart",
                paper_bgcolor=COLOR_BG,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Radar chart error: {e}")
        return None


def create_bubble_chart(df, kpi_name):
    """Create Bubble Chart"""
    try:
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            # Use third column as size if available
            size_col = df.columns[2] if len(df.columns) >= 3 else y_col
            
            fig = px.scatter(
                df.head(50),
                x=x_col,
                y=y_col,
                size=size_col,
                color=y_col,
                color_continuous_scale='Viridis',
                template='plotly_dark',
                hover_data=df.columns.tolist()
            )
            
            fig.update_layout(
                title=f"{kpi_name} - Bubble Chart",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Bubble chart error: {e}")
        return None


def create_sunburst_chart(df, kpi_name):
    """Create Sunburst Chart"""
    try:
        if len(df.columns) >= 2:
            # Prepare hierarchical data
            df_copy = df.head(10).copy()
            
            if 'percentage' in df.columns:
                values_col = 'percentage'
            elif 'count' in df.columns:
                values_col = 'count'
            else:
                values_col = df.columns[1]
            
            fig = px.sunburst(
                df_copy,
                path=[df.columns[0]],
                values=values_col,
                color=values_col,
                color_continuous_scale='RdYlGn',
                template='plotly_dark'
            )
            
            fig.update_layout(
                title=f"{kpi_name} - Sunburst",
                paper_bgcolor=COLOR_BG,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Sunburst error: {e}")
        return None


def create_waterfall_chart(df, kpi_name):
    """Create Waterfall Chart"""
    try:
        if len(df.columns) >= 2:
            categories = df[df.columns[0]].head(8).tolist()
            values = df[df.columns[1]].head(8).tolist()
            
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative"] * len(values),
                x=categories,
                y=values,
                connector={"line": {"color": COLOR_GOLD}},
                increasing={"marker": {"color": "#32CD32"}},
                decreasing={"marker": {"color": COLOR_ACCENT}},
            ))
            
            fig.update_layout(
                title=f"{kpi_name} - Waterfall",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Waterfall error: {e}")
        return None


def create_violin_chart(df, kpi_name):
    """Create Violin Plot"""
    try:
        if len(df.columns) >= 1:
            # Get numeric column
            numeric_col = df.select_dtypes(include=[np.number]).columns[0]
            
            fig = go.Figure()
            
            fig.add_trace(go.Violin(
                y=df[numeric_col].head(200),
                box_visible=True,
                meanline_visible=True,
                fillcolor=COLOR_ACCENT,
                line_color=COLOR_GOLD,
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f"{kpi_name} - Violin Plot",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500,
                yaxis_title=numeric_col
            )
            return fig
    except Exception as e:
        st.error(f"Violin plot error: {e}")
        return None


def create_gauge_chart(df, kpi_name):
    """Create Gauge Chart"""
    try:
        if len(df) > 0 and len(df.columns) >= 2:
            value = float(df[df.columns[1]].iloc[0])
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': kpi_name, 'font': {'color': TEXT_COLOR}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': TEXT_COLOR},
                    'bar': {'color': COLOR_GOLD},
                    'steps': [
                        {'range': [0, 33], 'color': COLOR_ACCENT},
                        {'range': [33, 66], 'color': "#FFD700"},
                        {'range': [66, 100], 'color': "#32CD32"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor=COLOR_BG,
                font={'color': TEXT_COLOR},
                height=400
            )
            return fig
    except Exception as e:
        st.error(f"Gauge error: {e}")
        return None


def create_histogram_chart(df, kpi_name):
    """Create Histogram"""
    try:
        if len(df.columns) >= 1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                
                fig = px.histogram(
                    df,
                    x=col,
                    nbins=30,
                    color_discrete_sequence=[COLOR_ACCENT],
                    template='plotly_dark'
                )
                
                fig.update_layout(
                    title=f"{kpi_name} - Distribution",
                    paper_bgcolor=COLOR_BG,
                    plot_bgcolor=COLOR_PANEL,
                    font_color=TEXT_COLOR,
                    height=500
                )
                return fig
    except Exception as e:
        st.error(f"Histogram error: {e}")
        return None


def create_doughnut_chart(df, kpi_name):
    """Create Doughnut/Pie Chart"""
    try:
        if len(df.columns) >= 2:
            labels = df[df.columns[0]].head(8)
            values = df[df.columns[1]].head(8)
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=NFL_COLORS
            )])
            
            fig.update_layout(
                title=f"{kpi_name} - Distribution",
                paper_bgcolor=COLOR_BG,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Doughnut error: {e}")
        return None


def create_funnel_chart(df, kpi_name):
    """Create Funnel Chart"""
    try:
        if len(df.columns) >= 2:
            stages = df[df.columns[0]].head(6).tolist()
            values = df[df.columns[1]].head(6).tolist()
            
            fig = go.Figure(go.Funnel(
                y=stages,
                x=values,
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": NFL_COLORS[:len(stages)]}
            ))
            
            fig.update_layout(
                title=f"{kpi_name} - Funnel",
                paper_bgcolor=COLOR_BG,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Funnel error: {e}")
        return None


def create_stacked_bar_chart(df, kpi_name):
    """Create Stacked Bar Chart"""
    try:
        if len(df.columns) >= 2:
            fig = px.bar(
                df.head(10),
                x=df.columns[0],
                y=df.columns[1],
                color=df.columns[0],
                template='plotly_dark',
                color_discrete_sequence=NFL_COLORS
            )
            
            fig.update_layout(
                title=f"{kpi_name} - Comparison",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500,
                barmode='stack'
            )
            return fig
    except Exception as e:
        st.error(f"Stacked bar error: {e}")
        return None


def create_time_series_chart(df, kpi_name):
    """Create Time Series Chart"""
    try:
        if len(df.columns) >= 2:
            fig = px.line(
                df.head(100),
                x=df.columns[0],
                y=df.columns[1],
                markers=True,
                template='plotly_dark',
                color_discrete_sequence=[COLOR_GOLD]
            )
            
            fig.update_layout(
                title=f"{kpi_name} - Trend",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Time series error: {e}")
        return None


def create_step_chart(df, kpi_name):
    """Create Step Chart"""
    try:
        if len(df.columns) >= 2:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df[df.columns[0]].head(50),
                y=df[df.columns[1]].head(50),
                mode='lines',
                line_shape='hv',
                line_color=COLOR_ACCENT,
                fill='tozeroy',
                fillcolor='rgba(193, 18, 31, 0.3)'
            ))
            
            fig.update_layout(
                title=f"{kpi_name} - Step Chart",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500
            )
            return fig
    except Exception as e:
        st.error(f"Step chart error: {e}")
        return None


# =======================================================
# 🎯 MASTER CHART ROUTER
# =======================================================

def visualize_kpi(kpi_name, kpi_data, chart_type_hint=None):
    """
    Route KPI data to appropriate visualization
    """
    if kpi_data is None or (isinstance(kpi_data, pd.DataFrame) and kpi_data.empty):
        st.warning(f"No data available for {kpi_name}")
        return None
    
    # Determine chart type based on KPI name or hint
    kpi_lower = kpi_name.lower()
    
    if 'heat' in kpi_lower or 'coverage' in kpi_lower:
        return create_heatmap_chart(kpi_data, kpi_name)
    
    elif 'route' in kpi_lower or 'efficiency' in kpi_lower:
        return create_radar_chart(kpi_data, kpi_name)
    
    elif 'separation' in kpi_lower:
        return create_bubble_chart(kpi_data, kpi_name)
    
    elif 'formation' in kpi_lower or 'tendency' in kpi_lower:
        return create_sunburst_chart(kpi_data, kpi_name)
    
    elif 'probability' in kpi_lower or 'ep_' in kpi_lower:
        return create_waterfall_chart(kpi_data, kpi_name)
    
    elif 'reaction' in kpi_lower or 'defense' in kpi_lower:
        return create_violin_chart(kpi_data, kpi_name)
    
    elif 'redzone' in kpi_lower or 'success' in kpi_lower:
        return create_gauge_chart(kpi_data, kpi_name)
    
    elif 'tempo' in kpi_lower or 'analysis' in kpi_lower:
        return create_time_series_chart(kpi_data, kpi_name)
    
    elif 'pass' in kpi_lower and 'result' in kpi_lower:
        return create_doughnut_chart(kpi_data, kpi_name)
    
    elif 'distribution' in kpi_lower or 'speed' in kpi_lower:
        return create_histogram_chart(kpi_data, kpi_name)
    
    elif 'coverage_type' in kpi_lower:
        return create_stacked_bar_chart(kpi_data, kpi_name)
    
    elif 'playaction' in kpi_lower or 'impact' in kpi_lower:
        return create_funnel_chart(kpi_data, kpi_name)
    
    elif 'timing' in kpi_lower:
        return create_step_chart(kpi_data, kpi_name)
    
    # Default fallback
    else:
        if len(kpi_data.columns) >= 2:
            return px.bar(
                kpi_data.head(10),
                x=kpi_data.columns[0],
                y=kpi_data.columns[1],
                template='plotly_dark',
                color_discrete_sequence=[COLOR_ACCENT]
            )
    
    return None
