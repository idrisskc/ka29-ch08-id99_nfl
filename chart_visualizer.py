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
    """Create Heatmap visualization with NFL Field Dimensions"""
    try:
        # Check if this is a field-based heatmap
        is_field_heatmap = 'x_zone' in df.columns or 'x_bin' in df.columns
        
        if is_field_heatmap:
            # NFL Field dimensions: 120 yards x 53.3 yards
            if 'x_zone' in df.columns and 'y_zone' in df.columns:
                x_col, y_col = 'x_zone', 'y_zone'
                z_col = 'density' if 'density' in df.columns else df.columns[-1]
            elif 'x_bin' in df.columns and 'y_bin' in df.columns:
                x_col, y_col = 'x_bin', 'y_bin'
                z_col = 'intensity' if 'intensity' in df.columns else 'frequency'
            else:
                x_col, y_col = df.columns[0], df.columns[1]
                z_col = df.columns[2] if len(df.columns) >= 3 else df.columns[1]
            
            # Create pivot table for heatmap
            pivot_df = df.pivot_table(
                index=y_col, 
                columns=x_col, 
                values=z_col, 
                aggfunc='sum',
                fill_value=0
            )
            
            # Create heatmap with NFL field styling
            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns * 10 if 'zone' in x_col else pivot_df.columns * 5,  # Convert to yards
                y=pivot_df.index * 5,  # Convert to yards (width)
                colorscale=[
                    [0, '#00FF00'],      # Green (cold/low activity)
                    [0.3, '#FFFF00'],    # Yellow (medium)
                    [0.6, '#FFA500'],    # Orange (warm)
                    [0.8, '#FF6347'],    # Red-Orange (hot)
                    [1, '#FF0000']       # Red (hottest/high activity)
                ],
                colorbar=dict(
                    title="Activity<br>Intensity",
                    titleside="right",
                    tickmode="linear",
                    tick0=0,
                    dtick=pivot_df.values.max() / 5,
                    tickfont=dict(color=TEXT_COLOR)
                ),
                hoverongaps=False,
                hovertemplate='<b>Field Position</b><br>' +
                             'Yards Downfield: %{x}<br>' +
                             'Yards From Sideline: %{y}<br>' +
                             'Intensity: %{z:.1f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add field markings
            fig.update_layout(
                title=f"{kpi_name} - NFL Field Heatmap",
                xaxis=dict(
                    title="Yards Downfield (0 = Own Goal Line, 120 = Opponent Goal Line)",
                    tickmode='linear',
                    tick0=0,
                    dtick=10,
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    range=[0, 120]
                ),
                yaxis=dict(
                    title="Field Width (Yards from Sideline)",
                    tickmode='linear',
                    tick0=0,
                    dtick=10,
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    range=[0, 53.3]
                ),
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=600,
                width=1200
            )
            
            # Add red zone marker
            fig.add_vrect(
                x0=100, x1=120,
                fillcolor="rgba(193, 18, 31, 0.1)",
                layer="below",
                line_width=0,
                annotation_text="Red Zone",
                annotation_position="top left"
            )
            
            return fig
            
        else:
            # Standard heatmap for non-field data
            if len(df.columns) >= 3:
                x_col = df.columns[0]
                y_col = df.columns[1]
                z_col = df.columns[2]
                
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
                    x=df.columns[0], 
                    y=df.columns[1],
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
            top_data = df.head(8)
            
            # Normalize values to 0-100 scale for better visualization
            max_val = top_data[top_data.columns[1]].max()
            normalized_values = (top_data[top_data.columns[1]] / max_val * 100).values
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=top_data[top_data.columns[0]].values,
                fill='toself',
                line_color=COLOR_ACCENT,
                fillcolor='rgba(193, 18, 31, 0.3)',
                name='Performance'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 100],
                        tickfont=dict(color=TEXT_COLOR)
                    ),
                    bgcolor=COLOR_PANEL,
                    angularaxis=dict(
                        tickfont=dict(color=TEXT_COLOR, size=10)
                    )
                ),
                showlegend=False,
                title=f"{kpi_name} - Performance Radar",
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
            x_col = df.columns[0] if df.columns[0] != 'play_id' else df.columns[1]
            y_col = df.columns[1] if df.columns[1] != 'play_id' else df.columns[2]
            
            # Use separation or relevant metric as size
            size_col = None
            for col in ['min_separation', 'separation', 'avg_separation']:
                if col in df.columns:
                    size_col = col
                    break
            
            if size_col is None:
                size_col = y_col
            
            # Limit data points for performance
            plot_df = df.head(100).copy()
            
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                size=size_col,
                color=y_col,
                color_continuous_scale='Viridis',
                template='plotly_dark',
                hover_data=plot_df.columns.tolist(),
                title=f"{kpi_name} - Bubble Analysis"
            )
            
            fig.update_layout(
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
            df_copy = df.head(12).copy()
            
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
                title=f"{kpi_name} - Formation Distribution",
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
            categories = df[df.columns[0]].head(10).tolist()
            values = df[df.columns[1]].head(10).tolist()
            
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative"] * len(values),
                x=categories,
                y=values,
                connector={"line": {"color": COLOR_GOLD}},
                increasing={"marker": {"color": "#32CD32"}},
                decreasing={"marker": {"color": COLOR_ACCENT}},
                text=[f"{v:.2f}" for v in values],
                textposition="outside"
            ))
            
            fig.update_layout(
                title=f"{kpi_name} - Impact Analysis",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500,
                showlegend=False
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
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
                
            numeric_col = numeric_cols[0]
            
            fig = go.Figure()
            
            # Add violin plot
            fig.add_trace(go.Violin(
                y=df[numeric_col].head(500),
                box_visible=True,
                meanline_visible=True,
                fillcolor=COLOR_ACCENT,
                line_color=COLOR_GOLD,
                opacity=0.7,
                name='Distribution'
            ))
            
            fig.update_layout(
                title=f"{kpi_name} - Distribution Analysis",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500,
                yaxis_title=numeric_col.replace('_', ' ').title(),
                showlegend=False
            )
            return fig
    except Exception as e:
        st.error(f"Violin plot error: {e}")
        return None


def create_gauge_chart(df, kpi_name):
    """Create Gauge Chart for Red Zone Success"""
    try:
        if len(df) > 0 and 'success_rate' in df.columns:
            # Calculate overall success rate
            value = float(df['success_rate'].mean())
        elif len(df) > 0 and len(df.columns) >= 2:
            value = float(df[df.columns[1]].iloc[0])
        else:
            value = 65.0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{kpi_name}<br><span style='font-size:0.8em'>Success Rate %</span>", 'font': {'color': TEXT_COLOR, 'size': 20}},
            delta={'reference': 60, 'increasing': {'color': "#32CD32"}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': TEXT_COLOR},
                'bar': {'color': COLOR_GOLD, 'thickness': 0.75},
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(193, 18, 31, 0.3)', 'name': 'Poor'},
                    {'range': [40, 60], 'color': 'rgba(255, 165, 0, 0.3)', 'name': 'Average'},
                    {'range': [60, 75], 'color': 'rgba(255, 215, 0, 0.3)', 'name': 'Good'},
                    {'range': [75, 100], 'color': 'rgba(50, 205, 50, 0.3)', 'name': 'Excellent'}
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
            # Find best numeric column
            if 'speed_range' in df.columns and 'count' in df.columns:
                # For speed distribution
                fig = px.bar(
                    df,
                    x='speed_range',
                    y='count',
                    color='count',
                    color_continuous_scale='Viridis',
                    template='plotly_dark',
                    title=f"{kpi_name} - Distribution"
                )
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    
                    fig = px.histogram(
                        df,
                        x=col,
                        nbins=30,
                        color_discrete_sequence=[COLOR_ACCENT],
                        template='plotly_dark',
                        title=f"{kpi_name} - Distribution"
                    )
                else:
                    return None
            
            fig.update_layout(
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500,
                showlegend=False
            )
            return fig
    except Exception as e:
        st.error(f"Histogram error: {e}")
        return None


def create_doughnut_chart(df, kpi_name):
    """Create Doughnut/Pie Chart"""
    try:
        if len(df.columns) >= 2:
            labels = df[df.columns[0]].head(10)
            values = df[df.columns[1]].head(10)
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=NFL_COLORS,
                textinfo='label+percent',
                textfont_size=12,
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title=f"{kpi_name} - Distribution",
                paper_bgcolor=COLOR_BG,
                font_color=TEXT_COLOR,
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                )
            )
            return fig
    except Exception as e:
        st.error(f"Doughnut error: {e}")
        return None


def create_funnel_chart(df, kpi_name):
    """Create Funnel Chart"""
    try:
        if len(df.columns) >= 2:
            stages = df[df.columns[0]].head(8).tolist()
            values = df[df.columns[1]].head(8).tolist()
            
            fig = go.Figure(go.Funnel(
                y=stages,
                x=values,
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": NFL_COLORS[:len(stages)]},
                connector={"line": {"color": COLOR_GOLD, "width": 2}}
            ))
            
            fig.update_layout(
                title=f"{kpi_name} - Performance Funnel",
                paper_bgcolor=COLOR_BG,
                font_color=TEXT_COLOR,
                height=500,
                showlegend=False
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
                df.head(12),
                x=df.columns[0],
                y=df.columns[1],
                color=df.columns[0],
                template='plotly_dark',
                color_discrete_sequence=NFL_COLORS,
                text=df.columns[1]
            )
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            
            fig.update_layout(
                title=f"{kpi_name} - Performance Comparison",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500,
                barmode='group',
                showlegend=True,
                xaxis_tickangle=-45
            )
            return fig
    except Exception as e:
        st.error(f"Stacked bar error: {e}")
        return None


def create_time_series_chart(df, kpi_name):
    """Create Time Series Chart"""
    try:
        if len(df.columns) >= 2:
            # For tempo analysis
            if 'tempo' in df.columns and 'count' in df.columns:
                fig = px.bar(
                    df,
                    x='tempo',
                    y='count',
                    color='count',
                    color_continuous_scale='Viridis',
                    template='plotly_dark'
                )
            else:
                fig = px.line(
                    df.head(100),
                    x=df.columns[0],
                    y=df.columns[1],
                    markers=True,
                    template='plotly_dark',
                    color_discrete_sequence=[COLOR_GOLD]
                )
            
            fig.update_layout(
                title=f"{kpi_name} - Trend Analysis",
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
    """Create Step Chart for Pass Timing"""
    try:
        if len(df.columns) >= 2:
            # For pass timing windows
            if 'timing_window' in df.columns:
                timing_summary = df['timing_window'].value_counts().reset_index()
                timing_summary.columns = ['timing_window', 'count']
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=timing_summary['timing_window'],
                    y=timing_summary['count'],
                    marker_color=NFL_COLORS[:len(timing_summary)],
                    text=timing_summary['count'],
                    textposition='outside'
                ))
            else:
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
                title=f"{kpi_name} - Timing Analysis",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500,
                xaxis_tickangle=-45
            )
            return fig
    except Exception as e:
        st.error(f"Step chart error: {e}")
        return None


def create_scatter_3d_chart(df, kpi_name):
    """Create 3D Scatter Plot for Separation Analysis"""
    try:
        if all(col in df.columns for col in ['x_position', 'y_position', 'min_separation']):
            fig = px.scatter_3d(
                df.head(100),
                x='x_position',
                y='y_position',
                z='min_separation',
                color='separation_category',
                size='min_separation',
                template='plotly_dark',
                title=f"{kpi_name} - 3D Field Position Analysis"
            )
            
            fig.update_layout(
                paper_bgcolor=COLOR_BG,
                font_color=TEXT_COLOR,
                height=600,
                scene=dict(
                    xaxis_title='Field Position (Yards)',
                    yaxis_title='Lateral Position (Yards)',
                    zaxis_title='Separation (Yards)',
                    bgcolor=COLOR_PANEL
                )
            )
            return fig
    except Exception as e:
        st.error(f"3D scatter error: {e}")
        return None


def create_box_plot(df, kpi_name):
    """Create Box Plot for Statistical Distribution"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=df[col].head(500),
                name=col.replace('_', ' ').title(),
                marker_color=COLOR_ACCENT,
                boxmean='sd'
            ))
            
            fig.update_layout(
                title=f"{kpi_name} - Statistical Distribution",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500,
                showlegend=False
            )
            return fig
    except Exception as e:
        st.error(f"Box plot error: {e}")
        return None


# =======================================================
# 🎯 MASTER CHART ROUTER
# =======================================================

def visualize_kpi(kpi_name, kpi_data, chart_type_hint=None):
    """
    Route KPI data to appropriate visualization
    Returns a Plotly figure or None
    """
    if kpi_data is None or (isinstance(kpi_data, pd.DataFrame) and kpi_data.empty):
        st.warning(f"No data available for {kpi_name}")
        return None
    
    # Determine chart type based on KPI name
    kpi_lower = kpi_name.lower()
    
    try:
        # Heatmaps for coverage and movement
        if 'heat' in kpi_lower or 'coverage' in kpi_lower or 'movement' in kpi_lower:
            return create_heatmap_chart(kpi_data, kpi_name)
        
        # Radar for route efficiency
        elif 'route' in kpi_lower or 'efficiency' in kpi_lower:
            return create_radar_chart(kpi_data, kpi_name)
        
        # Bubble/3D for separation
        elif 'separation' in kpi_lower:
            if 'separation_category' in kpi_data.columns:
                fig = create_scatter_3d_chart(kpi_data, kpi_name)
                if fig:
                    return fig
            return create_bubble_chart(kpi_data, kpi_name)
        
        # Sunburst for formations
        elif 'formation' in kpi_lower or 'tendency' in kpi_lower:
            return create_sunburst_chart(kpi_data, kpi_name)
        
        # Waterfall for probability and EPA
        elif 'probability' in kpi_lower or 'ep_' in kpi_lower or 'expected' in kpi_lower:
            return create_waterfall_chart(kpi_data, kpi_name)
        
        # Violin for reaction time
        elif 'reaction' in kpi_lower or ('defense' in kpi_lower and 'reaction' in str(kpi_data.columns)):
            return create_violin_chart(kpi_data, kpi_name)
        
        # Gauge for red zone success
        elif 'redzone' in kpi_lower or 'success' in kpi_lower:
            return create_gauge_chart(kpi_data, kpi_name)
        
        # Time series for tempo
        elif 'tempo' in kpi_lower or 'analysis' in kpi_lower:
            return create_time_series_chart(kpi_data, kpi_name)
        
        # Doughnut for pass results
        elif 'pass' in kpi_lower and 'result' in kpi_lower:
            return create_doughnut_chart(kpi_data, kpi_name)
        
        # Histogram for speed distribution
        elif 'distribution' in kpi_lower or 'speed' in kpi_lower:
            return create_histogram_chart(kpi_data, kpi_name)
        
        # Stacked bar for coverage type
        elif 'coverage_type' in kpi_lower or 'type' in kpi_lower:
            return create_stacked_bar_chart(kpi_data, kpi_name)
        
        # Funnel for play action
        elif 'playaction' in kpi_lower or 'impact' in kpi_lower:
            return create_funnel_chart(kpi_data, kpi_name)
        
        # Step chart for timing
        elif 'timing' in kpi_lower or 'window' in kpi_lower:
            return create_step_chart(kpi_data, kpi_name)
        
        # QB Pressure - use box plot or heatmap
        elif 'qb' in kpi_lower or 'pressure' in kpi_lower:
            if 'pressure_level' in kpi_data.columns:
                return create_stacked_bar_chart(kpi_data, kpi_name)
            return create_box_plot(kpi_data, kpi_name)
        
        # Default fallback - smart bar chart
        else:
            if len(kpi_data.columns) >= 2:
                fig = px.bar(
                    kpi_data.head(15),
                    x=kpi_data.columns[0],
                    y=kpi_data.columns[1],
                    template='plotly_dark',
                    color_discrete_sequence=[COLOR_ACCENT],
                    text=kpi_data.columns[1]
                )
                
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                
                fig.update_layout(
                    title=f"{kpi_name} - Overview",
                    paper_bgcolor=COLOR_BG,
                    plot_bgcolor=COLOR_PANEL,
                    font_color=TEXT_COLOR,
                    height=500,
                    xaxis_tickangle=-45
                )
                return fig
    
    except Exception as e:
        st.warning(f"Visualization error for {kpi_name}: {str(e)}")
        # Return simple fallback chart
        try:
            if len(kpi_data.columns) >= 2:
                fig = px.bar(
                    kpi_data.head(10),
                    x=kpi_data.columns[0],
                    y=kpi_data.columns[1],
                    template='plotly_dark'
                )
                fig.update_layout(
                    title=f"{kpi_name} (Simplified)",
                    paper_bgcolor=COLOR_BG,
                    plot_bgcolor=COLOR_PANEL,
                    font_color=TEXT_COLOR,
                    height=400
                )
                return fig
        except:
            pass
    
    return None


# =======================================================
# 📈 ADDITIONAL UTILITY FUNCTIONS
# =======================================================

def create_comparison_chart(df1, df2, kpi_name, labels=['Group A', 'Group B']):
    """Create side-by-side comparison chart"""
    try:
        fig = make_subplots(rows=1, cols=2, subplot_titles=labels)
        
        if len(df1.columns) >= 2:
            fig.add_trace(
                go.Bar(x=df1[df1.columns[0]].head(10), y=df1[df1.columns[1]].head(10), 
                       name=labels[0], marker_color=COLOR_ACCENT),
                row=1, col=1
            )
        
        if len(df2.columns) >= 2:
            fig.add_trace(
                go.Bar(x=df2[df2.columns[0]].head(10), y=df2[df2.columns[1]].head(10), 
                       name=labels[1], marker_color=COLOR_GOLD),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f"{kpi_name} - Comparison",
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_PANEL,
            font_color=TEXT_COLOR,
            height=500,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Comparison chart error: {e}")
        return None


def create_multi_metric_dashboard(kpi_dict, title="Multi-KPI Dashboard"):
    """Create a dashboard with multiple KPIs"""
    try:
        num_kpis = len(kpi_dict)
        rows = (num_kpis + 1) // 2
        
        fig = make_subplots(
            rows=rows, 
            cols=2,
            subplot_titles=list(kpi_dict.keys()),
            specs=[[{"type": "indicator"}, {"type": "indicator"}]] * rows
        )
        
        for idx, (kpi_name, kpi_value) in enumerate(kpi_dict.items()):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=kpi_value,
                    title={"text": kpi_name},
                    delta={"reference": kpi_value * 0.9}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            paper_bgcolor=COLOR_BG,
            font_color=TEXT_COLOR,
            height=300 * rows
        )
        
        return fig
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        return None
