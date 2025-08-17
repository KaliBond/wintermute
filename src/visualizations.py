"""
CAMS Framework Visualizations
Data visualization functions for Complex Adaptive Model State framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import networkx as nx

class CAMSVisualizer:
    """Visualization class for CAMS framework data"""
    
    def __init__(self):
        self.color_scheme = {
            'Executive': '#FF6B6B',
            'Army': '#4ECDC4', 
            'Priests': '#45B7D1',
            'Property Owners': '#96CEB4',
            'Trades/Professions': '#FFEAA7',
            'Proletariat': '#DDA0DD',
            'State Memory': '#98D8C8',
            'Shopkeepers/Merchants': '#F7DC6F'
        }
        
    def plot_system_health_timeline(self, df: pd.DataFrame, 
                                  nation: str = None) -> go.Figure:
        """Plot system health over time with critical thresholds"""
        from src.cams_analyzer import CAMSAnalyzer
        analyzer = CAMSAnalyzer()
        
        if nation:
            nation_col = analyzer._get_column_name(df, 'nation')
            if nation_col and nation_col in df.columns:
                df = df[df[nation_col] == nation]
            
        year_col = analyzer._get_column_name(df, 'year')
        years = sorted(df[year_col].unique())
        health_values = []
        
        for year in years:
            health = analyzer.calculate_system_health(df, year)
            health_values.append(health)
            
        fig = go.Figure()
        
        # Add health timeline
        fig.add_trace(go.Scatter(
            x=years,
            y=health_values,
            mode='lines+markers',
            name='System Health',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        # Add critical thresholds
        fig.add_hline(y=2.3, line_dash="dash", line_color="red", 
                     annotation_text="Critical: Forced Reorganization")
        fig.add_hline(y=2.5, line_dash="dash", line_color="orange",
                     annotation_text="Warning: Collapse Risk")
        fig.add_hline(y=5.0, line_dash="dash", line_color="yellow",
                     annotation_text="Instability Threshold")
        
        fig.update_layout(
            title=f'System Health Timeline - {nation or "Analysis"}',
            xaxis_title='Year',
            yaxis_title='System Health H(t)',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_node_heatmap(self, df: pd.DataFrame, metric: str = 'Coherence',
                         nation: str = None) -> go.Figure:
        """Create heatmap of node metrics over time"""
        from src.cams_analyzer import CAMSAnalyzer
        analyzer = CAMSAnalyzer()
        
        if nation:
            nation_col = analyzer._get_column_name(df, 'nation')
            if nation_col and nation_col in df.columns:
                df = df[df[nation_col] == nation]
            
        # Get column names using analyzer
        try:
            year_col = analyzer._get_column_name(df, 'year')
            node_col = analyzer._get_column_name(df, 'node')
        except KeyError as e:
            raise ValueError(f"Required columns not found: {e}")
        
        # Check if metric column exists and map to actual column name
        metric_mapping = {
            'coherence': 'coherence',
            'capacity': 'capacity', 
            'stress': 'stress',
            'abstraction': 'abstraction',
            'node value': 'node_value',
            'bond strength': 'bond_strength'
        }
        
        metric_key = metric.lower()
        if metric_key in metric_mapping:
            try:
                metric_col = analyzer._get_column_name(df, metric_mapping[metric_key])
                metric = metric_col
            except KeyError:
                if metric not in df.columns:
                    raise ValueError(f"Metric column '{metric}' not found in dataframe")
        elif metric not in df.columns:
            raise ValueError(f"Metric column '{metric}' not found in dataframe")
        
        pivot_data = df.pivot_table(
            values=metric, 
            index=node_col, 
            columns=year_col, 
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlBu_r',
            colorbar=dict(title=metric)
        ))
        
        fig.update_layout(
            title=f'{metric} Heatmap by Node - {nation or "Analysis"}',
            xaxis_title='Year',
            yaxis_title='Societal Node',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def plot_stress_distribution(self, df: pd.DataFrame, year: int = None,
                               nation: str = None) -> go.Figure:
        """Plot stress distribution across nodes"""
        from src.cams_analyzer import CAMSAnalyzer
        analyzer = CAMSAnalyzer()
        
        if nation:
            nation_col = analyzer._get_column_name(df, 'nation')
            if nation_col and nation_col in df.columns:
                df = df[df[nation_col] == nation]
            
        try:
            year_col = analyzer._get_column_name(df, 'year')
        except KeyError as e:
            raise ValueError(f"Year column not found: {e}")
        
        if year:
            year_data = df[df[year_col] == year]
        else:
            year_data = df[df[year_col] == df[year_col].max()]
            
        fig = go.Figure()
        
        # Get stress and node columns using analyzer
        try:
            stress_col = analyzer._get_column_name(year_data, 'stress')
            node_col = analyzer._get_column_name(year_data, 'node')
        except KeyError as e:
            raise ValueError(f"Required columns not found: {e}")
        
        fig.add_trace(go.Bar(
            x=year_data[node_col],
            y=np.abs(year_data[stress_col]),
            marker_color=[self.color_scheme.get(node, '#888888') 
                         for node in year_data[node_col]],
            name='Stress Level'
        ))
        
        fig.update_layout(
            title=f'Stress Distribution by Node - {year or "Latest"} - {nation or "Analysis"}',
            xaxis_title='Societal Node',
            yaxis_title='Stress Level (Absolute)',
            template='plotly_white',
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_four_dimensions_radar(self, df: pd.DataFrame, year: int = None,
                                 nation: str = None) -> go.Figure:
        """Create radar chart for four CAMS dimensions"""
        from src.cams_analyzer import CAMSAnalyzer
        analyzer = CAMSAnalyzer()
        
        if nation:
            nation_col = analyzer._get_column_name(df, 'nation')
            if nation_col and nation_col in df.columns:
                df = df[df[nation_col] == nation]
            
        try:
            year_col = analyzer._get_column_name(df, 'year')
        except KeyError as e:
            raise ValueError(f"Year column not found: {e}")
        
        if year:
            year_data = df[df[year_col] == year]
        else:
            year_data = df[df[year_col] == df[year_col].max()]
            
        # Calculate averages for each dimension using analyzer
        dimension_types = ['coherence', 'capacity', 'stress', 'abstraction']
        dimension_names = ['Coherence', 'Capacity', 'Stress', 'Abstraction']
        values = []
        
        for i, dim_type in enumerate(dimension_types):
            try:
                dim_col = analyzer._get_column_name(year_data, dim_type)
                if dim_type == 'stress':
                    # Use absolute value and invert for better visualization
                    avg_val = 10 - np.mean(np.abs(year_data[dim_col]))
                else:
                    avg_val = np.mean(year_data[dim_col])
                values.append(avg_val)
            except KeyError:
                # If dimension not found, use 0 as default
                values.append(0.0)
                print(f"Warning: {dimension_names[i]} column not found, using 0")
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=dimension_names + [dimension_names[0]],
            fill='toself',
            name=f'{nation or "Society"} Profile'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title=f'CAMS Four Dimensions Profile - {year or "Latest"} - {nation or "Analysis"}',
            template='plotly_white'
        )
        
        return fig
    
    def plot_node_network(self, df: pd.DataFrame, year: int = None,
                         nation: str = None) -> go.Figure:
        """Create network visualization of node interactions"""
        from src.cams_analyzer import CAMSAnalyzer
        analyzer = CAMSAnalyzer()
        
        if nation:
            nation_col = analyzer._get_column_name(df, 'nation')
            if nation_col and nation_col in df.columns:
                df = df[df[nation_col] == nation]
            
        try:
            year_col = analyzer._get_column_name(df, 'year')
        except KeyError as e:
            raise ValueError(f"Year column not found: {e}")
        
        if year:
            year_data = df[df[year_col] == year]
        else:
            year_data = df[df[year_col] == df[year_col].max()]
            
        # Get required columns using analyzer
        try:
            node_col = analyzer._get_column_name(year_data, 'node')
            node_value_col = analyzer._get_column_name(year_data, 'node_value')
            bond_strength_col = analyzer._get_column_name(year_data, 'bond_strength')
        except KeyError as e:
            raise ValueError(f"Required columns not found: {e}")
            
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for _, row in year_data.iterrows():
            G.add_node(row[node_col], 
                      size=row[node_value_col],
                      bond_strength=row[bond_strength_col],
                      color=self.color_scheme.get(row[node_col], '#888888'))
        
        # Add edges based on bond strengths (simplified)
        nodes = list(G.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Add edge if both nodes have sufficient bond strength
                bs1 = G.nodes[node1]['bond_strength']
                bs2 = G.nodes[node2]['bond_strength']
                if min(bs1, bs2) > 5:  # Threshold for connection
                    G.add_edge(node1, node2, weight=min(bs1, bs2))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=[s*2 for s in node_sizes],  # Scale for visibility
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>Size: %{marker.size}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'Node Network - {year or "Latest"} - {nation or "Analysis"}',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Node size represents Node Value, connections show bond strength relationships",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white'
        )
        
        return fig
    
    def create_dashboard_layout(self, df: pd.DataFrame, 
                              nation: str = None) -> Dict:
        """Create complete dashboard with all visualizations"""
        from src.cams_analyzer import CAMSAnalyzer
        analyzer = CAMSAnalyzer()
        
        dashboard = {}
        
        try:
            dashboard['health_timeline'] = self.plot_system_health_timeline(df, nation)
        except Exception as e:
            print(f"Warning: Could not create health timeline: {e}")
            
        try:
            dashboard['coherence_heatmap'] = self.plot_node_heatmap(df, 'coherence', nation)
        except Exception as e:
            print(f"Warning: Could not create coherence heatmap: {e}")
            
        try:
            dashboard['capacity_heatmap'] = self.plot_node_heatmap(df, 'capacity', nation)
        except Exception as e:
            print(f"Warning: Could not create capacity heatmap: {e}")
            
        try:
            dashboard['stress_distribution'] = self.plot_stress_distribution(df, nation=nation)
        except Exception as e:
            print(f"Warning: Could not create stress distribution: {e}")
            
        try:
            dashboard['dimensions_radar'] = self.plot_four_dimensions_radar(df, nation=nation)
        except Exception as e:
            print(f"Warning: Could not create dimensions radar: {e}")
            
        try:
            dashboard['node_network'] = self.plot_node_network(df, nation=nation)
        except Exception as e:
            print(f"Warning: Could not create node network: {e}")
        
        return dashboard