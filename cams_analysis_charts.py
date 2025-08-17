"""
CAMS Analysis Charts - Comprehensive Visualization Suite
Advanced charts for stress dynamics, phase attractors, and system evolution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob

# Configure page
st.set_page_config(page_title="CAMS Analysis Charts", layout="wide", initial_sidebar_state="expanded")

st.title("📊 CAMS Analysis Charts - Comprehensive Suite")
st.markdown("**Advanced visualization and analysis of stress dynamics across civilizations**")

# === Data Loading Section ===
@st.cache_data
def load_all_cams_datasets():
    """Load all available CAMS datasets"""
    csv_files = glob.glob("*.csv")
    datasets = {}
    
    country_mapping = {
        'australia cams cleaned': 'Australia',
        'usa cams cleaned': 'USA', 
        'france cams cleaned': 'France',
        'denmark cams cleaned': 'Denmark',
        'germany1750 2025': 'Germany',
        'italy cams cleaned': 'Italy',
        'iran cams cleaned': 'Iran',
        'iraq cams cleaned': 'Iraq',
        'lebanon cams cleaned': 'Lebanon',
        'japan 1850 2025': 'Japan',
        'thailand 1850_2025': 'Thailand',
        'netherlands mastersheet': 'Netherlands',
        'canada_cams_2025': 'Canada',
        'saudi arabia master file': 'Saudi Arabia'
    }
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0 and 'Node' in df.columns and 'Coherence' in df.columns:
                base_name = file.replace('.csv', '').lower().replace('_', ' ')
                country_name = country_mapping.get(base_name, base_name.title())
                
                # Ensure numeric columns
                for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['Coherence', 'Capacity', 'Stress', 'Abstraction'])
                
                if len(df) > 0:
                    datasets[country_name] = df
        except:
            continue
    
    return datasets

# Load datasets
with st.spinner("🔄 Loading CAMS datasets..."):
    datasets = load_all_cams_datasets()

if not datasets:
    st.error("No CAMS datasets found! Please ensure CSV files are available.")
    st.stop()

st.success(f"✅ Loaded {len(datasets)} civilizations: {', '.join(datasets.keys())}")

# === Control Panel ===
st.sidebar.markdown("## 🎛️ Analysis Controls")

# Dataset selection
selected_countries = st.sidebar.multiselect(
    "Select Civilizations:",
    options=list(datasets.keys()),
    default=list(datasets.keys())[:3] if len(datasets) >= 3 else list(datasets.keys()),
    help="Choose civilizations for comparative analysis"
)

if not selected_countries:
    st.warning("Please select at least one civilization")
    st.stop()

# Chart type selection
chart_categories = [
    "📈 Time Series Analysis",
    "🌀 Phase Space Dynamics", 
    "🔥 Stress Analysis",
    "⚖️ Node Comparisons",
    "🧠 Meta-Cognitive Functions",
    "📊 Statistical Distributions",
    "🎯 Attractor Analysis",
    "🌐 Multi-Dimensional Analysis",
    "🕸️ Network Analysis"
]

selected_category = st.sidebar.selectbox("Chart Category:", chart_categories)

# Parameters
st.sidebar.markdown("### ⚙️ Parameters")
tau = st.sidebar.slider("Stress Tolerance τ", 2.5, 3.5, 3.0, 0.1)
lambda_param = st.sidebar.slider("Resilience λ", 0.3, 0.7, 0.5, 0.1)
analysis_year = st.sidebar.selectbox("Analysis Year", ["Latest", "All Years"])

# === Data Processing Functions ===
def calculate_enhanced_metrics(df):
    """Calculate all CAMS metrics for a dataset"""
    # Node Fitness
    stress_impact = 1 + np.exp((np.abs(df['Stress']) - tau) / lambda_param)
    df['Fitness'] = (df['Coherence'] * df['Capacity']) / stress_impact * (1 + df['Abstraction']/10)
    
    # Processing Efficiency
    df['ProcessingEfficiency'] = df['Fitness'] / (np.abs(df['Stress']) + 1e-6)
    
    # Coherence Asymmetry
    if 'Year' in df.columns:
        def calc_ca(group):
            cc = group['Coherence'] * group['Capacity']
            return np.std(cc) / (np.mean(cc) + 1e-9)
        df['CoherenceAsymmetry'] = df.groupby('Year')['Fitness'].transform(lambda x: np.std(x) / (np.mean(x) + 1e-9))
    else:
        cc = df['Coherence'] * df['Capacity']
        ca = np.std(cc) / (np.mean(cc) + 1e-9)
        df['CoherenceAsymmetry'] = ca
    
    # Meta-cognitive functions
    df['Monitoring'] = df['Coherence'] * (1 - np.abs(df['Stress']) / 10)
    df['Control'] = df['Capacity'] * np.exp(-np.abs(df['Stress']) / 5)
    df['Reflection'] = df['Abstraction'] * df['Fitness'] / (df['Fitness'].max() + 1e-6)
    
    return df

# Process all selected datasets
processed_data = {}
for country in selected_countries:
    df = datasets[country].copy()
    df = calculate_enhanced_metrics(df)
    processed_data[country] = df

# === Chart Generation Functions ===

def create_time_series_charts():
    """Time series analysis charts"""
    st.markdown("## 📈 Time Series Analysis")
    
    # Combine all countries for time series
    time_series_data = []
    for country, df in processed_data.items():
        if 'Year' in df.columns:
            yearly_metrics = df.groupby('Year').agg({
                'Fitness': 'mean',
                'Stress': 'mean', 
                'Coherence': 'mean',
                'Capacity': 'mean',
                'ProcessingEfficiency': 'mean'
            }).reset_index()
            yearly_metrics['Country'] = country
            time_series_data.append(yearly_metrics)
    
    if time_series_data:
        combined_ts = pd.concat(time_series_data, ignore_index=True)
        
        # Multi-metric time series
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['System Fitness Evolution', 'Stress Trajectories', 
                          'Coherence & Capacity', 'Processing Efficiency'],
            vertical_spacing=0.12
        )
        
        for country in selected_countries:
            country_data = combined_ts[combined_ts['Country'] == country]
            
            # Fitness
            fig.add_trace(go.Scatter(
                x=country_data['Year'], y=country_data['Fitness'],
                name=f'{country} Fitness', mode='lines+markers'
            ), row=1, col=1)
            
            # Stress
            fig.add_trace(go.Scatter(
                x=country_data['Year'], y=country_data['Stress'],
                name=f'{country} Stress', mode='lines'
            ), row=1, col=2)
            
            # Coherence & Capacity
            fig.add_trace(go.Scatter(
                x=country_data['Year'], y=country_data['Coherence'],
                name=f'{country} Coherence', mode='lines'
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=country_data['Year'], y=country_data['Capacity'],
                name=f'{country} Capacity', mode='lines', line=dict(dash='dash')
            ), row=2, col=1)
            
            # Processing Efficiency
            fig.add_trace(go.Scatter(
                x=country_data['Year'], y=country_data['ProcessingEfficiency'],
                name=f'{country} SPE', mode='lines+markers'
            ), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Temporal Evolution Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical trend analysis
        st.markdown("### 📊 Trend Statistics")
        trend_stats = []
        for country in selected_countries:
            country_data = combined_ts[combined_ts['Country'] == country]
            if len(country_data) > 1:
                fitness_slope = stats.linregress(country_data['Year'], country_data['Fitness']).slope
                stress_slope = stats.linregress(country_data['Year'], country_data['Stress']).slope
                
                trend_stats.append({
                    'Country': country,
                    'Fitness_Trend': 'Rising' if fitness_slope > 0.01 else 'Falling' if fitness_slope < -0.01 else 'Stable',
                    'Stress_Trend': 'Rising' if stress_slope > 0.01 else 'Falling' if stress_slope < -0.01 else 'Stable',
                    'Fitness_Rate': f"{fitness_slope:.4f}/year",
                    'Stress_Rate': f"{stress_slope:.4f}/year"
                })
        
        if trend_stats:
            st.dataframe(pd.DataFrame(trend_stats), use_container_width=True)

def create_phase_space_charts():
    """Phase space dynamics visualization"""
    st.markdown("## 🌀 Phase Space Dynamics")
    
    # 3D Phase space
    fig_3d = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(selected_countries)]
    
    for i, (country, df) in enumerate(processed_data.items()):
        latest_data = df.tail(8) if 'Year' in df.columns else df
        
        fig_3d.add_trace(go.Scatter3d(
            x=latest_data['Coherence'],
            y=latest_data['Capacity'], 
            z=latest_data['Stress'],
            mode='markers+text',
            text=latest_data['Node'],
            textposition='top center',
            marker=dict(
                size=latest_data['Fitness'] * 3,
                color=colors[i],
                opacity=0.8,
                line=dict(color='black', width=1)
            ),
            name=country
        ))
    
    fig_3d.update_layout(
        title="3D Phase Space: Coherence-Capacity-Stress",
        scene=dict(
            xaxis_title="Coherence",
            yaxis_title="Capacity", 
            zaxis_title="Stress",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        height=600
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Phase attractors analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Coherence vs Capacity
        fig_cc = go.Figure()
        for i, (country, df) in enumerate(processed_data.items()):
            latest_data = df.tail(8) if 'Year' in df.columns else df
            fig_cc.add_trace(go.Scatter(
                x=latest_data['Coherence'],
                y=latest_data['Capacity'],
                mode='markers+text',
                text=latest_data['Node'],
                textposition='top center',
                marker=dict(size=10, color=colors[i]),
                name=country
            ))
        
        fig_cc.update_layout(
            title="Coherence-Capacity Phase Space",
            xaxis_title="Coherence",
            yaxis_title="Capacity"
        )
        st.plotly_chart(fig_cc, use_container_width=True)
    
    with col2:
        # Stress vs Fitness
        fig_sf = go.Figure()
        for i, (country, df) in enumerate(processed_data.items()):
            latest_data = df.tail(8) if 'Year' in df.columns else df
            fig_sf.add_trace(go.Scatter(
                x=latest_data['Stress'],
                y=latest_data['Fitness'],
                mode='markers+text',
                text=latest_data['Node'],
                textposition='top center',
                marker=dict(size=10, color=colors[i]),
                name=country
            ))
        
        fig_sf.update_layout(
            title="Stress-Fitness Phase Space",
            xaxis_title="Stress",
            yaxis_title="Fitness"
        )
        st.plotly_chart(fig_sf, use_container_width=True)

def create_stress_analysis_charts():
    """Comprehensive stress analysis"""
    st.markdown("## 🔥 Stress Analysis")
    
    # Stress distribution comparison
    fig_dist = go.Figure()
    
    for country, df in processed_data.items():
        latest_data = df.tail(8) if 'Year' in df.columns else df
        fig_dist.add_trace(go.Box(
            y=latest_data['Stress'],
            name=country,
            boxpoints='all',
            jitter=0.3
        ))
    
    fig_dist.update_layout(
        title="Stress Distribution by Civilization",
        yaxis_title="Stress Level",
        height=500
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Stress heatmap by node
    col1, col2 = st.columns(2)
    
    with col1:
        # Create stress matrix
        stress_matrix = []
        node_names = []
        for country, df in processed_data.items():
            latest_data = df.tail(8) if 'Year' in df.columns else df
            if len(stress_matrix) == 0:
                node_names = latest_data['Node'].tolist()
            stress_matrix.append(latest_data['Stress'].tolist())
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=stress_matrix,
            x=node_names,
            y=selected_countries,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Stress Level")
        ))
        
        fig_heatmap.update_layout(
            title="Stress Heatmap: Countries vs Nodes",
            xaxis_title="Institutional Nodes",
            yaxis_title="Civilizations"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Stress vs Processing Efficiency
        fig_spe = go.Figure()
        
        for country, df in processed_data.items():
            latest_data = df.tail(8) if 'Year' in df.columns else df
            fig_spe.add_trace(go.Scatter(
                x=latest_data['Stress'],
                y=latest_data['ProcessingEfficiency'],
                mode='markers',
                marker=dict(size=12),
                name=country,
                text=latest_data['Node'],
                hovertemplate="<b>%{text}</b><br>Stress: %{x:.2f}<br>SPE: %{y:.2f}<extra></extra>"
            ))
        
        # Add trend line
        all_stress = []
        all_spe = []
        for df in processed_data.values():
            latest_data = df.tail(8) if 'Year' in df.columns else df
            all_stress.extend(latest_data['Stress'].tolist())
            all_spe.extend(latest_data['ProcessingEfficiency'].tolist())
        
        if len(all_stress) > 1:
            slope, intercept, r_value, _, _ = stats.linregress(all_stress, all_spe)
            line_x = np.array([min(all_stress), max(all_stress)])
            line_y = slope * line_x + intercept
            
            fig_spe.add_trace(go.Scatter(
                x=line_x, y=line_y,
                mode='lines',
                line=dict(dash='dash', color='red'),
                name=f'Trend (R²={r_value**2:.3f})'
            ))
        
        fig_spe.update_layout(
            title="Stress vs Processing Efficiency",
            xaxis_title="Stress",
            yaxis_title="Processing Efficiency"
        )
        st.plotly_chart(fig_spe, use_container_width=True)

def create_node_comparison_charts():
    """Node-level comparative analysis"""
    st.markdown("## ⚖️ Node Comparisons")
    
    # Radar chart for node capabilities
    all_nodes = set()
    for df in processed_data.values():
        all_nodes.update(df['Node'].tolist())
    
    selected_nodes = st.multiselect(
        "Select Nodes for Comparison:",
        options=sorted(list(all_nodes)),
        default=sorted(list(all_nodes))[:4]
    )
    
    if selected_nodes:
        # Multi-dimensional radar chart
        fig_radar = go.Figure()
        
        metrics = ['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Fitness']
        
        for country, df in processed_data.items():
            latest_data = df.tail(len(df)) if 'Year' in df.columns else df
            
            for node in selected_nodes:
                node_data = latest_data[latest_data['Node'] == node]
                if len(node_data) > 0:
                    node_values = [
                        node_data['Coherence'].iloc[0],
                        node_data['Capacity'].iloc[0],
                        10 - abs(node_data['Stress'].iloc[0]),  # Invert stress for radar
                        node_data['Abstraction'].iloc[0],
                        node_data['Fitness'].iloc[0]
                    ]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=node_values,
                        theta=metrics,
                        fill='toself',
                        name=f'{country}-{node}',
                        opacity=0.6
                    ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            title="Multi-Dimensional Node Analysis",
            height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Node performance ranking
        st.markdown("### 🏆 Node Performance Ranking")
        
        ranking_data = []
        for country, df in processed_data.items():
            latest_data = df.tail(len(df)) if 'Year' in df.columns else df
            for _, row in latest_data.iterrows():
                ranking_data.append({
                    'Country': country,
                    'Node': row['Node'],
                    'Fitness': row['Fitness'],
                    'Coherence': row['Coherence'],
                    'Capacity': row['Capacity'],
                    'Stress': abs(row['Stress']),
                    'Abstraction': row['Abstraction']
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Fitness', ascending=False)
        
        st.dataframe(ranking_df, use_container_width=True)

def create_metacognitive_charts():
    """Meta-cognitive function analysis"""
    st.markdown("## 🧠 Meta-Cognitive Functions")
    
    # Meta-cognitive function comparison
    fig_meta = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Monitoring', 'Control', 'Reflection'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for country, df in processed_data.items():
        latest_data = df.tail(8) if 'Year' in df.columns else df
        
        # Monitoring
        fig_meta.add_trace(go.Bar(
            x=latest_data['Node'],
            y=latest_data['Monitoring'],
            name=f'{country} Mon.',
            opacity=0.7
        ), row=1, col=1)
        
        # Control
        fig_meta.add_trace(go.Bar(
            x=latest_data['Node'],
            y=latest_data['Control'],
            name=f'{country} Ctrl',
            opacity=0.7
        ), row=1, col=2)
        
        # Reflection
        fig_meta.add_trace(go.Bar(
            x=latest_data['Node'],
            y=latest_data['Reflection'],
            name=f'{country} Refl.',
            opacity=0.7
        ), row=1, col=3)
    
    fig_meta.update_layout(height=500, title_text="Meta-Cognitive Function Analysis")
    st.plotly_chart(fig_meta, use_container_width=True)
    
    # Meta-cognitive balance analysis
    st.markdown("### ⚖️ Meta-Cognitive Balance")
    
    balance_data = []
    for country, df in processed_data.items():
        latest_data = df.tail(8) if 'Year' in df.columns else df
        
        total_monitoring = latest_data['Monitoring'].sum()
        total_control = latest_data['Control'].sum()
        total_reflection = latest_data['Reflection'].sum()
        total = total_monitoring + total_control + total_reflection
        
        if total > 0:
            balance_data.append({
                'Country': country,
                'Monitoring_Ratio': total_monitoring / total,
                'Control_Ratio': total_control / total,
                'Reflection_Ratio': total_reflection / total,
                'Balance_Score': 1 - np.std([total_monitoring/total, total_control/total, total_reflection/total])
            })
    
    balance_df = pd.DataFrame(balance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stacked bar for ratios
        fig_balance = go.Figure()
        
        fig_balance.add_trace(go.Bar(
            name='Monitoring',
            x=balance_df['Country'],
            y=balance_df['Monitoring_Ratio']
        ))
        
        fig_balance.add_trace(go.Bar(
            name='Control',
            x=balance_df['Country'],
            y=balance_df['Control_Ratio']
        ))
        
        fig_balance.add_trace(go.Bar(
            name='Reflection',
            x=balance_df['Country'],
            y=balance_df['Reflection_Ratio']
        ))
        
        fig_balance.update_layout(
            barmode='stack',
            title='Meta-Cognitive Function Distribution',
            yaxis_title='Proportion'
        )
        st.plotly_chart(fig_balance, use_container_width=True)
    
    with col2:
        # Balance scores
        fig_score = go.Figure(data=go.Bar(
            x=balance_df['Country'],
            y=balance_df['Balance_Score'],
            marker_color='lightblue'
        ))
        
        fig_score.update_layout(
            title='Meta-Cognitive Balance Score',
            yaxis_title='Balance Score (higher = more balanced)'
        )
        st.plotly_chart(fig_score, use_container_width=True)

def create_statistical_charts():
    """Statistical distribution analysis"""
    st.markdown("## 📊 Statistical Distributions")
    
    # Correlation matrix
    all_data = []
    for country, df in processed_data.items():
        latest_data = df.tail(8) if 'Year' in df.columns else df
        country_data = latest_data[['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Fitness']].copy()
        country_data['Country'] = country
        all_data.append(country_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Correlation heatmap
    corr_matrix = combined_data[['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Fitness']].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig_corr.update_layout(
        title="Correlation Matrix: CAMS Dimensions",
        height=500
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Distribution comparisons
    col1, col2 = st.columns(2)
    
    with col1:
        # Fitness distributions
        fig_dist = go.Figure()
        
        for country in selected_countries:
            country_data = combined_data[combined_data['Country'] == country]
            fig_dist.add_trace(go.Histogram(
                x=country_data['Fitness'],
                name=country,
                opacity=0.7,
                nbinsx=15
            ))
        
        fig_dist.update_layout(
            title="Fitness Distribution Comparison",
            xaxis_title="Fitness",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # PCA Analysis
        numeric_data = combined_data[['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Fitness']]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        fig_pca = go.Figure()
        
        for country in selected_countries:
            country_indices = combined_data[combined_data['Country'] == country].index
            fig_pca.add_trace(go.Scatter(
                x=pca_result[country_indices, 0],
                y=pca_result[country_indices, 1],
                mode='markers',
                marker=dict(size=10),
                name=country
            ))
        
        fig_pca.update_layout(
            title=f"PCA Analysis (Explained Variance: {sum(pca.explained_variance_ratio_):.1%})",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

def create_attractor_analysis_charts():
    """Phase attractor identification and analysis"""
    st.markdown("## 🎯 Attractor Analysis")
    
    # Define attractor regions
    def identify_attractor(health, spe, ca):
        if health > 3.5 and spe > 2.0 and ca < 0.3:
            return "Adaptive"
        elif 2.5 <= health <= 3.5:
            return "Authoritarian"
        elif 1.5 <= health <= 2.5:
            return "Fragmented"  
        else:
            return "Collapse"
    
    # Calculate system-level metrics
    attractor_data = []
    for country, df in processed_data.items():
        latest_data = df.tail(8) if 'Year' in df.columns else df
        
        # System health (geometric mean of fitness)
        safe_fitness = np.maximum(latest_data['Fitness'], 1e-6)
        system_health = np.exp(np.mean(np.log(safe_fitness)))
        
        # System processing efficiency
        spe = np.mean(latest_data['ProcessingEfficiency'])
        
        # Coherence asymmetry
        cc = latest_data['Coherence'] * latest_data['Capacity'] 
        ca = np.std(cc) / (np.mean(cc) + 1e-9)
        
        attractor = identify_attractor(system_health, spe, ca)
        
        attractor_data.append({
            'Country': country,
            'SystemHealth': system_health,
            'ProcessingEfficiency': spe,
            'CoherenceAsymmetry': ca,
            'Attractor': attractor
        })
    
    attractor_df = pd.DataFrame(attractor_data)
    
    # Attractor landscape
    fig_landscape = go.Figure()
    
    attractor_colors = {
        'Adaptive': 'green',
        'Authoritarian': 'orange', 
        'Fragmented': 'red',
        'Collapse': 'darkred'
    }
    
    for attractor in attractor_df['Attractor'].unique():
        subset = attractor_df[attractor_df['Attractor'] == attractor]
        fig_landscape.add_trace(go.Scatter(
            x=subset['SystemHealth'],
            y=subset['ProcessingEfficiency'],
            mode='markers+text',
            text=subset['Country'],
            textposition='top center',
            marker=dict(
                size=15,
                color=attractor_colors.get(attractor, 'gray'),
                line=dict(color='black', width=1)
            ),
            name=attractor
        ))
    
    # Add attractor regions as background
    fig_landscape.add_shape(
        type="rect",
        x0=3.5, y0=2.0, x1=10, y1=10,
        fillcolor="green", opacity=0.1,
        line=dict(color="green", width=2)
    )
    
    fig_landscape.add_annotation(
        x=5, y=4,
        text="Adaptive Region",
        showarrow=False,
        font=dict(color="green", size=14)
    )
    
    fig_landscape.update_layout(
        title="Civilizational Attractor Landscape",
        xaxis_title="System Health Ψ(t)",
        yaxis_title="Processing Efficiency SPE(t)",
        height=600
    )
    st.plotly_chart(fig_landscape, use_container_width=True)
    
    # Attractor summary
    st.markdown("### 📋 Attractor Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(attractor_df, use_container_width=True)
    
    with col2:
        # Attractor distribution
        attractor_counts = attractor_df['Attractor'].value_counts()
        
        fig_pie = go.Figure(data=go.Pie(
            labels=attractor_counts.index,
            values=attractor_counts.values,
            marker_colors=[attractor_colors.get(a, 'gray') for a in attractor_counts.index]
        ))
        
        fig_pie.update_layout(
            title="Attractor Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def create_multidimensional_analysis():
    """Multi-dimensional analysis with advanced techniques"""
    st.markdown("## 🌐 Multi-Dimensional Analysis")
    
    # Prepare comprehensive dataset
    all_features = []
    for country, df in processed_data.items():
        latest_data = df.tail(8) if 'Year' in df.columns else df
        
        feature_vector = {
            'Country': country,
            'Mean_Coherence': latest_data['Coherence'].mean(),
            'Mean_Capacity': latest_data['Capacity'].mean(),
            'Mean_Stress': latest_data['Stress'].abs().mean(),
            'Mean_Abstraction': latest_data['Abstraction'].mean(),
            'Mean_Fitness': latest_data['Fitness'].mean(),
            'Std_Coherence': latest_data['Coherence'].std(),
            'Std_Capacity': latest_data['Capacity'].std(),
            'Std_Stress': latest_data['Stress'].std(),
            'Max_Fitness': latest_data['Fitness'].max(),
            'Min_Fitness': latest_data['Fitness'].min(),
            'Node_Count': len(latest_data)
        }
        all_features.append(feature_vector)
    
    features_df = pd.DataFrame(all_features)
    
    # Parallel coordinates plot
    fig_parallel = go.Figure(data=go.Parcoords(
        line=dict(color=np.arange(len(features_df)),
                 colorscale='Viridis',
                 showscale=True),
        dimensions=[
            dict(range=[features_df['Mean_Coherence'].min(), features_df['Mean_Coherence'].max()],
                 label='Coherence', values=features_df['Mean_Coherence']),
            dict(range=[features_df['Mean_Capacity'].min(), features_df['Mean_Capacity'].max()],
                 label='Capacity', values=features_df['Mean_Capacity']),
            dict(range=[features_df['Mean_Stress'].min(), features_df['Mean_Stress'].max()],
                 label='Stress', values=features_df['Mean_Stress']),
            dict(range=[features_df['Mean_Abstraction'].min(), features_df['Mean_Abstraction'].max()],
                 label='Abstraction', values=features_df['Mean_Abstraction']),
            dict(range=[features_df['Mean_Fitness'].min(), features_df['Mean_Fitness'].max()],
                 label='Fitness', values=features_df['Mean_Fitness'])
        ]
    ))
    
    fig_parallel.update_layout(
        title="Multi-Dimensional Civilization Profiles",
        height=600
    )
    st.plotly_chart(fig_parallel, use_container_width=True)
    
    # Civilization similarity matrix
    numeric_features = features_df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    
    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(scaled_features, metric='euclidean')
    distance_matrix = squareform(distances)
    
    fig_similarity = go.Figure(data=go.Heatmap(
        z=distance_matrix,
        x=features_df['Country'],
        y=features_df['Country'],
        colorscale='Viridis_r',
        colorbar=dict(title="Similarity<br>(lower = more similar)")
    ))
    
    fig_similarity.update_layout(
        title="Civilization Similarity Matrix",
        height=500
    )
    st.plotly_chart(fig_similarity, use_container_width=True)

def create_network_analysis_charts():
    """Network analysis and resilience visualization"""
    st.markdown("## 🕸️ CAMS Network Analysis")
    
    # Network parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        layout_type = st.selectbox("Network Layout", 
                                  ["spring", "circular", "kamada_kawai", "random"])
    
    with col2:
        node_size_scale = st.slider("Node Size Scale", 50, 500, 200, 50)
        
    with col3:
        edge_weight_scale = st.slider("Edge Weight Scale", 0.1, 3.0, 1.0, 0.1)
    
    # Define meta-learning and capacity nodes based on CAMS framework
    meta_learning_nodes = ["Executive", "Priesthood", "State Memory", "StateMemory"]
    capacity_nodes = ["Army", "Craft", "Trades", "Property Owners", "Proletariat", 
                     "Merchants", "Stewards", "Flow", "Hands"]
    
    # Color mapping
    color_map = {
        "meta": "#FF6B6B",      # Red for meta-learning
        "capacity": "#4ECDC4",   # Teal for capacity
        "mixed": "#45B7D1"      # Blue for mixed/other
    }
    
    # Create network visualization function
    def plot_cams_resilience_network(df, nation_name, ax):
        """Plot CAMS resilience network for a single nation"""
        G = nx.Graph()
        
        # Add nodes with classification
        for _, row in df.iterrows():
            node_name = row["Node"]
            if node_name in meta_learning_nodes:
                node_type = "meta"
            elif node_name in capacity_nodes:
                node_type = "capacity"
            else:
                node_type = "mixed"
                
            # Use fitness as node value, fallback to coherence*capacity
            if 'Fitness' in row:
                node_value = max(row['Fitness'], 0.1)
            else:
                node_value = max(row['Coherence'] * row['Capacity'], 0.1)
            
            G.add_node(node_name, 
                      value=node_value,
                      stress=abs(row["Stress"]), 
                      node_type=node_type,
                      coherence=row["Coherence"],
                      capacity=row["Capacity"])
        
        # Add edges with stress-resilience weighting
        nodes_list = list(df.iterrows())
        for i, (idx1, row1) in enumerate(nodes_list):
            for j, (idx2, row2) in enumerate(nodes_list):
                if i < j:  # Avoid duplicate edges
                    avg_stress = (abs(row1["Stress"]) + abs(row2["Stress"])) / 2
                    # Weight based on stress interaction - higher stress = stronger connections
                    weight = (avg_stress + 0.1) * edge_weight_scale
                    G.add_edge(row1["Node"], row2["Node"], weight=weight)
        
        # Choose layout
        if layout_type == "spring":
            pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        elif layout_type == "circular":
            pos = nx.circular_layout(G)
        elif layout_type == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G, seed=42)
        
        # Node properties
        node_sizes = [G.nodes[node]["value"] * node_size_scale for node in G.nodes()]
        node_colors = [color_map[G.nodes[node]["node_type"]] for node in G.nodes()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                              alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, 
                              edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(f"{nation_name} Resilience Network\n(Meta-Learning vs Capacity Nodes)", 
                    fontsize=12, pad=20)
        ax.axis("off")
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["meta"], 
                      markersize=10, label='Meta-Learning'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["capacity"], 
                      markersize=10, label='Capacity'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["mixed"], 
                      markersize=10, label='Mixed/Other')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        return G
    
    # Multi-nation network comparison
    st.markdown("### 🌐 Multi-Nation Network Comparison")
    
    if len(selected_countries) <= 4:
        # Create matplotlib subplot grid
        n_countries = len(selected_countries)
        if n_countries == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = [axes]
        elif n_countries == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        elif n_countries == 3:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        
        networks = {}
        for idx, country in enumerate(selected_countries):
            df = processed_data[country]
            latest_data = df.tail(8) if 'Year' in df.columns else df
            
            if idx < len(axes):
                G = plot_cams_resilience_network(latest_data, country, axes[idx])
                networks[country] = G
        
        # Hide unused subplots
        for idx in range(len(selected_countries), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("CAMS Network Comparison - Stress Resilience Analysis", 
                    fontsize=16, y=0.95)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Network metrics analysis
        st.markdown("### 📊 Network Metrics Analysis")
        
        metrics_data = []
        for country, G in networks.items():
            if len(G.nodes()) > 0:
                # Calculate network metrics
                density = nx.density(G)
                try:
                    avg_clustering = nx.average_clustering(G)
                except:
                    avg_clustering = 0
                
                # Centrality measures
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                
                # Node type distribution
                meta_count = sum(1 for node in G.nodes() if G.nodes[node]['node_type'] == 'meta')
                capacity_count = sum(1 for node in G.nodes() if G.nodes[node]['node_type'] == 'capacity')
                mixed_count = sum(1 for node in G.nodes() if G.nodes[node]['node_type'] == 'mixed')
                
                # Stress resilience metrics
                avg_stress = np.mean([G.nodes[node]['stress'] for node in G.nodes()])
                avg_value = np.mean([G.nodes[node]['value'] for node in G.nodes()])
                
                metrics_data.append({
                    'Country': country,
                    'Nodes': len(G.nodes()),
                    'Edges': len(G.edges()),
                    'Density': density,
                    'Avg_Clustering': avg_clustering,
                    'Meta_Nodes': meta_count,
                    'Capacity_Nodes': capacity_count,
                    'Mixed_Nodes': mixed_count,
                    'Avg_Stress': avg_stress,
                    'Avg_Node_Value': avg_value,
                    'Resilience_Ratio': avg_value / (avg_stress + 1e-6)
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Network comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Network density comparison
            fig_density = go.Figure()
            fig_density.add_trace(go.Bar(
                x=metrics_df['Country'],
                y=metrics_df['Density'],
                marker_color='lightblue',
                text=metrics_df['Density'].round(3),
                textposition='auto'
            ))
            fig_density.update_layout(
                title="Network Density Comparison",
                yaxis_title="Density",
                height=400
            )
            st.plotly_chart(fig_density, use_container_width=True)
        
        with col2:
            # Resilience ratio comparison
            fig_resilience = go.Figure()
            fig_resilience.add_trace(go.Bar(
                x=metrics_df['Country'],
                y=metrics_df['Resilience_Ratio'],
                marker_color='lightgreen',
                text=metrics_df['Resilience_Ratio'].round(2),
                textposition='auto'
            ))
            fig_resilience.update_layout(
                title="Resilience Ratio (Value/Stress)",
                yaxis_title="Resilience Ratio",
                height=400
            )
            st.plotly_chart(fig_resilience, use_container_width=True)
        
        # Individual network analysis
        st.markdown("### 🔍 Individual Network Analysis")
        
        selected_network = st.selectbox("Select Country for Detailed Analysis:", 
                                       selected_countries)
        
        if selected_network in networks:
            G = networks[selected_network]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Node Centrality**")
                degree_cent = nx.degree_centrality(G)
                centrality_data = [(node, cent) for node, cent in 
                                 sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)]
                
                for node, cent in centrality_data[:5]:
                    st.write(f"- {node}: {cent:.3f}")
            
            with col2:
                st.markdown("**High Stress Nodes**")
                stress_data = [(node, G.nodes[node]['stress']) for node in G.nodes()]
                stress_data.sort(key=lambda x: x[1], reverse=True)
                
                for node, stress in stress_data[:5]:
                    st.write(f"- {node}: {stress:.2f}")
            
            with col3:
                st.markdown("**High Value Nodes**")
                value_data = [(node, G.nodes[node]['value']) for node in G.nodes()]
                value_data.sort(key=lambda x: x[1], reverse=True)
                
                for node, value in value_data[:5]:
                    st.write(f"- {node}: {value:.2f}")
        
    else:
        st.warning("Please select 4 or fewer countries for network visualization")
    
    # Interactive network (using Plotly for single country)
    if len(selected_countries) == 1:
        st.markdown("### 🎯 Interactive Network Visualization")
        
        country = selected_countries[0]
        df = processed_data[country]
        latest_data = df.tail(8) if 'Year' in df.columns else df
        
        # Create networkx graph
        G = nx.Graph()
        
        for _, row in latest_data.iterrows():
            node_name = row["Node"]
            if node_name in meta_learning_nodes:
                node_type = "meta"
            elif node_name in capacity_nodes:
                node_type = "capacity"  
            else:
                node_type = "mixed"
                
            node_value = max(row.get('Fitness', row['Coherence'] * row['Capacity']), 0.1)
            
            G.add_node(node_name, 
                      value=node_value,
                      stress=abs(row["Stress"]), 
                      node_type=node_type,
                      coherence=row["Coherence"],
                      capacity=row["Capacity"])
        
        # Add edges
        nodes_list = list(latest_data.iterrows())
        for i, (idx1, row1) in enumerate(nodes_list):
            for j, (idx2, row2) in enumerate(nodes_list):
                if i < j:
                    avg_stress = (abs(row1["Stress"]) + abs(row2["Stress"])) / 2
                    weight = (avg_stress + 0.1) * edge_weight_scale
                    G.add_edge(row1["Node"], row2["Node"], weight=weight)
        
        # Create Plotly network visualization
        pos = nx.spring_layout(G, seed=42)
        
        # Edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} - {edge[1]}: Weight {weight:.2f}")
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=2, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            node_info = G.nodes[node]
            text = f"{node}<br>Type: {node_info['node_type']}<br>"
            text += f"Value: {node_info['value']:.2f}<br>"
            text += f"Stress: {node_info['stress']:.2f}<br>"
            text += f"Coherence: {node_info['coherence']:.2f}<br>"
            text += f"Capacity: {node_info['capacity']:.2f}"
            node_text.append(text)
            
            # Color by type
            node_color.append(color_map[node_info['node_type']])
            node_size.append(node_info['value'] * 20)
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=[node for node in G.nodes()],
                               textposition="middle center",
                               hovertext=node_text,
                               marker=dict(size=node_size,
                                         color=node_color,
                                         line=dict(width=2, color='black')))
        
        # Create figure
        fig_interactive = go.Figure(data=[edge_trace, node_trace],
                                   layout=go.Layout(
                                       title=f'Interactive Network: {country}',
                                       titlefont_size=16,
                                       showlegend=False,
                                       hovermode='closest',
                                       margin=dict(b=20,l=5,r=5,t=40),
                                       annotations=[ dict(
                                           text="Node size = Value, Color = Type (Red=Meta, Teal=Capacity)",
                                           showarrow=False,
                                           xref="paper", yref="paper",
                                           x=0.005, y=-0.002,
                                           xanchor="left", yanchor="bottom",
                                           font=dict(color="#888", size=12)
                                       )],
                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       height=600))
        
        st.plotly_chart(fig_interactive, use_container_width=True)

# === Main Chart Rendering ===
if selected_category == "📈 Time Series Analysis":
    create_time_series_charts()
elif selected_category == "🌀 Phase Space Dynamics":
    create_phase_space_charts()
elif selected_category == "🔥 Stress Analysis":
    create_stress_analysis_charts()
elif selected_category == "⚖️ Node Comparisons":
    create_node_comparison_charts()
elif selected_category == "🧠 Meta-Cognitive Functions":
    create_metacognitive_charts()
elif selected_category == "📊 Statistical Distributions":
    create_statistical_charts()
elif selected_category == "🎯 Attractor Analysis":
    create_attractor_analysis_charts()
elif selected_category == "🌐 Multi-Dimensional Analysis":
    create_multidimensional_analysis()
elif selected_category == "🕸️ Network Analysis":
    create_network_analysis_charts()

# === Footer ===
st.markdown("---")
st.markdown("### 📚 Analysis Framework")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Implemented Charts:**
    - Time series evolution
    - 3D phase space dynamics
    - Stress distribution analysis
    - Node performance comparison
    """)

with col2:
    st.markdown("""
    **Advanced Analytics:**
    - Meta-cognitive functions
    - Statistical correlations
    - PCA dimensionality reduction
    - Attractor classification
    """)

with col3:
    st.markdown("""
    **Multi-Dimensional:**
    - Parallel coordinates
    - Similarity matrices  
    - Cross-civilization comparison
    - Network analysis & resilience
    - Interactive visualizations
    """)

st.success("🎉 Comprehensive CAMS analysis charts ready for exploration!")