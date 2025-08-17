"""
FIXED CAMS Framework Analysis Dashboard
Complex Adaptive Model State (CAMS) Framework - Analyzing societies as Complex Adaptive Systems
Developed by Kari McKern | Fixed version with proper data loading and display
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="CAMS Framework Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Enhanced CSS Styling ===
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-healthy {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: #2d5a27;
        font-weight: bold;
    }
    .status-stressed {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
    }
    .status-moderate {
        background: linear-gradient(135deg, #ffd93d 0%, #ff6b35 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: #8b4513;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.8rem;">🧠 CAMS Framework Analysis Dashboard</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Complex Adaptive Model State (CAMS) Framework - Analyzing societies as Complex Adaptive Systems
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.8;">
        <strong>Developed by Kari McKern</strong> | Quantifying Coherence, Capacity, Stress, and Abstraction across societal nodes
    </p>
</div>
""", unsafe_allow_html=True)

# === Data Loading Functions ===
@st.cache_data
def load_all_cams_datasets():
    """Load all available CAMS datasets with enhanced error handling"""
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
            if len(df) > 0 and 'Node' in df.columns:
                base_name = file.replace('.csv', '').replace('.CSV', '').lower().strip()
                base_name = base_name.replace('_', ' ').replace(' (2)', '').replace(' - ', ' ')
                
                country_name = country_mapping.get(base_name, base_name.title())
                
                # Enhanced column detection and cleaning
                required_cols = ['Coherence', 'Capacity', 'Stress', 'Abstraction']
                available_cols = []
                
                for col in required_cols:
                    if col in df.columns:
                        available_cols.append(col)
                    else:
                        # Try alternative column names
                        alternatives = {
                            'Coherence': ['coherence', 'Coh', 'C'],
                            'Capacity': ['capacity', 'Cap', 'K', 'Capability'],
                            'Stress': ['stress', 'S'],
                            'Abstraction': ['abstraction', 'Abstract', 'A', 'Abs']
                        }
                        for alt in alternatives.get(col, []):
                            if alt in df.columns:
                                df[col] = df[alt]
                                available_cols.append(col)
                                break
                
                # Only proceed if we have the core columns
                if len(available_cols) >= 3:
                    # Ensure numeric columns
                    for col in available_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove rows with missing data
                    df = df.dropna(subset=available_cols)
                    
                    if len(df) > 0:
                        # Add missing columns with defaults if needed
                        for col in required_cols:
                            if col not in df.columns:
                                if col == 'Coherence':
                                    df[col] = 5.0  # Default coherence
                                elif col == 'Capacity':
                                    df[col] = 5.0  # Default capacity
                                elif col == 'Stress':
                                    df[col] = 0.0  # Default stress
                                elif col == 'Abstraction':
                                    df[col] = 5.0  # Default abstraction
                        
                        # Get time information
                        if 'Year' in df.columns:
                            years_range = f"{int(df['Year'].min())}-{int(df['Year'].max())}"
                        else:
                            years_range = 'Current'
                        
                        datasets[country_name] = {
                            'data': df,
                            'records': len(df),
                            'years': years_range,
                            'filename': file
                        }
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
            continue
    
    return datasets

# === CAMS Analysis Functions ===
def calculate_node_fitness(coherence, capacity, stress, abstraction, tau=3.0, lambda_param=0.5):
    """Calculate node fitness using CAMS formula"""
    # Ensure inputs are numeric
    coherence = float(coherence) if coherence is not None else 5.0
    capacity = float(capacity) if capacity is not None else 5.0
    stress = float(stress) if stress is not None else 0.0
    abstraction = float(abstraction) if abstraction is not None else 5.0
    
    # Stress impact calculation (avoid overflow)
    stress_normalized = min(max((abs(stress) - tau) / lambda_param, -10), 10)
    stress_impact = 1 + np.exp(stress_normalized)
    
    # Node fitness calculation
    fitness = (coherence * capacity / stress_impact) * (1 + abstraction / 10)
    return max(fitness, 0.001)  # Ensure positive

def calculate_system_health(df):
    """Calculate system health as geometric mean of node fitness"""
    if df is None or len(df) == 0:
        return 0.0
    
    fitness_values = []
    for _, row in df.iterrows():
        fitness = calculate_node_fitness(
            row.get('Coherence', 5.0),
            row.get('Capacity', 5.0), 
            row.get('Stress', 0.0),
            row.get('Abstraction', 5.0)
        )
        fitness_values.append(fitness)
    
    if not fitness_values:
        return 0.0
    
    # Geometric mean
    return np.exp(np.mean(np.log(fitness_values)))

def classify_civilization_type(health, avg_stress, coherence_std):
    """Classify civilization based on CAMS metrics"""
    if health > 4.0 and avg_stress < 2.0:
        return "🟢 Adaptive/Flourishing"
    elif health > 3.0 and coherence_std < 1.5:
        return "🟡 Stable/Coordinated"
    elif health > 2.0:
        return "🟠 Stressed/Transitional"
    elif health > 1.0:
        return "🔴 Fragmented/Critical"
    else:
        return "⚫ Collapse/Terminal"

def calculate_esd(df):
    """Calculate Effective Stress Distribution"""
    if df is None or len(df) == 0:
        return 0.0
    
    stress_values = df['Stress'].values
    weights = df['Capacity'].values / (df['Capacity'].sum() + 1e-6)
    
    # Weighted stress distribution
    weighted_stress = np.sum(weights * np.abs(stress_values))
    return weighted_stress

# === Load Data ===
with st.spinner("🔄 Loading CAMS datasets..."):
    datasets = load_all_cams_datasets()

if not datasets:
    st.error("❌ No CAMS datasets found! Please ensure CSV files are available.")
    
    # Create sample data for demonstration
    st.warning("🔧 Using sample data for demonstration")
    sample_data = pd.DataFrame({
        'Node': ['Executive', 'Army', 'StateMemory', 'Priesthood', 'Stewards', 'Craft', 'Flow', 'Hands'],
        'Coherence': [6.5, 7.2, 5.8, 6.9, 5.5, 6.1, 4.8, 5.2],
        'Capacity': [7.1, 8.0, 6.2, 6.5, 6.8, 7.3, 5.9, 6.0],
        'Stress': [2.1, 1.8, 3.2, 1.5, 2.8, 2.3, 3.5, 2.9],
        'Abstraction': [7.8, 5.9, 8.5, 8.2, 6.1, 6.4, 5.7, 4.8]
    })
    datasets = {'Sample Civilization': {
        'data': sample_data,
        'records': len(sample_data),
        'years': 'Current',
        'filename': 'sample_data.csv'
    }}

st.success(f"✅ Loaded {len(datasets)} civilizations: {', '.join(datasets.keys())}")

# === Sidebar Controls ===
st.sidebar.markdown("## 🎛️ Analysis Controls")

# Dataset selection
selected_country = st.sidebar.selectbox(
    "Select Civilization:",
    options=list(datasets.keys()),
    help="Choose a civilization for CAMS analysis"
)

# Get selected data
dataset_info = datasets[selected_country]
country_data = dataset_info['data'].copy()

# Year selection (if available)
if 'Year' in country_data.columns and len(country_data['Year'].unique()) > 1:
    available_years = sorted(country_data['Year'].unique())
    selected_year = st.sidebar.selectbox(
        "Select Year:",
        options=['Latest'] + list(available_years),
        help="Choose a specific year or latest data"
    )
    
    if selected_year == 'Latest':
        current_data = country_data[country_data['Year'] == country_data['Year'].max()].copy()
    else:
        current_data = country_data[country_data['Year'] == selected_year].copy()
else:
    current_data = country_data.copy()
    selected_year = 'Current'

# CAMS Parameters
st.sidebar.markdown("### ⚙️ CAMS Parameters")
tau = st.sidebar.slider("Stress Tolerance τ", 2.0, 4.0, 3.0, 0.1,
                       help="Stress tolerance threshold")
lambda_param = st.sidebar.slider("Resilience Factor λ", 0.3, 0.8, 0.5, 0.05,
                                 help="Resilience decay factor")

st.sidebar.markdown("### 📊 Dataset Information")
st.sidebar.info(f"""
**Civilization:** {selected_country}  
**Records:** {dataset_info['records']}  
**Time Period:** {dataset_info['years']}  
**Current Analysis:** {len(current_data)} nodes  
**File:** {dataset_info['filename']}
""")

# === Main Analysis ===

# Calculate current system metrics
system_health = calculate_system_health(current_data)
avg_stress = current_data['Stress'].abs().mean()
coherence_std = current_data['Coherence'].std()
civilization_type = classify_civilization_type(system_health, avg_stress, coherence_std)
esd = calculate_esd(current_data)

# Health trajectory (simplified - would need time series for real calculation)
health_trajectory = "Stable" if system_health > 3.0 else "Declining" if system_health < 2.0 else "Variable"

# === Current System Health Section ===
st.markdown("## 📊 Current System Health")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin: 0; font-size: 2.5rem;">{system_health:.2f}</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">System Health</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status_class = "status-healthy" if "🟢" in civilization_type else "status-moderate" if "🟡" in civilization_type else "status-stressed"
    st.markdown(f"""
    <div class="{status_class}">
        <h3 style="margin: 0;">Civilization Type</h3>
        <p style="margin: 0.25rem 0 0 0;">{civilization_type}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    trajectory_class = "status-healthy" if health_trajectory == "Stable" else "status-stressed"
    st.markdown(f"""
    <div class="{trajectory_class}">
        <h3 style="margin: 0;">Health Trajectory</h3>
        <p style="margin: 0.25rem 0 0 0;">{health_trajectory}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin: 0; font-size: 2rem;">{esd:.3f}</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Stress Distribution (ESD)</p>
    </div>
    """, unsafe_allow_html=True)

# === Visualizations ===
st.markdown("## 📈 System Health Timeline")

# Create timeline (if historical data available)
if 'Year' in country_data.columns and len(country_data['Year'].unique()) > 1:
    timeline_data = []
    for year in sorted(country_data['Year'].unique()):
        year_data = country_data[country_data['Year'] == year]
        if len(year_data) > 0:
            year_health = calculate_system_health(year_data)
            year_stress = year_data['Stress'].abs().mean()
            timeline_data.append({
                'Year': year,
                'SystemHealth': year_health,
                'AvgStress': year_stress
            })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        
        fig_timeline = go.Figure()
        
        # System Health line
        fig_timeline.add_trace(go.Scatter(
            x=timeline_df['Year'],
            y=timeline_df['SystemHealth'],
            name='System Health',
            line=dict(color='blue', width=3),
            hovertemplate="Year: %{x}<br>Health: %{y:.2f}<extra></extra>"
        ))
        
        # Average Stress line (secondary axis)
        fig_timeline.add_trace(go.Scatter(
            x=timeline_df['Year'],
            y=timeline_df['AvgStress'],
            name='Average Stress',
            line=dict(color='red', width=2, dash='dash'),
            yaxis='y2',
            hovertemplate="Year: %{x}<br>Stress: %{y:.2f}<extra></extra>"
        ))
        
        fig_timeline.update_layout(
            title=f"CAMS Timeline Analysis - {selected_country}",
            xaxis_title="Year",
            yaxis_title="System Health",
            yaxis2=dict(
                title="Average Stress",
                overlaying='y',
                side='right'
            ),
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("📊 Single time point data - timeline analysis not available")

# === Four Dimensions Profile ===
st.markdown("## 🎯 Four Dimensions Profile")

# Calculate averages
avg_coherence = current_data['Coherence'].mean()
avg_capacity = current_data['Capacity'].mean()
avg_stress_abs = current_data['Stress'].abs().mean()
avg_abstraction = current_data['Abstraction'].mean()

# Radar chart
categories = ['Coherence', 'Capacity', 'Stress', 'Abstraction']
values = [avg_coherence, avg_capacity, avg_stress_abs, avg_abstraction]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name=selected_country,
    line_color='rgba(0,100,255,0.8)',
    fillcolor='rgba(0,100,255,0.3)'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(10, max(values) * 1.2)]
        )
    ),
    showlegend=False,
    title=f"CAMS Four Dimensions Profile - {selected_country}",
    height=500
)

st.plotly_chart(fig_radar, use_container_width=True)

# === Node Network ===
st.markdown("## 🔗 Node Network")

# Create network graph
if len(current_data) > 1:
    # Calculate node connections based on similarity
    G = nx.Graph()
    
    # Add nodes
    for _, row in current_data.iterrows():
        fitness = calculate_node_fitness(row['Coherence'], row['Capacity'], row['Stress'], row['Abstraction'])
        G.add_node(row['Node'], 
                  fitness=fitness,
                  coherence=row['Coherence'],
                  capacity=row['Capacity'],
                  stress=row['Stress'],
                  abstraction=row['Abstraction'])
    
    # Add edges based on similarity (simplified)
    nodes_list = list(current_data.iterrows())
    for i, (idx1, row1) in enumerate(nodes_list):
        for j, (idx2, row2) in enumerate(nodes_list):
            if i < j:
                # Similarity based on CAMS dimensions
                coherence_sim = 1 / (1 + abs(row1['Coherence'] - row2['Coherence']))
                capacity_sim = 1 / (1 + abs(row1['Capacity'] - row2['Capacity']))
                stress_sim = 1 / (1 + abs(row1['Stress'] - row2['Stress']))
                
                similarity = (coherence_sim + capacity_sim + stress_sim) / 3
                
                if similarity > 0.6:  # Threshold for connection
                    G.add_edge(row1['Node'], row2['Node'], weight=similarity)
    
    # Create network visualization
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, seed=42)
        
        # Node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_info = G.nodes[node]
            text = f"{node}<br>Fitness: {node_info['fitness']:.2f}<br>"
            text += f"Coherence: {node_info['coherence']:.1f}<br>"
            text += f"Capacity: {node_info['capacity']:.1f}<br>"
            text += f"Stress: {node_info['stress']:.1f}<br>"
            text += f"Abstraction: {node_info['abstraction']:.1f}"
            node_text.append(text)
            
            node_color.append(node_info['fitness'])
        
        # Edge traces
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig_network = go.Figure()
        
        # Add edges
        fig_network.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig_network.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_text,
            text=[node for node in G.nodes()],
            textposition="middle center",
            marker=dict(
                size=30,
                color=node_color,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Node Fitness"),
                line=dict(width=2, color='black')
            )
        ))
        
        fig_network.update_layout(
            title="CAMS Node Network Analysis",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text="Node size and color represent fitness levels",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(color="#888", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(fig_network, use_container_width=True)

# === Node Analysis Heatmaps ===
st.markdown("## 🌡️ Node Analysis Heatmaps")

# Metric selection
selected_metric = st.selectbox(
    "Select Metric for Heatmap:",
    options=['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Fitness'],
    help="Choose which metric to visualize in the heatmap"
)

# Calculate fitness if selected
if selected_metric == 'Fitness':
    current_data['Fitness'] = [
        calculate_node_fitness(row['Coherence'], row['Capacity'], row['Stress'], row['Abstraction'])
        for _, row in current_data.iterrows()
    ]

# Create heatmap data
heatmap_data = current_data.pivot_table(
    values=selected_metric,
    index='Node',
    aggfunc='mean'
).reset_index()

if len(heatmap_data) > 0:
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=[heatmap_data[selected_metric].values],
        x=heatmap_data['Node'],
        y=[selected_metric],
        colorscale='RdYlBu_r' if selected_metric != 'Stress' else 'Reds',
        colorbar=dict(title=selected_metric),
        text=np.round([heatmap_data[selected_metric].values], 2),
        texttemplate="%{text}",
        hovertemplate="Node: %{x}<br>" + selected_metric + ": %{z:.2f}<extra></extra>"
    ))
    
    fig_heatmap.update_layout(
        title=f"CAMS {selected_metric} Analysis - {selected_country}",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

# === Current Stress Distribution ===
st.markdown("## ⚡ Current Stress Distribution")

# Stress analysis
stress_data = current_data.copy()
stress_data['StressAbs'] = stress_data['Stress'].abs()
stress_data['StressCategory'] = stress_data['StressAbs'].apply(
    lambda x: '🔴 High' if x > 3.0 else '🟡 Moderate' if x > 1.5 else '🟢 Low'
)

# Stress bar chart
fig_stress = go.Figure()

colors = ['red' if x > 3.0 else 'orange' if x > 1.5 else 'green' 
          for x in stress_data['StressAbs']]

fig_stress.add_trace(go.Bar(
    x=stress_data['Node'],
    y=stress_data['StressAbs'],
    marker_color=colors,
    text=stress_data['StressAbs'].round(2),
    textposition='auto',
    hovertemplate="Node: %{x}<br>Stress: %{y:.2f}<br>Category: " + 
                 stress_data['StressCategory'].astype(str) + "<extra></extra>"
))

fig_stress.update_layout(
    title=f"Stress Distribution Analysis - {selected_country}",
    xaxis_title="Institutional Nodes",
    yaxis_title="Stress Level",
    height=500
)

st.plotly_chart(fig_stress, use_container_width=True)

# Stress summary
col1, col2, col3 = st.columns(3)

with col1:
    high_stress = len(stress_data[stress_data['StressAbs'] > 3.0])
    st.metric("High Stress Nodes", f"{high_stress}/{len(stress_data)}")

with col2:
    max_stress = stress_data['StressAbs'].max()
    max_stress_node = stress_data.loc[stress_data['StressAbs'].idxmax(), 'Node']
    st.metric("Highest Stress", f"{max_stress:.2f}", delta=max_stress_node)

with col3:
    avg_stress_metric = stress_data['StressAbs'].mean()
    st.metric("Average Stress", f"{avg_stress_metric:.2f}")

# === Detailed Data Table ===
st.markdown("## 📋 Detailed Node Analysis")

# Enhanced data table
display_data = current_data.copy()
if 'Fitness' not in display_data.columns:
    display_data['Fitness'] = [
        calculate_node_fitness(row['Coherence'], row['Capacity'], row['Stress'], row['Abstraction'])
        for _, row in display_data.iterrows()
    ]

# Round numerical columns
for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Fitness']:
    if col in display_data.columns:
        display_data[col] = display_data[col].round(2)

st.dataframe(
    display_data[['Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction', 'Fitness']],
    use_container_width=True
)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 1rem; margin: 2rem 0;">
    <h3 style="color: #666;">CAMS Framework Dashboard</h3>
    <p style="color: #888; margin: 0;">
        Developed based on <strong>Kari McKern's</strong> Complex Adaptive Model State framework<br>
        📧 Contact: <a href="mailto:kari.freyr.4@gmail.com">kari.freyr.4@gmail.com</a> | 
        📚 Learn more: <a href="#">Pearls and Irritations</a>
    </p>
</div>
""", unsafe_allow_html=True)