"""
Inter-Institutional Bond Network Analysis
Advanced analysis of institutional relationships and bond strengths in CAMS framework
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import glob
import json

# Configure page
st.set_page_config(page_title="Inter-Institutional Bond Network", layout="wide", initial_sidebar_state="expanded")

st.title("🔗 Inter-Institutional Bond Network Analysis")
st.markdown("**Deep analysis of institutional relationships, bond strengths, and network dynamics**")

# === Data Loading ===
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
st.sidebar.markdown("## 🎛️ Bond Network Controls")

# Dataset selection
selected_country = st.sidebar.selectbox(
    "Select Civilization:",
    options=list(datasets.keys()),
    help="Choose a civilization for bond network analysis"
)

# Analysis parameters
st.sidebar.markdown("### ⚙️ Bond Parameters")

bond_metric = st.sidebar.selectbox(
    "Bond Strength Metric:",
    ["Coherence Similarity", "Capacity Synergy", "Stress Correlation", "Multi-Dimensional", "Custom Formula"],
    help="Method for calculating inter-institutional bond strength"
)

threshold = st.sidebar.slider("Bond Threshold", 0.0, 1.0, 0.3, 0.05, 
                             help="Minimum bond strength to display connection")

network_layout = st.sidebar.selectbox(
    "Network Layout:",
    ["spring", "circular", "kamada_kawai", "hierarchical", "shell"],
    help="Layout algorithm for network visualization"
)

# CAMS parameters
tau = st.sidebar.slider("Stress Tolerance τ", 2.5, 3.5, 3.0, 0.1)
lambda_param = st.sidebar.slider("Resilience Factor λ", 0.3, 0.7, 0.5, 0.1)

# === Data Processing ===
@st.cache_data
def calculate_enhanced_metrics(df, tau_val, lambda_val):
    """Calculate enhanced CAMS metrics"""
    df = df.copy()
    
    # Node Fitness
    stress_impact = 1 + np.exp((np.abs(df['Stress']) - tau_val) / lambda_val)
    df['Fitness'] = (df['Coherence'] * df['Capacity']) / stress_impact * (1 + df['Abstraction']/10)
    
    # Processing Efficiency
    df['ProcessingEfficiency'] = df['Fitness'] / (np.abs(df['Stress']) + 1e-6)
    
    # Node Stability (inverse stress sensitivity)
    df['Stability'] = 1 / (1 + np.abs(df['Stress']))
    
    # Adaptive Capacity (coherence * abstraction)
    df['AdaptiveCapacity'] = df['Coherence'] * df['Abstraction']
    
    return df

# Process selected dataset
country_data = datasets[selected_country].copy()

# Get latest data
if 'Year' in country_data.columns:
    latest_year = country_data['Year'].max()
    current_data = country_data[country_data['Year'] == latest_year].copy()
else:
    current_data = country_data.copy()

current_data = calculate_enhanced_metrics(current_data, tau, lambda_param)

st.success(f"✅ Analyzing {len(current_data)} institutional nodes from {selected_country}")

# === Bond Strength Calculations ===
def calculate_bond_matrix(df, metric_type):
    """Calculate inter-institutional bond strength matrix"""
    n_nodes = len(df)
    bond_matrix = np.zeros((n_nodes, n_nodes))
    node_names = df['Node'].tolist()
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                node_i = df.iloc[i]
                node_j = df.iloc[j]
                
                if metric_type == "Coherence Similarity":
                    # Bond based on coherence similarity
                    diff = abs(node_i['Coherence'] - node_j['Coherence'])
                    bond = 1 / (1 + diff)
                
                elif metric_type == "Capacity Synergy":
                    # Bond based on capacity synergy potential
                    synergy = np.sqrt(node_i['Capacity'] * node_j['Capacity'])
                    max_capacity = max(df['Capacity'])
                    bond = synergy / max_capacity if max_capacity > 0 else 0
                
                elif metric_type == "Stress Correlation":
                    # Bond inversely related to stress differential
                    stress_diff = abs(node_i['Stress'] - node_j['Stress'])
                    bond = np.exp(-stress_diff / 2)  # Exponential decay
                
                elif metric_type == "Multi-Dimensional":
                    # Comprehensive multi-dimensional bond
                    coh_sim = 1 / (1 + abs(node_i['Coherence'] - node_j['Coherence']))
                    cap_syn = np.sqrt(node_i['Capacity'] * node_j['Capacity']) / 10
                    stress_comp = np.exp(-abs(node_i['Stress'] - node_j['Stress']) / 2)
                    abs_align = 1 / (1 + abs(node_i['Abstraction'] - node_j['Abstraction']))
                    
                    # Weighted combination
                    bond = 0.3 * coh_sim + 0.25 * cap_syn + 0.25 * stress_comp + 0.2 * abs_align
                
                elif metric_type == "Custom Formula":
                    # Custom CAMS-based formula
                    fitness_product = np.sqrt(node_i['Fitness'] * node_j['Fitness'])
                    stability_factor = (node_i['Stability'] + node_j['Stability']) / 2
                    adaptive_resonance = np.sqrt(node_i['AdaptiveCapacity'] * node_j['AdaptiveCapacity'])
                    
                    bond = (fitness_product * stability_factor * adaptive_resonance) / 100
                
                bond_matrix[i, j] = max(0, min(1, bond))  # Clamp to [0,1]
    
    return bond_matrix, node_names

# Calculate bond matrix
bond_matrix, node_names = calculate_bond_matrix(current_data, bond_metric)

# === Main Analysis Display ===
st.markdown("## 🕸️ Inter-Institutional Bond Network")

# Network metrics summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_bond_strength = np.mean(bond_matrix[bond_matrix > 0])
    st.metric("Average Bond Strength", f"{avg_bond_strength:.3f}")

with col2:
    strong_bonds = np.sum(bond_matrix > 0.7)
    st.metric("Strong Bonds (>0.7)", strong_bonds)

with col3:
    total_connections = np.sum(bond_matrix > threshold)
    st.metric(f"Active Connections (>{threshold})", total_connections)

with col4:
    network_density = total_connections / (len(node_names) * (len(node_names) - 1))
    st.metric("Network Density", f"{network_density:.3f}")

# === Bond Matrix Heatmap ===
st.markdown("### 🔥 Bond Strength Matrix")

fig_heatmap = go.Figure(data=go.Heatmap(
    z=bond_matrix,
    x=node_names,
    y=node_names,
    colorscale='Viridis',
    colorbar=dict(title="Bond Strength"),
    text=np.round(bond_matrix, 3),
    texttemplate="%{text}",
    textfont={"size": 10},
    hovertemplate="From: %{y}<br>To: %{x}<br>Bond: %{z:.3f}<extra></extra>"
))

fig_heatmap.update_layout(
    title=f"Inter-Institutional Bond Matrix - {selected_country} ({bond_metric})",
    height=600,
    xaxis_title="Target Institution",
    yaxis_title="Source Institution"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# === Network Graph Visualization ===
st.markdown("### 🌐 Network Graph Visualization")

# Create NetworkX graph
G = nx.Graph()

# Add nodes with attributes
for i, node in enumerate(node_names):
    node_data = current_data.iloc[i]
    G.add_node(node,
               fitness=node_data['Fitness'],
               coherence=node_data['Coherence'],
               capacity=node_data['Capacity'],
               stress=abs(node_data['Stress']),
               abstraction=node_data['Abstraction'])

# Add edges above threshold
for i in range(len(node_names)):
    for j in range(i+1, len(node_names)):
        bond_strength = bond_matrix[i, j]
        if bond_strength > threshold:
            G.add_edge(node_names[i], node_names[j], weight=bond_strength)

st.info(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges (threshold: {threshold})")

# Choose layout
if network_layout == "spring":
    pos = nx.spring_layout(G, seed=42, k=2, iterations=100)
elif network_layout == "circular":
    pos = nx.circular_layout(G)
elif network_layout == "kamada_kawai":
    pos = nx.kamada_kawai_layout(G) if len(G.nodes()) > 2 else nx.spring_layout(G)
elif network_layout == "hierarchical":
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if len(G.nodes()) > 0 else nx.spring_layout(G)
elif network_layout == "shell":
    pos = nx.shell_layout(G)

# Create Plotly network visualization
edge_x, edge_y = [], []
edge_weights = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_weights.append(G[edge[0]][edge[1]]['weight'])

# Edge trace
edge_trace = go.Scatter(x=edge_x, y=edge_y,
                       line=dict(width=2, color='rgba(125,125,125,0.5)'),
                       hoverinfo='none',
                       mode='lines')

# Node trace
node_x, node_y = [], []
node_text, node_color, node_size = [], [], []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    
    node_info = G.nodes[node]
    text = f"{node}<br>"
    text += f"Fitness: {node_info['fitness']:.2f}<br>"
    text += f"Coherence: {node_info['coherence']:.2f}<br>"
    text += f"Capacity: {node_info['capacity']:.2f}<br>"
    text += f"Stress: {node_info['stress']:.2f}<br>"
    text += f"Abstraction: {node_info['abstraction']:.2f}"
    node_text.append(text)
    
    # Color by fitness, size by degree centrality
    node_color.append(node_info['fitness'])
    degree = G.degree(node)
    node_size.append(20 + degree * 5)

node_trace = go.Scatter(x=node_x, y=node_y,
                       mode='markers+text',
                       hoverinfo='text',
                       hovertext=node_text,
                       text=list(G.nodes()),
                       textposition="middle center",
                       marker=dict(size=node_size,
                                 color=node_color,
                                 colorscale='RdYlBu_r',
                                 showscale=True,
                                 colorbar=dict(title="Fitness"),
                                 line=dict(width=2, color='black')))

fig_network = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Inter-Institutional Bond Network - {selected_country}<br>Bond Metric: {bond_metric}',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=70),
                           annotations=[dict(
                               text=f"Node size = Centrality, Color = Fitness, Edge thickness = Bond strength",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=700))

st.plotly_chart(fig_network, use_container_width=True)

# === Network Analysis ===
st.markdown("### 📊 Network Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🎯 Centrality Analysis**")
    
    if len(G.nodes()) > 0:
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G) if nx.is_connected(G) else {}
        
        centrality_data = []
        for node in G.nodes():
            centrality_data.append({
                'Node': node,
                'Degree': degree_centrality.get(node, 0),
                'Betweenness': betweenness_centrality.get(node, 0),
                'Closeness': closeness_centrality.get(node, 0),
                'Connections': G.degree(node)
            })
        
        centrality_df = pd.DataFrame(centrality_data)
        centrality_df = centrality_df.sort_values('Degree', ascending=False)
        st.dataframe(centrality_df, use_container_width=True)

with col2:
    st.markdown("**🔗 Bond Strength Distribution**")
    
    # Bond strength histogram
    bonds = bond_matrix[bond_matrix > 0].flatten()
    
    if len(bonds) > 0:
        fig_hist = go.Figure(data=go.Histogram(
            x=bonds,
            nbinsx=20,
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Threshold: {threshold}")
        
        fig_hist.update_layout(
            title="Bond Strength Distribution",
            xaxis_title="Bond Strength",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

# === Clustering Analysis ===
st.markdown("### 🎭 Institutional Clusters")

if len(current_data) >= 3:  # Need at least 3 nodes for clustering
    
    # Prepare features for clustering
    features = current_data[['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Fitness']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means clustering
    n_clusters = min(4, len(current_data) // 2 + 1)  # Reasonable number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Add cluster info to dataframe
    cluster_data = current_data.copy()
    cluster_data['Cluster'] = clusters
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster visualization
        fig_clusters = go.Figure()
        
        colors = px.colors.qualitative.Set1[:n_clusters]
        
        for i in range(n_clusters):
            cluster_nodes = cluster_data[cluster_data['Cluster'] == i]
            
            fig_clusters.add_trace(go.Scatter(
                x=cluster_nodes['Coherence'],
                y=cluster_nodes['Capacity'],
                mode='markers+text',
                text=cluster_nodes['Node'],
                textposition='top center',
                marker=dict(size=12, color=colors[i]),
                name=f'Cluster {i+1}',
                hovertemplate="<b>%{text}</b><br>Coherence: %{x:.2f}<br>Capacity: %{y:.2f}<extra></extra>"
            ))
        
        fig_clusters.update_layout(
            title="Institutional Clusters (Coherence vs Capacity)",
            xaxis_title="Coherence",
            yaxis_title="Capacity",
            height=500
        )
        
        st.plotly_chart(fig_clusters, use_container_width=True)
    
    with col2:
        # Cluster summary
        st.markdown("**📋 Cluster Summary**")
        
        cluster_summary = []
        for i in range(n_clusters):
            cluster_nodes = cluster_data[cluster_data['Cluster'] == i]
            
            cluster_summary.append({
                'Cluster': f'Cluster {i+1}',
                'Nodes': len(cluster_nodes),
                'Members': ', '.join(cluster_nodes['Node'].tolist()),
                'Avg_Fitness': cluster_nodes['Fitness'].mean(),
                'Avg_Stress': abs(cluster_nodes['Stress']).mean(),
                'Coherence_Range': f"{cluster_nodes['Coherence'].min():.1f}-{cluster_nodes['Coherence'].max():.1f}"
            })
        
        cluster_df = pd.DataFrame(cluster_summary)
        st.dataframe(cluster_df, use_container_width=True)

# === Advanced Bond Analysis ===
st.markdown("### 🔬 Advanced Bond Analysis")

# Bond strength vs node properties
analysis_data = []
for i in range(len(node_names)):
    for j in range(i+1, len(node_names)):
        if bond_matrix[i, j] > 0:
            node_i_data = current_data.iloc[i]
            node_j_data = current_data.iloc[j]
            
            analysis_data.append({
                'Node_A': node_names[i],
                'Node_B': node_names[j],
                'Bond_Strength': bond_matrix[i, j],
                'Fitness_Product': node_i_data['Fitness'] * node_j_data['Fitness'],
                'Coherence_Diff': abs(node_i_data['Coherence'] - node_j_data['Coherence']),
                'Capacity_Sum': node_i_data['Capacity'] + node_j_data['Capacity'],
                'Stress_Diff': abs(node_i_data['Stress'] - node_j_data['Stress']),
                'Bond_Type': 'Strong' if bond_matrix[i, j] > 0.7 else 'Medium' if bond_matrix[i, j] > 0.4 else 'Weak'
            })

if analysis_data:
    bond_analysis_df = pd.DataFrame(analysis_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bond strength vs fitness correlation
        fig_corr = go.Figure()
        
        colors = {'Strong': 'red', 'Medium': 'orange', 'Weak': 'lightblue'}
        
        for bond_type in ['Strong', 'Medium', 'Weak']:
            subset = bond_analysis_df[bond_analysis_df['Bond_Type'] == bond_type]
            if len(subset) > 0:
                fig_corr.add_trace(go.Scatter(
                    x=subset['Fitness_Product'],
                    y=subset['Bond_Strength'],
                    mode='markers',
                    name=bond_type,
                    marker=dict(color=colors[bond_type], size=8),
                    hovertemplate="<b>%{text}</b><br>Fitness Product: %{x:.2f}<br>Bond: %{y:.3f}<extra></extra>",
                    text=[f"{row['Node_A']} - {row['Node_B']}" for _, row in subset.iterrows()]
                ))
        
        fig_corr.update_layout(
            title="Bond Strength vs Fitness Product",
            xaxis_title="Fitness Product",
            yaxis_title="Bond Strength",
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Top bonds table
        st.markdown("**🏆 Strongest Institutional Bonds**")
        
        top_bonds = bond_analysis_df.nlargest(10, 'Bond_Strength')[
            ['Node_A', 'Node_B', 'Bond_Strength', 'Bond_Type']
        ]
        
        st.dataframe(top_bonds, use_container_width=True)

# === Export and Summary ===
st.markdown("### 📁 Export & Summary")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Bond Matrix"):
        bond_df = pd.DataFrame(bond_matrix, columns=node_names, index=node_names)
        csv = bond_df.to_csv()
        st.download_button(
            label="Download Bond Matrix CSV",
            data=csv,
            file_name=f"{selected_country}_bond_matrix.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🕸️ Export Network Data"):
        if analysis_data:
            network_csv = pd.DataFrame(analysis_data).to_csv(index=False)
            st.download_button(
                label="Download Network CSV",
                data=network_csv,
                file_name=f"{selected_country}_network_analysis.csv",
                mime="text/csv"
            )

with col3:
    if st.button("📈 Export Metrics"):
        if len(G.nodes()) > 0:
            metrics_csv = centrality_df.to_csv(index=False)
            st.download_button(
                label="Download Metrics CSV",
                data=metrics_csv,
                file_name=f"{selected_country}_network_metrics.csv", 
                mime="text/csv"
            )

# === Summary Statistics ===
st.markdown("### 📋 Network Summary")

summary_stats = {
    "Total Institutions": len(node_names),
    "Active Bonds": len([b for b in bonds if b > threshold]) if 'bonds' in locals() else 0,
    "Average Bond Strength": f"{avg_bond_strength:.3f}",
    "Network Density": f"{network_density:.3f}",
    "Most Connected Node": max(G.nodes(), key=lambda x: G.degree(x)) if len(G.nodes()) > 0 else "N/A",
    "Bond Metric Used": bond_metric,
    "Analysis Threshold": f"{threshold}"
}

col1, col2 = st.columns(2)

with col1:
    for key, value in list(summary_stats.items())[:4]:
        st.write(f"**{key}:** {value}")

with col2:
    for key, value in list(summary_stats.items())[4:]:
        st.write(f"**{key}:** {value}")

st.markdown("---")
st.success("🎉 Inter-Institutional Bond Network Analysis Complete!")

st.markdown("""
**Analysis Capabilities:**
- 🔥 Bond strength matrix with multiple calculation methods
- 🌐 Interactive network visualization with customizable layouts  
- 📊 Centrality analysis and network metrics
- 🎭 Institutional clustering and pattern detection
- 🔬 Advanced bond correlation analysis
- 📁 Complete data export functionality
""")