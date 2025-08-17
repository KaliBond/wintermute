"""
CAMS–CAN v3.4 Stress Dynamics Explorer
Complex Adaptive Management Systems - Catch-All Network v3.4
Interactive visualization and exploration of stress dynamics and attractor behaviors in civilizational systems

Implementing the complete CAMS-CAN v3.4 specification:
- 8 Institutional Nodes Analysis (Executive, Army, Priesthood, Property, Trades, Proletariat, StateMemory, Merchants)
- 32-Dimensional Phase Space Visualization with PCA/t-SNE/UMAP reduction
- 5 Key Metrics: Individual Node Fitness, System Health, Coherence Asymmetry, Critical Transition Risk, Network Coherence
- 6 Interactive Visualizations: Node Stress Trajectories, Phase Space Scatter Plots, Attractor Basin Visualization, 
  Bond Strength Matrix, System Health & Risk Trends, Coordinated Interactive Visualization
- Time Window Filtering, Institution Filtering, Animation Controls
- Attractor Behavior Detection with transition highlighting
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    st.warning("UMAP not available. Install with: pip install umap-learn")
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -- Utility Functions: CAMS–CAN metrics --

def node_fitness(C, K, S, A, tau=3.0, lam=0.5):
    """
    Hi(t) = Ci*Ki / (1+exp((|Si|-τ)/λ)) * (1+Ai/10)
    Enhanced node fitness with stress tolerance
    """
    # Ensure arrays for vectorized operations
    C, K, S, A = map(np.asarray, [C, K, S, A])
    
    # Stress penalty with tau threshold and lambda decay
    penalty = 1 + np.exp((np.abs(S) - tau) / lam)
    
    # Base fitness with abstraction bonus
    fitness = (C * K / penalty) * (1 + A / 10)
    
    return fitness

def system_health_H(H_mat, phi_network, stress_var, K_avg):
    """
    Psi(t) = (prod_i Hi)^(1/8) * (1+phi_network) * R(t)
    R = exp(-Var(S)/<K>^2) - resilience factor
    """
    # Resilience factor based on stress variance and capacity
    R = np.exp(-stress_var / (K_avg**2 + 1e-9))
    
    # Geometric mean of node fitness values
    H_mat_safe = np.maximum(H_mat, 1e-9)  # Prevent log(0)
    geometric_mean = np.exp(np.mean(np.log(H_mat_safe)))
    
    # Combined system health
    return geometric_mean * (1 + phi_network) * R

def coherence_asymmetry(C, K):
    """
    CA = std(Ci*Ki) / mean(Ci*Ki)
    Measures institutional coordination asymmetry
    """
    x = C * K
    return np.std(x) / (np.mean(x) + 1e-9)

def risk_index(CA, psi):
    """
    Risk index: Λ = CA / (1 + ψ)
    Higher asymmetry and lower health increase risk
    """
    return CA / (1 + psi)

def bond_strength_matrix(Cs, sigma=5.0):
    """
    B(i,j) = exp(-|Ci-Cj|/σ) 
    Inter-institutional bond strengths based on coherence similarity
    """
    Cs = np.asarray(Cs)
    diff = np.abs(Cs.reshape(-1, 1) - Cs.reshape(1, -1))
    return np.exp(-diff / sigma)

def phi_network(CMat):
    """
    Φ_network: mean of all off-diagonal bond strength entries
    Global network coherence measure
    """
    n = CMat.shape[0]
    if n <= 1:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return np.mean(CMat[mask])

def detect_phase_attractor(psi, ca, spe=None):
    """
    Classify current phase space attractor based on system metrics
    """
    if psi > 3.5 and ca < 0.3:
        return "Adaptive", "🟢"
    elif psi >= 2.5 and psi <= 3.5:
        return "Authoritarian", "🟡" 
    elif psi >= 1.5 and psi <= 2.5 and ca > 0.4:
        return "Fragmented", "🟠"
    elif psi < 1.5:
        return "Collapse", "🔴"
    else:
        return "Transitional", "🔵"

def stress_processing_efficiency(H_arr, S_arr, A_arr):
    """
    SPE(t) = Σ(Hi) / Σ(|Si|*Ai)
    Collective stress processing capability
    """
    numerator = np.sum(H_arr)
    denominator = np.sum(np.abs(S_arr) * A_arr)
    return numerator / (denominator + 1e-9)

def calculate_critical_transition_risk(df):
    """
    Calculate Critical Transition Risk CTR(t)
    Based on stress variance, system health, and coherence asymmetry
    """
    if df is None or len(df) == 0:
        return 0.0
    
    H_arr = df["H"].values if "H" in df.columns else node_fitness(df["C"], df["K"], df["S"], df["A"])
    stress_var = np.var(df["S"].abs())
    ca = coherence_asymmetry(df["C"], df["K"])
    system_health = np.exp(np.mean(np.log(np.maximum(H_arr, 1e-9))))
    
    # Risk increases with high stress variance, high asymmetry, low health
    ctr = (stress_var * (1 + ca)) / (system_health + 1e-6)
    return min(ctr, 10.0)  # Cap at 10

def calculate_network_coherence(df):
    """
    Calculate Network Coherence NC(t)
    Based on inter-institutional coherence correlation
    """
    if df is None or len(df) == 0:
        return 0.0
    
    coherence_values = df["C"].values
    capacity_values = df["K"].values
    
    if len(coherence_values) > 1:
        correlation = np.corrcoef(coherence_values, capacity_values)[0, 1]
        return max(0, correlation)  # Return only positive correlation
    else:
        return 0.5  # Neutral for single institution

def create_32d_phase_space_features(df, standard_nodes):
    """
    Create 32-dimensional feature vector (8 nodes × 4 dimensions)
    """
    features = []
    
    for node in standard_nodes:
        node_data = df[df["Node"] == node]
        if len(node_data) == 1:
            row = node_data.iloc[0]
            features.extend([row["C"], row["K"], row["S"], row["A"]])
        else:
            # Missing node - use defaults
            features.extend([5.0, 5.0, 0.0, 5.0])  # Default values
    
    return np.array(features)

def apply_dimensionality_reduction(features_array, method="PCA", n_components=2):
    """
    Apply dimensionality reduction to 32D feature space
    """
    if len(features_array) <= 1:
        return np.zeros((len(features_array), n_components))
    
    if method == "PCA":
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(features_array)
        variance_info = f"Explained Variance: {reducer.explained_variance_ratio_.sum():.3f}"
    elif method == "t-SNE":
        perplexity = min(30, len(features_array) - 1)
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        reduced = reducer.fit_transform(features_array)
        variance_info = "t-SNE embedding"
    elif method == "UMAP" and UMAP_AVAILABLE:
        n_neighbors = min(15, len(features_array) - 1)
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
        reduced = reducer.fit_transform(features_array)
        variance_info = "UMAP embedding"
    else:
        # Fallback to PCA
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(features_array)
        variance_info = f"PCA Fallback - Explained Variance: {reducer.explained_variance_ratio_.sum():.3f}"
    
    return reduced, variance_info

def identify_attractor_transitions(sys_df):
    """
    Identify transitions between attractors
    """
    if len(sys_df) <= 1:
        return []
    
    transitions = []
    for i in range(1, len(sys_df)):
        if sys_df.iloc[i]['attractor'] != sys_df.iloc[i-1]['attractor']:
            transitions.append({
                'year': sys_df.iloc[i]['Year'],
                'from': sys_df.iloc[i-1]['attractor'],
                'to': sys_df.iloc[i]['attractor'],
                'health_change': sys_df.iloc[i]['system_health'] - sys_df.iloc[i-1]['system_health']
            })
    
    return transitions

# -- Data Loading Functions --

@st.cache_data
def load_available_datasets():
    """Load all available CAMS datasets"""
    # Look for CSV files in multiple locations
    csv_files = (glob.glob("*.csv") + 
                 glob.glob("data/cleaned/*.csv") + 
                 glob.glob("cleaned_datasets/*.csv"))
    datasets = {}
    
    # Enhanced country name mapping for cleaned datasets
    country_mapping = {
        'australia cleaned': 'Australia',
        'usa cleaned': 'USA', 
        'france cleaned': 'France',
        'italy cleaned': 'Italy',
        'italy19002025 cleaned': 'Italy (1900-2025)',
        'germany cleaned': 'Germany',
        'denmark cleaned': 'Denmark',
        'iran cleaned': 'Iran',
        'iraq cleaned': 'Iraq',
        'lebanon cleaned': 'Lebanon',
        'japan cleaned': 'Japan',
        'thailand 1850 2025 thailand 1850 2025 cleaned': 'Thailand',
        'netherlands cleaned': 'Netherlands',
        'canada cleaned': 'Canada',
        'saudi arabia cleaned': 'Saudi Arabia',
        'hong kong cleaned': 'Hong Kong',
        'hongkong manual cleaned': 'Hong Kong (Manual)',
        'england cleaned': 'England',
        'france 1785 1800 cleaned': 'France (1785-1800)',
        'france master 3 france 1785 1790 1795 1800 cleaned': 'France (Master)',
        'india cleaned': 'India',
        'indonesia cleaned': 'Indonesia',
        'israel cleaned': 'Israel',
        'new rome ad 5y rome 0 bce 5ad 10ad 15ad 20 ad cleaned': 'Ancient Rome',
        'pakistan cleaned': 'Pakistan',
        'russia cleaned': 'Russia',
        'singapore cleaned': 'Singapore',
        'syria cleaned': 'Syria',
        'usa highres cleaned': 'USA (HighRes)',
        'usa master cleaned': 'USA (Master)',
        'usa reconstructed cleaned': 'USA (Reconstructed)',
        'usa maximum 1790-2025 us high res 2025 (1) cleaned': 'USA (Maximum)',
        # Legacy mappings
        'australia cams cleaned': 'Australia',
        'usa cams cleaned': 'USA', 
        'france cams cleaned': 'France',
        'germany1750 2025': 'Germany',
        'denmark cams cleaned': 'Denmark',
        'iran cams cleaned': 'Iran',
        'iraq cams cleaned': 'Iraq',
        'lebanon cams cleaned': 'Lebanon',
        'japan 1850 2025': 'Japan',
        'thailand 1850_2025': 'Thailand',
        'netherlands mastersheet': 'Netherlands',
        'canada_cams_2025': 'Canada',
        'saudi arabia master file': 'Saudi Arabia',
        'eqmasterrome': 'Roman Empire',
        'new rome ad 5y': 'Early Rome'
    }
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                # Extract filename and remove path
                filename = file.split('/')[-1].split('\\')[-1]
                base_name = filename.replace('.csv', '').replace('.CSV', '').lower().strip()
                base_name = base_name.replace('_', ' ').replace(' (2)', '').replace(' - ', ' ')
                
                country_name = country_mapping.get(base_name, base_name.title())
                
                # Check if it's a node-based dataset with required columns
                required_cols = ['Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']
                has_required = all(col in df.columns for col in required_cols)
                has_year = 'Year' in df.columns
                
                if has_required and has_year:
                    # Standardize column names
                    df_clean = df.copy()
                    df_clean = df_clean.rename(columns={
                        'Coherence': 'C',
                        'Capacity': 'K', 
                        'Stress': 'S',
                        'Abstraction': 'A'
                    })
                    
                    # Ensure numeric values
                    for col in ['C', 'K', 'S', 'A']:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                    # Remove rows with missing data
                    df_clean = df_clean.dropna(subset=['C', 'K', 'S', 'A'])
                    
                    if len(df_clean) > 0:
                        datasets[country_name] = {
                            'filename': file,
                            'data': df_clean,
                            'records': len(df_clean),
                            'years': f"{int(df_clean['Year'].min())}-{int(df_clean['Year'].max())}",
                            'nodes': sorted(df_clean['Node'].unique())
                        }
                        
        except Exception as e:
            continue
    
    return datasets

def create_demo_dataset():
    """Create demonstration dataset if no real data available"""
    years = list(range(2000, 2025))
    nodes = ['Executive', 'Army', 'StateMemory', 'Priesthood', 'Stewards', 'Craft', 'Flow', 'Hands']
    
    data = []
    np.random.seed(42)  # Reproducible demo data
    
    for year in years:
        # Simulate gradual stress increase over time
        stress_trend = (year - 2000) / 25.0  # 0 to 1 over 25 years
        
        for i, node in enumerate(nodes):
            # Base values with some node-specific characteristics
            base_c = 5 + np.random.normal(0, 0.5)
            base_k = 5 + np.random.normal(0, 0.5)
            base_s = 3 + stress_trend * 2 + np.random.normal(0, 0.3)  # Increasing stress
            base_a = 5 + np.random.normal(0, 0.5)
            
            # Node-specific modifiers
            if node == 'Executive':
                base_c += 1  # Higher coherence
                base_s += 0.5  # More stress
            elif node == 'Army':
                base_k += 1  # Higher capacity
            elif node == 'Priesthood':
                base_a += 1  # Higher abstraction
            
            data.append({
                'Year': year,
                'Node': node,
                'C': max(0.1, base_c),
                'K': max(0.1, base_k), 
                'S': base_s,
                'A': max(0.1, base_a)
            })
    
    return pd.DataFrame(data)

# -- Main Application --

st.set_page_config(layout="wide", page_title="CAMS-CAN v3.4", page_icon="🧠")

# Header
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">🧠 CAMS–CAN v3.4 Stress Dynamics Explorer</h1>
    <p style="color: #666; font-size: 1.1rem;">Interactive Analysis of Societal Stress as Meta-Cognition</p>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([3, 1])

# Data loading section
with col2:
    st.markdown("### 📊 Data Selection")
    
    # Load available datasets
    datasets = load_available_datasets()
    
    if datasets:
        dataset_names = list(datasets.keys())
        selected_dataset = st.selectbox(
            "Choose civilization:",
            options=dataset_names,
            help="Select a civilization dataset for analysis"
        )
        
        df = datasets[selected_dataset]['data'].copy()
        
        # Display dataset info
        info = datasets[selected_dataset]
        st.markdown(f"""
        **Dataset:** {selected_dataset}  
        **Records:** {info['records']}  
        **Years:** {info['years']}  
        **Nodes:** {len(info['nodes'])}
        """)
        
    else:
        st.warning("No compatible datasets found. Using demo data.")
        selected_dataset = "Demo Civilization"
        df = create_demo_dataset()
    
    # Parameter adjustment
    st.markdown("### ⚙️ CAMS-CAN v3.4 Parameters")
    tau = st.slider("Stress Tolerance (τ)", 1.0, 5.0, 3.0, 0.1)
    lam = st.slider("Resilience Decay (λ)", 0.1, 1.0, 0.5, 0.1)
    
    # Analysis options
    st.markdown("### 🎛️ Analysis Options")
    show_risk_threshold = st.checkbox("Show Risk Threshold", True)
    show_attractors = st.checkbox("Show Phase Attractors", True)
    show_32d_analysis = st.checkbox("32D Phase Space Analysis", True)
    
    # Dimensionality reduction method
    dim_reduction_method = st.selectbox(
        "32D Reduction Method:",
        options=["PCA", "t-SNE"] + (["UMAP"] if UMAP_AVAILABLE else []),
        help="Method for reducing 32-dimensional phase space"
    )
    
    # Animation controls
    enable_animation = st.checkbox("Enable Time Animation", False)
    if enable_animation:
        animation_speed = st.slider("Animation Speed (ms)", 500, 3000, 1000, 100)

# Main analysis section
with col1:
    if df is not None:
        # Time range selection
        years = sorted(df["Year"].unique())
        year_min, year_max = int(min(years)), int(max(years))
        
        st.markdown("### 📅 Time Range & Node Selection")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if len(years) > 1:
                start_yr, end_yr = st.select_slider(
                    "Year Range:", 
                    options=years, 
                    value=(year_min, year_max)
                )
            else:
                start_yr = end_yr = years[0]
                st.write(f"Single year dataset: {start_yr}")
        
        with col_b:
            nodes = sorted(df["Node"].unique())
            sel_nodes = st.multiselect(
                "Nodes to Display:", 
                options=nodes, 
                default=nodes
            )
        
        # Filter data
        mask = (df["Year"] >= start_yr) & (df["Year"] <= end_yr) & (df["Node"].isin(sel_nodes))
        dfv = df[mask].copy()
        
        if len(dfv) == 0:
            st.error("No data matches the selected filters.")
        else:
            # Compute node fitness
            dfv["H"] = node_fitness(dfv["C"], dfv["K"], dfv["S"], dfv["A"], tau=tau, lam=lam)
            
            # Standard 8 institutional nodes for CAMS-CAN v3.4
            standard_nodes = ['Executive', 'Army', 'Priesthood', 'Property', 'Trades', 'Proletariat', 'StateMemory', 'Merchants']
            
            # Compute system metrics by year
            plot_data = []
            phase_space_features = []
            
            for y in sorted(dfv["Year"].unique()):
                ydf = dfv[dfv["Year"] == y]
                
                H_arr = ydf["H"].values
                C_arr = ydf["C"].values
                K_arr = ydf["K"].values
                S_arr = ydf["S"].values
                A_arr = ydf["A"].values
                
                # Bond strength matrix and network coherence
                phi_mat = bond_strength_matrix(C_arr)
                phi_net = phi_network(phi_mat)
                
                # System health
                psi = system_health_H(H_arr, phi_net, np.var(S_arr), np.mean(K_arr))
                
                # Coherence asymmetry and traditional risk
                ca = coherence_asymmetry(C_arr, K_arr)
                traditional_risk = risk_index(ca, psi)
                
                # New CAMS-CAN v3.4 metrics
                critical_transition_risk = calculate_critical_transition_risk(ydf)
                network_coherence = calculate_network_coherence(ydf)
                
                # Stress processing efficiency
                spe = stress_processing_efficiency(H_arr, S_arr, A_arr)
                
                # Phase attractor
                attractor, attractor_icon = detect_phase_attractor(psi, ca, spe)
                
                # Create 32D phase space feature vector
                features_32d = create_32d_phase_space_features(ydf, standard_nodes)
                phase_space_features.append(features_32d)
                
                plot_data.append({
                    "Year": y, 
                    "system_health": psi, 
                    "asymmetry": ca, 
                    "risk": traditional_risk,
                    "critical_transition_risk": critical_transition_risk,
                    "network_coherence": network_coherence,
                    "phi_net": phi_net,
                    "spe": spe,
                    "attractor": attractor,
                    "attractor_icon": attractor_icon,
                    "mean_stress": np.mean(np.abs(S_arr)),
                    "mean_fitness": np.mean(H_arr),
                    "individual_fitness": np.mean(H_arr)
                })
            
            sys_df = pd.DataFrame(plot_data)
            
            # Display current system status - CAMS-CAN v3.4 Key Metrics Dashboard
            if len(sys_df) > 0:
                latest = sys_df.iloc[-1]
                st.markdown("### 📊 CAMS-CAN v3.4 Key Metrics Dashboard")
                
                col_1, col_2, col_3, col_4, col_5 = st.columns(5)
                
                with col_1:
                    st.metric(
                        "Individual Node Fitness", 
                        f"{latest['individual_fitness']:.3f}",
                        help="Average fitness across all institutional nodes"
                    )
                
                with col_2:
                    st.metric(
                        "System Health Ω(S,t)", 
                        f"{latest['system_health']:.3f}",
                        help="Overall system resilience and coordination"
                    )
                
                with col_3:
                    st.metric(
                        "Coherence Asymmetry CA(t)", 
                        f"{latest['asymmetry']:.3f}",
                        help="Institutional coordination asymmetry measure"
                    )
                
                with col_4:
                    st.metric(
                        "Critical Transition Risk CTR(t)", 
                        f"{latest['critical_transition_risk']:.3f}",
                        delta="⚠️ HIGH" if latest['critical_transition_risk'] > 3.0 else "✅ MODERATE" if latest['critical_transition_risk'] > 1.5 else "🟢 LOW"
                    )
                
                with col_5:
                    st.metric(
                        "Network Coherence NC(t)", 
                        f"{latest['network_coherence']:.3f}",
                        help="Inter-institutional coherence correlation"
                    )
                
                # Current Attractor Status
                st.markdown("### 🎯 Current Attractor State")
                attractor_colors = {
                    "Adaptive": "background-color: rgba(76, 175, 80, 0.1); border: 2px solid #4CAF50;",
                    "Authoritarian": "background-color: rgba(255, 193, 7, 0.1); border: 2px solid #FFC107;", 
                    "Fragmented": "background-color: rgba(255, 87, 34, 0.1); border: 2px solid #FF5722;",
                    "Collapse": "background-color: rgba(244, 67, 54, 0.1); border: 2px solid #F44336;",
                    "Transitional": "background-color: rgba(33, 150, 243, 0.1); border: 2px solid #2196F3;"
                }
                
                attractor_style = attractor_colors.get(latest['attractor'], "")
                st.markdown(f"""
                <div style="{attractor_style} padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                    <h3 style="margin: 0;">{latest['attractor_icon']} {latest['attractor']} Attractor</h3>
                    <p style="margin: 0.5rem 0;"><strong>Processing Efficiency:</strong> {latest['spe']:.2f}</p>
                    <p style="margin: 0;"><strong>Network Coherence:</strong> {latest['network_coherence']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("### 📈 Stress Dynamics Analysis")
            
            # 1. Stress Trajectories
            st.markdown("#### Individual Node Stress Trajectories S(t)")
            figS = px.line(
                dfv, x="Year", y="S", color="Node", 
                markers=True, title=f"Stress Evolution: {selected_dataset}"
            )
            figS.add_hline(y=tau, line_dash="dash", line_color="red", 
                          annotation_text=f"Stress Tolerance τ={tau}")
            st.plotly_chart(figS, use_container_width=True)
            
            # 2. System Health and Risk
            st.markdown("#### System Health Ψ(t) and Risk Index Λ(t)")
            figHR = go.Figure()
            
            figHR.add_trace(go.Scatter(
                x=sys_df.Year, y=sys_df.system_health, 
                name="System Health Ψ", line=dict(color="green", width=3),
                hovertemplate="Year: %{x}<br>Health: %{y:.2f}<extra></extra>"
            ))
            
            figHR.add_trace(go.Scatter(
                x=sys_df.Year, y=sys_df.risk, 
                name="Risk Λ", line=dict(color="red", width=2),
                hovertemplate="Year: %{x}<br>Risk: %{y:.3f}<extra></extra>"
            ))
            
            if show_risk_threshold:
                figHR.add_hline(
                    y=0.5, line_dash="dash", line_color="red", 
                    annotation_text="Collapse Risk (Λ=0.5)"
                )
            
            figHR.update_layout(
                title="System Health and Risk Evolution",
                xaxis_title="Year",
                yaxis_title="Value",
                hovermode="x unified"
            )
            
            st.plotly_chart(figHR, use_container_width=True)
            
            # 3. Phase Space Analysis
            st.markdown("#### Phase Space: Coherence vs Stress")
            
            if len(sel_nodes) > 0:
                node_opt = st.selectbox("Node for Phase Space:", options=sel_nodes)
                sub = dfv[dfv.Node == node_opt]
                
                figPS = px.scatter(
                    sub, x="C", y="S", color="Year", 
                    size="H", hover_data=["K", "A"],
                    title=f"{node_opt}: Coherence vs Stress Phase Space",
                    labels={"C": "Coherence", "S": "Stress"}
                )
                
                # Add stress tolerance line
                figPS.add_hline(y=tau, line_dash="dash", line_color="red")
                figPS.add_hline(y=-tau, line_dash="dash", line_color="red")
                
                st.plotly_chart(figPS, use_container_width=True)
            
            # 4. Processing Efficiency and Attractors
            if show_attractors:
                st.markdown("#### Stress Processing Efficiency & Phase Attractors")
                
                fig_spe = go.Figure()
                
                # Color-code by attractor type
                attractor_colors = {
                    "Adaptive": "green",
                    "Authoritarian": "orange", 
                    "Fragmented": "red",
                    "Collapse": "darkred",
                    "Transitional": "blue"
                }
                
                for attractor in sys_df['attractor'].unique():
                    mask_att = sys_df['attractor'] == attractor
                    subset = sys_df[mask_att]
                    
                    fig_spe.add_trace(go.Scatter(
                        x=subset['Year'], y=subset['spe'],
                        mode='lines+markers', 
                        name=f"{attractor}",
                        line=dict(color=attractor_colors.get(attractor, 'gray')),
                        hovertemplate="Year: %{x}<br>SPE: %{y:.2f}<br>Attractor: " + attractor + "<extra></extra>"
                    ))
                
                fig_spe.add_hline(y=2.0, line_dash="dash", line_color="green",
                                 annotation_text="High Efficiency (SPE=2.0)")
                
                fig_spe.update_layout(
                    title="Stress Processing Efficiency by Phase Attractor",
                    xaxis_title="Year",
                    yaxis_title="Processing Efficiency SPE(t)",
                    showlegend=True
                )
                
                st.plotly_chart(fig_spe, use_container_width=True)
            
            # 5. Bond Strength Matrix
            st.markdown("#### Inter-Institutional Bond Strength Matrix")
            
            # Use most recent year
            last_year = sys_df.Year.iloc[-1]
            last_df = dfv[dfv.Year == last_year].set_index("Node")
            
            if len(last_df) > 1:
                Bmat = bond_strength_matrix(last_df["C"].values)
                
                figB = px.imshow(
                    Bmat, 
                    text_auto=".2f",
                    x=last_df.index, 
                    y=last_df.index,
                    color_continuous_scale="RdBu_r",
                    zmin=0, zmax=1,
                    title=f"Bond Strength Matrix B(i,j) - Year {int(last_year)}"
                )
                
                st.plotly_chart(figB, use_container_width=True)
            
            # 6. 32-Dimensional Phase Space Analysis with Dimensionality Reduction
            if show_32d_analysis and len(phase_space_features) > 1:
                st.markdown("#### 32-Dimensional Phase Space Analysis")
                
                # Apply dimensionality reduction
                features_array = np.array(phase_space_features)
                reduced_features, variance_info = apply_dimensionality_reduction(
                    features_array, method=dim_reduction_method, n_components=2
                )
                
                # Create phase space dataframe
                phase_df = pd.DataFrame({
                    'PC1': reduced_features[:, 0],
                    'PC2': reduced_features[:, 1],
                    'Year': sys_df['Year'],
                    'Attractor': sys_df['attractor'],
                    'SystemHealth': sys_df['system_health'],
                    'CriticalRisk': sys_df['critical_transition_risk']
                })
                
                # Create 32D phase space plot
                fig_32d = go.Figure()
                
                # Add attractor regions (conceptual)
                attractor_regions = [
                    {'center': [1, 1], 'name': 'A₁ Adaptive', 'color': 'green'},
                    {'center': [0, 0], 'name': 'A₂ Authoritarian', 'color': 'orange'},
                    {'center': [-1, -1], 'name': 'A₃ Fragmented', 'color': 'red'},
                    {'center': [-2, -2], 'name': 'A₄ Collapse', 'color': 'darkred'}
                ]
                
                for region in attractor_regions:
                    fig_32d.add_shape(
                        type="circle",
                        x0=region['center'][0]-0.5, y0=region['center'][1]-0.5,
                        x1=region['center'][0]+0.5, y1=region['center'][1]+0.5,
                        fillcolor=region['color'], opacity=0.1,
                        line=dict(color=region['color'], width=1)
                    )
                    fig_32d.add_annotation(
                        x=region['center'][0], y=region['center'][1],
                        text=region['name'], showarrow=False,
                        font=dict(color=region['color'], size=10)
                    )
                
                # Add trajectory
                if len(phase_df) > 1:
                    fig_32d.add_trace(go.Scatter(
                        x=phase_df['PC1'], y=phase_df['PC2'],
                        mode='markers+lines',
                        marker=dict(
                            size=[h*3 + 5 for h in phase_df['SystemHealth']],
                            color=phase_df['CriticalRisk'],
                            colorscale='RdYlBu_r',
                            showscale=True,
                            colorbar=dict(title="Critical Risk CTR(t)")
                        ),
                        line=dict(color='blue', width=2),
                        text=[f"{row['Year']}<br>{row['Attractor']}" for _, row in phase_df.iterrows()],
                        name=selected_dataset,
                        hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Health: %{marker.size}<br>Risk: %{marker.color:.3f}<extra></extra>"
                    ))
                else:
                    fig_32d.add_trace(go.Scatter(
                        x=phase_df['PC1'], y=phase_df['PC2'],
                        mode='markers+text',
                        marker=dict(size=20, color=phase_df['CriticalRisk'].iloc[0]),
                        text=[selected_dataset],
                        textposition='top center',
                        name=selected_dataset
                    ))
                
                fig_32d.update_layout(
                    title=f"32D Phase Space Analysis - {dim_reduction_method} Reduction<br><sub>{variance_info}</sub>",
                    xaxis_title=f"{dim_reduction_method} Component 1",
                    yaxis_title=f"{dim_reduction_method} Component 2",
                    height=600
                )
                
                st.plotly_chart(fig_32d, use_container_width=True)
            
            # 7. System Health & Risk Trends
            st.markdown("#### System Health & Risk Trends")
            
            if len(sys_df) > 1:
                # Multi-metric trends plot
                fig_trends = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'System Health Ω(S,t)', 'Critical Transition Risk CTR(t)',
                        'Coherence Asymmetry CA(t)', 'Network Coherence NC(t)'
                    ]
                )
                
                # System Health
                fig_trends.add_trace(
                    go.Scatter(x=sys_df['Year'], y=sys_df['system_health'],
                              name='System Health', line=dict(color='blue', width=3)),
                    row=1, col=1
                )
                
                # Critical Transition Risk
                fig_trends.add_trace(
                    go.Scatter(x=sys_df['Year'], y=sys_df['critical_transition_risk'],
                              name='Critical Risk', line=dict(color='red', width=3)),
                    row=1, col=2
                )
                
                # Coherence Asymmetry
                fig_trends.add_trace(
                    go.Scatter(x=sys_df['Year'], y=sys_df['asymmetry'],
                              name='Coherence Asymmetry', line=dict(color='orange', width=3)),
                    row=2, col=1
                )
                
                # Network Coherence
                fig_trends.add_trace(
                    go.Scatter(x=sys_df['Year'], y=sys_df['network_coherence'],
                              name='Network Coherence', line=dict(color='green', width=3)),
                    row=2, col=2
                )
                
                fig_trends.update_layout(
                    height=800,
                    title_text=f"CAMS-CAN v3.4 System Health & Risk Trends - {selected_dataset}",
                    showlegend=False
                )
                
                st.plotly_chart(fig_trends, use_container_width=True)
                
                # Attractor Transitions Analysis
                transitions = identify_attractor_transitions(sys_df)
                if transitions:
                    st.markdown("#### 🔄 Attractor Transitions Detected")
                    for i, trans in enumerate(transitions[:5]):  # Show first 5 transitions
                        direction = "📈 Positive" if trans['health_change'] > 0 else "📉 Negative" if trans['health_change'] < 0 else "⚖️ Neutral"
                        st.write(f"**{trans['year']}:** {trans['from']} → {trans['to']} ({direction} health change: {trans['health_change']:+.3f})")
            
            # 8. Original PCA Trajectory (Enhanced)
            if len(sys_df) > 2:  # Need multiple time points for meaningful PCA
                st.markdown("#### Phase Space Trajectory (PCA)")
                
                # Create phase vector for each year (all nodes' [C,K,S,A])
                all_years = sorted(dfv["Year"].unique())
                phase_vecs = []
                
                for y in all_years:
                    year_data = dfv[dfv["Year"] == y]
                    phase_vec = []
                    
                    for node in nodes:  # Use all original nodes for consistency
                        node_data = year_data[year_data["Node"] == node]
                        if len(node_data) == 1:
                            phase_vec.extend(node_data[["C", "K", "S", "A"]].values[0])
                        else:
                            phase_vec.extend([0, 0, 0, 0])  # Missing node data
                    
                    phase_vecs.append(phase_vec)
                
                if len(phase_vecs) > 1:
                    X = np.array(phase_vecs)
                    
                    # PCA reduction to 2D
                    pca = PCA(n_components=2)
                    X2 = pca.fit_transform(X)
                    
                    # Create trajectory plot
                    pca_df = pd.DataFrame({
                        'PC1': X2[:, 0],
                        'PC2': X2[:, 1], 
                        'Year': all_years
                    })
                    
                    # Add attractor information
                    pca_df = pca_df.merge(
                        sys_df[['Year', 'attractor', 'system_health']], 
                        on='Year', 
                        how='left'
                    )
                    
                    figPCA = px.scatter(
                        pca_df, x='PC1', y='PC2', 
                        color='attractor', size='system_health',
                        hover_data=['Year'],
                        title="Phase Space Attractor Trajectory (PCA Projection)",
                        labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", 
                               "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)"}
                    )
                    
                    # Add trajectory line
                    figPCA.add_trace(go.Scatter(
                        x=X2[:, 0], y=X2[:, 1],
                        mode='lines', 
                        line=dict(color='gray', width=1, dash='dot'),
                        name='Trajectory',
                        showlegend=False
                    ))
                    
                    st.plotly_chart(figPCA, use_container_width=True)
                    
                    # PCA interpretation
                    st.markdown(f"""
                    **PCA Interpretation:**
                    - PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance
                    - PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance  
                    - Total explained: {sum(pca.explained_variance_ratio_):.1%}
                    """)
            
            # Enhanced CAMS-CAN v3.4 Summary Analysis
            if len(sys_df) > 1:
                st.markdown("### 📊 CAMS-CAN v3.4 Analysis Summary")
                
                col_1, col_2, col_3, col_4 = st.columns(4)
                
                with col_1:
                    st.markdown("**System Health Trajectory**")
                    health_change = sys_df['system_health'].iloc[-1] - sys_df['system_health'].iloc[0]
                    health_trend = "📈 Improving" if health_change > 0.1 else "📉 Declining" if health_change < -0.1 else "⚖️ Stable"
                    st.write(f"Trend: {health_trend}")
                    st.write(f"Change: {health_change:+.3f}")
                    st.write(f"Average: {sys_df['system_health'].mean():.3f}")
                    st.write(f"Volatility: {sys_df['system_health'].std():.3f}")
                
                with col_2:
                    st.markdown("**Critical Risk Assessment**")
                    max_ctr = sys_df['critical_transition_risk'].max()
                    current_ctr = sys_df['critical_transition_risk'].iloc[-1]
                    high_risk_periods = len(sys_df[sys_df['critical_transition_risk'] > 3.0])
                    st.write(f"Peak CTR: {max_ctr:.3f}")
                    st.write(f"Current CTR: {current_ctr:.3f}")
                    st.write(f"High-Risk Periods: {high_risk_periods}/{len(sys_df)}")
                    risk_trend = "⚠️ Increasing" if sys_df['critical_transition_risk'].iloc[-1] > sys_df['critical_transition_risk'].iloc[0] else "✅ Decreasing"
                    st.write(f"Trend: {risk_trend}")
                
                with col_3:
                    st.markdown("**Network Coherence**") 
                    avg_coherence = sys_df['network_coherence'].mean()
                    coherence_trend = sys_df['network_coherence'].iloc[-1] - sys_df['network_coherence'].iloc[0]
                    st.write(f"Average NC: {avg_coherence:.3f}")
                    st.write(f"Current NC: {sys_df['network_coherence'].iloc[-1]:.3f}")
                    st.write(f"Change: {coherence_trend:+.3f}")
                    coherence_stability = "🟢 High" if avg_coherence > 0.7 else "🟡 Moderate" if avg_coherence > 0.4 else "🔴 Low"
                    st.write(f"Stability: {coherence_stability}")
                
                with col_4:
                    st.markdown("**Attractor Analysis**") 
                    attractor_counts = sys_df['attractor'].value_counts()
                    dominant = attractor_counts.index[0]
                    dominant_pct = attractor_counts.iloc[0] / len(sys_df) * 100
                    transitions = identify_attractor_transitions(sys_df)
                    st.write(f"Dominant: {dominant}")
                    st.write(f"Stability: {dominant_pct:.1f}%")
                    st.write(f"Transitions: {len(transitions)}")
                    
                    # Current attractor assessment
                    current_attractor = sys_df['attractor'].iloc[-1]
                    attractor_assessment = {
                        "Adaptive": "🟢 Optimal",
                        "Authoritarian": "🟡 Stable",
                        "Fragmented": "🟠 Unstable", 
                        "Collapse": "🔴 Critical",
                        "Transitional": "🔵 Dynamic"
                    }
                    st.write(f"Status: {attractor_assessment.get(current_attractor, '❓ Unknown')}")
            
            st.markdown("---")
            st.info("💡 **Tip:** All charts update interactively with your time range, node selection, and parameter adjustments. Try focusing on crisis periods or comparing institutional stress patterns.")
    
    else:
        st.warning("Please select a dataset to begin analysis.")

# Sidebar with methodology
st.sidebar.markdown("## 📖 Methodology")
st.sidebar.markdown("""
### Core Equations

**Node Fitness:**
```
Hi(t) = Ci*Ki / (1+exp((|Si|-τ)/λ)) * (1+Ai/10)
```

**System Health:**
```
Ψ(t) = (∏Hi)^(1/n) * (1+Φ_net) * R(t)
```

**Risk Index:**
```
Λ = CA / (1 + Ψ)
```

**Processing Efficiency:**
```
SPE = Σ(Hi) / Σ(|Si|*Ai)
```

### Phase Attractors
- 🟢 **Adaptive:** Ψ>3.5, CA<0.3
- 🟡 **Authoritarian:** Ψ∈[2.5,3.5] 
- 🟠 **Fragmented:** Ψ∈[1.5,2.5], CA>0.4
- 🔴 **Collapse:** Ψ<1.5
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 CAMS-CAN v3.4 Features

**Core Capabilities:**
- ✅ 8 Institutional Nodes Analysis
- ✅ 32-Dimensional Phase Space Visualization  
- ✅ 5 Key Metrics Dashboard
- ✅ 6 Interactive Visualization Components
- ✅ Time Window & Institution Filtering
- ✅ Attractor Behavior Detection
- ✅ Transition Analysis & Risk Assessment

**Dimensionality Reduction:**
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation)

**Mathematical Framework:**
Formal implementation of stress as societal meta-cognition
with comprehensive attractor dynamics and phase space analysis.
""")

st.sidebar.markdown("---")  
st.sidebar.markdown("**CAMS-CAN v3.4** | *Complex Adaptive Management Systems*")
st.sidebar.markdown("*Stress as Societal Meta-Cognition Framework*")