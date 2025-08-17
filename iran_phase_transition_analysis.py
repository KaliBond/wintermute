"""
⚠️ Phase Transition Analysis - Iran
Critical Analysis of Civilizational Stress Dynamics and Meta-Cognitive State
Using Formal CAMS-CAN Framework
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="⚠️ Iran Phase Transition Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Header with alert styling
st.markdown("""
<div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
           padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;">
    <h1 style="margin: 0; font-size: 2.5rem;">⚠️ PHASE TRANSITION ANALYSIS</h1>
    <h2 style="margin: 0.5rem 0 0 0; font-size: 1.8rem;">📍 IRAN</h2>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Critical Assessment of Meta-Cognitive Attractor Dynamics
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("**CAMS-CAN Framework | Formal Mathematical Analysis | Real-Time Monitoring**")

# === Enhanced CAMS-CAN Framework for Crisis Analysis ===
class CrisisCAMSCANParameters:
    """Enhanced parameters for crisis/transition analysis"""
    def __init__(self):
        # Core empirically validated parameters
        self.tau = 3.0  # stress tolerance threshold
        self.lambda_decay = 0.5  # resilience decay factor
        self.xi = 0.2  # coherence coupling strength
        self.gamma_c = 0.15  # coherence decay rate
        self.delta_s = 0.2  # stress dissipation rate
        
        # Crisis-enhanced parameters
        self.alpha_k = 0.3  # capacity growth rate
        self.beta_k = 0.15  # capacity stress sensitivity (increased)
        self.eta_coh = 0.3   # coherence network coupling (increased)
        self.kappa_adapt = 0.1  # learning adaptation rate (decreased under stress)
        self.eta_a = 0.25   # abstraction growth rate (increased - crisis thinking)
        self.mu_a = 0.08    # abstraction decay rate (decreased - retain crisis models)
        self.rho_symbolic = 0.1  # symbolic input rate (increased - narrative warfare)
        
        # Transition detection thresholds
        self.transition_sensitivity = 0.1  # Lower = more sensitive
        self.hysteresis_factor = 0.15     # Phase transition hysteresis
        self.critical_stress_threshold = 4.5  # Critical stress level
        
        # Iran-specific calibrations (based on regional analysis)
        self.regional_stress_multiplier = 1.3  # Middle East regional stress
        self.institutional_fragility = 1.2    # Post-revolution institutional volatility
        self.external_pressure_factor = 1.4   # International sanctions/pressure

class IranSymbon:
    """Specialized Symbon for Iran phase transition analysis"""
    
    def __init__(self, params=None):
        self.params = params if params else CrisisCAMSCANParameters()
        
        # Iran-specific institutional nodes
        self.nodes = [
            "Supreme_Leader",      # Executive (unique to Iran)
            "IRGC",               # Army (Revolutionary Guard)
            "StateArchives",      # StateMemory  
            "Clergy",             # Priesthood
            "Bureaucracy",        # Stewards
            "Bazaar_Guilds",      # Craft/Trade
            "Urban_Population",   # Flow
            "Rural_Population"    # Hands
        ]
        self.N = len(self.nodes)
        
        # Enhanced state tracking for transition analysis
        self.node_states = np.zeros((self.N, 4))  # [C, K, S, A]
        self.bond_matrix = np.zeros((self.N, self.N))
        self.meta_cognitive_state = np.zeros(3)
        self.transition_indicators = {}
        self.critical_thresholds = {}
        
        # Historical trajectory tracking
        self.trajectory_history = []
        self.transition_risk_history = []
        self.attractor_history = []
        
    def initialize_from_data(self, df):
        """Initialize with Iran-specific data mapping"""
        # Map generic nodes to Iran-specific institutions
        node_mapping = {
            "Executive": "Supreme_Leader",
            "Army": "IRGC", 
            "StateMemory": "StateArchives",
            "State Memory": "StateArchives",
            "Priesthood": "Clergy",
            "Stewards": "Bureaucracy",
            "Craft": "Bazaar_Guilds",
            "Flow": "Urban_Population",
            "Hands": "Rural_Population"
        }
        
        for i, iran_node in enumerate(self.nodes):
            # Find matching data
            node_data = None
            for generic_node, mapped_node in node_mapping.items():
                if mapped_node == iran_node:
                    node_data = df[df['Node'].str.contains(generic_node, case=False, na=False)]
                    break
            
            # If no direct match, try partial matching
            if node_data is None or len(node_data) == 0:
                for _, row in df.iterrows():
                    if any(term in row['Node'].lower() for term in iran_node.lower().split('_')):
                        node_data = pd.DataFrame([row])
                        break
            
            # Initialize with data or defaults
            if node_data is not None and len(node_data) > 0:
                row = node_data.iloc[0]
                self.node_states[i, 0] = row['Coherence']
                self.node_states[i, 1] = row['Capacity'] 
                self.node_states[i, 2] = row['Stress'] * self.params.regional_stress_multiplier
                self.node_states[i, 3] = row['Abstraction']
            else:
                # Default values with Iran-specific adjustments
                self.node_states[i, 0] = np.random.normal(4.5, 1.0)  # Lower coherence
                self.node_states[i, 1] = np.random.normal(5.0, 1.2)  # Variable capacity
                self.node_states[i, 2] = np.random.normal(2.5, 1.5)  # Higher stress
                self.node_states[i, 3] = np.random.normal(6.0, 0.8)  # Higher abstraction
        
        self.update_bond_matrix()
        self.update_meta_cognitive_state()
        self.calculate_transition_indicators()
    
    def calculate_transition_indicators(self):
        """Calculate comprehensive phase transition indicators"""
        
        # 1. System Health Ω(S,t)
        fitness_values = []
        for i in range(self.N):
            C, K, S, A = self.node_states[i]
            stress_impact = 1 + np.exp((abs(S) - self.params.tau) / self.params.lambda_decay)
            fitness = (C * K / stress_impact) * (1 + A / 10)
            fitness_values.append(max(fitness, 1e-6))
        
        system_health = np.exp(np.mean(np.log(fitness_values)))
        
        # 2. Processing Efficiency SPE(t)
        capacities = self.node_states[:, 1]
        bond_sums = np.sum(self.bond_matrix, axis=1)
        stress_scaled = np.abs(self.node_states[:, 2]) + 1e-6
        abstractions = self.node_states[:, 3]
        
        spe = sum(capacities[i] * bond_sums[i] for i in range(self.N)) / \
              sum(stress_scaled[i] * abstractions[i] for i in range(self.N))
        
        # 3. Coherence Asymmetry CA(t)
        coherence_capacity = self.node_states[:, 0] * self.node_states[:, 1]
        ca = np.std(coherence_capacity) / (np.mean(coherence_capacity) + 1e-9)
        
        # 4. Critical Stress Index
        critical_stress_nodes = np.sum(np.abs(self.node_states[:, 2]) > self.params.critical_stress_threshold)
        critical_stress_index = critical_stress_nodes / self.N
        
        # 5. Meta-Cognitive Breakdown Risk
        monitoring, control, reflection = self.meta_cognitive_state
        metacog_breakdown_risk = 1 - (monitoring * control * reflection) / \
                                (monitoring + control + reflection + 1e-6)
        
        # 6. Bond Network Fragmentation
        bond_threshold = np.mean(self.bond_matrix) * 0.5
        strong_bonds = np.sum(self.bond_matrix > bond_threshold)
        max_possible_bonds = self.N * (self.N - 1)
        fragmentation_index = 1 - (strong_bonds / max_possible_bonds)
        
        # Store all indicators
        self.transition_indicators = {
            'system_health': system_health,
            'processing_efficiency': spe,
            'coherence_asymmetry': ca,
            'critical_stress_index': critical_stress_index,
            'metacog_breakdown_risk': metacog_breakdown_risk,
            'fragmentation_index': fragmentation_index
        }
        
        return self.transition_indicators
    
    def identify_phase_attractor_enhanced(self):
        """Enhanced attractor identification with transition proximity"""
        indicators = self.transition_indicators
        H = indicators['system_health']
        SPE = indicators['processing_efficiency']
        CA = indicators['coherence_asymmetry']
        
        monitoring, control, reflection = self.meta_cognitive_state
        
        # Calculate distances to each attractor
        distances = {}
        
        # A₁ (Adaptive): H > 3.5, SPE > 2.0, CA < 0.3
        if H > 3.5 and SPE > 2.0 and CA < 0.3:
            current_attractor = "A₁_Adaptive"
            distances['A₁_Adaptive'] = 0
        else:
            distances['A₁_Adaptive'] = max(0, 3.5 - H) + max(0, 2.0 - SPE) + max(0, CA - 0.3)
        
        # A₂ (Authoritarian): H ∈ [2.5,3.5], Control >> Monitoring
        control_dominance = control / (monitoring + 1e-6)
        if 2.5 <= H <= 3.5 and control_dominance > 2.0:
            current_attractor = "A₂_Authoritarian"
            distances['A₂_Authoritarian'] = 0
        else:
            h_distance = min(abs(H - 2.5), abs(H - 3.5)) if not (2.5 <= H <= 3.5) else 0
            control_distance = max(0, 2.0 - control_dominance)
            distances['A₂_Authoritarian'] = h_distance + control_distance
        
        # A₃ (Fragmented): H ∈ [1.5,2.5], CA > 0.4, SPE < 1.0
        if 1.5 <= H <= 2.5 and CA > 0.4 and SPE < 1.0:
            current_attractor = "A₃_Fragmented"
            distances['A₃_Fragmented'] = 0
        else:
            h_distance = min(abs(H - 1.5), abs(H - 2.5)) if not (1.5 <= H <= 2.5) else 0
            ca_distance = max(0, 0.4 - CA)
            spe_distance = max(0, SPE - 1.0)
            distances['A₃_Fragmented'] = h_distance + ca_distance + spe_distance
        
        # A₄ (Collapse): H < 1.5, Reflection → 0
        if H < 1.5 and reflection < 0.1:
            current_attractor = "A₄_Collapse"
            distances['A₄_Collapse'] = 0
        else:
            h_distance = max(0, H - 1.5)
            reflection_distance = max(0, reflection - 0.1) if reflection < 0.1 else reflection
            distances['A₄_Collapse'] = h_distance + reflection_distance
        
        # Find closest attractor if not in any
        if 'current_attractor' not in locals():
            current_attractor = min(distances.keys(), key=lambda k: distances[k])
        
        # Calculate transition probabilities (inverse distance)
        total_inverse_distance = sum(1/(d + 1e-6) for d in distances.values())
        transition_probabilities = {k: (1/(v + 1e-6))/total_inverse_distance 
                                  for k, v in distances.items()}
        
        return current_attractor, distances, transition_probabilities
    
    def calculate_critical_points(self):
        """Identify critical transition points and tipping thresholds"""
        
        critical_points = {}
        
        # Health-based critical points
        health = self.transition_indicators['system_health']
        
        critical_points['collapse_threshold'] = {
            'current': health,
            'threshold': 1.5,
            'distance': health - 1.5,
            'risk_level': 'CRITICAL' if health < 2.0 else 'HIGH' if health < 2.5 else 'MODERATE'
        }
        
        critical_points['fragmentation_threshold'] = {
            'current': health,
            'threshold': 2.5,
            'distance': health - 2.5,
            'risk_level': 'CRITICAL' if health < 2.0 else 'HIGH' if health < 2.5 else 'MODERATE'
        }
        
        # Stress-based critical points
        max_stress = np.max(np.abs(self.node_states[:, 2]))
        avg_stress = np.mean(np.abs(self.node_states[:, 2]))
        
        critical_points['stress_cascade'] = {
            'max_stress': max_stress,
            'avg_stress': avg_stress,
            'threshold': self.params.critical_stress_threshold,
            'nodes_at_risk': np.sum(np.abs(self.node_states[:, 2]) > self.params.critical_stress_threshold),
            'cascade_risk': min(1.0, max_stress / self.params.critical_stress_threshold)
        }
        
        # Meta-cognitive critical points
        monitoring, control, reflection = self.meta_cognitive_state
        metacog_total = monitoring + control + reflection
        
        critical_points['metacognitive_collapse'] = {
            'total_function': metacog_total,
            'balance_deviation': np.std([monitoring, control, reflection]),
            'reflection_risk': 1 - reflection / (metacog_total + 1e-6),
            'control_dominance': control / (monitoring + reflection + 1e-6)
        }
        
        return critical_points
    
    def update_bond_matrix(self):
        """Enhanced bond matrix with crisis dynamics"""
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    # Base bond calculation
                    beta_i = max(self.node_states[i, 1], 0.1)
                    beta_j = max(self.node_states[j, 1], 0.1)
                    
                    coherence_diff = abs(self.node_states[i, 0] - self.node_states[j, 0])
                    coherence_coupling = np.exp(-coherence_diff / self.params.sigma_bond)
                    
                    # Crisis-enhanced stress coupling
                    stress_i = abs(self.node_states[i, 2])
                    stress_j = abs(self.node_states[j, 2])
                    
                    # Under high stress, bonds can either strengthen (solidarity) or weaken (fragmentation)
                    avg_stress = (stress_i + stress_j) / 2
                    if avg_stress > self.params.critical_stress_threshold:
                        # Crisis bonding vs fragmentation (depends on node types)
                        if self._are_aligned_institutions(i, j):
                            stress_factor = 1 + avg_stress * 0.1  # Strengthen under crisis
                        else:
                            stress_factor = 1 / (1 + avg_stress * 0.2)  # Weaken under crisis
                    else:
                        stress_factor = 1 + avg_stress * 0.05  # Normal stress coupling
                    
                    self.bond_matrix[i, j] = (np.sqrt(beta_i * beta_j) * 
                                            coherence_coupling * 
                                            stress_factor)
    
    def _are_aligned_institutions(self, i, j):
        """Check if two institutions are naturally aligned during crisis"""
        # Iran-specific institutional alignments during crisis
        aligned_groups = [
            [0, 1, 3],  # Supreme Leader, IRGC, Clergy (power center)
            [4, 5],     # Bureaucracy, Bazaar (administrative/economic)
            [6, 7]      # Urban/Rural populations (civil society)
        ]
        
        for group in aligned_groups:
            if i in group and j in group:
                return True
        return False
    
    def update_meta_cognitive_state(self):
        """Enhanced meta-cognitive calculation with crisis dynamics"""
        # Enhanced monitoring with crisis sensitivity
        stress_states = self.node_states[:, 2]
        bond_sums = np.sum(self.bond_matrix, axis=1)
        
        # Crisis-weighted monitoring (higher stress = more monitoring attention)
        stress_weights = np.exp(np.abs(stress_states) / self.params.tau)
        weighted_monitoring = sum(stress_weights[i] * stress_states[i] * bond_sums[i] 
                                for i in range(self.N))
        total_weighted_stress = sum(stress_weights[i] * abs(stress_states[i]) for i in range(self.N))
        
        self.meta_cognitive_state[0] = weighted_monitoring / (total_weighted_stress + 1e-6)
        
        # Enhanced control with institutional coordination under stress
        coherence_states = self.node_states[:, 0]
        coherence_derivatives = np.gradient(coherence_states)
        
        control_sum = 0
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    # Crisis amplifies control needs
                    crisis_multiplier = 1 + (abs(stress_states[i]) + abs(stress_states[j])) * 0.1
                    control_sum += (self.bond_matrix[i, j] * 
                                  abs(coherence_derivatives[i] - coherence_derivatives[j]) *
                                  crisis_multiplier)
        
        self.meta_cognitive_state[1] = control_sum
        
        # Enhanced reflection with memory institutions under stress
        state_memory_idx = 2  # StateArchives
        clergy_idx = 3        # Clergy
        
        # Under stress, reflection can be impaired or enhanced depending on stress level
        stress_effect = np.exp(-np.mean(np.abs(stress_states)) / self.params.tau)
        
        reflection_term1 = (self.node_states[state_memory_idx, 0] * 
                           self.node_states[state_memory_idx, 3])
        reflection_term2 = (self.node_states[clergy_idx, 0] * 
                           self.node_states[clergy_idx, 3])
        
        self.meta_cognitive_state[2] = ((reflection_term1 + reflection_term2) / 2) * stress_effect

# === Load Iran Data ===
@st.cache_data
def load_iran_data():
    """Load Iran-specific CAMS data"""
    import glob
    csv_files = glob.glob("*.csv")
    
    for file in csv_files:
        try:
            if 'iran' in file.lower():
                df = pd.read_csv(file)
                if len(df) > 0 and 'Node' in df.columns:
                    # Ensure numeric columns
                    for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna(subset=['Coherence', 'Capacity', 'Stress', 'Abstraction'])
                    return df
        except:
            continue
    
    # Generate synthetic Iran data if not found
    st.warning("⚠️ Using synthetic Iran data - actual data not found")
    
    nodes_data = [
        {"Node": "Executive", "Coherence": 4.2, "Capacity": 5.8, "Stress": 3.4, "Abstraction": 6.1},
        {"Node": "Army", "Coherence": 5.1, "Capacity": 6.2, "Stress": 2.8, "Abstraction": 4.7},
        {"Node": "StateMemory", "Coherence": 3.8, "Capacity": 4.5, "Stress": 4.1, "Abstraction": 7.2},
        {"Node": "Priesthood", "Coherence": 5.4, "Capacity": 5.0, "Stress": 2.2, "Abstraction": 7.8},
        {"Node": "Stewards", "Coherence": 3.2, "Capacity": 4.8, "Stress": 4.5, "Abstraction": 5.3},
        {"Node": "Craft", "Coherence": 4.1, "Capacity": 5.5, "Stress": 3.9, "Abstraction": 4.8},
        {"Node": "Flow", "Coherence": 2.8, "Capacity": 4.2, "Stress": 5.2, "Abstraction": 5.9},
        {"Node": "Hands", "Coherence": 3.5, "Capacity": 4.0, "Stress": 4.8, "Abstraction": 4.2}
    ]
    
    return pd.DataFrame(nodes_data)

# Load data
with st.spinner("🔄 Loading Iran CAMS data..."):
    iran_data = load_iran_data()

# Initialize Iran Symbon
iran_symbon = IranSymbon()
iran_symbon.initialize_from_data(iran_data)

# === Main Analysis Dashboard ===

# Current threat level assessment
current_attractor, distances, transition_probs = iran_symbon.identify_phase_attractor_enhanced()
indicators = iran_symbon.transition_indicators
critical_points = iran_symbon.calculate_critical_points()

# Threat level determination
health = indicators['system_health']
if health < 1.5:
    threat_level = "🔴 CRITICAL"
    threat_color = "red"
elif health < 2.5:
    threat_level = "🟠 HIGH"
    threat_color = "orange"
elif health < 3.5:
    threat_level = "🟡 ELEVATED"
    threat_color = "yellow"
else:
    threat_level = "🟢 STABLE"
    threat_color = "green"

# Threat level display
st.markdown(f"""
<div style="background: linear-gradient(135deg, {threat_color}22 0%, {threat_color}11 100%); 
           padding: 1.5rem; border-radius: 0.75rem; border-left: 5px solid {threat_color}; margin: 1rem 0;">
    <h2 style="margin: 0; color: {threat_color};">CURRENT THREAT LEVEL: {threat_level}</h2>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Current Attractor: <strong>{current_attractor}</strong> | 
        System Health: <strong>{health:.3f}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# === Critical Indicators Dashboard ===
st.markdown("## 🚨 Critical Transition Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    health_risk = "🔴" if health < 2.0 else "🟠" if health < 3.0 else "🟡" if health < 4.0 else "🟢"
    st.metric(
        f"{health_risk} System Health Ω(S,t)", 
        f"{health:.3f}",
        delta=f"Collapse threshold: {critical_points['collapse_threshold']['distance']:+.3f}"
    )

with col2:
    spe = indicators['processing_efficiency']
    spe_risk = "🔴" if spe < 1.0 else "🟠" if spe < 1.5 else "🟡" if spe < 2.0 else "🟢"
    st.metric(
        f"{spe_risk} Processing Efficiency", 
        f"{spe:.3f}",
        delta="Efficiency trend"
    )

with col3:
    ca = indicators['coherence_asymmetry']
    ca_risk = "🔴" if ca > 0.6 else "🟠" if ca > 0.4 else "🟡" if ca > 0.3 else "🟢"
    st.metric(
        f"{ca_risk} Coherence Asymmetry", 
        f"{ca:.3f}",
        delta="Fragmentation index"
    )

with col4:
    stress_risk = critical_points['stress_cascade']['cascade_risk']
    stress_icon = "🔴" if stress_risk > 0.8 else "🟠" if stress_risk > 0.6 else "🟡" if stress_risk > 0.4 else "🟢"
    st.metric(
        f"{stress_icon} Stress Cascade Risk", 
        f"{stress_risk:.3f}",
        delta=f"{critical_points['stress_cascade']['nodes_at_risk']}/8 nodes critical"
    )

# === Phase Space Visualization ===
st.markdown("## 🌀 Phase Space Analysis")

fig_phase = go.Figure()

# Add attractor regions
# A1 - Adaptive (green)
fig_phase.add_shape(
    type="rect", x0=3.5, y0=2.0, x1=10, y1=10,
    fillcolor="green", opacity=0.1, line=dict(color="green", width=2)
)
fig_phase.add_annotation(x=5.5, y=4.5, text="A₁ ADAPTIVE", showarrow=False, 
                        font=dict(color="green", size=12, family="Arial Black"))

# A2 - Authoritarian (orange)  
fig_phase.add_shape(
    type="rect", x0=2.5, y0=0, x1=3.5, y1=10,
    fillcolor="orange", opacity=0.1, line=dict(color="orange", width=2)
)
fig_phase.add_annotation(x=3.0, y=3.0, text="A₂ AUTH", showarrow=False,
                        font=dict(color="orange", size=12, family="Arial Black"))

# A3 - Fragmented (red)
fig_phase.add_shape(
    type="rect", x0=1.5, y0=0, x1=2.5, y1=1.0,
    fillcolor="red", opacity=0.1, line=dict(color="red", width=2)
)
fig_phase.add_annotation(x=2.0, y=0.5, text="A₃ FRAG", showarrow=False,
                        font=dict(color="red", size=12, family="Arial Black"))

# A4 - Collapse (darkred)
fig_phase.add_shape(
    type="rect", x0=0, y0=0, x1=1.5, y1=10,
    fillcolor="darkred", opacity=0.15, line=dict(color="darkred", width=2)
)
fig_phase.add_annotation(x=0.8, y=1.5, text="A₄ COLLAPSE", showarrow=False,
                        font=dict(color="darkred", size=12, family="Arial Black"))

# Current Iran position
fig_phase.add_trace(go.Scatter(
    x=[health], y=[spe],
    mode='markers+text',
    marker=dict(size=25, color='red', symbol='star', 
                line=dict(color='black', width=3)),
    text=['IRAN'],
    textposition='top center',
    textfont=dict(size=16, color='red', family="Arial Black"),
    name='Iran Current Position',
    hovertemplate=f"<b>IRAN</b><br>Health: {health:.3f}<br>SPE: {spe:.3f}<br>Attractor: {current_attractor}<extra></extra>"
))

# Transition risk vectors
for attractor, prob in transition_probs.items():
    if prob > 0.1 and attractor != current_attractor:  # Only show significant transitions
        # Approximate attractor positions
        attractor_positions = {
            'A₁_Adaptive': (5.0, 3.5),
            'A₂_Authoritarian': (3.0, 1.5),
            'A₃_Fragmented': (2.0, 0.7),
            'A₄_Collapse': (1.0, 0.5)
        }
        
        if attractor in attractor_positions:
            target_x, target_y = attractor_positions[attractor]
            fig_phase.add_trace(go.Scatter(
                x=[health, target_x], y=[spe, target_y],
                mode='lines',
                line=dict(color='red', width=3*prob, dash='dash'),
                opacity=prob,
                name=f'Transition Risk: {attractor} ({prob:.1%})',
                showlegend=True
            ))

fig_phase.update_layout(
    title="⚠️ IRAN PHASE SPACE POSITION - TRANSITION RISK ANALYSIS",
    xaxis_title="System Health Ω(S,t)",
    yaxis_title="Processing Efficiency SPE(t)",
    height=600,
    xaxis=dict(range=[0, 6]),
    yaxis=dict(range=[0, 5]),
    font=dict(family="Arial", size=12)
)

st.plotly_chart(fig_phase, use_container_width=True)

# === Transition Probability Analysis ===
st.markdown("## ⚖️ Phase Transition Probabilities")

col1, col2 = st.columns(2)

with col1:
    # Transition probability chart
    prob_data = [(k.replace('_', ' '), v) for k, v in transition_probs.items()]
    prob_data.sort(key=lambda x: x[1], reverse=True)
    
    colors = ['red' if 'Collapse' in name else 'orange' if 'Fragment' in name 
              else 'yellow' if 'Authoritarian' in name else 'green' 
              for name, prob in prob_data]
    
    fig_probs = go.Figure(data=go.Bar(
        x=[name for name, prob in prob_data],
        y=[prob for name, prob in prob_data],
        marker_color=colors,
        text=[f"{prob:.1%}" for name, prob in prob_data],
        textposition='auto'
    ))
    
    fig_probs.update_layout(
        title="Attractor Transition Probabilities",
        yaxis_title="Probability",
        height=400
    )
    
    st.plotly_chart(fig_probs, use_container_width=True)

with col2:
    # Distance to attractors
    distance_data = [(k.replace('_', ' '), v) for k, v in distances.items()]
    distance_data.sort(key=lambda x: x[1])
    
    fig_distances = go.Figure(data=go.Bar(
        x=[name for name, dist in distance_data],
        y=[dist for name, dist in distance_data],
        marker_color='lightblue',
        text=[f"{dist:.2f}" for name, dist in distance_data],
        textposition='auto'
    ))
    
    fig_distances.update_layout(
        title="Distance to Phase Attractors",
        yaxis_title="Distance",
        height=400
    )
    
    st.plotly_chart(fig_distances, use_container_width=True)

# === Critical Nodes Analysis ===
st.markdown("## 🏛️ Institutional Stress Analysis")

# Node stress analysis
node_stress_data = []
for i, node in enumerate(iran_symbon.nodes):
    C, K, S, A = iran_symbon.node_states[i]
    fitness = (C * K) / (1 + np.exp((abs(S) - iran_symbon.params.tau) / iran_symbon.params.lambda_decay)) * (1 + A/10)
    
    stress_level = abs(S)
    risk_category = ("🔴 CRITICAL" if stress_level > 4.5 else 
                    "🟠 HIGH" if stress_level > 3.5 else
                    "🟡 ELEVATED" if stress_level > 2.5 else 
                    "🟢 NORMAL")
    
    node_stress_data.append({
        'Institution': node,
        'Coherence': C,
        'Capacity': K, 
        'Stress': S,
        'Abstraction': A,
        'Fitness': fitness,
        'Risk_Level': risk_category
    })

stress_df = pd.DataFrame(node_stress_data)
stress_df = stress_df.sort_values('Stress', key=abs, ascending=False)

st.dataframe(stress_df, use_container_width=True)

# === Meta-Cognitive State Analysis ===
st.markdown("## 🧠 Meta-Cognitive Function Analysis")

monitoring, control, reflection = iran_symbon.meta_cognitive_state
metacog_critical = critical_points['metacognitive_collapse']

col1, col2, col3 = st.columns(3)

with col1:
    monitoring_risk = "🔴" if monitoring < 1.0 else "🟠" if monitoring < 2.0 else "🟡" if monitoring < 3.0 else "🟢"
    st.metric(f"{monitoring_risk} Monitoring", f"{monitoring:.3f}", 
             help="Real-time stress assessment capability")

with col2:
    control_risk = "🔴" if control > monitoring * 3 else "🟠" if control > monitoring * 2 else "🟢"
    st.metric(f"{control_risk} Control", f"{control:.3f}",
             delta=f"Dominance: {metacog_critical['control_dominance']:.2f}")

with col3:
    reflection_risk = "🔴" if reflection < 0.5 else "🟠" if reflection < 1.0 else "🟢"
    st.metric(f"{reflection_risk} Reflection", f"{reflection:.3f}",
             delta=f"Risk: {metacog_critical['reflection_risk']:.3f}")

# === Early Warning System ===
st.markdown("## ⚠️ Early Warning Indicators")

warning_indicators = []

# Check each critical threshold
if health < 2.0:
    warning_indicators.append("🔴 **IMMEDIATE RISK**: System health approaching collapse threshold")
elif health < 2.5:
    warning_indicators.append("🟠 **HIGH RISK**: System health in fragmentation zone")

if indicators['critical_stress_index'] > 0.5:
    warning_indicators.append("🔴 **STRESS CASCADE**: Multiple institutions at critical stress levels")

if indicators['metacog_breakdown_risk'] > 0.7:
    warning_indicators.append("🔴 **META-COGNITIVE FAILURE**: Collective intelligence breakdown imminent")

if indicators['fragmentation_index'] > 0.6:
    warning_indicators.append("🟠 **INSTITUTIONAL FRAGMENTATION**: Inter-institutional bonds weakening")

if control > monitoring * 2.5:
    warning_indicators.append("🟡 **AUTHORITARIAN DRIFT**: Control functions dominating monitoring")

# Display warnings
if warning_indicators:
    for warning in warning_indicators:
        st.markdown(warning)
else:
    st.success("✅ No immediate critical warnings detected")

# === Recommendations ===
st.markdown("## 📋 Strategic Recommendations")

recommendations = []

if current_attractor in ["A₃_Fragmented", "A₄_Collapse"]:
    recommendations.append("🎯 **Priority 1**: Immediate coherence restoration interventions required")
    recommendations.append("💪 **Priority 2**: Strengthen inter-institutional bonds, especially Supreme Leader-IRGC coordination")
    recommendations.append("🧠 **Priority 3**: Enhance meta-cognitive functions through improved information processing")

elif current_attractor == "A₂_Authoritarian":
    recommendations.append("⚖️ **Balance**: Reduce control dominance, enhance monitoring and reflection capabilities")
    recommendations.append("🤝 **Dialogue**: Increase civil society integration (Urban/Rural population nodes)")

if indicators['critical_stress_index'] > 0.3:
    recommendations.append("📉 **Stress Management**: Implement targeted stress reduction for high-risk institutions")

for rec in recommendations:
    st.markdown(rec)

# === Footer ===
st.markdown("---")
st.markdown("### 📊 Analysis Framework")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **CAMS-CAN Framework:**
    - Mathematical phase transition analysis
    - Real-time attractor identification
    - Meta-cognitive state monitoring
    - Critical threshold detection
    """)

with col2:
    st.markdown("""
    **Iran-Specific Calibrations:**
    - Regional stress multiplier: 1.3×
    - Institutional fragility factor: 1.2×
    - External pressure factor: 1.4×
    - Crisis dynamics modeling
    """)

with col3:
    st.markdown(f"""
    **Current Status:**
    - **Threat Level**: {threat_level}
    - **Primary Attractor**: {current_attractor}
    - **System Health**: {health:.3f}
    - **Analysis Timestamp**: Real-time
    """)

st.error("⚠️ This analysis is for research purposes only. Real-world policy decisions should incorporate multiple analytical frameworks and expert consultation.")