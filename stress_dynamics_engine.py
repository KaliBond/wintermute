"""
🧠 CAMS-CAN Stress Dynamics Engine
Mathematical Framework: Stress as Societal Meta-Cognition

Implementation of the complete mathematical formalization for stress as distributed 
meta-cognitive function within complex adaptive social systems.

Version: 2.1.0
Classification: Open Research
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. THEORETICAL FOUNDATIONS
# ================================

@dataclass
class CAMSCANParameters:
    """Empirically validated parameter set from historical analysis"""
    # Stress tolerance and resilience parameters
    tau: float = 3.0  # ± 0.2 stress tolerance threshold
    lambda_decay: float = 0.5  # ± 0.1 resilience decay factor
    xi: float = 0.2  # ± 0.05 coherence coupling strength
    gamma_c: float = 0.15  # ± 0.03 coherence decay rate
    delta_s: float = 0.2  # ± 0.04 stress dissipation rate
    
    # Node evolution parameters
    alpha_k: float = 0.3  # Capacity growth rate
    beta_k: float = 0.1  # Capacity stress degradation
    kappa_adapt: float = 0.25  # Learning adaptation rate
    eta_coh: float = 0.15  # Network coherence coupling
    eta_a: float = 0.2  # Abstraction development rate
    mu_a: float = 0.1  # Abstraction decay rate
    rho_symbolic: float = 0.05  # External symbol influence
    
    # Bond and network parameters
    sigma_bond: float = 1.0  # Bond strength decay parameter
    diffusion_coeff: float = 0.1  # Inter-institutional diffusion

class SymbonSystem:
    """
    Societal Symbon: Meta-organism implementation
    S = (N, E, Φ, Ψ, Θ, Ω, T, M)
    """
    
    def __init__(self, params: CAMSCANParameters = None):
        self.params = params or CAMSCANParameters()
        
        # Node definitions (8 institutional nodes)
        self.nodes = {
            0: "Executive",
            1: "Army", 
            2: "StateMemory",
            3: "Priesthood",
            4: "Stewards",
            5: "Craft",
            6: "Flow",
            7: "Hands"
        }
        
        # Initialize node states [C, K, S, A]
        self.node_states = np.zeros((8, 4))
        
        # Initialize bond strength matrix
        self.bond_matrix = np.eye(8)
        
        # Meta-cognitive function vector [Monitoring, Control, Reflection]
        self.meta_cognitive_state = np.zeros(3)
        
        # Time series storage
        self.history = {'t': [], 'states': [], 'meta_cog': [], 'health': [], 'spe': []}
        
    def initialize_from_data(self, cams_data: pd.DataFrame):
        """Initialize system state from CAMS dataset"""
        try:
            # Extract latest state for each node
            for i, node_name in self.nodes.items():
                # Try to find matching node in data
                node_data = None
                for _, row in cams_data.iterrows():
                    node_str = str(row.get('Node', '')).strip().lower()
                    if (node_name.lower() in node_str or 
                        node_str in node_name.lower() or
                        self._match_node_alias(node_name, node_str)):
                        node_data = row
                        break
                
                if node_data is not None:
                    self.node_states[i, 0] = float(pd.to_numeric(node_data.get('Coherence', 5.0), errors='coerce') or 5.0)
                    self.node_states[i, 1] = float(pd.to_numeric(node_data.get('Capacity', 5.0), errors='coerce') or 5.0)
                    self.node_states[i, 2] = float(pd.to_numeric(node_data.get('Stress', 5.0), errors='coerce') or 5.0)
                    self.node_states[i, 3] = float(pd.to_numeric(node_data.get('Abstraction', 5.0), errors='coerce') or 5.0)
                else:
                    # Default values if node not found
                    self.node_states[i] = [5.0, 5.0, 5.0, 5.0]
            
            # Calculate initial bond strengths
            self._update_bond_matrix()
            
            # Calculate initial meta-cognitive state
            self._update_meta_cognitive_state()
            
        except Exception as e:
            st.warning(f"Error initializing from data: {e}. Using default values.")
            # Set reasonable defaults
            self.node_states = np.random.uniform(3, 7, (8, 4))
            self._update_bond_matrix()
            self._update_meta_cognitive_state()
    
    def _match_node_alias(self, canonical_name: str, data_name: str) -> bool:
        """Match node names with common aliases"""
        aliases = {
            'executive': ['government', 'admin', 'leadership'],
            'army': ['military', 'defense', 'security'],
            'statememory': ['archive', 'records', 'memory', 'state memory'],
            'priesthood': ['priests', 'religious', 'clergy', 'lore'],
            'stewards': ['bureaucracy', 'civil service', 'administration'],
            'craft': ['artisans', 'trades', 'professions', 'guilds'],
            'flow': ['merchants', 'trade', 'commerce', 'shopkeepers'],
            'hands': ['workers', 'labor', 'proletariat', 'peasants']
        }
        
        canonical_lower = canonical_name.lower()
        if canonical_lower in aliases:
            return any(alias in data_name for alias in aliases[canonical_lower])
        return False

# ================================
# 2. CORE MATHEMATICAL STRUCTURES  
# ================================

    def stress_information_isomorphism(self, t: float) -> np.ndarray:
        """
        Theorem 2.1: Stress-Information Duality
        I: Stress(S,t) ↔ Information(S,t)
        """
        alpha, beta, gamma = 0.1, 0.05, 0.02
        
        stress_field = self.node_states[:, 2]  # S(t) vector
        coherence_field = self.node_states[:, 0]  # C(t) vector
        
        # Compute stress gradients (discrete approximation)
        grad_S = np.gradient(stress_field)
        grad_C = np.gradient(coherence_field)
        laplace_S = np.gradient(grad_S)
        
        # Information density evolution: ∂I/∂t = α∇²S + β(∇S·∇C) + γS
        information_density = (alpha * laplace_S + 
                             beta * grad_S * grad_C + 
                             gamma * stress_field)
        
        return information_density
    
    def _update_meta_cognitive_state(self):
        """Calculate meta-cognitive function vector M(S,t) = [Monitoring, Control, Reflection]"""
        
        # Extract current states
        C = self.node_states[:, 0]  # Coherence
        S = self.node_states[:, 2]  # Stress 
        A = self.node_states[:, 3]  # Abstraction
        
        # Bond strengths for weighting
        BS = np.diag(self.bond_matrix)
        
        # Definition 2.1: Monitoring Function
        # Real-time stress assessment capability
        stress_sum = np.sum(S)
        if stress_sum > 0:
            monitoring = np.sum(BS * S) / stress_sum
        else:
            monitoring = 0.5  # Default when no stress
        
        # Definition 2.2: Control Function  
        # Coordination of institutional responses (coherence synchronization)
        coherence_variations = []
        for i in range(8):
            for j in range(i+1, 8):
                if self.bond_matrix[i,j] > 0:
                    coherence_variations.append(self.bond_matrix[i,j] * abs(C[i] - C[j]))
        control = np.mean(coherence_variations) if coherence_variations else 0.0
        
        # Definition 2.3: Reflection Function
        # Historical pattern integration (StateMemory + Priesthood)
        state_memory_idx = 2  # StateMemory node
        priesthood_idx = 3    # Priesthood node
        reflection = (C[state_memory_idx] * A[state_memory_idx] + 
                     C[priesthood_idx] * A[priesthood_idx]) / 2
        
        self.meta_cognitive_state = np.array([monitoring, control, reflection])
        
    def stress_processing_efficiency(self) -> float:
        """
        Definition 2.4: Collective Processing Efficiency
        SPE(t) = Σᵢ(Kᵢ×BSᵢ) / Σᵢ(S_scaled,i×Aᵢ)
        """
        K = self.node_states[:, 1]  # Capacity
        S = np.abs(self.node_states[:, 2])  # Absolute stress
        A = self.node_states[:, 3]  # Abstraction
        BS = np.diag(self.bond_matrix)  # Bond strengths
        
        numerator = np.sum(K * BS)
        
        # Scale stress to prevent division issues
        S_scaled = np.maximum(S, 0.1)  # Minimum stress floor
        denominator = np.sum(S_scaled * A)
        
        if denominator > 0:
            spe = numerator / denominator
        else:
            spe = 1.0  # Default efficiency
            
        return spe
    
    def _update_bond_matrix(self):
        """Calculate inter-institutional bond strengths using stress coupling"""
        C = self.node_states[:, 0]  # Coherence
        
        # Intrinsic stress-processing capacity (based on capacity and coherence)
        K = self.node_states[:, 1]  # Capacity
        beta = np.sqrt(K * C)  # βᵢ = √(Kᵢ × Cᵢ)
        
        # Update bond matrix: B(i,j,t) = √(βᵢ×βⱼ) × exp(-|Cᵢ-Cⱼ|/σ) × Coupling
        for i in range(8):
            for j in range(8):
                if i != j:
                    coherence_diff = abs(C[i] - C[j])
                    bond_strength = (np.sqrt(beta[i] * beta[j]) * 
                                   np.exp(-coherence_diff / self.params.sigma_bond))
                    
                    # Add stress coupling based on node type relationships
                    stress_coupling = self._get_stress_coupling(i, j)
                    self.bond_matrix[i, j] = bond_strength * stress_coupling
                else:
                    self.bond_matrix[i, j] = 1.0  # Self-bond
    
    def _get_stress_coupling(self, i: int, j: int) -> float:
        """Define stress coupling relationships between institutional nodes"""
        # Define institutional coupling matrix based on CAMS theory
        coupling_matrix = np.array([
            #Ex  Ar  SM  Pr  St  Cr  Fl  Ha
            [1.0, 0.8, 0.9, 0.6, 0.9, 0.5, 0.6, 0.4], # Executive
            [0.8, 1.0, 0.7, 0.4, 0.6, 0.3, 0.4, 0.5], # Army  
            [0.9, 0.7, 1.0, 0.8, 0.8, 0.6, 0.5, 0.3], # StateMemory
            [0.6, 0.4, 0.8, 1.0, 0.5, 0.7, 0.4, 0.6], # Priesthood
            [0.9, 0.6, 0.8, 0.5, 1.0, 0.7, 0.8, 0.7], # Stewards
            [0.5, 0.3, 0.6, 0.7, 0.7, 1.0, 0.9, 0.8], # Craft
            [0.6, 0.4, 0.5, 0.4, 0.8, 0.9, 1.0, 0.7], # Flow
            [0.4, 0.5, 0.3, 0.6, 0.7, 0.8, 0.7, 1.0]  # Hands
        ])
        
        return coupling_matrix[i, j]

# ================================
# 3. DYNAMIC EVOLUTION EQUATIONS
# ================================

    def node_evolution_equations(self, t: float, state_vector: np.ndarray, 
                               external_stress: np.ndarray = None) -> np.ndarray:
        """
        System of differential equations for node state evolution
        Returns: d/dt[C₁,K₁,S₁,A₁,...,C₈,K₈,S₈,A₈]
        """
        # Reshape state vector to 8x4 matrix
        states = state_vector.reshape(8, 4)
        C, K, S, A = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        
        # External stress inputs (default to small random fluctuations)
        if external_stress is None:
            external_stress = 0.1 * np.random.normal(0, 1, 8)
        
        # Calculate network coherence field
        phi_network = self._calculate_network_coherence_field(states)
        
        # Calculate learning rates
        learning_rates = self._calculate_learning_rates(states)
        
        # Initialize derivatives
        dC_dt = np.zeros(8)
        dK_dt = np.zeros(8)  
        dS_dt = np.zeros(8)
        dA_dt = np.zeros(8)
        
        # Evolution equations for each node
        for i in range(8):
            # Coherence Evolution: dCᵢ/dt = ξᵢ×Φ_network - γc,i×Cᵢ×|Sᵢ| + network_coupling
            network_coupling = sum(self.bond_matrix[i,j] * (C[j] - C[i]) 
                                 for j in range(8) if j != i)
            
            dC_dt[i] = (self.params.xi * phi_network - 
                       self.params.gamma_c * C[i] * abs(S[i]) +
                       self.params.eta_coh * network_coupling)
            
            # Capacity Evolution: dKᵢ/dt = αₖ×Cᵢ - βₖ×Kᵢ×S²ᵢ + κ_adapt×Learning
            dK_dt[i] = (self.params.alpha_k * C[i] - 
                       self.params.beta_k * K[i] * S[i]**2 +
                       self.params.kappa_adapt * learning_rates[i])
            
            # Stress Evolution: dSᵢ/dt = εₑₓₜ + stress_transfer - dissipation - processing
            stress_transfer = sum(self._stress_transfer_function(i, j) * S[j] 
                                for j in range(8) if j != i)
            processing_rate = self._stress_processing_rate(i, states)
            
            dS_dt[i] = (external_stress[i] + stress_transfer - 
                       self.params.delta_s * S[i] - processing_rate)
            
            # Abstraction Evolution: dAᵢ/dt = ηₐ×Kᵢ×Cᵢ - μₐ×Aᵢ + external_symbols
            external_symbols = self.params.rho_symbolic * np.sin(t * 0.1 + i)  # Symbolic environment
            
            dA_dt[i] = (self.params.eta_a * K[i] * C[i] - 
                       self.params.mu_a * A[i] + external_symbols)
        
        # Flatten back to vector form
        derivatives = np.column_stack([dC_dt, dK_dt, dS_dt, dA_dt]).flatten()
        return derivatives
    
    def _calculate_network_coherence_field(self, states: np.ndarray) -> float:
        """Calculate global coherence field Φ_network(t)"""
        C = states[:, 0]
        S = states[:, 2]
        
        total_coherence = 0.0
        total_weight = 0.0
        
        for i in range(8):
            for j in range(8):
                if i != j and self.bond_matrix[i,j] > 0:
                    # Stress alignment factor
                    stress_alignment = np.exp(-abs(S[i] - S[j]) / 2.0)
                    # Phase alignment (simplified as coherence similarity)
                    theta_ij = abs(C[i] - C[j]) / 10.0  # Normalized phase difference
                    cos_theta = np.cos(theta_ij)
                    
                    weight = self.bond_matrix[i,j] * cos_theta * stress_alignment
                    total_coherence += weight * (C[i] + C[j]) / 2
                    total_weight += weight
        
        if total_weight > 0:
            phi_network = total_coherence / total_weight
        else:
            phi_network = np.mean(C)  # Fallback to average coherence
            
        return phi_network
    
    def _calculate_learning_rates(self, states: np.ndarray) -> np.ndarray:
        """Calculate individual node learning rates based on stress exposure"""
        C, K, S, A = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        
        # Learning is optimal at moderate stress levels (inverted U-curve)
        optimal_stress = 3.0
        learning_rates = K * np.exp(-((S - optimal_stress)**2) / 2.0)
        
        return learning_rates
    
    def _stress_transfer_function(self, i: int, j: int) -> float:
        """Stress transfer function Θ(i,j) between nodes"""
        # Transfer rate depends on bond strength and stress gradient
        transfer_rate = 0.1 * self.bond_matrix[i,j]
        return transfer_rate
    
    def _stress_processing_rate(self, i: int, states: np.ndarray) -> float:
        """Calculate stress processing rate for node i"""
        C, K, S, A = states[i, 0], states[i, 1], states[i, 2], states[i, 3]
        
        # Processing rate depends on capacity and coherence, modulated by abstraction
        processing_efficiency = C * K * (1 + 0.1 * A)
        processing_rate = processing_efficiency * abs(S) / (abs(S) + 1.0)  # Saturation
        
        return processing_rate

# ================================
# 4. PHASE SPACE ANALYSIS
# ================================

    def system_health(self) -> float:
        """Calculate system health Ω(S,t)"""
        C = self.node_states[:, 0]
        K = self.node_states[:, 1] 
        S = self.node_states[:, 2]
        
        # Weighted sum based on node importance and stress processing capability
        node_weights = np.array([0.15, 0.12, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12])
        node_health = C * K / (1 + np.abs(S))  # Health decreases with stress
        
        health = np.sum(node_weights * node_health)
        return health
    
    def identify_phase_attractor(self) -> Tuple[str, Dict[str, float]]:
        """Identify current phase space attractor"""
        health = self.system_health()
        spe = self.stress_processing_efficiency()
        monitoring, control, reflection = self.meta_cognitive_state
        
        # Calculate coherence asymmetry
        C = self.node_states[:, 0]
        coherence_asymmetry = np.std(C) / (np.mean(C) + 0.01)
        
        metrics = {
            'health': health,
            'spe': spe,  
            'coherence_asymmetry': coherence_asymmetry,
            'monitoring': monitoring,
            'control': control,
            'reflection': reflection
        }
        
        # Attractor classification based on empirical boundaries
        if health > 3.5 and spe > 2.0 and coherence_asymmetry < 0.3:
            return "Adaptive", metrics
        elif health >= 2.5 and health <= 3.5 and control > monitoring:
            return "Authoritarian", metrics
        elif health >= 1.5 and health <= 2.5 and coherence_asymmetry > 0.4 and spe < 1.0:
            return "Fragmented", metrics
        elif health < 1.5:
            return "Collapse", metrics
        else:
            return "Transitional", metrics
    
    def detect_phase_transition(self, history_length: int = 10) -> bool:
        """Detect if system is approaching phase transition"""
        if len(self.history['health']) < history_length:
            return False
        
        recent_health = self.history['health'][-history_length:]
        
        # Check for critical slowing down (increased variance and autocorrelation)
        variance_increase = np.var(recent_health[-5:]) > 2 * np.var(recent_health[:5])
        
        # Simple autocorrelation check
        if len(recent_health) > 3:
            autocorr = np.corrcoef(recent_health[:-1], recent_health[1:])[0,1]
            high_autocorr = autocorr > 0.8
        else:
            high_autocorr = False
            
        return variance_increase and high_autocorr

# ================================
# 5. SIMULATION AND ANALYSIS
# ================================

    def simulate_evolution(self, time_span: Tuple[float, float], 
                          external_stress_function: Callable = None,
                          dt: float = 0.1) -> Dict:
        """Run complete system evolution simulation"""
        t_start, t_end = time_span
        t_eval = np.arange(t_start, t_end, dt)
        
        # Initial state vector (flatten 8x4 matrix)
        initial_state = self.node_states.flatten()
        
        # External stress function (default to small random perturbations)
        if external_stress_function is None:
            external_stress_function = lambda t: 0.1 * np.random.normal(0, 1, 8)
        
        # Solve ODE system
        try:
            sol = solve_ivp(
                fun=lambda t, y: self.node_evolution_equations(t, y, external_stress_function(t)),
                t_span=(t_start, t_end),
                y0=initial_state,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            # Process results
            results = {
                'time': sol.t,
                'states': sol.y.reshape(32, -1),  # 32D state space
                'health': [],
                'spe': [],
                'meta_cognitive': [],
                'attractors': []
            }
            
            # Calculate derived quantities for each time point
            for i, t in enumerate(sol.t):
                # Update system state
                self.node_states = sol.y[:, i].reshape(8, 4)
                self._update_bond_matrix()
                self._update_meta_cognitive_state()
                
                # Record metrics
                results['health'].append(self.system_health())
                results['spe'].append(self.stress_processing_efficiency())
                results['meta_cognitive'].append(self.meta_cognitive_state.copy())
                
                attractor, _ = self.identify_phase_attractor()
                results['attractors'].append(attractor)
            
            return results
            
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return {'error': str(e)}
    
    def run_stress_shock_analysis(self, shock_magnitude: float = 2.0, 
                                 shock_duration: float = 5.0) -> Dict:
        """Analyze system response to stress shock"""
        
        def stress_shock(t):
            """External stress function with shock"""
            base_stress = 0.1 * np.random.normal(0, 1, 8)
            
            if 10.0 <= t <= (10.0 + shock_duration):
                # Apply shock primarily to critical nodes (Executive, Army, Stewards)
                shock_stress = np.zeros(8)
                shock_stress[0] = shock_magnitude  # Executive
                shock_stress[1] = shock_magnitude * 0.8  # Army
                shock_stress[4] = shock_magnitude * 0.6  # Stewards
                return base_stress + shock_stress
            else:
                return base_stress
        
        # Run simulation with shock
        results = self.simulate_evolution(
            time_span=(0, 30),  # 30 time units
            external_stress_function=stress_shock,
            dt=0.1
        )
        
        return results

# ================================
# 6. VISUALIZATION SYSTEM
# ================================

def create_stress_dynamics_dashboard(symbon: SymbonSystem, simulation_results: Dict = None):
    """Create comprehensive stress dynamics visualization dashboard"""
    
    st.markdown("## 🧠 CAMS-CAN Stress Dynamics Analysis")
    st.markdown("*Mathematical Framework: Stress as Societal Meta-Cognition*")
    
    # System state overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health = symbon.system_health()
        st.metric("System Health (Ω)", f"{health:.2f}", 
                 help="Overall system health based on stress-cognition coupling")
    
    with col2:
        spe = symbon.stress_processing_efficiency()
        st.metric("Processing Efficiency", f"{spe:.2f}", 
                 help="Collective stress processing capability")
    
    with col3:
        monitoring, control, reflection = symbon.meta_cognitive_state
        st.metric("Meta-Cognitive State", f"M:[{monitoring:.1f},{control:.1f},{reflection:.1f}]", 
                 help="[Monitoring, Control, Reflection] functions")
    
    with col4:
        attractor, metrics = symbon.identify_phase_attractor()
        attractor_colors = {"Adaptive": "🟢", "Authoritarian": "🟡", 
                          "Fragmented": "🟠", "Collapse": "🔴", "Transitional": "🔵"}
        st.metric("Phase Attractor", f"{attractor_colors.get(attractor, '⚪')} {attractor}",
                 help="Current phase space attractor classification")
    
    # Node state matrix visualization
    st.markdown("### 📊 Node State Matrix [C, K, S, A]")
    
    # Create node state heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=symbon.node_states,
        x=['Coherence', 'Capacity', 'Stress', 'Abstraction'],
        y=[symbon.nodes[i] for i in range(8)],
        colorscale='RdBu_r',
        colorbar=dict(title="State Value")
    ))
    
    fig_heatmap.update_layout(
        title="Current System State Matrix",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Bond strength network
    st.markdown("### 🕸️ Inter-Institutional Bond Network")
    
    # Create network visualization
    G = nx.Graph()
    
    # Add nodes
    for i, node_name in symbon.nodes.items():
        G.add_node(i, name=node_name, 
                  coherence=symbon.node_states[i, 0],
                  stress=symbon.node_states[i, 2])
    
    # Add edges (bonds above threshold)
    threshold = 0.3
    for i in range(8):
        for j in range(i+1, 8):
            if symbon.bond_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=symbon.bond_matrix[i, j])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create plotly network
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                           line=dict(width=1, color='#888'), hoverinfo='none')
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [symbon.nodes[node] for node in G.nodes()]
    node_stress = [symbon.node_states[node, 2] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="middle center",
        marker=dict(size=20, color=node_stress, colorscale='Reds',
                   colorbar=dict(title="Stress Level")),
        hovertemplate='<b>%{text}</b><br>Stress: %{marker.color:.2f}<extra></extra>'
    )
    
    fig_network = go.Figure(data=[edge_trace, node_trace])
    fig_network.update_layout(
        title="Inter-Institutional Bond Network (Bond Strength > 0.3)",
        showlegend=False, hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    st.plotly_chart(fig_network, use_container_width=True)
    
    # Meta-cognitive radar chart
    st.markdown("### 🎯 Meta-Cognitive Function Analysis")
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=[monitoring, control, reflection],
        theta=['Monitoring', 'Control', 'Reflection'],
        fill='toself',
        name='Current State'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10])
        ),
        title="Meta-Cognitive Function Vector M(S,t)",
        height=400
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Simulation results visualization
    if simulation_results and 'error' not in simulation_results:
        st.markdown("### 📈 Temporal Evolution Analysis")
        
        # Time series plots
        fig_evolution = make_subplots(
            rows=2, cols=2,
            subplot_titles=('System Health Ω(t)', 'Processing Efficiency SPE(t)',
                          'Meta-Cognitive Functions', 'Phase Attractors')
        )
        
        time = simulation_results['time']
        
        # System health
        fig_evolution.add_trace(
            go.Scatter(x=time, y=simulation_results['health'], name='Health'),
            row=1, col=1
        )
        
        # Processing efficiency
        fig_evolution.add_trace(
            go.Scatter(x=time, y=simulation_results['spe'], name='SPE'),
            row=1, col=2
        )
        
        # Meta-cognitive functions
        meta_cog = np.array(simulation_results['meta_cognitive'])
        fig_evolution.add_trace(
            go.Scatter(x=time, y=meta_cog[:, 0], name='Monitoring'),
            row=2, col=1
        )
        fig_evolution.add_trace(
            go.Scatter(x=time, y=meta_cog[:, 1], name='Control'),
            row=2, col=1
        )
        fig_evolution.add_trace(
            go.Scatter(x=time, y=meta_cog[:, 2], name='Reflection'),
            row=2, col=1
        )
        
        # Phase attractor timeline
        attractor_numeric = []
        attractor_map = {"Adaptive": 4, "Authoritarian": 3, "Fragmented": 2, 
                        "Collapse": 1, "Transitional": 2.5}
        
        for att in simulation_results['attractors']:
            attractor_numeric.append(attractor_map.get(att, 0))
        
        fig_evolution.add_trace(
            go.Scatter(x=time, y=attractor_numeric, mode='lines+markers', name='Attractor'),
            row=2, col=2
        )
        
        fig_evolution.update_layout(height=600, title="System Evolution Analysis")
        st.plotly_chart(fig_evolution, use_container_width=True)
    
    return symbon

def create_stress_shock_analysis_dashboard(shock_results: Dict):
    """Visualize stress shock analysis results"""
    if 'error' in shock_results:
        st.error(f"Shock analysis failed: {shock_results['error']}")
        return
    
    st.markdown("### ⚡ Stress Shock Response Analysis")
    
    time = shock_results['time']
    health = shock_results['health']
    spe = shock_results['spe']
    
    # Identify shock period
    shock_start, shock_end = 10.0, 15.0
    
    fig_shock = make_subplots(
        rows=2, cols=1,
        subplot_titles=('System Health Response to Stress Shock',
                       'Stress Processing Efficiency Response')
    )
    
    # Add shock region
    fig_shock.add_vrect(
        x0=shock_start, x1=shock_end, 
        fillcolor="red", opacity=0.2,
        annotation_text="Stress Shock", annotation_position="top left",
        row=1, col=1
    )
    fig_shock.add_vrect(
        x0=shock_start, x1=shock_end, 
        fillcolor="red", opacity=0.2,
        row=2, col=1
    )
    
    # Health response
    fig_shock.add_trace(
        go.Scatter(x=time, y=health, name='System Health', line=dict(color='blue')),
        row=1, col=1
    )
    
    # SPE response  
    fig_shock.add_trace(
        go.Scatter(x=time, y=spe, name='Processing Efficiency', line=dict(color='green')),
        row=2, col=1
    )
    
    fig_shock.update_layout(height=600, title="Stress Shock Analysis")
    fig_shock.update_xaxes(title_text="Time")
    fig_shock.update_yaxes(title_text="System Health", row=1, col=1)
    fig_shock.update_yaxes(title_text="Processing Efficiency", row=2, col=1)
    
    st.plotly_chart(fig_shock, use_container_width=True)
    
    # Recovery analysis
    pre_shock_health = np.mean([h for t, h in zip(time, health) if t < shock_start])
    min_shock_health = min([h for t, h in zip(time, health) if shock_start <= t <= shock_end])
    post_shock_health = np.mean([h for t, h in zip(time, health) if t > shock_end + 5])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pre-Shock Health", f"{pre_shock_health:.2f}")
    
    with col2:
        health_drop = pre_shock_health - min_shock_health
        st.metric("Max Health Drop", f"{health_drop:.2f}", f"-{health_drop:.2f}")
    
    with col3:
        recovery = post_shock_health / pre_shock_health if pre_shock_health > 0 else 0
        recovery_pct = (recovery - 1) * 100
        st.metric("Recovery Rate", f"{recovery:.2f}", f"{recovery_pct:+.1f}%")