"""
Unified CAMS-CAN + 13 Laws Mathematical Framework
Complete thermodynamic implementation with inter-institutional bond dynamics
Simulation-ready continuous/discrete time system

NOTE: The neural network hypothesis has been falsified (December 2025).
CAMS now focuses on thermodynamic principles, entropy flows, and phase transitions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import networkx as nx
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Unified CAMS-CAN + 13 Laws Framework", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🌡️⚖️ Unified CAMS-CAN + 13 Laws Framework")
st.markdown("**Complete thermodynamic implementation with inter-institutional bond dynamics**")
st.markdown("*Simulation-ready continuous/discrete time system*")
st.info("⚠️ **Scientific Update (Dec 2025)**: The neural network hypothesis has been falsified. CAMS now focuses on thermodynamic principles, entropy flows, and phase transitions.")

# === Core Framework Implementation ===

class UnifiedCAMS13Laws:
    """
    Complete implementation of the unified mathematical framework
    """
    
    def __init__(self, n_nodes=8):
        self.n = n_nodes
        
        # Node labels (institutional types)
        self.node_labels = [
            "Executive", "Army", "StateMemory", "Priesthood", 
            "Stewards", "Craft", "Flow", "Hands"
        ]
        
        # === Core State Variables ===
        # Node states: N_i(t) = (χ_i, κ_i, σ_i, α_i)
        self.chi = np.zeros(self.n)      # Coherence χ_i ∈ [0,1]
        self.kappa = np.zeros(self.n)    # Capacity κ_i ≥ 0
        self.sigma = np.zeros(self.n)    # Stress σ_i ≥ 0
        self.alpha = np.zeros(self.n)    # Abstraction α_i ∈ [0,1]
        
        # Stress decomposition: σ_i = σ_i^ch + σ_i^ac
        self.sigma_chronic = np.zeros(self.n)  # Chronic stress
        self.sigma_acute = np.zeros(self.n)    # Acute stress
        
        # === Inter-Institutional Bond Components ===
        self.w = np.random.normal(0, 0.1, (self.n, self.n))  # Bond weights w_ij
        self.B = np.zeros((self.n, self.n))                  # Macro bonds B_ij
        self.theta_0 = np.random.normal(0.5, 0.1, self.n)    # Base thresholds θ_i^0
        self.activation = np.zeros(self.n)                    # Node activations a_i(t)
        
        # === Memory and Integration ===
        self.M = np.zeros(self.n)        # Memory M_i(t) = ∫ α_i(τ)χ_i(τ) dτ
        self.iota = 0.0                  # Innovation ι(t)
        
        # === System-Level Metrics ===
        self.H = 0.0                     # System Health H(t)
        self.L = 0.0                     # Legitimacy L(t)
        self.Psi = 0.0                   # Grand System Metric Ψ(t)
        self.CA = 0.0                    # Coherence Asymmetry CA(t)
        self.E_chi = 0.0                 # Coherence Entropy E_χ(t)
        self.I_info = 0.0                # Information Integration I(t)
        self.P_path = 1.0                # Path Dependence P(t)
        
        # === Parameters ===
        self.setup_parameters()
        
        # === History Tracking ===
        self.history = {
            'time': [],
            'chi': [], 'kappa': [], 'sigma': [], 'alpha': [],
            'H': [], 'L': [], 'Psi': [], 'CA': [], 'E_chi': [],
            'activation': [], 'B': [], 'M': [], 'iota': []
        }
    
    def setup_parameters(self):
        """Initialize all model parameters"""

        # === Thermodynamic Bond Parameters ===
        self.beta = np.random.uniform(0.1, 0.5, self.n)      # Stress modulation β_i
        self.alpha_acute = 3.0                                # Acute stress multiplier
        self.eta_sigma = 0.3                                  # Stress threshold impact
        self.eta_chi = 0.2                                    # Coherence threshold impact
        
        # === State Dynamics Parameters ===
        # Capacity dynamics
        self.r_kappa = 0.2                  # Capacity growth rate
        self.phi_kappa = 0.15               # Capacity diffusion
        self.xi_kappa = 0.1                 # Stress drag on capacity
        self.zeta_kappa = 0.05              # Coordination load
        
        # Coherence dynamics (Coherence Decay Law)
        self.lambda_chi = 0.1               # Coherence decay λ_χ
        self.mu_chi = 0.3                   # Maintenance effectiveness
        self.rho_chi = 0.2                  # Stress impact on coherence
        
        # Stress dynamics (Law 4 + CAMS)
        self.lambda_sigma = 0.15            # Stress dissipation λ_σ
        self.rho_sigma = 0.25               # Stress propagation ρ_σ
        self.upsilon_sigma = 0.1            # Memory load on stress
        self.omega_sigma = 0.3              # Activation stress reduction
        
        # Abstraction dynamics (Law 5 + Abstraction Paradox)
        self.gamma_alpha = 0.2              # Memory to abstraction γ_α
        self.delta_alpha = 0.15             # Stress penalty on abstraction
        
        # === 13 Laws Constraint Parameters ===
        self.gamma_coord = 1.2              # Coordination requirement (Law 11)
        self.S_crit = 0.5                   # Critical synchronization threshold
        self.V_crit = 2.0                   # Critical stress variance
        self.eta_I = 0.3                    # Information integration factor
        
        # Innovation parameters (Law 13)
        self.rho_iota = 0.1                 # Innovation growth
        self.delta_iota = 0.2               # Stress penalty on innovation
        
        # Elite adaptation (Law 10)
        self.rho_e = 0.1                    # Elite stress penalty
        
        # Entropy dynamics (Law 2)
        self.lambda_E = 0.05                # Entropy growth rate
        
        # Path dependence (CAMS)
        self.alpha_P = 0.02                 # Path dependence growth
        
        # === Health and Legitimacy Parameters ===
        self.w_health = np.ones(self.n) / self.n  # Node weights in health
        self.beta_L = np.array([0.4, 0.3, 0.3])   # Legitimacy components [H, I, χ̄]
        self.omega_Psi = np.array([0.3, 0.2, 0.3, 0.2])  # Grand metric weights
        
        # === Critical Thresholds ===
        self.H_crit = 1.5
        self.L_crit = 1.0
        self.E_crit = 2.0
        self.Psi_crit = 2.0
    
    def initialize_from_data(self, df):
        """Initialize system state from empirical data"""
        for i, node_label in enumerate(self.node_labels):
            # Find matching data
            node_data = df[df['Node'].str.contains(node_label, case=False, na=False)]
            if len(node_data) == 0:
                # Try partial matching
                for _, row in df.iterrows():
                    if any(term.lower() in row['Node'].lower() 
                          for term in node_label.lower().split()):
                        node_data = pd.DataFrame([row])
                        break
            
            if len(node_data) > 0:
                row = node_data.iloc[0]
                # Map CAMS data to unified variables
                self.chi[i] = max(0, min(1, row['Coherence'] / 10))  # Normalize to [0,1]
                self.kappa[i] = max(0, row['Capacity'])
                self.sigma[i] = max(0, abs(row['Stress']))
                self.alpha[i] = max(0, min(1, row['Abstraction'] / 10))  # Normalize to [0,1]
            else:
                # Default initialization
                self.chi[i] = np.random.uniform(0.3, 0.8)
                self.kappa[i] = np.random.uniform(2, 8)
                self.sigma[i] = np.random.uniform(0.5, 3.0)
                self.alpha[i] = np.random.uniform(0.2, 0.7)
        
        # Initialize stress decomposition
        self.sigma_chronic = 0.7 * self.sigma
        self.sigma_acute = 0.3 * self.sigma
        
        # Calculate initial bonds and metrics
        self.update_bonds()
        self.update_system_metrics()
    
    def update_bonds(self):
        """Update bond matrix B_ij based on current state"""
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Bond strength based on coherence similarity and capacity synergy
                    coherence_sim = np.exp(-abs(self.chi[i] - self.chi[j]) / 0.2)
                    capacity_syn = np.sqrt(self.kappa[i] * self.kappa[j]) / 10
                    stress_coupling = 1 + 0.1 * (self.sigma[i] + self.sigma[j])
                    
                    self.B[i, j] = coherence_sim * capacity_syn * stress_coupling
    
    def sigmoid_activation(self, x):
        """Sigmoid activation function σ_act(x) = 1/(1+e^(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def update_bond_dynamics(self, dt):
        """Update inter-institutional bond dynamics - equation (1)"""

        # Calculate effective thresholds θ_i(t)
        theta_eff = (self.theta_0 +
                    self.eta_sigma * self.sigma -
                    self.eta_chi * self.chi)

        # Stress-modulated input u_i(t)
        u = np.zeros(self.n)
        for i in range(self.n):
            # Weighted inputs from other nodes
            weighted_input = np.sum(self.w[i, :] * self.activation)
            
            # Stress modulation term
            stress_mod = self.beta[i] * (
                self.sigma_chronic[i] + 
                self.alpha_acute * self.sigma_acute[i]
            )
            
            u[i] = weighted_input - theta_eff[i] + stress_mod
        
        # Update activations a_i(t)
        self.activation = self.sigmoid_activation(u)
    
    def state_derivatives(self, t, state):
        """
        Calculate derivatives for all state variables - equations (2)
        State vector: [χ, κ, σ, α, M, ι]
        """
        
        # Unpack state
        chi = state[:self.n]
        kappa = state[self.n:2*self.n]
        sigma = state[2*self.n:3*self.n]
        alpha = state[3*self.n:4*self.n]
        M = state[4*self.n:5*self.n]
        iota = state[5*self.n]
        
        # Update internal state
        self.chi, self.kappa, self.sigma, self.alpha, self.M, self.iota = chi, kappa, sigma, alpha, M, iota
        
        # Decompose stress
        self.sigma_chronic = 0.7 * sigma
        self.sigma_acute = 0.3 * sigma
        
        # Update bonds and thermodynamic dynamics
        self.update_bonds()
        self.update_bond_dynamics(0.01)
        
        # Calculate system-level metrics for constraints
        C_complexity = np.sum(np.abs(self.B))
        K_coord = np.sum(kappa)
        
        derivatives = np.zeros_like(state)
        
        # === Capacity Evolution dκ_i/dt ===
        for i in range(self.n):
            capacity_growth = self.r_kappa * self.activation[i]
            capacity_diffusion = self.phi_kappa * np.sum(self.B[i, :] * (kappa - kappa[i]))
            stress_drag = self.xi_kappa * sigma[i]
            coord_load = self.zeta_kappa * C_complexity
            
            derivatives[i] = capacity_growth + capacity_diffusion - stress_drag - coord_load
        
        # === Coherence Evolution dχ_i/dt (Coherence Decay Law) ===
        maintenance = np.random.uniform(0.1, 0.8, self.n)  # Placeholder for maint_i(t)
        
        for i in range(self.n):
            coherence_decay = self.lambda_chi * chi[i]
            maintenance_term = self.mu_chi * maintenance[i]
            bond_coupling = np.sum(self.B[i, :] * (chi[i] - chi))
            stress_impact = self.rho_chi * sigma[i] * chi[i]
            
            derivatives[self.n + i] = (-coherence_decay + maintenance_term - 
                                     bond_coupling - stress_impact)
        
        # === Stress Evolution dσ_i/dt (Law 4 + CAMS) ===
        epsilon_shocks = np.random.normal(0, 0.1, self.n)  # External shocks
        
        for i in range(self.n):
            stress_dissipation = self.lambda_sigma * sigma[i]
            stress_propagation = self.rho_sigma * np.sum(self.B[i, :] * (sigma - sigma[i]))
            memory_load = self.upsilon_sigma * M[i]
            activation_reduction = self.omega_sigma * self.activation[i]
            
            derivatives[2*self.n + i] = (-stress_dissipation + stress_propagation + 
                                       epsilon_shocks[i] + memory_load - activation_reduction)
        
        # === Abstraction Evolution dα_i/dt (Law 5 + Abstraction Paradox) ===
        for i in range(self.n):
            memory_boost = self.gamma_alpha * M[i]
            stress_penalty = self.delta_alpha * sigma[i]
            
            derivatives[3*self.n + i] = memory_boost - stress_penalty
        
        # === Memory Evolution dM_i/dt ===
        for i in range(self.n):
            derivatives[4*self.n + i] = alpha[i] * chi[i]
        
        # === Innovation Evolution dι/dt (Law 13) ===
        # Innovation function F({κ_i, α_i, B_ij}) - simplified
        F_innovation = np.mean(kappa * alpha) + 0.1 * np.mean(self.B)
        sigma_bar = np.mean(sigma)
        chi_bar = np.mean(chi)
        
        innovation_growth = self.rho_iota * F_innovation
        stress_penalty = self.delta_iota * sigma_bar
        
        # Integration constraint |dι/dt| < f(χ̄, B)
        integration_limit = chi_bar * np.mean(self.B)
        raw_derivative = innovation_growth - stress_penalty
        
        derivatives[5*self.n] = np.sign(raw_derivative) * min(abs(raw_derivative), integration_limit)
        
        return derivatives
    
    def update_system_metrics(self):
        """Calculate all system-level metrics - equations (5-6)"""
        
        # === Coherence Asymmetry CA(t) ===
        chi_kappa_product = self.chi * self.kappa
        self.CA = np.var(chi_kappa_product) / (np.mean(chi_kappa_product) + 1e-9)
        
        # === Coherence Entropy E_χ(t) (Law 2) ===
        chi_safe = np.clip(self.chi, 1e-9, 1)  # Avoid log(0)
        self.E_chi = -np.sum(chi_safe * np.log(chi_safe))
        
        # === Information Integration I(t) (Law 8) ===
        self.I_info = self.eta_I * np.sum(self.alpha * self.kappa)
        
        # === Path Dependence P(t) ===
        CA_mean = self.CA  # Simplified for real-time
        self.P_path = 1.0 * np.exp(self.alpha_P * CA_mean)
        
        # === Node Health Calculation h_i(t) ===
        sigma_bar_ch = np.mean(self.sigma_chronic)
        sigma_bar_ac = np.mean(self.sigma_acute)
        
        D_i = (1 + np.sum(self.B, axis=1)) * (sigma_bar_ch + 2 * sigma_bar_ac + 1e-6)
        h_i = (self.chi * self.kappa / D_i) * (1 + 0.5 * self.alpha)
        
        # === Penalty Term P_t ===
        sigma_sum = np.sum(self.sigma)
        chi_sum = np.sum(self.chi) + 1e-6
        P_t = min(self.CA * (sigma_sum / chi_sum), 0.75)
        
        # === System Health H(t) ===
        self.H = np.sum(self.w_health * h_i) * (1 - P_t)
        
        # === Legitimacy L(t) (Law 9) ===
        chi_bar = np.mean(self.chi)
        self.L = (self.beta_L[0] * self.H + 
                 self.beta_L[1] * self.I_info + 
                 self.beta_L[2] * chi_bar)
        
        # === Elite Adaptation A_e(t) (Law 10) - simplified for all nodes ===
        A_e_mean = np.mean(self.kappa) + np.mean(self.alpha) - self.rho_e * np.mean(self.sigma)
        
        # === Grand System Metric Ψ(t) ===
        E_chi_inv = 1 / (self.E_chi + 1e-6)
        self.Psi = (self.omega_Psi[0] * self.H + 
                   self.omega_Psi[1] * E_chi_inv + 
                   self.omega_Psi[2] * self.L + 
                   self.omega_Psi[3] * A_e_mean)
    
    def check_safety_constraints(self):
        """Check all safety constraints from equation (7)"""
        violations = []
        
        if self.H < self.H_crit:
            violations.append(f"Health critical: {self.H:.3f} < {self.H_crit}")
        
        if self.L < self.L_crit:
            violations.append(f"Legitimacy critical: {self.L:.3f} < {self.L_crit}")
        
        if self.E_chi >= self.E_crit:
            violations.append(f"Entropy critical: {self.E_chi:.3f} ≥ {self.E_crit}")
        
        if self.Psi < self.Psi_crit:
            violations.append(f"System metric critical: {self.Psi:.3f} < {self.Psi_crit}")
        
        # Synchronization check (simplified)
        S_sync = np.std([np.gradient(self.chi), np.gradient(self.kappa), 
                        np.gradient(self.sigma), np.gradient(self.alpha)])
        if S_sync >= self.S_crit:
            violations.append(f"Synchronization critical: {S_sync:.3f} ≥ {self.S_crit}")
        
        # Stress variance check
        V_sigma = np.var(self.sigma)
        if V_sigma >= self.V_crit:
            violations.append(f"Stress variance critical: {V_sigma:.3f} ≥ {self.V_crit}")
        
        return violations
    
    def simulate_discrete(self, n_steps=100, dt=0.1):
        """Run discrete-time simulation - equation (8)"""
        
        # Initialize state vector
        state = np.concatenate([
            self.chi, self.kappa, self.sigma, self.alpha, self.M, [self.iota]
        ])
        
        # Clear history
        for key in self.history:
            self.history[key] = []
        
        for step in range(n_steps):
            t = step * dt
            
            # Calculate derivatives
            derivatives = self.state_derivatives(t, state)
            
            # Euler integration: x(t+Δt) = x(t) + Δt·dx/dt
            state = state + dt * derivatives
            
            # Ensure bounds
            state[:self.n] = np.clip(state[:self.n], 0, 1)        # χ ∈ [0,1]
            state[self.n:2*self.n] = np.maximum(state[self.n:2*self.n], 0)  # κ ≥ 0
            state[2*self.n:3*self.n] = np.maximum(state[2*self.n:3*self.n], 0)  # σ ≥ 0
            state[3*self.n:4*self.n] = np.clip(state[3*self.n:4*self.n], 0, 1)  # α ∈ [0,1]
            
            # Update internal state
            self.chi = state[:self.n]
            self.kappa = state[self.n:2*self.n]
            self.sigma = state[2*self.n:3*self.n]
            self.alpha = state[3*self.n:4*self.n]
            self.M = state[4*self.n:5*self.n]
            self.iota = state[5*self.n]
            
            # Update system metrics
            self.update_system_metrics()
            
            # Store history
            self.history['time'].append(t)
            self.history['chi'].append(self.chi.copy())
            self.history['kappa'].append(self.kappa.copy())
            self.history['sigma'].append(self.sigma.copy())
            self.history['alpha'].append(self.alpha.copy())
            self.history['H'].append(self.H)
            self.history['L'].append(self.L)
            self.history['Psi'].append(self.Psi)
            self.history['CA'].append(self.CA)
            self.history['E_chi'].append(self.E_chi)
            self.history['activation'].append(self.activation.copy())
            self.history['B'].append(self.B.copy())
            self.history['M'].append(self.M.copy())
            self.history['iota'].append(self.iota)
        
        return self.history

# === Streamlit Interface ===

# Load data
@st.cache_data
def load_available_datasets():
    """Load all available CAMS datasets"""
    csv_files = glob.glob("*.csv")
    datasets = {}
    
    country_mapping = {
        'australia cams cleaned': 'Australia',
        'usa cams cleaned': 'USA',
        'iran cams cleaned': 'Iran',
        'denmark cams cleaned': 'Denmark',
        'france cams cleaned': 'France',
        'italy cams cleaned': 'Italy',
        'iraq cams cleaned': 'Iraq'
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
with st.spinner("🔄 Loading datasets..."):
    datasets = load_available_datasets()

if not datasets:
    st.warning("No datasets found - using synthetic data")
    datasets = {"Synthetic": pd.DataFrame({
        'Node': ['Executive', 'Army', 'StateMemory', 'Priesthood'],
        'Coherence': [5, 6, 4, 7], 'Capacity': [6, 7, 5, 6],
        'Stress': [2, 3, 4, 1], 'Abstraction': [6, 4, 8, 7]
    })}

st.success(f"✅ Loaded {len(datasets)} datasets: {', '.join(datasets.keys())}")

# === Control Panel ===
st.sidebar.markdown("## 🎛️ Unified Framework Controls")

# Dataset selection
selected_country = st.sidebar.selectbox(
    "Select Dataset:", 
    options=list(datasets.keys())
)

# Simulation parameters
n_steps = st.sidebar.slider("Simulation Steps", 50, 500, 200, 25)
dt = st.sidebar.slider("Time Step (dt)", 0.01, 0.2, 0.05, 0.01)

# Thermodynamic bond parameters
st.sidebar.markdown("### 🌡️ Thermodynamic Parameters")
beta_stress = st.sidebar.slider("Stress Modulation (β)", 0.1, 1.0, 0.3, 0.05)
alpha_acute = st.sidebar.slider("Acute Stress Multiplier", 1.0, 10.0, 3.0, 0.5)

# Analysis mode
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode:",
    ["Full Simulation", "State Analysis", "Constraint Monitoring", "Phase Dynamics"]
)

# === Initialize System ===
with st.spinner("🔧 Initializing unified framework..."):
    system = UnifiedCAMS13Laws()
    
    # Get latest data
    country_data = datasets[selected_country]
    if 'Year' in country_data.columns:
        latest_year = country_data['Year'].max()
        latest_data = country_data[country_data['Year'] == latest_year]
    else:
        latest_data = country_data
    
    # Initialize from data
    system.initialize_from_data(latest_data)
    
    # Update parameters from sidebar
    system.beta = np.full(system.n, beta_stress)
    system.alpha_acute = alpha_acute

st.success(f"✅ Unified system initialized for {selected_country}")

# === Main Analysis ===

if analysis_mode == "Full Simulation":
    st.markdown("## 🚀 Full System Simulation")
    
    if st.button("▶️ Run Unified Simulation", type="primary"):
        
        # Display current state
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Health H(t)", f"{system.H:.3f}")
        with col2:
            st.metric("Legitimacy L(t)", f"{system.L:.3f}")
        with col3:
            st.metric("Grand Metric Ψ(t)", f"{system.Psi:.3f}")
        with col4:
            st.metric("Coherence Asymmetry", f"{system.CA:.3f}")
        
        # Run simulation
        with st.spinner("Running unified CAMS-CAN + 13 Laws simulation..."):
            history = system.simulate_discrete(n_steps=n_steps, dt=dt)
        
        st.success("✅ Simulation completed!")
        
        # === Results Visualization ===
        
        # System-level metrics
        fig_system = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'System Health H(t) Evolution',
                'Legitimacy L(t) & Grand Metric Ψ(t)',
                'Coherence Asymmetry CA(t)',
                'Innovation ι(t) & Information I(t)'
            ]
        )
        
        # System Health
        fig_system.add_trace(
            go.Scatter(x=history['time'], y=history['H'],
                      name='System Health H(t)', line=dict(color='green', width=3)),
            row=1, col=1
        )
        fig_system.add_hline(y=system.H_crit, line_dash="dash", line_color="red",
                            annotation_text="Critical", row=1, col=1)
        
        # Legitimacy & Grand Metric
        fig_system.add_trace(
            go.Scatter(x=history['time'], y=history['L'],
                      name='Legitimacy L(t)', line=dict(color='blue')),
            row=1, col=2
        )
        fig_system.add_trace(
            go.Scatter(x=history['time'], y=history['Psi'],
                      name='Grand Metric Ψ(t)', line=dict(color='purple')),
            row=1, col=2
        )
        
        # Coherence Asymmetry
        fig_system.add_trace(
            go.Scatter(x=history['time'], y=history['CA'],
                      name='CA(t)', line=dict(color='red')),
            row=2, col=1
        )
        
        # Innovation & Information
        I_history = [system.eta_I * np.sum(alpha * kappa) 
                    for alpha, kappa in zip(history['alpha'], history['kappa'])]
        fig_system.add_trace(
            go.Scatter(x=history['time'], y=history['iota'],
                      name='Innovation ι(t)', line=dict(color='orange')),
            row=2, col=2
        )
        fig_system.add_trace(
            go.Scatter(x=history['time'], y=I_history,
                      name='Information I(t)', line=dict(color='cyan')),
            row=2, col=2
        )
        
        fig_system.update_layout(height=800, title_text="Unified Framework: System Metrics")
        st.plotly_chart(fig_system, use_container_width=True)
        
        # Node-level dynamics
        st.markdown("### 🏛️ Institutional Node Dynamics")
        
        fig_nodes = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Coherence χ_i(t) Evolution',
                'Capacity κ_i(t) Evolution', 
                'Stress σ_i(t) Evolution',
                'Abstraction α_i(t) Evolution'
            ]
        )
        
        colors = px.colors.qualitative.Set1[:system.n]
        
        for i, (node, color) in enumerate(zip(system.node_labels, colors)):
            # Coherence
            chi_history = [chi[i] for chi in history['chi']]
            fig_nodes.add_trace(
                go.Scatter(x=history['time'], y=chi_history,
                          name=f'{node} χ', line=dict(color=color)),
                row=1, col=1
            )
            
            # Capacity
            kappa_history = [kappa[i] for kappa in history['kappa']]
            fig_nodes.add_trace(
                go.Scatter(x=history['time'], y=kappa_history,
                          name=f'{node} κ', line=dict(color=color)),
                row=1, col=2
            )
            
            # Stress
            sigma_history = [sigma[i] for sigma in history['sigma']]
            fig_nodes.add_trace(
                go.Scatter(x=history['time'], y=sigma_history,
                          name=f'{node} σ', line=dict(color=color)),
                row=2, col=1
            )
            
            # Abstraction
            alpha_history = [alpha[i] for alpha in history['alpha']]
            fig_nodes.add_trace(
                go.Scatter(x=history['time'], y=alpha_history,
                          name=f'{node} α', line=dict(color=color)),
                row=2, col=2
            )
        
        fig_nodes.update_layout(height=800, title_text="Node State Evolution")
        st.plotly_chart(fig_nodes, use_container_width=True)
        
        # Final system assessment
        st.markdown("### 🎯 Final System Assessment")
        
        violations = system.check_safety_constraints()
        
        if violations:
            st.error("🚨 **SAFETY CONSTRAINT VIOLATIONS:**")
            for violation in violations:
                st.write(f"- {violation}")
        else:
            st.success("✅ All safety constraints satisfied")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            final_H = history['H'][-1]
            H_trend = "↗️" if final_H > history['H'][0] else "↘️"
            st.metric("Final Health", f"{final_H:.3f}", delta=H_trend)
        
        with col2:
            final_L = history['L'][-1]
            L_trend = "↗️" if final_L > history['L'][0] else "↘️"
            st.metric("Final Legitimacy", f"{final_L:.3f}", delta=L_trend)
        
        with col3:
            final_Psi = history['Psi'][-1]
            Psi_trend = "↗️" if final_Psi > history['Psi'][0] else "↘️"
            st.metric("Final Grand Metric", f"{final_Psi:.3f}", delta=Psi_trend)

elif analysis_mode == "State Analysis":
    st.markdown("## 📊 Current State Analysis")
    
    # Current state display
    state_data = []
    for i, node in enumerate(system.node_labels):
        state_data.append({
            'Node': node,
            'Coherence_χ': system.chi[i],
            'Capacity_κ': system.kappa[i],
            'Stress_σ': system.sigma[i],
            'Abstraction_α': system.alpha[i],
            'Memory_M': system.M[i],
            'Activation_a': system.activation[i] if len(system.activation) > i else 0
        })
    
    state_df = pd.DataFrame(state_data)
    st.dataframe(state_df, use_container_width=True)
    
    # State visualization
    fig_state = go.Figure(data=go.Heatmap(
        z=np.array([system.chi, system.kappa, system.sigma, system.alpha]),
        x=system.node_labels,
        y=['Coherence χ', 'Capacity κ', 'Stress σ', 'Abstraction α'],
        colorscale='RdBu_r',
        text=np.round(np.array([system.chi, system.kappa, system.sigma, system.alpha]), 3),
        texttemplate="%{text}",
        hovertemplate="Node: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>"
    ))
    
    fig_state.update_layout(title="Current Node States", height=400)
    st.plotly_chart(fig_state, use_container_width=True)
    
    # Bond matrix
    st.markdown("### 🔗 Bond Matrix B_ij")
    
    fig_bonds = go.Figure(data=go.Heatmap(
        z=system.B,
        x=system.node_labels,
        y=system.node_labels,
        colorscale='Viridis',
        text=np.round(system.B, 3),
        texttemplate="%{text}",
        hovertemplate="From: %{y}<br>To: %{x}<br>Bond: %{z:.3f}<extra></extra>"
    ))
    
    fig_bonds.update_layout(title="Inter-Institutional Bonds", height=500)
    st.plotly_chart(fig_bonds, use_container_width=True)

elif analysis_mode == "Constraint Monitoring":
    st.markdown("## ⚖️ 13 Laws Constraint Monitoring")
    
    # Check all constraints
    violations = system.check_safety_constraints()
    
    # Constraint status
    constraints = {
        "System Health": {"value": system.H, "threshold": system.H_crit, "direction": "≥"},
        "Legitimacy": {"value": system.L, "threshold": system.L_crit, "direction": "≥"},
        "Coherence Entropy": {"value": system.E_chi, "threshold": system.E_crit, "direction": "≤"},
        "Grand Metric": {"value": system.Psi, "threshold": system.Psi_crit, "direction": "≥"},
        "Coordination": {"value": np.sum(system.kappa), 
                        "threshold": system.gamma_coord * np.sum(np.abs(system.B)), 
                        "direction": "≥"}
    }
    
    constraint_data = []
    for name, info in constraints.items():
        status = "✅ SAFE" if (
            (info["direction"] == "≥" and info["value"] >= info["threshold"]) or
            (info["direction"] == "≤" and info["value"] <= info["threshold"])
        ) else "🚨 VIOLATION"
        
        margin = abs(info["value"] - info["threshold"])
        
        constraint_data.append({
            'Constraint': name,
            'Current': f"{info['value']:.3f}",
            'Threshold': f"{info['threshold']:.3f}",
            'Required': info['direction'],
            'Status': status,
            'Margin': f"{margin:.3f}"
        })
    
    st.dataframe(pd.DataFrame(constraint_data), use_container_width=True)
    
    # Constraint visualization
    fig_constraints = go.Figure()
    
    for i, (name, info) in enumerate(constraints.items()):
        color = 'green' if (
            (info["direction"] == "≥" and info["value"] >= info["threshold"]) or
            (info["direction"] == "≤" and info["value"] <= info["threshold"])
        ) else 'red'
        
        fig_constraints.add_trace(go.Bar(
            x=[name], y=[info["value"]],
            marker_color=color, opacity=0.7,
            name=f"{name}: {info['value']:.3f}"
        ))
        
        # Add threshold line
        fig_constraints.add_hline(
            y=info["threshold"], line_dash="dash",
            annotation_text=f"Threshold: {info['threshold']:.3f}"
        )
    
    fig_constraints.update_layout(
        title="Constraint Compliance Status",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig_constraints, use_container_width=True)

elif analysis_mode == "Phase Dynamics":
    st.markdown("## 🌀 Phase Space Dynamics")
    
    # Run short simulation for phase trajectory
    with st.spinner("Computing phase trajectory..."):
        history = system.simulate_discrete(n_steps=50, dt=0.05)
    
    # Phase space plot (H vs L)
    fig_phase = go.Figure()
    
    fig_phase.add_trace(go.Scatter(
        x=history['H'], y=history['L'],
        mode='markers+lines',
        marker=dict(
            size=8,
            color=history['time'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time")
        ),
        name='Phase Trajectory',
        hovertemplate="Health: %{x:.3f}<br>Legitimacy: %{y:.3f}<br>Time: %{marker.color:.1f}<extra></extra>"
    ))
    
    # Add critical boundaries
    fig_phase.add_hline(y=system.L_crit, line_dash="dash", line_color="red",
                       annotation_text="Legitimacy Critical")
    fig_phase.add_vline(x=system.H_crit, line_dash="dash", line_color="red",
                       annotation_text="Health Critical")
    
    # Add regions
    fig_phase.add_shape(
        type="rect", x0=system.H_crit, y0=system.L_crit, x1=10, y1=10,
        fillcolor="green", opacity=0.1, line=dict(color="green")
    )
    fig_phase.add_annotation(x=3, y=3, text="STABLE REGION", 
                           showarrow=False, font=dict(color="green", size=16))
    
    fig_phase.update_layout(
        title="Phase Space: Health vs Legitimacy",
        xaxis_title="System Health H(t)",
        yaxis_title="Legitimacy L(t)",
        height=600
    )
    
    st.plotly_chart(fig_phase, use_container_width=True)
    
    # 3D phase space
    fig_3d = go.Figure(data=go.Scatter3d(
        x=history['H'],
        y=history['L'], 
        z=history['Psi'],
        mode='markers+lines',
        marker=dict(
            size=4,
            color=history['time'],
            colorscale='Plasma',
            showscale=True
        ),
        line=dict(color='blue', width=3),
        name='3D Trajectory'
    ))
    
    fig_3d.update_layout(
        title="3D Phase Space: H-L-Ψ",
        scene=dict(
            xaxis_title="Health H(t)",
            yaxis_title="Legitimacy L(t)",
            zaxis_title="Grand Metric Ψ(t)"
        ),
        height=600
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

# === Footer ===
st.markdown("---")
st.markdown("### 📋 Unified Framework Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **CAMS-CAN Components:**
    - Thermodynamic inter-institutional bonds
    - Node state dynamics (χ,κ,σ,α)
    - Meta-cognitive emergence
    - Information integration
    """)

with col2:
    st.markdown("""
    **13 Laws Integration:**
    - Constraint architecture
    - Safety monitoring
    - Elite adaptation
    - Path dependence
    """)

with col3:
    st.markdown(f"""
    **Current System:**
    - Dataset: {selected_country}
    - Nodes: {system.n}
    - Health: {system.H:.3f}
    - Status: {"🟢 Safe" if not system.check_safety_constraints() else "🚨 Violations"}
    """)

st.success("🎉 Unified CAMS-CAN + 13 Laws Framework fully operational!")