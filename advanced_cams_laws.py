"""
Advanced CAMS Framework with 13 Universal Laws
Comprehensive Civilizational Dynamics Simulator
Created: July 26, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

@dataclass
class NodeState:
    """Represents the state of a single node in the CAMS system"""
    coherence: float    # C - Internal organization and unity
    capacity: float     # K - Ability to perform functions  
    stress: float       # S - Pressure and strain
    abstraction: float  # A - Distance from operational reality
    
    def as_vector(self) -> np.ndarray:
        return np.array([self.coherence, self.capacity, self.stress, self.abstraction])

class CAMSLaws:
    """Implementation of the CAMS 13 Universal Laws"""
    
    def __init__(self, nodes: np.ndarray, node_names: List[str]):
        """
        nodes: shape (8,4) representing [Coherence, Capacity, Stress, Abstraction]
        node_names: names of the 8 nodes
        """
        self.nodes = nodes  # shape (8,4)
        self.node_names = node_names
        self.n_nodes = len(nodes)
        
    def law_1_capacity_stress_balance(self) -> Dict:
        """Law 1: Capacity-Stress Balance Law"""
        capacity = self.nodes[:, 1]
        stress = self.nodes[:, 2]
        balance = capacity - stress
        system_balance = np.sum(balance)
        
        return {
            'system_balance': system_balance,
            'node_balances': dict(zip(self.node_names, balance)),
            'imbalanced_nodes': [name for name, bal in zip(self.node_names, balance) if bal < -1.0]
        }
    
    def law_2_coherence_capacity_coupling(self) -> Dict:
        """Law 2: Coherence-Capacity Coupling Law"""
        coherence = self.nodes[:, 0]
        capacity = self.nodes[:, 1]
        coupling_strength = coherence * capacity
        
        return {
            'coupling_strengths': dict(zip(self.node_names, coupling_strength)),
            'weak_coupling_nodes': [name for name, strength in zip(self.node_names, coupling_strength) if strength < 20]
        }
    
    def law_3_coherence_decay(self, prev_coherence: np.ndarray, dt: float = 0.1) -> Dict:
        """Law 3: Coherence Decay Law"""
        current_coherence = self.nodes[:, 0]
        natural_decay_rate = 0.01 * dt
        expected_decay = prev_coherence * natural_decay_rate
        actual_decay = prev_coherence - current_coherence
        
        excess_decay = actual_decay - expected_decay
        violations = excess_decay > natural_decay_rate
        
        return {
            'excess_decay': dict(zip(self.node_names, excess_decay)),
            'violated_nodes': [name for name, viol in zip(self.node_names, violations) if viol],
            'decay_rate': dict(zip(self.node_names, actual_decay / prev_coherence))
        }
    
    def law_4_stress_propagation(self, bond_matrix: np.ndarray, alpha: float = 0.05) -> Dict:
        """Law 4: Stress Propagation Law"""
        stress = self.nodes[:, 2]
        propagated_stress = alpha * bond_matrix.dot(stress)
        new_stress = stress + propagated_stress
        
        return {
            'original_stress': dict(zip(self.node_names, stress)),
            'propagated_stress': dict(zip(self.node_names, propagated_stress)),
            'new_stress': dict(zip(self.node_names, new_stress)),
            'stress_multipliers': dict(zip(self.node_names, new_stress / np.maximum(stress, 0.1)))
        }
    
    def law_5_abstraction_drift(self, operational_baseline: np.ndarray = None) -> Dict:
        """Law 5: Abstraction Drift Law"""
        if operational_baseline is None:
            operational_baseline = np.ones(self.n_nodes) * 2.0  # Baseline operational level
            
        abstraction = self.nodes[:, 3]
        drift = abstraction - operational_baseline
        
        return {
            'abstraction_drift': dict(zip(self.node_names, drift)),
            'excessive_drift_nodes': [name for name, d in zip(self.node_names, drift) if d > 3.0],
            'drift_severity': np.mean(np.maximum(0, drift - 1.0))
        }
    
    def law_6_system_fitness(self) -> Dict:
        """Law 6: System Fitness Law"""
        coherence = self.nodes[:, 0]
        capacity = self.nodes[:, 1]
        stress = self.nodes[:, 2]
        abstraction = self.nodes[:, 3]
        
        # Fitness function: balance of coherence, capacity vs stress, abstraction
        tau, lam = 3.0, 0.5
        fitness_components = (coherence * capacity) / (1 + np.exp((np.abs(stress) - tau) / lam))
        abstraction_penalty = 1 / (1 + 0.1 * np.maximum(0, abstraction - 2.0)**2)
        
        node_fitness = fitness_components * abstraction_penalty
        system_fitness = np.mean(node_fitness)
        
        return {
            'system_fitness': system_fitness,
            'node_fitness': dict(zip(self.node_names, node_fitness)),
            'fitness_components': dict(zip(self.node_names, fitness_components)),
            'low_fitness_nodes': [name for name, fit in zip(self.node_names, node_fitness) if fit < 15]
        }
    
    def law_7_elite_circulation(self, elite_indices: List[int] = [0, 1, 3, 6]) -> Dict:
        """Law 7: Elite Circulation Law"""
        # Default elite nodes: Executive, Army, Property Owners, State Memory
        elite_nodes = self.nodes[elite_indices]
        elite_names = [self.node_names[i] for i in elite_indices]
        
        # Elite vitality = Coherence + Capacity - 0.5 * Stress - 0.2 * Abstraction
        vitality = elite_nodes[:, 0] + elite_nodes[:, 1] - 0.5 * elite_nodes[:, 2] - 0.2 * elite_nodes[:, 3]
        
        return {
            'elite_vitality': dict(zip(elite_names, vitality)),
            'average_elite_vitality': np.mean(vitality),
            'stagnant_elites': [name for name, v in zip(elite_names, vitality) if v < 8.0],
            'circulation_needed': np.mean(vitality) < 10.0
        }
    
    def law_8_bond_strength_matrix(self) -> Tuple[np.ndarray, Dict]:
        """Law 8: Bond Strength Law - Calculate inter-node bond strengths"""
        coherence = self.nodes[:, 0]
        abstraction = self.nodes[:, 3]
        
        # Bond strength based on coherence similarity and abstraction compatibility
        bond_matrix = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    coh_similarity = 1 - abs(coherence[i] - coherence[j]) / 10.0
                    abs_compatibility = 1 - abs(abstraction[i] - abstraction[j]) / 5.0
                    bond_matrix[i, j] = max(0, coh_similarity * abs_compatibility)
        
        # Network metrics
        avg_bond_strength = np.mean(bond_matrix[bond_matrix > 0])
        weak_bonds = np.sum(bond_matrix < 0.3) - self.n_nodes  # Exclude diagonal
        
        return bond_matrix, {
            'average_bond_strength': avg_bond_strength,
            'weak_bond_count': weak_bonds,
            'network_density': np.sum(bond_matrix > 0.5) / (self.n_nodes * (self.n_nodes - 1)),
            'most_connected_node': self.node_names[np.argmax(np.sum(bond_matrix, axis=1))]
        }
    
    def law_9_synchronization_threshold(self, bond_matrix: np.ndarray) -> Dict:
        """Law 9: Synchronization Threshold Law"""
        coherence = self.nodes[:, 0]
        
        # Calculate synchronization potential
        sync_forces = bond_matrix.dot(coherence)
        sync_potential = sync_forces / np.maximum(coherence, 0.1)
        
        return {
            'synchronization_forces': dict(zip(self.node_names, sync_forces)),
            'sync_potential': dict(zip(self.node_names, sync_potential)),
            'synchronized_nodes': [name for name, pot in zip(self.node_names, sync_potential) if pot > 1.2],
            'system_sync_level': np.mean(sync_potential)
        }
    
    def law_10_innovation_threshold(self) -> Dict:
        """Law 10: Innovation Threshold Law"""
        abstraction = self.nodes[:, 3]
        stress = self.nodes[:, 2]
        capacity = self.nodes[:, 1]
        
        # Innovation = Abstraction × Stress × Freedom (freedom ∝ capacity/stress)
        freedom = capacity / (1 + stress)
        innovation_rate = abstraction * stress * freedom / 100  # Normalized
        
        return {
            'innovation_rates': dict(zip(self.node_names, innovation_rate)),
            'system_innovation': np.mean(innovation_rate),
            'innovation_leaders': [name for name, rate in zip(self.node_names, innovation_rate) if rate > np.mean(innovation_rate) * 1.5],
            'innovation_potential': np.sum(innovation_rate)
        }
    
    def law_11_stress_cascade_analysis(self, bond_matrix: np.ndarray) -> Dict:
        """Law 11: Stress Cascade Law"""
        stress = self.nodes[:, 2]
        
        # Simulate stress cascade through network
        cascade_stress = stress.copy()
        for _ in range(3):  # 3 propagation steps
            cascade_stress += 0.1 * bond_matrix.dot(cascade_stress)
        
        cascade_amplification = cascade_stress / np.maximum(stress, 0.1)
        
        return {
            'original_stress': dict(zip(self.node_names, stress)),
            'cascaded_stress': dict(zip(self.node_names, cascade_stress)),
            'amplification_factor': dict(zip(self.node_names, cascade_amplification)),
            'cascade_vulnerable_nodes': [name for name, amp in zip(self.node_names, cascade_amplification) if amp > 2.0]
        }
    
    def law_12_metastability_detection(self) -> Dict:
        """Law 12: Metastability Law"""
        coherence = self.nodes[:, 0]
        capacity = self.nodes[:, 1]
        stress = self.nodes[:, 2]
        
        # Metastability indicators
        stability_index = coherence * capacity / (1 + stress**2)
        variance_coherence = np.var(coherence)
        
        # High individual stability but high system variance suggests metastability
        metastable = (np.mean(stability_index) > 15) and (variance_coherence > 2.0)
        
        return {
            'stability_indices': dict(zip(self.node_names, stability_index)),
            'coherence_variance': variance_coherence,
            'system_metastable': metastable,
            'metastability_risk': variance_coherence / np.mean(stability_index) if np.mean(stability_index) > 0 else 0
        }
    
    def law_13_transformation_potential(self) -> Dict:
        """Law 13: Transformation Potential Law"""
        # Combination of innovation, stress, and system fitness
        innovation_data = self.law_10_innovation_threshold()
        fitness_data = self.law_6_system_fitness()
        
        transformation_score = (
            innovation_data['system_innovation'] * 0.4 +
            (7.0 - fitness_data['system_fitness']) / 7.0 * 0.3 +  # Normalized inverse fitness
            min(np.mean(self.nodes[:, 2]) / 5.0, 1.0) * 0.3  # Normalized stress
        )
        
        return {
            'transformation_score': transformation_score,
            'transformation_likely': transformation_score > 0.6,
            'transformation_type': 'Mutation' if transformation_score > 0.6 else 'Stabilization',
            'key_drivers': {
                'innovation': innovation_data['system_innovation'],
                'fitness_pressure': (7.0 - fitness_data['system_fitness']) / 7.0,
                'stress_level': min(np.mean(self.nodes[:, 2]) / 5.0, 1.0)
            }
        }

class CAMSNodeSimulator:
    """Advanced CAMS simulator with full node-level dynamics and 13 Universal Laws"""
    
    # Node definitions
    NODE_NAMES = [
        'Executive', 'Army', 'Priesthood', 'Property_Owners',
        'Trades_Professions', 'Proletariat', 'State_Memory', 'Merchants'
    ]
    
    # Critical thresholds
    HEALTH_COLLAPSE_THRESHOLD = 2.5
    COHERENCE_ASYMMETRY_THRESHOLD = 0.5
    STRESS_THRESHOLD = 7.0
    ABSTRACTION_DRIFT_THRESHOLD = 3.0
    
    def __init__(self, initial_nodes: np.ndarray, coupling_strength: float = 0.1):
        """
        initial_nodes: shape (8,4) - [Coherence, Capacity, Stress, Abstraction] for each node
        """
        self.initial_nodes = initial_nodes
        self.coupling_strength = coupling_strength
        self.bond_matrix = None
        
    def compute_node_dynamics(self, t: float, nodes_flat: np.ndarray, 
                             external_shocks: np.ndarray = None) -> np.ndarray:
        """Compute time derivatives for all nodes with integrated Laws 2 & 3"""
        nodes = nodes_flat.reshape((8, 4))
        
        if external_shocks is None:
            external_shocks = np.zeros_like(nodes)
        
        # Extract variables
        C, K, S, A = nodes[:, 0], nodes[:, 1], nodes[:, 2], nodes[:, 3]
        
        # Update bond matrix based on current state
        laws = CAMSLaws(nodes, self.NODE_NAMES)
        self.bond_matrix, _ = laws.law_8_bond_strength_matrix()
        
        # Calculate system entropy (Law 2: Terminal Horizons)
        entropy = -np.sum(C * np.log(np.maximum(C/10.0, 1e-6)))  # Normalized
        entropy_pressure = 0.01 * entropy if entropy > 15.0 else 0.0
        
        # Coherence dynamics with Law 3 (Coherence Decay) integrated
        # Natural decay + entropy pressure + stress + capacity support + network sync
        sync_force = self.bond_matrix.dot(C) * 0.02
        coherence_decay = 0.012 * C  # Law 3: Natural coherence decay
        dC = (-0.1 * S + 0.05 * K + sync_force 
              - coherence_decay - entropy_pressure + external_shocks[:, 0])
        
        # Capacity dynamics - Law 2 coupling with coherence
        capacity_coherence_coupling = 0.03 * C * K / np.maximum(C + K, 1.0)  # Law 2
        dK = (0.08 * C - 0.06 * S - 0.02 * A 
              + capacity_coherence_coupling + external_shocks[:, 1])
        
        # Stress dynamics - accumulates from low capacity, propagates through network
        stress_propagation = self.bond_matrix.dot(S) * 0.03
        dS = (0.15 * (1 / np.maximum(K, 0.1)) + 0.08 * A + stress_propagation 
              - 0.05 * S + 0.005 * entropy_pressure + external_shocks[:, 2])
        
        # Abstraction dynamics - grows with stress and capacity, natural decay
        dA = 0.04 * S + 0.06 * K - 0.03 * A + external_shocks[:, 3]
        
        # Add small random noise for realism
        noise = np.random.normal(0, 0.001, (8, 4))
        derivatives = np.column_stack([dC, dK, dS, dA]) + noise
        
        return derivatives.flatten()
    
    def simulate(self, time_span: Tuple[float, float] = (0, 100), 
                dt: float = 0.1, external_shocks_timeline: Dict = None) -> Tuple:
        """Run the full node-level simulation"""
        t_eval = np.arange(time_span[0], time_span[1], dt)
        
        def dynamics_wrapper(t, y):
            nodes = y.reshape((8, 4))
            shocks = np.zeros_like(nodes)
            
            if external_shocks_timeline:
                for shock_time, shock_vals in external_shocks_timeline.items():
                    if abs(t - shock_time) < dt:
                        if isinstance(shock_vals, dict):
                            # Convert dict format to array
                            for node_idx, node_name in enumerate(self.NODE_NAMES):
                                if node_name in shock_vals:
                                    shocks[node_idx] = shock_vals[node_name]
                        else:
                            shocks = shock_vals
                        break
            
            return self.compute_node_dynamics(t, y, shocks)
        
        solution = solve_ivp(
            dynamics_wrapper,
            time_span,
            self.initial_nodes.flatten(),
            t_eval=t_eval,
            method='RK45',
            max_step=dt
        )
        
        # Reshape solution
        times = solution.t
        node_trajectories = solution.y.T.reshape((-1, 8, 4))
        
        return times, node_trajectories
    
    def compute_system_metrics(self, nodes: np.ndarray) -> Dict:
        """Compute system-level aggregated metrics from node states"""
        laws = CAMSLaws(nodes, self.NODE_NAMES)
        
        # System Health (weighted average of node fitness)
        fitness_data = laws.law_6_system_fitness()
        system_health = fitness_data['system_fitness'] / 5.0  # Normalize to ~0-5 scale
        
        # Coherence Asymmetry (coefficient of variation in coherence-capacity product)
        coherence = nodes[:, 0]
        capacity = nodes[:, 1]
        cc_product = coherence * capacity
        coherence_asymmetry = np.std(cc_product) / np.mean(cc_product) if np.mean(cc_product) > 0 else 1.0
        
        # System Stress (weighted by node importance)
        # Give more weight to critical nodes (Executive, Army, Property Owners)
        weights = np.array([1.5, 1.3, 1.0, 1.4, 1.0, 0.8, 1.2, 1.0])
        system_stress = np.average(nodes[:, 2], weights=weights)
        
        # System Abstraction (average abstraction drift)
        abstraction_data = laws.law_5_abstraction_drift()
        system_abstraction = abstraction_data['drift_severity']
        
        # System Entropy (Law 2: Terminal Horizons)
        entropy = -np.sum(coherence * np.log(np.maximum(coherence/10.0, 1e-6)))
        
        # Psi System Attractor Metric (overall system resilience)
        # Ψ(t) = w1*H(t) + w2/E(t) + w3*Innovation(t) + w4*SyncLevel(t)
        innovation_data = laws.law_10_innovation_threshold()
        sync_data = laws.law_9_synchronization_threshold(laws.law_8_bond_strength_matrix()[0])
        
        w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2  # Weights
        psi_attractor = (w1 * system_health + 
                        w2 / max(entropy/10.0, 0.1) + 
                        w3 * innovation_data['system_innovation'] * 10 + 
                        w4 * sync_data['system_sync_level'])
        
        return {
            'system_health': system_health,
            'coherence_asymmetry': coherence_asymmetry,
            'system_stress': system_stress,
            'system_abstraction': system_abstraction,
            'system_entropy': entropy,
            'psi_attractor': psi_attractor,
            'fitness_data': fitness_data,
            'abstraction_data': abstraction_data,
            'innovation_data': innovation_data,
            'sync_data': sync_data
        }
    
    def analyze_laws_compliance(self, nodes: np.ndarray, prev_nodes: np.ndarray = None) -> Dict:
        """Analyze compliance with all 13 CAMS laws"""
        laws = CAMSLaws(nodes, self.NODE_NAMES)
        bond_matrix, bond_data = laws.law_8_bond_strength_matrix()
        
        analysis = {
            'law_1_capacity_stress': laws.law_1_capacity_stress_balance(),
            'law_2_coherence_capacity': laws.law_2_coherence_capacity_coupling(),
            'law_4_stress_propagation': laws.law_4_stress_propagation(bond_matrix),
            'law_5_abstraction_drift': laws.law_5_abstraction_drift(),
            'law_6_system_fitness': laws.law_6_system_fitness(),
            'law_7_elite_circulation': laws.law_7_elite_circulation(),
            'law_8_bond_strength': bond_data,
            'law_9_synchronization': laws.law_9_synchronization_threshold(bond_matrix),
            'law_10_innovation': laws.law_10_innovation_threshold(),
            'law_11_stress_cascade': laws.law_11_stress_cascade_analysis(bond_matrix),
            'law_12_metastability': laws.law_12_metastability_detection(),
            'law_13_transformation': laws.law_13_transformation_potential()
        }
        
        if prev_nodes is not None:
            analysis['law_3_coherence_decay'] = laws.law_3_coherence_decay(prev_nodes[:, 0])
        
        return analysis

def create_default_initial_state() -> np.ndarray:
    """Create a reasonable initial state for the 8 nodes"""
    return np.array([
        [7.5, 7.0, 2.5, 3.0],  # Executive
        [7.0, 6.5, 3.0, 2.5],  # Army  
        [6.5, 6.0, 2.0, 4.0],  # Priesthood
        [7.2, 7.5, 3.5, 3.5],  # Property Owners
        [6.8, 6.2, 2.8, 3.2],  # Trades/Professions
        [5.5, 5.5, 4.5, 2.0],  # Proletariat
        [7.0, 6.8, 2.2, 3.8],  # State Memory
        [6.5, 6.0, 3.2, 3.0]   # Merchants
    ])

def plot_network_bonds(bond_matrix: np.ndarray, node_names: List[str], title: str = "Node Network Bonds"):
    """Plot network bond strength matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(bond_matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(node_names)))
    ax.set_yticks(range(len(node_names)))
    ax.set_xticklabels([name[:8] for name in node_names], rotation=45, ha='right')
    ax.set_yticklabels([name[:8] for name in node_names])
    
    # Add text annotations
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if i != j:
                text = ax.text(j, i, f'{bond_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white" if bond_matrix[i, j] > 0.5 else "black")
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Bond Strength')
    plt.tight_layout()
    return fig

# Integration functions for existing CAMS system
def analyze_real_data_with_laws(df: pd.DataFrame, nation: str = None) -> Dict:
    """Analyze real CAMS data using the 13 Universal Laws"""
    import sys
    sys.path.append('src')
    from cams_analyzer import CAMSAnalyzer
    
    analyzer = CAMSAnalyzer()
    
    # Filter data
    if nation:
        nation_col = analyzer._get_column_name(df, 'nation')
        if nation_col and nation_col in df.columns:
            df = df[df[nation_col] == nation]
    
    # Get column mappings
    year_col = analyzer._get_column_name(df, 'year')
    node_col = analyzer._get_column_name(df, 'node')
    coherence_col = analyzer._get_column_name(df, 'coherence')
    capacity_col = analyzer._get_column_name(df, 'capacity')
    stress_col = analyzer._get_column_name(df, 'stress')
    abstraction_col = analyzer._get_column_name(df, 'abstraction')
    
    # Get latest year data
    latest_year = df[year_col].max()
    year_data = df[df[year_col] == latest_year]
    
    # Convert to node matrix format (8 nodes x 4 dimensions)
    nodes = np.zeros((8, 4))
    node_mapping = {
        'Executive': 0, 'Army': 1, 'Priesthood': 2, 'Property_Owners': 3,
        'Trades_Professions': 4, 'Proletariat': 5, 'State_Memory': 6, 'Merchants': 7
    }
    
    for _, row in year_data.iterrows():
        node_name = row[node_col]
        if node_name in node_mapping:
            idx = node_mapping[node_name]
            nodes[idx] = [
                pd.to_numeric(row[coherence_col], errors='coerce'),
                pd.to_numeric(row[capacity_col], errors='coerce'),
                pd.to_numeric(row[stress_col], errors='coerce'),
                pd.to_numeric(row[abstraction_col], errors='coerce')
            ]
    
    # Apply 13 Laws analysis
    laws = CAMSLaws(nodes, CAMSNodeSimulator.NODE_NAMES)
    
    # Comprehensive analysis
    analysis_results = {
        'nation': nation or 'Unknown',
        'year': latest_year,
        'raw_nodes': nodes,
        'laws_analysis': {
            'law_1_capacity_stress': laws.law_1_capacity_stress_balance(),
            'law_2_coherence_capacity': laws.law_2_coherence_capacity_coupling(),
            'law_5_abstraction_drift': laws.law_5_abstraction_drift(),
            'law_6_system_fitness': laws.law_6_system_fitness(),
            'law_7_elite_circulation': laws.law_7_elite_circulation(),
            'law_10_innovation': laws.law_10_innovation_threshold(),
            'law_12_metastability': laws.law_12_metastability_detection(),
            'law_13_transformation': laws.law_13_transformation_potential()
        }
    }
    
    # Add bond analysis
    bond_matrix, bond_data = laws.law_8_bond_strength_matrix()
    analysis_results['laws_analysis']['law_8_bond_strength'] = bond_data
    analysis_results['laws_analysis']['law_9_synchronization'] = laws.law_9_synchronization_threshold(bond_matrix)
    analysis_results['laws_analysis']['law_11_stress_cascade'] = laws.law_11_stress_cascade_analysis(bond_matrix)
    
    return analysis_results

if __name__ == "__main__":
    print("Advanced CAMS Framework with 13 Universal Laws")
    print("Module loaded successfully - ready for analysis!")