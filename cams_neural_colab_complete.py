# CAMS Neural Network Analysis Tool for Google Colab - COMPLETE VERSION
# Implements the CAMS-CAN framework for societal dynamics analysis
# Features: Core CAMS equations, visualizations, predictions, policy recommendations, and specialized modules

# Install dependencies
!pip install numpy pandas plotly tensorflow scikit-learn networkx

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import networkx as nx
from google.colab import files
import io
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Core CAMS Equations
def calculate_node_value(coherence, capacity, stress, abstraction):
    """Calculate node value based on CAMS formulation: C + K + |S| + A"""
    return coherence + capacity + abs(stress) + abstraction

def calculate_bond_strength(coherence, capacity, stress, abstraction):
    """Calculate bond strength (simplified as node value * 0.6 for consistency with provided data)"""
    return calculate_node_value(coherence, capacity, stress, abstraction) * 0.6

def calculate_spe(nodes, bond_strengths):
    """Stress Processing Efficiency: Σ(Capacity × BondStrength) / Σ(|Stress| × Abstraction)"""
    if not nodes or not bond_strengths:
        return 0
    
    numerator = 0
    denominator = 0
    
    for node in nodes:
        # Get bond strengths for this node
        node_bonds = [b['strength'] for b in bond_strengths if b['source'] == node['id'] or b['target'] == node['id']]
        avg_bond = np.mean(node_bonds) if node_bonds else 1.0
        
        numerator += node['Capacity'] * avg_bond
        denominator += abs(node['Stress']) * node['Abstraction']
    
    return numerator / denominator if denominator > 0 else 0

def calculate_ns(nodes):
    """Network Synchronization: 1 - σ(Coherence) / μ(Coherence)"""
    if not nodes:
        return 0
    coherence_values = [node['Coherence'] for node in nodes]
    mean = np.mean(coherence_values)
    std = np.std(coherence_values)
    return 1 - (std / mean) if mean > 0 else 0

def calculate_api(nodes, bond_strengths, prev_nodes=None, prev_bond_strengths=None):
    """Adaptive Plasticity Index: Δ(Abstraction) × Δ(BondStrength) / Δ(Stress)²"""
    if prev_nodes is None or prev_bond_strengths is None:
        return 0.15  # Default value
    
    delta_abstraction = sum(node['Abstraction'] - prev_nodes[i]['Abstraction'] for i, node in enumerate(nodes) if i < len(prev_nodes))
    
    current_bond_avg = np.mean([b['strength'] for b in bond_strengths]) if bond_strengths else 0
    prev_bond_avg = np.mean([b['strength'] for b in prev_bond_strengths]) if prev_bond_strengths else 0
    delta_bond_strength = current_bond_avg - prev_bond_avg
    
    delta_stress = sum(abs(node['Stress']) - abs(prev_nodes[i]['Stress']) for i, node in enumerate(nodes) if i < len(prev_nodes))
    
    return (delta_abstraction * delta_bond_strength) / (delta_stress ** 2 + 1e-6) if delta_stress != 0 else 0.15

def calculate_system_health(nodes, bond_strengths, polarization=0):
    """System Health: Σ(w_i × Coherence_i × Capacity_i) / (1 + total_stress)"""
    if not nodes:
        return 0
    
    total_stress = sum(abs(node['Stress']) for node in nodes)
    health = 0
    
    for node in nodes:
        node_bonds = [b['strength'] for b in bond_strengths if b['source'] == node['id'] or b['target'] == node['id']]
        weight = np.mean(node_bonds) if node_bonds else 1.0
        health += node['Coherence'] * node['Capacity'] * weight
    
    return health / (1 + total_stress) if total_stress >= 0 else health

def calculate_coherence_asymmetry(nodes):
    """Coherence Asymmetry: Measures inequality in coherence distribution"""
    if not nodes:
        return 0
    coherence_values = [node['Coherence'] for node in nodes]
    mean_coherence = np.mean(coherence_values)
    return np.std(coherence_values) / mean_coherence if mean_coherence > 0 else 0

def stress_propagation(nodes, bond_strengths, stress_type, intensity):
    """Simulate stress propagation across nodes"""
    updated_nodes = nodes.copy()
    for node in updated_nodes:
        impact = intensity * (1.5 if node['id'] in ['military', 'proletariat'] else 0.7)  # Differential sensitivity
        node['Stress'] = min(10, max(-10, node['Stress'] + impact))
    return updated_nodes

# Data Processing
def load_data():
    """Load and validate CSV data"""
    print("Upload your CSV file (format: Society, Year, Node, Coherence, Capacity, Stress, Abstraction)")
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file uploaded")
    
    df = pd.read_csv(io.BytesIO(list(uploaded.values())[0]))
    required_columns = ['Society', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Calculate Node Value and Bond Strength if not provided
    if 'Node Value' not in df.columns:
        df['Node Value'] = df.apply(lambda row: calculate_node_value(row['Coherence'], row['Capacity'], row['Stress'], row['Abstraction']), axis=1)
    if 'Bond Strength' not in df.columns:
        df['Bond Strength'] = df.apply(lambda row: calculate_bond_strength(row['Coherence'], row['Capacity'], row['Stress'], row['Abstraction']), axis=1)
    
    return df

# Calculate metrics history for all years
def calculate_metrics_history(df):
    """Calculate CAMS metrics for all years in the dataset"""
    years = sorted(df['Year'].unique())
    metrics_history = {}
    prev_nodes = None
    prev_bond_strengths = None
    
    for year in years:
        year_df = df[df['Year'] == year]
        nodes = [{'id': row['Node'], 'Coherence': row['Coherence'], 'Capacity': row['Capacity'], 
                 'Stress': row['Stress'], 'Abstraction': row['Abstraction']}
                for _, row in year_df.iterrows()]
        
        bond_strengths = [{'source': row['Node'], 'target': 'system', 'strength': row['Bond Strength']}
                         for _, row in year_df.iterrows()]
        
        metrics_history[year] = {
            'SPE': calculate_spe(nodes, bond_strengths),
            'NS': calculate_ns(nodes),
            'API': calculate_api(nodes, bond_strengths, prev_nodes, prev_bond_strengths),
            'CA': calculate_coherence_asymmetry(nodes),
            'System Health': calculate_system_health(nodes, bond_strengths)
        }
        
        prev_nodes = nodes.copy()
        prev_bond_strengths = bond_strengths.copy()
    
    return metrics_history

# Visualization Dashboard
def create_dashboard(df, metrics_history, predictions, society):
    """Create a 12-panel interactive dashboard using Plotly"""
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            "System Health Over Time", "Network Synchronization", "Stress Processing Efficiency",
            "Adaptive Plasticity Index", "Coherence Asymmetry", "Stress Variance",
            "Network Graph", "Stress Propagation Heatmap", "Phase Space (C vs S)",
            "Bond Strength Evolution", "Node Coherence Trends", "Early Warning Signals"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    years = sorted(df['Year'].unique())
    
    # 1. System Health
    health_values = [metrics_history[year]['System Health'] for year in years]
    fig.add_trace(go.Scatter(x=years, y=health_values, mode='lines+markers', 
                            name='System Health', line=dict(color='#10B981')), row=1, col=1)

    # 2. Network Synchronization
    ns_values = [metrics_history[year]['NS'] for year in years]
    fig.add_trace(go.Scatter(x=years, y=ns_values, mode='lines+markers', 
                            name='NS', line=dict(color='#3B82F6')), row=1, col=2)

    # 3. Stress Processing Efficiency
    spe_values = [metrics_history[year]['SPE'] for year in years]
    fig.add_trace(go.Scatter(x=years, y=spe_values, mode='lines+markers', 
                            name='SPE', line=dict(color='#F59E0B')), row=1, col=3)

    # 4. Adaptive Plasticity Index
    api_values = [metrics_history[year]['API'] for year in years]
    fig.add_trace(go.Scatter(x=years, y=api_values, mode='lines+markers', 
                            name='API', line=dict(color='#8B5CF6')), row=2, col=1)

    # 5. Coherence Asymmetry
    ca_values = [metrics_history[year]['CA'] for year in years]
    fig.add_trace(go.Scatter(x=years, y=ca_values, mode='lines+markers', 
                            name='CA', line=dict(color='#EF4444')), row=2, col=2)

    # 6. Stress Variance
    stress_variance = [np.var([row['Stress'] for _, row in df[df['Year'] == year].iterrows()]) for year in years]
    fig.add_trace(go.Scatter(x=years, y=stress_variance, mode='lines+markers', 
                            name='Stress Variance', line=dict(color='#DC2626')), row=2, col=3)

    # 7. Network Graph (Latest Year)
    G = nx.Graph()
    latest_year_df = df[df['Year'] == years[-1]]
    nodes = latest_year_df[['Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']].to_dict('records')
    
    for node in nodes:
        G.add_node(node['Node'], size=abs(node['Coherence']) + abs(node['Capacity']))
    
    # Add edges between all nodes (simplified network)
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            bond_strength = (latest_year_df.iloc[i]['Bond Strength'] + latest_year_df.iloc[j]['Bond Strength']) / 2
            G.add_edge(node1['Node'], node2['Node'], weight=bond_strength)
    
    # Position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                            line=dict(width=1, color='#6B7280'), 
                            showlegend=False), row=3, col=1)
    
    # Draw nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_sizes = [G.nodes[node]['size'] * 3 for node in G.nodes()]
    
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', 
                            marker=dict(size=node_sizes, color='#EF4444'),
                            text=list(G.nodes()), textposition="middle center",
                            showlegend=False), row=3, col=1)

    # 8. Stress Propagation Heatmap
    nodes_list = df['Node'].unique()
    heatmap_data = []
    for year in years:
        year_stress = [df[(df['Year'] == year) & (df['Node'] == node)]['Stress'].iloc[0] 
                      if len(df[(df['Year'] == year) & (df['Node'] == node)]) > 0 else 0 
                      for node in nodes_list]
        heatmap_data.append(year_stress)
    
    fig.add_trace(go.Heatmap(z=heatmap_data, x=nodes_list, y=years, 
                            colorscale='RdYlGn_r', showlegend=False), row=3, col=2)

    # 9. Phase Space (Coherence vs Stress)
    latest_year_data = df[df['Year'] == years[-1]]
    fig.add_trace(go.Scatter(x=latest_year_data['Coherence'], y=latest_year_data['Stress'], 
                            mode='markers+text', marker=dict(size=10, color='#9333EA'),
                            text=latest_year_data['Node'], textposition="top center",
                            showlegend=False), row=3, col=3)

    # 10. Bond Strength Evolution
    bond_strengths = [df[df['Year'] == year]['Bond Strength'].mean() for year in years]
    fig.add_trace(go.Scatter(x=years, y=bond_strengths, mode='lines+markers', 
                            name='Bond Strength', line=dict(color='#06B6D4')), row=4, col=1)

    # 11. Node Coherence Trends
    colors = ['#EF4444', '#10B981', '#3B82F6', '#F59E0B', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316']
    for i, node in enumerate(df['Node'].unique()):
        node_data = df[df['Node'] == node]
        fig.add_trace(go.Scatter(x=node_data['Year'], y=node_data['Coherence'], 
                                mode='lines', name=node, 
                                line=dict(color=colors[i % len(colors)])), row=4, col=2)

    # 12. Early Warning Signals
    warning_signals = []
    for year in years:
        warning_score = 0
        if metrics_history[year]['SPE'] < 1.5:
            warning_score += 1
        if metrics_history[year]['NS'] < 0.6:
            warning_score += 1
        if metrics_history[year]['API'] < 0.1:
            warning_score += 1
        if metrics_history[year]['CA'] > 0.4:
            warning_score += 1
        warning_signals.append(warning_score)
    
    fig.add_trace(go.Scatter(x=years, y=warning_signals, mode='lines+markers', 
                            name='Warning Level', line=dict(color='#EF4444', width=3)), row=4, col=3)

    # Layout
    fig.update_layout(height=1400, width=1400, showlegend=True, 
                     title_text=f"{society} CAMS-CAN v3.4 Analysis Dashboard",
                     title_x=0.5, title_font_size=20)
    
    fig.show()
    fig.write_html("cams_dashboard.html")
    files.download("cams_dashboard.html")
    return fig

# Predictive Model
def train_predictive_model(df):
    """Train a simple neural network for 5-year state predictions"""
    years = sorted(df['Year'].unique())
    nodes = sorted(df['Node'].unique())
    
    X, y = [], []
    sequence_length = 3  # Use 3 years to predict next state
    
    for i in range(len(years) - sequence_length):
        # Input: 3 consecutive years of data
        input_sequence = []
        for j in range(sequence_length):
            year_data = df[df['Year'] == years[i + j]].sort_values('Node')
            if len(year_data) == len(nodes):
                year_features = year_data[['Coherence', 'Capacity', 'Stress', 'Abstraction']].values.flatten()
                input_sequence.extend(year_features)
        
        # Output: Next year's data
        next_year_data = df[df['Year'] == years[i + sequence_length]].sort_values('Node')
        if len(next_year_data) == len(nodes) and len(input_sequence) == sequence_length * len(nodes) * 4:
            output_features = next_year_data[['Coherence', 'Capacity', 'Stress', 'Abstraction']].values.flatten()
            X.append(input_sequence)
            y.append(output_features)
    
    if len(X) < 2:
        return None
    
    X, y = np.array(X), np.array(y)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(y.shape[1])
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train with validation split
    history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=1, batch_size=4)
    
    return model, nodes

def predict_future_states(model, df, nodes):
    """Predict future states for 5 years"""
    if model is None:
        return []
    
    years = sorted(df['Year'].unique())
    sequence_length = 3
    
    # Use the last 3 years as input
    input_sequence = []
    for i in range(sequence_length):
        year_data = df[df['Year'] == years[-(sequence_length - i)]].sort_values('Node')
        year_features = year_data[['Coherence', 'Capacity', 'Stress', 'Abstraction']].values.flatten()
        input_sequence.extend(year_features)
    
    predictions = []
    current_input = np.array([input_sequence])
    
    for i in range(1, 6):  # Predict next 5 years
        pred = model.predict(current_input, verbose=0)
        pred_year = years[-1] + i
        
        # Reshape prediction back to node format
        pred_reshaped = pred[0].reshape(-1, 4)  # 4 features per node
        
        node_predictions = []
        for j, node in enumerate(nodes):
            node_predictions.append({
                'id': node,
                'Coherence': float(pred_reshaped[j][0]),
                'Capacity': float(pred_reshaped[j][1]),
                'Stress': float(pred_reshaped[j][2]),
                'Abstraction': float(pred_reshaped[j][3])
            })
        
        predictions.append({
            'Year': pred_year,
            'Nodes': node_predictions
        })
        
        # Update input for next prediction (shift window)
        new_input = current_input[0][len(nodes) * 4:].tolist()  # Remove oldest year
        new_input.extend(pred[0].tolist())  # Add prediction
        current_input = np.array([new_input])
    
    return predictions

# Policy Recommendations
def generate_policy_recommendations(metrics, nodes):
    """Generate policy recommendations based on CAMS metrics"""
    recommendations = []
    
    latest_metrics = metrics[max(metrics.keys())]
    
    if latest_metrics['SPE'] < 1.5:
        recommendations.append("🎯 **Stress Processing**: Increase capacity in high-stress nodes through resource allocation and institutional strengthening.")
    
    if latest_metrics['NS'] < 0.6:
        recommendations.append("🤝 **Network Synchronization**: Enhance inter-institutional coordination and communication to improve network synchronization.")
    
    if latest_metrics['API'] < 0.1:
        recommendations.append("🚀 **Adaptive Plasticity**: Invest in abstraction capacity (education, innovation, R&D) to boost adaptive plasticity.")
    
    if latest_metrics['CA'] > 0.4:
        recommendations.append("⚖️ **Coherence Balance**: Address institutional coherence asymmetry by redistributing resources and trust.")
    
    if latest_metrics['System Health'] < 2.0:
        recommendations.append("🚨 **System Health Critical**: Immediate intervention required - consider emergency stabilization measures.")
    
    # Node-specific recommendations would require node data
    return recommendations

# Cross-civilizational Comparison (COMPLETED)
def cross_civilizational_comparison(dfs, societies):
    """Compare CAMS metrics across multiple societies"""
    comparison_data = []
    
    for society, df in zip(societies, dfs):
        metrics_history = calculate_metrics_history(df)
        years = sorted(df['Year'].unique())
        
        society_metrics = {
            'Society': society,
            'Years': f"{min(years)}-{max(years)}",
            'Avg_System_Health': np.mean([metrics_history[year]['System Health'] for year in years]),
            'Avg_SPE': np.mean([metrics_history[year]['SPE'] for year in years]),
            'Avg_NS': np.mean([metrics_history[year]['NS'] for year in years]),
            'Avg_API': np.mean([metrics_history[year]['API'] for year in years]),
            'Avg_CA': np.mean([metrics_history[year]['CA'] for year in years]),
            'Max_System_Health': max([metrics_history[year]['System Health'] for year in years]),
            'Min_System_Health': min([metrics_history[year]['System Health'] for year in years]),
            'Health_Volatility': np.std([metrics_history[year]['System Health'] for year in years])
        }
        comparison_data.append(society_metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison visualization
    fig = make_subplots(rows=2, cols=3, 
                       subplot_titles=('System Health', 'SPE', 'Network Synchronization',
                                     'Adaptive Plasticity', 'Coherence Asymmetry', 'Health Volatility'))
    
    for i, metric in enumerate(['Avg_System_Health', 'Avg_SPE', 'Avg_NS', 'Avg_API', 'Avg_CA', 'Health_Volatility']):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        fig.add_trace(go.Bar(x=comparison_df['Society'], y=comparison_df[metric], 
                            name=metric, showlegend=False), row=row, col=col)
    
    fig.update_layout(height=800, width=1200, title_text="Cross-Civilizational CAMS Comparison")
    fig.show()
    
    return comparison_df

# Early Warning System
def early_warning_system(metrics_history, df):
    """Implement early warning system for societal instability"""
    years = sorted(metrics_history.keys())
    warnings = {}
    
    for year in years:
        warning_level = 0
        warning_details = []
        
        metrics = metrics_history[year]
        
        # Critical thresholds
        if metrics['System Health'] < 1.5:
            warning_level += 3
            warning_details.append("Critical system health")
        
        if metrics['SPE'] < 1.0:
            warning_level += 2
            warning_details.append("Poor stress processing")
        
        if metrics['NS'] < 0.4:
            warning_level += 2
            warning_details.append("Low network synchronization")
        
        if metrics['CA'] > 0.6:
            warning_level += 2
            warning_details.append("High coherence asymmetry")
        
        # Stress level warnings
        year_df = df[df['Year'] == year]
        max_stress = year_df['Stress'].abs().max()
        if max_stress > 8:
            warning_level += 2
            warning_details.append(f"Extreme stress level: {max_stress:.1f}")
        
        # Trend warnings (if we have previous years)
        if len([y for y in years if y < year]) >= 2:
            prev_years = [y for y in years if y < year][-2:]
            health_trend = [metrics_history[y]['System Health'] for y in prev_years] + [metrics['System Health']]
            if all(health_trend[i] > health_trend[i+1] for i in range(len(health_trend)-1)):
                warning_level += 1
                warning_details.append("Declining health trend")
        
        warnings[year] = {
            'level': min(warning_level, 5),  # Cap at 5
            'details': warning_details,
            'status': 'CRITICAL' if warning_level >= 5 else 'WARNING' if warning_level >= 3 else 'STABLE'
        }
    
    return warnings

# Main Analysis Function
def run_cams_analysis():
    """Main function to run complete CAMS analysis"""
    print("🧠 CAMS-CAN v3.4 Neural Network Analysis Tool")
    print("=" * 50)
    
    # Load data
    df = load_data()
    society = df['Society'].iloc[0]
    
    print(f"Analyzing {society}...")
    print(f"Data range: {df['Year'].min()}-{df['Year'].max()}")
    print(f"Nodes: {', '.join(df['Node'].unique())}")
    
    # Calculate metrics
    metrics_history = calculate_metrics_history(df)
    
    # Train predictive model
    print("Training predictive model...")
    model_result = train_predictive_model(df)
    if model_result:
        model, nodes = model_result
        predictions = predict_future_states(model, df, nodes)
        print(f"Generated predictions for {len(predictions)} future years")
    else:
        predictions = []
        print("Insufficient data for predictive modeling")
    
    # Generate dashboard
    print("Creating interactive dashboard...")
    dashboard = create_dashboard(df, metrics_history, predictions, society)
    
    # Policy recommendations
    recommendations = generate_policy_recommendations(metrics_history, [])
    print("\n📋 Policy Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    # Early warning analysis
    warnings = early_warning_system(metrics_history, df)
    latest_year = max(warnings.keys())
    latest_warning = warnings[latest_year]
    
    print(f"\n⚠️ Current Status: {latest_warning['status']}")
    print(f"Warning Level: {latest_warning['level']}/5")
    if latest_warning['details']:
        print("Concerns:")
        for detail in latest_warning['details']:
            print(f"  • {detail}")
    
    # Summary statistics
    latest_metrics = metrics_history[max(metrics_history.keys())]
    print(f"\n📊 Latest Metrics Summary:")
    print(f"  System Health: {latest_metrics['System Health']:.2f}")
    print(f"  Stress Processing Efficiency: {latest_metrics['SPE']:.2f}")
    print(f"  Network Synchronization: {latest_metrics['NS']:.2f}")
    print(f"  Adaptive Plasticity: {latest_metrics['API']:.2f}")
    print(f"  Coherence Asymmetry: {latest_metrics['CA']:.2f}")
    
    return {
        'df': df,
        'metrics_history': metrics_history,
        'predictions': predictions,
        'recommendations': recommendations,
        'warnings': warnings,
        'dashboard': dashboard
    }

# Run the analysis
if __name__ == "__main__":
    try:
        results = run_cams_analysis()
        print("\n✅ Analysis completed successfully!")
        print("Dashboard saved as 'cams_dashboard.html'")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your data format and try again.")