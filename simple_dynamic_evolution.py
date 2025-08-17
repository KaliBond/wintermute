"""
Simple Dynamic Evolution Simulation that actually works
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import glob

st.set_page_config(page_title="CAMS Dynamic Evolution", layout="wide")

st.title("📈 CAMS Dynamic Evolution Simulation")
st.success("✅ Simplified version guaranteed to produce results!")

# Load datasets
@st.cache_data
def load_datasets():
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
        'iraq cams cleaned': 'Iraq'
    }
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0 and 'Node' in df.columns and 'Coherence' in df.columns:
                base_name = file.replace('.csv', '').lower().replace('_', ' ')
                country_name = country_mapping.get(base_name, base_name.title())
                datasets[country_name] = {
                    'data': df,
                    'records': len(df),
                    'years': f"{int(df['Year'].min())}-{int(df['Year'].max())}" if 'Year' in df.columns else 'Current'
                }
        except:
            continue
    
    return datasets

datasets = load_datasets()

# Selection interface
col1, col2 = st.columns(2)

with col1:
    if datasets:
        selected_country = st.selectbox("Select Civilization:", list(datasets.keys()))
    else:
        st.error("No datasets found")
        st.stop()

with col2:
    info = datasets[selected_country]
    st.info(f"**{selected_country}**\nRecords: {info['records']}\nPeriod: {info['years']}")

# Load and prepare data
country_data = datasets[selected_country]['data'].copy()

# Get latest data
if 'Year' in country_data.columns:
    latest_year = country_data['Year'].max()
    current_data = country_data[country_data['Year'] == latest_year].copy()
else:
    current_data = country_data.copy()

# Clean data
for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
    if col in current_data.columns:
        current_data[col] = pd.to_numeric(current_data[col], errors='coerce')

current_data = current_data.dropna(subset=['Coherence', 'Capacity', 'Stress', 'Abstraction'])

st.success(f"✅ Loaded {len(current_data)} nodes from {selected_country}")

# Simulation parameters
st.markdown("### ⚙️ Simulation Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    sim_duration = st.slider("Duration (years)", 5, 30, 15)

with col2:
    external_stress = st.slider("External Stress", 0.0, 2.0, 0.5, 0.1)

with col3:
    volatility = st.slider("Volatility", 0.0, 1.0, 0.3, 0.1)

# Run simulation button
if st.button("🚀 RUN DYNAMIC EVOLUTION", type="primary", use_container_width=True):
    
    st.markdown("## 🎯 SIMULATION RUNNING...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate initial system state from real data
    initial_fitness = []
    initial_stress = []
    initial_coherence = []
    
    for _, row in current_data.iterrows():
        C = row['Coherence']
        K = row['Capacity']  
        S = row['Stress']
        A = row['Abstraction']
        
        # Simple fitness calculation
        fitness = (C * K) / (1 + abs(S)) * (1 + A/10)
        initial_fitness.append(fitness)
        initial_stress.append(abs(S))
        initial_coherence.append(C)
    
    # Calculate initial system metrics
    safe_fitness = np.clip(initial_fitness, 1e-6, None)
    initial_health = np.exp(np.mean(np.log(safe_fitness)))
    
    coherence_capacity = np.array(initial_coherence) * current_data['Capacity'].values
    initial_ca = np.std(coherence_capacity) / (np.mean(coherence_capacity) + 1e-9)
    
    initial_risk = initial_ca / (1 + initial_health)
    initial_spe = np.sum(initial_fitness) / (np.sum(initial_stress) + 1e-9)
    
    # Run simulation
    results = []
    node_evolution = {node: [] for node in current_data['Node'].values}
    
    for step in range(sim_duration):
        progress = (step + 1) / sim_duration
        progress_bar.progress(progress)
        status_text.text(f"Simulating {selected_country} - Year {step + 1}/{sim_duration}")
        
        # Apply external stress and evolution
        stress_impact = external_stress * np.random.normal(0, 0.2)
        volatility_impact = volatility * np.random.normal(0, 0.1)
        
        # System health evolution
        health_decline = step * 0.01  # Gradual decline
        year_health = initial_health * (1 - health_decline) + stress_impact
        year_health = max(0.1, year_health)
        
        # Risk evolution
        year_risk = initial_risk + (step * 0.005) + abs(stress_impact) * 0.1
        year_risk = max(0, min(1, year_risk))
        
        # Processing efficiency
        year_spe = initial_spe * (1 - external_stress * 0.05 - step * 0.005)
        year_spe = max(0.1, year_spe)
        
        # Determine phase attractor
        if year_health > 3.5 and year_risk < 0.3:
            attractor = "Adaptive"
        elif 2.5 <= year_health <= 3.5:
            attractor = "Authoritarian" 
        elif 1.5 <= year_health <= 2.5:
            attractor = "Fragmented"
        else:
            attractor = "Collapse"
        
        # Store results
        results.append({
            'Year': step + 1,
            'SystemHealth': year_health,
            'RiskIndex': year_risk,
            'ProcessingEfficiency': year_spe,
            'PhaseAttractor': attractor,
            'ExternalStress': external_stress + volatility_impact
        })
        
        # Evolve individual nodes
        for i, (_, row) in enumerate(current_data.iterrows()):
            node_name = row['Node']
            base_fitness = initial_fitness[i]
            
            # Apply stress and evolution to node
            node_stress_effect = stress_impact + volatility_impact
            evolved_fitness = base_fitness * (1 - step * 0.008) + node_stress_effect
            evolved_fitness = max(0.1, evolved_fitness)
            
            node_evolution[node_name].append({
                'Year': step + 1,
                'Fitness': evolved_fitness,
                'Stress': initial_stress[i] + abs(node_stress_effect)
            })
        
        time.sleep(0.1)  # Visible progress
    
    progress_bar.progress(1.0)
    status_text.text("✅ Simulation Complete!")
    
    # Convert results
    results_df = pd.DataFrame(results)
    
    st.markdown("# 🎉 DYNAMIC EVOLUTION RESULTS")
    st.success(f"Successfully simulated {sim_duration} years of {selected_country} evolution!")
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    initial_health_val = results_df['SystemHealth'].iloc[0]
    final_health_val = results_df['SystemHealth'].iloc[-1]
    health_change = final_health_val - initial_health_val
    
    with col1:
        st.metric("Final System Health", f"{final_health_val:.2f}", 
                 delta=f"{health_change:+.2f}")
    
    with col2:
        peak_risk = results_df['RiskIndex'].max()
        st.metric("Peak Risk", f"{peak_risk:.3f}")
    
    with col3:
        avg_spe = results_df['ProcessingEfficiency'].mean()
        st.metric("Avg Processing Efficiency", f"{avg_spe:.2f}")
    
    with col4:
        final_attractor = results_df['PhaseAttractor'].iloc[-1]
        attractor_icons = {"Adaptive": "🟢", "Authoritarian": "🟡", "Fragmented": "🟠", "Collapse": "🔴"}
        st.metric("Final Attractor", f"{attractor_icons.get(final_attractor, '⚪')} {final_attractor}")
    
    # Create comprehensive plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'System Health Evolution',
            'Risk Index & Processing Efficiency', 
            'Phase Attractor Timeline',
            'Individual Node Fitness'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # System Health
    fig.add_trace(
        go.Scatter(x=results_df['Year'], y=results_df['SystemHealth'],
                  name='System Health', line=dict(color='green', width=3)),
        row=1, col=1
    )
    
    # Risk and SPE
    fig.add_trace(
        go.Scatter(x=results_df['Year'], y=results_df['RiskIndex'],
                  name='Risk Index', line=dict(color='red', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=results_df['Year'], y=results_df['ProcessingEfficiency'],
                  name='Processing Efficiency', line=dict(color='blue', width=2)),
        row=1, col=2, secondary_y=True
    )
    
    # Phase attractors
    attractor_numeric = {'Adaptive': 4, 'Authoritarian': 3, 'Fragmented': 2, 'Collapse': 1}
    y_attractors = [attractor_numeric[a] for a in results_df['PhaseAttractor']]
    
    fig.add_trace(
        go.Scatter(x=results_df['Year'], y=y_attractors, mode='markers+lines',
                  name='Phase Attractor', marker=dict(size=8)),
        row=2, col=1
    )
    
    # Node fitness evolution (show top 4 nodes)
    top_nodes = list(node_evolution.keys())[:4]
    for node_name in top_nodes:
        node_data = node_evolution[node_name]
        years = [d['Year'] for d in node_data]
        fitness = [d['Fitness'] for d in node_data]
        
        fig.add_trace(
            go.Scatter(x=years, y=fitness, name=node_name, mode='lines'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text=f"{selected_country} - Dynamic Evolution Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.markdown("### 📊 Evolution Analysis")
    
    trend_analysis = "Improving" if health_change > 0.1 else "Declining" if health_change < -0.1 else "Stable"
    risk_trend = "Increasing" if results_df['RiskIndex'].iloc[-1] > results_df['RiskIndex'].iloc[0] else "Decreasing"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Health Trajectory**")
        st.write(f"Trend: {trend_analysis}")
        st.write(f"Change: {health_change:+.2f}")
        st.write(f"Volatility: {results_df['SystemHealth'].std():.2f}")
    
    with col2:
        st.markdown("**Risk Assessment**") 
        st.write(f"Trend: {risk_trend}")
        st.write(f"Peak Risk: {peak_risk:.3f}")
        st.write(f"Final Risk: {results_df['RiskIndex'].iloc[-1]:.3f}")
    
    with col3:
        st.markdown("**Attractor Analysis**")
        attractor_counts = results_df['PhaseAttractor'].value_counts()
        dominant = attractor_counts.index[0]
        st.write(f"Dominant: {dominant}")
        st.write(f"Transitions: {len(results_df) - len(results_df[results_df['PhaseAttractor'] == results_df['PhaseAttractor'].shift(1)])}")
    
    # Data table
    st.markdown("### 📋 Complete Results")
    st.dataframe(results_df, use_container_width=True)
    
    st.balloons()
    st.success("🎉 Dynamic Evolution Simulation Complete!")

else:
    st.info("👆 Configure parameters above and click 'RUN DYNAMIC EVOLUTION' to start simulation")

# Show current data preview
st.markdown("### 👀 Current System State")
st.dataframe(current_data[['Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']].head(8), 
            use_container_width=True)

st.markdown("---")
st.markdown("**Status:** ✅ Simplified Dynamic Evolution | **Guaranteed Results**")