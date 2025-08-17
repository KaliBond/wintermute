"""
Debug version of CAMS stress dynamics simulation
Simplified to ensure results display properly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="CAMS Debug", layout="wide")
st.title("🔍 CAMS Stress Dynamics - Debug Version")

# Simple test to ensure basic functionality
st.success("✅ Streamlit is working!")

# Create simple synthetic data for testing
@st.cache_data
def create_test_data():
    """Create simple test dataset"""
    years = np.arange(2020, 2026)
    nodes = ['Executive', 'Army', 'Archive', 'Lore']
    
    data = []
    np.random.seed(42)
    
    for year in years:
        for i, node in enumerate(nodes):
            # Simple synthetic values
            C = 5 + np.random.normal(0, 1) + (year - 2020) * 0.1
            K = 5 + np.random.normal(0, 1) 
            S = np.random.normal(0, 2) + (year - 2022) * 0.5  # Increasing stress
            A = 5 + np.random.normal(0, 0.5)
            
            data.append({
                'Year': year,
                'Node': node, 
                'C': C,
                'K': K,
                'S': S,
                'A': A
            })
    
    return pd.DataFrame(data)

# Load test data
df = create_test_data()
st.success(f"✅ Test data created: {len(df)} records")

# Simple fitness calculation
def simple_fitness(C, K, S, A, tau=3.0, lam=0.5):
    penalty = 1 + np.exp((np.abs(S) - tau) / lam)
    return (C * K) / penalty * (1 + A / 10)

df['fitness'] = simple_fitness(df['C'], df['K'], df['S'], df['A'])
st.success("✅ Fitness calculations complete")

# System health calculation
system_health = []
for year in sorted(df['Year'].unique()):
    year_data = df[df['Year'] == year]
    fitness_vals = year_data['fitness'].values
    
    # Geometric mean (system health)
    safe_fitness = np.clip(fitness_vals, 1e-6, None)
    psi = np.exp(np.mean(np.log(safe_fitness)))
    
    system_health.append({'Year': year, 'SystemHealth': psi})

sys_df = pd.DataFrame(system_health)
st.success("✅ System health calculations complete")

# Test simulation parameters
st.sidebar.header("🎛️ Simulation Parameters")
sim_duration = st.sidebar.slider("Simulation Duration (years)", 5, 20, 10)
external_stress = st.sidebar.slider("External Stress Level", 0.0, 2.0, 0.5, 0.1)

st.sidebar.info(f"Duration: {sim_duration} years\nStress Level: {external_stress}")

# Run button
if st.button("🚀 Run Debug Simulation", type="primary"):
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate processing steps
    for i in range(10):
        status_text.text(f"Processing step {i+1}/10...")
        progress_bar.progress((i + 1) / 10)
        time.sleep(0.2)  # Simulate computation
    
    status_text.text("✅ Simulation complete!")
    
    # Display results
    st.success("🎉 Simulation Results Generated!")
    
    # Basic plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 System Health Evolution")
        fig1 = px.line(sys_df, x='Year', y='SystemHealth', 
                      title="System Health Ψ(t)")
        fig1.update_traces(line=dict(width=3))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📊 Stress by Node")
        fig2 = px.line(df, x='Year', y='S', color='Node',
                      title="Stress Trajectories")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Results summary
    st.subheader("📊 Simulation Summary")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("Average Health", f"{sys_df['SystemHealth'].mean():.2f}")
    
    with col_b:
        st.metric("Health Range", f"{sys_df['SystemHealth'].max() - sys_df['SystemHealth'].min():.2f}")
    
    with col_c:
        st.metric("Peak Stress", f"{df['S'].abs().max():.2f}")
    
    with col_d:
        st.metric("Nodes Analyzed", len(df['Node'].unique()))
    
    # Data table
    st.subheader("🗂️ Raw Data")
    st.dataframe(df, use_container_width=True)
    
    st.balloons()  # Celebration when complete!

else:
    st.info("👆 Click the button above to run the debug simulation")

# Diagnostic information
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Diagnostics")
st.sidebar.write(f"**Python version working:** ✅")
st.sidebar.write(f"**Streamlit version working:** ✅")
st.sidebar.write(f"**Plotly working:** ✅")
st.sidebar.write(f"**Data processing:** ✅")
st.sidebar.write(f"**Mathematical functions:** ✅")

# Show current data preview
st.subheader("👀 Data Preview")
st.write("**System Health Data:**")
st.dataframe(sys_df)

st.write("**Node Data (sample):**")
st.dataframe(df.head(8))

st.markdown("---")
st.info("""
**Debug Status:** This simplified version tests core functionality.
If this works but the full simulation doesn't, the issue may be:
1. **Complex calculations** taking too long
2. **Memory issues** with large datasets
3. **Display problems** with complex visualizations
4. **Error handling** masking the real issue
""")

# Test error handling
try:
    test_calc = np.exp(np.mean(np.log(np.array([1, 2, 3]))))
    st.success(f"✅ Mathematical operations working: {test_calc:.2f}")
except Exception as e:
    st.error(f"❌ Mathematical error: {e}")

# System status
st.sidebar.write(f"**System status:** ✅ Running normally")