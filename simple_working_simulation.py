"""
Simple Working CAMS Stress Dynamics Simulation
Guaranteed to display results
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="CAMS Simple", layout="wide")

st.title("🧠 CAMS Stress Dynamics - Working Version")
st.success("✅ Application loaded successfully!")

# Load available datasets
@st.cache_data
def load_available_datasets():
    """Load all available CAMS datasets"""
    import glob
    csv_files = glob.glob("*.csv")
    datasets = {}
    
    country_mapping = {
        'australia cams cleaned': 'Australia',
        'usa cams cleaned': 'USA',
        'france cams cleaned': 'France',
        'italy cams cleaned': 'Italy',
        'germany1750 2025': 'Germany',
        'denmark cams cleaned': 'Denmark',
        'iran cams cleaned': 'Iran',
        'iraq cams cleaned': 'Iraq',
        'lebanon cams cleaned': 'Lebanon',
        'japan 1850 2025': 'Japan',
        'thailand 1850_2025': 'Thailand',
        'netherlands mastersheet': 'Netherlands',
        'canada_cams_2025': 'Canada'
    }
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0 and 'Node' in df.columns and 'Coherence' in df.columns:
                base_name = file.replace('.csv', '').replace('.CSV', '').lower().strip()
                base_name = base_name.replace('_', ' ').replace(' (2)', '').replace(' - ', ' ')
                
                country_name = country_mapping.get(base_name, base_name.title())
                
                datasets[country_name] = {
                    'filename': file,
                    'data': df,
                    'records': len(df),
                    'years': f"{int(df['Year'].min())}-{int(df['Year'].max())}" if 'Year' in df.columns else 'Unknown'
                }
        except:
            continue
    
    return datasets

# Load datasets
datasets = load_available_datasets()

# Nation selection
st.markdown("### 🌍 Select Nation for Analysis")

if datasets:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_country = st.selectbox(
            "Choose Civilization:",
            options=list(datasets.keys()),
            help="Select a nation to analyze using real CAMS data"
        )
    
    with col2:
        if selected_country:
            info = datasets[selected_country]
            st.info(f"**{selected_country}**\nRecords: {info['records']}\nPeriod: {info['years']}")
    
    # Load selected country data
    country_data = datasets[selected_country]['data'].copy()
    
    # Prepare data for analysis
    if 'Year' in country_data.columns:
        latest_year = country_data['Year'].max()
        df = country_data[country_data['Year'] == latest_year].copy()
    else:
        df = country_data.copy()
    
    # Standardize column names
    df = df.rename(columns={
        'Coherence': 'Coherence',
        'Capacity': 'Capacity', 
        'Stress': 'Stress',
        'Abstraction': 'Abstraction'
    })
    
    # Ensure numeric values
    for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Coherence', 'Capacity', 'Stress', 'Abstraction'])
    
    # Calculate fitness
    df['Fitness'] = (df['Coherence'] * df['Capacity']) / (1 + np.abs(df['Stress'])) * (1 + df['Abstraction']/10)
    
    st.success(f"✅ Loaded {len(df)} institutional nodes from {selected_country}")
    
else:
    st.warning("No CAMS datasets found. Using synthetic data.")
    selected_country = "Demo Nation"
    
    # Generate simple data as fallback
    np.random.seed(42)
    nodes = ['Executive', 'Army', 'Archive', 'Lore']
    
    data = []
    for node in nodes:
        C = 5 + np.random.normal(0, 1)
        K = 5 + np.random.normal(0, 1)  
        S = np.random.normal(0, 2)
        A = 5 + np.random.normal(0, 0.5)
        
        fitness = (C * K) / (1 + abs(S)) * (1 + A/10)
        
        data.append({
            'Node': node,
            'Coherence': C,
            'Capacity': K, 
            'Stress': S,
            'Abstraction': A,
            'Fitness': fitness
        })
    
    df = pd.DataFrame(data)

st.success(f"✅ Generated {len(df)} data points")

# Parameters
col1, col2 = st.columns(2)

with col1:
    duration = st.slider("Simulation Duration", 5, 15, 10)
    
with col2: 
    stress_level = st.slider("External Stress", 0.0, 2.0, 0.5)

# Big obvious button
if st.button("🚀 RUN SIMULATION NOW", type="primary", use_container_width=True):
    
    st.markdown(f"## 🎯 SIMULATION RUNNING - {selected_country}")
    
    # Progress bar that actually works
    progress = st.progress(0)
    status = st.empty()
    
    # Calculate initial system health from real data
    initial_fitness = df['Fitness'].values
    safe_fitness = np.clip(initial_fitness, 1e-6, None)
    initial_health = np.exp(np.mean(np.log(safe_fitness)))
    
    # Calculate initial coherence asymmetry
    coherence_capacity = df['Coherence'] * df['Capacity']
    initial_ca = np.std(coherence_capacity) / (np.mean(coherence_capacity) + 1e-9)
    
    initial_risk = initial_ca / (1 + initial_health)
    initial_spe = np.sum(initial_fitness) / (np.sum(np.abs(df['Stress']) * df['Abstraction']) + 1e-9)
    
    # Simulate evolution based on real data
    results = []
    
    for step in range(duration):
        # Update progress
        progress.progress((step + 1) / duration)
        status.text(f"Simulating {selected_country} - Step {step + 1}/{duration}...")
        
        # Evolve system based on initial conditions and external stress
        stress_effect = stress_level * np.random.normal(0, 0.3)
        
        # Health evolution (tends toward initial value with perturbations)
        year_health = initial_health * (1 - step * 0.02) + stress_effect
        year_health = max(0.1, year_health)
        
        # Risk evolution (increases with stress and time)
        year_risk = initial_risk + (step * 0.01) + (stress_level * 0.05)
        year_risk = max(0, min(1, year_risk))
        
        # Processing efficiency (declines under stress)
        year_spe = initial_spe * (1 - stress_level * 0.1 - step * 0.01)
        year_spe = max(0.1, year_spe)
        
        results.append({
            'Step': step + 1,
            'SystemHealth': year_health,
            'RiskIndex': year_risk,
            'ProcessingEfficiency': year_spe
        })
        
        time.sleep(0.15)  # Visible progress
    
    status.text("✅ Simulation Complete!")
    results_df = pd.DataFrame(results)
    
    # RESULTS SECTION - BIG AND OBVIOUS
    st.markdown("# 🎉 SIMULATION RESULTS")
    st.success("Results generated successfully!")
    
    # Metrics in big boxes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        final_health = results_df['SystemHealth'].iloc[-1]
        st.metric("Final System Health", f"{final_health:.2f}", 
                 delta=f"{final_health - results_df['SystemHealth'].iloc[0]:.2f}")
    
    with col2:
        peak_risk = results_df['RiskIndex'].max()
        st.metric("Peak Risk", f"{peak_risk:.2f}")
        
    with col3:
        avg_efficiency = results_df['ProcessingEfficiency'].mean()
        st.metric("Avg Efficiency", f"{avg_efficiency:.2f}")
        
    with col4:
        health_volatility = results_df['SystemHealth'].std()
        st.metric("Health Volatility", f"{health_volatility:.2f}")
    
    # Simple matplotlib plots (more reliable than Plotly)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('CAMS Simulation Results', fontsize=16, fontweight='bold')
    
    # System Health
    axes[0,0].plot(results_df['Step'], results_df['SystemHealth'], 'g-', linewidth=3)
    axes[0,0].set_title('System Health Evolution')
    axes[0,0].set_ylabel('Health')
    axes[0,0].grid(True)
    
    # Risk Index  
    axes[0,1].plot(results_df['Step'], results_df['RiskIndex'], 'r-', linewidth=3)
    axes[0,1].axhline(y=0.5, color='red', linestyle='--', label='Critical Risk')
    axes[0,1].set_title('Risk Index Evolution')
    axes[0,1].set_ylabel('Risk')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # Processing Efficiency
    axes[1,0].plot(results_df['Step'], results_df['ProcessingEfficiency'], 'b-', linewidth=3)
    axes[1,0].set_title('Processing Efficiency')
    axes[1,0].set_xlabel('Simulation Step')
    axes[1,0].set_ylabel('Efficiency')
    axes[1,0].grid(True)
    
    # Phase diagram
    axes[1,1].scatter(results_df['SystemHealth'], results_df['RiskIndex'], 
                     c=results_df['Step'], cmap='viridis', s=50)
    axes[1,1].set_title('Phase Diagram: Health vs Risk')
    axes[1,1].set_xlabel('System Health')
    axes[1,1].set_ylabel('Risk Index')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Data table
    st.subheader("📊 Detailed Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Analysis summary
    st.subheader("🎯 Analysis Summary")
    
    # Determine trends
    health_trend = "Declining" if final_health < results_df['SystemHealth'].iloc[0] else "Improving"
    risk_trend = "Increasing" if results_df['RiskIndex'].iloc[-1] > results_df['RiskIndex'].iloc[0] else "Decreasing"
    
    st.write(f"**Health Trend:** {health_trend}")
    st.write(f"**Risk Trend:** {risk_trend}")
    st.write(f"**Simulation Duration:** {duration} steps")
    st.write(f"**External Stress Level:** {stress_level}")
    
    # Success celebration
    st.balloons()
    st.success("🎉 Simulation completed successfully! All results displayed above.")

else:
    # Show preview data while waiting
    st.subheader("📋 Data Preview")
    st.write("Sample data that will be used in simulation:")
    st.dataframe(df.head(8))
    
    st.info("👆 Click the big blue button above to run the simulation!")

# Footer
st.markdown("---")
st.markdown("**Status:** ✅ All systems operational | **Framework:** CAMS-CAN v3.4")