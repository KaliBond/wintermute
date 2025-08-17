"""
⚠️ Iran Phase Transition Analysis - WORKING VERSION
Critical Analysis of Iranian Civilizational Stress Dynamics
Using Real CAMS Data and Formal Mathematical Framework
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="⚠️ Iran Phase Transition Analysis", 
    layout="wide"
)

# Header with alert styling
st.markdown("""
<div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
           padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;">
    <h1 style="margin: 0; font-size: 2.5rem;">⚠️ PHASE TRANSITION ANALYSIS</h1>
    <h2 style="margin: 0.5rem 0 0 0; font-size: 1.8rem;">📍 IRAN</h2>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Critical Assessment Using Real CAMS Data
    </p>
</div>
""", unsafe_allow_html=True)

# === Load and Process Iran Data ===
@st.cache_data
def load_iran_data():
    """Load Iran CAMS data"""
    try:
        # Try to load Iran data
        df = pd.read_csv("Iran_CAMS_Cleaned.csv")
        
        # Clean and process
        if len(df) > 0:
            # Ensure numeric columns
            for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing data
            df = df.dropna(subset=['Coherence', 'Capacity', 'Stress', 'Abstraction'])
            
            # Get latest year data
            if 'Year' in df.columns:
                latest_year = df['Year'].max()
                latest_data = df[df['Year'] == latest_year].copy()
                
                st.success(f"✅ Loaded Iran CAMS data | Latest Year: {latest_year} | {len(latest_data)} institutions")
                return latest_data, df
            else:
                st.success(f"✅ Loaded Iran CAMS data | {len(df)} institutions")
                return df, df
                
    except FileNotFoundError:
        st.error("❌ Iran_CAMS_Cleaned.csv not found!")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading Iran data: {str(e)}")
        return None, None

# Load data
with st.spinner("🔄 Loading Iran CAMS data..."):
    latest_data, full_data = load_iran_data()

if latest_data is None:
    st.stop()

# === Core CAMS Analysis Functions ===
def calculate_node_fitness(row, tau=3.0, lambda_param=0.5):
    """Calculate node fitness H_i(t)"""
    C, K, S, A = row['Coherence'], row['Capacity'], row['Stress'], row['Abstraction']
    
    # Stress impact calculation
    stress_impact = 1 + np.exp((abs(S) - tau) / lambda_param)
    
    # Node fitness
    fitness = (C * K / stress_impact) * (1 + A / 10)
    return max(fitness, 1e-6)  # Ensure positive

def calculate_system_health(df):
    """Calculate system health Ω(S,t) - geometric mean of fitness"""
    fitness_values = []
    for _, row in df.iterrows():
        fitness = calculate_node_fitness(row)
        fitness_values.append(fitness)
    
    # Geometric mean
    return np.exp(np.mean(np.log(fitness_values)))

def calculate_processing_efficiency(df):
    """Calculate SPE(t)"""
    total_capacity = df['Capacity'].sum()
    total_stress = df['Stress'].abs().sum() + 1e-6
    total_abstraction = df['Abstraction'].sum()
    
    return total_capacity / (total_stress * total_abstraction) * 10

def calculate_coherence_asymmetry(df):
    """Calculate coherence asymmetry CA(t)"""
    coherence_capacity = df['Coherence'] * df['Capacity']
    return np.std(coherence_capacity) / (np.mean(coherence_capacity) + 1e-9)

def identify_phase_attractor(health, spe, ca, control_ratio=1.0):
    """Identify current phase attractor"""
    
    # Attractor classification logic
    if health > 3.5 and spe > 2.0 and ca < 0.3:
        return "🟢 A₁ (Adaptive)", "green", {
            "description": "High stress processing efficiency with coordinated response",
            "stability": "HIGH",
            "recommendation": "Maintain current institutional coordination"
        }
    elif 2.5 <= health <= 3.5 and control_ratio > 1.5:
        return "🟡 A₂ (Authoritarian)", "orange", {
            "description": "Centralized stress processing with reduced monitoring",
            "stability": "MODERATE",
            "recommendation": "Balance control with monitoring capabilities"
        }
    elif 1.5 <= health <= 2.5 and ca > 0.4 and spe < 1.0:
        return "🟠 A₃ (Fragmented)", "red", {
            "description": "Distributed but inefficient stress processing",
            "stability": "LOW",
            "recommendation": "Urgent: Restore institutional coherence"
        }
    else:
        return "🔴 A₄ (Collapse)", "darkred", {
            "description": "Meta-cognitive breakdown and system failure",
            "stability": "CRITICAL",
            "recommendation": "IMMEDIATE INTERVENTION REQUIRED"
        }

# === Calculate Current System Metrics ===
current_health = calculate_system_health(latest_data)
current_spe = calculate_processing_efficiency(latest_data)
current_ca = calculate_coherence_asymmetry(latest_data)

# Mock control ratio (would need time series for real calculation)
control_ratio = 1.2  # Slightly authoritarian tendency

# Identify current attractor
current_attractor, attractor_color, attractor_info = identify_phase_attractor(
    current_health, current_spe, current_ca, control_ratio
)

# === Main Dashboard ===
st.markdown("## 🚨 Current System Status")

# Threat level assessment
if current_health < 1.5:
    threat_level = "🔴 CRITICAL"
    threat_bg = "#ff4444"
elif current_health < 2.5:
    threat_level = "🟠 HIGH RISK"  
    threat_bg = "#ff8800"
elif current_health < 3.5:
    threat_level = "🟡 ELEVATED"
    threat_bg = "#ffaa00"
else:
    threat_level = "🟢 STABLE"
    threat_bg = "#00aa00"

# Current status display
st.markdown(f"""
<div style="background: {threat_bg}33; padding: 1.5rem; border-radius: 0.75rem; 
           border-left: 5px solid {threat_bg}; margin: 1rem 0;">
    <h2 style="margin: 0; color: {threat_bg};">THREAT LEVEL: {threat_level}</h2>
    <h3 style="margin: 0.5rem 0 0 0; color: {attractor_color};">{current_attractor}</h3>
    <p style="margin: 0.5rem 0 0 0;"><strong>Description:</strong> {attractor_info['description']}</p>
    <p style="margin: 0.25rem 0 0 0;"><strong>Stability:</strong> {attractor_info['stability']}</p>
</div>
""", unsafe_allow_html=True)

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    health_delta = current_health - 2.5  # Distance from fragmentation threshold
    st.metric(
        "System Health Ω(S,t)", 
        f"{current_health:.3f}",
        delta=f"Fragmentation: {health_delta:+.3f}"
    )

with col2:
    st.metric(
        "Processing Efficiency SPE(t)", 
        f"{current_spe:.3f}",
        delta="Efficiency status"
    )

with col3:
    st.metric(
        "Coherence Asymmetry CA(t)", 
        f"{current_ca:.3f}",
        delta="Fragmentation index"
    )

with col4:
    # Calculate nodes at high stress
    high_stress_nodes = len(latest_data[latest_data['Stress'].abs() > 3.0])
    st.metric(
        "High Stress Institutions", 
        f"{high_stress_nodes}/{len(latest_data)}",
        delta="Critical threshold: >3.0"
    )

# === Phase Space Visualization ===
st.markdown("## 🌀 Phase Space Analysis")

fig_phase = go.Figure()

# Add attractor regions
# A1 - Adaptive (green zone)
fig_phase.add_shape(
    type="rect", x0=3.5, y0=2.0, x1=6, y1=5,
    fillcolor="green", opacity=0.15, line=dict(color="green", width=2)
)
fig_phase.add_annotation(x=4.7, y=3.5, text="A₁<br>ADAPTIVE", showarrow=False, 
                        font=dict(color="green", size=14, family="Arial Black"))

# A2 - Authoritarian (yellow zone)
fig_phase.add_shape(
    type="rect", x0=2.5, y0=0.5, x1=3.5, y1=3.5,
    fillcolor="orange", opacity=0.15, line=dict(color="orange", width=2)
)
fig_phase.add_annotation(x=3.0, y=2.0, text="A₂<br>AUTH", showarrow=False,
                        font=dict(color="orange", size=14, family="Arial Black"))

# A3 - Fragmented (red zone)
fig_phase.add_shape(
    type="rect", x0=1.5, y0=0, x1=2.5, y1=1.0,
    fillcolor="red", opacity=0.15, line=dict(color="red", width=2)
)
fig_phase.add_annotation(x=2.0, y=0.5, text="A₃<br>FRAG", showarrow=False,
                        font=dict(color="red", size=14, family="Arial Black"))

# A4 - Collapse (dark red zone)
fig_phase.add_shape(
    type="rect", x0=0, y0=0, x1=1.5, y1=5,
    fillcolor="darkred", opacity=0.2, line=dict(color="darkred", width=2)
)
fig_phase.add_annotation(x=0.75, y=2.5, text="A₄<br>COLLAPSE", showarrow=False,
                        font=dict(color="darkred", size=14, family="Arial Black"))

# Current Iran position
fig_phase.add_trace(go.Scatter(
    x=[current_health], y=[current_spe],
    mode='markers+text',
    marker=dict(size=30, color=attractor_color, symbol='star', 
                line=dict(color='black', width=3)),
    text=['🇮🇷 IRAN'],
    textposition='top center',
    textfont=dict(size=16, color='black', family="Arial Black"),
    name='Iran Current Position',
    hovertemplate=f"<b>IRAN</b><br>Health: {current_health:.3f}<br>SPE: {current_spe:.3f}<br>{current_attractor}<extra></extra>"
))

fig_phase.update_layout(
    title="⚠️ IRAN PHASE SPACE POSITION",
    xaxis_title="System Health Ω(S,t)",
    yaxis_title="Processing Efficiency SPE(t)", 
    height=600,
    xaxis=dict(range=[0, 6]),
    yaxis=dict(range=[0, 5]),
    font=dict(family="Arial", size=12)
)

st.plotly_chart(fig_phase, use_container_width=True)

# === Institutional Analysis ===
st.markdown("## 🏛️ Institutional Stress Analysis")

# Enhanced institutional data
institutional_data = []
for _, row in latest_data.iterrows():
    fitness = calculate_node_fitness(row)
    stress_level = abs(row['Stress'])
    
    # Risk categorization
    if stress_level > 4.0:
        risk = "🔴 CRITICAL"
    elif stress_level > 3.0:
        risk = "🟠 HIGH"
    elif stress_level > 2.0:
        risk = "🟡 ELEVATED"
    else:
        risk = "🟢 NORMAL"
    
    institutional_data.append({
        'Institution': row['Node'],
        'Coherence': row['Coherence'],
        'Capacity': row['Capacity'],
        'Stress': row['Stress'], 
        'Abstraction': row['Abstraction'],
        'Fitness': fitness,
        'Risk_Level': risk,
        'Stress_Magnitude': stress_level
    })

# Sort by stress level (highest first)
institutional_df = pd.DataFrame(institutional_data)
institutional_df = institutional_df.sort_values('Stress_Magnitude', ascending=False)

# Display institutional analysis
st.dataframe(institutional_df[['Institution', 'Risk_Level', 'Fitness', 'Coherence', 'Capacity', 'Stress', 'Abstraction']], 
            use_container_width=True)

# === Stress Distribution Visualization ===
st.markdown("### 📊 Stress Distribution Analysis")

fig_stress = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Institutional Stress Levels', 'Fitness vs Stress Correlation']
)

# Stress levels bar chart
stress_colors = ['red' if abs(s) > 3.0 else 'orange' if abs(s) > 2.0 else 'yellow' if abs(s) > 1.0 else 'green' 
                for s in latest_data['Stress']]

fig_stress.add_trace(
    go.Bar(x=latest_data['Node'], 
           y=latest_data['Stress'].abs(),
           marker_color=stress_colors,
           name='Stress Level',
           text=latest_data['Stress'].round(2),
           textposition='auto'),
    row=1, col=1
)

# Fitness vs Stress scatter
fitness_values = [calculate_node_fitness(row) for _, row in latest_data.iterrows()]

fig_stress.add_trace(
    go.Scatter(x=latest_data['Stress'].abs(),
               y=fitness_values,
               mode='markers+text',
               text=latest_data['Node'],
               textposition='top center',
               marker=dict(size=10, color='blue'),
               name='Fitness vs Stress'),
    row=1, col=2
)

fig_stress.update_layout(height=500, title_text="Iran Institutional Stress Analysis")
st.plotly_chart(fig_stress, use_container_width=True)

# === Time Series Analysis (if available) ===
if 'Year' in full_data.columns and len(full_data['Year'].unique()) > 1:
    st.markdown("## 📈 Historical Trend Analysis")
    
    # Calculate historical metrics
    historical_metrics = []
    for year in sorted(full_data['Year'].unique()):
        year_data = full_data[full_data['Year'] == year]
        
        if len(year_data) > 0:
            health = calculate_system_health(year_data)
            spe = calculate_processing_efficiency(year_data)
            ca = calculate_coherence_asymmetry(year_data)
            
            historical_metrics.append({
                'Year': year,
                'SystemHealth': health,
                'ProcessingEfficiency': spe,
                'CoherenceAsymmetry': ca
            })
    
    if historical_metrics:
        hist_df = pd.DataFrame(historical_metrics)
        
        fig_history = make_subplots(
            rows=2, cols=2,
            subplot_titles=['System Health Over Time', 'Processing Efficiency', 
                          'Coherence Asymmetry', 'Phase Space Trajectory']
        )
        
        # System Health
        fig_history.add_trace(
            go.Scatter(x=hist_df['Year'], y=hist_df['SystemHealth'],
                      name='System Health', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Processing Efficiency
        fig_history.add_trace(
            go.Scatter(x=hist_df['Year'], y=hist_df['ProcessingEfficiency'],
                      name='SPE', line=dict(color='green', width=3)),
            row=1, col=2
        )
        
        # Coherence Asymmetry
        fig_history.add_trace(
            go.Scatter(x=hist_df['Year'], y=hist_df['CoherenceAsymmetry'],
                      name='CA', line=dict(color='red', width=3)),
            row=2, col=1
        )
        
        # Phase space trajectory
        fig_history.add_trace(
            go.Scatter(x=hist_df['SystemHealth'], y=hist_df['ProcessingEfficiency'],
                      mode='markers+lines',
                      marker=dict(color=hist_df['Year'], colorscale='Viridis', 
                                showscale=True, colorbar=dict(title="Year")),
                      name='Trajectory'),
            row=2, col=2
        )
        
        fig_history.update_layout(height=800, title_text="Iran Historical Analysis")
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Trend analysis
        if len(hist_df) > 1:
            recent_health = hist_df['SystemHealth'].iloc[-1]
            previous_health = hist_df['SystemHealth'].iloc[-2]
            health_trend = "IMPROVING" if recent_health > previous_health else "DECLINING"
            
            st.markdown(f"### 📊 Recent Trend Analysis")
            st.write(f"**System Health Trend:** {health_trend}")
            st.write(f"**Current Health:** {recent_health:.3f}")
            st.write(f"**Change from Previous Period:** {recent_health - previous_health:+.3f}")

# === Strategic Assessment ===
st.markdown("## 🎯 Strategic Assessment")

st.markdown(f"**Current Attractor:** {current_attractor}")
st.markdown(f"**Recommendation:** {attractor_info['recommendation']}")

# Risk assessment
if current_health < 2.0:
    st.error("🔴 **CRITICAL RISK**: Iran is approaching or in the collapse attractor. Immediate intervention required.")
elif current_health < 2.5:
    st.warning("🟠 **HIGH RISK**: Iran is in the fragmented attractor zone. Urgent coherence restoration needed.")
elif current_health < 3.5:
    st.info("🟡 **MODERATE RISK**: Iran shows authoritarian tendencies. Monitor and balance control mechanisms.")
else:
    st.success("🟢 **STABLE**: Iran is in the adaptive attractor. Maintain current coordination.")

# Specific recommendations
st.markdown("### 📋 Specific Recommendations")

recommendations = []

# High stress institutions
high_stress_institutions = institutional_df[institutional_df['Stress_Magnitude'] > 3.0]
if len(high_stress_institutions) > 0:
    recommendations.append(f"🎯 **Priority 1**: Address high stress in {', '.join(high_stress_institutions['Institution'].tolist())}")

# Low fitness institutions  
low_fitness_institutions = institutional_df[institutional_df['Fitness'] < 2.0]
if len(low_fitness_institutions) > 0:
    recommendations.append(f"💪 **Priority 2**: Strengthen {', '.join(low_fitness_institutions['Institution'].tolist())}")

# System-level recommendations
if current_ca > 0.4:
    recommendations.append("⚖️ **Coherence**: Reduce institutional asymmetry through coordination mechanisms")

if current_spe < 1.5:
    recommendations.append("🧠 **Efficiency**: Improve information processing and decision-making capabilities")

# Display recommendations
for i, rec in enumerate(recommendations[:5], 1):  # Top 5 recommendations
    st.markdown(f"{i}. {rec}")

# === Footer ===
st.markdown("---")
st.markdown("### ℹ️ Analysis Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Data Source:** Iran_CAMS_Cleaned.csv  
    **Institutions Analyzed:** {len(latest_data)}  
    **Time Period:** {full_data['Year'].min() if 'Year' in full_data.columns else 'Current'} - {full_data['Year'].max() if 'Year' in full_data.columns else 'Current'}  
    **Framework:** CAMS-CAN v2.1  
    """)

with col2:
    st.markdown(f"""
    **Current Metrics:**  
    - System Health: {current_health:.3f}  
    - Processing Efficiency: {current_spe:.3f}  
    - Coherence Asymmetry: {current_ca:.3f}  
    - Attractor: {current_attractor.split()[1]}  
    """)

st.warning("⚠️ This analysis is for research and educational purposes. Policy decisions should incorporate multiple analytical frameworks and expert consultation.")