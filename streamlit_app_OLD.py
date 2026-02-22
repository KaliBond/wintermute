# streamlit_app.py
# CAMS GTS EV v2.0 — Thermodynamic Cognitive Dashboard
# December 2025

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime
from cams_engine import run_cams_engine
from model_fitter import render_model_fitter_tab
from society_defaults import get_defaults
import json

# =============================================================================
# CONFIG & STYLE
# =============================================================================
st.set_page_config(page_title="CAMS GTS EV v2.0", layout="wide")
st.title("🧠 CAMS GTS EV v2.0 — Thermodynamic Cognitive Dashboard")
st.markdown("**Canonical Engine • Multi-Society • Export-Ready • Dec 2025**")

if not os.path.exists("exports"):
    os.makedirs("exports")

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_all_data():
    data = {}
    mapping = {
        "USA": "USA", "Rome": "Rome", "Australia": "Australia",
        "China": "China", "United Kingdom": "UK", "Ukraine": "Ukraine",
        "Russia": "Russia"
    }

    # Check for cleaned_datasets directory first
    data_dir = "cleaned_datasets" if os.path.exists("cleaned_datasets") else "data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return {}

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(f"{data_dir}/{file}")
                # Try to identify society name from file
                name = next((v for k,v in mapping.items() if k.lower() in file.lower()), file.split("_")[0])
                data[name] = df
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
    return data

data = load_all_data()
societies = list(data.keys())

if not societies:
    st.error("No data files found. Please add CSV files to the 'data' or 'cleaned_datasets' directory.")
    st.info("Expected format: Society,Year,Node,Coherence,Capacity,Stress,Abstraction")
    st.stop()

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.header("Analysis Controls")
    society = st.selectbox("Society", societies, index=societies.index("USA") if "USA" in societies else 0)
    df_full = data[society]

    # Get real-world defaults for this society
    defaults = get_defaults(society)

    min_y, max_y = int(df_full['Year'].min()), int(df_full['Year'].max())
    year_range = st.slider("Analysis Period", min_y, max_y, (max(min_y, max_y-50), max_y))
    start_year, end_year = year_range

    # Global thermodynamic parameters (real-world defaults)
    st.markdown("### Thermodynamic Parameters")
    st.caption(f"📊 Real-world 2025 estimates: {defaults['notes']}")

    EROEI = st.slider("Societal EROEI", 2.0, 50.0, float(defaults['EROEI']), 0.1,
                     help="Energy Return on Energy Invested - ratio of energy delivered to energy required to obtain it")
    phi_export = st.slider("Φ_export (entropy offload W·K⁻¹·cap⁻¹)", 0.0, 10.0, float(defaults['phi_export']), 0.01,
                          help="Entropy exported per capita via fossil fuel/resource extraction for other nations")

# =============================================================================
# MAIN PROCESSING
# =============================================================================
df_period = df_full[(df_full['Year'] >= start_year) & (df_full['Year'] <= end_year)].copy()

# Run engine for every year in range
results = []
for year in range(start_year, end_year + 1):
    df_y = df_full[df_full['Year'] == year]
    if df_y.empty:
        continue
    pop_M = df_y.iloc[0].get('Pop_M', 330 if "USA" in society else 26)
    res = run_cams_engine(df_y, EROEI, pop_M, phi_export)
    if res:
        res['Society'] = society
        res['Year'] = year
        results.append(res)

if not results:
    st.error("No data in selected range")
    st.stop()

results_df = pd.DataFrame(results)

# Current year (last in range)
current = results_df.iloc[-1]
prev = results_df.iloc[-2] if len(results_df) > 1 else current

# =============================================================================
# TOP SUMMARY ROW
# =============================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("R = Φ/Ψ", f"{current['R']:.2f}", f"{current['R']-prev['R']:+.2f}")
c2.metric("Health H%", f"{current['H%']:.1f}%", f"{current['H%']-prev['H%']:+.1f}")
c3.metric("⟨B⟩ Bond Strength", f"{current['⟨B⟩']:.2f}", f"{current['⟨B⟩']-prev['⟨B⟩']:+.2f}")
c4.metric("Ψ Deliberative", f"{current['Ψ']:.1f}", delta=f"{current['Ψ']-prev['Ψ']:+.1f}")
c5.metric("10-yr Crisis Probability", current['CrisisProb'])

st.markdown(f"## **{current['Class']}** — {society} {end_year}")

# =============================================================================
# TABBED INTERFACE
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Single Society Analysis", "Multi-Society Comparison", "Data Export", "Model Archaeology"])

with tab1:
    # =============================================================================
    # 1. DUAL-MODE TIME SERIES
    # =============================================================================
    fig_ts = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           subplot_titles=("Ψ vs Φ (Dual-Mode Load)", "R = Φ/Ψ Ratio", "Health Battery H%", "⟨B⟩ Bond Strength"),
                           vertical_spacing=0.06)

    fig_ts.add_trace(go.Scatter(x=results_df['Year'], y=results_df['Ψ'], name="Ψ Deliberative", line=dict(color="#1f77b4", width=3)), row=1, col=1)
    fig_ts.add_trace(go.Scatter(x=results_df['Year'], y=results_df['Φ'], name="Φ Reactive", line=dict(color="#ff7f0e", width=3)), row=1, col=1)

    fig_ts.add_trace(go.Scatter(x=results_df['Year'], y=results_df['R'], name="R Ratio", line=dict(color="purple")), row=2, col=1)
    for thresh, col in zip([1.0, 2.2, 4.5], ["green", "yellow", "red"]):
        fig_ts.add_hline(y=thresh, line_dash="dash", line_color=col, row=2, col=1)

    fig_ts.add_trace(go.Scatter(x=results_df['Year'], y=results_df['H%'], name="H%", line=dict(color="#2ca02c", width=3)), row=3, col=1)
    fig_ts.add_hline(y=65, line_color="lightgreen", line_dash="dot", row=3, col=1)
    fig_ts.add_hline(y=35, line_color="orange", line_dash="dot", row=3, col=1)
    fig_ts.add_hline(y=15, line_color="red", line_dash="dot", row=3, col=1)

    fig_ts.add_trace(go.Scatter(x=results_df['Year'], y=results_df['⟨B⟩'], name="Bond Strength", line=dict(color="#9467bd")), row=4, col=1)

    fig_ts.update_layout(height=900, title_text=f"{society} — Thermodynamic Trajectory {start_year}–{end_year}")
    st.plotly_chart(fig_ts, use_container_width=True)

    # =============================================================================
    # 2. BOND STRENGTH HEATMAP
    # =============================================================================
    st.subheader("Node Coupling — Bond Strength Matrix (Latest Year)")
    latest_year_df = df_full[df_full['Year'] == end_year].drop_duplicates(subset=['Node'], keep='first')

    if len(latest_year_df) == 8:
        C = latest_year_df['Coherence'].values
        # Create K array for all 8 nodes
        eta_i = np.ones(8) * 0.95
        K_calc = eta_i * EROEI * 1.2e3 / 12e3
        K = 10.0 * np.clip(K_calc, 0, 1.0)
        B_matrix = C[:, None] * C[None, :] * np.exp(-np.abs(K[:, None] - K[None, :])) * 0.88  # simplified D_KL

        node_names = latest_year_df['Node'].str.replace("Priesthood / Knowledge Workers", "Lore", regex=False).tolist()

        fig_heat = go.Figure(data=go.Heatmap(
            z=B_matrix,
            x=node_names,
            y=node_names,
            colorscale="Viridis",
            zmin=0, zmax=4,
            text=B_matrix,
            texttemplate="%{text:.2f}",
            textfont={"size":10}
        ))
        fig_heat.update_layout(title="Institutional Bond Strength Matrix", height=600)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning(f"Cannot display bond matrix - expected 8 nodes but found {len(latest_year_df)}")

with tab2:
    # =============================================================================
    # 3. MULTI-SOCIETY COMPARISON
    # =============================================================================
    st.subheader("Compare Multiple Societies (Health-Normalised)")
    compare_societies = st.multiselect("Add societies", societies, default=[s for s in ["USA", "Rome", "China", "Russia"] if s in societies])

    if compare_societies:
        fig_comp = go.Figure()
        for soc in compare_societies:
            dfc = data[soc]
            min_year = int(dfc['Year'].min())
            max_year = int(dfc['Year'].max())
            traj = []
            for y in range(min_year, max_year + 1):
                dy = dfc[dfc['Year'] == y]
                if dy.empty:
                    continue
                pop = dy.iloc[0].get('Pop_M', 300)
                res = run_cams_engine(dy, EROEI, pop, phi_export)
                if res:
                    traj.append({"Year": y, "H": res['H%'], "R": res['R'], "Society": soc})
            traj_df = pd.DataFrame(traj)
            if not traj_df.empty:
                traj_df['H_norm'] = traj_df['H'] / traj_df['H'].max() * 100
                fig_comp.add_trace(go.Scatter(x=traj_df['Year'], y=traj_df['H_norm'],
                                             name=f"{soc} (peak=100%)", mode="lines"))
        fig_comp.update_layout(title="Health Trajectories (Normalised to Historical Peak)", height=600)
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Select at least one society to compare")

with tab3:
    # =============================================================================
    # 4. EXPORT FUNCTIONALITY
    # =============================================================================
    st.subheader("Export Analysis")
    col1, col2 = st.columns(2)
    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Full Time-Series (CSV)", csv, f"CAMS_{society}_{start_year}-{end_year}.csv", "text/csv")
    with col2:
        json_out = results_df.to_dict(orient="records")
        st.download_button("📥 Download Full Time-Series (JSON)", json.dumps(json_out, indent=2),
                           f"CAMS_{society}_{start_year}-{end_year}.json", "application/json")

    st.markdown("### One-Row Canonical Output (Latest Year)")
    one_row = f"{society},{end_year},{EROEI:.1f},{current['⟨C⟩']:.2f},{current['⟨K⟩']:.2f},{current['⟨S⟩']:.2f},{current['⟨A⟩']:.2f},{current['⟨NV⟩']:.2f},{current['⟨B⟩']:.2f},0.12,{current['Ψ']:.1f},{current['Φ']:.1f},{current['R']:.2f},{current['H%']:.0f},{phi_export:.2f},{current['Class']},{current['CrisisProb']}"
    st.code(one_row, language="text")

    st.markdown("### Full Results Table")
    st.dataframe(results_df, use_container_width=True)

with tab4:
    # =============================================================================
    # 5. MODEL ARCHAEOLOGY & CIVILIZATION TYPE
    # =============================================================================
    render_model_fitter_tab(data, societies)

st.caption("CAMS GTS EV v2.0 • Thermodynamic Cognitive Dual-Mode Theory • neuralnations.org • Dec 2025")
