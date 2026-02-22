# Redirect to cams_advanced_analysis.py
# This file exists to maintain compatibility with Streamlit Cloud app settings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# Import custom modules
from src.ddig_analysis import compute_influence_bundle, compute_dDIG_for_metric
from src.cams_dyad_cld import (
    pivot_cams_long_to_wide, compute_M, compute_Y,
    normalise_B, compute_Omega, DELTA, EPS
)
from src.cams_attractor import (
    compute_fields_from_wide, smooth, plot_attractor_3d_plotly,
    plot_density_projection_plotly, detect_regimes
)

st.set_page_config(page_title="CAMS Advanced Analysis", layout="wide", page_icon="🔬")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 5px;}
    h1 {color: #3498db;}
    h2 {color: #e74c3c;}
    h3 {color: #f39c12;}
</style>
""", unsafe_allow_html=True)

st.title("🔬 CAMS Advanced Analysis Dashboard")
st.markdown("**Directed Information Gain (dDIG), Dyad Field Analysis & Phase-Space Attractors**")

# Sidebar - Data Selection
st.sidebar.header("📁 Data Selection")

data_source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Select from cleaned datasets"])

df = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CAMS CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    data_dir = Path("cleaned_datasets")
    if not data_dir.exists():
        data_dir = Path("data/cleaned")

    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            file_names = {f.stem.replace("_cleaned", "").replace("_", " ").title(): f for f in csv_files}
            selected = st.sidebar.selectbox("Select dataset:", sorted(file_names.keys()))
            df = pd.read_csv(file_names[selected])
            st.sidebar.success(f"Loaded: {selected}")
        else:
            st.sidebar.warning("No CSV files found in cleaned_datasets directory")
    else:
        st.sidebar.warning("No data directory found. Please upload a CSV file.")

if df is None:
    st.info("👈 Select a dataset or upload a CSV file to begin analysis")
    st.markdown("""
    ### Expected CSV Format
    Long-format with columns:
    - `Society` (optional)
    - `Year` (required)
    - `Node` (required)
    - `Coherence` (required)
    - `Capacity` (required)
    - `Stress` (required)
    - `Abstraction` (required)
    - `Bond Strength` (optional)

    ### Available Analysis Tools
    - **dDIG Analysis**: Directed Information Gain for causal influence ranking
    - **Dyad Field Analysis**: M-Y dynamics and metabolic-mythic tension
    - **Combined Insights**: Multi-metric influence rankings
    - **Phase-Space Attractor**: 3D trajectory visualization with regime detection
    """)
    st.stop()

# Main analysis tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 dDIG Analysis", "🌀 Dyad Field Analysis", "📈 Combined Insights", "🌌 Phase-Space Attractor"])

with tab1:
    st.header("📊 Directed Information Gain (dDIG) Analysis")
    st.markdown("""
    Compute **I(X→Y|Z)** - the conditional mutual information measuring how much institutional node X
    predicts changes in metric Y, given context Z.
    """)

    # Get available nodes
    if 'Node' not in df.columns:
        st.error("Dataset must have a 'Node' column")
        st.stop()

    nodes = sorted(df['Node'].unique())

    st.subheader("Configuration")
    col1, col2 = st.columns(2)

    with col1:
        target_metric = st.selectbox("Target Metric (Y):",
                                     ["Coherence", "Capacity", "Stress", "Abstraction"],
                                     index=0)
        n_bins = st.slider("Number of quantile bins:", 2, 10, 5)

    with col2:
        context_vars = st.multiselect("Context Variables (Z):",
                                      ["Regime", "Shock", "Era"],
                                      default=[])

    if st.button("🔬 Compute dDIG", type="primary"):
        with st.spinner("Computing directed information gain..."):
            try:
                result = compute_dDIG_for_metric(df, target_metric, nodes,
                                                context_vars=context_vars if context_vars else None,
                                                n_bins=n_bins)

                if result is not None and not result.empty:
                    # Filter out NaN values before plotting
                    result_valid = result.dropna(subset=['nDIG', 'dDIG_bits', 'H(Y|Z)_bits']).reset_index()

                    if len(result_valid) > 0:
                        st.success("✅ Analysis complete!")

                        # Top influencers
                        st.subheader(f"🎯 Top Influencers on {target_metric}")
                        top_n = min(5, len(result_valid))
                        top = result_valid.nlargest(top_n, 'nDIG')[['Node', 'dDIG_bits', 'nDIG']]

                        if not top.empty and not top['nDIG'].isna().all():
                            for idx, row in top.iterrows():
                                if pd.notna(row['nDIG']) and pd.notna(row['dDIG_bits']):
                                    st.metric(
                                        label=f"{row['Node']}",
                                        value=f"{row['dDIG_bits']:.3f} bits",
                                        delta=f"nDIG: {row['nDIG']:.3f}"
                                    )
                        else:
                            st.warning("No valid influence scores computed")

                        # Visualization
                        st.subheader("📊 Information Transfer Landscape")

                        # Ensure nDIG has positive values for size
                        result_valid['size_val'] = result_valid['nDIG'].clip(lower=0.001)

                        fig = px.scatter(
                            result_valid,
                            x='H(Y|Z)_bits', y='dDIG_bits',
                            text='Node',
                            title=f'Information Transfer: Nodes → {target_metric}',
                            size='size_val',
                            color='nDIG',
                            color_continuous_scale='Plasma',
                            labels={
                                'H(Y|Z)_bits': 'Baseline Uncertainty H(Y|Z) [bits]',
                                'dDIG_bits': 'Information Gain I(X→Y|Z) [bits]',
                                'nDIG': 'Normalized dDIG'
                            }
                        )
                        fig.update_traces(textposition='top center')
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)

                        # Full results table
                        st.subheader("📋 Full Results")
                        st.dataframe(result_valid.sort_values('nDIG', ascending=False),
                                   use_container_width=True)

                        # Export
                        csv = result_valid.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name=f"dDIG_{target_metric}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No valid data points after filtering NaN values. Try different parameters.")
                else:
                    st.error("Could not compute dDIG. Check that dataset has sufficient temporal data.")

            except Exception as e:
                st.error(f"Error computing dDIG: {e}")
                import traceback
                st.code(traceback.format_exc())

with tab2:
    st.header("🌀 Dyad Field Analysis")
    st.markdown("""
    Analyze the **M-Y dynamics** - the tension between metabolic load (material stress)
    and mythic integration (ideological coherence).
    """)

    try:
        wide_df = pivot_cams_long_to_wide(df)

        M = compute_M(wide_df)
        Y = compute_Y(wide_df)
        D = M - DELTA * Y

        B = wide_df.get('B', pd.Series(index=wide_df.index, dtype=float))
        Bn = normalise_B(B)
        R = D / (EPS + Bn)

        Omega = compute_Omega(wide_df)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean M (Metabolic)", f"{M.mean():.2f}")
        col2.metric("Mean Y (Mythic)", f"{Y.mean():.2f}")
        col3.metric("Mean Mismatch (D)", f"{D.mean():.2f}")
        col4.metric("Max Risk (R)", f"{R.max():.2f}")

        # Time series
        st.subheader("📈 M-Y Dynamics Over Time")
        fig_dyad = go.Figure()

        fig_dyad.add_trace(go.Scatter(x=M.index, y=M, name='M (Metabolic Load)',
                                     line=dict(color='#e74c3c', width=2)))
        fig_dyad.add_trace(go.Scatter(x=Y.index, y=Y, name='Y (Mythic Integration)',
                                     line=dict(color='#3498db', width=2)))
        fig_dyad.add_trace(go.Scatter(x=D.index, y=D, name='D (Mismatch)',
                                     line=dict(color='#f39c12', width=2, dash='dash')))

        fig_dyad.update_layout(
            title="Metabolic-Mythic Field Dynamics",
            xaxis_title="Year",
            yaxis_title="Field Strength",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_dyad, use_container_width=True)

        # Phase space
        st.subheader("🌀 M-Y Phase Space")
        fig_phase = px.scatter(x=M, y=Y, color=M.index,
                              labels={'x': 'M (Metabolic Load)', 'y': 'Y (Mythic Integration)', 'color': 'Year'},
                              title='M-Y Phase Space Trajectory',
                              color_continuous_scale='Viridis')
        fig_phase.update_traces(marker=dict(size=8))
        fig_phase.update_layout(height=500)
        st.plotly_chart(fig_phase, use_container_width=True)

        # Risk analysis
        st.subheader("⚠️ Risk Assessment")
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=R.index, y=R, name='Risk (R)',
                                     fill='tozeroy', line=dict(color='#e74c3c')))
        fig_risk.add_hline(y=2.0, line_dash="dash", line_color="orange",
                          annotation_text="Warning Threshold")
        fig_risk.update_layout(title="System Risk Over Time", height=400)
        st.plotly_chart(fig_risk, use_container_width=True)

    except Exception as e:
        st.error(f"Error in dyad field analysis: {e}")
        import traceback
        st.code(traceback.format_exc())

with tab3:
    st.header("📈 Combined Influence Insights")
    st.markdown("""
    Unified influence rankings across **all CAMS metrics** (Coherence, Capacity, Stress, Abstraction).
    """)

    if st.button("🔬 Compute Full Influence Bundle", type="primary"):
        with st.spinner("Computing influence across all metrics..."):
            try:
                nodes = sorted(df['Node'].unique())
                bundle = compute_influence_bundle(df, nodes, n_bins=5)

                if bundle:
                    st.success("✅ Multi-metric analysis complete!")

                    # Cognitive vs Affective influence
                    st.subheader("🧠 Cognitive vs Affective Influence")

                    col1, col2 = st.columns(2)

                    with col1:
                        if 'cognitive_influence' in bundle:
                            cog = bundle['cognitive_influence'].dropna().sort_values(ascending=False)
                            st.markdown("**Top Cognitive Influencers**")
                            for node, score in cog.head(5).items():
                                st.metric(node, f"{score:.3f}")

                    with col2:
                        if 'affective_influence' in bundle:
                            aff = bundle['affective_influence'].dropna().sort_values(ascending=False)
                            st.markdown("**Top Affective Influencers**")
                            for node, score in aff.head(5).items():
                                st.metric(node, f"{score:.3f}")

                    # Heatmap of all metrics
                    st.subheader("🔥 Influence Heatmap Across All Metrics")

                    metrics_data = []
                    for metric in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
                        if metric in bundle:
                            result = bundle[metric]
                            if result is not None and not result.empty:
                                for node in nodes:
                                    if node in result.index:
                                        ndig = result.loc[node, 'nDIG']
                                        if pd.notna(ndig):
                                            metrics_data.append({
                                                'Node': node,
                                                'Metric': metric,
                                                'nDIG': ndig
                                            })

                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        pivot = metrics_df.pivot(index='Node', columns='Metric', values='nDIG')

                        fig_heat = px.imshow(
                            pivot,
                            labels=dict(x="Target Metric", y="Source Node", color="nDIG"),
                            color_continuous_scale='RdYlGn',
                            title="Normalized dDIG Scores per Node-Metric Pair"
                        )
                        fig_heat.update_layout(height=600)
                        st.plotly_chart(fig_heat, use_container_width=True)

                        # Export
                        st.subheader("📥 Export Results")
                        export_data = {
                            'cognitive_influence': bundle.get('cognitive_influence', pd.Series()).to_dict(),
                            'affective_influence': bundle.get('affective_influence', pd.Series()).to_dict(),
                            'per_metric': {k: v.to_dict() if v is not None else {}
                                         for k, v in bundle.items()
                                         if k not in ['cognitive_influence', 'affective_influence']}
                        }

                        st.download_button(
                            label="📥 Download Full Bundle (JSON)",
                            data=json.dumps(export_data, indent=2),
                            file_name="cams_influence_bundle.json",
                            mime="application/json"
                        )
                    else:
                        st.warning("No valid metric data found")
                else:
                    st.error("Could not compute influence bundle")

            except Exception as e:
                st.error(f"Error computing influence bundle: {e}")
                import traceback
                st.code(traceback.format_exc())

with tab4:
    st.header("🌌 Phase-Space Attractor")
    st.markdown("""
    Visualize institutional dynamics as trajectories through **M-Y-B space**
    (Metabolic Load × Mythic Integration × Bond Strength).
    """)

    try:
        # Pivot to wide format
        wide_df = pivot_cams_long_to_wide(df)

        # Compute fields
        M, Y, B = compute_fields_from_wide(wide_df)

        # Check if we have valid B data
        has_B_data = B.notna().sum() >= 2

        if has_B_data:
            common = M.dropna().index.intersection(Y.dropna().index).intersection(B.dropna().index)
        else:
            # No B data - use 2D mode (M-Y only)
            common = M.dropna().index.intersection(Y.dropna().index)
            st.warning("⚠️ No Bond Strength data available - using 2D mode (M-Y only)")

        if len(common) < 2:
            st.error("❌ Insufficient data for attractor visualization (need at least 2 valid years)")

            # Diagnostic information
            st.info(f"""
            **Diagnostic Information:**
            - M (Metabolic Load): {M.notna().sum()} valid values
            - Y (Mythic Integration): {Y.notna().sum()} valid values
            - B (Bond Strength): {B.notna().sum()} valid values
            - Common valid indices: {len(common)}
            """)
            st.stop()

        # Filter to common indices
        M = M.loc[common]
        Y = Y.loc[common]
        if has_B_data:
            B = B.loc[common]
        years = common.values

        st.success(f"✅ Loaded {len(common)} time points from {years.min()} to {years.max()}")

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            sigma = st.slider("Smoothing (σ)", 0.0, 10.0, 2.0, 0.5,
                            help="Gaussian smoothing strength - higher = smoother trajectory")
        with col2:
            n_regimes = st.slider("Number of regimes", 2, 6, 3,
                                help="K-means clusters for regime detection")

        # Apply smoothing
        M_s = smooth(M, sigma=sigma)
        Y_s = smooth(Y, sigma=sigma)
        if has_B_data:
            B_s = smooth(B, sigma=sigma)

        # Regime detection
        if has_B_data:
            regime_labels = detect_regimes(M_s, Y_s, B_s, n_clusters=n_regimes)

            # 3D Attractor Plot
            st.subheader("🌌 3D Phase-Space Attractor")
            fig_3d = plot_attractor_3d_plotly(M_s, Y_s, B_s, years, regime_labels)
            st.plotly_chart(fig_3d, use_container_width=True)

            # 2D Projections
            st.subheader("📊 2D Projections")
            fig_proj = plot_density_projection_plotly(M_s, Y_s, B_s)
            st.plotly_chart(fig_proj, use_container_width=True)
        else:
            # 2D mode - M-Y only
            regime_labels = detect_regimes(M_s, Y_s, None, n_clusters=n_regimes)

            st.subheader("📊 2D Phase-Space Trajectory (M-Y)")

            # Create 2D trajectory plot
            fig_2d = go.Figure()

            # Add trajectory line
            fig_2d.add_trace(go.Scatter(
                x=M_s, y=Y_s,
                mode='lines+markers',
                line=dict(color='rgba(150,150,150,0.3)', width=1),
                marker=dict(
                    size=6,
                    color=years,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Year")
                ),
                text=[f"Year: {y:.0f}<br>M: {m:.2f}<br>Y: {yval:.2f}"
                      for y, m, yval in zip(years, M_s, Y_s)],
                hovertemplate='%{text}<extra></extra>',
                name='Trajectory'
            ))

            # Add regime colors
            if regime_labels is not None:
                colors = px.colors.qualitative.Set1[:n_regimes]
                for regime in range(n_regimes):
                    mask = regime_labels == regime
                    if mask.sum() > 0:
                        fig_2d.add_trace(go.Scatter(
                            x=M_s[mask], y=Y_s[mask],
                            mode='markers',
                            marker=dict(size=10, color=colors[regime],
                                      line=dict(width=2, color='white')),
                            name=f'Regime {regime}',
                            showlegend=True
                        ))

            fig_2d.update_layout(
                title="M-Y Phase Space (2D Mode)",
                xaxis_title="M (Metabolic Load)",
                yaxis_title="Y (Mythic Integration)",
                height=600,
                hovermode='closest'
            )

            st.plotly_chart(fig_2d, use_container_width=True)

        # Regime statistics
        st.subheader("📊 Regime Statistics")

        regime_stats = []
        for r in range(n_regimes):
            mask = regime_labels == r
            years_in_regime = years[mask]
            regime_stats.append({
                'Regime': r,
                'Years': len(years_in_regime),
                'Mean M': M_s[mask].mean(),
                'Mean Y': Y_s[mask].mean(),
                'Mean B': B_s[mask].mean() if has_B_data else np.nan,
                'Period': f"{years_in_regime.min():.0f}-{years_in_regime.max():.0f}" if len(years_in_regime) > 0 else "N/A"
            })

        regime_df = pd.DataFrame(regime_stats)
        st.dataframe(regime_df, use_container_width=True)

        # Field components over time
        st.subheader("📈 Field Components Over Time")

        fig_components = go.Figure()
        fig_components.add_trace(go.Scatter(x=years, y=M_s, name='M (Metabolic)',
                                          line=dict(color='#e74c3c', width=2)))
        fig_components.add_trace(go.Scatter(x=years, y=Y_s, name='Y (Mythic)',
                                          line=dict(color='#3498db', width=2)))
        if has_B_data:
            fig_components.add_trace(go.Scatter(x=years, y=B_s, name='B (Bond)',
                                              line=dict(color='#2ecc71', width=2)))

        # Add regime backgrounds
        if regime_labels is not None:
            colors = px.colors.qualitative.Set1[:n_regimes]
            for r in range(n_regimes):
                mask = regime_labels == r
                if mask.sum() > 0:
                    regime_years = years[mask]
                    fig_components.add_vrect(
                        x0=regime_years.min(), x1=regime_years.max(),
                        fillcolor=colors[r], opacity=0.1,
                        layer="below", line_width=0
                    )

        fig_components.update_layout(
            title="Field Components with Regime Backgrounds",
            xaxis_title="Year",
            yaxis_title="Field Strength",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_components, use_container_width=True)

        # Phase velocity
        st.subheader("⚡ Phase Velocity (Rate of Change)")

        # Velocity (rate of change in phase space)
        dM = np.diff(M_s)
        dY = np.diff(Y_s)
        if has_B_data:
            dB = np.diff(B_s)
            velocity = np.sqrt(dM**2 + dY**2 + dB**2)
        else:
            velocity = np.sqrt(dM**2 + dY**2)

        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=years[1:], y=velocity,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#9b59b6', width=2),
            name='Velocity'
        ))
        fig_vel.update_layout(
            title="System Change Velocity (higher = faster transitions)",
            xaxis_title="Year",
            yaxis_title="Velocity",
            height=300
        )
        st.plotly_chart(fig_vel, use_container_width=True)

        # Export data
        st.subheader("📥 Export Attractor Data")

        export_df = pd.DataFrame({
            'Year': years,
            'M': M_s,
            'Y': Y_s,
            'Regime': regime_labels
        })

        if has_B_data:
            export_df['B'] = B_s

        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Attractor Data CSV",
            data=csv,
            file_name="phase_space_attractor.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error in phase-space analysis: {e}")
        import traceback
        st.code(traceback.format_exc())

st.divider()
st.caption("🧠 CAMS Advanced Analysis Dashboard | Neural Nations Research | 2025")
