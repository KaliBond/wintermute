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
    normalise_B, compute_Omega, METABOLIC_NODES, MYTH_NODES
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
    data_dir = Path("data/cleaned")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            selected_file = st.sidebar.selectbox(
                "Select dataset:",
                csv_files,
                format_func=lambda x: x.stem
            )
            df = pd.read_csv(selected_file)
            st.sidebar.success(f"Loaded: {selected_file.stem}")
        else:
            st.sidebar.warning("No datasets found in data/cleaned/")
    else:
        st.sidebar.warning("data/cleaned/ directory not found")

if df is not None:
    # Display basic info
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Records", len(df))
    if 'Year' in df.columns:
        years = pd.to_numeric(df['Year'], errors='coerce').dropna()
        st.sidebar.metric("Year Range", f"{int(years.min())} - {int(years.max())}")
    if 'Society' in df.columns:
        st.sidebar.metric("Society", df['Society'].iloc[0] if len(df) > 0 else "Unknown")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 dDIG Analysis", "🌀 Dyad Field Analysis", "📈 Combined Insights", "🌌 Phase-Space Attractor"])

    # ==================== TAB 1: dDIG Analysis ====================
    with tab1:
        st.header("Directed Information Gain (dDIG) Analysis")
        st.markdown("Measures how changes in one institutional node predict changes in others, controlling for context (regime, shock intensity, era).")

        # Prepare wide format
        try:
            wide = pivot_cams_long_to_wide(df)

            # Auto-detect nodes from the data (extract unique node names from column names)
            detected_nodes = []
            for col in wide.columns:
                if col.startswith('Coherence_'):
                    node = col.replace('Coherence_', '')
                    detected_nodes.append(node)

            nodes = detected_nodes if detected_nodes else ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

            st.sidebar.info(f"**Detected {len(nodes)} nodes:** {', '.join(nodes)}")

        except ValueError as e:
            st.error(f"❌ Data format error: {str(e)}")
            st.info("**Expected CAMS format:**\n- Required columns: Society/Nation, Year, Node, Coherence, Capacity, Stress, Abstraction\n- Optional: Bond Strength, Node Value")
            wide = None

        if wide is not None:
            try:
                # Regime classification (simple heuristic)
                if 'B' in wide.columns and wide['B'].notna().sum() > 0:
                    try:
                        regime = pd.cut(wide['B'], bins=3, labels=['Low', 'Med', 'High'], duplicates='drop')
                    except ValueError:
                        # If all values are the same or insufficient variation
                        regime = pd.Series('Uniform', index=wide.index)
                else:
                    regime = pd.Series('Unknown', index=wide.index)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Configuration")
                    metric = st.selectbox("Select metric:", ["Coherence", "Capacity", "Stress", "Abstraction"])
                    qx = st.slider("Quantile bins for X (node change):", 2, 5, 3)
                    qy = st.slider("Quantile bins for Y (system response):", 2, 5, 3)

                with col2:
                    st.subheader("Context Controls")
                    qshock = st.slider("Shock intensity bins:", 2, 5, 3)
                    era = st.selectbox("Era binning:", ["decade", "none"])

                if st.button("🚀 Compute dDIG", type="primary"):
                    with st.spinner("Computing dDIG..."):
                        try:
                            result = compute_dDIG_for_metric(
                                wide, metric, nodes, regime,
                                qx=qx, qy=qy, qshock=qshock, era=era
                            )
                        except Exception as e:
                            st.error(f"❌ dDIG computation failed: {str(e)}")
                            st.info("**Troubleshooting tips:**\n- Ensure dataset has sufficient time series data (50+ years recommended)\n- Check that all 8 institutional nodes are present\n- Verify numeric values in Coherence, Capacity, Stress, Abstraction columns")
                            result = None

                        if result is not None and not result.empty:
                            st.success("✓ Analysis complete!")

                            # Display results
                            st.subheader(f"dDIG Results for {metric}")
                            st.dataframe(result.style.format({
                                'dDIG_bits': '{:.3f}',
                                'H(Y|Z)_bits': '{:.3f}',
                                'nDIG': '{:.3f}'
                            }), use_container_width=True)

                            # Visualizations
                            col1, col2 = st.columns(2)

                            with col1:
                                # nDIG bar chart
                                fig = px.bar(
                                    result.reset_index(),
                                    x='Node', y='nDIG',
                                    title=f'Normalized Directed Information Gain (nDIG) - {metric}',
                                    color='nDIG',
                                    color_continuous_scale='Viridis'
                                )
                                fig.update_layout(showlegend=False, height=400)
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                # dDIG vs H(Y|Z) scatter - filter out NaN values
                                result_valid = result.dropna(subset=['nDIG', 'dDIG_bits', 'H(Y|Z)_bits']).reset_index()

                                if len(result_valid) > 0:
                                    # Ensure nDIG has positive values for size
                                    result_valid['size_val'] = result_valid['nDIG'].clip(lower=0.001)

                                    fig = px.scatter(
                                        result_valid,
                                        x='H(Y|Z)_bits', y='dDIG_bits',
                                        text='Node',
                                        title='Information Transfer Landscape',
                                        size='size_val',
                                        color='nDIG',
                                        color_continuous_scale='Plasma'
                                    )
                                    fig.update_traces(textposition='top center')
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No valid nDIG values to plot")

                            # Top influencers
                            st.subheader("🎯 Key Insights")
                            result_valid = result.dropna(subset=['nDIG'])
                            if len(result_valid) > 0:
                                top3 = result_valid.nlargest(3, 'nDIG')
                                for i, (node, row) in enumerate(top3.iterrows(), 1):
                                    st.metric(
                                        f"#{i} {node}",
                                        f"{row['nDIG']:.3f}",
                                        f"{row['dDIG_bits']:.3f} bits"
                                    )
                            else:
                                st.info("No valid nDIG values computed (insufficient data or variation)")
                        else:
                            st.error("No results generated. Check data format.")

            except Exception as e:
                st.error(f"Error in dDIG analysis: {e}")
                st.exception(e)

    # ==================== TAB 2: Dyad Field Analysis ====================
    with tab2:
        st.header("Dyad Field Analysis: M-Y Dynamics")
        st.markdown("**M (Metabolic)**: Stress in material subsystem (Hands, Flow, Shield)")
        st.markdown("**Y (Mythic)**: Coherence/abstraction in meaning subsystem (Lore, Archive, Stewards)")

        try:
            wide = pivot_cams_long_to_wide(df)

            # Compute dyad fields
            M = compute_M(wide)
            Y = compute_Y(wide)
            D = M - Y  # Mismatch

            if 'B' in wide.columns:
                Bn = normalise_B(wide['B'])
                R = D / (0.05 + Bn)  # Risk
            else:
                Bn = pd.Series(np.nan, index=wide.index)
                R = pd.Series(np.nan, index=wide.index)

            Om = compute_Omega(wide)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean M (Metabolic)", f"{M.mean():.2f}")
            col2.metric("Mean Y (Mythic)", f"{Y.mean():.2f}")
            col3.metric("Mean Mismatch (D)", f"{D.mean():.2f}")
            col4.metric("Max Risk (R)", f"{R.max():.2f}")

            # Time series plot
            st.subheader("📈 Dyad Field Evolution")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=M.index, y=M.values, name='M (Metabolic)',
                                     line=dict(color='#e74c3c', width=2)))
            fig.add_trace(go.Scatter(x=Y.index, y=Y.values, name='Y (Mythic)',
                                     line=dict(color='#3498db', width=2)))
            fig.add_trace(go.Scatter(x=D.index, y=D.values, name='D (Mismatch)',
                                     line=dict(color='#f39c12', width=2, dash='dash')))

            fig.update_layout(
                title='M-Y Dyad Dynamics Over Time',
                xaxis_title='Year',
                yaxis_title='Field Value',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Phase space plot
            st.subheader("🌀 M-Y Phase Space")

            fig = px.scatter(
                x=Y.values, y=M.values,
                color=wide.index.values,
                labels={'x': 'Y (Mythic)', 'y': 'M (Metabolic)', 'color': 'Year'},
                title='Dyad Phase Space Trajectory',
                color_continuous_scale='Viridis'
            )
            fig.add_shape(type="line", x0=Y.min(), y0=Y.min(), x1=Y.max(), y1=Y.max(),
                         line=dict(color="red", width=2, dash="dash"),
                         name="Balance Line (M=Y)")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Export JSON
            st.subheader("💾 Export Analysis")
            if st.button("Generate CLD JSON"):
                summary = {
                    "years": [int(wide.index.min()), int(wide.index.max())],
                    "mean_M": float(M.mean()),
                    "mean_Y": float(Y.mean()),
                    "mean_D": float(D.mean()),
                    "mean_Bn": float(Bn.mean()),
                    "max_R": float(R.max()),
                    "mean_Omega": float(Om.mean()),
                }

                cld_json = {
                    "metadata": {
                        "topic": "CAMS Dyad: M–Y mismatch, coupling, risk",
                        "computed_summary": summary
                    },
                    "nodes": [
                        {"id": "M", "label": "Metabolic Stress", "type": "stock"},
                        {"id": "Y", "label": "Mythic Coherence", "type": "stock"},
                        {"id": "D", "label": "Dyad Mismatch", "type": "variable"},
                        {"id": "R", "label": "System Risk", "type": "variable"}
                    ],
                    "edges": [],
                    "loops": [],
                    "archetypes": []
                }

                st.json(cld_json)
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(cld_json, indent=2),
                    file_name="cams_dyad_cld.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"Error in dyad analysis: {e}")
            st.exception(e)

    # ==================== TAB 3: Combined Insights ====================
    with tab3:
        st.header("🔍 Combined Multi-Metric Analysis")
        st.markdown("Unified view of dDIG influence scores across all CAMS metrics")

        try:
            wide = pivot_cams_long_to_wide(df)

            # Auto-detect nodes
            detected_nodes = []
            for col in wide.columns:
                if col.startswith('Coherence_'):
                    node = col.replace('Coherence_', '')
                    detected_nodes.append(node)
            nodes = detected_nodes if detected_nodes else ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

            if 'B' in wide.columns and wide['B'].notna().sum() > 0:
                try:
                    regime = pd.cut(wide['B'], bins=3, labels=['Low', 'Med', 'High'], duplicates='drop')
                except ValueError:
                    regime = pd.Series('Uniform', index=wide.index)
            else:
                regime = pd.Series('Unknown', index=wide.index)

            if st.button("🎯 Compute Full Influence Bundle", type="primary"):
                with st.spinner("Computing influence scores across all metrics..."):
                    dig_tables, rollup = compute_influence_bundle(wide, regime, nodes, wc=0.5, wa=0.5)

                    st.success("✓ Analysis complete!")

                    # Display rollup
                    st.subheader("Institutional Influence Rankings")
                    st.dataframe(rollup.style.format({
                        'nDIG_cog': '{:.3f}',
                        'nDIG_aff': '{:.3f}',
                        'Influence': '{:.3f}'
                    }).background_gradient(subset=['Influence'], cmap='RdYlGn'), use_container_width=True)

                    # Heatmap of per-metric nDIG
                    st.subheader("Per-Metric nDIG Heatmap")

                    heatmap_data = []
                    for metric in ["Coherence", "Capacity", "Stress", "Abstraction"]:
                        if metric in dig_tables and not dig_tables[metric].empty:
                            heatmap_data.append(dig_tables[metric]['nDIG'].rename(metric))

                    if heatmap_data:
                        heatmap_df = pd.concat(heatmap_data, axis=1).T

                        fig = px.imshow(
                            heatmap_df,
                            labels=dict(x="Node", y="Metric", color="nDIG"),
                            x=heatmap_df.columns,
                            y=heatmap_df.index,
                            color_continuous_scale='Viridis',
                            aspect="auto"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    # Radar chart
                    st.subheader("Influence Profile (Top 5 Nodes)")
                    top5 = rollup.nlargest(5, 'Influence')

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=top5['Influence'].values,
                        theta=top5.index.tolist(),
                        fill='toself',
                        name='Influence'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, top5['Influence'].max()*1.1])),
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error in combined analysis: {e}")
            st.exception(e)

    # ==================== TAB 4: Phase-Space Attractor ====================
    with tab4:
        st.header("🌌 Phase-Space Attractor: M-Y-B Dynamics")
        st.markdown("""
        **3D trajectory visualization** of institutional dynamics:
        - **X-axis**: Metabolic Load (M) - stress in material subsystem (Hands, Flow, Shield)
        - **Y-axis**: Mythic Integration (Y) - coherence in meaning subsystem (Lore, Archive, Stewards)
        - **Z-axis**: Bond Strength (B) - coupling between institutions

        Reveals attractor basins, regime transitions, and system stability.
        """)

        try:
            wide = pivot_cams_long_to_wide(df)

            # Compute M, Y, B fields
            M, Y, B = compute_fields_from_wide(wide)

            # Diagnostic info
            st.sidebar.markdown("---")
            st.sidebar.subheader("Diagnostic Info")
            st.sidebar.metric("M (Metabolic) valid years", M.notna().sum())
            st.sidebar.metric("Y (Mythic) valid years", Y.notna().sum())
            st.sidebar.metric("B (Bond) valid years", B.notna().sum())

            # Align indices (remove NaN)
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

                # Show detailed diagnostics
                st.markdown("**Diagnostic Information:**")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**M (Metabolic Load)**")
                    stress_cols = [c for c in wide.columns if c.startswith('Stress_')]
                    if stress_cols:
                        st.success(f"✓ Found {len(stress_cols)} Stress columns")
                        for col in stress_cols[:5]:
                            st.caption(f"- {col}")
                        st.metric("Valid M values", M.notna().sum())
                    else:
                        st.error("✗ No Stress columns found")

                with col2:
                    st.markdown("**Y (Mythic Integration)**")
                    coherence_cols = [c for c in wide.columns if c.startswith('Coherence_')]
                    abstraction_cols = [c for c in wide.columns if c.startswith('Abstraction_')]
                    if coherence_cols and abstraction_cols:
                        st.success(f"✓ Found {len(coherence_cols)} Coherence + {len(abstraction_cols)} Abstraction")
                        st.metric("Valid Y values", Y.notna().sum())
                    else:
                        st.error(f"✗ Missing: {'Coherence' if not coherence_cols else 'Abstraction'}")

                with col3:
                    st.markdown("**B (Bond Strength)**")
                    if 'B' in wide.columns:
                        st.success("✓ Found pre-computed B column")
                        st.metric("Valid B values", B.notna().sum())
                    else:
                        bond_cols = [c for c in wide.columns if c.startswith('Bond_')]
                        if bond_cols:
                            st.success(f"✓ Found {len(bond_cols)} Bond columns")
                            st.metric("Valid B values", B.notna().sum())
                        else:
                            st.error("✗ No Bond Strength data")

                st.info("""
                **Possible fixes:**
                - Ensure dataset has Stress, Coherence, Abstraction columns for multiple nodes
                - Ensure Bond Strength data is present (or B column in wide format)
                - Check that data covers multiple years (not just a single year)
                - Try a different dataset with more complete time series
                """)
            else:
                M = M.loc[common]
                Y = Y.loc[common]
                if has_B_data:
                    B = B.loc[common]
                years = common.values

                # Summary metrics
                if has_B_data:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Time Span", f"{len(years)} years")
                    col2.metric("M Range", f"{M.min():.2f} - {M.max():.2f}")
                    col3.metric("Y Range", f"{Y.min():.2f} - {Y.max():.2f}")
                    col4.metric("B Range", f"{B.min():.2f} - {B.max():.2f}")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Time Span", f"{len(years)} years")
                    col2.metric("M Range", f"{M.min():.2f} - {M.max():.2f}")
                    col3.metric("Y Range", f"{Y.min():.2f} - {Y.max():.2f}")

                # Smoothing control
                st.sidebar.markdown("---")
                st.sidebar.subheader("Attractor Settings")
                sigma = st.sidebar.slider("Smoothing (σ)", 0, 10, 2, help="Gaussian smoothing sigma (0=no smoothing)")

                # Apply smoothing
                M_s = smooth(M, sigma=sigma)
                Y_s = smooth(Y, sigma=sigma)
                if has_B_data:
                    B_s = smooth(B, sigma=sigma)

                # 3D Attractor Plot (or 2D if no B data)
                if has_B_data:
                    st.subheader("🌌 3D Phase-Space Trajectory (M-Y-B)")
                    fig_3d = plot_attractor_3d_plotly(M_s, Y_s, B_s, years,
                                                      f"{df['Society'].iloc[0] if 'Society' in df.columns else 'CAMS'} Attractor")
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.subheader("📈 2D Phase-Space Trajectory (M-Y only)")
                    st.info("💡 Bond Strength data not available - showing M-Y trajectory only")

                    # Create 2D trajectory plot
                    fig_2d = go.Figure()
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

                    # Mark start and end
                    fig_2d.add_trace(go.Scatter(
                        x=[M_s[0]], y=[Y_s[0]],
                        mode='markers',
                        marker=dict(size=15, color='green', symbol='diamond'),
                        name=f'Start ({years[0]:.0f})',
                        hovertemplate=f'Start: {years[0]:.0f}<extra></extra>'
                    ))
                    fig_2d.add_trace(go.Scatter(
                        x=[M_s[-1]], y=[Y_s[-1]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='diamond'),
                        name=f'End ({years[-1]:.0f})',
                        hovertemplate=f'End: {years[-1]:.0f}<extra></extra>'
                    ))

                    fig_2d.update_layout(
                        title=f"{df['Society'].iloc[0] if 'Society' in df.columns else 'CAMS'} M-Y Trajectory",
                        xaxis_title="Metabolic Load (M)",
                        yaxis_title="Mythic Integration (Y)",
                        height=600
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)

                # Density Projection
                st.subheader("🔥 Phase Density Projection (M-Y plane)")
                st.markdown("*Heatmap shows time spent in different regions of phase space (attractor basins)*")
                fig_density = plot_density_projection_plotly(M_s, Y_s,
                                                             f"{df['Society'].iloc[0] if 'Society' in df.columns else 'CAMS'} Density")
                st.plotly_chart(fig_density, use_container_width=True)

                # Regime Detection
                st.subheader("🎯 Regime Detection")
                n_regimes = st.slider("Number of regimes", 2, 5, 3, help="K-means clustering of phase-space trajectory")

                if st.button("🔍 Detect Regimes"):
                    with st.spinner("Clustering phase-space trajectory..."):
                        if has_B_data:
                            labels, centers = detect_regimes(M_s, Y_s, B_s, n_regimes=n_regimes)
                            center_cols = ['M', 'Y', 'B']
                        else:
                            # 2D regime detection
                            from sklearn.cluster import KMeans
                            X = np.column_stack([M_s, Y_s])
                            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                            labels = kmeans.fit_predict(X)
                            centers = kmeans.cluster_centers_
                            center_cols = ['M', 'Y']

                        if labels is not None and centers is not None:
                            st.success(f"✓ Detected {n_regimes} distinct regimes!")

                            # Show regime centers
                            regime_df = pd.DataFrame(centers, columns=center_cols)
                            regime_df.index = [f"Regime {i+1}" for i in range(n_regimes)]
                            st.dataframe(regime_df.style.format('{:.2f}'), use_container_width=True)

                            # Regime occupancy
                            regime_counts = pd.Series(labels).value_counts().sort_index()
                            regime_pct = (regime_counts / len(labels) * 100).round(1)

                            st.markdown("**Regime Occupancy:**")
                            for i, (count, pct) in enumerate(zip(regime_counts, regime_pct)):
                                st.metric(f"Regime {i+1}", f"{count} years ({pct}%)")

                            # Plot with regime colors (3D or 2D)
                            fig_regimes = go.Figure()

                            if has_B_data:
                                # 3D plot with regime colors
                                for regime_id in range(n_regimes):
                                    mask = labels == regime_id
                                    regime_years = years[mask]
                                    regime_M = M_s[mask]
                                    regime_Y = Y_s[mask]
                                    regime_B = B_s[mask]

                                    fig_regimes.add_trace(go.Scatter3d(
                                        x=regime_M, y=regime_Y, z=regime_B,
                                        mode='markers',
                                        marker=dict(size=6),
                                        name=f'Regime {regime_id+1}',
                                        text=[f"Year: {y:.0f}" for y in regime_years],
                                        hovertemplate='%{text}<extra></extra>'
                                    ))

                                # Add cluster centers
                                fig_regimes.add_trace(go.Scatter3d(
                                    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                                    mode='markers',
                                    marker=dict(size=15, color='black', symbol='diamond'),
                                    name='Regime Centers',
                                    hoverinfo='skip'
                                ))
                            else:
                                # 2D plot with regime colors
                                for regime_id in range(n_regimes):
                                    mask = labels == regime_id
                                    regime_years = years[mask]
                                    regime_M = M_s[mask]
                                    regime_Y = Y_s[mask]

                                    fig_regimes.add_trace(go.Scatter(
                                        x=regime_M, y=regime_Y,
                                        mode='markers',
                                        marker=dict(size=8),
                                        name=f'Regime {regime_id+1}',
                                        text=[f"Year: {y:.0f}" for y in regime_years],
                                        hovertemplate='%{text}<extra></extra>'
                                    ))

                                # Add cluster centers
                                fig_regimes.add_trace(go.Scatter(
                                    x=centers[:, 0], y=centers[:, 1],
                                    mode='markers',
                                    marker=dict(size=20, color='black', symbol='diamond'),
                                    name='Regime Centers',
                                    hoverinfo='skip'
                                ))

                            if has_B_data:
                                fig_regimes.update_layout(
                                    title="Regime-Colored Phase Space (3D)",
                                    scene=dict(
                                        xaxis_title="Metabolic Load (M)",
                                        yaxis_title="Mythic Integration (Y)",
                                        zaxis_title="Bond Strength (B)"
                                    ),
                                    height=700
                                )
                            else:
                                fig_regimes.update_layout(
                                    title="Regime-Colored Phase Space (2D)",
                                    xaxis_title="Metabolic Load (M)",
                                    yaxis_title="Mythic Integration (Y)",
                                    height=600
                                )

                            st.plotly_chart(fig_regimes, use_container_width=True)

                        else:
                            st.error("❌ Regime detection failed (insufficient data)")

                # Trajectory statistics
                st.subheader("📊 Trajectory Statistics")

                # Velocity (rate of change in phase space)
                dM = np.diff(M_s)
                dY = np.diff(Y_s)
                dB = np.diff(B_s)
                velocity = np.sqrt(dM**2 + dY**2 + dB**2)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Phase Velocity** (rate of change)")
                    st.line_chart(pd.DataFrame({
                        'Velocity': velocity
                    }, index=years[:-1]))

                    st.metric("Mean Velocity", f"{velocity.mean():.3f}")
                    st.metric("Max Velocity", f"{velocity.max():.3f} (Year {years[:-1][velocity.argmax()]:.0f})")

                with col2:
                    st.markdown("**Field Components Over Time**")
                    components_df = pd.DataFrame({
                        'M (Metabolic)': M_s,
                        'Y (Mythic)': Y_s,
                        'B (Bond)': B_s
                    }, index=years)
                    st.line_chart(components_df)

        except Exception as e:
            st.error(f"Error in attractor analysis: {e}")
            st.exception(e)

else:
    st.info("👈 Please select or upload a CAMS dataset to begin analysis")

    # Show example data format
    with st.expander("📋 Expected CAMS Data Format"):
        st.code("""
Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength
USA,2025,Helm,5,6,3,7,12.5,2.1
USA,2025,Shield,5,7,2,6,13.0,2.2
USA,2025,Lore,7,8,1,8,18.5,2.8
...
        """, language="csv")

st.sidebar.markdown("---")
st.sidebar.markdown("**CAMS Advanced Analysis v1.0**")
st.sidebar.markdown("*Directed Information Gain & Dyad Fields*")
