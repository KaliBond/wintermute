# cams_gt_app.py
# CAMS GT Dual-Mode Analyzer - Enhanced Version
# December 2025

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import base64
from datetime import datetime
import os

st.set_page_config(page_title="CAMS GT Dual-Mode Analyzer", layout="wide", page_icon="🧠")

# ============================================================================
# CAMS GT MODEL (Thermodynamically Grounded Dual-Mode)
# ============================================================================

class CAMSGT:
    def __init__(self):
        self.epsilon = 1e-6
        self.thresholds = {
            "H_crit": 3.0,
            "Psi_crit": 0.4,
            "CA_crit": 0.7,
            "V_crit": 3.0,
            "S_sync_crit": 0.8
        }

    def node_value(self, C, K, S, A):
        """Calculate node value: NV = C + K - |S| + 0.5A"""
        return C + K - abs(S) + 0.5 * A

    def bond_strength(self, row1, row2):
        """Calculate bond strength between two nodes"""
        return ((row1["coherence"] + row2["coherence"]) * 0.6 +
                (row1["abstraction"] + row2["abstraction"]) * 0.4) / (1 + abs(row1["stress"] + row2["stress"]) / 2)

    def system_health(self, df):
        """Calculate system health with coherence asymmetry penalty"""
        if df.empty:
            return np.nan
        nv = df.apply(lambda r: self.node_value(r["coherence"], r["capacity"], r["stress"], r["abstraction"]), axis=1)
        CA = self.coherence_asymmetry(df)
        penalty = min(CA * abs(df["stress"].mean()) / max(df["coherence"].mean(), self.epsilon), 0.75)
        return nv.mean() * (1 - penalty)

    def coherence_asymmetry(self, df):
        """Measure institutional imbalance via coherence*capacity dispersion"""
        prod = df["coherence"] * df["capacity"]
        if prod.mean() < self.epsilon:
            return 0
        return prod.std() / prod.mean()

    def legitimacy(self, df, H):
        """Calculate legitimacy from abstraction, capacity, and coherence"""
        I = (df["abstraction"] * df["capacity"]).mean() / 100
        chi = df["coherence"].mean() / 10
        return 0.4 * H + 0.3 * I + 0.3 * chi

    def stress_variance(self, df):
        """Calculate stress variance (early warning signal)"""
        return df["stress"].var()

    def synchronization(self, df):
        """Measure coherence synchronization across nodes"""
        c = df["coherence"]
        return 1 - c.std() / max(c.mean(), self.epsilon)

    def entropy(self, df):
        """Shannon entropy of coherence distribution"""
        c = df["coherence"] / df["coherence"].sum()
        c = c[c > 0]
        return - (c * np.log(c)).sum() if len(c) > 0 else 0

    def grand_metric_psi(self, H, L, E):
        """Grand metric Ψ (Deliberative Mode Strength)"""
        if pd.isna([H, L, E]).any():
            return np.nan
        return 0.3 * H + 0.3 * L + 0.2 / max(E, self.epsilon) + 0.2

    def phase(self, metrics):
        """Classify system phase based on thresholds"""
        score = 0
        if metrics["H"] < self.thresholds["H_crit"]: score += 1
        if metrics["CA"] > self.thresholds["CA_crit"]: score += 1
        if metrics["V_sigma"] > self.thresholds["V_crit"]: score += 1
        if metrics["S_sync"] > self.thresholds["S_sync_crit"]: score += 1
        if metrics["Psi"] < self.thresholds["Psi_crit"]: score += 1

        if score >= 4: return "CRITICAL", "red"
        if score >= 3: return "FRAGILE", "orange"
        if score >= 1: return "STRESSED", "yellow"
        return "STABLE", "green"

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

@st.cache_data
def load_available_datasets():
    """Load all datasets from cleaned_datasets or data directory"""
    datasets = {}
    data_dir = "cleaned_datasets" if os.path.exists("cleaned_datasets") else "data"

    if not os.path.exists(data_dir):
        return datasets

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(f"{data_dir}/{file}")
                name = file.replace("_cleaned.csv", "").replace(".csv", "").replace("_", " ").title()
                datasets[name] = df
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")

    return datasets

def normalize_columns(df):
    """Normalize column names to standard format"""
    col_map = {
        "Society": "society", "Nation": "society",
        "Year": "year",
        "Node": "node", "node": "node",
        "Coherence": "coherence", "C": "coherence",
        "Capacity": "capacity", "K": "capacity",
        "Stress": "stress", "S": "stress",
        "Abstraction": "abstraction", "A": "abstraction",
        "Node Value": "node_value", "NV": "node_value",
        "Node_Value": "node_value",
        "Bond Strength": "bond_strength", "BS": "bond_strength",
        "Bond_Strength": "bond_strength"
    }
    df = df.rename(columns=lambda x: col_map.get(x.strip(), x.lower()))
    return df

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.title("🧠 CAMS GT – Thermodynamic Dual-Mode Societal Analyzer")
st.markdown("**Complex Adaptive Model of Society – General Theory (Dec 2025)**  \nAnalyze civilizational dynamics through thermodynamic lens")

model = CAMSGT()

# Load pre-existing datasets
available_datasets = load_available_datasets()

# Data source selection
data_source = st.radio("Data Source:", ["Upload CSV", "Select Pre-loaded Dataset"], horizontal=True)

df = None
society_name = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CAMS CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            df = normalize_columns(raw_df)
            society_name = uploaded_file.name.split(".")[0].replace("_", " ").title()
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
else:
    if available_datasets:
        selected = st.selectbox("Select Society:", list(available_datasets.keys()))
        raw_df = available_datasets[selected]
        df = normalize_columns(raw_df)
        society_name = selected
    else:
        st.warning("No datasets found. Upload a CSV file instead.")
        st.stop()

if df is not None:
    # Validate required columns
    required = ["year", "node", "coherence", "capacity", "stress", "abstraction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.info(f"Available columns: {list(df.columns)}")
        st.stop()

    # Convert to numeric and clean
    for col in ["coherence", "capacity", "stress", "abstraction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=required, inplace=True)

    if df.empty:
        st.error("No valid data after cleaning")
        st.stop()

    # Year selection
    years = sorted(df["year"].unique())

    col_year, col_range = st.columns([2, 1])
    with col_year:
        selected_year = st.select_slider("Select Year", options=years, value=years[-1])
    with col_range:
        analyze_all = st.checkbox("Analyze Full Timeline", value=True)

    # ========================================================================
    # CALCULATE METRICS FOR ALL YEARS
    # ========================================================================

    yearly = []
    for y in years:
        sub = df[df["year"] == y]
        if sub.empty or len(sub) < 3:  # Need at least 3 nodes
            continue

        H = model.system_health(sub)
        CA = model.coherence_asymmetry(sub)
        L = model.legitimacy(sub, H)
        V = model.stress_variance(sub)
        S = model.synchronization(sub)
        E = model.entropy(sub)
        Psi = model.grand_metric_psi(H, L, E)
        phase, color = model.phase({"H": H, "CA": CA, "V_sigma": V, "S_sync": S, "Psi": Psi})

        yearly.append({
            "year": y, "H": H, "L": L, "CA": CA, "V_sigma": V, "S_sync": S,
            "E": E, "Psi": Psi, "Phase": phase, "Color": color
        })

    if not yearly:
        st.error("No valid data for any year")
        st.stop()

    metrics_df = pd.DataFrame(yearly)

    # Current year data
    current = df[df["year"] == selected_year]
    cur_metrics = metrics_df[metrics_df["year"] == selected_year].iloc[0]

    # ========================================================================
    # DASHBOARD LAYOUT
    # ========================================================================

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("System Health H", f"{cur_metrics['H']:.2f}",
                 delta=f"{cur_metrics['H'] - metrics_df.iloc[-2]['H']:.2f}" if len(metrics_df) > 1 else None)
    with col2:
        st.metric("Grand Metric Ψ", f"{cur_metrics['Psi']:.3f}",
                 delta=f"{cur_metrics['Psi'] - metrics_df.iloc[-2]['Psi']:.3f}" if len(metrics_df) > 1 else None)
    with col3:
        phase_emoji = {"CRITICAL": "🔴", "FRAGILE": "🟠", "STRESSED": "🟡", "STABLE": "🟢"}
        st.metric("Phase", f"{phase_emoji.get(cur_metrics['Phase'], '')} {cur_metrics['Phase']}")
    with col4:
        st.metric("Coherence Asymmetry", f"{cur_metrics['CA']:.2f}",
                 delta="⚠️ HIGH" if cur_metrics['CA'] > 0.7 else None)
    with col5:
        st.metric("Synchronization", f"{cur_metrics['S_sync']:.2f}")

    st.markdown(f"## {society_name} – Year {selected_year}")

    # ========================================================================
    # MAIN VISUALIZATIONS
    # ========================================================================

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔗 Network Analysis", "📈 Time Series", "📥 Export"])

    with tab1:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Multi-panel figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("System Health H(t)", "Grand Metric Ψ(t) - Deliberative Mode",
                               "Node Positions (C vs K)", "Stress Distribution"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "scatter"}, {"type": "box"}]]
            )

            # Health over time
            fig.add_trace(go.Scatter(
                x=metrics_df["year"],
                y=metrics_df["H"],
                name="Health H",
                line=dict(color="blue", width=3),
                fill='tozeroy',
                fillcolor='rgba(0,100,255,0.1)'
            ), row=1, col=1)
            fig.add_hline(y=model.thresholds["H_crit"], line_dash="dash",
                         line_color="red", annotation_text="Critical", row=1, col=1)

            # Psi over time with phase colors
            fig.add_trace(go.Scatter(
                x=metrics_df["year"],
                y=metrics_df["Psi"],
                name="Ψ (Deliberative)",
                line=dict(color="purple", width=3),
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.1)'
            ), row=1, col=2)
            fig.add_hline(y=model.thresholds["Psi_crit"], line_dash="dash",
                         line_color="orange", annotation_text="Threshold", row=1, col=2)

            # Coherence vs Capacity scatter
            fig.add_trace(go.Scatter(
                x=current["capacity"],
                y=current["coherence"],
                mode="markers+text",
                text=current["node"],
                textposition="top center",
                marker=dict(
                    size=25,
                    color=current["stress"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="Stress", x=0.46)
                ),
                name="Nodes"
            ), row=2, col=1)

            # Stress box plot
            fig.add_trace(go.Box(
                y=current["stress"],
                name="Stress",
                marker_color="indianred",
                boxmean='sd'
            ), row=2, col=2)

            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("🎯 Current State")
            st.write(f"**Year**: {selected_year}")
            st.write(f"**Nodes**: {len(current)}")
            st.write(f"**Phase**: {cur_metrics['Phase']}")

            st.divider()

            st.subheader("📊 Key Metrics")
            st.write(f"**Health (H)**: {cur_metrics['H']:.2f}")
            st.write(f"**Legitimacy (L)**: {cur_metrics['L']:.2f}")
            st.write(f"**Entropy (E)**: {cur_metrics['E']:.2f}")
            st.write(f"**Stress Variance**: {cur_metrics['V_sigma']:.2f}")

            st.divider()

            st.subheader("⚠️ Early Warnings")
            warnings = []
            if cur_metrics["CA"] > model.thresholds["CA_crit"]:
                warnings.append("🔴 High coherence asymmetry")
            if cur_metrics["V_sigma"] > model.thresholds["V_crit"]:
                warnings.append("🔴 High stress variance")
            if cur_metrics["S_sync"] > model.thresholds["S_sync_crit"]:
                warnings.append("🔴 Low synchronization")
            if cur_metrics["Psi"] < model.thresholds["Psi_crit"]:
                warnings.append("🟡 Weak deliberative mode")

            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("✅ No critical warnings")

            st.divider()

            st.subheader("💡 Recommendations")
            if cur_metrics["Phase"] == "CRITICAL":
                st.error("🔴 **CRITICAL STATE**")
                st.write("- Emergency coordination required")
                st.write("- Reduce systemic stress immediately")
                st.write("- Focus on weakest nodes")
            elif cur_metrics["Phase"] == "FRAGILE":
                st.warning("🟠 **FRAGILE STATE**")
                st.write("- Rebalance capacity across nodes")
                st.write("- Address high-stress institutions")
                st.write("- Build surplus buffers")
            elif cur_metrics["Phase"] == "STRESSED":
                st.info("🟡 **STRESSED STATE**")
                st.write("- Monitor stress trends closely")
                st.write("- Strengthen institutional bonds")
                st.write("- Maintain abstraction capacity")
            else:
                st.success("🟢 **STABLE STATE**")
                st.write("- Maintain institutional health")
                st.write("- Continue deliberative processes")
                st.write("- Build long-term resilience")

    with tab2:
        st.subheader(f"🔗 Institutional Network – Year {selected_year}")

        # Build network graph
        G = nx.Graph()
        nodes = current["node"].unique()
        for n in nodes:
            G.add_node(n)

        # Add edges based on bond strength
        bond_threshold = st.slider("Bond Strength Threshold", 0.0, 2.0, 0.5, 0.1)

        for i in range(len(current)):
            for j in range(i+1, len(current)):
                b = model.bond_strength(current.iloc[i], current.iloc[j])
                if b > bond_threshold:
                    G.add_edge(current.iloc[i]["node"], current.iloc[j]["node"], weight=b)

        if len(G.edges()) > 0:
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

            # Create network visualization
            edge_x, edge_y, edge_weights = [], [], []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2]['weight'])

            node_x = [pos[n][0] for n in nodes]
            node_y = [pos[n][1] for n in nodes]
            node_color = [current[current["node"] == n]["coherence"].iloc[0] for n in nodes]
            node_size = [current[current["node"] == n]["capacity"].iloc[0] * 5 for n in nodes]

            fig_net = go.Figure()

            # Add edges
            fig_net.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=2, color="lightgray"),
                hoverinfo="none",
                showlegend=False
            ))

            # Add nodes
            fig_net.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=nodes,
                textposition="top center",
                marker=dict(
                    size=node_size,
                    color=node_color,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Coherence"),
                    line=dict(width=2, color="white")
                ),
                hovertemplate='<b>%{text}</b><br>Coherence: %{marker.color:.2f}<extra></extra>',
                showlegend=False
            ))

            fig_net.update_layout(
                title=f"Network with {len(G.edges())} bonds (threshold: {bond_threshold})",
                height=600,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )

            st.plotly_chart(fig_net, use_container_width=True)

            # Network statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Network Density", f"{nx.density(G):.2f}")
            with col2:
                if nx.is_connected(G):
                    st.metric("Avg Path Length", f"{nx.average_shortest_path_length(G):.2f}")
                else:
                    st.metric("Connected Components", len(list(nx.connected_components(G))))
            with col3:
                st.metric("Clustering Coefficient", f"{nx.average_clustering(G):.2f}")
        else:
            st.info(f"No bonds above threshold {bond_threshold}. Lower the threshold to see connections.")

    with tab3:
        st.subheader("📈 Full Timeline Analysis")

        # Create comprehensive time series
        fig_timeline = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=("Grand Metric Ψ (Deliberative Mode)",
                           "System Health H",
                           "Phase Classification",
                           "Early Warning Signals"),
            vertical_spacing=0.08
        )

        # Psi
        fig_timeline.add_trace(go.Scatter(
            x=metrics_df["year"], y=metrics_df["Psi"],
            name="Ψ", line=dict(color="purple", width=3),
            fill='tozeroy'
        ), row=1, col=1)
        fig_timeline.add_hline(y=0.4, line_dash="dash", line_color="orange", row=1, col=1)

        # Health
        fig_timeline.add_trace(go.Scatter(
            x=metrics_df["year"], y=metrics_df["H"],
            name="H", line=dict(color="blue", width=3),
            fill='tozeroy'
        ), row=2, col=1)
        fig_timeline.add_hline(y=3.0, line_dash="dash", line_color="red", row=2, col=1)

        # Phase (colored bar)
        fig_timeline.add_trace(go.Bar(
            x=metrics_df["year"], y=[1]*len(metrics_df),
            marker_color=metrics_df["Color"],
            name="Phase",
            showlegend=False
        ), row=3, col=1)

        # Early warnings
        fig_timeline.add_trace(go.Scatter(
            x=metrics_df["year"], y=metrics_df["CA"],
            name="Coherence Asymmetry", line=dict(color="red")
        ), row=4, col=1)
        fig_timeline.add_trace(go.Scatter(
            x=metrics_df["year"], y=metrics_df["S_sync"],
            name="Synchronization", line=dict(color="green")
        ), row=4, col=1)

        fig_timeline.update_layout(height=1000, title=f"{society_name} – Complete Timeline")
        st.plotly_chart(fig_timeline, use_container_width=True)

    with tab4:
        st.subheader("📥 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            # Export metrics CSV
            csv_metrics = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv_metrics,
                file_name=f"cams_metrics_{society_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            # Export raw data
            csv_raw = current.to_csv(index=False)
            st.download_button(
                label="📥 Download Current Year Data",
                data=csv_raw,
                file_name=f"cams_nodes_{society_name}_{selected_year}.csv",
                mime="text/csv"
            )

        st.divider()

        st.subheader("📊 Summary Statistics")
        st.dataframe(metrics_df.describe(), use_container_width=True)

        st.subheader("📋 Full Metrics Table")
        st.dataframe(metrics_df, use_container_width=True, height=400)

else:
    st.info("👈 Upload a CAMS CSV file or select a pre-loaded dataset to begin analysis")
    if available_datasets:
        st.success(f"✅ {len(available_datasets)} datasets available in library")
    st.markdown("""
    ### Supported Format
    CSV with columns: `Year, Node, Coherence, Capacity, Stress, Abstraction`

    ### Example Datasets
    - Rome (0-476 CE)
    - USA (1790-2025)
    - China, Russia, Australia, Singapore, Denmark, etc.
    """)

st.divider()
st.caption("CAMS GT Dual-Mode Framework v2.0 – December 2025 | Built on Prigogine, Haken, and 5000+ years of civilizational data")
