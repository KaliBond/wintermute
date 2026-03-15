#!/usr/bin/env python3
"""
CAMS Archetype Phase Engine (CAPE) — Streamlit App
----------------------------------------------------
Framework: CAMS v2.1
Author: Kari McKern
Draft implementation

Converts CAMS node datasets into archetype phase space
and plots structural trajectories over time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

CAMS_NODES = [
    "Helm", "Shield", "Lore", "Stewards",
    "Craft", "Hands", "Archive", "Flow"
]

LEGACY_NODE_MAP = {
    "Executive": "Helm",
    "Army": "Shield",
    "Priesthood / Knowledge Workers": "Lore",
    "Property Owners": "Stewards",
    "Trades / Professions": "Craft",
    "Proletariat": "Hands",
    "State Memory": "Archive",
    "Merchants / Shopkeepers": "Flow"
}

ARCHETYPE_COLORS = {
    "Temple": "#8B0000",
    "Marketplace": "#DAA520",
    "Workshop": "#2E8B57",
    "Commons": "#4682B4",
    "Throne": "#800080",
    "Fortress": "#A9A9A9",
    "Library": "#D2691E",
    "Guild": "#006400",
}

DATASETS_DIR = "cleaned_datasets"

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

def load_dataset(path_or_buffer, entity_name=None):
    df = pd.read_csv(path_or_buffer)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Society": "entity",
        "Company": "entity",
        "Node": "node",
        "Coherence": "C",
        "Capacity": "K",
        "Stress": "S",
        "Abstraction": "A"
    }
    df = df.rename(columns=rename_map)
    df["node"] = df["node"].replace(LEGACY_NODE_MAP)
    df = df[df["node"].isin(CAMS_NODES)]
    df[["C", "K", "S", "A"]] = df[["C", "K", "S", "A"]].astype(float)

    if "entity" not in df.columns:
        df["entity"] = entity_name if entity_name else "Unknown"

    return df[["entity", "Year", "node", "C", "K", "S", "A"]]


# ------------------------------------------------------------
# NODE COGNITION
# ------------------------------------------------------------

def compute_node_cognition(df):
    df = df.copy()
    df["cognition"] = (df["C"] * df["A"]) / (1 + df["S"])
    return df


# ------------------------------------------------------------
# ARCHETYPE COORDINATES
# ------------------------------------------------------------

def compute_phase_coordinates(df):
    rows = []
    for (entity, year), grp in df.groupby(["entity", "Year"]):
        node_cog = grp.groupby("node")["cognition"].mean()
        node_cog = node_cog.reindex(CAMS_NODES, fill_value=0)
        cog_total = node_cog.sum()

        X = (node_cog["Flow"] + node_cog["Stewards"]) - (node_cog["Lore"] + node_cog["Archive"])
        Y = (node_cog["Helm"] + node_cog["Shield"]) - (node_cog["Hands"] + node_cog["Craft"])

        dominant_node = node_cog.idxmax()
        archetype = assign_archetype(X, Y, dominant_node, node_cog, cog_total)

        rows.append({
            "entity": entity,
            "Year": year,
            "X": X,
            "Y": Y,
            "Cognition_Total": cog_total,
            "Archetype": archetype,
            "Dominant_Node": dominant_node
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# ARCHETYPE ASSIGNMENT
# ------------------------------------------------------------

def assign_archetype(X, Y, dominant_node, node_cog, cog_total):
    intermediate_map = {
        "Helm": "Throne",
        "Shield": "Fortress",
        "Archive": "Library",
        "Craft": "Guild"
    }
    if X <= 0 and Y > 0:
        quadrant = "Temple"
    elif X > 0 and Y > 0:
        quadrant = "Marketplace"
    elif X > 0 and Y <= 0:
        quadrant = "Workshop"
    else:
        quadrant = "Commons"

    dominance = node_cog[dominant_node] / cog_total if cog_total > 0 else 0
    if dominant_node in intermediate_map and dominance > 0.20:
        return intermediate_map[dominant_node]
    return quadrant


# ------------------------------------------------------------
# INSTABILITY DETECTION
# ------------------------------------------------------------

def detect_instability(phase):
    results = []
    for entity, grp in phase.groupby("entity"):
        grp = grp.sort_values("Year")
        for i in range(1, len(grp)):
            prev = grp.iloc[i - 1]
            curr = grp.iloc[i]
            dx = curr["X"] - prev["X"]
            dy = curr["Y"] - prev["Y"]
            velocity = np.sqrt(dx**2 + dy**2)
            flags = []
            if curr["Archetype"] != prev["Archetype"]:
                flags.append("ARCHETYPE_SHIFT")
            if curr["Cognition_Total"] < prev["Cognition_Total"] and dy > 0:
                flags.append("SCISSORS_SIGNATURE")
            results.append({
                "entity": entity,
                "Year": curr["Year"],
                "velocity": round(velocity, 3),
                "flags": ";".join(flags)
            })
    return pd.DataFrame(results)


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

def plot_archetype_map(phase):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    for label, (x0, y0, x1, y1) in {
        "Temple": (-1, 0, 0, 1), "Marketplace": (0, 0, 1, 1),
        "Workshop": (0, -1, 1, 0), "Commons": (-1, -1, 0, 0)
    }.items():
        pass  # quadrant labels via text below

    quadrant_labels = [
        (-0.6, 0.6, "Temple\n(Mythic / Authority)"),
        (0.6, 0.6, "Marketplace\n(Economic / Authority)"),
        (0.6, -0.6, "Workshop\n(Economic / Production)"),
        (-0.6, -0.6, "Commons\n(Mythic / Production)"),
    ]
    for qx, qy, ql in quadrant_labels:
        ax.text(qx, qy, ql, ha="center", va="center",
                fontsize=8, color="#cccccc", style="italic")

    latest = phase.sort_values("Year").groupby("entity").last().reset_index()
    for _, row in latest.iterrows():
        color = ARCHETYPE_COLORS.get(row["Archetype"], "black")
        ax.scatter(row["X"], row["Y"], s=140, color=color, zorder=5, edgecolors="white", linewidths=0.8)
        ax.text(row["X"] + 0.05, row["Y"] + 0.05, row["entity"], fontsize=8)

    ax.set_xlabel("← Symbolic / Memory      Economic / Logistic →")
    ax.set_ylabel("← Production      Authority →")
    ax.set_title("CAMS Archetype Phase Map (latest year per entity)", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_trajectories(phase, selected_entities):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    for entity, grp in phase.groupby("entity"):
        if entity not in selected_entities:
            continue
        grp = grp.sort_values("Year")
        ax.plot(grp["X"], grp["Y"], marker="o", markersize=4, label=entity, linewidth=1.5)
        # label start and end
        ax.text(grp.iloc[0]["X"], grp.iloc[0]["Y"], str(int(grp.iloc[0]["Year"])),
                fontsize=7, alpha=0.6)
        ax.text(grp.iloc[-1]["X"], grp.iloc[-1]["Y"], str(int(grp.iloc[-1]["Year"])),
                fontsize=7, fontweight="bold")

    ax.set_xlabel("← Symbolic / Memory      Economic / Logistic →")
    ax.set_ylabel("← Production      Authority →")
    ax.set_title("CAMS Archetype Trajectories", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------------------

st.set_page_config(page_title="CAPE — CAMS Archetype Phase Engine", layout="wide")

st.title("CAMS Archetype Phase Engine (CAPE)")
st.caption("Framework: CAMS v2.1 · Author: Kari McKern · Draft implementation")

st.markdown("""
Maps CAMS node datasets into an 8-archetype phase space — **Temple, Marketplace, Workshop, Commons**
(quadrant archetypes) and **Throne, Fortress, Library, Guild** (dominant-node intermediates) —
and plots structural trajectories over time.
""")

# ------------------------------------------------------------
# DATA SOURCE
# ------------------------------------------------------------

st.sidebar.header("Data Source")
source = st.sidebar.radio("Load from", ["Repo cleaned_datasets/", "Upload CSV files"])

frames = []

if source == "Repo cleaned_datasets/":
    if os.path.isdir(DATASETS_DIR):
        available = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")])
        selected_files = st.sidebar.multiselect("Select datasets", available, default=available[:5])
        for fname in selected_files:
            path = os.path.join(DATASETS_DIR, fname)
            try:
                entity_name = fname.replace(".csv", "")
                df = load_dataset(path, entity_name=entity_name)
                frames.append(df)
            except Exception as e:
                st.sidebar.warning(f"{fname}: {e}")
    else:
        st.sidebar.error(f"Directory '{DATASETS_DIR}' not found.")

else:
    uploaded = st.sidebar.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True)
    for f in uploaded:
        try:
            entity_name = f.name.replace(".csv", "")
            df = load_dataset(io.StringIO(f.read().decode("utf-8")), entity_name=entity_name)
            frames.append(df)
        except Exception as e:
            st.sidebar.warning(f"{f.name}: {e}")

# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

if not frames:
    st.info("Select or upload at least one dataset to begin.")
    st.stop()

combined = pd.concat(frames, ignore_index=True)
combined = compute_node_cognition(combined)
phase = compute_phase_coordinates(combined)
instab = detect_instability(phase)

entities = sorted(phase["entity"].unique())

# ------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Phase Map", "Trajectories", "Instability", "Data"])

with tab1:
    st.subheader("Archetype Phase Map")
    st.caption("Latest year per entity. Colour = archetype.")

    col1, col2 = st.columns([3, 1])
    with col1:
        fig1 = plot_archetype_map(phase)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.markdown("**Archetype legend**")
        for arch, color in ARCHETYPE_COLORS.items():
            st.markdown(
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:0.85em;">{arch}</span>',
                unsafe_allow_html=True
            )
            st.write("")

        latest = phase.sort_values("Year").groupby("entity").last().reset_index()
        st.markdown("**Current archetypes**")
        st.dataframe(latest[["entity", "Archetype", "Dominant_Node"]].set_index("entity"),
                     use_container_width=True)

with tab2:
    st.subheader("Archetype Trajectories")
    selected = st.multiselect("Entities to plot", entities, default=entities[:6])
    if selected:
        fig2 = plot_trajectories(phase, selected)
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Select at least one entity.")

with tab3:
    st.subheader("Instability Detection")
    st.caption("ARCHETYPE_SHIFT = quadrant change between years. SCISSORS_SIGNATURE = falling cognition + rising Y.")

    flagged = instab[instab["flags"] != ""].sort_values("velocity", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total transitions", len(instab))
    col2.metric("Flagged events", len(flagged))
    col3.metric("Entities", instab["entity"].nunique())

    if not flagged.empty:
        filter_entity = st.selectbox("Filter by entity", ["All"] + sorted(flagged["entity"].unique().tolist()))
        df_show = flagged if filter_entity == "All" else flagged[flagged["entity"] == filter_entity]
        st.dataframe(df_show.reset_index(drop=True), use_container_width=True)
    else:
        st.success("No instability flags detected in this dataset.")

with tab4:
    st.subheader("Phase Coordinates")
    entity_filter = st.selectbox("Entity", ["All"] + entities)
    df_view = phase if entity_filter == "All" else phase[phase["entity"] == entity_filter]
    st.dataframe(df_view.sort_values(["entity", "Year"]).reset_index(drop=True), use_container_width=True)

    st.subheader("Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download phase results CSV",
                           phase.to_csv(index=False).encode(),
                           "cape_phase_results.csv", "text/csv")
    with col2:
        st.download_button("Download instability results CSV",
                           instab.to_csv(index=False).encode(),
                           "cape_instability_results.csv", "text/csv")
