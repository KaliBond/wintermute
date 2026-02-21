"""
CAMS Phase-Space Attractor Generator

Visualizes institutional dynamics as 3D phase-space trajectories:
- x = Metabolic Load (M) - stress in material subsystem
- y = Mythic Integration (Y) - coherence in meaning subsystem
- z = Mean Bond Strength (B) - coupling between institutions

Reveals attractor basins, regime transitions, and system stability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# -----------------------------
# CONFIGURATION
# -----------------------------

METABOLIC_NODES = ["Hands", "Flow", "Shield"]
MYTHIC_NODES    = ["Lore", "Archive", "Stewards"]
SMOOTH_SIGMA    = 2


# -----------------------------
# LOAD + CLEAN
# -----------------------------

def load_cams(filepath):
    """Load and clean CAMS CSV file"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["Year", "Node"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    # force numeric conversion
    for col in ["Coherence", "Capacity", "Stress", "Abstraction", "Bond Strength"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -----------------------------
# FIELD CONSTRUCTION
# -----------------------------

def compute_fields(df):
    """
    Compute M, Y, B fields from CAMS long-format dataframe

    Returns: (M, Y, B) as pandas Series indexed by Year
    """
    wide = df.pivot_table(
        index="Year",
        columns="Node",
        values=["Coherence", "Capacity", "Stress", "Abstraction", "Bond Strength"],
        aggfunc="mean"
    )

    wide.columns = ["_".join(col).strip() for col in wide.columns]

    # ---- Metabolic Load (M) ----
    M_list = []
    for node in METABOLIC_NODES:
        col = f"Stress_{node}"
        if col in wide.columns:
            M_list.append(wide[col])

    M = pd.concat(M_list, axis=1).mean(axis=1) if M_list else pd.Series(dtype=float)

    # ---- Mythic Integration (Y) ----
    Y_list = []
    for node in MYTHIC_NODES:
        C = f"Coherence_{node}"
        A = f"Abstraction_{node}"
        if C in wide.columns and A in wide.columns:
            Y_list.append(0.5 * wide[C] + 0.5 * wide[A])

    Y = pd.concat(Y_list, axis=1).mean(axis=1) if Y_list else pd.Series(dtype=float)

    # ---- Bond Strength (B) ----
    bond_cols = [c for c in wide.columns if c.startswith("Bond Strength_")]
    B = wide[bond_cols].mean(axis=1) if bond_cols else pd.Series(dtype=float)

    return M, Y, B


def compute_fields_from_wide(wide_df):
    """
    Compute M, Y, B fields from already-pivoted wide-format dataframe
    (for integration with existing cams_dyad_cld.pivot_cams_long_to_wide)

    Auto-detects nodes from column names instead of hardcoding.

    Returns: (M, Y, B) as pandas Series indexed by Year
    """
    # Auto-detect nodes from column names
    detected_nodes = []
    for col in wide_df.columns:
        if col.startswith('Stress_'):
            node = col.replace('Stress_', '')
            detected_nodes.append(node)

    # ---- Metabolic Load (M) ----
    # Use ALL available Stress columns (not just hardcoded nodes)
    M_list = []
    for node in detected_nodes:
        col = f"Stress_{node}"
        if col in wide_df.columns:
            M_list.append(wide_df[col])

    M = pd.concat(M_list, axis=1).mean(axis=1) if M_list else pd.Series(index=wide_df.index, dtype=float)

    # ---- Mythic Integration (Y) ----
    # Use ALL available Coherence + Abstraction columns
    Y_list = []
    for node in detected_nodes:
        C = f"Coherence_{node}"
        A = f"Abstraction_{node}"
        if C in wide_df.columns and A in wide_df.columns:
            Y_list.append(0.5 * wide_df[C] + 0.5 * wide_df[A])

    Y = pd.concat(Y_list, axis=1).mean(axis=1) if Y_list else pd.Series(index=wide_df.index, dtype=float)

    # ---- Bond Strength (B) ----
    if 'B' in wide_df.columns and wide_df['B'].notna().sum() > 0:
        B = wide_df['B']
    else:
        # Try to find individual node bond strengths
        bond_cols = [c for c in wide_df.columns if 'Bond' in c and c != 'B']
        if bond_cols:
            B = wide_df[bond_cols].mean(axis=1)
        else:
            # No bond data - return NaN series
            B = pd.Series(index=wide_df.index, dtype=float)

    return M, Y, B


# -----------------------------
# SMOOTHING
# -----------------------------

def smooth(series, sigma=SMOOTH_SIGMA):
    """Apply Gaussian smoothing to time series"""
    if len(series) < 2:
        return series.values
    return gaussian_filter1d(series.values, sigma=sigma)


# -----------------------------
# 3D ATTRACTOR (Matplotlib)
# -----------------------------

def plot_attractor_3d(M, Y, B, title="CAMS Phase-Space Attractor"):
    """
    Plot 3D phase-space trajectory using matplotlib

    Note: Requires interactive backend for rotation
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color by time (progression)
    colors = np.linspace(0, 1, len(M))
    scatter = ax.scatter(M, Y, B, c=colors, cmap='viridis', s=20, alpha=0.6)
    ax.plot(M, Y, B, linewidth=0.5, alpha=0.3, color='gray')

    ax.set_xlabel("Metabolic Load (M)", fontsize=12)
    ax.set_ylabel("Mythic Integration (Y)", fontsize=12)
    ax.set_zlabel("Bond Strength (B)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar for time progression
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Time Progression', fontsize=10)

    plt.tight_layout()
    return fig


# -----------------------------
# 3D ATTRACTOR (Plotly - Interactive)
# -----------------------------

def plot_attractor_3d_plotly(M, Y, B, years=None, title="CAMS Phase-Space Attractor"):
    """
    Plot interactive 3D phase-space trajectory using Plotly

    Returns: plotly Figure object
    """
    import plotly.graph_objects as go

    if years is None:
        years = np.arange(len(M))

    fig = go.Figure()

    # Trajectory line
    fig.add_trace(go.Scatter3d(
        x=M, y=Y, z=B,
        mode='lines+markers',
        line=dict(color=years, colorscale='Viridis', width=3),
        marker=dict(
            size=4,
            color=years,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Year")
        ),
        text=[f"Year: {y:.0f}<br>M: {m:.2f}<br>Y: {y_val:.2f}<br>B: {b:.2f}"
              for y, m, y_val, b in zip(years, M, Y, B)],
        hovertemplate='%{text}<extra></extra>',
        name='Trajectory'
    ))

    # Mark start and end points
    fig.add_trace(go.Scatter3d(
        x=[M[0]], y=[Y[0]], z=[B[0]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name=f'Start ({years[0]:.0f})',
        hovertemplate=f'Start: {years[0]:.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter3d(
        x=[M[-1]], y=[Y[-1]], z=[B[-1]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name=f'End ({years[-1]:.0f})',
        hovertemplate=f'End: {years[-1]:.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Metabolic Load (M)",
            yaxis_title="Mythic Integration (Y)",
            zaxis_title="Bond Strength (B)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        height=700
    )

    return fig


# -----------------------------
# DENSITY PROJECTION (LESS SPAGHETTI)
# -----------------------------

def plot_density_projection(M, Y, title="Phase Density Projection"):
    """
    Plot 2D density heatmap of M-Y phase space
    Shows attractor basins and regime occupancy
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    hist = ax.hist2d(M, Y, bins=50, cmap='hot')

    ax.set_xlabel("Metabolic Load (M)", fontsize=12)
    ax.set_ylabel("Mythic Integration (Y)", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(hist[3], ax=ax)
    cbar.set_label('Density (time spent)', fontsize=10)

    plt.tight_layout()
    return fig


def plot_density_projection_plotly(M, Y, title="Phase Density Projection"):
    """
    Plot interactive 2D density heatmap using Plotly
    """
    import plotly.graph_objects as go

    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(M, Y, bins=50)

    fig = go.Figure(data=go.Heatmap(
        z=hist.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale='Hot',
        colorbar=dict(title="Density")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Metabolic Load (M)",
        yaxis_title="Mythic Integration (Y)",
        height=600
    )

    return fig


# -----------------------------
# REGIME DETECTION
# -----------------------------

def detect_regimes(M, Y, B, n_regimes=3):
    """
    Cluster phase-space trajectory into discrete regimes using k-means

    Returns: (regime_labels, cluster_centers)
    """
    from sklearn.cluster import KMeans

    # Prepare data
    X = np.column_stack([M, Y, B])
    X = X[~np.isnan(X).any(axis=1)]  # Remove NaN rows

    if len(X) < n_regimes:
        return None, None

    # Fit k-means
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    return labels, centers


# -----------------------------
# MASTER FUNCTION
# -----------------------------

def generate_cams_attractor(filepath, society_name, use_plotly=True):
    """
    Generate complete phase-space attractor visualization from CAMS CSV

    Args:
        filepath: Path to CAMS CSV file
        society_name: Name for plot titles
        use_plotly: If True, use Plotly (interactive), else matplotlib

    Returns:
        (fig_3d, fig_density, M, Y, B, years)
    """
    df = load_cams(filepath)
    M, Y, B = compute_fields(df)

    # Align indices
    common = M.dropna().index.intersection(Y.dropna().index).intersection(B.dropna().index)

    M = M.loc[common]
    Y = Y.loc[common]
    B = B.loc[common]
    years = common.values

    if len(M) < 2:
        raise ValueError("Insufficient data points for attractor visualization")

    # Smoothing
    M_s = smooth(M)
    Y_s = smooth(Y)
    B_s = smooth(B)

    # Generate plots
    if use_plotly:
        fig_3d = plot_attractor_3d_plotly(M_s, Y_s, B_s, years, f"{society_name} CAMS Attractor")
        fig_density = plot_density_projection_plotly(M_s, Y_s, f"{society_name} Density Projection")
    else:
        fig_3d = plot_attractor_3d(M_s, Y_s, B_s, f"{society_name} CAMS Attractor")
        fig_density = plot_density_projection(M_s, Y_s, f"{society_name} Density Projection")

    return fig_3d, fig_density, M_s, Y_s, B_s, years


# -----------------------------
# EXAMPLE USAGE
# -----------------------------

if __name__ == "__main__":
    # Example: generate_cams_attractor("path_to_dataset.csv", "United States")
    print("CAMS Phase-Space Attractor Generator")
    print("Usage: from src.cams_attractor import generate_cams_attractor")
    print("       fig_3d, fig_density, M, Y, B, years = generate_cams_attractor('data.csv', 'USA')")
