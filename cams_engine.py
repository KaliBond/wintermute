# cams_engine.py
# CAMS GTS EV v1.95 - Canonical Thermodynamic Engine
# December 2025

import numpy as np
import pandas as pd

# Immutable physical constants (from canonical spec)
SIGMA_0 = 1.0
EPS_REF = 1.2e3   # W/capita reference
EPS_CRIT = 12e3   # W/capita critical
KAPPA = 1e-14
OMEGA = 2.5
CHI = 0.08

def run_cams_engine(df_year: pd.DataFrame, EROEI: float, pop_M: float, phi_export: float = 0.0):
    """
    Run CAMS GTS EV v1.95 engine on a single year of data.

    Parameters:
    -----------
    df_year : pd.DataFrame
        DataFrame containing node data for a single year (8 nodes expected)
    EROEI : float
        Energy Return on Energy Invested (societal level)
    pop_M : float
        Population in millions
    phi_export : float
        Entropy export/offload parameter (W·K⁻¹·cap⁻¹)

    Returns:
    --------
    dict : Dictionary containing all CAMS metrics or None if invalid data
    """
    df = df_year.copy()
    pop = pop_M * 1e6

    # Map node names (flexible to handle different column naming)
    node_col = df.columns[df.columns.str.contains('Node', case=False)][0]

    # Remove duplicates - take first occurrence of each node
    df = df.drop_duplicates(subset=[node_col], keep='first')

    nodes = df[node_col].unique()
    if len(nodes) != 8:
        return None

    # Ensure we have exactly 8 rows
    if len(df) != 8:
        return None

    # Extract C, K_raw, S, A
    C = df['Coherence'].values
    K_raw = df['Capacity'].values
    S = df['Stress'].values
    A = df['Abstraction'].values

    # Thermodynamic Capacity correction
    eta_i = np.ones(8) * 0.95  # assume equal allocation for now
    K = 10 * np.minimum(1, eta_i * EROEI * EPS_REF / EPS_CRIT)

    # Node Value
    NV = C + K - S + 0.5 * A

    # Bond strength (simplified D_KL = 0.12 average, adjustable later)
    D_KL = np.ones((8,8)) * 0.12
    np.fill_diagonal(D_KL, 0)
    B_matrix = C[:, None] * C[None, :] * np.exp(-np.abs(K[:, None] - K[None, :])) * (1 - OMEGA * D_KL)
    B_mean = B_matrix.mean()

    # Dual-mode scalars
    E_total = 3.5e3 * pop / 1e9  # rough average ~3.5 kW/cap global
    E_net = E_total * (1 - 1/EROEI)
    P_Abs = KAPPA * pop * (A * 1e8).sum()
    E_star = 0.1e3 * pop / 1e9   # 0.1 kW/cap reference
    Psi = np.log(max(E_net - P_Abs, 1e-9) / E_star) * (C * A).sum()
    Phi = (K * (11 - S)).sum()
    Phi -= CHI * phi_export * pop / 1e9  # entropy export correction

    R = Phi / Psi if Psi > 0 else 999
    H = 100 * (C * K).sum() / ((C * K).max() * 8)  # normalized to theoretical max

    # Classification
    if R < 1.0 and H > 65 and B_mean > 2.0:
        classification = "Resilient Frontier"
    elif R < 1.0 and H > 65:
        classification = "Stable Core"
    elif 1 <= R <= 2.2 and 35 <= H <= 65:
        classification = "Transitional"
    elif 2.2 < R <= 4.5 and H < 35:
        classification = "Fragile"
    else:
        classification = "Terminal"

    crisis_prob = min(99, int(5 + 25 * (R - 1)**2 / 10 + (100 - H)/2))

    result = {
        '⟨C⟩': C.mean(), '⟨K⟩': K.mean(), '⟨S⟩': S.mean(), '⟨A⟩': A.mean(),
        '⟨NV⟩': NV.mean(), '⟨B⟩': B_mean, 'Ψ': Psi, 'Φ': Phi, 'R': R,
        'H%': H, 'Φ_export': phi_export, 'Class': classification,
        'CrisisProb': f"{crisis_prob}%"
    }
    return result
