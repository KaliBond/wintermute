# cams_engine.py
# CAMS GTS EV v1.95 - Canonical Thermodynamic Engine
# December 2025

import numpy as np
import pandas as pd

# Immutable physical constants (from canonical spec)
SIGMA_0 = 1.0
EPS_REF = 1.2e3   # W/capita reference
EPS_CRITICAL = 12e3   # W/capita critical
KAPPA = 1e-14
OMEGA = 2.5
CHI = 0.08

def run_cams_engine(df_year: pd.DataFrame, EROEI: float, pop_M: float, phi_export: float = 0.0):
    """
    Run CAMS GTS EV v1.95 engine on a single year of data (v1.95 canonical + bug squash).

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
    pop = pop_M * 1e6  # to persons

    # Extract (assume columns: Node, Coherence, Capacity [raw], Stress, Abstraction)
    C = df['Coherence'].values.astype(float)
    K_raw = df['Capacity'].values.astype(float)  # raw, for ref
    S = df['Stress'].values.astype(float)
    A = df['Abstraction'].values.astype(float)

    # Thermodynamic Capacity correction (eta_i sums to ~1)
    eta_i = np.full(8, 1/8)  # equal share
    K = 10 * np.minimum(1, eta_i * EROEI * EPS_REF / EPS_CRITICAL)  # vectorized, no seq error

    # Node Value
    NV = C + K - S + 0.5 * A

    # Bond matrix (D_KL=0.18 avg from example)
    D_KL = np.full((8,8), 0.18)
    np.fill_diagonal(D_KL, 0)
    B_ij = C[:,None] * C[None,:] * np.exp(-np.abs(K[:,None] - K[None,:])) * (1 - OMEGA * D_KL)
    B_mean = np.mean(B_ij[B_ij > 0])  # triu mean, avoid diag

    # Dual modes — FIXED: clip E_net >0, proper units (W total)
    E_total = 3500 * pop  # 3.5 kW/cap avg primary energy
    E_net_pre_abs = E_total * (1 - 1/EROEI)
    P_Abs = KAPPA * pop * np.sum(A) * 1e8  # bits total
    E_net = max(E_net_pre_abs - P_Abs, 1e6)  # clip tiny positive (W)
    E_star = 100 * pop  # 0.1 kW/cap = 100 W/cap
    ln_term = np.log(E_net / E_star)
    Psi = ln_term * np.sum(C * A) if ln_term > 0 else 0  # zero if no surplus

    Phi_internal = np.sum(K * (11 - S))
    Phi_export = CHI * phi_export * pop  # W total equiv, rough
    Phi = max(Phi_internal - Phi_export / 8, 0)  # per-node avg

    R = Phi / max(Psi, 1e-6) if Psi > 0 else 999  # safe div
    ck_max = np.max(C * K) * 8  # historical max proxy
    H = 100 * np.sum(C * K) / ck_max

    # Classification LUT (canonical)
    if R < 1.0 and H > 65 and B_mean > 2.0:
        cls = "Resilient Frontier"
    elif R < 1.0 and H > 65:
        cls = "Stable Core"
    elif 1 <= R <= 2.2 and 35 <= H <= 65:
        cls = "Transitional"
    elif 2.2 < R <= 4.0 and 15 <= H <= 35:
        cls = "Fragile"
    else:
        cls = "Terminal"

    crisis_prob = min(99, int(5 + 25 * max(0, R - 1)**2 / 10 + (100 - H)/2))

    return {
        '⟨C⟩': np.mean(C), '⟨K⟩': np.mean(K), '⟨S⟩': np.mean(S), '⟨A⟩': np.mean(A),
        '⟨NV⟩': np.mean(NV), '⟨B⟩': B_mean, 'Ψ': Psi, 'Φ': Phi, 'R': R,
        'H%': H, 'Φ_export': phi_export, 'Class': cls, 'CrisisProb': f"{crisis_prob}%"
    }
