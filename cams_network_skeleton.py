#!/usr/bin/env python3
"""
CAMS / JUNO Network Model Skeleton
----------------------------------
8 nodes × 4 metrics = 32-dimensional dynamical system on a weighted digraph.
Integrates the exact JUNO node-value formula with stress/capacity transport
and a simple game-theoretic payoff layer.

Dependencies: networkx, numpy, scipy
"""

import numpy as np
from scipy.integrate import solve_ivp
import networkx as nx
from typing import Dict, Tuple

# ------------------------------------------------------------------
# 1. Canonical node order (never change this order)
# ------------------------------------------------------------------
NODES = ["Helm", "Shield", "Archive", "Lore", "Stewards", "Craft", "Hands", "Flow"]
N = len(NODES)          # 8
METRICS = ["C", "K", "S", "A"]   # Coherence, Capacity, Stress, Abstraction
M = len(METRICS)        # 4
STATE_DIM = N * M       # 32

# ------------------------------------------------------------------
# 2. Exact JUNO node-value (v1.2-Final)
# ------------------------------------------------------------------
def juno_V(C: float, K: float, S: float, A: float) -> float:
    """V = C + K - S + A/2   (6 d.p. policy elsewhere)"""
    return C + K - S + 0.5 * A

def juno_V_vector(state: np.ndarray) -> np.ndarray:
    """Compute V for all 8 nodes from a flattened 32-vector."""
    V = np.zeros(N)
    for i in range(N):
        C, K, S, A = state[i*M : (i+1)*M]
        V[i] = juno_V(C, K, S, A)
    return V

# ------------------------------------------------------------------
# 3. Network construction
# ------------------------------------------------------------------
def make_graph(B: np.ndarray, w: np.ndarray) -> nx.DiGraph:
    """Create a directed NetworkX graph from bond matrix B and weights w."""
    G = nx.DiGraph()
    for i, name in enumerate(NODES):
        G.add_node(name, weight=float(w[i]))
    for i in range(N):
        for j in range(N):
            if B[i, j] > 0:
                G.add_edge(NODES[j], NODES[i], weight=float(B[i, j]))
    return G

# Example bond matrix (placeholder – replace with real calibrated values)
# Rows = receiver, columns = sender
B_example = np.array([
    # H    S    A    L   St    C   Ha    F
    [0.4, 0.6, 0.3, 0.5, 0.2, 0.1, 0.1, 0.2],  # Helm
    [0.5, 0.3, 0.2, 0.4, 0.1, 0.1, 0.2, 0.1],  # Shield
    [0.2, 0.1, 0.5, 0.6, 0.3, 0.2, 0.1, 0.1],  # Archive
    [0.4, 0.3, 0.5, 0.4, 0.2, 0.1, 0.1, 0.2],  # Lore
    [0.2, 0.1, 0.3, 0.2, 0.4, 0.3, 0.4, 0.3],  # Stewards
    [0.1, 0.1, 0.2, 0.1, 0.3, 0.5, 0.4, 0.3],  # Craft
    [0.1, 0.2, 0.1, 0.1, 0.4, 0.3, 0.4, 0.5],  # Hands
    [0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.3],  # Flow
], dtype=float)

# Relative institutional weights (sum ≈ 1)
w_example = np.array([0.08, 0.10, 0.09, 0.09, 0.12, 0.14, 0.20, 0.18])

# ------------------------------------------------------------------
# 4. ODE right-hand side
# ------------------------------------------------------------------
def _clamp_deriv(value: float, deriv: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """
    Project the derivative at a state-space boundary: if the state is
    already at or past a bound and the raw derivative points further out,
    zero it. This is what actually guarantees every metric stays in its
    canonical [0,10] range -- coefficient tuning (gamma, delta, ...) only
    ever makes a *specific* bond matrix well-behaved, never guarantees it
    for whatever matrix gets plugged in next.
    """
    if value <= lo and deriv < 0.0:
        return 0.0
    if value >= hi and deriv > 0.0:
        return 0.0
    return deriv

def cams_ode(t: float, y: np.ndarray, B: np.ndarray, w: np.ndarray,
             gamma: float = 0.30, delta: float = 0.08,
             alpha: float = 0.25, beta: float = 0.12,
             S_max: float = 10.0) -> np.ndarray:
    """
    dy/dt for the 32-dimensional state. y is flattened [C0,K0,S0,A0, C1,K1,...].

    NOTE ON STABILITY: every metric is boundary-clamped to [0,10] via
    _clamp_deriv below, so the returned derivatives never push a metric
    out of its canonical range regardless of B/w/gamma. This does not mean
    the *interior* dynamics are well-calibrated -- gamma=0.30 is only
    tuned to damp the specific B_example/w_example matrices below (the
    linear part of the stress-transport term alone has a dominant
    eigenvalue of +0.13 at gamma=0.15, i.e. unstable in the interior even
    before hitting a boundary). Check the dominant eigenvalue of
    (B*w - gamma*I) before trusting a new bond matrix's interior behaviour;
    the clamp is a safety net, not a substitute for calibration.
    """
    dydt = np.zeros_like(y)
    V = juno_V_vector(y)

    for i in range(N):
        C, K, S, A = y[i*M : (i+1)*M]

        # --- Stress transport ---
        inflow_S = np.sum(B[i, :] * w * y[2::M])          # S components
        dS = inflow_S - gamma * S
        # endogenous crisis-acceleration term, logistically saturated at S_max
        # so it accelerates stress growth above threshold 6 without diverging
        if S > 6.0:
            dS += 0.3 * (S - 6.0) * max(0.0, 1.0 - S / S_max)

        # --- Capacity transport ---
        outflow_K = np.sum(B[:, i] * w[i] * y[1::M])      # K sent downstream
        # Growth term is logistically self-capped at K_max=10. A raw
        # K*(1-K/K_max) logistic term is only stable for K >= 0 -- if K ever
        # goes negative (it can, via outflow_K or the -delta*K*S term), the
        # factor (1-K/K_max) exceeds 1 and the term flips sign, driving K to
        # -inf in finite time. Clamping the base to max(K,0) and both
        # modulating factors to [0,1] keeps growth well-defined for any K.
        K_max = 10.0
        K_pos = max(K, 0.0)
        growth_factor = np.clip(1.0 - K_pos / K_max, 0.0, 1.0)
        stress_factor = np.clip(1.0 - S / 10.0, 0.0, 1.0)
        dK = -outflow_K + 0.4 * K_pos * growth_factor * stress_factor - delta * K * S
        # local production term (Craft & Hands heavier)
        if NODES[i] in ("Craft", "Hands", "Stewards"):
            dK += 0.15 * max(0, 8 - S)

        # --- Coherence (slow alignment + stress damage) ---
        neighbour_C = np.sum(B[i, :] * y[0::M]) / (np.sum(B[i, :]) + 1e-9)
        dC = alpha * (neighbour_C - C) - beta * S

        # --- Abstraction (very slow) ---
        dA = 0.02 * max(0, C - 5) - 0.01 * S

        # Boundary clamp: no metric may be driven outside [0,10] regardless
        # of which interior terms or bond matrix produced the raw derivative
        dydt[i*M + 0] = _clamp_deriv(C, dC)
        dydt[i*M + 1] = _clamp_deriv(K, dK)
        dydt[i*M + 2] = _clamp_deriv(S, dS)
        dydt[i*M + 3] = _clamp_deriv(A, dA)

    return dydt

# ------------------------------------------------------------------
# 5. Convenience helpers
# ------------------------------------------------------------------
def pack_state(metrics: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Turn a nested dict into a 32-vector."""
    y = np.zeros(STATE_DIM)
    for i, node in enumerate(NODES):
        for j, m in enumerate(METRICS):
            y[i*M + j] = metrics[node][m]
    return y

def unpack_state(y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Turn a 32-vector back into nested dict."""
    out = {}
    for i, node in enumerate(NODES):
        out[node] = {m: float(y[i*M + j]) for j, m in enumerate(METRICS)}
    return out

def clip_state(y: np.ndarray, lo: float = 0.0, hi: float = 10.0) -> np.ndarray:
    """
    Second safety layer: clip a state vector (or a full trajectory array)
    into [0,10]. The _clamp_deriv boundary guard in cams_ode should keep
    RK45 from leaving this range in the first place, but the solver's
    intermediate stage evaluations and tolerance-driven overshoot can still
    produce tiny excursions (see the -0.0000 floating-point noise observed
    in practice) -- this belt-and-suspenders clip makes the guarantee
    exact rather than approximate. Apply to sol.y (shape STATE_DIM x n_times)
    or any single state vector.
    """
    return np.clip(y, lo, hi)

def print_snapshot(y: np.ndarray, label: str = ""):
    """Pretty-print current V and metrics."""
    print(f"\n=== {label} ===")
    V = juno_V_vector(y)
    for i, node in enumerate(NODES):
        C, K, S, A = y[i*M:(i+1)*M]
        print(f"{node:10s}  V={V[i]:6.3f}   C={C:5.2f}  K={K:5.2f}  S={S:5.2f}  A={A:5.2f}")

# ------------------------------------------------------------------
# 6. Minimal runnable example
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --- Initial condition (toy numbers – replace with real scores) ---
    init = {
        "Helm":     {"C": 5.2, "K": 6.1, "S": 4.8, "A": 5.5},
        "Shield":   {"C": 6.0, "K": 5.8, "S": 5.5, "A": 4.9},
        "Archive":  {"C": 5.8, "K": 5.5, "S": 4.2, "A": 6.2},
        "Lore":     {"C": 4.9, "K": 5.0, "S": 5.8, "A": 5.7},
        "Stewards": {"C": 5.5, "K": 6.4, "S": 4.0, "A": 5.1},
        "Craft":    {"C": 6.2, "K": 7.0, "S": 3.8, "A": 5.4},
        "Hands":    {"C": 5.0, "K": 6.8, "S": 5.2, "A": 4.3},
        "Flow":     {"C": 5.7, "K": 6.5, "S": 4.5, "A": 4.8},
    }
    y0 = pack_state(init)

    print_snapshot(y0, "t = 0 (initial)")

    # --- Integrate ---
    t_span = (0.0, 12.0)
    sol = solve_ivp(
        fun=lambda t, y: cams_ode(t, y, B_example, w_example),
        t_span=t_span,
        y0=y0,
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
        dense_output=True
    )

    y_final = clip_state(sol.y[:, -1])
    print_snapshot(y_final, f"t = {sol.t[-1]:.1f}")

    # Optional: build the NetworkX graph for further analysis
    G = make_graph(B_example, w_example)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Strongly connected components:", list(nx.strongly_connected_components(G)))
