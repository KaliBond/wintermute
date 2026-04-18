"""
cams_spectral.py
================
Spectral coherence extension for CAMS v2.3.

Replaces (and supplements) the mean-bond-strength Lambda with the Fiedler
eigenvalue of the weighted bond-graph Laplacian — a topologically richer
measure of system integration.

WHY THIS MATTERS
----------------
Mean Lambda is the average pairwise bond strength across all 28 node pairs.
It tells you "on average, how well-coupled are the eight institutions?"

Spectral Lambda (Fiedler value) asks a harder question:
"Could the system fragment — even if average coupling looks fine?"

A bottleneck node can hold two otherwise-healthy clusters together by a
single weak bond. Mean Lambda averages over all 28 pairs and masks the
bottleneck. The Fiedler value is the *minimum* flow capacity across any
partition of the graph — it goes low exactly when a bottleneck is present.

HIGH MEAN, LOW SPECTRAL → structural fragility signature:
  The system looks coordinated on average but is one node-failure
  away from institutional fragmentation.

METRICS PROVIDED
----------------
Lambda_mean       : original CAMS mean bond strength (unchanged)
Lambda_spectral   : Fiedler eigenvalue of the normalised Laplacian ∈ [0, n/(n-1)]
Lambda_spec_norm  : Lambda_spectral / (n/(n-1)) ∈ [0, 1] — directly comparable
                   to Lambda_mean
divergence        : Lambda_mean − Lambda_spec_norm (>0 = fragility premium)
fragile           : divergence exceeds user-set threshold
cluster_A/B       : Fiedler-vector bipartition — which nodes are on each side
                   of the structural fault line
bottleneck_nodes  : nodes whose Fiedler-vector component is near zero —
                   the load-bearing institutional links

USAGE
-----
    from cams_v23_tests import panel_to_year_matrix, series_lambda_and_health
    from cams_spectral  import (lambda_spectral, series_spectral,
                                diagnose_structural_fragility,
                                print_spectral_report)

    ym   = panel_to_year_matrix(your_df)
    spec = series_spectral(ym)           # per-year DataFrame
    diag = diagnose_structural_fragility(ym)
    print_spectral_report(diag, name="Rome")

Compatible with: cams_v23_tests.py (no changes to that file required).
Requires: numpy, pandas (no new deps).

Kari McKern / CAMS v2.3 spectral extension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Optional


# ── re-import core CAMS arithmetic ─────────────────────────────────────────
# (duplicated here so this file is self-contained if needed)

def _bond_strength(Vi: float, Vj: float) -> float:
    """CAMS v2.3 canonical bond: Bij = sqrt(max(Vi+8,0)*max(Vj+8,0)) / 32"""
    return float(np.sqrt(max(Vi + 8.0, 0.0) * max(Vj + 8.0, 0.0)) / 32.0)


def _build_bond_matrix(node_values: np.ndarray) -> np.ndarray:
    """Build the full n×n symmetric weighted adjacency matrix of bond strengths."""
    n = len(node_values)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i, j] = _bond_strength(node_values[i], node_values[j])
    return W


def _normalised_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Normalised graph Laplacian: L_norm = D^{-1/2} (D - W) D^{-1/2}
    Eigenvalues lie in [0, 2].  Second-smallest = Fiedler value.
    """
    d = W.sum(axis=1)
    D = np.diag(d)
    L = D - W
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    Di = np.diag(d_inv_sqrt)
    return Di @ L @ Di


# ── Core spectral functions ─────────────────────────────────────────────────

def lambda_spectral(node_values: np.ndarray,
                    normalised: bool = True) -> float:
    """
    Algebraic connectivity (Fiedler eigenvalue) of the weighted bond-graph Laplacian.

    Parameters
    ----------
    node_values : 1-D array of 8 node V-scores for a single year.
    normalised  : If True (default), use the normalised Laplacian.
                  Eigenvalues ∈ [0, 2]; comparable across different bond-strength
                  scales and society sizes.
                  If False, use the combinatorial Laplacian (scale-dependent).

    Returns
    -------
    float : Fiedler value.  Higher = more robustly integrated.
            0 = graph is disconnected (catastrophic coordination failure).

    Interpretation (normalised Laplacian, n=8 nodes):
        0.00–0.20  : severe structural fragility, near-disconnected
        0.20–0.50  : weak integration, bottleneck likely
        0.50–0.90  : moderate integration
        0.90–1.14  : strong integration (max ≈ n/(n-1) = 8/7 ≈ 1.143)
    """
    W = _build_bond_matrix(np.asarray(node_values, dtype=float))
    if normalised:
        L = _normalised_laplacian(W)
    else:
        d = W.sum(axis=1)
        L = np.diag(d) - W
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def fiedler_vector(node_values: np.ndarray,
                   normalised: bool = True) -> np.ndarray:
    """
    Fiedler eigenvector — the algebraic partition of the bond graph.

    Positive components form cluster A; negative components form cluster B.
    Components near zero identify bottleneck nodes: institutions sitting on
    the structural fault line between the two clusters.

    Returns
    -------
    np.ndarray of shape (n,) — one component per node in the same order
    as node_values.
    """
    W = _build_bond_matrix(np.asarray(node_values, dtype=float))
    if normalised:
        L = _normalised_laplacian(W)
    else:
        d = W.sum(axis=1)
        L = np.diag(d) - W
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[1]
    return eigvecs[:, idx]


def lambda_spec_normalised(node_values: np.ndarray) -> float:
    """
    Spectral Lambda rescaled to [0, 1] by dividing by the theoretical maximum
    n/(n-1) for a complete graph on n nodes.

    This makes it directly comparable to Lambda_mean (also roughly ∈ [0, 1]).
    """
    n = len(node_values)
    raw = lambda_spectral(node_values, normalised=True)
    max_fiedler = n / (n - 1)   # 8/7 ≈ 1.143 for n=8
    return float(raw / max_fiedler)


# ── Time-series variants ────────────────────────────────────────────────────

def series_spectral(year_matrix: pd.DataFrame,
                    bottleneck_tol: float = 0.15) -> pd.DataFrame:
    """
    For each year in a CAMS year-matrix (output of panel_to_year_matrix),
    compute the full suite of spectral coherence metrics.

    Parameters
    ----------
    year_matrix     : year-indexed DataFrame, one column per node (V values).
    bottleneck_tol  : Fiedler-vector components with |x| < tol are classified
                      as bottleneck nodes (sitting on the structural fault line).

    Returns
    -------
    DataFrame with columns:
        Lambda_mean       : original mean bond strength
        Lambda_spectral   : raw Fiedler eigenvalue (normalised Laplacian)
        Lambda_spec_norm  : Fiedler value rescaled to [0, 1]
        divergence        : Lambda_mean − Lambda_spec_norm  (fragility premium)
        cluster_A         : nodes in the positive Fiedler partition
        cluster_B         : nodes in the negative Fiedler partition
        bottleneck_nodes  : nodes sitting near the structural fault line
    """
    nodes = list(year_matrix.columns)
    n = len(nodes)
    max_fiedler = n / (n - 1)
    rows = []

    for yr, row in year_matrix.iterrows():
        vals = row.values.astype(float)

        # Mean Lambda (original)
        pairs = [_bond_strength(vals[i], vals[j])
                 for i, j in combinations(range(n), 2)]
        lam_mean = float(np.mean(pairs))

        # Spectral Lambda
        lam_spec = lambda_spectral(vals)
        lam_norm = lam_spec / max_fiedler

        # Fiedler vector → partition
        fv = fiedler_vector(vals)
        cluster_a = [nodes[i] for i in range(n) if fv[i] >= bottleneck_tol]
        cluster_b = [nodes[i] for i in range(n) if fv[i] <= -bottleneck_tol]
        bottleneck = [nodes[i] for i in range(n) if abs(fv[i]) < bottleneck_tol]

        rows.append({
            'Year':             yr,
            'Lambda_mean':      lam_mean,
            'Lambda_spectral':  lam_spec,
            'Lambda_spec_norm': lam_norm,
            'divergence':       lam_mean - lam_norm,
            'cluster_A':        '+'.join(sorted(cluster_a)) or '—',
            'cluster_B':        '+'.join(sorted(cluster_b)) or '—',
            'bottleneck_nodes': '+'.join(sorted(bottleneck)) or 'none',
        })

    return pd.DataFrame(rows).set_index('Year')


def diagnose_structural_fragility(year_matrix: pd.DataFrame,
                                   divergence_threshold: float = 0.12,
                                   spec_low_threshold: float = 0.40,
                                   bottleneck_tol: float = 0.15) -> pd.DataFrame:
    """
    Full structural fragility diagnostic.

    A year is flagged `fragile` when BOTH conditions hold:
        1. divergence  > divergence_threshold   (mean looks fine, spectral low)
        2. Lambda_spec_norm < spec_low_threshold (absolute structural weakness)

    This double-gate avoids false positives in genuinely low-Lambda periods
    where both measures agree the system is struggling.

    Returns the series_spectral DataFrame augmented with a `fragile` boolean
    and a `fragility_score` ∈ [0, 1] combining both signals.
    """
    spec = series_spectral(year_matrix, bottleneck_tol=bottleneck_tol)

    # Normalise each signal to [0,1] for composite score
    div_norm  = (spec['divergence']        / divergence_threshold).clip(0, 1)
    spec_norm = (spec_low_threshold - spec['Lambda_spec_norm']).clip(0, None)
    spec_norm = (spec_norm / spec_low_threshold).clip(0, 1)

    spec['fragility_score'] = ((div_norm + spec_norm) / 2).round(3)
    spec['fragile'] = (
        (spec['divergence']        > divergence_threshold) &
        (spec['Lambda_spec_norm']  < spec_low_threshold)
    )
    return spec


# ── Reporting ───────────────────────────────────────────────────────────────

def print_spectral_report(diag: pd.DataFrame,
                          name: str = "society",
                          top_n: int = 5) -> None:
    """
    Print a readable summary of the structural fragility diagnostic.

    Shows:
      • Overall spectral vs. mean Lambda statistics
      • Top-N highest fragility years with their fault-line partition
      • Most commonly implicated bottleneck nodes
    """
    print(f"\n{'='*72}")
    print(f"  CAMS Spectral Coherence Report — {name}")
    print(f"{'='*72}")

    print(f"\n  Years analysed : {diag.index.min()}–{diag.index.max()}  "
          f"({len(diag)} observations)")
    print(f"\n  Lambda (mean bond strength)  ─ original CAMS measure")
    print(f"    mean={diag['Lambda_mean'].mean():.3f}  "
          f"min={diag['Lambda_mean'].min():.3f}  "
          f"max={diag['Lambda_mean'].max():.3f}")
    print(f"\n  Lambda_spec_norm (Fiedler, rescaled)  ─ NEW spectral measure")
    print(f"    mean={diag['Lambda_spec_norm'].mean():.3f}  "
          f"min={diag['Lambda_spec_norm'].min():.3f}  "
          f"max={diag['Lambda_spec_norm'].max():.3f}")
    print(f"\n  Structural divergence (mean − spectral norm)")
    print(f"    mean={diag['divergence'].mean():.3f}  "
          f"max={diag['divergence'].max():.3f}")

    fragile_years = diag[diag['fragile']]
    print(f"\n  Fragility alerts : {len(fragile_years)} of {len(diag)} years "
          f"({len(fragile_years)/len(diag):.1%})")

    if len(fragile_years) > 0:
        print(f"\n  Top-{min(top_n, len(fragile_years))} highest-fragility years:")
        top = fragile_years.nlargest(top_n, 'fragility_score')
        for yr, row in top.iterrows():
            print(f"    {yr:5d}  fragility={row['fragility_score']:.2f}  "
                  f"divergence={row['divergence']:+.3f}  "
                  f"spectral={row['Lambda_spec_norm']:.3f}  "
                  f"fault-line: [{row['cluster_A']}] | [{row['cluster_B']}]  "
                  f"bottleneck: {row['bottleneck_nodes']}")

    # Most common bottleneck nodes
    all_bottlenecks = []
    for bn in diag['bottleneck_nodes']:
        if bn != 'none':
            all_bottlenecks.extend(bn.split('+'))
    if all_bottlenecks:
        from collections import Counter
        counts = Counter(all_bottlenecks)
        print(f"\n  Most-implicated bottleneck nodes (all years):")
        for node, cnt in counts.most_common(4):
            print(f"    {node:12s}  {cnt} years ({cnt/len(diag):.1%})")

    print(f"\n{'='*72}\n")


# ── Integration with run_battery ────────────────────────────────────────────

def run_spectral_test(df: pd.DataFrame,
                      name: str = "society",
                      divergence_threshold: float = 0.12,
                      spec_low_threshold: float = 0.40,
                      save_csv: bool = True) -> pd.DataFrame:
    """
    Drop-in replacement / augmentation for the Lambda section of run_battery.

    Computes both measures, prints the spectral report, and returns the full
    diagnostic DataFrame for further analysis or Granger testing.

    To wire into your existing run_full_test_battery:

        from cams_spectral import run_spectral_test
        ...
        spec_diag = run_spectral_test(df, name=name)

    The returned DataFrame has all columns of series_spectral plus fragility
    columns — you can pass spec_diag['Lambda_spec_norm'] to walk_forward_predict
    or the Granger functions in place of (or alongside) the original Lambda.
    """
    from cams_v23_tests import panel_to_year_matrix  # lazy import, avoids circular dep

    ym   = panel_to_year_matrix(df)
    diag = diagnose_structural_fragility(
        ym,
        divergence_threshold=divergence_threshold,
        spec_low_threshold=spec_low_threshold,
    )
    print_spectral_report(diag, name=name)

    if save_csv:
        fname = f"cams_spectral_{name.replace(' ', '_').lower()}.csv"
        diag.to_csv(fname)
        print(f"  Saved → {fname}")

    return diag


# ── Granger-ready helper ─────────────────────────────────────────────────────

def spectral_granger_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: return a year-indexed DataFrame with BOTH Lambda columns
    ready for grangercausalitytests.

    Columns: Lambda_mean, Lambda_spec_norm, divergence

    Typical use:
        from cams_spectral import spectral_granger_frame
        frame = spectral_granger_frame(your_rome_df)

        # Does spectral fragility LEAD mean coherence breakdown?
        grangercausalitytests(frame[['Lambda_mean', 'divergence']], maxlag=5)
    """
    from cams_v23_tests import panel_to_year_matrix
    ym   = panel_to_year_matrix(df)
    spec = series_spectral(ym)
    return spec[['Lambda_mean', 'Lambda_spec_norm', 'divergence']]


# ── Quick synthetic sanity-check ─────────────────────────────────────────────

def _sanity_check() -> None:
    """
    Verify the spectral functions on two toy cases:

    Case A — perfectly balanced 8-node system (all V = 5.0)
      All bonds identical → Fiedler value should be n/(n-1) = 8/7 ≈ 1.143
      Lambda_spec_norm ≈ 1.0

    Case B — one severely degraded node (V_0 = -8.0, rest = 5.0)
      Node 0's bonds are near-zero → creates a bottleneck
      Lambda_spec_norm should drop well below Lambda_mean
    """
    n = 8
    max_fiedler = n / (n - 1)

    print("  Sanity check A — uniform system (all V=5.0)")
    vals_a = np.full(n, 5.0)
    lm_a   = np.mean([_bond_strength(vals_a[i], vals_a[j])
                       for i, j in combinations(range(n), 2)])
    ls_a   = lambda_spectral(vals_a)
    print(f"    Lambda_mean      = {lm_a:.4f}")
    print(f"    Lambda_spectral  = {ls_a:.4f}  (expected ≈ {max_fiedler:.4f})")
    print(f"    Lambda_spec_norm = {ls_a/max_fiedler:.4f}  (expected ≈ 1.000)")

    print("\n  Sanity check B — one degraded node (V_0 = -8.0, rest = 5.0)")
    vals_b = np.array([-8.0] + [5.0] * 7)
    lm_b   = np.mean([_bond_strength(vals_b[i], vals_b[j])
                       for i, j in combinations(range(n), 2)])
    ls_b   = lambda_spectral(vals_b)
    fv_b   = fiedler_vector(vals_b)
    print(f"    Lambda_mean      = {lm_b:.4f}  ← still looks reasonable")
    print(f"    Lambda_spectral  = {ls_b:.4f}")
    print(f"    Lambda_spec_norm = {ls_b/max_fiedler:.4f}  ← divergence = {lm_b - ls_b/max_fiedler:+.4f}")
    print(f"    Fiedler vector   : {np.round(fv_b, 3)}")
    print(f"    Node 0 component : {fv_b[0]:.4f}  "
          f"({'bottleneck' if abs(fv_b[0]) < 0.15 else 'in cluster'})")
    print()

    assert ls_a / max_fiedler > 0.95, "Case A spectral norm should be near 1.0"
    assert (lm_b - ls_b / max_fiedler) > 0.10, "Case B divergence should be > 0.10"
    print("  ✓ Both sanity checks passed.\n")


if __name__ == "__main__":
    print(f"{'='*72}")
    print("  cams_spectral.py — self-test")
    print(f"{'='*72}\n")
    _sanity_check()
