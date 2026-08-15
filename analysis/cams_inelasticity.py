"""
CAMS Temporal Inelasticity Analysis
====================================
Computes pair-specific coupling measures from longitudinal Block 1 CAMS data.

The current JUNO bond formula B_ij = w_i * w_j is rank-1 — every bond is
algebraically determined by node-level weights, so there is no pair-specific
information. Temporal inelasticity I_ij measures instead whether the *relative
configuration* of two nodes resists independent displacement through time.

Outputs
-------
1. <prefix>_inelasticity.csv  — I_ij per pair per society, plus coupling class
2. <prefix>_cond_response.csv — CR_{j←i}: mean response of j when i moved >1σ
3. <prefix>_asymmetry.csv     — |CR_{j←i} - CR_{i←j}|, ranked by pair

Usage
-----
    python cams_inelasticity.py block1.csv [block1b.csv ...]
    python cams_inelasticity.py --glob "analysis/*_block1.csv"
    python cams_inelasticity.py --help

References
----------
I_ij = 1 - Var(ΔR_ij) / (Var(Δx_i) + Var(Δx_j))
  where R_ij(t) = x_i(t) - x_j(t), x = Node Value, Δ = year-on-year change.
I_ij > 0.7  → Rigid
I_ij 0.3–0.7 → Loosely associated (or Directionally constrained if |asym| > 1.0)
I_ij < 0.3  → Weakly coupled

CR_{j←i} = mean Δx_j(t+1) over years where |Δx_i(t)| > σ(Δx_i)
Asymmetry = |CR_{j←i}| - |CR_{i←j}|  (positive = i drives j more than j drives i)
"""

import argparse
import glob as glob_module
import sys
from pathlib import Path
from itertools import permutations, combinations

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

NODES = ['Helm', 'Shield', 'Lore', 'Stewards', 'Craft', 'Hands', 'Archive', 'Flow']

SLOW = {'Helm', 'Lore', 'Archive', 'Stewards'}
FAST = {'Shield', 'Craft', 'Hands', 'Flow'}

LOOP_TYPE = {
    (True,  True):  'Slow–Slow',
    (False, False): 'Fast–Fast',
    (True,  False): 'Cross-loop',
    (False, True):  'Cross-loop',
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_block1(path: str) -> pd.DataFrame:
    """
    Load a Block 1 CSV and return a long-format DataFrame with columns:
    Society, Year, Node, Node Value
    """
    df = pd.read_csv(path)
    # Normalise society column (some files use 'Entity' instead of 'Society')
    if 'Entity' in df.columns and 'Society' not in df.columns:
        df = df.rename(columns={'Entity': 'Society'})
    required = {'Society', 'Year', 'Node', 'Node Value'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df['Year'] = df['Year'].astype(int)
    df['Node Value'] = df['Node Value'].astype(float)
    return df[['Society', 'Year', 'Node', 'Node Value']]


def pivot_to_wide(df: pd.DataFrame, society: str) -> pd.DataFrame:
    """
    Pivot long-format data for one society to wide format:
    index = Year, columns = Node, values = Node Value
    Sorted by Year, restricted to known NODES.
    """
    sub = df[df['Society'] == society].copy()
    wide = sub.pivot_table(index='Year', columns='Node', values='Node Value')
    cols = [n for n in NODES if n in wide.columns]
    wide = wide[cols].sort_index()
    return wide


# ── Core calculations ─────────────────────────────────────────────────────────

def inelasticity_matrix(wide: pd.DataFrame, society: str) -> pd.DataFrame:
    """
    Compute I_ij for all 28 unique node pairs.

    I_ij = 1 - Var(ΔR_ij) / (Var(Δx_i) + Var(Δx_j))
    where R_ij(t) = x_i(t) - x_j(t), Δ = year-on-year first difference.

    Values near 1 → relative configuration extremely stable (rigid coupling).
    Values near 0 → relative configuration as variable as the nodes themselves.
    Negative values → relationship amplifies divergence (anti-correlated movement).
    """
    nodes = list(wide.columns)
    X = wide.values.astype(float)          # (T, n_nodes)
    dX = np.diff(X, axis=0)               # (T-1, n_nodes)
    T = len(dX)

    rows = []
    for ni, nj in combinations(nodes, 2):
        i, j = nodes.index(ni), nodes.index(nj)
        dR   = dX[:, i] - dX[:, j]        # ΔR_ij = Δx_i - Δx_j

        var_dR  = float(np.var(dR,        ddof=1)) if T > 1 else np.nan
        var_dXi = float(np.var(dX[:, i], ddof=1)) if T > 1 else np.nan
        var_dXj = float(np.var(dX[:, j], ddof=1)) if T > 1 else np.nan
        denom   = var_dXi + var_dXj

        I_ij = (1.0 - var_dR / denom) if denom > 1e-9 else np.nan

        # Loop classification
        is_slow_i = ni in SLOW
        is_slow_j = nj in SLOW
        loop = LOOP_TYPE[(is_slow_i, is_slow_j)]

        # Coupling class
        if np.isnan(I_ij):
            coupling = 'Undefined'
        elif I_ij > 0.7:
            coupling = 'Rigid'
        elif I_ij < 0.3:
            coupling = 'Weakly coupled'
        else:
            coupling = 'Loosely associated'   # may be updated to Directional later

        rows.append({
            'Society':   society,
            'Node_i':    ni,
            'Node_j':    nj,
            'Loop':      loop,
            'I_ij':      round(I_ij, 4) if not np.isnan(I_ij) else np.nan,
            'Var_dR':    round(var_dR,  4) if not np.isnan(var_dR)  else np.nan,
            'Var_dXi':   round(var_dXi, 4) if not np.isnan(var_dXi) else np.nan,
            'Var_dXj':   round(var_dXj, 4) if not np.isnan(var_dXj) else np.nan,
            'n_years':   T,
            'Coupling':  coupling,
        })

    return pd.DataFrame(rows)


def conditional_response(wide: pd.DataFrame, society: str,
                         threshold_sigma: float = 1.0,
                         lag: int = 1) -> pd.DataFrame:
    """
    Compute CR_{j←i}: mean response of j when i moved > threshold.

    For each ordered pair (i → j):
      - Find years t where |Δx_i(t)| > threshold * σ(Δx_i).
      - Record Δx_j(t + lag) for those years.
      - CR_{j←i} = mean of those responses.

    A positive CR_{j←i} means j tends to move in the same direction as i did.
    Asymmetry: |CR_{j←i}| - |CR_{i←j}| > 0 means i drives j more than vice versa.
    """
    nodes = list(wide.columns)
    X = wide.values.astype(float)
    dX = np.diff(X, axis=0)
    T = len(dX)

    rows = []
    for ni in nodes:
        i = nodes.index(ni)
        sigma_i = float(np.std(dX[:, i], ddof=1)) if T > 1 else 0.0
        if sigma_i < 1e-9:
            # node never moved — no events
            for nj in nodes:
                if ni == nj:
                    continue
                rows.append({
                    'Society': society, 'Driver': ni, 'Responder': nj,
                    'CR': np.nan, 'CR_pos_frac': np.nan, 'n_events': 0,
                    'threshold_sigma': threshold_sigma, 'lag': lag,
                })
            continue

        # Event indices: years where |Δx_i| > threshold
        events = np.where(np.abs(dX[:, i]) > threshold_sigma * sigma_i)[0]
        # Keep only events where lag-step response is available
        events = events[events + lag < T]

        for nj in nodes:
            if ni == nj:
                continue
            j = nodes.index(nj)
            if len(events) == 0:
                cr = np.nan
                pos_frac = np.nan
            else:
                responses = dX[events + lag, j]
                cr = float(np.mean(responses))
                pos_frac = float(np.mean(responses > 0))

            rows.append({
                'Society':         society,
                'Driver':          ni,
                'Responder':       nj,
                'CR':              round(cr, 4) if not np.isnan(cr) else np.nan,
                'CR_pos_frac':     round(pos_frac, 3) if not np.isnan(pos_frac) else np.nan,
                'n_events':        len(events),
                'threshold_sigma': threshold_sigma,
                'lag':             lag,
            })

    return pd.DataFrame(rows)


def asymmetry_table(cr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-society, per-pair asymmetry table from the conditional-response output.

    Asymmetry = |CR_{j←i}| - |CR_{i←j}|
    Positive = i drives j more strongly than j drives i (i is the dominant node).
    """
    rows = []
    for society, grp in cr_df.groupby('Society'):
        cr_lookup = {(row['Driver'], row['Responder']): row['CR']
                     for _, row in grp.iterrows()}
        seen = set()
        for (ni, nj), cr_ji in cr_lookup.items():
            if ni == nj:
                continue
            pair = tuple(sorted([ni, nj]))
            if pair in seen:
                continue
            seen.add(pair)
            cr_ij = cr_lookup.get((nj, ni), np.nan)

            abs_ji = abs(cr_ji) if not np.isnan(cr_ji) else np.nan
            abs_ij = abs(cr_ij) if not np.isnan(cr_ij) else np.nan

            if not np.isnan(abs_ji) and not np.isnan(abs_ij):
                asym = abs_ji - abs_ij
                dominant = ni if asym > 0 else nj
            else:
                asym = np.nan
                dominant = 'Unknown'

            rows.append({
                'Society':   society,
                'Node_i':    ni,
                'Node_j':    nj,
                'CR_j_from_i': round(cr_ji, 4) if not np.isnan(cr_ji) else np.nan,
                'CR_i_from_j': round(cr_ij, 4) if not np.isnan(cr_ij) else np.nan,
                'Asymmetry': round(asym, 4) if not np.isnan(asym) else np.nan,
                'Dominant':  dominant,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['Society', 'Asymmetry'], ascending=[True, False])
    return df


def annotate_coupling_with_asymmetry(inel_df: pd.DataFrame,
                                     asym_df: pd.DataFrame) -> pd.DataFrame:
    """
    Upgrade 'Loosely associated' to 'Directionally constrained' where |Asymmetry| > 1.0.
    """
    if asym_df.empty:
        return inel_df

    asym_lookup = {}
    for _, row in asym_df.iterrows():
        key = (row['Society'], row['Node_i'], row['Node_j'])
        asym_lookup[key] = row.get('Asymmetry', np.nan)
        key2 = (row['Society'], row['Node_j'], row['Node_i'])
        asym_lookup[key2] = row.get('Asymmetry', np.nan)

    def _update_class(row):
        if row['Coupling'] != 'Loosely associated':
            return row['Coupling']
        key = (row['Society'], row['Node_i'], row['Node_j'])
        asym = asym_lookup.get(key, np.nan)
        if not np.isnan(asym) and abs(asym) > 1.0:
            return 'Directionally constrained'
        return 'Loosely associated'

    inel_df = inel_df.copy()
    inel_df['Coupling'] = inel_df.apply(_update_class, axis=1)
    return inel_df


# ── Summary reporting ─────────────────────────────────────────────────────────

def print_summary(inel_df: pd.DataFrame, asym_df: pd.DataFrame) -> None:
    """Print a human-readable summary to stdout."""
    societies = inel_df['Society'].unique()

    for soc in societies:
        print(f"\n{'='*60}")
        print(f"  {soc}")
        print(f"{'='*60}")

        sub = inel_df[inel_df['Society'] == soc].copy()

        # Coupling class counts
        counts = sub['Coupling'].value_counts()
        print(f"\n  Coupling classes ({len(sub)} pairs):")
        for cls, n in counts.items():
            print(f"    {cls:<28} {n:>3} pairs")

        # Loop type breakdown
        print(f"\n  Mean I_ij by loop type:")
        for loop, grp in sub.groupby('Loop'):
            mean_i = grp['I_ij'].mean()
            print(f"    {loop:<20} mean I = {mean_i:.3f}  (n={len(grp)})")

        # Top 5 most rigid pairs
        rigid = sub.nlargest(5, 'I_ij')[['Node_i','Node_j','Loop','I_ij','Coupling']]
        print(f"\n  Most rigid pairs:")
        for _, r in rigid.iterrows():
            print(f"    {r['Node_i']:10}–{r['Node_j']:10}  I={r['I_ij']:+.3f}  [{r['Loop']}]")

        # Top 5 most elastic pairs
        elastic = sub.nsmallest(5, 'I_ij')[['Node_i','Node_j','Loop','I_ij','Coupling']]
        print(f"\n  Most elastic pairs:")
        for _, r in elastic.iterrows():
            print(f"    {r['Node_i']:10}–{r['Node_j']:10}  I={r['I_ij']:+.3f}  [{r['Loop']}]")

        # Top asymmetric pairs
        if not asym_df.empty:
            asoc = asym_df[asym_df['Society'] == soc].head(5)
            if not asoc.empty:
                print(f"\n  Most directionally asymmetric pairs:")
                for _, r in asoc.iterrows():
                    dom = r['Dominant']
                    print(f"    {r['Node_i']:10}–{r['Node_j']:10}  asym={r['Asymmetry']:+.3f}  dominant={dom}")


# ── Cross-society comparison ──────────────────────────────────────────────────

def cross_society_comparison(inel_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each node pair, show mean and std of I_ij across all societies.
    High std = pair is society-specific.  Low std = pair is universal.
    """
    grp = inel_df.groupby(['Node_i', 'Node_j'])['I_ij'].agg(['mean','std','count'])
    grp.columns = ['I_mean', 'I_std', 'n_societies']
    grp = grp.reset_index()
    grp['Loop'] = grp.apply(
        lambda r: LOOP_TYPE[(r['Node_i'] in SLOW, r['Node_j'] in SLOW)], axis=1
    )
    grp = grp.sort_values('I_mean', ascending=False)
    return grp


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Compute CAMS temporal inelasticity from Block 1 CSV(s).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('files', nargs='*', help='Block 1 CSV file(s)')
    p.add_argument('--glob', help='Glob pattern for Block 1 files (e.g. "analysis/*_block1.csv")')
    p.add_argument('--out', default='analysis/inelasticity', help='Output prefix (default: analysis/inelasticity)')
    p.add_argument('--threshold', type=float, default=1.0, help='σ threshold for event detection (default: 1.0)')
    p.add_argument('--lag', type=int, default=1, help='Response lag in years (default: 1)')
    p.add_argument('--min-years', type=int, default=10, help='Minimum years required per society (default: 10)')
    p.add_argument('--no-summary', action='store_true', help='Suppress printed summary')
    return p.parse_args()


def main():
    args = parse_args()

    # Collect input files
    input_files = list(args.files)
    if args.glob:
        input_files += glob_module.glob(args.glob)
    if not input_files:
        print("No input files specified. Use: python cams_inelasticity.py block1.csv [...]", file=sys.stderr)
        sys.exit(1)

    # Load and concatenate
    frames = []
    for f in input_files:
        try:
            frames.append(load_block1(f))
            print(f"Loaded: {f}")
        except Exception as e:
            print(f"Warning: skipping {f} — {e}", file=sys.stderr)

    if not frames:
        print("No valid files loaded.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    societies = df['Society'].unique()
    print(f"\n{len(societies)} societies: {', '.join(sorted(societies))}")

    # ── Per-society computation ───────────────────────────────────────────────
    all_inel = []
    all_cr   = []

    for soc in sorted(societies):
        wide = pivot_to_wide(df, soc)
        if len(wide) < args.min_years:
            print(f"  {soc}: only {len(wide)} years — skipping (--min-years {args.min_years})")
            continue
        print(f"  {soc}: {len(wide)} years ({wide.index.min()}–{wide.index.max()}) × {len(wide.columns)} nodes")

        inel = inelasticity_matrix(wide, soc)
        cr   = conditional_response(wide, soc, threshold_sigma=args.threshold, lag=args.lag)
        all_inel.append(inel)
        all_cr.append(cr)

    if not all_inel:
        print("No societies with sufficient data.", file=sys.stderr)
        sys.exit(1)

    inel_df = pd.concat(all_inel, ignore_index=True)
    cr_df   = pd.concat(all_cr,   ignore_index=True)

    # ── Asymmetry ─────────────────────────────────────────────────────────────
    asym_df = asymmetry_table(cr_df)
    inel_df = annotate_coupling_with_asymmetry(inel_df, asym_df)

    # ── Cross-society comparison ──────────────────────────────────────────────
    cross_df = cross_society_comparison(inel_df)

    # ── Output ────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    inel_path = f"{args.out}_matrix.csv"
    cr_path   = f"{args.out}_cond_response.csv"
    asym_path = f"{args.out}_asymmetry.csv"
    cross_path= f"{args.out}_cross_society.csv"

    inel_df.to_csv(inel_path, index=False)
    cr_df.to_csv(cr_path,     index=False)
    asym_df.to_csv(asym_path, index=False)
    cross_df.to_csv(cross_path, index=False)

    print(f"\nOutputs written:")
    print(f"  {inel_path}   — inelasticity matrix (I_ij per pair per society)")
    print(f"  {cr_path}     — conditional response CR_{{j←i}}")
    print(f"  {asym_path}   — directional asymmetry ranking")
    print(f"  {cross_path}  — cross-society comparison")

    # ── Summary ───────────────────────────────────────────────────────────────
    if not args.no_summary:
        print_summary(inel_df, asym_df)

        print(f"\n{'='*60}")
        print("  Cross-society: most universally rigid pairs")
        print(f"{'='*60}")
        top = cross_df[cross_df['n_societies'] > 1].head(10)
        for _, r in top.iterrows():
            print(f"  {r['Node_i']:10}–{r['Node_j']:10}  mean I={r['I_mean']:.3f}  std={r['I_std']:.3f}  [{r['Loop']}]")

        print(f"\n  Cross-society: most variable pairs (society-specific coupling)")
        bot = cross_df[cross_df['n_societies'] > 1].sort_values('I_std', ascending=False).head(10)
        for _, r in bot.iterrows():
            print(f"  {r['Node_i']:10}–{r['Node_j']:10}  mean I={r['I_mean']:.3f}  std={r['I_std']:.3f}  [{r['Loop']}]")


if __name__ == '__main__':
    main()
