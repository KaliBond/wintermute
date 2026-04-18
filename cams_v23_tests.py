"""
cams_v23_tests.py
=================
Prototype validation + extension module for CAMS v2.3.

Implements five tests using the public-repo formulae:
  Vi   = Ci + Ki + Ai/2 - Si          (node value, v2.3 canonical)
  Bij  = sqrt(max(Vi+8,0)*max(Vj+8,0)) / 32   (bond strength)
  Lambda(t) = mean over distinct node pairs of Bij(t)  (coherence / coordination)

Tests:
  1. bootstrap_corr         — bootstrap CI + p-value for entropy-vs-health or any pair
  2. monte_carlo_noise      — LLM-typical noise perturbation on raw scores
  3. walk_forward_predict   — fit trend on early window, forecast Lambda, score vs actual
  4. what_if_shock          — policy-shock simulator on node C/K/S/A
  5. seshat_benchmark       — Seshat/Turchin structural merge with bidirectional
                              Granger causality (per-lag F + p table, configurable
                              max_lag) against the Equinox2020 snapshot

Optional spectral extension:
  If cams_spectral.py is in the same directory, run_full_test_battery will
  also report the Fiedler eigenvalue (algebraic connectivity) and flag
  structural fragility years automatically.

Written for tight, auditable code — no external deps beyond numpy/pandas/scipy/
statsmodels (all standard scientific Python).

Kari McKern / CAMS v2.3 framework. Caveat OD-001 applies to bond formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

# Optional spectral extension — graceful fallback if not present
try:
    from cams_spectral import run_spectral_test, spectral_granger_frame
    _SPECTRAL_AVAILABLE = True
except ImportError:
    _SPECTRAL_AVAILABLE = False


# ----------------------------------------------------------------------
# Core CAMS v2.3 arithmetic
# ----------------------------------------------------------------------

def node_value(C, K, S, A):
    """Vi = C + K + A/2 - S   (CAMS v2.3 canonical form)"""
    return C + K + 0.5 * A - S


def bond_strength(Vi, Vj):
    """Bij = sqrt(max(Vi+8,0)*max(Vj+8,0)) / 32"""
    return np.sqrt(np.maximum(Vi + 8, 0) * np.maximum(Vj + 8, 0)) / 32.0


def lambda_coherence(node_values: np.ndarray) -> float:
    """
    Lambda(t) = mean bond strength across all distinct node pairs.
    node_values: 1-D array of 8 V_i scores for a single year.
    """
    pairs = [bond_strength(node_values[i], node_values[j])
             for i, j in combinations(range(len(node_values)), 2)]
    return float(np.mean(pairs))


def system_health(node_values: np.ndarray) -> float:
    """
    Simple scalar health proxy: mean V with a modest variance penalty.
    H = mean(V) - 0.5 * std(V)   (penalises node imbalance)
    """
    return float(np.mean(node_values) - 0.5 * np.std(node_values))


def panel_to_year_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape long CAMS panel → year-indexed DataFrame with one column per node (V)."""
    req = {'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction'}
    assert req.issubset(df.columns), f"missing cols, need {req}"
    df = df.copy()
    df['V'] = node_value(df['Coherence'], df['Capacity'], df['Stress'], df['Abstraction'])
    return df.pivot_table(index='Year', columns='Node', values='V', aggfunc='mean').sort_index()


def series_lambda_and_health(year_matrix: pd.DataFrame) -> pd.DataFrame:
    """For each year, compute Lambda(t) and H(t)."""
    out = pd.DataFrame(index=year_matrix.index, columns=['Lambda', 'H'], dtype=float)
    for yr, row in year_matrix.iterrows():
        vals = row.values.astype(float)
        out.at[yr, 'Lambda'] = lambda_coherence(vals)
        out.at[yr, 'H']      = system_health(vals)
    return out


# ----------------------------------------------------------------------
# TEST 1 — Bootstrap correlation (CI + p-value)
# ----------------------------------------------------------------------

@dataclass
class BootResult:
    r: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    n: int
    n_boot: int


def bootstrap_corr(x: np.ndarray, y: np.ndarray,
                   n_boot: int = 5000, seed: int = 42,
                   alpha: float = 0.05) -> BootResult:
    """
    Paired bootstrap 95% CI for Pearson r, plus permutation p-value.
    Replaces the single point-estimate r = -0.958 style claim with
    an honest uncertainty band.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    r_obs = float(np.corrcoef(x, y)[0, 1])

    # Bootstrap CI (resample paired observations with replacement)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            rs[b] = np.nan
        else:
            rs[b] = np.corrcoef(xb, yb)[0, 1]
    rs = rs[np.isfinite(rs)]
    lo, hi = np.quantile(rs, [alpha/2, 1-alpha/2])

    # Permutation p-value (shuffle y, recompute r)
    perm = np.empty(n_boot)
    for b in range(n_boot):
        yp = rng.permutation(y)
        perm[b] = np.corrcoef(x, yp)[0, 1]
    p = float((np.abs(perm) >= abs(r_obs)).mean())

    return BootResult(r=r_obs, ci_low=float(lo), ci_high=float(hi),
                      p_two_sided=p, n=n, n_boot=n_boot)


# ----------------------------------------------------------------------
# TEST 2 — Monte Carlo noise robustness on Lambda(t)
# ----------------------------------------------------------------------

def monte_carlo_noise(df: pd.DataFrame,
                      sigma: float = 0.5,
                      n_runs: int = 500,
                      seed: int = 7) -> pd.DataFrame:
    """
    Perturb C, K, S, A per node-year with Gaussian noise (sigma in raw-score units,
    0.5 ~= typical LLM inter-model SD) and re-compute Lambda(t) each run.
    Returns per-year {mean, std, p5, p95} of Lambda — directly shows where the
    Lambda = 0.45 coordination-failure threshold is or isn't robust.
    """
    rng = np.random.default_rng(seed)
    years = sorted(df['Year'].unique())
    lam = np.full((n_runs, len(years)), np.nan)
    base = df.copy()
    cols = ['Coherence', 'Capacity', 'Stress', 'Abstraction']

    for r in range(n_runs):
        noisy = base.copy()
        noise = rng.normal(0, sigma, size=(len(noisy), 4))
        noisy[cols] = np.clip(noisy[cols].values + noise, 0, 10)  # keep 0-10 bounds
        noisy['V'] = node_value(noisy['Coherence'], noisy['Capacity'],
                                noisy['Stress'], noisy['Abstraction'])
        pivot = noisy.pivot_table(index='Year', columns='Node',
                                  values='V', aggfunc='mean').sort_index()
        for j, yr in enumerate(years):
            lam[r, j] = lambda_coherence(pivot.loc[yr].values.astype(float))

    return pd.DataFrame({
        'Year':       years,
        'Lambda_mean': np.nanmean(lam, axis=0),
        'Lambda_std':  np.nanstd(lam,  axis=0),
        'Lambda_p5':   np.nanpercentile(lam,  5, axis=0),
        'Lambda_p95':  np.nanpercentile(lam, 95, axis=0),
    })


# ----------------------------------------------------------------------
# TEST 3 — Walk-forward predictive validation
# ----------------------------------------------------------------------

def walk_forward_predict(lam_series: pd.Series,
                         train_frac: float = 0.6,
                         threshold: float = 0.45) -> dict:
    """
    Very simple OOS test: fit a linear trend + AR(1) residual on the early
    `train_frac` of the Lambda(t) series, forecast the rest, and ask:
    does the model predict sub-threshold crossings where they actually occur?
    Returns RMSE, directional-hit rate, and threshold-crossing confusion.
    """
    s = lam_series.dropna().astype(float)
    n = len(s)
    split = int(n * train_frac)
    train, test = s.iloc[:split], s.iloc[split:]

    # Linear trend on train
    t_train = np.arange(len(train))
    slope, intercept = np.polyfit(t_train, train.values, 1)

    # AR(1) on detrended residuals
    resid = train.values - (intercept + slope * t_train)
    if len(resid) > 2 and np.std(resid[:-1]) > 0:
        phi = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    else:
        phi = 0.0

    # Forecast
    t_test = np.arange(len(train), n)
    trend_fc = intercept + slope * t_test
    r_last = resid[-1] if len(resid) else 0.0
    ar_fc = np.array([phi**(k+1) * r_last for k in range(len(test))])
    fc = trend_fc + ar_fc

    rmse = float(np.sqrt(np.mean((fc - test.values) ** 2)))

    # Directional hit rate
    if len(test) > 1:
        dir_actual = np.sign(np.diff(test.values))
        dir_pred   = np.sign(np.diff(fc))
        hits = (dir_actual == dir_pred) & (dir_actual != 0)
        hit_rate = float(hits.sum() / max((dir_actual != 0).sum(), 1))
    else:
        hit_rate = np.nan

    # Threshold-crossing confusion
    a_below = (test.values < threshold).astype(int)
    p_below = (fc < threshold).astype(int)
    tp = int(((a_below == 1) & (p_below == 1)).sum())
    tn = int(((a_below == 0) & (p_below == 0)).sum())
    fp = int(((a_below == 0) & (p_below == 1)).sum())
    fn = int(((a_below == 1) & (p_below == 0)).sum())

    return {
        'train_years': (int(train.index.min()), int(train.index.max())),
        'test_years':  (int(test.index.min()),  int(test.index.max())),
        'slope': float(slope), 'phi_ar1': phi,
        'rmse': rmse, 'dir_hit_rate': hit_rate,
        'threshold': threshold,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'forecast': pd.Series(fc, index=test.index, name='Lambda_pred'),
        'actual':   test,
    }


# ----------------------------------------------------------------------
# TEST 4 — What-if shock simulator
# ----------------------------------------------------------------------

def what_if_shock(df: pd.DataFrame,
                  from_year: int,
                  shocks: dict,
                  horizon: int = 20,
                  diffusion: float = 0.15,
                  noise_sigma: float = 0.0,
                  seed: int = 0) -> pd.DataFrame:
    """
    Apply a one-off policy shock (delta on C/K/S/A of chosen nodes) at `from_year`
    and simulate `horizon` years forward with simple bond-weighted diffusion:
        V_i(t+1) = V_i(t) + diffusion * sum_j B_ij * (V_j - V_i) + noise
    shocks example: {'Lore': {'C': +2}, 'Hands': {'S': -1}}
    Returns a DataFrame of V-per-node plus Lambda and H per simulated year.
    """
    rng = np.random.default_rng(seed)
    ym  = panel_to_year_matrix(df)
    if from_year not in ym.index:
        from_year = int(ym.index[ym.index <= from_year].max())
    base = ym.loc[from_year].copy()

    # Apply shocks to raw C/K/S/A at from_year and recompute V
    row = df[df['Year'] == from_year].set_index('Node')[
        ['Coherence', 'Capacity', 'Stress', 'Abstraction']].copy()
    for node, deltas in shocks.items():
        if node not in row.index:
            continue
        for k, dv in deltas.items():
            col = {'C': 'Coherence', 'K': 'Capacity',
                   'S': 'Stress', 'A': 'Abstraction'}.get(k, k)
            row.at[node, col] = np.clip(row.at[node, col] + dv, 0, 10)
    shocked_V = pd.Series(
        node_value(row['Coherence'], row['Capacity'], row['Stress'], row['Abstraction']),
        index=row.index
    ).reindex(base.index).fillna(base)

    nodes = list(base.index)
    V = shocked_V.values.astype(float).copy()
    history = [V.copy()]

    for _ in range(horizon):
        B = np.zeros((len(V), len(V)))
        for i in range(len(V)):
            for j in range(len(V)):
                if i == j:
                    continue
                B[i, j] = bond_strength(V[i], V[j])
        delta = np.array([
            diffusion * np.sum(B[i] * (V - V[i])) / max(np.sum(B[i]), 1e-9)
            for i in range(len(V))
        ])
        noise = rng.normal(0, noise_sigma, size=len(V)) if noise_sigma > 0 else 0.0
        V = V + delta + noise
        history.append(V.copy())

    sim = pd.DataFrame(history, columns=nodes)
    sim.index = range(from_year, from_year + len(sim))
    sim.index.name = 'Year'
    sim['Lambda'] = [lambda_coherence(v) for v in sim[nodes].values]
    sim['H']      = [system_health(v)    for v in sim[nodes].values]
    return sim


# ----------------------------------------------------------------------
# TEST 5 — Seshat / Turchin structural merge
#           Bidirectional Granger causality with per-lag F + p reporting
# ----------------------------------------------------------------------

def load_seshat_equinox(path: str) -> pd.DataFrame:
    """
    Load Equinox2020 (or later) Seshat snapshot (CSV or XLSX).
    Auto-handles common column-name variants across Seshat releases.
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    rename_map = {
        'NGA': 'nga', 'Polity': 'polity', 'Year': 'year', 'year_from': 'year',
        'elite_competition': 'elite_overproduction',
        'elite_overproduction': 'elite_overproduction',
        'state_fiscal_stress': 'state_fiscal_stress',
        'state_capacity': 'state_capacity',
        'military_technology': 'military_technology',
        'military_size': 'military_size',
        'population_immiseration': 'population_immiseration',
        'wellbeing': 'wellbeing',
        'social_complexity_index': 'social_complexity_index',
        'internal_warfare': 'internal_warfare',
        'instability': 'instability',
        'political_centralization': 'political_centralization',
    }
    df = df.rename(columns=rename_map)

    core_cols = [
        'nga', 'polity', 'year',
        'elite_overproduction', 'state_fiscal_stress', 'state_capacity',
        'military_technology', 'population_immiseration',
        'social_complexity_index', 'internal_warfare', 'instability',
    ]
    available = [c for c in core_cols if c in df.columns]
    return df[available].copy()


# Keep old name as alias for backwards compatibility
load_seshat_equinoX = load_seshat_equinox


def map_seshat_to_cams(seshat_df: pd.DataFrame,
                       cams_df: pd.DataFrame,
                       society_name: str) -> pd.DataFrame:
    """
    Align a Seshat subset with a CAMS panel by year.
    Rome / Latin Europe gets the best NGA match automatically.
    Returns an inner-joined DataFrame with both Lambda/H and Seshat variables.
    """
    if any(x in society_name.lower() for x in ['rome', 'roman', 'latin']):
        seshat_subset = seshat_df[
            seshat_df['nga'].str.contains('Latin Europe|Rome', na=False, case=False)
        ]
    else:
        seshat_subset = seshat_df[
            seshat_df['polity'].str.contains(society_name, na=False, case=False)
        ]

    if len(seshat_subset) == 0:
        return pd.DataFrame()

    seshat_subset = (seshat_subset
                     .sort_values('year')
                     .groupby('year').mean(numeric_only=True)
                     .reset_index())

    ym   = panel_to_year_matrix(cams_df)
    lamH = series_lambda_and_health(ym).reset_index()

    merged = pd.merge(lamH, seshat_subset,
                      left_on='Year', right_on='year', how='inner')
    return merged.drop(columns=['year'], errors='ignore')


def run_granger_expanded(merged: pd.DataFrame,
                         var: str,
                         max_lag: int = 5) -> dict:
    """
    Bidirectional Granger causality between Lambda and a Seshat variable,
    with per-lag F-statistic and p-value reported for lags 1..max_lag.

    Tests both:
      Seshat→Λ  (does the cliodynamic stressor predict future Lambda?)
      Λ→Seshat  (does CAMS's cross-layer decoupling lead the stressor?)

    Returns nested dict: {direction: {lagN: "F=x.xxx, p=x.xxxx"}}
    """
    if len(merged) < max_lag + 10:
        return {'error': f'Insufficient observations ({len(merged)}) for max_lag={max_lag}'}

    results = {}
    directions = [
        ('Seshat→Λ', var,      'Lambda'),
        ('Λ→Seshat', 'Lambda', var),
    ]

    for label, cause, effect in directions:
        results[label] = {}
        try:
            gc = grangercausalitytests(
                merged[[effect, cause]], maxlag=max_lag, verbose=False
            )
            for lag in range(1, max_lag + 1):
                f_stat = gc[lag][0]['ssr_ftest'][0]
                p_val  = gc[lag][0]['ssr_ftest'][1]
                results[label][f'lag{lag}'] = f'F={f_stat:.3f}, p={p_val:.4f}'
        except Exception as e:
            results[label] = {'error': str(e)}

    return results


def run_seshat_benchmark(cams_df: pd.DataFrame,
                         seshat_df: pd.DataFrame,
                         society_name: str,
                         max_lag: int = 5) -> Tuple[dict, pd.DataFrame]:
    """
    Full Test 5: Pearson correlation + bidirectional Granger (per-lag) for
    each available Seshat variable against CAMS Lambda.

    Saves merged CSV for downstream dashboard use.
    Returns (results_dict, merged_DataFrame).
    """
    merged = map_seshat_to_cams(seshat_df, cams_df, society_name)
    if len(merged) < 10:
        return (f'Insufficient overlap for {society_name} ({len(merged)} years)',
                pd.DataFrame())

    results = {}
    seshat_vars = [
        'elite_overproduction', 'state_fiscal_stress', 'state_capacity',
        'internal_warfare', 'social_complexity_index',
    ]

    for var in seshat_vars:
        if var not in merged.columns:
            continue

        # Pearson correlation
        r, p = pearsonr(merged['Lambda'], merged[var])
        results[f'r(Λ, {var})'] = f'{r:.3f} (p={p:.4f})'

        # Expanded bidirectional Granger
        gr = run_granger_expanded(merged, var, max_lag=max_lag)
        results[f'Granger {var}'] = gr

    merged.to_csv(f'merged_{society_name}_cams_seshat.csv', index=False)
    return results, merged


# ----------------------------------------------------------------------
# Convenience: run the whole battery on one loaded dataframe
# ----------------------------------------------------------------------

def run_battery(df: pd.DataFrame, name: str = 'society') -> dict:
    """Run Tests 1–4 and print a summary report."""
    print(f"\n{'='*72}\n  CAMS v2.3 test battery — {name}\n{'='*72}")

    ym   = panel_to_year_matrix(df)
    lamH = series_lambda_and_health(ym)
    print(f'Years: {ym.index.min()}–{ym.index.max()}  ({len(ym)} obs)')
    print(f'Lambda mean={lamH["Lambda"].mean():.3f}  '
          f'min={lamH["Lambda"].min():.3f}  max={lamH["Lambda"].max():.3f}')
    print(f'H      mean={lamH["H"].mean():.3f}  '
          f'min={lamH["H"].min():.3f}  max={lamH["H"].max():.3f}')

    print('\n[1] Bootstrap r(Lambda, H)')
    b = bootstrap_corr(lamH['Lambda'].values, lamH['H'].values, n_boot=3000)
    print(f'    r = {b.r:+.3f}  95% CI [{b.ci_low:+.3f}, {b.ci_high:+.3f}]  '
          f'p = {b.p_two_sided:.4f}  (n={b.n})')

    print('\n[2] Monte Carlo noise (sigma=0.5, 150 runs)')
    mc = monte_carlo_noise(df, sigma=0.5, n_runs=150)
    below = (mc['Lambda_p5'] < 0.45).sum()
    print(f'    Years where even p5 stays above 0.45 threshold: '
          f'{len(mc) - below}/{len(mc)}')
    print(f'    Mean Lambda noise SD across years: {mc["Lambda_std"].mean():.4f}')

    print('\n[3] Walk-forward predict')
    wf = walk_forward_predict(lamH['Lambda'])
    print(f'    Train {wf["train_years"]}  → Test {wf["test_years"]}')
    print(f'    RMSE={wf["rmse"]:.3f}  dir_hit={wf["dir_hit_rate"]:.2%}  '
          f'confusion TP/TN/FP/FN = {wf["tp"]}/{wf["tn"]}/{wf["fp"]}/{wf["fn"]}')

    print('\n[4] What-if shock  (+2 Coherence to Lore, from latest year, 20yr horizon)')
    latest = int(ym.index.max())
    sim = what_if_shock(df, from_year=latest - 1,
                        shocks={'Lore': {'C': +2}}, horizon=20)
    d_lam = sim['Lambda'].iloc[-1] - sim['Lambda'].iloc[0]
    d_H   = sim['H'].iloc[-1]      - sim['H'].iloc[0]
    print(f'    After 20yr simulated diffusion: ΔLambda={d_lam:+.3f}  ΔH={d_H:+.3f}')

    return {'lamH': lamH, 'boot': b, 'mc': mc, 'wf': wf, 'sim': sim}


def run_full_test_battery(df: pd.DataFrame,
                          name: str = 'society',
                          seshat_path: Optional[str] = None,
                          max_lag: int = 5,
                          run_spectral: bool = True) -> dict:
    """
    Run the complete CAMS v2.3 test battery (Tests 1–5) plus optional
    spectral coherence analysis.

    Parameters
    ----------
    df            : long-format CAMS panel (Year / Node / Coherence / Capacity /
                    Stress / Abstraction columns required).
    name          : society label for output files and headings.
    seshat_path   : path to Seshat Equinox CSV or XLSX (activates Test 5).
    max_lag       : maximum Granger lag (Test 5 + spectral Granger).
    run_spectral  : if True and cams_spectral.py is available, run the Fiedler
                    eigenvalue structural fragility diagnostic.

    Returns
    -------
    dict with keys: lamH, boot, mc, wf, sim, and optionally seshat, spectral.
    """
    results = run_battery(df, name)

    # TEST 5 — Seshat / Turchin merge
    if seshat_path:
        print(f"\n{'='*72}\n  TEST 5 — Seshat/Turchin merge "
              f"(max_lag={max_lag}) for {name}\n{'='*72}")
        seshat = load_seshat_equinox(seshat_path)
        bench, merged = run_seshat_benchmark(df, seshat, name, max_lag=max_lag)
        for k, v in bench.items():
            if isinstance(v, dict):
                print(f'\n  {k}:')
                for direction, lags in v.items():
                    if isinstance(lags, dict):
                        print(f'    {direction}:')
                        for lag_key, stat in lags.items():
                            print(f'      {lag_key}: {stat}')
                    else:
                        print(f'    {direction}: {lags}')
            else:
                print(f'  {k}: {v}')
        results['seshat'] = bench

    # SPECTRAL — Fiedler eigenvalue structural fragility diagnostic
    if run_spectral and _SPECTRAL_AVAILABLE:
        print(f"\n{'='*72}\n  SPECTRAL — Algebraic connectivity (Fiedler) for {name}\n{'='*72}")
        spec_diag = run_spectral_test(df, name=name, save_csv=True)
        results['spectral'] = spec_diag
    elif run_spectral and not _SPECTRAL_AVAILABLE:
        print('\n  [spectral] cams_spectral.py not found — skipping Fiedler analysis.')

    return results


if __name__ == '__main__':
    pass
