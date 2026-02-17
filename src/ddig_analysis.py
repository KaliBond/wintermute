import numpy as np
import pandas as pd

EPS_MI = 1e-12

def _qbin(s: pd.Series, q=3):
    """Quantile bin into q discrete labels 0..q-1. Falls back safely."""
    s = s.astype(float)
    if s.nunique(dropna=True) < 2:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index)
    try:
        b = pd.qcut(s.rank(method="first"), q=q, labels=False, duplicates="drop")
        return b.astype("Int64").fillna(0).astype(int)
    except Exception:
        # fallback: median split
        med = np.nanmedian(s.values)
        return (s > med).fillna(False).astype(int)

def _combine_context(*cols: pd.Series) -> pd.Series:
    """Combine discrete context columns into a single discrete label."""
    df = pd.concat(cols, axis=1)
    return df.astype(str).agg("|".join, axis=1)

def conditional_mutual_information(x, y, z, base=2.0, pseudocount=0.0, min_samples=25):
    """
    Discrete CMI: I(X;Y|Z). x,y,z must be integer-coded series aligned on index.
    Returns CMI in bits (base=2) by default.
    """
    data = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(data) < min_samples:
        return np.nan

    # counts
    n = len(data)
    c_xyz = data.groupby(["x","y","z"]).size().astype(float)
    c_xz  = data.groupby(["x","z"]).size().astype(float)
    c_yz  = data.groupby(["y","z"]).size().astype(float)
    c_z   = data.groupby(["z"]).size().astype(float)

    # optional Laplace-like smoothing (tiny)
    if pseudocount > 0:
        c_xyz = c_xyz + pseudocount
        c_xz  = c_xz  + pseudocount
        c_yz  = c_yz  + pseudocount
        c_z   = c_z   + pseudocount

    # normalise to probabilities
    p_xyz = c_xyz / c_xyz.sum()
    p_xz  = c_xz  / c_xz.sum()
    p_yz  = c_yz  / c_yz.sum()
    p_z   = c_z   / c_z.sum()

    # compute sum p(x,y,z) * log( p(x,y|z) / (p(x|z)p(y|z)) )
    cmi = 0.0
    for (xv, yv, zv), p in p_xyz.items():
        pxz = p_xz.get((xv, zv), 0.0)
        pyz = p_yz.get((yv, zv), 0.0)
        pz  = p_z.get(zv, 0.0)
        if p <= 0 or pxz <= 0 or pyz <= 0 or pz <= 0:
            continue
        ratio = (p * pz) / (pxz * pyz + EPS_MI)
        cmi += p * np.log(ratio + EPS_MI)

    if base == 2.0:
        cmi /= np.log(2.0)
    return float(cmi)

def conditional_entropy_y_given_z(y, z, base=2.0):
    """H(Y|Z) for discrete y,z."""
    data = pd.DataFrame({"y": y, "z": z}).dropna()
    if len(data) == 0:
        return np.nan
    c_yz = data.groupby(["y","z"]).size().astype(float)
    c_z  = data.groupby(["z"]).size().astype(float)
    p_yz = c_yz / c_yz.sum()
    p_z  = c_z  / c_z.sum()

    h = 0.0
    for (yv, zv), p in p_yz.items():
        pz = p_z.get(zv, 0.0)
        if p <= 0 or pz <= 0:
            continue
        py_given_z = p / pz
        h -= p * np.log(py_given_z + EPS_MI)

    if base == 2.0:
        h /= np.log(2.0)
    return float(h)

def compute_dDIG_for_metric(df_wide: pd.DataFrame, metric: str, nodes: list,
                            regime: pd.Series,
                            qx=3, qy=3, qshock=3, era="decade"):
    """
    Returns a table with dDIG_i and nDIG_i for a given metric (Coherence/Capacity/Stress/Abstraction).
    """
    # deltas per node
    deltas = {}
    for n in nodes:
        col = f"{metric}_{n}"
        if col in df_wide.columns:
            deltas[n] = df_wide[col].diff()
    if len(deltas) < 3:
        return pd.DataFrame()

    D = pd.DataFrame(deltas)

    # shock proxy S(t) = median_j |Δx_j(t)|
    shock = D.abs().median(axis=1)
    shock_bin = _qbin(shock, q=qshock)

    # era bin
    if era == "decade":
        era_bin = (df_wide.index // 10) * 10
    else:
        era_bin = pd.Series("all", index=df_wide.index)

    # context Z(t)
    Z = _combine_context(regime.fillna("Unknown"), shock_bin, era_bin)

    out = []
    for i in D.columns:
        # Y_-i(t+1)
        Y_next = D.drop(columns=[i]).abs().mean(axis=1).shift(-1)

        # align X(t) with Y(t+1)
        X = D[i]
        Xb = _qbin(X, q=qx)
        Yb = _qbin(Y_next, q=qy)
        Zc = Z.shift(0)  # explicit

        cmi = conditional_mutual_information(Xb, Yb, Zc, base=2.0, pseudocount=0.0, min_samples=25)
        hyz = conditional_entropy_y_given_z(Yb, Zc, base=2.0)
        ndig = cmi / (hyz + 1e-9) if (cmi is not None and not np.isnan(cmi) and hyz is not None and not np.isnan(hyz)) else np.nan

        out.append({"Node": i, "dDIG_bits": cmi, "H(Y|Z)_bits": hyz, "nDIG": ndig})

    return pd.DataFrame(out).set_index("Node").sort_values("nDIG", ascending=False)

def compute_CTT_for_metric(df_wide: pd.DataFrame, metric: str, nodes: list, q_event=0.8, q_trans=0.8):
    """
    Optional trigger overlay: CTT_i for a given metric.
    Uses percentile thresholds; returns log-risk lift with Laplace smoothing.
    """
    deltas = {}
    for n in nodes:
        col = f"{metric}_{n}"
        if col in df_wide.columns:
            deltas[n] = df_wide[col].diff()
    if len(deltas) < 3:
        return pd.DataFrame()

    D = pd.DataFrame(deltas)
    xbar = df_wide[[f"{metric}_{n}" for n in D.columns]].mean(axis=1)
    T = (xbar.diff().abs().shift(-1) > xbar.diff().abs().quantile(q_trans)).astype(int)

    out = []
    for i in D.columns:
        th = D[i].abs().quantile(q_event)
        E = (D[i].abs() > th).astype(int)

        # align E(t) with T(t+1) already shifted
        data = pd.DataFrame({"E": E, "T": T}).dropna()
        if len(data) < 25:
            out.append({"Node": i, "CTT": np.nan})
            continue

        # Laplace smoothing
        a = ((data.E == 1) & (data.T == 1)).sum()
        b = ((data.E == 1) & (data.T == 0)).sum()
        c = ((data.E == 0) & (data.T == 1)).sum()
        d = ((data.E == 0) & (data.T == 0)).sum()
        p1 = (a + 1) / (a + b + 2)
        p0 = (c + 1) / (c + d + 2)
        ctt = np.log((p1 + 1e-9) / (p0 + 1e-9))
        out.append({"Node": i, "CTT": float(ctt)})

    return pd.DataFrame(out).set_index("Node").sort_values("CTT", ascending=False)

def compute_influence_bundle(df_wide, regime, nodes, wc=0.5, wa=0.5):
    """
    Produces:
      - per-metric nDIG tables
      - cog/aff rollups
      - optional CTT overlays
    """
    metrics = ["Coherence", "Capacity", "Stress", "Abstraction"]
    dig_tables = {m: compute_dDIG_for_metric(df_wide, m, nodes, regime) for m in metrics}

    # rollups
    def get_nDIG(metric, node):
        t = dig_tables.get(metric, pd.DataFrame())
        return t.loc[node, "nDIG"] if (not t.empty and node in t.index) else np.nan

    rows = []
    for n in nodes:
        cog = np.nanmean([get_nDIG("Coherence", n), get_nDIG("Capacity", n), get_nDIG("Abstraction", n)])
        aff = get_nDIG("Stress", n)
        influence = wc * cog + wa * aff
        rows.append({"Node": n, "nDIG_cog": cog, "nDIG_aff": aff, "Influence": influence})

    rollup = pd.DataFrame(rows).set_index("Node").sort_values("Influence", ascending=False)
    return dig_tables, rollup
