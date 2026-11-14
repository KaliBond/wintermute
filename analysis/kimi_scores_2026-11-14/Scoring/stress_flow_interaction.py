"""
Stress x Flow interaction test on the CAMS/JUNO panel
======================================================
Hypothesis (from "The Meta-Animal and Its Nodes", 2026):
    If Flow acts as pressure relief, the association between high Stress and
    subsequent loss of Coherence should WEAKEN as Flow Capacity rises.

Model:
    dC = C(t+h) - C(t)  for horizon h in {1, 5} years
    dC ~ Stress(t) + FlowCap(t) + Stress(t)*FlowCap(t)  (+ society fixed effects)

Sign expectations:
    beta_Stress        < 0   (stress precedes coherence loss)
    beta_interaction   > 0   (flow capacity moderates the loss)

Data: JUNO_Unified_Dataset.csv (36 societies, 32,361 node-year rows).
Unit of analysis: society-year. Node scores are averaged to society-year level:
overall Stress = mean of Stress across the 8 nodes; Coherence likewise;
Flow Capacity = Capacity of the Flow node only (the relief organ).

Caveats (reported honestly in output):
  * Scores are partly carry-forward smoothed, so consecutive years are not
    independent. Standard errors are clustered by society; effective N is
    smaller than the row count suggests.
  * This is an observational panel association, not a causal estimate.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANEL = "JUNO_Unified_Dataset.csv"
OUT_TABLE = "stress_flow_interaction_results.csv"
OUT_PLOT = "stress_flow_interaction.png"

# ---------------------------------------------------------------- load & shape
df = pd.read_csv(PANEL)
df = df.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------- carry-forward audit
# (protocol flag: societies with >30% of years identical to the previous
# year are potential smoothing artefacts; reported and used for sensitivity)
print("=" * 70)
print("CARRY-FORWARD AUDIT per society (fraction of society-years identical")
print("to the previous year across all 8 nodes x C,K,S,A)")
cf_rates = {}
for s, g in df.groupby("Society"):
    w = g.pivot(index="Year", columns="Node",
                values=["Coherence", "Capacity", "Stress", "Abstraction"])
    r = float((w == w.shift()).all(axis=1).mean())
    cf_rates[s] = r
flagged = [s for s, r in cf_rates.items() if r > 0.30]
print(pd.Series(cf_rates).round(3).sort_values(ascending=False).to_string())
print(f"flagged (>30%): {flagged}")

df = df[["Society", "Year", "Node", "Coherence", "Capacity", "Stress"]]

# society-year aggregates
sy = (
    df.groupby(["Society", "Year"], as_index=False)
      .agg(Coh=("Coherence", "mean"),
           Stress=("Stress", "mean"),
           K_all=("Capacity", "mean"))
)
flow = df[df.Node == "Flow"][["Society", "Year", "Capacity"]].rename(
    columns={"Capacity": "K_flow"})
sy = sy.merge(flow, on=["Society", "Year"], how="left").dropna()
print(f"society-years: {len(sy)} across {sy.Society.nunique()} societies")

# sensitivity frame: drop society-years that are exact carry-forward repeats
# (mask computed on the full panel including Abstraction, same criterion as audit)
w = pd.read_csv(PANEL).pivot_table(index=["Society", "Year"], columns="Node",
                   values=["Coherence", "Capacity", "Stress", "Abstraction"])
repeat_mask = (w == w.shift()).all(axis=1)
repeat_mask.index.names = ["Society", "Year"]
keep = ~repeat_mask.reindex(
    pd.MultiIndex.from_frame(sy[["Society", "Year"]])).fillna(False).values
sy_clean = sy[keep].reset_index(drop=True)
print(f"after dropping carry-forward-repeat years: {len(sy_clean)} society-years "
      f"({len(sy) - len(sy_clean)} dropped)")

# ------------------------------------------------- build horizon change scores
def add_change(frame, h):
    """Merge C(t+h) onto t via Year+h; returns frame with dC column."""
    fut = frame[["Society", "Year", "Coh"]].copy()
    fut["Year"] = fut["Year"] - h
    out = frame.merge(fut.rename(columns={"Coh": f"Coh_fwd{h}"}),
                      on=["Society", "Year"], how="inner")
    out[f"dC{h}"] = out[f"Coh_fwd{h}"] - out["Coh"]
    return out

rows = []
def fit_models(frame, frame_tag, h):
    """Fit signed-change and volatility interaction models on one frame."""
    d = add_change(frame, h)
    d[f"absdC{h}"] = d[f"dC{h}"].abs()
    specs = [(f"{frame_tag}:pooled",
              smf.ols(f"dC{h} ~ Stress * K_flow", data=d)),
             (f"{frame_tag}:society_FE",
              smf.ols(f"dC{h} ~ Stress * K_flow + C(Society)", data=d)),
             (f"{frame_tag}:volatility_pooled",
              smf.ols(f"absdC{h} ~ Stress * K_flow", data=d)),
             (f"{frame_tag}:volatility_society_FE",
              smf.ols(f"absdC{h} ~ Stress * K_flow + C(Society)", data=d))]
    for tag, mod in specs:
        m = mod.fit(cov_type="cluster", cov_kwds={"groups": d["Society"]})
        rows.append({
            "horizon_years": h, "model": tag, "n": int(len(d)),
            "beta_Stress": m.params["Stress"],
            "se_Stress": m.bse["Stress"],
            "beta_Kflow": m.params["K_flow"],
            "se_Kflow": m.bse["K_flow"],
            "beta_interaction": m.params["Stress:K_flow"],
            "se_interaction": m.bse["Stress:K_flow"],
            "p_interaction": m.pvalues["Stress:K_flow"],
            "R2": m.rsquared,
        })
    # simple slopes at low/high Flow Capacity (25th/75th pct), signed model
    m = specs[1][1].fit(cov_type="cluster", cov_kwds={"groups": d["Society"]})
    k25, k75 = d["K_flow"].quantile([0.25, 0.75])
    for label, kv in [("flow_p25", k25), ("flow_p75", k75)]:
        eff = m.params["Stress"] + m.params["Stress:K_flow"] * kv
        rows.append({
            "horizon_years": h, "model": f"{frame_tag}:society_FE:{label}",
            "n": int(len(d)),
            "beta_Stress": eff, "se_Stress": np.nan,
            "beta_Kflow": np.nan, "se_Kflow": np.nan,
            "beta_interaction": np.nan, "se_interaction": np.nan,
            "p_interaction": np.nan, "R2": np.nan,
        })

for h in (1, 5):
    fit_models(sy, "full", h)
    fit_models(sy_clean, "no_repeat", h)

res = pd.DataFrame(rows)
res.to_csv(OUT_TABLE, index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))

# ------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, h in zip(axes, (1, 5)):
    d = add_change(sy, h)
    # bin flow capacity into terciles for a visual read
    d["flow_tercile"] = pd.qcut(d["K_flow"], 3,
                                labels=["low Flow K", "mid", "high Flow K"])
    colours = {"low Flow K": "#c0392b", "mid": "#7f8c8d", "high Flow K": "#2980b9"}
    for terc, g in d.groupby("flow_tercile", observed=True):
        ax.scatter(g["Stress"], g[f"dC{h}"], s=8, alpha=0.25,
                   color=colours[terc], label=terc)
        # binned trend line
        bins = pd.qcut(g["Stress"], 6, duplicates="drop")
        mid = g.groupby(bins, observed=True)["Stress"].mean()
        mean = g.groupby(bins, observed=True)[f"dC{h}"].mean()
        ax.plot(mid, mean, color=colours[terc], lw=2)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("overall Stress (t)")
    ax.set_title(f"horizon h = {h} yr")
    if h == 1:
        ax.set_ylabel("change in overall Coherence  C(t+h) - C(t)")
    else:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Stress -> subsequent Coherence change, by Flow Capacity tercile "
             "(CAMS/JUNO panel, 36 societies)", y=1.02)
fig.tight_layout()
fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nwrote {OUT_TABLE} and {OUT_PLOT}")
