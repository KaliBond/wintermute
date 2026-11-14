"""
Audit of "SUPPORTED CONCLUSIONS?" document against JUNO_Blind_motif.csv
(12 societies x 61 years x 8 nodes). Verifies headline claims computationally.
Protocol conventions per CAMS Blind Analysis skill (carry-forward audit etc.).
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("JUNO_Blind_motif.csv")
NODES = ["Archive", "Craft", "Flow", "Hands", "Helm", "Lore", "Shield", "Stewards"]

# wide K-S sign matrices per society: rows = years, cols = nodes
def wide(metric_expr):
    d = df.copy()
    d["X"] = metric_expr(d)
    return {s: g.pivot(index="T_Offset", columns="Node", values="X")[NODES].sort_index()
            for s, g in d.groupby("Society_ID")}

KS = wide(lambda d: d["K"] - d["S"])
SGN = {s: (m > 0).astype(int) for s, m in KS.items()}
CFG = {s: m.apply(lambda r: "".join(r.astype(str)), axis=1) for s, m in SGN.items()}
M = {s: m.mean(axis=1) for s, m in KS.items()}          # aggregate M = mean(K-S)

print("=" * 70)
print("0. CARRY-FORWARD AUDIT (protocol flag: >30% = artefact concern)")
for s in KS:
    m = df[df.Society_ID == s].pivot(index="T_Offset", columns="Node",
                                     values=["C", "K", "S", "A"])
    rate = (m == m.shift()).all(axis=1).mean()
    print(f"  {s}: {rate:.1%}", end="  ")
print()

print("=" * 70)
print("1. REPERTOIRE RESTRICTION (claim: 13-27 states, p<0.0002 vs circular-shift null)")
rng = np.random.default_rng(7)
rep_obs, rep_null = {}, {}
for s, cfg in CFG.items():
    obs = cfg.nunique()
    nulls = []
    for b in range(200):
        # circular-shift each node's sign series independently (destroys joint config,
        # preserves own persistence) -> count distinct configs
        sh = SGN[s].apply(lambda col: np.roll(col.values, rng.integers(0, len(col))))
        nulls.append(sh.apply(lambda r: "".join(r.values.astype(str)), axis=1).nunique())
    rep_obs[s] = obs
    rep_null[s] = np.mean(nulls)
    p = (np.array(nulls) <= obs).mean()
    print(f"  {s}: observed {obs:3d} states | null mean {np.mean(nulls):6.1f} | "
          f"null<=obs {p:.4f}")
print(f"  range: {min(rep_obs.values())}-{max(rep_obs.values())} (claim: 13-27)")

print("=" * 70)
print("2. FIRST-PC VARIANCE OF K-S (claim: ~82.9% average)")
pcs = {}
for s, m in KS.items():
    X = (m - m.mean()) / m.std(ddof=0)
    X = X.fillna(0)
    _, sv, _ = np.linalg.svd(X, full_matrices=False)
    ve = sv**2 / (sv**2).sum()
    pcs[s] = ve[0]
    print(f"  {s}: PC1 = {ve[0]:.1%}  PC2 = {ve[1]:.1%}")
print(f"  mean PC1 = {np.mean(list(pcs.values())):.1%} (claim ~82.9%)")

print("=" * 70)
print("3. ONE-FACTOR SIGN RECONSTRUCTION (claim: ~89% of node signs correct)")
acc = []
acc3 = []
for s, m in KS.items():
    X = ((m - m.mean()) / m.std(ddof=0)).fillna(0)
    U, sv, Vt = np.linalg.svd(X, full_matrices=False)
    f1 = U[:, [0]] * sv[0]                    # first factor scores
    load = Vt[0]                               # node loadings
    recon = pd.DataFrame(f1 @ load.reshape(1, -1), index=m.index, columns=m.columns)
    correct = (recon > 0) == (m > 0)
    acc.append(correct.values.mean())
    f3 = U[:, :3] * sv[:3]
    recon3 = pd.DataFrame(f3 @ Vt[:3], index=m.index, columns=m.columns)
    acc3.append(((recon3 > 0) == (m > 0)).values.mean())
    print(f"  {s}: 1-factor {correct.values.mean():.1%} | "
          f"3-factor {((recon3 > 0) == (m > 0)).values.mean():.1%}")
print(f"  mean 1-factor = {np.mean(acc):.1%} (claim ~89%) | "
      f"mean 3-factor = {np.mean(acc3):.1%} (claim '>90% with a few dimensions')")

print("=" * 70)
print("4. RECOVERY / CONFIGURATION MISMATCH (claims: 40/44 split recoveries;")
print("   34/36 = 94.4% return with different config; 58% same-M pairs differ;")
print("   53.4% of M>0 obs have >=1 non-positive node)")
n_rec_split = n_rec_split_nonalign = 0
rec_cfg = []      # (pre_cfg, post_cfg, pre_M, post_M)
allM_pos_with_nonpos = 0
allM_pos = 0
for s, m in M.items():
    pos = (m > 0).astype(int).values
    cfg = CFG[s].values
    yrs = m.index.values
    # find recovery episodes: M<=0 -> M>0
    up = np.where(np.diff(pos) == 1)[0]
    for i in up:
        t0, t1 = yrs[i], yrs[i + 1]                      # last non-pos, first pos
        # end of positive spell
        j = i + 1
        while j + 1 < len(pos) and pos[j + 1] == 1:
            j += 1
        t_end = yrs[j]
        # 'split recovery': at the recovery year at least one node still non-positive
        sgn_t1 = SGN[s].loc[t1]
        if (sgn_t1 <= 0).any():
            n_rec_split += 1
            # does it ever reach full alignment during the positive spell?
            aligned = False
            for t in range(t1, t_end + 1):
                if (SGN[s].loc[t] > 0).all():
                    aligned = True
                    break
            if not aligned:
                n_rec_split_nonalign += 1
        # config change pre/post (pre = last year <=0, post = last year of spell)
        rec_cfg.append((cfg[i], cfg[j], m.iloc[i], m.iloc[j]))
    allM_pos += int((m > 0).sum())
    allM_pos_with_nonpos += int(((m > 0) & ((KS[s] <= 0).any(axis=1))).sum())

estim = [(a, b, pm, qm) for a, b, pm, qm in rec_cfg]
diff = sum(a != b for a, b, _, _ in estim)
print(f"  split recoveries (>=1 node non-positive at recovery): {n_rec_split} "
      f"(claim 44)")
print(f"  of those, never reach full alignment: {n_rec_split_nonalign} "
      f"(claim 19/40)")
print(f"  estimable recovery episodes: {len(estim)} | different config: {diff} "
      f"= {diff/len(estim):.1%} (claim 34/36 = 94.4%)")
sameM = [(a, b) for a, b, pm, qm in estim if abs(pm - qm) < 0.25]
if sameM:
    d2 = sum(a != b for a, b in sameM)
    print(f"  near-identical M recoveries: {len(sameM)} | different config: {d2} "
          f"(claim 14/14)")
print(f"  M>0 observations with >=1 non-positive node: {allM_pos_with_nonpos}/"
      f"{allM_pos} = {allM_pos_with_nonpos/allM_pos:.1%} (claim 53.4%)")
# same-M temporally separated pairs (claim: 58% differ)
n_same = n_same_diff = 0
for s, m in M.items():
    vals = m.round(2)
    for v, grp in vals.groupby(vals):
        idx = grp.index.values
        for a_i in range(len(idx)):
            for b_i in range(a_i + 1, len(idx)):
                if idx[b_i] - idx[a_i] >= 5:      # temporally separated
                    n_same += 1
                    if CFG[s].loc[idx[a_i]] != CFG[s].loc[idx[b_i]]:
                        n_same_diff += 1
print(f"  same-M separated pairs: {n_same} | different config: "
      f"{n_same_diff/max(n_same,1):.1%} (claim ~58%)")

print("=" * 70)
print("5. REPERTOIRE GEOMETRY (claim: effective span 4.96-20.40; switching")
print("   0.283-0.700; rho(switch, span) ~ 0.91)")
def entropy_eff(cfg_series):
    p = cfg_series.value_counts(normalize=True).values
    return float(np.exp(-(p * np.log(p)).sum()))
spans, switches = [], []
for s, cfg in CFG.items():
    e = entropy_eff(cfg)
    sw = (cfg != cfg.shift()).mean()
    spans.append(e); switches.append(sw)
    print(f"  {s}: effective span {e:5.2f} | switching {sw:.3f}")
print(f"  span range {min(spans):.2f}-{max(spans):.2f} (claim 4.96-20.40)")
print(f"  switching range {min(switches):.3f}-{max(switches):.3f} (claim 0.283-0.700)")
rho, pv = stats.spearmanr(spans, switches)
print(f"  rho(span, switch) = {rho:.2f} p={pv:.2g} (claim ~0.91)")

print("=" * 70)
print("6. NODE ROLE CONCORDANCE ACROSS SOCIETIES (claim: Kendall W ~0.33, p~1e-4)")
# role = mean K-S per node per society -> rank nodes -> W across 12 societies
R = []
for s, m in KS.items():
    R.append(m.mean().rank(ascending=False).reindex(NODES).values)
R = np.array(R)
k = len(NODES); n = len(R)
S = ((R.sum(axis=0) - n * (k + 1) / 2) ** 2).sum()
W = 12 * S / (n**2 * (k**3 - k))
# permutation null
cnt = 0
perm = np.random.default_rng(3)
for b in range(20000):
    Rp = np.array([perm.permutation(r) for r in R])
    Sp = ((Rp.sum(axis=0) - n * (k + 1) / 2) ** 2).sum()
    if 12 * Sp / (n**2 * (k**3 - k)) >= W:
        cnt += 1
print(f"  Kendall's W = {W:.3f} | perm p = {cnt/20000:.2g} (claim W~0.33, p~1e-4)")
means = pd.DataFrame({s: KS[s].mean() for s in KS}).T
print("  node mean K-S (positive-leaning vs negative-leaning):")
print(means.mean().sort_values(ascending=False).round(2).to_string())

print("=" * 70)
print("7. SHARED THRESHOLD CHAIN (claim: ~72% of split observations within one")
print("   node flip of Shield>Lore>Archive>Craft>Helm>Flow>Stewards>Hands; 14% exact)")
CHAIN = ["Shield", "Lore", "Archive", "Craft", "Helm", "Flow", "Stewards", "Hands"]
def flip_dist(cfg_bits, order):
    """min flips to make positive set a prefix of `order` (nested chain)."""
    pos = set(np.array(order)[np.array(cfg_bits) == 1])
    best = 8
    for k in range(9):
        prefix = set(order[:k])
        best = min(best, len(pos ^ prefix))
    return best
dists = []
for s in SGN:
    X = SGN[s][CHAIN]
    for _, row in X.iterrows():
        npos = int(row.sum())
        if 0 < npos < 8:                      # "split" observation: mixed signs
            dists.append(flip_dist(row.values, CHAIN))
dists = np.array(dists)
print(f"  n split obs = {len(dists)} | within 1 flip: {(dists<=1).mean():.1%} "
      f"(claim ~72%) | exact on chain: {(dists==0).mean():.1%} (claim ~14%)")
# null: chains with matched node marginals (shuffle order respecting positivity rate)
rnd = np.random.default_rng(11)
nullw1, nullex = [], []
order_by_rate = None
for b in range(200):
    order = list(rnd.permutation(CHAIN))
    dd = np.array([flip_dist(row.values, order) for _, row in
                   pd.concat([SGN[s][CHAIN] for s in SGN]).iterrows()
                   if 0 < row.sum() < 8][::3])
    nullw1.append((dd <= 1).mean()); nullex.append((dd == 0).mean())
print(f"  random-chain null (split obs): within1 {np.mean(nullw1):.1%}, "
      f"exact {np.mean(nullex):.1%}")
# best-fitting chain learned from positivity rates (common-field nested order)
rate = pd.DataFrame({s: SGN[s][CHAIN].mean() for s in SGN}).mean(axis=1).sort_values(
    ascending=False)
print("  chain by overall positivity rate:")
print("   ", " -> ".join(rate.index.tolist()))

print("=" * 70)
print("8. NODE-LOCAL PERSISTENCE vs CROSS-NODE GRAMMAR (light held-out check)")
print("   predict K-S sign(t) from [factor] / [factor+own lag] / [factor+all lags]")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
res_rows = []
for s, m in KS.items():
    X = ((m - m.mean()) / m.std(ddof=0)).fillna(0)
    U, sv, Vt = np.linalg.svd(X, full_matrices=False)
    F = U[:, :3] * sv[:3]                       # few latent dimensions
    ny = len(m)
    y = (m > 0).astype(int).values
    own = np.vstack([np.zeros((1, 8)), (m > 0).astype(int).values[:-1]])
    alllag = np.vstack([np.zeros((1, 8)), X.values[:-1]])
    y_flat = y.ravel()                                   # year-major (t, node)
    F_flat = np.repeat(F, 8, axis=0)
    own_flat = own.ravel()
    all_flat = np.repeat(alllag, 8, axis=0)
    for tag, feats in [("factor", F_flat),
                       ("factor+own", np.hstack([F_flat, own_flat[:, None]])),
                       ("factor+alllags", np.hstack([F_flat, all_flat]))]:
        tscv = TimeSeriesSplit(n_splits=5)
        aucs = []
        for tr, te in tscv.split(feats):
            if y_flat[tr].sum() in (0, len(tr)) or y_flat[te].sum() in (0, len(te)):
                continue
            clf = LogisticRegression(max_iter=2000).fit(feats[tr], y_flat[tr])
            aucs.append(roc_auc_score(y_flat[te],
                                      clf.predict_proba(feats[te])[:, 1]))
        res_rows.append((s, tag, np.mean(aucs)))
r = pd.DataFrame(res_rows, columns=["society", "model", "auc"]).pivot(
    index="society", columns="model", values="auc")
print(r.round(3).to_string())
own_gain = (r["factor+own"] - r["factor"])
all_gain = (r["factor+alllags"] - r["factor+own"])
print(f"  own-lag AUC gain: median {own_gain.median():+.3f}, "
      f"positive in {(own_gain>0).sum()}/12 (claim: substantial in all 12)")
print(f"  cross-node extra gain over own-lag: median {all_gain.median():+.3f}, "
      f"positive in {(all_gain>0).sum()}/12 (claim: not consistent)")
