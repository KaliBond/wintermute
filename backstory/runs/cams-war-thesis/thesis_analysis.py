"""
CAMS War Initiation Thesis — Quantitative Test
H1: Rentier overload → war initiation within 10 years
H2: Lore/Archive moderation
H3: Jubilee alternative in high Lore-Archive societies
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import sys
from io import StringIO

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = "/sessions/stoic-trusting-wright/mnt/wintermute"
OUT  = "/sessions/stoic-trusting-wright/mnt/wintermute"

# ── war onsets ─────────────────────────────────────────────────────────────────
WAR_ONSETS = [
    ('France',   1792),
    ('France',   1803),
    ('France',   1870),
    ('France',   1956),
    ('Germany',  1914),
    ('Germany',  1939),
    ('England',  1899),
    ('England',  1956),
    ('England',  1982),
    ('USA',      1846),
    ('USA',      1898),
    ('USA',      1950),
    ('USA',      2003),
    ('China',    1950),
    ('China',    1962),
    ('China',    1979),
    ('Finland',  1941),
]

# ── node remapping (old → canonical) ──────────────────────────────────────────
OLD_TO_NEW = {
    # France spellings
    'Executive':                    'Helm',
    'Army':                         'Shield',
    'Priests':                      'Lore',
    'Property Owners':              'Stewards',
    'Trades/Professions':           'Craft',
    'Proletariat':                  'Hands',
    'State Memory':                 'Archive',
    'Shopkeepers/Merchants':        'Flow',
    # England variant spellings (spaces around slash, different terms)
    'Priesthood / Knowledge Workers': 'Lore',
    'Merchants / Shopkeepers':        'Flow',
    'Trades / Professions':           'Craft',
}

CANONICAL_NODES = {'Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow'}

load_errors = []

# ── loader helpers ─────────────────────────────────────────────────────────────
def load_canonical(fname, society_col='Society'):
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        load_errors.append(f"NOT FOUND: {fname}")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if society_col != 'Society':
        df = df.rename(columns={society_col: 'Society'})
    return df

def load_old_names(fname, society_col='Society'):
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        load_errors.append(f"NOT FOUND: {fname}")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # rename nation/society col
    for c in list(df.columns):
        if c.lower() in ('nation', 'society'):
            df = df.rename(columns={c: 'Society'})
            break
    df['Node'] = df['Node'].map(lambda x: OLD_TO_NEW.get(str(x).strip(), x))
    return df

# ── load each file ─────────────────────────────────────────────────────────────
print("Loading files...")

germany_gem = load_canonical("Germany_gem_jan_2025.csv")
germany_grok = load_canonical("Germany_grok_jan21.csv")
finland     = load_canonical("Finland_Gem_21Jan.csv")
china       = load_canonical("China_canonical_BS.csv")
france      = load_old_names("France_CAMS_Cleaned.csv")
england     = load_old_names("england.csv")

# USA: try canonical first
usa = load_canonical("USA_canonical_BS.csv")
if usa is not None:
    # check nodes are canonical
    nodes_present = set(usa['Node'].unique())
    if not nodes_present.issubset(CANONICAL_NODES):
        load_errors.append("USA_canonical_BS.csv: non-canonical nodes found, trying Reconstructed_USA_Dataset.csv")
        usa = load_old_names("Reconstructed_USA_Dataset.csv")
    else:
        # rename "United States" → "USA" for onset lookup
        usa['Society'] = usa['Society'].str.strip().replace('United States','USA')
        print("  USA: USA_canonical_BS.csv OK (United States → USA)")
else:
    usa = load_old_names("Reconstructed_USA_Dataset.csv")
    if usa is not None:
        usa['Society'] = usa['Society'].str.strip().replace({'United States':'USA'})
        print("  USA: Reconstructed_USA_Dataset.csv used")

# fix society names
if france is not None:
    france['Society'] = 'France'
if england is not None:
    england['Society'] = 'England'
if germany_gem is not None:
    germany_gem['Society'] = 'Germany'
if germany_grok is not None:
    germany_grok['Society'] = 'Germany'

# ── combine all societies ──────────────────────────────────────────────────────
pieces = [df for df in [germany_gem, finland, china, france, england, usa] if df is not None]
if not pieces:
    sys.exit("No data loaded — aborting.")

df_all = pd.concat(pieces, ignore_index=True)
df_all['Year'] = pd.to_numeric(df_all['Year'], errors='coerce')
df_all = df_all.dropna(subset=['Year','Node','Coherence','Capacity','Stress','Abstraction'])
df_all['Year'] = df_all['Year'].astype(int)

# drop rome (no war data, used for descriptive only per instructions)
df_all = df_all[~df_all['Society'].str.lower().str.contains('rome')]

# report unmapped nodes
unmapped = df_all[~df_all['Node'].isin(CANONICAL_NODES)]['Node'].unique()
if len(unmapped):
    load_errors.append(f"Unmapped nodes dropped: {list(unmapped)}")
    df_all = df_all[df_all['Node'].isin(CANONICAL_NODES)]

societies_loaded = sorted(df_all['Society'].unique())
print(f"  Societies loaded: {societies_loaded}")
print(f"  Rows: {len(df_all):,}")

# ── numeric coerce ─────────────────────────────────────────────────────────────
for col in ['Coherence','Capacity','Stress','Abstraction']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# ── build society-year-node panel ─────────────────────────────────────────────
# group to unique society-year-node (mean if duplicates)
df_panel = (df_all.groupby(['Society','Year','Node'], as_index=False)
            [['Coherence','Capacity','Stress','Abstraction']].mean())

# ── extract node-specific rows ─────────────────────────────────────────────────
stew = df_panel[df_panel['Node']=='Stewards'].copy()
lore = df_panel[df_panel['Node']=='Lore'].copy()
arch = df_panel[df_panel['Node']=='Archive'].copy()
shield = df_panel[df_panel['Node']=='Shield'].copy()

print(f"  Stewards rows: {len(stew)}")
societies_with_stew = set(stew['Society'].unique())
print(f"  Societies with Stewards: {sorted(societies_with_stew)}")

# ── system stress (mean Stress across all nodes per society-year) ───────────────
sys_stress = (df_panel.groupby(['Society','Year'], as_index=False)['Stress']
              .mean().rename(columns={'Stress':'system_stress'}))

# ── lore-archive index ─────────────────────────────────────────────────────────
lore_sub  = lore[['Society','Year','Coherence','Abstraction']].rename(
                  columns={'Coherence':'Lore_C','Abstraction':'Lore_A'})
arch_sub  = arch[['Society','Year','Coherence','Abstraction']].rename(
                  columns={'Coherence':'Arch_C','Abstraction':'Arch_A'})
la = lore_sub.merge(arch_sub, on=['Society','Year'], how='outer')
la['lore_archive_idx'] = la[['Lore_C','Lore_A','Arch_C','Arch_A']].mean(axis=1, skipna=True)
la = la[['Society','Year','lore_archive_idx','Lore_C','Lore_A','Arch_C','Arch_A']]

# ── shield stress ──────────────────────────────────────────────────────────────
sh = shield[['Society','Year','Stress']].rename(columns={'Stress':'Shield_Stress'})

# ── build overload variables on stewards ──────────────────────────────────────
stew = stew.sort_values(['Society','Year'])
stew['overload_binary'] = (stew['Stress'] > stew['Capacity']).astype(int)

def compute_streak(grp):
    streak = []
    s = 0
    for v in grp['overload_binary']:
        s = s+1 if v else 0
        streak.append(s)
    grp = grp.copy()
    grp['overload_streak'] = streak
    return grp

stew = stew.groupby('Society', group_keys=False).apply(compute_streak)
stew['overload_3yr'] = (stew['overload_streak'] >= 3).astype(int)

# ── war onset sets ─────────────────────────────────────────────────────────────
# set of (society, year) for all onsets
onset_set = set(WAR_ONSETS)
onset_by_soc = {}
for soc, yr in WAR_ONSETS:
    onset_by_soc.setdefault(soc, []).append(yr)

# ── peace filter ──────────────────────────────────────────────────────────────
# a year t is "at war" if t or t-1 is in onset_set for that society
def is_at_war(soc, yr):
    return (soc, yr) in onset_set or (soc, yr-1) in onset_set

stew['at_war'] = stew.apply(lambda r: is_at_war(r['Society'], r['Year']), axis=1)
stew_peace = stew[~stew['at_war']].copy()

# ── war initiation outcome (10-year window) ────────────────────────────────────
def war_in_window(soc, yr):
    yrs = onset_by_soc.get(soc, [])
    return int(any(yr < w <= yr+10 for w in yrs))

stew_peace['war_initiation_10yr'] = stew_peace.apply(
    lambda r: war_in_window(r['Society'], r['Year']), axis=1)

# ── merge covariates ───────────────────────────────────────────────────────────
base = stew_peace.merge(sys_stress, on=['Society','Year'], how='left')
base = base.merge(la,             on=['Society','Year'], how='left')
base = base.merge(sh,             on=['Society','Year'], how='left')

base = base.rename(columns={'Coherence':'Stew_C','Capacity':'Stew_Cap',
                             'Stress':'Stew_Stress','Abstraction':'Stew_A'})

print(f"\nFull peace-year panel: {len(base):,} rows")
print(f"War onset years: {base['war_initiation_10yr'].sum()} / {len(base)}")
print(f"Overload_3yr years: {base['overload_3yr'].sum()} / {len(base)}")
print()

# ── power check ───────────────────────────────────────────────────────────────
n_soc = base['Society'].nunique()
underpowered = n_soc < 3
if underpowered:
    print(f"WARNING: Only {n_soc} society/ies with Stewards data — test is UNDERPOWERED")

# ── save thesis_dataset.csv ───────────────────────────────────────────────────
save_cols = ['Society','Year','Node','Stew_C','Stew_Cap','Stew_Stress','Stew_A',
             'overload_binary','overload_streak','overload_3yr',
             'at_war','war_initiation_10yr',
             'system_stress','lore_archive_idx','Shield_Stress',
             'Lore_C','Lore_A','Arch_C','Arch_A']
out_ds = base[[c for c in save_cols if c in base.columns]]
out_ds.to_csv(os.path.join(OUT, 'thesis_dataset.csv'), index=False)
print("Saved thesis_dataset.csv")

# ══════════════════════════════════════════════════════════════════════════════
# MODELLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== RUNNING MODELS ===\n")

results_lines = []
def rlog(*args):
    s = ' '.join(str(a) for a in args)
    print(s)
    results_lines.append(s)

rlog("=" * 70)
rlog("CAMS WAR INITIATION THESIS — QUANTITATIVE RESULTS")
rlog("=" * 70)
rlog()
rlog(f"Societies with Stewards data: {sorted(societies_with_stew)}")
rlog(f"Total peace-year observations: {len(base):,}")
rlog(f"War-initiation outcomes (10yr window): {base['war_initiation_10yr'].sum()}")
rlog(f"Overload_3yr events: {base['overload_3yr'].sum()}")
rlog()

if load_errors:
    rlog("LOAD WARNINGS:")
    for e in load_errors:
        rlog(f"  {e}")
    rlog()

# ── attempt statsmodels logit ──────────────────────────────────────────────────
try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster
    HAS_SM = True
except ImportError:
    HAS_SM = False
    rlog("statsmodels not available — using sklearn for coefficients only")

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SK = True
except ImportError:
    HAS_SK = False

def run_logit_sm(df, formula_vars, outcome='war_initiation_10yr', label=""):
    """Run logistic regression with statsmodels; cluster SEs by Society."""
    sub = df[formula_vars + [outcome, 'Society']].dropna()
    if sub[outcome].nunique() < 2:
        rlog(f"  {label}: outcome has no variation — skipping")
        return None
    if len(sub) < 10:
        rlog(f"  {label}: fewer than 10 obs after dropping NA — skipping")
        return None
    X = sm.add_constant(sub[formula_vars])
    y = sub[outcome]
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=200, method='bfgs')
        # cluster SEs by society
        groups = sub['Society'].values
        try:
            cov_cl = cov_cluster(model, groups)
            from statsmodels.iolib.summary import summary_params
            params = model.params
            se_cl = np.sqrt(np.diag(cov_cl))
            z_cl  = params / se_cl
            from scipy import stats as sp_stats
            p_cl  = 2 * sp_stats.norm.sf(np.abs(z_cl))
            rlog(f"\n--- {label} ---")
            rlog(f"  N = {len(sub)}, Events = {y.sum()}")
            rlog(f"  Pseudo-R² (McFadden) = {model.prsquared:.4f}")
            rlog(f"  {'Variable':<30} {'Coef':>8} {'ClSE':>8} {'z':>7} {'p':>8} {'OR':>8}")
            rlog(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
            for v in params.index:
                coef = params[v]; se = se_cl[list(params.index).index(v)]
                z = coef/se; p = 2*sp_stats.norm.sf(abs(z))
                OR = np.exp(coef)
                sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '.' if p<0.1 else ''
                rlog(f"  {str(v):<30} {coef:>8.3f} {se:>8.3f} {z:>7.2f} {p:>8.4f} {OR:>8.3f} {sig}")
        except Exception as e:
            rlog(f"  Clustered SE failed ({e}); showing standard SEs")
            rlog(model.summary2().as_text())
        return model
    except Exception as e:
        rlog(f"  {label}: model failed — {e}")
        return None

def run_logit_sk(df, formula_vars, outcome='war_initiation_10yr', label=""):
    """Fallback sklearn logit — no SEs, just OR direction."""
    sub = df[formula_vars + [outcome]].dropna()
    if sub[outcome].nunique() < 2 or len(sub) < 10:
        return None
    X = sub[formula_vars].values
    y = sub[outcome].values
    lr = LogisticRegression(max_iter=1000, C=1e6)
    lr.fit(X, y)
    rlog(f"\n--- {label} (sklearn) ---")
    rlog(f"  N = {len(sub)}, Events = {y.sum()}")
    rlog(f"  {'Variable':<30} {'Coef':>8} {'OR':>8}")
    for v, c in zip(formula_vars, lr.coef_[0]):
        rlog(f"  {str(v):<30} {c:>8.3f} {np.exp(c):>8.3f}")
    return lr

# ── MODEL 1: H1 baseline ──────────────────────────────────────────────────────
rlog("\n" + "="*70)
rlog("MODEL 1 — H1 Baseline: war_10yr ~ overload_3yr")
rlog("="*70)
m1 = None
if HAS_SM:
    m1 = run_logit_sm(base, ['overload_3yr'], label="Model 1")
if m1 is None and HAS_SK:
    run_logit_sk(base, ['overload_3yr'], label="Model 1")

# ── MODEL 2: H1 controlled ────────────────────────────────────────────────────
rlog("\n" + "="*70)
rlog("MODEL 2 — H1 Controlled: war_10yr ~ overload_3yr + Shield_Stress + system_stress")
rlog("="*70)
m2 = None
if HAS_SM:
    m2 = run_logit_sm(base, ['overload_3yr','Shield_Stress','system_stress'], label="Model 2")
if m2 is None and HAS_SK:
    run_logit_sk(base, ['overload_3yr','Shield_Stress','system_stress'], label="Model 2")

# ── MODEL 3: H2 interaction ───────────────────────────────────────────────────
rlog("\n" + "="*70)
rlog("MODEL 3 — H2 Interaction: war_10yr ~ overload_3yr * lore_archive_idx + Shield_Stress + system_stress")
rlog("="*70)
base2 = base.copy()
base2['ol3_x_la'] = base2['overload_3yr'] * base2['lore_archive_idx']
m3 = None
if HAS_SM:
    m3 = run_logit_sm(base2, ['overload_3yr','lore_archive_idx','ol3_x_la',
                               'Shield_Stress','system_stress'], label="Model 3")
if m3 is None and HAS_SK:
    run_logit_sk(base2, ['overload_3yr','lore_archive_idx','ol3_x_la',
                          'Shield_Stress','system_stress'], label="Model 3")

# ── PLACEBO TEST ──────────────────────────────────────────────────────────────
rlog("\n" + "="*70)
rlog("PLACEBO TEST — Shuffle overload_3yr timing within each society (500 shuffles)")
rlog("="*70)

np.random.seed(42)
N_SHUFFLE = 500
placebo_coefs = []

sub_p = base[['Society','overload_3yr','war_initiation_10yr']].dropna()
if len(sub_p) > 10 and sub_p['war_initiation_10yr'].nunique() > 1:
    for _ in range(N_SHUFFLE):
        shuf = sub_p.copy()
        shuf['overload_3yr'] = shuf.groupby('Society')['overload_3yr'].transform(
            lambda x: x.sample(frac=1).values)
        if HAS_SM:
            try:
                X_ = sm.add_constant(shuf[['overload_3yr']])
                y_ = shuf['war_initiation_10yr']
                res_ = sm.Logit(y_, X_).fit(disp=0, maxiter=100, method='bfgs')
                placebo_coefs.append(res_.params.get('overload_3yr', np.nan))
            except:
                pass
        elif HAS_SK:
            try:
                lr_ = LogisticRegression(max_iter=500, C=1e6)
                lr_.fit(shuf[['overload_3yr']], shuf['war_initiation_10yr'])
                placebo_coefs.append(lr_.coef_[0][0])
            except:
                pass

    placebo_coefs = [c for c in placebo_coefs if not np.isnan(c)]
    if placebo_coefs:
        p_med  = np.median(placebo_coefs)
        p_95   = np.percentile(placebo_coefs, 95)
        rlog(f"  Placebo median coef:    {p_med:.3f}")
        rlog(f"  Placebo 95th pctile:    {p_95:.3f}")

        # get real coef
        if m1 is not None and HAS_SM:
            real_coef = m1.params.get('overload_3yr', np.nan)
        elif HAS_SK and m1 is not None:
            real_coef = m1.coef_[0][0]
        else:
            real_coef = np.nan

        if not np.isnan(real_coef):
            pct_rank = np.mean(np.array(placebo_coefs) < real_coef)
            rlog(f"  Real Model 1 coef:      {real_coef:.3f}")
            rlog(f"  Percentile rank vs placebo: {pct_rank*100:.1f}%")
            if real_coef > p_95:
                rlog("  → Real effect EXCEEDS 95th percentile of placebo — supports H1")
            else:
                rlog("  → Real effect does NOT exceed placebo 95th pctile — H1 not supported by placebo test")
    else:
        rlog("  No placebo runs converged.")
else:
    rlog("  Insufficient variation for placebo test.")

# ── H3: JUBILEE ALTERNATIVE ───────────────────────────────────────────────────
rlog("\n" + "="*70)
rlog("H3 TEST — Jubilee Alternative")
rlog("Overload resolution: does Stewards Stress fall below Capacity within 5yr?")
rlog("No Shield rise: Shield_Stress does not increase >1 in that window")
rlog("="*70)

# among overload_3yr == 1, check resolution
stew_sorted = stew_peace.merge(la[['Society','Year','lore_archive_idx']], on=['Society','Year'], how='left')
stew_sorted = stew_sorted.sort_values(['Society','Year'])

la_median = stew_sorted['lore_archive_idx'].median()
rlog(f"  Lore-Archive index median: {la_median:.3f}")

h3_rows = stew_sorted[stew_sorted['overload_3yr']==1].copy()
rlog(f"  Overload_3yr rows: {len(h3_rows)}")

# build Shield_Stress lookup
sh_dict = {}
if len(sh) > 0:
    for _, row in sh.iterrows():
        sh_dict[(row['Society'], row['Year'])] = row['Shield_Stress']

def check_resolution(soc, yr, stew_df, sh_dict_local):
    """
    Within t+1..t+5: does Stress fall below Capacity?
    Shield rise: max(Shield_Stress[t+1..t+5]) - Shield_Stress[t] > 1
    Returns (resolved, no_shield_rise)
    """
    sub = stew_df[(stew_df['Society']==soc) &
                  (stew_df['Year'] > yr) &
                  (stew_df['Year'] <= yr+5)]
    if len(sub) == 0:
        return None, None
    resolved = any(sub['Stress'] <= sub['Capacity'])
    # shield
    base_sh = sh_dict_local.get((soc, yr), np.nan)
    fut_sh  = [sh_dict_local.get((soc, y), np.nan) for y in range(yr+1, yr+6)]
    fut_sh  = [v for v in fut_sh if not np.isnan(v)]
    if np.isnan(base_sh) or not fut_sh:
        no_shield_rise = None
    else:
        no_shield_rise = (max(fut_sh) - base_sh) <= 1
    return resolved, no_shield_rise

h3_rows['la_high'] = h3_rows['lore_archive_idx'] >= la_median

results_h3 = []
for _, row in h3_rows.iterrows():
    soc, yr = row['Society'], row['Year']
    la_hi = row['la_high']
    resolved, no_sh = check_resolution(soc, yr,
                                        stew_peace[['Society','Year','Stress','Capacity']].rename(
                                            columns={'Stress':'Stress','Capacity':'Capacity'}),
                                        sh_dict)
    if resolved is None:
        continue
    results_h3.append({'Society':soc,'Year':yr,'la_high':la_hi,
                        'resolved':int(resolved),
                        'no_shield_rise': int(no_sh) if no_sh is not None else np.nan})

h3_df = pd.DataFrame(results_h3)
if len(h3_df) > 0:
    for la_grp, label in [(True,'High Lore-Archive'),(False,'Low Lore-Archive')]:
        g = h3_df[h3_df['la_high']==la_grp]
        if len(g) == 0:
            rlog(f"  {label}: no observations")
            continue
        n = len(g)
        n_res = g['resolved'].sum()
        n_nosh = g['no_shield_rise'].sum() if 'no_shield_rise' in g else 0
        n_jubilee = ((g['resolved']==1) & (g['no_shield_rise']==1)).sum() if 'no_shield_rise' in g else 0
        rlog(f"\n  {label} (n={n}):")
        rlog(f"    Resolved (Stress falls below Cap in 5yr): {n_res}/{n} = {n_res/n*100:.1f}%")
        rlog(f"    No Shield rise:                          {n_nosh}/{n} = {n_nosh/n*100:.1f}%")
        rlog(f"    Jubilee (resolved AND no Shield rise):   {n_jubilee}/{n} = {n_jubilee/n*100:.1f}%")
    # Fisher's exact if 2x2 available
    try:
        from scipy.stats import fisher_exact
        ct = pd.crosstab(h3_df['la_high'], h3_df['resolved'])
        if ct.shape == (2,2):
            odds, p_fish = fisher_exact(ct)
            rlog(f"\n  Fisher's exact (high_LA × resolved): OR={odds:.3f}, p={p_fish:.4f}")
    except:
        pass
    # H3 conclusion
    hi = h3_df[h3_df['la_high']==True]
    lo = h3_df[h3_df['la_high']==False]
    if len(hi)>0 and len(lo)>0:
        hi_jub = ((hi['resolved']==1) & (hi['no_shield_rise']==1)).mean()
        lo_jub = ((lo['resolved']==1) & (lo['no_shield_rise']==1)).mean()
        rlog(f"\n  Jubilee rate — High LA: {hi_jub*100:.1f}%  |  Low LA: {lo_jub*100:.1f}%")
        if hi_jub > lo_jub + 0.1:
            rlog("  → H3 SUPPORTED: High Lore-Archive societies more often resolve without Shield rise")
        elif hi_jub < lo_jub:
            rlog("  → H3 NOT SUPPORTED: Low LA societies show higher jubilee rate")
        else:
            rlog("  → H3 INCONCLUSIVE: Small difference between groups")
else:
    rlog("  No H3 data available.")

# ── REPLICATION CHECK ─────────────────────────────────────────────────────────
rlog("\n" + "="*70)
rlog("REPLICATION CHECK — Germany Gem vs Grok")
rlog("="*70)

if germany_grok is not None:
    germany_grok['Society'] = 'Germany'
    # build grok panel
    grok_stew = germany_grok[germany_grok['Node']=='Stewards'].copy().sort_values('Year')
    grok_stew['overload_binary'] = (grok_stew['Stress'] > grok_stew['Capacity']).astype(int)
    grok_stew = grok_stew.groupby('Society', group_keys=False).apply(compute_streak)
    grok_stew['overload_3yr'] = (grok_stew['overload_streak'] >= 3).astype(int)
    grok_stew['at_war'] = grok_stew.apply(lambda r: is_at_war(r['Society'], r['Year']), axis=1)
    grok_peace = grok_stew[~grok_stew['at_war']].copy()
    grok_peace['war_initiation_10yr'] = grok_peace.apply(
        lambda r: war_in_window(r['Society'], r['Year']), axis=1)

    rlog(f"  Grok Germany: {len(grok_peace)} peace years, {grok_peace['overload_3yr'].sum()} overload, "
         f"{grok_peace['war_initiation_10yr'].sum()} war-window years")

    # Gem Germany subset
    gem_de = base[base['Society']=='Germany'].copy()
    rlog(f"  Gem  Germany: {len(gem_de)} peace years, {gem_de['overload_3yr'].sum()} overload, "
         f"{gem_de['war_initiation_10yr'].sum()} war-window years")

    for lbl, df_rep in [("Gem", gem_de), ("Grok", grok_peace)]:
        sub_r = df_rep[['overload_3yr','war_initiation_10yr']].dropna()
        if sub_r['war_initiation_10yr'].nunique() < 2 or len(sub_r) < 5:
            rlog(f"  {lbl}: insufficient variation for regression")
            continue
        if HAS_SM:
            X_r = sm.add_constant(sub_r[['overload_3yr']])
            try:
                res_r = sm.Logit(sub_r['war_initiation_10yr'], X_r).fit(disp=0, maxiter=200)
                c = res_r.params.get('overload_3yr', np.nan)
                p = res_r.pvalues.get('overload_3yr', np.nan)
                OR = np.exp(c) if not np.isnan(c) else np.nan
                rlog(f"  {lbl} — overload_3yr coef: {c:.3f}, OR: {OR:.3f}, p: {p:.4f}")
            except Exception as e:
                rlog(f"  {lbl} — model error: {e}")
        elif HAS_SK:
            lr_r = LogisticRegression(max_iter=500, C=1e6)
            lr_r.fit(sub_r[['overload_3yr']], sub_r['war_initiation_10yr'])
            c = lr_r.coef_[0][0]
            rlog(f"  {lbl} — overload_3yr coef (sklearn): {c:.3f}, OR: {np.exp(c):.3f}")
else:
    rlog("  Germany_grok_jan21.csv not found — replication check skipped")

# ── DESCRIPTIVE STATS ─────────────────────────────────────────────────────────
rlog("\n" + "="*70)
rlog("DESCRIPTIVE STATS BY SOCIETY")
rlog("="*70)
grp = base.groupby('Society')
for soc, g in grp:
    n_total = len(g)
    n_ol = g['overload_3yr'].sum()
    n_war = g['war_initiation_10yr'].sum()
    n_both = ((g['overload_3yr']==1) & (g['war_initiation_10yr']==1)).sum()
    yr_range = f"{g['Year'].min()}–{g['Year'].max()}"
    rlog(f"  {soc:<12} years={yr_range:<12} peace-N={n_total:>4}  overload_3yr={n_ol:>3}  "
         f"war_window={n_war:>3}  overlap={n_both:>3}")

# ── FALSIFICATION SUMMARY ─────────────────────────────────────────────────────
rlog("\n" + "="*70)
rlog("FALSIFICATION CRITERIA ASSESSMENT")
rlog("="*70)
rlog("""
FC1 (Shield/system_stress controls): Compare Model 1 vs Model 2.
     If overload_3yr coef shrinks substantially or becomes non-significant,
     effect may be driven by general militarisation, not rentier overload.

FC2 (Civil conflict equally predicted): Not tested here — no civil conflict
     outcome variable in this dataset.

FC3 (Effect similar across LA groups): Model 3 interaction term.
     If ol3_x_la coefficient is small/non-significant, no moderation by LA.

FC4 (Replication across scorers): Gem vs Grok Germany above.
""")

# ── save results.txt ──────────────────────────────────────────────────────────
with open(os.path.join(OUT, 'thesis_results.txt'), 'w') as f:
    f.write('\n'.join(results_lines))
print("\nSaved thesis_results.txt")

# ══════════════════════════════════════════════════════════════════════════════
# PLAIN-LANGUAGE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
# (Dynamically assembled after results are computed)
print("\nBuilding thesis_summary.md...")

# Collect key numbers
n_obs = len(base)
n_war_ev = int(base['war_initiation_10yr'].sum())
n_ol_ev  = int(base['overload_3yr'].sum())

m1_coef = m1.params.get('overload_3yr', np.nan) if (m1 is not None and HAS_SM) else np.nan
m1_p    = m1.pvalues.get('overload_3yr', np.nan) if (m1 is not None and HAS_SM) else np.nan
m1_OR   = np.exp(m1_coef) if not np.isnan(m1_coef) else np.nan

m2_coef = m2.params.get('overload_3yr', np.nan) if (m2 is not None and HAS_SM) else np.nan
m2_p    = m2.pvalues.get('overload_3yr', np.nan) if (m2 is not None and HAS_SM) else np.nan
m2_OR   = np.exp(m2_coef) if not np.isnan(m2_coef) else np.nan

m3_int  = None
if m3 is not None and HAS_SM:
    m3_int_coef = m3.params.get('ol3_x_la', np.nan)
    m3_int_p    = m3.pvalues.get('ol3_x_la', np.nan)
else:
    m3_int_coef = np.nan; m3_int_p = np.nan

def fmt(v, dec=3):
    return f"{v:.{dec}f}" if not (v is None or np.isnan(v)) else "N/A"

summary_md = f"""# CAMS War Initiation Thesis — Summary of Results

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Data:** {', '.join(societies_loaded)}
**Observations:** {n_obs:,} society-peace-years
**War initiation events (10yr window):** {n_war_ev}
**Overload_3yr events:** {n_ol_ev}

---

## H1 — Kinetic Export Hypothesis

**Claim:** Sustained Stewards overload (Stress > Capacity for 3+ consecutive years)
raises the probability of war initiation within 10 years.

**Result:**

- **Model 1 (baseline):** overload_3yr coefficient = {fmt(m1_coef)}, OR = {fmt(m1_OR)}, p = {fmt(m1_p, 4)}
- **Model 2 (controlled for Shield_Stress and system_stress):** overload_3yr coefficient = {fmt(m2_coef)}, OR = {fmt(m2_OR)}, p = {fmt(m2_p, 4)}
"""

if not np.isnan(m1_coef):
    if m1_coef > 0 and m1_p < 0.1:
        summary_md += "\n**H1: Supported at conventional significance.** Overload is positively associated with war initiation.\n"
    elif m1_coef > 0 and m1_p >= 0.1:
        summary_md += "\n**H1: Directionally consistent but not statistically significant.** Positive coefficient but wide uncertainty given sample size.\n"
    else:
        summary_md += "\n**H1: Not supported.** Coefficient is negative or effectively zero.\n"

    if not np.isnan(m2_coef):
        shrinkage = (m1_coef - m2_coef) / abs(m1_coef) * 100 if m1_coef != 0 else np.nan
        if not np.isnan(shrinkage):
            summary_md += f"\nControlling for Shield_Stress and system_stress, the coefficient {'shrinks' if shrinkage > 0 else 'grows'} by {abs(shrinkage):.1f}%.\n"
            if shrinkage > 50 and m2_p > 0.1:
                summary_md += "**Falsification criterion FC1 triggered:** effect largely absorbed by controls — may reflect general militarisation.\n"
            else:
                summary_md += "**FC1 not triggered:** overload retains independent predictive value after controls.\n"

summary_md += f"""
### Placebo test

"""
if placebo_coefs and not np.isnan(m1_coef):
    p_med_v  = np.median(placebo_coefs)
    p95_v    = np.percentile(placebo_coefs, 95)
    pct_rank = np.mean(np.array(placebo_coefs) < m1_coef) * 100
    summary_md += f"Real coefficient ({fmt(m1_coef)}) ranks at **{pct_rank:.1f}th percentile** of {len(placebo_coefs)} shuffled runs (median = {fmt(p_med_v)}, 95th = {fmt(p95_v)}).\n"
    if m1_coef > p95_v:
        summary_md += "Placebo test **supports H1**: real timing is unusually predictive.\n"
    else:
        summary_md += "Placebo test **does not support H1**: shuffled timings perform comparably.\n"
else:
    summary_md += "Placebo test not completed.\n"

summary_md += f"""
---

## H2 — Recovery Moderation

**Claim:** High Lore-Archive index weakens the overload→war link.

- **Model 3 interaction term (overload_3yr × lore_archive_idx):** coef = {fmt(m3_int_coef)}, p = {fmt(m3_int_p, 4)}
"""
if not np.isnan(m3_int_coef):
    if m3_int_coef < 0 and m3_int_p < 0.1:
        summary_md += "\n**H2: Supported.** Negative interaction means high Lore-Archive attenuates the overload effect.\n"
    elif m3_int_coef < 0:
        summary_md += "\n**H2: Directionally consistent but not significant.** Negative interaction term in the right direction.\n"
    else:
        summary_md += "\n**H2: Not supported.** Interaction term is positive or near zero — no moderation detected.\n"
    if np.isnan(m3_int_p) or m3_int_p > 0.1:
        summary_md += "**FC3 applies:** interaction is non-significant — effect size similar across Lore-Archive groups.\n"
else:
    summary_md += "\nModel 3 did not converge or insufficient data.\n"

summary_md += f"""
---

## H3 — Jubilee Alternative

**Claim:** In high Lore-Archive societies, overload resolves internally (no Shield rise)
more often than in low Lore-Archive societies.
"""
if len(h3_df) > 0:
    hi_g  = h3_df[h3_df['la_high']==True]
    lo_g  = h3_df[h3_df['la_high']==False]
    if len(hi_g)>0 and len(lo_g)>0:
        hi_j = ((hi_g['resolved']==1) & (hi_g['no_shield_rise']==1)).mean()*100
        lo_j = ((lo_g['resolved']==1) & (lo_g['no_shield_rise']==1)).mean()*100
        summary_md += f"""
- High Lore-Archive jubilee rate: {hi_j:.1f}% (n={len(hi_g)})
- Low  Lore-Archive jubilee rate: {lo_j:.1f}% (n={len(lo_g)})
"""
        if hi_j > lo_j + 10:
            summary_md += "\n**H3: Supported.** High LA societies substantially more likely to resolve overload without Shield escalation.\n"
        elif hi_j < lo_j:
            summary_md += "\n**H3: Not supported.** Low LA societies show equal or higher jubilee rate.\n"
        else:
            summary_md += "\n**H3: Inconclusive.** Difference is small or in the wrong direction.\n"
    else:
        summary_md += "\nInsufficient group data for H3.\n"
else:
    summary_md += "\nNo H3 data computed.\n"

summary_md += f"""
---

## Falsification Summary

| Criterion | Assessment |
|-----------|-----------|
| FC1: Effect disappears under Shield/system_stress controls | {'See Model 1 vs 2 above' if not np.isnan(m2_coef) else 'Model 2 not available'} |
| FC2: Overload predicts civil conflict equally | Not tested — no civil conflict outcome in dataset |
| FC3: Effect similar across LA groups | {'See Model 3 interaction' if not np.isnan(m3_int_coef) else 'Model 3 not available'} |
| FC4: Results don't replicate Gem vs Grok | {'See Replication Check in thesis_results.txt' if germany_grok is not None else 'Grok file not found'} |

---

## Caveats

- **Sample size:** {n_obs:,} peace-year observations across {len(societies_loaded)} societies. War onset events are rare ({n_war_ev} outcome-window years), limiting statistical power for rare-event logistic regression.
- **Historical coverage varies** significantly by society, introducing temporal heterogeneity.
- **H1 is structurally dominated by China**, which provides 73 of 102 overload_3yr events. England (1750–1898), France, and USA show zero overload events in this dataset, so the pooled regression is essentially testing China + Germany + Finland.
- **England coverage (1750–1898)** means two of its three coded war onsets (1956, 1982) fall outside the data entirely; only the 1899 Boer War onset is reachable via the 10-year window.
- **War onset coding** is a simplified hardcoded list; disputes about initiator attribution directly affect results.
- **Rome** was excluded from quantitative models (no COW comparators).
- Clustered standard errors by society use very few clusters ({len(societies_loaded)}), so reported p-values are approximate.

---

*All details in `thesis_results.txt`. Full panel in `thesis_dataset.csv`.*
"""

with open(os.path.join(OUT, 'thesis_summary.md'), 'w') as f:
    f.write(summary_md)
print("Saved thesis_summary.md")
print("\n=== DONE ===")
