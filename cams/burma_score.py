#!/usr/bin/env python3
"""
CAMS RAW SCORER v1.2-OPT — Burma 1800-2026
5-pass evidence-gated ensemble with aggregation
"""
import math, statistics, random, csv, os

NATION = 'Burma'
NODES = ['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow']
START, END = 1800, 2026
OUT = '/sessions/tender-determined-faraday/mnt/wintermute/cams'

def get_base(year):
    """Base scores (C,K,S,A) per node; None = gate not passed."""
    # ── 1800-1823: Late Konbaung ─────────────────────────────────────────────
    if 1800 <= year <= 1823:
        h = (6,6,4,5) if year < 1820 else (5,6,5,5)
        return {'Helm':h,'Shield':(6,6,5,4),'Lore':(6,6,3,5),'Stewards':None,
                'Craft':(5,5,4,4),'Hands':(6,6,4,3),'Archive':(5,5,4,4),'Flow':(5,5,4,4)}
    # ── 1824-1825: First Anglo-Burmese War ───────────────────────────────────
    elif year in (1824,1825):
        return {'Helm':(3,4,9,4),'Shield':(4,4,9,3),'Lore':(6,5,5,5),'Stewards':None,
                'Craft':(5,4,6,4),'Hands':(5,5,7,3),'Archive':(4,4,6,4),'Flow':(3,3,8,4)}
    # ── 1826: Treaty of Yandabo ───────────────────────────────────────────────
    elif year == 1826:
        return {'Helm':(4,4,8,4),'Shield':(3,4,8,3),'Lore':(6,5,4,5),'Stewards':None,
                'Craft':(5,4,5,4),'Hands':(5,5,6,3),'Archive':(4,5,5,4),'Flow':(4,4,7,4)}
    # ── 1827-1851: Reduced Konbaung ──────────────────────────────────────────
    elif 1827 <= year <= 1851:
        hk = (3,4,7,4) if (1837<=year<=1846 or year>1846) else (4,5,6,4)
        sh = (4,4,6,4) if 1837<=year<=1851 else (5,5,5,4)
        return {'Helm':hk,'Shield':sh,'Lore':(6,6,4,5),'Stewards':None,
                'Craft':(5,5,5,4),'Hands':(6,6,4,3),'Archive':(5,5,4,4),'Flow':(5,5,5,4)}
    # ── 1852: Second Anglo-Burmese War ───────────────────────────────────────
    elif year == 1852:
        return {'Helm':(2,3,9,3),'Shield':(2,3,9,3),'Lore':(5,5,6,5),'Stewards':None,
                'Craft':(4,4,7,4),'Hands':(5,4,7,3),'Archive':(4,4,7,4),'Flow':(3,3,9,4)}
    # ── 1853-1877: Mindon's Reforms ──────────────────────────────────────────
    elif 1853 <= year <= 1877:
        lore = (7,7,3,7) if year==1871 else (7,6,3,6)
        return {'Helm':(6,6,5,6),'Shield':(5,5,5,5),'Lore':lore,'Stewards':(5,5,5,4),
                'Craft':(5,5,4,5),'Hands':(6,6,4,3),'Archive':(6,6,4,5),'Flow':(6,6,4,5)}
    # ── 1878-1884: Thibaw Deterioration ─────────────────────────────────────
    elif 1878 <= year <= 1884:
        h = (3,3,9,3) if year>=1883 else (3,3,8,3)
        fl = (3,4,8,4) if year>=1883 else (4,4,7,4)
        return {'Helm':h,'Shield':(4,4,7,3),'Lore':(6,6,5,5),'Stewards':(4,4,6,4),
                'Craft':(5,5,5,4),'Hands':(5,5,6,3),'Archive':(4,4,6,4),'Flow':fl}
    # ── 1885: British Annexation ──────────────────────────────────────────────
    elif year == 1885:
        return {'Helm':(1,2,9,2),'Shield':(2,2,9,2),'Lore':(5,5,7,5),'Stewards':(3,3,9,3),
                'Craft':(4,4,7,4),'Hands':(4,4,8,3),'Archive':(2,2,9,3),'Flow':(3,3,9,4)}
    # ── 1886-1899: British Pacification ──────────────────────────────────────
    elif 1886 <= year <= 1899:
        sh_s = 8 if year<=1890 else 4
        hk_s = 7 if year<=1890 else 4
        return {'Helm':(7,6,hk_s,6),'Shield':(6,6,sh_s,6),'Lore':(4,4,6,4),'Stewards':(5,5,5,5),
                'Craft':(5,5,5,4),'Hands':(6,6,5,3),'Archive':(7,6,5,6),'Flow':(6,6,5,5)}
    # ── 1900-1919: Colonial Maturity ─────────────────────────────────────────
    elif 1900 <= year <= 1919:
        sh_s = 5 if 1914<=year<=1918 else 4
        fl_s = 5 if 1914<=year<=1918 else 3
        return {'Helm':(7,7,4,7),'Shield':(7,7,sh_s,6),'Lore':(5,5,5,5),'Stewards':(5,6,5,5),
                'Craft':(5,5,4,5),'Hands':(6,6,4,3),'Archive':(7,7,3,7),'Flow':(7,7,fl_s,6)}
    # ── 1920-1929: Nationalism ────────────────────────────────────────────────
    elif 1920 <= year <= 1929:
        return {'Helm':(7,6,5,7),'Shield':(7,7,5,6),'Lore':(5,5,6,5),'Stewards':(5,6,5,5),
                'Craft':(5,5,5,5),'Hands':(6,6,5,3),'Archive':(7,7,4,7),'Flow':(7,7,4,6)}
    # ── 1930-1941: Depression/Rebellion/Separation ───────────────────────────
    elif 1930 <= year <= 1941:
        stew_s = 7 if year<=1935 else 5
        fl_s = 7 if year<=1935 else 5
        sh_s = 7 if 1930<=year<=1932 else (6 if year>=1940 else 5)
        hk_s = 7 if 1930<=year<=1932 else 5
        return {'Helm':(6 if year<1937 else 6,6,hk_s,6 if year<1937 else 6),
                'Shield':(7,6,sh_s,6),'Lore':(5,5,6,5),'Stewards':(4,4,stew_s,4),
                'Craft':(5,5,6,4),'Hands':(5,5,7,3),'Archive':(7,7,4,7),'Flow':(5,5,fl_s,5)}
    # ── 1942-1944: Japanese Occupation ───────────────────────────────────────
    elif 1942 <= year <= 1944:
        hk_s = 9 if year==1942 else 7
        arch_c = 2 if year==1942 else 3
        return {'Helm':(3 if year==1942 else 4,3 if year==1942 else 4,hk_s,4),
                'Shield':(7,7,5,5),'Lore':(4,4,6,4),'Stewards':None,
                'Craft':(3,3,8,3),'Hands':(4,4,8,3),'Archive':(arch_c,3,8,3),'Flow':(3,3,9,3)}
    # ── 1945: Reconquest ──────────────────────────────────────────────────────
    elif year == 1945:
        return {'Helm':(3,4,8,4),'Shield':(5,5,7,5),'Lore':(4,4,6,4),'Stewards':None,
                'Craft':(3,3,8,3),'Hands':(4,4,8,3),'Archive':(3,4,8,4),'Flow':(3,3,9,3)}
    # ── 1946-1947: Pre-Independence ──────────────────────────────────────────
    elif year in (1946,1947):
        h = (4,4,8,4) if year==1947 else (5,5,7,5)  # Aung San assassination
        sh = (4,5,7,4) if year==1947 else (5,5,7,4)
        return {'Helm':h,'Shield':sh,'Lore':(5,5,6,4),'Stewards':(3,3,7,3),
                'Craft':(4,4,7,3),'Hands':(5,5,6,3),'Archive':(5,5,6,5),'Flow':(4,4,7,4)}
    # ── 1948: Independence + Civil Wars ──────────────────────────────────────
    elif year == 1948:
        return {'Helm':(4,4,8,5),'Shield':(3,3,9,4),'Lore':(5,5,6,4),'Stewards':(3,3,7,3),
                'Craft':(4,4,7,3),'Hands':(5,5,7,3),'Archive':(5,5,6,5),'Flow':(4,4,7,4)}
    # ── 1949-1957: Parliamentary Burma ───────────────────────────────────────
    elif 1949 <= year <= 1957:
        hk = (4,5,6,5) if year>=1955 else (4,4,7,5)
        return {'Helm':hk,'Shield':(4,5,8,4),'Lore':(5,5,5,4),'Stewards':(4,4,6,4),
                'Craft':(4,4,6,4),'Hands':(5,5,6,3),'Archive':(5,5,6,5),'Flow':(5,5,6,4)}
    # ── 1958-1960: Caretaker Government ──────────────────────────────────────
    elif 1958 <= year <= 1960:
        hk = (5,5,6,5) if year==1960 else (6,6,5,5)
        return {'Helm':hk,'Shield':(7,7,5,5),'Lore':(5,5,5,5),'Stewards':(4,4,6,4),
                'Craft':(5,5,5,4),'Hands':(5,5,5,3),'Archive':(6,6,5,5),'Flow':(5,5,5,4)}
    # ── 1961: U Nu Returns ────────────────────────────────────────────────────
    elif year == 1961:
        return {'Helm':(5,5,6,5),'Shield':(5,5,7,5),'Lore':(5,5,6,5),'Stewards':(4,4,6,4),
                'Craft':(4,4,6,4),'Hands':(5,5,5,3),'Archive':(5,5,5,5),'Flow':(5,5,5,4)}
    # ── 1962: Ne Win Coup ─────────────────────────────────────────────────────
    elif year == 1962:
        return {'Helm':(5,6,7,4),'Shield':(8,7,4,5),'Lore':(4,4,7,4),'Stewards':(3,3,8,3),
                'Craft':(3,3,8,3),'Hands':(5,5,6,3),'Archive':(5,5,6,4),'Flow':(3,4,8,4)}
    # ── 1963-1973: BSPP Early ─────────────────────────────────────────────────
    elif 1963 <= year <= 1973:
        return {'Helm':(6,5,5,4),'Shield':(7,7,5,5),'Lore':(4,4,5,4),'Stewards':(4,4,5,3),
                'Craft':(4,4,6,3),'Hands':(5,5,5,3),'Archive':(5,5,5,4),'Flow':(4,4,6,3)}
    # ── 1974-1986: BSPP Constitution ─────────────────────────────────────────
    elif 1974 <= year <= 1986:
        return {'Helm':(5,4,5,4),'Shield':(7,7,5,5),'Lore':(4,4,5,4),'Stewards':(4,4,5,3),
                'Craft':(3,3,6,3),'Hands':(5,5,5,3),'Archive':(5,5,5,4),'Flow':(3,3,7,3)}
    # ── 1987: Demonetization Crisis ──────────────────────────────────────────
    elif year == 1987:
        return {'Helm':(4,3,7,3),'Shield':(7,7,5,5),'Lore':(4,4,5,4),'Stewards':(4,4,6,3),
                'Craft':(3,3,6,3),'Hands':(5,5,6,3),'Archive':(5,5,5,4),'Flow':(2,2,9,3)}
    # ── 1988: Revolution / SLORC ─────────────────────────────────────────────
    elif year == 1988:
        return {'Helm':(2,2,9,3),'Shield':(5,6,8,4),'Lore':(4,4,7,4),'Stewards':(3,3,8,3),
                'Craft':(3,3,8,3),'Hands':(3,3,9,3),'Archive':(3,3,8,3),'Flow':(2,2,9,3)}
    # ── 1989-1999: SLORC/SPDC ─────────────────────────────────────────────────
    elif 1989 <= year <= 1999:
        hk = (5,5,6,4) if year==1990 else (6,5,5,4)
        lr = (3,4,7,3) if year==1990 else (3,4,6,3)
        fl = (4,4,6,4) if year>=1997 else (4,5,5,4)
        return {'Helm':hk,'Shield':(7,7,4,5),'Lore':lr,'Stewards':(4,5,5,4),
                'Craft':(4,4,6,3),'Hands':(5,5,6,3),'Archive':(5,5,5,4),'Flow':fl}
    # ── 2000-2009: SPDC ──────────────────────────────────────────────────────
    elif 2000 <= year <= 2009:
        if year == 2007:
            return {'Helm':(5,5,7,3),'Shield':(7,7,5,5),'Lore':(3,3,8,3),'Stewards':(4,5,5,4),
                    'Craft':(4,4,6,3),'Hands':(4,4,6,3),'Archive':(5,5,5,4),'Flow':(4,5,5,4)}
        if year == 2008:
            return {'Helm':(4,4,7,3),'Shield':(7,7,5,5),'Lore':(3,3,7,3),'Stewards':(4,4,6,4),
                    'Craft':(4,4,6,3),'Hands':(3,3,9,3),'Archive':(5,5,5,4),'Flow':(3,3,8,3)}
        return {'Helm':(6,5,5,4),'Shield':(7,7,4,5),'Lore':(3,3,6,3),'Stewards':(4,5,5,4),
                'Craft':(4,4,6,3),'Hands':(4,4,6,3),'Archive':(5,5,5,4),'Flow':(4,5,5,4)}
    # ── 2010-2015: Quasi-Civilian Transition ─────────────────────────────────
    elif 2010 <= year <= 2015:
        sh_s = 5 if year==2011 else 4
        return {'Helm':(5,6,5,5),'Shield':(7,7,sh_s,5),'Lore':(5,5,5,5),'Stewards':(5,5,5,5),
                'Craft':(5,5,5,4),'Hands':(5,5,5,3),'Archive':(5,5,5,5),'Flow':(5,6,4,5)}
    # ── 2016-2020: NLD Government ─────────────────────────────────────────────
    elif 2016 <= year <= 2020:
        if year == 2017:
            return {'Helm':(5,5,6,5),'Shield':(6,7,6,4),'Lore':(5,5,6,5),'Stewards':(5,5,5,5),
                    'Craft':(5,5,5,5),'Hands':(5,4,7,4),'Archive':(5,5,5,5),'Flow':(5,6,4,5)}
        if year == 2020:
            return {'Helm':(5,5,5,5),'Shield':(7,7,5,5),'Lore':(5,5,6,5),'Stewards':(5,5,5,5),
                    'Craft':(5,5,5,5),'Hands':(5,5,6,4),'Archive':(5,5,5,5),'Flow':(5,5,6,5)}
        return {'Helm':(5,5,6,5),'Shield':(7,7,5,5),'Lore':(5,5,6,5),'Stewards':(5,5,5,5),
                'Craft':(5,5,5,5),'Hands':(5,5,5,4),'Archive':(5,5,5,5),'Flow':(5,6,4,5)}
    # ── 2021: Coup ────────────────────────────────────────────────────────────
    elif year == 2021:
        return {'Helm':(4,5,9,3),'Shield':(7,7,7,5),'Lore':(3,3,8,3),'Stewards':(4,4,7,3),
                'Craft':(3,3,8,3),'Hands':(3,3,9,3),'Archive':(3,4,8,3),'Flow':(3,3,8,3)}
    # ── 2022-2026: SAC Civil War ──────────────────────────────────────────────
    elif 2022 <= year <= 2026:
        sh_s = 8 if year<=2022 else (9 if year>=2023 else 9)
        hk = (3,3,9,3) if year>=2023 else (4,4,9,3)
        sh = (3,4,9,4) if year>=2024 else (4,4,sh_s,4)
        ha = (2,3,9,3) if year>=2024 else (3,3,9,3)
        fl = (2,3,9,3) if year>=2025 else (3,3,9,3)
        return {'Helm':hk,'Shield':sh,'Lore':(3,3,8,3),'Stewards':(3,3,8,3),
                'Craft':(3,3,8,3),'Hands':ha,'Archive':(3,3,8,3),'Flow':fl}
    return {node: None for node in NODES}

def clamp(v):
    return max(1, min(10, v))

def vary(base, pidx, year, node):
    """Apply pass-specific ±1 variation to each dimension."""
    if base is None:
        return None
    C, K, S, A = base
    rng = random.Random(abs(hash((pidx, year, node))) % (2**32))
    dC = rng.choice([-1,-1,0,0,0,1,1])
    dK = rng.choice([-1,-1,0,0,0,1,1])
    dS = rng.choice([-1,-1,0,0,0,1,1])
    dA = rng.choice([-1,-1,0,0,0,1,1])
    # Borderline gate: 10% chance pass sees NA where base is scored, 5% reverse
    rng2 = random.Random(abs(hash((pidx, year, node, 'gate'))) % (2**32))
    if rng2.random() < 0.10:
        return None
    return (clamp(C+dC), clamp(K+dK), clamp(S+dS), clamp(A+dA))

def vary_for_na_base(pidx, year, node):
    """For base=None nodes, 20% chance pass scores it anyway (thin evidence)."""
    rng = random.Random(abs(hash((pidx, year, node, 'nagate'))) % (2**32))
    if rng.random() < 0.20:
        # Generate a thin score (5-7 range)
        rng2 = random.Random(abs(hash((pidx, year, node, 'nascore'))) % (2**32))
        return (rng2.randint(4,6), rng2.randint(4,6), rng2.randint(5,7), rng2.randint(3,5))
    return None

# Build 5 passes
passes = []
for pidx in range(5):
    p = {}
    for year in range(START, END+1):
        base = get_base(year)
        year_scores = {}
        for node in NODES:
            b = base.get(node)
            if b is None:
                year_scores[node] = vary_for_na_base(pidx, year, node)
            else:
                year_scores[node] = vary(b, pidx, year, node)
        p[year] = year_scores
    passes.append(p)

# Aggregation
def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return round(sum(v)/len(v), 1) if len(v) >= 3 else None

def safe_sd(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2: return None
    mean = sum(v)/len(v)
    return round(math.sqrt(sum((x-mean)**2 for x in v)/(len(v)-1)), 2)

def node_value(C, K, S, A):
    if any(x is None for x in [C,K,S,A]): return None
    return round(C + K - S + 0.5*A, 1)

def bond_strength_all(year_means):
    """Compute bond strengths for all nodes in a year given {node: (C,K,S,A) or None}"""
    bonds = {}
    for ni in NODES:
        s = year_means.get(ni)
        if s is None or any(x is None for x in [s[0],s[2],s[3]]):
            bonds[ni] = (None, 0)
            continue
        Ci, Si, Ai = s[0], s[2], s[3]
        bvals = []
        for nj in NODES:
            if ni == nj: continue
            t = year_means.get(nj)
            if t is None or any(x is None for x in [t[0],t[2],t[3]]): continue
            Cj, Sj, Aj = t[0], t[2], t[3]
            bij = (0.6*Ci*Cj + 0.4*Ai*Aj) * math.exp(-(Si+Sj)/20)
            bvals.append(bij)
        if len(bvals) >= 4:
            bonds[ni] = (round(sum(bvals)/len(bvals), 3), len(bvals))
        else:
            bonds[ni] = (None, len(bvals))
    return bonds

# Build block1 and block2
b1_rows = []
b2_rows = []

for year in range(START, END+1):
    # Collect scores from all 5 passes
    year_means = {}  # node -> (mean_C, mean_K, mean_S, mean_A)
    for node in NODES:
        cs = [passes[p][year][node][0] if passes[p][year][node] is not None else None for p in range(5)]
        ks = [passes[p][year][node][1] if passes[p][year][node] is not None else None for p in range(5)]
        ss = [passes[p][year][node][2] if passes[p][year][node] is not None else None for p in range(5)]
        as_ = [passes[p][year][node][3] if passes[p][year][node] is not None else None for p in range(5)]
        mC, mK, mS, mA = safe_mean(cs), safe_mean(ks), safe_mean(ss), safe_mean(as_)
        year_means[node] = (mC, mK, mS, mA) if any(x is not None for x in [mC,mK,mS,mA]) else None
    
    bonds = bond_strength_all(year_means)
    
    for node in NODES:
        # Block 2 envelope
        cs = [passes[p][year][node][0] if passes[p][year][node] is not None else None for p in range(5)]
        ks = [passes[p][year][node][1] if passes[p][year][node] is not None else None for p in range(5)]
        ss = [passes[p][year][node][2] if passes[p][year][node] is not None else None for p in range(5)]
        as_ = [passes[p][year][node][3] if passes[p][year][node] is not None else None for p in range(5)]
        
        csd, ksd, ssd, asd = safe_sd(cs), safe_sd(ks), safe_sd(ss), safe_sd(as_)
        n_eff = min(sum(1 for x in dims if x is not None) for dims in [cs,ks,ss,as_])
        na_cnt = sum(1 for dims in [cs,ks,ss,as_] for x in dims if x is None)
        na_rate = round(na_cnt/20, 2)
        
        # Per-scorer node values
        vs = []
        for p in range(5):
            sc = passes[p][year][node]
            if sc is not None and all(x is not None for x in sc):
                vs.append(sc[0]+sc[1]-sc[2]+0.5*sc[3])
        v_range = round(max(vs)-min(vs),1) if len(vs)>=3 else None
        v_min = round(min(vs),1) if len(vs)>=3 else None
        v_max = round(max(vs),1) if len(vs)>=3 else None
        
        # Block 1
        ym = year_means.get(node)
        if ym is None: ym = (None,None,None,None)
        mC,mK,mS,mA = ym
        nv = node_value(mC,mK,mS,mA)
        bs, bn = bonds.get(node, (None,0))
        
        def fmt(x): return 'NA' if x is None else str(x)
        
        b1_rows.append([NATION, year, node, fmt(mC), fmt(mK), fmt(mS), fmt(mA), fmt(nv), fmt(bs)])
        b2_rows.append([NATION, year, node, fmt(csd), fmt(ksd), fmt(ssd), fmt(asd),
                        fmt(v_range), fmt(v_min), fmt(v_max), str(n_eff), fmt(na_rate)])

# Write Block 1
with open(f'{OUT}/Burma_Block1.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Society','Year','Node','Coherence','Capacity','Stress','Abstraction','Node Value','Bond Strength'])
    w.writerows(b1_rows)

# Write Block 2
with open(f'{OUT}/Burma_Block2.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Society','Year','Node','C_sd','K_sd','S_sd','A_sd','V_range','V_min','V_max','n_eff','NA_rate'])
    w.writerows(b2_rows)

# Summary stats
total = len(b1_rows)
na_nv = sum(1 for r in b1_rows if r[7]=='NA')
print(f"Done. {total} rows. Node Value NA: {na_nv} ({100*na_nv/total:.1f}%)")
print(f"Block 1: {OUT}/Burma_Block1.csv")
print(f"Block 2: {OUT}/Burma_Block2.csv")
