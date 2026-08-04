# JUNO / CAMS — Quick Reference Card

## The Eight Nodes (1–10 integer scores on C/K/S/A)

| Node | Function |
|---|---|
| Helm | Strategic direction, executive coordination |
| Shield | Security, order, coercive force |
| Lore | Knowledge, culture, legitimation |
| Stewards | Resources, capital allocation |
| Craft | Skilled production, professional delivery |
| Hands | Labour, basic throughput |
| Archive | Memory, records, institutional continuity |
| Flow | Commerce, trade, logistics |

**Slow loop** (reflective): Helm, Lore, Archive, Stewards  
**Fast loop** (reactive): Shield, Craft, Hands, Flow

## Four Dimensions

| Dim | What it measures |
|---|---|
| C — Coherence | Internal consistency; parts working in concert |
| K — Capacity | Demonstrated delivery under real conditions |
| S — Stress | Entropy production rate; observed breakdown |
| A — Abstraction | Quality of formal models, rules, planning |

Score **operational truth**. Stress = disorder produced, not external pressure.

## JUNO v1.2-Final Formulas

```
Node Value:   V  = C + K − S + 0.5·A

quality:      q  = (0.6·C + 0.4·A) / 10

pairwise bond B_ij = sqrt(q_i · q_j) · 2^(−(S_i+S_j)/10)

Bond Strength BS_i = mean(B_ij for j ≠ i)   [7 pairwise values]
```

## System Metrics

```
S/K ratio     = mean(S) / mean(K)         # >1.4 = critical
praetorian    = V(Shield) − V(Helm)       # >2.0 = military overreach
cog_gap       = mean(V_slow) − mean(V_fast)  # <−3 = cognitive capture
system_bond Λ = mean(BS across 8 nodes)   # <0.25 = bond decay
```

## Basin Classification

| Basin | Key Condition |
|---|---|
| Sovereign Optimum | sk<0.8, Λ>0.45 |
| Managed Tension | sk 0.8–1.0, Λ>0.35 |
| Praetorian Drift | praetorian>2.0 |
| Cognitive Capture | cog_gap<−3.0 |
| Bond Decay | Λ<0.25 |
| Fragmentation | sk>1.2, Λ<0.30 |
| Collapse Cascade | sk>1.4, multiple V<0 |

## Robson Gauge η_soil

```
η = (BS_Lore · BS_Archive · C_Hands) / (S_Hands · σ_V + 2.0)
σ_V = std of 8 Node Values (ddof=0)
```

| Path | Condition |
|---|---|
| Ruptured | η < 0.005 |
| Failed Transplant | η < 0.02 and σ_V > 3.5 |
| Standard Transplant | η < 0.08 and σ_V > 2.0 |
| Intermediate | η 0.08–0.15 |
| Young Ancient Forest | 0.08≤η≤0.15 and σ_V < 2.0 |
| Deep Ancient Forest | η > 0.15 and σ_V < 2.0 |
| Greenhouse Garden | η > 0.15 and σ_V ≥ 2.0 |

## Output Format (Block 1 CSV)

```
Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength
```

Eight rows per year. Compute V and BS from the formulas above.

## Corporate Mapping

Helm=Board, Shield=Risk/Legal, Lore=R&D/Culture, Stewards=Finance,  
Craft=Operations, Hands=Front-line, Archive=Data/KM, Flow=Sales/Supply

## City Mapping

Helm=Council/Planning, Shield=Police/Emergency, Lore=Education/Culture,  
Stewards=Municipal Finance, Craft=Infrastructure, Hands=Services,  
Archive=Records/Permits, Flow=Commerce/Transport

## Python Snippet

```python
import math
NODES = ['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow']

def juno(scores):
    q = {n: (0.6*scores[n]['C']+0.4*scores[n]['A'])/10 for n in NODES}
    out = {}
    for i,n in enumerate(NODES):
        V = scores[n]['C']+scores[n]['K']-scores[n]['S']+0.5*scores[n]['A']
        bs = [math.sqrt(q[n]*q[m])*(2**(-(scores[n]['S']+scores[m]['S'])/10))
              for j,m in enumerate(NODES) if j!=i]
        out[n] = {'V': round(V,3), 'BS': round(sum(bs)/7,5)}
    return out
```
