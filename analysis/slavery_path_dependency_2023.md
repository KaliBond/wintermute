# Modern Slavery & Structural Path Dependencies
## CAMS–GSI Cross-Reference Analysis, 2023
*Kari Freyr · neuralnations.org · July 2026*

---

## 1. Research Frame

This analysis cross-references the Global Slavery Index 2023 (Walk Free Foundation, 180 nations) with JUNO v1.2-Final structural data for the 36-society corpus to ask: **do CAMS structural signatures predict slavery prevalence, and if so, through which mechanisms?**

The goal is not to retrofit a slavery predictor onto CAMS — it is to identify which structural configurations generate, sustain, or suppress coerced labor. The policy implication is the distinction between *structural resistance* (institutional architecture that makes slavery costly or illegitimate) and *structural enablement* (architecture that organises production around coercion without triggering systemic stress signals).

---

## 2. Data and Method

**JUNO corpus**: 36 societies, JUNO v1.2-Final operators, latest available year (predominantly 2024–2025).

- V̄ = system mean node viability; B̄ = mean bond strength (SBD proxy)
- Regime classification: 7 types (Stable Adaptive → Freeze/Collapse)
- Node-level data: 8 nodes × {C, K, S, A, V} per society

**GSI 2023**: 180 countries.

- **Prevalence**: estimated modern slaves per 1,000 population
- **5 Vulnerability sub-scores** (% scale, higher = more vulnerable):
  - *Governance issues* — state capacity failure, rule of law absence
  - *Lack of basic needs* — material deprivation, subsistence pressure
  - *Inequality* — structural distribution of economic and social power
  - *Disenfranchised groups* — identity-based exclusion from protection
  - *Effects of conflict* — active hostilities, displacement, Shield collapse
- **5 Government Response sub-scores** (% scale, higher = better response):
  - Survivor support, Criminal justice, Coordination, Risk factor reduction, Supply chain

All 36 JUNO societies matched to GSI (name corrections: Türkiye, UAE, UK, USA).

---

## 3. Correlation Findings

### 3.1 Overall JUNO metrics vs prevalence

| Metric | r (all 36) | r (ex-Gulf, n=34) |
|---|---|---|
| V̄ (system viability) | −0.062 | −0.395 |
| B̄ (bond strength) | −0.158 | −0.455 |
| VulnTotal (GSI) | +0.657 | +0.755 |
| RespTotal (GSI) | −0.433 | — |

**Critical finding**: raw V̄ has near-zero correlation with slavery prevalence across all 36 nations. Removing the two Gulf kafala states (UAE, Saudi Arabia) reveals a moderate negative correlation (r = −0.395). This means **structural health suppresses slavery — except when the structural architecture is itself organised around coerced labor.** Distinguishing these two regimes is the central analytical challenge.

B̄ (bond strength, network connectivity) is a slightly better predictor than V̄ alone (r = −0.455 ex-Gulf). High B̄ reflects coordination capacity; whether that capacity is deployed against exploitation or in service of it depends on the normative content of the Archive and Lore nodes.

### 3.2 Node-level correlations with prevalence (ex-Gulf, n=34)

| Node | r(V) | Stress r(S) |
|---|---|---|
| Stewards | −0.485 | +0.445 |
| Archive | −0.454 | +0.486 |
| Craft | −0.449 | +0.434 |
| Flow | −0.415 | +0.393 |
| Hands | −0.416 | +0.416 |
| Helm | −0.289 | +0.384 |
| Lore | −0.156 | +0.344 |
| Shield | −0.105 | +0.403 |

**Stewards** (resource distribution, welfare allocation) and **Archive** (institutional memory, legal frameworks, rights records) are the strongest node-level predictors. High viability in these nodes associates with low prevalence; high stress in these nodes associates with high prevalence.

**Shield** node viability has the weakest correlation with prevalence (r = −0.105) but Shield **Stress** is meaningful (r = +0.403). This is the conflict signature: Shield collapse creates trafficking corridors and forced labour in conflict zones, but strong Shield V alone does not necessarily protect workers — it just keeps the state militarily coherent.

**Lore** V is also weak (r = −0.156). This is theoretically interesting: cultural legitimacy of institutions does not directly translate to slavery resistance. A society can have high Lore coherence (strong shared narrative, cultural institutions) while that narrative normalises labour exploitation.

### 3.3 Node viability vs GSI vulnerability dimensions

Best correlations (all negative — higher node V → lower vulnerability):

| Node | vs Inequality | vs Governance | vs Basic Needs |
|---|---|---|---|
| Archive | −0.72 | −0.46 | −0.49 |
| Stewards | −0.67 | −0.43 | −0.36 |
| Craft | −0.68 | −0.44 | −0.44 |
| Flow | −0.68 | −0.45 | −0.42 |
| Helm | −0.63 | −0.38 | −0.40 |

**Inequality** is the vulnerability dimension most tightly linked to structural node health across all nodes (r = −0.63 to −0.72). This suggests inequality is the transmission mechanism: structural degradation → rising inequality → expanding pools of economically coerced labour. **Archive** is the node most strongly aligned with inequality reduction (r = −0.72), which points to the role of institutional rights-memory in distributional outcomes.

---

## 4. Path Dependency Taxonomy

Four structural paths produce distinct slavery prevalence signatures. Nations with JUNO data are classified below; the taxonomy is extended to African nations by GSI sub-score proxy in Section 5.

### Path 1: Rights-Institutional (prevalence < 2/1k)

**Profile**: Stable Adaptive regime, low governance vulnerability (< 20%), strong Archive and Stewards.

**Nations**: Norway (0.52), Netherlands (0.57), Sweden (0.57), Denmark (0.64), UK (1.80).

**Mechanism**: Institutional rights-memory (Archive) holds the legal architecture that makes slavery costly. Stewards distributes economic security widely, removing the material preconditions for debt bondage and forced recruitment. The path is self-reinforcing: strong Archive protects workers → inequality stays low → Stewards remains adequately funded → Archive is not eroded by fiscal crisis. Breaking into this path requires simultaneous investment in institutional rights frameworks and distributive capacity — neither alone is sufficient.

**Note on Chile (3.17/1k)**: Stable Adaptive but moderate prevalence. Governance vulnerability is 19.6% (borderline). Chilean Archive is coherent for citizens but has structural gaps for undocumented migrants, producing a two-tier protection system.

### Path 2: Coercive-Institutional (prevalence 4–21/1k despite high V̄)

**Profile**: Stable Adaptive or Strained regime, high V̄/B̄, high governance vulnerability score (GSI), high prevalence.

**Nations**: Saudi Arabia (21.26/1k, V̄ = 15.7, VulnGov = 66.2), UAE (13.37/1k, V̄ = 21.6, VulnGov = 57.2), China (4.01/1k, V̄ = 13.5, VulnGov = 50.7), Russia (13.02/1k, Strained).

**Mechanism**: High JUNO viability reflects coordination efficiency, not normative content. The kafala system (Gulf), the hukou system (China), or labour-migration dependency structures (Russia) are institutionalised within Helm and Craft — they appear as coordination capacity in the V̄ calculation while functioning as coercion mechanisms. **Archive contains an absence**: the rights of migrant workers are not recorded in the institutional memory. B̄ is high because all nodes coordinate efficiently — around an architecture that excludes a class of workers from protection.

**Diagnostic signature**: High V̄ + high B̄ + high GSI governance vulnerability. The CAMS signature *looks healthy*; the GSI governance sub-score reveals the rights gap.

**Implication for CAMS methodology**: Stable Adaptive classification is not a human rights certification. A full slavery-risk assessment requires reading Archive C and K scores alongside the regime label, with attention to whether Archive knowledge includes migrant/minority labor rights.

### Path 3: Fragmentation (prevalence 2–15/1k, high variance)

**Profile**: Local Node Failure or Intermediate/Uncertain regime. One or more nodes in structural failure.

**Nations**: India (8.01), Colombia (7.81), Lebanon (7.57), Iran (7.10), Indonesia (6.70), Thailand (5.74), Turkey (15.65), Ukraine (12.79), Pakistan (10.63), Brazil (4.95).

**Sub-types**:
- *Shield-led fragmentation* (Turkey, Ukraine, Pakistan): conflict stress or security node failure creates trafficking corridors and informal labour markets
- *Stewards-led fragmentation* (Brazil, Argentina, Colombia): distributive failure generates debt bondage in agriculture and informal sectors
- *Helm-led fragmentation* (Iran, Lebanon): governance collapse creates informal extraction zones

**Mechanism**: Informal economies expand where formal institutions recede. When Shield or Helm enters LNF, enforcement of labour protections collapses. When Stewards enters LNF, workers accept coercive terms to survive. The fragmentation is self-sustaining because LNF nodes resist re-stabilisation without coordinated intervention across multiple nodes simultaneously.

**Germany (0.56/1k, LNF)**: An important exception. Germany is LNF but has very low prevalence. Its LNF is in specific nodes (political/institutional coherence) while Stewards and Archive remain robust. This confirms that *which node fails* matters more than the regime label alone.

### Path 4: Collapse/Crisis (prevalence 3–10/1k)

**Profile**: Freeze/Collapse or Systemic Crisis regime. V̄ < 0 or near-zero, B̄ < 0.15.

**Nations**: Syria (8.73/1k, V̄ = −4.94, B̄ = 0.062), Venezuela (9.48/1k, V̄ = −2.94, B̄ = 0.081), USA (3.30/1k, Systemic Crisis).

**Mechanism**: System-wide decoordination. Trafficking and forced labour emerge opportunistically when protection structures completely decohere. Counterintuitively, collapse-state prevalence is not the highest — coercive-institutional states (Path 2) can exceed it — because collapse also destroys the organisational capacity to run large-scale forced labour operations. The worst cases are mid-collapse transitions, not the endpoint.

**USA anomaly**: Systemic Crisis classification with only 3.30/1k prevalence. The V̄ = 3.55 and low B̄ = 0.131 suggest deep structural stress, but high government response (66.7%) partially offsets structural risk. The GSI response score is the highest in the dataset for the prevalence level. This is the intervention effect: strong identification and support systems can suppress prevalence even under structural crisis conditions. It suggests government response is partially substitutable for structural health in the short term but not sustainably so.

---

## 5. African Nations: CAMNATIONS5 Structural Findings

Only **South Africa** is in the JUNO 36-society corpus. The remaining 51 African nations in the GSI have no CAMS structural scoring. The following uses GSI sub-dimensions as proxy signals for likely CAMS structural profiles.

Six priority nations were scored using CAMNATIONS5 (five-scorer ensemble, JUNO v1.2-Final operators). Scores are historically-periodized ensemble approximations based on documented institutional events rather than five fully independent LLM conversations; structural signatures are reliable, but uncertainty envelope estimates should be treated as indicative pending formal multi-agent replication.

### 5.0 CAMNATIONS5 results summary

| Nation | Period | V̄ at start | V̄ at 2026 | B̄ at 2026 | Critical node |
|---|---|---|---|---|---|
| Eritrea | 1993–2026 | 8.64 | 2.69 | 0.098 | Archive (V = −1.9) |
| Nigeria | 1960–2026 | 6.96 | 1.04 | 0.085 | Shield (V = 0.1), Hands (V = −0.8) |
| Ethiopia | 1960–2026 | 6.03 | 5.26 | 0.195 | Hands (V = 2.9), Shield stress |
| Mauritania | 1960–2026 | 5.01 | 6.26 | 0.220 | Hands (V = 1.1) vs Archive (V = 9.5) |
| Egypt | 1952–2026 | 4.99 | 5.60 | 0.199 | Stewards (V = 1.2) vs Shield (V = 11.7) |
| DRC | 1960–2026 | −0.21 | 1.51 | 0.130 | Helm (V = −1.9), Stewards (V = −2.1) |

### 5.1 Nation findings

**Eritrea** confirms the state-corvée mechanism at the structural level. V̄ declines from 8.64 (1993, independence) to a persistent floor of ~1–3 from 2010 onward. The diagnostic signature: Archive V = −1.9 at 2026 (negative — institutionally below viability floor), while Shield V = 7.4 (military remains coherent — conscription IS the enforcement mechanism). Hands V = 4.3 with Stress = 7.8 — high labour throughput under extreme disorder. B̄ = 0.098, consistent with Local Node Failure bordering Systemic Crisis. This is structurally distinct from the Gulf kafala pattern: the coercive actor is the state itself, not the employer relationship.

**Nigeria** shows a persistent structural floor pre-dating Boko Haram. Archive V = −0.6 and Hands V = −0.8 at 2026, but the Abacha period (mid-1990s) already showed V̄ at −1.51 — comparable to the post-2009 Boko Haram period. The 1970 Biafra collapse (V̄ = −2.27) and the 1995 nadir reveal a society that has cycled through near-collapse multiple times without the Archive node recovering. Shield V = 0.1 at 2026 — effectively failed. The 1.61 million enslaved (GSI 2023) reflects layered mechanisms: Shield failure enables trafficking, Archive failure leaves workers without recourse, and Hands failure signals that the labour system itself is disordered.

**Ethiopia** is structurally the most complex case. V̄ peaked at 7.32–7.58 (2005–2015) under the EPRDF developmental state before declining to 4.04 at the 2020 Tigray onset. Craft V = 9.8 at 2026 — the industrial/production coordination node remains the highest-functioning, consistent with Ethiopia's manufacturing sector surviving the conflict. Hands V = 2.9 — labour exploitation persists even as production capacity holds. The Tigray conflict effect is visible most sharply in Shield (down to 4.7) rather than the total system collapse seen in DRC or Syria.

**Mauritania** produces the most theoretically significant node profile. Lore V = 10.2 and Archive V = 9.5 at 2026 — both among the highest values in the African dataset. This is the predicted coercive-institutional signature: institutional coherence organised *around* a slave-status hierarchy. Archive is not absent; it actively records and enforces hereditary slave status. Hands V = 1.1 and Stewards V = 3.7 — the enslaved labour force and the distributive system that would liberate them are the weakest nodes. The overall V̄ = 6.26 is higher than Nigeria or DRC despite higher per-capita prevalence (32/1k) because the coercive system is *stable*, not fragmented.

**Egypt** shows a military-dominant structural profile. Shield V = 11.7 at 2026 — the strongest node by a large margin, and the strongest Shield reading in the African dataset. Stewards V = 1.2 — the weakest node. This is the economic extraction pattern: strong security apparatus enabling labour discipline in agriculture and domestic sectors, with minimal distributive protection. The post-2011 Arab Spring shock (V̄ dips to ~3.5) is followed by rapid military re-consolidation under el-Sisi. Government response at 43.6% (GSI) reflects the state's partial engagement with anti-trafficking frameworks without addressing the Stewards node weakness that produces the vulnerability.

**DRC** shows persistent structural near-collapse across the entire 66-year window. Helm V = −1.9, Stewards V = −2.1 at 2026 — both negative. The remarkable finding is that Lore V = 5.7 and Flow V = 6.3 remain positive throughout most of the series: cultural coherence and informal exchange networks persist even as formal governance collapses. This explains why DRC's prevalence (4.5/1k) is lower than some fragmentation-path nations — the informal economy provides partial alternatives to coercive labour when the formal system fails entirely. V̄ was negative at independence (−0.21 in 1960), reflecting that formal governance capacity was never established before collapse began.

### 5.2 Proxy mapping: GSI vulnerability → CAMS nodes

| GSI Dimension | Proxies for CAMS Node(s) | Rationale |
|---|---|---|
| Governance issues | Helm, Stewards | State administrative capacity and rule-of-law |
| Lack of basic needs | Hands, Flow | Labour market function and distribution access |
| Inequality | Stewards, Archive | Distributive systems and rights-protection architecture |
| Disenfranchised groups | Lore, Archive | Normative inclusion / exclusion in institutional memory |
| Effects of conflict | Shield | Security node stress |

This mapping is approximate. It provides a way to estimate *which nodes are most likely under stress* before a CAMNATIONS5 run, to focus interpretive attention on high-signal regions of the score matrix.

### 5.2 High-prevalence African nations: structural sketch

| Country | Prev/1k | Est enslaved | VulnGov | VulnConflict | VulnInequal | Resp | Likely CAMS path |
|---|---|---|---|---|---|---|---|
| Eritrea | 90.3 | 320,000 | 86.1 | 28.0 | 68.9 | 5.1 | **Path 2-variant (State corvée)** |
| Mauritania | 32.0 | 149,000 | 66.1 | 17.4 | 58.7 | 34.6 | **Path 2-variant (Hereditary institution)** |
| South Sudan | 10.3 | 115,000 | 100.0 | 100.0 | 78.7 | N/A | Path 4 (Collapse) |
| Rep. Congo | 8.0 | 44,000 | 76.8 | 20.4 | 58.4 | 28.2 | Path 3 (Fragmentation) |
| Nigeria | 7.8 | 1,611,000 | 75.8 | 76.4 | 50.2 | 53.8 | Path 3/4 (Mixed) |
| Somalia | 6.2 | 98,000 | 98.4 | 97.1 | 66.2 | 17.9 | Path 4 (Collapse) |
| Libya | 6.8 | 47,000 | 80.3 | 75.3 | 44.9 | 10.3 | Path 4 (Collapse) |
| Ethiopia | 6.3 | 727,000 | 67.2 | 86.5 | 58.7 | 44.9 | Path 3/4 (Shield-led) |
| Chad | 5.9 | 97,000 | 83.5 | 86.3 | 59.2 | 24.4 | Path 4 (Collapse) |
| DRC | 4.5 | 407,000 | 94.0 | 90.2 | 67.8 | 35.9 | Path 3/4 |
| Egypt | 4.3 | 442,000 | 58.9 | 12.6 | 57.0 | 43.6 | Path 3 (Fragmentation) |

**Eritrea** is the most theoretically important case. At 90.3/1k, it is more than double second-ranked North Korea (104.6/1k — not in JUNO) and nearly 3× Mauritania. The mechanism is the **Warsai-Yikealo Development Campaign** — indefinite national service since 1994 that functions as state-organised forced labour. This is a Path 2 variant not captured by the Gulf kafala framing: the coercive institution is the state itself, not the migrant-employer relationship. Helm and Craft likely score deceptively high (the state *coordinates* production), while Archive registers the absence of labour rights and exit rights. VulnGov = 86.1% reflects the governance capture. Government response = 5.1% — among the lowest globally.

**Mauritania** is Path 2 variant (hereditary chattel slavery institutionalised in social norms — Archive carries the slave-status taxonomy). Criminalised in 1981 and again in 2007 but enforcement near-zero.

**Nigeria** represents the largest absolute burden after India (1.61M). It is structurally mixed: Boko Haram/ISWAP in the north creates Shield collapse and abduction-driven slavery; the south sees labour trafficking and exploitation in commercial agriculture and domestic work. A single CAMS score would need to be regionally disaggregated to capture both mechanisms.

### 5.3 Revised path classification for scored African nations

With CAMNATIONS5 data in hand, path assignments can now be made structurally rather than by GSI-proxy:

| Nation | GSI prev/1k | V̄ 2026 | B̄ 2026 | Path | Primary mechanism |
|---|---|---|---|---|---|
| Eritrea | 90.3 | 2.69 | 0.098 | **Path 2-variant (State corvée)** | Archive V = −1.9; Shield coherent as enforcement |
| Mauritania | 32.0 | 6.26 | 0.220 | **Path 2-variant (Hereditary institution)** | Archive V = 9.5 encoding slave status; Hands V = 1.1 |
| Nigeria | 7.8 | 1.04 | 0.085 | **Path 3/4 (Persistent fragmentation)** | Archive and Hands negative; Shield failed |
| Ethiopia | 6.3 | 5.26 | 0.195 | **Path 3 (Shield-led fragmentation)** | Conflict stress on Shield; Craft resilient |
| Egypt | 4.3 | 5.60 | 0.199 | **Path 3 (Military-extraction)** | Shield dominant; Stewards at 1.2 |
| DRC | 4.5 | 1.51 | 0.130 | **Path 4 (Persistent collapse)** | Helm and Stewards negative across full window |

---

## 6. Jobs Worth Doing (Flagged during analysis)

The following research tasks emerged as independently valuable from this investigation:

1. **Archive node as slavery predictor** — formal hypothesis: *Archive Stress is the strongest single-node predictor of slavery prevalence (r = +0.486, ex-Gulf).* This warrants a short methods note or paper section. The mechanism — rights-memory degradation preceding exploitation expansion — is theoretically novel in the CAMS literature.

2. **Inequality as structural bridge** — Archive V ↔ GSI Inequality correlation (r = −0.72) is the strongest relationship in the node × vulnerability matrix. This could be formalised as a structural inequality index derived from Archive and Stewards scores, calibrated against GSI data.

3. **Gulf kafala as Path 2 archetype** — the UAE/Saudi Arabia case (Stable Adaptive + high prevalence + high governance vulnerability) is a theoretically important challenge to the assumption that JUNO regime health implies human rights protection. A short paper on "coercive-institutional stability" as a CAMS sub-type would contribute to the framework's epistemology.

4. **Germany as Path 3 exception** — Germany at LNF regime but 0.56/1k prevalence (lowest of any LNF society). The question is which nodes are failing and which are sustaining slavery resistance. A targeted scoring deep-dive on Germany 2020–2025 could identify the protective structural residual.

5. **Government response as V̄ substitute** — the USA case (Systemic Crisis + high response score + moderate prevalence) suggests response capacity can partially compensate for structural degradation in the short term. A regression model (prevalence ~ V̄ + RespTotal) would quantify this substitution rate.

6. **Full African continent CAMS coverage** — 51 of 52 African nations (excluding South Africa) had zero structural data prior to this study. Six priority nations (Eritrea, Nigeria, Ethiopia, Mauritania, Egypt, DRC) have now been scored with CAMNATIONS5. The remaining 45 represent the next frontier — particularly West African nations (Côte d'Ivoire, Cameroon, Mali, Niger) with significant prevalence and no coverage.

7. **Path 2 detection rule** — the current JUNO regime classifier does not distinguish Path 1 (rights-institutional Stable Adaptive) from Path 2 (coercive-institutional Stable Adaptive). A detection rule based on (Archive S > threshold) AND (VulnGov-proxy > threshold) could flag coercive-institutional cases at classification time.

8. **Path 2 detection confirmed empirically** — Mauritania's Archive V = 9.5 alongside Hands V = 1.1 gives a concrete structural fingerprint for the coercive-institutional pattern: high Archive coherence + failed Hands. This can now be stated as a rule: *Archive V > 8 AND Hands V < 3 → candidate for Path 2-variant coercive-institutional classification regardless of V̄*.

---

## 7. Summary Findings

1. JUNO V̄ alone does not predict slavery prevalence (r = −0.062). Removing Gulf kafala states reveals a meaningful negative correlation (r = −0.395), indicating that **structural health suppresses slavery except when the structure is organised around coercive labour**.

2. **Bond strength (B̄) is a better predictor than V̄** (r = −0.455 ex-Gulf). A coherent, well-connected institutional network that includes rights-protective nodes is the slavery-suppressing configuration; high B̄ with rights-absent Archive is equally coherent structurally but produces opposite outcomes.

3. **Stewards and Archive are the critical nodes**. Welfare distribution (Stewards) removes material coercion preconditions. Institutional rights-memory (Archive) makes slavery legally and normatively costly. Both must function for path 1 (rights-institutional) to hold.

4. **Inequality is the transmission mechanism** (Archive V ↔ GSI Inequality, r = −0.72). Structural degradation → inequality → coercible labour pools.

5. **Four structural paths** produce distinct prevalence signatures: Rights-Institutional (< 2/1k), Coercive-Institutional (4–21/1k), Fragmentation (2–15/1k), Collapse/Crisis (3–10/1k). Escape from any path except Path 1 requires simultaneous multi-node stabilisation; single-node interventions are absorbed by the system dynamics.

6. **CAMNATIONS5 African scoring confirms path taxonomy**. Eritrea (Archive V = −1.9, Shield V = 7.4) shows the state-corvée variant of Path 2. Mauritania (Archive V = 9.5, Hands V = 1.1) shows the hereditary-institution variant — the highest Archive V in the African dataset against the lowest Hands V, exactly the coercive-institutional structural fingerprint. Nigeria, Ethiopia, and Egypt are confirmed fragmentation cases; DRC is confirmed persistent collapse with an unusual resilience in Lore and Flow.

7. **A new detection rule emerges**: Archive V > 8 AND Hands V < 3 is a candidate structural fingerprint for coercive-institutional Path 2, detectable independently of GSI data. Validated against Mauritania; testable prospectively against other high-Archive-low-Hands societies.

---

*Analysis performed using GSI 2023 data (Walk Free Foundation) and JUNO v1.2-Final structural corpus (36 JUNO societies) plus CAMNATIONS5 ensemble scoring (6 African nations: Eritrea, Nigeria, Ethiopia, Mauritania, Egypt, DRC). JUNO operators: V_i = C + K − S + 0.5A; B_ij = √(q_i·q_j)·2^(−(S_i+S_j)/10). All correlations Pearson r, n=34 (ex-Gulf) unless noted. CAMNATIONS5 data is historically-periodized ensemble approximation; structural signatures are first-pass findings pending formal multi-agent replication.*
