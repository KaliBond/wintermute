# CAMS FORMULATIONS — HISTORY AND LINEAGE

**Maintainer:** Kari Freyr McKern
**Assembled:** 2026-04-09
**Status:** Living reference — audit of formula evolution across all CAMS versions
**Authority:** CAMS-CANONICAL.md (v1.0-RC1) takes precedence over all earlier formulations recorded here.

---

## PURPOSE

This document records the full developmental history of CAMS (Common Adaptive Model of Society / Civilizational Analysis and Measurement System) formulations from earliest inception through the current canonical standard. It is not a scoring reference — it is a forensic audit and lineage record.

Where formulations conflict across versions, conflicts are explicitly flagged. Nothing here has been silently reconciled. The canonical resolution for any conflict is CAMS-CANONICAL.md.

---

## FRAMEWORK IDENTITY

| Field | Value |
|---|---|
| Full canonical name | Common Adaptive Model of Society — Catch-All Network |
| Short form | CAMS-CAN |
| Originator | Kari Freyr McKern |
| Origin date | 27 September 2024 |
| First public appearance | *Pearls and Irritations*, October 2024 |
| Predictive accuracy (validated) | 78% overall; 87% for collapse within 25-year window |
| Framework class | Complex adaptive systems analysis; civilisational diagnostics |
| Epistemological character | *Keplerian, not Newtonian*: empirically calibrated rather than derived from first principles |
| Intellectual ancestry | Comte's positivism; cliodynamics; complexity science; Socratic realism |
| Positioning | Third major unification attempt in social science, after Comte and cliodynamics |

### Origin Context

CAMS originated in mid-2024 through collaboration with AI as a forcing function for logical rigour, emerging from epistemological frustration with the fact-value distinction in social science. The framework was conceived as a Keplerian enterprise: start from empirical observation of what societies do, build metrics that describe those behaviours, and only then derive laws from the patterns — not from first principles assumed in advance.

Key publications positioning CAMS appeared in *Pearls and Irritations* (October 2024 – March 2024), with geopolitical prescience validation drawn from the author's 2022–2024 essay series.

---

## THEORETICAL FOUNDATION

Societies are complex adaptive systems (CAS). CAMS operationalises them as distributed networks in which institutional nodes perform specialised functions, interact through measurable bonds, accumulate or shed adaptive capacity over time, and undergo phase transitions when stress exceeds systemic resilience.

The **Sybond Hypothesis** is the theoretical core: a Sybond is a collective intelligence greater than the sum of its parts, emerging from interactions among individuals, institutions, and cultural forces. The Sybond is not a metaphor — it is a claim about what societies actually are: entities that can be healthy or sick, adaptive or brittle, and that undergo genuine phase transitions, not merely surface-level changes, when they reach critical thresholds. CAMS-CAN treats societal nodes as the organs of this emergent entity.

The full state description of any society at any time is **32-dimensional**: 8 nodes × 4 metrics = 32 independent measurements per time-slice.

---

## SECTION 1 — NODE ARCHITECTURE HISTORY

### 1.1 Earliest descriptive node set (pre-Sep 2024)

Before formal naming, nodes were described in functional-descriptive language:

```
Leadership, Warrior Class, Priesthood, Property Owners,
Skilled Producers, Workers, Merchants, Collective Memory
```

Also attested in early layer as:
```
Leadership, Warrior Class, Collective Memory, Priests,
Property Owners, Skilled Producers, Workers, Merchants
```

**Dependency wording used:** "Leadership can only function if supported by Warriors, Priests, Money, Workers, Producers, and Memory."

**Confidence:** MEDIUM for exact order; HIGH for the set composition.

---

### 1.2 Classical eight-node set (Sep–Nov 2024, first code implementation Feb 2025)

First standardised implementation. Confirmed in `cams-scorer.tsx` (2025-02-08):

```javascript
const nodes = [
  'Executive', 'Army', 'Priests', 'Property Owners',
  'Trades/Professions', 'Proletariat', 'State Memory', 'Merchants'
];
```

Also attested in variant forms:
- `Priests` → `Priesthood` (later standardisation)
- `Merchants` → `Storekeepers` (transitional)
- `Storekeepers/Merchants` → `Shopkeepers/Merchants` (concurrent variant)

**Transitional note:** "The Storekeepers" were added as a distinct node to represent the merchandising/goods-selling function, leading to brief confusion between `Merchants`, `Shopkeepers/Merchants`, and `Storekeepers` as labels.

**Nine-node aberration (2025-05-01):** One secondary synthesis document (`The CAMS-CAN Framework_ Kari McKern's Complex Adap (1).md`) mistakenly lists nine nodes by treating "Abstraction systems" as a separate node. This is not canonical and is flagged as a secondary synthesis error.

---

### 1.3 Ten-node variant (superseded July 2025)

An earlier development phase used a ten-node model. This was formally superseded in July 2025 (RD-002) in favour of the eight-node canonical model on grounds of parsimony and functional completeness across 32+ societies. The ten-node variant's exact composition is not fully recovered from accessible source files.

---

### 1.4 Canonical classical set (2025, pre-renaming)

| # | Canonical name | Functional role |
|---|---|---|
| 1 | Executive | Leadership, governance, decision-making |
| 2 | Army | Defence, coercion, strategic enforcement |
| 3 | Priests / Knowledge | Ideology, religion, legitimacy, intellectual infrastructure |
| 4 | Property Owners | Landholding, asset control, economic anchoring |
| 5 | Trades / Professions | Skilled labour, professional expertise, innovation |
| 6 | Proletariat | Mass labour, productive base |
| 7 | State Memory | Laws, institutions, archives, continuity mechanisms |
| 8 | Merchants / Shopkeepers | Commerce, trade, wealth circulation |

### 1.4a Gemini variant node names (CAMS-CAN-GEMINI.md, Apr 2026)

A parallel canonical reference produced in the Gemini environment uses shortened functional labels as primary names, with the archetypal names as aliases:

| # | Gemini canonical name | Historical / Analogous labels |
|---|---|---|
| 1 | Executive | Helm, Government, Leadership, Monarchy, Oligarchy |
| 2 | Army | Shield, Military, Defense, Warriors, Armed Forces |
| 3 | Knowledge | Lore, Priests, Clergy, Academy, Knowledge Class |
| 4 | Property | Stewards, Landlords, Merchant-princes, Owners |
| 5 | Trades | Craft, Professions, Guild, Factory Workers, Producers |
| 6 | Labor | Hands, Workers, Servants, Labourers, Commons |
| 7 | Memory | Archive, State Memory, Keepers, Historians, Traditions |
| 8 | Commerce | Flow, Merchants, Banking, Markets |

**Note:** This variant diverges from CAMS-CANONICAL.md primarily in preferring shorter functional labels (Knowledge, Labor, Commerce) over the compound classical forms (Priests/Knowledge, Proletariat, Merchants/Shopkeepers). The archetypal names (Helm, Shield, Lore, etc.) are treated as aliases rather than primary labels in this variant.

---

### 1.4 Archetypal renamed set (2025–2026)

Culturally neutral, archetypal terminology introduced for essays, reports, and visualisations.

| Classical name | Archetypal name |
|---|---|
| Executive | **Helm** |
| Army | **Shield** |
| Priesthood / Knowledge Workers | **Lore** |
| Property Owners | **Stewards** |
| Trades / Professions | **Craft** |
| Proletariat | **Hands** |
| State Memory | **Archive** |
| Merchants / Shopkeepers | **Flow** |

**Reason stated:** Culturally neutral, archetypal terminology for essays, reports, and visualisations.
**Confirmed in:** CAPE branch (`mythos.txt`, March 14 2026), v3.0 (`formulation3.0.txt`, March 13 2026), CAMS-CANONICAL.md.

---

### 1.5 Deprecated node variants

| Deprecated label | Context | Canonical equivalent | Status |
|---|---|---|---|
| Checksums / Dissent | Early CAMS pre-2025 | Merged into Archive + Lore | **Deprecated** |
| Storekeepers | Very early variant | Flow | **Deprecated** |
| External Environment | instructionsa.txt (Gemini Gem variant) | Exogenous context field only | **Not canonical** |
| Nine-node model | Secondary synthesis 2025-05-01 | Eight-node model is canonical | **Deprecated** |

---

## SECTION 2 — METRIC HISTORY

### 2.1 Earliest metric variant (Sep–Nov 2024)

**Metric set:** Coherence, Capacity, Stress, **Impact**

This was the original fourth metric. Impact was later replaced by Abstraction. The transition is documented in project memory but no exact date is confirmed. The reason for the change: Impact measured reach or effect — what a node produced externally — whereas Abstraction captures the depth of strategic and symbolic thinking within the node itself. The shift reflected a recognition that what most distinguished adaptive, resilient societies was not simply their reach but their internal sophistication.

**Confidence:** MEDIUM for exact timing; HIGH for the existence of this variant.

---

### 2.2 Standard four-metric set (Feb 2025 onwards)

**Confirmed in** `cams-scorer.tsx`:
```javascript
const metrics = ['Coherence', 'Capacity', 'Stress', 'Abstraction'];
```

Scale: `min=1, max=10`

| Metric | Definition | Scale |
|---|---|---|
| **Coherence (C)** | Internal unity, alignment, legitimacy, organisational consistency | 1–10 |
| **Capacity (K)** | Material, organisational, financial, operational, human ability to act | 1–10 |
| **Stress (S)** | Pressure, strain, crisis load, destabilising burden | See Section 2.3 |
| **Abstraction (A)** | Strategic sophistication, symbolic reach, intellectual depth, planning horizon | 1–10 |

---

### 2.3 Stress convention history

Stress has been expressed three ways across CAMS versions. This is one of the most persistent sources of conflict in the record, but it now has a formal resolved decision.

**Convention A — Negative scale (Feb 2025, early canonical scoring):**
Range: −10 to 0. Zero = no stress; more negative = greater demonstrated resilience under pressure.
Source: `instructions.txt`, `CAMS_NodeValue_BondStrength.py`, CAMS-CANONICAL.md.

**Convention B — Positive scale (positive-up, unsigned):**
Higher value = more stress. Used in scorecard-period readings (late 2025), where "Stress = entropy, contradiction, external pressure."
Source: visible shared project history; cams-scorer.tsx context.

**Convention C — Signed bidirectional (±10):**
Coherence and Capacity on [−10, +10]; Abstraction on [0, 10].
Source: `Give me a mathematical formalisation..md` (2025-03-26).

---

**✅ FORMALLY RESOLVED — RD-001 (November 2025):**

> Stress is **always positive** (scale 1–10). Resilience under pressure is captured through system-level dynamics (bond strength, abstraction evolution), **not** via negative stress encoding. This eliminates the thermodynamic inconsistency identified in the October 2025 review and aligns with the physical interpretation of stress as load/pressure.
>
> *All scoring post-November 2025 uses positive Stress only.*

**Implementation:** Scale 1–10 (integer, always positive).
- 1: Minimal — no appreciable pressure or constraint
- 4–6: Elevated — sustained pressure, resources strained but not yet critical
- 7–8: High — severe pressure, crisis conditions, resource exhaustion imminent
- 9–10: Extreme — existential threat, organisational collapse risk

**⚠ RESIDUAL CONFLICT:** CAMS-CANONICAL.md (v1.0-RC1, April 2026) still formally states Stress as scale −10 to 0 (Convention A). This creates a direct conflict between CAMS-CANONICAL.md and the resolved decision RD-001 documented in CAMS-CAN-GEMINI.md. Until CAMS-CANONICAL.md is updated to reflect the positive convention, all scoring must explicitly document which convention is in use. The Gemini file's positive convention is the operationally implemented form as of November 2025.

---

### 2.4 Early component definitions (Darwin appendix, Nov 2025 retrospective)

These explicit product-form definitions were used in the earliest formal phase:

```
C(t) = Cultural_Depth × Shared_Values × [1 − Internal_Conflict]
K(t) = Resource_Control × Skill_Level × Infrastructure_Quality
S(t) = (External_Pressure + Internal_Tension) / Adaptive_Capacity
A(t) = Learning_Rate × Knowledge_Integration × Innovation_Capacity
```

These were not carried forward into the additive scoring regime.

---

## SECTION 3 — NODE VALUE FORMULA HISTORY

### 3.1 Multiplicative phase (Feb 2025, `cams-formalization.md`)

**First formally recorded Node Value:**

```
V(nᵢ, t) = [C(t)ᵢ × P(t)ᵢ × A(t)ᵢ] / max(1, |S(t)ᵢ|)
```

Where P(t) appears to be a Polarisation or Performance variable distinct from current metrics.

**⚠ CONFLICT:** This is a multiplicative formula. All later canonical Node Value formulas are additive. The shift from multiplicative to additive is not explicitly documented.

**Confidence:** HIGH for the formula text; LOW for the transition reason.

---

### 3.2 Early additive phase with generic weight w (Nov 2025 retrospective)

Found in the Darwin appendix:

```
NV = C_i + K_i − S_i + w × A_i
```

The same document also gives the fixed form:
```
NV = C + K − S + 0.5A
```

Worked example:
```
3.8 + 7.2 − 4.5 + 0.5×5.0 = 9.0
```

**⚠ CONFLICT:** Generic weight `w` vs fixed weight `0.5` coexist in the same document. No reason for fixing `w = 0.5` is given here.

**Confidence:** HIGH.

---

### 3.3 Canonical additive formula (Feb 2025 upload, confirmed to present)

**Canonical Node Value:**

```
NV = C + K − S + (0.5 × A)
```

Or in full notation:
```
Vᵢ(t) = Cᵢ + Kᵢ − Sᵢ + (Aᵢ × 0.5)
```

**Python implementation** (`CAMS_NodeValue_BondStrength.py`):
```python
def calculate_node_value(coherence, capacity, stress, abstraction):
    """Calculate the Node Value based on CAMS formula."""
    node_value = coherence + capacity - stress + 0.5 * abstraction
    return round(node_value, 2)
```

**Validation examples:**

| C | K | S | A | NV |
|---|---|---|---|---|
| 8 | 7 | −3 | 6 | 8 + 7 − (−3) + (0.5 × 6) = **21.0** |
| 4 | 5 | 0 | 4 | 4 + 5 − 0 + (0.5 × 4) = **11.0** |
| 9 | 8 | −7 | 8 | 9 + 8 − (−7) + (0.5 × 8) = **28.0** |

**Rationale:** Coherence and Capacity contribute directly; Abstraction contributes at half weight, reflecting its moderating rather than primary metabolic role; Stress is subtracted (under the positive-stress convention, higher S reduces NV; under the negative-stress convention, negative S values thus add to NV).

**Confirmed range (positive stress convention, 32+ datasets):**
- Theoretical floor: 1 + 1 − 10 + (0.5 × 1) = **−7.5**
- Theoretical ceiling: 10 + 10 − 1 + (0.5 × 10) = **28.5**
- Observed empirical range: **3 to 21**
- Typical operating range: **8 to 18**
- Crisis signal: NV drops 2–4 points during Phase 3 transition

**Interpretive note (Mar 2026):** "Abstraction as performance modifier, not equal metabolic addend."

---

### 3.4 Archetypal form (v3.0 / v3.1, Mar 2026)

```
V_i(t) = C_i + K_i − S_i + (A_i × 0.5)
```

Mean node value:
```
V̄(t) = (1/8) × Σᵢ Vᵢ(t)
```

Bond-weighted mean node value (v3.1):
```
β_i = B_i / Σⱼ Bⱼ
Ṽ(t) = Σᵢ βᵢ(t) × Vᵢ(t)
```

This is the current operational form.

---

## SECTION 4 — BOND STRENGTH FORMULA HISTORY

Bond Strength has the most contested formula lineage in the CAMS record. At least four distinct families exist.

### 4.1 Weighted-average family (Feb–Mar 2025)

**Formula:**
```
BS(A,B) = ((Coh_A + Coh_B) × 0.6 + (Abs_A + Abs_B) × 0.4) / (1 + |Stress_Average|)
```

Where `Stress_Average = (|S_A| + |S_B|) / 2`.

**Python implementation:**
```python
def calculate_bond_strength(coh_a, abs_a, stress_a, coh_b, abs_b, stress_b):
    """Calculate Bond Strength based on CAMS formula."""
    stress_avg = (stress_a + stress_b) / 2
    bond_strength = ((0.6 * (coh_a + coh_b)) + (0.4 * (abs_a + abs_b))) / (1 + abs(stress_avg))
    return round(bond_strength, 2)
```

**⚠ CONFLICT — WORKED EXAMPLE DISCREPANCY:**
- Example in source document: `Bond Strength = 10.8`
- Formula recalculation (Node A: C=7, A=5, S=−3; Node B: C=6, A=4, S=−2):
  - Numerator: `(13 × 0.6) + (9 × 0.4) = 7.8 + 3.6 = 11.4`
  - Denominator: `1 + 2.5 = 3.5`
  - Result: `≈ 3.26`, not `10.8`
- The Python code and the formula agree with each other; neither agrees with the worked example. The `10.8` figure is unrecovered.

**Confidence:** HIGH for code and formula; HIGH that the worked example `10.8` is internally inconsistent.

---

### 4.2 Product family / canonical v1.0-RC1 form (CAMS-CANONICAL.md)

**Canonical formula:**

```
BS(A,B) = (C_A × C_B × 0.6) + (A_A × A_B × 0.4) / (1 + S̄_{A,B})
```

Where `S̄_{A,B} = (|S_A| + |S_B|) / 2`.

**Note:** The shift from *sum* form (`(C_A + C_B) × 0.6`) to *product* form (`C_A × C_B × 0.6`) is a substantive change. Products penalise asymmetric pairings more severely than sums.

**⚠ OPEN ITEM (CAMS-CANONICAL.md):** Whether `S̄` uses raw signed values or absolute values in the denominator is not yet frozen. Current canonical assumption: absolute values.

**Diagnostic thresholds:**

| BS value | Interpretation |
|---|---|
| > 7.0 | Strong coupling; robust inter-node function |
| 4.0–7.0 | Moderate coupling; functional but fragile |
| < 3.0 | Bond failure risk; substitution dynamics likely |

**Substitution signal:** When BS(Helm–Shield) > 7.0 simultaneously with BS(Helm–Lore) < 4.0, coercion is substituting for legitimacy — validated precursor to autocratisation (Weimar Germany 1930–33, Turkey 2013–2024).

---

### 4.3 Thermodynamic / exponential family (Nov 2025 GTS EV redraft)

```
B_ij = C_i × C_j × exp(−|K_i − K_j|)
```

**Rationale:** Coupling is strongest between nodes with similar Capacity levels; dissimilar Capacities decay the bond exponentially.

**Stress propagation equation (paired with this bond form):**
```
dS/dt = Σ B_ij × (S_j − S_i)
```

This is a diffusion equation: stress flows down gradients between coupled nodes.

**⚠ CONFLICT:** This is a fundamentally different Bond Strength family. It is not additive, uses Capacity rather than Abstraction as the coupling decay variable, and is embedded in a differential equation framework. It was introduced as part of the `CAMS GTS EV` thermodynamic redraft that explicitly deprecated the prior neural-architecture (`camsnn1`).

---

### 4.4 Similarity / distance family (Mar 2026 conceptual branch)

```
K_eff = K + A/2
dist = Euclidean distance in (C, K_eff, A) space
sim = exp(−dist / sigma)
```

Bond Strength as similarity in effective-capacity space, with sigma set to median pairwise Euclidean distance.

**Code implementation (Mar 2026 audit, `MARKER001_check_report.txt`):**
```
Bond Strength = node degree D_ii from the yearly CAMS coupling matrix
(Coupling matrix built from z-scored [C, K, S, A] within each year)
```

**⚠ CONFLICT:** This family treats Bond Strength as a graph-theoretic edge weight derived from normalised state-vector proximity. It is incompatible with the weighted-average formula families.

---

### 4.5 Bond Strength — exponential stress penalty variant (CAMS-CAN-GEMINI.md, v1.0-RC1)

This is the provisional canonical form as of April 2026 in the Gemini reference:

```
B_ij = [0.6 × (C_i × C_j) + 0.4 × (A_i × A_j)] × stress_penalty

stress_penalty = exp(−λ × (S_i + S_j) / 20)
where λ ≈ 1.0 (calibrated, provisional)
```

**Observed range:** [1.6, 4.3] across datasets.

**Rationale:** The exponential penalty replaces the denominator-based stress adjustment. As combined stress increases, the penalty decays smoothly toward zero rather than producing a hard division. This is the form consistent with the positive-stress convention.

**⚠ CONFLICT:** This differs from the CAMS-CANONICAL.md product form:
`BS(A,B) = (C_A × C_B × 0.6) + (A_A × A_B × 0.4) / (1 + S̄_{A,B})`
The denominator form and the exponential penalty form are not algebraically equivalent and will produce different numerical outputs.

### 4.5a System Bond Density (B̄)

Introduced in CAMS-CAN-GEMINI.md as a formal system-level aggregate:

```
B̄ = (1/28) × Σ B_ij  for all i ≠ j
```

(28 = number of unique pairs in an 8-node system)

**Per-node aggregate:**
```
B_i = (1/7) × Σ B_ij  for all j ≠ i
```

**Phase-linked ranges (observed):**
- Phase 1: B_i > 3.5 for most nodes
- Phase 2: B_i ∈ [2.5, 3.5]
- Phase 3: B_i ∈ [1.5, 2.5]
- Phase 4: B_i < 1.8

**Integrity thresholds (B̄ relative to baseline B̄₀):**
- B̄/B̄₀ > 0.95: Full network integrity
- 0.60–0.95: Network thinning; isolated subsystems emerging
- < 0.70: Severe fragmentation; near-complete network collapse
- Bond loss > 40%: Irreversibility threshold (see Section 6.5)

### 4.6 Bond Strength range conflict

**Architecture document (Book, Mar 27 2026):** States B_ij lies "between zero and one."
**Datasets and v3.0 trigger rules:** Use Bond Strength values of 1.2, 1.4, 1.5, 2+, 3+.
**Resolution:** The [0, 1] constraint appears to apply only when Bond Strength is explicitly normalised (as in the dyad branch). The canonical pairwise BS formula produces values outside this range and is not normalised by default.

---

## SECTION 5 — SYSTEM HEALTH FORMULA HISTORY

System Health H(t) has the widest divergence of any CAMS formula. At least five distinct families are recorded.

### 5.1 Earliest formal H(t) — product-ratio with exponential decay (~2024, preserved Nov 2025)

```
H(t) = N(t) / D(t) × [1 − P(t)] × exp(−S(t) / T)
```

Where:
- N(t): number of functioning nodes
- D(t): dysfunction factor
- P(t): polarisation, scale 0–1
- S(t): aggregate stress level
- T: system resilience constant ("temperature")

**Thresholds:** H(t) < 2.5; H(t) < 2.3; P(t) > 0.7; Abstraction drift > 3.0.

---

### 5.2 Multiplicative-form H(t) (Feb 2025, `cams-formalization.md`)

```
H(t) = [Σᵢ C(t)ᵢ × Σᵢ P(t)ᵢ] / [Σᵢ S(t)ᵢ × V(t)]
```

Where V(t) is a volatility measure and P(t)ᵢ appears to be a performance-level variable.

**Bond Strength evolution linked to this form:**
```
∂B/∂t = γ[H(t) − H₀]B(t)
```

This is a logistic growth equation: Bond Strength grows when H exceeds a baseline, decays below it.

---

### 5.3 Instruction-set aggregate H(t) (Mar 2025, `Civilizational_Analysis_Instruction_Set.docx`)

```
H(t) = (Σ(φᵢ(t) − sᵢ(t))) / (n(1 + Σ B(i,j,t)))
```

**Bond Strength aggregate:**
```
B(t) = (1/n) × Σ B(i,j,t)
```

**Unified Resilience Index:**
```
URI = H(t) × B(t) × τ(t)
```

Where τ(t) is a technological amplification factor.

**Type thresholds:**
- Type I: H > 0.75, B > 0.6, URI > 0.5
- Type II: 0.5 < H ≤ 0.75, B > 0.5
- Type III: 0.2 < H ≤ 0.5, B < 0.5
- Type IV: H ≤ 0.2, B < 0.3

**⚠ CONFLICT:** Third incompatible H(t) family. These type thresholds are also inconsistent with later threshold values.

---

### 5.4 Named-law H(t) with stress decomposition (Mar 2025, `sciencecams.txt`)

```
H(t) = [Σ(wᵢ × Cᵢ × Kᵢ)] / [Σ((S_chronic + 2S_acute)(1 + 0.5√Aᵢ)) × (1−P(t))]
```

Named indices introduced in this phase:
- `Abstraction Inversion Index (AII)`
- `Pragmatic Abstraction Deficit (PAD)`
- `Cross-Domain Coherence (CDC)`

**⚠ CONFLICT:** Placement of (1−P(t)) in the denominator here vs as a multiplier of the full fraction in other formulations.

---

### 5.5 Ratio-form H(t) — primary formalised version (Mar–May 2025)

First fully specified ratio form, appearing in `Give me a mathematical formalisation..md` (2025-03-26), `Elaboration on Primal Roots and CAM.txt` (2025-05-08), Architecture manuscript (Mar 2026):

```
H(t) = [N(t) / D(t)] × (1 − P(t))
```

Where:
```
N(t) = Σᵢ wᵢ(t) × Cᵢ(t) × Kᵢ(t)

D(t) = Σᵢ (S_chronic,i + 2 × S_acute,i) × (1 + 0.5√Aᵢ(t))

P(t) = min(std(Cᵢ × Kᵢ) / (2 × mean(Cᵢ × Kᵢ)), 0.5)
```

Node weights:
```
wᵢ(t) = BS_i(t) × mean(BS_j≠i(t)) / Σₖ BS_k(t)
```

**Stress decomposition:**
```
S_total = √(S_chronic² + S_acute²)
```

**Critical thresholds:**
- H(t) < 2.5 → risk of system collapse increases
- H(t) < 2.3 → system reorganisation is likely
- |ΔH/Δt| > 0.5 → acute instability

**Validated against:** Western Roman Empire (retrospective), Australia 1900–2024, China 1900–2025, Denmark 1752–2025.

---

### 5.6 CAMS-CANONICAL.md H(t) (Apr 2026, current)

```
H(t) = [C̄(t) × K̄(t) − |S̄(t)|] / [A_sys(t) × (1 + 0.5√A_sys(t))]
```

Where C̄, K̄, S̄ = mean Coherence, Capacity, Stress across all 8 nodes; A_sys = mean Abstraction across all 8 nodes.

**Thresholds:**
| H value | System status |
|---|---|
| > 4.0 | Healthy; robust adaptive capacity |
| 2.5–4.0 | Caution zone; monitoring required |
| 2.3–2.5 | Reorganisation threshold; high collapse risk |
| < 2.3 | Basin transition imminent; critical vulnerability |

**The Abstraction–Resilience Paradox:** The denominator encodes the finding that rising institutional sophistication amplifies effective stress even as it expands capability. The most sophisticated societies are simultaneously the most capable and the most fragile.

---

### 5.7 Grand System Metric — simplified scalar (Mar 31 2026)

```
Grand System Metric = (Average Node Value × Average Bond Strength) / 10
```

A compressed aggregate scalar used for quick diagnostic readings. Not a replacement for H(t).

---

### 5.8 Simplified health forms (various)

```
H = (C × K − S) / A
```

```
H = (C × K − S) / (A × (1 + 0.5√A))
```

```
Health = (Coherence × Capacity) / (Stress × (1 + 0.5√Abstraction))
```

These appear in secondary synthesis documents and applied readings. The last form is the "primal-band" simplified health used when working with pre-state societies.

---

## SECTION 6 — DERIVED OPERATORS AND INDICES

### 6.1 Node-level operators

| Operator | Formula | Origin |
|---|---|---|
| Social Cognition (early) | SC_i = A_i / C_i | Scorecard period, late 2025 |
| National Affect | NA_i = K_i − S_i | Scorecard period, late 2025 |
| Abstraction Ratio (v3.0/v3.1) | G_i(t) = A_i / (C_i + ε) | Mar 2026 |
| Affect (v3.0/v3.1) | α_i(t) = K_i − S_i | Mar 2026 |
| Cognition (soul-reading) | COGNITION = A_i × C_i | v3.0 Appendix A |
| Affect (soul-reading) | AFFECT = K_i − S_i | v3.0 Appendix A |
| Cognition (CAPE branch) | cognition = (C × A) / (1 + S) | mythos.txt, Mar 14 2026 |

**⚠ CONFLICT:** Three incompatible "cognition" operators across branches: A/C, A×C, and (C×A)/(1+S). They are not equivalent and should not be used interchangeably.

---

### 6.2 Thermodynamic / mode operators (Jan 2026 dyad branch)

From `dyad.txt`:

```
M(t) = (1/3) × Σ [S_n(t) + α × ΔK_n(t)]
ΔK_n = max(0, K*_n − K_n)

Y(t) = (1/3) × Σ [β×C_n(t) + (1−β)×A_n(t) − γ×S_n(t)]
Y_capacity(t) = (1/3) × Σ [β×C_n(t) + (1−β)×A_n(t)]
Y_drag(t) = (1/3) × Σ [γ×S_n(t)]

D(t) = M(t) − δ×Y(t)
R(t) = D(t) / (ε + B_norm(t))
```

Field weights: ALPHA = 0.5, BETA = 0.5, GAMMA = 0.3, DELTA = 1.0, EPSILON = 0.05

Mode node groups:
- METABOLIC_NODES = ['Hands', 'Flow', 'Shield']
- MYTH_NODES = ['Lore', 'Archive', 'Stewards']

Regime classifier:
```python
def classify_regime(D, B_norm, R, D_threshold=2.0, B_threshold=0.3, R_threshold=20):
```

Crisis condition: `R > 20`

**Note:** Helm is omitted from both node groups in this branch — an acknowledged gap.

---

### 6.3 System-level aggregate operators (Mar 2026 v3.0/v3.1)

```
Φ = Σ(C × A)         [deliberative capacity]
F = Σ(K × S)         [reactive force]
Ω = Φ − F            [net system mode]
T = F / Φ            [tension ratio]
S_X defined by Φ = F [equilibrium crossing point]
```

Interpretation:
- Ω > 0: deliberative-dominant
- Ω < 0: reactive-dominant
- T > 1.5: severe systemic strain

---

### 6.4 Network Synchronisation (CAMS-CANONICAL.md)

```
NS(t) = 1 − σ(Cᵢ) / μ(Cᵢ)
```

Where σ and μ are the standard deviation and mean of Coherence scores across all 8 nodes.

**Thresholds by system type:**

| System type | NS threshold | Stability mechanism |
|---|---|---|
| Federal / Distributed (Australia, Denmark) | > 0.7 for stability | Consensus and inter-institutional coordination |
| Centralised Command (China, Soviet) | 0.5–0.6 is normal | Authority substitutes for consensus |

---

### 6.5 Coherence Asymmetry and Critical Transition Risk

```
CA(t) = σ(Cᵢ × Kᵢ) / μ(Cᵢ × Kᵢ)
```

CA > 0.4: High fracture risk; power imbalance between nodes is destabilising.

```
Risk(t) = CA(t) × (1 / H(t))
```

High CA combined with low H is the canonical early-warning signature of impending phase transition.

---

### 6.6 Stress dispersion and instability operators (v3.0/v3.1)

```
Δσᵢ(t) = Sᵢ(t) − Sᵢ(t−1)
O(t) = StdDev({Δσᵢ(t) : i ∈ N})
R(t) = λ_max(t) / λ₂(t)       [spectral ratio]
J(t) = O(t) × |R(t)|
```

High J indicates dangerous shear: high stress dispersion relative to coupling.

**Additional threshold rules:**
- Lifetime mean S > 6: chronic stress danger threshold
- DR = A/C; optimal band ~0.85–1.05

---

### 6.7 Multiscale Viability Operator / D-functional (Dec 25 2025)

Introduced because "formal theorems assume clean structure; real datasets reveal noise, reversals, and scale-dependent behaviour."

```
SNR_i^Δt(t) = |trend_i(t)| / σ_i^Δt(t)
A(SNR) = SNR² / (1 + SNR²)
w_i^Δt(t) = wᵢ × A(SNR_i^Δt(t))
D^Δt(t) = Σᵢ w_i^Δt(t) × penalty_i(t)

ξ(t) = (A(t) + ε) / (C̃(t) + ε)
τ(t) = K̃(t) / (S̃(t) + ε)
```

Scale: timescales Δt ∈ {1, 2, 5, 10, 20, 50, 100} years

**⚠ NOTE:** This phase relabels components as "Cognitive C, Affective K, Metabolic S, Structural B" — overloading earlier CAMS symbols. Context must be confirmed when reading this branch.

---

### 6.8 Kuramoto / synchronisation bridge (v3.1, Mar 23 2026)

```
K_c = 2σ_Ω / (π × g(0))
μᵢ(t) = (Cᵢ + Aᵢ) / 2
ω_μ(t) = StdDev({μᵢ(t) − μᵢ(t−1) : i ∈ N})
κ(t) = ω_μ(t) / B(t)
```

**⚠ CONFLICT — INTERNAL TO v3.1:** The same document also states:

```
κ(t) = B(t) / ω(t)
```

These two forms are inverses of each other. One or the other must be chosen. This conflict is unresolved in v3.1.

---

### 6.9 Reynolds analogue / societal viscosity (Mar 21 2026)

```
Re_S,i(t) = Cᵢ(t) × Kᵢ(t) × Aᵢ(t) / Sᵢ(t)
μ_social ≈ f(Sᵢ⁻¹, B(t))
ξ(t) = σ_Ω / B    [criticality synthesis]
```

**⚠ CONFLICT with v3.0 summary:** v3.0 states `ξ = B/σO`; the fluid-dynamics bridge states `ξ = σ_Ω/B`. Inversion unresolved.

---

### 6.10 Overdamping / late abstraction operators

```
Σᵢ(t) = max(0, Aᵢ(t) − A_thresh) × max(0, Sᵢ(t) − S_thresh)
```

Where A_thresh = 7, S_thresh = 6.

This detects the condition where high Abstraction coincides with high Stress — the "Late Abstraction Collapse" signature.

---

## SECTION 6b — THERMODYNAMIC FRAMEWORK (formalised v0.9, Jan 2026)

This framework was initiated in v0.7 (Sep 2025) and completed in v0.9 (Jan 2026). It treats societies as dissipative thermodynamic systems and introduces three scalar system-level diagnostics and a unified Decay Index.

---

### 6b.1 Affective Tone — τ (Surplus Capacity Ratio)

**Definition:** The ratio of mean system Capacity to mean Stress.

```
τ = mean(K) / mean(S)
```

**Interpretation:**
- τ > 1.5: Healthy surplus; system can absorb perturbations
- 1.2–1.5: Strained but viable; reserves being drawn
- τ ≈ 1.0: Critical stress; minimal reserves
- τ < 0.7: Catastrophic; system at metabolic minimum
- τ < 0.6: Historical collapse point (observed: Rome, France 1789)

**Physics:** As τ declines, the system loses capacity to respond to shocks.

---

### 6b.2 Cognitive Style Ratio — ε (Abstraction-to-Reality Coupling)

**Definition:** The ratio of mean system Abstraction to Affective Tone.

```
ε = mean(A) / τ
```

**Interpretation:**
- 0.9–1.1: Balanced cognition; abstraction matched to material capacity
- ε > 1.15: Cognitive overreach; abstractions exceed material grounding
- ε > 1.3: Critical overreach; "brittle sophistication" developing
- ε > 1.4 or ε < 1.0: Hollow or collapsed cognition

**Physics:** When τ declines but Abstraction continues to rise, the system develops brittle sophistication — elaborate institutions that fail under stress. High A + Low C = "Brittle Sophistication" signature (Phase 2 warning).

---

### 6b.3 Mode Ratio — R (Deliberative vs. Reactive Balance)

**Definition:** The ratio of reactive to deliberative modes in system operation.

```
R = reactive_mode / deliberative_mode
```

**Interpretation:**
- R < 1: Deliberative dominant; system planning ahead
- 1–3: Mixed modes; planning maintained under some crisis response
- R > 3–4: Reactive dominant; long-horizon planning capacity lost
- R > 5: Nearly pure reaction; bifurcation imminent

**Physics:** High Stress and declining Capacity force reactive mode. Once R > 3, bifurcation approaches.

---

### 6b.4 Decay Index — D(t)

**Definition:** Unified measure of system decay and fragmentation risk.

```
D(t) = w₁·O + w₂·T + w₃·L + w₄·max(0, R − 1)
```

**Components:**

```
[O] Overreach Penalty:
    O = max(0, ε − 1)
    Interpretation: Cognitive gap severity

[T] Strain Penalty:
    T = 1 − τ = 1 − (mean(K) / mean(S))
    Interpretation: Metabolic deficit severity

[L] Connectivity Loss:
    L = 1 − (B̄ / B̄₀)
    where B̄₀ = baseline bond density (reference period)
    Interpretation: Network fragmentation (0 = full integrity; 1 = total collapse)

[Mode Imbalance]:
    max(0, R − 1) = penalty for reactive overweight
```

**Provisional weights (calibration ongoing):**
```
w₁ = 1.0 (Overreach)
w₂ = 1.0 (Strain)
w₃ = 1.0 (Connectivity loss)
w₄ = 0.5 (Mode imbalance)
```

**Normalised output:** D(t) ∈ [0, 10]

**Phase thresholds:**
- D ≤ 1: Healthy integration (Phase 1)
- 2 < D < 4: Brittle sophistication (Phase 2)
- 4 ≤ D ≤ 6: Critical slowing; bifurcation onset (Phase 3)
- D ≥ 6: **Bifurcation threshold crossed; reversibility questionable**
- D > 7: Fragmentation imminent (Phase 4)

---

### 6b.5 Dynamical Evolution Equations

#### ε evolution (Abstraction growth under strain):

```
ε_{t+1} = ε_t + a·(1 − τ_t)·(1 − B̄_t)
```

Parameters: a = learning/adaptation rate (typically 0.01–0.1; not yet calibrated)

**Physics:** When material capacity declines (τ↓) and nodes are isolated (B̄↓), nodes adopt more abstract internal solutions. This increases ε as a compensatory strategy, ultimately making the system more brittle.

#### τ evolution (Surplus capacity decay):

```
τ_{t+1} = τ_t · [1 − ξ·(ε_t − 1)]
```

Parameters: ξ = erosion rate (typically 0.01–0.1; not yet calibrated)

**Physics:** Cognitive overreach erodes capacity. The more the system over-abstracts relative to material grounding, the faster it depletes reserves through maladapted strategies.

**Nonlinear feedback loop:** As τ declines → ε rises (via ε equation) → τ declines faster (via τ equation). This is the mathematical form of the Abstraction–Resilience Paradox.

---

### 6b.6 Causal Pathways to Collapse

**Path 1 — Stress → Surplus Depletion → Decay:**
```
S ↑ → τ ↓ → D ↑ → R ↑ (reactive mode)
```
Rising stress exhausts capacity. As reserves deplete, deliberative mode is lost and decay accelerates.

**Path 2 — Isolation → Cognitive Overreach → Decay:**
```
B̄ ↓ → ε ↑ → τ ↓ (via overreach erosion) → D ↑
```
Network fragmentation forces nodes into isolation. Isolated nodes develop internal abstract solutions that erode actual capacity.

**Path 3 — Network Collapse Feedback (Nonlinear Amplification):**
```
D ↑ → (ε_{t+1}, τ_{t+1} feedback) → D ↑↑
```
Once thresholds are crossed, decay self-amplifies. Small shocks produce large responses.

---

### 6b.7 Recovery Impossibility Boundary

**Irreversibility threshold:**
```
IF: (τ < 0.6) AND (Bond loss > 40%)
THEN: D > 6 — recovery probability → 0 even with energy influx
```

**Historical validations:**
- Rome (late 4th century): τ < 0.6; >40% bond loss
- France (1789): τ ≈ 0.55; structural fragmentation
- XXU (terminal phase): τ < 0.5; complete network dissolution

---

## SECTION 6c — PHASE SPACE (formalised v0.9, Jan 2026)

Four phases describe the complete lifecycle of a society's systemic health. Phase assignment is determined primarily by D and confirmed by τ, ε, and B̄/B̄₀.

### Phase 1: High Integration (Stable Equilibrium)

```
Conditions: ε ∈ [0.9, 1.1] AND τ > 1.4 AND B̄/B̄₀ > 0.95
D ≤ 1
```

Characteristics: Coherent planning; strong surplus capacity; intact network integration; low bifurcation risk; system absorbs perturbations without phase change.

### Phase 2: Brittle Sophistication (Unstable Equilibrium)

```
Conditions: ε > 1.15 (rising) AND τ ∈ [1.2, 1.4] (declining) AND B̄: −5% to −15% loss
2 < D < 4
```

Characteristics: Cognitive overreach begins; institutional elaboration increases; surplus capacity declining but still present; network beginning to thin.
**Early warning signature:** D > 2.

**Risk profile:** Stable but fragile. Margin for error shrinks.

### Phase 3: Critical Slowing (Bifurcation Onset)

```
Conditions: ε > 1.3 (peak) AND τ ≈ 1.0 AND R rising sharply AND recovery time variance increasing
4 < D < 6
```

Characteristics: Approaching metabolic minimum; cognitive overreach at peak; recovery from shocks takes longer; reactive mode dominant.
**Bifurcation point:** D ≥ 6 (reversibility lost).

**Risk profile:** Critical slowing — shocks produce disproportionate responses. Recovery trajectory unpredictable.

### Phase 4: Fragmentation / Dissociative Equilibrium (Breakdown)

```
Conditions: [ε > 1.4 (hollow) OR ε < 1.0 (collapsed)] AND τ < 0.7 AND B̄/B̄₀ < 0.7
D > 7
```

Characteristics: Network disintegration; nodes operate independently; abstraction either hollow or collapsed; severe metabolic constraint.
**Irreversibility:** >40% bond loss means recovery is not possible.

**Historical signatures:** Rome (late 4th century), France (1789–1795), XXU (terminal phase).

---

## SECTION 7 — TIMESCALE PARTITIONS

### 7.1 v3.0 partition (Mar 13 2026)

```
Slow loop: Archive, Lore, Helm, Stewards
Fast loop: Craft, Flow, Hands, Shield
```

### 7.2 v3.1 partition (Mar 23 2026) — CANONICAL

```
Slow loop: Archive, Lore, Stewards
Fast loop: Helm, Shield, Flow, Hands, Craft
```

**Reason for change (explicitly stated):** "Abandoning theoretical symmetry for structural reality." Helm and Shield were reclassified as reactive and much faster than previously assumed.

### 7.3 Dyad branch partition (Jan 2026)

```
Metabolic: Hands, Flow, Shield
Myth:      Lore, Archive, Stewards
[Helm: unassigned in this branch]
```

---

## SECTION 8 — NAMED LAWS, RULES, AND SIGNATURES

### 8.1 Early law-set (Darwin appendix, Nov 2025)

- Scale-Dependent Governance Law: `|dT/dt − dS/dt| < K`
- Meta-Cognition Threshold Law: `M(t) > C_int(t) + C_ext(t)`
- Coherence–Efficiency Tradeoff
- Information Integration Limit
- Abstraction Evolution Law
- Technology–Society Coevolution Law
- Modularity Principle
- Complexity Management Law
- Constraint Satisfaction Dynamics
- Memory–Innovation Balance
- Stress Distribution Law
- Fractal Governance Principle
- Extended Phenotype Principle

---

### 8.2 Mid-period named indices (sciencecams.txt, Mar 2025)

- Abstraction Inversion / Abstraction Inversion Index (AII)
- Pragmatic Abstraction Deficit (PAD)
- Cross-Domain Coherence (CDC)

---

### 8.3 v3.0 / v3.1 canonical signatures and theorems

**Crisis signatures with exact trigger conditions:**

| Name | Trigger condition |
|---|---|
| Executive Decoupling | `V_Helm < 6 AND B_Helm,Archive < 1.4 AND B_Helm,Hands < 1.2` |
| Praetorian Condition | `V_Shield > 12 AND V_Shield > 1.5 × V̄ AND B_Shield,Flow < 1.5` |
| Mythic-Material Decoupling | `|V_Lore − V_Hands| > 8 AND sign(V_Lore) ≠ sign(V_Hands)` |
| Flow Collapse | `V_Flow < 0 AND ∂V_Flow/∂t < −2/year` |
| Thermodynamic Freeze | `σ_V < 1.5 AND V̄ < 8` |
| Late Abstraction Collapse | `A_Helm > 8 AND S_Helm > 7 AND dA/dt < 0` |

**Theoretical corollaries:**
- Coordination Phase Transition (CPT) / Coordination Phase Transition Theorem
- Non-Agentic Crisis
- Coupling Primacy
- Sisu operator (resilience under extreme stress)

**Substitution signal (from Bond Strength diagnostics):**
- When BS(Helm–Shield) > 7.0 simultaneously with BS(Helm–Lore) < 4.0: coercion substituting for legitimacy

---

### 8.4 CAPE archetype coordinates (mythos.txt, Mar 14 2026)

```
X = (Flow + Stewards) − (Lore + Archive)    [material vs mythic axis]
Y = (Helm + Shield) − (Hands + Craft)       [authority vs labour axis]
```

Named archetypes in the resulting phase space:
`Temple, Marketplace, Workshop, Commons, Throne, Fortress, Library, Guild`

Instability rule: `SCISSORS: declining cognition + authority drift`

---

### 8.5 Soul-reading diagnostics (v3.0 Appendix A)

Node-level archetypes mapped from COGNITION (A×C) and AFFECT (K−S):

- Generative
- Dissociative
- Terminal
- Late hypertrophy
- Dissociative hypertrophy
- Generative coherence
- Bifurcation crisis
- Thermodynamic death

---

## SECTION 9 — CONFLICTS REGISTER

This section aggregates all explicitly flagged conflicts from across the record.

| # | Conflict | Source A | Source B | Resolution |
|---|---|---|---|---|
| 1 | Bond Strength worked example | Written result: 10.8 | Formula + code implies: ~3.26 | **Unresolved — 10.8 appears erroneous** |
| 2 | Stress scale convention | −10 to 0 (CAMS-CANONICAL.md) | 1–10 positive (CAMS-CAN-GEMINI.md RD-001) | **⚠ LIVE CONFLICT: RD-001 formally resolves to positive 1–10 (Nov 2025), but CAMS-CANONICAL.md still states −10 to 0. Positive convention is operationally implemented; CAMS-CANONICAL.md requires update.** |
| 3 | Stress prose vs scale | "Positive stress penalises" | Scale was −10 to 0 | **Resolved by RD-001: positive convention adopted Nov 2025** |
| 4 | Fourth metric name | "Impact" (early) | "Abstraction" (standard) | **Abstraction is canonical** |
| 5 | Node Value form | Multiplicative (Feb 2025 formal) | Additive (all subsequent) | **Additive is canonical** |
| 6 | Node Value Abstraction weight | Generic w | Fixed 0.5 | **0.5 is canonical** |
| 7 | Bond Strength formula family | Weighted sum / product form (CAMS-CANONICAL.md) | Exponential stress penalty (CAMS-CAN-GEMINI.md) | **⚠ LIVE CONFLICT: Two current canonical documents use different Bond Strength families. Not yet resolved.** |
| 8 | Bond Strength range | B_ij ∈ [0,1] (architecture doc) | Values > 1 in datasets and trigger rules | **[0,1] applies only to normalised variants** |
| 9 | System Health form | ≥ 5 distinct H(t) formulas | — | **CAMS-CANONICAL.md ratio-form governs** |
| 10 | Cognition operator | A/C | A×C | (C×A)/(1+S) | **Context-dependent; soul-reading uses A×C** |
| 11 | κ(t) criticality direction | B/ω | ω/B | **Internal v3.1 conflict; unresolved** |
| 12 | ξ(t) direction | σ_Ω/B (fluid bridge) | B/σO (v3.0 summary) | **Unresolved** |
| 13 | Timescale partition | Helm in slow loop (v3.0) | Helm in fast loop (v3.1) | **v3.1 is current: Helm in fast loop** |
| 14 | System Health P(t) placement | In denominator (sciencecams.txt) | Multiplies full fraction (ratio-form) | **Ratio-form is canonical** |
| 15 | Nine-node vs eight-node | Nine nodes (secondary synthesis) | Eight nodes (all primary documents) | **Eight nodes is canonical** |

---

## SECTION 10 — ITEMS NOT RECOVERED

The following formula names or types were explicitly requested in prior audit sessions but were not found in recoverable source files:

- `ARG` (formula family)
- `EFI` (formula family)
- `ESD` (formula family)
- `URI` (only the symbol was found; no closed-form derivation recovered)
- `SPE` (formula family)
- Dirichlet-form bond equation
- Fully closed Bond Strength equation using `dist`, `sigma`, and `K_eff` simultaneously
- Complete derivation of the Peloponnesian ratio-form `H(t) = N(t)/D(t) × (1−P(t))` from first principles
- Exact wording of: Abstraction–Stress Inversion, Tipping Point Law, Coherence-Capacity Trade-Off (as named laws)

These items may exist in inaccessible project files (separate ChatGPT projects, early Gemini Gems, local files not uploaded). This record will be updated when source material becomes available.

---

## SECTION 11 — CANONICAL STATE (AS OF CAMS-CANONICAL.md v1.0-RC1)

For operational use, the following are the current locked formulas:

**Node Value:**
```
NV = C + K − S + (0.5 × A)
```

**Bond Strength:**
```
BS(A,B) = (C_A × C_B × 0.6) + (A_A × A_B × 0.4) / (1 + S̄_{A,B})
where S̄ = (|S_A| + |S_B|) / 2
```
*(Bond Strength denominator convention pending final confirmation before v1.0 freeze.)*

**System Health:**
```
H(t) = [C̄(t) × K̄(t) − |S̄(t)|] / [A_sys(t) × (1 + 0.5√A_sys(t))]
```

**Network Synchronisation:**
```
NS(t) = 1 − σ(Cᵢ) / μ(Cᵢ)
```

**Coherence Asymmetry:**
```
CA(t) = σ(Cᵢ × Kᵢ) / μ(Cᵢ × Kᵢ)
```

**Critical Transition Risk:**
```
Risk(t) = CA(t) × (1 / H(t))
```

**Node set:** Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow

**Metrics and scales:**
- Coherence: 1–10
- Capacity: 1–10
- Stress: −10 to 0 (negative; justified by demonstrated resilience)
- Abstraction: 1–10

---

## SECTION 12 — DECISION REGISTER

### Open Decisions

**OD-001: Final Bond Strength Equation**
Status: Under final confirmation. Options: (1) Finalise exponential form `B_ij = [0.6×(C_i×C_j) + 0.4×(A_i×A_j)] × exp(−λ·(S_i+S_j)/20)`; (2) Retain denominator form from CAMS-CANONICAL.md; (3) Explore Dirichlet or other normalisation forms. Decision required before v1.0-final. Owner: Kari McKern.

**OD-002: External Environment Node Status**
Status: Currently "contextual only." Options: (1) Keep as exogenous context layer; (2) Elevate to Node 9 in v1.1 (requires formula extensions); (3) Create separate Environment Metrics subsystem. Decision required before v1.1 planning. Owner: Kari McKern.

**OD-003: Confidence Score Requirement**
Status: Recommended but optional. Options: (1) Mandatory in all records from v1.0-final; (2) Keep optional; (3) Create tiered framework (High / Medium / Low / Very Low). Decision required before v1.0-final. Owner: Kari McKern.

**OD-004: Ontology Standardisation**
Status: Not yet formalised. Options: (1) Create separate CAMS-Ontology.md with node mapping examples for 50+ societies; (2) Embed in main reference; (3) Maintain separate examples folder. Decision required before v1.0-final. Owner: Kari McKern.

**OD-005: κ(t) direction conflict**
Status: Unresolved. v3.1 contains both B/ω and ω/B as definitions of the same operator. Requires explicit selection and deletion of incorrect form.

**OD-006: CAMS-CANONICAL.md Stress convention update**
Status: Urgent. CAMS-CANONICAL.md still states Stress scale as −10 to 0, conflicting with the formally resolved positive convention (RD-001, Nov 2025). CAMS-CANONICAL.md requires updating to reflect the resolved decision.

### Resolved Decisions

**RD-001: Stress Metric Sign Convention (November 2025)**
Decision: Stress is always positive (1–10 range). Resilience is captured through system-level dynamics (bond strength, abstraction evolution), not via negative encoding. Rationale: eliminates thermodynamic inconsistency; aligns with physical interpretation of stress as load/pressure. Implementation: all scoring post-November 2025 uses positive Stress only.

**RD-002: Eight-Node Canonical Model (July 2025)**
Decision: Adopt eight functional nodes as canonical, superseding the earlier ten-node variant. Nodes: Executive, Army, Knowledge, Property, Trades, Labor, Memory, Commerce. Rationale: parsimony and functional completeness across 32+ societies.

---

## SECTION 13 — VERSION HISTORY

This section records the formal version history as documented in CAMS-CAN-GEMINI.md. It supersedes the informal dating in the Appendix lineage table where they conflict.

**v0.1-draft — September 2024**
Framework origin; foundational definitions established. Node architecture proposed (eight primary nodes). Four canonical metrics defined. Node Value formula introduced. Initial scoring on 8 test societies.

**v0.5 — July 2025**
State space formalisation begun. Eight-node canonical architecture confirmed (RD-002, superseding earlier ten-node variant).

**v0.6 — August 2025**
CAMS-CAN applications developed for corporate foresight: Boeing (supply chain fragmentation), GM (labour ecosystem stress modelling), BYD (capacity planning under geopolitical constraint), Tesla (supply chain and labour cohesion analysis).

**v0.7 — September 2025**
Thermodynamic formalisation initiated. τ (Affective Tone), ε (Cognitive Style Ratio), and R (Mode Ratio) introduced. CAMS-CAN terminology standardised (eliminated inconsistencies between "Helm/Executive" dual naming, etc.).

**v0.8 — November 2025**
Critical correction: Stress metric sign convention clarified (RD-001). Stress is always positive (1–10). Resilience captured in system dynamics, not negative stress encoding. Fixes thermodynamic inconsistency identified in October 2025 review.

**v0.9 — January 2026**
Thermodynamic formalisation completed. Critical Slowing and Bifurcation thresholds documented. 32+ society validation dataset completed. Michael Hudson debt-militarisation analysis completed.

**v1.0-RC1 — 8 April 2026**
Final refinement before v1.0 release. Confirmed eight-node canonical architecture. Confirmed NV = C + K − S + 0.5A. Finalised thermodynamic formalisation (τ, ε, R). Finalised Decay Index (D) with phase thresholds. Confirmed dynamical equations for τ and ε evolution. Marked Bond Strength algebra for final confirmation (denominator pending). Established irreversibility boundary: D ≥ 6, Bond Loss > 40%. Documented all observed numerical thresholds from 32+ dataset validation.

**Pending before v1.0-final:**
- Final confirmation of Bond Strength equation
- Empirical recalibration of Decay Index weights
- Decision on External Environment node status
- Finalisation of confidence score requirements
- Update CAMS-CANONICAL.md to reflect positive Stress convention (RD-001)

---

## APPENDIX A — FORMULA LINEAGE SUMMARY TABLE

| Period | Source | Formula / Feature | Status |
|---|---|---|---|
| Pre-Sep 2024 | Oral/pre-formal | Descriptive node set (Leadership, Warrior Class…) | Superseded |
| Sep–Nov 2024 | Project memory | Metrics: Coherence, Capacity, Stress, Impact | Deprecated (Impact → Abstraction) |
| Sep–Nov 2024 | Project memory | 8-node set with Storekeepers | Superseded |
| Feb 2025 | cams-formalization.md | Multiplicative Node Value | Superseded |
| Feb 2025 | cams-scorer.tsx | Additive NV; 8-node classical set in code | Superseded (renaming) |
| Feb 2025 | CAMS_NodeValue_BondStrength.py | NV = C+K−S+0.5A; BS weighted-sum form | NV canonical; BS superseded |
| Mar 2025 | sciencecams.txt | H(t) with AII/PAD; (1−P) in denominator | Superseded |
| Mar 2025 | Give me a mathematical formalisation..md | Ratio H(t); stress decomposition | Partially canonical |
| Mar 2025 | Civilizational_Analysis_Instruction_Set.docx | H via instruction-set aggregate | Superseded |
| May 2025 | CAMS-CAN Framework synthesis | Nine-node aberration; simplified H | Superseded (nine-node deprecated) |
| May 2025 | Primal Roots elaboration | Primal-band simplified H | Research use only |
| Nov 2025 | Darwin appendix (retrospective) | Early named laws; component formulas | Historical reference |
| Nov 2025 | GTS EV thermodynamic redraft | Exponential bond B_ij; ODE system | Specialist branch, not canonical for scoring |
| Dec 2025 | MVO / D-functional note | Multiscale viability operator | Research branch |
| Jan 2026 | dyad.txt | Thermodynamic dyad M(t)/Y(t)/D(t)/R(t) | Specialist branch |
| Mar 2026 | formulation3.0.txt | v3.0 formalisation; 4–4 loop split; soul-reading | Superseded in timescale partition |
| Mar 2026 | mythos.txt (CAPE) | Archetypal node names; archetype coordinates | Canonical (renaming) |
| Mar 2026 | v3.1 Pasted markdown.md | 3–5 loop split; κ conflict | Current loop partition canonical |
| Mar 2026 | Architecture book doc | H(t) ratio-form fully specified; B_ij [0,1] statement | Mostly canonical |
| Mar 2026 | MARKER001 audit | Code confirms NV; graph-based BS | Corroborative |
| Mar 2026 | Pasted text / fluid bridge | Reynolds analogue; ξ conflict | Research branch |
| Apr 2026 | CAMS-CANONICAL.md v1.0-RC1 | Full canonical specification | **CURRENT CANONICAL** |
| Apr 2026 | CAMS-CAN-GEMINI.md v1.0-RC1 | Parallel canonical reference; positive Stress convention; thermodynamic τ/ε/R/D framework; phase space; exponential Bond Strength penalty | **Operationally implemented; conflicts with CAMS-CANONICAL.md on Stress convention and Bond Strength form** |

---

*This document is a living audit. When new source material becomes available — particularly from early ARG/EFI/ESD-era files, inaccessible ChatGPT projects, or Gemini Gem variants — the relevant sections should be updated and conflicts re-evaluated. The canonical standard remains CAMS-CANONICAL.md at all times.*

**© Kari Freyr McKern — All rights reserved**
