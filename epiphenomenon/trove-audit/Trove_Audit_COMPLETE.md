# Measurement-Validity Audit — Complete

**Completed 26 July 2026.** 420 items, ten node-periods, six newspapers, blind-coded on heading + snippet.
Batch A (252 items: Shield, Hands, Stewards) under `PROTOCOL_Measurement_Audit.md`.
Batch B (168 items: Helm, Lore) under `AMENDMENT_02_Audit_Helm_Lore.md`, frozen before sampling.

---

## 1. Precision, all ten node-periods

| Node | Year | **Precision** | 95% CI | concept-match | ROUT | NONE | foreign | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Shield** | **1916** | **78.6%** | 64.1–88.3 | 81.0% | 7.1% | 9.5% | 52.4% | **ADEQUATE** |
| Shield | 1914 | 50.0% | 35.5–64.5 | 50.0% | 35.7% | 28.6% | 38.1% | fail |
| Helm | 1916 | 33.3% | 21.0–48.4 | 52.4% | 26.2% | 14.3% | 33.3% | fail |
| Helm | 1914 | 31.0% | 19.1–46.0 | 42.9% | 26.2% | 19.0% | 35.7% | fail |
| Hands | 1893 | 19.0% | 10.0–33.3 | 19.0% | 45.2% | 38.1% | 11.9% | fail |
| Lore | 1916 | 11.9% | 5.2–25.0 | 11.9% | 38.1% | 33.3% | 23.8% | fail |
| Lore | 1914 | 9.5% | 3.8–22.1 | 9.5% | 38.1% | 33.3% | 28.6% | fail |
| Hands | 1888 | 7.1% | 2.5–19.0 | 7.1% | 61.9% | 71.4% | 4.8% | fail |
| Stewards | 1888 | 7.1% | 2.5–19.0 | 21.4% | 57.1% | 47.6% | 11.9% | fail |
| Stewards | 1893 | 7.1% | 2.5–19.0 | 9.5% | 81.0% | 66.7% | 7.1% | fail |

**Pooled by node:** Stewards 7.1% · Lore 10.7% · Hands 13.1% · Helm 32.1% · **Shield 64.3%**.

**Nine of ten node-periods fail. One dictionary out of five works, in one year.**

### Pre-declared expectations, scored

Amendment 02 recorded two predictions before sampling:

- *"Helm is expected to perform better than Hands and Stewards but worse than Shield."* — **correct** (32.1%, between 13.1% and 64.3%).
- *"Lore is expected to perform worst of the five."* — **wrong.** Lore is 10.7%; Stewards is 7.1%. Recorded as a failed expectation.

---

## 2. Cross-node confusion

Retrieving node (rows) × independently coded concept (columns), row-normalised, all 420 items:

| Retrieving ↓ | SHIELD | HELM | LORE | HANDS | STEWARDS | NONE |
|---|---|---|---|---|---|---|
| **Shield** | **65.5** | 7.1 | 1.2 | 2.4 | 4.8 | 19.0 |
| **Helm** | 21.4 | **47.6** | 3.6 | 6.0 | 4.8 | 16.7 |
| **Lore** | 20.2 | 22.6 | **10.7** | 8.3 | 4.8 | 33.3 |
| **Hands** | 1.2 | 20.2 | 1.2 | **13.1** | 9.5 | 54.8 |
| **Stewards** | 2.4 | 20.2 | 0.0 | 4.8 | **15.5** | 57.1 |

Three structural facts:

**Only Shield's diagonal dominates.** For Helm the diagonal (47.6) is the largest cell but a fifth of its retrievals are Shield-domain war reporting. For Lore, Hands and Stewards the modal cell is not the diagonal at all — it is `NONE` for Hands and Stewards, and Lore's diagonal (10.7) is *smaller* than its leakage into Shield (20.2) and Helm (22.6).

**Leakage into Helm is systematic and one-directional.** Every non-Helm dictionary sends ~20% of its retrievals to the Helm domain (Shield 7.1 is the exception). Helm sends only 3.6% to Lore and 4.8–6.0% to Hands and Stewards. Governmental language is a sink, not a source. Newspapers of this period report institutional activity mainly as government activity, so any dictionary with a public-affairs term lands there.

**Lore is not separable from Helm or Shield.** Lore→HELM 22.6%, Lore→SHIELD 20.2%, Lore→LORE 10.7%. The Lore dictionary retrieves other nodes' domains twice as often as its own. Helm→LORE is only 3.6%, so the confusion is asymmetric: Lore fails to be distinct from Helm, not the reverse.

---

## 3. Term-level attribution

Stem-prefix match of dictionary terms in heading + snippet, cross-tabulated against codes. **Indicative only** — a term often occurs in the body and not the snippet, so counts are small and the "term not visible" rows are large.

| Node | Terms that hold up | Terms that import junk |
|---|---|---|
| **Shield** | `enemy` 94.1% (n=17) · `defence` 90.0% (n=10) · `recruiting` 100% (n=6) · `enlistment` 100% (n=2) · `military` 71.4% (n=7) | none identified |
| **Helm** | `parliament` 75.0% (n=4) · `referendum` 66.7% (n=3) · `minister` 42.9% (n=21) | **`government` 23.1% (n=26)** · **`authority` 16.7% (n=6)** |
| **Lore** | `loyalty` 100% (n=2) · `empire` 66.7% (n=6) | **`nation` 0.0% (n=4)** · `duty` 33.3% (n=3) · `betrayal` never appeared |
| **Hands** | `labour` 100% (n=1) · `strike` 66.7% (n=3) · `employment` 66.7% (n=3) | **`worker` 27.3% (n=11)** |
| **Stewards** | none | `property`, `investment`, `capital` all 0% and 100% ROUT; `landowner` and `finance` never visible |

The diagnostic that matters most is the contrast between "term visible in snippet" and "term not visible":

| Node | Precision when a term is visible | Precision when no term visible | n (not visible) |
|---|---|---|---|
| Shield | 71–100% | 42.2% | 45 |
| Helm | 17–75% | 31.2% | 32 |
| Lore | 0–100% | **2.9%** | 69 |
| Hands | 27–100% | **4.6%** | 65 |
| Stewards | 0% | **7.5%** | 80 |

For Stewards, 80 of 84 items matched on a term that never surfaced in the opening lines — the match is buried in price tables, auction lists and advertising bodies. That is the mechanism of its failure, made visible.

`government` deserves separate note: it appears in 26 items and carries 23.1% precision with 42.3% ROUT. It is not a Helm term in practice; it is a general-purpose modifier attached to reports of everything from tramway concessions to cocoa duties.

---

## 4. What this does to the record

**Nothing statistical is retracted.** Both registered tests ran as specified. Episode 1: T = +0.4514, p ≤ 3.7 × 10⁻⁷ on the six-title panel. Episode 2: T = +0.1125, p = 0.00156. Leave-one-title-out 6/6. Those numbers stand.

What changes is what may be *said* about them.

**Episode 1's node-level claim shrinks to one node.** The predicted-dominant set was {Shield, Helm, Lore}. Helm is 33.3% and Lore is 11.9% in 1916 — both measurement failures. The Episode 1 result was already known to be carried by Shield (lift 2.06 against Helm 1.25 and Lore 1.21); the audit now shows that the two minor contributors were not measuring what their labels claim. The defensible statement is:

> During the conscription referenda, a dictionary shown to retrieve coercion-and-defence discourse at 78.6% precision rose sharply relative to two dictionaries that retrieve mostly routine and non-institutional text.

**Episode 2's node-level claim is withdrawn entirely,** as previously recorded. Neither Hands (19.0%) nor Stewards (7.1%) can support substantive interpretation.

**The surviving cross-episode claim is unchanged and is the strongest result in the programme:**

> Shield lift 2.06 (conscription) → 0.89 (depression, registered 3-title panel) / 0.84 (6-title Amendment-01 extension), both below the mechanical null, in the one dictionary with demonstrated precision in its critical year. The instrument discriminates crisis morphologies and is not measuring historical salience. Quote both, labelled; do not overwrite one with the other.

**Two standing qualifications on that number.** Shield's 1914 baseline precision is 50.0% against 78.6% in 1916, so the measured lift compounds a frequency change with a semantic-composition change; 2.06 is an upper bound. And Shield's foreign share runs 38% → 52%, so much of what rose is cable war reporting rather than Australian institutional discourse — a substantive question about what the node denotes, not a coding error.

---

## 5. Evidence base for Version 2

Recorded, **not acted on**. Any revision creates instrument Version 2, testable only on episodes not yet run.

What the audited evidence supports:

1. **Restrict the harvest to `l-category=Article`.** Stewards' failure is 57–81% routine content concentrated in advertising and detailed lists. The earlier exploratory check showed this improves both episodes (Episode 1 T +0.4228 → +0.4590; Episode 2 Stewards-alone p 0.2024 → 0.0032).
2. **Drop or replace the four identified poison terms:** `government`, `authority`, `nation`, `worker`. Each imports large volumes of non-node material and each is a high-frequency generic word or a permissive stem class.
3. **Stewards needs rebuilding, not repair.** No Stewards term survived. Its vocabulary is the standing language of commercial notices. Candidate replacements would need to be event terms — insolvency, suspension, foreclosure, liquidation, receivership — rather than asset terms.
4. **Lore needs a separability criterion, not just better terms.** Its failure is 43% leakage into Shield and Helm, which better vocabulary alone will not fix. Version 2 should require that each node's dictionary be tested for cross-node leakage before use, with a declared maximum.
5. **Shield can stand, with its baseline-precision problem documented.**
6. **Adopt a precision floor as a release gate.** No node should enter a confirmatory episode below a stated precision, measured on a fresh audit sample.

What is **not** supported: any re-analysis of Episodes 1 or 2 with revised dictionaries.

---

## 6. Limitations

1. **Single coder, no reliability estimate.** Both subsample packs are prepared (`audit_subsample_second_coder.csv`, 50 items; `audit_subsample_second_coder_B.csv`, 34 items; shared guide). Until returned, every number here rests on one set of judgements — mine, on an instrument I helped build. Blinding removed knowledge of the retrieving node, which is the failure mode that matters most, but it is not a substitute for independent coding.
2. **Headings and snippets only.** Full text is not retrievable. This most plausibly *understates* precision, so it is conservative against the failing nodes and inflates nothing. It also means the term-attribution table under-counts term presence, which is why the "not visible" rows are large.
3. **OCR failure remains unmeasured.** No item was coded GARB. The failures documented here are semantic, not optical — the dictionaries retrieve perfectly legible text about the wrong things.
4. **Contestable coding rules.** Subject-not-actor resolved cases like "miners' union meeting about conscription" (→ HANDS) and "council debating a finance report" (→ HELM). A different rule moves numbers, particularly at the Helm boundary.
5. **Small per-term counts** in section 3. Treat the direction as informative and the magnitudes as provisional.

---

## 7. Where the programme stands

| Stage | Status |
|---|---|
| Episode 1 — conscription | statistically overwhelming; node-level claim reduced to Shield alone |
| Control window 1908–12 | passed; first out-of-sample check |
| Episode 2 — depression | statistically robust (p = 0.00156, 6/6 leave-one-out); node-level claim withdrawn |
| Panel extension | passed; instability repaired |
| **Measurement audit** | **complete; 9 of 10 node-periods fail; Shield alone validated** |
| Second-coder reliability | **outstanding — the one thing blocking closure** |
| Version 2 dictionaries | evidence assembled, revision not made |
| Episode 3 | correctly deferred until Version 2 is frozen |

The programme has a reproducible, robust, sign-flipped lexical result and exactly one validated node. That is a smaller claim than it looked like three steps ago, and a much more secure one.

---

## Files

| File | Contents |
|---|---|
| `AMENDMENT_02_Audit_Helm_Lore.md` | Batch B pre-registration |
| `audit_sample_blind_B.csv` · `audit_codes_coder1_B.csv` · `audit_key_B.csv` | Batch B sample, codes, key |
| `audit_coded_all_420.csv` | All 420 items, merged and unblinded |
| `audit_subsample_second_coder.csv` (50) · `_B.csv` (34) · `audit_second_coder_guide.md` | Reliability packs |
| `Trove_Measurement_Audit_Report.md` | Batch A report, superseded by this document for pooled figures |
