# Measurement-Validity Audit — Results

**Run:** 26 July 2026 · Protocol frozen in `PROTOCOL_Measurement_Audit.md` before any item was viewed.
**Sample:** 252 items — 6 node-periods × 42, stratified 7 per newspaper across the six-title panel, drawn by random date (not by relevance).
**Coding:** heading + snippet, blind to node, year, paper, term and hypothesis. Single coder. Second-coder subsample pending.

---

## Headline

**Five of six node-periods fail the pre-registered 70% precision threshold. Only Shield 1916 passes.**

| Node | Period | n | **Precision** | 95% CI | Verdict |
|---|---|---|---|---|---|
| **Shield** | **1916** | 42 | **78.6%** | 64.1 – 88.3 | **ADEQUATE** |
| Shield | 1914 | 42 | 50.0% | 35.5 – 64.5 | measurement failure |
| Hands | 1893 | 42 | 19.0% | 10.0 – 33.3 | measurement failure |
| Hands | 1888 | 42 | 7.1% | 2.5 – 19.0 | measurement failure |
| Stewards | 1893 | 42 | 7.1% | 2.5 – 19.0 | measurement failure |
| Stewards | 1888 | 42 | 7.1% | 2.5 – 19.0 | measurement failure |

Precision = the blind coder independently assigned the item to that node's institutional domain **and** judged the treatment substantive.

This is the most consequential result in the programme so far, and it is negative.

---

## What the nodes actually retrieved

Concept assigned by the blind coder, pooled across both periods (n = 84 per node):

| Retrieving node | SHIELD | HELM | LORE | HANDS | STEWARDS | NONE |
|---|---|---|---|---|---|---|
| Shield | **55** | 6 | 1 | 2 | 4 | 16 |
| Hands | 1 | 17 | 1 | **11** | 8 | **46** |
| Stewards | 2 | 17 | 0 | 4 | **13** | **48** |

For Hands and Stewards, the modal retrieved item is not the node's own domain — it is **nothing institutional at all**. More items retrieved by the Hands dictionary were coded HELM (17) than HANDS (11).

Breakdown of the 42 Stewards 1893 items: 27 routine non-institutional, 5 routine governmental, **3 substantive Stewards**, and the remainder scattered. Breakdown of Hands 1893: 14 routine non-institutional, 8 substantive Hands, 8 substantive Helm, 5 substantive Stewards.

## Why

**Hands — the stem problem, confirmed.** `worker` stems to `work` / `working` / `works`, verified directly against the API (all return 14,959 in SMH 1917). "Work" appears in council reports, mining returns, court lists and almost every column of a newspaper. The Hands dictionary is not measuring labour discourse; it is measuring a near-ubiquitous English verb, with a labour signal buried inside it.

**Stewards — routine commerce, confirmed.** 81% of Stewards 1893 items were coded ROUT. Advertising share is 23.8% (1888) and 28.6% (1893) against a corpus baseline near 17%; `Detailed lists` adds more. `property`, `capital`, `investment` and `finance` are the standing vocabulary of auction notices, prospectuses, company names and share tables. This confirms **measurement failure** (hypothesis 1/2 from your framing), not the boom-baseline hypothesis: Stewards precision is *identical* at 7.1% in 1888 and 1893, so the 1888 baseline is not full of high-quality speculation discourse. It is full of auction reports, exactly as 1893 is.

**Shield — mostly works, and works better under crisis.** 55 of 84 items were independently coded SHIELD. Precision rises from 50.0% in 1914 to 78.6% in 1916. Foreign content is high (38% → 52%), reflecting cable war reporting — a substantive question about what "Australian Shield discourse" means, not a coding error.

---

## What this does and does not do to the two episodes

**It does not retract either registered result.** Both were run as specified; the statistics stand. What the audit removes is the *interpretation* of the node labels.

**Episode 1 (conscription) survives largely intact.** The result was carried by Shield, with a lift of 2.06 against Helm's 1.25 and Lore's 1.21. Shield is the one node that passes, in the one year that matters, at 78.6%. The finding "Shield-type discourse rose sharply, and rose relative to Hands and Stewards" is supported by the audit.

Two qualifications. Shield's 1914 precision is only 50%, so the *baseline* is noisier than the target; and since precision rises with the crisis, part of the measured lift is a composition shift — the retrieved set becomes more genuinely Shield in 1916, not just larger. That inflates the apparent lift by an unknown amount. Directionally it does not change the conclusion; quantitatively, 2.06 should be treated as an upper bound.

**Episode 2 (depression) cannot be interpreted at node level.** The registered test passed on the joint {Stewards, Hands} contrast, and the extension strengthened it to p = 0.00156. But the exploratory decomposition attributed the effect to Hands — and Hands 1893 precision is 19.0%. **Four in five items counted as Hands were not about labour.** The sentence "Hands became relatively more prominent" is not defensible on this evidence. What is defensible is narrower and stranger:

> During the depression window, a lexical set that mostly retrieves routine and governmental text held its share of the corpus better than a lexical set that mostly retrieves military text. Shield fell below the mechanical null; the Hands and Stewards sets did not fall as far.

That is a real, replicated, sign-flipped result. It is not evidence about labour discourse.

**The directional disconfirmation is the part that survives cleanly.** Shield going from 2.06 to 0.89 (registered 3-title) / 0.84 (6-title extension) rests on the node with 78.6% / 50% precision, not on the failed ones. Both panel figures are correct and must stay labelled. "The instrument distinguishes military mobilisation from economic-social crisis" holds. "It does so by detecting a rise in labour discourse" does not.

---

## Consequences under the frozen rules

Per `PROTOCOL_Measurement_Audit.md`, a node flagged as a measurement failure has its substantive result withheld until the dictionary is revised. Applied:

| Node | Status |
|---|---|
| Shield | provisionally adequate at crisis-year precision; baseline precision poor; retained |
| Hands | **measurement failure — substantive interpretation withdrawn** |
| Stewards | **measurement failure — substantive interpretation withdrawn** |
| Helm, Lore | **not audited.** No claim about them is supported either way. |

Per the consequence rule, any dictionary revision creates **instrument version 2**, testable only on episodes not yet run. Episodes 1 and 2 stand as run under version 1 and may not be re-analysed with revised dictionaries.

The unaudited Helm and Lore nodes are now the largest untested assumption in the programme. Helm was the modal *mis*-assignment for both failed nodes (17 of 84 in each), which suggests governmental language is being picked up promiscuously by dictionaries that are not aiming at it — and says nothing yet about whether the Helm dictionary itself works.

---

## Limitations of this audit

1. **Single coder.** The 50-item subsample and guide are prepared (`audit_subsample_second_coder.csv`, `audit_second_coder_guide.md`). Until a second coder returns them there is no reliability estimate, and every number above rests on one set of judgements. The ROUT/SUB boundary in particular is a judgement call that materially drives the Stewards result.
2. **Headings and snippets only.** Full article text is not retrievable. A snippet can misrepresent an article whose substance lies beyond the first sentences. This most likely *understates* precision — the effect is conservative for the failing nodes and inflates nothing.
3. **OCR failure not measured.** The GARB category was dropped as uncodeable from snippets; no item was coded GARB. The OCR question remains entirely open. Note that low precision here is a *semantic* failure, not an OCR failure — the dictionaries are retrieving perfectly legible text about the wrong things.
4. **Coder is the same agent that built the instrument.** Blinding removed knowledge of which node retrieved each item, which is the failure mode that matters most, but it cannot remove every incentive. The second-coder check is not optional for publication.
5. **Concept assignment involves contestable calls** — notably whether a miners' union meeting about conscription is HANDS or SHIELD, and whether a council finance debate is HELM or STEWARDS. These were resolved by a rule stated in the guide (subject, not actor) applied consistently, but a different rule would move numbers.

---

## Recommended next steps

1. **Get the second coder through the 50-item subsample.** Everything above is provisional until then.
2. **Do not revise the dictionaries yet.** Revision is warranted, but it should be informed by the reliability check and designed once, not iterated. When it happens: replace `worker` with quoted phrases or a stem-controlled set; replace `property` / `capital` / `finance` with terms that do not saturate advertising; and restrict the harvest to `l-category=Article`, which the earlier exploratory check showed improves both episodes.
3. **Audit Helm and Lore before Episode 3.** They are unmeasured and were the commonest false positives for the failing nodes.
4. **Episode 3 should wait** for instrument version 2. Running it on version 1 would add a third result whose node-level interpretation is already known to be unsafe.

---

## Files

| File | Contents |
|---|---|
| `PROTOCOL_Measurement_Audit.md` | Protocol, frozen before sampling |
| `build_audit_sample.py` | Date-stratified sampler, seeded and resumable |
| `audit_sample_blind.csv` | 252 blinded items as coded |
| `audit_codes_coder1.csv` | Coder 1 codes |
| `audit_key.csv` | Unblinding key, with `troveUrl` per item |
| `audit_coded_unblinded.csv` | Merged, with derived precision |
| `audit_subsample_second_coder.csv` + `audit_second_coder_guide.md` | Reliability pack, 50 items |
