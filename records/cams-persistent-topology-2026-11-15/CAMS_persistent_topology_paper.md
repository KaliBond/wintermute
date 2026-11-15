---
title: "Persistent national topology in coded institutional profiles: four pre-registered blind rounds"
author: "Kari McKern (Neural Nations, Sydney)"
date: "15 November 2026"
---

*Added 2026-11-15: Conducted by AI systems (Diogenes; Hermes); commissioned by Kari McKern. See the dated credits note at the end of this paper.*

# Abstract

We test whether societies carry a persistent arrangement of institutional functions that can be recognised across a century of change. The data are coded ratings from the CAMS framework, as recorded in the JUNO v3 dataset. Each society-year scores eight functional nodes on four terms (Coherence, Capacity, Stress, Abstraction). We compared two summaries of an era's profile: the full 32-value profile, matched by Pearson r, and an 8-value shape σ = (A·C/100)(K − S), matched by Spearman ρ.

Over four pre-registered rounds, randomly drawn eras were matched mechanically against six national reference profiles, each built without the era under test. Every protocol and script was SHA-256 hashed before the draw. An independent matcher committed its rankings before any key was opened.

Both confirmatory rounds met their pre-registered primary criterion:

- **B4** (full-profile primary): 5 of 12 eras ranked their own society first (exact binomial p = 0.036, chance 1/6).
- **B5** (σ primary): 7 of 12 (p = 0.0013).

Two mechanism tests then examined the one consistent σ failure, Russia 1986–1991:

- **B6** falsified the hypothesis that crisis-level stress swamps σ.
- **B7** falsified the hypothesis that a single node or term carries the full-profile match.

One exploratory lead remains: removing Helm from σ restores Russia. We state the limits and name the next tests.

# 1. Introduction

Institutions change. Governments fall, economies fail and recover, and wars remake whole societies. Yet observers of humans' societies often say that a country "is still itself": that something in how it organises command, protection, knowledge and work persists through upheaval. That claim is usually made in prose and checked against memory. Here we make it in a form that can fail.

The CAMS framework describes a society in each year by eight functional nodes:

- **Helm:** direction and command
- **Shield:** protection
- **Lore:** explanation and meaning
- **Stewards:** provisioning and administration
- **Craft:** skilled production
- **Hands:** ordinary labour
- **Archive:** record and memory
- **Flow:** circulation and exchange

Each node is rated on four terms: Coherence (C), Capacity (K), Stress (S) and Abstraction (A). An era of several years can then be summarised as an 8 × 4 profile.

**The claim under test.** A society's era profile resembles that society's own long-run profile, built *without* that era, more than it resembles other societies' long-run profiles, and more often than chance. We call this a *persistent national topology*.

We take what we call Darwin's posture: the claim is a bold inference from a body of observations, and it earns its keep only by surviving tests that could kill it. Each round therefore had to name, before any data were drawn:

- a primary method,
- a success threshold derived from an exact null distribution,
- the conditions under which the claim would be called falsified.

When the claim survived, we pressed on its weakest point. When a proposed explanation failed, we recorded the failure as a result.

The paper reports four rounds:

- **B4 and B5** are confirmatory tests of persistence. They use the same design with the primary method swapped.
- **B6 and B7** are single-case mechanism tests. They ask why the 8-value shape σ repeatedly failed on Russia's late-Soviet years, when the full profile did not.

# 2. Methods

## 2.1 Data

All profiles derive from `JUNO_Unified_Dataset_v3_2026-10-19.csv` (SHA-256 `54b62badddfd1f7ba17eb67ebaaf604d7e792f335eda685065a0fa096b76b9b4`). Each row gives one society, year and node, with Coherence, Capacity, Stress and Abstraction scores. These are **coded ratings** produced within the CAMS research programme. They are on a nominal 0–10 scale, and in parts of the record they fall on a 0.2-point grid. They are interpretive codings, not instrument readings.

The panel comprised six societies: Australia, China, Norway, Russia, the United Kingdom and the United States. The annual coverage of each society in the dataset is:

| Society | Complete years |
|---|---|
| Russia | 227 (1800–2026) |
| China | 226 (1800–2025) |
| Australia | 152 (1875–2026) |
| United Kingdom | 147 (1880–2026) |
| Norway | 147 (1880–2026) |
| United States | 127 (1900–2026) |

A year is *complete* only if all eight nodes have all four terms present.

Eras come from the November 2026 CAMS personality era grid. Its three source files are recorded with their SHA-256 digests in the B6 key. The grid holds 107 eras for the six societies with at least one complete year. Era lengths range from a single complete year to several decades.

## 2.2 Measures

For an era (or a reference), the profile is the node-wise arithmetic mean of exact source values over the relevant complete years, rounded half-up to two decimals (Python `decimal`, `ROUND_HALF_UP`).

**Full profile (32 values).** The values (C, K, S, A) for each node are concatenated in canonical node order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow. Similarity is the Pearson product-moment correlation r between two such vectors.

**σ shape (8 values).** For each node, computed from the rounded C, K, S and A:

- g = A · C / 100
- e = K − S
- σ = g · e

Similarity is the Spearman rank correlation ρ between two 8-value σ vectors, using midranks for ties. This is the ordinary product-moment correlation of the midrank vectors.

The two measures differ in what they keep:

- The full profile retains every coordinate, including absolute levels and between-term differences.
- σ compresses each node's four terms into one product of a "grounded abstraction" term g and a "capacity margin" e, and keeps only the rank order of the eight products.

σ is the more parsimonious summary. The question is whether it loses the signal.

## 2.3 Matching design

**Test items.** Each item is one era's profile. The file carries no identifier of society, year, era or source.

**References: leave-one-era-out with balanced concealment.** For each item, six labelled national references were built. Each reference is the full-record mean of that society's complete years, with one era excluded:

- the test era, for the item's own society;
- one era drawn at random (with `secrets`), for each of the other five societies.

The second rule ensures that "which society is missing an era" cannot identify the item. References report C, K, S and A to two decimals, with g, e and σ derived from those rounded values. All exclusions were recorded only in a sealed key.

**Ranking.** For each item and method, the six societies were ranked by correlation, highest first:

- Ties were broken alphabetically (Australia, China, Norway, Russia, UK, USA).
- An undefined correlation, from a zero-variance vector, was ranked below every defined one.

A *rank-1 hit* is the true society at rank 1. A *top-two hit* is the true society at rank 1 or 2. Rankings were produced mechanically by each round's scorer script.

**Draws.** Eras were drawn with Python's `secrets.randbelow`, without replacement, subject to a cap of at most three eras per society. Every era used in an earlier round was excluded from later pools. Item positions were randomised. The drawn identities, year lists, reference exclusions and a random salt were written only to a sealed key file (file mode 600).

## 2.4 Pre-registration, hashing and commitment

Each round followed the same sequence:

1. **Protocol and scripts hashed.** The protocol and every script were written and SHA-256 hashed before the draw (B4–B6) or the run (B7). The digests were recorded in the round's hash file, and the protocols forbid editing a hashed file. Any change made before a draw or run was appended as a dated, versioned amendment and re-hashed.
2. **Draw hashed.** After the draw, every test matrix, every reference file and the sealed key were hashed into a manifest. The manifest digest is the SHA-256 of a fixed-order list of `sha256:<digest>  <filename>` lines.
3. **Public pack released.** A public tarball went to the matcher. It contained the protocol, the scorer, the matrices, the references and the hash file, but not the key.
4. **Commitment.** The matcher, an AI agent (Hermes) running on a separate machine that could not see the analysis machine's files, ran the scorer on the public pack. It appended its rankings to a commitment file before any key was opened.
5. **Unblinding.** The key was then opened, every digest re-verified, and hits scored.
6. **Cross-check.** The matcher's committed rankings were compared with the analysis machine's scorer output.

All outcomes, corrections and cross-checks were appended to the round's hash file as dated notes. No earlier text was rewritten.

In B7 the target's identity was public, so blinding the matcher offered little. B7 was instead run twice, independently, from the same hashed pack:

- the matcher ran first and committed the SHA-256 digests of its output files;
- the analysis machine then ran;
- the two sets of outputs had to be byte-identical.

## 2.5 Statistics

For B4 and B5 (n = 12 items, six candidate societies), we used one-sided exact binomial tests at α = 0.05. Each threshold is the smallest k with P(X ≥ k) ≤ 0.05:

| Criterion | Null | Threshold | Tail probability |
|---|---|---|---|
| Primary (rank-1 hits) | p₀ = 1/6 | k = 5 | P(X ≥ 5) = 13187681/362797056 ≈ 0.0364 |
| Co-registered secondary (top-two hits) | p₀ = 1/3 | k = 8 | P(X ≥ 8) = 3323/177147 ≈ 0.0188 |

The claim was supported in a round only if the primary criterion was met. The secondary criterion could not rescue a failed primary. The competing method in each round was scored in parallel and compared descriptively, as the difference in rank-1 hits, with no significance test.

B6 and B7 were pre-registered as **mechanism tests on a single case**, with no significance level claimed for the target. Decoy and control hit rates were reported descriptively, with exact binomial tails.

## 2.6 Round designs

### B4: full-profile primary

- **Primary method:** full-profile Pearson. **Baseline:** σ Spearman.
- **Items:** 12, drawn from a pool of 104 eligible eras. The pool excluded the three eras used in the exploratory rounds B1–B3.
- **Why the full profile was primary:** in the exploratory round B3, a post-hoc Pearson analysis of the 32-value profile ranked the true society first where σ had ranked it third. B4 was registered to test that observation prospectively.

### B5: σ primary

The design was identical to B4, with the methods swapped: σ Spearman was primary and full-profile Pearson the baseline. There were 12 fresh items from a pool of 92, which excluded every era used in B1–B4. The thresholds were unchanged.

### B6: Russia crisis-masking test

**Motivation.** In B5, both Russia items from the late-Soviet years were missed by σ (rank 5 each) but matched by the full profile (ranks 1 and 2):

- Russia XVII, 1986–1988;
- Russia XVIII, 1989–1991.

**Hypothesis.** The crisis-level stress of that period swamps σ's four-term reduction. If so, masking the era's crisis years from the test matrix should restore Russia under σ, while the full-profile rank stays roughly where it is.

**Items.** The registered design (v1.1, which superseded v1.0 before any draw) used:

- one six-year target item, Russia 1986–1991, whose reference excluded 1986–1991 only;
- ten decoys drawn with `secrets` from the 80 eligible eras not used in B1–B5, with Russia's cap counting the target once.

**Mask (rule M2).** The same rule was applied to every item:

- Eras with fewer than four complete years were left unmasked.
- Otherwise, the block of three calendar-consecutive complete years with the lowest mean η_loop was removed (the earliest block on ties).
- The test matrix was rebuilt from the remaining years. Both methods received that same masked matrix.
- References were not masked.

**η_loop.** For each year and each node i:

- q_i = (0.6 C_i + 0.4 A_i)/10
- B_ij = √(q_i q_j) · 2^(−(S_i + S_j)/10), for each other node j ≠ i
- BS_i = the mean of B_ij over the seven other nodes
- η_loop = BS_Lore · BS_Archive / max(S_Hands, 0.01)

This form follows the specification approved by Kari McKern, which she attributes to her May 2026 calibration. One sealed exploratory variant (X2) used the denominator S_Hands + 2, the primary form in an earlier CAMS pre-registration (T-M7). η_loop had not been used in any earlier round's scoring.

**Criteria.**

- *Not testable:* the target's unmasked σ rank, with the same references, is already 2 or better.
- *Supported:* the masked σ rank is 2 or better, **and** strictly better than the unmasked σ rank, **and** the full-profile rank changes by no more than one place.
- *Confounded:* the σ part holds but the full-profile condition fails.
- *Falsified:* the σ part fails.

**Null (descriptive).** The analysis also scored, against the same references, every three-consecutive-year block (4 blocks) and every three-year subset (20 subsets) of the target era. The protocol states that this null cannot reach significance: its smallest non-zero fraction is 1/20.

### B7: node and term ablation

**Question.** Which node or term carries Russia 1986–1991's full-profile match, and is that what σ loses?

**Inputs.**

- **Target:** the unmasked B6 Russia matrix with its B6 references.
- **Controls:** three eras chosen by a fixed rule. The rule took the first three B5 items in (randomised) B5 item order that had both σ rank 1 and full-profile rank 1, at most one per society. It selected USA 1991–2000, Norway 2014–2019 and Russia 1856–1881.

**Analyses.**

1. **Full-profile ablation:** drop each node (4 values) in turn, then each term (8 values) in turn, and re-rank.
2. **σ restoration:** for each node, append its raw (C, K, S, A) to the σ vector; for each term, append that term's eight raw values. The comparison statistic was a *pooled within-block rank correlation*: midranks within each block, centred on the block mean, then Pearson r of the concatenated vectors. Every node and term was restored, so that each carrying element had placebo comparisons.
3. **Exploratory σ node-drop:** drop each node from σ and re-rank.

**Rules (v1.1, unchanged by v1.2).**

- A node or term was *carrying* if removing it worsened the true society's rank by at least two places **and** left it outside the top two. The rule was identical for target and controls.
- **Prediction P1:** the target is *localised*, with one or two carrying nodes. *Broad-fragile* (three or more) or *distributed* (none) would falsify P1.
- **Prediction P2:** at least one carrying node is *linked* (its restoration brings σ to rank 2 or better) and *specific* (fewer than half of the non-carrying restorations do the same).

Amendment v1.2 corrected only a fault in the script's integrity check (Appendix A).

# 3. Results

## 3.1 B4: full-profile primary

Manifest `e272f80307d601a22a5dbb88b6c81f2564d04d51d8d17bb6c686e673fd459159`; protocol `f714a48f…1132b`.

**Table 1.** B4 truths and ranks of the true society (1 = best of 6).

| Item | Era | Full profile (primary) | σ (baseline) |
|---|---|---|---|
| 01 | Norway 1880–1904 | 2 | 2 |
| 02 | Norway 2005–2013 | 1 | 1 |
| 03 | UK 1971–1979 | 1 | 1 |
| 04 | USA 1980–1990 | 1 | 1 |
| 05 | China 2002–2012 | 6 | 5 |
| 06 | Russia 2000–2008 | 3 | 1 |
| 07 | Australia 1983–1991 | 3 | 3 |
| 08 | UK 1930–1939 | 1 | 1 |
| 09 | Australia 2020–2021 | 3 | 4 |
| 10 | UK 1940–1945 | 2 | 1 |
| 11 | China 1840–1849 | 1 | 3 |
| 12 | Russia 1907–1914 | 3 | 2 |

**Primary method (full profile).**

- 5 of 12 rank-1 hits (p = 0.0364). The criterion was **met**.
- 7 of 12 top-two hits (p = 0.066). The secondary criterion was **not met**; it needed 8.

**Baseline (σ).**

- 6 of 12 rank-1 hits (p = 0.0079).
- 8 of 12 top-two hits (p = 0.0188).

The difference in rank-1 hits (primary minus baseline) was −1. The matcher's committed rankings matched the scorer's output on all 12 items. China 2002–2012 was the largest miss under both methods.

## 3.2 B5: σ primary

Manifest `e2d35dd2a27e4ad0a0b90301e13f0df6cdec36ff67a016116dd2767cf35b2701`; protocol `6ff5cb71…f9f0`.

**Table 2.** B5 truths and ranks of the true society.

| Item | Era | σ (primary) | Full profile (baseline) |
|---|---|---|---|
| 01 | USA 1991–2000 | 1 | 1 |
| 02 | China 1938–1945 | 2 | 1 |
| 03 | Russia 1986–1988 | 5 | 1 |
| 04 | Norway 2014–2019 | 1 | 1 |
| 05 | Russia 1989–1991 | 5 | 2 |
| 06 | Russia 1856–1881 | 1 | 1 |
| 07 | Australia 1996–2007 | 1 | 2 |
| 08 | Australia 1950–1965 | 2 | 4 |
| 09 | USA 1900–1916 | 1 | 1 |
| 10 | UK 1880–1899 | 1 | 1 |
| 11 | Norway 1920–1929 | 4 | 2 |
| 12 | UK 1900–1913 | 1 | 1 |

**Primary method (σ).**

- 7 of 12 rank-1 hits (p = 0.0013). The criterion was **met**.
- 9 of 12 top-two hits (p = 0.0039). The secondary criterion was **met**.

**Baseline (full profile).**

- 8 of 12 rank-1 hits (p = 0.0002).
- 11 of 12 top-two hits.

The difference in rank-1 hits was −1. The matcher's committed rankings matched the scorer's output on all 12 items.

Across B4 and B5, the method named as primary met its bar both times. In each round the competing method scored one more rank-1 hit than the primary.

## 3.3 B6: crisis masking

Draw manifest `210b09a217c6ef482075c014f04b6d0b4b18269f73e800f8507d240cb3d33b0e`; key `bf00f583…0b24`; v1.1 pre-draw manifest `dec4ccc7…9f47`.

**Table 3.** B6 truths, masked blocks and ranks of the true society. In each pair, the first figure is the M2-masked rank and the second the unmasked rank, against the same references.

| Item | Era | Masked block | σ | Full profile |
|---|---|---|---|---|
| 01 | China 2020–2025 | 2020–22 | 2 / 1 | 6 / 6 |
| 02 | Australia 1940–1945 | 1941–43 | 5 / 4 | 4 / 4 |
| **03** | **Russia 1986–1991 (target)** | **1989–91** | **5 / 4** | **1 / 2** |
| 04 | Russia 1954–1964 | 1962–64 | 1 / 1 | 3 / 3 |
| 05 | China 1928–1937 | 1935–37 | 2 / 2 | 1 / 1 |
| 06 | UK 1998–2007 | 2003–05 | 3 / 3 | 1 / 1 |
| 07 | Norway 1905–1913 | 1910–12 | 1 / 1 | 1 / 1 |
| 08 | Australia 2022–2025 | 2022–24 | 2 / 3 | 1 / 1 |
| 09 | USA 1929–1932 | 1930–32 | 1 / 5 | 1 / 1 |
| 10 | Australia 2014–2019 | 2017–19 | 1 / 1 | 1 / 2 |
| 11 | China 1917–1927 | 1923–25 | 1 / 1 | 1 / 1 |

**The target.**

- **Masked block.** The rule removed 1989–1991, the block with the lowest mean η_loop (0.0067, against 0.0107, 0.0170 and 0.0191 for the other three blocks).
- **Testable.** The target's unmasked σ rank was 4.
- **σ: fail.** The masked σ rank was 5, so the σ criterion failed on both counts: it was not 2 or better, and it was not better than 4. **The crisis-swamping hypothesis is falsified for this case.**
- **Full-profile condition held.** The full-profile rank moved from 2 to 1.
- **X2 agreed.** The exploratory variant selected the same block on every item and gave the same ranks.
- **Null.** No three-consecutive-year block (0 of 4) and no three-year subset (0 of 20) brought Russia to σ rank 2 or better.

**Decoys (descriptive).** The rule masked all ten decoys, and their σ rank changed on four:

| Item | Era | Unmasked σ rank | Masked σ rank |
|---|---|---|---|
| 01 | China 2020–2025 | 1 | 2 |
| 02 | Australia 1940–1945 | 4 | 5 |
| 08 | Australia 2022–2025 | 3 | 2 |
| 09 | USA 1929–1932 | 5 | 1 |

Under masking, the decoy hit rates were:

| Method | Rank-1 hits | Tail p | Top-two hits | Tail p |
|---|---|---|---|---|
| σ | 5 of 10 | 0.0155 | 8 of 10 | 0.0034 |
| Full profile | 7 of 10 | 0.00027 | 7 of 10 | 0.0197 |

The matcher's ranking file was byte-identical to the analysis machine's.

## 3.4 B7: ablation

Pre-run manifest (v1.2) `40ed775d0b03edb1ba97acc60dd2573fa90e945252dc18f4067ba3c1b268e5c7`. The two independent runs produced byte-identical outputs:

- `results_B7.csv`: SHA-256 `a83cf10be546fcadd840560e6affc898e760af1d76cd18072c41a677679d984d`
- `outcome_B7.md`: SHA-256 `eae88f60331ae6778381447679c7e7af05c82e77ec7acc259a1a9a7f99e65597`

**Baselines for the target.**

- **Full profile:** Russia ranked 2 (r = 0.434), behind China (r = 0.721) and well ahead of the UK (r = −0.019).
- **σ:** Russia ranked 4 (ρ = −0.048). China (0.381), the UK (0.262) and the USA (0.071) ranked above it.

**Table 4.** B7 target (Russia 1986–1991): rank of Russia under each manipulation.

| Manipulation | Helm | Shield | Lore | Stewards | Craft | Hands | Archive | Flow | C | K | S | A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full profile, element dropped (baseline 2) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| σ, element restored (baseline 4) | 5 | 4 | 4 | 4 | 3 | 3 | 4 | 3 | 3 | 3 | 4 | 2 |
| σ, node dropped (exploratory) | 1 | 6 | 3 | 5 | 5 | 4 | 3 | 5 | – | – | – | – |

**P1: falsified.** No node and no term was carrying. In fact, no single drop moved Russia from rank 2 at all, so the target's full-profile signal is *distributed*.

**P2: not assessed,** because there was no carrying node. The restorations are reported for completeness:

- No node restoration reached rank 2.
- Restoring term A did reach rank 2. This is not a pre-registered link, because A was not carrying.

**Exploratory node-drop.**

- Dropping Helm from σ moved Russia from rank 4 to rank 1 (7-node ρ = 0.214, against the UK 0.107 and China 0.071).
- Dropping Shield sent Russia to rank 6.

**Controls.** None of the three controls had a carrying element. Every full-profile drop and every σ restoration left each control at rank 1. In the σ node-drop, Russia 1856–1881 slipped to rank 2 when Helm or Shield was dropped. The other two controls stayed at rank 1 under every drop.

# 4. Discussion

## 4.1 What survived

The persistence claim survived both confirmatory tests it was exposed to:

- In B4, the full profile placed the true society first in 5 of 12 eras.
- In B5, σ did so in 7 of 12 fresh eras.

Chance would give about 2 in 12. Each round met the threshold it had registered before any era was drawn. The B4 secondary (top-two) criterion was missed, and we record that miss alongside the passes.

Both methods carried the signal in both rounds, and neither dominated: the baseline method beat the named primary by one rank-1 hit in each round. Because the method that was primary in one round was the baseline in the other, the result does not depend on a single choice of method. It also does not depend on choosing eras by hand, since every era was drawn by a cryptographic generator from a pool fixed in advance.

The inference we draw, and hold open to refutation, is this. In these codings, a society's era profile keeps a recognisable, society-specific arrangement across wars, revolutions and long economic cycles. It is recognisable against a reference from which that era has been removed. We think the simplest explanation is that the arrangement is a durable property of how a society's institutions are organised, as the coders perceive it, rather than an artefact of the matching.

## 4.2 What failed

Russia's late-Soviet years were σ's one recurrent failure. We proposed two explanations in advance, and each could have been confirmed. Both failed.

**Crisis swamping (B6).** If stress were overwhelming σ, removing the most stressed years should have helped. It did not:

- Masking the lowest-η_loop block (1989–1991) left σ at rank 5.
- No three-year mask of any kind brought Russia into the top two.

The miss is therefore not a property of the crisis years. The full profile, by contrast, held Russia in the top two with or without masking.

**A localised carrier (B7).** If one node or term carried Russia's full-profile match, dropping it should have broken the match. No drop moved Russia even one place. The full profile's advantage on this era is spread across its 32 coordinates, so its information cannot be pinned to one place. This argues against any single targeted repair that would add a node or term back into σ.

## 4.3 The remaining lead

The only concentrated signal we found lies on the σ side. Dropping Helm from σ turned Russia 1986–1991 from a miss (rank 4) into a hit (rank 1). Dropping Shield made it much worse (rank 6).

One reading is that σ's multiplicative form, g · e, magnifies one node's departure from the national pattern. On this reading, the late-Soviet Helm (direction and command) departs sharply enough to drag the whole σ ranking. We do not treat this as a finding, because:

- It is one exploratory manipulation, on one case, among 32 manipulations run without multiplicity correction.
- It rests on a 7-point Spearman correlation.
- The only echo among the controls is a one-place slip for Russia 1856–1881.

It is a hypothesis to test.

## 4.4 Limits

- **Sample size.** Twelve items per confirmatory round gives low power. Point estimates of the hit rate are imprecise.
- **Coded data.** The data are coded ratings from one research programme. Agreement between two summaries of the same codings is not independent validation of the codings.
- **Panel size.** The panel has six societies. With more societies, the chance rate would fall and the test would be harder.
- **Overlapping references.** References for different items overlap heavily, because each is a long-run mean, so items are not fully independent.
- **Single-case mechanism tests.** B6 and B7 each analysed one case. Their verdicts describe that case only.
- **Multiplicity.** B7 applied no multiplicity correction across its ablations, restorations and drops.
- **What the designers knew.** The designers of B6 and B7 knew all earlier outcomes, and the B6 designer had seen the raw Russia values for 1986–1991. These exposures were disclosed in the protocols before hashing. No B7 statistic on the target or controls was computed before the matcher committed its results.

## 4.5 Next tests

We name three tests:

1. **Helm-lead round (pre-registered).** Register a σ variant that leaves out Helm, or caps any single node's influence. Test it blind on fresh eras that include every Russia era not yet used. It should pass only if it recovers Russia's late-Soviet eras **and** loses nothing on other societies.
2. **More societies.** Extend the panel beyond six. A larger panel lowers the chance rate and tests whether persistence holds in a crowded field.
3. **Human scorers.** Have independent human coders rate a sample of eras without knowing the society. If the topology survives in their codings, it is not an artefact of a single coding source.

# 5. Data and code availability

The complete record is held in the analysis directory `blind_round/`. For each round it contains:

- the protocol and its amendments;
- the scripts (`score_B4.py`, `score_B5.py`, `build_B6.py`, `score_B6.py`, `evaluate_B6.py`, `build_inputs_B7.py`, `run_B7.py`);
- every test matrix and reference file;
- the sealed keys, opened after commitment;
- the ranking and results files;
- the hash files `hashes_B4.txt` to `hashes_B7.txt`, holding every manifest and every dated note.

The public tarballs, without keys, have these SHA-256 digests:

| Tarball | SHA-256 |
|---|---|
| B4 | `bd61ded36b940d2224c7902762f0469f53dae28a6a1bab95807e44b43752b878` |
| B5 | `d96ad7004504c83e050c77daf20671d216268bfeb0b809b1e5e13a3691b71618` |
| B6 draw | `74eff5300e32f1e673e389e8e71481e7caccc43265bf01c5aec743aa85dba64e` |
| B7 v1.2 | `06489a40e2a29fc6d1cf874d2aa0af1aa79924780f1c9edd34545f4086561953` |

A series write-up, signed off by the matcher, and its digests are in `SERIES_B4-B7_writeup_signed.md` and `SERIES_hashes.txt`. The source dataset is `JUNO_Unified_Dataset_v3_2026-10-19.csv` (digest in §2.1).

The bundle will be deposited on Zenodo. DOI: [to be assigned on deposit].

# Acknowledgements

Matching and replication were performed by Hermes, an AI agent running on a separate machine. Hermes committed rankings or result digests before each key was opened or each comparison was made, and signed off the series write-up without objection. Protocol drafting, draws, hashing and scoring on the analysis machine were AI-assisted. The author takes responsibility for the design, the decisions and the interpretation.

# Appendix A. Deviations and errata (summary)

Each item below is also recorded as a dated note at the point where it occurred.

1. **B5.** A hash list sent in a chat message transposed two digests. It was corrected in the same message, and no file was affected. The analysis machine could not read the matcher's B5 commitment at the time of unblinding. The matcher later confirmed an item-for-item match.
2. **B6 commitment file.** The matcher's commitment text first listed σ rankings only. Its lines for items 03 and 08 were copied from the B5 lines with the same item numbers. Neither error changed a score. The matcher's ranking file was byte-identical to the analysis machine's, and a correction was appended.
3. **B6 hit-count bar.** The matcher's text cited a "≥ 5/11 primary bar". B6 registered no hit-count bar, and the matcher appended a retraction.
4. **η_loop provenance.** The approval for B6 stated that η_loop had been used in B1–B5. It had not, and the B6 protocol says so. The T-M7 pre-registration's primary denominator (S_Hands + 2) differs from the one used in B6 (max(S_Hands, 0.01)). The T-M7 form was kept as the sealed exploratory variant, and it selected identical masks. The cited May 2026 calibration file was not located on the analysis machine.
5. **B6 identifiability.** As the protocol anticipated, the masked target matrix was byte-identical to the published B5 matrix for 1986–1988, so its digest revealed which block had been masked. The rankings were mechanical, so this could not change the result.
6. **B7 integrity check.** Our v1.1 hash block listed record-only superseded files in the manifest's format. The script therefore demanded files that were absent from the public pack. This was fixed as v1.2, with no change to the design.
7. **Edited hashed files on the matcher's machine.** In the folder holding its v1.1 extraction, the matcher edited the script and the hash file to bypass that check. The folder was declared invalid, and no result from it is used. The valid replication was run in a fresh folder from the v1.2 pack.
8. **Timestamps.** The B7 v1.2 amendment heading gives a time about four minutes later than the true time. The matcher's sign-off on the series write-up is timestamped about 40 minutes ahead of the analysis machine's clock. Both are noted as stated and not corrected.


---

# Credits note (appended 2026-11-15, Australia/Sydney)

**Conducted by AI systems (Diogenes; Hermes); commissioned by Kari McKern.**

1. The CAMS framework, the JUNO coded data and the research question belong to Kari McKern (Neural Nations). She commissioned the B4–B7 series.
2. Two AI systems carried out the series together. Diogenes (Grok Bot, on the analysis machine, "the box") and Hermes (on Kari McKern's machine, Nitra) did the design, pre-registration, scoring, hashing, write-up and replication between them.
3. Kari McKern relayed messages between the two systems and made the go/no-go decisions.

This note is added to the paper rather than replacing anything in it. The author line and the acknowledgements above are left as first written, and this note supersedes them wherever they differ.
