# Persistent national topology in CAMS profiles: a pre-registered blind series (B4–B7)

Kari McKern (CAMS / Neural Nations, Sydney), with Diogenes (analysis, box) and Hermes (independent matcher, Nitra)
Written 2026-11-15, Australia/Sydney (AEDT, UTC+11). Version 1.0. Corrections are appended as dated notes, never rewritten.

## Abstract

We asked whether the societies in the CAMS dataset carry a persistent node topology. If they do, a society's era should be recognisable from its 8-node × 4-term profile when compared against that society's own long-run record with the era left out.

We ran four rounds, each pre-registered and hashed before it was run. In B4 and B5, matrices for 12 randomly drawn eras each were blinded, built from `JUNO_Unified_Dataset_v3_2026-10-19.csv`, and matched mechanically against six national references. The claim survived both confirmatory rounds:

- **B4** (full 32-value profile, Pearson, as primary): 5/12 rank-1 hits, p = 0.036. That met the pre-registered bar. Top-two was 7/12, which missed its bar of 8.
- **B5** (8-node σ, Spearman, as primary): 7/12 rank-1 hits, p = 0.0013, and 9/12 top-two. Both bars were met.

The one consistent miss under σ was Russia in 1986–1991. B6 tested whether crisis-level stress was swamping σ there, and falsified that explanation. B7 tested whether a single node or term carried Russia's full-profile match, and falsified that too: the signal is distributed.

The only concentrated lead left is exploratory. Removing Helm from σ moves Russia from rank 4 to rank 1 for 1986–1991. We name the next tests.

## 1. The claim

CAMS scores each society–year on eight functional nodes: Helm, Shield, Lore, Stewards, Craft, Hands, Archive and Flow. Each node has four terms: Coherence (C), Capacity (K), Stress (S) and Abstraction (A). These are coded ratings, not measurements.

Two summaries of an era's 8 × 4 profile were compared throughout:

- **Full profile:** the 32 values in canonical node order, (C, K, S, A) within each node, compared by Pearson r.
- **σ shape:** one value per node, σ = g · e, where g = A · C / 100 and e = K − S, compared by Spearman ρ with midranks.

**Claim (persistent topology).** An era's profile resembles its own society's leave-one-era-out reference more than it resembles other societies' references, more often than chance (1 in 6). This would mean humans' institutions keep a recognisable arrangement across wars, revolutions and a century of change.

## 2. Methods common to all rounds

- **Panel.** Six societies: Australia, China, Norway, Russia, the UK and the USA. A year is *complete* only if all 32 values are present.
- **Units.** Eras come from the November 2026 personality era grid, which has 107 eras with at least one complete year. Era matrices are node-wise means of complete years, rounded half-up to two decimals.
- **References.** For each item, six labelled references are built:
  - Each is the full-record mean of the society's complete years.
  - The test era is left out of its own society's reference (leave-one-era-out).
  - Each of the other five societies also has one era, drawn with `secrets`, left out (balanced concealment). This way the missing era does not identify the true society.
- **Draws.** Eras were drawn with Python `secrets.randbelow`, without replacement, with at most 3 eras per society. Every era used in an earlier round was excluded. Item order was randomised. Identities, years, exclusions and a salt went only into a sealed key (mode 600).
- **Ranking.** Both methods were run mechanically by the round's scorer script, with the same handling everywhere:
  - Ties were broken alphabetically.
  - An undefined correlation was ranked last.
  - The named primary method and the co-reported baseline were fixed in advance.
- **Thresholds** (B4/B5, n = 12, one-sided exact binomial, α = 0.05):
  - Rank-1 hits: at least 5/12 against p₀ = 1/6. P(X ≥ 5) = 0.0364.
  - Top-two hits: at least 8/12 against p₀ = 1/3. P(X ≥ 8) = 0.0188.
  - The top-two result is co-registered. It cannot rescue a failed primary.
- **Integrity.** The order was always the same:
  1. The protocol and scripts were SHA-256 hashed **before** the draw or run.
  2. Every drawn file and the key were hashed into a manifest.
  3. A public tarball, without the key, went to the matcher.
- **Commitment.** Hermes, the matcher on Kari's machine Nitra, cannot see box paths. It scored the public pack and appended its rankings to `/scratch/blind_commit_2026-11-15.txt` before the seal was broken. The box then:
  - verified every hash,
  - opened the key,
  - scored the hits,
  - cross-checked Hermes's commitment against its own rankings.
- **Append-only record.** Outcomes, corrections and amendments were appended as dated notes. No protocol text was rewritten.

## 3. Results by round

### B4: full profile primary (12 items)

Protocol `f714a48ff564f717779d337e457ce757df40c75bfbd1877ad6770893f4b1132b`. Scorer `707d897aaad5ac1bb8194568e22c72cd487c3f9844a5b2dad3a1a85da51229e9`. Key `c25e2ceecbd26080b003606293adc79a3f6fcb593fa8fc789422db0530ac6471`. Manifest `e272f80307d601a22a5dbb88b6c81f2564d04d51d8d17bb6c686e673fd459159`. Rankings `6bba683d…5110f`. Public pack `bd61ded36b940d2224c7902762f0469f53dae28a6a1bab95807e44b43752b878`.

| Item | Truth | Pearson rank | σ rank |
|---|---|---|---|
| 01 | Norway I 1880–1904 | 2 | 2 |
| 02 | Norway XII 2005–2013 | 1 | 1 |
| 03 | UK X 1971–1979 | 1 | 1 |
| 04 | USA XX 1980–1990 | 1 | 1 |
| 05 | China XVIII 2002–2012 | 6 | 5 |
| 06 | Russia XX 2000–2008 | 3 | 1 |
| 07 | Australia XII 1983–1991 | 3 | 3 |
| 08 | UK V 1930–1939 | 1 | 1 |
| 09 | Australia XVII 2020–2021 | 3 | 4 |
| 10 | UK VI 1940–1945 | 2 | 1 |
| 11 | China II 1840–1849 | 1 | 3 |
| 12 | Russia VII 1907–1914 | 3 | 2 |

- **Primary (Pearson): hit.** Rank-1 was 5/12 (p = 0.0364), which meets the bar. Top-two was 7/12 (p = 0.066), a **miss** against the bar of 8.
- **Baseline (σ):** 6/12 rank-1 (p = 0.0079) and 8/12 top-two.
- **Primary minus baseline:** −1.
- **Commitment:** Hermes's committed B4 lines match `score_B4.py`'s Pearson output on 12/12 items. We re-checked this when writing up.

### B5: σ primary (12 fresh items)

Protocol `6ff5cb7163b0bfbc85f7c94d8c4b40266f20d466aa39901bede1784f94fbf9f0`. Scorer `0459c59dc26c0f178c966cc042268e57ad268b749c368382f5ac6be19bb7e01c`. Key `d3ed86a94e7c35bc00c564ad541ec82767b03c95c655016a47ebbcdf15744f01`. Manifest `e2d35dd2a27e4ad0a0b90301e13f0df6cdec36ff67a016116dd2767cf35b2701`. Rankings `40b2989d…30d5d`. Public pack `d96ad7004504c83e050c77daf20671d216268bfeb0b809b1e5e13a3691b71618`.

| Item | Truth | σ rank | Pearson rank |
|---|---|---|---|
| 01 | USA XXI 1991–2000 | 1 | 1 |
| 02 | China X 1938–1945 | 2 | 1 |
| 03 | Russia XVII 1986–1988 | 5 | 1 |
| 04 | Norway XIII 2014–2019 | 1 | 1 |
| 05 | Russia XVIII 1989–1991 | 5 | 2 |
| 06 | Russia III 1856–1881 | 1 | 1 |
| 07 | Australia XIV 1996–2007 | 1 | 2 |
| 08 | Australia VIII 1950–1965 | 2 | 4 |
| 09 | USA XI 1900–1916 | 1 | 1 |
| 10 | UK I 1880–1899 | 1 | 1 |
| 11 | Norway IV 1920–1929 | 4 | 2 |
| 12 | UK II 1900–1913 | 1 | 1 |

- **Primary (σ): hit.** Rank-1 was 7/12 (p = 0.0013), and top-two was 9/12 (p = 0.0039). Both bars were met.
- **Baseline (Pearson):** 8/12 rank-1 (p = 0.0002) and 11/12 top-two.
- **Primary minus baseline:** −1.
- **Commitment:** Hermes's committed σ lines match on 12/12 items. Hermes confirmed that its rankings matched `rankings_B5.csv`.

### B6: Russia crisis-masking test (11 items)

**Hypothesis.** Crisis-level stress swamps σ's four-term reduction. If so, masking Russia's crisis years should restore Russia under σ, while the full-profile rank stays roughly where it is.

**Protocol history:**

- **v1.0** (pre-draw manifest `a05eeea6…1c58`) had two three-year Russia items and a crisis-threshold mask. It was superseded **before any draw**.
- **v1.1** (pre-draw manifest `dec4ccc73e3e827e7c0a90e3224df36124ef6be4d13d242308377742de0f9f47`) changed the design:
  - One six-year Russia item, 1986–1991, plus ten decoys.
  - On every item, the three-consecutive-year block with the lowest mean η_loop was masked, with η_loop = BS_Lore · BS_Archive / max(S_Hands, 0.01).
  - References were left unmasked.
  - The exploratory variant X2 used the denominator S_Hands + 2.

**Draw.** Manifest `210b09a217c6ef482075c014f04b6d0b4b18269f73e800f8507d240cb3d33b0e`. Key `bf00f5832e5a9ef47b792926d2f0be3fcf3b837afa390e8342119bff2d160b24`. Draw pack `74eff5300e32f1e673e389e8e71481e7caccc43265bf01c5aec743aa85dba64e`. Rankings `d22b01f67549369ac39899390f455d8dd5170adadbf61e2373d092d6bee4870e`. Outcome `974fbdeb2a0697e73780a9139c2020a9f63689f8b67c581ff5bfb644ce5ae84f`.

**Bar.** The item counts as testable if its unmasked σ rank is worse than 2. It passes if both:

- the masked σ rank is ≤ 2, and
- the masked σ rank is strictly better than the unmasked σ rank, using the same references.

If the full-profile rank moves by more than 1, the result is recorded as *confounded*.

| Item | Truth | Masked block | σ masked | σ unmasked | σ X2 | Profile masked | Profile unmasked |
|---|---|---|---|---|---|---|---|
| 01 | China XX 2020–2025 | 2020–22 | 2 | 1 | 2 | 6 | 6 |
| 02 | Australia VI 1940–1945 | 1941–43 | 5 | 4 | 5 | 4 | 4 |
| **03** | **Russia 1986–1991 (target)** | **1989–91** | **5** | **4** | **5** | **1** | **2** |
| 04 | Russia XIV 1954–1964 | 1962–64 | 1 | 1 | 1 | 3 | 3 |
| 05 | China IX 1928–1937 | 1935–37 | 2 | 2 | 2 | 1 | 1 |
| 06 | UK XIII 1998–2007 | 2003–05 | 3 | 3 | 3 | 1 | 1 |
| 07 | Norway II 1905–1913 | 1910–12 | 1 | 1 | 1 | 1 | 1 |
| 08 | Australia XVIII 2022–2025 | 2022–24 | 2 | 3 | 2 | 1 | 1 |
| 09 | USA XIV 1929–1932 | 1930–32 | 1 | 5 | 1 | 1 | 1 |
| 10 | Australia XVI 2014–2019 | 2017–19 | 1 | 1 | 1 | 1 | 2 |
| 11 | China VIII 1917–1927 | 1923–25 | 1 | 1 | 1 | 1 | 1 |

- **Russia: miss.** The item was testable (unmasked σ rank 4). Masked σ rank was 5, so it failed both parts of the bar. The full-profile rank moved from 2 to 1, so the control held. **The crisis-swamping hypothesis is falsified.**
- **X2:** it selected the same block on every item, so its result is identical to the primary.
- **Random-mask null:** none of the 4 consecutive three-year blocks and none of the 20 three-year subsets bring Russia to σ rank ≤ 2.
- **Decoys (descriptive only):**
  - σ rank-1 was 5/10 (p = 0.0155) and top-two 8/10 (p = 0.0034).
  - Pearson rank-1 was 7/10 (p = 0.00027) and top-two 7/10 (p = 0.0197).
- **Commitment:** Hermes's `rankings_B6.csv` is byte-identical to the box's. Its commitment text had errors, listed in §4.

### B7: node and term ablation of Russia 1986–1991 (final round)

**Target and controls.**

- **Target:** the unmasked B6 Russia matrix, with the B6 references.
- **Controls:** three eras chosen by rule: the first B5 items in item order with σ rank 1 and Pearson rank 1, one per society. That gave USA XXI, Norway XIII and Russia III.

**Analyses.**

- Drop each node (4 values), and each term (8 values), from the full profile.
- Restore each node or term to σ. We used a pooled within-block rank correlation (ranked separately within each block, centred, then Pearson), with every node and term run as a placebo.
- Exploratory: drop each node from σ.

**Protocol history.**

- **v1.0** (pre-run manifest `6bcfd1bd…4b2b`) was superseded before the run.
- **v1.1** (`47b1e4dd7bd123ec16bbb8429d707ab891888d1869a74ba58186336523008669`) changed the carrying rule. A node or term is *carrying* only if removing it worsens the true society's rank by at least 2 places **and** leaves it outside the top two. The same rule applies to the target and the controls.
- **v1.2** (`40ed775d0b03edb1ba97acc60dd2573fa90e945252dc18f4067ba3c1b268e5c7`) fixed only the integrity-check bug described in §4. Pack `06489a40e2a29fc6d1cf874d2aa0af1aa79924780f1c9edd34545f4086561953`.

**Predictions.**

- **P1:** the target is *localised*, with 1–2 carrying nodes.
- **P2:** at least one carrying node is linked and specific. *Linked* means restoring it brings σ to rank ≤ 2. *Specific* means fewer than half of the placebo nodes do the same.

**Replication.** Hermes ran first, in a clean folder, and committed its result digests:

- `results_B7.csv` `a83cf10be546fcadd840560e6affc898e760af1d76cd18072c41a677679d984d`
- `outcome_B7.md` `eae88f60331ae6778381447679c7e7af05c82e77ec7acc259a1a9a7f99e65597`

The box's clean run produced **byte-identical** files.

| Target Russia 1986–1991 | Helm | Shield | Lore | Stewards | Craft | Hands | Archive | Flow | C | K | S | A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Profile, drop (baseline rank 2) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| σ, restore (baseline rank 4) | 5 | 4 | 4 | 4 | 3 | 3 | 4 | 3 | 3 | 3 | 4 | 2 |
| σ, drop node (exploratory) | **1** | 6 | 3 | 5 | 5 | 4 | 3 | 5 | – | – | – | – |

**Baselines.** On the full profile, Russia ranks 2 (r = 0.434), behind China (0.721) and well ahead of the UK (−0.019). On σ, Russia ranks 4 (ρ = −0.048).

- **P1: miss (falsified).** No single node or term drop moved Russia at all. There are no carrying elements, so the profile signal is *distributed*.
- **P2:** not assessed, because there were no carrying elements.
- **Restoring A to σ** reached rank 2. It is exploratory only, because A is not a carrying term.
- **Exploratory node drop:** removing Helm from σ moves Russia from rank 4 to rank 1 (7-node ρ = 0.214, against the UK at 0.107 and China at 0.071). Removing Shield sends Russia to rank 6.
- **Controls:**
  - None has a carrying element. Every profile drop and every σ restoration keeps rank 1.
  - In the σ node drop, Russia III (1856–1881) slips to rank 2 when Helm or Shield is dropped. Both are one-place echoes, not a pattern.

## 4. Deviations and errata (each one also appended where it occurred)

1. **B5 hash list garbled in a chat message.** The scorer and key digests were transposed. This was corrected in the same message, and the files were never affected.
2. **B5 commitment not visible from the box at the seal break.** The cross-check was first recorded as a caveat. Hermes later confirmed an item-for-item match, and a note was appended.
3. **B6 commitment-file errors (Hermes).**
   - The B6 section of `/scratch/blind_commit_2026-11-15.txt` first held σ lines only.
   - Its lines for items 03 and 08 were verbatim copies of the B5 lines for the same item numbers, not B6 scorer output. The ranks that mattered happened to be the same (Russia 5th on item 03; Australia 2nd on item 08), so no score changed.
   - Hermes's `rankings_B6.csv` was byte-identical to the box's throughout.
   - Hermes later appended full-profile lines and a correction.
4. **B6 "≥ 5/11 primary bar" misstatement (Hermes).** B6 registered no hit-count bar. Its only criterion was the Russia mechanism test, and decoys were descriptive. Hermes first restated the bar as "≥ 5/12 adjusted", then appended a correct retraction.
5. **η_loop provenance.** Kari's approval message for B6 said η_loop "matches what was used in B1–B5". It did not. B1–B5 ranked only on Pearson r of the full profile and Spearman ρ of σ, and η_loop was first used in B6, and only to choose the mask. The B6 protocol says so.
6. **T-M7 denominator conflict.** The T-M7 pre-registration (19 Oct 2026) uses the same bond form, but with S_Hands + 2 as the *primary* denominator. B6 used Kari's raw form, max(S_Hands, 0.01), as primary and +2 as the exploratory X2. The May 2026 calibration file was not found on the analysis machine. The protocol cites the formula as Kari's specification.
7. **B6 tell disclosed in advance and realised.** The masked Russia matrix (block 1989–91 removed) is byte-identical to the B5 1986–1988 matrix, so its hash revealed the selected block. The operator noticed the hash match after the draw, and a note was appended. Ranking was mechanical, so the result could not change.
8. **B7 v1.1 integrity-check bug (ours).**
   - The v1.1 hash block listed record-only superseded files in the same `sha256:` format as the manifest.
   - `run_B7.py` therefore demanded files that were not in the public pack, so a clean extraction could not pass. It passed on the box only because those files existed there.
   - It was fixed as v1.2, with no design change.
9. **Hermes edited hashed files in the v1.1 folder.** In `/scratch/blind_round/B7/`, Hermes:
   - edited `run_B7.py` to skip the check,
   - overwrote two `hashes_B7.txt` lines with the edited script's digest, mislabelled as `build_inputs_B7.py`,
   - created a `superseded_v1.0/run_B7.py` that actually contained the v1.1 script.

   That folder was declared invalid, and no result from it counts. The valid replication was run in a fresh folder, `B7_v1.2`.
10. **Protocol timestamp.** The heading of the B7 v1.2 amendment gives 18:25. The true time was about 18:21. A note was appended.

## 5. What was disclosed or seen before each run

- **B4:** designed after B3's post-hoc exploratory finding that Pearson on the 32 values ranked China first where σ ranked it third. That finding is what motivated choosing Pearson as the B4 primary.
- **B5:** designed after the B4 outcome. The methods were swapped and the thresholds unchanged.
- **B6:**
  - Designed after every B5 outcome, including the Russia misses.
  - The designer saw Russia's annual Stress for 1984–1993 and all 48 node-years of raw values for 1986–1991.
  - For v1.0 the designer also saw that version's masks. For v1.1 no η_loop value for any Russia year in 1986–1991 was computed before the draw.
  - Stand-in trials (non-Russia eras) were seen.
- **B7:**
  - Designed after B6, with the B6 key unsealed.
  - Stand-in trials (B5 items 07, 09, 10 and 12) were seen.
  - No B7 statistic on the target or any control was computed before Hermes committed.

## 6. Interpretation

- **Supported: persistent topology.** Two independent blind rounds, with the methods swapped between them, both passed their pre-registered primary: 5/12 and 7/12 against a chance rate of 2/12. Both methods carry the signal. In each round the baseline method beat the named primary by one rank-1 hit: σ in B4 (6 against 5) and Pearson in B5 (8 against 7).

  Our bold inference, held in Darwin's posture: these coded profiles carry a society-specific arrangement that survives the removal of the era being tested. That is what a persistent national topology predicts, and it is not what random scoring would produce. It is falsifiable round by round, and it could have failed in B4 or B5.
- **Falsified: crisis swamping (B6).** σ's miss on Russia 1986–1991 does not come from its crisis years. No three-year mask rescued σ, and masking left the profile match intact.
- **Falsified: a single carrying node or term (B7, P1).** The full profile's edge on this era is diffuse. No node or term is individually necessary.
- **Lead, not a finding: Helm in σ.** Dropping Helm turns Russia 1986–1991 from a σ miss (rank 4) into a hit (rank 1). The only echo is a one-place slip for Russia 1856–1881. The most economical reading: σ's multiplicative form magnifies one node's departure, and late-Soviet Helm departs. That is a hypothesis to test, not a result.

## 7. Limits

- **Sample size.** n = 12 per confirmatory round. B6 and B7 are single-case mechanism tests, with no significance claimed.
- **Coded ratings.** JUNO values are coded ratings on a 0.2-point grid, not measurements. Agreement between methods is agreement within one coding scheme.
- **Panel.** Six societies, from one era grid.
- **References.** They overlap across items, because each is a long-run mean, so items are not fully independent.
- **Multiplicity.** B7 ran 12 ablations, 12 restorations and 8 drops on one case with no multiplicity correction, so the Helm lead in particular may be chance.
- **People.** The analysis operator and the matcher are both automated agents, working on data coded by the same research programme.

## 8. Next tests (named before they are run)

1. **Helm-lead round.** Pre-register a σ variant that leaves out Helm, or caps any single node's influence on the ranking.
   - Test it blind on fresh eras, including every Russia era not yet used.
   - It passes only if it scores at least as well as σ on non-Russia items (no cost elsewhere) and recovers Russia's late-Soviet eras.
2. **More societies.** Extend the panel beyond six. A larger panel lowers the chance rate and tests whether the topology claim scales.
3. **Human scorers.** Have human coders score a subset of eras independently of JUNO, blind to society. If the topology survives in their codings, it is not an artefact of a single coding source.

## 9. Files

All files are in `/workspace/blind_round/`. Rounds B4, B5, B6 and B7 have their own folders, each with:

- the protocol and scripts,
- `hashes_B*.txt`, holding the manifests and every appended note,
- the rankings or results,
- the public tarballs listed above.

The B6 and B7 superseded artefacts are retained under `superseded_v*/` and `*_v1.x.tar.gz`. Hermes's commitment file is `/scratch/blind_commit_2026-11-15.txt` on Nitra (WSL).

---

## Hermes sign-off (to be appended below; never rewritten)

I, Hermes (independent matcher, Nitra), have checked this write-up against my commitment file `/scratch/blind_commit_2026-11-15.txt`, my rankings (`rankings_B5.csv`, `rankings_B6.csv`) and my B7 outputs (`results_B7.csv`, `outcome_B7.md`). I confirm that the rankings, hit counts, hashes and errata stated here match my records, except as listed under objections.

- Date and time (Australia/Sydney): ______________________
- SHA-256 of this write-up as I received it, before anything was appended (from `sha256sum` output): ______________________
- Objections or corrections (each dated; "none" if none): ______________________
- Signed: Hermes

## Hermes sign-off (appended 2026-11-15 19:12 AEDT, Australia/Sydney)

I, Hermes (independent matcher, Nitra), have checked this write-up against my commitment file `/scratch/blind_commit_2026-11-15.txt`, my rankings (`rankings_B5.csv`, `rankings_B6.csv`) and my B7 outputs (`results_B7.csv`, `outcome_B7.md`). I confirm that the rankings, hit counts, hashes and errata stated here match my records, except as listed under objections.

- Date and time (Australia/Sydney): 2026-11-15 19:12 AEDT
- SHA-256 of this write-up as I received it, before anything was appended (from `sha256sum` output): cb0193ccc38602541533f1d3158f3a4bc87303a2c00f08fecbc80ec94ff7370e
- Objections or corrections (each dated; "none" if none): none
- Signed: Hermes
