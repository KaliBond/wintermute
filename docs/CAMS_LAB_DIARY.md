# CAMS Lab Diary

*Kari McKern. Written 5 November 2026, from the git history of `KaliBond/wintermute`.*

> **How this was made.** This diary was drafted by Claude (Sonnet 5.5) from the repository's git history, in my voice, for me to read, correct and own. I read every one of the 834 commit subjects on `main` (up to `709342c`), and the full text of the commits and files named below. I did not open every file in every commit. Where a date comes from a document and not from git, I say so. Where the record does not tell me why I did something, I say that too, and I mark it **[unclear]**. Where I could not check something, I mark it **[not checked]**. No statistics here are new: every number is quoted from a commit message or a file in the repo, with the short SHA beside it.
>
> **Addendum, 8 November 2026.** Section 12a was added by Claude (Opus 5.5) from the 22 non-merge commits on `main` after `709342c`, up to the merge `aebb28a`, and extended the same day to `b8fbd82` for the Zenodo lodgement. Every commit message was read in full; files were opened only where a number is quoted. Where an earlier entry has been overtaken, I have added a dated qualification beside it and left the original wording standing.
>
> The house rule applies, and it is the reason this diary exists: **we don't edit the past, we qualify it.** Nothing in the repo has been changed to make this story tidier. Where two documents disagree, I leave both standing and say which I currently trust.

---

## Contents

0. Reading notes
1. Phase 0: Before the repo (mid 2024)
2. Phase 1: Text files and a nine-node model (Oct 2024 to Jun 2025)
3. Phase 2: THERMO, CAN, v2.1 and a website (Aug to Sep 2025)
4. Phase 3: Dashboards, and the neural network goes (Dec 2025)
5. Phase 4: Datasets, tools and the name keeps changing (Feb to Apr 2026)
6. Phase 5: Papers, telescopes and a rename in both directions (Apr to May 2026)
7. Phase 6: JUNO arrives and v3.2 is replaced (Jun 2026)
8. Phase 7: Blind tests, v1.2-Final and the first sceptical audits (Jul 2026)
9. Phase 8: Audits, a bond bug and the great clean-up (Aug 2026)
10. Phase 9: Thermometer, not crystal ball (Sep 2026)
11. Phase 10: Deep time, v3 and the erratum (Oct 2026)
12. Phase 11: The first week of November
12a. Addendum: 5 to 8 November (written 8 November)
13. Taxonomy ledger (nodes, metrics, formulas, names, categories) up to JUNO v1.2
14. Retired, failed, reversed and withdrawn: a list
15. What I still do not know

---

## 0. Reading notes

- **Dates.** Commit dates are the author dates git reports. Merge commits sometimes carry a date earlier than the commit they merge (for example `9103057`, 27 Aug, merged a fix made on 28 Aug), because of how the PR was recorded. I have used the date of the work, not of the merge.
- **Two sources for the early story.** The first commit is 4 October 2024 (`09ce7f5`). Everything before that, and every version number before `v1.0-RC1`, comes from `docs/VERSION_HISTORY.md` (20 August 2026) and `canonical/CAMS-CAN-MASTER-REFERENCE.md`, which are documents written later, not from dated git objects. I treat them as my own account, to be qualified by git where git can speak.
- **Voice.** I have kept it plain. Where something was funny at the time, I let it be.

---

## 1. Phase 0: Before the repo (mid 2024)

**What the record says.** `docs/VERSION_HISTORY.md` records, citing the project-history page of April 2026, that CAMS began "around July 2024", in conversation with an AI, from a Socratic itch: how do you tell knowledge from persuasion? It was not a research plan. The same document records a repository origin of 27 September 2024 (taken from the master reference) and a first public appearance in *Pearls and Irritations* on 14 October 2024, "independently verified externally" by that document. **[not checked]** by me here.

**What git can say.** Nothing earlier than 4 October 2024. The "27 September" in the master reference and the 4 October of the first commit are a week apart; `VERSION_HISTORY.md` itself notes that the first commit is 4 October. I take 4 October 2024 as the first dated object and nothing earlier as dated.

---

## 2. Phase 1: Text files and a nine-node model (Oct 2024 to Jun 2025)

### 4 October 2024 `09ce7f5`: "Add files via upload"

Three text files, 507 lines. No code. This is the origin of the repository and, as far as git goes, of CAMS. What was in them matters for the taxonomy story, so I read them:

- **Nine nodes, not eight.** `parameters performance .txt` and `experimental approach .txt` list: the Executive, the Priesthood, Property Owners, the Army, Trades and Professions, the Proletariat, State Memory, **State Checksums** (public discourse, intellectual merit) and the Storekeepers. That is nine.
- **Four dimensions, with a different fourth.** Node Value is defined as `NV = Coherence + Capacity + Stress + Impact`. A plain sum. Not Abstraction, not yet. "Capacity for Rational Abstraction" appears, but as a society-level quantity (writing systems, record-keeping, number of active scholars), not as a node dimension.
- **A loose, evaluative vocabulary.** Nodes are scored for how well they perform, with words I would no longer use in a definition (see the style note in section 15).
- **Biology and cybernetics.** The framing is explicitly evolutionary: selection pressure, emergence, subsystems "akin to organs", cellular automata and game theory. The Executive is "Brain", the Army is "Muscles".

### 7 November 2024 `d7dd654`: the checksums node is deleted

"Update parameters performance .txt" removes the entire section 8, *Checksums of Intellectual Integrity*, and renumbers the Storekeepers from 9 to 8. **This is the dated git evidence for the nine-to-eight reduction.** It matches what I told `VERSION_HISTORY.md` on 20 August 2026 (the predecessor was a nine-node model with a "checksums" node that did not work). It does not match the date that document gives for the eight-node adoption ("Jul 2025, RD-002"). The honest reading is that the checksums node left the working text in November 2024 and the eight-node set was only declared canonical in the master reference in 2025. I did not find a commit that records why it "didn't work". **[unclear]**

### 15 November 2024 `5b1e00b`: the first README

It spells the acronym "Common Adaptive Model State" and says the framework has "Nodes: Coherence, Capacity, Stress, Activity". Two things are off. Those are metrics, not nodes. And the fourth is "Activity", where the October text said "Impact". I cannot tell from the record whether "Activity" was a slip, a rename or a third candidate. **[unclear]** It is the first of many small naming wobbles.

### 1 March 2025 `cdecc11`, `b48fea0`: "Quantification" and "Quant2"

`Quantification` rescores the dimensions as −5 to +5 and drops to **three** metrics: Coherence, Capacity, Stress. The node names are the sociological set: Executive, Army, Proletariat, Priesthood. And look at the polarity: "**Stress: +5 Resilient and stable; −5 under significant strain.**" That is the opposite sign convention to anything later. High Stress was good. `Quant2` is an AI reply pasted into the repo (it opens "Your document presents a fascinating conceptual framework"), proposing richer definitions. I kept it. It is a record of how I was working then: ask, paste, keep.

*Qualification for later:* `VERSION_HISTORY.md` says stress was "never negative as a construct" and only recorded negative as a calculation convenience. The March 2025 text reads more like a polarity in its own right (+5 means resilient). I do not think the two accounts are in conflict, but they are not the same thing, and a reader should know both exist. **[unclear]**

### 5 May 2025 `61b18b7`, `4e6c966`, then `4982071` on 15 May

On 5 May I added 62 lines to `experimental approach .txt`, then removed 3 of them on the same day (`4e6c966`). The removed lines included a claim of "83% predictive accuracy" for Greece and China 1900 to 2025 and a leftover system-prompt line from a pasted Grok session. So the 83% figure first appears in git as part of pasted AI output, and I took it out of the working text within the day. It then reappears, as a stated result, in the research diary (see Phase 3). I did not find where it was computed. **[unclear]** `4982071` ("more fixes", 15 May) rewrote `Quantification` and `quantitative .txt` by about 218 lines. **[not checked]** what changed.

### 13 June 2025 `d08cfa3`: the first dataset

`Australia_CAMS_Cleaned.csv`, 985 lines. This is the first scored data in git, and Australia is where I started.

**Phase 1 in one line:** nine nodes became eight on 7 November 2024, there were at least three candidate fourth metrics (Impact, Activity, and later Abstraction), stress pointed the other way, and the acronym had already appeared in a form I would later abandon.

---

## 3. Phase 2: THERMO, CAN, v2.1 and a website (Aug to Sep 2025)

### 5 to 7 August 2025: `e897075`, `1b39b0d`, `36858fc`

`CAMS-THERMO` (real-time thermodynamic monitoring, with parameters ALPHA 1.2, BETA 1.0, GAMMA 0.8, DELTA 0.9) and then `CAMS-CAN`, "stress dynamics" as **societal meta-cognition**, with a stress-information isomorphism, a three-part meta-cognitive vector (Monitoring, Control, Reflection), four attractors (Adaptive, Authoritarian, Fragmented, Collapse) and 8 institutional nodes. The THERMO commit message says negative stress values were handled as "beneficial stress" in the Australia data. That is a different gloss from "calculation convenience". Three descriptions of negative stress now exist in the record (polarity, beneficial stress, arithmetic convenience). **[unclear]** which, if any, I held at the time.

### 17 August 2025 `46460eb`: "CAMS Framework v2.1 Academic Implementation"

A big tidy-up commit: MIT licence, `CITATION.cff`, 74 files into `data/`, 32 cleaned datasets, "30,856+ validated records", and a framework described as "stress-modulated neural network dynamics". The code is the first place the **mnemonic names appear in git**: the node list reads `Executive  # Helm`, `Army  # Shield`, and so on, with Lore glossed as "Knowledge Workers" and Stewards as "Property Owners". So **both label sets are present together from the first appearance of Helm in git**, which supports what I said on 20 August 2026: two parallel label sets, not a rename. It sits uneasily with the dating in `VERSION_HISTORY.md`, which puts the Helm/Executive pairing in "Sep 2025 (v0.7)". The git date is earlier than the document date. I trust git on this one.

Same file: Abstraction is now a node dimension (`abstraction A in [0, 10]`), and Coherence and Capacity are normalised from a −10 to +10 range. So by August 2025 the metrics are C, K, S, A, but the scale conventions in the code still differ from later ones.

### 20 August to 18 September 2025: the website (`7cad5f8` to `2217775`)

neuralnations.org, a backstory page, attractor pages, a 3D attractor space (`561b43d`), a dashboard (`f8c0946`). Two side steps worth keeping: I added a traffic-tracking system on 25 August (`d625d98`) and removed visitor tracking two days later (`e290a60`); and on 4 September I converted the whole site to Australian spelling (`de68e6e`), which is why this document is not an embarrassment to me.

---

## 4. Phase 3: Dashboards, and the neural network goes (Dec 2025)

### 2 December 2025: Streamlit and the "engine bug squash"

`e28d09d` adds "CAMS GTS EV v2.0" with "Model Archaeology & Real-World Data". `875b96f` is titled "Fix critical bugs in CAMS engine v1.95 (canonical bug squash)", and `7293614` adds validation to stop array-dimension errors. Version numbers are already untidy: v2.1 in August, v1.95 and v2.0 in December. **[unclear]** how I intended these to order.

### 7 December 2025 `f4c1ec1`, `f671ef2`: the neural network hypothesis is falsified

I put a red banner on the research diary, rewrote the subtitle from "Societal Neural Network Discovery" to "Thermodynamic Societal Analysis", renamed `update_neural_dynamics()` to `update_bond_dynamics()`, and left the January 2025 entry in place with a correction notice. The commit message says the empirical findings "remain valid" (the 83% accuracy, the universal thresholds). It says the formulas were "preserved (reinterpreted as thermodynamic metrics)". `docs/VERSION_HISTORY.md` records what that entry originally claimed in full.

Two things I want on the record:

1. **I do not know what the falsification test was.** `VERSION_HISTORY.md` says, in so many words, that no external description of it was found. The commit announces a falsification; it does not contain one. **[unclear]**
2. **The 83% survived the hypothesis that produced it.** That is a qualification I would apply to myself: a number that outlives the model it was reported under needs its own provenance. It has none in the repo.

The word "bond" is already in the function name by this point, which is when the bond idea moves from neurons to thermodynamics. I do not read intent into the rename. **[unclear]**

---

## 5. Phase 4: Datasets, tools and the name keeps changing (Feb to Apr 2026)

### February 2026: 54 datasets and a sign problem (`1b88ae6`, 17 Feb)

54 CAMS-format CSVs, and a fix for "853 negative Stress values in Germany datasets (all 4 locations) by applying +10 offset; stress range now 0-9". So negative stress was still in the data in February 2026, nearly a year after the March 2025 polarity text. Then on 11 March, `f8a3864` adds India and Saudi Arabia to the datasets page with "(signed Stress convention)" in the subject. The signed convention is still in use for some files on 11 March 2026. The 20 August 2026 explanation (a convenience, standardised by RD-001 in Nov 2025) is therefore not the whole history of the data in the repo. **[unclear]** how many files in `data/` still carry signed stress. The July 2026 JUNO work uses unsigned stress. I note that as a thing the corpus has lived through, not as a defect I have audited.

Also in February: dDIG and the Dyad Field analysis (`ffaf0f1`), CLD Studio (`0f1f364`), a phase-space attractor viewer (`5a61f48`) and a long run of small fixes (`8e76386` to `c0f5c5a`), several of which are the same bug turning up in new places: hard-coded node names (`72cf9c0`: "auto-detect nodes instead of hardcoding"), missing columns, a Streamlit redeploy (`4aaf0f0`). That cluster is what tool-building looks like when the node list is not fixed in one place.

### 7 to 11 March: v2.3 and the housekeeping of names

`659962a` and `bafb8a3` add the "CAMS v2.3 formal specification". On 8 March I stripped the old GTS EV and GT labels and made the dashboard simply "CAMS Dashboard" (`1444596`, `9b4a236`, `d5db973`, `e9b56d6`). I also removed "unverified accuracy claims" from the CAMS page (`90ef8f0`). On 10 March the Explore page went from 46 artefacts (`a5d5ae1`) to 103 (`6a37d4f`) to 146 (`8f78489`) in one day. I remember none of it, which is a fair measure of how that day felt.

### 13 to 14 March: "Why Eight Nodes?" and a link that came and went

`acd8bb4` adds a rate-separation paper on why eight; `99d76e4` adds an appendix on the eight-node optimum; `969704d` and `9fd62e1` remove the appendix link from Explore on 14 March. The appendix file is still in the repo (`appendix-eight-node-optimum.md`). I do not know why I pulled the link. **[unclear]** What I can say is that the eight-node claim has been argued from three directions in the repo (rate separation in March, coordination problems in the May taxonomy paper, and clustering in the same paper), and the first of those has the weakest public standing.

### 21 to 22 March: the Reconstructor, and its civilisation types

`f8156bc` adds the CAMS 3.0 Reconstructor. In one day it gains a five-category signal engine (`25d20c6`), a civilisation classifier with **Types I to IV** (`68fa5aa`), an auto-run analysis (`dc7e2e4`) that I reverted within the day (`e62ac13`), and a reworked classifier (`5c9b18b`) that replaces the Types with a **CAPE archetype** (Temple, Marketplace, Workshop, Commons, computed from `c_n = (C x A)/(1 + S)` and two differences across node pairs) plus a "dyadic state-variable" type. On 22 March, `2aeaf70` calibrates thresholds "from corpus (n=3494 society-years)". The tool was then renamed Interpreter (`ff4880e`, `1c6f8e9`) with a redirect (`72050b2`). That is at least three category schemes for societies in two days (Types I to IV, CAPE archetype, dyadic type), and I do not think any of them survived under those names. See the ledger in section 13.

The society count in this period also wobbles: 42 nations on 22 March (`f0ade07`), then "Fix society count to 38 across all pages" on 26 March (`940aece`). **[unclear]** whether that is a deduplication or an error in one of the two counts.

### 30 March to 23 April: scale invariance, then scale covariance

A "Scale Invariance Experiment" with blinded datasets (`b97fcb0`, `86feeeb`, report `d5cd37d`). On 23 April, `b204fa6` replaces "scale-invariant" with "scale-covariant" everywhere. **I did not find the reason in the commit.** **[unclear]** The qualification is the word itself: I stopped claiming invariance. (Note for honesty: the October thesis, `docs/theory/CAMS_Organisational_Pathologies_Thesis.md`, line 6, still says "scale-invariant properties". The vocabulary relapsed. I have not edited it.)

### 6 April: the name, once

`3ed9782`: "Correct framework name to Complex Adaptive Model State (CAMS)", replacing "Complex Adaptive Model of Societies". I will come back to this. It is round one.

### 9 April `62c6795`, `c78cebb`, `ea75848`: the canonical folder

`canonical/` appears, with `CAMSFORMULATIONS.MD` (a formulations history and lineage document) and `CAMS-CAN-MASTER-REFERENCE.md` (the "single source of truth", release candidate, last updated 8 April). This is where the version table that `VERSION_HISTORY.md` later leans on was first written down, which means the early version dates (v0.1 to v0.9) are my later reconstruction and not contemporaneous.

---

## 6. Phase 5: Papers, telescopes and a rename in both directions (Apr to May 2026)

### 15 to 19 April: the cognition model and the focus/energy pair

`324ed89` adds a Subbrain Activation tab. `974d564` (18 April) changes the explorer charts: **"radar = A x C, bar = K vs S"**. This is the first place in git where the two node-level quantities I now use for social cognition, A times C and K minus S, appear side by side as the way to look at a node. `93839bf` adds "v3.2-R metrics: ESCH sigma, kappa, headroom, attractor + node table". Per `VERSION_HISTORY.md`, v3.2/v3.2-R was "the social cognition model", with ESCH (σ) as its main metric. The Zeitgeist Detector follows (`ee77995`, `ec2648c`). JUNO would later replace it outright (Phase 6).

### 29 to 30 April, 1 to 2 May: Papers 1 to 3

Paper 1 (`6e33f42`), Paper 3 on the Extended Evolutionary Synthesis (`55ff6f0`), two Grok session summaries on paper development and taxonomy (`9742b42`, `823bb7f`), and **Paper 2, "A Functional Taxonomy of Human Social Organisation"** (`b9409ed`, `c88fc8f`, `c6a7a44`). That paper does the thing I had not done before: it argues the nodes from the bottom. One node per recurring coordination problem (direction, boundary, meaning, allocation, technical reproduction, labour reproduction, memory, circulation), geography as the thing that makes some problems acute, path dependency as the way local answers harden. It reports eight clusters from Ward clustering of 229 polities and lists seven falsifiability tests. No commit I read reports any of them run. **[not checked]** whether any have been.

### 28 April `53bf019`: the USA prospective test

Sealed on 28 April, adjudication window 2026 to 2028. Its bond numbers are on a scale the 20 October erratum would later say cannot be reconciled with anything else in the repo (Phase 10).

### 3 to 6 May: the Telescope

`telescope_1.html`, `telescope_2.html` with extra societies injected, a CLAUDE.md (`723c8a3`), a roll button, then a React build deployed to `cams-telescope/` (`8d7853b`) and a restoration of "verbose dossier text" (`424c445`). It is a good example of a tool I rebuilt three times because the data shape kept moving under it.

### 17 May `5ecee1d`: the global rename, round two

"Global rename: CAMS = Complex Adaptive Model System." 55 files, 77 replacements, replacing both "Model State" and "Model of Societies". So after 6 April chose "State", 17 May chose "System".

### 29 May `36f7745`: the global rename, round three

"standardise CAMS acronym expansion to 'Complex Adaptive Model of Societies' sitewide": 61 HTML files, two instances kept and wrapped as "historical label". So the 6 April removal of "of Societies" was reversed 53 days later, and the 17 May "System" was replaced twelve days after that.

### 20 August `d5b1b38`: the name, resolved

"Author confirmed 2026-08-20", canonical: **Complex Adaptive Model State**. `docs/VERSION_HISTORY.md` records my explanation: the variants were produced by different AI assistants renaming the acronym on their own initiative while drafting, which I called "a bit of a joke". The git record shows what that looks like in practice: three global find-and-replace commits in under eight weeks (`3ed9782`, `5ecee1d`, `36f7745`), all committed under my name, the first two with a Claude co-author trailer, each described as the correct, canonical or standard name. I accept the explanation; I also accept that I approved the commits. Both are true. The six variants stay in the table in `VERSION_HISTORY.md`. Even so, **the "Short History" page (`3367cc4`, 15 September) still calls it "Complex Adaptive Model of Societies"**, which is the one I thought we had finished with. I have not edited it.

### May 2026: Epiphenomenon@Trove, and the datasets learn to be pairs

`2fbe78c` (16 May) to `54e528a` (21 May, "Falsification Results v5"). I will not summarise those results here. **[not checked]** the contents of that paper for this diary. On 27 May the datasets were renamed to an `X_ENS_YYYY_YYYY` / `X_ENV_YYYY_YYYY` convention (`30f1b77`), ensemble means paired with envelopes, which carries the five-scorer ensembles through to everything after.

### 29 May `590d091`, `9188b4e`, `c3d537f`: the v2.3 pipeline and a bond specification

`cams_framework_v2_3.py` becomes the "canonical NV and BS computation pipeline", and `bond-strength-spec.html` the "canonical v2.3 Bond Strength specification". The fix a day later corrects the spec's "editor note, Step 2 formula, version history". The v2.3 bond was not stable even in its own spec page. It would be retired on 9 June.

### 23 to 31 May: attractors, social cognition, affect

A redesigned attractors page in "M, Y, Lambda" coordinates (`0f161f1`, `021a0aa`), the Cognitive Affect Map (`3915a7d`, `94bd707`) and, on 2 June, **"Social Cognition in Real Time"** (`2a16a6a`), a canvas instrument for the per-node quantity that `canonical-definitions.html` now writes as σ = A·C·(K − S).

---

## 7. Phase 6: JUNO arrives and v3.2 is replaced (Jun 2026)

### 9 June 2026: the day the formulas changed

The sequence matters: `2b7384a` "Pre-Cowork baseline", `20949c6` "pre-recalc", then **`1083b2c`: "recalc: CAMS v2.4 (JUNO) batch recompute"**, `5a89073` "post-recalc-v2.4", `444605b` "calibrate". The commit message of `1083b2c` is the cleanest statement of the formulas at that moment:

- Node Value: `V_i = C + K - S + 0.5*A`
- Bond Strength: `B_ij = sqrt(q_i*q_j) * 2^(-(S_i+S_j)/10)`, bounded to [0, 1]
- ESCH: `sigma_i = (A*C/100)*(K-S)`, "floor only at K==S"
- "v2.3 formula retired; frozen in `cams_framework_v2_3.py`"
- Three data tiers: **CANONICAL** (15 ENS and 14 ENV pairs), **USP** (20 files, valid single-pass, standard nodes), and **DEV** (34 files, "legacy pre-CAMS-8-node taxonomy").

That last tier is worth sitting on. As late as June 2026, 34 files in the corpus were classed as using an earlier node taxonomy than the eight. I did not list in the commit which taxonomy they used or how they differed. **[unclear]** It is the only place in git where "pre-8-node" is named as a data category. It tells me that the nine-to-eight change of November 2024 did not clean the data. It only fenced it.

`444605b` then validates the σ gate at −0.85 against "2,377 deduplicated society-years, 23 societies". `VERSION_HISTORY.md` adds that the pre-v2.4 ESCH had a blanket clamp that forced the metric positive and silently disabled the −0.85 Local-Node-Failure trigger, and that the clamp's introduction date is not recorded. **[unclear]** how long the clamp had been in place.

### What JUNO replaced, in the author's words

Per `VERSION_HISTORY.md`: JUNO replaced v3.2 "outright", and was named for Juno, who sees through clouds, with a nod to the NASA probe. It is a succession and not an upgrade. I note that the version numbers go *down* here (v3.2-R in April, v2.4 in June, "v1.0" for JUNO later in June). That is because JUNO started a new lineage. It looks like a regression in a list, and it is not one, but it is confusing and I have not tidied it.

### 17 to 20 June: the toggle that wasn't

`5843587` adds a JUNO-1.0 formalism toggle. On 17 June I was wiring it into the explorer (`8ebf7b0`), the zeitgeist, interpreter and compare tools (`5486549`), then reverting it from the explorer and interpreter because the `cams_index.json` schema was incompatible (`ab4970d`), re-adding it to the explorer (`483862b`), restoring it to the interpreter "with correct schema" (`a34a5a9`), and on the 18th wiring "5 stub toggle tools" (`fae08a5`). On 19 June, `dde073e` puts the "v1.0-Final formulation" at the top of the model page. All the same, a revert in the middle of a feature is normal; I am putting it here because it was the first time the data schema and the formulation visibly disagreed, and it won't be the last.

### 16 June: the lavender problem (a side step)

A run of commits (`145c32a` to `ea867a6`, and `b092ece`) goes through the site removing cream and teal because on my display the cream `#F5F1E4` read as lavender. I replaced it with white (`e386107`). The telescope's CLAUDE.md still specifies the cream palette as the design token. I did not reconcile those. **[unclear]**

---

## 8. Phase 7: Blind tests, v1.2-Final and the first sceptical audits (Jul 2026)

### 30 June to 5 July: the blind series

JUNO Blind Test and Series B (`e4a476c`, `567e931`); the **v1.1 Blind Test Assessment** (`b8d37f2`, dated 1 July): 8 societies unmasked (Germany, Norway, Russia, UK, Chile, Poland, Argentina, Australia), 1,201 society-years, 232 inflection points, "formulas validated at machine precision". A public blind challenge (`fab0e57`) and panel results (`cae1db1`, `09d7d75`). Those are claims about internal coherence of the operators, which `JUNO_v1.2-Final_Formalism.md` later said plainly: "compression consistency, not external validity".

### 1 July `b0668b2`: a falsification test pre-lodged

The annual panarchy and Holling-cycle test for Russia, China, USA and Germany: phase kinematics, K-dwell release depth, a revolt-cascade shuffle null, **registered before the analysis was run**. I did not follow the result through in this pass. **[not checked]**

### 6 July `464222f`: JUNO v1.2-Final

The 3,002 society-year blind-corpus test of v1.1 led to:

- the "s-arm" (`s_min <= -0.80`) deleted from Local Node Failure, because it was unreachable (zero firings on 3,002 society-years, everything it caught was already caught by Phantom Type II or higher);
- the Strained floor lowered from V̄ ≥ 6 to V̄ ≥ 4, closing a gap in which six society-years had fallen through to "Unclassified";
- a mandatory numerical policy: aggregates rounded to 6 decimal places before every threshold comparison, since inputs carry one decimal and thresholds like V_min = 4.0 were landing on floating-point noise;
- section 4's "validated instrument bounds" relabelled "reference ranges from audited panels", and λ₂ downgraded to a derived diagnostic, with its 0.972 correlation with mean bond strength written into the text.

That is a version that makes its own earlier wording smaller. I am pleased with that and I want it understood that it is the opposite of a polish.

The same day's `98305b8` publishes Blind Protocol v1.1.1; `295df96` adds the TPV-1 third-party validation protocol. The v1.5 pre-registration programme (registered 6 July per `VERSION_HISTORY.md`, added to the site on 14 July by `d39ab33`) retired the v14 `crisis_size = base x 1.5^duration` construction (divergent at a duration of 46 years).

### 12 July `a01a69c`: out-of-sample predictions, 2027 to 2028

A certification page and a PDF memorandum. **This is the document the 20 October erratum later qualifies** (bond scale, the word "guarantees"). It is also the document I would be most embarrassed to lose track of, so I am putting it in the diary now.

### 16 July `d959be4`: Stewards is corrected (a taxonomy fix in the docs)

`model.html` and `cams-validation-2026.html` had labelled Stewards "Bureaucratic administration / Civil service, regulators, fiscal apparatus". That contradicts the definition used everywhere else: capital, land and asset ownership. I also dropped a stray "bureaucratic" qualifier from Archive's tooltip in `cams-coordination.html`. This is the second time. `VERSION_HISTORY.md` records an earlier "canonical node-mapping correction" (v0.9, January 2026) fixing Shield as military and defence, not welfare institutions, and Stewards as landowners and asset management, not the welfare state, "changing the interpretation of Sweden's 1990s banking crisis". So the Stewards definition has been wrongly written as welfare or civil service twice, in the data (January) and in the documentation (July). I name this because it means any result involving Stewards before January 2026 must be read against that correction. **[not checked]** which dataset results were affected.

### 20 July `4289140`: the Paladin layer meets its audit

"When a Metaphor Earns Its Keep". This is the first time in git that "Paladin" appears. The report's claim, which I am quoting from the commit message and the page: **three of the four Paladin renderings** (the node that "dreams", the node that "can act", the node on which "the burden falls") **are relabelled maxima and minima**. The fourth, alignment (the correlation across the eight nodes between A·C/100 and K − S), is non-redundant and stable year to year, **but non-predictive: AUC about 0.48**. It was audited on the 3,002 society-year blind corpus and the 32,361 node-year Unified Dataset. The audit also caught its own mistake (a data-keying error that spliced unrelated society-years and made alignment look like noise), and said so.

The rule it states is the one I have tried to keep: *the operators measure; the metaphor renders.* I record this as a failed AUC in my own favourite layer.

### 29 to 30 July: PR-GAP-2 gets honest

`3bfd7e0` publishes PR-GAP-2 with a combined AUC of 0.652 on pooled ruptures. Then `39370b7` rewrites the site pages to match the stricter manuscript: pooled over all 126 ruptures, leave-one-society-out gives **AUC 0.563, not significant**; splitting by the sign of the JUNO expressed-cognition measure into deficit crises (n = 76) and coordination transitions (n = 41) lifts the deficit-crisis LOSO AUC to 0.740 (p < .001; 26 of 29 countries above chance). So the first number I published was 0.652, the one I now stand behind is 0.563 overall, and the interesting number, 0.740, is a post-hoc split that I made after seeing 0.563. I have said so on the page. It is still a post-hoc split. **[unclear]** whether the split was planned before the pooled result. Then `427b39a` (30 July) qualifies an overreach in the US study: an arithmetic error in the Archive node, a sourcing discrepancy on a survey figure, "confirmed" changed to "supported", and "causes" changed to "precedes" throughout.

---

## 9. Phase 8: Audits, a bond bug and the great clean-up (Aug 2026)

### 1 August `be95d1c`: three unbounded-growth modes

The CAMS/JUNO network ODE skeleton had three ways to blow up. I fixed all three. **[not checked]** what they were.

### 15 August `2e713ba`: the structural audit

Four findings, quoting the commit message: (1) the JUNO bond `B_ij` is **algebraically rank-1**, which is to say it holds no pair-specific information beyond per-node scalars; (2) a three-society inelasticity pilot found Stewards-Flow the most universal bond (I = 0.822) and Shield anomalously weak; (3) node response coefficient beta is stable across 36 societies (69.1% common-mode, 30.9% relational, 74.9% LOSO pair accuracy); (4) Kimi's 88% Archive-Lore FRACTURE/FREEZE claim **does not replicate at scale** (max 51.2%), though a 9.9x enrichment does; and Stewards confirmed as the leading political-economy crisis marker, AUC 0.930.

Number (1) is the structural problem with my bond idea, and I have not fixed it. `juno/CANONICAL_STATUS.md` says network visuals that treat B_ij as measured topology "should be labelled illustrative". My network visualisations (`ca058f1`, `67ba089`, `c85b905`) predate that statement. **[not checked]** whether they now carry the "illustrative" label.

### 17 August `6a84a2a`: Phase A external checks

SIPRI, V-Dem and Maddison, against JUNO_36: T1 (SIPRI military spending against Shield bond) **conditional pass** with real data (r = +0.251 cross-sectional, +0.121 within society), superseding a World Bank proxy that had returned RESPECIFY; T2 (labour periphery) pass internal; T3 (V-Dem against Shield and λ₂) strong pass; T4 (Maddison structural breaks) pass; T5 pass.

### 20 August: the housekeeping day

Sixteen commits (`74a0a5a` to `c33940b`) in which I answered the open questions in `VERSION_HISTORY.md` one at a time:

- the acronym (`d5b1b38`, above);
- the naming drift was AI-driven, not deliberate (`071d9d1`);
- v3.2 and v3.2-R are not a rename of anything; JUNO replaced them (`727e2ef`);
- why "JUNO" (`a522594`);
- RD-002's "ten-node" claim is wrong: there was a nine-node predecessor with a checksums node (`aba8b92`);
- stress sign: a calculation convenience, not a construct (`0204cbe`);
- RD-003: I do not recall what it was (`9098a36`);
- **node naming: "never a rename, always parallel labels"** (`ac04ea3`);
- the 0 to 5 to 1 to 10 scale transition: date genuinely unresolved (`6360a2b`);
- DOIs minted (`c33940b`).

Two of those I would phrase differently today. The Short History page says "functional renaming" (`3367cc4`, 15 September), which is the opposite of `ac04ea3`. And the stress-sign answer sits beside Phase 1's polarity text (+5 means resilient). I leave all of it standing. **We don't edit the past; we qualify it.**

### 28 August `898b0d6`, `83f0e67`, `c79bcec`, `aad6218`: the bond column bug

This one is a real error of mine and a useful one. `Bond_Strength_Calc` in the Unified Dataset held "the right multiset of per-node mean bonds but almost never attached to the row's Node": exact node alignment in **5 of 5,425** complete society-years, and those five are years where all eight means are identical. The sorted values matched 5,425 of 5,425. The fix recomputes from raw C, K, S, A under JUNO v1.2, keeps the broken column as `Bond_Strength_Calc_legacy` for rollback, and backs up the old CSV. The consequence recorded in `CANONICAL_STATUS.md` is that any analysis that slices bond strength by node label must be re-run. That includes Phase A's Shield x SIPRI test (the addendum `c79bcec` re-runs it). I do not know whether the 15 August inelasticity pilot used the misaligned column. **[unclear]** The same day `aad6218` freezes the 2026 to 2028 holdout predictions as a dated freeze that later work may score and must not edit.

---

## 10. Phase 9: Thermometer, not crystal ball (Sep 2026)

### 10 to 15 September

The Hansard horse-race note (`e552f61`, `e38fdab`); "thermometer framing; universality from giving ground" (`72cb8ac`, 12 September); Shield depression lifts of 0.89 (3-title) and 0.84 (6-title) labelled (`b29c26b`); an agreed Jul to Sep lab update with the April hindcast qualified (`8f284fd`); and **A Short History of CAMS** (`3367cc4`). It says the instrument is "diagnostic, not a predictive oracle", that vocabulary and formulas changed, and that a hypothesised "merchant civilisation" stress-suppression signature was checked four independent ways and falsified in every one. I have not located the underlying analysis. **[not checked]**

### 17 September: a note exactly 2800 characters long

`908e7bd` adds a USA CAMNATIONSMFIN 2015 to 2026 panel and a September note. Then `2cb6c34`, `32d6c81`, `57e4ed5` each fix the note to be **exactly** 2,800 characters. Three commits for a character count. I am keeping them in because they are what hitting a limit looks like.

### 18 to 23 September

Burma added; Italy and Netherlands rescored; a separate run kept out of the unified dataset (`b125b65`, `d132ad2`); "A Core Response to Outright Dismissal" (`079bb2a`); "The Meta-Animal Under the Microscope" (`8d11f02`); JUNO coverage rises from 36 to 53 societies (`f1a07f9`). The homepage opening was changed to a meta-animal and Spencer paragraph (`2e15d11`), and the Spencer/eugenics sentence removed from it the same day (`a70c991`). A sitemap rebuild from 79 to 170 URLs (`3e7f793`) and a searchable All Pages index (`b416b53`) are housekeeping I had stopped being able to do by hand.

---

## 11. Phase 10: Deep time, v3 and the erratum (Oct 2026)

### 2 to 9 October: the backstory (deep-time) test

`614af9e`: a blind multi-model deep-time test (Claude 15 passes, Grok 4, GPT 3), `e5be317`: adoption of the DT-2 rubric amendment and a Kimi blind pass. **DT-2** exists because GPT treated the absence of writing as a reason to leave Lore and Archive unscored for societies without written records, while Claude and Grok scored them substantially. An Archive that equals "writing" is a rubric that measures the scorer's taste. DT-2 explicitly recognises oral tradition, and separates "function demonstrably absent" from "evidence insufficient".

`4bd2381` (rev. 6 October): *the experiment CAMS*, which calls the eight nodes **the Paladins**, proposes P = C·A and H = K − S as derived coordinates "whose interpretation and measurement validity remain open", and lists six hypotheses with their defeat conditions. It records a failed hypothesis from the deep-time study's separate register (**DT-H4**, Mesopotamian debt-relief edicts and labour stress: the registered prediction failed under all three model families) and leaves it as a failure. It also reports the pooled LOSO AUC of 0.563 (mean across 37 societies, all 126 ruptures; t(36) = 1.76, p = .087) with its denominator attached.

### 4 to 14 October: "Geopolitics as Group Adaptation"

`d537e5b` publishes it as a "draft, for review"; the draft badge and the byline "draft" come off on 14 October (`d5ff4d5`, `85fe812`, `446928c`). **[not checked]** whether anything in the text changed when it stopped being a draft.

### 20 October: the big day

- **JUNO v3** (`ffd108d`): 41,914 rows, 42 societies. Columns now `Society, Year, Node, Coherence, Capacity, Stress, Abstraction, Node_Value_v2 (C+K+A/2-S), Source_File, Source_Note`. Legacy `_Calc` columns dropped. UAE removed ("ceiling scores 10/10/3-4/10 in every source") pending rescore. A patch after audit: "Palestine legacy node names mapped to canonical 8". So **legacy node names were still turning up in the corpus on 19 October 2026.**
- **Signature Pack v1**, in the same commit; then three commits renaming "Idiots in charge" to **Helm Incompetence** (`361088e`, `7c9f5d6`, `d861308`). The rule is Helm ΔK ≤ −1.0, Helm ΔS ≥ +2.0, Helm ΔA ≥ 0 over three years with Craft and Flow capacity holding. The informal name is still in the thesis as a gloss. I changed the label, not the rule.
- **The thesis**, "Organisational Pathologies as Recurring Functional Topologies" (`25878a2`, `079619a`, redraft `e77df64`, page `c4d434b`, rewritten `7374ff5`, `b4a20b6`, `232f519`). Its warrant for "pathology is a relationship" is a frozen six-signature pack with Holm-corrected window tests.
- **Retire 22 superseded Claude artifacts from Explore** (`172b68a`). I did not record which 22 in the commit body. **[unclear]**
- **Canonical Definitions** (`8237f2e`). See the ledger: Stress is now defined as the rate of entropy production within a function, "not the severity of an external challenge", and Abstraction as the degree to which symbolic activity is operationally integrated with function, "coherent cognition". Compare Phase 1.
- **The erratum** (`2c9750e`, `docs/errata/ERRATUM_2026-10-20_bond_scale_and_labels.md`). In three parts. (1) The July memorandum's bond thresholds, the China claim that edge weights "consistently average above 30.0", and the USA prospective test's bond means (B̄ = 8.986 for 2026) sit on **incompatible scales**, and neither repo kernel can produce 8.986 because both are bounded by 1; the per-node formula behind it is "undetermined from the repo" and was attributed to a v3.0 `formulation3.0.txt` that is not in the repository. (2) The word "guarantees" about the Helm node's decline is withdrawn: it was a forecast. (3) `JUNO_B_EP_Predictions.csv` (197 rows, 7 societies, 1801 to 2022) is relabelled **retrospective episode labels, not predictions**, and must not be counted as forecasts or hits. Files qualified, not edited. I also drafted an event-graded prospective register v1 and did not lodge it.

### 23 to 29 October

*The Synchronised Slide* (`882f927`), a common-factor analysis of post-2000 stress on JUNO v3. v1.2-OPT blind rescores added to `data/nations` (`31cc441`). A **Shield-inversion two-mechanism split**, registered as exploratory only (`2786373`): "not preregistered; not validated", with a 274-versus-268 firing-year discrepancy against the Signature Pack **unresolved**. A research note on an accidental repeatability test using a GPT blind rater (`3d7ca03`, 29 October). **[not checked]** its contents.

### 30 to 31 October

The Human-AI Scientific Record page (`4406bd7`) with a Zenodo DOI (`112a43d`) and a statement of Claude's part in it (`b1ccaf7`). The book: first *The Dynamics of Organisations* (`4977a15`), retitled *The Dynamics of Social Organisation* the same day (`3fdff90`), with a new subtitle, *A field guide to dreaming vertebrates* (`709185c`), and an acknowledgements page that was then rebuilt four more times by 3 November (`8d306ce`, `68f22cd`, `8ac6536`, `e2aa8a3`). I will not claim that the PDF checksums were the problem. They were the reason each rebuild had to be committed.

---

## 12. Phase 11: The first week of November

- **2 November `c6fe436`: retire the legacy USA 1861 v2.1 fixed-value validation** from CI. The commit message says only "(Kari, 2 Nov 2026)". The USA 1861 benchmark itself comes from the v2.1 commit (`46460eb`, 17 August 2025: Ψ = 0.724). **[not checked]** when it entered CI. **[unclear]** why I retired it: the commit does not say, and I would rather record that than guess.
- **4 and 5 November `859ebe4`, `99696c2`, `d6000aa`, `e0576fd`**: ADVERSARY-02, a five-pass reconstruction of 2010 to 2020 for China and seven firms, with a results report and Part 2 data; the last commit corrects a manifest hash in section 7. **[not checked]** the results. I am not summarising a report I have not read.
- **5 November, unmerged** (`c0d8798`, `d51e14e` on `crisis-typology-esch-cognition-5nov2026`): a crisis typology rule v1.0 (cluster gap in V between {Helm, Lore} and {Hands, Stewards, Flow}) reported as 13 of 15 correct on canonical ensemble-mean files, and an ESCH cognitive-personality extension defining **Focus = A x C** and **Energy = K − S** at node level, with five "personality types" for the USA 1900 to 2026. Neither is on `main`. The draft PR note itself says the rule is a lagging indicator and that the leading-indicator variant is "proposed but not yet sealed". I am treating all of it as exploratory.
  - *Qualified 8 November:* rebuilt on top of `main` and merged the same day as PR #43 (`ce1d558`, `3dbdb46`, merge `205424a`). Merging it does not change its status: it is still exploratory, and the rule is still lagging.

---

## 12a. Addendum: 5 to 8 November (written 8 November)

Four days, 22 commits, and most of them are about one experiment. In order of the work:

### 5 November `950baf6`, `60bfef2`: this diary

The diary and its companion paper, `docs/CAMS_IN_HPS_AND_COMPLEX_SYSTEMS.md`, were committed as drafts for my review, with an Explore card ("How CAMS Got Here, and Where It Sits"). They were merged as PRs #41 and #42.

### 5 November `ce1d558`, `3dbdb46`: the crisis typology goes to `main`

The work described as unmerged in section 12 was rebuilt on current `main` and merged (PR #43). The commit records the cluster-gap rule as sealed at threshold 1.0, with V = C + K − S. It reports 13 of 15 correct on canonical ensemble data and USA 2008 reading economic (−4.93). Its ESCH reading names **Focus = A × C** and **Energy = K − S**. The commit itself carries the caveat: "cluster-gap V rule is lagging indicator; Focus differential is leading". The leading variant is still not sealed. Note the formula: this V omits the A/2 term of the canonical Node Value (V = C + K + A/2 − S). **[unclear]** whether that was deliberate.

### 6 November `6ed1b8f`, `75dc644`: ADVERSARY-02, the scorecard holds

This is the answer to item 8 in section 15. Fifteen predictions were sealed before any 2021–2024 data was looked at. China and seven firms were then scored. On the primary pool (7 runs, 20 passes) the result is **6 PASS, 9 FAIL**: P1, P4, P5, P10, P12 and C2 pass, the rest fail, and the built-in kill-switch FC5 is **not triggered**. A best-two-scorer check (Claude agent and Gemini) gives the same 6/9 on every item. A knife-edge is recorded rather than rounded up: P8 is FAIL because the Vanke ratio is 4.998818728522337 against a sealed ≥ 5.0. If Kimi web took Gemini's seat, P6 would flip to PASS (giving 7 PASS, 8 FAIL), but the canonical pair and the headline do not change. Kimi Work was added as a secondary scorer only (`analysis/adversary_02/part2/best2_scorers_sensitivity.md`).

### 6 to 7 November `4d512eb` and after: the personality replication round

The largest deposit of the week. The closed CAMS personality AI replication round (Arms A to C) was deposited byte for byte. It came with the full report, score tables, an evidence manifest and a checker, `verify_evidence.py` (`analysis/cams_personality_independent_ai_replication_2026-11-07/`). The question: can eight node functions, read only through Cog = A × C, Aff = K − S and σ = Cog × Aff, give era portraits that AI systems recognise against sealed eras? There were 126 eras: a 56-era core of Russia, Australia and the USA, a 51-era extension, and 19 sparse Denmark eras. From the report's own table:

| Model | Core | Denmark | Extension | Note |
|---|---|---|---|---|
| Claude | 56/56 | 13/19 | 45/51 | Arm A file absent; lineage unverified |
| Gemini | 56/56 | 17/19 | not in the drop | claimed digests admitted non-cryptographic: descriptive only |
| Manus | 56/56 | 15/19 | 16/51 | flagged: Arm A read the tables in the same session |
| Kimi | 52/56 | 12/19 | 8/51 | |
| Grok | 19/56 | no file | no file | still clears the pre-registered bar |
| Codex | 3/56 | 1/19 | 3/51 | at chance; fails the bar |

The report states the limits itself: "There is no clean validation yet." The pre-registered nulls were never returned, and several cells were never run.

What followed was mostly about publishing it honestly:

- **`9fab4ce`, `b4bb2de`: a public repeat kit.** It contains the Arm A, B and C packets, with a deterministic participant zip. The closed-round answer keys are kept out, and a stale Arm B count note was deliberately left unedited.
- **`ecd647f`, `60fb074`, `f081eac`: an on-site page for the case** (`cams-personality-experiment.html`). JUNO v3 and the personality experiment are kept as separate series, and recognition counts sit behind a disclosure that starts closed. The last commit states meaning and metabolism as a *proposed* unification, with the eight-pole attractor kept distinct from the measured results.
- **`bb35b0e`: a short essay on "the Gemini hypothesis".** The commit says it does "not treat unrun controls as passed falsification tests".
- **`2347cf9`, `cb77c5c`, `9447491`: *The Anatomy of the Meta-Organism*, a general essay.** The second commit "stop[s] equating Cog with focus" and lists three unrun falsifiers. Note the tension with the crisis-typology commit two days earlier, which defines Focus = A × C. **[unclear]** which wording I now hold. **[not checked]** either essay beyond its opening and the commit messages.

### 4 to 7 November `0edacce`: the book, again

Faith Lightfoot was added beside Tod Tommo in the acknowledgements. The PDF and EPUB were patched directly, since the book's source is not in this repo, and the checksums were updated (PR #38). That is the fifth rebuild of the acknowledgements page.

### 7 November `b044055`; 8 November `aebb28a`: Story Maker, and a σ correction

The Story Maker (`story-maker.html`, PR #52) builds name- or date-masked packets from the kit's era means for any AI system to read. Building it exposed a definitional point worth writing down. **In the kit, σ is the era mean of yearly Cog × Aff, not the product of the era means of Cog and Aff. Rank is the era mean of yearly σ ranks, so it can be fractional.** The first version checked σ = Cog × Aff on the era rows and failed on real data, so the check was dropped. The packet wording still implied the product, and one AI review then concluded that σ was redundant and rank was derived from it. Both conclusions were wrong, and the wording was corrected on 8 November (PR #54).

### 8 November `b10e6c1` to `fa17d10`: *The Natural History of Civilisations*

*The Human Metasystem: A Natural History of Civilisation* and its supporting document were published as the master statement of the framework (`natural-history-of-civilisations.html`, PR #53), with links to the latest findings. The name changed twice within the hour. The page first carried both expansions. It was then set to "of Societies" (`8947c63`) and finally to "Model State" (`fa17d10`). I said then that "Complex Adaptive Civilisational Model" is the likely future name. Nothing in the repo has changed to it yet.

Two loose ends from the supporting document. It cites failures (the Stress Entrainment Hypothesis, the Abstraction Rent Hypothesis, the Persistent Shear audit, the T-M archive audit) and a blind morphology result of 13 of 24. **None of these has a file in this repo** that I could find on 8 November. They are claims about the record that the record here cannot yet show. I have not added them to section 14 for that reason.

### 8 November: the 2027–2028 register is lodged at Zenodo (`2cb0404`, PR #56)

*CAMS/JUNO v1.2 prospective predictions 2027–2028: United States, China and Australia* (v1.0.0) was published at Zenodo on 8 November 2026 as DOI **10.5281/zenodo.24046639**. It is the third prospective lodgement. Like the other two, it is scored separately:
- the 12 July 2026 six-nation memo (`predictions.html`)
- the frozen 28 August 2026 register (`juno/OOS_PREDICTIONS_2026-2028.md`, P-01 to P-12)

It makes 32 probability-weighted claims about the *scored* state of the USA (12), China (11) and Australia (9) under JUNO v1.2-Final. 11 are Tier A, stable across datasets; 20 are Tier B, dependent on the dataset; 1 is conditional. It supersedes the working drafts Registers B, C and D. Its inputs are a snapshot of this repo at `b4227c2`, plus a USA MARKER ensemble I supplied separately. Every probability is regenerated by one script with seed 2026.

The register states its own limits:
- The datasets for each society disagree by more than the ±1 NV margin, so Tier B claims get a single verdict only if every dataset agrees.
- The simulation has no directional edge.
- The claims are correlated.
- JUNO v1.2 is not externally validated; scoring this register is part of that test.

The lodged web page and the package went into the repo byte for byte, each checked against the record's md5:
- the page: `predictions-2027-2028.html`, which the record's "Is documented by" link points to
- the package: `records/cams-juno-prospective-predictions-2027-2028-v1-0-0-2026-11-08/`

From here the rule is the same as for the August register: **no edits**. Corrections go in a dated erratum, and changed claims need a new version. The PDF is linked from the Predictions page and Explore rather than from the lodged page, so the lodged bytes stay as deposited.

---

## 13. Taxonomy ledger (up to JUNO v1.2)

This is the part I most wanted to get right. All rows give the evidence type: **git** (a dated commit), **doc** (a document dated later), or **author** (my statement in `VERSION_HISTORY.md`, 20 August 2026).

### 13.1 The nodes

| When | Node set | Evidence |
|---|---|---|
| 4 Oct 2024 | **Nine**: Executive, Priesthood, Property Owners, Army, Trades and Professions, Proletariat, State Memory, State Checksums, Storekeepers | git `09ce7f5` |
| 7 Nov 2024 | **Eight**: State Checksums removed; Storekeepers renumbered 8 | git `d7dd654` |
| 1 Mar 2025 | Executive, Army, Proletariat, Priesthood used as the node examples | git `cdecc11` |
| 17 Aug 2025 | Mnemonic names appear beside the sociological ones in code: Helm (Executive), Shield (Army), Lore (Knowledge Workers), Stewards (Property Owners), and so on | git `46460eb` |
| "Jul 2025" | Eight functional nodes "adopted as canonical" under Executive, Army, Knowledge, Property, Trades, Labor, Memory, Commerce | doc (master reference). Author: its "ten-node predecessor" claim is wrong; there was a nine-node one |
| "Sep 2025" | Helm=Executive, Shield=Army, Stewards=Property, Lore=Knowledge, Archive=Memory, Craft=Trades, Hands=Labor, Flow=Commerce | doc/author. **Not a rename: two parallel label sets "from the start"**; mnemonic set became public-facing as "more universal" |
| Jan 2026 | "Canonical node-mapping correction": Shield is military and defence, not welfare institutions; Stewards is landowners and asset management, not the welfare state | doc (`VERSION_HISTORY.md`, citing the project-history page) |
| Mar 2026 | Mapped to eight coordination problems: Helm = direction, Shield = boundary, Lore = meaning, Stewards = allocation, Craft = technical reproduction, Hands = labour reproduction, Archive = memory, Flow = circulation | git `c6a7a44`, `functional_taxonomy.html` |
| 9 Jun 2026 | 34 files classed **DEV: legacy pre-CAMS-8-node taxonomy** | git `1083b2c` |
| 16 Jul 2026 | Stewards wrongly labelled "Bureaucratic administration / Civil service" in `model.html` and the validation page, corrected; stray "bureaucratic" removed from Archive tooltip | git `d959be4` |
| 20 Jul 2026 | "Paladins" used as the rendering names; Lore glossed "Meaning, identity, narrative imagination"; Archive "Memory, law, evidence, continuity" | git `4289140` |
| 6 Oct 2026 | Paladins with images: Helm (the Throne), Shield (the Battlefield), Lore (the Temple), Archive (the Library), Craft (the Workshop), Flow (the Agora), Stewards (the Manor), Hands (the Fields) | git `4bd2381` |
| 19 Oct 2026 | "Palestine legacy node names mapped to canonical 8" in JUNO v3 | git `ffd108d` |

**My own summary:** the count went from nine to eight once (November 2024). The *names* did not change in sequence; two sets coexisted. The *definitions* changed more than the names did: Stewards and Shield were each corrected after they were wrongly widened to welfare or civil service, and Stewards was wrongly widened a second time in documentation. And the note that the Short History says "renamed" is a qualification I have not resolved.

One more small thing on the **order** of the eight. It is not stable across the repo. The 2024 list starts with the Executive and ends with the Storekeepers. The canonical definitions page lists Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow. The Short History page lists Helm, Shield, Lore, Archive, Craft, Flow, Stewards, Hands. The Paladin table in the rev-6-October paper has yet another order. None of this changes a calculation. It changes a reader's ability to compare tables by eye.

### 13.2 The metrics

| When | Dimensions | Scale | Evidence |
|---|---|---|---|
| 4 Oct 2024 | Coherence, Capacity, Stress, **Impact** | not stated; NV is a plain sum | git `09ce7f5` |
| 15 Nov 2024 | Coherence, Capacity, Stress, **Activity** (called "Nodes" in the README) | not stated | git `5b1e00b`. **[unclear]** |
| 1 Mar 2025 | Coherence, Capacity, Stress (three) | **−5 to +5**; Stress +5 = resilient | git `cdecc11` |
| "early to mid 2025" | "Impact" replaced by **Abstraction** | | doc: Short History. The Short History's reason: Abstraction is "internal symbolic honesty, not external reach" |
| 17 Aug 2025 | C, K, S, **A** in code | C and K normalised from −10..10; S and A from 0..10 | git `46460eb` |
| Feb to Mar 2026 | Signed stress still in some files (Germany +10 offset fix; India and Saudi Arabia "signed Stress convention") | | git `1b88ae6`, `f8a3864` |
| "Nov 2025 (RD-001)" | Stress standardised to **unsigned**; metrics originally 0 to 5, later 1 to 10 | **Date of 0..5 to 1..10 transition unknown, closed as unresolved** | doc/author (`6360a2b`) |
| 9 Jun 2026 | C, K, S, A on 1 to 10 | | git `1083b2c` |
| 20 Oct 2026 | Definitions: **C** internal alignment and coordination clarity; **K** demonstrated ability to perform the function; **S** *rate of entropy production within the function, not the severity of an external challenge*; **A** degree to which symbolic activity is operationally integrated with function ("coherent cognition") | | git `8237f2e` |

*Qualification on Stress:* in October 2024 Stress "captures external or internal pressures". In October 2026 it is "not the severity of an external challenge". Same word, different thing. Results from before the redefinition are labelled against the earlier wording and I have not tried to re-derive them. **[not checked]** whether the scoring prompts changed correspondingly. The scoring protocol itself was revised several times (v2.1 at `4fc77a3`, discipline v1.1 at `46a607a`, v1.2-OPT at `9948fd0`, with evidence-gated NA later). Whether the Stress wording moved with them I have not checked.

*Qualification on Abstraction:* in 2024 it was a society-level capacity (writing systems, scholars); by 2025 it was a node-level dimension; by 2026 it is "coherent cognition", with high A meaning sophisticated and grounded, and drifted symbolic activity scoring low. Three meanings, one letter.

### 13.3 Formulas

| When | Quantity | Form | Evidence |
|---|---|---|---|
| Oct 2024 | Node Value | C + Cap + Stress + Impact | git `09ce7f5` |
| Aug 2025 | System score, normalised | `h_i = 0.5 C' + 0.5 K'`, weighted sum over nodes (from the module docstring) | git `46460eb` |
| to Apr 2026 | Node Value | `V = C + K + S + 0.5A` with signed S (equivalently `C + K − S + 0.5A` with unsigned S) | doc/author |
| Mar to May 2026 | v2.3 Bond Strength | pipeline in `cams_framework_v2_3.py`; spec page `9188b4e`, repaired `c3d537f` | git |
| Apr 2026 | ESCH / σ (v3.2-R, social cognition) | with a clamp that forced it positive | doc (`VERSION_HISTORY.md`); date of clamp **[unclear]** |
| 9 Jun 2026 | JUNO v2.4 | `V = C + K − S + 0.5A`; `B_ij = sqrt(q_i q_j) 2^(−(S_i+S_j)/10)`; `σ = (A C/100)(K − S)` | git `1083b2c` |
| 6 Jul 2026 | JUNO v1.2-Final | as above, `q_i = (0.6 C + 0.4 A)/10`, `s_i` sign-preserving with `max(abs(K − S), 0.1)`; λ₂ of the Laplacian of `W_ij = B_ij`; rounding to 6 dp | git `464222f` |
| 20 Oct 2026 | Canonical Definitions | `V = C + K + A/2 − S`; `σ = A·C·(K − S)`; metabolic load `Φ = ΣK·S`; dyad fields 𝒫 and 𝒴 | git `8237f2e` |

*Two details to keep apart.* The JUNO σ has the divisor 100 (`A C/100`); the October Canonical Definitions page gives σ without it. They differ by a constant and the same ordering results, but they are not the same number. And the Paladin "dream" is A·C/100 while the October paper's P is C·A. In both cases a scale factor of 100 is the only difference.

*Bond:* v3.0 `formulation3.0.txt` (not in the repo), a Gaussian-RBF kernel in the telescope's `cams-compute.js`, the v2.3 spec, the v2.4/JUNO "coupling-quality form" and the v1.2 rank-1 identification are five different things called bond strength. The erratum says the first two cannot produce the numbers published under the name. Only the last is currently supported by the code and the status note.

### 13.4 The name of the thing

| Date | Expansion | Evidence |
|---|---|---|
| 15 Nov 2024 | Common Adaptive Model State | git `5b1e00b` |
| 2025 (per doc) | Complex Adaptive Metrics of Society | `CITATION.cff` (prior), dated 2025-08-15 inside the file |
| 17 Aug 2025 | Complex Adaptive Management Systems, Catch-All Network ("CAMS-CAN v3.4") | `CAMS_INDEX.md`, git `46460eb` |
| 6 Apr 2026 | **Complex Adaptive Model State** (replacing "of Societies") | git `3ed9782` |
| 8 Apr 2026 | Common Adaptive Model of Society, Catch-All Network | master reference |
| 17 May 2026 | **Complex Adaptive Model System** (replacing both) | git `5ecee1d` |
| 29 May 2026 | **Complex Adaptive Model of Societies** sitewide | git `36f7745` |
| 31 Jul 2026 | Complex Adaptive Model of Societies | `analysis/operator_portability_synthesis.md` |
| 13 Aug 2026 | Complex Adaptive Meta-System | `analysis/pr_gap2_manuscript_draft.md` |
| 20 Aug 2026 | **Complex Adaptive Model State** declared canonical | git `d5b1b38` |
| 15 Sep 2026 | The Short History still reads "Complex Adaptive Model of Societies" | git `3367cc4` |
| 8 Nov 2026 | Synthesis page set to "of Societies", then back to **Complex Adaptive Model State** within the hour | git `8947c63`, `fa17d10` |
| 8 Nov 2026 | "Complex Adaptive Civilisational Model" named as a likely future rename; not adopted | author |

### 13.5 Categories of societies and of crisis

| When | Scheme | Evidence |
|---|---|---|
| 2024 | Agricultural versus conqueror societies; resilient versus sensitive | git `09ce7f5` |
| Aug 2025 | Four attractors: Adaptive, Authoritarian, Fragmented, Collapse | git `1b39b0d` |
| 21 Mar 2026 | Civilisation **Types I to IV**, then replaced the same day by **CAPE archetype** (Temple / Marketplace / Workshop / Commons) plus a dyadic type | git `68fa5aa`, `5c9b18b` |
| 22 Mar 2026 | Thresholds calibrated on n = 3,494 society-years | git `2aeaf70` |
| May 2026 | Eight clusters of 229 polities (Imperial mode; Modern high-function; Stressed regional; Upper-mid stable; Upper-mid stressed; Broad middle; Fragile; Extractive / failed) | git `c6a7a44`, `functional_taxonomy.html` |
| 9 Jun 2026 | Regimes: Stable Adaptive 38.2%, Strained 27.6%, Local Node Failure 22.1%, Systemic Crisis 10.3%, Freeze/Collapse 1.6%, Phantom Type II 0.2% (on 2,377 deduplicated society-years, 23 societies) | git `444605b` |
| 2026 (Short History) | Regime names given as Stable Adaptive, Strained, **Fracture** | doc `3367cc4`. "Fracture" is not a JUNO v1.2 regime name. **[unclear]** |
| 6 Jul 2026 | v1.2-Final: Freeze/Collapse, Systemic Crisis, Phantom Type II, Local Node Failure, Stable Adaptive, Strained, plus **Unclassified** kept as a data tripwire | git `464222f` |
| 20 Oct 2026 | Six signatures: Archive amnesia, Helm Incompetence (was "Idiots in charge"), Helm isolation, Shield inversion, Late abstraction collapse, Narrative-material gap | git `ffd108d`, `361088e` |
| 5 Nov 2026 | Crisis types: epistemological versus economic (unmerged) | git `c0d8798` |

Note one point of continuity that I find reassuring. The regime table of 9 June (Strained 27.6%, and so on) is the same family as the v1.2 table, and v1.2 changed it only at the edges: one branch deleted, one floor lowered. The categories of *societies* changed four times in a year. The categories of *regimes* changed once.

---

## 14. Retired, failed, reversed and withdrawn: a list

Each of these is in the record; nothing was deleted to make this list shorter.

| Item | What happened | Evidence |
|---|---|---|
| State Checksums node | Dropped Nov 2024; "didn't work" | `d7dd654` |
| 83% predictive accuracy | Appeared in pasted AI text; cut within a day; reappears later; **no provenance** | `4e6c966` |
| Neural network hypothesis | Falsified Dec 2025; correction left on original entry | `f4c1ec1` |
| Visitor tracking | Added 25 Aug 2025, removed 27 Aug | `d625d98`, `e290a60` |
| GTS EV / GT dashboard labels | Renamed to CAMS, dual-dashboard framing removed | `1444596`, `e9b56d6` |
| "Unverified accuracy claims" on CAMS page | Removed 8 Mar 2026 | `90ef8f0` |
| Eight-node optimum appendix link | Added and removed within a day | `99d76e4`, `969704d` |
| Auto-run analysis on load | Added and reverted the same day | `dc7e2e4`, `e62ac13` |
| Civilisation Types I to IV | Replaced the same day | `68fa5aa`, `5c9b18b` |
| "Scale-invariant" | Replaced by "scale-covariant" (23 Apr 2026); the October thesis still says invariant | `b204fa6` |
| v2.3 bond formula | Retired 9 Jun 2026, frozen | `1083b2c` |
| v3.2 social-cognition model with clamped ESCH | Replaced outright by JUNO | `VERSION_HISTORY.md` |
| JUNO toggle in explorer and interpreter | Reverted, re-added, schema mismatch | `ab4970d`, `483862b`, `a34a5a9` |
| v14 `crisis_size` | Retired as a "construction failure" | pre-registration v1.5 |
| s-arm of Local Node Failure | Deleted: zero firings in 3,002 society-years | `464222f` |
| "Predicts civilisational resilience or collapse" | Narrowed to "does not predict the future" | `VERSION_HISTORY.md` |
| Paladin dream / can-act / burden | Shown to be relabelled extrema | `4289140` |
| Paladin alignment | Real structure, AUC about 0.48, non-predictive | `4289140` |
| PR-GAP-2 combined AUC 0.652 | Replaced by LOSO AUC 0.563, not significant | `39370b7` |
| Kimi's 88% Archive-Lore claim | Max 51.2% at scale; 9.9x enrichment survives | `2e713ba` |
| WB-proxy SIPRI test | RESPECIFY; superseded by real SIPRI data (conditional pass) | `6a84a2a` |
| `Bond_Strength_Calc` | Misaligned to nodes in all but 5 of 5,425 society-years; fixed with a legacy column | `898b0d6` |
| "Guarantees" (USA Helm breach) | Withdrawn as a forecast, not a certainty | erratum, `2c9750e` |
| `JUNO_B_EP_Predictions.csv` | Relabelled retrospective, not predictions | erratum |
| "Average above 30.0" (China bond) | Withdrawn as unsupported by any formula in the repo | erratum |
| USA prospective bond thresholds P1, P7 | Qualified as on an undocumented scale; retired from the draft register as self-scored | erratum |
| DT-H4, debt-relief edicts | Failed in all three model families; left as failed | `4bd2381` |
| Documentation of Stewards as civil service | Corrected (twice, data and docs) | `d959be4` |
| "Idiots in charge" | Renamed Helm Incompetence | `361088e` |
| 22 Claude artifacts on Explore | Retired as superseded | `172b68a` |
| USA 1861 v2.1 fixed-value CI check | Retired 2 Nov 2026 | `c6fe436` |
| Merchant-civilisation stress signature | Falsified four ways, per the Short History | `3367cc4` |
| UAE in JUNO v3 | Removed pending rescore: ceiling scores | `ffd108d` |
| ADVERSARY-02 sealed predictions | 9 of 15 failed (P2, P3, P6, P7, P8, P9, P11, C1, C3); P8 by 0.0012; left as failed | `6ed1b8f` |
| Story Maker σ = Cog × Aff check | Dropped: kit σ is the mean of yearly products | `b044055` |
| Packet wording "σ = Cog × Aff" of era means | Corrected to era averages of yearly values | PR #54, `aebb28a` |

---

## 15. What I still do not know

1. **Why checksums did not work**, and whether the failure was conceptual (it measured the observer, not a function) or practical (it could not be scored). Only the removal is dated.
2. **How many files still use signed stress** and whether any current result depends on one.
3. **What the pre-8-node DEV files used** as their taxonomy.
4. **What the December 2025 falsification test of the neural network hypothesis actually was**, and where the 83% came from.
5. **Whether "renamed" (Short History) or "never renamed" (`VERSION_HISTORY.md`) is the better description.** Git shows both label sets present together from August 2025; it does not show a switch. I lean to "never renamed", and I would add that the *definitions* moved.
6. **Whether the 15 August inelasticity pilot used the misaligned bond column** (fixed on 28 August).
7. **Which 22 artifacts** were retired on 20 October, and why USA 1861 was retired on 2 November.
8. **ADVERSARY-02's results** (I have not read them for this diary) and the contents of several papers named above.
9. **Whether the unmerged crisis typology and ESCH personality work survives its own next test.** The note on the branch says the rule is lagging, and the leading-indicator variant is not sealed.
*Qualified 8 November:* item 8 is now answered (section 12a). Item 9 is half answered: the work is merged, but its next test has not been run.

10. **Vocabulary.** The 2024 and 2025 files, the v2.1 code and some commit messages describe societies in a loosely evaluative register. The 2026 definitions do not. I leave the old uses where they are and I do not carry that register forward; this diary avoids it, including in the tables above, where I describe the formula and not the old label.

*Closing note for the next diary entry.* The pattern across fourteen months is not that the idea changed shape. It is that each time a number or a name got more confident than its provenance, a later commit shrank it back to what the record could carry. The nine-node model became eight. The neural network became thermodynamics. "Predicts" became "does not predict". "Guarantees" became "forecast". "Validated bounds" became "reference ranges". I would like the next phase to do less shrinking, which means making fewer claims in the first place.

*Addendum, 8 November.* The first days of the next phase went partly that way. The replication round was published with its own failures in the first paragraph. ADVERSARY-02 left a 4.9988 as a FAIL. The Story Maker had its wording shrunk within a day of shipping. The new open questions:

11. **What V the crisis typology uses.** It writes V = C + K − S, not the canonical C + K + A/2 − S. Is that deliberate?
12. **Focus or not Focus.** Is A × C "Focus" (crisis typology, 5 November) or not to be equated with focus (`cb77c5c`, 6 November)?
13. **Where the failure record cited in the supporting document lives.** The cited items are SEH, ARH, Persistent Shear, the T-M audit and the 13 of 24 morphology result.
14. **The null models for the personality round.** They are the next step named by the report itself.
15. **How the three prospective lodgements will be scored together.** These are the 12 July memo, the 28 August register and the 8 November 2027–2028 register. Each has its own rules, and some claims overlap: US-4 overlaps the memo's Helm arm. The register says the overlap is "not a restatement".
