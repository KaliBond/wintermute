# Backstory run log

Every run, decision and deviation is recorded here, newest first.

## 2026-10-02 — Write-up: DT-2 adoption and Kimi blind pass

- `results/DT2_adoption_and_Kimi_blind_2026-10-02.md`: the DT-2 rulings and hashes; four items to settle before the DT-2 rescore (R1's one-sentence NA rule conflicts with the CSV-only contract; worked examples name two scored cells and two models; the prompts-only zip is still DT-1; run numbering); Kimi results.
- Kimi (single pass, DT-1) reproduces H2a, H2b (Aegean), H3 (Egypt) and H4's failure; H1b holds in direction; H1c Flow 3rd; H1a +2.03.
- **H5 fails under Kimi** (Energy ratio 0.36, Cognition 0.69) even though Kimi scores oral Lore and Archive at 7–8 and NAs nothing in the non-state cases. NA choices therefore do not explain the H5 split on their own: Kimi rates the non-state societies' other functions lower and books more Stress. Single pass; needs a Kimi ensemble.

## 2026-10-02 — Run 10 prep: DT-2 adopted

**Decision (Kari McKern):** DT-2 adopted. The three rulings R1–R3 from `prompts/DT-2-amendment.md` were appended verbatim, worked example included, to both scorer prompts under a new section `## DT-2 amendments (2026-10-02)`. The DT-1 text above that section is byte-for-byte unchanged. Source record kept as `prompts/DT-2-amendment-draft.md`, identical to `prompts/DT-2-amendment.md`. No rescoring yet; Kari will say when.

| Prompt | DT-1 SHA-256 (record) | DT-2 SHA-256 (current) |
| --- | --- | --- |
| A, `prompts/scorer_prompt_deeptime_v1.2-OPT.txt` | 64009fc6… (`64009fc68f5a86b01c697da9a824300e8ea40dca580dfb4a3714e029fe010ee2`) | `f4bfc5b812f3cb5f304a5c6840d2f3bafad64fa6ce5fa9d22ec43b9f0d7e2b62` |
| B, `prompts/scorer_prompt_deeptime_v1.2-OPT_windows2.txt` | 90b68399… (`90b683991da9128efad03dbedb605cde59e814db3f684985ccdfb058eb3d2052`) | `0652e5376c877f5bb9e09c883ab619040c0154c51438918a7c29d61988833aac` |

- DT-1 originals archived as `prompts/archive/scorer_prompt_deeptime_v1.2-OPT_DT1.txt` and `prompts/archive/scorer_prompt_deeptime_v1.2-OPT_windows2_DT1.txt`; their hashes match the DT-1 record above. Every run before this entry was scored under DT-1.
- The rulings: **R1** NA means unknown; evidenced absence is scored low. **R2** Archive and Lore are functions, not media; oral systems are admissible evidence, and a pass that NAs a documented oral system is flagged at intake as a rubric-interpretation error. **R3** Abstraction scores the method that exists; coercion and unused method book to Stress.
- Any DT-1 vs DT-2 differences found in the rescore are to be reported cell by cell as findings, not corrections.

**Also imported (Kimi, fourth model lineage):** `runs/kimi_blind/` (RUN-KIMI-BLIND-1, one blind pass per prompt, sealed) and its reports in `results/kimi/`. Copied verbatim from the Kimi workspace (`blind-draw/`). On import, both raw-score hashes in `runs/kimi_blind/pass1/SHA256SUMS` and all 14 entries in `RUN-KIMI-BLIND-1-SEAL.txt` were checked against the source files and match. The seal file is unedited, so its report paths still read `../results-kimi/`; those reports now sit in `results/kimi/`. Per the Kimi `RUN.txt`, this is a single pass with prompts A and B returned together, so A/B isolation is not confirmed, and the model version is not recorded. Scored under DT-1.

## 2026-10-02 — Complete results write-up

- Paper tab "Complete results" filled: run scope, all-20-window chart, window table, hypothesis verdicts by run, node-level Node Value table for Claude 15 / Grok 4 / GPT 3.
- New file: `results/node_means_three_models.csv` (480 rows: model, window, node, C, K, S, A, NV; NA under pooled rule).
- H1a reclassified from "not a finding" to "holds in direction; small and NA-sensitive" (pooled Claude +0.82, Grok +0.40, GPT +0.71; run 2/3 −0.03 against).
- Section 4 tally now: 4 held under all three models, 2 held in direction, 2 scorer-dependent.

## 2026-10-02 — Status at wrap-up
- Working paper drafted: "Reading Societies Without Their Stories" (Claude Docs, https://claude.ai/code/artifact/b4dc6c9f-e102-4dd6-9607-7a3a0047a1cc), covering runs 1–10.
- Corrected counts used in the paper: Claude left Harappan Helm NA in 12/15 passes and Shield in 14/15 (earlier log lines said 14–15).
- Pending: Grok passes 5–6; GPT version; R6 model confirmation; DT-2 rubric; nothing committed to git yet.

## 2026-10-02 — Run 10: Grok 4.7 blind pass 4

**Source:** Kari; Grok RUN.txt in `runs/grok-4.7_blind/pass4/`. Each prompt in its own conversation, prompt file only, one tool call each (file read). The first prompt-B scorer stalled and was replaced; the replacement is the file kept. Hashes A 87389bfc…, B f0b4b000…, both new. Grok ensemble now 4 passes (`runs/grok-4.7_blind/ensemble4/`); spec checks passed.

**Grok (4) vs Claude (15):** MAE A 0.39–0.53, B 0.41–0.57; ranking 0.92 / 1.00; offset +0.2 to +0.47 (Capacity largest). Unchanged picture.
**Hypotheses (Grok 4-pass):** H2 fall 8/8, 7/8; H2 recovery Aegean yes, Egypt no; H3 Egypt yes, Aegean no; H4 fails; H1b Minoan +2.1, Māori +1.2 vs −0.62 (holds); H1c Flow 5th (fails, firmer than at 3 passes); H1a +0.40 (small yes); **H5 fails at the threshold: Energy ratio 1.66 / 2.08 = 0.80, Cognition ratio 0.92.** H5 under Grok sits exactly on the cut-off and should be reported as indeterminate for Grok.
NA (4 passes): Harappan Helm 0/4, Shield 1/4; Xiongnu Archive 3/4; Longshan Archive 4/4; Aegean Archive 4/4 in both prompts.

## 2026-10-02 — Run 9: GPT three-pass set ("Backstory_Three_Separate_Passes")

**Source:** zip from Kari, 3 passes × 2 prompts, all six hashes new (A 16095d94…, 9250e914…, eee7afcd…; B e9ef37df…, a8bbd7be…, 4adadc0a…). **Model:** GPT via ChatGPT Work/Codex. Exact model version not exposed to Kari: **unverified**.
**Pass isolation (as recorded by Kari):** three separate sub-agent contexts, each started with no inherited conversation history; each received the same ZIP and instructions and was told not to inspect other scores. All three used the same inherited model (one model, three passes). Accurate description: "three fresh isolated agent contexts", not "three separate user conversations".
**Within each pass:** prompts A and B ran in the same context, so the overlap windows (B05, B09) are not independent between A and B within a pass, and B may have been scored with A in view.
Saved in `runs/gpt_3pass/`; spec checks passed.

**GPT differs from both Claude and Grok.**
| vs | Prompt A MAE | A offset (C K S A) | A ranking | Prompt B MAE | B ranking |
| --- | --- | --- | --- | --- | --- |
| Claude 15-pass | 0.89–1.64 | +0.90 +0.94 +0.64 +1.64 | 0.81 | 0.84–1.80 | 0.99 |
| Grok 3-pass | 0.59–1.40 | +0.71 +0.56 +0.56 +1.40 | 0.85 | 0.51–1.45 | 0.99 |

Large upward offset, especially Abstraction (+1.6 to +1.8 vs Claude). Internal agreement tight (SD 0.12–0.36, V range 1.0–1.2).

**Distinctive NA pattern:** NA on Lore and/or Archive for every non-literate society — Aboriginal Australia Helm, Shield, Lore and Archive (4 of 8 nodes, all passes); Māori Lore and Archive; Xiongnu Lore and Archive; Longshan Lore and Archive; Harappa Shield, Lore and Archive. Claude and Grok score Aboriginal and Māori Lore and Archive high (7–8) from oral tradition. GPT appears to treat oral transmission as insufficient evidence for the gate. That is a rubric-interpretation difference DT-1 must settle (it says oral Archive counts, but this scorer did not apply it).

**Hypotheses:** H2 fall 8/8, 7/8 (holds); H2 recovery slower — Aegean yes and, unlike Claude/Grok, Egypt yes too; H3 Egypt yes, Aegean no (same split); H4 fails; H1b direction holds (Minoan +2.7, Māori +0.8 vs −0.80); H1c Flow 3rd; H5 fails (Energy ratio 0.72) — **but computed on only half the Aboriginal nodes because of NA, so H5 is not testable on this run.**

## 2026-10-02 — Run 8: Grok 4.7 blind, three-pass ensemble

**Source:** two further Grok passes per prompt from Kari (hashes A 085c928b…, 14444205…; B 7b50c6a0…, bda47667…), all unique. No RUN.txt with these two; assumed same protocol as run 7 (separate conversation per prompt, prompt text only) — to be confirmed. Combined with run 7 into a 3-pass Grok ensemble (`runs/grok-4.7_blind/ensemble3/`); spec checks passed. Three passes meets the n_eff ≥ 3 minimum but is smaller than Claude's 15.

**Grok internal agreement:** SD 0.19–0.44, V range 1.39–1.65 — similar to Claude's. Grok is as self-consistent as Claude; the gap between them is a model difference, not noise.

### Grok (3) vs Claude (15)
| | Prompt A | Prompt B |
| --- | --- | --- |
| MAE | 0.36–0.47 | 0.43–0.55 |
| Spearman | 0.94–0.95 | 0.94–0.96 |
| Case ranking | 0.92 | 1.00 |
| Offset Grok − Claude | C +0.17, K +0.38, S +0.12, A +0.26 | C +0.37, K +0.36, S +0.43, A +0.34 |

Grok runs ~0.3 higher across the board; rank order is preserved. NA: Grok never NAs Harappa (0 of 3), Claude NAs its Helm and Shield almost always. Both NA Longshan Archive, Aegean-collapse Archive, Xiongnu Archive, Protogeometric Shield and Archive.

### Hypotheses: Claude 15-pass vs Grok 3-pass
| Hypothesis | Claude | Grok | Status |
| --- | --- | --- | --- |
| H1a river Stewards+Archive | unstable | for (+0.19, small) | Not a finding |
| H1b maritime Flow > Helm | Minoan +4.1, Māori +1.1 vs −0.68 | Minoan +1.9, Māori +0.5 vs −0.55 | **Direction holds across models; Grok effect smaller** |
| H1c steppe: Flow 3rd | 3rd | 4th | Weak / model-dependent |
| H2 fall together | 8/8, 8/8 | 8/8, 7/8 | **Robust** |
| H2 recovery slower | Aegean yes, Egypt no | Aegean yes, Egypt no | **Robust split** |
| H3 Helm–Lore leads collapse | Egypt yes, Aegean no | Egypt yes, Aegean no | **Robust split** |
| H4 relief edicts | fails | fails | **Robust failure** |
| H5 no ladder | holds (Energy ratio 0.92) | fails, narrowly (Energy ratio 0.78 vs 0.80 threshold; Cognition ratio 0.94) | **Model-dependent and threshold-sensitive** |

## 2026-10-02 — Run 7: Grok 4.7 blind single pass (first clean cross-model test)

**Source:** Kari, Grok 4.7; each prompt in its own conversation from the prompt text only, zero tool calls, scores not edited (Grok's RUN.txt in `runs/grok-4.7_blind/`). SHA-256 checked: A 2fb9d450…, B 5b3523f5…. Single pass per prompt, not an ensemble.
**Reference:** new pooled Claude ensemble of all 15 independent Claude passes per prompt (`runs/claude_grand/`, runs 2/3, 5, 6).
**Noise baseline:** a single Claude pass against the other two Claude runs differs by MAE 0.26–0.29. Grok differences must be read against that.

### Agreement with the 15-pass Claude ensemble
| | Prompt A | Prompt B |
| --- | --- | --- |
| MAE per dimension | 0.37–0.50 | 0.42–0.53 |
| Within 1 point | 93–98% | 90–96% |
| Spearman per dimension | 0.90–0.92 | 0.91–0.94 |
| Case-ranking Spearman | 0.87 | 0.98 |
| Mean offset (Grok − Claude) | C +0.03, K +0.35, S 0.00, A +0.20 | C +0.42, K +0.33, S +0.22, A +0.20 |

Grok's gap to Claude (≈0.45) is larger than Claude's own pass-to-pass noise (≈0.27). Most of the excess is a steady upward offset, chiefly Capacity. Blind vs non-blind Grok: MAE 0.22–0.42, so opening the project files earlier did not change Grok's scores much.

**NA:** blind Grok scored all eight Harappan nodes (Claude leaves Helm and Shield NA in 14–15 of 15 passes). Agreed NA: Longshan Archive, Minoan Shield, Xiongnu Archive, Protogeometric Archive. The hidden-Helm reading of Harappa is a Claude property, not a model-independent one.

### Hypotheses: Claude (15 passes) vs blind Grok
| Hypothesis | Claude 15-pass | Grok blind | Status |
| --- | --- | --- | --- |
| H1a river Stewards+Archive | unstable across runs | for (+0.66) | Not a finding |
| H1b maritime Flow > Helm | Minoan +4.1, Māori +1.1 vs −0.68 | Minoan +2.0, Māori −1.0 vs −1.06 | Holds for Minoan only across models |
| H1c steppe Shield 1st, Flow 3rd | yes | Shield 1st, Flow 5th | Model-dependent |
| H2 fall: all nodes drop together | 8/8, 8/8 | 8/8, 7/8 | **Holds across models** |
| H2 recovery slower than fall | Aegean yes, Egypt no | Aegean yes, Egypt no | **Same split across models** |
| H3 Helm–Lore falls first | Egypt yes, Aegean no | Egypt yes, Aegean no | **Same split across models** |
| H4 relief edicts (series) | against | against | **Fails across models** |
| H5 no ladder | holds (Energy 1.69 vs 1.84) | **fails** (1.62 vs 2.29, below the 0.8 threshold) | **Model-dependent** |

### Reading
- Robust across two model families: collapse drains every node at once; the Aegean recovers far more slowly than it fell while Egypt does not; Egypt's ruler–meaning coherence falls before its surplus does, the Aegean's does not; debt-relief edicts do not show lower labour stress.
- Not robust: H1a, H1c, H5, and the hidden-Helm NA pattern in Harappa. H5 ("no ladder") passes under Claude and fails under Grok, which scores state societies higher. That is precisely the kind of corpus or model bias the project set out to expose; it should not be published as a CAMS finding until more models are run.
- Caveat: one Grok pass is not an ensemble. A five-pass blind Grok run would settle whether the H5 and H1c differences are Grok's view or one pass's noise.

## 2026-10-02 — Run 6: second independent replication (R6)

**Source:** 10 files uploaded by Kari (5 × prompt A, 5 × prompt B). **No RUN.txt supplied, so the model and blind conditions are not recorded.** Pass numbering follows upload order. Saved verbatim in `runs/replication_R6/` with SHA256SUMS.
**Integrity:** all 10 hashes unique; none matches the earlier replication (`blind_replication_claude`), unlike the duplicate upload earlier today, which was logged as a duplicate and not counted. All aggregation spec checks passed.

### Agreement
| R6 compared with | Prompt A MAE | Prompt A Spearman | Prompt A ranking | Prompt B MAE | Prompt B ranking |
| --- | --- | --- | --- | --- | --- |
| Run 2 Claude ensemble | 0.17–0.23 | 0.97–0.99 | 0.98 | 0.14–0.31 | 1.00 |
| Run 5 Claude replication | 0.14–0.17 | 0.98–0.99 | 0.99 | 0.14–0.31 | 1.00 |
| Grok 4.7 (single pass) | 0.42–0.60 | 0.88–0.93 | 0.92 | 0.36–0.53 | 1.00 |

R6 sits with the two Claude ensembles, not with Grok, and shows the same small Grok offset (+0.3 on Capacity and Stress). Most likely another Claude run; to be confirmed by Kari.
Scorer SD 0.21–0.32; V range 1.77–1.79. NA in the same places: Longshan Archive, Harappan Helm and Shield, Minoan Shield, Protogeometric Archive.

### Hypotheses: three independent Claude ensembles (runs 2/3, 5, 6)
| Hypothesis | Run 2/3 | Run 5 | Run 6 | Verdict |
| --- | --- | --- | --- | --- |
| H1a river Stewards+Archive | against | for | for (+0.67) | Unstable (NA-sensitive) |
| H1b maritime Flow > Helm | for | for | for | Holds 3/3 |
| H1c steppe Shield 1st, Flow 3rd | yes | yes | yes | Holds 3/3 within Claude (Grok differs) |
| H2 fall: all nodes drop | 8/8, 8/8 | 8/8, 8/8 | 8/8, 8/8 | Holds 3/3 |
| H2 recovery slower | Aegean only | Aegean only | Aegean only | Aegean 3/3; Egypt 0/3 |
| H3 Helm–Lore leads collapse | Egypt yes, Aegean no | Egypt yes, Aegean NA | Egypt yes, Aegean no | Egypt 3/3; Aegean 0/3 |
| H4 relief edicts (series) | against | against | against | Fails 3/3 |
| H5 no ladder | holds | holds | holds | Holds 3/3 |

## 2026-10-02 — Run 5: blind replication of the Claude ensemble (separate session)

**Source:** run by Kari in a separate session from `Backstory_Prompts_Only_2026-10-02.zip`; five fresh-context Claude Opus 5.5 subagents per prompt, prompts never combined, no project files, web barred (see `runs/blind_replication_claude/RUN.txt`). Prompt files verified identical to ours (SHA-256 64009fc6…, 90b68399…). Raw passes saved verbatim with SHA256SUMS; aggregated here with `aggregate_ensemble.py` (all spec checks passed). Upload order taken as pass1–pass5.
**Note:** `hypothesis_check_series.py` changed after this run to report an H3 series as *not testable* when Helm or Lore is NA (previously printed 'against'). Logic otherwise unchanged; earlier verdicts unaffected.

### Test–retest (same model, independent session)
| | Prompt A vs run 2 | Prompt B vs run 3 |
| --- | --- | --- |
| MAE per dimension | 0.18–0.23 | 0.11–0.22 |
| Within 1 point | 99–100% | 99–100% |
| Spearman per dimension | 0.97–0.99 | 0.97–0.99 |
| Case-ranking Spearman | 0.97 | 1.00 |
| Mean offset | −0.10 to −0.02 | −0.11 to +0.04 |

The Claude ensemble reproduces almost exactly in a separate session. Scorer SD 0.19–0.30, V range 1.6–1.7, as before.

**NA reproduces in the same places:** Longshan Archive, Harappan Helm and Shield, Minoan Shield, Protogeometric Archive (4 of 5), Mycenaean LH IIIA Lore (3 of 5). The Aegean-collapse Archive was scored 1/1/9–10/1–2 by all five here (as in run 2), confirming the function-absent vs evidence-absent ambiguity seen in run 3 is a property of context, not of the case.

### Hypotheses: four independent ensembles/passes
| Hypothesis | Run 2 ens. | Run 5 replication | Grok 4.7 (not blind) | Status |
| --- | --- | --- | --- | --- |
| H1a river: Stewards+Archive | against (−0.03) | for (+1.20) | for (+0.03) | Sensitive to which Harappan nodes are NA; **not a finding** |
| H1b maritime Flow > Helm | for | for | for | **Robust** |
| H1c steppe Shield–Flow | Flow 3rd | Flow 3rd | Flow 6th | Claude-stable, model-dependent |
| H2 fall together | 8/8, 8/8 | 8/8, 8/8 | 8/8, 7/8 | **Robust** |
| H2 recovery slower | Egypt no / Aegean yes | Egypt no / Aegean yes | both yes | **Aegean robust; Egypt recovers as fast as it fell** |
| H3 Helm–Lore falls first | Egypt yes / Aegean no | Egypt yes / Aegean not testable (Lore NA) | Egypt yes / Aegean no | **Egypt robust; Aegean no or untestable** |
| H4 relief edicts (series) | against | against | against | **Against, robust** |
| H5 no ladder | holds | holds | holds | **Robust** |

## 2026-10-02 — Run 4: cross-model check, Grok 4.7 (single pass, not blind)

**Source:** files supplied by Kari, scored by Grok 4.7 from `Backstory_Prompts_Only_2026-10-02.zip`. Saved verbatim in `runs/grok-4.7/` with Grok's own RUN.txt; SHA-256 checked on receipt (prompt A 666f98d3…, prompt B 87926726…) and matching.
**Limits stated by the Grok run itself:** one pass only; both prompts scored in one session; the session had already opened README, TESTING.md and HYPOTHESES.md before scoring. So this is **not blind and not an ensemble**. Agreement on hypothesis verdicts may partly reflect having read them.

### Agreement with the Claude five-scorer ensemble
| | Prompt A (12 societies) | Prompt B (10 windows) |
| --- | --- | --- |
| MAE per dimension | 0.42–0.53 | 0.36–0.55 |
| Within 1 point | 92–99% | 87–97% |
| Spearman per dimension | 0.89–0.93 | 0.91–0.95 |
| Case-ranking Spearman | 0.89 | 1.00 |
| Mean offset (Grok − Claude) | +0.14 to +0.36 | +0.25 to +0.39 |

- **Calibration offset:** Grok scores about 0.3 higher on every dimension, most on Capacity. Consistent direction = offset, not disagreement.
- **Where the models differ most: Hands.** The largest gaps are labour nodes (Mycenaean, Old Kingdom, FIP, Aegean collapse, Protogeometric): Grok gives labourers higher Capacity and much higher Abstraction (e.g. Old Kingdom Hands A 6 vs 3.8). Claude reads coerced labour as low-sophistication; Grok does not. This is a substantive interpretive difference about how the instrument sees labour, worth a rubric note.
- **NA:** both agree Longshan Archive, Minoan Shield, Xiongnu Archive and Aegean-collapse Archive are unscoreable. On Harappa they agree evidence is thin but disagree *which* nodes: Grok scored Helm and Lore, NA'd Stewards and Archive; Claude mostly the reverse. The hidden-Helm case is unstable across models.

### Hypotheses across three scorers
| Hypothesis | Claude in-session | Claude blind ensemble | Grok 4.7 | Status |
| --- | --- | --- | --- | --- |
| H1a river states: Stewards+Archive | for | against (11.29 v 11.32) | for (11.86 v 11.83) | Null: differences ≈0 in both ensembles; not a finding |
| H1b maritime Flow > Helm | for | for | for | **Holds across models** |
| H1c steppe Shield–Flow coupling | Flow 3rd | Flow 3rd | Flow 6th | Scorer-dependent |
| H2 fall: Energy and Cognition drop together | 8/8, 8/8 | 8/8, 8/8 | 8/8, 7/8 | **Holds across models** |
| H2 recovery slower than fall | — | Aegean yes; Egypt borderline | yes in both | Aegean robust; Egypt leans yes |
| H3 Helm–Lore coherence falls first | — | Egypt yes, Aegean no | Egypt yes, Aegean no | **Same pattern across models** |
| H4 relief edicts lower Hands' Stress | against | for (single window) / against (series) | against | **Against** |
| H5 no ladder | holds | holds | holds | **Holds across models** |

### Next
- A blind Grok ensemble (five fresh conversations, nothing opened first) is needed before any of the cross-model agreements can be cited.
- Add a Hands rubric note to DT-2: how to score Abstraction for coerced but skilled mass labour.

## 2026-10-02 — Run 3: extra time windows, five-scorer ensemble

**Design:** as run 2 — five Claude subagents in separate contexts, prompt `prompts/scorer_prompt_deeptime_v1.2-OPT_windows2.txt` only, no tools (all five confirmed zero tool use). Eight new windows plus B05 and B09 rescored as overlap anchors. Tests were fixed and hashed before scoring (entry below).
**Files:** `runs/ensemble_windows2_2026-10-02/` (scorer_1–5 verbatim, SHA256SUMS, block1, block2); `runs/combined_2026-10-02/` (all 20 windows, seam report, series checks under two overlap rules).

### Instrument checks
- Scorer agreement again tight: mean SD 0.21 (C), 0.22 (K), 0.32 (S), 0.41 (A); mean V range 1.64.
- **Seam** (run 2 vs run 3 ensemble means on B05 and B09, 60 cells): mean |diff| 0.24, max 1.0. Run 3 slightly lower on Coherence (−0.16) and Abstraction (−0.21). Small: the ensemble is stable across contexts.
- **NA ambiguity found:** Aegean Archive at −1175 was scored 1/1/9/1 by all five run-2 scorers ('collapsed') but NA by four of five run-3 scorers ('no evidence'). Protogeometric Archive is NA in all five. The gate does not distinguish *function absent* from *evidence absent*. Recommendation for DT-2: add an explicit rule — when the absence of a function is itself well evidenced (e.g. loss of literacy), score it low; reserve NA for unknown.

### Series (primary rule: run-2 values for overlap cases)
| Window | Year | Energy | Cognition | Helm–Lore C |
| --- | --- | --- | --- | --- |
| Egypt Old Kingdom | −2500 | 3.65 | 44.4 | 8.0 |
| Egypt late 5th dyn | −2400 | 2.60 | 37.8 | 6.6 |
| Egypt late 6th dyn | −2250 | 0.03 | 26.5 | 5.0 |
| Egypt FIP (breakdown) | −2150 | −2.75 | 15.5 | 3.6 |
| Egypt Middle Kingdom | −1900 | 3.33 | 45.0 | 7.0 |
| Mycenaean LH IIIA | −1350 | 2.12 | 31.6 | 5.5 |
| Mycenaean LH IIIB | −1250 | 0.70 | 32.5 | 5.6 |
| Aegean collapse (breakdown) | −1175 | −4.62 | 10.1 | 3.0 |
| Protogeometric | −1000 | −1.15 | 14.2 | 3.9 |
| Late Geometric | −750 | 0.65 | 26.8 | 5.4 |

### Pre-registered verdicts
| Hypothesis | Egypt | Aegean | Mesopotamia | Overall |
| --- | --- | --- | --- | --- |
| H2 recovery slower than fall | against (primary) / supports (alt overlap rule): fall and recovery rates nearly equal | supports strongly: fall −7.1/century, recovery +2.0; still below LH IIIB after 425 years | — | **Mixed; depends on case.** Robust in the Aegean only. |
| H3 Helm–Lore coherence falls first | supports: 8.0 → 6.6 → 5.0 while Energy stays positive | against: flat 5.5 → 5.6 before collapse | — | **Mixed.** |
| H4 relief edicts keep Hands' Stress low for longer | — | — | against: 7.0, 6.2, 7.2 vs comparator 5.93; 0 of 3 windows below | **Against.** |

### Reading (interpretation, not a test)
- Egypt shows the CAMS signature most cleanly: coherence between rulers and meaning-makers drains for two centuries while surplus is still positive, then Energy turns negative. If this holds in other cases it is a leading indicator. One case is not enough.
- The Aegean shows a different shape: no visible coherence lead, a sudden fall (sea-borne contagion, network failure) and a very slow recovery. Egypt, an inland river state, rebuilt its macro-order in under three centuries. That contrast is worth its own hypothesis (H6, below) — stated after seeing the data, so it must be tested on new cases.
- H4 as framed is probably confounded: debt-relief edicts are issued *because* labour is under debt stress, so high Hands' Stress in edict-issuing states is expected. A fair test needs states with and without edicts under comparable debt load. Recorded as a design flaw, not a rescue: H4 stands as **against**.

### New hypothesis (post hoc, flagged as such; not tested)
- **H6:** Recovery speed after breakdown depends on whether the setting forces re-coordination. Single river-valley states (Egypt, Mesopotamia, China) rebuild the macro-order faster than dispersed maritime networks (Aegean). Test cases: Mesopotamia after Ur III, Shang–Zhou transition, Late Bronze Age Levant, post-Harappan Indus.

## 2026-10-02 07:11 UTC — Pre-registration for run 3 (before any run-3 scoring)

- Series tests fixed in `scripts/hypothesis_check_series.py`, SHA-256 353aaeab264196584464d89618e8f46e4f9857d271617db57909ef0c71cf541f.
  - H2 recovery: per-century change in mean Energy and Cognition, fall leg vs first recovery leg; supports if both recover more slowly than they fell.
  - H3: Helm–Lore coherence (mean of Helm C and Lore C) must fall by at least 0.5 from the earliest to the last pre-collapse window, in each series.
  - H4: Mesopotamian Hands Stress (Ur III, Hammurabi, Ammisaduqa) below the mean of six non-collapse palace windows without attested debt relief, in all three windows.
- New windows (prompt `prompts/scorer_prompt_deeptime_v1.2-OPT_windows2.txt`, SHA-256 90b68399…): Ur III −2050; Egypt −2400, −2250, −1900; late Old Babylonian −1650; Mycenaean LH IIIA −1350; Greece −1000, −750.
- B05 (−2150) and B09 (−1175) are rescored as overlap anchors to measure seam (same scorer design, new context).
- Windows are listed out of chronological order within the prompt to reduce trajectory anchoring.

## 2026-10-02 — Run 2: five-scorer ensemble (Claude subagents)

**Design:** CAMNATIONSM5N paste-and-spawn, executed as five Claude subagents launched in parallel from the Cowork session. Each received only the DT-1 prompt (`prompts/scorer_prompt_deeptime_v1.2-OPT.txt`, SHA-256 64009fc6…) plus one identical wrapper line: answer from own knowledge, no tools, CSV only. No scorer saw the plan, hypotheses, evidence log or run 1. None was told it was one of five. All five reported zero tool use.
**Model:** same configured model as run 1 (`claude-opus-5-5`); so this is a same-model ensemble, not a cross-model test.
**Aggregation:** `scripts/aggregate_ensemble.py`, following the CAMNATIONSM5N spec (dimension-level NA, mean only if n_eff_d >= 3, sample SD, per-scorer V spread, bond_n >= 4). No batch overlap, so no SEAM line. All spec verification checks passed.

### Files
| File | Contents |
| --- | --- |
| `runs/ensemble_claude-subagents_2026-10-02/scorer_A.csv` … `scorer_E.csv` | Raw outputs, verbatim; hashes in `SHA256SUMS` |
| `…/block1_ensemble_mean.csv` | Ensemble mean C K S A, Node Value, Bond Strength |
| `…/block2_envelope.csv` | SD per dimension, V range, n_eff, NA_rate, bond_n |
| `…/node_metrics.csv`, `system_metrics.csv` | Cognition, Energy and case summaries from the means |
| `…/hypothesis_check.md` | H1–H5 on the ensemble |
| `results/comparison_pass1_vs_ensemble_2026-10-02.md` | Run 1 vs ensemble |

### Results
- **Scorer agreement is tight:** mean SD 0.21–0.30 per dimension; mean per-scorer Node Value range 1.57 (max 3.5: Xiongnu Craft, Mycenaean Flow, Aboriginal Shield). Same-model scorers share a corpus, so a tight envelope is expected and is *not* evidence of validity.
- **NA is honest and concentrated:** Longshan Archive and Minoan Shield NA in all five; Harappan Helm, Shield and Lore NA in four of five; Xiongnu Lore NA in four of five, Archive in three. These are the cases with no ruler, no deciphered script, or no own writing.
- **Run 1 vs ensemble:** MAE 0.33–0.48 per dimension, Spearman 0.93–0.95, case ranking Spearman 0.93. Run 1 was more extreme: collapses deeper (Aegean Stewards, Shield, Flow; FIP Hands, Stewards) and Old Kingdom higher. Abstraction +0.34 higher in the ensemble. Likely cause: run 1 was scored with the plan's narrative in context — the narrative-leakage risk the safeguards table names.

### Hypotheses: run 1 vs ensemble
| Hypothesis | Run 1 (in-session) | Ensemble (blind) | Status |
| --- | --- | --- | --- |
| H1a Stewards+Archive stronger in river states | supports (12.81 vs 11.45) | against (11.29 vs 11.32) | Scorer-dependent; not a finding |
| H1b Maritime Flow > Helm | Minoan NA, Māori +2.0 | Minoan +4.1, Māori +0.6, others −1.37 | Direction holds in both |
| H1c Xiongnu Shield–Flow | Shield 1st, Flow 3rd | Shield 1st, Flow 3rd | Holds in both |
| H2 Fall: Energy and Cognition drop together | 8/8 nodes both collapses | 8/8 nodes both collapses | Holds in both (fall leg only) |
| H3 Helm–Lore coherence leads collapse | not testable | not testable | Needs earlier windows |
| H4 Relief edicts lower Hands' Stress | against (7 vs 6.33) | supports (6.2 vs 6.73) | Scorer-dependent; not a finding |
| H5 No ladder | holds | holds | Holds in both |

### Still to do
- Cross-model test: run the same prompt in another model (see `TESTING.md`), and compare against `block1_ensemble_mean.csv`, which now replaces run 1 as the reference.
- Add pre-collapse windows (Egypt late 5th/6th dynasty; Mycenaean c.1350) and recovery windows (Middle Kingdom; Geometric Greece) to make H2 recovery and H3 testable.

## 2026-10-02 — Run 1: claude-opus-5-5, pass 1, all twelve cases

**Who/what:** scored by Claude (configured model ID `claude-opus-5-5`; the serving model may differ) in a Cowork session for Kari McKern.
**Rubric:** CAMS RAW SCORER v1.2-OPT with the deep-time adapter DT-1 (`prompts/scorer_prompt_deeptime_v1.2-OPT.txt`).
**Scope:** 12 cases x 8 nodes = 96 node cells, one anchor year per case.

### Files produced
| File | Contents |
| --- | --- |
| `runs/claude-opus-5-5_pass1/raw_scores.csv` | Raw C, K, S, A per node (96 rows, 9 NA nodes) |
| `runs/claude-opus-5-5_pass1/evidence_log.csv` | Gate decision per node: performance indicator, strain indicator, mechanism |
| `runs/claude-opus-5-5_pass1/node_metrics.csv` | + Cognition (A x C), Energy (K - S), Node Value, Bond Strength, bond_n |
| `runs/claude-opus-5-5_pass1/system_metrics.csv` | Case-level means and lowest-Energy node |
| `runs/claude-opus-5-5_pass1/hypothesis_check.md` | Preliminary H1-H5 reading |

### Steps
1. Anchor years fixed per case (see prompt table). Choices: Taosi phase for Longshan; Wu Ding era for Shang; Giza era for Old Kingdom; Hammurabi for Old Babylonian; LH IIIB for Mycenaean; 1700 CE for pre-contact Aboriginal Australia; 1500 CE for Māori; c.170 BCE for Xiongnu.
2. Each node passed through the three-box evidence gate; failures scored NA on all four dimensions and logged.
3. Scores entered; derived metrics computed with `scripts/backstory_calc.py`.
4. Hypotheses checked with `scripts/hypothesis_check.py`.
5. Blind prompt and comparison script prepared for cross-model testing (`TESTING.md`).

### Deviations from v1.2-OPT (adapter DT-1)
- Year = window anchor, not 31 December; t-1 baseline rule and the 2-point change constraint dropped (no time series within a case).
- Node examples reworded to ancient equivalents (priesthood, scribes, corvée); node boundaries kept functional.
- Oral record systems admitted as Archive evidence.
- Source posture extended to conquerors' accounts, later traditions and colonisers' ethnography.

### Known weaknesses of this run (read before citing)
- **Not blind.** The same model wrote the plan, the hypotheses and these scores in one session. H1-H5 were stated before scoring, but the operationalisations in `hypothesis_check.py` were written after. This is why the cross-model test matters.
- **Not independent.** One pass only; no scorer spread. Five independent passes (camnationsm5n) are still needed.
- **Prompt written after scoring.** The DT-1 prompt codifies the rules applied in this pass, but this pass was not produced by pasting that prompt into a fresh context. A fresh-context Claude pass should be added as the true like-for-like control.
- **Single anchors.** H2 recovery, H3 and the "for longer" part of H4 cannot be tested yet.
- **Evidence gate applied leniently** for 'limitation' as the strain indicator in stable cases (e.g. Harappan Craft: limited bronze use). Stricter reading would raise NA counts.

### Headline readings (pilot only)
- NA is concentrated where it should be: Harappan Helm/Shield/Lore/Stewards and Minoan Helm/Shield/Hands — the 'hidden Helm' societies.
- Both collapses (Egypt FIP, Aegean c.1175) show every node losing Energy and Cognition together.
- H4 (relief devices) reads against at the level of a single window: Old Babylonian Hands Stress 7 vs palace-state mean 6.33.
- H5 (no ladder) holds: non-state cases sit close to stable state cases on Energy and Cognition.
