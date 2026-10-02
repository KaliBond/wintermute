# Backstory: import index for Kimi (state as of 2026-10-02, after DT-2 adoption)

Root: `C:\Users\julie\wintermute\backstory\`  (all paths below are relative to it)
Bundle: `Backstory_Methods_Results_2026-10-02.zip` (everything listed here, one file)

## BLINDNESS WARNING: read first
If Kimi is going to SCORE the cases as an independent blind model, do NOT import this folder first.
Give Kimi only `Backstory_Prompts_Only_2026-10-02.zip`, collect its scores, and import the rest afterwards.
Everything below (hypotheses, results, other models' scores) would contaminate a blind pass.

## 1. Read in this order
| # | Path | What it is |
|---|---|---|
| 1 | `README.md` | Project thesis, method, open-project commitments, layout |
| 2 | `HYPOTHESES.md` | Pre-registered H1–H5, each with what would count against it |
| 3 | `RUNLOG.md` | Every run, decision, deviation and result, newest first (runs 1–9 + pre-registration) |
| 4 | `TESTING.md` | How cross-model tests are run and read |
| 5 | `cases.csv` | All 20 society-windows (B01–B20) and what each tests |

## 2. Method
| Path | What it is |
|---|---|
| `prompts/scorer_prompt_deeptime_v1.2-OPT.txt` | Prompt A: 12 societies, now DT-2 (SHA-256 f4bfc5b8…) |
| `prompts/scorer_prompt_deeptime_v1.2-OPT_windows2.txt` | Prompt B: 10 extra windows incl. overlap B05, B09, now DT-2 (SHA-256 0652e537…) |
| `prompts/DT-2-amendment.md` | DT-2 rulings R1–R3, adopted 2026-10-02 (`DT-2-amendment-draft.md` = identical source record) |
| `prompts/archive/…_DT1.txt` | DT-1 originals (A 64009fc6…, B 90b68399…); every run so far, including Kimi's, was scored under DT-1 |
| `scripts/aggregate_ensemble.py` | N scorer CSVs → ensemble mean (block1) + envelope (block2); CAMNATIONSM5N spec, NA-aware |
| `scripts/backstory_calc.py` | Cognition (A×C), Energy (K−S), Node Value, Bond Strength |
| `scripts/hypothesis_check.py` | H1–H5 single-window tests |
| `scripts/hypothesis_check_series.py` | H2 recovery, H3, H4 series tests (pre-registered, hash in RUNLOG) |
| `scripts/compare_models.py` | Cell-by-cell model comparison: MAE, Spearman, NA agreement, offsets |
| `runs/claude-opus-5-5_pass1/evidence_log.csv` | Evidence-gate reasoning for every node (run 1) |

## 3. Raw scores (verbatim, with SHA256SUMS)
| Path | Model | Passes | Notes |
|---|---|---|---|
| `runs/claude-opus-5-5_pass1/raw_scores.csv` | Claude | 1 | In-session, NOT blind; superseded |
| `runs/ensemble_claude-subagents_2026-10-02/scorer_A..E.csv` | Claude | 5 | Prompt A, blind subagents |
| `runs/ensemble_windows2_2026-10-02/scorer_1..5.csv` | Claude | 5 | Prompt B, blind subagents |
| `runs/blind_replication_claude/pass1..5/` | Claude Opus 5.5 | 5 | Independent session replication |
| `runs/replication_R6/pass1..5/` | probably Claude (unconfirmed) | 5 | Second replication |
| `runs/grok-4.7/` | Grok 4.7 | 1 | NOT blind (had read hypotheses) |
| `runs/grok-4.7_blind/` + `pass2/`, `pass3/`, `pass4/` | Grok 4.7 | 4 | Blind, one conversation per prompt |
| `runs/gpt_3pass/pass1..3/` | GPT (ChatGPT Work/Codex, version unverified) | 3 | Isolated agent contexts; A and B in same context |
| `runs/kimi_blind/pass1/` | Kimi (version not recorded) | 1 | RUN-KIMI-BLIND-1, sealed (`RUN-KIMI-BLIND-1-SEAL.txt`); A and B returned together, isolation not confirmed |

## 4. Ensembles and results
| Path | What it is |
|---|---|
| `runs/claude_grand/promptA/`, `promptB/` | Pooled Claude reference, 15 independent passes per prompt |
| `runs/claude_grand/series_check.md` | Claude 15-pass series tests (H2, H3, H4) |
| `runs/claude_grand/promptA/hypothesis_check.md` | Claude 15-pass H1–H5 |
| `runs/grok-4.7_blind/ensemble4/` | Grok 4-pass ensemble (current), hypothesis_check.md, series_check.md; `ensemble3/` kept for history |
| `runs/gpt_3pass/A/`, `B/`, `hypothesis_check.md`, `series_check.md` | GPT 3-pass ensemble and tests |
| `results/grok3_vs_claude15_promptA.md`, `…promptB.md` | Grok vs Claude |
| `results/three_vs_claude_A.md`, `…_B.md` | GPT vs Claude (files named before GPT was identified) |
| `results/three_vs_grok_A.md`, `…_B.md` | GPT vs Grok |
| `results/replication_prompt*.md`, `R6_prompt*_vs_all.md` | Claude test–retest |
| `runs/kimi_blind/A_calc/`, `B_calc/`, `series/` | Kimi single-pass metrics |
| `results/kimi/` | Kimi vs Claude 15 (A, B), vs Grok 4 (A), vs GPT 3 (A); Kimi hypothesis and series checks |

## 5. Headline results so far (details in RUNLOG.md)
Holding across Claude (15), Grok (4) and GPT (3):
- H2 (fall): in both collapses, Energy and Cognition fall together in all or almost all nodes.
- H2 (recovery): the Aegean recovers far more slowly than it fell; Egypt recovers about as fast (GPT alone says Egypt slower too).
- H3: Helm–Lore coherence falls before Energy turns negative in Egypt, not in the Aegean.
- H4: debt-relief edicts do NOT show lower Hands Stress (likely confounded by design; recorded as a failure).
- H1b: maritime societies show Flow above Helm (direction holds; effect smaller in Grok).
Model-dependent or not a finding: H1a (river Stewards+Archive), H1c (steppe Flow rank), H5 (no ladder: holds in Claude, exactly on the 0.80 cut-off in Grok at 0.798, untestable in GPT because of NA). H1c: Flow 3rd in Claude and GPT, 5th in Grok.

Scorer differences: Grok ≈ +0.3 above Claude; GPT ≈ +1 (Abstraction +1.7). Claude NAs Harappan Helm and Shield; Grok never does; GPT NAs Lore and Archive in all non-literate societies (treats oral tradition as no evidence).

## 6. Open items
- DT-2 adopted 2026-10-02 (R1 evidenced absence scored low, NA = unknown; R2 oral systems count for Lore and Archive; R3 coercion books to Stress, not Abstraction). See RUNLOG "Run 10 prep: DT-2 adopted".
- **Next step: all-model rescore under DT-2** (Claude, Grok, GPT, Kimi; fresh blind passes), reporting DT-1 vs DT-2 differences cell by cell. Not started: waiting for Kari's go.
- Grok passes 5–6 pending; five-pass blind GPT run; confirm R6 model.
- H6 (post hoc, untested): recovery speed depends on whether the setting forces re-coordination.

## 7. Write-up
Working paper (Claude Docs): "Reading Societies Without Their Stories", https://claude.ai/code/artifact/b4dc6c9f-e102-4dd6-9607-7a3a0047a1cc — reflects runs 1–10.

- `results/node_means_three_models.csv` — node-level ensemble means for all three models (added with the Complete results write-up). H1a now reads "holds in direction; small and NA-sensitive".
