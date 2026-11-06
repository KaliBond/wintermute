# ADVERSARY-02 Part 2: scorer assessment (one page)
Written 2026-11-04 (AEDT); Kimi Work assessment added 2026-11-06 (AEDT). Evidence: analysis_v2_manus_2026-11-04 (personality_metrics.csv, personality_pass_level.csv, input_validation.csv) and each scorer's run note. Kimi Work: analysis_v3_kimiwork_2026-11-05 (scoring_table.csv, kimiwork_vs_pool.csv) and runs/part2/kimiwork/look_report.md (MANIFEST 95f9357d…; analysis MANIFEST 4d3413ba…). These are my judgments from that evidence, not a measurement of truth. Calibration is against the other scorers' pool, which is itself not ground truth: a harsh but consistent scorer is not necessarily wrong.

## Rubric
Five criteria, graded A to D. Overall = equal-weight mean of the graded criteria (A 4, B 3, C 2, D 1, with plus and minus steps), banded back to a letter. Reliability needs at least two passes, so single-pass scorers are ungraded there (n/a) and their overall averages four criteria.
- Reliability: agreement between the scorer's own passes (pass-to-pass V difference and correlation, spread of pass means, carry-forward stickiness).
- Rule compliance: run note present and accurate, mode followed, disclosures, timestamps, counts that match the files.
- Independence: isolation between passes, family overlap with other scorers, anything in context that could carry framework knowledge.
- Calibration: mean V offset from the leave-one-out pool and mean absolute V distance from it (pool includes Manus).
- NA discipline: sparing, consistent, explained blanks that match the run note.

## Grades
| Scorer | Reliability | Compliance | Independence | Calibration | NA discipline | Overall |
|---|---|---|---|---|---|---|
| Claude agent | A- | A | B | A- | B+ | **A-** |
| Gemini | n/a | C | A- | B | A- | **B** |
| Kimi web | C+ | B- | B | B- | A- | **B** |
| Claude chat | n/a | B | C+ | B+ | B | **B** |
| Codex | B+ | A- | A- | C+ | D | **B-** |
| Manus | B- | B | B+ | D+ | A- | **B-** |
| Kimi Work (secondary) | C+ | B+ | D | C+ | B+ | **C+** |
| Perplexity (unconfirmed) | n/a | D | C | D+ | B | **C-** |
| Grok | D | D | D | D+ | C | **D+** |

## One-line verdicts and the evidence behind them
- **Claude agent, A-.** Most repeatable (pass means span 0.30, pass-pair V correlation 0.976) and closest to the pool (V -0.73, mean distance 0.90); its blank cells rise from 12 in 2021 to 32 in 2024; same family as Claude chat.
- **Gemini, B.** Sound level (V -0.43) and sparing blanks (11 node-years, a subset of the common 12), but its note's 23:05 timestamp is impossible against the 22:59:55 handover, and 41.5% of its node-years repeat the prior year; one pass only.
- **Kimi web, B.** Completes almost everything (0.9% blank) with a modest +0.45 offset, but its two passes differ by 1.49 in V (search pass 2 is 1.03 above pass 1), its note miscounts corporate cells, and the session memory index was in context.
- **Claude chat, B.** Close to the pool (V +0.53, distance 1.07) and honest in its note, which disclosed an account-level user profile and a list of memory file names (one on CAMS terminology) in context with no file opened; that disclosure also makes it the least clean memory-only run and it shares a family with Claude agent. My earlier box summary dropped the disclosure; corrected 23:37 AEDT.
- **Codex, B-.** Compliant and independent, with consistent passes (correlation 0.975), but it blanks 52% of corporate node-years across every entity and then scores generously (V +0.93, A +1.15 above the pool), so it is indeterminate on most items alone.
- **Manus, B-.** Isolated Mode A passes with search, candid note (mode misread in pass 1, no zero-exposure certificate, discard counts 24/17/47/60/47), disciplined blanks; but it is the harshest scorer (V -1.81) and its passes disagree more than the memory Mode A runs (pass-pair 1.20, Evergrande 2021 M from -4.86 to -1.14). Model undisclosed.
- **Kimi Work, C+ (secondary only).** Five Mode A passes had clean formats and NA counts of 64/60/60/32/60; 58 of 69 blank node-years were on Workforce/Hands. Pairwise repeatability was r = 0.912 (0.892–0.932) and carry-forward was 18–28%, the lowest in the set. All passes read one shared November 2026 evidence pack, China evidence came from a single collector with about 12 memory-dated items, and only passes 4–5 searched for themselves: independence fails the protocol despite honest declaration. The run was harsh on strong firms (CATL −2.37 V, Huawei −2.03, SMIC −1.77, BYD −1.49). The claimed model name `k2d8-preview` is unverified. Treat it as a separate run from Kimi web; its look report is `kimiwork_look_report.md`.
- **Perplexity (unconfirmed), C-.** Valid grid and the standard blank pattern, but no run note, unconfirmed attribution, the most generous level (V +1.39), the widest spread (SD 2.31) and the farthest from the pool cell by cell (2.14).
- **Grok, D+.** Weakest agreement with the pool (V correlation 0.91), 75.8% of node-years carried forward, a level that drifts 2.20 across three passes run in one conversation, a wrong blank count in its note, and it is not independent of this agent family; secondary pool only.

## Secondary-pool sensitivity

Adding Kimi Work to the primary pool (e) flips **P6** from FAIL (13/16) to PASS (14/16) and **P8** from FAIL (4.35/5.32) to PASS (5.03/5.99, limit 5.0). In the hypothetical replacement pool (f), Kimi Work substitutes for Kimi web: P6 stays FAIL (13/16) and P8 passes at 5.08/5.94. All other item verdicts are unchanged; FC5 is NOT TRIGGERED. These checks do not change the primary 6 PASS / 9 FAIL headline or primary-pool membership. Full metrics: `scoring_table.md` and `scoring_table.csv`.

## Best-two-scorer sensitivity (post-hoc)

If scoring had used only the two highest-graded scorers, the canonical pair would be **Claude agent (A−) and Gemini (B)**, the top overall grade then the first B-tier scorer with sound calibration. Equal run weights and primary thresholds give **6 PASS / 9 FAIL with no indeterminates**, identical to the primary scorecard, and FC5 NOT TRIGGERED. P6 remains FAIL at 13 satisfied / 3 not; P8 remains FAIL at rounded ratios 5.00/5.91 because the unrounded Vanke ratio is 4.998819 (unrounded 4.998818728522337), below the sealed ≥5.0 threshold. If Kimi web occupied the second seat instead, P6 would pass (14/16), giving 7 PASS / 8 FAIL, while P8 would remain FAIL at 4.30/5.23. This is supplementary and changes neither grades nor primary-pool membership. See `best2_scorers_sensitivity.md`.

## Caveats
Grades are provisional where only one pass exists. Perplexity's independence grade reflects that it cannot be verified. Manus's Calibration grade penalises distance from the pool, and its harsh level moved one verdict (P6) when added. Kimi Work's r is not directly comparable with independently gathered passes because all five shared evidence. It remains secondary alongside Grok and does not join or replace any run in the primary pool. Part 2 results should be read with the entity-level spread, where all scorers who could score a company place it in the same quadrant for 5 of 8 entities in every year.
