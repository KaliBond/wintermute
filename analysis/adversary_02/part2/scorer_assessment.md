# ADVERSARY-02 Part 2: scorer assessment (one page)
Written 2026-11-04 (AEDT) by the executor for Kari. Evidence: analysis_v2_manus_2026-11-04 (personality_metrics.csv, personality_pass_level.csv, input_validation.csv) and each scorer's run note. These are my judgments from that evidence, not a measurement of truth. Calibration is against the other scorers' pool, which is itself not ground truth: a harsh but consistent scorer is not necessarily wrong.

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
| Kimi | C+ | B- | B | B- | A- | **B** |
| Claude chat | n/a | B | C+ | B+ | B | **B** |
| Codex | B+ | A- | A- | C+ | D | **B-** |
| Manus | B- | B | B+ | D+ | A- | **B-** |
| Perplexity (unconfirmed) | n/a | D | C | D+ | B | **C-** |
| Grok | D | D | D | D+ | C | **D+** |

## One-line verdicts and the evidence behind them
- **Claude agent, A-.** Most repeatable (pass means span 0.30, pass-pair V correlation 0.976) and closest to the pool (V -0.73, mean distance 0.90); its blank cells rise from 12 in 2021 to 32 in 2024; same family as Claude chat.
- **Gemini, B.** Sound level (V -0.43) and sparing blanks (11 node-years, a subset of the common 12), but its note's 23:05 timestamp is impossible against the 22:59:55 handover, and 41.5% of its node-years repeat the prior year; one pass only.
- **Kimi, B.** Completes almost everything (0.9% blank) with a modest +0.45 offset, but its two passes differ by 1.49 in V (search pass 2 is 1.03 above pass 1), its note miscounts corporate cells, and the session memory index was in context.
- **Claude chat, B.** Close to the pool (V +0.53, distance 1.07) and honest in its note, which disclosed an account-level user profile and a list of memory file names (one on CAMS terminology) in context with no file opened; that disclosure also makes it the least clean memory-only run and it shares a family with Claude agent. My earlier box summary dropped the disclosure; corrected 23:37 AEDT.
- **Codex, B-.** Compliant and independent, with consistent passes (correlation 0.975), but it blanks 52% of corporate node-years across every entity and then scores generously (V +0.93, A +1.15 above the pool), so it is indeterminate on most items alone.
- **Manus, B-.** Isolated Mode A passes with search, candid note (mode misread in pass 1, no zero-exposure certificate, discard counts 24/17/47/60/47), disciplined blanks; but it is the harshest scorer (V -1.81) and its passes disagree more than the memory Mode A runs (pass-pair 1.20, Evergrande 2021 M from -4.86 to -1.14). Model undisclosed.
- **Perplexity (unconfirmed), C-.** Valid grid and the standard blank pattern, but no run note, unconfirmed attribution, the most generous level (V +1.39), the widest spread (SD 2.31) and the farthest from the pool cell by cell (2.14).
- **Grok, D+.** Weakest agreement with the pool (V correlation 0.91), 75.8% of node-years carried forward, a level that drifts 2.20 across three passes run in one conversation, a wrong blank count in its note, and it is not independent of this agent family; secondary pool only.

## Caveats
Grades are provisional where only one pass exists. Perplexity's independence grade reflects that it cannot be verified. Manus's Calibration grade penalises distance from the pool, and its harsh level moved one verdict (P6) when added. Part 2 results should be read with the entity-level spread, where all scorers who could score a company place it in the same quadrant for 5 of 8 entities in every year.
