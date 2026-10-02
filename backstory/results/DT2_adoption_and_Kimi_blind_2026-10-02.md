# DT-2 adoption and the Kimi blind pass

2 October 2026 · Backstory (CAMS deep time) · status note between the DT-1 runs and the DT-2 rescore

## Summary

- **The rubric has moved from DT-1 to DT-2.** Three rulings (R1–R3) close the gaps that cross-model scoring exposed: NA versus evidenced absence, oral evidence, and coerced labour. Every result so far was scored under DT-1. Nothing has been rescored yet.
- **A fourth model lineage, Kimi, has scored blind under DT-1.** It is a single pass, so it is a reading, not an ensemble.
- **Kimi reproduces the four results the other three models agreed on.** Collapse drains every node at once. The Aegean recovers far more slowly than it falls. Helm–Lore coherence falls before Egypt's breakdown and not before the Aegean's. Debt-relief edicts do not lower labour stress.
- **Kimi weakens the paper's explanation for H5.** The working paper traced the split on H5 ("no ladder") mainly to how models treat oral evidence. Kimi accepts oral evidence fully and still puts non-state societies well below the state societies. That is the clearest H5 failure in the data so far.

## 1. DT-2: what changed in the rubric

Adopted 2026-10-02 by Kari McKern. R1–R3 were appended verbatim to both scorer prompts under `## DT-2 amendments (2026-10-02)`. The DT-1 text is unchanged above that section, and the full rulings are in `prompts/DT-2-amendment.md`.

| Ruling | Rule | Gap it closes |
| --- | --- | --- |
| R1 | NA means *unknown*. A function whose absence is itself evidenced, such as lost literacy, is scored low. | Aegean-collapse Archive was scored 1 in one context and NA in another. |
| R2 | Archive and Lore are functions, not media. Oral systems are admissible on the same gate terms as writing. A pass that NAs a documented oral system is flagged at intake. | GPT left Lore and Archive NA for every non-literate society. |
| R3 | Abstraction scores the method that exists. Coercion, and method left unused, book to Stress. | The biggest Claude–Grok gap, Old Kingdom Hands Abstraction (3.8 vs 6). |

| Prompt | DT-1 SHA-256 (archived) | DT-2 SHA-256 (current) |
| --- | --- | --- |
| A, 12 societies | `64009fc6…` (`prompts/archive/…_DT1.txt`) | `f4bfc5b8…` |
| B, 10 windows | `90b68399…` (`prompts/archive/…_windows2_DT1.txt`) | `0652e537…` |

### Open before the rescore

1. **R1 conflicts with the output contract.** R1 asks scorers to "state — one sentence —" which NA case holds. The same prompt says "Return ONLY CSV. No prose… No evidence notes." Scorers will either break the CSV or skip R1. This needs a decision: apply R1 silently, or add an NA-reason column.
2. **The worked examples are not blind.** R1 says the Aegean-collapse Archive "is scored low". R3 explains the Old Kingdom Hands disagreement and names Claude and Grok. Scorers would see the expected answer for two scored cells and learn that other models were compared. Either use neutral examples in the prompts (keeping the full text in `DT-2-amendment.md`), or treat those two cells as not independent in the rescore.
3. **The blind bundle is stale.** `Backstory_Prompts_Only_2026-10-02.zip` still holds the DT-1 prompts, so a new DT-2 bundle is needed before any model scores.
4. **Run numbering.** RUNLOG already uses "Run 10" for Grok blind pass 4. The DT-2 rescore needs its own run number.

## 2. Kimi blind pass (RUN-KIMI-BLIND-1)

**Provenance:**
- One pass per prompt, scored in a fresh Kimi session that had no access to the lab conversation.
- Prompts A and B came back together, so isolation between them is not confirmed.
- The model version was not recorded.
- Scored under DT-1. The run is sealed; all 14 hashes in `RUN-KIMI-BLIND-1-SEAL.txt` and both raw-score hashes were verified on import.

Files: `runs/kimi_blind/`, `results/kimi/`.

### Agreement with the other models

| Kimi vs | MAE per dimension (A / B) | Society ranking (A / B) | Mean offset, Kimi − reference (C, K, S, A) |
| --- | --- | --- | --- |
| Claude, 15 passes | 0.52–0.90 / 0.65–1.08 | 0.77 / 0.93 | A: +0.82, +0.54, +0.07, −0.49 · B: +0.98, +0.67, +0.38, −0.49 |
| Grok, 4 passes (A only) | 0.45–0.92 | 0.83 | +0.60, +0.06, −0.10, −0.82 |
| GPT, 3 passes (A only) | 0.44–2.09 | 0.80 | −0.02, −0.40, −0.60, −2.09 |

One Kimi pass sits further from Claude than Grok's four-pass ensemble does, but a single pass carries its own noise: one Claude pass differs from the other Claude runs by MAE 0.26–0.29. Kimi's profile is distinctive. It scores **Coherence high and Abstraction low**: it judges institutions as more unified but less methodologically sophisticated than the other models do. That is the opposite of GPT, which scores Abstraction about 2 points above Kimi.

### What Kimi counts as evidence

| | Claude | Grok | GPT | Kimi |
| --- | --- | --- | --- | --- |
| Harappan Helm / Shield | NA (12 and 14 of 15) | scored | Shield NA | scored |
| Aboriginal and Māori Lore / Archive (oral) | scored 7–8 | scored | NA | **scored 7–8** |
| Aegean-collapse Archive | 1 or NA, by context | NA | NA | **scored low (3, 3, 7, 2)** |
| Longshan Archive, Minoan Shield | NA | NA | NA | NA |

Kimi left only two nodes NA in 176 cells, the two that every model leaves NA. It already behaves the way DT-2's R1 and R2 require: it scored the collapsed Aegean Archive low rather than NA, and scored oral Lore and Archive.

### Hypotheses under Kimi (single pass)

| Hypothesis | Kimi | Consistent with the other three? |
| --- | --- | --- |
| H2a: collapse drains every node | 8/8 Egypt, 8/8 Aegean | Yes |
| H2b: recovery slower than fall | Aegean yes (Energy −6.83/century fall, 0.00 recovery to 1000 BCE); Egypt no | Yes, same split |
| H3: Helm–Lore coherence falls first | Egypt yes (8.5 → 5.5); Aegean no (7.0 → 7.0) | Yes, same split |
| H4: relief edicts lower labour stress | Fails, series (0 of 3) and level (6.0 vs 5.67) | Yes |
| H1b: maritime Flow above Helm | Minoan +1.5, Māori +1.0 vs −0.19 | Yes, direction |
| H1a: river Stewards + Archive | +2.03 | Direction, and larger than the other models |
| H1c: steppe Shield first, Flow high | Shield 1st, Flow 3rd | Matches Claude and GPT; Grok has Flow 5th |
| H5: no ladder | **Fails: Energy ratio 0.36 (0.92 vs 2.59); Cognition ratio 0.69** | No; the clearest failure so far |

### Why the H5 result matters

The working paper argued that H5 changed verdict between models mainly because of NA choices. GPT could not test it because it left oral Lore and Archive NA. Kimi rules that out as the whole story: it scores oral Lore and Archive at 7–8, NAs nothing in the non-state societies, and still puts them far below the states. The difference is in Kimi's other nodes. It gives Aboriginal Stewards, Craft and Hands Capacity 5 and Abstraction 3–4, where Claude gives Capacity 6.0–6.8 and Abstraction 5.0–6.9. It also books more Stress on every Aboriginal node except Archive (4–5, against Claude's 3.0–4.1).

So there are now two separate ways a model can fail H5:
- leave oral institutions unscored (GPT);
- score them but rate the non-state society's other functions as weak and unsophisticated (Kimi).

R2 addresses the first. Nothing in DT-2 addresses the second, and arguably nothing should: it may be a genuine judgement, or the progress-ladder bias the project set out to detect. That is exactly the question H5 was registered to test. A single Kimi pass cannot settle it. A Kimi ensemble and the DT-2 rescore can.

R3 is also already visible in Kimi's Old Kingdom Hands: Abstraction 5 with Stress 5. That sits between Claude's 3.8 and Grok's 6 on Abstraction. The DT-2 rescore will show whether R3 moves all models to the same booking.

## 3. Next steps

1. Resolve the four open items in section 1, then build a DT-2 prompts-only bundle.
2. Run the all-model rescore under DT-2 (Claude, Grok, GPT, Kimi; fresh blind passes, A and B separate), reporting DT-1 vs DT-2 differences cell by cell as findings. Waiting for Kari's go.
3. Bring Kimi to at least three passes so it meets the ensemble minimum (n_eff ≥ 3) and its H5 result can be read against its own noise.

*All Kimi figures come from `results/kimi/` (single pass, DT-1). The other models' figures come from the working paper and RUNLOG.*
