# DT-2 — Deep-Time Adapter, Second Issue (draft for author confirmation)

*Drafted 2026-10-02 in the CAMS lab from the unification memo (DT2-UIS-Unification-Memo.md). Resolves the three DT-2 items named in the Backstory run log and in "Reading Societies Without Their Stories" §8. Wording identical to UIS v1.7 R1–R3, so both research lines carry the same text. Status: DRAFT — K. McKern to confirm; on confirmation, append to prompts/scorer_prompt_deeptime_v1.2-OPT*.txt as DT-2 and rescore per the working paper's next-step 3.*

## Context

Three rubric gaps were found by cross-model scoring (runs 1–9 and RUN-KIMI-BLIND-1):
1. **Absent vs unknown confused:** Aegean-collapse Archive was scored 1/1/9/1 by five scorers in one context and NA by four of five in another. These mean different things; the gate did not separate them.
2. **Oral evidence rule too quiet:** DT-1 admits oral record systems ("continuity across time (written or oral)"), but one lineage NA'd Lore/Archive for every non-literate society regardless. The rule existed and was not applied.
3. **Coerced skilled labour:** the largest cross-model divergence (Old Kingdom Hands Abstraction 3.8 vs 6) showed scorers booking coercion against Abstraction rather than Stress.

## R1 — NA means unknown; evidenced absence is scored low

NA is reserved for *unknown*: neither a performance nor a strain indicator exists for the function. A function whose **absence is itself evidenced** (e.g. documented loss of literacy, abandoned offices, deliberately suppressed institutions) is **scored, low**, with the absence cited as the anchor. Before writing NA the scorer must state — one sentence — which case holds: "no evidence found" or "absence evidenced and scored." Under R1, the Aegean-collapse Archive is scored low (loss of literacy is well evidenced), and the context-dependent flip between 1 and NA disappears.

## R2 — Archive and Lore are functions, not media

Oral transmission systems (songlines, recitation traditions, mnemonic offices, customary law recitals) are admissible evidence for Archive and Lore on the same gate terms as written records: a concrete performance indicator and a strain indicator. A scorer who NAs a documented oral system has misapplied the gate. At intake, such passes are **flagged as rubric-interpretation errors**, not silently averaged into ensembles.

## R3 — Method exists in Abstraction; coercion and disuse book to Stress

Abstraction measures whether codified, transmissible method **exists** — including method for organising coerced or unfree labour. Two symmetric bookings:
1. Where a node's performance rests on coercion, the coercion books to that node's **Stress**, never as a discount on Abstraction.
2. Where a method demonstrably existed but was **not used** when the need was evident, the failure books to **Stress**, not Abstraction — symmetrically, whatever the society.

**Worked example:** Old Kingdom pyramid construction shows codified, transmissible method (surveying, logistics, crew rotation) → Abstraction scores the method; the corvée books to Hands Stress. The Claude–Grok gap dissolves: Claude taxed Abstraction for coercion (wrong metric); Grok failed to book the Stress penalty (missed obligation).

## Adoption plan (from the working paper)

1. Append R1–R3 to both prompt files as DT-2; rehash and log the new SHA-256s in RUNLOG.
2. Rescore all models under DT-2 (fresh blind passes), testing whether the NA differences and H5 converge once the rule is explicit.
3. Report DT-1 vs DT-2 deltas cell-by-cell; changes are findings, not corrections.

*Cross-reference: UIS v1.7 (REPLICATION-KIT/UIS-v1.7-amendment.md) carries the same three rulings for the evidence-packet line. Sealed DT runs are unaffected.*
