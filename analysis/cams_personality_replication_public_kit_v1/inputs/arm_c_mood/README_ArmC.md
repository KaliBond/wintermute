# CAMS Arm C: independent mood continuity

Prepared for Kari McKern (Neural Nations / CAMS) by Reportmister.
Packet time zone: Australia/Sydney (AEDT).

## Purpose

Arm C independently checks Q3 mood continuity against historical sources. Web research is allowed in this arm only. Use your own hashed Arm A pockets, not the original analyst pockets. This packet contains no sealed Q3 grades and no answer key.

Run CORE first: Russia, Australia and USA, 56 eras total. EXTENSION is optional next: China, UK and Norway, 51 eras. Run LIMIT Denmark last, 19 graded eras. Denmark IX (1901-1913) has no observations, no Arm A pocket and no sealed Q3 grade; omit it rather than inventing a result.

## Preconditions

1. Arm A must already be hashed and submitted. Record its sha256 in `arm_a_sha256` on every Arm C return.
2. Use your own sealed Arm A JSONL and the era ids and year ranges in your existing standard-packet CSVs: `era_means.csv`, then optional `era_means_extension.csv`, then `era_means_denmark.csv`.
3. Read `output_schema_ArmC.md` before grading. Return one JSONL record per era, with popular, institutional and official mood assessed separately.
4. Do not change, regenerate or rehash Arm A pockets during Arm C. If a pocket is missing, omit that era and disclose the gap; never substitute an analyst pocket.

## For each era

1. Open historical sources for that entity and those years only. Use Britannica, Wikipedia and period press where available. Add polls, institutional records or official statements when they help separate the three mood levels. Prefer more than one source and distinguish retrospective reference histories from contemporaneous reporting.
2. Compare your own hashed pocket's leadership, constraints and emotional register with the sourced record. Keep popular feeling, organised institutional mood and official messaging separate. Press under censorship is evidence about the public arena and its constraints, not a clean plebiscite.
3. Assign one Q3 grade:
   - **close:** leadership, constraint and emotional register all fit, with no material contradiction.
   - **broad:** the central configuration fits, but one meaningful feature is missing or underplayed.
   - **partial:** the structure is recognisable, but the emotional register or one central function substantially differs.
   - **mixed:** sources describe opposing official and popular moods, or the pocket fits only one important social current.
   - **direct_disagreement:** the leading configuration or central emotional register is wrong.
4. For long eras, check whether the average conceals a turning point. Qualify the grade instead of counting one famous episode as a win. Positive activation is not happiness; positive Shield can be coercion, positive Helm can be state self-confidence, and operational continuity need not imply popular confidence.
5. Write a short evidence note naming source titles and dates or URLs where practical. No long quotes.

## Return and hash

1. Save CORE as `armC_CORE.jsonl`; save EXTENSION and LIMIT separately if run.
2. Verify the output against `output_schema_ArmC.md`, then compute sha256 for each file.
3. Return the JSONL, source-linked evidence notes and a ledger line: `model`, `arm=C`, `entity_set`, `arm_a_sha256`, `sha256_input` (this packet zip), `sha256_output`, `timestamp_aedt`, and coverage gaps.
4. Do not request, consult or infer the sealed analyst Q3 key. Central scoring happens only after your output hash is recorded.

Pre-registered central criterion: majority exact-or-adjacent Q3, with no overturning disagreement pattern. Arm C does not reopen Arm A or Arm B results.

## Files

- `README_ArmC.md`
- `output_schema_ArmC.md`
- `SHORT_PASTE.md`
- `MANIFEST.sha256`

The tester uses the era CSVs and Arm A pockets already received with the standard packet; they are deliberately not duplicated here.
