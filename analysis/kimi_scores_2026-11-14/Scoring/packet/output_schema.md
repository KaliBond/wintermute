# Output schema

Return one record per era. Prefer JSONL (one JSON object per line). Markdown tables are acceptable if JSON is unavailable, but must carry the same fields.

## Required fields (Arm A pocket)
```json
{
  "model": "string (model name and version)",
  "arm": "A",
  "entity": "Russia|Australia|USA|China|UK|Norway|Denmark",
  "era_id": "string matching CSV",
  "year_start": 1800,
  "year_end": 1825,
  "leaders": ["Node", "Node"],
  "constraints": ["Node", "Node"],
  "pocket": "Two short sentences. Leadership pair plus constraint. Functional language only.",
  "notes": "optional; keep short",
  "sha256_input_csv": "sha256 of the era_means file you read",
  "sha256_this_output": "sha256 of this file after writing, or leave blank and hash the whole submission",
  "timestamp_aedt": "YYYY-MM-DD HH:MM AEDT"
}
```

## Pocket rules
- Exactly two sentences when the configuration permits.
- Sentence 1: leading one or two functions and what they are doing together.
- Sentence 2: one or two constraints (weak, negative, isolated, or carrying work the leader cannot perform).
- No era titles, no Wikipedia paraphrase, no personality-trait vocabulary.
- No em dashes in pocket text.

## Arm B (blind era ID)
After Arm A pockets are hashed and submitted, the coordinator may send unlabeled pockets. Return:
```json
{
  "arm": "B",
  "model": "...",
  "pocket_id": "opaque id from coordinator",
  "guessed_entity": "...",
  "guessed_era_id_or_years": "...",
  "confidence": "high|medium|low",
  "rationale": "two linked correspondences: leadership + constraint",
  "timestamp_aedt": "..."
}
```

## Arm C (mood continuity)
After Arm A is sealed, open historical sources for that entity and era years only. Grade each era:
```json
{
  "arm": "C",
  "model": "...",
  "entity": "...",
  "era_id": "...",
  "year_start": 0,
  "year_end": 0,
  "q3_grade": "close|broad|partial|mixed|direct_disagreement",
  "mood_levels": {
    "popular": "agree|partial|disagree|unclear",
    "institutional": "agree|partial|disagree|unclear",
    "official": "agree|partial|disagree|unclear"
  },
  "evidence_note": "one or two sentences; cite source titles, not long quotes",
  "timestamp_aedt": "..."
}
```

## Hashing
1. Write Arm A outputs.
2. Compute sha256 of the output file.
3. Record that hash in your ledger before any web search for Arm C.
4. Do not edit Arm A pockets after hashing.

## Ledger fields (every submission)
model, arm, entity_set, sha256_input, sha256_output, timestamp_aedt, operator_notes
