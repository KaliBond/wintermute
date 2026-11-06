# Arm C JSONL output schema

Return exactly one JSON object per line, one record per era. Do not wrap the JSONL in Markdown fences or add commentary inside the file. Grade your own hashed Arm A pocket. Never revise that pocket in Arm C.

```json
{
  "arm": "C",
  "model": "model name and version",
  "entity": "Russia|Australia|USA|China|UK|Norway|Denmark",
  "era_id": "era id matching the existing era-means CSV",
  "year_start": 1800,
  "year_end": 1825,
  "arm_a_sha256": "sha256 of your submitted Arm A file containing this pocket",
  "q3_grade": "close|broad|partial|mixed|direct_disagreement",
  "mood_levels": {
    "popular": "agree|partial|disagree|unclear",
    "institutional": "agree|partial|disagree|unclear",
    "official": "agree|partial|disagree|unclear"
  },
  "evidence_note": "one or two sentences naming the historical sources; no long quotes",
  "sources": [
    {
      "title": "source title",
      "url": "https://...",
      "accessed_aedt": "YYYY-MM-DD"
    }
  ],
  "timestamp_aedt": "YYYY-MM-DD HH:MM AEDT"
}
```

## Validation

- Required: every field shown above. `sources` must contain at least one source with title and URL; use more than one source where available.
- `entity`, `era_id` and year range must exactly match one row-set in the applicable standard-packet CSV. `arm_a_sha256` must match the submitted Arm A file containing that era's pocket.
- The five `q3_grade` values and four mood-level values are closed vocabularies; use lower case exactly as shown.
- Assess all three mood levels independently. `unclear` means the evidence is insufficient, not agreement.
- CORE expected coverage is 56 records: Russia 23, Australia 18, USA 15. Optional EXTENSION is 51: China 20, UK 16, Norway 15. LIMIT is 19: Denmark I-VIII and X-XX. Denmark IX is not graded and has no Arm A pocket.
- Do not include sealed analyst grades, original analyst pockets, film material or Arm D content.

## Ledger

For each returned file, include: `model`, `arm=C`, `entity_set`, `arm_a_sha256`, `sha256_input` (Arm C packet zip), `sha256_output` (returned JSONL), `timestamp_aedt`, and `operator_notes` with omissions or source limitations. Hash the final JSONL after validation; do not ask for the HOLD key.
